#!/usr/bin/env python3
"""Classify new molecules using pre-trained pipeline models."""

import os, sys, json, pickle
import numpy as np

for pkg in ['dscribe', 'ase']:
    try: __import__(pkg)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

from ase import Atoms
from ase.io import read as ase_read
from dscribe.descriptors import SOAP

MODEL_DIR = './models'


def load_pipeline(model_dir=MODEL_DIR):
    """Load all pre-trained pipeline components."""
    print("[LOAD] Loading pipeline...")

    with open(os.path.join(model_dir, 'config.json')) as f:
        config = json.load(f)

    soap_cfg = config['soap']
    soap = SOAP(
        species=soap_cfg['species'], r_cut=soap_cfg['r_cut'],
        n_max=soap_cfg['n_max'], l_max=soap_cfg['l_max'],
        sigma=soap_cfg['sigma'], average=soap_cfg['average'], periodic=False,
    )

    def _load(name):
        with open(os.path.join(model_dir, name), 'rb') as f:
            return pickle.load(f)

    pipeline = {
        'config': config, 'soap': soap,
        'scaler': _load('scaler.pkl'),
        'ipca': _load('ipca.pkl'),
        'kmeans': _load('kmeans.pkl'),
        'iso_forest': _load('isolation_forest.pkl') if os.path.exists(
            os.path.join(model_dir, 'isolation_forest.pkl')) else None,
        'ocsvm': _load('ocsvm.pkl') if os.path.exists(
            os.path.join(model_dir, 'ocsvm.pkl')) else None,
    }

    print(f"[LOAD] Ready: {config['n_molecules']} mols, "
          f"K={config['clustering']['best_k']}")
    return pipeline


def predict(molecules, pipeline):
    """Predict cluster labels for molecules. Returns dict with labels and anomaly flags."""
    soap = pipeline['soap']
    scaler = pipeline['scaler']

    features = soap.create(molecules)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    std = np.sqrt(np.maximum(scaler['M2'] / (scaler['n'] - 1), 1e-10))
    X_scaled = (features - scaler['mean']) / std
    X_pca = pipeline['ipca'].transform(X_scaled)

    is_anomaly = np.zeros(len(molecules), dtype=bool)
    if pipeline['iso_forest'] and pipeline['ocsvm']:
        is_anomaly = (pipeline['iso_forest'].predict(X_pca) == -1) & \
                     (pipeline['ocsvm'].predict(X_pca) == -1)

    return {
        'cluster_labels': pipeline['kmeans'].predict(X_pca),
        'is_anomaly': is_anomaly,
        'pca_features': X_pca,
    }


def parse_xyz_file(filepath):
    try:
        return ase_read(filepath)
    except Exception:
        with open(filepath) as f:
            lines = f.read().strip().split('\n')
        n = int(lines[0])
        symbols, positions = [], []
        for i in range(2, 2 + n):
            parts = lines[i].split()
            symbols.append(parts[0].replace('*', ''))
            positions.append([float(c.replace('*^', 'e')) for c in parts[1:4]])
        return Atoms(symbols=symbols, positions=positions)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <file.xyz | directory/ | --demo>")
        sys.exit(0)

    pipeline = load_pipeline()

    if sys.argv[1] == '--demo':
        mol = Atoms('CH4', positions=[[0,0,0],[1,0,0],[0,1,0],[0,0,1],[-1,0,0]])
        result = predict([mol], pipeline)
        print(f"  CH4 → Cluster {result['cluster_labels'][0]}, "
              f"Anomaly: {'Yes' if result['is_anomaly'][0] else 'No'}")

    elif os.path.isfile(sys.argv[1]):
        mol = parse_xyz_file(sys.argv[1])
        result = predict([mol], pipeline)
        print(f"  {sys.argv[1]} → Cluster {result['cluster_labels'][0]}")

    elif os.path.isdir(sys.argv[1]):
        files = sorted(f for f in os.listdir(sys.argv[1]) if f.endswith('.xyz'))
        mols = [parse_xyz_file(os.path.join(sys.argv[1], f)) for f in files]
        result = predict(mols, pipeline)
        for f, c, a in zip(files, result['cluster_labels'], result['is_anomaly']):
            print(f"  {f:<35s}  Cluster {c}  {'ANOMALY' if a else ''}")
