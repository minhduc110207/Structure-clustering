#!/usr/bin/env python3
"""
Export trained models from the pipeline to models/ directory.
Run this after the pipeline completes if not using the integrated Stage 8.
"""

import os, json, pickle
import numpy as np

MODEL_DIR = './models'


def export_models(scaler, ipca, ipca_full, iso_forest, ocsvm,
                  best_model, best_k, kmeans_results,
                  n_mols, n_feat, n_pca_components, cumvar,
                  anomaly_stats, config):
    os.makedirs(MODEL_DIR, exist_ok=True)

    def _save(name, obj):
        with open(os.path.join(MODEL_DIR, name), 'wb') as f:
            pickle.dump(obj, f)
        print(f"[EXPORT] ✓ {name}")

    _save('scaler.pkl', {'n': scaler.n, 'mean': scaler.mean, 'M2': scaler.M2})
    _save('ipca.pkl', ipca)
    _save('ipca_full.pkl', ipca_full)
    _save('isolation_forest.pkl', iso_forest)
    _save('ocsvm.pkl', ocsvm)
    _save('kmeans.pkl', best_model)

    np.save(os.path.join(MODEL_DIR, 'cumulative_variance.npy'), cumvar)

    meta = {
        'pipeline': 'Molecular Structural Classification',
        'dataset': 'QM9', 'n_molecules': n_mols,
        'soap': {'n_max': config.SOAP_NMAX, 'l_max': config.SOAP_LMAX,
                 'r_cut': config.SOAP_RCUT, 'sigma': config.SOAP_SIGMA,
                 'species': config.SOAP_SPECIES, 'n_features': n_feat},
        'pca': {'n_components': n_pca_components,
                'variance': float(cumvar[n_pca_components - 1])},
        'anomaly': anomaly_stats,
        'clustering': {'best_k': best_k},
    }
    with open(os.path.join(MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    total = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR))
    print(f"\n[EXPORT] All saved to {MODEL_DIR}/ ({total/(1024**2):.1f} MB)")


if __name__ == '__main__':
    print("Call export_models() from the main pipeline after training.")
