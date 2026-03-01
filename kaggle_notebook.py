#!/usr/bin/env python3
"""
Molecular Structural Classification — Unsupervised Pipeline (with Checkpoint/Resume)
Based on: "Phân loại cấu trúc" research report

Pipeline: SOAP → Welford Scaler → IPCA → Anomaly Detection → K-means → Analysis → Export
"""

# === SETUP ===

import subprocess, sys

def install_packages():
    for pkg in ['dscribe', 'ase', 'h5py']:
        try:
            __import__(pkg)
        except ImportError:
            print(f"[SETUP] Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

install_packages()

# === IMPORTS ===

import os, io, json, time, pickle, tarfile, warnings, requests
import numpy as np
import h5py
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

from ase import Atoms
from ase.io import read as ase_read
from dscribe.descriptors import SOAP

from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# === CONFIGURATION ===

class Config:
    DATA_DIR        = './data'
    QM9_URLS        = [
        'https://springernature.figshare.com/ndownloader/files/3195389',
        'https://ndownloader.figshare.com/files/3195389',
        'https://figshare.com/ndownloader/files/3195389',
    ]
    QM9_TAR         = 'dsgdb9nsd.xyz.tar.bz2'
    QM9_MIN_SIZE    = 50 * 1024 * 1024
    MAX_MOLECULES   = None   # None = all ~134k; set 5000 for quick test
    HDF5_PATH       = './qm9_soap_features.h5'
    BATCH_SIZE      = 2048

    SOAP_NMAX       = 8
    SOAP_LMAX       = 8
    SOAP_RCUT       = 6.0
    SOAP_SIGMA      = 1.0
    SOAP_SPECIES    = ['C', 'H', 'O', 'N', 'F']
    SOAP_AVERAGE    = 'inner'

    PCA_VARIANCE    = 0.95

    IF_CONTAMINATION = 0.05
    OCSVM_NU         = 0.05
    OCSVM_KERNEL     = 'rbf'

    K_RANGE          = [3, 5, 8, 10, 15, 20, 30]
    KMEANS_BATCH     = 2048
    KMEANS_NINIT     = 10

    RESULTS_DIR      = './results'
    CHECKPOINT_DIR   = './checkpoints'
    CHECKPOINT_FILE  = './checkpoints/pipeline_state.json'

cfg = Config()

# === CHECKPOINT MANAGER ===

class CheckpointManager:
    """Saves pipeline state after each stage for crash-resilient execution."""

    STAGES = [
        'stage0_data', 'stage1_soap', 'stage2_scaler',
        'stage3_pca', 'stage4_anomaly', 'stage5_kmeans',
    ]

    def __init__(self, config: Config):
        self.config = config
        self.state = {}
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.config.CHECKPOINT_FILE):
            try:
                with open(self.config.CHECKPOINT_FILE, 'r') as f:
                    self.state = json.load(f)
                print(f"[CKPT] Loaded checkpoint: "
                      f"{len(self.state.get('completed', {}))} stage(s) done")
            except (json.JSONDecodeError, IOError):
                print("[CKPT] Corrupt checkpoint file, starting fresh")
                self.state = {}
        else:
            print("[CKPT] No checkpoint found — starting from scratch")
            self.state = {}
        if 'completed' not in self.state:
            self.state['completed'] = {}

    def _save(self):
        with open(self.config.CHECKPOINT_FILE, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)

    def is_stage_done(self, stage_name: str) -> bool:
        info = self.state['completed'].get(stage_name)
        if info is None:
            return False
        for fpath in info.get('output_files', []):
            if not os.path.exists(fpath):
                print(f"[CKPT] Stage '{stage_name}': '{fpath}' missing → will re-run")
                return False
        return True

    def mark_done(self, stage_name: str, output_files=None, metadata=None):
        self.state['completed'][stage_name] = {
            'timestamp': datetime.now().isoformat(),
            'output_files': output_files or [],
            'metadata': metadata or {},
        }
        self._save()
        print(f"[CKPT] ✓ Stage '{stage_name}' checkpointed")

    def get_metadata(self, stage_name: str) -> dict:
        return self.state['completed'].get(stage_name, {}).get('metadata', {})

    def save_object(self, name: str, obj):
        path = os.path.join(self.config.CHECKPOINT_DIR, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def load_object(self, name: str):
        path = os.path.join(self.config.CHECKPOINT_DIR, f'{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def get_resume_point(self) -> str:
        for stage in self.STAGES:
            if not self.is_stage_done(stage):
                return stage
        return None

    def print_status(self):
        print("\n" + "═"*60)
        print("  CHECKPOINT STATUS")
        print("═"*60)
        resume = self.get_resume_point()
        for stage in self.STAGES:
            done = self.is_stage_done(stage)
            icon = "✓" if done else ("→" if stage == resume else "○")
            info = self.state['completed'].get(stage, {})
            ts = info.get('timestamp', '')
            if ts:
                ts = f"  ({ts[:19]})"
            print(f"  {icon}  {stage:<20s}{ts}")
        if resume:
            print(f"\n  ★ Will resume from: {resume}")
        else:
            print(f"\n  ★ All stages complete!")
        print("═"*60 + "\n")


ckpt = CheckpointManager(cfg)
ckpt.print_status()

print("=" * 60)
print("  Molecular Structural Classification Pipeline")
print("=" * 60)
print(f"  SOAP: n_max={cfg.SOAP_NMAX}, l_max={cfg.SOAP_LMAX}, "
      f"r_cut={cfg.SOAP_RCUT}Å, σ={cfg.SOAP_SIGMA}")
print(f"  PCA variance: {cfg.PCA_VARIANCE*100:.0f}%  |  "
      f"Anomaly: IF({cfg.IF_CONTAMINATION})+OCSVM({cfg.OCSVM_NU})")
print(f"  K candidates: {cfg.K_RANGE}")
print("=" * 60)

# === DATA LOADING ===

def _verify_tar(tar_path: str) -> bool:
    try:
        with tarfile.open(tar_path, 'r:bz2') as tar:
            if tar.next() is None:
                return False
        return True
    except (tarfile.TarError, EOFError, OSError):
        return False


def _try_download_url(url: str, tar_path: str, min_size: int) -> bool:
    tmp_path = tar_path + '.tmp'
    try:
        print(f"[DATA]   Trying: {url[:80]}...")
        response = requests.get(
            url, stream=True, timeout=300, allow_redirects=True,
            headers={'Accept': 'application/octet-stream, application/x-bzip2, */*'}
        )
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' in content_type:
            print(f"[DATA]   ⚠ Server returned HTML — skipping")
            return False

        total = int(response.headers.get('content-length', 0))
        if 0 < total < min_size:
            print(f"[DATA]   ⚠ File too small ({total} bytes) — skipping")
            return False

        downloaded = 0
        with open(tmp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    print(f"\r[DATA]   {downloaded/(1024**2):.1f}/{total/(1024**2):.1f} MB "
                          f"({pct:.1f}%)", end='')
        print()

        if downloaded < min_size:
            print(f"[DATA]   ⚠ File too small ({downloaded/(1024**2):.1f} MB)")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            return False

        os.replace(tmp_path, tar_path)
        print(f"[DATA]   Complete ({downloaded/(1024**2):.1f} MB)")
        return True

    except (requests.RequestException, IOError) as e:
        print(f"\n[DATA]   ⚠ Failed: {e}")
        if os.path.exists(tmp_path): os.remove(tmp_path)
        return False


def download_qm9(config: Config):
    """Download QM9 with multi-mirror support and integrity verification."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    tar_path = os.path.join(config.DATA_DIR, config.QM9_TAR)

    if os.path.exists(tar_path):
        print(f"[DATA] Found cached QM9 at {tar_path}")
        print("[DATA] Verifying...", end=' ')
        if _verify_tar(tar_path):
            print(f"OK ({os.path.getsize(tar_path)/(1024**2):.1f} MB)")
            return tar_path
        else:
            print("CORRUPT — re-downloading")
            os.remove(tar_path)

    print("[DATA] Downloading QM9 dataset...")
    for i, url in enumerate(config.QM9_URLS):
        print(f"[DATA] Mirror {i+1}/{len(config.QM9_URLS)}:")
        if _try_download_url(url, tar_path, config.QM9_MIN_SIZE):
            print("[DATA] Verifying...", end=' ')
            if _verify_tar(tar_path):
                print("OK ✓")
                return tar_path
            else:
                print("CORRUPT — next mirror")
                if os.path.exists(tar_path): os.remove(tar_path)

    raise RuntimeError(
        "\n" + "="*60 + "\n"
        "  Failed to download QM9 from all mirrors.\n"
        "  Try adding QM9 as Kaggle Dataset input instead.\n"
        "  https://quantum-machine.org/datasets/\n"
        + "="*60
    )


def parse_qm9_xyz(content: str) -> Atoms:
    """Parse a single QM9 .xyz file into ASE Atoms."""
    lines = content.strip().split('\n')
    n_atoms = int(lines[0].strip())
    symbols, positions = [], []
    for i in range(2, 2 + n_atoms):
        parts = lines[i].strip().split()
        symbols.append(parts[0].replace('*', ''))
        positions.append([float(c.replace('*^', 'e')) for c in parts[1:4]])
    return Atoms(symbols=symbols, positions=positions)


def load_qm9_molecules(config: Config):
    """Load QM9 molecules from tar.bz2 archive."""
    tar_path = download_qm9(config)
    print("[DATA] Extracting molecules from archive...")
    molecules, failed = [], 0

    with tarfile.open(tar_path, 'r:bz2') as tar:
        members = [m for m in tar.getmembers() if m.name.endswith('.xyz')]
        total = len(members)
        if config.MAX_MOLECULES is not None:
            members = members[:config.MAX_MOLECULES]
            print(f"[DATA] Using first {config.MAX_MOLECULES} of {total}")
        else:
            print(f"[DATA] Loading all {total} molecules")

        for i, member in enumerate(members):
            try:
                f = tar.extractfile(member)
                if f is not None:
                    molecules.append(parse_qm9_xyz(f.read().decode('utf-8')))
            except Exception:
                failed += 1

            if (i + 1) % 10000 == 0 or (i + 1) == len(members):
                print(f"\r[DATA] Parsed {i+1}/{len(members)} (failed: {failed})", end='')

    print(f"\n[DATA] Loaded {len(molecules)} molecules ({failed} failures)")
    return molecules


# --- STAGE 0: DATA LOADING ---

print("\n" + "─"*50)
print("  STAGE 0: DATA LOADING")
print("─"*50)

molecules = None

if ckpt.is_stage_done('stage0_data'):
    print("[CKPT] ✓ Stage 0 done — data cached")
else:
    tar_path = download_qm9(cfg)
    ckpt.mark_done('stage0_data',
                   output_files=[os.path.join(cfg.DATA_DIR, cfg.QM9_TAR)])

# --- STAGE 1: SOAP FEATURE EXTRACTION ---

print("\n" + "─"*50)
print("  STAGE 1: SOAP FEATURE EXTRACTION")
print("─"*50)


def compute_soap_to_hdf5(molecules, config: Config):
    """Compute SOAP descriptors and save to HDF5 in chunks."""
    soap = SOAP(
        species=config.SOAP_SPECIES, r_cut=config.SOAP_RCUT,
        n_max=config.SOAP_NMAX, l_max=config.SOAP_LMAX,
        sigma=config.SOAP_SIGMA, average=config.SOAP_AVERAGE, periodic=False,
    )
    n_mols = len(molecules)
    n_feat = soap.get_number_of_features()
    batch_size = config.BATCH_SIZE

    print(f"[SOAP] {n_mols} molecules, {n_feat} features, batch={batch_size}")

    with h5py.File(config.HDF5_PATH, 'w') as hf:
        dset = hf.create_dataset(
            'soap_features', shape=(n_mols, n_feat), dtype='float32',
            chunks=(min(batch_size, n_mols), n_feat),
            compression='gzip', compression_opts=4,
        )
        start_time = time.time()
        for start in range(0, n_mols, batch_size):
            end = min(start + batch_size, n_mols)
            batch_features = soap.create(molecules[start:end])
            if batch_features.ndim == 1:
                batch_features = batch_features.reshape(1, -1)
            dset[start:end] = batch_features.astype(np.float32)

            elapsed = time.time() - start_time
            rate = end / elapsed if elapsed > 0 else 0
            eta = (n_mols - end) / rate if rate > 0 else 0
            print(f"\r[SOAP] {end}/{n_mols} ({end/n_mols*100:.1f}%) "
                  f"— {rate:.0f} mol/s — ETA: {eta:.0f}s", end='')

        print(f"\n[SOAP] Saved to {config.HDF5_PATH} "
              f"({os.path.getsize(config.HDF5_PATH)/(1024**2):.1f} MB)")

    return n_mols, n_feat


if ckpt.is_stage_done('stage1_soap'):
    print("[CKPT] ✓ Stage 1 done — loading metadata")
    meta = ckpt.get_metadata('stage1_soap')
    n_mols, n_feat = meta['n_molecules'], meta['n_features']
    print(f"[SOAP] Cached: {n_mols} mols, {n_feat} features")
else:
    if molecules is None:
        molecules = load_qm9_molecules(cfg)

    all_elements = []
    n_atoms_list = []
    for mol in molecules:
        all_elements.extend(mol.get_chemical_symbols())
        n_atoms_list.append(len(mol))
    print(f"[STATS] {len(molecules)} molecules, "
          f"atoms/mol: {min(n_atoms_list)}-{max(n_atoms_list)} "
          f"(mean {np.mean(n_atoms_list):.1f})")
    print(f"[STATS] Elements: {dict(Counter(all_elements))}")

    n_mols, n_feat = compute_soap_to_hdf5(molecules, cfg)
    ckpt.mark_done('stage1_soap',
                   output_files=[cfg.HDF5_PATH],
                   metadata={'n_molecules': n_mols, 'n_features': n_feat})
    del molecules
    molecules = None

# --- STAGE 2: WELFORD SCALER ---

print("\n" + "─"*50)
print("  STAGE 2: ONLINE STANDARDSCALER (Welford)")
print("─"*50)

class WelfordScaler:
    """Online StandardScaler using Welford's algorithm for batch processing."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None
        self._fitted = False

    def partial_fit(self, batch: np.ndarray):
        if self.mean is None:
            self.mean = np.zeros(batch.shape[1], dtype=np.float64)
            self.M2 = np.zeros(batch.shape[1], dtype=np.float64)
        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
        self._fitted = True
        return self

    @property
    def variance(self):
        return self.M2 / (self.n - 1) if self.n >= 2 else np.ones_like(self.mean)

    @property
    def std(self):
        return np.sqrt(np.maximum(self.variance, 1e-10))

    def transform(self, batch: np.ndarray) -> np.ndarray:
        assert self._fitted, "Scaler not fitted yet."
        return (batch - self.mean) / self.std

    def fit_transform_batched(self, hdf5_path, dataset_name, batch_size):
        """Two-pass: compute stats → transform."""
        scaled_path = hdf5_path.replace('.h5', '_scaled.h5')

        print("[SCALER] Pass 1/2: Computing statistics...")
        with h5py.File(hdf5_path, 'r') as hf:
            dset = hf[dataset_name]
            n_samples, n_features = dset.shape
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                self.partial_fit(dset[start:end][:])
                print(f"\r  [{end}/{n_samples}]", end='')

        print(f"\n[SCALER] mean=[{self.mean.min():.4f}, {self.mean.max():.4f}], "
              f"std=[{self.std.min():.4f}, {self.std.max():.4f}]")

        print("[SCALER] Pass 2/2: Transforming...")
        with h5py.File(hdf5_path, 'r') as hf_in, \
             h5py.File(scaled_path, 'w') as hf_out:
            dset_in = hf_in[dataset_name]
            dset_out = hf_out.create_dataset(
                'scaled_features', shape=dset_in.shape, dtype='float32',
                chunks=(min(batch_size, n_samples), n_features), compression='gzip',
            )
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                dset_out[start:end] = self.transform(dset_in[start:end][:]).astype(np.float32)
                print(f"\r  [{end}/{n_samples}]", end='')

        print(f"\n[SCALER] Saved to {scaled_path}")
        return scaled_path


scaled_hdf5 = cfg.HDF5_PATH.replace('.h5', '_scaled.h5')

if ckpt.is_stage_done('stage2_scaler'):
    print("[CKPT] ✓ Stage 2 done")
    scaler = ckpt.load_object('welford_scaler')
    if scaler is None:
        scaler = WelfordScaler()
else:
    scaler = WelfordScaler()
    scaled_hdf5 = scaler.fit_transform_batched(cfg.HDF5_PATH, 'soap_features', cfg.BATCH_SIZE)
    ckpt.save_object('welford_scaler', scaler)
    ckpt.mark_done('stage2_scaler',
                   output_files=[scaled_hdf5],
                   metadata={'scaled_path': scaled_hdf5})

# --- STAGE 3: INCREMENTAL PCA ---

print("\n" + "─"*50)
print("  STAGE 3: INCREMENTAL PCA")
print("─"*50)


def fit_incremental_pca(scaled_path: str, config: Config):
    """Two-pass IPCA: analyze variance → fit optimal → transform."""
    batch_size = config.BATCH_SIZE

    with h5py.File(scaled_path, 'r') as hf:
        n_samples, n_features = hf['scaled_features'].shape

    # Pass 1: Full IPCA for variance analysis
    print(f"[PCA] Pass 1: {n_samples} samples, {n_features} features")
    max_components = min(n_features, batch_size, 300)
    ipca_full = IncrementalPCA(n_components=max_components)

    with h5py.File(scaled_path, 'r') as hf:
        dset = hf['scaled_features']
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = dset[start:end][:]
            if batch.shape[0] >= max_components:
                ipca_full.partial_fit(batch)
            print(f"\r  [{end}/{n_samples}]", end='')

    cumvar = np.cumsum(ipca_full.explained_variance_ratio_)
    n_components = min(int(np.searchsorted(cumvar, config.PCA_VARIANCE) + 1), max_components)

    print(f"\n[PCA] {n_features} → {n_components} components "
          f"({cumvar[n_components-1]*100:.2f}% variance)")

    # Pass 2: Fit optimal and transform
    print(f"[PCA] Pass 2: Fitting n={n_components}...")
    ipca = IncrementalPCA(n_components=n_components)

    with h5py.File(scaled_path, 'r') as hf:
        dset = hf['scaled_features']
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = dset[start:end][:]
            if batch.shape[0] >= n_components:
                ipca.partial_fit(batch)
            print(f"\r  [{end}/{n_samples}]", end='')

    print(f"\n[PCA] Transforming...")
    pca_path = scaled_path.replace('_scaled.h5', '_pca.h5')

    with h5py.File(scaled_path, 'r') as hf_in, \
         h5py.File(pca_path, 'w') as hf_out:
        dset_in = hf_in['scaled_features']
        dset_out = hf_out.create_dataset(
            'pca_features', shape=(n_samples, n_components), dtype='float32',
            chunks=(min(batch_size, n_samples), n_components), compression='gzip',
        )
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            dset_out[start:end] = ipca.transform(dset_in[start:end][:]).astype(np.float32)
            print(f"\r  [{end}/{n_samples}]", end='')

    print(f"\n[PCA] Saved to {pca_path}")
    return pca_path, ipca, ipca_full, n_components, cumvar


pca_path = cfg.HDF5_PATH.replace('.h5', '_pca.h5')

if ckpt.is_stage_done('stage3_pca'):
    print("[CKPT] ✓ Stage 3 done")
    meta = ckpt.get_metadata('stage3_pca')
    n_pca_components = meta['n_components']
    cumvar = np.array(meta['cumulative_variance'])
    ipca_full = ckpt.load_object('ipca_full')
    ipca = ckpt.load_object('ipca_optimal')
else:
    pca_path, ipca, ipca_full, n_pca_components, cumvar = fit_incremental_pca(scaled_hdf5, cfg)
    ckpt.save_object('ipca_full', ipca_full)
    ckpt.save_object('ipca_optimal', ipca)
    ckpt.mark_done('stage3_pca',
                   output_files=[pca_path],
                   metadata={
                       'pca_path': pca_path,
                       'n_components': n_pca_components,
                       'cumulative_variance': cumvar.tolist(),
                   })

# PCA Variance Plot
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(range(1, len(cumvar) + 1), cumvar * 100, 'b-', linewidth=2)
ax.axhline(y=cfg.PCA_VARIANCE * 100, color='r', linestyle='--',
           label=f'{cfg.PCA_VARIANCE*100:.0f}% threshold')
ax.axvline(x=n_pca_components, color='g', linestyle='--',
           label=f'n_components = {n_pca_components}')
ax.set_xlabel('Number of Components', fontsize=12)
ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
ax.set_title('Incremental PCA — Cumulative Variance', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'pca_variance.png'), dpi=150)
plt.show()
print("[PLOT] Saved pca_variance.png")

# --- STAGE 4: ANOMALY DETECTION ---

print("\n" + "─"*50)
print("  STAGE 4: ANOMALY DETECTION")
print("─"*50)


def detect_anomalies(pca_path: str, config: Config):
    """Consensus anomaly detection: Isolation Forest + One-class SVM."""
    with h5py.File(pca_path, 'r') as hf:
        X_pca = hf['pca_features'][:]

    n_samples = X_pca.shape[0]
    print(f"[ANOMALY] {n_samples} samples, {X_pca.shape[1]} dims")

    print("[ANOMALY] Isolation Forest...")
    iso_forest = IsolationForest(
        contamination=config.IF_CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1)
    labels_if = iso_forest.fit_predict(X_pca)
    n_if = np.sum(labels_if == -1)
    print(f"  → IF anomalies: {n_if} ({n_if/n_samples*100:.2f}%)")

    MAX_OCSVM = 20000
    if n_samples > MAX_OCSVM:
        print(f"[ANOMALY] OCSVM (subsampling {MAX_OCSVM})...")
        idx = np.random.choice(n_samples, MAX_OCSVM, replace=False)
        ocsvm = OneClassSVM(kernel=config.OCSVM_KERNEL, nu=config.OCSVM_NU)
        ocsvm.fit(X_pca[idx])
        labels_ocsvm = ocsvm.predict(X_pca)
    else:
        print("[ANOMALY] One-class SVM...")
        ocsvm = OneClassSVM(kernel=config.OCSVM_KERNEL, nu=config.OCSVM_NU)
        labels_ocsvm = ocsvm.fit_predict(X_pca)

    n_svm = np.sum(labels_ocsvm == -1)
    print(f"  → OCSVM anomalies: {n_svm} ({n_svm/n_samples*100:.2f}%)")

    consensus = (labels_if == -1) & (labels_ocsvm == -1)
    n_cons = np.sum(consensus)
    normal_mask = ~consensus

    print(f"  → Consensus removed: {n_cons} ({n_cons/n_samples*100:.2f}%)")
    print(f"  → Clean remaining: {np.sum(normal_mask)}")

    X_clean = X_pca[normal_mask]
    clean_path = pca_path.replace('_pca.h5', '_clean.h5')
    with h5py.File(clean_path, 'w') as hf:
        hf.create_dataset('clean_features', data=X_clean, compression='gzip')

    stats = {
        'if_anomalies': int(n_if), 'ocsvm_anomalies': int(n_svm),
        'consensus_anomalies': int(n_cons), 'clean_samples': int(np.sum(normal_mask)),
    }
    return clean_path, X_clean, normal_mask, stats, iso_forest, ocsvm


clean_path = cfg.HDF5_PATH.replace('.h5', '_clean.h5')

if ckpt.is_stage_done('stage4_anomaly'):
    print("[CKPT] ✓ Stage 4 done")
    anomaly_stats = ckpt.get_metadata('stage4_anomaly')
    with h5py.File(clean_path, 'r') as hf:
        X_clean = hf['clean_features'][:]
    iso_forest = ckpt.load_object('iso_forest')
    ocsvm = ckpt.load_object('ocsvm')
    normal_mask = ckpt.load_object('normal_mask')
else:
    clean_path, X_clean, normal_mask, anomaly_stats, iso_forest, ocsvm = detect_anomalies(pca_path, cfg)
    ckpt.save_object('iso_forest', iso_forest)
    ckpt.save_object('ocsvm', ocsvm)
    ckpt.save_object('normal_mask', normal_mask)
    ckpt.mark_done('stage4_anomaly', output_files=[clean_path], metadata=anomaly_stats)

# --- STAGE 5: MINI-BATCH K-MEANS ---

print("\n" + "─"*50)
print("  STAGE 5: MINI-BATCH K-MEANS")
print("─"*50)


def optimize_kmeans(X: np.ndarray, config: Config):
    """Run K-means for multiple K, evaluate with Silhouette and DBI."""
    k_range = config.K_RANGE
    n_samples = X.shape[0]
    sil_size = min(10000, n_samples)

    results = {'k_values': [], 'inertia': [], 'silhouette': [], 'dbi': [], 'models': []}
    print(f"[KMEANS] K={k_range}, {n_samples} samples (sil sample={sil_size})\n")

    for k in k_range:
        print(f"  K={k:3d}  ", end='', flush=True)
        t0 = time.time()
        km = MiniBatchKMeans(
            n_clusters=k, batch_size=config.KMEANS_BATCH,
            n_init=config.KMEANS_NINIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)

        if n_samples > sil_size:
            idx = np.random.choice(n_samples, sil_size, replace=False)
            sil = silhouette_score(X[idx], labels[idx])
            dbi = davies_bouldin_score(X[idx], labels[idx])
        else:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)

        results['k_values'].append(k)
        results['inertia'].append(float(km.inertia_))
        results['silhouette'].append(float(sil))
        results['dbi'].append(float(dbi))
        results['models'].append(km)

        print(f"WCSS={km.inertia_:.2e}  Sil={sil:.4f}  DBI={dbi:.4f}  ({time.time()-t0:.1f}s)")

    best_idx = np.argmax(results['silhouette'])
    best_k = results['k_values'][best_idx]
    best_model = results['models'][best_idx]
    print(f"\n[KMEANS] ★ Best K={best_k} (Sil={results['silhouette'][best_idx]:.4f})")

    return results, best_k, best_model


if ckpt.is_stage_done('stage5_kmeans'):
    print("[CKPT] ✓ Stage 5 done")
    meta = ckpt.get_metadata('stage5_kmeans')
    best_k = meta['best_k']
    best_model = ckpt.load_object('best_kmeans_model')
    kmeans_results = ckpt.load_object('kmeans_results')
    final_labels = best_model.predict(X_clean)
else:
    kmeans_results, best_k, best_model = optimize_kmeans(X_clean, cfg)
    kr_save = {k: kmeans_results[k] for k in ['k_values', 'inertia', 'silhouette', 'dbi']}
    ckpt.save_object('best_kmeans_model', best_model)
    ckpt.save_object('kmeans_results', kr_save)
    final_labels = best_model.predict(X_clean)
    ckpt.mark_done('stage5_kmeans', output_files=[], metadata={
        'best_k': best_k, **kr_save})

cluster_counts = Counter(final_labels)
print(f"\n[KMEANS] Cluster distribution (K={best_k}):")
for c in sorted(cluster_counts):
    print(f"  Cluster {c}: {cluster_counts[c]} ({cluster_counts[c]/len(final_labels)*100:.1f}%)")

# --- STAGE 6: VISUALIZATION ---

print("\n" + "─"*50)
print("  STAGE 6: VISUALIZATION")
print("─"*50)

kr = kmeans_results if isinstance(kmeans_results, dict) and 'models' not in kmeans_results else {
    k: kmeans_results[k] for k in ['k_values', 'inertia', 'silhouette', 'dbi']}

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]
ax.plot(kr['k_values'], kr['inertia'], 'bo-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('WCSS'); ax.set_title('Elbow Method')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(kr['k_values'], kr['silhouette'], 'go-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('Silhouette'); ax.set_title('Silhouette Coefficient')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(kr['k_values'], kr['dbi'], 'rs-', linewidth=2, markersize=8)
ax.axvline(x=best_k, color='b', linestyle='--', alpha=0.7, label=f'Best K={best_k}')
ax.set_xlabel('K'); ax.set_ylabel('DBI (lower=better)'); ax.set_title('Davies-Bouldin Index')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
scatter = ax.scatter(X_clean[:, 0], X_clean[:, 1], c=final_labels, cmap='tab10', s=2, alpha=0.5)
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.set_title(f'Clusters (K={best_k}) — PCA Projection')
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.grid(True, alpha=0.3)

plt.suptitle('Molecular Structural Classification — Results', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'clustering_results.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[PLOT] Saved clustering_results.png")

# Anomaly summary plot
fig, ax = plt.subplots(figsize=(8, 5))
methods = ['Isolation\nForest', 'One-class\nSVM', 'Consensus\n(Both)', 'Clean\nData']
counts = [anomaly_stats['if_anomalies'], anomaly_stats['ocsvm_anomalies'],
          anomaly_stats['consensus_anomalies'], anomaly_stats['clean_samples']]
colors = ['#e74c3c', '#e67e22', '#c0392b', '#27ae60']
bars = ax.bar(methods, counts, color=colors, edgecolor='black', linewidth=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Samples'); ax.set_title('Anomaly Detection Summary', fontsize=14)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'anomaly_summary.png'), dpi=150)
plt.show()
print("[PLOT] Saved anomaly_summary.png")

# --- FINAL SUMMARY ---

ckpt.print_status()
print("=" * 60)
print("  ★ PIPELINE COMPLETE")
print("=" * 60)
print(f"""
  Dataset:      QM9 ({n_mols} molecules)
  SOAP:         {n_feat} dims
  PCA:          {n_pca_components} components ({cfg.PCA_VARIANCE*100:.0f}% variance)
  Anomalies:    {anomaly_stats['consensus_anomalies']} removed
  Clean data:   {anomaly_stats['clean_samples']} samples
  Best K:       {best_k}
  Silhouette:   {kr['silhouette'][np.argmax(kr['silhouette'])]:.4f}
  DBI:          {kr['dbi'][np.argmax(kr['silhouette'])]:.4f}
""")
print("=" * 60)

# --- STAGE 7: CLUSTER ANALYSIS ---

print("\n" + "─"*50)
print("  STAGE 7: CLUSTER ANALYSIS")
print("─"*50)

ATOMIC_MASS = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998}


def analyze_clusters(molecules, normal_mask, final_labels, config):
    """Compute per-cluster molecular properties."""
    clean_indices = np.where(normal_mask)[0]
    properties = {
        'cluster': final_labels,
        'n_atoms': [], 'n_heavy': [], 'mol_weight': [], 'r_gyration': [],
        'frac_H': [], 'frac_C': [], 'frac_N': [], 'frac_O': [], 'frac_F': [],
    }

    for idx in clean_indices:
        mol = molecules[idx]
        symbols = mol.get_chemical_symbols()
        positions = mol.positions
        n = len(symbols)

        properties['n_atoms'].append(n)
        properties['n_heavy'].append(sum(1 for s in symbols if s != 'H'))
        properties['mol_weight'].append(sum(ATOMIC_MASS.get(s, 0) for s in symbols))

        elem_count = Counter(symbols)
        for elem in ['H', 'C', 'N', 'O', 'F']:
            properties[f'frac_{elem}'].append(elem_count.get(elem, 0) / n)

        center = positions.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((positions - center)**2, axis=1)))
        properties['r_gyration'].append(rg)

    for key in properties:
        properties[key] = np.array(properties[key])
    return properties


print("[ANALYSIS] Loading molecules...")
if molecules is None:
    molecules = load_qm9_molecules(cfg)

if 'normal_mask' not in dir() or normal_mask is None:
    with h5py.File(pca_path, 'r') as hf:
        n_pca = hf['pca_features'].shape[0]
    normal_mask = np.ones(n_pca, dtype=bool)

props = analyze_clusters(molecules, normal_mask, final_labels, cfg)

# Statistics table
print(f"\n{'='*75}")
print(f"  CLUSTER CHARACTERIZATION (K={best_k})")
print(f"{'='*75}")
print(f"{'Metric':<25s}", end='')
for c in range(best_k):
    print(f"{'Cluster '+str(c):>15s}", end='')
print()
print("─"*75)

for label, key, agg in [
    ('Samples','cluster','count'), ('Avg atoms','n_atoms','mean'),
    ('Avg heavy atoms','n_heavy','mean'), ('Avg mol weight','mol_weight','mean'),
    ('Avg R_gyration','r_gyration','mean'), ('Avg %H','frac_H','mean'),
    ('Avg %C','frac_C','mean'), ('Avg %N','frac_N','mean'),
    ('Avg %O','frac_O','mean'), ('Avg %F','frac_F','mean'),
]:
    print(f"  {label:<23s}", end='')
    for c in range(best_k):
        mask_c = props['cluster'] == c
        vals = props[key][mask_c] if key != 'cluster' else mask_c
        if agg == 'count':
            print(f"{np.sum(vals):>15d}", end='')
        else:
            print(f"{np.mean(vals):>15.2f}", end='')
    print()
print(f"{'='*75}")

# Cluster analysis plots
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
cluster_colors = plt.cm.tab10(np.linspace(0, 1, best_k))[:best_k]

ax = axes[0, 0]
for c in range(best_k):
    mask_c = props['cluster'] == c
    ax.hist(props['n_atoms'][mask_c], bins=range(2, 32), alpha=0.6,
            label=f'Cluster {c} (n={np.sum(mask_c)})', color=cluster_colors[c])
ax.set_xlabel('Number of Atoms'); ax.set_ylabel('Count')
ax.set_title('Atom Count Distribution'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
elements = ['H', 'C', 'N', 'O', 'F']
x_pos = np.arange(best_k)
for i, elem in enumerate(elements):
    means = [np.mean(props[f'frac_{elem}'][props['cluster'] == c]) for c in range(best_k)]
    ax.bar(x_pos + i*0.15, means, 0.15, label=elem, alpha=0.85)
ax.set_xlabel('Cluster'); ax.set_ylabel('Avg Fraction')
ax.set_title('Element Composition'); ax.set_xticks(x_pos + 0.3)
ax.set_xticklabels([f'C{c}' for c in range(best_k)]); ax.legend(); ax.grid(True, axis='y', alpha=0.3)

ax = axes[1, 0]
bp = ax.boxplot([props['mol_weight'][props['cluster']==c] for c in range(best_k)],
                labels=[f'C{c}' for c in range(best_k)], patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_xlabel('Cluster'); ax.set_ylabel('Mol. Weight (g/mol)')
ax.set_title('Molecular Weight'); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
bp = ax.boxplot([props['r_gyration'][props['cluster']==c] for c in range(best_k)],
                labels=[f'C{c}' for c in range(best_k)], patch_artist=True, notch=True)
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color); patch.set_alpha(0.6)
ax.set_xlabel('Cluster'); ax.set_ylabel('Rg (Å)')
ax.set_title('Structural Compactness'); ax.grid(True, alpha=0.3)

plt.suptitle('Cluster Analysis', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'cluster_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()
print("[PLOT] Saved cluster_analysis.png")

# Cluster interpretation
print("\n" + "─"*50)
print("  CLUSTER INTERPRETATION")
print("─"*50)
for c in range(best_k):
    mask_c = props['cluster'] == c
    n_c = np.sum(mask_c)
    avg_atoms = np.mean(props['n_atoms'][mask_c])
    avg_heavy = np.mean(props['n_heavy'][mask_c])
    avg_mw = np.mean(props['mol_weight'][mask_c])
    avg_rg = np.mean(props['r_gyration'][mask_c])
    dom = max(['C','N','O','F'], key=lambda e: np.mean(props[f'frac_{e}'][mask_c]))

    size = "Small" if avg_atoms < 12 else ("Medium" if avg_atoms < 20 else "Large")
    shape = "compact" if avg_rg < 1.5 else ("moderate" if avg_rg < 2.0 else "extended")

    print(f"\n  ★ Cluster {c} ({n_c} mols, {n_c/len(final_labels)*100:.1f}%):")
    print(f"    {avg_atoms:.1f} atoms ({avg_heavy:.1f} heavy), {avg_mw:.1f} g/mol, Rg={avg_rg:.2f}Å")
    print(f"    → {size}, {shape}, dominant: {dom}")

print("\n" + "=" * 60)
print("  Analysis complete! ✓")
print("=" * 60)

# --- STAGE 8: EXPORT MODELS ---

print("\n" + "─"*50)
print("  STAGE 8: EXPORT MODELS")
print("─"*50)

EXPORT_DIR = './models'
os.makedirs(EXPORT_DIR, exist_ok=True)

scaler_export = {'n': scaler.n, 'mean': scaler.mean, 'M2': scaler.M2}
with open(os.path.join(EXPORT_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler_export, f)

with open(os.path.join(EXPORT_DIR, 'ipca.pkl'), 'wb') as f:
    pickle.dump(ipca, f)
with open(os.path.join(EXPORT_DIR, 'ipca_full.pkl'), 'wb') as f:
    pickle.dump(ipca_full, f)

with open(os.path.join(EXPORT_DIR, 'isolation_forest.pkl'), 'wb') as f:
    pickle.dump(iso_forest, f)
with open(os.path.join(EXPORT_DIR, 'ocsvm.pkl'), 'wb') as f:
    pickle.dump(ocsvm, f)

with open(os.path.join(EXPORT_DIR, 'kmeans.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

np.save(os.path.join(EXPORT_DIR, 'cumulative_variance.npy'), cumvar)

export_meta = {
    'pipeline': 'Molecular Structural Classification',
    'dataset': 'QM9',
    'n_molecules': n_mols,
    'soap': {
        'n_max': cfg.SOAP_NMAX, 'l_max': cfg.SOAP_LMAX,
        'r_cut': cfg.SOAP_RCUT, 'sigma': cfg.SOAP_SIGMA,
        'species': cfg.SOAP_SPECIES, 'average': cfg.SOAP_AVERAGE,
        'n_features': n_feat,
    },
    'pca': {
        'n_components': n_pca_components,
        'variance_retained': float(cumvar[n_pca_components - 1]),
    },
    'anomaly': anomaly_stats,
    'clustering': {
        'best_k': best_k,
        'k_values': [int(k) for k in kr['k_values']],
        'silhouette': [float(s) for s in kr['silhouette']],
        'dbi': [float(d) for d in kr['dbi']],
        'inertia': [float(i) for i in kr['inertia']],
    },
}
with open(os.path.join(EXPORT_DIR, 'config.json'), 'w') as f:
    json.dump(export_meta, f, indent=2, default=str)

total_bytes = sum(os.path.getsize(os.path.join(EXPORT_DIR, f)) for f in os.listdir(EXPORT_DIR))
print(f"\n{'='*60}")
print(f"  ★ MODELS EXPORTED — {EXPORT_DIR}/ ({total_bytes/(1024**2):.1f} MB)")
print(f"{'='*60}")
for f in sorted(os.listdir(EXPORT_DIR)):
    print(f"    {f:<30s} {os.path.getsize(os.path.join(EXPORT_DIR, f))/(1024):.1f} KB")
print(f"{'='*60}")
