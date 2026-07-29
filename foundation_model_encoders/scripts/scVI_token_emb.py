# ============================================================
# Imports
# ============================================================
import os
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Iterable, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    import scvi
    import torch
    SCVI_AVAILABLE = True
except Exception as e:
    SCVI_AVAILABLE = False
    SCVI_IMPORT_ERROR = repr(e)

print("scanpy:", sc.__version__)
print("anndata:", ad.__version__)
print("scVI available:", SCVI_AVAILABLE)
if not SCVI_AVAILABLE:
    print("scVI import error:", SCVI_IMPORT_ERROR)

from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

# ============================================================
# USER CONFIG
# ============================================================

# Dataset-specific pseudo-control root. Use the dataset-level /single folder, not the global /ibex/project/c2366/Perturb_data root.
DATASET_ID = env_str("PPFM_DATASET_ID", "dataset")
PERTURBED_GROUP = env_str("PPFM_PERTURBED_GROUP", "single")
PSEUDO_SEARCH_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

VARIANTS = env_list("PPFM_VARIANTS", [])

# Output is external to individual variants but still stored beside the pseudo-control dataset folder.
OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_SEARCH_ROOT / "_hvg_scvi_embeddings")

# If True, recompute even when output files already exist.
OVERWRITE = env_bool("PPFM_OVERWRITE", False)

# Matrix source.
# None means use adata.X. Otherwise set to a layer name, e.g. "counts" or "raw_counts".
COUNTS_LAYER = env_optional_str("PPFM_COUNTS_LAYER", None)

# Optional metadata keys. For pseudo-control h5ad files, these may not exist.
# HVG and scVI will only use BATCH_KEY if it exists in adata.obs and has >=2 levels.
BATCH_KEY = None      # e.g. "batch", "assay_batch", or None
LABEL_KEY = "perturbation_label"  # only copied to metadata if present; not required for embeddings

# Which methods to run.
RUN_HVG_PCA = env_bool("PPFM_RUN_HVG_PCA", True)
RUN_SCVI = env_bool("PPFM_RUN_SCVI", True)

# Whether to write full h5ad files with embeddings in obsm.
# External .npy embeddings are always written.
WRITE_H5AD_WITH_EMB = env_bool("PPFM_WRITE_H5AD", True)

# ============================================================
# HVG-PCA settings
# ============================================================
HVG_OBSM_KEY = "X_pca_hvg"
N_HVGS = 2000
N_PCS = 50
HVG_FLAVOR = "seurat_v3"   # fallback to cell_ranger/seurat if seurat_v3 dependencies fail
SCALE_MAX_VALUE = 10
SCALE_ZERO_CENTER = False   # safer for sparse/large pseudo-control matrices

# ============================================================
# scVI settings
# ============================================================
SCVI_OBSM_KEY = "X_scVI"
SCVI_N_LATENT = 30
SCVI_N_HIDDEN = 128
SCVI_N_LAYERS = 2
SCVI_GENE_LIKELIHOOD = "nb"
SCVI_TRAIN_MAX_EPOCHS = 200
SCVI_TRAIN_BATCH_SIZE = env_int("PPFM_BATCH_SIZE", 256)
SCVI_USE_GPU = env_bool("PPFM_USE_GPU", True)
SEED = 42

# ============================================================
# Output directories
# ============================================================
HVG_EMB_DIR = OUT_ROOT / "hvg_pca" / "embeddings"
HVG_H5AD_DIR = OUT_ROOT / "hvg_pca" / "h5ad_with_hvg_pca"

SCVI_EMB_DIR = OUT_ROOT / "scvi" / "embeddings"
SCVI_H5AD_DIR = OUT_ROOT / "scvi" / "h5ad_with_scvi"
SCVI_MODEL_DIR = OUT_ROOT / "scvi" / "models"

MANIFEST_DIR = OUT_ROOT / "manifests"

for d in [HVG_EMB_DIR, HVG_H5AD_DIR, SCVI_EMB_DIR, SCVI_H5AD_DIR, SCVI_MODEL_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("PSEUDO_SEARCH_ROOT:", PSEUDO_SEARCH_ROOT)
print("OUT_ROOT:", OUT_ROOT)

# ============================================================
# Reproducibility and basic helpers
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    if SCVI_AVAILABLE:
        scvi.settings.seed = seed
        if torch.cuda.is_available():
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def sanitize_variant_name(name: str) -> str:
    return (
        str(name)
        .strip("/")
        .replace("/", "__")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("&", "and")
    )


def infer_variant_name(pseudo_h5ad: Path, root: Path) -> str:
    pseudo_h5ad = Path(pseudo_h5ad)
    root = Path(root)
    rel_parent = pseudo_h5ad.parent.relative_to(root)
    return sanitize_variant_name(str(rel_parent))


def variant_matches(variant_name: str, pseudo_path: Path, filters: List[str]) -> bool:
    if not filters:
        return True
    text_options = [
        variant_name,
        variant_name.replace("__", "/"),
        str(pseudo_path),
    ]
    for flt in filters:
        flt = str(flt).strip()
        if not flt:
            continue
        flt2 = sanitize_variant_name(flt)
        if any(flt in t or flt2 in t for t in text_options):
            return True
    return False


def discover_pseudo_h5ads(root: Path, variants: Optional[List[str]] = None) -> pd.DataFrame:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Pseudo-control search root does not exist: {root}")

    paths = sorted(root.rglob("pseudo_control_aligned_to_perturbed.h5ad"))
    rows = []
    seen = set()
    for p in paths:
        # Avoid accidentally rediscovering outputs from this notebook.
        if "_hvg_scvi_embeddings" in p.parts:
            continue
        variant_name = infer_variant_name(p, root)
        if variant_name in seen:
            raise RuntimeError(
                f"Variant name collision: {variant_name}. Check root={root} and path={p}"
            )
        seen.add(variant_name)
        if variant_matches(variant_name, p, variants or []):
            rows.append({
                "variant": variant_name,
                "variant_path_style": variant_name.replace("__", "/"),
                "pseudo_h5ad": str(p),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No pseudo_control_aligned_to_perturbed.h5ad files matched under {root}. "
            f"Check PSEUDO_SEARCH_ROOT and VARIANTS."
        )
    return df


def get_matrix_from_adata(adata: ad.AnnData, counts_layer: Optional[str] = None):
    if counts_layer is None:
        return adata.X
    if counts_layer not in adata.layers:
        raise KeyError(f"counts_layer '{counts_layer}' not found in adata.layers")
    return adata.layers[counts_layer]


def copy_selected_matrix_to_x(adata: ad.AnnData, counts_layer: Optional[str] = None) -> ad.AnnData:
    adata = adata.copy()
    X = get_matrix_from_adata(adata, counts_layer=counts_layer)
    if sp.issparse(X):
        adata.X = X.copy().tocsr()
    else:
        adata.X = np.asarray(X, dtype=np.float32)
    return adata


def choose_valid_batch_key(adata: ad.AnnData, batch_key: Optional[str]) -> Optional[str]:
    if batch_key is None:
        return None
    if batch_key not in adata.obs.columns:
        print(f"[Info] batch_key={batch_key!r} not found. Running without batch key.")
        return None
    if adata.obs[batch_key].nunique(dropna=False) < 2:
        print(f"[Info] batch_key={batch_key!r} has <2 levels. Running without batch key.")
        return None
    return batch_key


def write_json(path: Path, obj: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def append_manifest_row(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        # Keep latest row for a method/variant pair.
        if {"method", "variant"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["method", "variant"], keep="last")
    df.to_csv(path, index=False)


pseudo_df = discover_pseudo_h5ads(PSEUDO_SEARCH_ROOT, VARIANTS)

# ============================================================
# HVG-PCA embedding
# ============================================================

def _run_highly_variable_genes_with_fallback(
    adata: ad.AnnData,
    n_top_genes: int,
    flavor: str,
    batch_key: Optional[str],
):
    flavors = [flavor]
    for fallback in ["cell_ranger", "seurat"]:
        if fallback not in flavors:
            flavors.append(fallback)

    last_error = None
    for flv in flavors:
        try:
            kwargs = dict(
                n_top_genes=min(n_top_genes, adata.n_vars),
                flavor=flv,
                subset=False,
                inplace=True,
            )
            if batch_key is not None:
                kwargs["batch_key"] = batch_key
            sc.pp.highly_variable_genes(adata, **kwargs)
            return flv
        except Exception as e:
            last_error = e
            print(f"[Warn] HVG flavor {flv!r} failed: {e}")
    raise RuntimeError(f"All HVG flavors failed. Last error: {last_error}")


def build_hvg_pca_embedding_for_pseudo(
    pseudo_h5ad: Path,
    variant: str,
    counts_layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    n_hvgs: int = 2000,
    n_pcs: int = 50,
    hvg_flavor: str = "seurat_v3",
    write_h5ad: bool = True,
    overwrite: bool = False,
) -> Dict:
    print("=" * 100)
    print(f"[HVG-PCA] Variant: {variant}")
    print(f"[HVG-PCA] Input:   {pseudo_h5ad}")
    print("=" * 100)

    variant_dir = HVG_EMB_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    emb_npy = variant_dir / "X_pca_hvg.npy"
    obs_csv = variant_dir / "obs_names.csv"
    meta_json = variant_dir / "hvg_pca_metadata.json"
    out_h5ad = HVG_H5AD_DIR / f"{variant}_with_hvg_pca.h5ad"

    if emb_npy.exists() and meta_json.exists() and not overwrite:
        print(f"[HVG-PCA] Existing output found. Skipping: {emb_npy}")
        return {
            "method": "hvg_pca",
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "embedding_npy": str(emb_npy),
            "obs_csv": str(obs_csv),
            "metadata_json": str(meta_json),
            "h5ad_with_embedding": str(out_h5ad) if out_h5ad.exists() else "",
            "status": "skipped_existing",
        }

    adata = sc.read_h5ad(pseudo_h5ad)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = copy_selected_matrix_to_x(adata, counts_layer=counts_layer)

    valid_batch_key = choose_valid_batch_key(adata, batch_key)

    print(f"[HVG-PCA] Original shape: {adata.shape}")
    print(f"[HVG-PCA] Matrix source: {'adata.X' if counts_layer is None else counts_layer}")
    print(f"[HVG-PCA] Batch key used: {valid_batch_key}")

    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr()

    used_flavor = _run_highly_variable_genes_with_fallback(
        adata=adata,
        n_top_genes=n_hvgs,
        flavor=hvg_flavor,
        batch_key=valid_batch_key,
    )

    n_hvgs_found = int(np.sum(adata.var["highly_variable"].values))
    if n_hvgs_found == 0:
        raise RuntimeError("No HVGs were selected.")

    adata = adata[:, adata.var["highly_variable"].values].copy()
    print(f"[HVG-PCA] HVG flavor used: {used_flavor}")
    print(f"[HVG-PCA] Shape after HVG subset: {adata.shape}")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=SCALE_MAX_VALUE, zero_center=SCALE_ZERO_CENTER)

    n_pcs_use = min(n_pcs, adata.n_vars - 1, adata.n_obs - 1)
    if n_pcs_use < 2:
        raise RuntimeError(
            f"Too few dimensions for PCA: n_obs={adata.n_obs}, n_vars={adata.n_vars}"
        )

    sc.tl.pca(adata, n_comps=n_pcs_use, svd_solver="arpack")
    emb = np.asarray(adata.obsm["X_pca"], dtype=np.float32)
    adata.obsm[HVG_OBSM_KEY] = emb

    np.save(emb_npy, emb)
    pd.DataFrame({"obs_name": adata.obs_names.astype(str)}).to_csv(obs_csv, index=False)

    metadata = {
        "method": "hvg_pca",
        "variant": variant,
        "pseudo_h5ad": str(pseudo_h5ad),
        "counts_layer": counts_layer,
        "batch_key_requested": batch_key,
        "batch_key_used": valid_batch_key,
        "n_obs": int(adata.n_obs),
        "n_vars_after_hvg": int(adata.n_vars),
        "n_hvgs_requested": int(n_hvgs),
        "n_hvgs_found": int(n_hvgs_found),
        "hvg_flavor_used": used_flavor,
        "n_pcs": int(n_pcs_use),
        "obsm_key": HVG_OBSM_KEY,
        "embedding_shape": list(emb.shape),
    }
    write_json(meta_json, metadata)

    h5ad_path_str = ""
    if write_h5ad:
        # Keep the HVG-subset h5ad. This is enough for embedding inspection and avoids copying full matrix twice.
        adata.write_h5ad(out_h5ad)
        h5ad_path_str = str(out_h5ad)
        print(f"[HVG-PCA] Saved h5ad with embedding: {out_h5ad}")

    print(f"[HVG-PCA] Saved embedding: {emb_npy}")
    print(f"[HVG-PCA] Embedding shape: {emb.shape}")

    row = {
        "method": "hvg_pca",
        "variant": variant,
        "pseudo_h5ad": str(pseudo_h5ad),
        "embedding_npy": str(emb_npy),
        "obs_csv": str(obs_csv),
        "metadata_json": str(meta_json),
        "h5ad_with_embedding": h5ad_path_str,
        "status": "ok",
        "n_obs": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
    }

    del adata, emb
    gc.collect()
    return row

# ============================================================
# scVI embedding
# ============================================================

def prepare_adata_for_scvi(adata: ad.AnnData, counts_layer: Optional[str] = None) -> ad.AnnData:
    adata = adata.copy()
    X = get_matrix_from_adata(adata, counts_layer=counts_layer)
    if sp.issparse(X):
        adata.X = X.copy().tocsr()
    else:
        adata.X = np.asarray(X, dtype=np.float32)
    return adata


def train_scvi_for_pseudo(
    pseudo_h5ad: Path,
    variant: str,
    counts_layer: Optional[str] = None,
    batch_key: Optional[str] = None,
    write_h5ad: bool = True,
    overwrite: bool = False,
) -> Dict:
    if not SCVI_AVAILABLE:
        raise ImportError(f"scVI is not available in this environment: {SCVI_IMPORT_ERROR}")

    print("=" * 100)
    print(f"[scVI] Variant: {variant}")
    print(f"[scVI] Input:   {pseudo_h5ad}")
    print("=" * 100)

    variant_dir = SCVI_EMB_DIR / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    emb_npy = variant_dir / "X_scVI.npy"
    obs_csv = variant_dir / "obs_names.csv"
    meta_json = variant_dir / "scvi_metadata.json"
    out_h5ad = SCVI_H5AD_DIR / f"{variant}_with_scvi.h5ad"
    model_out = SCVI_MODEL_DIR / f"{variant}_scvi_model"

    if emb_npy.exists() and meta_json.exists() and not overwrite:
        print(f"[scVI] Existing output found. Skipping: {emb_npy}")
        return {
            "method": "scvi",
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "embedding_npy": str(emb_npy),
            "obs_csv": str(obs_csv),
            "metadata_json": str(meta_json),
            "h5ad_with_embedding": str(out_h5ad) if out_h5ad.exists() else "",
            "model_dir": str(model_out) if model_out.exists() else "",
            "status": "skipped_existing",
        }

    set_seed(SEED)

    adata = sc.read_h5ad(pseudo_h5ad)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata = prepare_adata_for_scvi(adata, counts_layer=counts_layer)

    valid_batch_key = choose_valid_batch_key(adata, batch_key)

    print(f"[scVI] Shape: {adata.shape}")
    print(f"[scVI] Matrix source: {'adata.X' if counts_layer is None else counts_layer}")
    print(f"[scVI] Batch key used: {valid_batch_key}")

    if valid_batch_key is None:
        scvi.model.SCVI.setup_anndata(adata)
    else:
        scvi.model.SCVI.setup_anndata(adata, batch_key=valid_batch_key)

    model = scvi.model.SCVI(
        adata,
        n_latent=SCVI_N_LATENT,
        n_hidden=SCVI_N_HIDDEN,
        n_layers=SCVI_N_LAYERS,
        gene_likelihood=SCVI_GENE_LIKELIHOOD,
    )

    if SCVI_USE_GPU and torch.cuda.is_available():
        print("[scVI] Training with GPU.")
        try:
            model.train(
                max_epochs=SCVI_TRAIN_MAX_EPOCHS,
                batch_size=SCVI_TRAIN_BATCH_SIZE,
                accelerator="gpu",
                devices=1,
            )
        except TypeError:
            model.train(
                max_epochs=SCVI_TRAIN_MAX_EPOCHS,
                batch_size=SCVI_TRAIN_BATCH_SIZE,
                use_gpu=True,
            )
    else:
        print("[scVI] Training with CPU.")
        try:
            model.train(
                max_epochs=SCVI_TRAIN_MAX_EPOCHS,
                batch_size=SCVI_TRAIN_BATCH_SIZE,
                accelerator="cpu",
            )
        except TypeError:
            model.train(
                max_epochs=SCVI_TRAIN_MAX_EPOCHS,
                batch_size=SCVI_TRAIN_BATCH_SIZE,
                use_gpu=False,
            )

    latent = model.get_latent_representation(adata=adata)
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape[0] != adata.n_obs:
        raise ValueError(f"Embedding cell number mismatch: {latent.shape[0]} vs {adata.n_obs}")

    adata.obsm[SCVI_OBSM_KEY] = latent

    model.save(model_out, overwrite=True)
    np.save(emb_npy, latent)
    pd.DataFrame({"obs_name": adata.obs_names.astype(str)}).to_csv(obs_csv, index=False)

    metadata = {
        "method": "scvi",
        "variant": variant,
        "pseudo_h5ad": str(pseudo_h5ad),
        "counts_layer": counts_layer,
        "batch_key_requested": batch_key,
        "batch_key_used": valid_batch_key,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_latent": int(SCVI_N_LATENT),
        "n_hidden": int(SCVI_N_HIDDEN),
        "n_layers": int(SCVI_N_LAYERS),
        "gene_likelihood": SCVI_GENE_LIKELIHOOD,
        "train_max_epochs": int(SCVI_TRAIN_MAX_EPOCHS),
        "train_batch_size": int(SCVI_TRAIN_BATCH_SIZE),
        "obsm_key": SCVI_OBSM_KEY,
        "embedding_shape": list(latent.shape),
        "model_dir": str(model_out),
    }
    write_json(meta_json, metadata)

    h5ad_path_str = ""
    if write_h5ad:
        adata.write_h5ad(out_h5ad)
        h5ad_path_str = str(out_h5ad)
        print(f"[scVI] Saved h5ad with embedding: {out_h5ad}")

    print(f"[scVI] Saved model:     {model_out}")
    print(f"[scVI] Saved embedding: {emb_npy}")
    print(f"[scVI] Embedding shape: {latent.shape}")

    row = {
        "method": "scvi",
        "variant": variant,
        "pseudo_h5ad": str(pseudo_h5ad),
        "embedding_npy": str(emb_npy),
        "obs_csv": str(obs_csv),
        "metadata_json": str(meta_json),
        "h5ad_with_embedding": h5ad_path_str,
        "model_dir": str(model_out),
        "status": "ok",
        "n_obs": int(latent.shape[0]),
        "embedding_dim": int(latent.shape[1]),
    }

    del model, adata, latent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row

# ============================================================
# Run selected pseudo-control variants
# ============================================================

hvg_manifest = MANIFEST_DIR / "hvg_pca_embedding_manifest.csv"
scvi_manifest = MANIFEST_DIR / "scvi_embedding_manifest.csv"
combined_manifest = MANIFEST_DIR / "hvg_scvi_embedding_manifest.csv"

all_rows = []

for _, r in pseudo_df.iterrows():
    variant = r["variant"]
    pseudo_h5ad = Path(r["pseudo_h5ad"])

    if RUN_HVG_PCA:
        try:
            row = build_hvg_pca_embedding_for_pseudo(
                pseudo_h5ad=pseudo_h5ad,
                variant=variant,
                counts_layer=COUNTS_LAYER,
                batch_key=BATCH_KEY,
                n_hvgs=N_HVGS,
                n_pcs=N_PCS,
                hvg_flavor=HVG_FLAVOR,
                write_h5ad=WRITE_H5AD_WITH_EMB,
                overwrite=OVERWRITE,
            )
        except Exception as e:
            row = {
                "method": "hvg_pca",
                "variant": variant,
                "pseudo_h5ad": str(pseudo_h5ad),
                "status": "failed",
                "error": repr(e),
            }
            print(f"[HVG-PCA][ERROR] {variant}: {e}")
        append_manifest_row(hvg_manifest, row)
        all_rows.append(row)

    if RUN_SCVI:
        try:
            row = train_scvi_for_pseudo(
                pseudo_h5ad=pseudo_h5ad,
                variant=variant,
                counts_layer=COUNTS_LAYER,
                batch_key=BATCH_KEY,
                write_h5ad=WRITE_H5AD_WITH_EMB,
                overwrite=OVERWRITE,
            )
        except Exception as e:
            row = {
                "method": "scvi",
                "variant": variant,
                "pseudo_h5ad": str(pseudo_h5ad),
                "status": "failed",
                "error": repr(e),
            }
            print(f"[scVI][ERROR] {variant}: {e}")
        append_manifest_row(scvi_manifest, row)
        all_rows.append(row)

if all_rows:
    pd.DataFrame(all_rows).to_csv(combined_manifest, index=False)

print("Done.")
print("HVG manifest:", hvg_manifest)
print("scVI manifest:", scvi_manifest)
print("Combined manifest:", combined_manifest)

# ============================================================
# Inspect manifests
# ============================================================

for p in [hvg_manifest, scvi_manifest, combined_manifest]:
    if p.exists():
        print("\n", p)
