import os
import sys
import gc
import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

# ============================================================
# USER CONFIG: scimilarity + pseudo-control paths
# ============================================================

SCIMILARITY_ROOT = str(env_path("PPFM_SCIMILARITY_REPO", required=True))
SCIMILARITY_MODEL_PATH = str(env_path("PPFM_SCIMILARITY_MODEL", required=True))

DATASET_ID = env_str("PPFM_DATASET_ID", "dataset")
PERTURBED_GROUP = env_str("PPFM_PERTURBED_GROUP", "single")
PSEUDO_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

VARIANTS = env_list("PPFM_VARIANTS", [])

OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_ROOT / "_scimilarity_embeddings")
EMB_ROOT = OUT_ROOT / "embeddings"
CHUNK_ROOT = OUT_ROOT / "embedding_chunks"
H5AD_ROOT = OUT_ROOT / "h5ad_with_scimilarity"
MANIFEST_DIR = OUT_ROOT / "manifests"

for d in [EMB_ROOT, CHUNK_ROOT, H5AD_ROOT, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

USE_GPU = env_bool("PPFM_USE_GPU", True)
SEED = 0
OBSM_KEY = "X_scimilarity"
COUNTS_SOURCE = env_str("PPFM_COUNTS_SOURCE", "X")
GENE_COL_CANDIDATES = ["gene_name", "gene_symbol", "symbol", "gene_id", "feature_name", "features", "index"]

# Memory controls.
CHUNK_CELLS = env_int("PPFM_CHUNK_CELLS", 4096)
BUFFER_SIZE = 2048
SUBSET_N_CELLS = env_optional_int("PPFM_SUBSET_N_CELLS", None)
WRITE_H5AD_WITH_EMB = env_bool("PPFM_WRITE_H5AD", False)
CONSOLIDATE_NPY = True


# ============================================================
# Import scimilarity
# ============================================================

prepend_repo(SCIMILARITY_ROOT)
try:
    from scimilarity.cell_embedding import CellEmbedding  # type: ignore
except ImportError:
    from src.scimilarity.cell_embedding import CellEmbedding  # type: ignore


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)


def safe_name_from_relpath(rel: Path) -> str:
    return "__".join(rel.parts)


def discover_pseudo_h5ads(pseudo_root: Path, variants: Optional[List[str]] = None) -> pd.DataFrame:
    pseudo_root = Path(pseudo_root)
    files = sorted(pseudo_root.rglob("pseudo_control_aligned_to_perturbed.h5ad"))
    rows = []
    variant_filters = [str(v).strip("/") for v in (variants or []) if str(v).strip()]
    for f in files:
        rel_parent = f.parent.relative_to(pseudo_root)
        rel_str = rel_parent.as_posix()
        if variant_filters:
            keep = any(v in rel_str or v.replace("__", "/") in rel_str for v in variant_filters)
            if not keep:
                continue
        rows.append({
            "variant_relpath": rel_str,
            "variant": safe_name_from_relpath(rel_parent),
            "h5ad": str(f),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No pseudo_control_aligned_to_perturbed.h5ad files found under {pseudo_root}")
    return df.sort_values("variant").reset_index(drop=True)


def get_counts_matrix(adata: ad.AnnData, counts_source: str):
    if counts_source == "X":
        return adata.X
    if counts_source not in adata.layers:
        raise KeyError(f"Layer '{counts_source}' not found in adata.layers")
    return adata.layers[counts_source]


def detect_gene_col(var: pd.DataFrame, candidates: List[str], model_gene_order: List[str]) -> Tuple[str, Dict[str, int]]:
    model_set = set(map(str, model_gene_order))
    scores = {}
    for col in candidates:
        if col == "index":
            vals = var.index.astype(str)
        elif col in var.columns:
            vals = var[col].astype(str)
        else:
            continue
        vals = vals.str.strip()
        scores[col] = int(vals.isin(model_set).sum())
    if not scores:
        raise KeyError(f"None of gene column candidates are available. var columns: {list(var.columns)}")
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        raise ValueError(f"No genes matched scimilarity gene_order. Candidate scores: {scores}")
    return best, scores


def get_gene_names(var: pd.DataFrame, gene_col: str) -> np.ndarray:
    if gene_col == "index":
        return var.index.astype(str).to_numpy()
    return var[gene_col].astype(str).str.strip().to_numpy()


def build_projection_to_model_genes(dataset_genes: np.ndarray, model_gene_order: List[str]) -> Tuple[sp.csr_matrix, int]:
    gene_to_model_idx = {g: i for i, g in enumerate(model_gene_order)}
    rows, cols, data = [], [], []
    seen = set()
    for src_i, gene in enumerate(map(str, dataset_genes)):
        j = gene_to_model_idx.get(gene)
        if j is not None:
            rows.append(src_i)
            cols.append(j)
            data.append(1.0)
            seen.add(j)
    proj = sp.csr_matrix((data, (rows, cols)), shape=(len(dataset_genes), len(model_gene_order)), dtype=np.float32)
    return proj, len(seen)


def lognorm_tp10k(X):
    if sp.issparse(X):
        X = X.tocsr(copy=True)
        cell_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        cell_sums[cell_sums == 0] = 1.0
        scale = 1e4 / cell_sums
        X = sp.diags(scale.astype(np.float32)) @ X
        X.data = np.log1p(X.data).astype(np.float32)
        return X.tocsr()
    X = np.asarray(X, dtype=np.float32).copy()
    cell_sums = X.sum(axis=1, keepdims=True)
    cell_sums[cell_sums == 0] = 1.0
    X = X / cell_sums * 1e4
    X = np.log1p(X)
    return X.astype(np.float32)


def slice_rows(X, start: int, end: int):
    if sp.issparse(X):
        return X[start:end]
    return X[start:end, :]


def align_chunk_to_model_order(X_chunk, projection: sp.csr_matrix):
    if sp.issparse(X_chunk):
        return (X_chunk.tocsr() @ projection).tocsr()
    # Dense pseudo-control chunks are converted only per chunk.
    X_np = np.asarray(X_chunk, dtype=np.float32)
    aligned = X_np @ projection.toarray().astype(np.float32)
    # scimilarity accepts sparse/dense; use sparse if mostly zeros.
    return sp.csr_matrix(aligned)


def consolidate_chunk_files(chunk_paths: List[Path], out_npy: Path) -> Tuple[int, int]:
    first = np.load(chunk_paths[0], mmap_mode="r")
    dim = first.shape[1]
    total = int(sum(np.load(p, mmap_mode="r").shape[0] for p in chunk_paths))
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32, shape=(total, dim))
    pos = 0
    for p in chunk_paths:
        x = np.load(p, mmap_mode="r")
        arr[pos:pos + x.shape[0]] = x
        pos += x.shape[0]
    del arr
    return total, dim


def write_h5ad_if_requested(in_h5ad: Path, emb_npy: Path, out_h5ad: Path):
    adata = sc.read_h5ad(in_h5ad)
    embs = np.load(emb_npy, mmap_mode="r")
    if adata.n_obs != embs.shape[0]:
        raise ValueError(f"Cell mismatch: h5ad={adata.n_obs}, embeddings={embs.shape[0]}")
    adata.obsm[OBSM_KEY] = np.asarray(embs, dtype=np.float32)
    adata.write_h5ad(out_h5ad)
    del adata, embs
    gc.collect()

# ============================================================
# Run one pseudo-control variant
# ============================================================

def run_one_variant_scimilarity(row: pd.Series, ce: CellEmbedding) -> Dict:
    variant = row["variant"]
    h5ad_path = Path(row["h5ad"])
    print("=" * 100)
    print(f"[scimilarity] Variant: {variant}")
    print(f"[scimilarity] H5AD:    {h5ad_path}")
    print("=" * 100)

    out_dir = EMB_ROOT / variant
    chunk_dir = CHUNK_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    obs_csv = out_dir / "obs_names.csv"
    final_npy = out_dir / "X_scimilarity.npy"
    metadata_json = out_dir / "scimilarity_embedding_metadata.json"
    out_h5ad = H5AD_ROOT / f"{variant}_with_scimilarity.h5ad"

    adata_b = sc.read_h5ad(h5ad_path, backed="r")
    n_total = adata_b.n_obs if SUBSET_N_CELLS is None else min(int(SUBSET_N_CELLS), adata_b.n_obs)
    var_df = adata_b.var.copy()
    obs_names = adata_b.obs_names[:n_total].astype(str).tolist()
    adata_b.file.close()

    gene_col, gene_scores = detect_gene_col(var_df, GENE_COL_CANDIDATES, ce.gene_order)
    dataset_genes = get_gene_names(var_df, gene_col)
    projection, matched = build_projection_to_model_genes(dataset_genes, ce.gene_order)
    print(f"[Info] Gene column selected: {gene_col}")
    print(f"[Info] Gene overlap scores: {gene_scores}")
    print(f"[Info] Matched scimilarity genes: {matched}/{len(ce.gene_order)}")

    adata = sc.read_h5ad(h5ad_path)
    X = get_counts_matrix(adata, COUNTS_SOURCE)
    if SUBSET_N_CELLS is not None:
        X = X[:n_total]

    chunk_paths: List[Path] = []
    chunk_id = 0

    for start in tqdm(range(0, n_total, CHUNK_CELLS), desc=f"{variant}: scimilarity chunks"):
        end = min(start + CHUNK_CELLS, n_total)
        X_chunk = slice_rows(X, start, end)
        X_aligned = align_chunk_to_model_order(X_chunk, projection)
        X_lognorm = lognorm_tp10k(X_aligned)

        embs = ce.get_embeddings(
            X_lognorm,
            num_cells=-1,
            buffer_size=BUFFER_SIZE,
        )
        embs = np.asarray(embs, dtype=np.float32)
        if embs.shape[0] != (end - start):
            raise ValueError(f"Embedding rows {embs.shape[0]} do not match chunk cells {end-start}")

        chunk_path = chunk_dir / f"chunk_{chunk_id:06d}.npy"
        np.save(chunk_path, embs)
        chunk_paths.append(chunk_path)
        chunk_id += 1

        del X_chunk, X_aligned, X_lognorm, embs
        gc.collect()

    del adata, X
    gc.collect()

    pd.DataFrame({"obs_name": obs_names}).to_csv(obs_csv, index=False)

    final_shape = None
    if CONSOLIDATE_NPY:
        n_rows, dim = consolidate_chunk_files(chunk_paths, final_npy)
        final_shape = [n_rows, dim]
    else:
        final_npy = None
        first = np.load(chunk_paths[0], mmap_mode="r")
        final_shape = [n_total, int(first.shape[1])]

    if WRITE_H5AD_WITH_EMB:
        if final_npy is None:
            raise ValueError("WRITE_H5AD_WITH_EMB requires CONSOLIDATE_NPY=True")
        write_h5ad_if_requested(h5ad_path, final_npy, out_h5ad)

    meta = {
        "variant": variant,
        "variant_relpath": row["variant_relpath"],
        "h5ad": str(h5ad_path),
        "method": "scimilarity",
        "obsm_key": OBSM_KEY,
        "gene_col": gene_col,
        "gene_scores": gene_scores,
        "matched_model_genes": int(matched),
        "n_model_genes": int(len(ce.gene_order)),
        "counts_source": COUNTS_SOURCE,
        "n_cells_input": int(n_total),
        "n_cells_embedded": int(n_total),
        "chunk_cells": int(CHUNK_CELLS),
        "buffer_size": int(BUFFER_SIZE),
        "chunk_files": [str(p) for p in chunk_paths],
        "embedding_npy": str(final_npy) if final_npy is not None else None,
        "embedding_shape": final_shape,
        "h5ad_with_embedding": str(out_h5ad) if WRITE_H5AD_WITH_EMB else None,
    }
    with open(metadata_json, "w") as f:
        json.dump(meta, f, indent=2)
    return meta

# ============================================================
# Execute
# ============================================================

set_seed(SEED)
print("Loading scimilarity model...")
ce = CellEmbedding(model_path=SCIMILARITY_MODEL_PATH, use_gpu=USE_GPU)
print(f"[Info] scimilarity model path: {SCIMILARITY_MODEL_PATH}")
print(f"[Info] Number of genes in model: {len(ce.gene_order)}")
try:
    print(f"[Info] Latent dimension: {ce.latent_dim}")
except Exception:
    pass

variants_df = discover_pseudo_h5ads(PSEUDO_ROOT, VARIANTS)
print(variants_df[["variant_relpath", "variant", "h5ad"]].to_string(index=False))

manifest_rows = []
for _, row in variants_df.iterrows():
    try:
        meta = run_one_variant_scimilarity(row, ce)
        meta["status"] = "success"
    except Exception as e:
        print(f"[ERROR] {row.get('variant', 'unknown')}: {type(e).__name__}: {e}")
        meta = {
            "variant": row.get("variant", None),
            "variant_relpath": row.get("variant_relpath", None),
            "h5ad": row.get("h5ad", None),
            "method": "scimilarity",
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
    manifest_rows.append(meta)
    pd.DataFrame(manifest_rows).to_csv(MANIFEST_DIR / "scimilarity_embedding_manifest.csv", index=False)
    gc.collect()

manifest = pd.DataFrame(manifest_rows)
manifest.to_csv(MANIFEST_DIR / "scimilarity_embedding_manifest.csv", index=False)
manifest
