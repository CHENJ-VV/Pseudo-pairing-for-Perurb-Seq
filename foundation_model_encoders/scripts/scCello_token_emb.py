from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

# ============================================================
# USER CONFIG
# ============================================================
import os
import sys
import gc
import json
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ----------------------------
# scCello paths
# ----------------------------
SCCELLO_ROOT = str(env_path("PPFM_SCCELLO_REPO", required=True))
TOKEN_DICT_PATH = str(env_path("PPFM_SCCELLO_TOKEN_DICT", Path(SCCELLO_ROOT) / "data" / "token_vocabulary" / "token_dictionary.pkl"))
PRETRAINED_CKPT = env_str("PPFM_SCCELLO_CHECKPOINT", "katarinayuan/scCello-zeroshot")

# ----------------------------
# Pseudo-control data paths
# ----------------------------
DATASET_ID = env_str("PPFM_DATASET_ID", "dataset")
PERTURBED_GROUP = env_str("PPFM_PERTURBED_GROUP", "single")
PSEUDO_SEARCH_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

VARIANTS = env_list("PPFM_VARIANTS", [])
MAX_VARIANTS = env_optional_int("PPFM_MAX_VARIANTS", None)
SUBSET_N_CELLS = env_optional_int("PPFM_SUBSET_N_CELLS", None)

# ----------------------------
# Input/tokenization options
# ----------------------------
COUNTS_SOURCE = "X"
GENE_COL = None  # None = auto-detect best token-vocabulary column
TRUNCATE_LENGTH = 2048
TOKEN_CHUNK_CELLS = env_int("PPFM_CHUNK_CELLS", 4096)
TARGET_SUM = 10000.0
PAD_TOKEN_ID = 0
METADATA_COLUMNS_TO_COPY = [
    "perturbation_label", "condition", "perturbation", "gene", "batch",
    "source_obs_name", "adata_order",
]

# The optimized version avoids HuggingFace Dataset.map/save_to_disk by default.
# It stores direct pickle token chunks and extracts embeddings from them.
OVERWRITE_PREPARED_H5AD = env_bool("PPFM_OVERWRITE", False)
OVERWRITE_TOKEN_CHUNKS = env_bool("PPFM_OVERWRITE", False)

# ----------------------------
# Embedding options
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FORWARD_BATCH_SIZE = env_int("PPFM_BATCH_SIZE", 64)
HIDDEN_LAYER = -2
OBSM_KEY = "X_scCello"
OVERWRITE_EMBEDDINGS = env_bool("PPFM_OVERWRITE", False)
WRITE_H5AD_WITH_EMBEDDING = env_bool("PPFM_WRITE_H5AD", False)
OVERWRITE_UPDATED_H5AD = env_bool("PPFM_OVERWRITE", False)
CONTINUE_ON_ERROR = env_bool("PPFM_CONTINUE_ON_ERROR", True)

# ----------------------------
# Output folders
# ----------------------------
OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_SEARCH_ROOT / "_sccello_embeddings_optimized_v2")
PREPARED_DIR = OUT_ROOT / "prepared_h5ad"
TOKEN_CHUNK_DIR = OUT_ROOT / "token_chunks"
EMB_DIR = OUT_ROOT / "embeddings"
UPDATED_H5AD_DIR = OUT_ROOT / "h5ad_with_sccello"
MANIFEST_DIR = OUT_ROOT / "manifests"

for d in [PREPARED_DIR, TOKEN_CHUNK_DIR, EMB_DIR, UPDATED_H5AD_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"[Config] Device: {DEVICE}")
print(f"[Config] Pseudo-control root: {PSEUDO_SEARCH_ROOT}")
print(f"[Config] Output root: {OUT_ROOT}")

# ============================================================
# Import scCello and token dictionary
# ============================================================
prepend_repo(SCCELLO_ROOT)

from sccello.src.model_prototype_contrastive import PrototypeContrastiveForMaskedLM
from sccello.src import utils

with open(TOKEN_DICT_PATH, "rb") as f:
    TOKEN_DICT = pickle.load(f)

SPECIAL_TOKENS = {"<pad>", "<mask>", "<cls>", "<eos>"}
VALID_TOKENS = set(k for k in TOKEN_DICT.keys() if k not in SPECIAL_TOKENS)
print(f"[Token dictionary] {len(VALID_TOKENS):,} usable scCello gene tokens")

# ============================================================
# Discovery helpers
# ============================================================
def sanitize_name(x: str) -> str:
    x = str(x).strip().strip("/").replace("\\", "/")
    return "__".join([p for p in x.split("/") if p])


def infer_variant_from_path(h5ad_path: Path, pseudo_root: Path) -> Tuple[str, str]:
    rel = Path(h5ad_path).resolve().relative_to(Path(pseudo_root).resolve())
    variant_path = rel.parent.as_posix()
    return variant_path, sanitize_name(variant_path)


def matches_variant(variant_path: str, variant_slug: str, filters: List[str]) -> bool:
    if not filters:
        return True
    vp = variant_path.strip("/")
    vs = variant_slug
    for f in filters:
        f = str(f).strip().strip("/")
        if not f:
            continue
        fs = sanitize_name(f)
        if f == vp or fs == vs or f in vp or fs in vs:
            return True
    return False


def discover_pseudo_control_h5ads(pseudo_root: Path, variants: List[str], max_variants: Optional[int]) -> pd.DataFrame:
    pseudo_root = Path(pseudo_root)
    if not pseudo_root.exists():
        raise FileNotFoundError(f"Pseudo-control root does not exist: {pseudo_root}")
    rows = []
    for p in sorted(pseudo_root.rglob("pseudo_control_aligned_to_perturbed.h5ad")):
        variant_path, variant_slug = infer_variant_from_path(p, pseudo_root)
        if matches_variant(variant_path, variant_slug, variants or []):
            rows.append({"variant_path": variant_path, "variant_slug": variant_slug, "pseudo_h5ad": str(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No pseudo-control h5ad files found under {pseudo_root}")
    if max_variants is not None:
        df = df.head(int(max_variants)).copy()
    df = df.reset_index(drop=True)
    print(f"[Discover] {len(df)} pseudo-control variant(s)")
    return df


pseudo_df = discover_pseudo_control_h5ads(PSEUDO_SEARCH_ROOT, VARIANTS, MAX_VARIANTS)

# ============================================================
# H5AD preparation and gene-column detection
# ============================================================
def get_counts_matrix(adata: ad.AnnData, counts_source: str):
    if counts_source == "X":
        return adata.X
    if counts_source not in adata.layers:
        raise KeyError(f"Layer '{counts_source}' not found. Available layers: {list(adata.layers.keys())}")
    return adata.layers[counts_source]


def compute_n_counts(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


def clean_gene_ids(values) -> pd.Series:
    return pd.Series(values).astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True)


def candidate_gene_columns(adata: ad.AnnData) -> List[str]:
    preferred = [
        "ensembl_id", "ensemblid", "ensembl", "gene_id", "gene_ids", "geneid",
        "gene_name", "gene_symbol", "gene_symbols", "symbol", "feature_name", "index",
    ]
    cols = set(map(str, adata.var.columns))
    out = [c for c in preferred if c == "index" or c in cols]
    for c in map(str, adata.var.columns):
        if c not in out:
            out.append(c)
    return out


def get_gene_series(adata: ad.AnnData, col: str) -> pd.Series:
    if col == "index":
        return pd.Series(adata.var_names.astype(str), index=adata.var_names)
    return pd.Series(adata.var[col].astype(str).values, index=adata.var_names)


def resolve_sccello_gene_column(adata: ad.AnnData, requested_col: Optional[str] = None):
    candidates = [requested_col] if requested_col is not None else candidate_gene_columns(adata)
    rows = []
    best = None
    for col in candidates:
        if col != "index" and col not in adata.var.columns:
            continue
        cleaned = clean_gene_ids(get_gene_series(adata, col))
        n_overlap = int(cleaned.isin(VALID_TOKENS).sum())
        row = {
            "candidate_column": col,
            "n_overlap": n_overlap,
            "frac_overlap": n_overlap / max(adata.n_vars, 1),
            "n_unique_after_cleaning": int(cleaned.nunique()),
        }
        rows.append(row)
        if best is None or n_overlap > best[0]:
            best = (n_overlap, col, cleaned)
    summary = pd.DataFrame(rows).sort_values("n_overlap", ascending=False)
    if best is None or best[0] == 0:
        raise ValueError(
            "No genes matched the scCello token dictionary. "
            f"Tried columns: {candidates}. Available var columns: {list(adata.var.columns)}"
        )
    print("[Gene column overlap]")
    return best[1], best[2], summary


def subset_rows(X, mask):
    if sp.issparse(X):
        return X[mask].tocsr()
    return np.asarray(X)[mask]


def subset_cols(X, mask):
    if sp.issparse(X):
        return X[:, mask].tocsr()
    return np.asarray(X)[:, mask]


def prepare_sccello_ready_h5ad(
    in_h5ad: Path,
    out_h5ad: Path,
    counts_source: str,
    gene_col: Optional[str],
    subset_n_cells: Optional[int] = None,
) -> Dict[str, Any]:
    if out_h5ad.exists() and not OVERWRITE_PREPARED_H5AD:
        print(f"[Prepare] Using existing prepared h5ad: {out_h5ad}")
        tmp = sc.read_h5ad(out_h5ad, backed="r")
        info = {
            "prepared_h5ad": str(out_h5ad),
            "n_cells_prepared": int(tmp.n_obs),
            "n_genes_prepared": int(tmp.n_vars),
            "chosen_gene_col": str(tmp.uns.get("chosen_gene_col", "unknown")),
            "n_genes_matched": int(tmp.uns.get("n_genes_matched", tmp.n_vars)),
        }
        tmp.file.close()
        return info

    print(f"[Prepare] Reading {in_h5ad}")
    adata = sc.read_h5ad(in_h5ad)
    if subset_n_cells is not None:
        adata = adata[:int(subset_n_cells)].copy()
        print(f"[Prepare] Debug subset: {adata.n_obs} cells")

    chosen_col, cleaned_gene_ids, overlap_df = resolve_sccello_gene_column(adata, requested_col=gene_col)
    X = get_counts_matrix(adata, counts_source)
    keep_gene_mask = cleaned_gene_ids.isin(VALID_TOKENS).values
    if int(keep_gene_mask.sum()) == 0:
        raise ValueError("No genes remained after scCello vocabulary filtering.")

    X = subset_cols(X, keep_gene_mask)
    var = adata.var.loc[keep_gene_mask].copy()
    var["sccello_gene_id"] = cleaned_gene_ids.loc[keep_gene_mask].values
    obs = adata.obs.copy()

    n_counts = compute_n_counts(X)
    keep_cell_mask = n_counts > 0
    print(f"[Prepare] Matched genes: {int(keep_gene_mask.sum())}/{adata.n_vars}")
    print(f"[Prepare] Nonzero cells after matched-gene filtering: {int(keep_cell_mask.sum())}/{adata.n_obs}")

    X = subset_rows(X, keep_cell_mask)
    obs = obs.loc[keep_cell_mask].copy()
    obs["cell_counts"] = n_counts[keep_cell_mask]
    obs["source_obs_name"] = obs.index.astype(str)
    obs["adata_order"] = np.arange(obs.shape[0], dtype=np.int64).astype(str)

    prepared = ad.AnnData(X=X, obs=obs, var=var)
    prepared.uns["chosen_gene_col"] = chosen_col
    prepared.uns["n_genes_matched"] = int(keep_gene_mask.sum())
    prepared.uns["counts_source"] = counts_source
    prepared.uns["tokenization_note"] = "Fast chunked scCello token builder; avoids slow HuggingFace map/save_to_disk stage."

    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(out_h5ad)
    print(f"[Prepare] Saved: {out_h5ad}; shape={prepared.shape}")

    info = {
        "prepared_h5ad": str(out_h5ad),
        "n_cells_prepared": int(prepared.n_obs),
        "n_genes_prepared": int(prepared.n_vars),
        "chosen_gene_col": chosen_col,
        "n_genes_matched": int(keep_gene_mask.sum()),
    }
    del adata, prepared, X, obs, var
    gc.collect()
    return info

# ============================================================
# Fast scCello token chunks with progress
# ============================================================
def save_pickle(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _rank_tokens_for_row(indices: np.ndarray, values: np.ndarray, token_ids: np.ndarray, max_length: int) -> List[int]:
    if len(indices) == 0:
        return []
    values = values.astype(np.float32, copy=False)
    if len(indices) > max_length:
        top = np.argpartition(-values, max_length - 1)[:max_length]
        indices = indices[top]
        values = values[top]
    order = np.argsort(-values, kind="mergesort")
    return token_ids[indices][order].astype(int).tolist()


def _tokenize_sparse_chunk(X_csr, token_ids: np.ndarray, max_length: int) -> List[List[int]]:
    out = []
    for i in range(X_csr.shape[0]):
        row = X_csr.getrow(i)
        out.append(_rank_tokens_for_row(row.indices, row.data, token_ids, max_length))
    return out


def _tokenize_dense_chunk(X_chunk, token_ids: np.ndarray, max_length: int) -> List[List[int]]:
    X_np = np.asarray(X_chunk)
    out = []
    for i in range(X_np.shape[0]):
        row = X_np[i]
        nz = np.flatnonzero(row)
        if len(nz) == 0:
            out.append([])
        else:
            out.append(_rank_tokens_for_row(nz, row[nz], token_ids, max_length))
    return out


def build_sccello_token_chunks_from_prepared_h5ad(
    prepared_h5ad: Path,
    variant_slug: str,
    token_chunk_root: Path,
    truncate_length: int = TRUNCATE_LENGTH,
    chunk_cells: int = TOKEN_CHUNK_CELLS,
) -> Dict[str, Any]:
    variant_token_dir = token_chunk_root / variant_slug
    manifest_path = variant_token_dir / "token_chunks_manifest.csv"
    metadata_path = variant_token_dir / "tokenization_metadata.json"

    if manifest_path.exists() and metadata_path.exists() and not OVERWRITE_TOKEN_CHUNKS:
        manifest = pd.read_csv(manifest_path)
        print(f"[Tokenize] Using existing token chunks: {variant_token_dir} ({len(manifest)} chunks)")
        with open(metadata_path) as f:
            return json.load(f)

    variant_token_dir.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(prepared_h5ad)
    gene_ids = adata.var["sccello_gene_id"].astype(str).tolist()
    token_ids = np.array([TOKEN_DICT[g] for g in gene_ids], dtype=np.int64)
    X = adata.X.tocsr() if sp.issparse(adata.X) else np.asarray(adata.X)

    rows = []
    n_cells = adata.n_obs
    start_time = time.time()
    pbar = tqdm(total=n_cells, desc=f"Tokenizing scCello {variant_slug}", unit="cell")

    for chunk_id, start in enumerate(range(0, n_cells, chunk_cells)):
        end = min(start + chunk_cells, n_cells)
        X_chunk = X[start:end]
        if sp.issparse(X_chunk):
            input_ids = _tokenize_sparse_chunk(X_chunk.tocsr(), token_ids, max_length=truncate_length)
        else:
            input_ids = _tokenize_dense_chunk(X_chunk, token_ids, max_length=truncate_length)

        lengths = [min(len(x), truncate_length) for x in input_ids]
        input_ids = [x[:truncate_length] for x in input_ids]
        obs_chunk = adata.obs.iloc[start:end].copy()

        payload = {
            "input_ids": input_ids,
            "length": lengths,
            "cell_counts": obs_chunk["cell_counts"].astype(float).tolist(),
            "source_obs_name": obs_chunk.get("source_obs_name", pd.Series(obs_chunk.index.astype(str))).astype(str).tolist(),
            "adata_order": obs_chunk.get("adata_order", pd.Series(np.arange(start, end).astype(str))).astype(str).tolist(),
        }
        # scCello reference metadata names; placeholders avoid later metadata-key assumptions.
        if "assay_batch" in obs_chunk.columns:
            payload["assay_batch"] = obs_chunk["assay_batch"].astype(str).tolist()
        elif "batch" in obs_chunk.columns:
            payload["assay_batch"] = obs_chunk["batch"].astype(str).tolist()
        else:
            payload["assay_batch"] = ["pseudo_control"] * (end - start)

        if "cell_type" in obs_chunk.columns:
            payload["cell_type"] = obs_chunk["cell_type"].astype(str).tolist()
        elif "perturbation_label" in obs_chunk.columns:
            payload["cell_type"] = obs_chunk["perturbation_label"].astype(str).tolist()
        else:
            payload["cell_type"] = ["pseudo_control"] * (end - start)

        for col in METADATA_COLUMNS_TO_COPY:
            if col in obs_chunk.columns and col not in payload:
                payload[col] = obs_chunk[col].astype(str).tolist()

        chunk_path = variant_token_dir / f"chunk_{chunk_id:05d}.pkl"
        save_pickle(payload, chunk_path)
        rows.append({
            "chunk_id": chunk_id,
            "chunk_path": str(chunk_path),
            "start": start,
            "end": end,
            "n_cells": end - start,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "max_length": int(np.max(lengths)) if lengths else 0,
        })
        pbar.update(end - start)
        pbar.set_postfix({"chunk": chunk_id, "mean_len": f"{np.mean(lengths):.0f}" if lengths else "0"})
        del X_chunk, input_ids, lengths, obs_chunk, payload
        gc.collect()
    pbar.close()

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, index=False)
    meta = {
        "variant_slug": variant_slug,
        "prepared_h5ad": str(prepared_h5ad),
        "token_chunk_dir": str(variant_token_dir),
        "manifest_path": str(manifest_path),
        "n_cells_tokenized": int(n_cells),
        "n_chunks": int(len(manifest)),
        "truncate_length": int(truncate_length),
        "chunk_cells": int(chunk_cells),
        "elapsed_seconds": float(time.time() - start_time),
        "tokenization_backend": "fast_ranked_chunked_no_hf_map",
    }
    with open(metadata_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Tokenize] Saved {len(manifest)} chunks to: {variant_token_dir}")

    del adata, X
    gc.collect()
    return meta

# ============================================================
# scCello model and chunked embedding extraction
# ============================================================
def load_sccello_model():
    print(f"[Model] Loading scCello checkpoint: {PRETRAINED_CKPT}")
    model = PrototypeContrastiveForMaskedLM.from_pretrained(PRETRAINED_CKPT)
    model.config.output_hidden_states = True
    model.config.return_dict = True
    model.eval()
    model.to(DEVICE)
    return model


def pad_batch_input_ids(batch_input_ids: List[List[int]], pad_token_id: int = PAD_TOKEN_ID):
    max_len = max([len(x) for x in batch_input_ids] + [1])
    padded = torch.full((len(batch_input_ids), max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)
    for i, ids in enumerate(batch_input_ids):
        ids = ids or [pad_token_id]
        L = len(ids)
        padded[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
    return padded, attn


def mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.device).float().unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def write_npy_from_chunks(chunk_npy_paths: List[Path], out_npy: Path) -> Tuple[int, int]:
    shapes = [np.load(p, mmap_mode="r").shape for p in chunk_npy_paths]
    n_total = int(sum(s[0] for s in shapes))
    dim = int(shapes[0][1])
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    mmap = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32, shape=(n_total, dim))
    cursor = 0
    for p, s in tqdm(list(zip(chunk_npy_paths, shapes)), desc=f"Concatenating {out_npy.name}"):
        arr = np.load(p, mmap_mode="r")
        mmap[cursor:cursor + s[0], :] = arr
        cursor += s[0]
    del mmap
    return n_total, dim


def extract_sccello_embeddings_from_token_chunks(
    model,
    token_meta: Dict[str, Any],
    variant_slug: str,
    out_root: Path,
    hidden_layer: int = HIDDEN_LAYER,
    forward_batch_size: int = FORWARD_BATCH_SIZE,
) -> Dict[str, Any]:
    emb_variant_dir = out_root / variant_slug
    emb_variant_dir.mkdir(parents=True, exist_ok=True)
    out_npy = emb_variant_dir / "X_scCello.npy"
    out_obs_csv = emb_variant_dir / "obs_metadata.csv"
    chunk_emb_dir = emb_variant_dir / "embedding_chunks"
    chunk_emb_dir.mkdir(parents=True, exist_ok=True)

    if out_npy.exists() and out_obs_csv.exists() and not OVERWRITE_EMBEDDINGS:
        arr = np.load(out_npy, mmap_mode="r")
        print(f"[Embed] Existing embedding kept: {out_npy}, shape={arr.shape}")
        return {"embedding_npy": str(out_npy), "obs_csv": str(out_obs_csv), "n_cells": int(arr.shape[0]), "embedding_dim": int(arr.shape[1])}

    token_manifest = pd.read_csv(token_meta["manifest_path"])
    all_obs_parts = []
    chunk_npy_paths = []
    start_time = time.time()

    for _, chunk_row in tqdm(token_manifest.iterrows(), total=len(token_manifest), desc="scCello token chunks"):
        payload = load_pickle(Path(chunk_row["chunk_path"]))
        input_ids = payload["input_ids"]
        n = len(input_ids)
        chunk_embs = []

        for start in range(0, n, forward_batch_size):
            end = min(start + forward_batch_size, n)
            padded, attn = pad_batch_input_ids(input_ids[start:end], pad_token_id=PAD_TOKEN_ID)
            padded = padded.to(DEVICE)
            attn = attn.to(DEVICE)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=padded,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )

            batch_embs = None
            layerwise_cls = getattr(outputs, "logits", None)
            if layerwise_cls is not None and hasattr(layerwise_cls, "dim") and layerwise_cls.dim() == 3:
                if abs(hidden_layer) > layerwise_cls.shape[1]:
                    raise ValueError(f"hidden_layer={hidden_layer} out of range for logits layers={layerwise_cls.shape[1]}")
                batch_embs = layerwise_cls[:, hidden_layer, :]
            if batch_embs is None:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise ValueError("scCello forward returned no usable logits or hidden_states.")
                hidden = hidden_states[hidden_layer]
                batch_embs = mean_pool_hidden(hidden, attn) if hidden.dim() == 3 else hidden

            chunk_embs.append(batch_embs.detach().cpu().numpy().astype(np.float32))
            del padded, attn, outputs, batch_embs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        chunk_embs = np.concatenate(chunk_embs, axis=0)
        chunk_npy = chunk_emb_dir / f"emb_{int(chunk_row['chunk_id']):05d}.npy"
        np.save(chunk_npy, chunk_embs)
        chunk_npy_paths.append(chunk_npy)

        meta_cols = [c for c in payload.keys() if c != "input_ids"]
        obs_part = pd.DataFrame({c: payload[c] for c in meta_cols if len(payload[c]) == n})
        all_obs_parts.append(obs_part)
        del payload, input_ids, chunk_embs, obs_part
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n_cells, dim = write_npy_from_chunks(chunk_npy_paths, out_npy)
    obs_df = pd.concat(all_obs_parts, axis=0, ignore_index=True)
    obs_df.to_csv(out_obs_csv, index=False)
    print(f"[Embed] Saved {out_npy}, shape=({n_cells}, {dim})")
    return {"embedding_npy": str(out_npy), "obs_csv": str(out_obs_csv), "n_cells": int(n_cells), "embedding_dim": int(dim), "elapsed_seconds": float(time.time() - start_time)}

# ============================================================
# H5AD writing and manifest helpers
# ============================================================
def append_manifest(row: Dict[str, Any], manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if manifest_path.exists():
        df.to_csv(manifest_path, mode="a", header=False, index=False)
    else:
        df.to_csv(manifest_path, index=False)


def write_embeddings_back_to_h5ad(prepared_h5ad: Path, emb_npy: Path, out_h5ad: Path, obsm_key: str = OBSM_KEY):
    if out_h5ad.exists() and not OVERWRITE_UPDATED_H5AD:
        print(f"[Write] Existing h5ad kept: {out_h5ad}")
        return
    adata = sc.read_h5ad(prepared_h5ad)
    embs = np.load(emb_npy)
    if embs.shape[0] != adata.n_obs:
        raise ValueError(f"Cell mismatch: embedding has {embs.shape[0]} rows, h5ad has {adata.n_obs}")
    adata.obsm[obsm_key] = embs
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad)
    print(f"[Write] Saved: {out_h5ad}")
    del adata, embs
    gc.collect()

# ============================================================
# Prepare and tokenize variants
# ============================================================
token_manifest_path = MANIFEST_DIR / "sccello_tokenization_manifest.csv"
token_rows = []

for _, row in tqdm(pseudo_df.iterrows(), total=len(pseudo_df), desc="Prepare + tokenize variants"):
    variant_slug = row["variant_slug"]
    pseudo_h5ad = Path(row["pseudo_h5ad"])
    prepared_h5ad = PREPARED_DIR / f"{variant_slug}_sccello_ready.h5ad"
    out_h5ad = UPDATED_H5AD_DIR / f"{variant_slug}_with_sccello.h5ad"
    manifest_row = {
        "dataset_id": DATASET_ID,
        "perturbed_group": PERTURBED_GROUP,
        "variant_path": row["variant_path"],
        "variant_slug": variant_slug,
        "pseudo_h5ad": str(pseudo_h5ad),
        "prepared_h5ad": str(prepared_h5ad),
        "h5ad_with_embedding": str(out_h5ad),
        "status": "started",
        "error": "",
    }
    try:
        prep_info = prepare_sccello_ready_h5ad(
            in_h5ad=pseudo_h5ad,
            out_h5ad=prepared_h5ad,
            counts_source=COUNTS_SOURCE,
            gene_col=GENE_COL,
            subset_n_cells=SUBSET_N_CELLS,
        )
        token_meta = build_sccello_token_chunks_from_prepared_h5ad(
            prepared_h5ad=prepared_h5ad,
            variant_slug=variant_slug,
            token_chunk_root=TOKEN_CHUNK_DIR,
            truncate_length=TRUNCATE_LENGTH,
            chunk_cells=TOKEN_CHUNK_CELLS,
        )
        manifest_row.update(prep_info)
        manifest_row.update(token_meta)
        manifest_row["status"] = "tokenized"
    except Exception as e:
        manifest_row["status"] = "failed"
        manifest_row["error"] = repr(e)
        print(f"[Error] Tokenization failed for {variant_slug}: {e}")
        if not CONTINUE_ON_ERROR:
            raise
    append_manifest(manifest_row, token_manifest_path)
    token_rows.append(manifest_row)

token_df = pd.DataFrame(token_rows)
print(f"[Manifest] Tokenization manifest: {token_manifest_path}")

# ============================================================
# Extract scCello embeddings from token chunks
# ============================================================
embedding_manifest_path = MANIFEST_DIR / "sccello_embedding_manifest.csv"
embedding_rows = []

successful_tokens = token_df[token_df["status"].eq("tokenized")].copy()
if successful_tokens.empty:
    raise RuntimeError("No successfully tokenized variants available for scCello embedding extraction.")

model = load_sccello_model()
try:
    for _, row in tqdm(successful_tokens.iterrows(), total=len(successful_tokens), desc="Embedding variants"):
        variant_slug = row["variant_slug"]
        manifest_row = {
            "dataset_id": DATASET_ID,
            "perturbed_group": PERTURBED_GROUP,
            "variant_path": row["variant_path"],
            "variant_slug": variant_slug,
            "model": "scCello",
            "checkpoint": PRETRAINED_CKPT,
            "token_chunk_dir": row["token_chunk_dir"],
            "status": "started",
            "error": "",
        }
        try:
            emb_info = extract_sccello_embeddings_from_token_chunks(
                model=model,
                token_meta=row.to_dict(),
                variant_slug=variant_slug,
                out_root=EMB_DIR,
                hidden_layer=HIDDEN_LAYER,
                forward_batch_size=FORWARD_BATCH_SIZE,
            )
            manifest_row.update(emb_info)
            manifest_row["status"] = "success"
            if WRITE_H5AD_WITH_EMBEDDING:
                out_h5ad = UPDATED_H5AD_DIR / f"{variant_slug}_with_sccello.h5ad"
                write_embeddings_back_to_h5ad(Path(row["prepared_h5ad"]), Path(emb_info["embedding_npy"]), out_h5ad)
        except Exception as e:
            manifest_row["status"] = "failed"
            manifest_row["error"] = repr(e)
            print(f"[Error] Embedding failed for {variant_slug}: {e}")
            if not CONTINUE_ON_ERROR:
                raise
        append_manifest(manifest_row, embedding_manifest_path)
        embedding_rows.append(manifest_row)
finally:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

embedding_df = pd.DataFrame(embedding_rows)
print(f"[Manifest] Embedding manifest: {embedding_manifest_path}")

# ============================================================
# Output summary
# ============================================================
print("Output root:", OUT_ROOT)
print("Token chunks:", TOKEN_CHUNK_DIR)
print("Embeddings:", EMB_DIR)
print("Manifests:", MANIFEST_DIR)
print("\nRecommended first debug settings:")
print("  SUBSET_N_CELLS = 1024")
print("  VARIANTS = ['S5_SEACell_OT_sampled_average/nmc_350/topk_05/seed_000']")
print("\nAfter debug succeeds:")
print("  SUBSET_N_CELLS = None")
print("  VARIANTS = []")
