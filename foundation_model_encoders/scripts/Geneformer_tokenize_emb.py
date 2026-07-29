"""
Chunked Geneformer tokenization and embedding extraction for pseudo-control h5ad files.

Why this version exists:
- The official TranscriptomeTokenizer can fail on very large pseudo-control h5ad files
  with: "Value ... too large to fit in C integer type" during HuggingFace/Arrow dataset creation.
- This happens when one tokenized dataset contains too many total token IDs, usually because
  hundreds of thousands of cells each have thousands of expressed genes.
- This script still uses the official TranscriptomeTokenizer, but splits each prepared h5ad
  into cell chunks and tokenizes each chunk separately.

Core Geneformer logic follows the provided script:
- TranscriptomeTokenizer
- BertForMaskedLM.from_pretrained(..., output_hidden_states=True)
- geneformer.perturber_utils.pad_tensor_list
- geneformer.perturber_utils.gen_attention_mask
- sort tokenized dataset by length
- extract hidden_states[EMB_LAYER], default -2
- mean-pool non-padding gene tokens
"""

import os
import sys
import gc
import json
import pickle
import shutil
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Union

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import torch
from datasets import load_from_disk
from transformers import BertForMaskedLM
from tqdm.auto import tqdm

from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

GENEFORMER_ROOT = str(env_path("PPFM_GENEFORMER_REPO", required=True))
MODEL_PATH = str(env_path("PPFM_GENEFORMER_MODEL", required=True))
TOKEN_DICT_PATH = str(env_path("PPFM_GENEFORMER_TOKEN_DICT", Path(GENEFORMER_ROOT) / "geneformer" / "token_dictionary.pkl"))

# Root containing pseudo-control strategy folders.
PSEUDO_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

# Dedicated output folder for the chunked run.
OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_ROOT / "_geneformer_embeddings_chunked")

PREPARED_DIR = OUT_ROOT / "prepared_h5ad"
PREPARED_CHUNK_DIR = OUT_ROOT / "prepared_h5ad_chunks"
TOKENIZED_CHUNK_DIR = OUT_ROOT / "tokenized_chunks"
TOKENIZER_TMP_DIR = OUT_ROOT / "tmp_tokenizer_input"
EMB_DIR = OUT_ROOT / "embeddings"
EMB_CHUNK_DIR = OUT_ROOT / "embedding_chunks"
UPDATED_H5AD_DIR = OUT_ROOT / "h5ad_with_embs"
MANIFEST_DIR = OUT_ROOT / "manifests"

for d in [
    PREPARED_DIR,
    PREPARED_CHUNK_DIR,
    TOKENIZED_CHUNK_DIR,
    TOKENIZER_TMP_DIR,
    EMB_DIR,
    EMB_CHUNK_DIR,
    UPDATED_H5AD_DIR,
    MANIFEST_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# IMPORT GENEFORMER MODULES
# ------------------------------------------------------------
prepend_repo(GENEFORMER_ROOT)

from geneformer.tokenizer import TranscriptomeTokenizer
from geneformer import perturber_utils as pu

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PSEUDO_H5AD_NAME = "pseudo_control_aligned_to_perturbed.h5ad"

# Use slash-style paths or slug-style names. Use [] to process all discovered variants.
VARIANTS = env_list("PPFM_VARIANTS", [])

# Use None for full data. Useful for debug runs.
SUBSET_N_CELLS = env_optional_int("PPFM_SUBSET_N_CELLS", None)

COUNTS_SOURCE = "X"
ENSEMBL_COL = None
ENSEMBL_COL_CANDIDATES = [
    "ensembl_id",
    "ensemblid",
    "gene_id",
    "gene_ids",
    "gene_symbol",
    "feature_id",
    "features",
    "index",
]

BATCH_KEY = None
LABEL_KEY = "perturbation_label"
EXTRA_OBS_ATTRS = [
    "condition",
    "perturbation",
    "perturbation_label",
    "perturbation_tokens",
    "nperts",
    "gene",
    "guide_id",
    "source_obs_name",
    "adata_order",
    "prepared_order",
    "chunk_id",
]

# Official Geneformer tokenizer settings.
TOKENIZE_NPROC = env_int("PPFM_TOKENIZE_NPROC", 8)
TOKENIZE_CHUNK_SIZE = 512
USE_GENERATOR = False

# Split prepared h5ad into chunks before tokenization to avoid Arrow int32 overflow.
# If the same error appears, reduce this to 5000 or 2000.
TOKENIZE_CELL_CHUNK_SIZE = env_int("PPFM_CHUNK_CELLS", 10000)

# Embedding extraction settings.
EMB_LAYER = -2
FORWARD_BATCH_SIZE = env_int("PPFM_BATCH_SIZE", 64)
PAD_TOKEN_ID = 0
OBSM_KEY = "X_geneformer"

# Output behavior.
OVERWRITE_PREPARED = env_bool("PPFM_OVERWRITE", False)
OVERWRITE_PREPARED_CHUNKS = env_bool("PPFM_OVERWRITE", False)
OVERWRITE_TOKENIZED_CHUNKS = env_bool("PPFM_OVERWRITE", False)
OVERWRITE_EMBEDDING_CHUNKS = env_bool("PPFM_OVERWRITE", False)
OVERWRITE_FINAL_EMBEDDINGS = env_bool("PPFM_OVERWRITE", False)
WRITE_H5AD_WITH_EMB = env_bool("PPFM_WRITE_H5AD", True)
OVERWRITE_H5AD_WITH_EMB = env_bool("PPFM_OVERWRITE", False)
CONTINUE_ON_ERROR = env_bool("PPFM_CONTINUE_ON_ERROR", True)

print(f"[Info] DEVICE = {DEVICE}")
print(f"[Info] PSEUDO_ROOT = {PSEUDO_ROOT}")
print(f"[Info] OUT_ROOT = {OUT_ROOT}")
print(f"[Info] TOKENIZE_CELL_CHUNK_SIZE = {TOKENIZE_CELL_CHUNK_SIZE}")

paths_to_check = {
    "GENEFORMER_ROOT": Path(GENEFORMER_ROOT),
    "MODEL_PATH": Path(MODEL_PATH),
    "TOKEN_DICT_PATH": Path(TOKEN_DICT_PATH),
    "PSEUDO_ROOT": PSEUDO_ROOT,
    "OUT_ROOT": OUT_ROOT,
}
for name, path in paths_to_check.items():
    print(f"{name}: {path} | exists={path.exists()}")

# ============================================================
def variant_to_slug(variant: Union[str, Path]) -> str:
    variant = str(variant).strip().strip("/")
    return variant.replace("/", "__")


def infer_variant_from_path(h5ad_path: Path, root: Path = PSEUDO_ROOT) -> str:
    rel_parent = h5ad_path.parent.relative_to(root)
    return str(rel_parent)


def matches_variant(variant: str, filters: List[str]) -> bool:
    if not filters:
        return True
    slug = variant_to_slug(variant)
    normalized = {str(v).strip().strip("/") for v in filters}
    normalized_slugs = {variant_to_slug(v) for v in normalized}
    return variant in normalized or slug in normalized_slugs


def discover_pseudo_control_h5ads(
    root: Path = PSEUDO_ROOT,
    filename: str = PSEUDO_H5AD_NAME,
    variants: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    for path in sorted(root.rglob(filename)):
        rel_parts = path.relative_to(root).parts
        if any(part.startswith("_") for part in rel_parts):
            continue
        variant = infer_variant_from_path(path, root=root)
        if not matches_variant(variant, variants or []):
            continue
        rows.append({"variant": variant, "variant_slug": variant_to_slug(variant), "h5ad": str(path)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No {filename} files found under {root} for variants={variants}")
    return df.sort_values(["variant"]).reset_index(drop=True)


pseudo_df = discover_pseudo_control_h5ads(PSEUDO_ROOT, PSEUDO_H5AD_NAME, VARIANTS)
print(f"[Discover] Found {len(pseudo_df)} pseudo-control h5ad file(s).")

# ============================================================
def load_token_dict(token_dict_path: str):
    with open(token_dict_path, "rb") as f:
        return pickle.load(f)


TOKEN_DICT = load_token_dict(TOKEN_DICT_PATH)
VALID_ENSG = set([k for k in TOKEN_DICT.keys() if k not in {"<pad>", "<mask>"}])
print(f"[Info] Loaded Geneformer token dictionary with {len(VALID_ENSG)} valid genes.")


def get_counts_matrix(adata: ad.AnnData, counts_source: str):
    if counts_source == "X":
        return adata.X
    if counts_source not in adata.layers:
        raise KeyError(f"Layer '{counts_source}' not found in adata.layers.")
    return adata.layers[counts_source]


def compute_n_counts(X):
    if sp.issparse(X):
        return np.asarray(X.sum(axis=1)).ravel()
    return np.asarray(X.sum(axis=1)).ravel()


def clean_ensembl_ids(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True)


def get_ensembl_source_series(
    adata: ad.AnnData,
    ensembl_col: Optional[str] = None,
    candidates: Optional[List[str]] = None,
) -> tuple[str, pd.Series]:
    candidates = candidates or ENSEMBL_COL_CANDIDATES
    if ensembl_col is not None:
        if ensembl_col == "index":
            return "index", pd.Series(adata.var_names.astype(str), index=adata.var_names)
        if ensembl_col not in adata.var.columns:
            raise KeyError(f"Ensembl source column '{ensembl_col}' not found in adata.var.")
        return ensembl_col, adata.var[ensembl_col]

    for col in candidates:
        if col == "index":
            s = pd.Series(adata.var_names.astype(str), index=adata.var_names)
        elif col in adata.var.columns:
            s = adata.var[col]
        else:
            continue
        cleaned = clean_ensembl_ids(s)
        n_match = int(cleaned.isin(VALID_ENSG).sum())
        if n_match > 0:
            return col, s

    raise ValueError(
        "Could not auto-detect an Ensembl ID column with overlap to Geneformer vocabulary. "
        f"Available adata.var columns: {list(adata.var.columns)}. Set ENSEMBL_COL manually."
    )


def existing_obs_attrs(adata: ad.AnnData, batch_key: Optional[str], label_key: Optional[str]) -> Dict[str, str]:
    attrs = []
    for key in [batch_key, label_key] + EXTRA_OBS_ATTRS:
        if key is not None and key not in attrs:
            attrs.append(key)
    return {key: key for key in attrs if key in adata.obs.columns}


def prepare_geneformer_input(
    in_h5ad: Path,
    out_h5ad: Path,
    ensembl_col: Optional[str],
    counts_source: str,
    batch_key: Optional[str],
    label_key: Optional[str],
):
    print(f"\n[Prepare] Reading: {in_h5ad}")
    adata = ad.read_h5ad(in_h5ad)

    if SUBSET_N_CELLS is not None:
        adata = adata[:SUBSET_N_CELLS].copy()
        print(f"[Prepare] Debug subset: first {adata.n_obs} cells")

    if "adata_order" not in adata.obs.columns:
        adata.obs["adata_order"] = np.arange(adata.n_obs, dtype=np.int64)
    if "source_obs_name" not in adata.obs.columns:
        adata.obs["source_obs_name"] = adata.obs_names.astype(str)

    if batch_key is not None and batch_key not in adata.obs.columns:
        print(f"[Prepare][Warn] Batch key '{batch_key}' not found in adata.obs. It will not be tokenized.")
    if label_key is not None and label_key not in adata.obs.columns:
        print(f"[Prepare][Warn] Label key '{label_key}' not found in adata.obs. It will not be tokenized.")

    resolved_ensembl_col, source_series = get_ensembl_source_series(
        adata,
        ensembl_col=ensembl_col,
        candidates=ENSEMBL_COL_CANDIDATES,
    )

    X = get_counts_matrix(adata, counts_source)
    obs = adata.obs.copy()
    var = adata.var.copy()
    var["ensembl_id"] = clean_ensembl_ids(source_series).values

    keep_gene_mask = var["ensembl_id"].isin(VALID_ENSG).values
    n_before = adata.n_vars
    n_keep = int(keep_gene_mask.sum())
    if n_keep == 0:
        raise ValueError(f"No genes matched Geneformer token dictionary for file: {in_h5ad}")

    print(f"[Prepare] Ensembl source column: {resolved_ensembl_col}")
    print(f"[Prepare] Genes before filtering: {n_before}")
    print(f"[Prepare] Genes matched Geneformer vocab: {n_keep}")

    if sp.issparse(X):
        X = X[:, keep_gene_mask].tocsr()
    else:
        X = X[:, keep_gene_mask]
    var = var.loc[keep_gene_mask].copy()

    n_counts = compute_n_counts(X)
    obs["n_counts"] = n_counts
    keep_cell_mask = n_counts > 0
    n_cells_before = X.shape[0]
    n_cells_keep = int(keep_cell_mask.sum())
    print(f"[Prepare] Cells before filtering: {n_cells_before}")
    print(f"[Prepare] Cells with nonzero counts after filtering: {n_cells_keep}")

    if sp.issparse(X):
        X = X[keep_cell_mask].tocsr()
    else:
        X = X[keep_cell_mask]
    obs = obs.loc[keep_cell_mask].copy()

    # Important: do NOT add filter_pass. It can trigger backed AnnData fancy-indexing errors.
    if "filter_pass" in obs.columns:
        obs = obs.drop(columns=["filter_pass"])

    obs["prepared_order"] = np.arange(obs.shape[0], dtype=np.int64)

    gf_adata = ad.AnnData(X=X, obs=obs, var=var)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    gf_adata.write_h5ad(out_h5ad)
    print(f"[Prepare] Saved Geneformer-ready h5ad to: {out_h5ad}")
    print(f"[Prepare] Final shape: {gf_adata.shape}")

    summary = {
        "input_h5ad": str(in_h5ad),
        "prepared_h5ad": str(out_h5ad),
        "ensembl_col": resolved_ensembl_col,
        "counts_source": counts_source,
        "n_genes_before": int(n_before),
        "n_genes_matched": int(n_keep),
        "n_cells_before": int(n_cells_before),
        "n_cells_keep": int(n_cells_keep),
    }
    del adata, gf_adata, X, obs, var
    gc.collect()
    return summary


def split_prepared_h5ad_into_chunks(
    prepared_h5ad: Path,
    chunk_dir: Path,
    chunk_cells: int,
    overwrite: bool = False,
) -> List[Dict]:
    """Split one Geneformer-ready h5ad into row chunks before official tokenization."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_manifest = chunk_dir / "chunk_manifest.csv"

    if chunk_manifest.exists() and not overwrite:
        df = pd.read_csv(chunk_manifest)
        if not df.empty and all(Path(p).exists() for p in df["chunk_h5ad"]):
            print(f"[Chunk] Reusing {len(df)} prepared chunks from: {chunk_manifest}")
            return df.to_dict("records")

    if overwrite and chunk_dir.exists():
        for item in chunk_dir.glob("chunk_*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    print(f"\n[Chunk] Reading prepared h5ad for splitting: {prepared_h5ad}")
    adata = ad.read_h5ad(prepared_h5ad)
    rows = []
    n = adata.n_obs
    n_chunks = int(np.ceil(n / chunk_cells))
    print(f"[Chunk] Splitting {n} cells into {n_chunks} chunk(s), chunk_cells={chunk_cells}")

    for chunk_id, start in enumerate(tqdm(range(0, n, chunk_cells), desc="Writing h5ad chunks")):
        end = min(start + chunk_cells, n)
        chunk_name = f"chunk_{chunk_id:05d}"
        this_dir = chunk_dir / chunk_name
        this_dir.mkdir(parents=True, exist_ok=True)
        chunk_h5ad = this_dir / f"{chunk_name}.h5ad"

        chunk = adata[start:end].copy()
        chunk.obs["chunk_id"] = chunk_name
        chunk.write_h5ad(chunk_h5ad)

        rows.append({
            "chunk_id": chunk_name,
            "chunk_index": int(chunk_id),
            "start": int(start),
            "end": int(end),
            "n_cells": int(end - start),
            "chunk_h5ad": str(chunk_h5ad),
        })
        del chunk
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(chunk_manifest, index=False)
    print(f"[Chunk] Saved chunk manifest: {chunk_manifest}")
    del adata
    gc.collect()
    return rows


def tokenize_geneformer_h5ad_chunk(
    chunk_h5ad: Path,
    tokenized_out_dir: Path,
    tokenizer_tmp_dir: Path,
    output_prefix: str,
    batch_key: Optional[str],
    label_key: Optional[str],
):
    tmp_input_dir = tokenizer_tmp_dir / output_prefix
    tmp_input_dir.mkdir(parents=True, exist_ok=True)

    tmp_h5ad = tmp_input_dir / f"{output_prefix}.h5ad"
    if tmp_h5ad.exists() or tmp_h5ad.is_symlink():
        tmp_h5ad.unlink()
    os.symlink(chunk_h5ad, tmp_h5ad)

    prepared = ad.read_h5ad(chunk_h5ad, backed="r")
    custom_attr = existing_obs_attrs(prepared, batch_key=batch_key, label_key=label_key)
    prepared.file.close()

    print(f"\n[Tokenize] Chunk: {output_prefix}")
    print(f"[Tokenize] custom_attr_name_dict = {custom_attr}")

    tk = TranscriptomeTokenizer(
        custom_attr_name_dict=custom_attr,
        nproc=TOKENIZE_NPROC,
        chunk_size=TOKENIZE_CHUNK_SIZE,
        token_dictionary_file=TOKEN_DICT_PATH,
    )
    tokenized_out_dir.mkdir(parents=True, exist_ok=True)

    tk.tokenize_data(
        data_directory=tmp_input_dir,
        output_directory=tokenized_out_dir,
        output_prefix=output_prefix,
        file_format="h5ad",
        use_generator=USE_GENERATOR,
    )

    tokenized_path = (tokenized_out_dir / output_prefix).with_suffix(".dataset")
    print(f"[Tokenize] Saved tokenized chunk dataset to: {tokenized_path}")
    return tokenized_path


def load_model(model_path: str):
    model = BertForMaskedLM.from_pretrained(
        model_path,
        output_hidden_states=True,
        output_attentions=False,
    )
    model.eval()
    model.to(DEVICE)
    return model


def mean_pool_nonpadding(hidden_states, lengths):
    pooled = []
    for i in range(hidden_states.size(0)):
        pooled.append(hidden_states[i, : lengths[i], :].mean(dim=0))
    return torch.stack(pooled, dim=0)


def extract_cell_embeddings_from_dataset(
    model,
    tokenized_dataset_path: Path,
    out_npy: Path,
    out_obs_csv: Path,
    emb_layer: int = -2,
    forward_batch_size: int = 64,
    pad_token_id: int = 0,
):
    print(f"\n[Embed] Loading tokenized dataset: {tokenized_dataset_path}")
    ds = load_from_disk(str(tokenized_dataset_path))
    ds = ds.sort("length")

    n = len(ds)
    print(f"[Embed] Number of cells: {n}")
    print(f"[Embed] Columns: {ds.column_names}")

    model_input_size = pu.get_model_input_size(model)
    all_embs = []
    obs_cols = [c for c in ds.column_names if c not in ["input_ids", "length"]]
    obs_df = pd.DataFrame({c: ds[c] for c in obs_cols}) if len(obs_cols) > 0 else pd.DataFrame(index=np.arange(n))

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    out_obs_csv.parent.mkdir(parents=True, exist_ok=True)

    for start in tqdm(range(0, n, forward_batch_size), desc=f"Embedding {tokenized_dataset_path.name}"):
        end = min(start + forward_batch_size, n)
        minibatch = ds.select(range(start, end))

        lengths = torch.tensor(minibatch["length"], device=DEVICE)
        max_len = int(max(minibatch["length"]))
        minibatch.set_format(type="torch")
        input_ids = minibatch["input_ids"]

        padded = pu.pad_tensor_list(input_ids, max_len, pad_token_id, model_input_size)
        attention_mask = pu.gen_attention_mask(minibatch)

        with torch.no_grad():
            outputs = model(input_ids=padded.to(DEVICE), attention_mask=attention_mask)

        hidden = outputs.hidden_states[emb_layer]
        pooled = mean_pool_nonpadding(hidden, lengths).cpu().numpy()
        all_embs.append(pooled)

        del minibatch, input_ids, padded, attention_mask, outputs, hidden, pooled, lengths
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    embs = np.concatenate(all_embs, axis=0)

    if "prepared_order" in obs_df.columns:
        obs_df["_sorted_embedding_row"] = np.arange(obs_df.shape[0], dtype=np.int64)
        order = np.argsort(obs_df["prepared_order"].astype(int).values)
        embs = embs[order]
        obs_df = obs_df.iloc[order].reset_index(drop=True)

    np.save(out_npy, embs)
    obs_df.to_csv(out_obs_csv, index=False)
    print(f"[Embed] Saved chunk embeddings: {out_npy}")
    print(f"[Embed] Saved chunk metadata:   {out_obs_csv}")
    print(f"[Embed] Chunk embedding shape: {embs.shape}")
    return embs, obs_df


def assemble_embedding_chunks(
    prepared_h5ad: Path,
    chunk_embedding_rows: List[Dict],
    final_npy: Path,
    final_obs_csv: Path,
    overwrite: bool = False,
):
    if final_npy.exists() and final_obs_csv.exists() and not overwrite:
        arr = np.load(final_npy, mmap_mode="r")
        shape = tuple(arr.shape)
        del arr
        print(f"[Assemble] Reusing final embeddings: {final_npy} shape={shape}")
        return shape

    if len(chunk_embedding_rows) == 0:
        raise ValueError("No chunk embedding rows available for assembly.")

    prepared = ad.read_h5ad(prepared_h5ad, backed="r")
    n_total = prepared.n_obs
    prepared.file.close()

    first = np.load(chunk_embedding_rows[0]["chunk_embedding_npy"], mmap_mode="r")
    emb_dim = int(first.shape[1])
    del first

    final_npy.parent.mkdir(parents=True, exist_ok=True)
    final_obs_csv.parent.mkdir(parents=True, exist_ok=True)

    # Use .dat temporary memmap then save as normal .npy for easy downstream use.
    memmap_path = final_npy.with_suffix(".float32.memmap.dat")
    mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n_total, emb_dim))

    obs_parts = []
    for row in tqdm(chunk_embedding_rows, desc="Assembling embedding chunks"):
        embs = np.load(row["chunk_embedding_npy"], mmap_mode="r")
        obs = pd.read_csv(row["chunk_obs_csv"])
        if "prepared_order" not in obs.columns:
            raise KeyError(f"prepared_order missing from chunk obs csv: {row['chunk_obs_csv']}")
        positions = obs["prepared_order"].astype(int).values
        mm[positions, :] = np.asarray(embs, dtype="float32")
        obs_parts.append(obs)
        del embs, obs
        gc.collect()

    mm.flush()
    final_arr = np.asarray(mm)
    np.save(final_npy, final_arr)
    del mm, final_arr
    if memmap_path.exists():
        memmap_path.unlink()

    final_obs = pd.concat(obs_parts, axis=0, ignore_index=True)
    if "prepared_order" in final_obs.columns:
        final_obs = final_obs.sort_values("prepared_order").reset_index(drop=True)
    final_obs.to_csv(final_obs_csv, index=False)

    shape = (n_total, emb_dim)
    print(f"[Assemble] Saved final embeddings: {final_npy}")
    print(f"[Assemble] Saved final metadata:   {final_obs_csv}")
    print(f"[Assemble] Final embedding shape: {shape}")
    return shape


def write_embeddings_back_to_h5ad(prepared_h5ad: Path, emb_npy: Path, out_h5ad: Path, obsm_key: str = "X_geneformer"):
    adata = ad.read_h5ad(prepared_h5ad)
    embs = np.load(emb_npy)
    if adata.n_obs != embs.shape[0]:
        raise ValueError(f"Cell number mismatch: prepared_h5ad has {adata.n_obs} cells, but embeddings have {embs.shape}")
    adata.obsm[obsm_key] = embs
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_h5ad)
    print(f"[Write] Saved h5ad with Geneformer embeddings: {out_h5ad}")
    del adata, embs
    gc.collect()


def run_one_dataset(name: str, cfg: Dict):
    print("=" * 80)
    print(f"Running pseudo-control variant: {name}")
    print("=" * 80)

    prepared_h5ad = PREPARED_DIR / f"{name}_geneformer_ready.h5ad"
    variant_prepared_chunk_dir = PREPARED_CHUNK_DIR / name
    variant_tokenized_chunk_dir = TOKENIZED_CHUNK_DIR / name
    variant_tokenizer_tmp_dir = TOKENIZER_TMP_DIR / name
    variant_emb_chunk_dir = EMB_CHUNK_DIR / name
    variant_emb_dir = EMB_DIR / name

    final_emb_npy = variant_emb_dir / f"{name}_geneformer.npy"
    final_obs_csv = variant_emb_dir / f"{name}_obs.csv"
    out_h5ad = UPDATED_H5AD_DIR / f"{name}_with_geneformer.h5ad"
    metadata_json = variant_emb_dir / f"{name}_geneformer_metadata.json"
    variant_emb_dir.mkdir(parents=True, exist_ok=True)

    prep_summary = None
    if prepared_h5ad.exists() and not OVERWRITE_PREPARED:
        print(f"[Prepare] Reusing existing prepared h5ad: {prepared_h5ad}")
    else:
        prep_summary = prepare_geneformer_input(
            in_h5ad=cfg["h5ad"],
            out_h5ad=prepared_h5ad,
            ensembl_col=cfg["ensembl_col"],
            counts_source=cfg["counts_source"],
            batch_key=cfg["batch_key"],
            label_key=cfg["label_key"],
        )

    chunk_rows = split_prepared_h5ad_into_chunks(
        prepared_h5ad=prepared_h5ad,
        chunk_dir=variant_prepared_chunk_dir,
        chunk_cells=TOKENIZE_CELL_CHUNK_SIZE,
        overwrite=OVERWRITE_PREPARED_CHUNKS,
    )

    token_rows = []
    for row in chunk_rows:
        chunk_id = row["chunk_id"]
        output_prefix = f"{name}__{chunk_id}"
        tokenized_path = (variant_tokenized_chunk_dir / output_prefix).with_suffix(".dataset")
        if tokenized_path.exists() and not OVERWRITE_TOKENIZED_CHUNKS:
            print(f"[Tokenize] Reusing tokenized chunk: {tokenized_path}")
        else:
            tokenized_path = tokenize_geneformer_h5ad_chunk(
                chunk_h5ad=Path(row["chunk_h5ad"]),
                tokenized_out_dir=variant_tokenized_chunk_dir,
                tokenizer_tmp_dir=variant_tokenizer_tmp_dir,
                output_prefix=output_prefix,
                batch_key=cfg["batch_key"],
                label_key=cfg["label_key"],
            )
        token_rows.append({**row, "tokenized_dataset": str(tokenized_path), "output_prefix": output_prefix})

    model = None
    chunk_embedding_rows = []
    try:
        for row in token_rows:
            chunk_id = row["chunk_id"]
            chunk_emb_npy = variant_emb_chunk_dir / chunk_id / f"{name}__{chunk_id}_geneformer.npy"
            chunk_obs_csv = variant_emb_chunk_dir / chunk_id / f"{name}__{chunk_id}_obs.csv"
            if chunk_emb_npy.exists() and chunk_obs_csv.exists() and not OVERWRITE_EMBEDDING_CHUNKS:
                print(f"[Embed] Reusing embedding chunk: {chunk_emb_npy}")
            else:
                if model is None:
                    model = load_model(MODEL_PATH)
                embs, obs_df = extract_cell_embeddings_from_dataset(
                    model=model,
                    tokenized_dataset_path=Path(row["tokenized_dataset"]),
                    out_npy=chunk_emb_npy,
                    out_obs_csv=chunk_obs_csv,
                    emb_layer=EMB_LAYER,
                    forward_batch_size=FORWARD_BATCH_SIZE,
                    pad_token_id=PAD_TOKEN_ID,
                )
                del embs, obs_df
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            chunk_embedding_rows.append({**row, "chunk_embedding_npy": str(chunk_emb_npy), "chunk_obs_csv": str(chunk_obs_csv)})
    finally:
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    emb_shape = assemble_embedding_chunks(
        prepared_h5ad=prepared_h5ad,
        chunk_embedding_rows=chunk_embedding_rows,
        final_npy=final_emb_npy,
        final_obs_csv=final_obs_csv,
        overwrite=OVERWRITE_FINAL_EMBEDDINGS,
    )

    if WRITE_H5AD_WITH_EMB:
        if out_h5ad.exists() and not OVERWRITE_H5AD_WITH_EMB:
            print(f"[Write] Reusing existing h5ad with embeddings: {out_h5ad}")
        else:
            write_embeddings_back_to_h5ad(prepared_h5ad, final_emb_npy, out_h5ad, obsm_key=OBSM_KEY)

    chunk_manifest_path = MANIFEST_DIR / f"{name}_chunk_manifest.csv"
    pd.DataFrame(chunk_embedding_rows).to_csv(chunk_manifest_path, index=False)

    metadata = {
        "variant_slug": name,
        "variant": cfg["variant"],
        "input_h5ad": str(cfg["h5ad"]),
        "prepared_h5ad": str(prepared_h5ad),
        "n_chunks": len(chunk_rows),
        "tokenize_cell_chunk_size": TOKENIZE_CELL_CHUNK_SIZE,
        "final_embedding_npy": str(final_emb_npy),
        "final_obs_csv": str(final_obs_csv),
        "h5ad_with_embedding": str(out_h5ad) if WRITE_H5AD_WITH_EMB else None,
        "chunk_manifest": str(chunk_manifest_path),
        "obsm_key": OBSM_KEY,
        "embedding_shape": list(emb_shape),
        "model_path": MODEL_PATH,
        "token_dict_path": TOKEN_DICT_PATH,
        "emb_layer": EMB_LAYER,
        "forward_batch_size": FORWARD_BATCH_SIZE,
        "prep_summary": prep_summary,
    }
    with open(metadata_json, "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def build_pseudo_dataset_configs(pseudo_df: pd.DataFrame) -> Dict[str, Dict]:
    datasets = {}
    for _, row in pseudo_df.iterrows():
        name = row["variant_slug"]
        datasets[name] = {
            "variant": row["variant"],
            "h5ad": Path(row["h5ad"]),
            "ensembl_col": ENSEMBL_COL,
            "counts_source": COUNTS_SOURCE,
            "batch_key": BATCH_KEY,
            "label_key": LABEL_KEY,
        }
    return datasets


DATASETS = build_pseudo_dataset_configs(pseudo_df)
print(f"[Config] Built DATASETS with {len(DATASETS)} variant(s).")
for k, v in DATASETS.items():
    print(k, "->", v["h5ad"])


def main():
    results = []
    for name, cfg in DATASETS.items():
        try:
            result = run_one_dataset(name, cfg)
            result["status"] = "done"
            result["error"] = ""
            results.append(result)
        except Exception as e:
            print(f"[Error] Failed variant {name}: {e}")
            results.append({
                "variant_slug": name,
                "variant": cfg.get("variant", ""),
                "input_h5ad": str(cfg.get("h5ad", "")),
                "status": "failed",
                "error": repr(e),
            })
            if not CONTINUE_ON_ERROR:
                raise
    manifest = pd.DataFrame(results)
    manifest_path = MANIFEST_DIR / "geneformer_chunked_embedding_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[Manifest] Saved: {manifest_path}")
    return manifest

if __name__ == "__main__":
    manifest = main()
