# ============================================================
# Imports
# ============================================================
import os
import gc
import sys
import json
import random
import warnings
from pathlib import Path
from typing import Dict, Optional, List, Union, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

import torch
from anndata import AnnData
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

# ============================================================
# USER CONFIG
# ============================================================

# -------------------------
# scGPT paths
# -------------------------
SCGPT_ROOT = str(env_path("PPFM_SCGPT_REPO", required=True))
MODEL_DIR = str(env_path("PPFM_SCGPT_MODEL_DIR", required=True))
VOCAB_FILE = str(env_path("PPFM_SCGPT_VOCAB_FILE", Path(SCGPT_ROOT) / "scgpt" / "tokenizer" / "default_gene_vocab.json"))

# -------------------------
# Pseudo-control dataset paths
# -------------------------
DATASET_ID = env_str("PPFM_DATASET_ID", "dataset")
PERTURBED_GROUP = env_str("PPFM_PERTURBED_GROUP", "single")
PSEUDO_SEARCH_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

# External output folder beside pseudo-control variants.
OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_SEARCH_ROOT / "_scgpt_embeddings")
EMB_DIR = OUT_ROOT / "embeddings"
UPDATED_H5AD_DIR = OUT_ROOT / "h5ad_with_scgpt"
MANIFEST_DIR = OUT_ROOT / "manifests"

for d in [OUT_ROOT, EMB_DIR, UPDATED_H5AD_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------
# Variant selection
# -------------------------
# Empty list means process all pseudo-control variants under PSEUDO_SEARCH_ROOT.
# For first debugging, keep only one variant.
VARIANTS = env_list("PPFM_VARIANTS", [])

# -------------------------
# scGPT extraction settings
# -------------------------
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OBSM_KEY = "X_scGPT"

MAX_LENGTH = 1200
BATCH_SIZE = env_int("PPFM_BATCH_SIZE", 64)
NUM_WORKERS = env_int("PPFM_NUM_WORKERS", 0)
USE_FAST_TRANSFORMER = True
ALLOW_FAST_TRANSFORMER_FALLBACK = True

# scGPT expects gene symbols. Use None for auto-detection.
# Common choices: "index", "gene_name", "gene_symbol", "symbol", "gene_id", "feature_name".
GENE_COL = None
GENE_COL_CANDIDATES = [
    "gene_name", "gene_symbol", "symbol", "gene_id", "feature_name", "features", "index"
]

# Use adata.X if None. Otherwise, use the given layer name, e.g. "counts" or "raw_counts".
COUNTS_LAYER = None

# Output options.
WRITE_H5AD_WITH_EMBEDDINGS = env_bool("PPFM_WRITE_H5AD", True)
OVERWRITE = env_bool("PPFM_OVERWRITE", False)
SAVE_FLOAT16 = False  # set True to reduce disk size, but float32 is safer for downstream models

MANIFEST_CSV = MANIFEST_DIR / "scgpt_embedding_manifest.csv"

print(f"Device: {DEVICE}")
print(f"Pseudo-control root: {PSEUDO_SEARCH_ROOT}")
print(f"Output root: {OUT_ROOT}")

# ============================================================
# Path setup for scGPT imports
# ============================================================

prepend_repo(SCGPT_ROOT)

from scgpt.data_collator import DataCollator
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import load_pretrained

# ============================================================
# General utilities
# ============================================================

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mkdir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_safe_name(x: str) -> str:
    x = str(x).replace("/", "__")
    x = x.replace(" ", "_")
    for ch in [":", "*", "?", "\"", "<", ">", "|"]:
        x = x.replace(ch, "_")
    return x


def normalize_for_match(x: Union[str, Path]) -> str:
    return str(x).replace("\\", "/").strip("/")


def infer_variant_name(pseudo_h5ad: Union[str, Path], root: Optional[Union[str, Path]] = None) -> str:
    """
    Infer variant name from the full folder hierarchy between the perturbation group
    folder (single/dual/multi) and pseudo_control_aligned_to_perturbed.h5ad.

    Example:
        .../single/S5_SEACell_OT_sampled_average/nmc_350/topk_05/seed_000/pseudo_control_aligned_to_perturbed.h5ad
    becomes:
        S5_SEACell_OT_sampled_average__nmc_350__topk_05__seed_000
    """
    p = Path(pseudo_h5ad)
    anchors = {"single", "dual", "multi"}
    parts = list(p.parts)

    anchor_idx = None
    for i, part in enumerate(parts[:-1]):
        if part in anchors:
            anchor_idx = i

    if anchor_idx is not None:
        rel_parts = parts[anchor_idx + 1 : -1]
    elif root is not None:
        try:
            rel_parts = list(p.parent.relative_to(Path(root)).parts)
        except Exception:
            rel_parts = [p.parent.name]
    else:
        rel_parts = [p.parent.name]

    drop = {
        "pseudo_control_shards", "ot_assignments", "membership", "_seacell_memberships",
        "_scgpt_embeddings", "h5ad_with_scgpt", "embeddings", "manifests",
    }
    rel_parts = [x for x in rel_parts if x not in drop]
    return "__".join(rel_parts) if rel_parts else p.parent.name


def discover_pseudo_h5ads(
    pseudo_search_root: Union[str, Path],
    variants: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Discover pseudo_control_aligned_to_perturbed.h5ad files."""
    root = Path(pseudo_search_root)
    if not root.exists():
        raise FileNotFoundError(f"Pseudo-search root does not exist: {root}")

    files = sorted(root.rglob("pseudo_control_aligned_to_perturbed.h5ad"))
    files = [p for p in files if "_scgpt_embeddings" not in normalize_for_match(p)]

    rows = []
    for p in files:
        variant_name = infer_variant_name(p, root=root)
        try:
            rel_path = p.relative_to(root)
        except Exception:
            rel_path = p
        rows.append({
            "variant": variant_name,
            "relative_path": normalize_for_match(rel_path),
            "pseudo_h5ad": str(p),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No pseudo_control_aligned_to_perturbed.h5ad files found under: {root}")

    if variants:
        wanted = [normalize_for_match(v) for v in variants]

        def keep(row):
            candidates = [
                normalize_for_match(row["variant"]),
                normalize_for_match(row["relative_path"]),
                normalize_for_match(row["pseudo_h5ad"]),
            ]
            return any(any(w in c for c in candidates) for w in wanted)

        df = df[df.apply(keep, axis=1)].copy()
        if df.empty:
            raise FileNotFoundError(
                "No pseudo h5ad matched VARIANTS.\n"
                f"Requested: {variants}\n"
                f"Search root: {root}"
            )

    return df.sort_values("variant").reset_index(drop=True)


set_seed(SEED)
pseudo_df = discover_pseudo_h5ads(PSEUDO_SEARCH_ROOT, VARIANTS)
print(f"Discovered {len(pseudo_df)} pseudo-control h5ad file(s).")

# ============================================================
# Matrix and gene-column utilities
# ============================================================

def get_matrix_from_adata(adata: AnnData, counts_layer: Optional[str] = None):
    if counts_layer is None:
        X = adata.X
    else:
        if counts_layer not in adata.layers:
            raise KeyError(f"counts_layer '{counts_layer}' not found in adata.layers")
        X = adata.layers[counts_layer]
    return X


def clean_gene_symbols(values: Sequence) -> pd.Series:
    return pd.Series(values).astype(str).str.strip()


def get_gene_values(adata: AnnData, gene_col: str) -> pd.Series:
    if gene_col == "index":
        return clean_gene_symbols(adata.var_names)
    if gene_col not in adata.var.columns:
        raise KeyError(f"gene_col '{gene_col}' not found in adata.var.columns")
    return clean_gene_symbols(adata.var[gene_col].values)


def choose_scgpt_gene_col(
    adata: AnnData,
    vocab,
    requested_gene_col: Optional[str] = None,
    candidates: Optional[Sequence[str]] = None,
) -> Tuple[str, int, int]:
    """Choose the gene column with highest overlap with the scGPT vocabulary."""
    if candidates is None:
        candidates = GENE_COL_CANDIDATES

    cols = []
    if requested_gene_col is not None:
        cols.append(requested_gene_col)
    for c in candidates:
        if c not in cols:
            cols.append(c)

    valid_cols = []
    for c in cols:
        if c == "index" or c in adata.var.columns:
            valid_cols.append(c)

    if not valid_cols:
        raise ValueError(
            "No valid gene column candidates found. "
            f"adata.var columns are: {list(adata.var.columns)}"
        )

    results = []
    for c in valid_cols:
        vals = get_gene_values(adata, c)
        matched = int(sum(v in vocab for v in vals))
        results.append((c, matched, len(vals)))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    best_col, best_match, total = results[0]

    print("[Gene column candidates]")
    for c, m, t in results:
        print(f"  {c:20s} matched {m}/{t}")

    if best_match == 0:
        raise RuntimeError(
            "No genes matched the scGPT vocabulary. "
            "Check whether var_names/var columns contain gene symbols rather than Ensembl IDs."
        )

    if requested_gene_col is not None and best_col != requested_gene_col:
        print(
            f"[Info] Requested gene_col={requested_gene_col!r}, but {best_col!r} has higher vocab overlap. "
            f"Using {best_col!r}."
        )

    return best_col, best_match, total


def filter_adata_to_scgpt_vocab(
    adata: AnnData,
    vocab,
    gene_col: str,
    counts_layer: Optional[str] = None,
) -> Tuple[AnnData, np.ndarray, pd.Series]:
    """Return adata subset to genes in scGPT vocab, preserving sparse matrix if possible."""
    adata = adata.copy()
    X = get_matrix_from_adata(adata, counts_layer)

    if counts_layer is not None:
        adata.X = X

    gene_values = get_gene_values(adata, gene_col)
    ids = np.array([vocab[g] if g in vocab else -1 for g in gene_values], dtype=np.int64)
    keep = ids >= 0

    if int(keep.sum()) == 0:
        raise RuntimeError("No genes left after scGPT vocabulary filtering.")

    adata = adata[:, keep].copy()
    gene_values = gene_values.loc[keep].reset_index(drop=True)
    gene_ids = ids[keep]

    adata.var["scgpt_gene_col_used"] = gene_col
    adata.var["scgpt_gene_symbol"] = gene_values.values
    adata.var["id_in_vocab"] = gene_ids

    return adata, gene_ids, gene_values

# ============================================================
# scGPT model loading
# ============================================================

def load_scgpt_model(
    model_dir: Union[str, Path],
    vocab_file: Union[str, Path],
    device: Union[str, torch.device] = "cuda",
    use_fast_transformer: bool = True,
    allow_fast_transformer_fallback: bool = True,
):
    model_dir = Path(model_dir)
    vocab_file = Path(vocab_file)

    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"

    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    if not model_config_file.exists():
        raise FileNotFoundError(f"Model config file not found: {model_config_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_file}")

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    vocab.set_default_index(vocab["<pad>"])

    def build_model(use_fast: bool):
        return TransformerModel(
            ntoken=len(vocab),
            d_model=model_configs["embsize"],
            nhead=model_configs["nheads"],
            d_hid=model_configs["d_hid"],
            nlayers=model_configs["nlayers"],
            nlayers_cls=model_configs["n_layers_cls"],
            n_cls=1,
            vocab=vocab,
            dropout=model_configs["dropout"],
            pad_token=model_configs["pad_token"],
            pad_value=model_configs["pad_value"],
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            explicit_zero_prob=False,
            use_fast_transformer=use_fast,
            fast_transformer_backend="flash" if use_fast else None,
            pre_norm=False,
        )

    try:
        model = build_model(use_fast_transformer)
    except Exception as e:
        if use_fast_transformer and allow_fast_transformer_fallback:
            print(f"[Warn] Fast transformer init failed: {e}")
            print("[Warn] Retrying with use_fast_transformer=False")
            model = build_model(False)
        else:
            raise

    state_dict = torch.load(model_file, map_location=device)
    load_pretrained(model, state_dict, verbose=False)
    model.to(device)
    model.eval()

    print(f"[Model] Loaded scGPT on {device}")
    print(f"[Model] embsize={model_configs['embsize']}, nlayers={model_configs['nlayers']}")
    return model, vocab, model_configs, device


model, vocab, model_configs, device = load_scgpt_model(
    model_dir=MODEL_DIR,
    vocab_file=VOCAB_FILE,
    device=DEVICE,
    use_fast_transformer=USE_FAST_TRANSFORMER,
    allow_fast_transformer_fallback=ALLOW_FAST_TRANSFORMER_FALLBACK,
)

# ============================================================
# Sparse-friendly scGPT embedding extraction
# ============================================================

class ScGPTCellDataset(torch.utils.data.Dataset):
    """Sparse-friendly dataset returning one nonzero gene-expression sequence per cell."""
    def __init__(self, X, gene_ids: np.ndarray, vocab, pad_value: float):
        self.X = X
        self.gene_ids = np.asarray(gene_ids, dtype=np.int64)
        self.vocab = vocab
        self.pad_value = pad_value
        self.is_sparse = sp.issparse(X)
        if self.is_sparse:
            self.X = X.tocsr()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.is_sparse:
            row = self.X.getrow(idx)
            nonzero_idx = row.indices
            values = row.data.astype(np.float32, copy=False)
        else:
            row = np.asarray(self.X[idx]).ravel()
            nonzero_idx = np.nonzero(row)[0]
            values = row[nonzero_idx].astype(np.float32, copy=False)

        genes = self.gene_ids[nonzero_idx]

        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.pad_value)

        return {
            "id": idx,
            "genes": torch.from_numpy(genes).long(),
            "expressions": torch.from_numpy(values).float(),
        }


def get_batch_cell_embeddings_sparse_friendly(
    adata: AnnData,
    model,
    vocab,
    model_configs: Dict,
    gene_ids: np.ndarray,
    max_length: int = 1200,
    batch_size: int = 64,
    num_workers: int = 0,
) -> np.ndarray:
    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    dataset = ScGPTCellDataset(
        X=X,
        gene_ids=gene_ids,
        vocab=vocab,
        pad_value=model_configs["pad_value"],
    )

    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[model_configs["pad_token"]],
        pad_value=model_configs["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = next(model.parameters()).device
    cell_embeddings = np.zeros((len(dataset), model_configs["embsize"]), dtype=np.float32)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        count = 0
        for data_dict in tqdm(data_loader, desc="scGPT embedding extraction"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[model_configs["pad_token"]])

            embeddings = model._encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=None,
            )

            # CLS token embedding
            embeddings = embeddings[:, 0, :].detach().cpu().numpy()
            cell_embeddings[count : count + len(embeddings)] = embeddings
            count += len(embeddings)

            del input_gene_ids, src_key_padding_mask, embeddings, data_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # L2 normalization, matching the reference script.
    norms = np.linalg.norm(cell_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cell_embeddings = cell_embeddings / norms
    return cell_embeddings

# ============================================================
# Run one pseudo-control variant
# ============================================================

def run_one_pseudo_control_variant(row: pd.Series) -> Dict:
    variant = row["variant"]
    pseudo_h5ad = Path(row["pseudo_h5ad"])
    safe_variant = make_safe_name(variant)

    variant_emb_dir = safe_mkdir(EMB_DIR / safe_variant)
    emb_npy = variant_emb_dir / "scgpt_embeddings.npy"
    obs_csv = variant_emb_dir / "obs_names.csv"
    meta_json = variant_emb_dir / "scgpt_embedding_metadata.json"
    out_h5ad = UPDATED_H5AD_DIR / f"{safe_variant}_with_scgpt.h5ad"

    if emb_npy.exists() and obs_csv.exists() and (not WRITE_H5AD_WITH_EMBEDDINGS or out_h5ad.exists()) and not OVERWRITE:
        print(f"[Skip] Existing output found for variant: {variant}")
        return {
            "dataset_id": DATASET_ID,
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "status": "skipped_existing",
            "embedding_npy": str(emb_npy),
            "obs_csv": str(obs_csv),
            "h5ad_with_scgpt": str(out_h5ad) if WRITE_H5AD_WITH_EMBEDDINGS else "",
        }

    print("=" * 100)
    print(f"[Variant] {variant}")
    print(f"[Input]   {pseudo_h5ad}")
    print("=" * 100)

    try:
        adata = sc.read_h5ad(pseudo_h5ad)
        adata.obs["source_obs_name"] = adata.obs_names.astype(str)
        adata.obs["adata_order"] = np.arange(adata.n_obs).astype(int)

        print(f"[Info] Original shape: {adata.shape}")
        print(f"[Info] adata.var columns: {list(adata.var.columns)}")
        print(f"[Info] first var_names: {list(adata.var_names[:5])}")

        selected_gene_col, n_match, n_total = choose_scgpt_gene_col(
            adata,
            vocab=vocab,
            requested_gene_col=GENE_COL,
            candidates=GENE_COL_CANDIDATES,
        )

        adata, gene_ids, gene_values = filter_adata_to_scgpt_vocab(
            adata=adata,
            vocab=vocab,
            gene_col=selected_gene_col,
            counts_layer=COUNTS_LAYER,
        )

        print(f"[Info] Shape after scGPT vocab filtering: {adata.shape}")
        print(f"[Info] Using gene_col: {selected_gene_col}")
        print(f"[Info] Using counts layer: {COUNTS_LAYER if COUNTS_LAYER else 'adata.X'}")

        embs = get_batch_cell_embeddings_sparse_friendly(
            adata=adata,
            model=model,
            vocab=vocab,
            model_configs=model_configs,
            gene_ids=gene_ids,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        if SAVE_FLOAT16:
            embs_to_save = embs.astype(np.float16)
        else:
            embs_to_save = embs.astype(np.float32)

        np.save(emb_npy, embs_to_save)
        pd.DataFrame({
            "obs_name": adata.obs_names.astype(str),
            "source_obs_name": adata.obs["source_obs_name"].astype(str).values,
            "adata_order": adata.obs["adata_order"].values,
        }).to_csv(obs_csv, index=False)

        metadata = {
            "dataset_id": DATASET_ID,
            "perturbed_group": PERTURBED_GROUP,
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "model": "scGPT",
            "model_dir": str(MODEL_DIR),
            "vocab_file": str(VOCAB_FILE),
            "obsm_key": OBSM_KEY,
            "gene_col_used": selected_gene_col,
            "n_genes_total_before_vocab_filter": int(n_total),
            "n_genes_matched_initial": int(n_match),
            "n_genes_after_vocab_filter": int(adata.n_vars),
            "n_cells": int(adata.n_obs),
            "embedding_shape": list(embs_to_save.shape),
            "max_length": int(MAX_LENGTH),
            "batch_size": int(BATCH_SIZE),
            "counts_layer": COUNTS_LAYER,
            "save_float16": bool(SAVE_FLOAT16),
        }
        with open(meta_json, "w") as f:
            json.dump(metadata, f, indent=2)

        if WRITE_H5AD_WITH_EMBEDDINGS:
            adata.obsm[OBSM_KEY] = embs_to_save
            adata.write_h5ad(out_h5ad)
            print(f"[Save] h5ad with scGPT embeddings: {out_h5ad}")

        print(f"[Save] embeddings: {emb_npy}")
        print(f"[Save] obs csv:    {obs_csv}")
        print(f"[Info] embedding shape: {embs_to_save.shape}")

        del adata, embs, embs_to_save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "dataset_id": DATASET_ID,
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "status": "success",
            "embedding_npy": str(emb_npy),
            "obs_csv": str(obs_csv),
            "metadata_json": str(meta_json),
            "h5ad_with_scgpt": str(out_h5ad) if WRITE_H5AD_WITH_EMBEDDINGS else "",
            "gene_col_used": selected_gene_col,
            "n_cells": int(metadata["n_cells"]),
            "n_genes_after_vocab_filter": int(metadata["n_genes_after_vocab_filter"]),
            "embedding_dim": int(metadata["embedding_shape"][1]),
        }

    except Exception as e:
        print(f"[Error] Variant failed: {variant}")
        print(f"[Error] {type(e).__name__}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "dataset_id": DATASET_ID,
            "variant": variant,
            "pseudo_h5ad": str(pseudo_h5ad),
            "status": "failed",
            "error_type": type(e).__name__,
            "error_message": str(e),
        }

# ============================================================
# Main execution
# ============================================================

rows = []
for _, row in pseudo_df.iterrows():
    result = run_one_pseudo_control_variant(row)
    rows.append(result)

manifest_df = pd.DataFrame(rows)
manifest_df.to_csv(MANIFEST_CSV, index=False)
print(f"[Done] Saved manifest: {MANIFEST_CSV}")
manifest_df
