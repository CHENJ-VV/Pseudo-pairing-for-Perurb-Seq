"""MLP-only evaluation pipeline for expression and precomputed embedding inputs.

This module is designed to be placed next to the existing evaluation files:
    eval_common.py, eval_mlp.py, mlp_models.py

Design goals
------------
1. Keep the expression baseline identical to the original eval_mlp.py path by
   delegating the expression branch to eval_mlp.run_mlp_evaluation(...).
   This avoids the split/gene-selection/model mismatch that caused the previous
   custom mlp_expr branch to diverge from older results.
2. Add embedding-input forward/inverse MLP tasks for hvg/scVI/scimilarity/scGPT/scCello
   using the same split, optimizer settings, early stopping, and inverse metrics.
3. Keep outputs compact and strategy/run metadata compatible with the existing pipeline.
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .eval_common import (
    add_metadata_columns,
    as_namespace,
    ensure_dir,
    get_config,
    load_pseudo_matrix_aligned,
    load_run_manifest,
    make_safe_id,
    mean_axis0,
    metadata_from_manifest_row,
    rowwise_cosine,
    rowwise_pearson,
    safe_cosine,
    safe_pearson,
    save_json,
    select_eval_genes,
    set_seed,
    slice_to_dense,
    standardize_manifest_columns,
    summarize_array,
    summarize_numeric,
    to_csr,
)
from .eval_mlp import (
    build_fixed_split,
    build_label_encoder,
    device_from_config,
    evaluate_inverse_model,
    finalize_per_perturbation_common_delta,
    train_inverse_model,
    run_mlp_evaluation as run_original_expression_mlp_evaluation,
)
from .mlp_models import InversePerturbationMLP


# ============================================================
# Default strategy/model labels
# ============================================================

EXPRESSION_MODEL_NAME = "mlp_expr"

DEFAULT_MODELS_TO_RUN = [
    "mlp_expr",
    "geneformer",
    "hvg",
    "scvi",
    "sccello",
    "scimilarity",
    "scgpt",
]

DEFAULT_OBSM_KEYS = {
    "geneformer": ["X_geneformer", "X_Geneformer"],
    "hvg": ["X_pca_hvg", "X_hvg_pca"],
    "scvi": ["X_scVI", "X_scvi"],
    "sccello": ["X_scCello", "X_sccello"],
    "scimilarity": ["X_scimilarity"],
    "scgpt": ["X_scGPT", "X_scgpt"],
}


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    return get_config(config, key, default)


def _copy_namespace(config: Any) -> SimpleNamespace:
    if isinstance(config, SimpleNamespace):
        return copy(config)
    if isinstance(config, Mapping):
        return SimpleNamespace(**dict(config))
    return copy(as_namespace(config))


def _to_dense_float32(X) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _read_obs_names_table(path: str | Path | None) -> pd.Index | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    for col in ["obs_name", "obs_names", "cell", "cell_id", "source_obs_name", "index"]:
        if col in df.columns:
            return pd.Index(df[col].astype(str).values)
    if df.shape[1] == 1:
        return pd.Index(df.iloc[:, 0].astype(str).values)
    return None


def _variant_slug_from_pseudo_path(pseudo_h5ad: str | Path, pseudo_group_root: str | Path | None) -> str:
    pseudo_h5ad = Path(pseudo_h5ad)
    variant_dir = pseudo_h5ad.parent
    if pseudo_group_root is not None:
        root = Path(pseudo_group_root)
        try:
            rel = variant_dir.relative_to(root)
            return str(rel).strip("/").replace("/", "__")
        except Exception:
            pass
    return variant_dir.name


# ============================================================
# Embedding path resolution
# ============================================================

def _default_embedding_specs(pseudo_group_root: str | Path) -> dict[str, dict[str, Any]]:
    """Candidate embedding paths following the pasted Replogle_RPE layout."""
    root = Path(pseudo_group_root)
    return {
        "geneformer": {
            "display_name": "Geneformer",
            "npy_candidates": [
                root / "_geneformer_embeddings_chunked" / "embeddings" / "{slug}" / "X_geneformer.npy",
            ],
            "obs_candidates": [
                root / "_geneformer_embeddings_chunked" / "embeddings" / "{slug}" / "geneformer_obs.csv",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["geneformer"],
        },
        "hvg": {
            "display_name": "HVG PCA",
            "npy_candidates": [
                root / "_hvg_scvi_embeddings" / "hvg_pca" / "embeddings" / "{slug}" / "X_pca_hvg.npy",
            ],
            "obs_candidates": [
                root / "_hvg_scvi_embeddings" / "hvg_pca" / "embeddings" / "{slug}" / "obs_names.csv",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["hvg"],
        },
        "scvi": {
            "display_name": "scVI",
            "npy_candidates": [
                root / "_hvg_scvi_embeddings" / "scvi" / "embeddings" / "{slug}" / "X_scVI.npy",
            ],
            "obs_candidates": [
                root / "_hvg_scvi_embeddings" / "scvi" / "embeddings" / "{slug}" / "obs_names.csv",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["scvi"],
        },
        "sccello": {
            "display_name": "scCello",
            "npy_candidates": [
                root / "_sccello_embeddings_optimized_v2" / "embeddings" / "{slug}" / "X_scCello.npy",
                root / "_sccello_embeddings" / "embeddings" / "{slug}" / "X_scCello.npy",
            ],
            "obs_candidates": [
                root / "_sccello_embeddings_optimized_v2" / "embeddings" / "{slug}" / "obs_metadata.csv",
                root / "_sccello_embeddings" / "embeddings" / "{slug}" / "obs_metadata.csv",
                root / "_sccello_embeddings" / "embeddings" / "{slug}" / "obs_names.csv",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["sccello"],
        },
        "scimilarity": {
            "display_name": "scimilarity",
            "npy_candidates": [
                root / "_scimilarity_embeddings" / "embeddings" / "{slug}" / "X_scimilarity.npy",
                root / "_scimilarity_embeddings_optimized" / "embeddings" / "{slug}" / "X_scimilarity.npy",
            ],
            "obs_candidates": [
                root / "_scimilarity_embeddings" / "embeddings" / "{slug}" / "obs_names.csv",
                root / "_scimilarity_embeddings_optimized" / "embeddings" / "{slug}" / "obs_names.csv",
            ],
            "chunk_candidates": [
                root / "_scimilarity_embeddings" / "embedding_chunks" / "{slug}",
                root / "_scimilarity_embeddings_optimized" / "embedding_chunks" / "{slug}",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["scimilarity"],
        },
        "scgpt": {
            "display_name": "scGPT",
            "npy_candidates": [
                root / "_scgpt_embeddings" / "embeddings" / "{slug}" / "scgpt_embeddings.npy",
                root / "_scgpt_embeddings" / "embeddings" / "{slug}" / "X_scGPT.npy",
            ],
            "obs_candidates": [
                root / "_scgpt_embeddings" / "embeddings" / "{slug}" / "obs_names.csv",
            ],
            "obsm_keys": DEFAULT_OBSM_KEYS["scgpt"],
        },
    }


def _resolve_template_path(path_template: str | Path, slug: str) -> Path:
    return Path(str(path_template).format(slug=slug))


def _first_existing(candidates: Sequence[str | Path], slug: str) -> Path | None:
    for templ in candidates:
        p = _resolve_template_path(templ, slug)
        if p.exists():
            return p
    return None


def _assemble_embedding_chunks(chunk_dir: Path, out_npy: Path, overwrite: bool = False) -> Path | None:
    if out_npy.exists() and not overwrite:
        return out_npy
    if not chunk_dir.exists():
        return None
    chunks = sorted(chunk_dir.glob("*.npy"))
    if not chunks:
        return None
    shapes = [np.load(p, mmap_mode="r").shape for p in chunks]
    dim = int(shapes[0][1])
    if any(int(s[1]) != dim for s in shapes):
        raise ValueError(f"Embedding chunk dimensions differ in {chunk_dir}: {shapes[:5]}")
    n_total = int(sum(int(s[0]) for s in shapes))
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(out_npy, mode="w+", dtype="float32", shape=(n_total, dim))
    cursor = 0
    for p, shape in zip(chunks, shapes):
        arr = np.load(p, mmap_mode="r")
        mm[cursor:cursor + int(shape[0])] = arr.astype("float32", copy=False)
        cursor += int(shape[0])
    del mm
    return out_npy


def resolve_pseudo_embedding_file(model_name: str, row: pd.Series, config: Any, runtime_cache: Path) -> tuple[Path, Path | None, str, str]:
    pseudo_group_root = _cfg(config, "pseudo_group_root", None)
    if pseudo_group_root is None:
        raise ValueError("Set config.pseudo_group_root to the group-level pseudo-control folder, e.g. .../Replogle_RPE/single")
    slug = _variant_slug_from_pseudo_path(row["pseudo_control_h5ad"], pseudo_group_root)
    specs = _cfg(config, "embedding_specs", None) or _default_embedding_specs(pseudo_group_root)
    if model_name not in specs:
        raise KeyError(f"No embedding spec for model_name={model_name!r}.")
    spec = specs[model_name]
    npy_path = _first_existing(spec.get("npy_candidates", []), slug)
    source = "npy"
    if npy_path is None:
        for templ in spec.get("chunk_candidates", []):
            chunk_dir = _resolve_template_path(templ, slug)
            if chunk_dir.exists():
                out_npy = runtime_cache / "assembled_embedding_chunks" / model_name / slug / f"X_{model_name}.npy"
                npy_path = _assemble_embedding_chunks(chunk_dir, out_npy, overwrite=False)
                source = "chunks"
                if npy_path is not None:
                    break
    obs_path = _first_existing(spec.get("obs_candidates", []), slug)
    if npy_path is None or not Path(npy_path).exists():
        raise FileNotFoundError(f"Missing pseudo embedding for {model_name}/{slug}; checked spec candidates.")
    return Path(npy_path), obs_path, source, slug


def _first_group_from_config(config: Any) -> str:
    groups = _cfg(config, "perturbed_groups_to_evaluate", None)
    if isinstance(groups, (list, tuple, set)) and len(groups) > 0:
        return str(list(groups)[0])
    return str(_cfg(config, "perturbed_group", "single"))


def _candidate_true_embedding_slugs(config: Any, role: str) -> list[str]:
    """Return candidate folder names for externally saved true embeddings.

    The pseudo embedding specs use an ``embeddings/{slug}/X_*.npy`` layout.  Your
    true perturbed/control embeddings may be saved in the same external embedding
    folders with slugs such as ``true_perturbed`` or the h5ad stem.  The exact
    names can be overridden in the execution config with
    ``true_perturbed_embedding_slugs`` and ``true_control_embedding_slugs``.
    """
    dataset_id = str(_cfg(config, "dataset_id", ""))
    group = _first_group_from_config(config)
    if role == "perturbed":
        explicit = _cfg(config, "true_perturbed_embedding_slugs", None)
        h5ad_stem = Path(str(_cfg(config, "perturbed_h5ad", ""))).stem
        defaults = [
            "true_perturbed",
            "perturbed",
            "TRUE_PERTURBED",
            "true_perturbed_cells",
            h5ad_stem,
            f"{dataset_id}_{group}",
            f"{dataset_id}_{group}_processed",
        ]
    elif role == "control":
        explicit = _cfg(config, "true_control_embedding_slugs", None)
        h5ad_stem = Path(str(_cfg(config, "control_h5ad", ""))).stem
        defaults = [
            "true_control",
            "control",
            "TRUE_CONTROL",
            "true_control_cells",
            h5ad_stem,
            f"{dataset_id}_control",
            f"{dataset_id}_control_processed",
        ]
    else:
        raise ValueError(f"Unknown true embedding role={role!r}")

    out: list[str] = []
    if explicit is not None:
        if isinstance(explicit, (str, Path)):
            out.append(str(explicit))
        else:
            out.extend([str(x) for x in explicit])
    out.extend([x for x in defaults if x])

    # Preserve order while removing duplicates/empty strings.
    seen = set()
    unique: list[str] = []
    for x in out:
        x = str(x).strip().strip("/")
        if x and x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def resolve_true_embedding_file(model_name: str, role: str, config: Any, runtime_cache: Path) -> tuple[Path | None, Path | None, str, str, list[str]]:
    """Resolve externally saved true perturbed/control embedding files.

    Returns ``(npy_path, obs_path, source, slug, checked_paths)``.  ``npy_path`` is
    ``None`` when no external file is found, allowing the caller to fall back to
    AnnData ``.obsm`` if available.
    """
    pseudo_group_root = _cfg(config, "pseudo_group_root", None)
    specs = _cfg(config, "embedding_specs", None) or _default_embedding_specs(pseudo_group_root)
    if model_name not in specs:
        raise KeyError(f"No embedding spec for model_name={model_name!r}.")
    spec = specs[model_name]
    checked: list[str] = []
    for slug in _candidate_true_embedding_slugs(config, role):
        npy_path = _first_existing(spec.get("npy_candidates", []), slug)
        source = "npy"
        if npy_path is not None:
            obs_path = _first_existing(spec.get("obs_candidates", []), slug)
            return Path(npy_path), obs_path, source, slug, checked
        for templ in spec.get("npy_candidates", []):
            checked.append(str(_resolve_template_path(templ, slug)))

        for templ in spec.get("chunk_candidates", []):
            chunk_dir = _resolve_template_path(templ, slug)
            checked.append(str(chunk_dir))
            if chunk_dir.exists():
                out_npy = runtime_cache / "assembled_true_embedding_chunks" / model_name / role / slug / f"X_{model_name}.npy"
                npy_path = _assemble_embedding_chunks(chunk_dir, out_npy, overwrite=False)
                source = "chunks"
                if npy_path is not None:
                    obs_path = _first_existing(spec.get("obs_candidates", []), slug)
                    return Path(npy_path), obs_path, source, slug, checked
    return None, None, "missing", "", checked


def load_aligned_embedding_file(
    embedding_path: str | Path,
    obs_path: str | Path | None,
    target_obs_names: pd.Index,
    role: str,
) -> np.ndarray:
    X = np.load(embedding_path, mmap_mode="r")
    if X.ndim != 2:
        raise ValueError(f"Embedding file must be 2D: {embedding_path}, got shape={X.shape}")
    if X.shape[0] != len(target_obs_names):
        raise ValueError(
            f"n_obs mismatch for {role} embedding {embedding_path}: "
            f"embedding={X.shape[0]}, expected={len(target_obs_names)}"
        )
    obs_names = _read_obs_names_table(obs_path)
    if obs_names is not None and len(obs_names) == len(target_obs_names):
        if not obs_names.astype(str).equals(target_obs_names.astype(str)):
            orderer = obs_names.astype(str).get_indexer(target_obs_names.astype(str))
            if np.any(orderer < 0):
                raise ValueError(f"obs_names in {obs_path} cannot be aligned to target obs_names for {role}.")
            X = np.asarray(X[orderer, :], dtype=np.float32)
            return X
    return X


def load_aligned_pseudo_embedding(
    embedding_path: str | Path,
    obs_path: str | Path | None,
    perturbed_obs_names: pd.Index,
) -> np.ndarray:
    return load_aligned_embedding_file(
        embedding_path=embedding_path,
        obs_path=obs_path,
        target_obs_names=perturbed_obs_names,
        role="pseudo",
    )


def load_true_representation_matrix(
    h5ad_path: str | Path,
    model_name: str,
    config: Any,
    role: str,
    expected_obs_names: pd.Index,
    runtime_cache: Path,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """Load true control/perturbed representation from external .npy or .obsm.

    External files are tried first by default because the Replogle_RPE embedding
    outputs are stored outside the processed h5ad files.  Set
    ``prefer_external_true_embeddings=False`` if you want to prefer ``.obsm``.
    """
    prefer_external = bool(_cfg(config, "prefer_external_true_embeddings", True))
    pseudo_group_root = _cfg(config, "pseudo_group_root", None)
    specs = _cfg(config, "embedding_specs", None) or _default_embedding_specs(pseudo_group_root)
    keys = list(specs.get(model_name, {}).get("obsm_keys", DEFAULT_OBSM_KEYS.get(model_name, [])))
    if not keys:
        raise KeyError(f"No obsm_keys defined for representation {model_name!r}.")

    checked_external: list[str] = []
    if prefer_external:
        npy_path, obs_path, source, slug, checked_external = resolve_true_embedding_file(model_name, role, config, runtime_cache)
        if npy_path is not None:
            arr = load_aligned_embedding_file(npy_path, obs_path, expected_obs_names, role=f"true_{role}")
            return np.asarray(arr, dtype=np.float32), f"external:{slug}", {
                f"true_{role}_embedding_path": str(npy_path),
                f"true_{role}_embedding_obs_path": str(obs_path) if obs_path is not None else "",
                f"true_{role}_embedding_source": source,
                f"true_{role}_embedding_slug": slug,
            }

    adata = sc.read_h5ad(str(h5ad_path))
    for key in keys:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key], dtype=np.float32)
            if arr.shape[0] != len(expected_obs_names):
                available = list(adata.obsm.keys())
                del adata
                gc.collect()
                raise ValueError(
                    f"True {role} {model_name} .obsm shape mismatch in {h5ad_path}: "
                    f"{arr.shape[0]} rows vs expected {len(expected_obs_names)}. Available obsm={available}"
                )
            del adata
            gc.collect()
            return arr, f"obsm:{key}", {
                f"true_{role}_embedding_path": str(h5ad_path),
                f"true_{role}_embedding_source": "obsm",
                f"true_{role}_obsm_key": key,
            }
    available = list(adata.obsm.keys())
    del adata
    gc.collect()

    if not prefer_external:
        npy_path, obs_path, source, slug, checked_external = resolve_true_embedding_file(model_name, role, config, runtime_cache)
        if npy_path is not None:
            arr = load_aligned_embedding_file(npy_path, obs_path, expected_obs_names, role=f"true_{role}")
            return np.asarray(arr, dtype=np.float32), f"external:{slug}", {
                f"true_{role}_embedding_path": str(npy_path),
                f"true_{role}_embedding_obs_path": str(obs_path) if obs_path is not None else "",
                f"true_{role}_embedding_source": source,
                f"true_{role}_embedding_slug": slug,
            }

    checked_preview = checked_external[:12]
    raise KeyError(
        f"Could not find true {role} {model_name} embedding. "
        f"Tried external slugs={_candidate_true_embedding_slugs(config, role)} and h5ad obsm_keys={keys}. "
        f"Available obsm in {h5ad_path}: {available}. "
        f"First checked external paths: {checked_preview}. "
        "Either set true_perturbed_embedding_slugs/true_control_embedding_slugs to the actual external folders, "
        "or set inverse_input_space='expression'."
    )


# ============================================================
# MLP models for embedding input
# ============================================================

class ForwardInputPerturbationMLP(nn.Module):
    """Forward MLP for arbitrary input dimension and arbitrary target dimension.

    For expression input with input_dim == output_dim, this is architecturally the
    same encoder/fusion policy as ForwardPerturbationMLP. The expression branch is
    still delegated to the original eval_mlp.py for exact result matching.
    """

    def __init__(
        self,
        input_dim: int,
        n_perturbations: int,
        output_dim: int,
        pert_emb_dim: int = 256,
        hidden_dim: int = 1024,
        latent_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.pert_embedding = nn.Embedding(int(n_perturbations), int(pert_emb_dim))
        self.x_encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(latent_dim)), nn.LayerNorm(int(latent_dim)), nn.GELU(), nn.Dropout(float(dropout)),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(latent_dim) + int(pert_emb_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, x0: torch.Tensor, pert_id: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.x_encoder(x0), self.pert_embedding(pert_id)], dim=1))


# ============================================================
# Datasets/loaders for embedding branches
# ============================================================

class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


def make_embedding_forward_collate_fn(X_input, X0_target, Xt_target, pert_ids):
    def collate_fn(batch_indices):
        idx = np.asarray(batch_indices, dtype=np.int64)
        x_input = np.asarray(X_input[idx, :], dtype=np.float32)
        x0 = slice_to_dense(X0_target, idx) if sp.issparse(X0_target) else np.asarray(X0_target[idx, :], dtype=np.float32)
        xt = slice_to_dense(Xt_target, idx) if sp.issparse(Xt_target) else np.asarray(Xt_target[idx, :], dtype=np.float32)
        t = pert_ids[idx].astype(np.int64)
        delta = xt - x0
        return torch.from_numpy(x_input), torch.from_numpy(t), torch.from_numpy(delta), torch.from_numpy(xt), torch.from_numpy(x0)
    return collate_fn


def make_embedding_inverse_collate_fn(X0_target, Xt_target, pert_ids, inverse_input_mode: str, control_mean_vec=None):
    if control_mean_vec is not None:
        control_mean_vec = np.asarray(control_mean_vec, dtype=np.float32).reshape(1, -1)

    def collate_fn(batch_indices):
        idx = np.asarray(batch_indices, dtype=np.int64)
        xt = slice_to_dense(Xt_target, idx) if sp.issparse(Xt_target) else np.asarray(Xt_target[idx, :], dtype=np.float32)
        if inverse_input_mode == "strategy_delta":
            x0 = slice_to_dense(X0_target, idx) if sp.issparse(X0_target) else np.asarray(X0_target[idx, :], dtype=np.float32)
            delta = xt - x0
        elif inverse_input_mode == "common_delta":
            if control_mean_vec is None:
                raise ValueError("control_mean_vec is required for common_delta")
            delta = xt - control_mean_vec
        else:
            raise ValueError(f"Unknown inverse_input_mode={inverse_input_mode!r}")
        t = pert_ids[idx].astype(np.int64)
        return torch.from_numpy(np.asarray(delta, dtype=np.float32)), torch.from_numpy(t)
    return collate_fn


def make_embedding_loaders(indices, X_input, X0_target, Xt_target, pert_ids, config, task: str, inverse_input_mode="strategy_delta", control_mean_vec=None):
    batch_size = int(_cfg(config, "batch_size", 256))
    num_workers = int(_cfg(config, "num_workers", 0))
    train_idx, val_idx, test_idx = indices
    if task == "forward":
        collate = make_embedding_forward_collate_fn(X_input, X0_target, Xt_target, pert_ids)
    elif task == "inverse":
        collate = make_embedding_inverse_collate_fn(X0_target, Xt_target, pert_ids, inverse_input_mode, control_mean_vec)
    else:
        raise ValueError(task)
    return (
        DataLoader(IndexDataset(train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, collate_fn=collate),
        DataLoader(IndexDataset(val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate),
        DataLoader(IndexDataset(test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, collate_fn=collate),
    )


# ============================================================
# Embedding-branch training/evaluation
# ============================================================

def train_forward_model_general(model, train_loader, val_loader, run_id: str, outdir: Path, config, device: torch.device):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(_cfg(config, "learning_rate", 1e-3)), weight_decay=float(_cfg(config, "weight_decay", 1e-5)))
    best_val = np.inf
    wait = 0
    hist = []
    best_state = None
    best_path = outdir / f"{run_id}_forward_best.pt"
    save_checkpoints = bool(_cfg(config, "save_mlp_checkpoints", False))
    if save_checkpoints:
        best_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in tqdm(range(1, int(_cfg(config, "forward_epochs", 30)) + 1), desc=f"{run_id} forward"):
        model.train()
        tr = []
        for x0, t, delta_true, _xt, _x0_target in train_loader:
            x0 = x0.to(device)
            t = t.to(device)
            delta_true = delta_true.to(device)
            opt.zero_grad()
            delta_pred = model(x0, t)
            loss = F.mse_loss(delta_pred, delta_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(_cfg(config, "max_grad_norm", 5.0)))
            opt.step()
            tr.append(loss.item())
        model.eval()
        va = []
        with torch.no_grad():
            for x0, t, delta_true, _xt, _x0_target in val_loader:
                x0 = x0.to(device)
                t = t.to(device)
                delta_true = delta_true.to(device)
                va.append(F.mse_loss(model(x0, t), delta_true).item())
        row = {"epoch": epoch, "train_loss": float(np.mean(tr)), "val_loss": float(np.mean(va))}
        hist.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_checkpoints:
                torch.save(model.state_dict(), best_path)
        else:
            wait += 1
        if wait >= int(_cfg(config, "early_stop_patience", 3)):
            break
    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(outdir / f"{run_id}_forward_training_history.csv", index=False)
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    elif save_checkpoints and best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    return model, hist_df


def evaluate_forward_model_general(model, test_loader, control_mean, n_classes, id_to_label, config, device):
    model.eval()
    control_mean = np.asarray(control_mean, dtype=np.float32).reshape(1, -1)
    sums = {
        "model_sse_xt": 0.0,
        "model_sae_xt": 0.0,
        "input_sse_xt": 0.0,
        "input_sae_xt": 0.0,
        "delta_sse": 0.0,
        "delta_sae": 0.0,
    }
    n_elem = 0
    corr = {
        "model_cell_pearson_xt": [],
        "input_cell_pearson_xt": [],
        "model_cell_cosine_xt": [],
        "input_cell_cosine_xt": [],
        "strategy_delta_cell_pearson": [],
        "strategy_delta_cell_cosine": [],
        "model_common_delta_cell_pearson": [],
        "input_common_delta_cell_pearson": [],
    }
    sum_pred = np.zeros((n_classes, control_mean.shape[1]), dtype=np.float64)
    sum_true = np.zeros((n_classes, control_mean.shape[1]), dtype=np.float64)
    counts = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for x_input, t, delta_true, xt_true, x0_target in tqdm(test_loader, desc="Evaluate forward", leave=False):
            t_np = t.numpy().astype(np.int64)
            delta_pred = model(x_input.to(device), t.to(device)).detach().cpu().numpy().astype(np.float32)
            x0_np = x0_target.numpy().astype(np.float32)
            delta_true_np = delta_true.numpy().astype(np.float32)
            xt_true_np = xt_true.numpy().astype(np.float32)
            xt_pred = x0_np + delta_pred
            model_err = xt_pred - xt_true_np
            input_err = x0_np - xt_true_np
            delta_err = delta_pred - delta_true_np
            sums["model_sse_xt"] += float(np.sum(model_err ** 2))
            sums["model_sae_xt"] += float(np.sum(np.abs(model_err)))
            sums["input_sse_xt"] += float(np.sum(input_err ** 2))
            sums["input_sae_xt"] += float(np.sum(np.abs(input_err)))
            sums["delta_sse"] += float(np.sum(delta_err ** 2))
            sums["delta_sae"] += float(np.sum(np.abs(delta_err)))
            n_elem += int(np.prod(xt_true_np.shape))
            corr["model_cell_pearson_xt"].extend(rowwise_pearson(xt_pred, xt_true_np).tolist())
            corr["input_cell_pearson_xt"].extend(rowwise_pearson(x0_np, xt_true_np).tolist())
            corr["model_cell_cosine_xt"].extend(rowwise_cosine(xt_pred, xt_true_np).tolist())
            corr["input_cell_cosine_xt"].extend(rowwise_cosine(x0_np, xt_true_np).tolist())
            corr["strategy_delta_cell_pearson"].extend(rowwise_pearson(delta_pred, delta_true_np).tolist())
            corr["strategy_delta_cell_cosine"].extend(rowwise_cosine(delta_pred, delta_true_np).tolist())
            true_common = xt_true_np - control_mean
            pred_common = xt_pred - control_mean
            input_common = x0_np - control_mean
            corr["model_common_delta_cell_pearson"].extend(rowwise_pearson(pred_common, true_common).tolist())
            corr["input_common_delta_cell_pearson"].extend(rowwise_pearson(input_common, true_common).tolist())
            for cls in np.unique(t_np):
                if cls < 0:
                    continue
                mask = t_np == cls
                sum_pred[cls] += pred_common[mask].sum(axis=0)
                sum_true[cls] += true_common[mask].sum(axis=0)
                counts[cls] += int(mask.sum())
    m = {
        "model_mse_xt": sums["model_sse_xt"] / max(n_elem, 1),
        "model_mae_xt": sums["model_sae_xt"] / max(n_elem, 1),
        "input_only_mse_xt": sums["input_sse_xt"] / max(n_elem, 1),
        "input_only_mae_xt": sums["input_sae_xt"] / max(n_elem, 1),
        "strategy_specific_delta_mse": sums["delta_sse"] / max(n_elem, 1),
        "strategy_specific_delta_mae": sums["delta_sae"] / max(n_elem, 1),
    }
    m["model_gain_mse_xt"] = m["input_only_mse_xt"] - m["model_mse_xt"]
    m["model_gain_mae_xt"] = m["input_only_mae_xt"] - m["model_mae_xt"]
    m["model_gain_mse_xt_fraction"] = m["model_gain_mse_xt"] / (m["input_only_mse_xt"] + 1e-12)
    for k, vals in corr.items():
        m.update(summarize_array(vals, k))
    per = finalize_per_perturbation_common_delta(
        sum_pred,
        sum_true,
        counts,
        id_to_label,
        topk_list=_cfg(config, "topk_effect_genes", [20, 50, 100]),
    )
    for col in [c for c in per.columns if c not in {"perturbation_id", "perturbation_label", "n_test_cells"} and pd.api.types.is_numeric_dtype(per[c])]:
        arr = per[col].astype(float).values
        m[f"per_perturbation_{col}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
        m[f"per_perturbation_{col}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
    m["n_perturbations_evaluated"] = int(per.shape[0])
    return m, per


# ============================================================
# Target preparation
# ============================================================

def _prepare_expression_target_matrices(control, perturbed, row, eval_genes):
    Xt_expr = to_csr(perturbed[:, eval_genes].X)
    X0_expr = load_pseudo_matrix_aligned(row, perturbed, eval_genes, require_full_alignment=True)
    X_control = to_csr(control[:, eval_genes].X)
    control_mean = mean_axis0(X_control).astype(np.float32)
    return X0_expr, Xt_expr, control_mean


def _prepare_representation_target_matrices(model_name: str, perturbed, control_h5ad: str | Path, row, config, runtime_cache: Path):
    pseudo_embedding_path, obs_path, source, slug = resolve_pseudo_embedding_file(model_name, row, config, runtime_cache)
    perturbed_obs = pd.Index(perturbed.obs_names.astype(str))
    X0_rep = load_aligned_pseudo_embedding(pseudo_embedding_path, obs_path, perturbed_obs)

    Xt_rep, pert_key, pert_meta = load_true_representation_matrix(
        h5ad_path=_cfg(config, "perturbed_h5ad"),
        model_name=model_name,
        config=config,
        role="perturbed",
        expected_obs_names=perturbed_obs,
        runtime_cache=runtime_cache,
    )

    _control_backed = sc.read_h5ad(str(control_h5ad), backed="r")
    control_obs = pd.Index(_control_backed.obs_names.astype(str))
    _control_backed.file.close()
    X_control_rep, ctrl_key, ctrl_meta = load_true_representation_matrix(
        h5ad_path=control_h5ad,
        model_name=model_name,
        config=config,
        role="control",
        expected_obs_names=control_obs,
        runtime_cache=runtime_cache,
    )

    if Xt_rep.shape[0] != perturbed.n_obs:
        raise ValueError(f"True perturbed representation n_obs mismatch for {model_name}: {Xt_rep.shape[0]} vs {perturbed.n_obs}")
    if X0_rep.shape[1] != Xt_rep.shape[1]:
        raise ValueError(f"Representation dimension mismatch for {model_name}: pseudo={X0_rep.shape[1]}, true={Xt_rep.shape[1]}")
    if X_control_rep.shape[1] != Xt_rep.shape[1]:
        raise ValueError(f"Control/perturbed representation dimension mismatch for {model_name}: control={X_control_rep.shape[1]}, perturbed={Xt_rep.shape[1]}")
    control_mean_rep = np.asarray(X_control_rep.mean(axis=0), dtype=np.float32)
    meta = {
        "pseudo_embedding_path": str(pseudo_embedding_path),
        "pseudo_embedding_source": source,
        "pseudo_embedding_slug": slug,
        "pseudo_embedding_obs_path": str(obs_path) if obs_path is not None else "",
        "true_perturbed_embedding_key": pert_key,
        "true_control_embedding_key": ctrl_key,
    }
    meta.update(pert_meta)
    meta.update(ctrl_meta)
    return X0_rep, Xt_rep.astype(np.float32, copy=False), control_mean_rep, meta


# ============================================================
# Main embedding branch
# ============================================================

def run_one_embedding_model(config: Any, model_name: str, base_outdir: Path) -> dict[str, Path]:
    config = as_namespace(config)
    model_outdir = ensure_dir(base_outdir / model_name)
    forward_dir = ensure_dir(model_outdir / "forward_outputs")
    inverse_dir = ensure_dir(model_outdir / "inverse_classification_outputs")
    split_dir = ensure_dir(Path(_cfg(config, "split_dir", base_outdir / "fixed_splits")))
    runtime_cache = ensure_dir(Path(_cfg(config, "runtime_cache_dir", base_outdir / "_runtime_cache")))

    runs_df = load_run_manifest(config)
    runs_df.to_csv(model_outdir / "discovered_run_manifest.csv", index=False)
    device = device_from_config(config)

    control = sc.read_h5ad(str(_cfg(config, "control_h5ad")))
    control.obs_names_make_unique()
    perturbed = sc.read_h5ad(str(_cfg(config, "perturbed_h5ad")))
    perturbed.obs_names_make_unique()
    key = _cfg(config, "perturbation_key", "perturbation_key")
    if key not in perturbed.obs.columns:
        raise KeyError(f"perturbation_key={key!r} not in perturbed.obs")

    labels_raw, y_all, label_to_id, id_to_label = build_label_encoder(
        perturbed,
        key,
        int(_cfg(config, "min_cells_per_perturbation", 20)),
        model_outdir,
    )
    indices = build_fixed_split(labels_raw, y_all, config, split_dir)
    n_classes = len(label_to_id)

    # Expression targets are used by default for forward tasks so the model_mse_xt/model_mae_xt
    # are directly comparable to the original expression MLP outputs.
    eval_genes = select_eval_genes(
        control,
        perturbed,
        runs_df,
        max_eval_genes=_cfg(config, "max_eval_genes", 3000),
        check_shared_genes_across_all_runs=False,
    )
    pd.Series(eval_genes).to_csv(model_outdir / "eval_genes.csv", index=False, header=["gene"])

    forward_target_space = str(_cfg(config, "forward_target_space", "expression"))
    inverse_input_space = str(_cfg(config, "inverse_input_space", "expression"))
    tasks = set(_cfg(config, "mlp_tasks", ["forward", "inverse_strategy_delta"]))
    skip_f = bool(_cfg(config, "skip_existing_forward", True))
    skip_i = bool(_cfg(config, "skip_existing_inverse", True))

    forward_records, forward_per_records, inverse_records, inverse_per_records = [], [], [], []

    for _, row in tqdm(runs_df.iterrows(), total=runs_df.shape[0], desc=f"MLP {model_name}"):
        metadata = metadata_from_manifest_row(row)
        run_id_base = str(metadata["run_id"])
        pseudo_embedding_path, obs_path, emb_source, emb_slug = resolve_pseudo_embedding_file(model_name, row, config, runtime_cache)
        X_input = load_aligned_pseudo_embedding(pseudo_embedding_path, obs_path, pd.Index(perturbed.obs_names.astype(str)))

        # Prepare forward target space.
        if forward_target_space == "expression":
            X0_fwd, Xt_fwd, control_mean_fwd = _prepare_expression_target_matrices(control, perturbed, row, eval_genes)
            target_dim = int(len(eval_genes))
        elif forward_target_space == "representation":
            X0_fwd, Xt_fwd, control_mean_fwd, _rep_meta = _prepare_representation_target_matrices(model_name, perturbed, _cfg(config, "control_h5ad"), row, config, runtime_cache)
            target_dim = int(Xt_fwd.shape[1])
        else:
            raise ValueError(f"Unknown forward_target_space={forward_target_space!r}")

        extra_meta = {
            "mlp_representation": model_name,
            "mlp_input_kind": "embedding",
            "forward_target_space": forward_target_space,
            "inverse_input_space": inverse_input_space,
            "pseudo_embedding_path": str(pseudo_embedding_path),
            "pseudo_embedding_source": emb_source,
            "pseudo_embedding_slug": emb_slug,
            "pseudo_embedding_dim": int(X_input.shape[1]),
            "forward_output_dim": target_dim,
        }
        metadata_with_rep = dict(metadata)
        metadata_with_rep.update(extra_meta)

        if "forward" in tasks:
            run_id = make_safe_id(f"{model_name}__{run_id_base}")
            jf = forward_dir / f"{run_id}_forward_test_metrics.json"
            pf = forward_dir / f"{run_id}_forward_per_perturbation_common_delta.csv"
            if skip_f and jf.exists() and pf.exists():
                fm = pd.read_json(jf, typ="series").to_dict()
                fp = pd.read_csv(pf)
            else:
                loaders = make_embedding_loaders(indices, X_input, X0_fwd, Xt_fwd, y_all, config, task="forward")
                set_seed(int(_cfg(config, "model_seed", 42)))
                model = ForwardInputPerturbationMLP(
                    input_dim=int(X_input.shape[1]),
                    n_perturbations=n_classes,
                    output_dim=target_dim,
                    pert_emb_dim=int(_cfg(config, "pert_emb_dim", 256)),
                    hidden_dim=int(_cfg(config, "hidden_dim", 1024)),
                    latent_dim=int(_cfg(config, "latent_dim", 512)),
                    dropout=float(_cfg(config, "dropout", 0.15)),
                )
                model, _hist = train_forward_model_general(model, loaders[0], loaders[1], run_id, forward_dir, config, device)
                fm, fp = evaluate_forward_model_general(model, loaders[2], control_mean_fwd, n_classes, id_to_label, config, device)
                fm.update(metadata_with_rep)
                fp = add_metadata_columns(fp, metadata_with_rep)
                save_json(fm, jf)
                fp.to_csv(pf, index=False)
                del model, loaders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            forward_records.append(fm)
            forward_per_records.append(fp)
            pd.DataFrame(forward_records).to_csv(model_outdir / "forward_mlp_run_summary_PARTIAL.csv", index=False)

        if "inverse_strategy_delta" in tasks:
            if inverse_input_space == "expression":
                X0_inv, Xt_inv, control_mean_inv = _prepare_expression_target_matrices(control, perturbed, row, eval_genes)
            elif inverse_input_space == "representation":
                X0_inv, Xt_inv, control_mean_inv, _rep_meta = _prepare_representation_target_matrices(model_name, perturbed, _cfg(config, "control_h5ad"), row, config, runtime_cache)
            else:
                raise ValueError(f"Unknown inverse_input_space={inverse_input_space!r}")
            inv_id = make_safe_id(f"{model_name}__{run_id_base}__inverse_strategy_delta")
            ji = inverse_dir / f"{inv_id}_inverse_classification_metrics.json"
            pi = inverse_dir / f"{inv_id}_inverse_classification_per_class.csv"
            if skip_i and ji.exists() and pi.exists():
                im = pd.read_json(ji, typ="series").to_dict()
                ip = pd.read_csv(pi)
            else:
                loaders = make_embedding_loaders(indices, X_input, X0_inv, Xt_inv, y_all, config, task="inverse", inverse_input_mode="strategy_delta", control_mean_vec=control_mean_inv)
                set_seed(int(_cfg(config, "model_seed", 42)))
                model = InversePerturbationMLP(
                    n_genes=int(Xt_inv.shape[1]),
                    n_perturbations=n_classes,
                    hidden_dim=int(_cfg(config, "hidden_dim", 1024)),
                    latent_dim=int(_cfg(config, "latent_dim", 512)),
                    dropout=float(_cfg(config, "dropout", 0.15)),
                )
                model, _hist = train_inverse_model(model, loaders[0], loaders[1], inv_id, inverse_dir, config, device)
                im, ip, pred = evaluate_inverse_model(model, loaders[2], n_classes, id_to_label, config, device)
                im.update(metadata_with_rep)
                im["inverse_input_mode"] = "strategy_delta"
                ip = add_metadata_columns(ip, metadata_with_rep)
                ip["inverse_input_mode"] = "strategy_delta"
                save_json(im, ji)
                ip.to_csv(pi, index=False)
                if bool(_cfg(config, "save_inverse_predictions", True)):
                    np.savez_compressed(inverse_dir / f"{inv_id}_inverse_classification_test_predictions.npz", **pred)
                del model, loaders
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            inverse_records.append(im)
            inverse_per_records.append(ip)
            pd.DataFrame(inverse_records).to_csv(model_outdir / "inverse_mlp_run_summary_PARTIAL.csv", index=False)

        del X_input
        gc.collect()

    if "inverse_common_delta" in tasks:
        # Common-reference inverse is one run per representation model, independent of pseudo strategy.
        common_meta = {
            "run_id": f"{model_name}__COMMON_REFERENCE_DELTA",
            "strategy": "COMMON_REFERENCE_DELTA",
            "strategy_id": "COMMON_REFERENCE_DELTA",
            "strategy_family": "common_reference_delta",
            "parameter_label": "xt_minus_mean_control",
            "mlp_representation": model_name,
            "mlp_input_kind": "embedding",
            "forward_target_space": forward_target_space,
            "inverse_input_space": inverse_input_space,
        }
        if inverse_input_space == "expression":
            Xt_inv = to_csr(perturbed[:, eval_genes].X)
            X0_inv = Xt_inv
            X_control = to_csr(control[:, eval_genes].X)
            control_mean_inv = mean_axis0(X_control).astype(np.float32)
        elif inverse_input_space == "representation":
            perturbed_obs = pd.Index(perturbed.obs_names.astype(str))
            control_obs = pd.Index(control.obs_names.astype(str))
            Xt_inv, _pert_key, _pert_meta = load_true_representation_matrix(
                h5ad_path=_cfg(config, "perturbed_h5ad"),
                model_name=model_name,
                config=config,
                role="perturbed",
                expected_obs_names=perturbed_obs,
                runtime_cache=runtime_cache,
            )
            X_control_inv, _ctrl_key, _ctrl_meta = load_true_representation_matrix(
                h5ad_path=_cfg(config, "control_h5ad"),
                model_name=model_name,
                config=config,
                role="control",
                expected_obs_names=control_obs,
                runtime_cache=runtime_cache,
            )
            X0_inv = Xt_inv
            control_mean_inv = np.asarray(X_control_inv.mean(axis=0), dtype=np.float32)
        else:
            raise ValueError(inverse_input_space)
        inv_id = make_safe_id(f"{model_name}__COMMON_REFERENCE_DELTA__inverse_common_delta")
        loaders = make_embedding_loaders(indices, Xt_inv, X0_inv, Xt_inv, y_all, config, task="inverse", inverse_input_mode="common_delta", control_mean_vec=control_mean_inv)
        set_seed(int(_cfg(config, "model_seed", 42)))
        model = InversePerturbationMLP(
            n_genes=int(Xt_inv.shape[1]),
            n_perturbations=n_classes,
            hidden_dim=int(_cfg(config, "hidden_dim", 1024)),
            latent_dim=int(_cfg(config, "latent_dim", 512)),
            dropout=float(_cfg(config, "dropout", 0.15)),
        )
        model, _hist = train_inverse_model(model, loaders[0], loaders[1], inv_id, inverse_dir, config, device)
        im, ip, pred = evaluate_inverse_model(model, loaders[2], n_classes, id_to_label, config, device)
        im.update(common_meta)
        im["inverse_input_mode"] = "common_delta"
        ip = add_metadata_columns(ip, common_meta)
        ip["inverse_input_mode"] = "common_delta"
        save_json(im, inverse_dir / f"{inv_id}_inverse_classification_metrics.json")
        ip.to_csv(inverse_dir / f"{inv_id}_inverse_classification_per_class.csv", index=False)
        if bool(_cfg(config, "save_inverse_predictions", True)):
            np.savez_compressed(inverse_dir / f"{inv_id}_inverse_classification_test_predictions.npz", **pred)
        inverse_records.append(im)
        inverse_per_records.append(ip)
        del model, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    paths: dict[str, Path] = {}
    if forward_records:
        fdf = pd.DataFrame(forward_records)
        fpath = model_outdir / "forward_mlp_run_summary.csv"
        fdf.to_csv(fpath, index=False)
        paths["forward_summary"] = fpath
        if forward_per_records:
            fppath = model_outdir / "forward_mlp_per_perturbation_common_delta.csv"
            pd.concat(forward_per_records, ignore_index=True).to_csv(fppath, index=False)
            paths["forward_per_perturbation"] = fppath
        summarize_numeric(fdf, [c for c in ["perturbed_group", "strategy", "strategy_family", "mlp_representation"] if c in fdf.columns], model_outdir / "summaries/forward_seed_averaged_by_strategy.csv")
    if inverse_records:
        idf = pd.DataFrame(inverse_records)
        ipath = model_outdir / "inverse_mlp_run_summary.csv"
        idf.to_csv(ipath, index=False)
        paths["inverse_summary"] = ipath
        if inverse_per_records:
            icpath = model_outdir / "inverse_mlp_per_class.csv"
            pd.concat(inverse_per_records, ignore_index=True).to_csv(icpath, index=False)
            paths["inverse_per_class"] = icpath
        group_cols = [c for c in ["perturbed_group", "strategy", "strategy_family", "inverse_input_mode", "mlp_representation"] if c in idf.columns]
        summarize_numeric(idf, group_cols, model_outdir / "summaries/inverse_seed_averaged_by_strategy.csv")
    save_json({
        "model_name": model_name,
        "n_runs": len(runs_df),
        "n_eval_genes": len(eval_genes),
        "n_perturbation_classes": len(label_to_id),
        "device": str(device),
        "forward_target_space": forward_target_space,
        "inverse_input_space": inverse_input_space,
    }, model_outdir / "mlp_config_summary.json")

    del control, perturbed
    gc.collect()
    return paths



# ============================================================
# Selected-variant manifest filtering
# ============================================================

def _safe_resolved_path(path: str | Path) -> str:
    """Resolve paths when possible, but do not fail on missing files."""
    path = Path(path)
    try:
        return str(path.resolve())
    except Exception:
        return str(path.absolute())


def _variant_rel_from_pseudo_path(pseudo_h5ad: str | Path, pseudo_group_root: str | Path) -> str:
    """Return variant directory relative to the group root, preserving slash syntax."""
    pseudo_h5ad = Path(pseudo_h5ad)
    variant_dir = pseudo_h5ad.parent
    try:
        return str(variant_dir.relative_to(Path(pseudo_group_root))).strip("/")
    except Exception:
        return ""


def _prepare_selected_variant_manifest(config: Any, base_outdir: Path) -> Any:
    """Create a temporary manifest containing only config.selected_variants.

    This is needed because the original eval_common.load_run_manifest filters by
    broad strategy names, but the current comparison requires exact variant paths
    such as S3_SEACell_metacell_average/nmc_350/k_10/seed_000.

    The returned config points manifest_path to the filtered manifest, so both
    mlp_expr, which delegates to eval_mlp.run_mlp_evaluation, and all embedding
    branches evaluate the exact same variant subset.
    """
    selected_variants = _cfg(config, "selected_variants", None)
    if selected_variants is None or len(selected_variants) == 0:
        return config

    pseudo_group_root = _cfg(config, "pseudo_group_root", None)
    if pseudo_group_root is None:
        raise ValueError("selected_variants requires config.pseudo_group_root.")

    selected_variants = [str(v).strip().strip("/") for v in selected_variants]
    selected_variant_set = set(selected_variants)
    pseudo_group_root = Path(pseudo_group_root)

    manifest_path = Path(_cfg(config, "manifest_path"))
    manifest_raw = pd.read_csv(manifest_path)
    manifest = standardize_manifest_columns(manifest_raw)

    expected_paths = {
        _safe_resolved_path(pseudo_group_root / variant / "pseudo_control_aligned_to_perturbed.h5ad"): variant
        for variant in selected_variants
    }

    manifest["_resolved_pseudo_control_h5ad"] = manifest["pseudo_control_h5ad"].map(_safe_resolved_path)
    manifest["_variant_rel"] = manifest["pseudo_control_h5ad"].map(
        lambda x: _variant_rel_from_pseudo_path(x, pseudo_group_root)
    )

    mask = manifest["_resolved_pseudo_control_h5ad"].isin(expected_paths.keys()) | manifest["_variant_rel"].isin(selected_variant_set)
    filtered = manifest.loc[mask].copy()

    found_variants = set(filtered["_variant_rel"].astype(str).tolist())
    # If exact rel-path extraction failed because the manifest root differs, recover found variants from expected paths.
    for resolved in filtered["_resolved_pseudo_control_h5ad"].astype(str).tolist():
        if resolved in expected_paths:
            found_variants.add(expected_paths[resolved])
    missing = [v for v in selected_variants if v not in found_variants]

    strict = bool(_cfg(config, "strict_selected_variants", True))
    if missing and strict:
        msg = "Selected variants missing from manifest or files:\n" + "\n".join(f"  - {v}" for v in missing)
        raise FileNotFoundError(msg)
    if missing:
        print("[Warn] Selected variants missing from manifest or files:")
        for v in missing:
            print(f"  - {v}")

    if filtered.empty:
        raise RuntimeError("No manifest rows left after selected_variants filtering.")

    filtered = filtered.drop(columns=["_resolved_pseudo_control_h5ad", "_variant_rel"], errors="ignore")
    out_manifest = base_outdir / "selected_variant_manifest.csv"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(out_manifest, index=False)

    selected_table = pd.DataFrame({
        "selected_variant": selected_variants,
        "expected_pseudo_control_h5ad": [
            str(pseudo_group_root / v / "pseudo_control_aligned_to_perturbed.h5ad") for v in selected_variants
        ],
        "found": [v not in missing for v in selected_variants],
    })
    selected_table.to_csv(base_outdir / "selected_variant_check.csv", index=False)

    new_config = _copy_namespace(config)
    new_config.manifest_path = str(out_manifest)
    new_config.max_runs_to_evaluate = None
    print(f"[Selected variants] Wrote filtered manifest with {filtered.shape[0]} rows: {out_manifest}")
    return new_config

# ============================================================
# Public entry point
# ============================================================

def run_mlp_task_model_evaluation(config: Mapping[str, Any] | SimpleNamespace) -> dict[str, Any]:
    """Run MLP-only evaluation over expression and embedding representations.

    Parameters expected in config
    -----------------------------
    Required, same as the original evaluation pipeline:
        control_h5ad, perturbed_h5ad, manifest_path, outdir, perturbation_key
    Additional for embedding models:
        pseudo_group_root: folder containing S0-S5 variant directories for this group.
        models_to_run: subset of [mlp_expr, geneformer, hvg, scvi, sccello, scimilarity, scgpt].
        forward_target_space: 'expression' or 'representation'. Default 'expression'.
        inverse_input_space: 'expression' or 'representation'. Default 'expression'.
            Recommended for your current setup: 'expression'. This predicts perturbation
            identity from true perturbed expression - pseudo-control expression, so it
            does not require true scVI/scGPT/scCello/scimilarity embeddings.
    """
    config = as_namespace(config)
    base_outdir = ensure_dir(Path(_cfg(config, "outdir")) / "downstream_mlp_task_models")
    config = _prepare_selected_variant_manifest(config, base_outdir)
    split_dir = ensure_dir(Path(_cfg(config, "split_dir", base_outdir / "fixed_splits")))
    runtime_cache = ensure_dir(Path(_cfg(config, "runtime_cache_dir", base_outdir / "_runtime_cache")))
    models_to_run = list(_cfg(config, "models_to_run", DEFAULT_MODELS_TO_RUN))

    outputs: dict[str, Any] = {}
    combined_forward = []
    combined_inverse = []

    for model_name in models_to_run:
        print("=" * 100)
        print(f"[MLP representation] {model_name}")
        print("=" * 100)
        if model_name == EXPRESSION_MODEL_NAME:
            # This is the critical branch for reproducing the original expression MLP results.
            # We do not use the embedding branch, custom label_df, or variance-gene selector here.
            expr_config = _copy_namespace(config)
            expr_config.outdir = base_outdir / EXPRESSION_MODEL_NAME
            expr_config.split_dir = split_dir
            expr_config.evaluation_tasks = ["mlp"]
            expr_paths = run_original_expression_mlp_evaluation(expr_config)
            outputs[model_name] = expr_paths
            fpath = expr_paths.get("forward_summary")
            ipath = expr_paths.get("inverse_summary")
            if fpath and Path(fpath).exists():
                df = pd.read_csv(fpath)
                df.insert(0, "mlp_representation", model_name)
                df.insert(1, "mlp_input_kind", "expression")
                combined_forward.append(df)
            if ipath and Path(ipath).exists():
                df = pd.read_csv(ipath)
                df.insert(0, "mlp_representation", model_name)
                df.insert(1, "mlp_input_kind", "expression")
                combined_inverse.append(df)
            continue

        try:
            paths = run_one_embedding_model(config, model_name, base_outdir)
            outputs[model_name] = paths
            fpath = paths.get("forward_summary")
            ipath = paths.get("inverse_summary")
            if fpath and Path(fpath).exists():
                combined_forward.append(pd.read_csv(fpath))
            if ipath and Path(ipath).exists():
                combined_inverse.append(pd.read_csv(ipath))
        except Exception as exc:
            if bool(_cfg(config, "continue_on_error", True)):
                print(f"[Error] {model_name} failed: {repr(exc)}")
                outputs[model_name] = {"error": repr(exc)}
            else:
                raise

    if combined_forward:
        fdf = pd.concat(combined_forward, ignore_index=True, sort=False)
        fpath = base_outdir / "combined_forward_mlp_run_summary.csv"
        fdf.to_csv(fpath, index=False)
        outputs["combined_forward_summary"] = fpath
        summarize_numeric(
            fdf,
            [c for c in ["mlp_representation", "perturbed_group", "strategy", "strategy_family"] if c in fdf.columns],
            base_outdir / "summaries/combined_forward_seed_averaged_by_strategy.csv",
        )
    if combined_inverse:
        idf = pd.concat(combined_inverse, ignore_index=True, sort=False)
        ipath = base_outdir / "combined_inverse_mlp_run_summary.csv"
        idf.to_csv(ipath, index=False)
        outputs["combined_inverse_summary"] = ipath
        summarize_numeric(
            idf,
            [c for c in ["mlp_representation", "perturbed_group", "strategy", "strategy_family", "inverse_input_mode"] if c in idf.columns],
            base_outdir / "summaries/combined_inverse_seed_averaged_by_strategy.csv",
        )

    if not bool(_cfg(config, "save_runtime_cache", False)) and runtime_cache.exists():
        shutil.rmtree(runtime_cache, ignore_errors=True)

    save_json({"outputs": {str(k): str(v) for k, v in outputs.items()}}, base_outdir / "mlp_task_model_outputs.json")
    return outputs
