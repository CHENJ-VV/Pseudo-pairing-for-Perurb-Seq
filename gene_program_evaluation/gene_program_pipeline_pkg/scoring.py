from __future__ import annotations

from pathlib import Path
from typing import Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .utils import chunk_slices, ensure_dir, get_expr_matrix, read_table, save_table


def load_programs(program_membership_path: str | Path) -> pd.DataFrame:
    membership = read_table(program_membership_path)
    required = {"program_id", "gene"}
    missing = required - set(membership.columns)
    if missing:
        raise KeyError(f"Program membership file missing columns: {missing}")
    membership["program_id"] = membership["program_id"].astype(str)
    membership["gene"] = membership["gene"].astype(str)
    return membership


def build_program_weight_matrix(
    adata_var_names: pd.Index,
    membership: pd.DataFrame,
    control_stats: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], pd.DataFrame]:
    genes = pd.Index(adata_var_names.astype(str))
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    stats = control_stats.copy()
    stats["gene"] = stats["gene"].astype(str)
    stats = stats.set_index("gene")
    program_ids = sorted(membership["program_id"].astype(str).unique())
    p_to_j = {p: j for j, p in enumerate(program_ids)}
    rows = []
    cols = []
    vals = []
    intercept = np.zeros(len(program_ids), dtype=np.float64)
    records = []
    for p, sub in membership.groupby("program_id", sort=True):
        p = str(p)
        usable = [g for g in sub["gene"].astype(str) if g in gene_to_idx and g in stats.index]
        if len(usable) == 0:
            continue
        scale = np.sqrt(float(len(usable)))
        j = p_to_j[p]
        for g in usable:
            mu = float(stats.loc[g, "mean"])
            sd = float(stats.loc[g, "std"])
            if not np.isfinite(sd) or sd <= 0:
                continue
            w = 1.0 / (sd * scale)
            rows.append(gene_to_idx[g])
            cols.append(j)
            vals.append(w)
            intercept[j] += mu * w
        records.append({"program_id": p, "n_genes_available": len(usable), "n_genes_defined": int(sub.shape[0])})
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(len(genes), len(program_ids)), dtype=np.float64)
    return W, intercept, program_ids, pd.DataFrame(records)


def score_adata_programs(
    adata: ad.AnnData,
    program_membership: pd.DataFrame,
    control_stats: pd.DataFrame,
    layer: str | None = None,
    chunk_size: int = 20000,
) -> pd.DataFrame:
    W, intercept, program_ids, availability = build_program_weight_matrix(adata.var_names, program_membership, control_stats)
    X = get_expr_matrix(adata, layer)
    chunks = []
    for sl in chunk_slices(int(adata.n_obs), int(chunk_size)):
        Xc = X[sl, :]
        sc = Xc @ W
        if sparse.issparse(sc):
            sc = sc.toarray()
        sc = np.asarray(sc, dtype=np.float64) - intercept[None, :]
        chunks.append(sc.astype(np.float32, copy=False))
    scores = np.vstack(chunks) if chunks else np.zeros((0, len(program_ids)), dtype=np.float32)
    df = pd.DataFrame(scores, index=adata.obs_names.astype(str), columns=program_ids)
    df.index.name = "cell_id"
    return df


def save_program_scores(scores: pd.DataFrame, obs: pd.DataFrame, out_path: str | Path, perturbation_key: str | None = None) -> Path:
    df = scores.reset_index()
    if perturbation_key and perturbation_key in obs.columns:
        labels = obs[perturbation_key].astype(str).reindex(scores.index)
        df.insert(1, perturbation_key, labels.values)
    return save_table(df, out_path)


def group_mean_scores(scores: pd.DataFrame, labels: pd.Series | np.ndarray) -> pd.DataFrame:
    labels = pd.Series(np.asarray(labels).astype(str), index=scores.index, name="perturbation")
    tmp = scores.copy()
    tmp["perturbation"] = labels.values
    return tmp.groupby("perturbation", sort=True).mean(numeric_only=True)
