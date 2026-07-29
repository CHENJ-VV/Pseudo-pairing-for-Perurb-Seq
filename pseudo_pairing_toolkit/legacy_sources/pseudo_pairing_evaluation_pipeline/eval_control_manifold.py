"""Control-manifold preservation evaluation for repeated pseudo-control datasets."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from eval_common import (
    add_metadata_columns,
    as_namespace,
    ensure_dir,
    get_config,
    load_pseudo_matrix_aligned,
    load_run_manifest,
    mean_var_axis0,
    metadata_from_manifest_row,
    read_dataframe,
    safe_pearson,
    save_dataframe,
    save_json,
    select_eval_genes,
    set_seed,
    summarize_numeric,
    to_dense_float32,
)


def sample_indices(n: int, max_n: int | None, rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(int(n), dtype=np.int64)
    if max_n is not None and len(idx) > int(max_n):
        idx = rng.choice(idx, size=int(max_n), replace=False)
        idx = np.sort(idx)
    return idx


def rbf_mmd2(A: np.ndarray, B: np.ndarray, gamma: float | None = None, max_n: int = 3000) -> float:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.nan
    n = min(A.shape[0], max_n)
    m = min(B.shape[0], max_n)
    A = A[:n]
    B = B[:m]
    if gamma is None:
        pooled = np.vstack([A, B])
        d = pairwise_distances(pooled[: min(1000, pooled.shape[0])], metric="sqeuclidean")
        positive = d[d > 0]
        gamma = 1.0 / (np.median(positive) + 1e-12) if positive.size else 1.0
    Kaa = np.exp(-gamma * pairwise_distances(A, A, metric="sqeuclidean"))
    Kbb = np.exp(-gamma * pairwise_distances(B, B, metric="sqeuclidean"))
    Kab = np.exp(-gamma * pairwise_distances(A, B, metric="sqeuclidean"))
    return float(Kaa.mean() + Kbb.mean() - 2.0 * Kab.mean())


def normalized_entropy(counts) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def compute_source_mixing_score(control_pca: np.ndarray, pseudo_pca: np.ndarray, k: int = 30) -> dict[str, float]:
    X = np.vstack([control_pca, pseudo_pca])
    labels = np.concatenate([np.zeros(control_pca.shape[0], dtype=int), np.ones(pseudo_pca.shape[0], dtype=int)])
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric="euclidean")
    nn.fit(X)
    _, ind = nn.kneighbors(X)
    ind = ind[:, 1:]
    neigh_labels = labels[ind]
    own = labels[:, None]
    opposite = np.mean(neigh_labels != own, axis=1)
    return {
        "source_mixing_opposite_neighbor_fraction_mean": float(np.mean(opposite)),
        "source_mixing_control_cells_mean": float(np.mean(opposite[labels == 0])) if np.any(labels == 0) else np.nan,
        "source_mixing_pseudo_cells_mean": float(np.mean(opposite[labels == 1])) if np.any(labels == 1) else np.nan,
    }


def compute_metacell_coverage(pair_metadata_path: str | Path | None, membership_path: str | Path | None) -> dict[str, Any]:
    if not pair_metadata_path or not membership_path:
        return {
            "metacell_coverage_n_covered": np.nan,
            "metacell_coverage_fraction": np.nan,
            "metacell_usage_entropy": np.nan,
            "metacell_max_usage_fraction": np.nan,
        }
    try:
        pair_df = read_dataframe(pair_metadata_path)
        membership_df = read_dataframe(membership_path)
    except Exception:
        return {
            "metacell_coverage_n_covered": np.nan,
            "metacell_coverage_fraction": np.nan,
            "metacell_usage_entropy": np.nan,
            "metacell_max_usage_fraction": np.nan,
        }
    total_mcs = membership_df["metacell_id"].astype(str).nunique() if "metacell_id" in membership_df.columns else np.nan
    mc_cols = [
        c for c in [
            "assigned_metacell_id", "dominant_metacell_id", "sampled_metacell_ids", "top_metacell_ids"
        ] if c in pair_df.columns
    ]
    used = []
    for col in mc_cols:
        for val in pair_df[col].dropna().astype(str):
            used.extend([x for x in val.replace(";", "|").replace(",", "|").split("|") if x])
    if not used:
        return {
            "metacell_coverage_n_covered": np.nan,
            "metacell_coverage_fraction": np.nan,
            "metacell_usage_entropy": np.nan,
            "metacell_max_usage_fraction": np.nan,
        }
    counts = pd.Series(used).value_counts()
    return {
        "metacell_coverage_n_covered": int(counts.shape[0]),
        "metacell_coverage_fraction": float(counts.shape[0] / total_mcs) if total_mcs and np.isfinite(total_mcs) else np.nan,
        "metacell_usage_entropy": normalized_entropy(counts.values),
        "metacell_max_usage_fraction": float(counts.max() / counts.sum()),
    }


def get_duplicate_stats(row: pd.Series, pair_metadata_path: str | Path | None) -> dict[str, Any]:
    if not pair_metadata_path:
        return {"unique_control_cells_used_if_available": np.nan, "duplicate_fraction_if_available": np.nan}
    try:
        pair_df = read_dataframe(pair_metadata_path)
    except Exception:
        return {"unique_control_cells_used_if_available": np.nan, "duplicate_fraction_if_available": np.nan}
    candidates = ["selected_control_cell_id", "selected_control_cell_pos"]
    for c in candidates:
        if c in pair_df.columns:
            n_unique = pair_df[c].nunique(dropna=True)
            return {
                "unique_control_cells_used_if_available": int(n_unique),
                "duplicate_fraction_if_available": float(1.0 - n_unique / max(pair_df.shape[0], 1)),
            }
    return {"unique_control_cells_used_if_available": np.nan, "duplicate_fraction_if_available": np.nan}


def evaluate_one_control_manifold_run(row: pd.Series, control, eval_genes, pca: PCA, control_mean, control_var,
                                      control_pca_full: np.ndarray, config) -> dict[str, Any]:
    rng = set_seed(int(get_config(config, "seed", 42)))
    pseudo = sc.read_h5ad(row["pseudo_control_h5ad"])
    pseudo.obs_names_make_unique()
    missing = pd.Index(eval_genes).difference(pd.Index(pseudo.var_names.astype(str)))
    if len(missing) > 0:
        raise ValueError(f"Missing eval genes in {row['pseudo_control_h5ad']}: {missing[:5].tolist()}")
    Xp = pseudo[:, eval_genes].X
    pseudo_mean, pseudo_var = mean_var_axis0(Xp)

    pca_sample_n = get_config(config, "n_pseudo_sample_for_overlap", 10000)
    pseudo_idx = sample_indices(pseudo.n_obs, pca_sample_n, rng)
    control_idx = sample_indices(control.n_obs, get_config(config, "n_control_sample_for_overlap", 10000), rng)
    pseudo_pca = pca.transform(to_dense_float32(Xp[pseudo_idx, :]))
    control_pca = control_pca_full[control_idx]

    rec = metadata_from_manifest_row(row)
    rec.update({
        "n_obs": int(pseudo.n_obs),
        "n_genes_eval": int(len(eval_genes)),
        "mean_expression_pearson": safe_pearson(control_mean, pseudo_mean),
        "mean_expression_r2": safe_pearson(control_mean, pseudo_mean) ** 2,
        "mean_expression_rmse": float(np.sqrt(np.mean((pseudo_mean - control_mean) ** 2))),
        "variance_pearson": safe_pearson(control_var, pseudo_var),
        "variance_r2": safe_pearson(control_var, pseudo_var) ** 2,
        "log1p_variance_pearson": safe_pearson(np.log1p(control_var), np.log1p(pseudo_var)),
        "log1p_variance_r2": safe_pearson(np.log1p(control_var), np.log1p(pseudo_var)) ** 2,
        "variance_rmse": float(np.sqrt(np.mean((pseudo_var - control_var) ** 2))),
        "median_gene_var_ratio_pseudo_over_control": float(np.nanmedian(pseudo_var / (control_var + 1e-12))),
        "pca_mean_distance": float(np.linalg.norm(pseudo_pca.mean(axis=0) - control_pca.mean(axis=0))),
        "pca_variance_ratio_pseudo_over_control": float(np.sum(np.var(pseudo_pca, axis=0)) / (np.sum(np.var(control_pca, axis=0)) + 1e-12)),
        "mmd_rbf_pca": rbf_mmd2(
            control_pca,
            pseudo_pca,
            max_n=int(get_config(config, "n_mmd_sample", 3000)),
        ),
    })
    rec.update(compute_source_mixing_score(control_pca, pseudo_pca, k=int(get_config(config, "source_mixing_k", 30))))
    rec.update(get_duplicate_stats(row, row.get("pair_metadata_path", "")))
    rec.update(compute_metacell_coverage(row.get("pair_metadata_path", ""), row.get("membership_path_for_metacell_coverage", "")))
    del pseudo
    gc.collect()
    return rec


def run_control_manifold_evaluation(config: Mapping[str, Any]) -> dict[str, Path]:
    config = as_namespace(config)
    outdir = ensure_dir(Path(get_config(config, "outdir")) / "control_manifold")
    runs_df = load_run_manifest(config)
    runs_df.to_csv(outdir / "discovered_run_manifest.csv", index=False)

    control = sc.read_h5ad(str(get_config(config, "control_h5ad")))
    control.obs_names_make_unique()
    eval_genes = select_eval_genes(
        control=control,
        perturbed=None,
        runs_df=runs_df,
        max_eval_genes=get_config(config, "max_eval_genes", 3000),
        check_shared_genes_across_all_runs=bool(get_config(config, "check_shared_genes_across_all_runs", False)),
    )
    pd.Series(eval_genes).to_csv(outdir / "eval_genes.csv", index=False, header=["gene"])

    Xc = control[:, eval_genes].X
    control_mean, control_var = mean_var_axis0(Xc)
    rng = set_seed(int(get_config(config, "seed", 42)))
    pca_idx = sample_indices(control.n_obs, get_config(config, "n_control_sample_for_pca", 10000), rng)
    pca = PCA(n_components=min(int(get_config(config, "n_pcs", 50)), len(eval_genes), len(pca_idx) - 1), random_state=int(get_config(config, "seed", 42)))
    pca.fit(to_dense_float32(Xc[pca_idx, :]))
    control_pca_full = pca.transform(to_dense_float32(Xc))

    records = []
    partial = outdir / "control_manifold_preservation_repeated_long_PARTIAL.csv"
    for _, row in tqdm(runs_df.iterrows(), total=runs_df.shape[0], desc="Control-manifold evaluation"):
        rec = evaluate_one_control_manifold_run(row, control, eval_genes, pca, control_mean, control_var, control_pca_full, config)
        records.append(rec)
        pd.DataFrame(records).to_csv(partial, index=False)
    long_df = pd.DataFrame(records)
    long_path = outdir / "control_manifold_preservation_repeated_long.csv"
    long_df.to_csv(long_path, index=False)
    summary_path = outdir / "control_manifold_preservation_repeated_summary.csv"
    summary = summarize_numeric(long_df, group_cols=["perturbed_group", "strategy", "strategy_family", "parameter_label"], output_path=summary_path)
    save_json({"n_runs": len(runs_df), "n_eval_genes": len(eval_genes)}, outdir / "control_manifold_config_summary.json")
    return {"long": long_path, "summary": summary_path}
