from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph
from sklearn.cluster import AgglomerativeClustering

from .utils import column_mean, column_mean_square, column_nonzero_fraction, dense_rows, ensure_dir, get_expr_matrix, save_json, save_table


def compute_control_gene_stats(control: ad.AnnData, layer: str | None, eps: float = 1e-6) -> pd.DataFrame:
    X = get_expr_matrix(control, layer)
    mean = column_mean(X)
    mean_sq = column_mean_square(X)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    std = np.maximum(std, float(eps))
    frac = column_nonzero_fraction(X)
    return pd.DataFrame({
        "gene": control.var_names.astype(str),
        "mean": mean,
        "variance": var,
        "std": std,
        "nonzero_fraction": frac,
    })


def select_genes_for_programs(stats: pd.DataFrame, max_genes: int, min_mean: float, min_frac: float) -> pd.DataFrame:
    df = stats.copy()
    df["passes_filters"] = (df["mean"] >= float(min_mean)) & (df["nonzero_fraction"] >= float(min_frac)) & np.isfinite(df["variance"])
    selected = df[df["passes_filters"]].sort_values("variance", ascending=False).head(int(max_genes)).copy()
    selected["selected_for_program_building"] = True
    out = df.merge(selected[["gene", "selected_for_program_building"]], on="gene", how="left")
    out["selected_for_program_building"] = out["selected_for_program_building"].fillna(False).astype(bool)
    return out


def make_control_zscore_sample(
    control: ad.AnnData,
    selected_stats: pd.DataFrame,
    layer: str | None,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    genes = selected_stats["gene"].astype(str).tolist()
    gene_to_idx = {g: i for i, g in enumerate(control.var_names.astype(str))}
    idx = np.array([gene_to_idx[g] for g in genes], dtype=int)
    n_obs = int(control.n_obs)
    if max_cells is not None and int(max_cells) > 0 and n_obs > int(max_cells):
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(np.arange(n_obs), size=int(max_cells), replace=False))
    else:
        rows = np.arange(n_obs, dtype=int)
    X = get_expr_matrix(control, layer)
    dense = dense_rows(X, rows, idx).astype(np.float32, copy=False)
    mean = selected_stats["mean"].to_numpy(dtype=np.float32)
    std = selected_stats["std"].to_numpy(dtype=np.float32)
    dense = (dense - mean[None, :]) / std[None, :]
    # Center sampled z-scores again to stabilize correlation estimates.
    dense -= dense.mean(axis=0, keepdims=True)
    col_std = dense.std(axis=0, ddof=1, keepdims=True)
    col_std[col_std <= 1e-8] = 1.0
    dense /= col_std
    return dense, genes


def compute_gene_correlation(z: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(z, rowvar=False)
    corr = np.asarray(corr, dtype=np.float32)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def cluster_genes_from_correlation(
    corr: np.ndarray,
    genes: list[str],
    corr_threshold: float,
    min_correlated_partners: int,
    cluster_method: str = "auto",
    leiden_resolution: float = 1.0,
) -> pd.DataFrame:
    n = len(genes)
    if n == 0:
        return pd.DataFrame(columns=["gene", "raw_cluster"])
    adj = corr >= float(corr_threshold)
    np.fill_diagonal(adj, False)
    degree = adj.sum(axis=0)
    keep = degree >= int(min_correlated_partners)
    if keep.sum() == 0:
        warnings.warn("No genes passed min_correlated_partners; using all selected genes with connected components.")
        keep[:] = True
    keep_idx = np.where(keep)[0]
    corr_keep = corr[np.ix_(keep_idx, keep_idx)]
    adj_keep = corr_keep >= float(corr_threshold)
    np.fill_diagonal(adj_keep, False)

    labels = np.full(n, -1, dtype=int)
    method = str(cluster_method).lower()
    used = "connected_components"
    if method in {"auto", "leiden"}:
        try:
            import igraph as ig
            import leidenalg
            sources, targets = np.where(np.triu(adj_keep, k=1))
            weights = corr_keep[sources, targets].astype(float).tolist()
            g = ig.Graph(n=int(len(keep_idx)), edges=list(zip(sources.tolist(), targets.tolist())), directed=False)
            if len(weights) > 0:
                g.es["weight"] = weights
            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight" if len(weights) > 0 else None,
                resolution_parameter=float(leiden_resolution),
            )
            labels_keep = np.asarray(part.membership, dtype=int)
            used = "leiden"
        except Exception as exc:
            if method == "leiden":
                raise RuntimeError(f"Leiden clustering failed: {exc!r}") from exc
            warnings.warn(f"Leiden unavailable/failed ({exc!r}); falling back to connected components.")
            n_components, labels_keep = csgraph.connected_components(sparse.csr_matrix(adj_keep), directed=False)
            used = "connected_components"
    elif method in {"connected", "connected_components"}:
        n_components, labels_keep = csgraph.connected_components(sparse.csr_matrix(adj_keep), directed=False)
    elif method in {"agglomerative", "hierarchical"}:
        dist = 1.0 - np.clip(corr_keep, -1, 1)
        try:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - float(corr_threshold), metric="precomputed", linkage="average")
        except TypeError:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - float(corr_threshold), affinity="precomputed", linkage="average")
        labels_keep = model.fit_predict(dist)
        used = "agglomerative"
    else:
        raise ValueError("cluster_method must be auto, leiden, connected_components, or agglomerative")
    labels[keep_idx] = labels_keep
    out = pd.DataFrame({"gene": genes, "raw_cluster": labels, "used_cluster_method": used, "correlated_partner_count": degree})
    return out


def refine_program_membership(
    cluster_df: pd.DataFrame,
    corr: np.ndarray,
    genes: list[str],
    min_program_size: int,
    max_program_size: int | None,
    min_gene_centroid_corr: float,
) -> pd.DataFrame:
    gene_to_pos = {g: i for i, g in enumerate(genes)}
    records = []
    program_counter = 0
    for raw_cluster, sub in cluster_df.groupby("raw_cluster", sort=True):
        if int(raw_cluster) < 0:
            continue
        members = sub["gene"].astype(str).tolist()
        if len(members) < int(min_program_size):
            continue
        positions = np.array([gene_to_pos[g] for g in members], dtype=int)
        subcorr = corr[np.ix_(positions, positions)]
        centroid_corr = (subcorr.sum(axis=1) - 1.0) / max(len(members) - 1, 1)
        keep = centroid_corr >= float(min_gene_centroid_corr)
        if keep.sum() < int(min_program_size):
            keep[:] = True
        kept_members = [members[i] for i in range(len(members)) if keep[i]]
        kept_centroid = [float(centroid_corr[i]) for i in range(len(members)) if keep[i]]
        order = np.argsort(kept_centroid)[::-1]
        if max_program_size is not None and int(max_program_size) > 0:
            order = order[: int(max_program_size)]
        kept_members = [kept_members[i] for i in order]
        kept_centroid = [kept_centroid[i] for i in order]
        if len(kept_members) < int(min_program_size):
            continue
        program_id = f"GP{program_counter:04d}"
        program_counter += 1
        for rank, (gene, cc) in enumerate(zip(kept_members, kept_centroid), start=1):
            records.append({
                "program_id": program_id,
                "gene": gene,
                "gene_rank_in_program": rank,
                "raw_cluster": int(raw_cluster),
                "gene_centroid_corr": cc,
                "program_size": len(kept_members),
            })
    return pd.DataFrame(records)


def build_gene_programs_for_dataset(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any]) -> dict[str, Path]:
    outdir = ensure_dir(Path(dataset_cfg["output_dir"]) / "gene_programs")
    layer = global_cfg.get("layer", None)
    eps = float(global_cfg.get("zscore_eps", 1e-6))
    seed = int(global_cfg.get("seed", 0))
    control = ad.read_h5ad(dataset_cfg["control_h5ad"])
    stats = compute_control_gene_stats(control, layer=layer, eps=eps)
    selection_stats = select_genes_for_programs(
        stats,
        max_genes=int(global_cfg.get("max_genes", 3000)),
        min_mean=float(global_cfg.get("min_mean", 0.01)),
        min_frac=float(global_cfg.get("min_frac", 0.01)),
    )
    selected_stats = selection_stats[selection_stats["selected_for_program_building"]].copy()
    z, genes = make_control_zscore_sample(
        control,
        selected_stats=selected_stats,
        layer=layer,
        max_cells=int(global_cfg.get("max_control_cells_for_programs", 50000)),
        seed=seed,
    )
    corr = compute_gene_correlation(z)
    cluster_df = cluster_genes_from_correlation(
        corr,
        genes,
        corr_threshold=float(global_cfg.get("corr_threshold", 0.35)),
        min_correlated_partners=int(global_cfg.get("min_correlated_partners", 5)),
        cluster_method=str(global_cfg.get("cluster_method", "auto")),
        leiden_resolution=float(global_cfg.get("leiden_resolution", 1.0)),
    )
    membership = refine_program_membership(
        cluster_df,
        corr,
        genes,
        min_program_size=int(global_cfg.get("min_program_size", 10)),
        max_program_size=global_cfg.get("max_program_size", None),
        min_gene_centroid_corr=float(global_cfg.get("min_gene_centroid_corr", 0.2)),
    )
    if membership.empty:
        raise RuntimeError("No gene programs were built. Try lowering corr_threshold/min_program_size/min_correlated_partners.")
    stats_path = save_table(stats, outdir / "control_zscore_stats.csv")
    selected_path = save_table(selection_stats, outdir / "control_gene_selection_stats.csv")
    cluster_path = save_table(cluster_df, outdir / "control_gene_clustering.csv")
    membership_path = save_table(membership, outdir / "gene_program_membership.csv")
    if bool(global_cfg.get("save_corr_matrix", False)):
        np.save(outdir / "selected_gene_correlation.npy", corr)
    summary = {
        "dataset_id": dataset_cfg.get("dataset_id"),
        "control_h5ad": dataset_cfg.get("control_h5ad"),
        "n_control_cells": int(control.n_obs),
        "n_genes_total": int(control.n_vars),
        "n_genes_selected": int(len(selected_stats)),
        "n_programs": int(membership["program_id"].nunique()),
        "program_size_min": int(membership.groupby("program_id").size().min()),
        "program_size_median": float(membership.groupby("program_id").size().median()),
        "program_size_max": int(membership.groupby("program_id").size().max()),
    }
    summary_path = save_json(summary, outdir / "gene_program_build_summary.json")
    return {
        "membership": membership_path,
        "control_zscore_stats": stats_path,
        "selection_stats": selected_path,
        "clustering": cluster_path,
        "summary": summary_path,
    }
