"""Perturbation-effect consistency evaluation for repeated pseudo-control datasets."""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from tqdm.auto import tqdm

from eval_common import (
    as_namespace,
    ensure_dir,
    get_config,
    load_run_manifest,
    mean_axis0,
    metadata_from_manifest_row,
    safe_cosine,
    safe_pearson,
    safe_spearman,
    save_json,
    select_eval_genes,
    summarize_numeric,
    to_csr,
)


def group_mean_from_matrix(X, labels: np.ndarray, target_labels: Sequence[str]) -> dict[str, np.ndarray]:
    out = {}
    labels = np.asarray(labels).astype(str)
    for lab in target_labels:
        idx = np.where(labels == str(lab))[0]
        if len(idx) == 0:
            continue
        out[str(lab)] = mean_axis0(X[idx, :]).astype(np.float64)
    return out


def topk_abs_indices(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x)
    k = min(int(k), x.size)
    if k <= 0:
        return np.array([], dtype=int)
    return np.argsort(np.abs(x))[-k:]


def sign_agreement(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.sign(a[mask]) == np.sign(b[mask])))


def compute_topk_metrics(strategy_delta: np.ndarray, common_delta: np.ndarray, topk_list: Sequence[int], prefix: str) -> dict[str, Any]:
    rec = {}
    abs_common = np.abs(common_delta)
    abs_strategy = np.abs(strategy_delta)
    for k in topk_list:
        k_eff = min(int(k), len(common_delta))
        true_idx = np.argsort(abs_common)[-k_eff:]
        pred_idx = np.argsort(abs_strategy)[-k_eff:]
        rec[f"top{k_eff}_{prefix}_pearson_true_genes"] = safe_pearson(strategy_delta[true_idx], common_delta[true_idx])
        rec[f"top{k_eff}_{prefix}_spearman_true_genes"] = safe_spearman(strategy_delta[true_idx], common_delta[true_idx])
        rec[f"top{k_eff}_{prefix}_cosine_true_genes"] = safe_cosine(strategy_delta[true_idx], common_delta[true_idx])
        rec[f"top{k_eff}_{prefix}_sign_agreement_true_genes"] = sign_agreement(strategy_delta[true_idx], common_delta[true_idx])
        rec[f"top{k_eff}_{prefix}_overlap_fraction"] = float(len(set(true_idx.tolist()).intersection(set(pred_idx.tolist()))) / max(k_eff, 1))
    return rec


def summarize_metric_columns(df: pd.DataFrame, metric_cols: Sequence[str]) -> dict[str, Any]:
    rec = {}
    for c in metric_cols:
        if c not in df.columns:
            continue
        arr = df[c].astype(float).values
        rec[f"{c}_mean"] = float(np.nanmean(arr)) if arr.size else np.nan
        rec[f"{c}_median"] = float(np.nanmedian(arr)) if arr.size else np.nan
        rec[f"{c}_std"] = float(np.nanstd(arr)) if arr.size else np.nan
        rec[f"{c}_sem"] = float(np.nanstd(arr) / np.sqrt(np.sum(np.isfinite(arr)))) if np.sum(np.isfinite(arr)) > 0 else np.nan
        rec[f"{c}_n_valid"] = int(np.sum(np.isfinite(arr)))
    return rec


def evaluate_one_effect_run(row: pd.Series, X_control, Xt, perturbed_labels: np.ndarray, eligible_labels: Sequence[str],
                            eval_genes: Sequence[str], control_mean: np.ndarray, perturbed_group_means: dict[str, np.ndarray], config) -> tuple[pd.DataFrame, dict[str, Any]]:
    pseudo = sc.read_h5ad(row["pseudo_control_h5ad"])
    pseudo.obs_names_make_unique()
    missing = pd.Index(eval_genes).difference(pd.Index(pseudo.var_names.astype(str)))
    if len(missing) > 0:
        raise ValueError(f"Missing eval genes in {row['pseudo_control_h5ad']}: {missing[:5].tolist()}")
    X0 = to_csr(pseudo[:, eval_genes].X)
    labels = np.asarray(perturbed_labels).astype(str)

    topk_list = get_config(config, "top_de_k_list", [20, 50, 100])
    records = []
    for lab in eligible_labels:
        idx = np.where(labels == str(lab))[0]
        if len(idx) == 0 or lab not in perturbed_group_means:
            continue
        pseudo_mean = mean_axis0(X0[idx, :]).astype(np.float64)
        true_mean = perturbed_group_means[lab]
        strategy_delta = true_mean - pseudo_mean
        common_delta = true_mean - control_mean
        rec = metadata_from_manifest_row(row)
        rec.update({
            "perturbation_label": str(lab),
            "n_cells": int(len(idx)),
            "strategy_delta_pearson_common": safe_pearson(strategy_delta, common_delta),
            "strategy_delta_spearman_common": safe_spearman(strategy_delta, common_delta),
            "strategy_delta_cosine_common": safe_cosine(strategy_delta, common_delta),
            "strategy_delta_rmse_common": float(np.sqrt(np.mean((strategy_delta - common_delta) ** 2))),
            "strategy_delta_mae_common": float(np.mean(np.abs(strategy_delta - common_delta))),
            "strategy_delta_norm": float(np.sqrt(np.mean(strategy_delta ** 2))),
            "common_delta_norm": float(np.sqrt(np.mean(common_delta ** 2))),
        })
        rec.update(compute_topk_metrics(strategy_delta, common_delta, topk_list, prefix="common_delta"))
        records.append(rec)
    per_pert = pd.DataFrame(records)
    metric_cols = [
        c for c in per_pert.columns
        if c not in {"perturbation_label", "run_id", "strategy", "strategy_id", "strategy_family", "parameter_label"}
        and pd.api.types.is_numeric_dtype(per_pert[c])
    ]
    summary = metadata_from_manifest_row(row)
    summary.update({"n_perturbations_evaluated": int(per_pert.shape[0]), "n_eval_genes": int(len(eval_genes))})
    summary.update(summarize_metric_columns(per_pert, metric_cols))
    del pseudo, X0
    gc.collect()
    return per_pert, summary


def run_perturbation_effect_evaluation(config: Mapping[str, Any]) -> dict[str, Path]:
    config = as_namespace(config)
    outdir = ensure_dir(Path(get_config(config, "outdir")) / "perturbation_effect")
    runs_df = load_run_manifest(config)
    runs_df.to_csv(outdir / "discovered_run_manifest.csv", index=False)

    control = sc.read_h5ad(str(get_config(config, "control_h5ad")))
    control.obs_names_make_unique()
    perturbed = sc.read_h5ad(str(get_config(config, "perturbed_h5ad")))
    perturbed.obs_names_make_unique()
    key = get_config(config, "perturbation_key", "perturbation_key")
    if key not in perturbed.obs.columns:
        raise KeyError(f"perturbation_key={key!r} not found in perturbed.obs. Available: {list(perturbed.obs.columns)}")

    eval_genes = select_eval_genes(control, perturbed, runs_df, max_eval_genes=get_config(config, "max_eval_genes", 3000),
                                   check_shared_genes_across_all_runs=bool(get_config(config, "check_shared_genes_across_all_runs", False)))
    pd.Series(eval_genes).to_csv(outdir / "eval_genes.csv", index=False, header=["gene"])
    X_control = to_csr(control[:, eval_genes].X)
    Xt = to_csr(perturbed[:, eval_genes].X)
    control_mean = mean_axis0(X_control).astype(np.float64)

    labels = perturbed.obs[key].astype(str).values
    counts = pd.Series(labels).value_counts()
    eligible = sorted(counts[counts >= int(get_config(config, "min_cells_per_perturbation", 20))].index.astype(str).tolist())
    perturbed_group_means = group_mean_from_matrix(Xt, labels, eligible)

    all_per = []
    run_summaries = []
    per_partial = outdir / "perturbation_effect_consistency_repeated_per_perturbation_PARTIAL.csv"
    summary_partial = outdir / "perturbation_effect_consistency_repeated_run_summary_PARTIAL.csv"
    for _, row in tqdm(runs_df.iterrows(), total=runs_df.shape[0], desc="Perturbation-effect evaluation"):
        per, summ = evaluate_one_effect_run(row, X_control, Xt, labels, eligible, eval_genes, control_mean, perturbed_group_means, config)
        all_per.append(per)
        run_summaries.append(summ)
        pd.concat(all_per, ignore_index=True).to_csv(per_partial, index=False)
        pd.DataFrame(run_summaries).to_csv(summary_partial, index=False)

    per_df = pd.concat(all_per, ignore_index=True) if all_per else pd.DataFrame()
    run_df = pd.DataFrame(run_summaries)
    per_path = outdir / "perturbation_effect_consistency_repeated_per_perturbation.csv"
    run_path = outdir / "perturbation_effect_consistency_repeated_run_summary.csv"
    per_df.to_csv(per_path, index=False)
    run_df.to_csv(run_path, index=False)
    summary_path = outdir / "perturbation_effect_consistency_repeated_summary.csv"
    summarize_numeric(run_df, group_cols=["perturbed_group", "strategy", "strategy_family", "parameter_label"], output_path=summary_path)
    save_json({"n_runs": len(runs_df), "n_eval_genes": len(eval_genes), "n_eligible_perturbations": len(eligible)}, outdir / "perturbation_effect_config_summary.json")
    return {"per_perturbation": per_path, "run_summary": run_path, "summary": summary_path}
