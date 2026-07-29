from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import anndata as ad
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .discovery import pseudo_file_index_from_dataset_cfg
from .metrics import matrix_metrics, per_perturbation_correlations, per_program_correlations, summarize_correlation_table
from .programs import build_gene_programs_for_dataset
from .scoring import group_mean_scores, load_programs, score_adata_programs
from .utils import ensure_dir, read_table, save_json, save_table


def _get_labels(adata: ad.AnnData, perturbation_key: str) -> pd.Series:
    if perturbation_key not in adata.obs.columns:
        candidates = ["perturbation_label", "condition", "gene", "target", "perturbation", "guide_identity"]
        found = next((c for c in candidates if c in adata.obs.columns), None)
        if found is None:
            raise KeyError(f"Could not find perturbation key {perturbation_key!r} or any fallback in AnnData.obs")
        perturbation_key = found
    return adata.obs[perturbation_key].astype(str)


def _write_effect_matrix(df: pd.DataFrame, path: Path, index_name: str = "perturbation") -> Path:
    out = df.copy()
    out.index.name = index_name
    return save_table(out.reset_index(), path)


def _read_effect_matrix(path: Path) -> pd.DataFrame:
    df = read_table(path)
    first = df.columns[0]
    return df.set_index(first)


def build_or_load_gene_programs(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any], force_build: bool = False) -> Path:
    membership_path = Path(dataset_cfg["output_dir"]) / "gene_programs" / "gene_program_membership.csv"
    if force_build or not membership_path.exists():
        build_gene_programs_for_dataset(dataset_cfg, global_cfg)
    if not membership_path.exists():
        raise FileNotFoundError(membership_path)
    return membership_path


def compute_true_effects(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any], membership_path: Path) -> dict[str, Path]:
    out_effect = ensure_dir(Path(dataset_cfg["output_dir"]) / "effects")
    out_scores = ensure_dir(Path(dataset_cfg["output_dir"]) / "scores")
    perturbation_key = str(global_cfg.get("perturbation_key", "perturbation_label"))
    layer = global_cfg.get("layer", None)
    chunk_size = int(global_cfg.get("chunk_size", 20000))
    programs = load_programs(membership_path)
    stats = read_table(Path(dataset_cfg["output_dir"]) / "gene_programs" / "control_zscore_stats.csv")

    control = ad.read_h5ad(dataset_cfg["control_h5ad"])
    perturbed = ad.read_h5ad(dataset_cfg["perturbed_h5ad"])
    control_scores = score_adata_programs(control, programs, stats, layer=layer, chunk_size=chunk_size)
    pert_scores = score_adata_programs(perturbed, programs, stats, layer=layer, chunk_size=chunk_size)

    control_mean = pd.DataFrame([control_scores.mean(axis=0)], index=["true_control"])
    pert_labels = _get_labels(perturbed, perturbation_key)
    pert_mean = group_mean_scores(pert_scores, pert_labels)
    true_effect = pert_mean.subtract(control_scores.mean(axis=0), axis=1)

    paths = {
        "true_control_program_expression": _write_effect_matrix(control_mean, out_effect / "true_control_program_expression.csv", index_name="group"),
        "true_perturbed_program_expression": _write_effect_matrix(pert_mean, out_effect / "true_perturbed_program_expression.csv"),
        "true_program_effects": _write_effect_matrix(true_effect, out_effect / "true_program_effects.csv"),
    }
    if bool(global_cfg.get("save_cell_level_scores", False)):
        save_table(control_scores.reset_index(), out_scores / "true_control_cell_program_scores.csv")
        tmp = pert_scores.reset_index()
        tmp.insert(1, perturbation_key, pert_labels.values)
        save_table(tmp, out_scores / "true_perturbed_cell_program_scores.csv")
    return paths


def evaluate_pseudo_files(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any], membership_path: Path) -> dict[str, Path]:
    output_dir = Path(dataset_cfg["output_dir"])
    out_effect = ensure_dir(output_dir / "effects")
    out_metrics = ensure_dir(output_dir / "metrics")
    out_scores = ensure_dir(output_dir / "scores")
    perturbation_key = str(global_cfg.get("perturbation_key", "perturbation_label"))
    layer = global_cfg.get("layer", None)
    chunk_size = int(global_cfg.get("chunk_size", 20000))
    programs = load_programs(membership_path)
    stats = read_table(output_dir / "gene_programs" / "control_zscore_stats.csv")
    true_effect = _read_effect_matrix(out_effect / "true_program_effects.csv")
    true_pert_mean = _read_effect_matrix(out_effect / "true_perturbed_program_expression.csv")
    pseudo_index = pseudo_file_index_from_dataset_cfg(dataset_cfg, global_cfg)
    if pseudo_index.empty:
        raise RuntimeError(f"No pseudo-control files found for {dataset_cfg.get('dataset_id')}")
    save_table(pseudo_index, out_metrics / "pseudo_file_index.csv")

    metric_rows = []
    per_prog_rows = []
    per_pert_rows = []

    for row_i, row in pseudo_index.iterrows():
        pseudo_path = Path(str(row["pseudo_control_h5ad"]))
        if not pseudo_path.exists():
            print(f"[Skip missing pseudo h5ad] {pseudo_path}")
            continue
        pseudo = ad.read_h5ad(pseudo_path)
        pseudo_scores = score_adata_programs(pseudo, programs, stats, layer=layer, chunk_size=chunk_size)
        try:
            pseudo_labels = _get_labels(pseudo, perturbation_key)
        except Exception:
            # If pseudo obs lacks perturbation label but has same row order as perturbed, reuse perturbed labels.
            perturbed = ad.read_h5ad(dataset_cfg["perturbed_h5ad"])
            pseudo_labels = _get_labels(perturbed, perturbation_key).iloc[: pseudo.n_obs].copy()
            pseudo_labels.index = pseudo_scores.index
        pseudo_mean = group_mean_scores(pseudo_scores, pseudo_labels)
        common = true_pert_mean.index.intersection(pseudo_mean.index)
        pseudo_effect = true_pert_mean.loc[common].subtract(pseudo_mean.loc[common], axis=0)

        vid = str(row.get("variant_id", f"pseudo_{row_i:04d}"))
        safe_vid = vid.replace("/", "_").replace(" ", "_")
        _write_effect_matrix(pseudo_mean, out_effect / f"pseudo_control_program_expression__{safe_vid}.csv")
        _write_effect_matrix(pseudo_effect, out_effect / f"pseudo_program_effects__{safe_vid}.csv")
        if bool(global_cfg.get("save_cell_level_scores", False)):
            tmp = pseudo_scores.reset_index()
            tmp.insert(1, perturbation_key, pseudo_labels.values)
            save_table(tmp, out_scores / f"pseudo_cell_program_scores__{safe_vid}.csv")

        m = matrix_metrics(true_effect, pseudo_effect)
        meta = row.to_dict()
        metric_rows.append({**meta, **m})
        pp = per_program_correlations(true_effect, pseudo_effect)
        pp.insert(0, "variant_id", vid)
        pp.insert(1, "strategy", row.get("strategy", np.nan))
        per_prog_rows.append(pp)
        pt = per_perturbation_correlations(true_effect, pseudo_effect)
        pt.insert(0, "variant_id", vid)
        pt.insert(1, "strategy", row.get("strategy", np.nan))
        per_pert_rows.append(pt)
        metric_rows[-1].update(summarize_correlation_table(pp, "per_program"))
        metric_rows[-1].update(summarize_correlation_table(pt, "per_perturbation"))
        print(f"[Evaluated] {vid}: RMSE={metric_rows[-1]['rmse']:.4g}, Pearson={metric_rows[-1]['pearson']:.4g}")

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        sort_cols = [c for c in ["strategy_order", "strategy", "variant_id", "sampling_seed"] if c in metrics_df.columns]
        if sort_cols:
            metrics_df = metrics_df.sort_values(sort_cols, na_position="last")
    metrics_path = save_table(metrics_df, out_metrics / "gene_program_metrics_by_variant_seed_level.csv")

    # Variant-level seed average table for direct result-analysis use.
    if not metrics_df.empty and "variant_id" in metrics_df.columns:
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        drop_numeric = {"sampling_seed"}
        numeric_cols = [c for c in numeric_cols if c not in drop_numeric]
        group_cols = [c for c in ["strategy", "variant_id", "display_variant_label"] if c in metrics_df.columns]
        avg = metrics_df.groupby(group_cols, dropna=False)[numeric_cols].mean().reset_index()
        count = metrics_df.groupby(group_cols, dropna=False).size().reset_index(name="n_seed_records")
        avg = avg.merge(count, on=group_cols, how="left")
        avg_path = save_table(avg, out_metrics / "gene_program_metrics_by_variant.csv")
    else:
        avg_path = save_table(pd.DataFrame(), out_metrics / "gene_program_metrics_by_variant.csv")

    per_prog_path = save_table(pd.concat(per_prog_rows, ignore_index=True) if per_prog_rows else pd.DataFrame(), out_metrics / "per_program_correlations.csv")
    per_pert_path = save_table(pd.concat(per_pert_rows, ignore_index=True) if per_pert_rows else pd.DataFrame(), out_metrics / "per_perturbation_correlations.csv")
    summary_path = save_json({
        "dataset_id": dataset_cfg.get("dataset_id"),
        "n_pseudo_files_indexed": int(pseudo_index.shape[0]),
        "n_pseudo_files_evaluated": int(len(metric_rows)),
        "metrics_seed_level": str(metrics_path),
        "metrics_variant_level": str(avg_path),
        "per_program_correlations": str(per_prog_path),
        "per_perturbation_correlations": str(per_pert_path),
    }, out_metrics / "evaluation_summary.json")
    return {"metrics": metrics_path, "metrics_variant_level": avg_path, "per_program": per_prog_path, "per_perturbation": per_pert_path, "summary": summary_path}


def run_dataset(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any], run_build: bool, run_evaluate: bool) -> dict[str, Any]:
    output_dir = ensure_dir(dataset_cfg["output_dir"])
    print("\n" + "=" * 120)
    print(f"[Gene-program dataset] {dataset_cfg.get('dataset_id')}")
    print(f"[Output] {output_dir}")
    print("=" * 120)
    membership_path = build_or_load_gene_programs(dataset_cfg, global_cfg, force_build=run_build)
    outputs: dict[str, Any] = {"dataset_id": dataset_cfg.get("dataset_id"), "membership_path": str(membership_path)}
    if run_evaluate:
        compute_true_effects(dataset_cfg, global_cfg, membership_path)
        eval_outputs = evaluate_pseudo_files(dataset_cfg, global_cfg, membership_path)
        outputs.update({k: str(v) for k, v in eval_outputs.items()})
    return outputs
