"""Shared definitions for pseudo-pairing result aggregation and plotting.

This file is synchronized with the current result-analysis pipeline:

- `aggregate_variants.py` builds one canonical strategy-variant row and averages
  only over random seeds.
- `run_result_analysis_pipeline.py` uses `STRATEGY_PLOT_LABELS` for final bar
  plot labels.
- The final metric set is restricted to the user-requested metrics.

Important convention:
    `sampling_seed`, `run_id`, h5ad paths, metadata paths, and task-specific
    row identifiers are never strategy-variant metadata. They are only source
    row metadata and must not define or split variants.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Strategy names, order, colors, and plot labels
# -----------------------------------------------------------------------------

STRATEGY_ORDER = [
    "S0_naive_mean_control_reference",
    "S1_random_single_control",
    "S2_random_average_controls",
    "S3_SEACell_metacell_average",
    "S4_SEACell_balanced_random_sample",
    "S5_SEACell_OT_sampled_average",
]
STRATEGY_ORDER_MAP = {s: i for i, s in enumerate(STRATEGY_ORDER)}

EXPECTED_VARIANT_COUNTS = {
    "S0_naive_mean_control_reference": 1,
    "S1_random_single_control": 1,
    "S2_random_average_controls": 1,
    "S3_SEACell_metacell_average": 15,
    "S4_SEACell_balanced_random_sample": 5,
    "S5_SEACell_OT_sampled_average": 15,
}

# Inverse MLP naive-baseline source.  The runner reads this JSON when present
# and uses n_perturbation_classes to draw 1 / class_number dashed baselines.
MLP_CONFIG_SUMMARY_FILENAME = "mlp_config_summary.json"
INVERSE_CLASS_COUNT_KEYS = [
    "n_perturbation_classes",
    "test_n_classes",
    "n_classes",
    "num_classes",
    "n_labels",
]

DEFAULT_STRATEGY_RENAME_MAP = {
    # final names
    "S0_naive_mean_control_reference": "S0_naive_mean_control_reference",
    "S1_random_single_control": "S1_random_single_control",
    "S2_random_average_controls": "S2_random_average_controls",
    "S3_SEACell_metacell_average": "S3_SEACell_metacell_average",
    "S4_SEACell_balanced_random_sample": "S4_SEACell_balanced_random_sample",
    "S5_SEACell_OT_sampled_average": "S5_SEACell_OT_sampled_average",

    # compact labels
    "S0": "S0_naive_mean_control_reference",
    "S1": "S1_random_single_control",
    "S2": "S2_random_average_controls",
    "S3": "S3_SEACell_metacell_average",
    "S4": "S4_SEACell_balanced_random_sample",
    "S5": "S5_SEACell_OT_sampled_average",

    # historical labels from previous Replogle/random scripts
    "S4_random_single_control_oracle": "S1_random_single_control",
    "S4_random_single_control": "S1_random_single_control",
    "strategy4_random_single_control": "S1_random_single_control",
    "strategy4_random_single_control_cell": "S1_random_single_control",

    "S3_random_average_controls": "S2_random_average_controls",
    "strategy3_random_average_controls": "S2_random_average_controls",
    "strategy3_random_average_control_cells": "S2_random_average_controls",

    "S5_random_metacell_average": "S3_SEACell_metacell_average",
    "S3_random_metacell_average": "S3_SEACell_metacell_average",
    "strategy5_random_metacell_average": "S3_SEACell_metacell_average",

    "S1_SEACell_balanced_random": "S4_SEACell_balanced_random_sample",
    "S4_SEACell_balanced_random": "S4_SEACell_balanced_random_sample",
    "strategy1_seacell_balanced_random_repeated": "S4_SEACell_balanced_random_sample",

    "S2_SEACell_OT_topk_sampled_average": "S5_SEACell_OT_sampled_average",
    "S2_SEACell_OT_topk_sampled_average_repeated": "S5_SEACell_OT_sampled_average",
    "strategy2_seacells_ot_topk_sampled_average": "S5_SEACell_OT_sampled_average",
    "strategy2_seacell_ot_topk_sampled_average_repeated": "S5_SEACell_OT_sampled_average",
}

# Base colors by base strategy only. Parenthesized variant labels are generated
# dynamically from selected settings, not used as color keys.
STRATEGY_BASE_COLORS = {
    "S0_naive_mean_control_reference": "#BDBDBD",
    "S1_random_single_control": "#D2E5E3",
    "S2_random_average_controls": "#A1CDD8",
    "S3_SEACell_metacell_average": "#8EBCBB",
    "S4_SEACell_balanced_random_sample": "#68A6A4",
    "S5_SEACell_OT_sampled_average": "#D49AB5",
}

# Optional color pool when multiple variants from the same strategy are selected.
STRATEGY_VARIANT_COLOR_POOLS = {
    "S3_SEACell_metacell_average": ["#8EBCBB", "#74AAA9", "#5E9796", "#A8CECD"],
    "S4_SEACell_balanced_random_sample": ["#68A6A4", "#4F8F8D", "#3B7775", "#8CBDBB"],
    "S5_SEACell_OT_sampled_average": ["#D49AB5", "#B66699", "#8F3E77", "#E8B8CB"],
}

# Final bar-plot labels. The runner applies these labels and adds a variant
# suffix only when multiple selected variants from the same strategy need
# disambiguation.
STRATEGY_PLOT_LABELS = {
    "S0_naive_mean_control_reference": "Naive mean\ncontrol",
    "S1_random_single_control": "Random\nsingle\ncontrol",
    "S2_random_average_controls": "Random\naverage\ncontrol",
    "S3_SEACell_metacell_average": "Random\nmetacell\naverage",
    "S4_SEACell_balanced_random_sample": "SEACell\nbalanced\nsample",
    "S5_SEACell_OT_sampled_average": "OT sampled\naverage",
}

# Columns preserved as strategy-variant metadata. This intentionally excludes:
# sampling_seed, run_id, pseudo_control_h5ad, pair_metadata_path, membership paths.
VARIANT_META_COLUMNS = [
    "strategy_order",
    "strategy",
    "strategy_family",
    "variant_id",
    "variant_label",
    "display_variant_label",
    "seacell_setting_id",
    "n_metacells_requested",
    "n_metacells_observed",
    "n_metacells",
    "top_k_metacells",
    "top_k",
    "n_control_cells_to_average",
    "n_metacells_to_average",
    "sampled_metacells_k",
    "sample_cells_per_metacell",
    "parameter_label",
]

# Columns that must never be considered metrics.
NON_METRIC_COLUMNS = set(VARIANT_META_COLUMNS + [
    "dataset_id",
    "perturbed_group",
    "evaluation_task",
    "run_id",
    "strategy_id",
    "strategy_old",
    "sampling_seed",
    "seed",
    "pair_selection_seed",
    "pseudo_control_h5ad",
    "pair_metadata_path",
    "membership_path",
    "membership_path_for_metacell_coverage",
    "source_control_h5ad",
    "source_perturbed_h5ad",
    "inverse_input_mode",
    "select_for_final",
    "manual_color",
    "final_strategy_label",
    "plot_label",
    "plot_color",
    "display_order",
    "naive_inverse_n_classes",
    "mlp_inverse_n_classes",
    "n_perturbation_classes",
    "test_n_classes",
    "n_classes",
    "class_number",
    "n_candidate_classes",
    "candidate_class_number",
    "num_classes",
    "n_labels",
])


# -----------------------------------------------------------------------------
# Final metric specifications from the current required metric list
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricSpec:
    name: str
    sources: tuple[str, ...] = ()
    formula: str | None = None
    direction: str = "higher"  # higher, lower, target1
    label: str | None = None
    ylabel: str | None = None
    digits: int = 4


FINAL_METRIC_SPECS: dict[str, list[MetricSpec]] = {
    "control_manifold": [
        MetricSpec("mean_expression_rmse", ("mean_expression_rmse",), direction="lower", label="Mean expression RMSE", ylabel="RMSE"),
        MetricSpec("mean_expression_correlation", ("mean_expression_pearson",), direction="higher", label="Mean expression correlation", ylabel="Pearson correlation"),
        MetricSpec("variance_pearson", ("variance_pearson",), direction="higher", label="Gene-wise variance Pearson correlation", ylabel="Pearson correlation"),
        MetricSpec("control_pseudo_local_mixing_score", ("source_mixing_opposite_neighbor_fraction_mean",), direction="higher", label="Control-pseudo local mixing score", ylabel="Opposite-neighbor fraction"),
        MetricSpec("pca_centroid_distance", ("pca_mean_distance",), direction="lower", label="PCA centroid distance", ylabel="Distance"),
        MetricSpec("mmd_pca", ("mmd_rbf_pca",), direction="lower", label="MMD in PCA space", ylabel="MMD"),
    ],
    "perturbation_effect": [
        MetricSpec("perturbation_effect_rmse", ("strategy_delta_rmse_common_mean",), direction="lower", label="Perturbation effect RMSE", ylabel="RMSE"),
        MetricSpec("perturbation_effect_pearson", ("strategy_delta_pearson_common_mean",), direction="higher", label="Perturbation effect Pearson correlation", ylabel="Pearson correlation"),
        MetricSpec("perturbation_effect_magnitude_ratio", ("strategy_delta_norm_mean", "common_delta_norm_mean"), formula="ratio", direction="target1", label="Perturbation effect magnitude ratio", ylabel="Strategy / common magnitude"),
        MetricSpec("top100_perturbation_effect_correlation", ("top100_common_delta_pearson_true_genes_mean",), direction="higher", label="Top 100 perturbation effect correlation", ylabel="Pearson correlation"),
    ],
    "mlp_forward": [
        MetricSpec("input_only_mse", ("input_only_mse_xt",), direction="lower", label="Input-only MSE", ylabel="MSE"),
        MetricSpec("model_mse", ("model_mse_xt",), direction="lower", label="Model MSE", ylabel="MSE"),
        MetricSpec("absolute_mse_improvement", ("model_gain_mse_xt", "input_only_mse_xt", "model_mse_xt"), formula="absolute_improvement", direction="higher", label="Absolute MSE improvement", ylabel="Input-only MSE - model MSE"),
        MetricSpec("relative_mse_reduction", ("model_gain_mse_xt_fraction", "input_only_mse_xt", "model_mse_xt"), formula="relative_reduction", direction="higher", label="Relative MSE reduction", ylabel="Fraction"),
        MetricSpec("input_only_mae", ("input_only_mae_xt",), direction="lower", label="Input-only MAE", ylabel="MAE"),
        MetricSpec("model_mae", ("model_mae_xt",), direction="lower", label="Model MAE", ylabel="MAE"),
        MetricSpec("absolute_mae_improvement", ("model_gain_mae_xt", "input_only_mae_xt", "model_mae_xt"), formula="absolute_improvement", direction="higher", label="Absolute MAE improvement", ylabel="Input-only MAE - model MAE"),
        MetricSpec("relative_mae_reduction", ("input_only_mae_xt", "model_mae_xt"), formula="relative_reduction", direction="higher", label="Relative MAE reduction", ylabel="Fraction"),
        MetricSpec("input_common_reference_correlation", ("input_common_delta_cell_pearson_mean",), direction="higher", label="Input-only correlation on common reference", ylabel="Pearson correlation"),
        MetricSpec("model_common_reference_correlation", ("model_common_delta_cell_pearson_mean",), direction="higher", label="Model correlation on common reference", ylabel="Pearson correlation"),
        MetricSpec("absolute_correlation_improvement", ("model_common_delta_cell_pearson_mean", "input_common_delta_cell_pearson_mean"), formula="correlation_gain", direction="higher", label="Absolute correlation improvement", ylabel="Correlation gain"),
        MetricSpec("relative_correlation_improvement", ("model_common_delta_cell_pearson_mean", "input_common_delta_cell_pearson_mean"), formula="relative_correlation_gain", direction="higher", label="Relative correlation improvement", ylabel="% headroom recovered"),
    ],
    "mlp_inverse": [
        MetricSpec("test_accuracy", ("test_accuracy",), direction="higher", label="Test accuracy", ylabel="Accuracy"),
        MetricSpec("macro_f1", ("test_macro_f1",), direction="higher", label="Macro F1", ylabel="Macro F1"),
        MetricSpec("recall", ("test_macro_recall",), direction="higher", label="Recall", ylabel="Macro recall"),
        MetricSpec("precision", ("test_macro_precision",), direction="higher", label="Precision", ylabel="Macro precision"),
        MetricSpec("macro_auc", ("test_macro_auc_ovr",), direction="higher", label="AUC", ylabel="Macro AUC"),
    ],
}

FINAL_METRICS_BY_TASK = {task: [spec.name for spec in specs] for task, specs in FINAL_METRIC_SPECS.items()}

METRIC_PLOT_LABELS = {
    spec.name: (spec.label or spec.name, spec.ylabel or spec.name, spec.digits)
    for specs in FINAL_METRIC_SPECS.values()
    for spec in specs
}

METRIC_DIRECTIONS = {
    spec.name: spec.direction
    for specs in FINAL_METRIC_SPECS.values()
    for spec in specs
}


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def as_namespace(config: Mapping[str, Any] | SimpleNamespace) -> SimpleNamespace:
    return config if isinstance(config, SimpleNamespace) else SimpleNamespace(**dict(config))


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def read_table(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix in {".txt", ".tsv"} else ","
        return pd.read_csv(path, sep=sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table file type: {path}")


def safe_filename(text: Any, max_len: int = 180) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.=+&-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len]


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    sx = str(x).strip()
    return sx == "" or sx.lower() in {"nan", "none", "null", "na"}


def to_num(x: Any) -> float:
    if is_missing(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def fmt_int_like(x: Any) -> str:
    val = to_num(x)
    if np.isfinite(val):
        if abs(val - round(val)) < 1e-8:
            return str(int(round(val)))
        return f"{val:g}"
    return str(x)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def get_numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def extract_number_from_text(text: Any, patterns: Sequence[str]) -> float:
    if is_missing(text):
        return np.nan
    s = str(text)
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan


# -----------------------------------------------------------------------------
# Task inputs
# -----------------------------------------------------------------------------

@dataclass
class TaskSpec:
    task_name: str
    input_path: str | Path
    metrics: Sequence[str] | None = None
    strategy_col: str | None = None
    dataset_id: str | None = None
    perturbed_group: str | None = None
    sheet_name: str | int | None = 0
    row_filter: Mapping[str, Any] = field(default_factory=dict)
    output_prefix: str | None = None

    def prefix(self) -> str:
        return self.output_prefix or safe_filename(self.task_name)


def _first_existing_path(candidates: Sequence[Path], allow_missing: bool = True) -> Path:
    for path in candidates:
        if path.exists():
            return path
    if allow_missing:
        # Return the preferred first candidate so the runner can print a clear
        # skip-missing message.
        return candidates[0]
    raise FileNotFoundError("None of the candidate files exists:\n" + "\n".join(map(str, candidates)))


def build_default_task_inputs(
    eval_root: str | Path,
    perturbed_group: str,
    tasks: Sequence[str] | None = None,
    allow_missing: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build current evaluation-output paths.

    For control manifold, this prefers variant-resolved run-level files if they
    exist. The older pre-averaged summary is used only as a fallback, because it
    may be too coarse for S3 variant-level merging.
    """
    eval_root = Path(eval_root)
    group_root = eval_root / perturbed_group
    tasks = list(tasks or ["control_manifold", "perturbation_effect", "mlp_forward", "mlp_inverse"])

    control_dir = group_root / "control_manifold"
    perturb_dir = group_root / "perturbation_effect"
    mlp_dir = group_root / "downstream_mlp"

    specs = {
        "control_manifold": {
            "input_path": _first_existing_path([
                control_dir / "control_manifold_preservation_repeated_run_summary.csv",
                control_dir / "control_manifold_preservation_repeated_long.csv",
                control_dir / "control_manifold_preservation_repeated_summary.csv",
            ], allow_missing=allow_missing),
            "strategy_col": "strategy",
            "metrics": FINAL_METRICS_BY_TASK["control_manifold"],
        },
        "perturbation_effect": {
            "input_path": _first_existing_path([
                perturb_dir / "perturbation_effect_consistency_repeated_run_summary.csv",
                perturb_dir / "perturbation_effect_consistency_repeated_summary.csv",
            ], allow_missing=allow_missing),
            "strategy_col": "strategy",
            "metrics": FINAL_METRICS_BY_TASK["perturbation_effect"],
        },
        "mlp_forward": {
            "input_path": _first_existing_path([
                mlp_dir / "forward_mlp_run_summary.csv",
                mlp_dir / "forward_mlp_repeated_run_summary.csv",
            ], allow_missing=allow_missing),
            "strategy_col": "strategy_id",
            "metrics": FINAL_METRICS_BY_TASK["mlp_forward"],
            "mlp_config_summary_path": mlp_dir / MLP_CONFIG_SUMMARY_FILENAME,
        },
        "mlp_inverse": {
            "input_path": _first_existing_path([
                mlp_dir / "inverse_mlp_run_summary.csv",
                mlp_dir / "inverse_mlp_repeated_strategy_delta_classification_summary.csv",
                mlp_dir / "inverse_mlp_repeated_run_summary.csv",
            ], allow_missing=allow_missing),
            "strategy_col": "strategy_id",
            "metrics": FINAL_METRICS_BY_TASK["mlp_inverse"],
            "row_filter": {"inverse_input_mode": "strategy_delta"},
            "mlp_config_summary_path": mlp_dir / MLP_CONFIG_SUMMARY_FILENAME,
        },
    }

    out = {}
    for task in tasks:
        if task not in specs:
            raise KeyError(f"Unknown task: {task}")
        path = Path(specs[task]["input_path"])
        if path.exists() or allow_missing:
            out[task] = specs[task]
        else:
            raise FileNotFoundError(path)
    return out
