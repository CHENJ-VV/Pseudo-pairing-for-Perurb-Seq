"""Seed-only aggregation for pseudo-pairing evaluation outputs.

This file is the critical place for controlling row counts.

The canonical variant key is:
    strategy + strategy-defining settings only

It never uses:
    run_id, sampling_seed, pair path, task name, or row index

Therefore the editable selected_variants_TEMPLATE_EDIT_ME.csv should contain
one row per strategy variant, not one row per seed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .common import (
    DEFAULT_STRATEGY_RENAME_MAP,
    FINAL_METRIC_SPECS,
    FINAL_METRICS_BY_TASK,
    STRATEGY_ORDER_MAP,
    VARIANT_META_COLUMNS,
    TaskSpec,
    ensure_dir,
    extract_number_from_text,
    fmt_int_like,
    get_numeric_series,
    is_missing,
    read_table,
)


# -----------------------------------------------------------------------------
# Row-level metadata: never variant-defining, never shown in selection template
# -----------------------------------------------------------------------------

ROW_LEVEL_META_COLUMNS = {
    "run_id",
    "sampling_seed",
    "seed",
    "pair_selection_seed",
    "pseudo_control_h5ad",
    "pair_metadata_path",
    "membership_path_for_metacell_coverage",
    "membership_path",
    "output_h5ad",
    "outdir",
    "assignment_path",
}

# Variant-level metadata to keep in the seed-averaged tables.  This deliberately
# excludes sampling_seed/run_id/path columns.  n_runs and n_random_seeds are added
# after aggregation.
OUTPUT_VARIANT_META_COLUMNS = [
    "dataset_id",
    "perturbed_group",
    "strategy_order",
    "strategy",
    "strategy_old",
    "strategy_id",
    "strategy_family",
    "variant_id",
    "variant_label",
    "display_variant_label",
    "parameter_label",
    "seacell_setting_id",
    "n_metacells",
    "n_metacells_requested",
    "n_metacells_observed",
    "top_k",
    "top_k_metacells",
    "sampled_metacells_k",
    "n_metacells_to_average",
    "n_control_cells_to_average",
    "sample_cells_per_metacell",
    "naive_inverse_n_classes",
    "mlp_inverse_n_classes",
    "n_perturbation_classes",
    "test_n_classes",
    "n_classes",
    "num_classes",
    "n_labels",
]

# Columns to show in selected_variants_TEMPLATE_EDIT_ME.csv before metrics.
SELECTION_BASE_COLUMNS = [
    "dataset_id",
    "perturbed_group",
    "strategy_order",
    "strategy",
    "strategy_family",
    "variant_id",
    "variant_label",
    "display_variant_label",
    "parameter_label",
    "seacell_setting_id",
    "n_metacells",
    "n_metacells_requested",
    "n_metacells_observed",
    "top_k",
    "top_k_metacells",
    "sampled_metacells_k",
    "n_metacells_to_average",
    "n_control_cells_to_average",
    "sample_cells_per_metacell",
    "n_runs",
    "n_random_seeds",
    "naive_inverse_n_classes",
    "mlp_inverse_n_classes",
    "n_perturbation_classes",
    "test_n_classes",
    "n_classes",
    "num_classes",
    "n_labels",
]


# -----------------------------------------------------------------------------
# Canonical strategy/variant parsing
# -----------------------------------------------------------------------------

def _infer_strategy_column(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None and requested in df.columns:
        return requested
    for col in ["strategy", "strategy_id", "pairing_strategy"]:
        if col in df.columns:
            return col
    raise KeyError(f"Could not infer strategy column. Available columns: {list(df.columns)}")


def _apply_row_filter(df: pd.DataFrame, row_filter: Mapping[str, Any] | None) -> pd.DataFrame:
    if not row_filter:
        return df
    out = df.copy()
    for col, value in row_filter.items():
        if col not in out.columns:
            continue
        if isinstance(value, (list, tuple, set)):
            out = out[out[col].astype(str).isin([str(v) for v in value])].copy()
        else:
            out = out[out[col].astype(str) == str(value)].copy()
    return out


def _fill_numeric_from_candidates(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> None:
    if target not in df.columns:
        df[target] = np.nan
    df[target] = pd.to_numeric(df[target], errors="coerce")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            df[target] = df[target].where(df[target].notna(), vals)


def _fill_from_text_patterns(df: pd.DataFrame, target: str, text_cols: Sequence[str], patterns: Sequence[str]) -> None:
    if target not in df.columns:
        df[target] = np.nan
    df[target] = pd.to_numeric(df[target], errors="coerce")
    for col in text_cols:
        if col not in df.columns:
            continue
        extracted = df[col].map(lambda x: extract_number_from_text(x, patterns))
        df[target] = df[target].where(df[target].notna(), extracted)


def _mode_or_first(values: pd.Series) -> Any:
    vals = values.dropna()
    if len(vals) == 0:
        return np.nan
    mode = vals.mode(dropna=True)
    if len(mode) > 0:
        return mode.iloc[0]
    return vals.iloc[0]


def canonicalize_input_table(
    raw: pd.DataFrame,
    task_spec: TaskSpec,
    strategy_rename_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return original rows with canonical strategy, variant, and metric fields.

    This function may receive true run-level tables or pre-aggregated tables.
    It does not aggregate yet.  It only creates stable canonical columns.
    """
    df = raw.copy()
    df = _apply_row_filter(df, task_spec.row_filter)
    if df.empty:
        return df

    rename_map = dict(DEFAULT_STRATEGY_RENAME_MAP)
    if strategy_rename_map:
        rename_map.update(strategy_rename_map)

    sc_col = _infer_strategy_column(df, task_spec.strategy_col)
    df["strategy_old"] = df[sc_col].astype(str)
    df["strategy"] = df["strategy_old"].map(rename_map).fillna(df["strategy_old"])
    df["strategy_order"] = df["strategy"].map(STRATEGY_ORDER_MAP).fillna(99).astype(int)

    if task_spec.dataset_id is not None and "dataset_id" not in df.columns:
        df.insert(0, "dataset_id", task_spec.dataset_id)
    if task_spec.perturbed_group is not None and "perturbed_group" not in df.columns:
        df.insert(1 if "dataset_id" in df.columns else 0, "perturbed_group", task_spec.perturbed_group)
    df["evaluation_task"] = task_spec.task_name

    # Harmonize metacell/top-k/sampled-k fields.
    _fill_numeric_from_candidates(df, "n_metacells", ["n_metacells_requested", "n_metacells_observed"])
    _fill_numeric_from_candidates(df, "n_metacells_requested", ["n_metacells"])
    _fill_numeric_from_candidates(df, "n_metacells_observed", ["n_metacells"])
    _fill_numeric_from_candidates(df, "top_k", ["top_k_metacells"])
    _fill_numeric_from_candidates(df, "top_k_metacells", ["top_k"])
    _fill_numeric_from_candidates(df, "sampled_metacells_k", ["n_metacells_to_average"])
    _fill_numeric_from_candidates(df, "n_metacells_to_average", ["sampled_metacells_k"])
    _fill_numeric_from_candidates(df, "n_control_cells_to_average", [])
    _fill_numeric_from_candidates(df, "sample_cells_per_metacell", [])

    text_cols = [c for c in ["parameter_label", "seacell_setting_id", "run_id", "strategy_old"] if c in df.columns]

    # nmc and topk are safe and strategy-defining for metacell strategies.
    _fill_from_text_patterns(df, "n_metacells", text_cols, [r"nmc[_=-]?(\d+)", r"metacell[s]?[_=-]?(\d+)"])
    _fill_from_text_patterns(df, "n_metacells_requested", text_cols, [r"nmc[_=-]?(\d+)"])
    _fill_from_text_patterns(df, "top_k", text_cols, [r"topk[_=-]?(\d+)", r"top_k[_=-]?(\d+)"])
    _fill_from_text_patterns(df, "top_k_metacells", text_cols, [r"topk[_=-]?(\d+)", r"top_k[_=-]?(\d+)"])

    # sampled_metacells_k only affects S3.  Do not let a generic "k_..." from
    # random average or top-k names split other strategies.
    _fill_from_text_patterns(
        df,
        "sampled_metacells_k",
        text_cols,
        [r"sampledMC[_=-]?(\d+)", r"sampled_metacells[_=-]?(\d+)", r"(?:^|__)k[_=-]?(\d+)"],
    )
    _fill_from_text_patterns(
        df,
        "n_metacells_to_average",
        text_cols,
        [r"sampledMC[_=-]?(\d+)", r"sampled_metacells[_=-]?(\d+)", r"(?:^|__)k[_=-]?(\d+)"],
    )

    if "seacell_setting_id" not in df.columns:
        df["seacell_setting_id"] = np.nan
    df["seacell_setting_id"] = df.apply(_make_seacell_setting_id, axis=1)

    if "parameter_label" not in df.columns:
        df["parameter_label"] = ""

    # Create canonical strategy variant identity.
    df["variant_label"] = df.apply(make_variant_label, axis=1)
    df["display_variant_label"] = df.apply(make_display_variant_label, axis=1)
    df["variant_id"] = df.apply(make_variant_id, axis=1)

    # Normalize seed only for counting; it will not enter variant_id or selection.
    if "sampling_seed" in df.columns:
        df["sampling_seed"] = pd.to_numeric(df["sampling_seed"], errors="coerce")

    # Normalize inverse class-count columns when present.  These are metadata for
    # drawing naive inverse baselines, never variant-defining settings.
    for col in [
        "naive_inverse_n_classes",
        "mlp_inverse_n_classes",
        "n_perturbation_classes",
        "test_n_classes",
        "n_classes",
        "num_classes",
        "n_labels",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _make_seacell_setting_id(row: Mapping[str, Any]) -> Any:
    existing = row.get("seacell_setting_id", np.nan)
    if not is_missing(existing):
        return existing
    strategy = str(row.get("strategy", ""))
    nmc = row.get("n_metacells", np.nan)
    if strategy in {
        "S3_SEACell_metacell_average",
        "S4_SEACell_balanced_random_sample",
        "S5_SEACell_OT_sampled_average",
    } and not is_missing(nmc):
        return f"nmc_{fmt_int_like(nmc)}"
    return np.nan


def make_variant_label(row: Mapping[str, Any]) -> str:
    """Canonical variant label.

    Expected complete output:
        S0: 1 row
        S1: 1 row
        S2: 1 row
        S3: n_metacells × sampled_metacells_k = 15 rows
        S4: n_metacells = 5 rows
        S5: n_metacells × top_k = 15 rows
    """
    strategy = str(row.get("strategy", ""))
    nmc = row.get("n_metacells", np.nan)
    topk = row.get("top_k", np.nan)
    sampled = row.get("sampled_metacells_k", np.nan)

    if strategy in {
        "S0_naive_mean_control_reference",
        "S1_random_single_control",
        "S2_random_average_controls",
    }:
        # S2 must be one record in the current design.  k/control-cell-count
        # settings are metadata only and do not split S2 into multiple variants.
        return "default"

    if strategy == "S3_SEACell_metacell_average":
        parts = []
        if not is_missing(nmc):
            parts.append(f"nmc_{fmt_int_like(nmc)}")
        if not is_missing(sampled):
            parts.append(f"sampledMC_{fmt_int_like(sampled)}")
        return "__".join(parts) if parts else "default"

    if strategy == "S4_SEACell_balanced_random_sample":
        return f"nmc_{fmt_int_like(nmc)}" if not is_missing(nmc) else "default"

    if strategy == "S5_SEACell_OT_sampled_average":
        parts = []
        if not is_missing(nmc):
            parts.append(f"nmc_{fmt_int_like(nmc)}")
        if not is_missing(topk):
            parts.append(f"topk_{fmt_int_like(topk)}")
        return "__".join(parts) if parts else "default"

    return "default"


def make_display_variant_label(row: Mapping[str, Any]) -> str:
    strategy = str(row.get("strategy", ""))
    nmc = row.get("n_metacells", np.nan)
    topk = row.get("top_k", np.nan)
    sampled = row.get("sampled_metacells_k", np.nan)

    if strategy == "S3_SEACell_metacell_average" and not is_missing(nmc) and not is_missing(sampled):
        return f"{strategy} ({fmt_int_like(nmc)}&{fmt_int_like(sampled)})"
    if strategy == "S4_SEACell_balanced_random_sample" and not is_missing(nmc):
        return f"{strategy} ({fmt_int_like(nmc)})"
    if strategy == "S5_SEACell_OT_sampled_average" and not is_missing(nmc) and not is_missing(topk):
        return f"{strategy} ({fmt_int_like(nmc)}&{fmt_int_like(topk)})"
    return strategy


def make_variant_id(row: Mapping[str, Any]) -> str:
    strategy = str(row.get("strategy", ""))
    label = make_variant_label(row)
    return strategy if label == "default" else f"{strategy}__{label}"


# -----------------------------------------------------------------------------
# Metric derivation
# -----------------------------------------------------------------------------

def _source_value(df: pd.DataFrame, name: str) -> pd.Series:
    """Return source metric value per row.

    Accepts raw metric columns such as `model_mse_xt` and pre-summary columns
    such as `model_mse_xt_mean`.
    """
    if name in df.columns:
        return get_numeric_series(df, name)
    if f"{name}_mean" in df.columns:
        return get_numeric_series(df, f"{name}_mean")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _source_std(df: pd.DataFrame, name: str) -> pd.Series:
    if f"{name}_std" in df.columns:
        return get_numeric_series(df, f"{name}_std")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _source_count(df: pd.DataFrame, name: str) -> pd.Series:
    for suffix in ["_n", "_count", "_n_valid"]:
        col = f"{name}{suffix}"
        if col in df.columns:
            return get_numeric_series(df, col)
    return pd.Series(np.nan, index=df.index, dtype="float64")


def derive_required_metrics(df: pd.DataFrame, task_name: str, strict: bool = True) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """Add standardized required metric columns to df."""
    out = df.copy()
    source_report: dict[str, dict[str, str]] = {}
    specs = FINAL_METRIC_SPECS[task_name]

    for spec in specs:
        source_report[spec.name] = {"formula": spec.formula or "direct", "sources": ",".join(spec.sources)}

        if spec.formula is None:
            src = spec.sources[0]
            vals = _source_value(out, src)

        elif spec.formula == "ratio":
            a = _source_value(out, spec.sources[0])
            b = _source_value(out, spec.sources[1])
            vals = a / b.replace(0, np.nan)

        elif spec.formula == "absolute_improvement":
            gain = _source_value(out, spec.sources[0]) if spec.sources else pd.Series(np.nan, index=out.index)
            if gain.notna().any():
                vals = gain
            else:
                vals = _source_value(out, spec.sources[1]) - _source_value(out, spec.sources[2])

        elif spec.formula == "relative_reduction":
            explicit = _source_value(out, spec.sources[0])
            if spec.sources[0].endswith("fraction") and explicit.notna().any():
                vals = explicit
            else:
                if len(spec.sources) >= 3 and spec.sources[0].endswith("fraction"):
                    input_v = _source_value(out, spec.sources[1])
                    model_v = _source_value(out, spec.sources[2])
                else:
                    input_v = _source_value(out, spec.sources[0])
                    model_v = _source_value(out, spec.sources[1])
                vals = (input_v - model_v) / input_v.replace(0, np.nan)
        
        elif spec.formula == "correlation_gain":
            vals = _source_value(out, spec.sources[0]) - _source_value(out, spec.sources[1])

        elif spec.formula == "relative_correlation_gain":
            model = _source_value(out, spec.sources[0])
            inp = _source_value(out, spec.sources[1])

            # Normalized correlation improvement:
            #
            # A direct relative change such as
            #     (model correlation - input correlation) / input correlation
            # is inappropriate because the input correlation can be close to zero
            # or even negative.
            #
            # Instead, calculate the fraction of available correlation headroom
            # recovered by the model:
            #
            #     100 * (model correlation - input correlation)
            #           / (1 - input correlation)
            #
            # 100% means the model reaches a correlation of 1.
            # 0% means no improvement over the input-only baseline.
            # Negative values mean the trained model performs worse.
            headroom = 1.0 - inp
            vals = 100.0 * (model - inp) / headroom.replace(0, np.nan)
            
        # elif spec.formula == "relative_correlation_gain":
        #     model = _source_value(out, spec.sources[0])
        #     inp = _source_value(out, spec.sources[1])
        #     vals = (model - inp) / inp.abs().replace(0, np.nan)

        else:
            raise ValueError(f"Unknown formula for {spec.name}: {spec.formula}")

        out[spec.name] = pd.to_numeric(vals, errors="coerce")

        # For direct metrics from pre-summary tables, preserve source std/count.
        if spec.formula is None:
            src = spec.sources[0]
            out[f"__pre_std__{spec.name}"] = _source_std(out, src)
            out[f"__pre_n__{spec.name}"] = _source_count(out, src)
        else:
            out[f"__pre_std__{spec.name}"] = np.nan
            out[f"__pre_n__{spec.name}"] = np.nan

        if strict and out[spec.name].isna().all():
            raise KeyError(
                f"Required metric '{spec.name}' for task '{task_name}' could not be derived from sources {spec.sources}. "
                f"Available columns: {list(out.columns)}"
            )

    return out, source_report


# -----------------------------------------------------------------------------
# Seed-only aggregation
# -----------------------------------------------------------------------------

def _metadata_first(g: pd.DataFrame, col: str) -> Any:
    if col not in g.columns:
        return np.nan
    vals = g[col].dropna()
    return vals.iloc[0] if len(vals) else np.nan


def _metadata_mode_or_first(g: pd.DataFrame, col: str) -> Any:
    if col not in g.columns:
        return np.nan
    return _mode_or_first(g[col])


def _infer_n_runs_and_seeds(g: pd.DataFrame, metrics: Sequence[str]) -> tuple[int, int]:
    """Infer seed count for run-level or pre-aggregated inputs."""
    if "sampling_seed" in g.columns and g["sampling_seed"].notna().any():
        return int(g["sampling_seed"].notna().sum()), int(g["sampling_seed"].nunique(dropna=True))

    if "seed" in g.columns and g["seed"].notna().any():
        return int(g["seed"].notna().sum()), int(g["seed"].nunique(dropna=True))

    existing_n_runs = _metadata_first(g, "n_runs")
    existing_n_seeds = _metadata_first(g, "n_random_seeds")
    if not is_missing(existing_n_runs):
        n_runs = int(existing_n_runs)
        n_seeds = int(existing_n_seeds) if not is_missing(existing_n_seeds) else n_runs
        return n_runs, n_seeds

    # Pre-summary tables often provide metric_count / metric_n_valid.  Use the
    # first available count as n_runs/n_random_seeds.
    for metric in metrics:
        for col in [f"__pre_n__{metric}", f"{metric}_n", f"{metric}_count", f"{metric}_n_valid"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                if len(vals) and vals.iloc[0] > 0:
                    n = int(vals.iloc[0])
                    return n, n

    # Fallback: one already summarized row.
    return int(len(g)), 0


def aggregate_seed_only_variants(
    canonical: pd.DataFrame,
    task_name: str,
    dataset_id: str | None = None,
    perturbed_group: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only over random seeds/runs; preserve one row per canonical variant."""
    if canonical.empty:
        raise ValueError("Cannot aggregate an empty table.")

    metrics = FINAL_METRICS_BY_TASK[task_name]
    grouped = canonical.groupby(["variant_id"], dropna=False, sort=False)

    records = []
    long_records = []

    for variant_id, g in grouped:
        if isinstance(variant_id, tuple):
            variant_id = variant_id[0]

        rec: dict[str, Any] = {}

        # Dataset/group metadata.
        rec["dataset_id"] = dataset_id if dataset_id is not None else _metadata_first(g, "dataset_id")
        rec["perturbed_group"] = perturbed_group if perturbed_group is not None else _metadata_first(g, "perturbed_group")

        # Stable variant metadata.  Use mode-or-first to avoid retaining a single
        # seed-specific row when a field has repeated identical values across seeds.
        for col in OUTPUT_VARIANT_META_COLUMNS:
            if col in {"dataset_id", "perturbed_group", "variant_id"}:
                continue
            rec[col] = _metadata_mode_or_first(g, col)

        # Recompute the canonical identity from the first row to guarantee stable labels.
        first = g.iloc[0].to_dict()
        rec["variant_id"] = str(variant_id)
        rec["strategy_order"] = int(_metadata_mode_or_first(g, "strategy_order")) if not is_missing(_metadata_mode_or_first(g, "strategy_order")) else 99
        rec["strategy"] = str(_metadata_mode_or_first(g, "strategy"))
        rec["variant_label"] = make_variant_label(first)
        rec["display_variant_label"] = make_display_variant_label(first)
        rec["parameter_label"] = rec["variant_label"]

        n_runs, n_random_seeds = _infer_n_runs_and_seeds(g, metrics)
        rec["n_runs"] = int(n_runs)
        rec["n_random_seeds"] = int(n_random_seeds)

        for metric in metrics:
            vals = pd.to_numeric(g[metric], errors="coerce") if metric in g.columns else pd.Series(dtype=float)
            pre_std = pd.to_numeric(g.get(f"__pre_std__{metric}", pd.Series(np.nan, index=g.index)), errors="coerce")
            pre_n = pd.to_numeric(g.get(f"__pre_n__{metric}", pd.Series(np.nan, index=g.index)), errors="coerce")

            rec[f"{metric}_mean"] = float(vals.mean(skipna=True)) if vals.notna().any() else np.nan

            if vals.notna().sum() > 1:
                rec[f"{metric}_std"] = float(vals.std(ddof=1))
            elif pre_std.notna().any():
                rec[f"{metric}_std"] = float(pre_std.dropna().iloc[0])
            else:
                rec[f"{metric}_std"] = np.nan

            if vals.notna().sum() > 0:
                rec[f"{metric}_n"] = int(vals.notna().sum())
            elif pre_n.notna().any():
                rec[f"{metric}_n"] = int(pre_n.dropna().iloc[0])
            else:
                rec[f"{metric}_n"] = 0

            long_records.append({
                "variant_id": str(variant_id),
                "strategy_order": rec["strategy_order"],
                "strategy": rec["strategy"],
                "variant_label": rec["variant_label"],
                "display_variant_label": rec["display_variant_label"],
                "metric": metric,
                "mean": rec[f"{metric}_mean"],
                "std": rec[f"{metric}_std"],
                "n": rec[f"{metric}_n"],
            })

        # Final safety: no seed/run/path-level metadata in the seed-averaged row.
        for col in ROW_LEVEL_META_COLUMNS:
            rec.pop(col, None)

        records.append(rec)

    wide = pd.DataFrame(records)
    long = pd.DataFrame(long_records)
    wide = sort_variant_table(wide)
    long = sort_variant_table(long) if not long.empty else long
    return wide, long


def sort_variant_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for col in [
        "strategy_order",
        "n_metacells",
        "n_metacells_requested",
        "n_metacells_observed",
        "top_k",
        "top_k_metacells",
        "sampled_metacells_k",
        "n_metacells_to_average",
        "n_control_cells_to_average",
        "sample_cells_per_metacell",
    ]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    sort_cols = [
        c for c in [
            "strategy_order",
            "n_metacells",
            "top_k",
            "sampled_metacells_k",
            "n_control_cells_to_average",
            "variant_label",
        ]
        if c in out.columns
    ]
    return out.sort_values(sort_cols, na_position="first").reset_index(drop=True)


def validate_unique_variants(wide: pd.DataFrame, task_name: str) -> None:
    dup = wide[wide["variant_id"].duplicated(keep=False)] if "variant_id" in wide.columns else pd.DataFrame()
    if not dup.empty:
        raise RuntimeError(
            f"Task '{task_name}' still has duplicated variant_id values after aggregation. "
            f"Examples:\n{dup[['strategy', 'variant_id', 'variant_label']].head(20).to_string(index=False)}"
        )


def expected_variant_count_report(df: pd.DataFrame) -> pd.Series:
    if "strategy" not in df.columns or "variant_id" not in df.columns:
        return pd.Series(dtype=int)
    return df.groupby("strategy", dropna=False)["variant_id"].nunique().reindex([
        "S0_naive_mean_control_reference",
        "S1_random_single_control",
        "S2_random_average_controls",
        "S3_SEACell_metacell_average",
        "S4_SEACell_balanced_random_sample",
        "S5_SEACell_OT_sampled_average",
    ]).fillna(0).astype(int)


def warn_if_unexpected_variant_counts(df: pd.DataFrame) -> None:
    """Print a clear warning if canonical variant counts look wrong."""
    expected = pd.Series({
        "S0_naive_mean_control_reference": 1,
        "S1_random_single_control": 1,
        "S2_random_average_controls": 1,
        "S3_SEACell_metacell_average": 15,
        "S4_SEACell_balanced_random_sample": 5,
        "S5_SEACell_OT_sampled_average": 15,
    })
    observed = expected_variant_count_report(df)
    print("[Observed variant counts]")
    print(observed.to_string())

    extra = observed[(expected > 0) & (observed > expected)]
    if not extra.empty:
        print("[Warning] More variants than expected for:")
        print(extra.to_string())
        print("This usually means a run-level field leaked into the variant key.")


# -----------------------------------------------------------------------------
# Public task aggregation and selection-template merge
# -----------------------------------------------------------------------------

def run_task_aggregation(
    task_spec: TaskSpec,
    outdir: str | Path,
    strategy_rename_map: Mapping[str, str] | None = None,
    strict_metrics: bool = True,
) -> dict[str, Any]:
    outdir = ensure_dir(outdir)
    task_dir = ensure_dir(outdir / task_spec.prefix())

    raw = read_table(task_spec.input_path, sheet_name=task_spec.sheet_name)
    canonical = canonicalize_input_table(raw, task_spec, strategy_rename_map=strategy_rename_map)
    if canonical.empty:
        raise RuntimeError(f"Task '{task_spec.task_name}' has no rows after filtering.")

    canonical, source_report = derive_required_metrics(canonical, task_spec.task_name, strict=strict_metrics)

    canonical_path = task_dir / f"{task_spec.prefix()}_canonical_input_with_required_metrics.csv"
    canonical.to_csv(canonical_path, index=False)

    summary_wide, summary_long = aggregate_seed_only_variants(
        canonical,
        task_name=task_spec.task_name,
        dataset_id=task_spec.dataset_id,
        perturbed_group=task_spec.perturbed_group,
    )
    summary_wide["evaluation_task"] = task_spec.task_name
    summary_long["evaluation_task"] = task_spec.task_name
    validate_unique_variants(summary_wide, task_spec.task_name)

    wide_path = task_dir / f"{task_spec.prefix()}_seed_averaged_by_strategy_variant_wide.csv"
    long_path = task_dir / f"{task_spec.prefix()}_seed_averaged_by_strategy_variant_long.csv"
    summary_wide.to_csv(wide_path, index=False)
    summary_long.to_csv(long_path, index=False)

    source_path = task_dir / f"{task_spec.prefix()}_required_metric_source_report.csv"
    pd.DataFrame([
        {"metric": m, **info} for m, info in source_report.items()
    ]).to_csv(source_path, index=False)

    print(f"[Saved] {wide_path}")
    print("[Variant counts]")
    print(expected_variant_count_report(summary_wide).to_string())

    return {
        "task_name": task_spec.task_name,
        "task_dir": task_dir,
        "canonical_path": canonical_path,
        "summary_wide_path": wide_path,
        "summary_long_path": long_path,
        "source_report_path": source_path,
        "summary_wide": summary_wide,
        "summary_long": summary_long,
    }


def _coalesce_base_metadata(base: pd.DataFrame) -> pd.DataFrame:
    """Coalesce metadata from multiple task summaries into one row per variant.

    Earlier versions used drop_duplicates(..., keep="first"), which could drop
    metadata available only in later tasks, such as inverse-MLP class counts.
    """
    if base.empty:
        return base

    records = []
    for variant_id, g in base.groupby("variant_id", dropna=False, sort=False):
        rec = {"variant_id": variant_id}
        for col in base.columns:
            if col == "variant_id":
                continue
            vals = g[col].dropna()
            vals = vals[vals.astype(str).str.strip() != ""]
            rec[col] = vals.iloc[0] if len(vals) else np.nan
        records.append(rec)
    return pd.DataFrame(records)


def combine_variant_selection_template(
    task_outputs: Mapping[str, Mapping[str, Any]],
    outdir: str | Path,
) -> Path:
    """Create one editable seed-averaged variant table.

    The table has exactly one row per canonical variant_id.  Metrics from each
    task are prefixed with `<task>__` to avoid name collisions.
    """
    outdir = ensure_dir(outdir)

    base_frames = []
    metric_frames = []

    for task_name, outputs in task_outputs.items():
        wide = outputs.get("summary_wide")
        if wide is None:
            wide = outputs.get("wide")

        if not isinstance(wide, pd.DataFrame) or wide.empty:
            continue

        if "variant_id" not in wide.columns:
            raise KeyError(f"Task '{task_name}' summary lacks variant_id.")

        if wide["variant_id"].duplicated().any():
            dup = wide.loc[wide["variant_id"].duplicated(keep=False), ["strategy", "variant_id", "variant_label"]]
            raise RuntimeError(f"Task '{task_name}' has duplicate variant rows before merge:\n{dup.head(20)}")

        local_base_cols = [c for c in SELECTION_BASE_COLUMNS if c in wide.columns]
        local_base = wide[local_base_cols].copy()

        # Enforce one row per variant and no seed/path columns in the editable table.
        local_base = local_base.drop(columns=[c for c in ROW_LEVEL_META_COLUMNS if c in local_base.columns], errors="ignore")
        local_base = local_base.drop_duplicates("variant_id", keep="first")
        base_frames.append(local_base)

        metric_cols = []
        for metric in FINAL_METRICS_BY_TASK[task_name]:
            for suffix in ["mean", "std", "n"]:
                col = f"{metric}_{suffix}"
                if col in wide.columns:
                    metric_cols.append(col)

        metric_local = wide[["variant_id"] + metric_cols].copy()
        metric_local = metric_local.rename(columns={c: f"{task_name}__{c}" for c in metric_cols})
        if metric_local["variant_id"].duplicated().any():
            raise RuntimeError(f"Task '{task_name}' metric frame has duplicated variant_id values.")
        metric_frames.append(metric_local)

    if not base_frames:
        raise RuntimeError("No task outputs were available to create selected variant template.")

    # Union of base metadata, one row per variant.
    base = pd.concat(base_frames, ignore_index=True, sort=False)
    base = base.drop(columns=[c for c in ROW_LEVEL_META_COLUMNS if c in base.columns], errors="ignore")
    base = _coalesce_base_metadata(base)
    base = sort_variant_table(base)

    merged = base.copy()
    for metric_local in metric_frames:
        merged = merged.merge(metric_local, on="variant_id", how="left", validate="one_to_one")

    merged = merged.drop(columns=[c for c in ROW_LEVEL_META_COLUMNS if c in merged.columns], errors="ignore")
    merged = sort_variant_table(merged)

    if "select_for_final" not in merged.columns:
        merged.insert(0, "select_for_final", False)
    if "manual_color" not in merged.columns:
        merged.insert(1, "manual_color", "")
    if "final_strategy_label" not in merged.columns:
        merged.insert(2, "final_strategy_label", merged["display_variant_label"])

    path = outdir / "selected_variants_TEMPLATE_EDIT_ME.csv"
    merged.to_csv(path, index=False)

    print(f"[Saved] editable selected-variant template: {path}")
    print("[Selection template variant counts]")
    warn_if_unexpected_variant_counts(merged)
    print(f"[Selection template shape] {merged.shape[0]} rows × {merged.shape[1]} columns")

    if "S2_random_average_controls" in set(merged["strategy"].astype(str)):
        s2_n = int((merged["strategy"].astype(str) == "S2_random_average_controls").sum())
        if s2_n != 1:
            raise RuntimeError(
                f"S2_random_average_controls should have exactly 1 selection row, but found {s2_n}. "
                "Check make_variant_label(): S2 must return 'default'."
            )

    return path
