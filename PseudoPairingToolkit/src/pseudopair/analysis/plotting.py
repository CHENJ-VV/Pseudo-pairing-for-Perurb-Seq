"""Plot seed-averaged variant heatmaps and final selected comparisons.

This module is intentionally aligned with the updated result-analysis pipeline:

1. Aggregation has already produced one row per canonical strategy variant.
2. ``sampling_seed`` and run-level paths are not used for plotting.
3. Final bar-plot labels are generated from ``STRATEGY_PLOT_LABELS`` in
   ``result_analysis_common.py`` with dynamic variant suffixes.
4. S0 and S1 are drawn as optional dashed reference lines in final plots.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    FINAL_METRICS_BY_TASK,
    METRIC_DIRECTIONS,
    METRIC_PLOT_LABELS,
    STRATEGY_BASE_COLORS,
    STRATEGY_ORDER,
    STRATEGY_PLOT_LABELS,
    STRATEGY_VARIANT_COLOR_POOLS,
    ensure_dir,
    fmt_int_like,
    is_missing,
    safe_filename,
)
from .aggregate import sort_variant_table


matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["xtick.major.width"] = 0.8
matplotlib.rcParams["ytick.major.width"] = 0.8


# -----------------------------------------------------------------------------
# Generic formatting helpers
# -----------------------------------------------------------------------------

def _is_finite_number(value: Any) -> bool:
    try:
        if is_missing(value):
            return False
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _fmt_setting(value: Any) -> str:
    return fmt_int_like(value) if _is_finite_number(value) else str(value)


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def format_value(value: Any) -> str:
    """Compact numeric string for heatmap annotations."""
    if is_missing(value):
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 0.1:
        return f"{value:.4f}"
    if abs_value >= 0.001:
        return f"{value:.5f}"
    return f"{value:.2e}"


def annotation_text(mean_value: Any, std_value: Any) -> str:
    """Return 'mean ± std' text for heatmap cells."""
    if is_missing(mean_value):
        return ""
    if is_missing(std_value):
        return format_value(mean_value)
    return f"{format_value(mean_value)}\n±{format_value(std_value)}"


def final_bar_value(value: Any, digits: int = 4) -> str:
    """Compact numeric string for labels above final bars."""
    if is_missing(value):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    if abs(value) >= 0.01:
        return f"{value:.{digits}f}"
    return f"{value:.2e}"


def cmap_for_metric(metric: str):
    """Return an intuitive colormap for raw-value heatmaps."""
    direction = METRIC_DIRECTIONS.get(metric, "higher")
    if direction == "lower":
        return "YlGnBu_r"
    if direction == "target1":
        return "YlOrBr"
    return "YlGnBu"


def metric_cols(df: pd.DataFrame, metric: str) -> tuple[str, str | None]:
    """Find mean/std columns for a standardized metric."""
    mean_col = f"{metric}_mean"
    if mean_col not in df.columns:
        if metric in df.columns:
            mean_col = metric
        else:
            raise KeyError(f"Missing metric column for {metric}: expected {metric}_mean")
    std_col = f"{metric}_std" if f"{metric}_std" in df.columns else None
    return mean_col, std_col


# -----------------------------------------------------------------------------
# Strategy display labels and colors
# -----------------------------------------------------------------------------

def _variant_suffix_for_strategy(row: pd.Series) -> str:
    """Build a dynamic variant suffix using canonical variant-setting columns.

    The base text always comes from STRATEGY_PLOT_LABELS. This suffix only adds
    the selected metacell/top-k settings, e.g. ``\n(350&5)``.
    """
    strategy = str(row.get("strategy", ""))
    nmc = row.get("n_metacells", row.get("n_metacells_requested", np.nan))
    if not _is_finite_number(nmc):
        nmc = row.get("n_metacells_observed", np.nan)

    if strategy == "S3_SEACell_metacell_average":
        sampled = row.get("sampled_metacells_k", row.get("n_metacells_to_average", np.nan))
        if _is_finite_number(nmc) and _is_finite_number(sampled):
            return f"\n({_fmt_setting(nmc)}&{_fmt_setting(sampled)})"
        if _is_finite_number(nmc):
            return f"\n({_fmt_setting(nmc)})"
        return ""

    if strategy == "S4_SEACell_balanced_random_sample":
        return f"\n({_fmt_setting(nmc)})" if _is_finite_number(nmc) else ""

    if strategy == "S5_SEACell_OT_sampled_average":
        topk = row.get("top_k", row.get("top_k_metacells", np.nan))
        if _is_finite_number(nmc) and _is_finite_number(topk):
            return f"\n({_fmt_setting(nmc)}&{_fmt_setting(topk)})"
        if _is_finite_number(nmc):
            return f"\n({_fmt_setting(nmc)})"
        if _is_finite_number(topk):
            return f"\n(topk={_fmt_setting(topk)})"
        return ""

    return ""


def _strategy_plot_label(row: pd.Series) -> str:
    """Return final x-axis label using STRATEGY_PLOT_LABELS.

    If the user manually edits ``final_strategy_label`` to something other than
    the automatically created display label, that manual label is respected.
    Otherwise, the label is reconstructed from STRATEGY_PLOT_LABELS.
    """
    strategy = str(row.get("strategy", ""))
    display_variant_label = str(row.get("display_variant_label", ""))
    manual_label = row.get("final_strategy_label", "")

    # Respect a truly manual label, but ignore automatically copied raw labels.
    if not is_missing(manual_label):
        manual_label = str(manual_label)
        if manual_label not in {strategy, display_variant_label}:
            return manual_label

    base = STRATEGY_PLOT_LABELS.get(strategy, strategy)
    return f"{base}{_variant_suffix_for_strategy(row)}"


def _assign_plot_colors(selected: pd.DataFrame) -> pd.Series:
    """Assign final bar colors with optional manual override."""
    colors: list[str] = []
    strategy_counts: dict[str, int] = {}
    for _, row in selected.iterrows():
        manual = row.get("manual_color", "")
        if not is_missing(manual):
            colors.append(str(manual))
            continue

        strategy = str(row.get("strategy", ""))
        idx = strategy_counts.get(strategy, 0)
        strategy_counts[strategy] = idx + 1

        pool = STRATEGY_VARIANT_COLOR_POOLS.get(strategy)
        if pool and selected["strategy"].astype(str).eq(strategy).sum() > 1:
            colors.append(pool[idx % len(pool)])
        else:
            colors.append(STRATEGY_BASE_COLORS.get(strategy, "#999999"))
    return pd.Series(colors, index=selected.index)


# -----------------------------------------------------------------------------
# Variant heatmap collections for selection
# -----------------------------------------------------------------------------

def _column_has_values(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns and df[col].notna().any()


def _axis_columns_for_strategy(strategy: str, data: pd.DataFrame) -> tuple[str, str]:
    """Choose x/y axis columns for variant heatmaps robustly.

    Preferred axes:
        S3: n_metacells × sampled_metacells_k
        S5: n_metacells × top_k
        S4: n_metacells × constant setting
        S2/S1/S0: one-cell heatmap, unless S2 has a real k setting.
    """
    if strategy == "S3_SEACell_metacell_average":
        x = "n_metacells" if _column_has_values(data, "n_metacells") else "n_metacells_requested"
        y = "sampled_metacells_k" if _column_has_values(data, "sampled_metacells_k") else "n_metacells_to_average"
        if _column_has_values(data, x) and _column_has_values(data, y):
            return x, y

    if strategy == "S5_SEACell_OT_sampled_average":
        x = "n_metacells" if _column_has_values(data, "n_metacells") else "n_metacells_requested"
        y = "top_k" if _column_has_values(data, "top_k") else "top_k_metacells"
        if _column_has_values(data, x) and _column_has_values(data, y):
            return x, y

    if strategy == "S4_SEACell_balanced_random_sample":
        x = "n_metacells" if _column_has_values(data, "n_metacells") else "n_metacells_requested"
        if _column_has_values(data, x):
            data["_setting"] = "S4"
            return x, "_setting"

    if strategy == "S2_random_average_controls" and _column_has_values(data, "n_control_cells_to_average"):
        data["_setting"] = "S2"
        return "n_control_cells_to_average", "_setting"

    data["_x"] = "default"
    data["_setting"] = "default"
    return "_x", "_setting"


def _axis_label(col: str) -> str:
    return {
        "n_metacells": "Number of SEACells",
        "n_metacells_requested": "Number of SEACells",
        "n_metacells_observed": "Observed SEACells",
        "top_k": "Top-k metacells",
        "top_k_metacells": "Top-k metacells",
        "sampled_metacells_k": "Sampled metacells per pseudo-control",
        "n_metacells_to_average": "Sampled metacells per pseudo-control",
        "n_control_cells_to_average": "Random controls averaged",
        "_setting": "Setting",
        "_x": "Setting",
    }.get(col, col)


def _sort_axis_values(values: Sequence[Any]) -> list[Any]:
    vals = list(pd.Index(values).dropna())
    if not vals:
        return vals
    numeric = pd.to_numeric(pd.Series(vals), errors="coerce")
    if numeric.notna().all():
        order = np.argsort(numeric.to_numpy(dtype=float))
        return [vals[i] for i in order]
    return sorted(vals, key=lambda x: str(x))


def _draw_one_heatmap(ax, data: pd.DataFrame, metric: str, x_col: str, y_col: str) -> None:
    mean_col, std_col = metric_cols(data, metric)
    plot = data.copy()
    plot[mean_col] = _as_numeric(plot[mean_col])
    if std_col is not None:
        plot[std_col] = _as_numeric(plot[std_col])
    else:
        plot["__std"] = np.nan
        std_col = "__std"

    plot = plot[plot[mean_col].notna()].copy()
    if plot.empty:
        ax.set_visible(False)
        return

    # One variant per x/y cell is expected. Collapse defensively if not.
    if plot.duplicated([x_col, y_col], keep=False).any():
        plot = (
            plot
            .groupby([x_col, y_col], dropna=False, as_index=False)
            .agg({mean_col: "mean", std_col: "mean"})
        )

    mean_grid = plot.pivot(index=y_col, columns=x_col, values=mean_col)
    x_order = _sort_axis_values(mean_grid.columns)
    y_order = _sort_axis_values(mean_grid.index)
    mean_grid = mean_grid.reindex(index=y_order, columns=x_order)
    std_grid = plot.pivot(index=y_col, columns=x_col, values=std_col).reindex(index=y_order, columns=x_order)

    arr = mean_grid.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap_for_metric(metric))

    for i in range(mean_grid.shape[0]):
        for j in range(mean_grid.shape[1]):
            text = annotation_text(mean_grid.iloc[i, j], std_grid.iloc[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=7)

    title, _, _ = METRIC_PLOT_LABELS.get(metric, (metric, metric, 4))
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(np.arange(mean_grid.shape[1]))
    ax.set_yticks(np.arange(mean_grid.shape[0]))
    ax.set_xticklabels([fmt_int_like(x) for x in mean_grid.columns], rotation=0)
    ax.set_yticklabels([fmt_int_like(y) for y in mean_grid.index], rotation=0)
    ax.set_xlabel(_axis_label(x_col))
    ax.set_ylabel(_axis_label(y_col))
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_strategy_metric_heatmap_collections(
    summary_wide: pd.DataFrame,
    task_name: str,
    output_dir: str | Path,
    metrics: Sequence[str] | None = None,
) -> list[Path]:
    """Create one all-metric heatmap collection figure for each strategy."""
    output_dir = ensure_dir(output_dir)
    metrics = list(metrics or FINAL_METRICS_BY_TASK[task_name])
    paths: list[Path] = []
    df = sort_variant_table(summary_wide)

    for strategy in STRATEGY_ORDER:
        sub = df[df["strategy"].astype(str) == strategy].copy()
        if sub.empty:
            continue
        available = []
        for metric in metrics:
            try:
                mean_col, _ = metric_cols(sub, metric)
            except KeyError:
                continue
            if _as_numeric(sub[mean_col]).notna().any():
                available.append(metric)
        if not available:
            continue

        x_col, y_col = _axis_columns_for_strategy(strategy, sub)
        n_cols = min(3, len(available))
        n_rows = int(math.ceil(len(available) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 4.1 * n_rows), squeeze=False)

        for ax, metric in zip(axes.flat, available):
            _draw_one_heatmap(ax, sub, metric, x_col, y_col)
        for ax in axes.flat[len(available):]:
            ax.set_visible(False)

        strategy_title = STRATEGY_PLOT_LABELS.get(strategy, strategy).replace("\n", " ")
        fig.suptitle(f"{task_name}: {strategy_title} seed-averaged variant metrics", fontsize=14, y=1.01)
        fig.tight_layout()
        stem = f"{safe_filename(task_name)}__{safe_filename(strategy)}__metric_heatmap_collection"
        png = output_dir / f"{stem}.png"
        pdf = output_dir / f"{stem}.pdf"
        fig.savefig(png, bbox_inches="tight", dpi=240)
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        paths.append(png)
    return paths


# -----------------------------------------------------------------------------
# Final selected comparison plots
# -----------------------------------------------------------------------------

def load_selection_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "select_for_final" not in df.columns:
        raise KeyError("selection table must contain select_for_final")
    mask = df["select_for_final"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return df.loc[mask].copy()


def prepare_selected_comparison_table(summary_wide: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """Return selected summary rows with plot labels/colors attached.

    The selected x-axis labels are generated from STRATEGY_PLOT_LABELS in the
    common file. Editable ``final_strategy_label`` values are respected only when
    they look manually changed rather than automatically copied from strategy.
    """
    if "variant_id" not in summary_wide.columns or "variant_id" not in selection.columns:
        raise KeyError("Both summary_wide and selection must contain variant_id")

    summary = summary_wide.copy()
    selection = selection.copy()
    summary["variant_id"] = summary["variant_id"].astype(str)
    selection["variant_id"] = selection["variant_id"].astype(str)

    if summary["variant_id"].duplicated().any():
        dup = summary.loc[summary["variant_id"].duplicated(keep=False), ["strategy", "variant_id"]]
        raise RuntimeError(f"summary_wide contains duplicated variant_id rows:\n{dup.head(20).to_string(index=False)}")
    if selection["variant_id"].duplicated().any():
        dup = selection.loc[selection["variant_id"].duplicated(keep=False), ["strategy", "variant_id"]]
        raise RuntimeError(f"selection contains duplicated variant_id rows:\n{dup.head(20).to_string(index=False)}")

    selection["_selection_order"] = np.arange(selection.shape[0], dtype=int)
    keep_cols = [c for c in ["variant_id", "final_strategy_label", "manual_color", "_selection_order"] if c in selection.columns]

    out = summary.merge(
        selection[keep_cols],
        on="variant_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_selected"),
    )
    if out.empty:
        return out

    out = out.sort_values("_selection_order").reset_index(drop=True)
    out["plot_label"] = [_strategy_plot_label(row) for _, row in out.iterrows()]
    out["plot_color"] = _assign_plot_colors(out)
    out["display_order"] = out["strategy_order"].astype(float) if "strategy_order" in out.columns else np.arange(out.shape[0])
    return out


def _reference_line_value(summary_wide: pd.DataFrame, strategy: str, metric: str) -> float:
    try:
        mean_col, _ = metric_cols(summary_wide, metric)
    except KeyError:
        return np.nan
    if "strategy" not in summary_wide.columns:
        return np.nan
    ref = summary_wide[summary_wide["strategy"].astype(str) == strategy].copy()
    if ref.empty:
        return np.nan
    return float(_as_numeric(ref[mean_col]).mean(skipna=True))


def _reference_label(strategy: str) -> str:
    base = STRATEGY_PLOT_LABELS.get(strategy, strategy).replace("\n", " ")
    if strategy == "S0_naive_mean_control_reference":
        return f"{base} reference"
    if strategy == "S1_random_single_control":
        return f"{base} mean"
    return base


# -----------------------------------------------------------------------------
# Naive inverse-MLP baseline helpers
# -----------------------------------------------------------------------------

# By default, the class-count chance baseline is only valid for top-1
# single-label accuracy.  Macro precision/recall/F1 depend on the dummy
# prediction rule, class prevalence, and zero-division convention, so they are
# not plotted automatically.  If desired, pass empirical dummy-classifier
# baselines via `inverse_metric_reference_values`.
INVERSE_CLASS_COUNT_REFERENCE_METRICS = {"test_accuracy"}


INVERSE_CLASS_COUNT_COLUMNS = [
    "naive_inverse_n_classes",
    "mlp_inverse_n_classes",
    "n_perturbation_classes",
    "n_perturbation_classes_mean",
    "test_n_classes",
    "test_n_classes_mean",
    "n_classes",
    "n_classes_mean",
    "class_number",
    "class_number_mean",
    "n_candidate_classes",
    "n_candidate_classes_mean",
    "candidate_class_number",
    "candidate_class_number_mean",
    "num_classes",
    "num_classes_mean",
    "n_labels",
    "n_labels_mean",
    "mlp_inverse__n_perturbation_classes",
    "mlp_inverse__n_perturbation_classes_mean",
    "mlp_inverse__test_n_classes",
    "mlp_inverse__test_n_classes_mean",
]


def _coerce_positive_class_count(value: Any) -> float:
    """Return a positive finite class count or NaN."""
    try:
        if is_missing(value):
            return np.nan
        value = float(value)
    except Exception:
        return np.nan
    if np.isfinite(value) and value > 0:
        return value
    return np.nan


def _infer_inverse_n_classes(
    *tables: pd.DataFrame,
    explicit_n_classes: Any = None,
) -> float:
    """Infer inverse-task perturbation class count.

    Priority:
    1. explicit value passed by the runner, usually from mlp_config_summary.json;
    2. class-count columns preserved in selected or summary tables.
    """
    explicit = _coerce_positive_class_count(explicit_n_classes)
    if np.isfinite(explicit):
        return explicit

    for table in tables:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for col in INVERSE_CLASS_COUNT_COLUMNS:
            if col not in table.columns:
                continue
            vals = pd.to_numeric(table[col], errors="coerce")
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 0]
            if len(vals):
                return float(np.nanmedian(vals.to_numpy(dtype=float)))
    return np.nan


def _naive_inverse_reference_value(
    metric: str,
    selected_df: pd.DataFrame,
    summary_wide: pd.DataFrame,
    naive_inverse_n_classes: Any = None,
    inverse_metric_reference_values: Mapping[str, Any] | None = None,
    inverse_metric_reference_labels: Mapping[str, str] | None = None,
) -> tuple[float, str | None]:
    """Return inverse-classification reference line when applicable.

    Default behavior:
        Only `test_accuracy` receives a class-count chance baseline.  For a
        uniform random top-1 guess over C candidate perturbation classes, the
        expected accuracy is 1/C.

    Precision, recall, and macro F1 are not assigned `1/C` automatically because
    their naive values depend on the dummy prediction rule, class prevalence,
    and zero-division convention.  To plot those baselines, provide explicit
    empirical values through `inverse_metric_reference_values`, for example:
        {"macro_f1": 0.012, "precision": 0.018, "recall": 0.0095}
    """
    custom_refs = dict(inverse_metric_reference_values or {})
    custom_labels = dict(inverse_metric_reference_labels or {})

    if metric in custom_refs:
        try:
            value = float(custom_refs[metric])
        except Exception:
            return np.nan, None
        if not np.isfinite(value):
            return np.nan, None
        label = custom_labels.get(metric)
        if label is None:
            title, _, _ = METRIC_PLOT_LABELS.get(metric, (metric, metric, 4))
            label = f"Naive {title.lower()} reference"
        return value, label

    if metric not in INVERSE_CLASS_COUNT_REFERENCE_METRICS:
        return np.nan, None

    n_classes = _infer_inverse_n_classes(
        selected_df,
        summary_wide,
        explicit_n_classes=naive_inverse_n_classes,
    )
    if not np.isfinite(n_classes) or n_classes <= 0:
        return np.nan, None

    naive = 1.0 / float(n_classes)
    class_text = fmt_int_like(n_classes)
    label = f"Naive accuracy = 1/{class_text}"
    return naive, label


def plot_metric_bar(
    selected_df: pd.DataFrame,
    summary_wide: pd.DataFrame,
    metric: str,
    output_dir: str | Path,
    file_prefix: str,
    plot_s0_reference: bool = True,
    plot_s1_reference: bool = True,
    plot_naive_inverse_reference: bool = True,
    naive_inverse_n_classes: Any = None,
    inverse_metric_reference_values: Mapping[str, Any] | None = None,
    inverse_metric_reference_labels: Mapping[str, str] | None = None,
) -> Path:
    output_dir = ensure_dir(output_dir)
    mean_col, std_col = metric_cols(selected_df, metric)
    plot_df = selected_df.copy()
    plot_df[mean_col] = _as_numeric(plot_df[mean_col])
    if std_col is not None:
        plot_df[std_col] = _as_numeric(plot_df[std_col])

    # Drop rows missing this metric. This allows one selected variant table to be
    # used across tasks even if a specific task lacks some strategy rows.
    plot_df = plot_df[plot_df[mean_col].notna()].copy()
    if plot_df.empty:
        raise KeyError(f"No finite values available for metric {metric}")

    title, ylabel, digits = METRIC_PLOT_LABELS.get(metric, (metric, metric, 4))
    x = np.arange(plot_df.shape[0])
    y = plot_df[mean_col].to_numpy(dtype=float)

    yerr = None
    if std_col is not None and plot_df[std_col].notna().any():
        yerr = plot_df[std_col].to_numpy(dtype=float)
        yerr = np.where(np.isfinite(yerr), yerr, 0.0)

    fig_width = max(6.2, 0.9 * max(len(plot_df), 1) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    bars = ax.bar(
        x,
        y,
        color=plot_df["plot_color"].tolist(),
        edgecolor="black",
        linewidth=0.5,
        width=0.72,
    )
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, capthick=0.8)

    ref_handles = []
    if plot_s0_reference:
        s0 = _reference_line_value(summary_wide, "S0_naive_mean_control_reference", metric)
        if np.isfinite(s0):
            ref_handles.append(ax.axhline(
                s0,
                color="#666666",
                linestyle="--",
                linewidth=1.1,
                alpha=0.85,
                label=_reference_label("S0_naive_mean_control_reference"),
            ))
    if plot_s1_reference:
        s1 = _reference_line_value(summary_wide, "S1_random_single_control", metric)
        if np.isfinite(s1):
            ref_handles.append(ax.axhline(
                s1,
                color="#333333",
                linestyle=(0, (3, 2)),
                linewidth=1.1,
                alpha=0.85,
                label=_reference_label("S1_random_single_control"),
            ))

    if plot_naive_inverse_reference:
        naive_value, naive_label = _naive_inverse_reference_value(
            metric=metric,
            selected_df=selected_df,
            summary_wide=summary_wide,
            naive_inverse_n_classes=naive_inverse_n_classes,
            inverse_metric_reference_values=inverse_metric_reference_values,
            inverse_metric_reference_labels=inverse_metric_reference_labels,
        )
        if np.isfinite(naive_value) and naive_label is not None:
            ref_handles.append(ax.axhline(
                naive_value,
                color="#8A8A8A",
                linestyle=(0, (1.5, 1.8)),
                linewidth=1.15,
                alpha=0.95,
                label=naive_label,
            ))

    if ref_handles:
        ax.legend(frameon=False, fontsize=8, loc="best")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"].astype(str), rotation=0, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    finite_y = y[np.isfinite(y)]
    if finite_y.size:
        y_min = float(np.nanmin(finite_y))
        y_max = float(np.nanmax(finite_y))
        y_range = y_max - y_min
        offset = y_range * 0.025 if y_range > 0 else abs(y_max) * 0.03 + 1e-3
        for bar, value in zip(bars, y):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset if value >= 0 else value - offset,
                final_bar_value(value, digits=digits),
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
                rotation=0,
            )

    fig.tight_layout()
    stem = f"{safe_filename(file_prefix)}__{safe_filename(metric)}"
    out_png = output_dir / f"{stem}.png"
    out_pdf = output_dir / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=260)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_png


def plot_selected_metrics(
    selected_df: pd.DataFrame,
    summary_wide: pd.DataFrame,
    metrics: Sequence[str],
    output_dir: str | Path,
    file_prefix: str,
    plot_s0_reference: bool = True,
    plot_s1_reference: bool = True,
    plot_naive_inverse_reference: bool = True,
    naive_inverse_n_classes: Any = None,
    inverse_metric_reference_values: Mapping[str, Any] | None = None,
    inverse_metric_reference_labels: Mapping[str, str] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    for metric in metrics:
        try:
            paths.append(plot_metric_bar(
                selected_df=selected_df,
                summary_wide=summary_wide,
                metric=metric,
                output_dir=output_dir,
                file_prefix=file_prefix,
                plot_s0_reference=plot_s0_reference,
                plot_s1_reference=plot_s1_reference,
                plot_naive_inverse_reference=plot_naive_inverse_reference,
                naive_inverse_n_classes=naive_inverse_n_classes,
                inverse_metric_reference_values=inverse_metric_reference_values,
                inverse_metric_reference_labels=inverse_metric_reference_labels,
            ))
        except KeyError as exc:
            print(f"[Skip plot] {metric}: {exc}")
    return paths
