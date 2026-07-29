"""High-level runner for aggregation, variant selection, heatmaps, and final plots."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd

from result_analysis_common import (
    FINAL_METRICS_BY_TASK,
    STRATEGY_PLOT_LABELS,
    TaskSpec,
    as_namespace,
    ensure_dir,
    save_json,
)
from aggregate_variants import (
    combine_variant_selection_template,
    expected_variant_count_report,
    run_task_aggregation,
)
from plot_comparison import (
    load_selection_table,
    plot_selected_metrics,
    plot_strategy_metric_heatmap_collections,
    prepare_selected_comparison_table,
)


def build_task_specs_from_config(config: SimpleNamespace) -> list[TaskSpec]:
    task_specs = []
    task_inputs = getattr(config, "task_inputs", {})
    for task_name, info in task_inputs.items():
        info = dict(info)
        task_specs.append(TaskSpec(
            task_name=task_name,
            input_path=info["input_path"],
            metrics=info.get("metrics", FINAL_METRICS_BY_TASK.get(task_name)),
            strategy_col=info.get("strategy_col"),
            dataset_id=info.get("dataset_id", getattr(config, "dataset_id", None)),
            perturbed_group=info.get("perturbed_group", getattr(config, "perturbed_group", None)),
            sheet_name=info.get("sheet_name", 0),
            row_filter=info.get("row_filter", {}),
            output_prefix=info.get("output_prefix"),
        ))
    return task_specs


def run_aggregation_stage(config: Mapping[str, Any] | SimpleNamespace) -> dict[str, Any]:
    config = as_namespace(config)
    outdir = ensure_dir(getattr(config, "outdir"))
    agg_dir = ensure_dir(outdir / "aggregated_by_task")
    task_specs = build_task_specs_from_config(config)
    if not task_specs:
        raise ValueError("config.task_inputs is empty.")

    records = []
    outputs = {}
    allow_missing_tasks = bool(getattr(config, "allow_missing_tasks", True))
    allow_task_failures = bool(getattr(config, "allow_task_failures", False))
    strict_metrics = bool(getattr(config, "strict_metrics", True))

    for spec in task_specs:
        print("\n" + "=" * 100)
        print(f"[Aggregate] {spec.task_name}")
        print("=" * 100)
        try:
            if not Path(spec.input_path).exists():
                if allow_missing_tasks:
                    print(f"[Skip missing task] {spec.task_name}: {spec.input_path}")
                    continue
                raise FileNotFoundError(spec.input_path)

            result = run_task_aggregation(
                task_spec=spec,
                outdir=agg_dir,
                strategy_rename_map=getattr(config, "strategy_rename_map", None),
                strict_metrics=strict_metrics,
            )
            outputs[spec.task_name] = result

            heatmap_dir = ensure_dir(result["task_dir"] / "heatmaps")
            heatmap_paths = plot_strategy_metric_heatmap_collections(
                summary_wide=result["summary_wide"],
                task_name=spec.task_name,
                output_dir=heatmap_dir,
                metrics=FINAL_METRICS_BY_TASK[spec.task_name],
            )

            records.append({
                "task_name": spec.task_name,
                "input_path": str(spec.input_path),
                "summary_wide_path": str(result["summary_wide_path"]),
                "summary_long_path": str(result["summary_long_path"]),
                "source_report_path": str(result["source_report_path"]),
                "n_heatmap_collections": len(heatmap_paths),
            })
        except Exception as exc:
            if allow_task_failures:
                print(f"[Task failed but continuing] {spec.task_name}: {repr(exc)}")
                continue
            raise

    if not outputs:
        raise RuntimeError("No task outputs were aggregated.")

    manifest = pd.DataFrame(records)
    manifest_path = agg_dir / "aggregation_outputs_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    selection_path = combine_variant_selection_template(outputs, outdir=outdir)
    selection_df = pd.read_csv(selection_path)

    save_json({
        "aggregation_manifest": str(manifest_path),
        "selection_template": str(selection_path),
        "variant_counts": expected_variant_count_report(selection_df).to_dict(),
        "note": "Edit selected_variants_TEMPLATE_EDIT_ME.csv: set select_for_final=True for chosen variants, then rerun with run_final_comparison=True.",
    }, outdir / "result_analysis_stage1_outputs.json")

    print("\n[Saved]", manifest_path)
    print("[Saved]", selection_path)
    return {"aggregation_manifest": manifest_path, "selection_template": selection_path, "outputs": outputs}


def _load_summary_wide_for_task(agg_dir: Path, task_name: str) -> pd.DataFrame | None:
    task_dir = agg_dir / task_name
    candidates = sorted(task_dir.glob(f"{task_name}_seed_averaged_by_strategy_variant_wide.csv"))
    if not candidates:
        candidates = sorted(task_dir.glob("*_seed_averaged_by_strategy_variant_wide.csv"))
    if not candidates:
        candidates = sorted(agg_dir.glob(f"*/{task_name}_seed_averaged_by_strategy_variant_wide.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])



def _variant_suffix_for_plot_label(row: pd.Series) -> str:
    """Return a compact variant suffix only when it is needed to disambiguate variants.

    The base label always comes from STRATEGY_PLOT_LABELS.  This suffix preserves
    selected metacell/top-k settings when multiple variants from the same strategy
    are selected for the final comparison.
    """
    strategy = str(row.get("strategy", ""))

    def _is_finite_number(value: Any) -> bool:
        try:
            return pd.notna(value) and str(value).strip() != "" and float(value) == float(value)
        except Exception:
            return False

    def _fmt(value: Any) -> str:
        value = float(value)
        return str(int(value)) if value.is_integer() else f"{value:g}"

    nmc = row.get("n_metacells", row.get("n_metacells_requested", pd.NA))
    if not _is_finite_number(nmc):
        nmc = row.get("n_metacells_observed", pd.NA)

    if strategy == "S2_random_average_controls":
        k = row.get("n_control_cells_to_average", pd.NA)
        return f"\n(k={_fmt(k)})" if _is_finite_number(k) else ""

    if strategy == "S3_SEACell_metacell_average":
        k = row.get("sampled_metacells_k", row.get("n_metacells_to_average", pd.NA))
        if _is_finite_number(nmc) and _is_finite_number(k):
            return f"\n({_fmt(nmc)}&{_fmt(k)})"
        if _is_finite_number(nmc):
            return f"\n({_fmt(nmc)})"
        return ""

    if strategy == "S4_SEACell_balanced_random_sample":
        return f"\n({_fmt(nmc)})" if _is_finite_number(nmc) else ""

    if strategy == "S5_SEACell_OT_sampled_average":
        topk = row.get("top_k", row.get("top_k_metacells", pd.NA))
        if _is_finite_number(nmc) and _is_finite_number(topk):
            return f"\n({_fmt(nmc)}&{_fmt(topk)})"
        if _is_finite_number(nmc):
            return f"\n({_fmt(nmc)})"
        if _is_finite_number(topk):
            return f"\n(topk={_fmt(topk)})"
        return ""

    return ""


def _apply_strategy_plot_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Force final bar-plot x labels to use STRATEGY_PLOT_LABELS.

    This function is intentionally placed in this runner so plot_comparison.py does
    not need to be modified.  It overwrites the `plot_label` column immediately
    before `plot_selected_metrics()` is called.
    """
    out = df.copy()
    if "strategy" not in out.columns:
        return out

    strategy_counts = out["strategy"].astype(str).value_counts().to_dict()
    labels = []
    for _, row in out.iterrows():
        strategy = str(row.get("strategy", ""))
        base_label = STRATEGY_PLOT_LABELS.get(strategy, strategy)
        # Keep duplicated selected variants distinguishable while still using
        # the common STRATEGY_PLOT_LABELS base label.
        suffix = _variant_suffix_for_plot_label(row) if strategy_counts.get(strategy, 0) > 1 else ""
        labels.append(f"{base_label}{suffix}")
    out["plot_label"] = labels
    return out



# -----------------------------------------------------------------------------
# Inverse-MLP naive baseline integration
# -----------------------------------------------------------------------------

INVERSE_CLASS_COUNT_CONFIG_KEYS = [
    "naive_inverse_n_classes",
    "mlp_inverse_n_classes",
    "n_perturbation_classes",
]

INVERSE_CLASS_COUNT_TABLE_COLUMNS = [
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
]


def _coerce_positive_class_count(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        value = float(value)
    except Exception:
        return None
    if value == value and value > 0:
        return value
    return None


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _first_class_count_from_table(df: pd.DataFrame | None) -> float | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for col in INVERSE_CLASS_COUNT_TABLE_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        vals = vals[vals.notna() & (vals > 0)]
        if len(vals):
            return float(vals.median())
    return None


def _candidate_mlp_config_paths(
    config: SimpleNamespace,
    task_info: Mapping[str, Any] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    task_info = dict(task_info or {})

    for key in ["mlp_config_summary_path", "mlp_config_path"]:
        if key in task_info:
            paths.append(Path(task_info[key]))
        if hasattr(config, key):
            paths.append(Path(getattr(config, key)))

    eval_root = getattr(config, "eval_root", None)
    perturbed_group = getattr(config, "perturbed_group", None)
    if eval_root is not None and perturbed_group is not None:
        paths.append(Path(eval_root) / str(perturbed_group) / "downstream_mlp" / "mlp_config_summary.json")

    input_path = task_info.get("input_path")
    if input_path is not None:
        input_path = Path(input_path)
        paths.append(input_path.parent / "mlp_config_summary.json")

    # Preserve order while removing duplicates.
    seen = set()
    unique_paths = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    return unique_paths


def _infer_inverse_n_classes(
    config: SimpleNamespace,
    task_info: Mapping[str, Any] | None = None,
    selected: pd.DataFrame | None = None,
    summary_wide: pd.DataFrame | None = None,
) -> float | None:
    """Infer inverse perturbation class count for the naive baseline.

    Priority:
    1. explicit CONFIG value, e.g. CONFIG.mlp_inverse_n_classes = 105;
    2. task input dictionary values;
    3. mlp_config_summary.json, especially key n_perturbation_classes;
    4. class-count columns preserved in selected or summary tables.
    """
    task_info = dict(task_info or {})

    for key in INVERSE_CLASS_COUNT_CONFIG_KEYS:
        if hasattr(config, key):
            value = _coerce_positive_class_count(getattr(config, key))
            if value is not None:
                return value
        if key in task_info:
            value = _coerce_positive_class_count(task_info[key])
            if value is not None:
                return value

    for path in _candidate_mlp_config_paths(config, task_info):
        data = _read_json(path)
        for key in ["n_perturbation_classes", "test_n_classes", "n_classes", "num_classes", "n_labels"]:
            value = _coerce_positive_class_count(data.get(key))
            if value is not None:
                return value

    value = _first_class_count_from_table(selected)
    if value is not None:
        return value
    value = _first_class_count_from_table(summary_wide)
    if value is not None:
        return value

    return None


def _inject_inverse_class_count(
    df: pd.DataFrame,
    n_classes: float | None,
) -> pd.DataFrame:
    """Attach class-count columns used by plot_comparison.py."""
    if n_classes is None:
        return df
    out = df.copy()
    for col in [
        "naive_inverse_n_classes",
        "mlp_inverse_n_classes",
        "n_perturbation_classes",
        "n_perturbation_classes_mean",
        "test_n_classes",
        "test_n_classes_mean",
    ]:
        out[col] = float(n_classes)
    return out

def run_final_comparison_stage(config: Mapping[str, Any] | SimpleNamespace) -> dict[str, Any]:
    config = as_namespace(config)
    outdir = ensure_dir(getattr(config, "outdir"))
    agg_dir = outdir / "aggregated_by_task"
    if not agg_dir.exists():
        raise FileNotFoundError(f"Aggregation directory does not exist: {agg_dir}. Run stage 1 first.")

    selection_path = Path(getattr(config, "selection_path", outdir / "selected_variants_TEMPLATE_EDIT_ME.csv"))
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection file does not exist: {selection_path}")
    selection = load_selection_table(selection_path)
    if selection.empty:
        raise RuntimeError("No rows selected in selection table. Set select_for_final=True for the desired variants.")

    final_dir = ensure_dir(outdir / "final_selected_comparison")
    records = []
    task_inputs = getattr(config, "task_inputs", {})
    final_metrics_by_task = getattr(config, "final_metrics_by_task", FINAL_METRICS_BY_TASK)
    allow_missing_tasks = bool(getattr(config, "allow_missing_tasks", True))

    for task_name in task_inputs.keys():
        print("\n" + "=" * 100)
        print(f"[Final comparison] {task_name}")
        print("=" * 100)
        summary_wide = _load_summary_wide_for_task(agg_dir, task_name)
        if summary_wide is None:
            if allow_missing_tasks:
                print(f"[Skip missing summary] {task_name}")
                continue
            raise FileNotFoundError(f"Cannot find summary for task {task_name} under {agg_dir}")

        selected = prepare_selected_comparison_table(summary_wide, selection)
        selected = _apply_strategy_plot_labels(selected)
        if selected.empty:
            print(f"[Skip] No selected variants available for task {task_name}")
            continue

        naive_inverse_n_classes = None
        if task_name == "mlp_inverse":
            task_info = task_inputs.get(task_name, {}) if isinstance(task_inputs, Mapping) else {}
            naive_inverse_n_classes = _infer_inverse_n_classes(
                config=config,
                task_info=task_info,
                selected=selected,
                summary_wide=summary_wide,
            )
            if naive_inverse_n_classes is not None:
                print(
                    f"[Inverse naive baseline] n_perturbation_classes = {naive_inverse_n_classes:g}; "
                    f"accuracy baseline = {1.0 / naive_inverse_n_classes:.6g}. "
                    "No automatic 1/n baseline is drawn for precision/recall/macro F1."
                )
                selected = _inject_inverse_class_count(selected, naive_inverse_n_classes)
                summary_wide = _inject_inverse_class_count(summary_wide, naive_inverse_n_classes)
            elif bool(getattr(config, "plot_naive_inverse_reference", True)):
                print("[Warning] Inverse naive baseline requested, but class count was not found. "
                      "Set CONFIG.mlp_inverse_n_classes or provide mlp_config_summary.json.")

        task_dir = ensure_dir(final_dir / task_name)
        table_path = task_dir / f"{task_name}_selected_final_compare.csv"
        selected.to_csv(table_path, index=False)
        figure_dir = ensure_dir(task_dir / "figures")
        metrics = final_metrics_by_task.get(task_name, FINAL_METRICS_BY_TASK.get(task_name, []))
        figure_paths = plot_selected_metrics(
            selected_df=selected,
            summary_wide=summary_wide,
            metrics=metrics,
            output_dir=figure_dir,
            file_prefix=f"{task_name}_selected",
            plot_s0_reference=bool(getattr(config, "plot_s0_reference", True)),
            plot_s1_reference=bool(getattr(config, "plot_s1_reference", True)),
            plot_naive_inverse_reference=(
                bool(getattr(config, "plot_naive_inverse_reference", True))
                and task_name == "mlp_inverse"
            ),
            naive_inverse_n_classes=naive_inverse_n_classes,
            inverse_metric_reference_values=getattr(config, "inverse_metric_reference_values", None),
            inverse_metric_reference_labels=getattr(config, "inverse_metric_reference_labels", None),
        )
        records.append({
            "task_name": task_name,
            "selected_table": str(table_path),
            "figure_dir": str(figure_dir),
            "n_figures": len(figure_paths),
        })
        print("[Saved]", table_path)
        print(f"[Saved] {len(figure_paths)} figures to {figure_dir}")

    manifest = pd.DataFrame(records)
    manifest_path = final_dir / "final_comparison_outputs_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("\n[Saved]", manifest_path)
    return {"final_manifest": manifest_path, "records": records}


def run_result_analysis_pipeline(config: Mapping[str, Any] | SimpleNamespace) -> dict[str, Any]:
    config = as_namespace(config)
    outputs = {}
    if bool(getattr(config, "run_aggregation", True)):
        outputs["aggregation"] = run_aggregation_stage(config)
    if bool(getattr(config, "run_final_comparison", False)):
        outputs["final_comparison"] = run_final_comparison_stage(config)
    return outputs


if __name__ == "__main__":
    raise SystemExit("Use execute_result_analysis_pipeline.ipynb or import run_result_analysis_pipeline(config).")
