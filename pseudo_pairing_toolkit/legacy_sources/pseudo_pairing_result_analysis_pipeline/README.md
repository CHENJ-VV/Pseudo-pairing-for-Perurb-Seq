# Pseudo-pairing result analysis pipeline

This package performs the post-evaluation analysis layer:

1. read the current evaluation-pipeline output files;
2. derive the requested final metrics using the exact available column names;
3. average only over random seeds;
4. preserve all strategy variant settings for manual selection;
5. create strategy-wise all-metric heatmap collections for variant selection;
6. plot final selected comparisons with S0 and S1 dashed reference lines.

## Expected input files

For a dataset/group such as:

```python
EVAL_ROOT = Path("/ibex/user/chenj0i/Perturbation/evaluation/NormanWeissman2019_pseudo_pairing_evaluation")
PERTURBED_GROUP = "single"
```

this pipeline reads:

```text
<EVAL_ROOT>/<PERTURBED_GROUP>/control_manifold/control_manifold_preservation_repeated_summary.csv
<EVAL_ROOT>/<PERTURBED_GROUP>/perturbation_effect/perturbation_effect_consistency_repeated_run_summary.csv
<EVAL_ROOT>/<PERTURBED_GROUP>/downstream_mlp/forward_mlp_run_summary.csv
<EVAL_ROOT>/<PERTURBED_GROUP>/downstream_mlp/inverse_mlp_run_summary.csv
<EVAL_ROOT>/<PERTURBED_GROUP>/downstream_mlp/mlp_config_summary.json
```

The `mlp_config_summary.json` file is used for inverse-MLP naive dashed baselines.
For your current Norman single-perturbation run, it contains:

```json
{
  "n_runs": 186,
  "n_eval_genes": 3000,
  "n_perturbation_classes": 105,
  "device": "cuda"
}
```

Thus the naive inverse-classification reference is `1 / 105 = 0.00952`.
Use `build_default_task_inputs(...)` to construct these paths automatically.

## Required final metrics

### Control manifold

- `mean_expression_rmse`
- `mean_expression_correlation`
- `variance_pearson`
- `control_pseudo_local_mixing_score`
- `pca_centroid_distance`
- `mmd_pca`

### Perturbation effect

- `perturbation_effect_rmse`
- `perturbation_effect_pearson`
- `perturbation_effect_magnitude_ratio`
- `top100_perturbation_effect_correlation`

### Forward MLP

- `input_only_mse`
- `model_mse`
- `absolute_mse_improvement`
- `relative_mse_reduction`
- `input_only_mae`
- `model_mae`
- `absolute_mae_improvement`
- `relative_mae_reduction`
- `input_common_reference_correlation`
- `model_common_reference_correlation`
- `absolute_correlation_improvement`
- `relative_correlation_improvement`

### Inverse MLP

- `test_accuracy`
- `macro_f1`
- `recall`
- `precision`

Inverse MLP is filtered to `inverse_input_mode == "strategy_delta"` by default.

## Aggregation rule

The aggregation unit is one canonical strategy variant:

```text
strategy + n_metacells + top_k + sampled_metacells_k
```

Only `sampling_seed` is averaged away.  The selected-variant table should have, for a complete S0-S5 run:

```text
S0 = 1 row
S1 = 1 row
S2 = 1 row
S3 = 15 rows
S4 = 5 rows
S5 = 15 rows
```

The key function enforcing this is:

```python
aggregate_variants.make_variant_id()
```

It does not use row index, run path, pair metadata path, task name, or `run_id`, so the selection table cannot expand to thousands of rows.

## Stage 1: aggregate and create heatmaps

```python
from pathlib import Path
from types import SimpleNamespace

from result_analysis_common import build_default_task_inputs, FINAL_METRICS_BY_TASK
from run_result_analysis_pipeline import run_result_analysis_pipeline

DATASET_ID = "NormanWeissman2019"
PERTURBED_GROUP = "single"

EVAL_ROOT = Path(
    "/ibex/user/chenj0i/Perturbation/evaluation/"
    "NormanWeissman2019_pseudo_pairing_evaluation"
)
OUTDIR = EVAL_ROOT / PERTURBED_GROUP / "result_analysis"

TASK_INPUTS = build_default_task_inputs(
    eval_root=EVAL_ROOT,
    perturbed_group=PERTURBED_GROUP,
    tasks=["control_manifold", "perturbation_effect", "mlp_forward", "mlp_inverse"],
    allow_missing=True,
)

CONFIG = SimpleNamespace(
    dataset_id=DATASET_ID,
    perturbed_group=PERTURBED_GROUP,
    eval_root=EVAL_ROOT,
    outdir=OUTDIR,
    task_inputs=TASK_INPUTS,
    run_aggregation=True,
    run_final_comparison=False,
    allow_missing_tasks=True,
    allow_task_failures=False,
    strict_metrics=True,
    final_metrics_by_task=FINAL_METRICS_BY_TASK,
    plot_s0_reference=True,
    plot_s1_reference=True,
    plot_naive_inverse_reference=True,
    # Optional explicit override. If omitted, the runner reads
    # <EVAL_ROOT>/<PERTURBED_GROUP>/downstream_mlp/mlp_config_summary.json.
    mlp_inverse_n_classes=105,
)

outputs = run_result_analysis_pipeline(CONFIG)
```

Stage 1 writes:

```text
<OUTDIR>/aggregated_by_task/<task>/<task>_seed_averaged_by_strategy_variant_wide.csv
<OUTDIR>/aggregated_by_task/<task>/<task>_seed_averaged_by_strategy_variant_long.csv
<OUTDIR>/aggregated_by_task/<task>/heatmaps/*metric_heatmap_collection.png
<OUTDIR>/selected_variants_TEMPLATE_EDIT_ME.csv
```

The editable selection table is a seed-averaged variant-performance table.  Edit only:

```text
select_for_final
manual_color
final_strategy_label
```

## Stage 2: final selected comparison

```python
CONFIG.run_aggregation = False
CONFIG.run_final_comparison = True
CONFIG.selection_path = OUTDIR / "selected_variants_TEMPLATE_EDIT_ME.csv"
final_outputs = run_result_analysis_pipeline(CONFIG)
```

Final comparison plots are saved as PNG files under:

```text
<OUTDIR>/final_selected_comparison/<task>/figures/
```

Each final bar plot adds dashed reference lines for:

- S0 mean-control reference, where applicable;
- S1 random-single-control mean, where applicable;
- inverse MLP naive top-1 accuracy baseline, where applicable.

For inverse MLP, the automatic class-count baseline is drawn **only for**
`test_accuracy`:

```text
naive accuracy = 1 / n_perturbation_classes
```

This is appropriate for a uniform random top-1 classifier.  It is **not**
automatically used for `precision`, `recall`, or `macro_f1`, because those
baselines depend on the dummy prediction rule, class prevalence, and
zero-division convention.  If you want to plot empirical dummy-classifier
baselines for those metrics, provide them explicitly:

```python
CONFIG.inverse_metric_reference_values = {
    # Example values from an empirical dummy classifier, not automatically assumed.
    # "precision": 0.010,
    # "recall": 0.0095,
    # "macro_f1": 0.008,
}
CONFIG.inverse_metric_reference_labels = {
    # "precision": "Empirical dummy precision",
    # "recall": "Empirical dummy recall",
    # "macro_f1": "Empirical dummy macro F1",
}
```

The class number for the accuracy baseline is read from `mlp_config_summary.json`
using the key `n_perturbation_classes`, or from `CONFIG.mlp_inverse_n_classes` if
explicitly set.
