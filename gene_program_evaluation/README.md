# Gene-program evaluation of pseudo-control strategies

This top-level repository folder evaluates pseudo-control perturbation effects at the level of control-derived gene programs. It is independent of the other folders at import time, but can reuse the h5ad files, generation manifest, and `run_config.json` written by `../pseudo_pairing_toolkit/`.

The refactor changes path resolution, configuration, packaging, and execution only. The uploaded gene-selection, correlation, clustering, program-scoring, effect-construction, and metric calculations are retained.

## Repository position

```text
single_cell_perturbation_workflows/
├── pseudo_pairing_toolkit/
├── foundation_model_encoders/
└── gene_program_evaluation/
    ├── configs/
    ├── gene_program_pipeline_pkg/
    ├── notebooks/legacy/
    ├── slurm/
    ├── tests/
    ├── pyproject.toml
    ├── run_gene_program_pipeline.py
    └── README.md
```

## What the workflow does

1. Resolves real-control, true-perturbed, and pseudo-control h5ad files.
2. Selects expressed and variable-enough control genes.
3. Computes control-cell gene-gene correlations.
4. Builds gene programs using Leiden, connected components, or agglomerative clustering.
5. Standardizes expression using real-control means and standard deviations.
6. Computes cell-level and perturbation-level program scores.
7. Defines true and pseudo perturbation effects at the gene-program level.
8. Compares pseudo effects with true effects using RMSE, MAE, Pearson, Spearman, cosine similarity, magnitude ratio, and per-program/per-perturbation correlations.
9. Averages repeated seeds at the variant level.

## Installation

From the repository root:

```bash
cd gene_program_evaluation
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

A minimal environment without notebooks can use:

```bash
python -m pip install -e .
```

Leiden clustering requires `igraph` and `leidenalg`. With `cluster_method: auto`, the workflow falls back to connected components when Leiden is unavailable.

## Configuration

Copy the template rather than editing the tracked example:

```bash
cp configs/gene_program.example.json configs/gene_program.local.json
```

Paths may be absolute or relative to the configuration file. `~` and environment variables are expanded. Standard shell variables such as `${PAIRING_ROOT}` are supported. The template also accepts the convenient form `${PAIRING_ROOT:-../work/pairing}` and uses the text after `:-` when the variable is unset.

### Shared pseudo-pairing mode

The default `path_mode` is:

```json
"path_mode": "shared_pseudo_pairing_pipeline"
```

For each source dataset, the pairing folder is resolved as:

```text
pairing_root / manifest_dataset_names[dataset_id] / dataset_id
```

The workflow searches that folder for:

```text
pseudo_pairing_repetition_manifest.csv
run_config.json
```

When `use_generation_run_config` is enabled, `control_h5ad` and group-specific `perturbed_h5ads` are recovered from `run_config.json`. Pseudo-control paths are read from the manifest. Relative paths inside either file are resolved against the file containing them.

### Explicit mode

For a standalone analysis that does not use the pairing-folder convention, use:

```json
{
  "global": {
    "path_mode": "explicit",
    "perturbation_key": "perturbation_label"
  },
  "datasets": [
    {
      "dataset_id": "example_single",
      "source_dataset_id": "example",
      "perturbed_group": "single",
      "control_h5ad": "../data/control.h5ad",
      "perturbed_h5ad": "../data/perturbed_single.h5ad",
      "manifest_path": "../pairing/pseudo_pairing_repetition_manifest.csv",
      "pseudo_root": "../pairing/single",
      "result_analysis_dir": "../evaluation/example/single/result_analysis",
      "output_dir": "../evaluation/example/single/result_analysis/gene_program_level"
    }
  ]
}
```

## Validate path resolution

Resolve inputs and write `<config-stem>__resolved.json` without loading AnnData:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --prepare-only
```

Equivalent installed command:

```bash
gene-program-eval --config configs/gene_program.local.json --prepare-only
```

Inspect the generated resolved JSON before launching a large job. Missing dataset/group combinations are recorded under `shared_path_resolution.skipped`.

## Run the analysis

Build or rebuild control-derived programs:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --build
```

Evaluate pseudo-control files using an existing membership table:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --evaluate
```

Build and evaluate:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --build --evaluate
```

Restrict execution to one or more source datasets:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --build --evaluate \
  --datasets NormanWeissman2019 Replogle_RPE
```

Evaluate only variants selected by the result-analysis table:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --build --evaluate \
  --use-selected-variants
```

Use all variants in the generation manifest regardless of the configuration setting:

```bash
python run_gene_program_pipeline.py \
  --config configs/gene_program.local.json \
  --build --evaluate \
  --all-manifest-variants
```

## SLURM

The supplied job script takes the configuration and run mode as arguments:

```bash
mkdir -p logs
sbatch slurm/run_gene_program.sbatch configs/gene_program.local.json all
```

Use `GENE_PROGRAM_PYTHON` to select a specific environment without editing the job script:

```bash
GENE_PROGRAM_PYTHON=/path/to/env/bin/python \
  sbatch slurm/run_gene_program.sbatch configs/gene_program.local.json all
```

Adjust partition, memory, CPU, wall time, and mail directives for the target cluster.

## Mathematical definition

For a gene program containing `n_P` genes, the cell-level score is:

```text
S_P = sum_g Z_g / sqrt(n_P)
```

Each `Z_g` uses the mean and standard deviation estimated from real control cells.

For perturbation `p`:

```text
true effect   = mean(true perturbed program score for p) - mean(real control program score)
pseudo effect = mean(true perturbed program score for p) - mean(pseudo-control program score for p)
```

## Output layout

For shared mode, the default output directory is:

```text
evaluation_root/<dataset_id>_pseudo_pairing_evaluation/<group>/result_analysis/gene_program_level/
```

Main files include:

```text
gene_programs/control_zscore_stats.csv
gene_programs/control_gene_selection_stats.csv
gene_programs/control_gene_clustering.csv
gene_programs/gene_program_membership.csv
gene_programs/gene_program_build_summary.json

effects/true_control_program_expression.csv
effects/true_perturbed_program_expression.csv
effects/true_program_effects.csv
effects/pseudo_program_effects__<variant_id>.csv

metrics/pseudo_file_index.csv
metrics/gene_program_metrics_by_variant_seed_level.csv
metrics/gene_program_metrics_by_variant.csv
metrics/per_program_correlations.csv
metrics/per_perturbation_correlations.csv
metrics/evaluation_summary.json
```

## Configuration guidance

- `max_genes` controls the maximum correlation-matrix dimension. Memory scales approximately quadratically with this value.
- `corr_threshold` and `min_correlated_partners` determine which gene-gene relationships enter clustering.
- `min_program_size`, `max_program_size`, and `min_gene_centroid_corr` control membership refinement.
- `max_control_cells_for_programs` limits cells used to estimate the gene correlation matrix.
- `chunk_size` controls program-score calculation batches.
- `layer: null` uses `AnnData.X`; otherwise the named AnnData layer is used.
- `require_existing_pseudo_files: true` filters stale manifest entries before evaluation.

## Legacy notebooks

The uploaded notebooks are retained under `notebooks/legacy/` for provenance. The maintained execution path is `run_gene_program_pipeline.py`; notebook path constants should not be used for new runs.

## Limitations

The workflow assumes control, true-perturbed, and pseudo-control objects use compatible gene identifiers. It aligns program genes by `var_names` but does not perform ortholog conversion or gene-symbol reconciliation. It also assumes pseudo-control cells can be assigned perturbation labels from their own `obs` or, when absent, by row-order alignment with the true perturbed object.
