# PseudoPairingToolkit

A reusable Python workflow that consolidates the uploaded scripts for:

1. public/local data acquisition;
2. AnnData preprocessing and control/perturbed group generation;
3. repeated pseudo-control construction with strategies S0–S5;
4. control-manifold, perturbation-effect, and downstream MLP evaluation;
5. seed-only aggregation, variant selection, heatmaps, and final comparisons.

The scientific implementations from the uploaded scripts are retained with minimal changes. The main refactor replaces folder-specific `sys.path` manipulation and separate execution notebooks with an installable package, one YAML configuration, and one CLI.

## What is included

```text
src/pseudopair/
  acquisition.py             generic download/copy + SHA-256 verification
  preprocessing.py           notebook preprocessing converted to functions
  pipeline.py                unified stage orchestration and run state
  pairing/                   S0–S5 implementations and repeated runner
  evaluation/                control, effect, forward MLP, inverse MLP
  analysis/                  seed-only aggregation and plotting
configs/
  pipeline.example.yaml      complete generic configuration
  replogle_rpe.ibex.example.yaml
slurm/
  run_pipeline.sbatch
legacy_notebooks/
  original uploaded execution notebooks for reference
```

## Installation

Create or activate the scientific environment used for the original scripts, then install the toolkit:

```bash
cd PseudoPairingToolkit
pip install -e ".[full]"
```

SEACells is still required for S3, S4, and S5 and should be installed in the same environment using the installation method already used on the cluster. S5 also requires POT, installed by the `ot` or `full` extra.

Check the environment:

```bash
pseudopair doctor
```

## Start from a configuration

```bash
pseudopair init pseudopair.yaml
pseudopair validate --config pseudopair.yaml
pseudopair plan --config pseudopair.yaml
```

Relative paths in YAML are resolved relative to the YAML file, not the shell's current directory. Environment variables and `~` are expanded.

## Run the workflow

Run every enabled stage:

```bash
pseudopair run --config pseudopair.yaml --stage all
```

Run stages separately, which is preferable for SLURM scheduling:

```bash
pseudopair run --config pseudopair.yaml --stage acquire
pseudopair run --config pseudopair.yaml --stage preprocess
pseudopair run --config pseudopair.yaml --stage pair
pseudopair run --config pseudopair.yaml --stage evaluate
pseudopair run --config pseudopair.yaml --stage aggregate
```

Use `--dry-run` to resolve the plan and output paths without loading Scanpy, SEACells, POT, or PyTorch:

```bash
pseudopair run --config pseudopair.yaml --stage all --dry-run
```

## Stage contracts

### Acquisition

`acquisition.files` accepts either a URL or a local `source_path`. Downloads are written through a `.part` file and atomically renamed. An optional SHA-256 checksum can be supplied.

### Preprocessing

The original Replogle preprocessing notebook is represented by configurable operations:

- perturbation multiplicity from `nperts` (`control`, `single`, `dual`, `multi`);
- standardized `perturbation_key` creation;
- counts-layer preservation;
- library-size normalization and log transformation;
- HVG selection, PCA, neighbors, Leiden, and UMAP;
- group-specific h5ad generation without subsetting the final expression feature space to HVGs.

The stage writes:

```text
<preprocessing.output_dir>/
  <dataset>_global_processed.h5ad
  groups/<dataset>_<group>_processed.h5ad
  preprocessing_manifest.csv
  preprocessing_summary.json
```

When explicit pairing paths are omitted, the pairing and evaluation stages resolve them from `preprocessing_summary.json`.

### Pairing

The retained strategies are:

- `S0_naive_mean_control_reference`
- `S1_random_single_control`
- `S2_random_average_controls`
- `S3_SEACell_metacell_average`
- `S4_SEACell_balanced_random_sample`
- `S5_SEACell_OT_sampled_average`

The pairing runner preserves the original parameter grids, repeat seeds, reusable SEACell memberships, reusable OT assignments, overwrite flags, per-run QC files, and top-level manifest. Existing outputs are reused by default.

```text
<pairing.output_root>/<dataset>/
  pseudo_pairing_repetition_manifest.csv
  _seacell_memberships/
  <group>/<strategy>/...
```

### Evaluation

The retained evaluation families are:

- control-manifold preservation;
- perturbation-effect consistency;
- forward MLP prediction of post-perturbation expression and perturbation effects;
- inverse MLP perturbation-identity classification.

The unified runner evaluates each requested perturbed group independently under:

```text
<evaluation.eval_root>/<group>/
```

Custom model factory functions remain supported through the direct Python API. See `examples/custom_mlp_models.py`.

### Aggregation and visualization

The analysis layer keeps the existing critical aggregation rule: only the random seed is averaged away. Strategy-defining parameters remain separate variants. The editable selection table therefore contains one row per canonical strategy variant rather than one row per run.

Stage 1:

```yaml
analysis:
  run_aggregation: true
  run_final_comparison: false
```

Edit only `select_for_final`, `manual_color`, and `final_strategy_label` in:

```text
<eval_root>/<group>/result_analysis/selected_variants_TEMPLATE_EDIT_ME.csv
```

Stage 2:

```yaml
analysis:
  run_aggregation: false
  run_final_comparison: true
  selection_path: /absolute/or/relative/path/to/selected_variants_TEMPLATE_EDIT_ME.csv
```

## Reproducibility and restart behavior

Each stage records its state synchronously in:

```text
<project.workdir>/pipeline_runs/<dataset>/pipeline_state.json
```

The file records the resolved configuration hash, Python/platform information, timestamps, status, outputs, and exceptions. It does not create a background process. Expensive pairing and MLP operations additionally retain their original skip-existing behavior.

## Direct Python API

```python
from pseudopair.config import load_config
from pseudopair.pipeline import run_pipeline

config = load_config("pseudopair.yaml")
outputs = run_pipeline(config, stages=["pair", "evaluate"])
```

The lower-level retained APIs are also available:

```python
from pseudopair.pairing import run_pseudo_pairing_repetition_plan
from pseudopair.evaluation import run_evaluation_pipeline
from pseudopair.analysis import run_result_analysis_pipeline
```

## Recommended HPC execution

Use separate jobs for pairing, evaluation, and aggregation. The included `slurm/run_pipeline.sbatch` accepts the configuration path and stage:

```bash
sbatch slurm/run_pipeline.sbatch configs/replogle_rpe.ibex.example.yaml pair
sbatch slurm/run_pipeline.sbatch configs/replogle_rpe.ibex.example.yaml evaluate
sbatch slurm/run_pipeline.sbatch configs/replogle_rpe.ibex.example.yaml aggregate
```

For full S0–S5 experiments, validate on a small subset first by setting `pairing.max_pairs_per_perturbation: 300` and `evaluation.max_runs_to_evaluate` to a small number.

## Important compatibility notes

- The scientific modules require Python 3.10 or newer.
- S3–S5 require an `X_pca`-like embedding in both control and perturbed h5ad objects.
- S3–S5 require SEACell membership generation or reusable membership files.
- S5 requires POT and sufficient memory for the selected OT problem.
- `hvg_flavor: seurat_v3` may require the optional `scikit-misc` dependency in the Scanpy environment.
- MLP evaluation requires PyTorch. GPU selection remains controlled by `evaluation.device`.
- Full numerical equivalence should be verified on the original cluster environment because the present packaging environment does not contain Scanpy, AnnData, SEACells, or POT.
