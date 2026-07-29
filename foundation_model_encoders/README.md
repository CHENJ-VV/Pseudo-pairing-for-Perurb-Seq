# Foundation-model encoders for pseudo-control evaluation

This top-level repository folder generates pseudo-control representations with multiple single-cell encoders and evaluates those representations in downstream perturbation-prediction tasks. It is a sibling of the pairing and gene-program workflows:

```text
single_cell_perturbation_workflows/
├── pseudo_pairing_toolkit/
├── foundation_model_encoders/
└── gene_program_evaluation/
```

The folder consumes pseudo-control h5ad files and manifests produced by `../pseudo_pairing_toolkit/`, but it does not import that package.

The original model computations are retained. The refactor is restricted to repository/checkpoint/data/output path handling, stage invocation, and downstream embedding discovery.

## Supported representations

| Name | Source implementation | Output |
|---|---|---|
| Geneformer | [Geneformer](https://huggingface.co/ctheodoris/Geneformer) | Rank-tokenized cell embeddings |
| scGPT | [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) | Transformer cell embeddings |
| scCello | [DeepGraphLearning/scCello](https://github.com/DeepGraphLearning/scCello) | Cell-ontology-guided embeddings |
| SCimilarity | [Genentech/scimilarity](https://github.com/Genentech/scimilarity) | Metric-learning cell embeddings |
| scVI | [scverse/scvi-tools](https://github.com/scverse/scvi-tools) | Learned latent representation |
| HVG-PCA | Scanpy/scVI branch | Linear baseline representation |
| Expression MLP | Existing evaluation branch | Unencoded expression baseline |

Model repositories and checkpoints are not vendored into this folder. Their local locations are supplied in YAML.

## Why separate environments are supported

The upstream repositories can require incompatible versions of PyTorch, Transformers, CUDA extensions, NumPy, or AnnData. The launcher therefore permits a different Python executable for every model:

```yaml
environments:
  python:
    geneformer: /path/to/envs/geneformer/bin/python
    scgpt: /path/to/envs/scgpt/bin/python
    sccello: /path/to/envs/sccello/bin/python
    scimilarity: /path/to/envs/scimilarity/bin/python
    scvi: /path/to/envs/scvi/bin/python
    mlp: /path/to/envs/evaluation/bin/python
```

The lightweight launcher environment only requires PyYAML:

```bash
python -m pip install -r requirements-launcher.txt
```

Install each model's dependencies according to its upstream repository in the corresponding environment. Repository checkouts are prepended to `sys.path` at runtime, so the scripts no longer assume a particular home directory.

## Configuration

Copy the example:

```bash
cd foundation_model_encoders
cp configs/encoders.example.yaml configs/encoders.yaml
```

Edit four groups of paths:

1. `dataset`: pseudo-control root, processed h5ad files, pairing manifest, and evaluation directory.
2. `repositories`: local checkouts of Geneformer, scGPT, scCello, and SCimilarity.
3. `checkpoints`: pretrained model and vocabulary locations.
4. `environments.python`: the Python executable used for each stage.

Validate the schema:

```bash
python launcher.py validate --config configs/encoders.yaml
```

Also verify that configured files and directories exist:

```bash
python launcher.py validate \
  --config configs/encoders.yaml \
  --check-paths
```

Display the resolved commands without running models:

```bash
python launcher.py show --config configs/encoders.yaml
```

## Input layout

`dataset.pseudo_root` must be the perturbation-group folder containing S0-S5 directories:

```text
<dataset>/<single-or-dual>/
├── S0_naive_mean_control_reference/
│   └── pseudo_control_aligned_to_perturbed.h5ad
├── S1_random_single_control/
│   └── seed_000/
│       └── pseudo_control_aligned_to_perturbed.h5ad
└── S5_SEACell_OT_sampled_average/
    └── nmc_200/topk_05/seed_000/
        └── pseudo_control_aligned_to_perturbed.h5ad
```

Set `dataset.variants: []` to discover all variants. Slash-style paths and generated slug names are accepted by the encoder discovery functions.

## Run encoders

Run one stage:

```bash
python launcher.py run --config configs/encoders.yaml --model geneformer
python launcher.py run --config configs/encoders.yaml --model scgpt
python launcher.py run --config configs/encoders.yaml --model sccello
python launcher.py run --config configs/encoders.yaml --model scimilarity
python launcher.py run --config configs/encoders.yaml --model scvi
```

The `scvi` stage runs both HVG-PCA and scVI by default. Disable either method under `settings.scvi`, or run only HVG-PCA with:

```bash
python launcher.py run --config configs/encoders.yaml --model hvg
```

Run all stages marked `true` under `enabled`:

```bash
python launcher.py run --config configs/encoders.yaml
```

Select several stages explicitly:

```bash
python launcher.py run \
  --config configs/encoders.yaml \
  --model scgpt \
  --model sccello \
  --model scimilarity
```

Preview without execution:

```bash
python launcher.py run \
  --config configs/encoders.yaml \
  --model all \
  --dry-run
```

## Default output layout

Keeping `outputs.<model>: null` preserves the expected automatic layout:

```text
<dataset.pseudo_root>/
├── _geneformer_embeddings_chunked/
│   ├── embeddings/<variant-slug>/X_geneformer.npy
│   └── manifests/
├── _scgpt_embeddings/
│   ├── embeddings/<variant-slug>/scgpt_embeddings.npy
│   └── manifests/
├── _sccello_embeddings_optimized_v2/
│   ├── embeddings/<variant-slug>/X_scCello.npy
│   └── manifests/
├── _scimilarity_embeddings/
│   ├── embeddings/<variant-slug>/X_scimilarity.npy
│   └── manifests/
└── _hvg_scvi_embeddings/
    ├── hvg_pca/embeddings/<variant-slug>/X_pca_hvg.npy
    ├── scvi/embeddings/<variant-slug>/X_scVI.npy
    └── manifests/
```

Every encoder also writes cell-order metadata and a manifest. The downstream evaluator aligns embeddings to perturbed cells using those metadata files rather than assuming that arbitrary arrays are ordered identically.

Custom output roots are supported. The MLP runner builds explicit path templates from the YAML `outputs` section, so custom paths do not require source-code edits.

## Downstream MLP evaluation

After generating embeddings, enable or run the `mlp` stage:

```bash
python launcher.py run --config configs/encoders.yaml --model mlp
```

The default evaluation includes:

```yaml
mlp_evaluation:
  models_to_run:
    - mlp_expr
    - geneformer
    - hvg
    - scvi
    - sccello
    - scimilarity
    - scgpt
  forward_target_space: expression
  inverse_input_space: expression
```

The forward task uses a pseudo-control representation plus perturbation identity to predict an expression-space perturbation effect and restored post-perturbation expression. The inverse expression-space task classifies perturbation identity from perturbed expression minus pseudo-control expression. Changing `inverse_input_space` to `representation` requires compatible true-control and true-perturbed representations.

Results are written under:

```text
<dataset.evaluation_root>/<perturbed_group>/downstream_mlp_task_models/
```

Combined forward and inverse summaries retain `mlp_representation`, strategy, seed, and perturbation-group metadata for later aggregation.

## UMAP visualization

Provide `dataset.perturbed_h5ad`, configure `settings.umap.models`, and run:

```bash
python launcher.py run --config configs/encoders.yaml --model umap
```

The UMAP stage searches all configured encoder output folders, reads the appropriate cell-order metadata file, filters to the requested variants and models, and writes coordinates plus PNG/PDF figures. The previous model-specific execution restriction was removed; all selected models are now processed.

## SLURM

Submit one stage:

```bash
mkdir -p logs
sbatch slurm/run_stage.sbatch configs/encoders.local.yaml scgpt
```

The SLURM job starts the launcher, while the launcher starts the stage with the model-specific Python executable configured in YAML. Adjust memory, GPU, CPU, and wall-time directives to the selected model and dataset size.

## Repository and checkpoint expectations

### Geneformer

Configure a local Geneformer checkout, a `transformers.from_pretrained()`-compatible model directory, and optionally a token dictionary. The script keeps the original official `TranscriptomeTokenizer` and chunked embedding extraction workflow.

### scGPT

Configure the scGPT checkout and a checkpoint directory containing at least `args.json` and `best_model.pt`. The vocabulary defaults to the checkout's `scgpt/tokenizer/default_gene_vocab.json` unless overridden.

### scCello

Configure the scCello checkout. The token dictionary defaults to `data/token_vocabulary/token_dictionary.pkl` inside that checkout. `sccello_checkpoint` may be a Hugging Face identifier or a local directory accepted by `from_pretrained()`.

### SCimilarity

Configure the repository checkout and extracted pretrained-model directory. The import layer supports both an editable `scimilarity` installation and the repository's `src.scimilarity` layout.

### scVI

No source checkout is required by the script; install `scvi-tools` in the Python environment selected for the `scvi` stage.

## Migration from the uploaded branch

| Original constant | YAML replacement |
|---|---|
| `GENEFORMER_ROOT`, `SCGPT_ROOT`, `SCCELLO_ROOT`, `SCIMILARITY_ROOT` | `repositories.*` |
| Model/checkpoint constants | `checkpoints.*` |
| `PSEUDO_ROOT` / `PSEUDO_SEARCH_ROOT` | `dataset.pseudo_root` |
| `PROCESSED_ROOT` | `dataset.control_h5ad` and `dataset.perturbed_h5ad` |
| `PAIRING_ROOT` and manifest constants | `dataset.pairing_manifest` |
| `EVAL_ROOT` | `dataset.evaluation_root` |
| Script-local `VARIANTS` lists | `dataset.variants` |
| Script-local output roots | `outputs.*` |
| Manually activated model environments | `environments.python.*` |

Do not edit the path constants inside individual scripts. Update the YAML and rerun validation instead.

## Provenance and limitations

The notebooks under `notebooks/legacy/` are retained as records of the exploratory workflow. The Python scripts are the maintained execution path.

A full numerical regression requires the original h5ad files, pretrained checkpoints, GPU software stack, and upstream model environments. Static compilation and configuration resolution can be tested without those assets, but successful model execution must be verified in the target HPC environment.
