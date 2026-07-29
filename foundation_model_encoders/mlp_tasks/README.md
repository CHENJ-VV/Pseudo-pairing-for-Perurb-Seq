# Encoder-aware MLP evaluation

This folder evaluates pseudo-control representations in the same forward and inverse tasks used by the expression-space MLP branch.

Supported representations:

- `mlp_expr`
- `geneformer`
- `hvg`
- `scvi`
- `sccello`
- `scimilarity`
- `scgpt`

Run from the parent folder:

```bash
python ../launcher.py run --config ../configs/encoders.yaml --model mlp
```

The runner derives control, perturbed, manifest, evaluation, pseudo-control, and embedding paths from the shared YAML file. Custom encoder output roots are translated into `embedding_specs`, so no source-code path edits are required.

The numerical training and evaluation functions remain in `eval_mlp.py` and `eval_mlp_task_models.py`. `run_mlp_tasks.py` is only the configuration and invocation layer.
