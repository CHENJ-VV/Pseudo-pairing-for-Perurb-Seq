# Pseudo-pairing evaluation pipeline

This package refactors the repeated evaluation notebooks into a reusable pipeline with three independent evaluation families:

1. `control_manifold`: evaluates whether pseudo-control cells preserve the real-control manifold.
2. `perturbation_effect`: evaluates whether strategy-specific perturbation effects agree with common-reference perturbation effects.
3. `mlp`: runs downstream forward and inverse MLP evaluations.

The pipeline expects a pseudo-pairing manifest such as:

```text
pseudo_pairing_repetition_manifest.csv
```

from the S0-S5 repeated pseudo-control generation workflow. The required manifest column is `pseudo_control_h5ad`. The pipeline also uses optional metadata columns such as `strategy`, `strategy_family`, `perturbed_group`, `sampling_seed`, `seacell_setting_id`, `parameter_label`, and `membership_path_for_metacell_coverage`.

## Basic usage in Jupyter

```python
from types import SimpleNamespace
from pathlib import Path
from run_evaluation_pipeline import run_evaluation_pipeline

CONFIG = SimpleNamespace(
    dataset_id="NormanWeissman2019",
    control_h5ad=".../NormanWeissman2019_control_processed.h5ad",
    perturbed_h5ad=".../NormanWeissman2019_single_processed.h5ad",
    manifest_path=".../pseudo_pairing_repetition_manifest.csv",
    outdir=Path(".../evaluation/NormanWeissman2019_pseudo_pairing_evaluation"),
    perturbation_key="perturbation_key",
    perturbed_groups_to_evaluate=["single"],
    evaluation_tasks=["control_manifold", "perturbation_effect", "mlp"],
    max_eval_genes=3000,
)

outputs = run_evaluation_pipeline(CONFIG)
```

## User-accessible MLP initialization

You can swap the MLP architecture without editing the training loop by passing factory functions:

```python
from mlp_models import ForwardPerturbationMLP, InversePerturbationMLP


def my_forward_model_init(n_genes, n_perturbations, config):
    return ForwardPerturbationMLP(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        pert_emb_dim=512,
        hidden_dim=2048,
        latent_dim=1024,
        dropout=0.10,
    )


def my_inverse_model_init(n_genes, n_perturbations, config):
    return InversePerturbationMLP(
        n_genes=n_genes,
        n_perturbations=n_perturbations,
        hidden_dim=2048,
        latent_dim=1024,
        dropout=0.10,
    )

CONFIG.forward_model_init_fn = my_forward_model_init
CONFIG.inverse_model_init_fn = my_inverse_model_init
```

Alternatively, keep default classes and only set:

```python
CONFIG.hidden_dim = 1024
CONFIG.latent_dim = 512
CONFIG.pert_emb_dim = 256
CONFIG.dropout = 0.15
```

## Output layout

```text
{outdir}/
  control_manifold/
    control_manifold_preservation_repeated_long.csv
    control_manifold_preservation_repeated_summary.csv
  perturbation_effect/
    perturbation_effect_consistency_repeated_per_perturbation.csv
    perturbation_effect_consistency_repeated_run_summary.csv
    perturbation_effect_consistency_repeated_summary.csv
  downstream_mlp/
    forward_mlp_run_summary.csv
    forward_mlp_per_perturbation_common_delta.csv
    inverse_mlp_run_summary.csv
    inverse_mlp_per_class.csv
    summaries/
```

## Dual-perturbation inverse prediction options

For `dual` perturbation h5ad files, you usually have labels such as `BCL2L11_BAK1`. There are three common inverse prediction formulations:

1. **Combination multiclass**: treat `BCL2L11_BAK1` as one class. This is the closest to the current inverse MLP. It is simple, but does not generalize to unseen gene combinations.
2. **Multi-label gene prediction**: output sigmoid probabilities over individual perturbed genes and evaluate whether the two true genes are recovered. This can generalize to unseen pairs.
3. **Factorized pair prediction**: predict two unordered gene heads or score all gene pairs from gene-level logits. This is more structured than multiclass and usually better when the number of observed pairs is sparse.

Recommended reporting for dual perturbations:

- combination top-1/top-5 accuracy and macro-F1;
- unordered exact pair match;
- component-gene recall@2 / recall@5;
- Jaccard index between predicted and true gene sets;
- per-gene AUROC/AUPRC/F1 for the multi-label formulation;
- held-out-combination split if you want to test compositional generalization.
