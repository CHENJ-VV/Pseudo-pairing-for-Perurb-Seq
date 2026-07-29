"""Entry point for the pseudo-pairing evaluation pipeline."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Any

from .control_manifold import run_control_manifold_evaluation
from .perturbation_effect import run_perturbation_effect_evaluation
from .mlp import run_mlp_evaluation


def run_evaluation_pipeline(config: Mapping[str, Any] | SimpleNamespace):
    tasks = getattr(config, "evaluation_tasks", None)
    if tasks is None and isinstance(config, Mapping):
        tasks = config.get("evaluation_tasks", None)
    if tasks is None:
        tasks = ["control_manifold", "perturbation_effect", "mlp"]

    outputs = {}
    if "control_manifold" in tasks:
        outputs["control_manifold"] = run_control_manifold_evaluation(config)
    if "perturbation_effect" in tasks:
        outputs["perturbation_effect"] = run_perturbation_effect_evaluation(config)
    if "mlp" in tasks:
        outputs["mlp"] = run_mlp_evaluation(config)
    return outputs


if __name__ == "__main__":
    # Edit this block for command-line execution, or use the notebook instead.
    CONFIG = SimpleNamespace(
        dataset_id="Replogle_RPE",
        control_h5ad="/ibex/user/chenj0i/Perturbation/data/processed_data/Replogle_RPE/groups/Replogle_RPE_control_processed.h5ad",
        perturbed_h5ad="/ibex/user/chenj0i/Perturbation/data/processed_data/Replogle_RPE/groups/Replogle_RPE_single_processed.h5ad",
        manifest_path="/ibex/project/c2366/Perturb_data/Replogle_rpe_data/Replogle_RPE/pseudo_pairing_repetition_manifest.csv",
        outdir=Path("/ibex/user/chenj0i/Perturbation/evaluation/Replogle_RPE_pseudo_pairing_evaluation"),
        perturbation_key="perturbation_key",
        perturbed_groups_to_evaluate=["single"],
        evaluation_tasks=["control_manifold", "perturbation_effect", "mlp"],
        max_eval_genes=3000,
        max_runs_to_evaluate=None,
        seed=42,
        # MLP settings
        mlp_tasks=["forward", "inverse_strategy_delta", "inverse_common_delta"],
        batch_size=256,
        forward_epochs=30,
        inverse_epochs=30,
        early_stop_patience=3,
        learning_rate=1e-3,
        weight_decay=1e-5,
        hidden_dim=1024,
        latent_dim=512,
        pert_emb_dim=256,
        dropout=0.15,
    )
    print(run_evaluation_pipeline(CONFIG))
