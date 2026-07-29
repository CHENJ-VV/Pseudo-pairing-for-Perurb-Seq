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
