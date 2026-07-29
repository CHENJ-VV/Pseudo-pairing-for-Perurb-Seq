"""Unified orchestration for acquisition, preprocessing, pairing, evaluation, and analysis."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .acquisition import run_acquisition
from .config import dump_resolved_config, enabled, get_project
from .preprocessing import run_preprocessing


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, SimpleNamespace):
        return {k: _jsonable(v) for k, v in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return value


def _merge(*blocks: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for block in blocks:
        out.update(dict(block))
    return out


def _load_preprocessing_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id, workdir = get_project(config)
    cfg = dict(config.get("preprocessing", {}))
    output_dir = Path(cfg.get("output_dir", workdir / "processed" / dataset_id))
    summary_path = output_dir / "preprocessing_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Preprocessing summary not found: {summary_path}. Run the preprocess stage or provide explicit pairing paths."
        )
    return json.loads(summary_path.read_text())


def _resolve_group_paths(config: Mapping[str, Any], preprocessing_output: Mapping[str, Any] | None = None) -> tuple[str, dict[str, str]]:
    pair_cfg = dict(config.get("pairing", {}))
    control_h5ad = pair_cfg.get("control_h5ad")
    perturbed = dict(pair_cfg.get("perturbed_h5ads", {}))
    if not control_h5ad or not perturbed:
        summary = dict(preprocessing_output or _load_preprocessing_summary(config))
        group_paths = dict(summary.get("group_paths", {}))
        control_h5ad = control_h5ad or group_paths.get(str(pair_cfg.get("control_group", "control")))
        selected_groups = pair_cfg.get("perturbed_groups")
        if not perturbed:
            if selected_groups:
                perturbed = {str(g): group_paths[str(g)] for g in selected_groups if str(g) in group_paths}
            else:
                control_group = str(pair_cfg.get("control_group", "control"))
                perturbed = {g: p for g, p in group_paths.items() if g != control_group}
    if not control_h5ad:
        raise ValueError("Could not resolve control_h5ad.")
    if not perturbed:
        raise ValueError("Could not resolve any perturbed_h5ads.")
    return str(control_h5ad), {str(k): str(v) for k, v in perturbed.items()}


class PipelineRun:
    """Stateful stage recorder. Scientific work is still executed synchronously."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.dataset_id, self.workdir = get_project(config)
        self.run_dir = self.workdir / "pipeline_runs" / self.dataset_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "pipeline_state.json"
        self.state = self._load_state()
        dump_resolved_config(config, self.run_dir / "resolved_config.yaml")

    def _load_state(self) -> dict[str, Any]:
        config_text = json.dumps(_jsonable(self.config), sort_keys=True)
        current_hash = hashlib.sha256(config_text.encode()).hexdigest()
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            previous_hash = state.get("config_sha256")
            if previous_hash and previous_hash != current_hash:
                state.setdefault("previous_config_sha256", []).append(previous_hash)
            state["config_sha256"] = current_hash
            return state
        return {
            "dataset_id": self.dataset_id,
            "created_at": _now(),
            "updated_at": _now(),
            "config_sha256": current_hash,
            "environment": {"python": sys.version, "platform": platform.platform()},
            "stages": {},
        }

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        self.state_path.write_text(json.dumps(_jsonable(self.state), indent=2))

    def start(self, stage: str) -> None:
        self.state["stages"][stage] = {"status": "running", "started_at": _now()}
        self._save()

    def finish(self, stage: str, output: Any) -> Any:
        self.state["stages"][stage] = {
            "status": "completed", "started_at": self.state["stages"][stage].get("started_at"),
            "finished_at": _now(), "output": _jsonable(output),
        }
        self._save()
        return output

    def fail(self, stage: str, exc: BaseException) -> None:
        self.state["stages"][stage] = {
            "status": "failed", "started_at": self.state["stages"].get(stage, {}).get("started_at"),
            "finished_at": _now(), "error": repr(exc),
        }
        self._save()

    def execute(self, stage: str, func):
        self.start(stage)
        try:
            return self.finish(stage, func())
        except Exception as exc:
            self.fail(stage, exc)
            raise


def run_pairing(config: Mapping[str, Any], preprocessing_output: Mapping[str, Any] | None = None):
    from .pairing.runner import run_pseudo_pairing_repetition_plan

    dataset_id, workdir = get_project(config)
    cfg = dict(config.get("pairing", {}))
    control_h5ad, perturbed_h5ads = _resolve_group_paths(config, preprocessing_output)
    output_root = Path(cfg.get("output_root", workdir / "pairing"))
    base = {
        "dataset_id": dataset_id,
        "control_h5ad": control_h5ad,
        "perturbed_h5ads": perturbed_h5ads,
        "outdir": str(output_root),
        "membership_root": str(Path(cfg.get("membership_root", output_root / dataset_id / "_seacell_memberships"))),
    }
    return run_pseudo_pairing_repetition_plan(SimpleNamespace(**_merge(cfg, base)))


def run_evaluation(config: Mapping[str, Any], preprocessing_output: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .evaluation.runner import run_evaluation_pipeline

    dataset_id, workdir = get_project(config)
    pair_cfg = dict(config.get("pairing", {}))
    cfg = dict(config.get("evaluation", {}))
    control_h5ad, perturbed_h5ads = _resolve_group_paths(config, preprocessing_output)
    pairing_root = Path(pair_cfg.get("output_root", workdir / "pairing"))
    manifest_path = Path(cfg.get("manifest_path", pairing_root / dataset_id / "pseudo_pairing_repetition_manifest.csv"))
    eval_root = Path(cfg.get("eval_root", workdir / "evaluation" / f"{dataset_id}_pseudo_pairing_evaluation"))
    groups = list(cfg.get("perturbed_groups_to_evaluate", perturbed_h5ads.keys()))
    outputs: dict[str, Any] = {}
    for group in groups:
        if str(group) not in perturbed_h5ads:
            raise KeyError(f"Evaluation group '{group}' is not available in perturbed_h5ads.")
        base = {
            "dataset_id": dataset_id,
            "control_h5ad": control_h5ad,
            "perturbed_h5ad": perturbed_h5ads[str(group)],
            "manifest_path": str(manifest_path),
            "outdir": eval_root / str(group),
            "perturbed_groups_to_evaluate": [str(group)],
        }
        outputs[str(group)] = run_evaluation_pipeline(SimpleNamespace(**_merge(cfg, base)))
    outputs["eval_root"] = str(eval_root)
    return outputs


def run_aggregation(config: Mapping[str, Any], evaluation_output: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from .analysis.common import FINAL_METRICS_BY_TASK, build_default_task_inputs
    from .analysis.runner import run_result_analysis_pipeline

    dataset_id, workdir = get_project(config)
    cfg = dict(config.get("analysis", {}))
    eval_cfg = dict(config.get("evaluation", {}))
    eval_root = Path(cfg.get("eval_root", eval_cfg.get("eval_root", workdir / "evaluation" / f"{dataset_id}_pseudo_pairing_evaluation")))
    configured_groups = cfg.get("perturbed_groups") or eval_cfg.get("perturbed_groups_to_evaluate")
    if configured_groups:
        groups = list(configured_groups)
    else:
        _, perturbed_h5ads = _resolve_group_paths(config)
        groups = list(perturbed_h5ads)
    outputs: dict[str, Any] = {}
    for group in groups:
        group = str(group)
        outdir = Path(cfg.get("output_root", eval_root)) / group / "result_analysis"
        tasks = cfg.get("tasks", ["control_manifold", "perturbation_effect", "mlp_forward", "mlp_inverse"])
        task_inputs = build_default_task_inputs(eval_root, group, tasks=tasks, allow_missing=bool(cfg.get("allow_missing_tasks", True)))
        selection = cfg.get("selection_path")
        if isinstance(selection, Mapping):
            selection = selection.get(group)
        run_final = bool(cfg.get("run_final_comparison", False))
        if run_final and not selection:
            selection = outdir / "selected_variants_TEMPLATE_EDIT_ME.csv"
        base = {
            "dataset_id": dataset_id,
            "perturbed_group": group,
            "eval_root": eval_root,
            "outdir": outdir,
            "task_inputs": task_inputs,
            "run_aggregation": bool(cfg.get("run_aggregation", True)),
            "run_final_comparison": run_final,
            "selection_path": selection,
            "final_metrics_by_task": cfg.get("final_metrics_by_task", FINAL_METRICS_BY_TASK),
        }
        outputs[group] = run_result_analysis_pipeline(SimpleNamespace(**_merge(cfg, base)))
    return outputs


def run_pipeline(config: Mapping[str, Any], stages: Sequence[str] | None = None) -> dict[str, Any]:
    """Run requested stages in dependency order."""
    requested = list(stages or ["acquire", "preprocess", "pair", "evaluate", "aggregate"])
    runner = PipelineRun(config)
    outputs: dict[str, Any] = {}

    if "acquire" in requested and enabled(config, "acquisition", False):
        outputs["acquire"] = runner.execute("acquire", lambda: run_acquisition(config))
    if "preprocess" in requested and enabled(config, "preprocessing", True):
        outputs["preprocess"] = runner.execute(
            "preprocess", lambda: run_preprocessing(config, outputs.get("acquire"))
        )
    if "pair" in requested and enabled(config, "pairing", True):
        outputs["pair"] = runner.execute(
            "pair", lambda: run_pairing(config, outputs.get("preprocess"))
        )
    if "evaluate" in requested and enabled(config, "evaluation", True):
        outputs["evaluate"] = runner.execute(
            "evaluate", lambda: run_evaluation(config, outputs.get("preprocess"))
        )
    if "aggregate" in requested and enabled(config, "analysis", True):
        outputs["aggregate"] = runner.execute(
            "aggregate", lambda: run_aggregation(config, outputs.get("evaluate"))
        )
    return outputs


def pipeline_plan(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id, workdir = get_project(config)
    pair_cfg = dict(config.get("pairing", {}))
    eval_cfg = dict(config.get("evaluation", {}))
    return {
        "dataset_id": dataset_id,
        "workdir": str(workdir),
        "stages": {
            "acquire": enabled(config, "acquisition", False),
            "preprocess": enabled(config, "preprocessing", True),
            "pair": enabled(config, "pairing", True),
            "evaluate": enabled(config, "evaluation", True),
            "aggregate": enabled(config, "analysis", True),
        },
        "pairing_manifest": str(Path(pair_cfg.get("output_root", workdir / "pairing")) / dataset_id / "pseudo_pairing_repetition_manifest.csv"),
        "evaluation_root": str(Path(eval_cfg.get("eval_root", workdir / "evaluation" / f"{dataset_id}_pseudo_pairing_evaluation"))),
        "pipeline_state": str(workdir / "pipeline_runs" / dataset_id / "pipeline_state.json"),
    }
