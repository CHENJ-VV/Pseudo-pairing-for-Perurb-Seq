"""Configuration loading, path normalization, and lightweight validation."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

STAGES = ("acquire", "preprocess", "pair", "evaluate", "aggregate")

_PATH_KEYS = {
    "workdir", "input", "input_h5ad", "output", "output_dir", "output_root",
    "raw_h5ad", "control_h5ad", "perturbed_h5ad", "manifest_path", "eval_root",
    "selection_path", "membership_root", "split_dir", "outdir",
}


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _resolve_paths(value: Any, base_dir: Path, key: str | None = None) -> Any:
    if isinstance(value, dict):
        if key in {"perturbed_h5ads", "group_paths"}:
            return {k: _resolve_paths(v, base_dir, "input_h5ad") for k, v in value.items()}
        return {k: _resolve_paths(v, base_dir, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(v, base_dir, key) for v in value]
    if isinstance(value, str) and key is not None:
        is_path = key in _PATH_KEYS or key.endswith(("_path", "_dir", "_root", "_h5ad"))
        if is_path and value and "://" not in value:
            path = Path(value)
            return str(path if path.is_absolute() else (base_dir / path).resolve())
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML/JSON config and resolve relative paths against its directory."""
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text()) or {}
    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
    else:
        raise ValueError("Configuration must be YAML or JSON.")
    if not isinstance(raw, Mapping):
        raise TypeError("Top-level configuration must be a mapping.")
    config = _resolve_paths(_expand(deepcopy(dict(raw))), path.parent)
    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)
    return config


def get_project(config: Mapping[str, Any]) -> tuple[str, Path]:
    project = dict(config.get("project", {}))
    dataset_id = str(project.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("project.dataset_id is required.")
    workdir = Path(project.get("workdir", "./pseudopair_work")).expanduser().resolve()
    return dataset_id, workdir


def enabled(config: Mapping[str, Any], section: str, default: bool = True) -> bool:
    block = config.get(section, {})
    return bool(block.get("enabled", default)) if isinstance(block, Mapping) else default


def validate_config(config: Mapping[str, Any], check_files: bool = False) -> list[str]:
    """Return validation errors without importing scientific dependencies."""
    errors: list[str] = []
    try:
        dataset_id, _ = get_project(config)
    except Exception as exc:
        errors.append(str(exc))
        dataset_id = ""

    if enabled(config, "acquisition", False):
        files = config.get("acquisition", {}).get("files", [])
        if not files:
            errors.append("acquisition.enabled=true but acquisition.files is empty.")
        for idx, item in enumerate(files):
            if not isinstance(item, Mapping):
                errors.append(f"acquisition.files[{idx}] must be a mapping.")
                continue
            if not item.get("url") and not item.get("source_path"):
                errors.append(f"acquisition.files[{idx}] needs url or source_path.")
            if not item.get("output"):
                errors.append(f"acquisition.files[{idx}].output is required.")

    if enabled(config, "preprocessing", True):
        pp = config.get("preprocessing", {})
        input_h5ad = pp.get("input_h5ad") or pp.get("input")
        if not input_h5ad and not enabled(config, "acquisition", False):
            errors.append("preprocessing.input_h5ad is required when acquisition is disabled.")
        if check_files and input_h5ad and not Path(input_h5ad).exists():
            errors.append(f"Preprocessing input does not exist: {input_h5ad}")

    if enabled(config, "pairing", True):
        pairing = config.get("pairing", {})
        strategies = pairing.get("strategies_to_run", [])
        valid = {
            "S0_naive_mean_control_reference", "S1_random_single_control",
            "S2_random_average_controls", "S3_SEACell_metacell_average",
            "S4_SEACell_balanced_random_sample", "S5_SEACell_OT_sampled_average",
        }
        unknown = sorted(set(strategies) - valid) if strategies else []
        if unknown:
            errors.append(f"Unknown pairing strategies: {unknown}")
        if any(str(s).startswith(("S3_", "S4_", "S5_")) for s in strategies):
            if not pairing.get("seacell_settings"):
                errors.append("S3/S4/S5 require pairing.seacell_settings.")
        if check_files:
            control = pairing.get("control_h5ad")
            if control and not Path(control).exists():
                errors.append(f"Pairing control_h5ad does not exist: {control}")
            for group, path in dict(pairing.get("perturbed_h5ads", {})).items():
                if not Path(path).exists():
                    errors.append(f"Pairing perturbed_h5ads[{group}] does not exist: {path}")

    if dataset_id and any(c in dataset_id for c in "/\\"):
        errors.append("project.dataset_id must be a name, not a path.")
    return errors


def dump_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in config.items() if not str(k).startswith("_")}
    path.write_text(yaml.safe_dump(clean, sort_keys=False))
    return path
