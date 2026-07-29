#!/usr/bin/env python3
"""Configuration-driven launcher for the foundation-model encoder branch."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
MLP_TASKS = ROOT / "mlp_tasks"

MODEL_SCRIPTS = {
    "geneformer": SCRIPTS / "Geneformer_tokenize_emb.py",
    "scgpt": SCRIPTS / "scGPT_token_emb.py",
    "sccello": SCRIPTS / "scCello_token_emb.py",
    "scimilarity": SCRIPTS / "Scimilarity_tokenize.py",
    "scvi": SCRIPTS / "scVI_token_emb.py",
    "hvg": SCRIPTS / "scVI_token_emb.py",
    "umap": SCRIPTS / "UMAP_plot.py",
    "mlp": MLP_TASKS / "run_mlp_tasks.py",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The configuration root must be a YAML mapping")
    return data


def nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def as_path(value: Any, *, base: Path) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def set_if(env: dict[str, str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (list, dict)):
        env[name] = json.dumps(value)
    elif isinstance(value, bool):
        env[name] = "true" if value else "false"
    else:
        env[name] = str(value)


def model_output_default(model: str, pseudo_root: str) -> str:
    suffixes = {
        "geneformer": "_geneformer_embeddings_chunked",
        "scgpt": "_scgpt_embeddings",
        "sccello": "_sccello_embeddings_optimized_v2",
        "scimilarity": "_scimilarity_embeddings",
        "scvi": "_hvg_scvi_embeddings",
        "hvg": "_hvg_scvi_embeddings",
    }
    return str(Path(pseudo_root) / suffixes[model])


def build_environment(config: dict[str, Any], config_path: Path, model: str) -> dict[str, str]:
    base = config_path.parent.resolve()
    env = os.environ.copy()
    # Scripts such as mlp_tasks/run_mlp_tasks.py import the top-level
    # foundation_model_encoders package. Add the repository root to PYTHONPATH
    # so execution is independent of the caller's current working directory.
    existing_pythonpath = env.get("PYTHONPATH", "")
    repo_root = str(ROOT.parent)
    env["PYTHONPATH"] = repo_root if not existing_pythonpath else repo_root + os.pathsep + existing_pythonpath
    dataset = config.get("dataset", {}) or {}
    repositories = config.get("repositories", {}) or {}
    checkpoints = config.get("checkpoints", {}) or {}
    outputs = config.get("outputs", {}) or {}
    settings = config.get("settings", {}) or {}
    model_settings = (settings.get(model, {}) or {}) if isinstance(settings, Mapping) else {}

    pseudo_root = as_path(dataset.get("pseudo_root"), base=base)
    if model not in {"umap", "mlp"} and not pseudo_root:
        raise ValueError("dataset.pseudo_root is required")

    set_if(env, "PPFM_CONFIG_PATH", str(config_path.resolve()))
    set_if(env, "PPFM_DATASET_ID", dataset.get("id", "dataset"))
    set_if(env, "PPFM_PERTURBED_GROUP", dataset.get("perturbed_group", "single"))
    set_if(env, "PPFM_PSEUDO_ROOT", pseudo_root)
    set_if(env, "PPFM_VARIANTS", dataset.get("variants", []))
    set_if(env, "PPFM_CONTINUE_ON_ERROR", settings.get("continue_on_error", True))

    if model in {"geneformer", "scgpt", "sccello", "scimilarity", "scvi", "hvg"}:
        configured_output = outputs.get(model)
        output_root = as_path(configured_output, base=base) if configured_output else model_output_default(model, pseudo_root)
        set_if(env, "PPFM_OUTPUT_ROOT", output_root)

    if model == "geneformer":
        set_if(env, "PPFM_GENEFORMER_REPO", as_path(repositories.get("geneformer"), base=base))
        set_if(env, "PPFM_GENEFORMER_MODEL", as_path(checkpoints.get("geneformer_model"), base=base))
        token_dict = checkpoints.get("geneformer_token_dictionary")
        set_if(env, "PPFM_GENEFORMER_TOKEN_DICT", as_path(token_dict, base=base) if token_dict else None)
    elif model == "scgpt":
        set_if(env, "PPFM_SCGPT_REPO", as_path(repositories.get("scgpt"), base=base))
        set_if(env, "PPFM_SCGPT_MODEL_DIR", as_path(checkpoints.get("scgpt_model_dir"), base=base))
        vocab = checkpoints.get("scgpt_vocab_file")
        set_if(env, "PPFM_SCGPT_VOCAB_FILE", as_path(vocab, base=base) if vocab else None)
    elif model == "sccello":
        set_if(env, "PPFM_SCCELLO_REPO", as_path(repositories.get("sccello"), base=base))
        token_dict = checkpoints.get("sccello_token_dictionary")
        set_if(env, "PPFM_SCCELLO_TOKEN_DICT", as_path(token_dict, base=base) if token_dict else None)
        set_if(env, "PPFM_SCCELLO_CHECKPOINT", checkpoints.get("sccello_checkpoint", "katarinayuan/scCello-zeroshot"))
    elif model == "scimilarity":
        set_if(env, "PPFM_SCIMILARITY_REPO", as_path(repositories.get("scimilarity"), base=base))
        set_if(env, "PPFM_SCIMILARITY_MODEL", as_path(checkpoints.get("scimilarity_model"), base=base))
    elif model in {"scvi", "hvg"}:
        if model == "hvg":
            set_if(env, "PPFM_RUN_HVG_PCA", True)
            set_if(env, "PPFM_RUN_SCVI", False)
        else:
            set_if(env, "PPFM_RUN_HVG_PCA", model_settings.get("run_hvg_pca", True))
            set_if(env, "PPFM_RUN_SCVI", model_settings.get("run_scvi", True))

    common_mapping = {
        "overwrite": "PPFM_OVERWRITE",
        "write_h5ad": "PPFM_WRITE_H5AD",
        "subset_n_cells": "PPFM_SUBSET_N_CELLS",
        "max_variants": "PPFM_MAX_VARIANTS",
        "use_gpu": "PPFM_USE_GPU",
        "batch_size": "PPFM_BATCH_SIZE",
        "num_workers": "PPFM_NUM_WORKERS",
        "chunk_cells": "PPFM_CHUNK_CELLS",
        "counts_source": "PPFM_COUNTS_SOURCE",
        "counts_layer": "PPFM_COUNTS_LAYER",
    }
    for key, env_name in common_mapping.items():
        if key in model_settings:
            set_if(env, env_name, model_settings[key])

    if model == "umap":
        set_if(env, "PPFM_PSEUDO_ROOT", pseudo_root)
        set_if(env, "PPFM_TRUE_PERTURBED_H5AD", as_path(dataset.get("perturbed_h5ad"), base=base))
        set_if(env, "PPFM_OUTPUT_ROOT", as_path(outputs.get("umap", ROOT / "outputs" / "umap"), base=base))
        set_if(env, "PPFM_MODELS", model_settings.get("models", []))
        set_if(env, "PPFM_LEIDEN_KEY", model_settings.get("leiden_key"))
        set_if(env, "PPFM_SUBSAMPLE_N_CELLS", model_settings.get("subsample_n_cells"))

    return env


def python_for_model(config: Mapping[str, Any], model: str, base: Path) -> str:
    executables = nested(config, "environments", "python", default={}) or {}
    value = executables.get(model) or executables.get("default") or sys.executable
    resolved = as_path(value, base=base) if ("/" in str(value) or str(value).startswith(".")) else str(value)
    return resolved or sys.executable


def validate(config: dict[str, Any], config_path: Path, check_paths: bool) -> list[str]:
    errors: list[str] = []
    dataset = config.get("dataset", {}) or {}
    if not dataset.get("pseudo_root"):
        errors.append("dataset.pseudo_root is required")
    if not dataset.get("id"):
        errors.append("dataset.id is required")

    for model in ["geneformer", "scgpt", "sccello", "scimilarity"]:
        if nested(config, "enabled", model, default=False):
            repo = nested(config, "repositories", model)
            if not repo:
                errors.append(f"repositories.{model} is required when {model} is enabled")

    required_checkpoints = {
        "geneformer": "geneformer_model",
        "scgpt": "scgpt_model_dir",
        "scimilarity": "scimilarity_model",
    }
    for model, key in required_checkpoints.items():
        if nested(config, "enabled", model, default=False) and not nested(config, "checkpoints", key):
            errors.append(f"checkpoints.{key} is required when {model} is enabled")

    if check_paths:
        base = config_path.parent.resolve()
        candidates: list[tuple[str, Any]] = [("dataset.pseudo_root", dataset.get("pseudo_root"))]
        for key, value in (config.get("repositories", {}) or {}).items():
            if value:
                candidates.append((f"repositories.{key}", value))
        for key, value in (config.get("checkpoints", {}) or {}).items():
            if value and key != "sccello_checkpoint" and not str(value).startswith(("http://", "https://")):
                candidates.append((f"checkpoints.{key}", value))
        for label, value in candidates:
            path = as_path(value, base=base)
            if path and not Path(path).exists():
                errors.append(f"{label} does not exist: {path}")
    return errors


def selected_models(config: Mapping[str, Any], requested: list[str]) -> list[str]:
    if requested:
        models = requested
    else:
        enabled = config.get("enabled", {}) or {}
        models = [name for name, is_enabled in enabled.items() if is_enabled]
    normalized: list[str] = []
    for model in models:
        name = model.lower()
        if name == "all":
            return ["geneformer", "scgpt", "sccello", "scimilarity", "scvi", "umap", "mlp"]
        if name not in MODEL_SCRIPTS:
            raise ValueError(f"Unknown model/stage {model!r}. Choices: {', '.join(MODEL_SCRIPTS)}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def command_for(config: dict[str, Any], config_path: Path, model: str) -> tuple[list[str], dict[str, str]]:
    base = config_path.parent.resolve()
    python = python_for_model(config, model, base)
    script = MODEL_SCRIPTS[model]
    command = [python, str(script)]
    if model == "mlp":
        command.extend(["--config", str(config_path.resolve())])
    return command, build_environment(config, config_path, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate the YAML configuration")
    validate_parser.add_argument("--config", required=True, type=Path)
    validate_parser.add_argument("--check-paths", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show resolved commands without executing them")
    show_parser.add_argument("--config", required=True, type=Path)
    show_parser.add_argument("--model", action="append", default=[])

    run_parser = subparsers.add_parser("run", help="Run one or more encoder/evaluation stages")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--model", action="append", default=[])
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)

    if args.command == "validate":
        errors = validate(config, config_path, args.check_paths)
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 2
        print(f"Configuration is valid: {config_path}")
        return 0

    models = selected_models(config, args.model)
    if not models:
        raise SystemExit("No stages selected. Enable stages in YAML or pass --model.")

    for model in models:
        command, env = command_for(config, config_path, model)
        print(f"[{model}] {shlex.join(command)}")
        if args.command == "show" or getattr(args, "dry_run", False):
            continue
        completed = subprocess.run(command, env=env, check=False)
        if completed.returncode != 0:
            print(f"[{model}] failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
