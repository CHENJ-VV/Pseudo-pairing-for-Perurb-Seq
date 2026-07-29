#!/usr/bin/env python3
"""Run expression and encoder-input MLP tasks from the folder-level YAML config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

# Support execution as a file while retaining package-relative imports internally.
FOLDER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FOLDER_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))



def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("The configuration root must be a mapping")
    return data


def resolve(value: Any, base: Path) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return str(path)


def default_output(pseudo_root: Path, name: str) -> Path:
    suffix = {
        "geneformer": "_geneformer_embeddings_chunked",
        "hvg": "_hvg_scvi_embeddings",
        "scvi": "_hvg_scvi_embeddings",
        "sccello": "_sccello_embeddings_optimized_v2",
        "scimilarity": "_scimilarity_embeddings",
        "scgpt": "_scgpt_embeddings",
    }[name]
    return pseudo_root / suffix


def embedding_specs(config: Mapping[str, Any], base: Path, pseudo_root: Path) -> dict[str, dict[str, Any]]:
    outputs = config.get("outputs", {}) or {}

    def out(name: str) -> Path:
        configured = outputs.get(name)
        return Path(resolve(configured, base)) if configured else default_output(pseudo_root, name)

    geneformer = out("geneformer")
    hvg_scvi = out("scvi") if outputs.get("scvi") else out("hvg")
    sccello = out("sccello")
    scimilarity = out("scimilarity")
    scgpt = out("scgpt")
    return {
        "geneformer": {
            "display_name": "Geneformer",
            "npy_candidates": [geneformer / "embeddings" / "{slug}" / "X_geneformer.npy"],
            "obs_candidates": [geneformer / "embeddings" / "{slug}" / "geneformer_obs.csv"],
            "obsm_keys": ["X_geneformer", "X_Geneformer"],
        },
        "hvg": {
            "display_name": "HVG PCA",
            "npy_candidates": [hvg_scvi / "hvg_pca" / "embeddings" / "{slug}" / "X_pca_hvg.npy"],
            "obs_candidates": [hvg_scvi / "hvg_pca" / "embeddings" / "{slug}" / "obs_names.csv"],
            "obsm_keys": ["X_pca_hvg", "X_hvg_pca"],
        },
        "scvi": {
            "display_name": "scVI",
            "npy_candidates": [hvg_scvi / "scvi" / "embeddings" / "{slug}" / "X_scVI.npy"],
            "obs_candidates": [hvg_scvi / "scvi" / "embeddings" / "{slug}" / "obs_names.csv"],
            "obsm_keys": ["X_scVI", "X_scvi"],
        },
        "sccello": {
            "display_name": "scCello",
            "npy_candidates": [sccello / "embeddings" / "{slug}" / "X_scCello.npy"],
            "obs_candidates": [sccello / "embeddings" / "{slug}" / "obs_metadata.csv"],
            "obsm_keys": ["X_scCello", "X_sccello"],
        },
        "scimilarity": {
            "display_name": "SCimilarity",
            "npy_candidates": [scimilarity / "embeddings" / "{slug}" / "X_scimilarity.npy"],
            "obs_candidates": [scimilarity / "embeddings" / "{slug}" / "obs_names.csv"],
            "chunk_candidates": [scimilarity / "embedding_chunks" / "{slug}"],
            "obsm_keys": ["X_scimilarity"],
        },
        "scgpt": {
            "display_name": "scGPT",
            "npy_candidates": [
                scgpt / "embeddings" / "{slug}" / "scgpt_embeddings.npy",
                scgpt / "embeddings" / "{slug}" / "X_scGPT.npy",
            ],
            "obs_candidates": [scgpt / "embeddings" / "{slug}" / "obs_names.csv"],
            "obsm_keys": ["X_scGPT", "X_scgpt"],
        },
    }


def build_config(config: dict[str, Any], config_path: Path) -> SimpleNamespace:
    base = config_path.parent.resolve()
    dataset = config.get("dataset", {}) or {}
    section = config.get("mlp_evaluation", {}) or {}
    pseudo_root_value = resolve(dataset.get("pseudo_root"), base)
    if not pseudo_root_value:
        raise ValueError("dataset.pseudo_root is required")
    pseudo_root = Path(pseudo_root_value)

    required_paths = {
        "control_h5ad": dataset.get("control_h5ad"),
        "perturbed_h5ad": dataset.get("perturbed_h5ad"),
        "manifest_path": dataset.get("pairing_manifest"),
        "evaluation_root": dataset.get("evaluation_root"),
    }
    resolved = {key: resolve(value, base) for key, value in required_paths.items()}
    missing = [key for key, value in resolved.items() if not value]
    if missing:
        raise ValueError("Missing dataset configuration for MLP evaluation: " + ", ".join(missing))

    defaults: dict[str, Any] = {
        "dataset_id": dataset.get("id", "dataset"),
        "control_h5ad": resolved["control_h5ad"],
        "perturbed_h5ad": resolved["perturbed_h5ad"],
        "manifest_path": resolved["manifest_path"],
        "outdir": str(Path(resolved["evaluation_root"]) / dataset.get("perturbed_group", "single")),
        "pseudo_group_root": str(pseudo_root),
        "selected_variants": dataset.get("variants", []),
        "strict_selected_variants": bool(section.get("strict_selected_variants", True)),
        "perturbation_key": dataset.get("perturbation_key", "perturbation_key"),
        "perturbed_groups_to_evaluate": [dataset.get("perturbed_group", "single")],
        "strategies_to_evaluate": None,
        "max_runs_to_evaluate": None,
        "max_eval_genes": 3000,
        "check_shared_genes_across_all_runs": False,
        "models_to_run": ["mlp_expr", "geneformer", "hvg", "scvi", "sccello", "scimilarity", "scgpt"],
        "forward_target_space": "expression",
        "inverse_input_space": "expression",
        "prefer_external_true_embeddings": False,
        "true_perturbed_embedding_slugs": [],
        "true_control_embedding_slugs": [],
        "mlp_tasks": ["forward", "inverse_strategy_delta", "inverse_common_delta"],
        "device": "auto",
        "split_seed": 42,
        "model_seed": 42,
        "train_frac": 0.70,
        "val_frac": 0.15,
        "test_frac": 0.15,
        "batch_size": 256,
        "num_workers": 0,
        "forward_epochs": 30,
        "inverse_epochs": 30,
        "early_stop_patience": 3,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "hidden_dim": 1024,
        "latent_dim": 512,
        "pert_emb_dim": 256,
        "dropout": 0.15,
        "max_grad_norm": 5.0,
        "min_cells_per_perturbation": 20,
        "inverse_topk_accuracy_list": [3, 5, 10],
        "topk_effect_genes": [20, 50, 100],
        "skip_existing_forward": True,
        "skip_existing_inverse": True,
        "save_inverse_predictions": True,
        "return_inverse_probabilities": False,
        "save_mlp_checkpoints": False,
        "save_runtime_cache": False,
        "continue_on_error": True,
        "embedding_specs": embedding_specs(config, base, pseudo_root),
    }
    for key, value in section.items():
        defaults[key] = value
    return SimpleNamespace(**defaults)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--show-config", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = build_config(load_yaml(config_path), config_path)
    if args.show_config:
        printable = {key: str(value) if isinstance(value, Path) else value for key, value in vars(config).items()}
        print(json.dumps(printable, indent=2, default=str))
        return 0
    from foundation_model_encoders.mlp_tasks.eval_mlp_task_models import run_mlp_task_model_evaluation

    outputs = run_mlp_task_model_evaluation(config)
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
