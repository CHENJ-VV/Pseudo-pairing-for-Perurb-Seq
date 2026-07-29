from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gene_program_pipeline_pkg.shared_paths import resolve_shared_config
from gene_program_pipeline_pkg.utils import load_config


def test_shared_mode_resolves_relative_manifest_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    pairing_root = tmp_path / "pairing"
    eval_root = tmp_path / "evaluation"
    processed_root = tmp_path / "processed"
    pair_dir = pairing_root / "Example_data" / "Example"
    pair_dir.mkdir(parents=True)
    processed_root.mkdir()

    control = processed_root / "control.h5ad"
    perturbed = processed_root / "perturbed.h5ad"
    pseudo = pair_dir / "single" / "S0_naive_mean_control_reference" / "pseudo_control.h5ad"
    pseudo.parent.mkdir(parents=True)
    for path in [control, perturbed, pseudo]:
        path.touch()

    (pair_dir / "run_config.json").write_text(json.dumps({
        "control_h5ad": str(control),
        "perturbed_h5ads": {"single": str(perturbed)},
    }))
    pd.DataFrame([{
        "perturbed_group": "single",
        "strategy": "S0_naive_mean_control_reference",
        "pseudo_control_h5ad": "single/S0_naive_mean_control_reference/pseudo_control.h5ad",
    }]).to_csv(pair_dir / "pseudo_pairing_repetition_manifest.csv", index=False)

    config_dir.mkdir()
    cfg_path = config_dir / "config.json"
    cfg_path.write_text(json.dumps({
        "global": {
            "path_mode": "shared_pseudo_pairing_pipeline",
            "pairing_root": "../pairing",
            "evaluation_root": "../evaluation",
            "processed_data_root": "../processed",
            "dataset_ids": ["Example"],
            "manifest_dataset_names": {"Example": "Example_data"},
            "perturbed_groups": ["single"],
            "only_existing_processed_files": True,
            "require_existing_pseudo_files": True,
        }
    }))

    resolved = resolve_shared_config(load_config(cfg_path))
    dataset = resolved["datasets"][0]
    assert Path(dataset["control_h5ad"]) == control.resolve()
    assert Path(dataset["perturbed_h5ad"]) == perturbed.resolve()
    assert Path(dataset["pseudo_files"][0]["pseudo_control_h5ad"]) == pseudo.resolve()
    assert Path(dataset["output_dir"]).is_absolute()


def test_explicit_mode_resolves_paths_from_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    cfg_path = config_dir / "explicit.json"
    cfg_path.write_text(json.dumps({
        "global": {"path_mode": "explicit"},
        "datasets": [{
            "dataset_id": "example",
            "control_h5ad": "../data/control.h5ad",
            "perturbed_h5ad": "../data/perturbed.h5ad",
            "output_dir": "../output",
            "pseudo_files": [{"pseudo_control_h5ad": "../pseudo/pseudo.h5ad"}],
        }],
    }))
    resolved = resolve_shared_config(load_config(cfg_path))
    dataset = resolved["datasets"][0]
    assert Path(dataset["control_h5ad"]) == (tmp_path / "data" / "control.h5ad").resolve()
    assert Path(dataset["output_dir"]) == (tmp_path / "output").resolve()
    assert Path(dataset["pseudo_files"][0]["pseudo_control_h5ad"]) == (tmp_path / "pseudo" / "pseudo.h5ad").resolve()
