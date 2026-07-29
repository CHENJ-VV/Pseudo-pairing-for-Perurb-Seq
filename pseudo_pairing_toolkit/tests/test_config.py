from pathlib import Path

from pseudopair.config import load_config, validate_config
from pseudopair.pipeline import pipeline_plan


def test_example_config_loads():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "pipeline.example.yaml")
    assert config["project"]["dataset_id"] == "Replogle_K562_essential"
    assert not validate_config(config, check_files=False)
    plan = pipeline_plan(config)
    assert plan["stages"]["pair"] is True
    assert plan["pairing_manifest"].endswith("pseudo_pairing_repetition_manifest.csv")


def test_nested_perturbed_paths_are_resolved(tmp_path):
    cfg = tmp_path / "x.yaml"
    cfg.write_text(
        """
project:
  dataset_id: toy
  workdir: ./work
preprocessing:
  enabled: false
  input_h5ad: ./raw.h5ad
pairing:
  enabled: true
  control_h5ad: ./control.h5ad
  perturbed_h5ads:
    single: ./single.h5ad
  strategies_to_run: [S0_naive_mean_control_reference]
evaluation:
  enabled: false
analysis:
  enabled: false
"""
    )
    loaded = load_config(cfg)
    assert loaded["pairing"]["perturbed_h5ads"]["single"] == str((tmp_path / "single.h5ad").resolve())
