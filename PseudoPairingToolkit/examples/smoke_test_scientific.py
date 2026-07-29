"""End-to-end synthetic smoke test for an environment with Scanpy/AnnData installed."""
from __future__ import annotations

import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from pseudopair.pipeline import run_pipeline


def make_data(root: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(7)
    genes = [f"g{i}" for i in range(30)]

    control_x = rng.poisson(2.0, size=(80, len(genes))).astype(np.float32)
    control = ad.AnnData(control_x, obs=pd.DataFrame(index=[f"c{i}" for i in range(80)]))
    control.var_names = genes
    control.obsm["X_pca"] = rng.normal(size=(80, 8)).astype(np.float32)

    labels = np.repeat(["pertA", "pertB"], 30)
    effects = np.vstack([
        np.r_[np.full(5, 1.5), np.zeros(len(genes) - 5)],
        np.r_[np.zeros(5), np.full(5, 1.5), np.zeros(len(genes) - 10)],
    ])
    perturbed_x = rng.poisson(2.0, size=(60, len(genes))).astype(np.float32)
    perturbed_x += effects[(labels == "pertB").astype(int)].astype(np.float32)
    perturbed = ad.AnnData(
        perturbed_x,
        obs=pd.DataFrame({"perturbation_key": labels}, index=[f"p{i}" for i in range(60)]),
    )
    perturbed.var_names = genes
    perturbed.obsm["X_pca"] = rng.normal(size=(60, 8)).astype(np.float32)

    control_path = root / "control.h5ad"
    perturbed_path = root / "single.h5ad"
    control.write_h5ad(control_path)
    perturbed.write_h5ad(perturbed_path)
    return control_path, perturbed_path


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="pseudopair-smoke-") as tmp:
        root = Path(tmp)
        control, perturbed = make_data(root)
        config = {
            "project": {"dataset_id": "toy", "workdir": str(root / "work")},
            "acquisition": {"enabled": False},
            "preprocessing": {"enabled": False, "input_h5ad": str(perturbed)},
            "pairing": {
                "enabled": True,
                "control_h5ad": str(control),
                "perturbed_h5ads": {"single": str(perturbed)},
                "output_root": str(root / "pairing"),
                "strategies_to_run": [
                    "S0_naive_mean_control_reference",
                    "S1_random_single_control",
                    "S2_random_average_controls",
                ],
                "random_seeds": [0],
                "s2_n_control_cells_to_average_values": [10],
                "perturbation_key": "perturbation_key",
                "embedding_key": "X_pca",
            },
            "evaluation": {
                "enabled": True,
                "eval_root": str(root / "evaluation"),
                "perturbed_groups_to_evaluate": ["single"],
                "evaluation_tasks": ["control_manifold", "perturbation_effect"],
                "perturbation_key": "perturbation_key",
                "max_eval_genes": 30,
                "n_pcs": 8,
                "n_control_sample_for_pca": 80,
                "n_control_sample_for_overlap": 80,
                "n_pseudo_sample_for_overlap": 60,
                "n_mmd_sample": 60,
                "source_mixing_k": 10,
                "min_cells_per_perturbation": 5,
            },
            "analysis": {
                "enabled": True,
                "run_aggregation": True,
                "run_final_comparison": False,
                "perturbed_groups": ["single"],
                "tasks": ["control_manifold", "perturbation_effect"],
                "allow_missing_tasks": False,
            },
        }
        outputs = run_pipeline(config, stages=["pair", "evaluate", "aggregate"])
        print(outputs)
        assert (root / "pairing" / "toy" / "pseudo_pairing_repetition_manifest.csv").exists()
        assert (root / "evaluation" / "single" / "result_analysis" / "selected_variants_TEMPLATE_EDIT_ME.csv").exists()


if __name__ == "__main__":
    main()
