"""Execution/orchestration layer for S0-S5 pseudo-control dataset generation.

This file is intentionally dataset-agnostic.  Edit the CONFIG block in the
companion notebook, or import run_pseudo_pairing_repetition_plan(config) from a
Jupyter notebook.
"""
from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pseudo_pairing_seacell import ensure_memberships_for_settings, membership_prefix_for_setting
from pseudo_pairing_strategies import (
    S0,
    S1,
    S2,
    S3,
    S4,
    S5,
    compute_ot_assignments_for_setting_topk,
    load_seacell_context,
    run_s0_naive_mean_control_reference,
    run_s1_random_single_control,
    run_s2_random_average_controls,
    run_s3_seacell_metacell_average,
    run_s4_seacell_balanced_random_sample,
    run_s5_seacell_ot_sampled_average,
)
from pseudo_pairing_utils import STRATEGY_ORDER, as_namespace, ensure_outdir, namespace_to_dict, save_json


DEFAULT_STRATEGIES = STRATEGY_ORDER.copy()


def _config_get(config: SimpleNamespace, name: str, default=None):
    return getattr(config, name, default)


def _make_base_args(config: SimpleNamespace, perturbed_group: str, perturbed_h5ad: str | Path) -> SimpleNamespace:
    """Build args passed to individual strategy functions."""
    fields = dict(
        dataset_id=str(config.dataset_id),
        perturbed_group=str(perturbed_group),
        control_h5ad=str(config.control_h5ad),
        perturbed_h5ad=str(perturbed_h5ad),
        expr_layer=_config_get(config, "expr_layer", "X"),
        embedding_key=_config_get(config, "embedding_key", "X_pca"),
        perturbation_key=_config_get(config, "perturbation_key", "auto"),
        require_all_genes=bool(_config_get(config, "require_all_genes", False)),
        batch_size=int(_config_get(config, "batch_size", _config_get(config, "matrix_batch_size", 4096))),
        matrix_batch_size=int(_config_get(config, "matrix_batch_size", _config_get(config, "batch_size", 4096))),
        max_pairs_per_perturbation=_config_get(config, "max_pairs_per_perturbation", None),
        pair_selection_seed=int(_config_get(config, "pair_selection_seed", 42)),
        sampling_replace=bool(_config_get(config, "sampling_replace", True)),
        store_sampled_control_positions=bool(_config_get(config, "store_sampled_control_positions", True)),
        seacell_key_prefix=str(_config_get(config, "seacell_key_prefix", "SEACell")),
        n_waypoint_eigs=int(_config_get(config, "n_waypoint_eigs", 10)),
        seacells_n_iter=int(_config_get(config, "seacells_n_iter", 50)),
        seacell_seed=int(_config_get(config, "seacell_seed", 42)),
        use_gpu_seacells=bool(_config_get(config, "use_gpu_seacells", False)),
        sample_metacells_with_replacement=bool(_config_get(config, "sample_metacells_with_replacement", True)),
        sample_cells_per_metacell=int(_config_get(config, "sample_cells_per_metacell", 10)),
        ot_reg=float(_config_get(config, "ot_reg", 0.05)),
        ot_max_iter=int(_config_get(config, "ot_max_iter", 2000)),
        ot_tol=float(_config_get(config, "ot_tol", 1e-7)),
        cost_metric=str(_config_get(config, "cost_metric", "sqeuclidean")),
        control_mass=str(_config_get(config, "control_mass", "size")),
        overwrite_assignments=bool(_config_get(config, "overwrite_assignments", False)),
        overwrite_sampled_outputs=bool(_config_get(config, "overwrite_sampled_outputs", False)),
    )
    return SimpleNamespace(**fields)


def _append_manifest(records: list[dict[str, Any]], manifest_path: Path) -> pd.DataFrame:
    manifest = pd.DataFrame(records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def run_pseudo_pairing_repetition_plan(config: Mapping[str, Any] | SimpleNamespace) -> pd.DataFrame:
    """Run S0-S5 pseudo-control construction over perturbed groups and repeat grids.

    Required config fields
    ----------------------
    dataset_id: str
    control_h5ad: str/path
    perturbed_h5ads: dict[str, str/path]
    outdir: str/path

    Common optional fields
    ----------------------
    strategies_to_run: list[str]
    random_seeds: list[int]
    s2_n_control_cells_to_average_values: list[int]
    seacell_settings: list[dict]
    s3_n_metacells_to_average_values: list[int]
    s5_top_k_values: list[int]
    """
    config = as_namespace(config)
    strategies = list(_config_get(config, "strategies_to_run", DEFAULT_STRATEGIES))
    invalid = [s for s in strategies if s not in STRATEGY_ORDER]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}. Valid strategies: {STRATEGY_ORDER}")

    out_root = ensure_outdir(Path(config.outdir) / str(config.dataset_id))
    save_json(namespace_to_dict(config), out_root / "run_config.json")

    random_seeds = [int(s) for s in _config_get(config, "random_seeds", [0, 1, 2, 3, 4])]
    s2_k_values = [int(k) for k in _config_get(config, "s2_n_control_cells_to_average_values", [100])]
    s3_k_values = [int(k) for k in _config_get(config, "s3_n_metacells_to_average_values", [3])]
    s5_top_k_values = [int(k) for k in _config_get(config, "s5_top_k_values", [1, 3, 5])]
    seacell_settings = list(_config_get(config, "seacell_settings", []))

    needs_seacells = any(s in strategies for s in [S3, S4, S5])
    if needs_seacells and len(seacell_settings) == 0:
        raise ValueError("S3/S4/S5 require config.seacell_settings to be non-empty.")

    manifest_records: list[dict[str, Any]] = []
    manifest_path = out_root / "pseudo_pairing_repetition_manifest.csv"

    perturbed_h5ads = dict(config.perturbed_h5ads)
    for group_name, perturbed_h5ad in perturbed_h5ads.items():
        print("\n" + "#" * 110)
        print(f"[Perturbed group] {group_name}")
        print(f"[Perturbed h5ad]  {perturbed_h5ad}")
        print("#" * 110)
        group_root = ensure_outdir(out_root / str(group_name))
        base_args = _make_base_args(config, group_name, perturbed_h5ad)

        membership_root = Path(_config_get(config, "membership_root", group_root / "_seacell_memberships"))
        membership_paths = {}
        if needs_seacells:
            membership_paths = ensure_memberships_for_settings(
                control_h5ad=base_args.control_h5ad,
                membership_root=membership_root,
                seacell_settings=seacell_settings,
                base_args=base_args,
                overwrite=bool(_config_get(config, "overwrite_memberships", False)),
            )

        # S0 deterministic baseline: one run per perturbed group.
        if S0 in strategies:
            run_outdir = group_root / S0
            result = run_s0_naive_mean_control_reference(base_args, run_outdir)
            qc = result["qc"]
            manifest_records.append(
                {
                    "dataset_id": config.dataset_id,
                    "perturbed_group": group_name,
                    "strategy": S0,
                    "sampling_seed": np.nan,
                    "parameter_label": "global_control_mean",
                    "pseudo_control_h5ad": str(result["output_h5ad"]),
                    "pair_metadata_path": str(result["pair_metadata_path"]),
                    "outdir": str(result["outdir"]),
                    **{k: qc.get(k) for k in ["n_pairs_written", "n_genes_output", "n_control_cells"]},
                }
            )
            _append_manifest(manifest_records, manifest_path)
            gc.collect()

        # S1 repeated random single controls.
        if S1 in strategies:
            for seed in random_seeds:
                run_outdir = group_root / S1 / f"seed_{seed:03d}"
                result = run_s1_random_single_control(base_args, run_outdir, seed=seed)
                qc = result["qc"]
                manifest_records.append(
                    {
                        "dataset_id": config.dataset_id,
                        "perturbed_group": group_name,
                        "strategy": S1,
                        "sampling_seed": seed,
                        "parameter_label": f"seed_{seed:03d}",
                        "pseudo_control_h5ad": str(result["output_h5ad"]),
                        "pair_metadata_path": str(result["pair_metadata_path"]),
                        "outdir": str(result["outdir"]),
                        **{k: qc.get(k) for k in ["n_pairs_written", "n_genes_output", "n_control_cells", "n_unique_control_cells_used"]},
                    }
                )
                _append_manifest(manifest_records, manifest_path)
                del result
                gc.collect()

        # S2 repeated random averaged controls.
        if S2 in strategies:
            for k in s2_k_values:
                for seed in random_seeds:
                    run_outdir = group_root / S2 / f"k_{k:03d}" / f"seed_{seed:03d}"
                    result = run_s2_random_average_controls(base_args, run_outdir, seed=seed, n_control_cells_to_average=k)
                    qc = result["qc"]
                    manifest_records.append(
                        {
                            "dataset_id": config.dataset_id,
                            "perturbed_group": group_name,
                            "strategy": S2,
                            "sampling_seed": seed,
                            "n_control_cells_to_average": k,
                            "parameter_label": f"k_{k}_seed_{seed:03d}",
                            "pseudo_control_h5ad": str(result["output_h5ad"]),
                            "pair_metadata_path": str(result["pair_metadata_path"]),
                            "outdir": str(result["outdir"]),
                            **{k2: qc.get(k2) for k2 in ["n_pairs_written", "n_genes_output", "n_control_cells", "n_unique_control_cells_used"]},
                        }
                    )
                    _append_manifest(manifest_records, manifest_path)
                    del result
                    gc.collect()

        # S3-S5 reuse one loaded context per SEACell setting.
        for setting in seacell_settings if needs_seacells else []:
            setting_id = str(setting["setting_id"])
            membership_prefix = membership_paths.get(setting_id, membership_prefix_for_setting(membership_root, setting_id))
            setting_args = deepcopy(base_args)
            setting_args.seacell_setting_id = setting_id
            setting_args.n_metacells = int(setting["n_metacells"])
            setting_args.n_waypoint_eigs = int(setting.get("n_waypoint_eigs", _config_get(config, "n_waypoint_eigs", 10)))
            setting_args.seacells_n_iter = int(setting.get("seacells_n_iter", _config_get(config, "seacells_n_iter", 50)))
            setting_args.seacell_seed = int(setting.get("seacell_seed", _config_get(config, "seacell_seed", 42)))

            context = load_seacell_context(setting_args, membership_prefix)
            n_metacells_observed = int(len(context["metacell_ids"]))

            if S3 in strategies:
                for k in s3_k_values:
                    for seed in random_seeds:
                        run_outdir = group_root / S3 / setting_id / f"k_{k:02d}" / f"seed_{seed:03d}"
                        result = run_s3_seacell_metacell_average(
                            setting_args,
                            context,
                            run_outdir,
                            seed=seed,
                            n_metacells_to_average=k,
                        )
                        qc = result["qc"]
                        manifest_records.append(
                            {
                                "dataset_id": config.dataset_id,
                                "perturbed_group": group_name,
                                "strategy": S3,
                                "seacell_setting_id": setting_id,
                                "n_metacells_requested": int(setting["n_metacells"]),
                                "n_metacells_observed": n_metacells_observed,
                                "n_metacells_to_average": k,
                                "sampling_seed": seed,
                                "parameter_label": f"{setting_id}_k_{k}_seed_{seed:03d}",
                                "membership_path_for_metacell_coverage": str(membership_prefix),
                                "pseudo_control_h5ad": str(result["output_h5ad"]),
                                "pair_metadata_path": str(result["pair_metadata_path"]),
                                "outdir": str(result["outdir"]),
                                **{k2: qc.get(k2) for k2 in ["n_pairs_written", "n_genes_output", "n_control_cells", "n_unique_metacells_used"]},
                            }
                        )
                        _append_manifest(manifest_records, manifest_path)
                        del result
                        gc.collect()

            if S4 in strategies:
                for seed in random_seeds:
                    run_outdir = group_root / S4 / setting_id / f"seed_{seed:03d}"
                    result = run_s4_seacell_balanced_random_sample(setting_args, context, run_outdir, seed=seed)
                    qc = result["qc"]
                    manifest_records.append(
                        {
                            "dataset_id": config.dataset_id,
                            "perturbed_group": group_name,
                            "strategy": S4,
                            "seacell_setting_id": setting_id,
                            "n_metacells_requested": int(setting["n_metacells"]),
                            "n_metacells_observed": n_metacells_observed,
                            "sampling_seed": seed,
                            "parameter_label": f"{setting_id}_seed_{seed:03d}",
                            "membership_path_for_metacell_coverage": str(membership_prefix),
                            "pseudo_control_h5ad": str(result["output_h5ad"]),
                            "pair_metadata_path": str(result["pair_metadata_path"]),
                            "outdir": str(result["outdir"]),
                            **{k2: qc.get(k2) for k2 in ["n_pairs_written", "n_genes_output", "n_control_cells", "n_unique_control_cells_used"]},
                        }
                    )
                    _append_manifest(manifest_records, manifest_path)
                    del result
                    gc.collect()

            if S5 in strategies:
                assignment_root = group_root / S5 / setting_id / "ot_assignments"
                for top_k in s5_top_k_values:
                    assignment_df, assignment_qc, assignment_path = compute_ot_assignments_for_setting_topk(
                        setting_args,
                        context,
                        top_k=top_k,
                        assignment_outdir=assignment_root,
                    )
                    for seed in random_seeds:
                        run_outdir = group_root / S5 / setting_id / f"topk_{top_k:02d}" / f"seed_{seed:03d}"
                        result = run_s5_seacell_ot_sampled_average(
                            setting_args,
                            context,
                            assignment_df,
                            run_outdir,
                            seed=seed,
                            top_k=top_k,
                        )
                        qc = result["qc"]
                        manifest_records.append(
                            {
                                "dataset_id": config.dataset_id,
                                "perturbed_group": group_name,
                                "strategy": S5,
                                "seacell_setting_id": setting_id,
                                "n_metacells_requested": int(setting["n_metacells"]),
                                "n_metacells_observed": n_metacells_observed,
                                "top_k_metacells": top_k,
                                "sample_cells_per_metacell": int(_config_get(config, "sample_cells_per_metacell", 10)),
                                "sampling_seed": seed,
                                "parameter_label": f"{setting_id}_topk_{top_k}_seed_{seed:03d}",
                                "membership_path_for_metacell_coverage": str(membership_prefix),
                                "assignment_path": str(assignment_path),
                                "pseudo_control_h5ad": str(result["output_h5ad"]),
                                "pair_metadata_path": str(result["pair_metadata_path"]),
                                "outdir": str(result["outdir"]),
                                **{k2: qc.get(k2) for k2 in [
                                    "n_pairs_written",
                                    "n_genes_output",
                                    "n_control_cells",
                                    "mean_dominant_weight",
                                    "mean_matching_entropy",
                                    "mean_dominant_distance",
                                ]},
                            }
                        )
                        _append_manifest(manifest_records, manifest_path)
                        del result
                        gc.collect()
                    del assignment_df
                    gc.collect()

            # Release one setting context before next setting.
            del context
            gc.collect()

    manifest = _append_manifest(manifest_records, manifest_path)
    print("\n" + "=" * 100)
    print(f"[Done] Manifest saved: {manifest_path}")
    print(f"[Done] Generated/recorded datasets: {manifest.shape[0]:,}")
    print("=" * 100)
    return manifest
