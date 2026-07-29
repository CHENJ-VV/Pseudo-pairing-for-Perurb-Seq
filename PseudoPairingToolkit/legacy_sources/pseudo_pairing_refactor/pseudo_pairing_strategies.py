"""Pseudo-control construction strategies S0-S5.

Strategy order follows the user-defined standardized names:

S0_naive_mean_control_reference
S1_random_single_control
S2_random_average_controls
S3_SEACell_metacell_average
S4_SEACell_balanced_random_sample
S5_SEACell_OT_sampled_average
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from pseudo_pairing_seacell import build_metacell_anndata, load_membership_groups
from pseudo_pairing_utils import (
    align_control_and_perturbed,
    align_control_to_perturbed_genes,
    basic_pair_metadata,
    build_average_by_index_matrix,
    build_output_obs,
    build_random_single_matrix,
    cleanup,
    dataframe_file_exists,
    ensure_outdir,
    entropy_from_counts,
    format_array_as_string,
    get_existing_dataframe_path,
    get_expr_matrix,
    infer_perturbation_key,
    load_perturbed_metadata,
    namespace_to_dict,
    read_dataframe,
    read_json,
    row_mean,
    save_dataframe,
    save_json,
    set_seed,
    sparse_usage_summary,
    to_jsonable,
    write_pseudo_control_h5ad,
)


S0 = "S0_naive_mean_control_reference"
S1 = "S1_random_single_control"
S2 = "S2_random_average_controls"
S3 = "S3_SEACell_metacell_average"
S4 = "S4_SEACell_balanced_random_sample"
S5 = "S5_SEACell_OT_sampled_average"


# -----------------------------------------------------------------------------
# Generic loading helpers
# -----------------------------------------------------------------------------

def _prepare_control_and_perturbed_metadata(args: SimpleNamespace):
    pert_meta = load_perturbed_metadata(
        args.perturbed_h5ad,
        perturbation_key=getattr(args, "perturbation_key", None),
        max_pairs_per_perturbation=getattr(args, "max_pairs_per_perturbation", None),
        pair_selection_seed=getattr(args, "pair_selection_seed", 42),
    )
    control = sc.read_h5ad(str(args.control_h5ad))
    control.obs_names_make_unique()
    control, output_var_names, output_var = align_control_to_perturbed_genes(
        control,
        pert_meta["var_names"],
        perturbed_var=pert_meta["var"],
        require_all_genes=bool(getattr(args, "require_all_genes", False)),
    )
    if output_var is None:
        output_var = pert_meta["var"].copy()
    return control, pert_meta, output_var


def _save_qc(qc: dict[str, Any], outdir: str | Path) -> None:
    save_json(qc, Path(outdir) / "pairing_qc_summary.json")

def _should_overwrite_existing_outputs(args: SimpleNamespace) -> bool:
    return any(
        bool(getattr(args, flag, False))
        for flag in [
            "overwrite_existing_outputs",
            "overwrite_pseudo_controls",
            "overwrite_sampled_outputs",
        ]
    )
    
def detect_existing_pseudo_control_output(args: SimpleNamespace, outdir: str | Path) -> dict[str, Any] | None:
    """Return manifest-ready result if pseudo-control h5ad and pair metadata already exist."""
    outdir = Path(outdir)
    if _should_overwrite_existing_outputs(args):
        return None

    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
    pair_metadata_prefix = outdir / "pair_metadata"

    if not output_h5ad.exists():
        return None
    if not dataframe_file_exists(pair_metadata_prefix):
        return None

    pair_metadata_path = get_existing_dataframe_path(pair_metadata_prefix)
    qc_path = outdir / "pairing_qc_summary.json"
    qc = read_json(qc_path) if qc_path.exists() else {}

    if not qc:
        qc = {
            "output_h5ad": str(output_h5ad),
            "pair_metadata": str(pair_metadata_path),
        }

    return {
        "outdir": outdir,
        "output_h5ad": output_h5ad,
        "pair_metadata_path": pair_metadata_path,
        "qc": qc,
        "skipped_existing": True,
    }


def _attach_existing_metacell_usage_path(result: dict[str, Any], outdir: str | Path) -> dict[str, Any]:
    """Attach existing metacell usage table path for S3/S4 if present."""
    outdir = Path(outdir)
    usage_prefix = outdir / "metacell_usage_summary"
    if dataframe_file_exists(usage_prefix):
        result["metacell_usage_path"] = get_existing_dataframe_path(usage_prefix)
    return result


# -----------------------------------------------------------------------------
# Existing-output detection for cheap manifest-only reruns
# -----------------------------------------------------------------------------

def _overwrite_existing_outputs(args: SimpleNamespace) -> bool:
    """Return True when pseudo-control outputs should be regenerated.

    The older S5 code already used ``overwrite_sampled_outputs``.  For S0-S2 we
    also accept a more general ``overwrite_existing_outputs`` flag so the runner
    can reuse already generated h5ad files by default while still allowing a
    forced rebuild when needed.
    """
    return bool(
        getattr(args, "overwrite_existing_outputs", False)
        or getattr(args, "overwrite_sampled_outputs", False)
        or getattr(args, "overwrite_pseudo_controls", False)
    )


def _fill_existing_qc_from_disk(qc: dict[str, Any], output_h5ad: Path, pair_metadata_path: Path, strategy: str) -> dict[str, Any]:
    """Fill minimal QC fields for an already generated pseudo-control dataset."""
    qc = dict(qc or {})
    qc.setdefault("strategy", strategy)
    qc.setdefault("output_h5ad", str(output_h5ad))
    qc.setdefault("pair_metadata", str(pair_metadata_path))
    qc.setdefault("skipped_existing", True)

    if "n_pairs_written" not in qc:
        try:
            qc["n_pairs_written"] = int(read_dataframe(pair_metadata_path).shape[0])
        except Exception:
            pass

    if "n_genes_output" not in qc or "n_pairs_written" not in qc:
        try:
            ad = sc.read_h5ad(str(output_h5ad), backed="r")
            qc.setdefault("n_pairs_written", int(ad.n_obs))
            qc.setdefault("n_genes_output", int(ad.n_vars))
            try:
                uns = dict(ad.uns)
                if "n_perturbed_cells_total" in uns:
                    qc.setdefault("n_perturbed_cells_total", int(uns["n_perturbed_cells_total"]))
                if "n_control_cells_averaged" in uns:
                    qc.setdefault("n_control_cells", int(uns["n_control_cells_averaged"]))
            finally:
                try:
                    ad.file.close()
                except Exception:
                    pass
        except Exception:
            pass

    return qc


def detect_existing_pseudo_control_output(
    args: SimpleNamespace,
    outdir: str | Path,
    strategy: str,
) -> dict[str, Any] | None:
    """Return an existing pseudo-control result dict when output files are present.

    This prevents expensive repeated generation for S0/S1/S2.  The caller can
    still append the returned paths/QC into the repetition manifest exactly as if
    the dataset had just been generated.

    Required files:
        - ``pseudo_control_aligned_to_perturbed.h5ad``
        - ``pair_metadata.csv`` or ``pair_metadata.parquet``

    Set ``overwrite_existing_outputs=True`` to force regeneration.
    """
    outdir = ensure_outdir(outdir)
    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
    pair_metadata_prefix = outdir / "pair_metadata"

    if _overwrite_existing_outputs(args):
        return None
    if not output_h5ad.exists() or output_h5ad.stat().st_size == 0:
        return None
    if not dataframe_file_exists(pair_metadata_prefix):
        return None

    pair_metadata_path = get_existing_dataframe_path(pair_metadata_prefix)
    qc_path = outdir / "pairing_qc_summary.json"
    qc = read_json(qc_path) if qc_path.exists() else {}
    qc = _fill_existing_qc_from_disk(qc, output_h5ad, pair_metadata_path, strategy)
    if not qc_path.exists():
        _save_qc(qc, outdir)

    print(f"[{strategy}] Existing output found; skip generation and reuse: {output_h5ad}")
    return {
        "outdir": outdir,
        "output_h5ad": output_h5ad,
        "pair_metadata_path": pair_metadata_path,
        "qc": qc,
        "skipped_existing": True,
    }


# -----------------------------------------------------------------------------
# S0: naive mean control reference
# -----------------------------------------------------------------------------

def run_s0_naive_mean_control_reference(args: SimpleNamespace, outdir: str | Path) -> dict[str, Any]:
    """Every perturbed cell receives the same global mean control expression."""
    outdir = ensure_outdir(outdir)
    save_json(namespace_to_dict(args), outdir / "strategy_config.json")

    existing = detect_existing_pseudo_control_output(args, outdir, S0)
    if existing is not None:
        return existing

    print("[S0] Loading control and perturbed metadata.")
    control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args)
    X_control = get_expr_matrix(control, getattr(args, "expr_layer", "X"))
    mean_profile = row_mean(X_control, np.arange(control.n_obs, dtype=np.int64))
    np.save(outdir / "global_control_mean_profile.npy", mean_profile)

    n_pairs = int(pert_meta["n_obs_selected"])
    n_genes = int(len(output_var))
    batch_size = int(getattr(args, "batch_size", getattr(args, "matrix_batch_size", 4096)))

    # S0 is dense by construction. Use a memmap to avoid holding the full matrix in RAM.
    memmap_path = outdir / "pseudo_control_dense_memmap.float32.dat"
    X_mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n_pairs, n_genes))
    for start in tqdm(range(0, n_pairs, batch_size), desc="[S0] Repeating global control mean"):
        end = min(start + batch_size, n_pairs)
        X_mm[start:end, :] = mean_profile[None, :]
    X_mm.flush()

    pair_metadata = basic_pair_metadata(pert_meta, S0)
    pair_metadata["n_control_cells_averaged"] = int(control.n_obs)
    pair_metadata["mean_profile_path"] = str(outdir / "global_control_mean_profile.npy")

    # AnnData can write the disk-backed memmap without materializing the full matrix first.
    result = write_pseudo_control_h5ad(
        X_pseudo=X_mm,
        pert_meta=pert_meta,
        output_var=output_var,
        outdir=outdir,
        strategy=S0,
        pair_metadata=pair_metadata,
        extra_uns={
            "source_control_h5ad": str(args.control_h5ad),
            "source_perturbed_h5ad": str(args.perturbed_h5ad),
            "expr_layer": getattr(args, "expr_layer", "X"),
            "n_control_cells_averaged": int(control.n_obs),
            "mean_profile_path": str(outdir / "global_control_mean_profile.npy"),
        },
        extra_obs_cols=["pair_id", "pairing_strategy"],
    )

    qc = {
        "strategy": S0,
        "n_control_cells": int(control.n_obs),
        "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
        "n_pairs_written": int(n_pairs),
        "n_genes_output": int(n_genes),
        "row_alignment": "same_order_as_perturbed_h5ad" if n_pairs == pert_meta["n_obs_total"] else "subset_sorted_by_original_perturbed_cell_order",
        "output_h5ad": str(result["output_h5ad"]),
        "pair_metadata": str(result["pair_metadata_path"]),
    }
    _save_qc(qc, outdir)
    result["qc"] = qc
    print(f"[S0] Done: {result['output_h5ad']}")
    return result


# -----------------------------------------------------------------------------
# S1: random single control
# -----------------------------------------------------------------------------

def run_s1_random_single_control(args: SimpleNamespace, outdir: str | Path, seed: int) -> dict[str, Any]:
    rng = set_seed(seed)
    outdir = ensure_outdir(outdir)
    args_i = SimpleNamespace(**vars(args), seed=int(seed))
    save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

    existing = detect_existing_pseudo_control_output(args_i, outdir, S1)
    if existing is not None:
        return existing

    print(f"[S1] seed={seed} Loading data.")
    control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args_i)
    X_control = get_expr_matrix(control, getattr(args_i, "expr_layer", "X"))

    n_pairs = int(pert_meta["n_obs_selected"])
    selected_control_pos = rng.choice(np.arange(control.n_obs), size=n_pairs, replace=True).astype(np.int64)
    X_pseudo = build_random_single_matrix(
        X_control,
        selected_control_pos,
        batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
    )

    pair_metadata = basic_pair_metadata(pert_meta, S1)
    pair_metadata["selected_control_cell_id"] = control.obs_names[selected_control_pos].astype(str)
    pair_metadata["selected_control_cell_pos"] = selected_control_pos.astype(int)
    pair_metadata["sampling_seed"] = int(seed)

    result = write_pseudo_control_h5ad(
        X_pseudo,
        pert_meta,
        output_var,
        outdir,
        S1,
        pair_metadata,
        extra_uns={
            "source_control_h5ad": str(args_i.control_h5ad),
            "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
            "expr_layer": getattr(args_i, "expr_layer", "X"),
            "sampling_seed": int(seed),
        },
        extra_obs_cols=["pair_id", "selected_control_cell_id", "selected_control_cell_pos", "sampling_seed"],
    )
    counts = pd.Series(selected_control_pos).value_counts()
    qc = {
        "strategy": S1,
        "sampling_seed": int(seed),
        "n_control_cells": int(control.n_obs),
        "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
        "n_pairs_written": int(n_pairs),
        "n_genes_output": int(len(output_var)),
        "n_unique_control_cells_used": int(counts.shape[0]),
        "control_cell_usage_mean": float(counts.mean()),
        "control_cell_usage_median": float(counts.median()),
        "control_cell_usage_max": int(counts.max()),
        "output_h5ad": str(result["output_h5ad"]),
        "pair_metadata": str(result["pair_metadata_path"]),
    }
    _save_qc(qc, outdir)
    result["qc"] = qc
    print(f"[S1] Done: {result['output_h5ad']}")
    return result


# -----------------------------------------------------------------------------
# S2: random average controls
# -----------------------------------------------------------------------------

def run_s2_random_average_controls(
    args: SimpleNamespace,
    outdir: str | Path,
    seed: int,
    n_control_cells_to_average: int,
) -> dict[str, Any]:
    rng = set_seed(seed)
    outdir = ensure_outdir(outdir)
    args_i = SimpleNamespace(**vars(args), seed=int(seed), n_control_cells_to_average=int(n_control_cells_to_average))
    save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

    existing = detect_existing_pseudo_control_output(args_i, outdir, S2)
    if existing is not None:
        return existing

    print(f"[S2] seed={seed}, k={n_control_cells_to_average} Loading data.")
    control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args_i)
    X_control = get_expr_matrix(control, getattr(args_i, "expr_layer", "X"))

    k = int(n_control_cells_to_average)
    if k <= 0:
        raise ValueError("n_control_cells_to_average must be positive.")
    replace = bool(getattr(args_i, "sampling_replace", True))
    if not replace and k > control.n_obs:
        raise ValueError("Cannot sample without replacement because k > number of control cells.")

    n_pairs = int(pert_meta["n_obs_selected"])
    sampled = rng.choice(np.arange(control.n_obs), size=(n_pairs, k), replace=replace).astype(np.int64)
    X_pseudo = build_average_by_index_matrix(
        X_control,
        sampled,
        batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
    )

    pair_metadata = basic_pair_metadata(pert_meta, S2)
    if bool(getattr(args_i, "store_sampled_control_positions", True)):
        pair_metadata["sampled_control_cell_positions"] = [format_array_as_string(row) for row in sampled]
    pair_metadata["n_random_control_cells_averaged"] = int(k)
    pair_metadata["sampling_seed"] = int(seed)
    pair_metadata["sampling_replace"] = bool(replace)

    result = write_pseudo_control_h5ad(
        X_pseudo,
        pert_meta,
        output_var,
        outdir,
        S2,
        pair_metadata,
        extra_uns={
            "source_control_h5ad": str(args_i.control_h5ad),
            "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
            "expr_layer": getattr(args_i, "expr_layer", "X"),
            "sampling_seed": int(seed),
            "n_random_control_cells_averaged": int(k),
            "sampling_replace": bool(replace),
        },
        extra_obs_cols=["pair_id", "n_random_control_cells_averaged", "sampling_seed", "sampling_replace"],
    )
    counts = pd.Series(sampled.reshape(-1)).value_counts()
    qc = {
        "strategy": S2,
        "sampling_seed": int(seed),
        "n_control_cells": int(control.n_obs),
        "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
        "n_pairs_written": int(n_pairs),
        "n_genes_output": int(len(output_var)),
        "n_random_control_cells_averaged": int(k),
        "sampling_replace": bool(replace),
        "n_unique_control_cells_used": int(counts.shape[0]),
        "control_cell_usage_mean": float(counts.mean()),
        "control_cell_usage_median": float(counts.median()),
        "control_cell_usage_max": int(counts.max()),
        "output_h5ad": str(result["output_h5ad"]),
        "pair_metadata": str(result["pair_metadata_path"]),
    }
    _save_qc(qc, outdir)
    result["qc"] = qc
    print(f"[S2] Done: {result['output_h5ad']}")
    return result


# -----------------------------------------------------------------------------
# Metacell context shared by S3-S5
# -----------------------------------------------------------------------------

def load_seacell_context(args: SimpleNamespace, membership_path: str | Path) -> dict[str, Any]:
    print(f"[Metacell context] membership={membership_path}")
    control = sc.read_h5ad(str(args.control_h5ad))
    perturbed = sc.read_h5ad(str(args.perturbed_h5ad))
    control.obs_names_make_unique()
    perturbed.obs_names_make_unique()
    control, perturbed = align_control_and_perturbed(
        control,
        perturbed,
        require_all_genes=bool(getattr(args, "require_all_genes", False)),
    )
    args.perturbation_key = infer_perturbation_key(perturbed.obs, getattr(args, "perturbation_key", None))
    metacell_ids, groups, membership_df, membership_file = load_membership_groups(control, membership_path)
    X_control = get_expr_matrix(control, getattr(args, "expr_layer", "X"))
    control_metacells = build_metacell_anndata(
        control,
        X_control,
        metacell_ids,
        groups,
        embedding_key=getattr(args, "embedding_key", None) if getattr(args, "embedding_key", None) in control.obsm else None,
    )
    return {
        "control": control,
        "perturbed": perturbed,
        "X_control": X_control,
        "metacell_ids": metacell_ids,
        "groups": groups,
        "membership_df": membership_df,
        "membership_file": membership_file,
        "control_metacells": control_metacells,
        "perturbation_key": args.perturbation_key,
    }


def _perturbed_subset_positions(perturbed: AnnData, perturbation_key: str, args: SimpleNamespace) -> np.ndarray:
    max_per = getattr(args, "max_pairs_per_perturbation", None)
    if max_per is None:
        return np.arange(perturbed.n_obs, dtype=np.int64)
    rng = set_seed(getattr(args, "pair_selection_seed", 42))
    values = perturbed.obs[perturbation_key].astype(str).values
    selected = []
    for pert_name in pd.Index(values).unique().astype(str):
        idx = np.where(values == pert_name)[0]
        if len(idx) > int(max_per):
            idx = rng.choice(idx, size=int(max_per), replace=False)
        selected.append(np.sort(idx).astype(np.int64))
    return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=np.int64)


def _context_pert_meta(context: dict[str, Any], args: SimpleNamespace) -> dict[str, Any]:
    perturbed = context["perturbed"]
    key = context["perturbation_key"]
    selected_positions = _perturbed_subset_positions(perturbed, key, args)
    return {
        "obs_all": perturbed.obs.copy(),
        "obs": perturbed.obs.iloc[selected_positions].copy(),
        "var": perturbed.var.copy(),
        "obs_names_all": pd.Index(perturbed.obs_names.astype(str)),
        "obs_names": pd.Index(perturbed.obs_names[selected_positions].astype(str)),
        "var_names": pd.Index(perturbed.var_names.astype(str)),
        "n_obs_total": int(perturbed.n_obs),
        "n_obs_selected": int(len(selected_positions)),
        "n_vars": int(perturbed.n_vars),
        "selected_positions": selected_positions,
        "perturbation_key": key,
    }


# -----------------------------------------------------------------------------
# S3: SEACell metacell average
# -----------------------------------------------------------------------------

def run_s3_seacell_metacell_average(
    args: SimpleNamespace,
    context: dict[str, Any],
    outdir: str | Path,
    seed: int,
    n_metacells_to_average: int,
) -> dict[str, Any]:
    rng = set_seed(seed)
    outdir = ensure_outdir(outdir)
    args_i = SimpleNamespace(**vars(args), seed=int(seed), n_metacells_to_average=int(n_metacells_to_average))
    save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

    # ------------------------------------------------------------
    # Existing-output detection
    # ------------------------------------------------------------
    existing = detect_existing_pseudo_control_output(args_i, outdir, S3)
    if existing is not None:
        existing = _attach_existing_metacell_usage_path(existing, outdir)
        print(f"[S3] Existing output found, skipping generation: {existing['output_h5ad']}")
        return existing

    control_mc = context["control_metacells"]
    X_mc = get_expr_matrix(control_mc, "X")
    metacell_ids = context["metacell_ids"]
    pert_meta = _context_pert_meta(context, args_i)
    n_pairs = int(pert_meta["n_obs_selected"])

    k = int(n_metacells_to_average)
    if k <= 0:
        raise ValueError("n_metacells_to_average must be positive.")

    replace = bool(getattr(args_i, "sample_metacells_with_replacement", True))
    if not replace and k > len(metacell_ids):
        raise ValueError("Cannot sample metacells without replacement because k > number of metacells.")

    sampled_mc = rng.choice(
        np.arange(len(metacell_ids)),
        size=(n_pairs, k),
        replace=replace,
    ).astype(np.int64)

    X_pseudo = build_average_by_index_matrix(
        X_mc,
        sampled_mc,
        batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
    )

    pair_metadata = basic_pair_metadata(pert_meta, S3)
    pair_metadata["sampled_metacell_indices"] = [
        format_array_as_string(row) for row in sampled_mc
    ]
    pair_metadata["sampled_metacell_ids"] = [
        format_array_as_string([metacell_ids[int(x)] for x in row])
        for row in sampled_mc
    ]
    pair_metadata["n_metacells_averaged"] = int(k)
    pair_metadata["sampling_seed"] = int(seed)
    pair_metadata["sample_metacells_with_replacement"] = bool(replace)
    pair_metadata["membership_path"] = str(context["membership_file"])

    result = write_pseudo_control_h5ad(
        X_pseudo,
        pert_meta,
        context["perturbed"].var.copy(),
        outdir,
        S3,
        pair_metadata,
        extra_uns={
            "source_control_h5ad": str(args_i.control_h5ad),
            "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
            "membership_path": str(context["membership_file"]),
            "n_metacells_observed": int(len(metacell_ids)),
            "n_metacells_averaged": int(k),
            "sampling_seed": int(seed),
            "sample_metacells_with_replacement": bool(replace),
        },
        extra_obs_cols=[
            "pair_id",
            "n_metacells_averaged",
            "sampling_seed",
            "sampled_metacell_ids",
        ],
    )

    usage = sparse_usage_summary(sampled_mc, metacell_ids, "n_sampled")
    usage_path = save_dataframe(usage, outdir / "metacell_usage_summary")

    qc = {
        "strategy": S3,
        "sampling_seed": int(seed),
        "membership_path": str(context["membership_file"]),
        "n_control_cells": int(context["control"].n_obs),
        "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
        "n_pairs_written": int(n_pairs),
        "n_genes_output": int(context["perturbed"].n_vars),
        "n_metacells_observed": int(len(metacell_ids)),
        "n_metacells_averaged": int(k),
        "n_unique_metacells_used": int(usage.shape[0]),
        "metacell_usage_entropy": entropy_from_counts(usage["n_sampled"].values),
        "metacell_max_usage_fraction": float(usage["n_sampled"].max() / usage["n_sampled"].sum()),
        "metacell_usage_summary": str(usage_path),
        "output_h5ad": str(result["output_h5ad"]),
        "pair_metadata": str(result["pair_metadata_path"]),
    }

    _save_qc(qc, outdir)
    result["qc"] = qc
    result["metacell_usage_path"] = usage_path

    print(f"[S3] Done: {result['output_h5ad']}")
    return result


# -----------------------------------------------------------------------------
# S4: SEACell balanced random sample
# -----------------------------------------------------------------------------

def balanced_metacell_sequence(n_pairs: int, n_metacells: int, rng: np.random.Generator) -> np.ndarray:
    if n_pairs <= 0:
        return np.array([], dtype=np.int32)
    base = np.tile(np.arange(n_metacells, dtype=np.int32), int(np.ceil(n_pairs / n_metacells)))[:n_pairs]
    rng.shuffle(base)
    return base.astype(np.int32)


def sample_one_control_per_assigned_metacell(
    groups: Sequence[np.ndarray],
    assigned_mc: np.ndarray,
    replace: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.empty(len(assigned_mc), dtype=np.int64)
    assigned_mc = np.asarray(assigned_mc, dtype=np.int64)
    for mc_idx in np.unique(assigned_mc):
        row_idx = np.where(assigned_mc == mc_idx)[0]
        members = groups[int(mc_idx)]
        if len(members) == 0:
            raise ValueError(f"Metacell index {mc_idx} has no member cells.")
        use_replace = bool(replace) or len(row_idx) > len(members)
        selected[row_idx] = rng.choice(members, size=len(row_idx), replace=use_replace)
    return selected


def run_s4_seacell_balanced_random_sample(
    args: SimpleNamespace,
    context: dict[str, Any],
    outdir: str | Path,
    seed: int,
) -> dict[str, Any]:
    rng = set_seed(seed)
    outdir = ensure_outdir(outdir)
    args_i = SimpleNamespace(**vars(args), seed=int(seed))
    save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

    # ------------------------------------------------------------
    # Existing-output detection
    # ------------------------------------------------------------
    existing = detect_existing_pseudo_control_output(args_i, outdir, S4)
    if existing is not None:
        existing = _attach_existing_metacell_usage_path(existing, outdir)
        print(f"[S4] Existing output found, skipping generation: {existing['output_h5ad']}")
        return existing

    perturbed = context["perturbed"]
    key = context["perturbation_key"]
    pert_meta = _context_pert_meta(context, args_i)
    selected_positions = pert_meta["selected_positions"]
    values = perturbed.obs[key].astype(str).values
    n_metacells = len(context["metacell_ids"])
    assigned_mc = np.empty(len(selected_positions), dtype=np.int64)

    # Balance metacell use independently within each perturbation identity.
    pos_to_row = {int(pos): i for i, pos in enumerate(selected_positions)}
    for pert_name in pd.Index(values[selected_positions]).unique().astype(str):
        pert_pos = selected_positions[values[selected_positions] == pert_name]
        local_rows = np.array([pos_to_row[int(p)] for p in pert_pos], dtype=np.int64)
        assigned_mc[local_rows] = balanced_metacell_sequence(len(local_rows), n_metacells, rng)

    selected_control_pos = sample_one_control_per_assigned_metacell(
        context["groups"],
        assigned_mc,
        replace=bool(getattr(args_i, "sampling_replace", False)),
        rng=rng,
    )

    X_pseudo = build_random_single_matrix(
        context["X_control"],
        selected_control_pos,
        batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
    )

    metacell_ids = context["metacell_ids"]
    pair_metadata = basic_pair_metadata(pert_meta, S4)
    pair_metadata["assigned_metacell_index"] = assigned_mc.astype(int)
    pair_metadata["assigned_metacell_id"] = [metacell_ids[int(i)] for i in assigned_mc]
    pair_metadata["selected_control_cell_id"] = context["control"].obs_names[selected_control_pos].astype(str)
    pair_metadata["selected_control_cell_pos"] = selected_control_pos.astype(int)
    pair_metadata["sampling_seed"] = int(seed)
    pair_metadata["membership_path"] = str(context["membership_file"])

    result = write_pseudo_control_h5ad(
        X_pseudo,
        pert_meta,
        context["perturbed"].var.copy(),
        outdir,
        S4,
        pair_metadata,
        extra_uns={
            "source_control_h5ad": str(args_i.control_h5ad),
            "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
            "membership_path": str(context["membership_file"]),
            "n_metacells_observed": int(n_metacells),
            "sampling_seed": int(seed),
        },
        extra_obs_cols=[
            "pair_id",
            "assigned_metacell_id",
            "assigned_metacell_index",
            "selected_control_cell_id",
            "selected_control_cell_pos",
            "sampling_seed",
        ],
    )

    selected_counts = pd.Series(selected_control_pos).value_counts()
    metacell_counts = pd.Series(assigned_mc).value_counts()

    usage = pd.DataFrame(
        {
            "metacell_index": metacell_counts.index.astype(int),
            "metacell_id": [metacell_ids[int(i)] for i in metacell_counts.index.astype(int)],
            "n_assigned_pairs": metacell_counts.values.astype(int),
        }
    ).sort_values("metacell_index")

    usage_path = save_dataframe(usage, outdir / "metacell_usage_summary")

    qc = {
        "strategy": S4,
        "sampling_seed": int(seed),
        "membership_path": str(context["membership_file"]),
        "n_control_cells": int(context["control"].n_obs),
        "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
        "n_pairs_written": int(len(selected_positions)),
        "n_genes_output": int(context["perturbed"].n_vars),
        "n_metacells_observed": int(n_metacells),
        "n_unique_control_cells_used": int(selected_counts.shape[0]),
        "control_cell_usage_mean": float(selected_counts.mean()),
        "control_cell_usage_median": float(selected_counts.median()),
        "control_cell_usage_max": int(selected_counts.max()),
        "metacell_usage_entropy_total": entropy_from_counts(metacell_counts.values),
        "metacell_usage_summary": str(usage_path),
        "output_h5ad": str(result["output_h5ad"]),
        "pair_metadata": str(result["pair_metadata_path"]),
    }

    _save_qc(qc, outdir)
    result["qc"] = qc
    result["metacell_usage_path"] = usage_path

    print(f"[S4] Done: {result['output_h5ad']}")
    return result


# -----------------------------------------------------------------------------
# S5: SEACell OT sampled average
# -----------------------------------------------------------------------------

def compute_control_mass(mc_sizes: np.ndarray, mode: str) -> np.ndarray:
    if mode == "size":
        a = mc_sizes.astype(np.float64)
        return a / a.sum()
    if mode == "uniform":
        a = np.ones(len(mc_sizes), dtype=np.float64)
        return a / a.sum()
    raise ValueError("control_mass must be 'size' or 'uniform'.")


def sinkhorn_ot(a: np.ndarray, b: np.ndarray, cost: np.ndarray, reg: float, max_iter: int, tol: float) -> np.ndarray:
    try:
        import ot
    except ImportError as exc:
        raise ImportError("POT is required for OT matching. Install with: pip install POT") from exc
    cost = cost.astype(np.float64, copy=False)
    try:
        G = ot.sinkhorn(a, b, cost, reg=float(reg), numItermax=int(max_iter), stopThr=float(tol), warn=False)
    except Exception:
        G = ot.bregman.sinkhorn_stabilized(
            a, b, cost, reg=float(reg), numItermax=int(max_iter), stopThr=float(tol), warn=False
        )
    if not np.all(np.isfinite(G)):
        raise FloatingPointError("OT transport matrix contains non-finite values.")
    return G


def topk_from_transport(G: np.ndarray, k: int):
    eps = 1e-12
    n_source, _ = G.shape
    k = min(int(k), n_source)
    W = G / np.maximum(G.sum(axis=0, keepdims=True), eps)
    entropy = -np.sum(W * np.log(W + eps), axis=0) / np.log(n_source)
    idx_unsorted = np.argpartition(W, kth=n_source - k, axis=0)[-k:, :]
    w_unsorted = np.take_along_axis(W, idx_unsorted, axis=0)
    order = np.argsort(-w_unsorted, axis=0)
    top_idx = np.take_along_axis(idx_unsorted, order, axis=0).T
    top_w = np.take_along_axis(w_unsorted, order, axis=0).T
    top_w = top_w / np.maximum(top_w.sum(axis=1, keepdims=True), eps)
    return top_idx.astype(np.int32), top_w.astype(np.float32), entropy.astype(np.float32)


def compute_ot_assignments_for_setting_topk(
    args: SimpleNamespace,
    context: dict[str, Any],
    top_k: int,
    assignment_outdir: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    assignment_outdir = ensure_outdir(assignment_outdir)
    assignment_prefix = assignment_outdir / f"ot_assignments_topk_{int(top_k):02d}"
    qc_path = assignment_outdir / f"ot_assignment_qc_topk_{int(top_k):02d}.json"
    if dataframe_file_exists(assignment_prefix) and qc_path.exists() and not bool(getattr(args, "overwrite_assignments", False)):
        print(f"[S5/OT] Reusing cached assignments: {get_existing_dataframe_path(assignment_prefix)}")
        return read_dataframe(assignment_prefix), read_json(qc_path), get_existing_dataframe_path(assignment_prefix)

    if getattr(args, "embedding_key", None) not in context["control_metacells"].obsm:
        raise KeyError(
            f"S5 requires metacell and perturbed embeddings under obsm['{args.embedding_key}']. "
            "Run preprocessing PCA first and ensure control/perturbed h5ad both contain this key."
        )
    if args.embedding_key not in context["perturbed"].obsm:
        raise KeyError(f"Perturbed AnnData does not contain obsm['{args.embedding_key}'].")

    rng_select = set_seed(getattr(args, "pair_selection_seed", 42))
    perturbed = context["perturbed"]
    key = context["perturbation_key"]
    values = perturbed.obs[key].astype(str).values
    unique_perts = pd.Index(values).unique().astype(str).tolist()
    metacell_ids = context["metacell_ids"]
    mc_sizes = context["control_metacells"].obs["n_control_cells"].values.astype(np.float64)
    a = compute_control_mass(mc_sizes, getattr(args, "control_mass", "size"))
    Z_mc = np.asarray(context["control_metacells"].obsm[args.embedding_key], dtype=np.float32)
    Z_pert = np.asarray(perturbed.obsm[args.embedding_key], dtype=np.float32)

    records = []
    usage_records = []
    total_pairs = 0
    print(f"[S5/OT] Computing assignments for top_k={top_k}; perturbations={len(unique_perts):,}")
    for pert_name in tqdm(unique_perts, desc="[S5/OT] Perturbation-wise OT"):
        pert_idx = np.where(values == pert_name)[0]
        max_per = getattr(args, "max_pairs_per_perturbation", None)
        if max_per is not None and len(pert_idx) > int(max_per):
            pert_idx = np.sort(rng_select.choice(pert_idx, size=int(max_per), replace=False))
        if len(pert_idx) == 0:
            continue
        cost_raw = pairwise_distances(Z_mc, Z_pert[pert_idx], metric=getattr(args, "cost_metric", "sqeuclidean")).astype(np.float64)
        positive = cost_raw[cost_raw > 0]
        scale = np.median(positive) if positive.size > 0 else 1.0
        cost = cost_raw / max(scale, 1e-12)
        b = np.ones(len(pert_idx), dtype=np.float64)
        b /= b.sum()
        G = sinkhorn_ot(
            a=a,
            b=b,
            cost=cost,
            reg=getattr(args, "ot_reg", 0.05),
            max_iter=getattr(args, "ot_max_iter", 2000),
            tol=getattr(args, "ot_tol", 1e-7),
        )
        top_idx, top_w, entropy = topk_from_transport(G, top_k)
        dominant_idx = top_idx[:, 0]
        dominant_weight = top_w[:, 0]
        dominant_distance = cost_raw[dominant_idx, np.arange(len(pert_idx))]
        usage = pd.DataFrame(
            {"perturbation": pert_name, "metacell_id": [metacell_ids[int(i)] for i in dominant_idx]}
        )
        usage = usage.groupby(["perturbation", "metacell_id"]).size().reset_index(name="n_dominant_assignments")
        usage_records.append(usage)
        batch = []
        for local_i, ppos in enumerate(pert_idx):
            rec = {
                "pair_id": f"pair_{total_pairs + local_i:012d}",
                "perturbed_cell_id": str(perturbed.obs_names[ppos]),
                "perturbed_cell_pos": int(ppos),
                "perturbation": str(pert_name),
                "dominant_metacell_id": str(metacell_ids[int(dominant_idx[local_i])]),
                "dominant_metacell_index": int(dominant_idx[local_i]),
                "dominant_weight": float(dominant_weight[local_i]),
                "dominant_distance": float(dominant_distance[local_i]),
                "matching_entropy": float(entropy[local_i]),
                "top_metacell_ids": format_array_as_string([metacell_ids[int(x)] for x in top_idx[local_i]]),
                "top_metacell_indices": format_array_as_string(top_idx[local_i]),
                "top_metacell_weights": format_array_as_string(top_w[local_i]),
                "pairing_strategy": S5,
                "top_k_metacells": int(top_k),
                "ot_reg": float(getattr(args, "ot_reg", 0.05)),
                "control_mass": getattr(args, "control_mass", "size"),
                "pair_selection_seed": int(getattr(args, "pair_selection_seed", 42)),
                "membership_path": str(context["membership_file"]),
            }
            for j in range(int(top_k)):
                rec[f"top_metacell_index_{j}"] = int(top_idx[local_i, j])
                rec[f"top_metacell_weight_{j}"] = float(top_w[local_i, j])
                rec[f"top_metacell_id_{j}"] = str(metacell_ids[int(top_idx[local_i, j])])
            batch.append(rec)
        records.append(pd.DataFrame(batch))
        total_pairs += len(batch)

    if not records:
        raise RuntimeError("No OT assignments were generated.")
    assignment_df = pd.concat(records, axis=0, ignore_index=True)
    assignment_df = assignment_df.sort_values("perturbed_cell_pos").reset_index(drop=True)
    # Regenerate pair IDs after sorting by perturbed position.
    assignment_df["pair_id"] = [f"pair_{i:012d}" for i in range(assignment_df.shape[0])]
    assignment_path = save_dataframe(assignment_df, assignment_prefix)
    if usage_records:
        usage_df = pd.concat(usage_records, axis=0, ignore_index=True)
        usage_path = save_dataframe(usage_df, assignment_outdir / f"metacell_usage_by_perturbation_topk_{int(top_k):02d}")
        usage_summary = usage_df.groupby("metacell_id")["n_dominant_assignments"].sum().reset_index()
        usage_summary_path = save_dataframe(usage_summary, assignment_outdir / f"metacell_usage_summary_topk_{int(top_k):02d}")
    else:
        usage_path = None
        usage_summary_path = None
    qc = {
        "strategy": S5,
        "membership_path": str(context["membership_file"]),
        "n_control_cells": int(context["control"].n_obs),
        "n_perturbed_cells_available": int(context["perturbed"].n_obs),
        "n_pairs_assigned": int(assignment_df.shape[0]),
        "n_metacells_observed": int(len(metacell_ids)),
        "top_k_metacells": int(top_k),
        "sample_cells_per_metacell": int(getattr(args, "sample_cells_per_metacell", 10)),
        "ot_reg": float(getattr(args, "ot_reg", 0.05)),
        "ot_max_iter": int(getattr(args, "ot_max_iter", 2000)),
        "ot_tol": float(getattr(args, "ot_tol", 1e-7)),
        "cost_metric": getattr(args, "cost_metric", "sqeuclidean"),
        "control_mass": getattr(args, "control_mass", "size"),
        "max_pairs_per_perturbation": None if getattr(args, "max_pairs_per_perturbation", None) is None else int(args.max_pairs_per_perturbation),
        "pair_selection_seed": int(getattr(args, "pair_selection_seed", 42)),
        "mean_dominant_weight": float(assignment_df["dominant_weight"].mean()),
        "median_dominant_weight": float(assignment_df["dominant_weight"].median()),
        "mean_matching_entropy": float(assignment_df["matching_entropy"].mean()),
        "median_matching_entropy": float(assignment_df["matching_entropy"].median()),
        "mean_dominant_distance": float(assignment_df["dominant_distance"].mean()),
        "median_dominant_distance": float(assignment_df["dominant_distance"].median()),
        "assignment_path": str(assignment_path),
        "usage_path": None if usage_path is None else str(usage_path),
        "usage_summary_path": None if usage_summary_path is None else str(usage_summary_path),
    }
    save_json(qc, qc_path)
    return assignment_df, qc, assignment_path


def sample_pseudo_controls_batch_from_ot(
    X_control,
    groups: Sequence[np.ndarray],
    top_idx_batch: np.ndarray,
    top_w_batch: np.ndarray,
    sample_cells_per_metacell: int,
    replace: bool,
    rng: np.random.Generator,
) -> sp.csr_matrix:
    n_rows = top_idx_batch.shape[0]
    n_vars = X_control.shape[1]
    out = np.zeros((n_rows, n_vars), dtype=np.float32)
    for i in range(n_rows):
        for mc_idx, weight in zip(top_idx_batch[i], top_w_batch[i]):
            members = groups[int(mc_idx)]
            if len(members) == 0:
                continue
            if replace:
                sampled = rng.choice(members, size=int(sample_cells_per_metacell), replace=True)
            else:
                n_sample = min(int(sample_cells_per_metacell), len(members))
                sampled = rng.choice(members, size=n_sample, replace=False)
            out[i] += float(weight) * row_mean(X_control, sampled)
    return sp.csr_matrix(out)


def build_adata_from_shards(
    perturbed: AnnData,
    pair_metadata: pd.DataFrame,
    shard_dir: str | Path,
    outdir: str | Path,
    strategy: str,
    extra_uns: dict[str, Any] | None = None,
) -> Path:
    outdir = ensure_outdir(outdir)
    shard_dir = Path(shard_dir)
    pair_metadata = pair_metadata.copy()
    pair_metadata["perturbed_cell_pos"] = pair_metadata["perturbed_cell_pos"].astype(int)
    pair_metadata["pseudo_control_row_in_shard"] = pair_metadata["pseudo_control_row_in_shard"].astype(int)
    covered_positions = np.sort(pair_metadata["perturbed_cell_pos"].unique())
    full = len(covered_positions) == perturbed.n_obs and np.array_equal(covered_positions, np.arange(perturbed.n_obs))
    if full:
        output_shape = (perturbed.n_obs, perturbed.n_vars)
        obs = perturbed.obs.copy()
        obs_names = pd.Index(perturbed.obs_names.astype(str))
        pos_to_row = {int(pos): int(pos) for pos in covered_positions}
    else:
        output_shape = (len(covered_positions), perturbed.n_vars)
        obs = perturbed.obs.iloc[covered_positions].copy()
        obs_names = pd.Index(perturbed.obs_names[covered_positions].astype(str))
        pos_to_row = {int(pos): i for i, pos in enumerate(covered_positions)}
    pair_metadata["_target_row"] = pair_metadata["perturbed_cell_pos"].map(pos_to_row).astype(int)

    data_parts = []
    row_parts = []
    col_parts = []
    for shard_name, df_shard in tqdm(list(pair_metadata.groupby("pseudo_control_shard", sort=False)), desc="Assembling shards"):
        shard_path = shard_dir / str(shard_name)
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard file: {shard_path}")
        X_shard = sp.load_npz(shard_path).tocsr()
        X_sel = X_shard[df_shard["pseudo_control_row_in_shard"].to_numpy()].tocoo()
        target_rows = df_shard["_target_row"].to_numpy()
        data_parts.append(X_sel.data)
        row_parts.append(target_rows[X_sel.row])
        col_parts.append(X_sel.col)
    X = sp.csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=output_shape,
        dtype=np.float32,
    )
    obs = obs.copy()
    obs.index = obs_names
    obs["pseudo_control_available"] = True
    obs["paired_perturbed_cell_id"] = obs_names.astype(str).values
    obs["paired_perturbed_cell_pos"] = covered_positions.astype(int) if not full else np.arange(perturbed.n_obs)
    obs["pairing_strategy"] = strategy
    metadata_aligned = pair_metadata.set_index("perturbed_cell_pos").loc[covered_positions].reset_index()
    obs_cols = [
        "pair_id",
        "perturbation",
        "dominant_metacell_id",
        "dominant_metacell_index",
        "dominant_weight",
        "dominant_distance",
        "matching_entropy",
        "top_metacell_ids",
        "top_metacell_indices",
        "top_metacell_weights",
        "top_k_metacells",
        "sampling_seed",
        "sample_cells_per_metacell",
        "sampling_replace",
        "ot_reg",
        "control_mass",
        "pair_selection_seed",
    ]
    for col in obs_cols:
        if col in metadata_aligned.columns:
            obs[f"pairing_{col}"] = metadata_aligned[col].values
    adata = AnnData(X=X, obs=obs, var=perturbed.var.copy())
    adata.uns["pairing_strategy"] = strategy
    adata.uns["row_alignment"] = "same_order_as_perturbed_h5ad" if full else "subset_sorted_by_original_perturbed_cell_order"
    if extra_uns:
        for k, v in extra_uns.items():
            adata.uns[k] = to_jsonable(v)
    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
    adata.write_h5ad(output_h5ad)
    return output_h5ad


def run_s5_seacell_ot_sampled_average(
    args: SimpleNamespace,
    context: dict[str, Any],
    assignment_df: pd.DataFrame,
    outdir: str | Path,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    rng = set_seed(seed)
    outdir = ensure_outdir(outdir)
    ensure_outdir(outdir / "pseudo_control_shards")
    args_i = SimpleNamespace(**vars(args), seed=int(seed), top_k_metacells=int(top_k))
    save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

    pair_metadata_prefix = outdir / "pair_metadata"
    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
    if output_h5ad.exists() and dataframe_file_exists(pair_metadata_prefix) and not bool(getattr(args_i, "overwrite_sampled_outputs", False)):
        print(f"[S5] Existing output found, skipping: {output_h5ad}")
        qc_path = outdir / "pairing_qc_summary.json"
        qc = read_json(qc_path) if qc_path.exists() else {}
        return {
            "outdir": outdir,
            "output_h5ad": output_h5ad,
            "pair_metadata_path": get_existing_dataframe_path(pair_metadata_prefix),
            "qc": qc,
            "skipped_existing": True,
        }

    idx_cols = [f"top_metacell_index_{j}" for j in range(int(top_k))]
    weight_cols = [f"top_metacell_weight_{j}" for j in range(int(top_k))]
    missing = [c for c in idx_cols + weight_cols if c not in assignment_df.columns]
    if missing:
        raise KeyError(f"Assignment table is missing columns: {missing}")

    all_pair_records = []
    shard_id = 0
    batch_size = int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 1024)))
    for start in tqdm(range(0, assignment_df.shape[0], batch_size), desc="[S5] Sampling pseudo-control shards"):
        end = min(start + batch_size, assignment_df.shape[0])
        df_batch = assignment_df.iloc[start:end].copy()
        top_idx_batch = df_batch[idx_cols].to_numpy(dtype=np.int32)
        top_w_batch = df_batch[weight_cols].to_numpy(dtype=np.float32)
        X_pseudo = sample_pseudo_controls_batch_from_ot(
            X_control=context["X_control"],
            groups=context["groups"],
            top_idx_batch=top_idx_batch,
            top_w_batch=top_w_batch,
            sample_cells_per_metacell=int(getattr(args_i, "sample_cells_per_metacell", 10)),
            replace=bool(getattr(args_i, "sampling_replace", False)),
            rng=rng,
        )
        shard_name = f"pseudo_control_shard_{shard_id:06d}.npz"
        sp.save_npz(outdir / "pseudo_control_shards" / shard_name, X_pseudo)
        df_batch["pseudo_control_shard"] = shard_name
        df_batch["pseudo_control_row_in_shard"] = np.arange(df_batch.shape[0], dtype=np.int32)
        df_batch["sampling_seed"] = int(seed)
        df_batch["sample_cells_per_metacell"] = int(getattr(args_i, "sample_cells_per_metacell", 10))
        df_batch["sampling_replace"] = bool(getattr(args_i, "sampling_replace", False))
        df_batch["pairing_strategy"] = S5
        all_pair_records.append(df_batch)
        shard_id += 1

    pair_metadata = pd.concat(all_pair_records, axis=0, ignore_index=True)
    pair_metadata_path = save_dataframe(pair_metadata, pair_metadata_prefix)
    final_h5ad = build_adata_from_shards(
        context["perturbed"],
        pair_metadata,
        outdir / "pseudo_control_shards",
        outdir,
        S5,
        extra_uns={
            "source_control_h5ad": str(args_i.control_h5ad),
            "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
            "membership_path": str(context["membership_file"]),
            "top_k_metacells": int(top_k),
            "sampling_seed": int(seed),
            "sample_cells_per_metacell": int(getattr(args_i, "sample_cells_per_metacell", 10)),
            "ot_reg": float(getattr(args_i, "ot_reg", 0.05)),
            "control_mass": getattr(args_i, "control_mass", "size"),
        },
    )
    qc = {
        "strategy": S5,
        "sampling_seed": int(seed),
        "membership_path": str(context["membership_file"]),
        "n_control_cells": int(context["control"].n_obs),
        "n_perturbed_cells_available": int(context["perturbed"].n_obs),
        "n_pairs_written": int(pair_metadata.shape[0]),
        "n_genes_output": int(context["perturbed"].n_vars),
        "n_metacells_observed": int(len(context["metacell_ids"])),
        "top_k_metacells": int(top_k),
        "sample_cells_per_metacell": int(getattr(args_i, "sample_cells_per_metacell", 10)),
        "sampling_replace": bool(getattr(args_i, "sampling_replace", False)),
        "pair_selection_seed": int(getattr(args_i, "pair_selection_seed", 42)),
        "ot_reg": float(getattr(args_i, "ot_reg", 0.05)),
        "control_mass": getattr(args_i, "control_mass", "size"),
        "mean_dominant_weight": float(pair_metadata["dominant_weight"].mean()),
        "median_dominant_weight": float(pair_metadata["dominant_weight"].median()),
        "mean_matching_entropy": float(pair_metadata["matching_entropy"].mean()),
        "median_matching_entropy": float(pair_metadata["matching_entropy"].median()),
        "mean_dominant_distance": float(pair_metadata["dominant_distance"].mean()),
        "median_dominant_distance": float(pair_metadata["dominant_distance"].median()),
        "n_shards": int(shard_id),
        "output_h5ad": str(final_h5ad),
        "pair_metadata": str(pair_metadata_path),
    }
    _save_qc(qc, outdir)
    print(f"[S5] Done: {final_h5ad}")
    return {
        "outdir": outdir,
        "output_h5ad": final_h5ad,
        "pair_metadata_path": pair_metadata_path,
        "qc": qc,
        "skipped_existing": False,
    }


# """Pseudo-control construction strategies S0-S5.

# Strategy order follows the user-defined standardized names:

# S0_naive_mean_control_reference
# S1_random_single_control
# S2_random_average_controls
# S3_SEACell_metacell_average
# S4_SEACell_balanced_random_sample
# S5_SEACell_OT_sampled_average
# """
# from __future__ import annotations

# import gc
# import json
# from pathlib import Path
# from types import SimpleNamespace
# from typing import Any, Sequence

# import numpy as np
# import pandas as pd
# import scanpy as sc
# import scipy.sparse as sp
# from anndata import AnnData
# from sklearn.metrics import pairwise_distances
# from tqdm.auto import tqdm

# from pseudo_pairing_seacell import build_metacell_anndata, load_membership_groups
# from pseudo_pairing_utils import (
#     align_control_and_perturbed,
#     align_control_to_perturbed_genes,
#     basic_pair_metadata,
#     build_average_by_index_matrix,
#     build_output_obs,
#     build_random_single_matrix,
#     cleanup,
#     dataframe_file_exists,
#     ensure_outdir,
#     entropy_from_counts,
#     format_array_as_string,
#     get_existing_dataframe_path,
#     get_expr_matrix,
#     infer_perturbation_key,
#     load_perturbed_metadata,
#     namespace_to_dict,
#     read_dataframe,
#     read_json,
#     row_mean,
#     save_dataframe,
#     save_json,
#     set_seed,
#     sparse_usage_summary,
#     to_jsonable,
#     write_pseudo_control_h5ad,
#     write_repeated_mean_profile_h5ad,
# )


# S0 = "S0_naive_mean_control_reference"
# S1 = "S1_random_single_control"
# S2 = "S2_random_average_controls"
# S3 = "S3_SEACell_metacell_average"
# S4 = "S4_SEACell_balanced_random_sample"
# S5 = "S5_SEACell_OT_sampled_average"


# # -----------------------------------------------------------------------------
# # Generic loading helpers
# # -----------------------------------------------------------------------------

# def _prepare_control_and_perturbed_metadata(args: SimpleNamespace):
#     pert_meta = load_perturbed_metadata(
#         args.perturbed_h5ad,
#         perturbation_key=getattr(args, "perturbation_key", None),
#         max_pairs_per_perturbation=getattr(args, "max_pairs_per_perturbation", None),
#         pair_selection_seed=getattr(args, "pair_selection_seed", 42),
#     )
#     control = sc.read_h5ad(str(args.control_h5ad))
#     control.obs_names_make_unique()
#     control, output_var_names, output_var = align_control_to_perturbed_genes(
#         control,
#         pert_meta["var_names"],
#         perturbed_var=pert_meta["var"],
#         require_all_genes=bool(getattr(args, "require_all_genes", False)),
#     )
#     if output_var is None:
#         output_var = pert_meta["var"].copy()
#     return control, pert_meta, output_var


# def _save_qc(qc: dict[str, Any], outdir: str | Path) -> None:
#     save_json(qc, Path(outdir) / "pairing_qc_summary.json")


# # -----------------------------------------------------------------------------
# # S0: naive mean control reference
# # -----------------------------------------------------------------------------

# def run_s0_naive_mean_control_reference(args: SimpleNamespace, outdir: str | Path) -> dict[str, Any]:
#     """Every perturbed cell receives the same global mean control expression."""
#     outdir = ensure_outdir(outdir)
#     save_json(namespace_to_dict(args), outdir / "strategy_config.json")

#     print("[S0] Loading control and perturbed metadata.")
#     control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args)
#     X_control = get_expr_matrix(control, getattr(args, "expr_layer", "X"))
#     mean_profile = row_mean(X_control, np.arange(control.n_obs, dtype=np.int64))
#     np.save(outdir / "global_control_mean_profile.npy", mean_profile)

#     n_pairs = int(pert_meta["n_obs_selected"])
#     n_genes = int(len(output_var))
#     batch_size = int(getattr(args, "batch_size", getattr(args, "matrix_batch_size", 4096)))

#     # S0 is dense by construction. Use a memmap to avoid holding the full matrix in RAM.
#     memmap_path = outdir / "pseudo_control_dense_memmap.float32.dat"
#     X_mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(n_pairs, n_genes))
#     for start in tqdm(range(0, n_pairs, batch_size), desc="[S0] Repeating global control mean"):
#         end = min(start + batch_size, n_pairs)
#         X_mm[start:end, :] = mean_profile[None, :]
#     X_mm.flush()

#     pair_metadata = basic_pair_metadata(pert_meta, S0)
#     pair_metadata["n_control_cells_averaged"] = int(control.n_obs)
#     pair_metadata["mean_profile_path"] = str(outdir / "global_control_mean_profile.npy")

#     # AnnData can write the disk-backed memmap without materializing the full matrix first.
#     # result = write_pseudo_control_h5ad(
#     #     X_pseudo=X_mm,
#     #     pert_meta=pert_meta,
#     #     output_var=output_var,
#     #     outdir=outdir,
#     #     strategy=S0,
#     #     pair_metadata=pair_metadata,
#     #     extra_uns={
#     #         "source_control_h5ad": str(args.control_h5ad),
#     #         "source_perturbed_h5ad": str(args.perturbed_h5ad),
#     #         "expr_layer": getattr(args, "expr_layer", "X"),
#     #         "n_control_cells_averaged": int(control.n_obs),
#     #         "mean_profile_path": str(outdir / "global_control_mean_profile.npy"),
#     #     },
#     #     extra_obs_cols=["pair_id", "pairing_strategy"],
#     # )
#     result = write_repeated_mean_profile_h5ad(
#         mean_profile=mean_profile,
#         pert_meta=pert_meta,
#         output_var=output_var,
#         outdir=outdir,
#         strategy=S0,
#         pair_metadata=pair_metadata,
#         extra_uns={
#             "source_control_h5ad": str(args.control_h5ad),
#             "source_perturbed_h5ad": str(args.perturbed_h5ad),
#             "expr_layer": getattr(args, "expr_layer", "X"),
#             "n_control_cells_averaged": int(control.n_obs),
#             "mean_profile_path": str(outdir / "global_control_mean_profile.npy"),
#         },
#         extra_obs_cols=["pair_id", "pairing_strategy"],
#         chunk_rows=512,
#         chunk_cols=2048,
#         compression="gzip",
#         compression_opts=4,
#     )

#     qc = {
#         "strategy": S0,
#         "n_control_cells": int(control.n_obs),
#         "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
#         "n_pairs_written": int(n_pairs),
#         "n_genes_output": int(n_genes),
#         "row_alignment": "same_order_as_perturbed_h5ad" if n_pairs == pert_meta["n_obs_total"] else "subset_sorted_by_original_perturbed_cell_order",
#         "output_h5ad": str(result["output_h5ad"]),
#         "pair_metadata": str(result["pair_metadata_path"]),
#     }
#     _save_qc(qc, outdir)
#     result["qc"] = qc
#     print(f"[S0] Done: {result['output_h5ad']}")
#     return result


# # -----------------------------------------------------------------------------
# # S1: random single control
# # -----------------------------------------------------------------------------

# def run_s1_random_single_control(args: SimpleNamespace, outdir: str | Path, seed: int) -> dict[str, Any]:
#     rng = set_seed(seed)
#     outdir = ensure_outdir(outdir)
#     args_i = SimpleNamespace(**vars(args), seed=int(seed))
#     save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

#     print(f"[S1] seed={seed} Loading data.")
#     control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args_i)
#     X_control = get_expr_matrix(control, getattr(args_i, "expr_layer", "X"))

#     n_pairs = int(pert_meta["n_obs_selected"])
#     selected_control_pos = rng.choice(np.arange(control.n_obs), size=n_pairs, replace=True).astype(np.int64)
#     X_pseudo = build_random_single_matrix(
#         X_control,
#         selected_control_pos,
#         batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
#     )

#     pair_metadata = basic_pair_metadata(pert_meta, S1)
#     pair_metadata["selected_control_cell_id"] = control.obs_names[selected_control_pos].astype(str)
#     pair_metadata["selected_control_cell_pos"] = selected_control_pos.astype(int)
#     pair_metadata["sampling_seed"] = int(seed)

#     result = write_pseudo_control_h5ad(
#         X_pseudo,
#         pert_meta,
#         output_var,
#         outdir,
#         S1,
#         pair_metadata,
#         extra_uns={
#             "source_control_h5ad": str(args_i.control_h5ad),
#             "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
#             "expr_layer": getattr(args_i, "expr_layer", "X"),
#             "sampling_seed": int(seed),
#         },
#         extra_obs_cols=["pair_id", "selected_control_cell_id", "selected_control_cell_pos", "sampling_seed"],
#     )
#     counts = pd.Series(selected_control_pos).value_counts()
#     qc = {
#         "strategy": S1,
#         "sampling_seed": int(seed),
#         "n_control_cells": int(control.n_obs),
#         "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
#         "n_pairs_written": int(n_pairs),
#         "n_genes_output": int(len(output_var)),
#         "n_unique_control_cells_used": int(counts.shape[0]),
#         "control_cell_usage_mean": float(counts.mean()),
#         "control_cell_usage_median": float(counts.median()),
#         "control_cell_usage_max": int(counts.max()),
#         "output_h5ad": str(result["output_h5ad"]),
#         "pair_metadata": str(result["pair_metadata_path"]),
#     }
#     _save_qc(qc, outdir)
#     result["qc"] = qc
#     print(f"[S1] Done: {result['output_h5ad']}")
#     return result


# # -----------------------------------------------------------------------------
# # S2: random average controls
# # -----------------------------------------------------------------------------

# def run_s2_random_average_controls(
#     args: SimpleNamespace,
#     outdir: str | Path,
#     seed: int,
#     n_control_cells_to_average: int,
# ) -> dict[str, Any]:
#     rng = set_seed(seed)
#     outdir = ensure_outdir(outdir)
#     args_i = SimpleNamespace(**vars(args), seed=int(seed), n_control_cells_to_average=int(n_control_cells_to_average))
#     save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

#     print(f"[S2] seed={seed}, k={n_control_cells_to_average} Loading data.")
#     control, pert_meta, output_var = _prepare_control_and_perturbed_metadata(args_i)
#     X_control = get_expr_matrix(control, getattr(args_i, "expr_layer", "X"))

#     k = int(n_control_cells_to_average)
#     if k <= 0:
#         raise ValueError("n_control_cells_to_average must be positive.")
#     replace = bool(getattr(args_i, "sampling_replace", True))
#     if not replace and k > control.n_obs:
#         raise ValueError("Cannot sample without replacement because k > number of control cells.")

#     n_pairs = int(pert_meta["n_obs_selected"])
#     sampled = rng.choice(np.arange(control.n_obs), size=(n_pairs, k), replace=replace).astype(np.int64)
#     X_pseudo = build_average_by_index_matrix(
#         X_control,
#         sampled,
#         batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
#     )

#     pair_metadata = basic_pair_metadata(pert_meta, S2)
#     if bool(getattr(args_i, "store_sampled_control_positions", True)):
#         pair_metadata["sampled_control_cell_positions"] = [format_array_as_string(row) for row in sampled]
#     pair_metadata["n_random_control_cells_averaged"] = int(k)
#     pair_metadata["sampling_seed"] = int(seed)
#     pair_metadata["sampling_replace"] = bool(replace)

#     result = write_pseudo_control_h5ad(
#         X_pseudo,
#         pert_meta,
#         output_var,
#         outdir,
#         S2,
#         pair_metadata,
#         extra_uns={
#             "source_control_h5ad": str(args_i.control_h5ad),
#             "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
#             "expr_layer": getattr(args_i, "expr_layer", "X"),
#             "sampling_seed": int(seed),
#             "n_random_control_cells_averaged": int(k),
#             "sampling_replace": bool(replace),
#         },
#         extra_obs_cols=["pair_id", "n_random_control_cells_averaged", "sampling_seed", "sampling_replace"],
#     )
#     counts = pd.Series(sampled.reshape(-1)).value_counts()
#     qc = {
#         "strategy": S2,
#         "sampling_seed": int(seed),
#         "n_control_cells": int(control.n_obs),
#         "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
#         "n_pairs_written": int(n_pairs),
#         "n_genes_output": int(len(output_var)),
#         "n_random_control_cells_averaged": int(k),
#         "sampling_replace": bool(replace),
#         "n_unique_control_cells_used": int(counts.shape[0]),
#         "control_cell_usage_mean": float(counts.mean()),
#         "control_cell_usage_median": float(counts.median()),
#         "control_cell_usage_max": int(counts.max()),
#         "output_h5ad": str(result["output_h5ad"]),
#         "pair_metadata": str(result["pair_metadata_path"]),
#     }
#     _save_qc(qc, outdir)
#     result["qc"] = qc
#     print(f"[S2] Done: {result['output_h5ad']}")
#     return result


# # -----------------------------------------------------------------------------
# # Metacell context shared by S3-S5
# # -----------------------------------------------------------------------------

# def load_seacell_context(args: SimpleNamespace, membership_path: str | Path) -> dict[str, Any]:
#     print(f"[Metacell context] membership={membership_path}")
#     control = sc.read_h5ad(str(args.control_h5ad))
#     perturbed = sc.read_h5ad(str(args.perturbed_h5ad))
#     control.obs_names_make_unique()
#     perturbed.obs_names_make_unique()
#     control, perturbed = align_control_and_perturbed(
#         control,
#         perturbed,
#         require_all_genes=bool(getattr(args, "require_all_genes", False)),
#     )
#     args.perturbation_key = infer_perturbation_key(perturbed.obs, getattr(args, "perturbation_key", None))
#     metacell_ids, groups, membership_df, membership_file = load_membership_groups(control, membership_path)
#     X_control = get_expr_matrix(control, getattr(args, "expr_layer", "X"))
#     control_metacells = build_metacell_anndata(
#         control,
#         X_control,
#         metacell_ids,
#         groups,
#         embedding_key=getattr(args, "embedding_key", None) if getattr(args, "embedding_key", None) in control.obsm else None,
#     )
#     return {
#         "control": control,
#         "perturbed": perturbed,
#         "X_control": X_control,
#         "metacell_ids": metacell_ids,
#         "groups": groups,
#         "membership_df": membership_df,
#         "membership_file": membership_file,
#         "control_metacells": control_metacells,
#         "perturbation_key": args.perturbation_key,
#     }


# def _perturbed_subset_positions(perturbed: AnnData, perturbation_key: str, args: SimpleNamespace) -> np.ndarray:
#     max_per = getattr(args, "max_pairs_per_perturbation", None)
#     if max_per is None:
#         return np.arange(perturbed.n_obs, dtype=np.int64)
#     rng = set_seed(getattr(args, "pair_selection_seed", 42))
#     values = perturbed.obs[perturbation_key].astype(str).values
#     selected = []
#     for pert_name in pd.Index(values).unique().astype(str):
#         idx = np.where(values == pert_name)[0]
#         if len(idx) > int(max_per):
#             idx = rng.choice(idx, size=int(max_per), replace=False)
#         selected.append(np.sort(idx).astype(np.int64))
#     return np.sort(np.concatenate(selected)) if selected else np.array([], dtype=np.int64)


# def _context_pert_meta(context: dict[str, Any], args: SimpleNamespace) -> dict[str, Any]:
#     perturbed = context["perturbed"]
#     key = context["perturbation_key"]
#     selected_positions = _perturbed_subset_positions(perturbed, key, args)
#     return {
#         "obs_all": perturbed.obs.copy(),
#         "obs": perturbed.obs.iloc[selected_positions].copy(),
#         "var": perturbed.var.copy(),
#         "obs_names_all": pd.Index(perturbed.obs_names.astype(str)),
#         "obs_names": pd.Index(perturbed.obs_names[selected_positions].astype(str)),
#         "var_names": pd.Index(perturbed.var_names.astype(str)),
#         "n_obs_total": int(perturbed.n_obs),
#         "n_obs_selected": int(len(selected_positions)),
#         "n_vars": int(perturbed.n_vars),
#         "selected_positions": selected_positions,
#         "perturbation_key": key,
#     }


# # -----------------------------------------------------------------------------
# # S3: SEACell metacell average
# # -----------------------------------------------------------------------------

# def run_s3_seacell_metacell_average(
#     args: SimpleNamespace,
#     context: dict[str, Any],
#     outdir: str | Path,
#     seed: int,
#     n_metacells_to_average: int,
# ) -> dict[str, Any]:
#     rng = set_seed(seed)
#     outdir = ensure_outdir(outdir)
#     args_i = SimpleNamespace(**vars(args), seed=int(seed), n_metacells_to_average=int(n_metacells_to_average))
#     save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

#     control_mc = context["control_metacells"]
#     X_mc = get_expr_matrix(control_mc, "X")
#     metacell_ids = context["metacell_ids"]
#     pert_meta = _context_pert_meta(context, args_i)
#     n_pairs = int(pert_meta["n_obs_selected"])
#     k = int(n_metacells_to_average)
#     if k <= 0:
#         raise ValueError("n_metacells_to_average must be positive.")
#     replace = bool(getattr(args_i, "sample_metacells_with_replacement", True))
#     if not replace and k > len(metacell_ids):
#         raise ValueError("Cannot sample metacells without replacement because k > number of metacells.")
#     sampled_mc = rng.choice(np.arange(len(metacell_ids)), size=(n_pairs, k), replace=replace).astype(np.int64)
#     X_pseudo = build_average_by_index_matrix(
#         X_mc,
#         sampled_mc,
#         batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
#     )

#     pair_metadata = basic_pair_metadata(pert_meta, S3)
#     pair_metadata["sampled_metacell_indices"] = [format_array_as_string(row) for row in sampled_mc]
#     pair_metadata["sampled_metacell_ids"] = [format_array_as_string([metacell_ids[int(x)] for x in row]) for row in sampled_mc]
#     pair_metadata["n_metacells_averaged"] = int(k)
#     pair_metadata["sampling_seed"] = int(seed)
#     pair_metadata["sample_metacells_with_replacement"] = bool(replace)
#     pair_metadata["membership_path"] = str(context["membership_file"])

#     result = write_pseudo_control_h5ad(
#         X_pseudo,
#         pert_meta,
#         context["perturbed"].var.copy(),
#         outdir,
#         S3,
#         pair_metadata,
#         extra_uns={
#             "source_control_h5ad": str(args_i.control_h5ad),
#             "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
#             "membership_path": str(context["membership_file"]),
#             "n_metacells_observed": int(len(metacell_ids)),
#             "n_metacells_averaged": int(k),
#             "sampling_seed": int(seed),
#             "sample_metacells_with_replacement": bool(replace),
#         },
#         extra_obs_cols=["pair_id", "n_metacells_averaged", "sampling_seed", "sampled_metacell_ids"],
#     )
#     usage = sparse_usage_summary(sampled_mc, metacell_ids, "n_sampled")
#     usage_path = save_dataframe(usage, outdir / "metacell_usage_summary")
#     qc = {
#         "strategy": S3,
#         "sampling_seed": int(seed),
#         "membership_path": str(context["membership_file"]),
#         "n_control_cells": int(context["control"].n_obs),
#         "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
#         "n_pairs_written": int(n_pairs),
#         "n_genes_output": int(context["perturbed"].n_vars),
#         "n_metacells_observed": int(len(metacell_ids)),
#         "n_metacells_averaged": int(k),
#         "n_unique_metacells_used": int(usage.shape[0]),
#         "metacell_usage_entropy": entropy_from_counts(usage["n_sampled"].values),
#         "metacell_max_usage_fraction": float(usage["n_sampled"].max() / usage["n_sampled"].sum()),
#         "metacell_usage_summary": str(usage_path),
#         "output_h5ad": str(result["output_h5ad"]),
#         "pair_metadata": str(result["pair_metadata_path"]),
#     }
#     _save_qc(qc, outdir)
#     result["qc"] = qc
#     result["metacell_usage_path"] = usage_path
#     print(f"[S3] Done: {result['output_h5ad']}")
#     return result


# # -----------------------------------------------------------------------------
# # S4: SEACell balanced random sample
# # -----------------------------------------------------------------------------

# def balanced_metacell_sequence(n_pairs: int, n_metacells: int, rng: np.random.Generator) -> np.ndarray:
#     if n_pairs <= 0:
#         return np.array([], dtype=np.int32)
#     base = np.tile(np.arange(n_metacells, dtype=np.int32), int(np.ceil(n_pairs / n_metacells)))[:n_pairs]
#     rng.shuffle(base)
#     return base.astype(np.int32)


# def sample_one_control_per_assigned_metacell(
#     groups: Sequence[np.ndarray],
#     assigned_mc: np.ndarray,
#     replace: bool,
#     rng: np.random.Generator,
# ) -> np.ndarray:
#     selected = np.empty(len(assigned_mc), dtype=np.int64)
#     assigned_mc = np.asarray(assigned_mc, dtype=np.int64)
#     for mc_idx in np.unique(assigned_mc):
#         row_idx = np.where(assigned_mc == mc_idx)[0]
#         members = groups[int(mc_idx)]
#         if len(members) == 0:
#             raise ValueError(f"Metacell index {mc_idx} has no member cells.")
#         use_replace = bool(replace) or len(row_idx) > len(members)
#         selected[row_idx] = rng.choice(members, size=len(row_idx), replace=use_replace)
#     return selected


# def run_s4_seacell_balanced_random_sample(
#     args: SimpleNamespace,
#     context: dict[str, Any],
#     outdir: str | Path,
#     seed: int,
# ) -> dict[str, Any]:
#     rng = set_seed(seed)
#     outdir = ensure_outdir(outdir)
#     args_i = SimpleNamespace(**vars(args), seed=int(seed))
#     save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

#     perturbed = context["perturbed"]
#     key = context["perturbation_key"]
#     pert_meta = _context_pert_meta(context, args_i)
#     selected_positions = pert_meta["selected_positions"]
#     values = perturbed.obs[key].astype(str).values
#     n_metacells = len(context["metacell_ids"])
#     assigned_mc = np.empty(len(selected_positions), dtype=np.int64)

#     # Balance metacell use independently within each perturbation identity.
#     pos_to_row = {int(pos): i for i, pos in enumerate(selected_positions)}
#     for pert_name in pd.Index(values[selected_positions]).unique().astype(str):
#         pert_pos = selected_positions[values[selected_positions] == pert_name]
#         local_rows = np.array([pos_to_row[int(p)] for p in pert_pos], dtype=np.int64)
#         assigned_mc[local_rows] = balanced_metacell_sequence(len(local_rows), n_metacells, rng)

#     selected_control_pos = sample_one_control_per_assigned_metacell(
#         context["groups"],
#         assigned_mc,
#         replace=bool(getattr(args_i, "sampling_replace", False)),
#         rng=rng,
#     )
#     X_pseudo = build_random_single_matrix(
#         context["X_control"],
#         selected_control_pos,
#         batch_size=int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 4096))),
#     )

#     metacell_ids = context["metacell_ids"]
#     pair_metadata = basic_pair_metadata(pert_meta, S4)
#     pair_metadata["assigned_metacell_index"] = assigned_mc.astype(int)
#     pair_metadata["assigned_metacell_id"] = [metacell_ids[int(i)] for i in assigned_mc]
#     pair_metadata["selected_control_cell_id"] = context["control"].obs_names[selected_control_pos].astype(str)
#     pair_metadata["selected_control_cell_pos"] = selected_control_pos.astype(int)
#     pair_metadata["sampling_seed"] = int(seed)
#     pair_metadata["membership_path"] = str(context["membership_file"])

#     result = write_pseudo_control_h5ad(
#         X_pseudo,
#         pert_meta,
#         context["perturbed"].var.copy(),
#         outdir,
#         S4,
#         pair_metadata,
#         extra_uns={
#             "source_control_h5ad": str(args_i.control_h5ad),
#             "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
#             "membership_path": str(context["membership_file"]),
#             "n_metacells_observed": int(n_metacells),
#             "sampling_seed": int(seed),
#         },
#         extra_obs_cols=[
#             "pair_id",
#             "assigned_metacell_id",
#             "assigned_metacell_index",
#             "selected_control_cell_id",
#             "selected_control_cell_pos",
#             "sampling_seed",
#         ],
#     )
#     selected_counts = pd.Series(selected_control_pos).value_counts()
#     metacell_counts = pd.Series(assigned_mc).value_counts()
#     usage = pd.DataFrame(
#         {
#             "metacell_index": metacell_counts.index.astype(int),
#             "metacell_id": [metacell_ids[int(i)] for i in metacell_counts.index.astype(int)],
#             "n_assigned_pairs": metacell_counts.values.astype(int),
#         }
#     ).sort_values("metacell_index")
#     usage_path = save_dataframe(usage, outdir / "metacell_usage_summary")
#     qc = {
#         "strategy": S4,
#         "sampling_seed": int(seed),
#         "membership_path": str(context["membership_file"]),
#         "n_control_cells": int(context["control"].n_obs),
#         "n_perturbed_cells_total": int(pert_meta["n_obs_total"]),
#         "n_pairs_written": int(len(selected_positions)),
#         "n_genes_output": int(context["perturbed"].n_vars),
#         "n_metacells_observed": int(n_metacells),
#         "n_unique_control_cells_used": int(selected_counts.shape[0]),
#         "control_cell_usage_mean": float(selected_counts.mean()),
#         "control_cell_usage_median": float(selected_counts.median()),
#         "control_cell_usage_max": int(selected_counts.max()),
#         "metacell_usage_entropy_total": entropy_from_counts(metacell_counts.values),
#         "metacell_usage_summary": str(usage_path),
#         "output_h5ad": str(result["output_h5ad"]),
#         "pair_metadata": str(result["pair_metadata_path"]),
#     }
#     _save_qc(qc, outdir)
#     result["qc"] = qc
#     result["metacell_usage_path"] = usage_path
#     print(f"[S4] Done: {result['output_h5ad']}")
#     return result


# # -----------------------------------------------------------------------------
# # S5: SEACell OT sampled average
# # -----------------------------------------------------------------------------

# def compute_control_mass(mc_sizes: np.ndarray, mode: str) -> np.ndarray:
#     if mode == "size":
#         a = mc_sizes.astype(np.float64)
#         return a / a.sum()
#     if mode == "uniform":
#         a = np.ones(len(mc_sizes), dtype=np.float64)
#         return a / a.sum()
#     raise ValueError("control_mass must be 'size' or 'uniform'.")


# def sinkhorn_ot(a: np.ndarray, b: np.ndarray, cost: np.ndarray, reg: float, max_iter: int, tol: float) -> np.ndarray:
#     try:
#         import ot
#     except ImportError as exc:
#         raise ImportError("POT is required for OT matching. Install with: pip install POT") from exc
#     cost = cost.astype(np.float64, copy=False)
#     try:
#         G = ot.sinkhorn(a, b, cost, reg=float(reg), numItermax=int(max_iter), stopThr=float(tol), warn=False)
#     except Exception:
#         G = ot.bregman.sinkhorn_stabilized(
#             a, b, cost, reg=float(reg), numItermax=int(max_iter), stopThr=float(tol), warn=False
#         )
#     if not np.all(np.isfinite(G)):
#         raise FloatingPointError("OT transport matrix contains non-finite values.")
#     return G


# def topk_from_transport(G: np.ndarray, k: int):
#     eps = 1e-12
#     n_source, _ = G.shape
#     k = min(int(k), n_source)
#     W = G / np.maximum(G.sum(axis=0, keepdims=True), eps)
#     entropy = -np.sum(W * np.log(W + eps), axis=0) / np.log(n_source)
#     idx_unsorted = np.argpartition(W, kth=n_source - k, axis=0)[-k:, :]
#     w_unsorted = np.take_along_axis(W, idx_unsorted, axis=0)
#     order = np.argsort(-w_unsorted, axis=0)
#     top_idx = np.take_along_axis(idx_unsorted, order, axis=0).T
#     top_w = np.take_along_axis(w_unsorted, order, axis=0).T
#     top_w = top_w / np.maximum(top_w.sum(axis=1, keepdims=True), eps)
#     return top_idx.astype(np.int32), top_w.astype(np.float32), entropy.astype(np.float32)


# def compute_ot_assignments_for_setting_topk(
#     args: SimpleNamespace,
#     context: dict[str, Any],
#     top_k: int,
#     assignment_outdir: str | Path,
# ) -> tuple[pd.DataFrame, dict[str, Any], Path]:
#     assignment_outdir = ensure_outdir(assignment_outdir)
#     assignment_prefix = assignment_outdir / f"ot_assignments_topk_{int(top_k):02d}"
#     qc_path = assignment_outdir / f"ot_assignment_qc_topk_{int(top_k):02d}.json"
#     if dataframe_file_exists(assignment_prefix) and qc_path.exists() and not bool(getattr(args, "overwrite_assignments", False)):
#         print(f"[S5/OT] Reusing cached assignments: {get_existing_dataframe_path(assignment_prefix)}")
#         return read_dataframe(assignment_prefix), read_json(qc_path), get_existing_dataframe_path(assignment_prefix)

#     if getattr(args, "embedding_key", None) not in context["control_metacells"].obsm:
#         raise KeyError(
#             f"S5 requires metacell and perturbed embeddings under obsm['{args.embedding_key}']. "
#             "Run preprocessing PCA first and ensure control/perturbed h5ad both contain this key."
#         )
#     if args.embedding_key not in context["perturbed"].obsm:
#         raise KeyError(f"Perturbed AnnData does not contain obsm['{args.embedding_key}'].")

#     rng_select = set_seed(getattr(args, "pair_selection_seed", 42))
#     perturbed = context["perturbed"]
#     key = context["perturbation_key"]
#     values = perturbed.obs[key].astype(str).values
#     unique_perts = pd.Index(values).unique().astype(str).tolist()
#     metacell_ids = context["metacell_ids"]
#     mc_sizes = context["control_metacells"].obs["n_control_cells"].values.astype(np.float64)
#     a = compute_control_mass(mc_sizes, getattr(args, "control_mass", "size"))
#     Z_mc = np.asarray(context["control_metacells"].obsm[args.embedding_key], dtype=np.float32)
#     Z_pert = np.asarray(perturbed.obsm[args.embedding_key], dtype=np.float32)

#     records = []
#     usage_records = []
#     total_pairs = 0
#     print(f"[S5/OT] Computing assignments for top_k={top_k}; perturbations={len(unique_perts):,}")
#     for pert_name in tqdm(unique_perts, desc="[S5/OT] Perturbation-wise OT"):
#         pert_idx = np.where(values == pert_name)[0]
#         max_per = getattr(args, "max_pairs_per_perturbation", None)
#         if max_per is not None and len(pert_idx) > int(max_per):
#             pert_idx = np.sort(rng_select.choice(pert_idx, size=int(max_per), replace=False))
#         if len(pert_idx) == 0:
#             continue
#         cost_raw = pairwise_distances(Z_mc, Z_pert[pert_idx], metric=getattr(args, "cost_metric", "sqeuclidean")).astype(np.float64)
#         positive = cost_raw[cost_raw > 0]
#         scale = np.median(positive) if positive.size > 0 else 1.0
#         cost = cost_raw / max(scale, 1e-12)
#         b = np.ones(len(pert_idx), dtype=np.float64)
#         b /= b.sum()
#         G = sinkhorn_ot(
#             a=a,
#             b=b,
#             cost=cost,
#             reg=getattr(args, "ot_reg", 0.05),
#             max_iter=getattr(args, "ot_max_iter", 2000),
#             tol=getattr(args, "ot_tol", 1e-7),
#         )
#         top_idx, top_w, entropy = topk_from_transport(G, top_k)
#         dominant_idx = top_idx[:, 0]
#         dominant_weight = top_w[:, 0]
#         dominant_distance = cost_raw[dominant_idx, np.arange(len(pert_idx))]
#         usage = pd.DataFrame(
#             {"perturbation": pert_name, "metacell_id": [metacell_ids[int(i)] for i in dominant_idx]}
#         )
#         usage = usage.groupby(["perturbation", "metacell_id"]).size().reset_index(name="n_dominant_assignments")
#         usage_records.append(usage)
#         batch = []
#         for local_i, ppos in enumerate(pert_idx):
#             rec = {
#                 "pair_id": f"pair_{total_pairs + local_i:012d}",
#                 "perturbed_cell_id": str(perturbed.obs_names[ppos]),
#                 "perturbed_cell_pos": int(ppos),
#                 "perturbation": str(pert_name),
#                 "dominant_metacell_id": str(metacell_ids[int(dominant_idx[local_i])]),
#                 "dominant_metacell_index": int(dominant_idx[local_i]),
#                 "dominant_weight": float(dominant_weight[local_i]),
#                 "dominant_distance": float(dominant_distance[local_i]),
#                 "matching_entropy": float(entropy[local_i]),
#                 "top_metacell_ids": format_array_as_string([metacell_ids[int(x)] for x in top_idx[local_i]]),
#                 "top_metacell_indices": format_array_as_string(top_idx[local_i]),
#                 "top_metacell_weights": format_array_as_string(top_w[local_i]),
#                 "pairing_strategy": S5,
#                 "top_k_metacells": int(top_k),
#                 "ot_reg": float(getattr(args, "ot_reg", 0.05)),
#                 "control_mass": getattr(args, "control_mass", "size"),
#                 "pair_selection_seed": int(getattr(args, "pair_selection_seed", 42)),
#                 "membership_path": str(context["membership_file"]),
#             }
#             for j in range(int(top_k)):
#                 rec[f"top_metacell_index_{j}"] = int(top_idx[local_i, j])
#                 rec[f"top_metacell_weight_{j}"] = float(top_w[local_i, j])
#                 rec[f"top_metacell_id_{j}"] = str(metacell_ids[int(top_idx[local_i, j])])
#             batch.append(rec)
#         records.append(pd.DataFrame(batch))
#         total_pairs += len(batch)

#     if not records:
#         raise RuntimeError("No OT assignments were generated.")
#     assignment_df = pd.concat(records, axis=0, ignore_index=True)
#     assignment_df = assignment_df.sort_values("perturbed_cell_pos").reset_index(drop=True)
#     # Regenerate pair IDs after sorting by perturbed position.
#     assignment_df["pair_id"] = [f"pair_{i:012d}" for i in range(assignment_df.shape[0])]
#     assignment_path = save_dataframe(assignment_df, assignment_prefix)
#     if usage_records:
#         usage_df = pd.concat(usage_records, axis=0, ignore_index=True)
#         usage_path = save_dataframe(usage_df, assignment_outdir / f"metacell_usage_by_perturbation_topk_{int(top_k):02d}")
#         usage_summary = usage_df.groupby("metacell_id")["n_dominant_assignments"].sum().reset_index()
#         usage_summary_path = save_dataframe(usage_summary, assignment_outdir / f"metacell_usage_summary_topk_{int(top_k):02d}")
#     else:
#         usage_path = None
#         usage_summary_path = None
#     qc = {
#         "strategy": S5,
#         "membership_path": str(context["membership_file"]),
#         "n_control_cells": int(context["control"].n_obs),
#         "n_perturbed_cells_available": int(context["perturbed"].n_obs),
#         "n_pairs_assigned": int(assignment_df.shape[0]),
#         "n_metacells_observed": int(len(metacell_ids)),
#         "top_k_metacells": int(top_k),
#         "sample_cells_per_metacell": int(getattr(args, "sample_cells_per_metacell", 10)),
#         "ot_reg": float(getattr(args, "ot_reg", 0.05)),
#         "ot_max_iter": int(getattr(args, "ot_max_iter", 2000)),
#         "ot_tol": float(getattr(args, "ot_tol", 1e-7)),
#         "cost_metric": getattr(args, "cost_metric", "sqeuclidean"),
#         "control_mass": getattr(args, "control_mass", "size"),
#         "max_pairs_per_perturbation": None if getattr(args, "max_pairs_per_perturbation", None) is None else int(args.max_pairs_per_perturbation),
#         "pair_selection_seed": int(getattr(args, "pair_selection_seed", 42)),
#         "mean_dominant_weight": float(assignment_df["dominant_weight"].mean()),
#         "median_dominant_weight": float(assignment_df["dominant_weight"].median()),
#         "mean_matching_entropy": float(assignment_df["matching_entropy"].mean()),
#         "median_matching_entropy": float(assignment_df["matching_entropy"].median()),
#         "mean_dominant_distance": float(assignment_df["dominant_distance"].mean()),
#         "median_dominant_distance": float(assignment_df["dominant_distance"].median()),
#         "assignment_path": str(assignment_path),
#         "usage_path": None if usage_path is None else str(usage_path),
#         "usage_summary_path": None if usage_summary_path is None else str(usage_summary_path),
#     }
#     save_json(qc, qc_path)
#     return assignment_df, qc, assignment_path


# def sample_pseudo_controls_batch_from_ot(
#     X_control,
#     groups: Sequence[np.ndarray],
#     top_idx_batch: np.ndarray,
#     top_w_batch: np.ndarray,
#     sample_cells_per_metacell: int,
#     replace: bool,
#     rng: np.random.Generator,
# ) -> sp.csr_matrix:
#     n_rows = top_idx_batch.shape[0]
#     n_vars = X_control.shape[1]
#     out = np.zeros((n_rows, n_vars), dtype=np.float32)
#     for i in range(n_rows):
#         for mc_idx, weight in zip(top_idx_batch[i], top_w_batch[i]):
#             members = groups[int(mc_idx)]
#             if len(members) == 0:
#                 continue
#             if replace:
#                 sampled = rng.choice(members, size=int(sample_cells_per_metacell), replace=True)
#             else:
#                 n_sample = min(int(sample_cells_per_metacell), len(members))
#                 sampled = rng.choice(members, size=n_sample, replace=False)
#             out[i] += float(weight) * row_mean(X_control, sampled)
#     return sp.csr_matrix(out)


# def build_adata_from_shards(
#     perturbed: AnnData,
#     pair_metadata: pd.DataFrame,
#     shard_dir: str | Path,
#     outdir: str | Path,
#     strategy: str,
#     extra_uns: dict[str, Any] | None = None,
# ) -> Path:
#     outdir = ensure_outdir(outdir)
#     shard_dir = Path(shard_dir)
#     pair_metadata = pair_metadata.copy()
#     pair_metadata["perturbed_cell_pos"] = pair_metadata["perturbed_cell_pos"].astype(int)
#     pair_metadata["pseudo_control_row_in_shard"] = pair_metadata["pseudo_control_row_in_shard"].astype(int)
#     covered_positions = np.sort(pair_metadata["perturbed_cell_pos"].unique())
#     full = len(covered_positions) == perturbed.n_obs and np.array_equal(covered_positions, np.arange(perturbed.n_obs))
#     if full:
#         output_shape = (perturbed.n_obs, perturbed.n_vars)
#         obs = perturbed.obs.copy()
#         obs_names = pd.Index(perturbed.obs_names.astype(str))
#         pos_to_row = {int(pos): int(pos) for pos in covered_positions}
#     else:
#         output_shape = (len(covered_positions), perturbed.n_vars)
#         obs = perturbed.obs.iloc[covered_positions].copy()
#         obs_names = pd.Index(perturbed.obs_names[covered_positions].astype(str))
#         pos_to_row = {int(pos): i for i, pos in enumerate(covered_positions)}
#     pair_metadata["_target_row"] = pair_metadata["perturbed_cell_pos"].map(pos_to_row).astype(int)

#     data_parts = []
#     row_parts = []
#     col_parts = []
#     for shard_name, df_shard in tqdm(list(pair_metadata.groupby("pseudo_control_shard", sort=False)), desc="Assembling shards"):
#         shard_path = shard_dir / str(shard_name)
#         if not shard_path.exists():
#             raise FileNotFoundError(f"Missing shard file: {shard_path}")
#         X_shard = sp.load_npz(shard_path).tocsr()
#         X_sel = X_shard[df_shard["pseudo_control_row_in_shard"].to_numpy()].tocoo()
#         target_rows = df_shard["_target_row"].to_numpy()
#         data_parts.append(X_sel.data)
#         row_parts.append(target_rows[X_sel.row])
#         col_parts.append(X_sel.col)
#     X = sp.csr_matrix(
#         (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(col_parts))),
#         shape=output_shape,
#         dtype=np.float32,
#     )
#     obs = obs.copy()
#     obs.index = obs_names
#     obs["pseudo_control_available"] = True
#     obs["paired_perturbed_cell_id"] = obs_names.astype(str).values
#     obs["paired_perturbed_cell_pos"] = covered_positions.astype(int) if not full else np.arange(perturbed.n_obs)
#     obs["pairing_strategy"] = strategy
#     metadata_aligned = pair_metadata.set_index("perturbed_cell_pos").loc[covered_positions].reset_index()
#     obs_cols = [
#         "pair_id",
#         "perturbation",
#         "dominant_metacell_id",
#         "dominant_metacell_index",
#         "dominant_weight",
#         "dominant_distance",
#         "matching_entropy",
#         "top_metacell_ids",
#         "top_metacell_indices",
#         "top_metacell_weights",
#         "top_k_metacells",
#         "sampling_seed",
#         "sample_cells_per_metacell",
#         "sampling_replace",
#         "ot_reg",
#         "control_mass",
#         "pair_selection_seed",
#     ]
#     for col in obs_cols:
#         if col in metadata_aligned.columns:
#             obs[f"pairing_{col}"] = metadata_aligned[col].values
#     adata = AnnData(X=X, obs=obs, var=perturbed.var.copy())
#     adata.uns["pairing_strategy"] = strategy
#     adata.uns["row_alignment"] = "same_order_as_perturbed_h5ad" if full else "subset_sorted_by_original_perturbed_cell_order"
#     if extra_uns:
#         for k, v in extra_uns.items():
#             adata.uns[k] = to_jsonable(v)
#     output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
#     adata.write_h5ad(output_h5ad)
#     return output_h5ad


# def run_s5_seacell_ot_sampled_average(
#     args: SimpleNamespace,
#     context: dict[str, Any],
#     assignment_df: pd.DataFrame,
#     outdir: str | Path,
#     seed: int,
#     top_k: int,
# ) -> dict[str, Any]:
#     rng = set_seed(seed)
#     outdir = ensure_outdir(outdir)
#     ensure_outdir(outdir / "pseudo_control_shards")
#     args_i = SimpleNamespace(**vars(args), seed=int(seed), top_k_metacells=int(top_k))
#     save_json(namespace_to_dict(args_i), outdir / "strategy_config.json")

#     pair_metadata_prefix = outdir / "pair_metadata"
#     output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
#     if output_h5ad.exists() and dataframe_file_exists(pair_metadata_prefix) and not bool(getattr(args_i, "overwrite_sampled_outputs", False)):
#         print(f"[S5] Existing output found, skipping: {output_h5ad}")
#         qc_path = outdir / "pairing_qc_summary.json"
#         qc = read_json(qc_path) if qc_path.exists() else {}
#         return {
#             "outdir": outdir,
#             "output_h5ad": output_h5ad,
#             "pair_metadata_path": get_existing_dataframe_path(pair_metadata_prefix),
#             "qc": qc,
#             "skipped_existing": True,
#         }

#     idx_cols = [f"top_metacell_index_{j}" for j in range(int(top_k))]
#     weight_cols = [f"top_metacell_weight_{j}" for j in range(int(top_k))]
#     missing = [c for c in idx_cols + weight_cols if c not in assignment_df.columns]
#     if missing:
#         raise KeyError(f"Assignment table is missing columns: {missing}")

#     all_pair_records = []
#     shard_id = 0
#     batch_size = int(getattr(args_i, "batch_size", getattr(args_i, "matrix_batch_size", 1024)))
#     for start in tqdm(range(0, assignment_df.shape[0], batch_size), desc="[S5] Sampling pseudo-control shards"):
#         end = min(start + batch_size, assignment_df.shape[0])
#         df_batch = assignment_df.iloc[start:end].copy()
#         top_idx_batch = df_batch[idx_cols].to_numpy(dtype=np.int32)
#         top_w_batch = df_batch[weight_cols].to_numpy(dtype=np.float32)
#         X_pseudo = sample_pseudo_controls_batch_from_ot(
#             X_control=context["X_control"],
#             groups=context["groups"],
#             top_idx_batch=top_idx_batch,
#             top_w_batch=top_w_batch,
#             sample_cells_per_metacell=int(getattr(args_i, "sample_cells_per_metacell", 10)),
#             replace=bool(getattr(args_i, "sampling_replace", False)),
#             rng=rng,
#         )
#         shard_name = f"pseudo_control_shard_{shard_id:06d}.npz"
#         sp.save_npz(outdir / "pseudo_control_shards" / shard_name, X_pseudo)
#         df_batch["pseudo_control_shard"] = shard_name
#         df_batch["pseudo_control_row_in_shard"] = np.arange(df_batch.shape[0], dtype=np.int32)
#         df_batch["sampling_seed"] = int(seed)
#         df_batch["sample_cells_per_metacell"] = int(getattr(args_i, "sample_cells_per_metacell", 10))
#         df_batch["sampling_replace"] = bool(getattr(args_i, "sampling_replace", False))
#         df_batch["pairing_strategy"] = S5
#         all_pair_records.append(df_batch)
#         shard_id += 1

#     pair_metadata = pd.concat(all_pair_records, axis=0, ignore_index=True)
#     pair_metadata_path = save_dataframe(pair_metadata, pair_metadata_prefix)
#     final_h5ad = build_adata_from_shards(
#         context["perturbed"],
#         pair_metadata,
#         outdir / "pseudo_control_shards",
#         outdir,
#         S5,
#         extra_uns={
#             "source_control_h5ad": str(args_i.control_h5ad),
#             "source_perturbed_h5ad": str(args_i.perturbed_h5ad),
#             "membership_path": str(context["membership_file"]),
#             "top_k_metacells": int(top_k),
#             "sampling_seed": int(seed),
#             "sample_cells_per_metacell": int(getattr(args_i, "sample_cells_per_metacell", 10)),
#             "ot_reg": float(getattr(args_i, "ot_reg", 0.05)),
#             "control_mass": getattr(args_i, "control_mass", "size"),
#         },
#     )
#     qc = {
#         "strategy": S5,
#         "sampling_seed": int(seed),
#         "membership_path": str(context["membership_file"]),
#         "n_control_cells": int(context["control"].n_obs),
#         "n_perturbed_cells_available": int(context["perturbed"].n_obs),
#         "n_pairs_written": int(pair_metadata.shape[0]),
#         "n_genes_output": int(context["perturbed"].n_vars),
#         "n_metacells_observed": int(len(context["metacell_ids"])),
#         "top_k_metacells": int(top_k),
#         "sample_cells_per_metacell": int(getattr(args_i, "sample_cells_per_metacell", 10)),
#         "sampling_replace": bool(getattr(args_i, "sampling_replace", False)),
#         "pair_selection_seed": int(getattr(args_i, "pair_selection_seed", 42)),
#         "ot_reg": float(getattr(args_i, "ot_reg", 0.05)),
#         "control_mass": getattr(args_i, "control_mass", "size"),
#         "mean_dominant_weight": float(pair_metadata["dominant_weight"].mean()),
#         "median_dominant_weight": float(pair_metadata["dominant_weight"].median()),
#         "mean_matching_entropy": float(pair_metadata["matching_entropy"].mean()),
#         "median_matching_entropy": float(pair_metadata["matching_entropy"].median()),
#         "mean_dominant_distance": float(pair_metadata["dominant_distance"].mean()),
#         "median_dominant_distance": float(pair_metadata["dominant_distance"].median()),
#         "n_shards": int(shard_id),
#         "output_h5ad": str(final_h5ad),
#         "pair_metadata": str(pair_metadata_path),
#     }
#     _save_qc(qc, outdir)
#     print(f"[S5] Done: {final_h5ad}")
#     return {
#         "outdir": outdir,
#         "output_h5ad": final_h5ad,
#         "pair_metadata_path": pair_metadata_path,
#         "qc": qc,
#         "skipped_existing": False,
#     }
