"""SEACell membership helpers for pseudo-control construction.

This module separates metacell membership generation/loading from the actual
pairing strategies.  The downstream strategy modules only need a membership
file with columns: control_cell_id, control_cell_pos, metacell_id.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

from pseudo_pairing_utils import (
    aggregate_by_groups,
    dataframe_file_exists,
    ensure_outdir,
    get_existing_dataframe_path,
    get_expr_matrix,
    read_dataframe,
    save_dataframe,
    save_json,
    set_seed,
)


def membership_prefix_for_setting(membership_root: str | Path, setting_id: str) -> Path:
    return Path(membership_root) / str(setting_id) / "membership" / "control_cell_to_metacell_membership"


def run_or_reuse_seacells(control: AnnData, args: SimpleNamespace) -> AnnData:
    """Fit SEACells unless args.seacell_key is already in control.obs."""
    if args.seacell_key in control.obs:
        print(f"[SEACells] Reusing control.obs['{args.seacell_key}'].")
        control.obs[args.seacell_key] = control.obs[args.seacell_key].astype(str)
        return control

    if args.embedding_key not in control.obsm:
        raise KeyError(
            f"control.obsm['{args.embedding_key}'] was not found. Available obsm keys: {list(control.obsm.keys())}"
        )

    print("[SEACells] Fitting SEACells on control cells.")
    print(f"[SEACells] setting_id = {getattr(args, 'seacell_setting_id', 'NA')}")
    print(f"[SEACells] n_metacells = {args.n_metacells}")
    print(f"[SEACells] build_kernel_on = {args.embedding_key}")

    import SEACells

    kwargs = dict(
        build_kernel_on=args.embedding_key,
        n_SEACells=int(args.n_metacells),
        n_waypoint_eigs=int(args.n_waypoint_eigs),
        convergence_epsilon=float(getattr(args, "seacells_convergence_epsilon", 1e-5)),
    )
    if bool(getattr(args, "use_gpu_seacells", False)):
        kwargs["use_gpu"] = True

    model = SEACells.core.SEACells(control, **kwargs)
    if hasattr(model, "construct_kernel_matrix"):
        model.construct_kernel_matrix()
    model.initialize_archetypes()
    try:
        model.fit(n_iter=int(args.seacells_n_iter))
    except TypeError:
        model.fit(min_iter=int(getattr(args, "seacells_min_iter", 10)), max_iter=int(args.seacells_n_iter))

    if "SEACell" not in control.obs:
        raise RuntimeError("SEACells fitting finished, but control.obs['SEACell'] was not found.")
    control.obs[args.seacell_key] = control.obs["SEACell"].astype(str)
    return control


def build_membership_from_control_obs(control: AnnData, seacell_key: str) -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
    labels = control.obs[seacell_key].astype(str).values
    metacell_ids = pd.Index(labels).unique().astype(str).tolist()
    groups: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for mc in metacell_ids:
        idx = np.where(labels == mc)[0].astype(np.int64)
        groups.append(idx)
        for pos in idx:
            records.append(
                {
                    "control_cell_id": str(control.obs_names[pos]),
                    "control_cell_pos": int(pos),
                    "metacell_id": str(mc),
                }
            )
    membership_df = pd.DataFrame(records)
    return metacell_ids, groups, membership_df


def load_membership_groups(control: AnnData, membership_path: str | Path) -> tuple[list[str], list[np.ndarray], pd.DataFrame, Path]:
    """Load membership and validate/fix cell positions against the current control AnnData."""
    membership_file = get_existing_dataframe_path(membership_path)
    membership_df = read_dataframe(membership_file).copy()
    required = ["control_cell_id", "control_cell_pos", "metacell_id"]
    missing = [c for c in required if c not in membership_df.columns]
    if missing:
        raise KeyError(f"Membership file is missing columns: {missing}")

    membership_df["control_cell_id"] = membership_df["control_cell_id"].astype(str)
    membership_df["metacell_id"] = membership_df["metacell_id"].astype(str)
    membership_df["control_cell_pos"] = membership_df["control_cell_pos"].astype(int)

    pos = membership_df["control_cell_pos"].to_numpy()
    valid_pos = np.logical_and(pos >= 0, pos < control.n_obs)
    use_id_mapping = False
    if not np.all(valid_pos):
        warnings.warn("Some control_cell_pos values are outside control.n_obs; falling back to control_cell_id mapping.")
        use_id_mapping = True
    else:
        observed_ids = control.obs_names[pos].astype(str)
        expected_ids = membership_df["control_cell_id"].to_numpy()
        agreement = float(np.mean(observed_ids == expected_ids))
        if agreement < 0.99:
            warnings.warn(
                f"Only {agreement:.3f} of membership positions agree with control_cell_id; falling back to ID mapping."
            )
            use_id_mapping = True

    if use_id_mapping:
        id_to_pos = {str(cid): i for i, cid in enumerate(control.obs_names.astype(str))}
        mapped_pos = membership_df["control_cell_id"].map(id_to_pos)
        if mapped_pos.isna().any():
            example = membership_df.loc[mapped_pos.isna(), "control_cell_id"].head().tolist()
            raise ValueError(f"Some membership cell IDs are absent from control AnnData. Examples: {example}")
        membership_df["control_cell_pos"] = mapped_pos.astype(int)

    metacell_ids = pd.Index(membership_df["metacell_id"]).unique().astype(str).tolist()
    groups = []
    for mc in metacell_ids:
        idx = membership_df.loc[membership_df["metacell_id"] == mc, "control_cell_pos"].to_numpy(dtype=np.int64)
        if len(idx) == 0:
            raise ValueError(f"Metacell {mc} is empty in membership {membership_file}")
        groups.append(idx)
    return metacell_ids, groups, membership_df, membership_file


def build_seacell_membership_for_setting(
    control_h5ad: str | Path,
    membership_root: str | Path,
    setting: dict[str, Any],
    base_args: SimpleNamespace,
    overwrite: bool = False,
) -> Path:
    """Build or reuse one fixed SEACell membership for one setting."""
    setting_id = str(setting["setting_id"])
    membership_prefix = membership_prefix_for_setting(membership_root, setting_id)
    ensure_outdir(membership_prefix.parent)
    config_path = membership_prefix.parent / "seacell_membership_config.json"

    if dataframe_file_exists(membership_prefix) and config_path.exists() and not overwrite:
        print(f"[Membership] Reusing existing membership for {setting_id}: {get_existing_dataframe_path(membership_prefix)}")
        return membership_prefix

    args = deepcopy(base_args)
    args.seacell_setting_id = setting_id
    args.n_metacells = int(setting["n_metacells"])
    args.n_waypoint_eigs = int(setting.get("n_waypoint_eigs", getattr(base_args, "n_waypoint_eigs", 10)))
    args.seacells_n_iter = int(setting.get("seacells_n_iter", getattr(base_args, "seacells_n_iter", 50)))
    args.seacell_seed = int(setting.get("seacell_seed", getattr(base_args, "seacell_seed", 42)))
    args.seacell_key = str(setting.get("seacell_key", f"{getattr(base_args, 'seacell_key_prefix', 'SEACell')}_{setting_id}"))

    set_seed(args.seacell_seed)
    print("=" * 90)
    print(f"[Membership] Building SEACells setting={setting_id}")
    print("=" * 90)
    control = sc.read_h5ad(str(control_h5ad))
    control.obs_names_make_unique()

    # Avoid accidental reuse of old generic columns unless the requested key exists.
    for col in ["SEACell"]:
        if col in control.obs.columns and col != args.seacell_key:
            del control.obs[col]

    control = run_or_reuse_seacells(control, args)
    metacell_ids, groups, membership_df = build_membership_from_control_obs(control, args.seacell_key)
    saved = save_dataframe(membership_df, membership_prefix)
    sizes = np.array([len(g) for g in groups], dtype=int)
    save_json(
        {
            "setting_id": setting_id,
            "seacell_key": args.seacell_key,
            "control_h5ad": str(control_h5ad),
            "n_metacells_requested": int(args.n_metacells),
            "n_metacells_observed": int(len(metacell_ids)),
            "n_waypoint_eigs": int(args.n_waypoint_eigs),
            "seacells_n_iter": int(args.seacells_n_iter),
            "seacell_seed": int(args.seacell_seed),
            "embedding_key": str(args.embedding_key),
            "membership_path": str(saved),
            "min_metacell_size": int(sizes.min()),
            "median_metacell_size": float(np.median(sizes)),
            "max_metacell_size": int(sizes.max()),
        },
        config_path,
    )
    print(f"[Membership] Saved: {saved}")
    return membership_prefix


def ensure_memberships_for_settings(
    control_h5ad: str | Path,
    membership_root: str | Path,
    seacell_settings: Sequence[dict[str, Any]],
    base_args: SimpleNamespace,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Build/reuse memberships for all settings and return setting_id -> prefix."""
    paths = {}
    for setting in seacell_settings:
        setting_id = str(setting["setting_id"])
        paths[setting_id] = build_seacell_membership_for_setting(
            control_h5ad=control_h5ad,
            membership_root=membership_root,
            setting=setting,
            base_args=base_args,
            overwrite=overwrite,
        )
    return paths


def build_metacell_anndata(
    control: AnnData,
    X_control,
    metacell_ids: Sequence[str],
    groups: Sequence[np.ndarray],
    embedding_key: str | None = None,
) -> AnnData:
    """Build mean-expression metacell AnnData and optional mean embedding."""
    X_mc = aggregate_by_groups(X_control, groups, mode="mean")
    mc_obs = pd.DataFrame(
        {"metacell_id": list(map(str, metacell_ids)), "n_control_cells": [len(g) for g in groups]},
        index=list(map(str, metacell_ids)),
    )
    mc = AnnData(X=X_mc, obs=mc_obs, var=control.var.copy())
    if embedding_key is not None:
        if embedding_key not in control.obsm:
            raise KeyError(f"control.obsm['{embedding_key}'] was not found. Available keys: {list(control.obsm.keys())}")
        Z = np.asarray(control.obsm[embedding_key], dtype=np.float32)
        mc.obsm[embedding_key] = np.vstack([Z[g].mean(axis=0) for g in groups]).astype(np.float32)
    return mc
