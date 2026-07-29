"""Configurable AnnData preprocessing extracted from the original notebook."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _require_scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError("Preprocessing requires scanpy and anndata. Install pseudopair[core].") from exc
    return sc


def _annotate_groups(adata, annotation: Mapping[str, Any]) -> None:
    obs = adata.obs
    nperts_key = annotation.get("nperts_key")
    condition_key = str(annotation.get("condition_key", "condition"))
    multiplicity_key = str(annotation.get("multiplicity_key", "perturbation_multiplicity"))
    group_key = str(annotation.get("analysis_group_key", "analysis_group"))

    if nperts_key and nperts_key in obs.columns:
        nperts = pd.to_numeric(obs[nperts_key], errors="raise").astype(int)
        obs[condition_key] = np.where(nperts == 0, "control", "perturbed")
        labels = np.full(adata.n_obs, "multi", dtype=object)
        labels[nperts.to_numpy() == 0] = "control"
        labels[nperts.to_numpy() == 1] = "single"
        labels[nperts.to_numpy() == 2] = "dual"
        obs[multiplicity_key] = labels
        obs[group_key] = labels
    elif group_key not in obs.columns:
        source_key = annotation.get("group_source_key")
        mapping = annotation.get("group_mapping", {})
        if not source_key or source_key not in obs.columns:
            raise KeyError(
                f"Cannot create '{group_key}'. Provide annotation.nperts_key or annotation.group_source_key."
            )
        obs[group_key] = obs[source_key].astype(str).map(mapping).fillna(obs[source_key].astype(str))

    perturbation_source = annotation.get("perturbation_source_key")
    standardized_key = str(annotation.get("standardized_perturbation_key", "perturbation_key"))
    if perturbation_source:
        if perturbation_source not in obs.columns:
            raise KeyError(f"Perturbation source column not found: {perturbation_source}")
        obs[standardized_key] = obs[perturbation_source].astype(str)
    control_label = str(annotation.get("control_perturbation_label", "control"))
    if standardized_key in obs.columns and group_key in obs.columns:
        obs.loc[obs[group_key].astype(str) == "control", standardized_key] = control_label

    for key in [condition_key, multiplicity_key, group_key]:
        if key in obs.columns:
            obs[key] = obs[key].astype("category")


def _compute_embedding(adata, cfg: Mapping[str, Any], prefix: str) -> None:
    sc = _require_scanpy()
    n_top = min(int(cfg.get("n_top_genes", 3000)), adata.n_vars)
    hvg_flavor = str(cfg.get("hvg_flavor", "seurat"))
    counts_layer = cfg.get("counts_layer", "counts")
    hvg_kwargs: dict[str, Any] = {"n_top_genes": n_top, "flavor": hvg_flavor}
    if hvg_flavor == "seurat_v3" and counts_layer in adata.layers:
        hvg_kwargs["layer"] = counts_layer
    sc.pp.highly_variable_genes(adata, **hvg_kwargs)
    hvg = adata[:, adata.var["highly_variable"]].copy()
    n_comps = min(int(cfg.get("n_pcs", 50)), hvg.n_obs - 1, hvg.n_vars - 1)
    if n_comps < 2:
        raise ValueError(f"Not enough observations/genes for PCA in {prefix}.")
    sc.pp.pca(hvg, n_comps=n_comps, use_highly_variable=False, svd_solver=str(cfg.get("svd_solver", "arpack")))
    adata.obsm[str(cfg.get("embedding_key", "X_pca"))] = hvg.obsm["X_pca"].copy()

    if bool(cfg.get("neighbors", True)):
        neighbors_key = f"neighbors_{prefix}"
        sc.pp.neighbors(
            adata,
            use_rep=str(cfg.get("embedding_key", "X_pca")),
            n_neighbors=min(int(cfg.get("n_neighbors", 30)), max(2, adata.n_obs - 1)),
            n_pcs=n_comps,
            key_added=neighbors_key,
        )
        if bool(cfg.get("leiden", True)):
            sc.tl.leiden(
                adata,
                resolution=float(cfg.get("leiden_resolution", 1.0)),
                neighbors_key=neighbors_key,
                key_added=f"leiden_{prefix}",
            )
        if bool(cfg.get("umap", True)):
            sc.tl.umap(adata, neighbors_key=neighbors_key)
            adata.obsm[f"X_umap_{prefix}"] = adata.obsm["X_umap"].copy()


def run_preprocessing(config: Mapping[str, Any], acquired_outputs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    sc = _require_scanpy()
    project = dict(config.get("project", {}))
    dataset_id = str(project["dataset_id"])
    workdir = Path(project.get("workdir", "./pseudopair_work"))
    cfg = dict(config.get("preprocessing", {}))
    input_h5ad = cfg.get("input_h5ad") or cfg.get("input")
    if not input_h5ad and acquired_outputs:
        acquisition_index = int(cfg.get("acquisition_output_index", 0))
        try:
            input_h5ad = acquired_outputs[acquisition_index]["output"]
        except IndexError as exc:
            raise IndexError(
                f"preprocessing.acquisition_output_index={acquisition_index} is outside the acquisition output list."
            ) from exc
    if not input_h5ad:
        raise ValueError("No preprocessing input_h5ad could be resolved.")
    input_h5ad = Path(input_h5ad)
    output_dir = Path(cfg.get("output_dir", workdir / "processed" / dataset_id))
    group_dir = output_dir / "groups"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(str(input_h5ad))
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    _annotate_groups(adata, dict(cfg.get("annotation", {})))

    counts_layer = str(cfg.get("counts_layer", "counts"))
    if bool(cfg.get("store_counts", True)) and counts_layer not in adata.layers:
        adata.layers[counts_layer] = adata.X.copy()
    if bool(cfg.get("normalize_total", True)):
        sc.pp.normalize_total(adata, target_sum=float(cfg.get("target_sum", 1e4)))
    if bool(cfg.get("log1p", True)):
        sc.pp.log1p(adata)
    if bool(cfg.get("compute_global_embedding", True)):
        _compute_embedding(adata, cfg, "global")

    global_path = output_dir / f"{dataset_id}_global_processed.h5ad"
    adata.write_h5ad(global_path, compression=str(cfg.get("compression", "gzip")))

    group_key = str(cfg.get("annotation", {}).get("analysis_group_key", "analysis_group"))
    if group_key not in adata.obs.columns:
        raise KeyError(f"Group key not found after annotation: {group_key}")
    requested_groups = cfg.get("groups")
    groups = list(requested_groups) if requested_groups else list(adata.obs[group_key].astype(str).unique())
    min_cells = int(cfg.get("min_cells_per_group", 500))
    records = []
    for group in groups:
        sub = adata[adata.obs[group_key].astype(str) == str(group)].copy()
        if sub.n_obs < min_cells:
            records.append({"group": group, "n_cells": sub.n_obs, "status": "skipped_too_few_cells", "h5ad": ""})
            continue
        if bool(cfg.get("recompute_group_embedding", True)):
            _compute_embedding(sub, cfg, "within_group")
        path = group_dir / f"{dataset_id}_{group}_processed.h5ad"
        sub.write_h5ad(path, compression=str(cfg.get("compression", "gzip")))
        records.append({"group": group, "n_cells": sub.n_obs, "n_genes": sub.n_vars, "status": "written", "h5ad": str(path)})

    manifest = pd.DataFrame(records)
    manifest_path = output_dir / "preprocessing_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "dataset_id": dataset_id,
        "input_h5ad": str(input_h5ad),
        "global_h5ad": str(global_path),
        "group_manifest": str(manifest_path),
        "group_paths": {str(r["group"]): str(r["h5ad"]) for r in records if r["status"] == "written"},
    }
    (output_dir / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
