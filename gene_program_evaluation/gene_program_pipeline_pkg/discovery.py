from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .shared_paths import canonicalize_index, discover_pseudo_files, pseudo_index_from_manifest, selected_variant_ids
from .utils import read_table


def pseudo_file_index_from_dataset_cfg(dataset_cfg: Mapping[str, Any], global_cfg: Mapping[str, Any]) -> pd.DataFrame:
    if dataset_cfg.get("pseudo_files"):
        df = pd.DataFrame(dataset_cfg["pseudo_files"])
        if df.empty:
            return df
        if "pseudo_control_h5ad" not in df.columns:
            raise KeyError("pseudo_files entries must contain pseudo_control_h5ad")
        return canonicalize_index(df)

    manifest_path = dataset_cfg.get("manifest_path")
    group = dataset_cfg.get("perturbed_group")
    selected_ids = None
    if bool(global_cfg.get("use_selected_variants", False)):
        table = Path(str(dataset_cfg.get("result_analysis_dir"))) / str(global_cfg.get("selection_table_name", "selected_variants_TEMPLATE_EDIT_ME.csv"))
        selected_ids = selected_variant_ids(table, include_s0=bool(global_cfg.get("include_s0_when_using_selected_variants", True)))
    if manifest_path and Path(str(manifest_path)).exists():
        return pseudo_index_from_manifest(
            Path(str(manifest_path)),
            str(group),
            selected_ids=selected_ids,
            require_existing=bool(global_cfg.get("require_existing_pseudo_files", False)),
        )

    pseudo_root = dataset_cfg.get("pseudo_root")
    if pseudo_root:
        return discover_pseudo_files(
            pseudo_root,
            pseudo_glob=str(global_cfg.get("pseudo_glob", "**/pseudo_control*.h5ad")),
            require_existing=bool(global_cfg.get("require_existing_pseudo_files", False)),
        )
    return pd.DataFrame()
