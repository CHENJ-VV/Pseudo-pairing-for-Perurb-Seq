"""Shared helpers for repeated pseudo-control evaluation pipelines.

This module is intentionally dataset-agnostic.  It expects a pseudo-pairing
manifest produced by the refactored S0-S5 generation workflow, but it also
normalizes older manifests when column names differ slightly.
"""
from __future__ import annotations

import json
import random
import re
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


STRATEGY_ORDER = [
    "S0_naive_mean_control_reference",
    "S1_random_single_control",
    "S2_random_average_controls",
    "S3_SEACell_metacell_average",
    "S4_SEACell_balanced_random_sample",
    "S5_SEACell_OT_sampled_average",
]


def as_namespace(config: Mapping[str, Any] | SimpleNamespace) -> SimpleNamespace:
    if isinstance(config, SimpleNamespace):
        return config
    return SimpleNamespace(**dict(config))


def get_config(config: Mapping[str, Any] | SimpleNamespace, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Index):
        return obj.astype(str).tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return obj


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> np.random.Generator:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    return np.random.default_rng(seed)


def make_safe_id(x: str, max_len: int = 180) -> str:
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:max_len]


def save_dataframe(df: pd.DataFrame, path_prefix: str | Path) -> Path:
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = path_prefix.with_suffix("") if path_prefix.suffix in {".csv", ".parquet"} else path_prefix
    parquet_path = stem.with_suffix(".parquet")
    csv_path = stem.with_suffix(".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as exc:
        warnings.warn(f"Could not write parquet because: {repr(exc)}. Falling back to CSV.")
        df.to_csv(csv_path, index=False)
        return csv_path


def dataframe_file_exists(path_prefix: str | Path) -> bool:
    path_prefix = Path(path_prefix)
    return path_prefix.exists() or path_prefix.with_suffix(".parquet").exists() or path_prefix.with_suffix(".csv").exists()


def get_existing_dataframe_path(path_prefix: str | Path) -> Path:
    path_prefix = Path(path_prefix)
    if path_prefix.exists():
        return path_prefix
    if path_prefix.with_suffix(".parquet").exists():
        return path_prefix.with_suffix(".parquet")
    if path_prefix.with_suffix(".csv").exists():
        return path_prefix.with_suffix(".csv")
    raise FileNotFoundError(f"Cannot find dataframe at prefix/path: {path_prefix}")


def read_dataframe(path_prefix: str | Path) -> pd.DataFrame:
    path = get_existing_dataframe_path(path_prefix)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe file type: {path}")


def to_csr(X) -> sp.csr_matrix:
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return sp.csr_matrix(np.asarray(X, dtype=np.float32))


def to_dense_float32(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def slice_to_dense(X, idx) -> np.ndarray:
    out = X[idx, :]
    if sp.issparse(out):
        out = out.toarray()
    return np.asarray(out, dtype=np.float32)


def mean_axis0(X) -> np.ndarray:
    if sp.issparse(X):
        return np.asarray(X.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(X.mean(axis=0)).ravel().astype(np.float32)


def mean_var_axis0(X) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        mean_sq = np.square(X).mean(axis=0)
    var = mean_sq - np.square(mean)
    var[var < 0] = 0
    return mean.astype(np.float64), var.astype(np.float64)


def safe_pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask] - np.mean(x[mask])
    y = y[mask] - np.mean(y[mask])
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def safe_cosine(x, y) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def safe_spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(x[mask], y[mask]).correlation)


def rowwise_pearson(A, B) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))
    out = np.sum(A * B, axis=1) / (denom + 1e-12)
    out[denom <= 0] = np.nan
    return out


def rowwise_cosine(A, B) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    denom = np.sqrt(np.sum(A * A, axis=1) * np.sum(B * B, axis=1))
    out = np.sum(A * B, axis=1) / (denom + 1e-12)
    out[denom <= 0] = np.nan
    return out


def summarize_array(values, prefix: str) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.nanmean(arr)) if arr.size else np.nan,
        f"{prefix}_median": float(np.nanmedian(arr)) if arr.size else np.nan,
        f"{prefix}_std": float(np.nanstd(arr)) if arr.size else np.nan,
        f"{prefix}_n": int(np.sum(np.isfinite(arr))) if arr.size else 0,
    }


def summarize_numeric(df: pd.DataFrame, group_cols: Sequence[str], output_path: str | Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metadata_numeric_cols = {
        "sampling_seed", "n_metacells_requested", "n_metacells_observed",
        "top_k_metacells", "n_control_cells_to_average", "n_metacells_to_average",
        "perturbation_id",
    }
    numeric_cols = [
        c for c in df.columns
        if c not in group_cols and c not in metadata_numeric_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        return pd.DataFrame()
    summary = df.groupby(list(group_cols), dropna=False)[numeric_cols].agg(["mean", "std", "median", "min", "max", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary


def standardize_manifest_columns(manifest: pd.DataFrame) -> pd.DataFrame:
    df = manifest.copy()
    if "pseudo_control_h5ad" not in df.columns:
        candidates = ["output_h5ad", "pseudo_h5ad", "h5ad_path"]
        for c in candidates:
            if c in df.columns:
                df["pseudo_control_h5ad"] = df[c]
                break
    if "pseudo_control_h5ad" not in df.columns:
        raise KeyError("Manifest must contain pseudo_control_h5ad or output_h5ad.")

    if "strategy" not in df.columns:
        if "strategy_id" in df.columns:
            df["strategy"] = df["strategy_id"]
        else:
            df["strategy"] = "UNKNOWN"

    if "strategy_id" not in df.columns:
        df["strategy_id"] = df["strategy"]
    if "strategy_family" not in df.columns:
        df["strategy_family"] = df["strategy"].astype(str).str.replace(r"^S\d+_", "", regex=True)
    if "parameter_label" not in df.columns:
        df["parameter_label"] = "default"
    if "sampling_seed" not in df.columns:
        df["sampling_seed"] = np.nan
    if "perturbed_group" not in df.columns:
        df["perturbed_group"] = "default"
    if "run_id" not in df.columns:
        df["run_id"] = [make_safe_id(f"{s}__{g}__{p}__seed_{seed}") for s, g, p, seed in zip(
            df["strategy"], df["perturbed_group"], df["parameter_label"], df["sampling_seed"]
        )]

    for col in ["seacell_setting_id", "pair_metadata_path", "membership_path_for_metacell_coverage"]:
        if col not in df.columns:
            df[col] = ""

    df["pseudo_control_h5ad"] = df["pseudo_control_h5ad"].astype(str)
    df = df[df["pseudo_control_h5ad"].map(lambda p: Path(p).exists())].copy()

    strategy_order = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    df["_order"] = df["strategy"].map(strategy_order).fillna(99)
    sort_cols = ["perturbed_group", "_order", "strategy", "parameter_label", "sampling_seed", "pseudo_control_h5ad"]
    df = df.sort_values(sort_cols).drop(columns=["_order"]).reset_index(drop=True)
    return df


def load_run_manifest(config: Mapping[str, Any] | SimpleNamespace) -> pd.DataFrame:
    manifest_path = get_config(config, "manifest_path", None)
    if manifest_path is None:
        raise ValueError("Set CONFIG.manifest_path to pseudo_pairing_repetition_manifest.csv.")
    df = pd.read_csv(manifest_path)
    df = standardize_manifest_columns(df)
    groups = get_config(config, "perturbed_groups_to_evaluate", None)
    if groups is not None:
        groups = {str(x) for x in groups}
        df = df[df["perturbed_group"].astype(str).isin(groups)].copy()
    strategies = get_config(config, "strategies_to_evaluate", None)
    if strategies is not None:
        strategies = {str(x) for x in strategies}
        df = df[df["strategy"].astype(str).isin(strategies)].copy()
    max_runs = get_config(config, "max_runs_to_evaluate", None)
    if max_runs is not None:
        df = df.iloc[:int(max_runs)].copy()
    if df.empty:
        raise RuntimeError("No pseudo-control runs left after manifest filtering.")
    return df.reset_index(drop=True)


def select_eval_genes(control, perturbed=None, runs_df: pd.DataFrame | None = None, max_eval_genes: int | None = None,
                      check_shared_genes_across_all_runs: bool = False) -> pd.Index:
    shared = pd.Index(control.var_names.astype(str))
    if perturbed is not None:
        shared = shared.intersection(pd.Index(perturbed.var_names.astype(str)))
    if check_shared_genes_across_all_runs and runs_df is not None:
        for p in runs_df["pseudo_control_h5ad"].astype(str).values:
            ad = sc.read_h5ad(p, backed="r")
            shared = shared.intersection(pd.Index(ad.var_names.astype(str)))
            ad.file.close()
    elif runs_df is not None and len(runs_df) > 0:
        ad = sc.read_h5ad(str(runs_df.iloc[0]["pseudo_control_h5ad"]), backed="r")
        shared = shared.intersection(pd.Index(ad.var_names.astype(str)))
        ad.file.close()

    if max_eval_genes is None:
        return pd.Index(shared)
    if "highly_variable" in control.var.columns:
        hvg_mask = control.var.loc[shared, "highly_variable"].astype(bool)
        hvg_genes = shared[hvg_mask.values]
        if len(hvg_genes) > 0:
            if "highly_variable_rank" in control.var.columns:
                ranked = control.var.loc[hvg_genes, ["highly_variable_rank"]].sort_values("highly_variable_rank")
                return pd.Index(ranked.index[:max_eval_genes].astype(str))
            return pd.Index(hvg_genes[:max_eval_genes].astype(str))
    return pd.Index(shared[:max_eval_genes].astype(str))


def load_pseudo_matrix_aligned(row: pd.Series, perturbed, eval_genes: Sequence[str], require_full_alignment: bool = True) -> sp.csr_matrix:
    pseudo_path = Path(row["pseudo_control_h5ad"])
    pseudo = sc.read_h5ad(pseudo_path)
    pseudo.obs_names_make_unique()
    if require_full_alignment:
        if pseudo.n_obs != perturbed.n_obs:
            raise ValueError(f"n_obs mismatch: pseudo={pseudo.n_obs}, perturbed={perturbed.n_obs}, file={pseudo_path}")
        if not np.array_equal(pseudo.obs_names.astype(str), perturbed.obs_names.astype(str)):
            raise ValueError(f"obs_names are not aligned to perturbed for {pseudo_path}")
    missing = pd.Index(eval_genes).difference(pd.Index(pseudo.var_names.astype(str)))
    if len(missing) > 0:
        raise ValueError(f"{len(missing)} eval genes missing in {pseudo_path}; examples: {missing[:5].tolist()}")
    X = to_csr(pseudo[:, eval_genes].X)
    return X


def metadata_from_manifest_row(row: pd.Series) -> dict[str, Any]:
    keys = [
        "run_id", "dataset_id", "perturbed_group", "strategy", "strategy_id", "strategy_family",
        "seacell_setting_id", "n_metacells_requested", "n_metacells_observed",
        "top_k_metacells", "n_control_cells_to_average", "n_metacells_to_average",
        "sample_cells_per_metacell", "sampling_seed", "parameter_label", "pseudo_control_h5ad",
        "pair_metadata_path", "membership_path_for_metacell_coverage",
    ]
    return {k: row.get(k, np.nan) for k in keys if k in row.index}


def add_metadata_columns(df: pd.DataFrame, metadata: Mapping[str, Any], insert_at: int = 0) -> pd.DataFrame:
    out = df.copy()
    for key, val in reversed(list(metadata.items())):
        if key not in out.columns:
            out.insert(insert_at, key, val)
    return out
