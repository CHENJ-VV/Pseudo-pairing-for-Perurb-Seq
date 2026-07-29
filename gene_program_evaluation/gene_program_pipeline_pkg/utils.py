from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expand_shell_defaults(value: str) -> str:
    """Expand ${NAME:-default} in addition to variables supported by expandvars."""
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):-([^}]*)\}")

    def replace(match: re.Match[str]) -> str:
        current = os.environ.get(match.group(1))
        return current if current not in {None, ""} else match.group(2)

    previous = value
    while True:
        expanded = pattern.sub(replace, previous)
        if expanded == previous:
            return expanded
        previous = expanded


def resolve_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Expand environment/user tokens and resolve relative paths against base_dir."""
    expanded = os.path.expandvars(_expand_shell_defaults(str(value)))
    path = Path(expanded).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path.resolve(strict=False)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON config and retain its directory for relative path resolution."""
    config_path = resolve_path(path)
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("The configuration root must be a JSON object")
    config.setdefault("global", {})
    if not isinstance(config["global"], dict):
        raise ValueError("config.global must be a JSON object")
    config["global"]["_config_dir"] = str(config_path.parent)
    config["global"]["_config_path"] = str(config_path)
    return config


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (float,)) and not math.isfinite(obj):
        return None
    return obj


def save_json(obj: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(to_jsonable(obj), f, indent=2)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception as exc:
            warnings.warn(f"Could not write parquet to {path}: {exc!r}; writing CSV instead.")
            path = path.with_suffix(".csv")
    if suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    else:
        df.to_csv(path, index=False)
    return path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def first_existing_path(paths: Sequence[str | Path]) -> Path | None:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def get_expr_matrix(adata: Any, layer: str | None = None):
    if layer is None or str(layer).lower() in {"none", "x"}:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} not found in AnnData.layers. Available: {list(adata.layers.keys())}")
    return adata.layers[layer]


def as_1d_array(x: Any, dtype=np.float64) -> np.ndarray:
    if sparse.issparse(x):
        x = x.A
    arr = np.asarray(x, dtype=dtype)
    return np.ravel(arr)


def column_mean(X: Any) -> np.ndarray:
    return as_1d_array(X.mean(axis=0), dtype=np.float64)


def column_mean_square(X: Any) -> np.ndarray:
    if sparse.issparse(X):
        return as_1d_array(X.multiply(X).mean(axis=0), dtype=np.float64)
    X = np.asarray(X)
    return np.mean(np.square(X), axis=0).astype(np.float64)


def column_nonzero_fraction(X: Any) -> np.ndarray:
    n = int(X.shape[0])
    if n == 0:
        return np.zeros(int(X.shape[1]), dtype=np.float64)
    if sparse.issparse(X):
        return np.asarray(X.getnnz(axis=0), dtype=np.float64) / float(n)
    return np.mean(np.asarray(X) != 0, axis=0).astype(np.float64)


def dense_rows(X: Any, row_idx: np.ndarray | Sequence[int], col_idx: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
    if col_idx is None:
        sub = X[row_idx, :]
    else:
        sub = X[row_idx, :][:, col_idx]
    if sparse.issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub)


def chunk_slices(n: int, chunk_size: int) -> Iterable[slice]:
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, int(n), chunk_size):
        yield slice(start, min(start + chunk_size, int(n)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    if np.allclose(x[mask], x[mask][0]) or np.allclose(y[mask], y[mask][0]):
        return np.nan
    return float(stats.spearmanr(x[mask], y[mask]).correlation)


def flatten_numeric(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float).ravel()


def normalize_group_name(group: str) -> str:
    return str(group).strip().lower()


def unique_preserve_order(values: Iterable[Any]) -> list[Any]:
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
