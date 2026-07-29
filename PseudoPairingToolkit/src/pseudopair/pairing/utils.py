"""Shared utilities for dataset-agnostic pseudo-control pairing workflows.

The functions here avoid Replogle-specific assumptions.  They operate on any
AnnData pair with a control h5ad and a perturbed h5ad, provided the expression
matrix and optional embeddings are available under configurable keys.
"""
from __future__ import annotations

import gc
import h5py
import json
import random
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from tqdm.auto import tqdm


STRATEGY_ORDER = [
    "S0_naive_mean_control_reference",
    "S1_random_single_control",
    "S2_random_average_controls",
    "S3_SEACell_metacell_average",
    "S4_SEACell_balanced_random_sample",
    "S5_SEACell_OT_sampled_average",
]

PERTURBATION_KEY_CANDIDATES = [
    "perturbation_key",
    "perturbation_label",
    "condition",
    "gene",
    "target_gene",
    "guide_identity",
    "perturbation",
    "gene_symbol",
]


def as_namespace(config: Mapping[str, Any] | SimpleNamespace) -> SimpleNamespace:
    """Return a SimpleNamespace with recursively JSON-like fields preserved."""
    if isinstance(config, SimpleNamespace):
        return config
    return SimpleNamespace(**dict(config))


def to_jsonable(obj: Any) -> Any:
    """Convert common scientific Python objects into JSON-serializable objects."""
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
        return [to_jsonable(v) for v in obj]
    return obj


def namespace_to_dict(args: SimpleNamespace) -> dict[str, Any]:
    return {k: to_jsonable(v) for k, v in vars(args).items()}


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def read_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def set_seed(seed: int) -> np.random.Generator:
    """Set common random seeds and return a NumPy Generator."""
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


def ensure_outdir(outdir: str | Path) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_dataframe(df: pd.DataFrame, path_prefix: str | Path) -> Path:
    """Save as parquet if possible; otherwise save CSV using the same prefix."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    if path_prefix.suffix in {".parquet", ".csv"}:
        stem = path_prefix.with_suffix("")
    else:
        stem = path_prefix
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
    if path_prefix.exists():
        return True
    return path_prefix.with_suffix(".parquet").exists() or path_prefix.with_suffix(".csv").exists()


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


def get_expr_matrix(adata: AnnData, expr_layer: str = "X"):
    """Return adata.X or adata.layers[expr_layer] as CSR or float32 ndarray."""
    if expr_layer == "X":
        X = adata.X
    else:
        if expr_layer not in adata.layers:
            raise KeyError(
                f"Layer '{expr_layer}' was not found. Available layers: {list(adata.layers.keys())}"
            )
        X = adata.layers[expr_layer]
    if sp.issparse(X):
        return X.tocsr().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def row_mean(X, row_indices: np.ndarray) -> np.ndarray:
    if len(row_indices) == 0:
        raise ValueError("row_indices is empty.")
    sub = X[row_indices]
    if sp.issparse(sub):
        return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(sub.mean(axis=0)).ravel().astype(np.float32)


def format_array_as_string(values: Iterable, precision: int = 6) -> str:
    out = []
    for value in values:
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, (np.integer, int)):
            out.append(str(int(value)))
        else:
            out.append(f"{float(value):.{precision}g}")
    return "|".join(out)


def infer_perturbation_key(obs: pd.DataFrame, requested_key: str | None = None) -> str:
    """Infer a perturbation-label column while allowing explicit override."""
    if requested_key is not None and str(requested_key) != "auto":
        if requested_key not in obs.columns:
            raise KeyError(
                f"Requested perturbation_key='{requested_key}' is absent. Available obs columns: {list(obs.columns)}"
            )
        return requested_key
    for key in PERTURBATION_KEY_CANDIDATES:
        if key in obs.columns:
            return key
    raise KeyError(
        "Could not infer perturbation_key. Set config['perturbation_key'] explicitly. "
        f"Available obs columns: {list(obs.columns)}"
    )


def load_perturbed_metadata(
    perturbed_h5ad: str | Path,
    perturbation_key: str | None = None,
    max_pairs_per_perturbation: int | None = None,
    pair_selection_seed: int = 42,
) -> dict[str, Any]:
    """Load metadata from a perturbed h5ad, optionally selecting a capped subset per perturbation."""
    perturbed = sc.read_h5ad(str(perturbed_h5ad), backed="r")
    obs_all = perturbed.obs.copy()
    var = perturbed.var.copy()
    obs_names_all = pd.Index(perturbed.obs_names.astype(str))
    var_names = pd.Index(perturbed.var_names.astype(str))
    n_total = int(perturbed.n_obs)
    n_vars = int(perturbed.n_vars)
    key = infer_perturbation_key(obs_all, perturbation_key)

    if max_pairs_per_perturbation is None:
        selected_positions = np.arange(n_total, dtype=np.int64)
    else:
        rng = set_seed(pair_selection_seed)
        selected = []
        values = obs_all[key].astype(str).values
        for pert_name in pd.Index(values).unique().astype(str):
            idx = np.where(values == pert_name)[0]
            if len(idx) > int(max_pairs_per_perturbation):
                idx = rng.choice(idx, size=int(max_pairs_per_perturbation), replace=False)
            selected.append(np.sort(idx).astype(np.int64))
        selected_positions = np.sort(np.concatenate(selected)) if selected else np.array([], dtype=np.int64)

    obs = obs_all.iloc[selected_positions].copy()
    obs_names = obs_names_all[selected_positions]
    perturbed.file.close()
    return {
        "obs_all": obs_all,
        "obs": obs,
        "var": var,
        "obs_names_all": obs_names_all,
        "obs_names": obs_names,
        "var_names": var_names,
        "n_obs_total": n_total,
        "n_obs_selected": int(len(selected_positions)),
        "n_vars": n_vars,
        "selected_positions": selected_positions,
        "perturbation_key": key,
    }


def align_control_to_perturbed_genes(
    control: AnnData,
    perturbed_var_names: Sequence[str],
    perturbed_var: pd.DataFrame | None = None,
    require_all_genes: bool = False,
) -> tuple[AnnData, pd.Index, pd.DataFrame | None]:
    """Align control genes to the perturbed gene order, using common genes if needed."""
    perturbed_var_names = pd.Index(perturbed_var_names).astype(str)
    control_var_names = pd.Index(control.var_names).astype(str)
    if np.array_equal(control_var_names.values, perturbed_var_names.values):
        output_var = None if perturbed_var is None else perturbed_var.copy()
        return control, perturbed_var_names, output_var

    common = perturbed_var_names.intersection(control_var_names)
    if len(common) == 0:
        raise ValueError("No shared genes between control and perturbed AnnData objects.")
    if require_all_genes and len(common) != len(perturbed_var_names):
        missing = perturbed_var_names.difference(control_var_names)[:10].tolist()
        raise ValueError(
            f"Control is missing {len(perturbed_var_names) - len(common)} perturbed genes. Examples: {missing}"
        )
    warnings.warn(
        f"Gene sets/order differ. Aligning control and output to {len(common)} common genes in perturbed order."
    )
    control_aligned = control[:, common].copy()
    output_var = None if perturbed_var is None else perturbed_var.loc[common].copy()
    return control_aligned, common, output_var


def align_control_and_perturbed(control: AnnData, perturbed: AnnData, require_all_genes: bool = False):
    """Align two in-memory AnnData objects to the common genes in perturbed order."""
    if np.array_equal(control.var_names.astype(str), perturbed.var_names.astype(str)):
        return control, perturbed
    common = pd.Index(perturbed.var_names.astype(str)).intersection(pd.Index(control.var_names.astype(str)))
    if len(common) == 0:
        raise ValueError("No shared genes between control and perturbed AnnData objects.")
    if require_all_genes and len(common) != perturbed.n_vars:
        raise ValueError("Gene sets differ and require_all_genes=True.")
    warnings.warn(f"Gene sets/order differ. Aligning to {len(common)} common genes.")
    return control[:, common].copy(), perturbed[:, common].copy()


def build_output_obs(
    pert_obs: pd.DataFrame,
    pert_obs_names: Sequence[str],
    selected_positions: np.ndarray,
) -> pd.DataFrame:
    obs = pert_obs.copy()
    obs.index = pd.Index(pert_obs_names).astype(str)
    obs["pseudo_control_available"] = True
    obs["paired_perturbed_cell_id"] = pd.Index(pert_obs_names).astype(str).values
    obs["paired_perturbed_cell_pos"] = selected_positions.astype(int)
    return obs


def make_pair_ids(n: int) -> list[str]:
    return [f"pair_{i:012d}" for i in range(int(n))]


def basic_pair_metadata(
    pert_meta: Mapping[str, Any],
    strategy: str,
) -> pd.DataFrame:
    key = pert_meta["perturbation_key"]
    obs = pert_meta["obs"]
    return pd.DataFrame(
        {
            "pair_id": make_pair_ids(pert_meta["n_obs_selected"]),
            "perturbed_cell_id": pd.Index(pert_meta["obs_names"]).astype(str).values,
            "perturbed_cell_pos": pert_meta["selected_positions"].astype(int),
            "perturbation": obs[key].astype(str).values if key in obs.columns else "NA",
            "pairing_strategy": strategy,
        }
    )


def write_pseudo_control_h5ad(
    X_pseudo,
    pert_meta: Mapping[str, Any],
    output_var: pd.DataFrame,
    outdir: str | Path,
    strategy: str,
    pair_metadata: pd.DataFrame,
    extra_uns: Mapping[str, Any] | None = None,
    extra_obs_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write a pseudo-control AnnData and companion metadata files."""
    outdir = ensure_outdir(outdir)
    obs = build_output_obs(
        pert_meta["obs"],
        pert_meta["obs_names"],
        pert_meta["selected_positions"],
    )
    if extra_obs_cols:
        # Align by current row order in pair_metadata.
        for col in extra_obs_cols:
            if col in pair_metadata.columns:
                obs[f"pairing_{col}"] = pair_metadata[col].values
    obs["pairing_strategy"] = strategy

    pseudo = AnnData(X=X_pseudo, obs=obs, var=output_var.copy())
    pseudo.uns["pairing_strategy"] = strategy
    pseudo.uns["row_alignment"] = (
        "same_order_as_perturbed_h5ad" if pert_meta["n_obs_selected"] == pert_meta["n_obs_total"] else "subset_sorted_by_original_perturbed_cell_order"
    )
    pseudo.uns["n_perturbed_cells_total"] = int(pert_meta["n_obs_total"])
    pseudo.uns["n_perturbed_cells_selected"] = int(pert_meta["n_obs_selected"])
    if extra_uns:
        for k, v in extra_uns.items():
            pseudo.uns[k] = to_jsonable(v)

    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"
    print(f"[Save] Writing h5ad: {output_h5ad}")
    pseudo.write_h5ad(output_h5ad)
    pair_metadata_path = save_dataframe(pair_metadata, outdir / "pair_metadata")
    return {
        "pseudo_control": pseudo,
        "output_h5ad": output_h5ad,
        "pair_metadata_path": pair_metadata_path,
        "outdir": outdir,
    }


def build_random_single_matrix(X_source, selected_pos: np.ndarray, batch_size: int = 4096) -> sp.csr_matrix:
    chunks = []
    for start in tqdm(range(0, len(selected_pos), int(batch_size)), desc="Building sampled rows"):
        end = min(start + int(batch_size), len(selected_pos))
        X_chunk = X_source[selected_pos[start:end], :]
        if sp.issparse(X_chunk):
            X_chunk = X_chunk.tocsr().astype(np.float32)
        else:
            X_chunk = sp.csr_matrix(np.asarray(X_chunk, dtype=np.float32))
        chunks.append(X_chunk)
    return sp.vstack(chunks, format="csr") if chunks else sp.csr_matrix((0, X_source.shape[1]))


def build_average_by_index_matrix(X_source, sampled_pos_matrix: np.ndarray, batch_size: int = 4096) -> sp.csr_matrix:
    """For each row i, average X_source[sampled_pos_matrix[i, :], :]."""
    n_rows, k = sampled_pos_matrix.shape
    chunks = []
    for start in tqdm(range(0, n_rows, int(batch_size)), desc="Building averaged rows"):
        end = min(start + int(batch_size), n_rows)
        pos_batch = sampled_pos_matrix[start:end]
        n_batch = pos_batch.shape[0]
        flat_pos = pos_batch.reshape(-1)
        X_flat = X_source[flat_pos, :]
        if sp.issparse(X_flat):
            X_flat = X_flat.tocsr()
            row_ids = np.repeat(np.arange(n_batch), k)
            col_ids = np.arange(n_batch * k)
            data = np.full(n_batch * k, 1.0 / k, dtype=np.float32)
            A = sp.csr_matrix((data, (row_ids, col_ids)), shape=(n_batch, n_batch * k))
            X_avg = (A @ X_flat).tocsr().astype(np.float32)
        else:
            X_flat = np.asarray(X_flat, dtype=np.float32)
            X_avg = sp.csr_matrix(X_flat.reshape(n_batch, k, X_flat.shape[1]).mean(axis=1))
        chunks.append(X_avg)
    return sp.vstack(chunks, format="csr") if chunks else sp.csr_matrix((0, X_source.shape[1]))


def aggregate_by_groups(X, groups: Sequence[np.ndarray], mode: str = "mean") -> sp.csr_matrix:
    rows = []
    for idx in tqdm(groups, desc=f"Aggregating groups by {mode}"):
        sub = X[idx]
        if mode == "mean":
            row = np.asarray(sub.mean(axis=0)).ravel()
        elif mode == "sum":
            row = np.asarray(sub.sum(axis=0)).ravel()
        else:
            raise ValueError("mode must be 'mean' or 'sum'.")
        rows.append(sp.csr_matrix(row.astype(np.float32)))
    return sp.vstack(rows, format="csr") if rows else sp.csr_matrix((0, X.shape[1]))


def sparse_usage_summary(indices: np.ndarray, id_values: Sequence[str], count_name: str) -> pd.DataFrame:
    flat = np.asarray(indices).reshape(-1)
    counts = pd.Series(flat).value_counts().sort_index()
    return pd.DataFrame(
        {
            "index": counts.index.astype(int),
            "id": [str(id_values[int(i)]) for i in counts.index.astype(int)],
            count_name: counts.values.astype(int),
        }
    )


def entropy_from_counts(counts: Sequence[int]) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def cleanup(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()

def write_repeated_mean_profile_h5ad(
    mean_profile: np.ndarray,
    pert_meta: dict,
    output_var: pd.DataFrame,
    outdir: str | Path,
    strategy: str,
    pair_metadata: pd.DataFrame,
    extra_uns: dict | None = None,
    extra_obs_cols: list[str] | None = None,
    chunk_rows: int = 512,
    chunk_cols: int = 2048,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
):
    """
    Write S0 pseudo-control h5ad without passing numpy.memmap to AnnData.

    S0 has the same global control mean profile repeated for every perturbed cell.
    This function writes the dense X matrix directly to the h5ad file in chunks.
    """

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"

    if output_h5ad.exists():
        output_h5ad.unlink()

    mean_profile = np.asarray(mean_profile, dtype=np.float32).ravel()

    n_obs = int(pert_meta["n_obs_selected"])
    n_obs_total = int(pert_meta["n_obs_total"])
    selected_positions = np.asarray(pert_meta["selected_positions"], dtype=int)
    n_vars = int(len(output_var))

    if mean_profile.shape[0] != n_vars:
        raise ValueError(
            f"mean_profile length {mean_profile.shape[0]} does not match n_vars={n_vars}"
        )

    if len(selected_positions) != n_obs:
        raise ValueError(
            f"selected_positions length {len(selected_positions)} does not match n_obs_selected={n_obs}"
        )

    obs = pert_meta["obs"].copy()
    obs.index = pd.Index(pert_meta["obs_names"]).astype(str)

    obs["pseudo_control_available"] = True
    obs["paired_perturbed_cell_id"] = pd.Index(pert_meta["obs_names"]).astype(str).values
    obs["paired_perturbed_cell_pos"] = selected_positions
    obs["pairing_strategy"] = strategy

    if extra_obs_cols is not None:
        pair_meta_indexed = pair_metadata.copy()

        if "perturbed_cell_pos" in pair_meta_indexed.columns:
            pair_meta_indexed = (
                pair_meta_indexed
                .set_index("perturbed_cell_pos")
                .loc[selected_positions]
                .reset_index()
            )

        for col in extra_obs_cols:
            if col in pair_meta_indexed.columns:
                obs[f"pairing_{col}"] = pair_meta_indexed[col].values

    pseudo = AnnData(
        X=None,
        obs=obs,
        var=output_var.copy(),
    )

    pseudo.uns["pairing_strategy"] = strategy
    pseudo.uns["row_alignment"] = (
        "same_order_as_perturbed_h5ad"
        if n_obs == n_obs_total and np.array_equal(selected_positions, np.arange(n_obs_total))
        else "subset_sorted_by_original_perturbed_cell_order"
    )
    pseudo.uns["n_perturbed_cells_total"] = n_obs_total
    pseudo.uns["n_perturbed_cells_selected"] = n_obs

    if extra_uns is not None:
        for k, v in extra_uns.items():
            pseudo.uns[k] = to_jsonable(v)

    print(f"[Save] Writing h5ad metadata shell: {output_h5ad}")
    pseudo.write_h5ad(output_h5ad)

    del pseudo
    gc.collect()

    print(f"[Save] Writing repeated mean X matrix in chunks: {n_obs:,} × {n_vars:,}")

    chunk_rows = min(int(chunk_rows), n_obs)
    chunk_cols = min(int(chunk_cols), n_vars)

    dataset_kwargs = dict(
        shape=(n_obs, n_vars),
        dtype="float32",
        chunks=(chunk_rows, chunk_cols),
    )

    if compression is not None:
        dataset_kwargs["compression"] = compression
        if compression_opts is not None:
            dataset_kwargs["compression_opts"] = compression_opts

    with h5py.File(output_h5ad, "a") as f:
        if "X" in f:
            del f["X"]

        ds = f.create_dataset("X", **dataset_kwargs)
        ds.attrs["encoding-type"] = "array"
        ds.attrs["encoding-version"] = "0.2.0"

        for start in range(0, n_obs, chunk_rows):
            end = min(start + chunk_rows, n_obs)

            block = np.empty((end - start, n_vars), dtype=np.float32)
            block[:, :] = mean_profile[None, :]

            ds[start:end, :] = block

            if start % (chunk_rows * 20) == 0:
                print(f"  wrote rows {start:,}–{end:,} / {n_obs:,}")

    pair_metadata_path = save_dataframe(pair_metadata, outdir / "pair_metadata")

    print(f"[Save] h5ad: {output_h5ad}")
    print(f"[Save] pair metadata: {pair_metadata_path}")

    return {
        "pseudo_control": None,
        "output_h5ad": output_h5ad,
        "pair_metadata_path": pair_metadata_path,
        "outdir": outdir,
    }

# def write_repeated_mean_profile_h5ad(
#     mean_profile: np.ndarray,
#     pert_meta: dict,
#     output_var: pd.DataFrame,
#     outdir: str | Path,
#     strategy: str,
#     pair_metadata: pd.DataFrame,
#     extra_uns: dict | None = None,
#     extra_obs_cols: list[str] | None = None,
#     chunk_rows: int = 512,
#     chunk_cols: int = 2048,
#     compression: str | None = "gzip",
#     compression_opts: int | None = 4,
# ):
#     """
#     Write S0 pseudo-control h5ad without passing numpy.memmap to AnnData.

#     S0 has the same global control mean profile repeated for every perturbed cell.
#     This function writes the dense X matrix directly to the h5ad file in chunks.

#     This avoids:
#         IORegistryError: No method registered for writing numpy.memmap
#     """

#     outdir = Path(outdir)
#     outdir.mkdir(parents=True, exist_ok=True)

#     output_h5ad = outdir / "pseudo_control_aligned_to_perturbed.h5ad"

#     # Remove incomplete/corrupted file from failed previous run.
#     if output_h5ad.exists():
#         output_h5ad.unlink()

#     mean_profile = np.asarray(mean_profile, dtype=np.float32).ravel()

#     n_obs = int(pert_meta["n_obs"])
#     n_vars = int(len(output_var))

#     if mean_profile.shape[0] != n_vars:
#         raise ValueError(
#             f"mean_profile length {mean_profile.shape[0]} does not match n_vars={n_vars}"
#         )

#     obs = pert_meta["obs"].copy()
#     obs.index = pert_meta["obs_names"]

#     obs["pseudo_control_available"] = True
#     obs["paired_perturbed_cell_id"] = pert_meta["obs_names"].astype(str)
#     obs["paired_perturbed_cell_pos"] = np.arange(n_obs, dtype=int)
#     obs["pairing_strategy"] = strategy

#     if extra_obs_cols is not None:
#         pair_meta_indexed = pair_metadata.copy()

#         if "perturbed_cell_pos" in pair_meta_indexed.columns:
#             pair_meta_indexed = (
#                 pair_meta_indexed
#                 .set_index("perturbed_cell_pos")
#                 .loc[np.arange(n_obs)]
#                 .reset_index()
#             )

#         for col in extra_obs_cols:
#             if col in pair_meta_indexed.columns:
#                 obs[col] = pair_meta_indexed[col].values

#     # Write AnnData shell first, with X=None.
#     # Then create X manually using h5py.
#     pseudo = AnnData(
#         X=None,
#         obs=obs,
#         var=output_var.copy(),
#     )

#     pseudo.uns["pairing_strategy"] = strategy
#     pseudo.uns["row_alignment"] = "same_order_as_perturbed_h5ad"

#     if extra_uns is not None:
#         for k, v in extra_uns.items():
#             pseudo.uns[k] = v

#     print(f"[Save] Writing h5ad metadata shell: {output_h5ad}")
#     pseudo.write_h5ad(output_h5ad)

#     del pseudo
#     gc.collect()

#     # Add dense X manually.
#     print(f"[Save] Writing repeated mean X matrix in chunks: {n_obs:,} × {n_vars:,}")

#     chunk_rows = min(int(chunk_rows), n_obs)
#     chunk_cols = min(int(chunk_cols), n_vars)

#     dataset_kwargs = dict(
#         shape=(n_obs, n_vars),
#         dtype="float32",
#         chunks=(chunk_rows, chunk_cols),
#     )

#     if compression is not None:
#         dataset_kwargs["compression"] = compression
#         if compression_opts is not None:
#             dataset_kwargs["compression_opts"] = compression_opts

#     with h5py.File(output_h5ad, "a") as f:
#         if "X" in f:
#             del f["X"]

#         ds = f.create_dataset("X", **dataset_kwargs)

#         # AnnData dense-array encoding metadata.
#         ds.attrs["encoding-type"] = "array"
#         ds.attrs["encoding-version"] = "0.2.0"

#         for start in range(0, n_obs, chunk_rows):
#             end = min(start + chunk_rows, n_obs)

#             block = np.empty((end - start, n_vars), dtype=np.float32)
#             block[:, :] = mean_profile[None, :]

#             ds[start:end, :] = block

#             if start % (chunk_rows * 20) == 0:
#                 print(f"  wrote rows {start:,}–{end:,} / {n_obs:,}")

#     pair_metadata_path = save_dataframe(pair_metadata, outdir / "pair_metadata")

#     print(f"[Save] h5ad: {output_h5ad}")
#     print(f"[Save] pair metadata: {pair_metadata_path}")

#     return {
#         "pseudo_control": None,
#         "output_h5ad": output_h5ad,
#         "pair_metadata_path": pair_metadata_path,
#         "outdir": outdir,
#     }