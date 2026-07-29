from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import safe_corr, safe_spearman


def align_effect_matrices(true_effects: pd.DataFrame, pseudo_effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_rows = true_effects.index.intersection(pseudo_effects.index)
    common_cols = true_effects.columns.intersection(pseudo_effects.columns)
    t = true_effects.loc[common_rows, common_cols].sort_index(axis=0).sort_index(axis=1)
    p = pseudo_effects.loc[common_rows, common_cols].sort_index(axis=0).sort_index(axis=1)
    return t, p


def matrix_metrics(true_effects: pd.DataFrame, pseudo_effects: pd.DataFrame) -> dict[str, Any]:
    t, p = align_effect_matrices(true_effects, pseudo_effects)
    x = t.to_numpy(dtype=float).ravel()
    y = p.to_numpy(dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return {
            "n_perturbations_eval": int(t.shape[0]),
            "n_programs_eval": int(t.shape[1]),
            "n_values_eval": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "pearson": np.nan,
            "spearman": np.nan,
            "magnitude_ratio": np.nan,
            "cosine_similarity": np.nan,
        }
    diff = y[mask] - x[mask]
    norm_true = float(np.linalg.norm(x[mask]))
    norm_pseudo = float(np.linalg.norm(y[mask]))
    cosine = np.nan if norm_true == 0 or norm_pseudo == 0 else float(np.dot(x[mask], y[mask]) / (norm_true * norm_pseudo))
    return {
        "n_perturbations_eval": int(t.shape[0]),
        "n_programs_eval": int(t.shape[1]),
        "n_values_eval": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "mae": float(np.mean(np.abs(diff))),
        "pearson": safe_corr(x, y),
        "spearman": safe_spearman(x, y),
        "magnitude_ratio": np.nan if norm_true == 0 else float(norm_pseudo / norm_true),
        "cosine_similarity": cosine,
    }


def per_program_correlations(true_effects: pd.DataFrame, pseudo_effects: pd.DataFrame) -> pd.DataFrame:
    t, p = align_effect_matrices(true_effects, pseudo_effects)
    rows = []
    for prog in t.columns:
        rows.append({
            "program_id": prog,
            "n_perturbations": int(np.isfinite(t[prog].to_numpy(dtype=float) + p[prog].to_numpy(dtype=float)).sum()),
            "pearson": safe_corr(t[prog].to_numpy(dtype=float), p[prog].to_numpy(dtype=float)),
            "spearman": safe_spearman(t[prog].to_numpy(dtype=float), p[prog].to_numpy(dtype=float)),
            "rmse": float(np.sqrt(np.nanmean((p[prog].to_numpy(dtype=float) - t[prog].to_numpy(dtype=float)) ** 2))),
        })
    return pd.DataFrame(rows)


def per_perturbation_correlations(true_effects: pd.DataFrame, pseudo_effects: pd.DataFrame) -> pd.DataFrame:
    t, p = align_effect_matrices(true_effects, pseudo_effects)
    rows = []
    for pert in t.index:
        rows.append({
            "perturbation": pert,
            "n_programs": int(np.isfinite(t.loc[pert].to_numpy(dtype=float) + p.loc[pert].to_numpy(dtype=float)).sum()),
            "pearson": safe_corr(t.loc[pert].to_numpy(dtype=float), p.loc[pert].to_numpy(dtype=float)),
            "spearman": safe_spearman(t.loc[pert].to_numpy(dtype=float), p.loc[pert].to_numpy(dtype=float)),
            "rmse": float(np.sqrt(np.nanmean((p.loc[pert].to_numpy(dtype=float) - t.loc[pert].to_numpy(dtype=float)) ** 2))),
        })
    return pd.DataFrame(rows)


def summarize_correlation_table(df: pd.DataFrame, prefix: str) -> dict[str, Any]:
    out = {}
    for col in ["pearson", "spearman", "rmse"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out[f"{prefix}_{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            out[f"{prefix}_{col}_median"] = float(vals.median()) if vals.notna().any() else np.nan
            out[f"{prefix}_{col}_n_finite"] = int(vals.notna().sum())
    return out
