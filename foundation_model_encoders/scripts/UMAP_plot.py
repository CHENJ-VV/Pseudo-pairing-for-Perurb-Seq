# ============================================================
# Imports
# ============================================================
import os
import re
import gc
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
sc.settings.verbosity = 2

from runtime_config import (
    env_bool, env_int, env_list, env_optional_int, env_optional_str,
    env_path, env_str, prepend_repo,
)

# ============================================================
# User configuration
# ============================================================
PSEUDO_ROOT = env_path("PPFM_PSEUDO_ROOT", required=True)

TRUE_PERTURBED_H5AD = env_path("PPFM_TRUE_PERTURBED_H5AD", required=True)

# Set to None for auto-detection. The notebook first tries exact 'leiden',
# then any obs column containing 'leiden'.
LEIDEN_KEY: Optional[str] = env_optional_str("PPFM_LEIDEN_KEY", None)

# Empty list means all discovered variants/models.
# Example: ["S5_SEACell_OT_sampled_average__nmc_350__topk_05__seed_000"]
VARIANTS = env_list("PPFM_VARIANTS", [])
# VARIANTS = [
#     "S0_naive_mean_control_reference", 
#     "S1_random_single_control/seed_000", 
#     "S2_random_average_controls/k_100/seed_000",
#     "S3_SEACell_metacell_average/nmc_100/k_3/seed_000",
#     "S4_SEACell_balanced_random_sample/nmc_100/seed_000", 
#     "S5_SEACell_OT_sampled_average/nmc_350/topk_05/seed_000",
#     "S5_SEACell_OT_sampled_average/nmc_500/topk_05/seed_000",
# ]

# Empty list means all discovered models.
# Supported detected model names include: geneformer, scgpt, sccello,
# scfoundation, scimilarity, hvg_pca, scvi.
MODELS: List[str] = env_list("PPFM_MODELS", [])

OUT_ROOT = env_path("PPFM_OUTPUT_ROOT", PSEUDO_ROOT / "_embedding_umaps")
FIG_DIR = OUT_ROOT / "figures"
COORD_DIR = OUT_ROOT / "umap_coordinates"
MANIFEST_DIR = OUT_ROOT / "manifests"
for d in [FIG_DIR, COORD_DIR, MANIFEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# UMAP settings. For full Replogle-scale embeddings, use subsampling first.
# Set SUBSAMPLE_N_CELLS = None to use all cells.
SUBSAMPLE_N_CELLS: Optional[int] = env_optional_int("PPFM_SUBSAMPLE_N_CELLS", 50000)
SUBSAMPLE_RANDOM_STATE = 0
N_NEIGHBORS = 15
MIN_DIST = 0.5
RANDOM_STATE = 0

# If embedding dimensionality is high, PCA before neighbors makes UMAP much faster.
USE_PCA_BEFORE_UMAP = True
N_PCS_FOR_UMAP = 50

# Plot settings
DPI = 300
POINT_SIZE = 4
SAVE_PDF = True
SAVE_PNG = True

# If True, skip UMAPs that already have both PNG and coordinates.
SKIP_EXISTING = not env_bool("PPFM_OVERWRITE", False)

print(f"PSEUDO_ROOT: {PSEUDO_ROOT}")
print(f"TRUE_PERTURBED_H5AD: {TRUE_PERTURBED_H5AD}")
print(f"OUT_ROOT: {OUT_ROOT}")

# ============================================================
# Path and label checks
# ============================================================
required_paths = {
    "PSEUDO_ROOT": PSEUDO_ROOT,
    "TRUE_PERTURBED_H5AD": TRUE_PERTURBED_H5AD,
}
for name, path in required_paths.items():
    print(f"{name}: {path} | exists={path.exists()}")

if not PSEUDO_ROOT.exists():
    raise FileNotFoundError(f"PSEUDO_ROOT does not exist: {PSEUDO_ROOT}")
if not TRUE_PERTURBED_H5AD.exists():
    raise FileNotFoundError(f"TRUE_PERTURBED_H5AD does not exist: {TRUE_PERTURBED_H5AD}")

# ============================================================
# Helper functions
# ============================================================
def safe_name(x: str) -> str:
    x = str(x)
    x = x.replace(os.sep, "__")
    x = re.sub(r"[^A-Za-z0-9_.=+\-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x or "unnamed"


def infer_model_from_embedding_path(path: Path) -> Optional[str]:
    name = path.name
    parts = set(path.parts)
    stem = path.stem

    if name == "X_geneformer.npy" or "geneformer" in parts:
        return "geneformer"
    if name in {"X_scGPT.npy", "scgpt_embeddings.npy"} or "_scgpt_embeddings" in parts:
        return "scgpt"
    if name == "X_scCello.npy" or "sccello" in str(path).lower():
        return "sccello"
    if name == "X_scfoundation.npy" or "scfoundation" in str(path).lower():
        return "scfoundation"
    if name == "X_scimilarity.npy" or "scimilarity" in str(path).lower():
        return "scimilarity"
    if name == "X_pca_hvg.npy" or stem == "X_pca_hvg":
        return "hvg_pca"
    if name == "X_scVI.npy" or stem == "X_scVI":
        return "scvi"
    return None


def infer_variant_from_embedding_path(path: Path, model: str) -> str:
    # Common layouts:
    #   .../embeddings/<variant>/geneformer/X_geneformer.npy
    #   .../embeddings/<variant>/X_scCello.npy
    #   .../hvg_pca/embeddings/<variant>/X_pca_hvg.npy
    parent = path.parent
    if parent.name in {"geneformer", "gfcab", "scgpt", "sccello", "scfoundation", "scimilarity"}:
        return parent.parent.name
    return parent.name


def discover_embedding_files(pseudo_root: Path) -> pd.DataFrame:
    candidates = []
    for p in pseudo_root.rglob("*.npy"):
        s = str(p)
        if "/embedding_chunks/" in s or "/token_chunks/" in s:
            continue
        if "/embeddings/" not in s:
            continue
        model = infer_model_from_embedding_path(p)
        if model is None:
            continue
        variant = infer_variant_from_embedding_path(p, model)
        candidates.append({
            "model": model,
            "variant": variant,
            "embedding_path": str(p),
        })
    df = pd.DataFrame(candidates).drop_duplicates()
    if df.empty:
        return df
    return df.sort_values(["variant", "model", "embedding_path"]).reset_index(drop=True)


def detect_leiden_key(obs: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested is not None:
        if requested not in obs.columns:
            raise KeyError(
                f"Requested LEIDEN_KEY='{requested}' not found. Available obs columns include: {list(obs.columns)[:30]}"
            )
        return requested
    if "leiden" in obs.columns:
        return "leiden"
    candidates = [c for c in obs.columns if "leiden" in c.lower()]
    if candidates:
        return sorted(candidates)[0]
    cluster_candidates = [c for c in obs.columns if "cluster" in c.lower()]
    if cluster_candidates:
        print("[Warn] No Leiden column found; using first cluster-like column instead.")
        return sorted(cluster_candidates)[0]
    raise KeyError(
        "Could not find a Leiden/cluster label in perturbed adata.obs. "
        "Set LEIDEN_KEY manually after inspecting perturbed.obs.columns."
    )


def load_obs_metadata_for_embedding(embedding_path: Path, n_rows: int) -> pd.DataFrame:
    # Search common metadata files in same folder.
    folder = embedding_path.parent
    candidates = [
        folder / "obs_metadata.csv",
        folder / "obs_names.csv",
        folder / "geneformer_obs.csv",
        folder / "scgpt_obs.csv",
        folder / "sccello_obs.csv",
    ]
    for c in candidates:
        if c.exists():
            df = pd.read_csv(c)
            if len(df) != n_rows:
                print(f"[Warn] Metadata row count mismatch for {c}: {len(df)} vs embedding {n_rows}")
            return df
    # fallback empty metadata; later we align positionally
    return pd.DataFrame(index=np.arange(n_rows))


def choose_cell_id_column(meta: pd.DataFrame) -> Optional[str]:
    candidates = [
        "source_obs_name",
        "adata_order",
        "obs_name",
        "obs_names",
        "cell_id",
        "cell",
        "index",
    ]
    for c in candidates:
        if c in meta.columns:
            return c
    if len(meta.columns) == 1:
        return meta.columns[0]
    return None


def align_labels_to_embedding(
    meta: pd.DataFrame,
    pert_obs: pd.DataFrame,
    label_key: str,
    n_rows: int,
) -> Tuple[pd.DataFrame, str]:
    label_series = pert_obs[label_key].astype(str)
    cell_col = choose_cell_id_column(meta)

    if cell_col is not None:
        ids = meta[cell_col].astype(str).values
        if np.isin(ids, pert_obs.index.astype(str)).mean() > 0.5:
            obs = pd.DataFrame(index=pd.Index(ids, name="cell_id"))
            obs[label_key] = label_series.reindex(obs.index).values
            obs["source_obs_name"] = obs.index.astype(str)
            missing = obs[label_key].isna().sum()
            if missing > 0:
                print(f"[Warn] {missing} cells could not be matched by '{cell_col}' and will be dropped.")
                keep = ~obs[label_key].isna()
                return obs.loc[keep].copy(), f"matched_by_{cell_col}"
            return obs, f"matched_by_{cell_col}"

    # Positional fallback. This is valid for debug subsets when embeddings were generated
    # from the first N cells in the pseudo-control/perturbed-aligned h5ad.
    if n_rows > pert_obs.shape[0]:
        raise ValueError(
            f"Cannot use positional alignment: embedding has {n_rows} rows but perturbed obs has {pert_obs.shape[0]} rows."
        )
    obs = pd.DataFrame(index=pert_obs.index[:n_rows].astype(str))
    obs[label_key] = label_series.iloc[:n_rows].values
    obs["source_obs_name"] = obs.index.astype(str)
    return obs, "positional_first_n"


def maybe_subsample(emb: np.ndarray, obs: pd.DataFrame, n: Optional[int], seed: int):
    if n is None or emb.shape[0] <= n:
        return emb, obs, np.arange(emb.shape[0])
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(emb.shape[0], size=n, replace=False))
    return emb[idx], obs.iloc[idx].copy(), idx

# ============================================================
# Load perturbed obs and discover embeddings
# ============================================================
print(f"Reading perturbed h5ad metadata: {TRUE_PERTURBED_H5AD}")
pert = sc.read_h5ad(TRUE_PERTURBED_H5AD, backed="r")
pert_obs = pert.obs.copy()
try:
    pert.file.close()
except Exception:
    pass

label_key = detect_leiden_key(pert_obs, LEIDEN_KEY)
print(f"Using color label from perturbed data: {label_key}")
print(pert_obs[label_key].astype(str).value_counts().head())

emb_df = discover_embedding_files(PSEUDO_ROOT)
if emb_df.empty:
    raise RuntimeError(f"No external embedding .npy files found under {PSEUDO_ROOT}")

if VARIANTS:
    VARIANTS = [var.replace("/", "__") for var in VARIANTS]
    variant_set = set(VARIANTS)
    emb_df = emb_df[emb_df["variant"].isin(variant_set)].copy()
if MODELS:
    model_set = set(MODELS)
    emb_df = emb_df[emb_df["model"].isin(model_set)].copy()

if emb_df.empty:
    raise RuntimeError("No embeddings remain after applying VARIANTS/MODELS filters.")

print(f"Discovered {len(emb_df)} embedding files to plot.")

# ============================================================
# UMAP plotting function
# ============================================================
def plot_one_embedding_umap(row: pd.Series) -> Dict:
    model = row["model"]
    variant = row["variant"]
    embedding_path = Path(row["embedding_path"])

    model_safe = safe_name(model)
    variant_safe = safe_name(variant)
    label_safe = safe_name(label_key)
    prefix = f"{variant_safe}__{model_safe}__umap_by_{label_safe}"
    png_path = FIG_DIR / f"{prefix}.png"
    pdf_path = FIG_DIR / f"{prefix}.pdf"
    coord_path = COORD_DIR / f"{prefix}_coords.csv"

    if SKIP_EXISTING and coord_path.exists() and (not SAVE_PNG or png_path.exists()) and (not SAVE_PDF or pdf_path.exists()):
        print(f"[Skip] Existing UMAP: {prefix}")
        return {
            "status": "skipped_existing",
            "model": model,
            "variant": variant,
            "embedding_path": str(embedding_path),
            "png_path": str(png_path) if png_path.exists() else None,
            "pdf_path": str(pdf_path) if pdf_path.exists() else None,
            "coord_path": str(coord_path),
        }

    print("=" * 100)
    print(f"[UMAP] Model:   {model}")
    print(f"[UMAP] Variant: {variant}")
    print(f"[UMAP] Emb:     {embedding_path}")

    emb = np.load(embedding_path)
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding matrix, got shape {emb.shape}: {embedding_path}")
    emb = np.asarray(emb, dtype=np.float32)

    meta = load_obs_metadata_for_embedding(embedding_path, emb.shape[0])
    obs, align_mode = align_labels_to_embedding(meta, pert_obs, label_key, emb.shape[0])

    # If label matching dropped cells, apply the same filtering to embeddings.
    if obs.shape[0] != emb.shape[0]:
        if align_mode.startswith("matched_by_"):
            # Reconstruct keep mask from metadata ID membership.
            cell_col = choose_cell_id_column(meta)
            ids = meta[cell_col].astype(str).values
            keep = pd.Index(ids).isin(obs.index)
            emb = emb[keep]
        else:
            emb = emb[: obs.shape[0]]

    emb, obs, subset_idx = maybe_subsample(
        emb=emb,
        obs=obs,
        n=SUBSAMPLE_N_CELLS,
        seed=SUBSAMPLE_RANDOM_STATE,
    )

    print(f"[UMAP] Alignment: {align_mode}")
    print(f"[UMAP] Final cells: {emb.shape[0]}, dims: {emb.shape[1]}")

    adata_plot = ad.AnnData(X=emb, obs=obs.copy())
    adata_plot.obs[label_key] = adata_plot.obs[label_key].astype(str).astype("category")

    use_rep = None
    if USE_PCA_BEFORE_UMAP and emb.shape[1] > N_PCS_FOR_UMAP and emb.shape[0] > N_PCS_FOR_UMAP + 1:
        n_comps = min(N_PCS_FOR_UMAP, emb.shape[0] - 1, emb.shape[1])
        sc.tl.pca(adata_plot, n_comps=n_comps, svd_solver="arpack")
        use_rep = "X_pca"
    sc.pp.neighbors(
        adata_plot,
        n_neighbors=N_NEIGHBORS,
        use_rep=use_rep,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata_plot, min_dist=MIN_DIST, random_state=RANDOM_STATE)

    coord_df = pd.DataFrame({
        "cell_id": adata_plot.obs.index.astype(str),
        "UMAP1": adata_plot.obsm["X_umap"][:, 0],
        "UMAP2": adata_plot.obsm["X_umap"][:, 1],
        label_key: adata_plot.obs[label_key].astype(str).values,
        "model": model,
        "variant": variant,
        "embedding_path": str(embedding_path),
        "alignment_mode": align_mode,
    })
    coord_df.to_csv(coord_path, index=False)

    # Plot with Scanpy for clean categorical legends.
    fig = sc.pl.umap(
        adata_plot,
        color=label_key,
        show=False,
        size=POINT_SIZE,
        frameon=False,
        title=f"{model} | {variant}",
        return_fig=True,
    )
    if SAVE_PNG:
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    n_clusters = int(adata_plot.obs[label_key].nunique())
    result = {
        "status": "plotted",
        "model": model,
        "variant": variant,
        "embedding_path": str(embedding_path),
        "n_cells_embedding": int(np.load(embedding_path, mmap_mode="r").shape[0]),
        "n_cells_plotted": int(adata_plot.n_obs),
        "embedding_dim": int(adata_plot.X.shape[1]),
        "label_key": label_key,
        "n_label_categories": n_clusters,
        "alignment_mode": align_mode,
        "png_path": str(png_path) if SAVE_PNG else None,
        "pdf_path": str(pdf_path) if SAVE_PDF else None,
        "coord_path": str(coord_path),
    }

    del emb, meta, obs, adata_plot, coord_df
    gc.collect()
    return result
# ============================================================
# Run UMAP plotting for all discovered embeddings
# ============================================================
results = []
for _, row in tqdm(emb_df.iterrows(), total=len(emb_df), desc="Plot UMAPs"):
    try:
        res = plot_one_embedding_umap(row)
    except Exception as e:
        print(f"[Error] Failed for {row['model']} / {row['variant']}: {e}")
        res = {
            "status": "failed",
            "model": row["model"],
            "variant": row["variant"],
            "embedding_path": row["embedding_path"],
            "error": repr(e),
        }
    results.append(res)

umap_manifest = pd.DataFrame(results)
umap_manifest_path = MANIFEST_DIR / "embedding_umap_manifest.csv"
umap_manifest.to_csv(umap_manifest_path, index=False)
print(f"Saved UMAP manifest: {umap_manifest_path}")

