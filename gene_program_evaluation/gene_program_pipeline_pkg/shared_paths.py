from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .utils import as_bool_series, ensure_dir, first_existing_path, is_missing, read_json, read_table, resolve_path, save_json, save_table


STRATEGY_ORDER = [
    "S0_naive_mean_control_reference",
    "S1_random_single_control",
    "S2_random_average_controls",
    "S3_SEACell_metacell_average",
    "S4_SEACell_balanced_random_sample",
    "S5_SEACell_OT_sampled_average",
]
STRATEGY_ORDER_MAP = {s: i for i, s in enumerate(STRATEGY_ORDER)}

STRATEGY_PLOT_LABELS = {
    "S0_naive_mean_control_reference": "Naive mean control",
    "S1_random_single_control": "Random single control",
    "S2_random_average_controls": "Random average control",
    "S3_SEACell_metacell_average": "Random metacell average",
    "S4_SEACell_balanced_random_sample": "SEACell balanced sample",
    "S5_SEACell_OT_sampled_average": "OT sampled average",
}

STRATEGY_RENAME_MAP = {
    "S0": "S0_naive_mean_control_reference",
    "S1": "S1_random_single_control",
    "S2": "S2_random_average_controls",
    "S3": "S3_SEACell_metacell_average",
    "S4": "S4_SEACell_balanced_random_sample",
    "S5": "S5_SEACell_OT_sampled_average",
    "strategy0_naive_mean_control_reference": "S0_naive_mean_control_reference",
    "strategy1_random_single_control": "S1_random_single_control",
    "strategy2_random_average_controls": "S2_random_average_controls",
    "strategy3_random_average_controls": "S2_random_average_controls",
    "strategy1_seacell_balanced_random_repeated": "S4_SEACell_balanced_random_sample",
    "strategy2_seacell_ot_topk_sampled_average_repeated": "S5_SEACell_OT_sampled_average",
    **{s: s for s in STRATEGY_ORDER},
}


def fmt_param(x: Any) -> str:
    if is_missing(x):
        return "nan"
    try:
        f = float(x)
        if np.isfinite(f) and abs(f - round(f)) < 1e-8:
            return str(int(round(f)))
        return f"{f:g}"
    except Exception:
        return str(x)


def extract_number(x: Any, patterns: Sequence[str]) -> float:
    if is_missing(x):
        return np.nan
    text = str(x)
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return np.nan


def canonical_strategy_name(x: Any) -> str:
    if is_missing(x):
        return "unknown"
    s = str(x)
    return STRATEGY_RENAME_MAP.get(s, s)


def infer_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def fill(target: str, cols: Sequence[str]):
        if target not in out.columns:
            out[target] = np.nan
        out[target] = pd.to_numeric(out[target], errors="coerce")
        for c in cols:
            if c in out.columns:
                out[target] = out[target].where(out[target].notna(), pd.to_numeric(out[c], errors="coerce"))

    fill("sampling_seed", ["seed", "random_seed", "sampling_seed_for_plot"])
    fill("n_metacells", ["n_seacells", "n_SEACells", "n_metacells_observed"])
    fill("top_k", ["topk", "s5_top_k", "ot_top_k"])
    fill("n_control_cells_to_average", ["k", "n_controls", "n_control_cells", "n_cells_to_average"])
    fill("sampled_metacells_k", ["n_metacells_to_average", "s3_n_metacells_to_average", "sampled_metacells"])
    fill("sample_cells_per_metacell", ["cells_per_metacell", "sample_cells"])

    text_cols = [c for c in ["parameter_label", "variant_label", "display_variant_label", "outdir", "pseudo_control_h5ad"] if c in out.columns]
    for c in text_cols:
        text = out[c]
        if "n_metacells" in out.columns:
            out["n_metacells"] = out["n_metacells"].where(
                out["n_metacells"].notna(),
                text.map(lambda v: extract_number(v, [r"nmc[_=-]?(\d+)", r"metacells[_=-]?(\d+)", r"SEACell[_=-]?(\d+)"]))
            )
        if "top_k" in out.columns:
            out["top_k"] = out["top_k"].where(
                out["top_k"].notna(),
                text.map(lambda v: extract_number(v, [r"top[_-]?k[_=-]?(\d+)", r"topk[_=-]?(\d+)"]))
            )
        if "sampled_metacells_k" in out.columns:
            out["sampled_metacells_k"] = out["sampled_metacells_k"].where(
                out["sampled_metacells_k"].notna(),
                text.map(lambda v: extract_number(v, [r"sampled[_-]?metacells[_=-]?(\d+)", r"mcavg[_=-]?(\d+)", r"k[_=-]?(\d+)"]))
            )
    return out


def make_variant_id(row: Mapping[str, Any]) -> str:
    strategy = canonical_strategy_name(row.get("strategy", "unknown"))
    if strategy in {"S0_naive_mean_control_reference", "S1_random_single_control"}:
        return strategy
    if strategy == "S2_random_average_controls":
        k = row.get("n_control_cells_to_average", np.nan)
        return strategy if is_missing(k) else f"{strategy}__k_{fmt_param(k)}"
    if strategy == "S3_SEACell_metacell_average":
        nmc = row.get("n_metacells", np.nan)
        k = row.get("sampled_metacells_k", row.get("n_metacells_to_average", np.nan))
        parts = [strategy]
        if not is_missing(nmc):
            parts.append(f"nmc_{fmt_param(nmc)}")
        if not is_missing(k):
            parts.append(f"k_{fmt_param(k)}")
        return "__".join(parts)
    if strategy == "S4_SEACell_balanced_random_sample":
        nmc = row.get("n_metacells", np.nan)
        return strategy if is_missing(nmc) else f"{strategy}__nmc_{fmt_param(nmc)}"
    if strategy == "S5_SEACell_OT_sampled_average":
        nmc = row.get("n_metacells", np.nan)
        top_k = row.get("top_k", np.nan)
        spm = row.get("sample_cells_per_metacell", np.nan)
        parts = [strategy]
        if not is_missing(nmc):
            parts.append(f"nmc_{fmt_param(nmc)}")
        if not is_missing(top_k):
            parts.append(f"topk_{fmt_param(top_k)}")
        if not is_missing(spm):
            parts.append(f"spm_{fmt_param(spm)}")
        return "__".join(parts)
    return strategy


def make_display_variant_label(row: Mapping[str, Any]) -> str:
    strategy = canonical_strategy_name(row.get("strategy", "unknown"))
    base = STRATEGY_PLOT_LABELS.get(strategy, strategy)
    if strategy in {"S0_naive_mean_control_reference", "S1_random_single_control"}:
        return base
    if strategy == "S2_random_average_controls":
        k = row.get("n_control_cells_to_average", np.nan)
        return base if is_missing(k) else f"{base} ({fmt_param(k)} cells)"
    if strategy == "S3_SEACell_metacell_average":
        nmc = row.get("n_metacells", np.nan)
        k = row.get("sampled_metacells_k", row.get("n_metacells_to_average", np.nan))
        return f"{base} ({fmt_param(nmc)}&{fmt_param(k)})" if not is_missing(nmc) or not is_missing(k) else base
    if strategy == "S4_SEACell_balanced_random_sample":
        nmc = row.get("n_metacells", np.nan)
        return base if is_missing(nmc) else f"{base} ({fmt_param(nmc)})"
    if strategy == "S5_SEACell_OT_sampled_average":
        nmc = row.get("n_metacells", np.nan)
        top_k = row.get("top_k", np.nan)
        return f"{base} ({fmt_param(nmc)}&{fmt_param(top_k)})" if not is_missing(nmc) or not is_missing(top_k) else base
    return base


def canonicalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "strategy" not in out.columns:
        if "strategy_name" in out.columns:
            out["strategy"] = out["strategy_name"]
        else:
            out["strategy"] = "unknown"
    out["strategy"] = out["strategy"].map(canonical_strategy_name)
    out = infer_numeric_columns(out)
    out["variant_id"] = out.apply(make_variant_id, axis=1)
    out["display_variant_label"] = out.apply(make_display_variant_label, axis=1)
    if "strategy_order" not in out.columns:
        out["strategy_order"] = out["strategy"].map(STRATEGY_ORDER_MAP).fillna(999).astype(int)
    return out


def _config_dir(global_cfg: Mapping[str, Any]) -> Path:
    return resolve_path(global_cfg.get("_config_dir", "."))


def _cfg_path(global_cfg: Mapping[str, Any], value: str | Path) -> Path:
    return resolve_path(value, _config_dir(global_cfg))


def pairing_dataset_dir(global_cfg: Mapping[str, Any], dataset_id: str) -> Path:
    if not global_cfg.get("pairing_root"):
        raise KeyError("global.pairing_root is required in shared_pseudo_pairing_pipeline mode")
    pairing_root = _cfg_path(global_cfg, str(global_cfg["pairing_root"]))
    manifest_dataset_names = dict(global_cfg.get("manifest_dataset_names", {}))
    layer = manifest_dataset_names.get(dataset_id, dataset_id)
    return pairing_root / layer / dataset_id


def find_manifest(pair_dir: Path) -> Path | None:
    candidates = [
        pair_dir / "pseudo_pairing_repetition_manifest.csv",
        pair_dir / "manifest.csv",
    ]
    found = first_existing_path(candidates)
    if found is not None:
        return found
    globbed = sorted(pair_dir.glob("**/pseudo_pairing_repetition_manifest.csv"))
    return globbed[0] if globbed else None


def load_generation_run_config(pair_dir: Path) -> dict[str, Any] | None:
    path = pair_dir / "run_config.json"
    if not path.exists():
        candidates = sorted(pair_dir.glob("**/run_config.json"))
        path = candidates[0] if candidates else path
    if not path.exists():
        return None
    data = read_json(path)
    data["_run_config_dir"] = str(path.parent.resolve())
    return data


def result_analysis_dir(global_cfg: Mapping[str, Any], dataset_id: str, group: str) -> Path:
    if not global_cfg.get("evaluation_root"):
        raise KeyError("global.evaluation_root is required in shared_pseudo_pairing_pipeline mode")
    evaluation_root = _cfg_path(global_cfg, str(global_cfg["evaluation_root"]))
    return evaluation_root / f"{dataset_id}_pseudo_pairing_evaluation" / str(group) / "result_analysis"


def output_dir_for(global_cfg: Mapping[str, Any], dataset_id: str, group: str) -> Path:
    return result_analysis_dir(global_cfg, dataset_id, group) / "gene_program_level"


def _candidate_processed_paths(global_cfg: Mapping[str, Any], dataset_id: str, group: str, kind: str) -> list[Path]:
    roots = []
    for key in ["processed_data_root", "preprocessed_data_root", "data_root"]:
        val = global_cfg.get(key)
        if val:
            roots.append(_cfg_path(global_cfg, str(val)))
    manifest_layer = dict(global_cfg.get("manifest_dataset_names", {})).get(dataset_id, dataset_id)
    names_control = [
        f"{dataset_id}_control_processed.h5ad",
        "control_processed.h5ad",
        f"{dataset_id}_control.h5ad",
        "control.h5ad",
    ]
    names_pert = [
        f"{dataset_id}_{group}_processed.h5ad",
        f"perturbed_{group}_processed.h5ad",
        f"{group}_processed.h5ad",
        f"{dataset_id}_perturbed_{group}_processed.h5ad",
    ]
    names = names_control if kind == "control" else names_pert
    paths = []
    for root in roots:
        for base in [root / dataset_id, root / manifest_layer / dataset_id, root]:
            for sub in [Path("groups"), Path("processed"), Path("")]:
                for name in names:
                    paths.append(base / sub / name)
    return paths


def resolve_control_perturbed_paths(global_cfg: Mapping[str, Any], dataset_id: str, group: str, pair_dir: Path, overrides: Mapping[str, Any] | None = None) -> tuple[str, str]:
    overrides = dict(overrides or {})
    run_cfg = load_generation_run_config(pair_dir) if bool(global_cfg.get("use_generation_run_config", True)) else None

    control = overrides.get("control_h5ad")
    perturbed = None
    if "perturbed_h5ads" in overrides:
        perturbed = dict(overrides["perturbed_h5ads"]).get(group)
    perturbed = overrides.get(f"perturbed_{group}_h5ad", perturbed)
    perturbed = overrides.get("perturbed_h5ad", perturbed)

    if control is not None:
        control = str(_cfg_path(global_cfg, str(control)))
    if perturbed is not None:
        perturbed = str(_cfg_path(global_cfg, str(perturbed)))

    if control is None and run_cfg is not None and run_cfg.get("control_h5ad") is not None:
        control = str(resolve_path(run_cfg["control_h5ad"], run_cfg.get("_run_config_dir", pair_dir)))
    if perturbed is None and run_cfg is not None:
        ph = run_cfg.get("perturbed_h5ads", {})
        if isinstance(ph, Mapping) and ph.get(group) is not None:
            perturbed = str(resolve_path(ph[group], run_cfg.get("_run_config_dir", pair_dir)))

    if control is None:
        found = first_existing_path(_candidate_processed_paths(global_cfg, dataset_id, group, "control"))
        control = str(found) if found is not None else str(_candidate_processed_paths(global_cfg, dataset_id, group, "control")[0])
    if perturbed is None:
        found = first_existing_path(_candidate_processed_paths(global_cfg, dataset_id, group, "perturbed"))
        perturbed = str(found) if found is not None else str(_candidate_processed_paths(global_cfg, dataset_id, group, "perturbed")[0])

    return str(control), str(perturbed)


def selected_variant_ids(selection_table: Path, include_s0: bool = True) -> set[str] | None:
    if not selection_table.exists():
        return None
    sel = read_table(selection_table)
    if "select_for_final" not in sel.columns:
        warnings.warn(f"Selection table has no select_for_final column: {selection_table}")
        return None
    sel = canonicalize_index(sel)
    keep = as_bool_series(sel["select_for_final"])
    selected = set(sel.loc[keep, "variant_id"].astype(str))
    if include_s0:
        selected.add("S0_naive_mean_control_reference")
    return selected


def pseudo_index_from_manifest(manifest_path: Path, group: str, selected_ids: set[str] | None = None, require_existing: bool = False) -> pd.DataFrame:
    df = read_table(manifest_path)
    if "perturbed_group" in df.columns:
        df = df[df["perturbed_group"].astype(str) == str(group)].copy()
    elif "group" in df.columns:
        df = df[df["group"].astype(str) == str(group)].copy()
    else:
        warnings.warn(f"Manifest has no perturbed_group/group column; using all rows: {manifest_path}")
    if df.empty:
        return df
    df = canonicalize_index(df)
    path_cols = ["pseudo_control_h5ad", "output_h5ad", "h5ad_path", "pseudo_h5ad"]
    existing_path_col = next((c for c in path_cols if c in df.columns), None)
    if existing_path_col is None:
        raise KeyError(f"Could not find pseudo-control h5ad path column in {manifest_path}. Tried {path_cols}")
    df["pseudo_control_h5ad"] = df[existing_path_col].map(lambda value: str(resolve_path(value, manifest_path.parent)))
    if selected_ids is not None:
        df = df[df["variant_id"].astype(str).isin(selected_ids)].copy()
    if require_existing:
        df = df[df["pseudo_control_h5ad"].map(lambda p: Path(p).exists())].copy()
    # Keep strategy_order in the returned index table. Earlier versions created
    # strategy_order during canonicalize_index(df) but accidentally dropped it
    # before sorting, which caused KeyError: 'strategy_order'.
    cols = [
        "dataset_id", "perturbed_group", "strategy_order", "strategy", "variant_id", "display_variant_label",
        "sampling_seed", "parameter_label", "pseudo_control_h5ad", "pair_metadata_path", "outdir",
        "n_metacells", "top_k", "n_control_cells_to_average", "sampled_metacells_k", "sample_cells_per_metacell",
    ]

    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    # Robust fallback in case the source manifest had unusual strategy labels or
    # selected/require-existing filters left an empty-but-columned dataframe.
    if "strategy_order" not in df.columns or df["strategy_order"].isna().all():
        df["strategy_order"] = df["strategy"].map(STRATEGY_ORDER_MAP).fillna(999).astype(int)
    else:
        df["strategy_order"] = pd.to_numeric(df["strategy_order"], errors="coerce").fillna(999).astype(int)

    if "perturbed_group" not in df.columns or df["perturbed_group"].isna().all():
        df["perturbed_group"] = str(group)

    sort_cols = [c for c in ["strategy_order", "variant_id", "sampling_seed"] if c in df.columns]
    return df[cols].sort_values(sort_cols, na_position="last").reset_index(drop=True)


def discover_pseudo_files(pseudo_root: str | Path, pseudo_glob: str = "**/pseudo_control*.h5ad", require_existing: bool = False) -> pd.DataFrame:
    root = resolve_path(pseudo_root)
    files = sorted(root.glob(pseudo_glob))
    rows = []
    for p in files:
        if require_existing and not p.exists():
            continue
        parts = p.parts
        strategy = next((canonical_strategy_name(part) for part in parts if canonical_strategy_name(part) in STRATEGY_ORDER), "unknown")
        rows.append({"strategy": strategy, "pseudo_control_h5ad": str(p), "outdir": str(p.parent)})
    return canonicalize_index(pd.DataFrame(rows)) if rows else pd.DataFrame()



def _normalize_explicit_config(config: Mapping[str, Any], only_datasets: Sequence[str] | None = None) -> dict[str, Any]:
    resolved = dict(config)
    global_cfg = dict(config.get("global", {}))
    resolved["global"] = global_cfg
    base = _config_dir(global_cfg)
    keep = set(map(str, only_datasets or []))
    datasets = []
    dataset_path_keys = {
        "control_h5ad", "perturbed_h5ad", "manifest_path", "pseudo_root",
        "result_analysis_dir", "output_dir", "pairing_dataset_dir",
    }
    for raw in config.get("datasets", []):
        ds = dict(raw)
        if keep and str(ds.get("dataset_id")) not in keep and str(ds.get("source_dataset_id")) not in keep:
            continue
        for key in dataset_path_keys:
            if ds.get(key) not in {None, ""}:
                ds[key] = str(resolve_path(ds[key], base))
        if ds.get("pseudo_files"):
            pseudo_files = []
            for raw_pseudo in ds["pseudo_files"]:
                row = dict(raw_pseudo)
                if row.get("pseudo_control_h5ad") not in {None, ""}:
                    row["pseudo_control_h5ad"] = str(resolve_path(row["pseudo_control_h5ad"], base))
                if row.get("pair_metadata_path") not in {None, ""}:
                    row["pair_metadata_path"] = str(resolve_path(row["pair_metadata_path"], base))
                if row.get("outdir") not in {None, ""}:
                    row["outdir"] = str(resolve_path(row["outdir"], base))
                pseudo_files.append(row)
            ds["pseudo_files"] = pseudo_files
        datasets.append(ds)
    resolved["datasets"] = datasets
    return resolved

def resolve_shared_config(config: Mapping[str, Any], only_datasets: Sequence[str] | None = None, prepare_only: bool = False) -> dict[str, Any]:
    global_cfg = dict(config.get("global", {}))
    path_mode = str(global_cfg.get("path_mode", "explicit"))
    if path_mode == "explicit":
        return _normalize_explicit_config(config, only_datasets=only_datasets)
    if path_mode != "shared_pseudo_pairing_pipeline":
        raise ValueError("global.path_mode must be 'shared_pseudo_pairing_pipeline' or 'explicit'")

    dataset_ids = list(global_cfg.get("dataset_ids", []))
    if only_datasets:
        keep = set(map(str, only_datasets))
        dataset_ids = [d for d in dataset_ids if d in keep]
    groups_default = list(global_cfg.get("perturbed_groups", ["single", "dual", "multi"]))
    dataset_overrides = dict(config.get("dataset_overrides", {}))
    require_existing_pseudo_files = bool(global_cfg.get("require_existing_pseudo_files", False))
    only_existing_processed = bool(global_cfg.get("only_existing_processed_files", True))
    use_selected = bool(global_cfg.get("use_selected_variants", False))
    include_s0 = bool(global_cfg.get("include_s0_when_using_selected_variants", True))
    selection_name = str(global_cfg.get("selection_table_name", "selected_variants_TEMPLATE_EDIT_ME.csv"))
    write_index = bool(global_cfg.get("write_pseudo_file_index", True))

    resolved_datasets = []
    skipped = []
    for dataset_id in dataset_ids:
        pair_dir = pairing_dataset_dir(global_cfg, dataset_id)
        manifest_path = find_manifest(pair_dir)
        overrides = dict(dataset_overrides.get(dataset_id, {}))
        groups = list(overrides.get("groups", groups_default))
        for group in groups:
            control_h5ad, perturbed_h5ad = resolve_control_perturbed_paths(global_cfg, dataset_id, group, pair_dir, overrides)
            if only_existing_processed and (not Path(control_h5ad).exists() or not Path(perturbed_h5ad).exists()):
                skipped.append({
                    "dataset_id": dataset_id,
                    "group": group,
                    "reason": "missing control or perturbed h5ad",
                    "control_h5ad": control_h5ad,
                    "perturbed_h5ad": perturbed_h5ad,
                })
                continue

            outdir = output_dir_for(global_cfg, dataset_id, group)
            pseudo_index = pd.DataFrame()
            if bool(global_cfg.get("use_manifest_pseudo_files", True)) and manifest_path is not None:
                selected_ids = None
                if use_selected:
                    selected_ids = selected_variant_ids(result_analysis_dir(global_cfg, dataset_id, group) / selection_name, include_s0=include_s0)
                pseudo_index = pseudo_index_from_manifest(manifest_path, group, selected_ids=selected_ids, require_existing=require_existing_pseudo_files)
            if pseudo_index.empty:
                pseudo_root_raw = overrides.get("pseudo_root", str(pair_dir / group))
                pseudo_root = _cfg_path(global_cfg, str(pseudo_root_raw)) if overrides.get("pseudo_root") else pair_dir / group
                pseudo_index = discover_pseudo_files(pseudo_root, str(global_cfg.get("pseudo_glob", "**/pseudo_control*.h5ad")), require_existing=require_existing_pseudo_files)

            if pseudo_index.empty:
                skipped.append({"dataset_id": dataset_id, "group": group, "reason": "no pseudo-control files found", "pair_dir": str(pair_dir), "manifest_path": str(manifest_path) if manifest_path else None})
                if bool(global_cfg.get("skip_datasets_without_pseudo", True)):
                    continue

            ensure_dir(outdir / "inputs")
            pseudo_files = []
            if not pseudo_index.empty:
                pseudo_index["dataset_id"] = dataset_id
                pseudo_index["perturbed_group"] = group
                if write_index:
                    save_table(pseudo_index, outdir / "inputs" / "pseudo_file_index_from_manifest.csv")
                pseudo_files = pseudo_index.to_dict(orient="records")

            resolved_datasets.append({
                "dataset_id": f"{dataset_id}_{group}",
                "source_dataset_id": dataset_id,
                "perturbed_group": group,
                "control_h5ad": control_h5ad,
                "perturbed_h5ad": perturbed_h5ad,
                "pairing_dataset_dir": str(pair_dir),
                "manifest_path": str(manifest_path) if manifest_path else None,
                "result_analysis_dir": str(result_analysis_dir(global_cfg, dataset_id, group)),
                "output_dir": str(outdir),
                "pseudo_files": pseudo_files,
                "pseudo_root": str(pair_dir / group),
            })

    resolved = dict(config)
    resolved["global"] = global_cfg
    resolved["datasets"] = resolved_datasets
    resolved["shared_path_resolution"] = {"skipped": skipped}
    return resolved


def write_resolved_config(config: Mapping[str, Any], original_config_path: str | Path | None = None) -> Path | None:
    if original_config_path is None:
        return None
    path = Path(original_config_path)
    out = path.with_name(path.stem + "__resolved.json")
    save_json(config, out)
    return out
