from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .shared_paths import resolve_shared_config, write_resolved_config
from .utils import save_json
from tqdm.auto import tqdm


def run_pipeline(
    config: Mapping[str, Any],
    run_build: bool = True,
    run_evaluate: bool = True,
    only_datasets: Sequence[str] | None = None,
    prepare_only: bool = False,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve shared pseudo-pairing paths and run gene-program evaluation."""
    resolved = resolve_shared_config(config, only_datasets=only_datasets, prepare_only=prepare_only)
    resolved_path = write_resolved_config(resolved, config_path)
    if resolved_path is not None:
        print(f"[Resolved config] {resolved_path}")
    if prepare_only:
        print("[Prepare only] Path resolution completed; no build/evaluate steps were run.")
        skipped = resolved.get("shared_path_resolution", {}).get("skipped", [])
        if skipped:
            print(f"[Prepare only] Skipped dataset/group entries: {len(skipped)}")
        return {"resolved_config": str(resolved_path) if resolved_path else None, "n_datasets": len(resolved.get("datasets", [])), "n_skipped": len(skipped)}

    # Import only when actual analysis is requested. This keeps --prepare-only usable
    # on login nodes/environments where anndata is not installed yet.
    from .evaluate import run_dataset

    global_cfg = dict(resolved.get("global", {}))
    datasets = list(resolved.get("datasets", []))
    if only_datasets:
        keep = set(map(str, only_datasets))
        datasets = [d for d in datasets if str(d.get("dataset_id")) in keep or str(d.get("source_dataset_id")) in keep]
    if not datasets:
        raise RuntimeError("No datasets are available after path resolution/filtering. Check resolved config and skipped entries.")

    results = []
    for ds in tqdm(datasets, desc='running datasets'):
        results.append(run_dataset(ds, global_cfg, run_build=run_build, run_evaluate=run_evaluate))
    summary = {"resolved_config": str(resolved_path) if resolved_path else None, "n_datasets_run": len(results), "datasets": results}
    if resolved_path is not None:
        save_json(summary, Path(resolved_path).with_name(Path(resolved_path).stem + "__run_summary.json"))
    return summary
