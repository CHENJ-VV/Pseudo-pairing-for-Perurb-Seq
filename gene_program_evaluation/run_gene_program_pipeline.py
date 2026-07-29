#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gene_program_pipeline_pkg.pipeline import run_pipeline
from gene_program_pipeline_pkg.utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build control-derived gene programs and evaluate pseudo-control perturbation effects at gene-program level."
    )
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--build", action="store_true", help="Build/rebuild gene programs from real control cells")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate pseudo-control strategies at gene-program level")
    parser.add_argument("--prepare-only", action="store_true", help="Only resolve shared paths and write a resolved config; do not build/evaluate")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset_id/source_dataset_id list to run")
    parser.add_argument("--use-selected-variants", action="store_true", help="Override config and evaluate only selected variants from result_analysis/selected_variants_TEMPLATE_EDIT_ME.csv")
    parser.add_argument("--all-manifest-variants", action="store_true", help="Override config and evaluate all pseudo-control files in the generation manifest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.prepare_only and not args.build and not args.evaluate:
        raise SystemExit("Specify at least one of --prepare-only, --build, or --evaluate")
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if args.use_selected_variants and args.all_manifest_variants:
        raise SystemExit("Use only one of --use-selected-variants or --all-manifest-variants")
    if args.use_selected_variants:
        config["global"]["use_selected_variants"] = True
    if args.all_manifest_variants:
        config["global"]["use_selected_variants"] = False
    run_pipeline(
        config,
        run_build=args.build,
        run_evaluate=args.evaluate,
        only_datasets=args.datasets,
        prepare_only=args.prepare_only,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
