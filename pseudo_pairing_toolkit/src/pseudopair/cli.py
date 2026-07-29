"""Command-line interface for the unified workflow."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .config import load_config, validate_config
from .doctor import dependency_report
from .pipeline import pipeline_plan, run_pipeline


def _print_json(value) -> None:
    print(json.dumps(value, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pseudopair", description="Pseudo-control pairing and evaluation workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Write an editable example configuration")
    init.add_argument("output", nargs="?", default="pseudopair.yaml")
    init.add_argument("--force", action="store_true")

    validate = sub.add_parser("validate", help="Validate configuration without running scientific code")
    validate.add_argument("--config", required=True)
    validate.add_argument("--check-files", action="store_true")

    plan = sub.add_parser("plan", help="Show resolved stages and output locations")
    plan.add_argument("--config", required=True)

    run = sub.add_parser("run", help="Run one stage or the full workflow")
    run.add_argument("--config", required=True)
    run.add_argument("--stage", choices=["all", "acquire", "preprocess", "pair", "evaluate", "aggregate"], default="all")
    run.add_argument("--dry-run", action="store_true")

    sub.add_parser("doctor", help="Report optional scientific dependencies")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        destination = Path(args.output)
        if destination.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {destination}; use --force.")
        template = Path(__file__).resolve().with_name("templates") / "pipeline.example.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template, destination)
        print(destination)
        return 0
    if args.command == "doctor":
        _print_json(dependency_report())
        return 0

    config = load_config(args.config)
    errors = validate_config(config, check_files=getattr(args, "check_files", False))
    if args.command == "validate":
        if errors:
            _print_json({"valid": False, "errors": errors})
            return 2
        _print_json({"valid": True, "plan": pipeline_plan(config)})
        return 0
    if errors:
        raise ValueError("Configuration errors:\n- " + "\n- ".join(errors))
    if args.command == "plan" or args.dry_run:
        _print_json(pipeline_plan(config))
        return 0
    stages = None if args.stage == "all" else [args.stage]
    _print_json(run_pipeline(config, stages=stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
