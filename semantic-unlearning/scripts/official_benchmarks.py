#!/usr/bin/env python3
"""Official-benchmark-first inventory, audit, plan, run, and aggregation CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from official_benchmarks.aggregation import aggregate_runs  # noqa: E402
from official_benchmarks.doctor import run_doctor  # noqa: E402
from official_benchmarks.planner import plan_tracks  # noqa: E402
from official_benchmarks.registry import (  # noqa: E402
    RegistryError,
    get_track,
    load_registry,
    select_tracks,
)
from official_benchmarks.runner import RunRefused, run_track  # noqa: E402


def _path(value: Optional[str]) -> Optional[Path]:
    return None if value is None else Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed official benchmark orchestration for Setting 5e + protected/active LM-head repair."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Registry JSON (defaults to config/official_benchmarks/registry.json).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory", help="Print the CPU-only benchmark/method inventory.")

    doctor = subparsers.add_parser("doctor", help="Audit pins, artifacts, roles, and compatibility.")
    doctor.add_argument("--suite", default="all")
    doctor.add_argument("--output-dir", type=Path, required=True)
    doctor.add_argument("--models-config", type=Path)
    doctor.add_argument("--source-lock", type=Path)

    plan = subparsers.add_parser("plan", help="Write setup/run commands without execution.")
    plan.add_argument("--suite", default="all")
    plan.add_argument("--method", default="our_method")
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--models-config", type=Path)
    plan.add_argument("--source-lock", type=Path)

    run = subparsers.add_parser("run", help="Dry-run or execute one resolved native track.")
    run.add_argument("--benchmark", required=True)
    run.add_argument("--method", default="our_method")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--models-config", type=Path)
    run.add_argument("--source-lock", type=Path)
    run.add_argument("--execute", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Collect native metrics without replacing them.")
    aggregate.add_argument("--runs-root", type=Path, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def print_inventory(registry: dict) -> None:
    print("BENCHMARK TRACKS (15)")
    print("id\tclassification\tinput_contract\tmethod_status")
    for track in registry["tracks"]:
        print(
            f"{track['id']}\t{track['classification']}\t"
            f"{track['input_contract']}\t{track['method_status']}"
        )
    print("\nEVALUATION PROFILES")
    for profile in registry["evaluation_profiles"]:
        print(f"{profile['id']}\t{profile['classification']}")
    print("\nBASELINE METHODS (not datasets)")
    for method in registry["baseline_methods"]:
        availability = method.get("availability", "optional_if_pinned")
        print(f"{method['id']}\t{method['classification']}\t{availability}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.registry)
        if args.command == "inventory":
            print_inventory(registry)
            return 0
        if args.command == "doctor":
            payload = run_doctor(
                select_tracks(registry, args.suite),
                output_dir=args.output_dir,
                models_path=args.models_config,
                source_lock_path=args.source_lock,
            )
            print(json.dumps(payload["summary"], indent=2, sort_keys=True))
            print(f"Audit: {args.output_dir / 'doctor.json'}")
            return 0
        if args.command == "plan":
            if args.method != "our_method":
                raise RegistryError(
                    "Stage 1 planning supports our_method only; baselines remain optional pinned metadata"
                )
            payload = plan_tracks(
                select_tracks(registry, args.suite),
                output_dir=args.output_dir,
                method=args.method,
                models_path=args.models_config,
                source_lock_path=args.source_lock,
            )
            print(f"Planned {len(payload['tracks'])} tracks: {args.output_dir / 'plan.json'}")
            return 0
        if args.command == "run":
            result = run_track(
                get_track(registry, args.benchmark),
                method=args.method,
                output_dir=args.output_dir,
                execute=args.execute,
                models_path=args.models_config,
                source_lock_path=args.source_lock,
            )
            print(("EXECUTE: " if args.execute else "DRY RUN: ") + result["command"])
            print(f"Manifest: {result['manifest']}")
            return 0
        if args.command == "aggregate":
            payload = aggregate_runs(args.runs_root, args.output_dir)
            print(f"Aggregated {payload['run_count']} manifests: {args.output_dir / 'aggregate.json'}")
            return 0
    except (RegistryError, RunRefused, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
