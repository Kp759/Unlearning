#!/usr/bin/env python3
"""Run reviewed original closed-form ZeroUnlearn on official MCF seeds 0-9.

This is a zero-only runner: it evaluates the untouched base model and the
in-memory original ZeroUnlearn edit for each requested seed, but it never
discovers or validates results from other methods. The original seed-0
comparison runner remains separately restricted to its full comparison
protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import run_zerounlearn_official_mcf as official


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_OUTPUT_ROOT = (
    official.SEMANTIC_ROOT / "outputs" / "zerounlearn_official_multiseed"
)
DEFAULT_SEEDS = tuple(range(10))
METRICS = ("Eff", "Gen", "Spe", "Spe_success", "PPL")
METHOD_ORDER = ("Base", official.METHOD)


@dataclass(frozen=True)
class SeedOutputPaths:
    seed_dir: Path
    base_result: Path
    zero_unlearn_result: Path
    provenance: Path

    def as_dict(self) -> Dict[str, str]:
        return {
            "seed_dir": str(self.seed_dir),
            "base_result": str(self.base_result),
            "zero_unlearn_result": str(self.zero_unlearn_result),
            "provenance": str(self.provenance),
        }


def seed_output_paths(output_root: Path | str, seed: int) -> SeedOutputPaths:
    root = Path(output_root)
    seed_dir = root / f"seed{seed}"
    return SeedOutputPaths(
        seed_dir=seed_dir,
        base_result=seed_dir / f"base_seed{seed}_official_eval.json",
        zero_unlearn_result=(seed_dir / f"zerounlearn_seed{seed}_official_eval.json"),
        provenance=seed_dir / f"zerounlearn_seed{seed}_provenance.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        metavar="SEED",
    )
    parser.add_argument("--model-path", default=official.DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--zero-unlearn-root",
        default=str(official.DEFAULT_ZERO_ROOT),
    )
    parser.add_argument("--hparams-path", default=str(official.DEFAULT_HPARAMS))
    parser.add_argument("--mcf-path", default=str(official.DEFAULT_MCF))
    parser.add_argument("--wikidata-dir", default=str(official.DEFAULT_WIKIDATA))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--forget-num",
        type=int,
        default=official.FORGET_NUM,
    )
    parser.add_argument(
        "--retain-num",
        type=int,
        default=official.RETAIN_NUM,
    )
    parser.add_argument(
        "--sample-mode",
        choices=[official.SAMPLE_MODE],
        default=official.SAMPLE_MODE,
    )
    parser.add_argument(
        "--dtype",
        choices=[official.DTYPE_NAME],
        default=official.DTYPE_NAME,
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip a seed only when both its ZeroUnlearn official evaluation "
            "and provenance JSON pass all protocol and provenance checks."
        ),
    )
    return parser


def validate_multiseed_args(args: argparse.Namespace) -> None:
    seeds = list(args.seeds)
    if not seeds:
        raise ValueError("--seeds requires at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds contains duplicates: {seeds}")
    for seed in seeds:
        official.validate_zero_only_protocol(
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            sample_mode=args.sample_mode,
            model_path=args.model_path,
            dtype=args.dtype,
        )


def _same_path(actual: Any, expected: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    return Path(actual).expanduser().resolve() == expected.resolve()


def validate_official_result(
    payload: Mapping[str, Any],
    *,
    path: Path,
    method: str,
    seed: int,
    forget_num: int,
    retain_num: int,
    sample_mode: str,
    model_path: Path,
    dtype: str,
    forget_case_ids: Sequence[int],
    retain_case_ids: Sequence[int],
) -> Dict[str, Optional[float]]:
    errors: List[str] = []
    metadata = official.protocol_metadata(payload)
    expected_metadata = {
        "seed": seed,
        "dataset": "MCF",
        "sample_mode": sample_mode,
        "forget_num": forget_num,
        "retain_num": retain_num,
    }
    for key, expected in expected_metadata.items():
        actual = metadata[key]
        if key == "dataset":
            matches = str(actual).upper() == expected
        elif key in {"seed", "forget_num", "retain_num"}:
            try:
                matches = int(actual) == expected
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    if payload.get("method") != method:
        errors.append(f"method: expected {method!r}, got {payload.get('method')!r}")
    if payload.get("model_revision") != official.MODEL_REVISION:
        errors.append(
            "model_revision: expected "
            f"{official.MODEL_REVISION!r}, got {payload.get('model_revision')!r}"
        )
    if payload.get("dtype") != dtype:
        errors.append(f"dtype: expected {dtype!r}, got {payload.get('dtype')!r}")
    if not _same_path(payload.get("model_path"), model_path):
        errors.append(
            f"model_path: expected {str(model_path)!r}, "
            f"got {payload.get('model_path')!r}"
        )

    stored_forget, stored_retain = official.extract_stored_case_ids(payload)
    if stored_forget != list(forget_case_ids):
        errors.append("forget_case_ids do not match the official seeded split")
    if stored_retain != list(retain_case_ids):
        errors.append("retain_case_ids do not match the official seeded split")
    if payload.get("case_ids_source") != f"official_sampler_seed{seed}":
        errors.append(
            "case_ids_source: expected "
            f"'official_sampler_seed{seed}', got "
            f"{payload.get('case_ids_source')!r}"
        )

    try:
        metrics = official.extract_result_metrics(payload)
    except ValueError as exc:
        errors.append(str(exc))
        metrics = {metric: None for metric in METRICS}
    if metrics.get("Spe_success") is None:
        errors.append("official result is missing finite Spe_success")

    if method == official.METHOD:
        if payload.get("model_dir") != f"in-memory:{official.METHOD}":
            errors.append(
                "ZeroUnlearn result must identify the edited model as in-memory"
            )
        if not isinstance(payload.get("zero_unlearn_runtime"), Mapping):
            errors.append("ZeroUnlearn result is missing runtime metadata")
    else:
        if not _same_path(payload.get("model_dir"), model_path):
            errors.append("Base result model_dir does not match the model snapshot")
        if payload.get("zero_unlearn_runtime") is not None:
            errors.append("Base result unexpectedly contains ZeroUnlearn runtime")

    if errors:
        raise ValueError(f"Invalid official result {path}:\n- " + "\n- ".join(errors))
    return metrics


def validate_provenance(
    payload: Mapping[str, Any],
    *,
    path: Path,
    result: Mapping[str, Any],
    result_path: Path,
    seed: int,
    forget_num: int,
    retain_num: int,
    sample_mode: str,
    model_path: Path,
    hparams_path: Path,
    dtype: str,
    forget_case_ids: Sequence[int],
    retain_case_ids: Sequence[int],
    expected_source_hashes: Optional[Mapping[str, str]],
) -> None:
    errors: List[str] = []
    expected_values = {
        "status": "completed",
        "method": official.METHOD,
        "algorithm_entrypoint": ("ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model"),
        "model_revision": official.MODEL_REVISION,
        "dtype": dtype,
        "zero_unlearn_compute_dtype": "float32",
        "seed": seed,
        "dataset": "MCF",
        "sample_mode": sample_mode,
        "forget_num": forget_num,
        "retain_num": retain_num,
        "edit_layer_nums": official.EDIT_LAYER_NUMS,
        "add_retain": official.ADD_RETAIN,
        "use_h": official.USE_H,
        "checkpoint_saved": False,
        "multi_gpu_device_map_used": False,
        "source_hashes_unchanged": True,
    }
    for key, expected in expected_values.items():
        actual = payload.get(key)
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    if not _same_path(payload.get("model_path"), model_path):
        errors.append("model_path does not match the requested model snapshot")
    if not _same_path(payload.get("hparams_path"), hparams_path):
        errors.append("hparams_path does not match the reviewed hparams")
    evaluation_path = payload.get("official_evaluation_path")
    if evaluation_path is not None and not _same_path(evaluation_path, result_path):
        errors.append("official_evaluation_path does not match the result file")

    if payload.get("forget_case_ids") != list(forget_case_ids):
        errors.append("provenance forget_case_ids do not match the official split")
    if payload.get("retain_case_ids") != list(retain_case_ids):
        errors.append("provenance retain_case_ids do not match the official split")
    if payload.get("case_ids_source") != f"official_sampler_seed{seed}":
        errors.append("provenance case_ids_source is not seed-specific")

    neutral = payload.get("neutral_target")
    if not isinstance(neutral, Mapping):
        errors.append("neutral_target provenance is missing")
    else:
        neutral_expected = {
            "source": "tokenizer.eos_token",
            "zero_unlearn_request_field": "target_new.str",
            "zero_unlearn_sensitive_request_field": "target_true",
            "zero_unlearn_sensitive_target_source": (
                "MCF requested_rewrite.target_new"
            ),
            "benchmark_forget_target": "MCF requested_rewrite.target_new",
            "benchmark_correct_target": "MCF requested_rewrite.target_true",
            "forget_request_count": forget_num,
            "retain_requests_modified": False,
            "official_evaluation_records_modified": False,
            "source_mcf_modified": False,
        }
        for key, expected in neutral_expected.items():
            if neutral.get(key) != expected:
                errors.append(
                    f"neutral_target.{key}: expected {expected!r}, "
                    f"got {neutral.get(key)!r}"
                )
        if not isinstance(neutral.get("token"), str) or not neutral.get("token"):
            errors.append("neutral_target.token is missing")
        if not isinstance(neutral.get("token_id"), int):
            errors.append("neutral_target.token_id is missing")

    before = payload.get("source_hashes_before")
    after = payload.get("source_hashes_after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        errors.append("source hash provenance is missing")
    else:
        if dict(before) != dict(after):
            errors.append("source hashes changed during the seed run")
        if expected_source_hashes is not None and dict(before) != dict(
            expected_source_hashes
        ):
            errors.append("source hashes do not match the current reviewed inputs")

    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        errors.append("completed provenance is missing runtime metadata")
    elif result.get("zero_unlearn_runtime") != runtime:
        errors.append("result and provenance runtime metadata differ")

    if errors:
        raise ValueError(
            f"Invalid ZeroUnlearn provenance {path}:\n- " + "\n- ".join(errors)
        )


def validate_completed_seed(
    paths: SeedOutputPaths,
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
    sample_mode: str,
    model_path: Path,
    hparams_path: Path,
    dtype: str,
    forget_case_ids: Sequence[int],
    retain_case_ids: Sequence[int],
    expected_source_hashes: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Validate only the two files that define resumable completion."""
    result = official.read_json(paths.zero_unlearn_result)
    provenance = official.read_json(paths.provenance)
    validate_official_result(
        result,
        path=paths.zero_unlearn_result,
        method=official.METHOD,
        seed=seed,
        forget_num=forget_num,
        retain_num=retain_num,
        sample_mode=sample_mode,
        model_path=model_path,
        dtype=dtype,
        forget_case_ids=forget_case_ids,
        retain_case_ids=retain_case_ids,
    )
    validate_provenance(
        provenance,
        path=paths.provenance,
        result=result,
        result_path=paths.zero_unlearn_result,
        seed=seed,
        forget_num=forget_num,
        retain_num=retain_num,
        sample_mode=sample_mode,
        model_path=model_path,
        hparams_path=hparams_path,
        dtype=dtype,
        forget_case_ids=forget_case_ids,
        retain_case_ids=retain_case_ids,
        expected_source_hashes=expected_source_hashes,
    )
    return result, provenance


def is_seed_complete(paths: SeedOutputPaths, **expected: Any) -> bool:
    try:
        validate_completed_seed(paths, **expected)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def result_row(
    method: str,
    seed: int,
    source: Path,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = official.extract_result_metrics(payload)
    return {
        "method": method,
        "seed": seed,
        **{metric: metrics[metric] for metric in METRICS},
        "source": str(source),
    }


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("method", official.METHOD)), []).append(row)

    aggregate: List[Dict[str, Any]] = []
    for method, method_rows in grouped.items():
        ordered = sorted(method_rows, key=lambda row: int(row["seed"]))
        output: Dict[str, Any] = {
            "method": method,
            "n_seeds": len(ordered),
            "seeds": ",".join(str(row["seed"]) for row in ordered),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in ordered]
            output[f"{metric}_mean"] = statistics.fmean(values)
            output[f"{metric}_std"] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )
            output[f"{metric}_n"] = len(values)
        aggregate.append(output)
    return sorted(
        aggregate,
        key=lambda row: (
            METHOD_ORDER.index(str(row["method"]))
            if str(row["method"]) in METHOD_ORDER
            else len(METHOD_ORDER),
            str(row["method"]),
        ),
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_per_seed_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Original ZeroUnlearn official MCF results by seed",
        "",
        "Eff and Gen are lower-is-better; Spe and Spe_success are "
        "higher-is-better; PPL is lower/stable-is-better.",
        "",
        "| Method | Seed | Eff ↓ | Gen ↓ | Spe ↑ | Spe_success ↑ | PPL ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {seed} | {Eff} | {Gen} | {Spe} | "
            "{Spe_success} | {PPL} |".format(
                **{key: _display(value) for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregate_markdown(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Original ZeroUnlearn aggregate official MCF results",
        "",
        "Values are mean ± population standard deviation across completed seeds.",
        "",
        "| Method | Seeds | Eff ↓ | Gen ↓ | Spe ↑ | Spe_success ↑ | PPL ↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        cells = [
            f"{float(row[f'{metric}_mean']):.3f} ± "
            f"{float(row[f'{metric}_std']):.3f}"
            for metric in METRICS
        ]
        lines.append(
            f"| {row['method']} | {row['n_seeds']} | " + " | ".join(cells) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_rows(
    *,
    output_root: Path,
    seeds: Sequence[int],
    forget_num: int,
    retain_num: int,
    sample_mode: str,
    model_path: Path,
    hparams_path: Path,
    mcf_path: Path,
    dtype: str,
    expected_source_hashes: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    base_issues: List[Dict[str, Any]] = []
    for seed in seeds:
        paths = seed_output_paths(output_root, seed)
        _, forget_records, retain_records = official.load_official_split(
            mcf_path,
            seed=seed,
            forget_num=forget_num,
            retain_num=retain_num,
            sample_mode=sample_mode,
        )
        forget_ids = official.case_ids(forget_records)
        retain_ids = official.case_ids(retain_records)
        zero_result, _ = validate_completed_seed(
            paths,
            seed=seed,
            forget_num=forget_num,
            retain_num=retain_num,
            sample_mode=sample_mode,
            model_path=model_path,
            hparams_path=hparams_path,
            dtype=dtype,
            forget_case_ids=forget_ids,
            retain_case_ids=retain_ids,
            expected_source_hashes=expected_source_hashes,
        )

        if paths.base_result.is_file():
            try:
                base_result = official.read_json(paths.base_result)
                validate_official_result(
                    base_result,
                    path=paths.base_result,
                    method="Base",
                    seed=seed,
                    forget_num=forget_num,
                    retain_num=retain_num,
                    sample_mode=sample_mode,
                    model_path=model_path,
                    dtype=dtype,
                    forget_case_ids=forget_ids,
                    retain_case_ids=retain_ids,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                base_issues.append({"seed": seed, "error": str(exc)})
            else:
                rows.append(result_row("Base", seed, paths.base_result, base_result))
        else:
            base_issues.append({"seed": seed, "error": f"missing {paths.base_result}"})
        rows.append(
            result_row(
                official.METHOD,
                seed,
                paths.zero_unlearn_result,
                zero_result,
            )
        )

    method_rank = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows.sort(
        key=lambda row: (
            int(row["seed"]),
            method_rank.get(str(row["method"]), len(method_rank)),
        )
    )
    return rows, base_issues


def write_summary_outputs(
    *,
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    protocol: Mapping[str, Any],
    seed_status: Sequence[Mapping[str, Any]],
    base_issues: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(output_root / "per_seed.csv", rows)
    write_csv(output_root / "aggregate.csv", aggregate)
    write_per_seed_markdown(output_root / "per_seed.md", rows)
    write_aggregate_markdown(output_root / "aggregate.md", aggregate)
    official.write_json(
        output_root / "results.json",
        {
            "protocol": dict(protocol),
            "standard_deviation": "population",
            "seed_status": list(seed_status),
            "base_result_issues": list(base_issues),
            "per_seed": list(rows),
            "aggregate": list(aggregate),
        },
    )


def run(
    args: argparse.Namespace,
    *,
    seed_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    validate_multiseed_args(args)
    seeds = list(args.seeds)
    model_path = Path(args.model_path).expanduser().resolve()
    zero_root = Path(args.zero_unlearn_root).expanduser().resolve()
    hparams_path = Path(args.hparams_path).expanduser().resolve()
    mcf_path = Path(args.mcf_path).expanduser().resolve()
    wikidata_dir = Path(args.wikidata_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    official.require_runtime_files(
        model_path,
        mcf_path,
        wikidata_dir,
        hparams_path,
        zero_root,
    )
    source_hashes = official.hash_protocol_inputs(
        mcf_path,
        hparams_path,
        zero_root,
    )
    official.validate_expected_protocol_hashes(
        source_hashes,
        mcf_path,
        hparams_path,
        zero_root,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    execute_seed = seed_runner or official.run_original_zerounlearn_mcf
    seed_status: List[Dict[str, Any]] = []

    for seed in seeds:
        paths = seed_output_paths(output_root, seed)
        _, forget_records, retain_records = official.load_official_split(
            mcf_path,
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            sample_mode=args.sample_mode,
        )
        completion_expected = {
            "seed": seed,
            "forget_num": args.forget_num,
            "retain_num": args.retain_num,
            "sample_mode": args.sample_mode,
            "model_path": model_path,
            "hparams_path": hparams_path,
            "dtype": args.dtype,
            "forget_case_ids": official.case_ids(forget_records),
            "retain_case_ids": official.case_ids(retain_records),
            "expected_source_hashes": source_hashes,
        }
        if args.skip_completed and is_seed_complete(
            paths,
            **completion_expected,
        ):
            print(f"Skipping completed and validated seed {seed}")
            seed_status.append(
                {"seed": seed, "status": "skipped_completed", **paths.as_dict()}
            )
            continue

        if args.skip_completed and (
            paths.zero_unlearn_result.exists() or paths.provenance.exists()
        ):
            print(f"Seed {seed} has incomplete or invalid outputs; rerunning it")
        paths.seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"Running original ZeroUnlearn official MCF seed {seed}")
        execute_seed(
            seed,
            args.forget_num,
            args.retain_num,
            args.sample_mode,
            model_path,
            paths.seed_dir,
            zero_unlearn_root=zero_root,
            hparams_path=hparams_path,
            mcf_path=mcf_path,
            wikidata_dir=wikidata_dir,
            dtype=args.dtype,
            exact_command=[
                sys.executable,
                str(SCRIPT_PATH),
                *sys.argv[1:],
            ],
        )
        validate_completed_seed(paths, **completion_expected)
        seed_status.append({"seed": seed, "status": "completed", **paths.as_dict()})

    rows, base_issues = collect_rows(
        output_root=output_root,
        seeds=seeds,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        sample_mode=args.sample_mode,
        model_path=model_path,
        hparams_path=hparams_path,
        mcf_path=mcf_path,
        dtype=args.dtype,
        expected_source_hashes=source_hashes,
    )
    aggregate = aggregate_rows(rows)
    protocol = {
        "dataset": "MCF",
        "seeds": seeds,
        "sample_mode": args.sample_mode,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "model_path": str(model_path),
        "model_revision": official.MODEL_REVISION,
        "dtype": args.dtype,
        "zero_unlearn_compute_dtype": "float32",
        "hparams_path": str(hparams_path),
        "mcf_path": str(mcf_path),
        "wikidata_dir": str(wikidata_dir),
        "zero_unlearn_root": str(zero_root),
        "source_hashes": source_hashes,
        "checkpoint_saved": False,
    }
    write_summary_outputs(
        output_root=output_root,
        rows=rows,
        aggregate=aggregate,
        protocol=protocol,
        seed_status=seed_status,
        base_issues=base_issues,
    )
    result = {
        "protocol": protocol,
        "seed_status": seed_status,
        "base_result_issues": base_issues,
        "per_seed": rows,
        "aggregate": aggregate,
    }
    print(f"Wrote multiseed ZeroUnlearn results to {output_root}")
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
