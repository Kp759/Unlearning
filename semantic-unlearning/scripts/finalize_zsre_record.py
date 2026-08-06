#!/usr/bin/env python3
"""Create an auditable ZsRE record with separate candidate and final tables."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


METRICS = (
    ("forget", "Eff"),
    ("forget", "Gen"),
    ("forget", "Spe"),
    ("retain", "Eff"),
    ("retain", "Gen"),
    ("retain", "Spe"),
    ("root", "PPL"),
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory contains no files: {path}")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def git(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def compact_value(
    block: Mapping[str, Any],
    split: str,
    metric: str,
) -> float | None:
    value = block.get(metric) if split == "root" else block.get(split, {}).get(metric)
    return None if value is None else float(value)


def summarize(values: Iterable[float | None]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "sample_sd": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.mean(finite),
        "sample_sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def aggregate(rows: list[dict[str, Any]], block_name: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split, metric in METRICS:
        key = f"{split}_{metric}" if split != "root" else metric
        output[key] = summarize(
            compact_value(row[block_name], split, metric) for row in rows
        )
    return output


def same_metrics(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    for split, metric in METRICS:
        left = compact_value(a, split, metric)
        right = compact_value(b, split, metric)
        if left is None or right is None:
            if left != right:
                return False
        elif abs(left - right) > 1e-9:
            return False
    return True


def dependency_versions() -> dict[str, str]:
    names = ("numpy", "safetensors", "tokenizers", "torch", "transformers")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--release-tag",
        default="zsre-setting5e-active-repair-seeds1-10",
    )
    args = parser.parse_args()

    project = args.project_root.resolve()
    source_root = args.source_root.resolve()
    candidate_root = args.candidate_root.resolve()
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    dataset_hashes: set[str] = set()
    model_paths: set[str] = set()
    seed_provenance: list[dict[str, Any]] = []

    for seed in range(1, 11):
        source_seed = source_root / f"seed{seed}"
        candidate_seed = candidate_root / f"seed{seed}"

        result_path = candidate_seed / "zsre_results.json"
        config_path = candidate_seed / "config_used.json"
        sampled_path = candidate_seed / "sampled_case_ids.json"
        if not sampled_path.is_file():
            sampled_path = source_seed / "sampled_case_ids.json"
        source_config_path = source_seed / "config_used.json"

        result = load_json(result_path)
        repair = result["repair"]
        setting5 = result["setting5e"]
        candidate = result["active_candidate"]
        selected = result["selected"]

        if result.get("dataset") != "ZsRE" or int(result.get("seed")) != seed:
            raise ValueError(f"Seed/dataset mismatch in {result_path}")
        if repair.get("candidate_accepted") is not False:
            raise ValueError(f"Seed {seed}: expected rejected candidate")
        if float(repair.get("selected_scale", -1)) != 0.0:
            raise ValueError(f"Seed {seed}: expected selected_scale=0")
        if compact_value(candidate, "forget", "Eff") != 0.0:
            raise ValueError(f"Seed {seed}: candidate Eff is not zero")
        if compact_value(candidate, "forget", "Gen") != 0.0:
            raise ValueError(f"Seed {seed}: candidate Gen is not zero")
        if not same_metrics(selected, setting5):
            raise ValueError(f"Seed {seed}: selected is not Setting 5e fallback")

        run_config = load_json(config_path)
        sampled = load_json(sampled_path)
        source_config = load_json(source_config_path)

        if int(run_config.get("retain_calibration_num", -1)) != 384:
            raise ValueError(f"Seed {seed}: repair calibration is not 384")
        if int(run_config.get("seed", -1)) != seed:
            raise ValueError(f"Seed {seed}: config seed mismatch")
        if int(sampled.get("seed", -1)) != seed:
            raise ValueError(f"Seed {seed}: sampled-case seed mismatch")

        dataset_hash = str(result["zsre_sha256"])
        if sampled.get("zsre_sha256") != dataset_hash:
            raise ValueError(f"Seed {seed}: ZsRE dataset hash mismatch")
        dataset_hashes.add(dataset_hash)

        model_path = str(source_config.get("model_path"))
        model_paths.add(model_path)

        source_checkpoint = source_seed / "setting5e" / "checkpoint"
        candidate_checkpoint = candidate_seed / "active_candidate_checkpoint"
        selected_checkpoint = candidate_seed / "selected_checkpoint"

        for checkpoint in (
            source_checkpoint,
            candidate_checkpoint,
            selected_checkpoint,
        ):
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            if not (checkpoint / "model.safetensors").is_file():
                raise FileNotFoundError(checkpoint / "model.safetensors")

        failed_gates = [
            name
            for name, check in repair["official_metric_gates"]["checks"].items()
            if not bool(check["passed"])
        ]

        rows.append(
            {
                "seed": seed,
                "setting5e": setting5,
                "raw_active_candidate": candidate,
                "final_selected": selected,
                "candidate_scale": repair.get("candidate_scale"),
                "selected_scale": repair.get("selected_scale"),
                "candidate_accepted": False,
                "failed_gates": failed_gates,
                "full_gate_report": repair["official_metric_gates"],
            }
        )

        seed_provenance.append(
            {
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "repair_config_path": str(config_path),
                "repair_config_sha256": sha256_file(config_path),
                "source_config_path": str(source_config_path),
                "source_config_sha256": sha256_file(source_config_path),
                "sampled_case_ids_path": str(sampled_path),
                "sampled_case_ids_sha256": sha256_file(sampled_path),
                "source_setting5_checkpoint": str(source_checkpoint),
                "source_setting5_model_sha256": sha256_file(
                    source_checkpoint / "model.safetensors"
                ),
                "source_setting5_checkpoint_tree_sha256": sha256_directory(
                    source_checkpoint
                ),
                "raw_candidate_checkpoint": str(candidate_checkpoint),
                "raw_candidate_model_sha256": sha256_file(
                    candidate_checkpoint / "model.safetensors"
                ),
                "raw_candidate_checkpoint_tree_sha256": sha256_directory(
                    candidate_checkpoint
                ),
                "final_selected_checkpoint": str(selected_checkpoint),
                "final_selected_model_sha256": sha256_file(
                    selected_checkpoint / "model.safetensors"
                ),
                "final_selected_checkpoint_tree_sha256": sha256_directory(
                    selected_checkpoint
                ),
            }
        )

    if len(dataset_hashes) != 1:
        raise ValueError(f"Multiple ZsRE dataset hashes: {sorted(dataset_hashes)}")
    if len(model_paths) != 1:
        raise ValueError(f"Multiple base-model paths: {sorted(model_paths)}")

    commit = git(project, "rev-parse", "HEAD")
    status = git(project, "status", "--porcelain")
    tag_commit = git(project, "rev-list", "-n", "1", args.release_tag)

    method_files = (
        "scripts/zsre_zero_unlearn_official_eval.py",
        "scripts/zsre_gagd_setting5e_active_repair.py",
        "scripts/zsre_bf16_safe_active_repair_v2.py",
        "scripts/run_zsre_bf16_safe_active_repair_v2.sh",
        "scripts/aggregate_zsre_gagd_results.py",
    )

    payload = {
        "schema_version": 2,
        "record_id": "zsre-official-setting5e-active-repair-seeds1-10-v2",
        "status": "RAW_CANDIDATES_REJECTED_FINAL_SELECTED_SETTING5E",
        "dataset": {
            "name": "ZsRE",
            "source_url": (
                "https://memit.baulab.info/data/dsets/zsre_mend_eval.json"
            ),
            "sha256": next(iter(dataset_hashes)),
            "sampling": (
                "seeded random.sample; forget sampled from second half before "
                "retain sampled from first half"
            ),
            "forget_num": 50,
            "retain_num": 1000,
            "seeds": list(range(1, 11)),
        },
        "model": {
            "id": "meta-llama/Llama-3.2-3B-Instruct",
            "local_snapshot_path": next(iter(model_paths)),
            "snapshot_revision_from_path": Path(next(iter(model_paths))).name,
        },
        "protocol_status": (
            "native_data_and_metrics_but_evaluation_conditioned_repair"
        ),
        "protocol_status_reason": (
            "Official ZsRE correctness and metric evidence participates in "
            "active-case and candidate-scale selection."
        ),
        "reporting": {
            "raw_active_candidate": (
                "diagnostic only; Eff/Gen reached zero but strict acceptance "
                "was 0/10"
            ),
            "final_selected": (
                "algorithm output after strict-gate rollback; Setting 5e for "
                "all seeds"
            ),
            "main_table_rule": (
                "Use final_selected for any final selected-checkpoint table."
            ),
            "diagnostic_table_rule": (
                "Raw candidate values require an explicit rejected and "
                "evaluation-conditioned label."
            ),
        },
        "aggregate_sample_sd": {
            "raw_active_candidate": aggregate(rows, "raw_active_candidate"),
            "final_selected": aggregate(rows, "final_selected"),
            "setting5e": aggregate(rows, "setting5e"),
            "strict_accepted_seeds": 0,
            "evaluated_seeds": 10,
        },
        "per_seed": rows,
        "provenance": {
            "repository_commit": commit,
            "repository_dirty": bool(status),
            "repository_status_porcelain": (
                [] if not status else status.splitlines()
            ),
            "original_release_tag": args.release_tag,
            "original_release_tag_commit": tag_commit,
            "dependency_versions_at_finalization": dependency_versions(),
            "method_file_sha256": {
                relative: sha256_file(project / relative)
                for relative in method_files
            },
            "per_seed": seed_provenance,
        },
        "scientific_conclusion": (
            "All ten raw active-repair candidates reached Eff=0 and Gen=0, "
            "but all ten failed at least one predefined strict relative-utility "
            "gate. Therefore the final selected checkpoints are the Setting 5e "
            "fallbacks; the zero-Eff/Gen candidate table is diagnostic, not the "
            "final method result."
        ),
    }

    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    def stat(block: str, key: str) -> str:
        value = payload["aggregate_sample_sd"][block][key]
        return f"{value['mean']:.6f} ± {value['sample_sd']:.6f}"

    lines = [
        "# ZsRE Setting 5e + active repair — authoritative v2 record",
        "",
        "## Final selected checkpoints",
        "",
        "| Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "|---:|---:|---:|---:|",
        (
            f"| {stat('final_selected', 'forget_Eff')} | "
            f"{stat('final_selected', 'forget_Gen')} | "
            f"{stat('final_selected', 'forget_Spe')} | "
            f"{stat('final_selected', 'PPL')} |"
        ),
        "",
        (
            "All 10 selected checkpoints are Setting 5e fallbacks because "
            "strict active-repair acceptance was 0/10."
        ),
        "",
        "## Raw active-repair candidates — rejected diagnostic",
        "",
        "| Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "|---:|---:|---:|---:|",
        (
            f"| {stat('raw_active_candidate', 'forget_Eff')} | "
            f"{stat('raw_active_candidate', 'forget_Gen')} | "
            f"{stat('raw_active_candidate', 'forget_Spe')} | "
            f"{stat('raw_active_candidate', 'PPL')} |"
        ),
        "",
        (
            "These candidates are evaluation-conditioned and failed one or "
            "more predefined strict relative-utility gates on every seed. "
            "They must not be presented as final selected checkpoints."
        ),
        "",
        f"Dataset SHA-256: `{payload['dataset']['sha256']}`",
        "",
        f"Repository commit: `{commit}`",
        "",
        (
            "The JSON companion contains per-seed gate reports, run-config "
            "hashes, sampled-case hashes, and source/candidate/selected "
            "checkpoint hashes."
        ),
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
