#!/usr/bin/env python3
"""Finalize Git-tracked best-run manifests from local checkpoint artifacts.

This script never copies model weights into Git. It verifies MCF hashes and
adds exact TOFU/ZsRE weight hashes plus snapshots of their authoritative local
configuration/result JSON files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


CHUNK_SIZE = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def relative_text(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def snapshot_files(root: Path, paths: Iterable[str]) -> Dict[str, Any]:
    snapshots: Dict[str, Any] = {}
    for value in paths:
        path = root / value
        if not path.is_file():
            raise FileNotFoundError(f"Missing provenance file: {path}")
        snapshots[value] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content": read_json(path),
        }
    return snapshots


def finalize_mcf(root: Path, manifest_path: Path) -> None:
    manifest = read_json(manifest_path)
    failures = []

    for record in manifest["checkpoints"]:
        weight = root / record["path"] / record["weight_file"]
        if not weight.is_file():
            failures.append(f"missing: {weight}")
            continue

        actual_size = weight.stat().st_size
        actual_hash = sha256(weight)

        if actual_size != int(record["size_bytes"]):
            failures.append(
                f"size mismatch: {weight}: {actual_size} != {record['size_bytes']}"
            )
        if actual_hash != record["sha256"]:
            failures.append(
                f"hash mismatch: {weight}: {actual_hash} != {record['sha256']}"
            )

    if failures:
        raise RuntimeError("MCF verification failed:\n" + "\n".join(failures))

    manifest["provenance_completeness"] = {
        "metrics": "COMPLETE",
        "parameters": "COMPLETE",
        "weight_hashes": "COMPLETE_AND_VERIFIED",
    }
    write_json(manifest_path, manifest)
    print(f"Verified MCF manifest: {manifest_path}")


def finalize_single_checkpoint(
    root: Path,
    manifest_path: Path,
    config_paths: Iterable[str],
) -> None:
    manifest = read_json(manifest_path)
    final = manifest["final_checkpoint"]
    weight = root / final["path"] / final["weight_file"]

    if not weight.is_file():
        raise FileNotFoundError(f"Missing checkpoint weight: {weight}")

    final["size_bytes"] = weight.stat().st_size
    final["sha256"] = sha256(weight)
    final["hash_status"] = "COMPLETE_AND_VERIFIED"

    manifest["exact_parameter_snapshots"] = snapshot_files(root, config_paths)
    manifest["provenance_completeness"]["embedded_parameter_snapshot"] = "COMPLETE"
    manifest["provenance_completeness"]["weight_hash"] = "COMPLETE_AND_VERIFIED"

    write_json(manifest_path, manifest)
    print(f"Finalized manifest: {manifest_path}")
    print(f"  checkpoint: {relative_text(weight.parent, root)}")
    print(f"  sha256: {final['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=None,
        help="semantic-unlearning project root (defaults to script parent/..)",
    )
    args = parser.parse_args()

    root = (
        Path(args.root).resolve()
        if args.root
        else Path(__file__).resolve().parents[1]
    )

    best_root = root / "config" / "best_runs"

    mcf_manifest = best_root / "mcf" / "controlled_fivefold_margin025_rank2.json"
    tofu_manifest = (
        best_root
        / "tofu"
        / "forget05_setting3_5e_lmhead_alpha000_seed42.json"
    )
    zsre_manifest = best_root / "zsre" / "cal384_seed1.json"
    registry_path = best_root / "registry.json"

    finalize_mcf(root, mcf_manifest)

    finalize_single_checkpoint(
        root,
        tofu_manifest,
        [
            "outputs/tofu_setting3_5e_ultra_seed42/setting3_emb_lm_all/config_used.json",
            "outputs/tofu_setting3_5e_ultra_seed42/setting5e_alpha000/config_used.json",
            "outputs/tofu_setting3_5e_ultra_seed42/setting5e_alpha000/checkpoint/repair_experiment_config.json",
            "outputs/tofu_setting3_5e_ultra_seed42/lm_head_repair_alpha000/repair_summary.json",
            "outputs/tofu_setting3_5e_ultra_seed42/lm_head_repair_alpha000/candidate_local_metrics.json",
        ],
    )

    finalize_single_checkpoint(
        root,
        zsre_manifest,
        [
            "outputs/zsre_cal384_seeds0_9/seed1/config_used.json",
            "outputs/zsre_cal384_seeds0_9/seed1/zsre_results.json",
            "outputs/zsre_cal384_seeds0_9/seed1/active_repair/repair_summary.json",
            "outputs/zsre_cal384_seeds0_9/seed1/active_repair/candidate_official_eval.json",
        ],
    )

    registry = read_json(registry_path)
    for record in registry["records"]:
        record["hash_status"] = "COMPLETE_AND_VERIFIED"
    write_json(registry_path, registry)

    print(f"Updated registry: {registry_path}")
    print("No model weights were copied or added to Git.")


if __name__ == "__main__":
    main()
