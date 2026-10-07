#!/usr/bin/env python3
"""MCF seed 1: one shared residual vs the existing 50-row IDK+EOS run.

Reuses the reference router and training hyperparameters, starts the shared
vector at zero, and evaluates both artifacts under the same current evaluator.
No baseline artifacts are modified. Use --dry-run for CPU-only preflight.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

import torch

from linear_router import ARCHITECTURE

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "outputs/compressed_multiseed_idk_eos_v1/mcf/seed1/L19/full"
TRAIN_KEYS = (
    "lr", "scale_lr", "batch_facts", "epochs", "eval_every",
    "max_training_seconds", "unknown_weight", "unknown_completion", "clip", "seed",
)
ROUTER_KEYS = (
    "architecture", "layer", "router_weight", "router_bias", "feature_mean",
    "feature_components", "threshold", "subject_patterns", "facts",
    "ambiguity_margin", "gate_mode", "per_head_thresholds", "bias_calibration", "head_index",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_reference(artifact, manifest, report):
    """Reject an unmatched baseline before allocating a model/GPU."""
    if artifact.get("architecture") != ARCHITECTURE:
        raise ValueError("Reference must use the existing linear-classifier router")
    if int(artifact["layer"]) != 19:
        raise ValueError("This experiment fixes read/write layer at 19")
    if len(artifact["facts"]) != 50 or artifact["rows"].shape[0] != 50:
        raise ValueError("Reference must contain exactly 50 MCF associations")
    if len({f["id"] for f in artifact["facts"]}) != 50:
        raise ValueError("Reference must contain 50 unique association IDs")
    if not manifest.get("mcf_path") or int(manifest.get("sampling", {}).get("seed", -1)) != 1:
        raise ValueError("Reference must explicitly record the MCF seed-1 split")
    if report.get("value_mode") != "full" or report.get("training_route") != "router":
        raise ValueError("Reference must be the joint full-row trainer under actual router routing")
    hp = report.get("hyperparameters", {})
    missing = [k for k in (*TRAIN_KEYS, "unknown_eos") if k not in hp]
    if missing:
        raise ValueError(f"Reference training report lacks hyperparameters: {missing}")
    if hp["unknown_eos"] is not True or float(hp["unknown_weight"]) <= 0:
        raise ValueError("Reference must have IDK loss enabled and unknown_eos=true")
    if str(hp["unknown_completion"]).strip() != "I don't know.":
        raise ValueError("Reference abstention text must be I don't know.")
    if int(hp["seed"]) != 1:
        raise ValueError("Reference optimization seed must also be 1")
    if report.get("training_coverage", {}).get("facts_trained") != 50:
        raise ValueError("Reference must have routed training views for all 50 facts")
    if not report.get("best_epoch"):
        raise ValueError("Reference has no selected trained checkpoint")
    return {k: hp[k] for k in TRAIN_KEYS}


def same_value(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    return a == b


def validate_shared(reference, shared, report):
    changed = [k for k in ROUTER_KEYS if not same_value(reference.get(k), shared.get(k))]
    if changed:
        raise ValueError(f"Router or association identity changed: {changed}")
    rows = shared["rows"]
    if rows.shape != reference["rows"].shape or not torch.isfinite(rows).all():
        raise ValueError("Shared rows have the wrong shape or nonfinite values")
    if not torch.equal(rows, rows[:1].expand_as(rows)):
        raise ValueError("All 50 exported rows must be exactly the same vector")
    compact = shared.get("compressed_values", {})
    vector = compact.get("compact_state", {}).get("shared_vector")
    if compact.get("mode") != "shared" or vector is None or not torch.equal(rows[0], vector):
        raise ValueError("Export does not reconstruct the trained shared vector")
    if compact.get("storage", {}).get("total_floats") != rows.shape[1]:
        raise ValueError("Shared residual must have exactly d trainable values")
    if report.get("training_coverage", {}).get("facts_trained") != 50 or not report.get("best_epoch"):
        raise ValueError("Shared run did not train all 50 facts or select a trained checkpoint")


def comparison_rows(official, generations, hidden_size):
    """Official scores retain their original units; generation rates are percent."""
    result = []
    for label, evaluation in official.items():
        row = {"arm": label, "residual_parameters": hidden_size * (50 if label == "full_50" else 1),
               "Eff": evaluation["forget"]["Eff"], "Gen": evaluation["forget"]["Gen"],
               "Spe": evaluation["forget"]["Spe"], "retain_Eff": evaluation["retain"]["Eff"],
               "PPL": evaluation["forget_PPL"]}
        for group in ("rewrite", "paraphrase", "neighborhood", "retain"):
            examples = [r for r in generations if r["group"] == group]
            n = len(examples)
            if not n:
                raise ValueError(f"Generation evaluation is missing {group}")
            row[f"{group}_n"] = n
            row[f"{group}_answer_pct"] = 100 * sum(r["runs"][label]["has_answer"] for r in examples) / n
            row[f"{group}_abstain_pct"] = 100 * sum(r["runs"][label]["abstains"] for r in examples) / n
            row[f"{group}_exact_idk_pct"] = 100 * sum(
                " ".join(r["runs"][label]["output"].strip().casefold().split()) == "i don't know."
                for r in examples) / n
            row[f"{group}_route_pct"] = 100 * sum(
                r["runs"][label]["routed_row"] is not None for r in examples) / n
        result.append(row)
    for row in generations:
        if row["runs"]["full_50"]["routed_row"] != row["runs"]["shared_1"]["routed_row"]:
            raise ValueError("Reference/shared generation route mismatch")
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run-dir", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mcf_shared_vector_seed1_idk_eos")
    p.add_argument("--wikidata-dir", type=Path, default=ROOT / "data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if a.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    reference, output = a.reference_run_dir.resolve(), a.output_dir.resolve()
    if output == reference or output in reference.parents or reference in output.parents:
        p.error("Use a separate output directory outside the reference run")
    artifact = torch.load(reference / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = read_json(reference / "association_manifest.json")
    report = read_json(reference / "training_report.json")
    hp = validate_reference(artifact, manifest, report)
    for key in ("model_path", "mcf_path"):
        if not Path(manifest[key]).exists():
            raise FileNotFoundError(f"Reference {key} is missing: {manifest[key]}")
    shared = output / "shared"
    flags = ["--device", a.device] + (["--local-files-only"] if a.local_files_only else [])
    train = [sys.executable, "-u", str(ROOT / "scripts/train_mcf_compressed_bank.py"),
             "--router-dir", str(reference), "--output-dir", str(shared),
             "--value-mode", "shared", "--training-route", "router", "--unknown-eos", *flags]
    for key, value in hp.items():
        train.extend(["--" + key.replace("_", "-"), str(value)])
    jobs = [(shared / "training_report.json", train)]
    for label, run in (("full_50", reference), ("shared_1", shared)):
        dest = output / f"official_{label}.json"
        jobs.append((dest, [sys.executable, "-u", str(ROOT / "scripts/evaluate_static_overlap_fact_association_embeddings_official.py"),
                           "--run-dir", str(run), "--mcf-path", manifest["mcf_path"],
                           "--wikidata-dir", str(a.wikidata_dir.resolve()), "--seed", "1",
                           "--dtype", a.dtype, "--out", str(dest), *flags]))
    generation = output / "generations"
    jobs.append((generation.with_suffix(".jsonl"), [sys.executable, "-u", str(ROOT / "scripts/generate_after_unlearning.py"),
                 "--run-dirs", str(reference), str(shared), "--labels", "full_50", "shared_1",
                 "--mcf-path", manifest["mcf_path"], "--groups", "rewrite", "paraphrase", "neighborhood", "retain",
                 "--neighborhood-per-fact", "10", "--retain", "0",
                 "--max-new-tokens", str(a.max_new_tokens), "--dtype", a.dtype, "--out", str(generation), *flags]))
    config = {
        "dataset": "MCF", "seed": 1, "forget_num": 50, "layer": 19,
        "reference_run_dir": str(reference), "hyperparameters": {**hp, "unknown_eos": True},
        "reference_sha256": {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                             for name in ("fact_association_embeddings.pt", "association_manifest.json", "training_report.json")},
        "commands": [cmd for _, cmd in jobs],
        "training_objective": "joint_forget_hinge_plus_IDK_and_EOS_NLL",
        "initialization": "zero; reference residual vectors are not reused",
        "training_coverage": "same correctly routed views as the full-row baseline",
        "inference": "every active route gets the same vector; no per-fact scaling",
        "evaluation": "all official seed-1 forget views, 10 neighbors/fact, all 1000 retain prompts; PPL enabled",
    }
    for _, cmd in jobs:
        print(shlex.join(cmd), flush=True)
    if a.dry_run:
        print("Preflight passed; no model loaded and no files written.")
        return 0
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "experiment_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("Existing experiment has different inputs/settings; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for index, (marker, command) in enumerate(jobs):
        if not marker.exists():
            if index == 0 and shared.exists():
                raise FileExistsError(f"Incomplete training directory: {shared}; use a new output directory")
            subprocess.run(command, cwd=ROOT, check=True)
        if index == 0:
            shared_artifact = torch.load(shared / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
            shared_report = read_json(marker)
            validate_shared(artifact, shared_artifact, shared_report)
            if shared_report["hyperparameters"] != report["hyperparameters"]:
                raise ValueError("Shared and full-row training hyperparameters differ")
            if shared_report["views_excluded_unrouted"] != report["views_excluded_unrouted"]:
                raise ValueError("Shared and full-row training coverage differs")
    with generation.with_suffix(".jsonl").open() as handle:
        header = json.loads(next(handle))
        generations = [json.loads(line) for line in handle if line.strip()]
    official = {label: read_json(output / f"official_{label}.json") for label in ("full_50", "shared_1")}
    rows = comparison_rows(official, generations, artifact["rows"].shape[1])
    (output / "comparison.json").write_text(json.dumps({
        "rows": rows, "generation_metadata": header["meta"], "identical_routes_verified": True,
        "units": "Official metrics unchanged; *_pct are percentages with *_n denominators.",
    }, indent=2, allow_nan=False) + "\n")
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)
    print(f"Comparison saved to {output / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
