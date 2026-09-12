#!/usr/bin/env python3
"""Bounded warm start for a saved fact-association embedding checkpoint.

This is intentionally NOT an exact optimizer resume: the parent artifact stores
association rows and frozen routing state but not Adam moments.  We warm-start
the exact saved rows and create fresh per-row Adam optimizers while preserving
the registered objective, threshold, learning rate, trust-radius schedule, and
training-visible examples.

No MCF benchmark file is opened by this continuation process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from run_static_overlap_mlp_pilot import emit
from static_overlap_data import Example
from static_overlap_extended_tokens_v2 import routed_metrics, train_row_wise
from static_overlap_fact_association_embeddings import (
    FactAssociationBank,
    FactAssociationEditor,
    make_unknown_examples,
)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def load_examples(path):
    return [
        Example(**row)
        for row in json.loads(Path(path).read_text())
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", required=True)
    parser.add_argument("--preflight-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--max-training-seconds", type=float, default=3600.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    parent = Path(args.parent_run).resolve()
    output = Path(args.output_dir).resolve()
    preflight_path = Path(args.preflight_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite continuation output: {output}")

    manifest_path = parent / "association_manifest.json"
    artifact_path = parent / "fact_association_embeddings.pt"
    examples_path = parent / "association_examples.json"
    report_path = parent / "training_report.json"
    for path in (manifest_path, artifact_path, examples_path, report_path, preflight_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (parent / "official_mcf_eval.json").exists():
        raise RuntimeError(
            "Parent run already contains official evaluation output; refusing "
            "to use held-out-informed continuation."
        )

    preflight = json.loads(preflight_path.read_text())
    if bool(preflight.get("official_evaluation_opened", True)):
        raise RuntimeError("Preflight is not training-safe")
    if not preflight.get("prefix_invariance", {}).get("passed", False):
        raise RuntimeError("Prompt-prefix routing invariance preflight did not pass")
    if Path(preflight.get("run_dir", "")).resolve() != parent:
        raise RuntimeError("Preflight belongs to a different parent run")

    manifest = json.loads(manifest_path.read_text())
    parent_report = json.loads(report_path.read_text())
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    examples = load_examples(examples_path)

    facts_manifest = [row["id"] for row in manifest["facts"]]
    facts_artifact = [row["id"] for row in artifact["facts"]]
    if facts_manifest != facts_artifact or len(facts_artifact) != 50:
        raise RuntimeError("Parent manifest/artifact fact order mismatch")
    if int(artifact["layer"]) != int(manifest["plan"]["layer"]):
        raise RuntimeError("Parent layer mismatch")
    if tuple(artifact["rows"].shape) != tuple(artifact["keys"].shape):
        raise RuntimeError("Parent rows/keys shape mismatch")

    plan = dict(manifest["plan"])
    plan["radius_schedule"] = tuple(
        tuple(value) for value in plan["radius_schedule"]
    )
    if args.steps <= 0 or args.steps % len(facts_artifact) != 0:
        raise ValueError("Continuation steps must be a positive number of complete 50-fact sweeps")
    if int(plan["check_every"]) % len(facts_artifact) != 0:
        raise ValueError("Parent check_every must end on complete fact sweeps")
    plan["steps"] = int(args.steps)
    plan["max_training_seconds"] = float(args.max_training_seconds)
    plan["log_phase"] = "fact_association_embedding_warm_start"
    plan["natural_prompt_behavior"] = (
        "prompt-prefix-only hierarchical association routing; "
        "warm-started independent fact vectors"
    )

    output.mkdir(parents=True, exist_ok=False)

    model_path = Path(manifest["model_path"]).resolve()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    bank = FactAssociationBank(
        base_model=model,
        layer=int(artifact["layer"]),
        keys=artifact["keys"],
        thresholds=artifact["thresholds"],
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        rows=artifact["rows"],
    )
    editor = FactAssociationEditor(model, bank)
    fact_to_row = {
        fact["id"]: index for index, fact in enumerate(artifact["facts"])
    }
    answer_map = {example.id: example for example in examples}
    unknown_map = make_unknown_examples(
        examples,
        tokenizer,
        int(plan["max_length"]),
        str(plan["unknown_completion"]),
    )

    # Recompute the parent state with the corrected prompt-only runtime before
    # spending a single continuation update.
    parent_recomputed = routed_metrics(
        editor.model,
        answer_map,
        unknown_map,
        float(plan["target_probability"]),
    )
    (output / "parent_recomputed_metrics.json").write_text(
        json.dumps(parent_recomputed, indent=2, allow_nan=False) + "\n"
    )
    emit(
        phase="warm_start_parent_recomputed",
        train=parent_recomputed["train"],
        development=parent_recomputed["development"],
    )

    provenance = {
        "kind": "fact_association_embedding_bounded_warm_start_v1",
        "parent_run": str(parent),
        "parent_artifact": str(artifact_path),
        "parent_artifact_sha256": sha256_file(artifact_path),
        "parent_training_report_sha256": sha256_file(report_path),
        "parent_selected_step": parent_report.get("best_step"),
        "parent_stop_reason": parent_report.get("stop_reason"),
        "warm_start": True,
        "exact_optimizer_resume": False,
        "optimizer_state": "fresh per-row Adam",
        "continuation_row_steps_budget": int(args.steps),
        "continuation_sweeps_budget": int(args.steps) // len(facts_artifact),
        "max_training_seconds": float(args.max_training_seconds),
        "git_head": git_head(),
        "official_evaluation_opened": False,
        "training_examples_reused_from_parent": str(examples_path),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "plan": {
            **plan,
            "radius_schedule": [list(value) for value in plan["radius_schedule"]],
        },
    }
    (output / "warm_start_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    # Preserve the exact training-visible example payload for provenance.
    (output / "association_examples.json").write_text(examples_path.read_text())
    continuation_manifest = dict(manifest)
    continuation_manifest.update({
        "parent_run": str(parent),
        "parent_artifact_sha256": provenance["parent_artifact_sha256"],
        "warm_start": True,
        "exact_optimizer_resume": False,
        "runtime_subject_scan_scope": "prompt_prefix_only",
        "official_evaluation_opened": False,
        "plan": provenance["plan"],
    })
    (output / "association_manifest.json").write_text(
        json.dumps(continuation_manifest, indent=2, allow_nan=False) + "\n"
    )

    report = train_row_wise(
        editor=editor,
        original_examples=examples,
        routed_answer=answer_map,
        routed_unknown=unknown_map,
        fact_to_row=fact_to_row,
        plan=plan,
        output=output,
    )
    final_metrics = routed_metrics(
        editor.model,
        answer_map,
        unknown_map,
        float(plan["target_probability"]),
    )
    selected_artifact = editor.artifact()
    final_artifact_path = output / "fact_association_embeddings.pt"
    torch.save(selected_artifact, final_artifact_path)

    # Reload the serialized selected state into the same unchanged base model
    # after removing the training hook, then reproduce authored metrics.
    bank.close()
    disk_artifact = torch.load(
        final_artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    reload_bank = FactAssociationBank(
        base_model=model,
        layer=int(disk_artifact["layer"]),
        keys=disk_artifact["keys"],
        thresholds=disk_artifact["thresholds"],
        subject_patterns=disk_artifact["subject_patterns"],
        facts=disk_artifact["facts"],
        rows=disk_artifact["rows"],
    )
    for row in reload_bank.rows:
        row.requires_grad_(False)
    from static_overlap_fact_association_embeddings import AssociationCausalLM
    reloaded = AssociationCausalLM(model, reload_bank)
    reload_metrics = routed_metrics(
        reloaded,
        answer_map,
        unknown_map,
        float(plan["target_probability"]),
    )

    report.update({
        "method": manifest["method"],
        "warm_start_provenance": provenance,
        "parent_recomputed_metrics": parent_recomputed,
        "final_metrics": final_metrics,
        "reload_metrics": reload_metrics,
        "reload_metrics_equal_selected_metrics": reload_metrics == final_metrics,
        "parent_selected_step": parent_report.get("best_step"),
        "continuation_selected_step": report.get("best_step"),
        "cumulative_selected_row_steps": (
            int(parent_report.get("best_step") or 0)
            + int(report.get("best_step") or 0)
        ),
        "artifact_sha256": sha256_file(final_artifact_path),
        "official_evaluation_started": False,
    })
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    emit(
        status="fact_association_embedding_warm_start_complete",
        stop_reason=report["stop_reason"],
        continuation_best_step=report["best_step"],
        cumulative_selected_row_steps=report["cumulative_selected_row_steps"],
        final_metrics=final_metrics,
        reload_metrics_equal=report["reload_metrics_equal_selected_metrics"],
        artifact=str(final_artifact_path),
        official_evaluation_started=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
