#!/usr/bin/env python3
"""Fresh, frozen-backbone answer-row regression; cache once, scan a finite grid."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from static_overlap_cached_head import (RetainMetricSolver, augment_contexts,
    audit_prefix_conflicts, cache_head, cached_measure, development_cache,
    prepare_independent_head, summarize)
from static_overlap_core import StaticEditor
from static_overlap_data import encode_bundle, endpoint_rows, load_bundle, text_fingerprints
from static_overlap_training import TrainConfig, export_verified, measure


ARCHITECTURE = "static_overlap_cached_head_v1"


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-untied-head", action="store_true",
                   help="Explicitly copy a shared LM head; freeze embeddings and save an untied checkpoint")
    p.add_argument("--training-only", action="store_true")
    p.add_argument("--development-protocol", help="Frozen protocol.json; requires a passing source parity audit")
    p.add_argument("--no-context-augmentation", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--target-probability", type=float, default=1e-6)
    p.add_argument("--nll-safety-margin", type=float, default=0.01)
    p.add_argument("--kl-safety-margin", type=float, default=0.002)
    p.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1, 1.0])
    p.add_argument("--ridges", type=float, nargs="+", default=[0.0001, 0.01])
    p.add_argument("--strengths", type=float, nargs="+", default=[0.25, 0.5, 1, 2, 4, 8, 16, 24, 32])
    args = p.parse_args(argv)
    for key in ("model_path", "training_bundle", "output_dir"):
        if not getattr(args, key).strip():
            p.error(f"--{key.replace('_', '-')} is empty; set its shell variable")
    if not Path(args.training_bundle).is_file():
        p.error("--training-bundle must name an existing file")
    if Path(args.output_dir).exists():
        p.error("--output-dir already exists; use a fresh directory")
    if args.max_length < 2:
        p.error("--max-length must be at least 2")
    for key in ("taus", "ridges", "strengths"):
        if any(not math.isfinite(x) or x < 0 or (key != "taus" and x == 0)
               for x in getattr(args, key)):
            p.error(f"Invalid --{key}")
    return args


@torch.no_grad()
def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args(argv)
    started = time.perf_counter()
    # Scientific limits are fixed. Only stricter internal margins are configurable.
    config = TrainConfig(target_probability=args.target_probability,
                         retain_nll_budget=0.05, retain_kl_budget=0.01,
                         retain_nll_safety_margin=args.nll_safety_margin,
                         retain_kl_safety_margin=args.kl_safety_margin,
                         fresh_start_only=True, select_best_valid_checkpoint=True, seed=args.seed)
    config.validate()
    bundle, facts, source_hash = load_bundle(args.training_bundle)
    protocol = None
    if args.development_protocol:
        from freeze_static_overlap_development import claim_training, check_parity, load_protocol
        protocol = load_protocol(args.development_protocol)
        check_parity(protocol["source_run"], protocol["files"]["parity_audit"]["path"])
        if source_hash != protocol["files"]["source_bundle"]["sha256"]:
            raise ValueError("Use the exact source bundle frozen by the development protocol")
        if not args.no_context_augmentation:
            raise ValueError("Development mode reuses existing augmented features; use --no-context-augmentation")
        plan = {k: getattr(args, k) for k in protocol["experiment"]}
        claim_training(args.development_protocol, args.output_dir, plan)
    elif not args.no_context_augmentation:
        bundle = augment_contexts(bundle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    bundle_file = output / ("source_bundle.json" if protocol else "training_bundle.json")
    write_json(bundle_file, bundle)
    bundle_hash = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
    torch.manual_seed(args.seed)
    emit({"phase": "load_original_base", "model": args.model_path,
          "method": ARCHITECTURE, "fresh_start": True})
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,
                                              local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32,
             local_files_only=args.local_files_only, attn_implementation="eager").to(args.device).eval()
    source_model_config = model.config.to_dict()
    examples = encode_bundle(bundle, tokenizer, args.max_length, "I don't know.")
    separation = prepare_independent_head(model, next(e for e in examples if e.split == "train"),
                                          allow_untie=args.allow_untied_head)
    write_json(output / "head_preparation.json", separation)
    emit({"phase": "head_preparation", **separation})
    _, rows = endpoint_rows(facts, examples, tokenizer, abstention_enabled=False)
    conflicts = audit_prefix_conflicts(examples)
    write_json(output / "prefix_conflicts.json", conflicts)
    emit({"phase": "cache_start", "examples": len(examples), "head_rows": len(rows),
          "exact_training_token_conflicts": len(conflicts)})
    if protocol:
        from audit_static_overlap_cached_head import load_saved_cache
        cache, _ = load_saved_cache(Path(protocol["source_run"]), tokenizer, args.device, args.max_length)
        cache = development_cache(cache)
        examples = cache.examples
        emit({"phase": "reuse_original_base_cache", "continuation_delta_applied": False,
              "development_preservation_examples": sum(e.split == "development" for e in examples),
              "final_test_features_loaded": False})
        write_json(output / "dataset_membership.json", [{"id": e.id, "split": e.split,
                   "role": e.role, "fact_id": e.fact_id} for e in examples])
        write_json(output / "development_examples.json", [asdict(e) for e in examples])
        write_json(output / "protocol.json", protocol)
    else:
        cache = cache_head(model, examples, rows, emit)
    # Save compact sufficient statistics, never full vocabulary caches or GPU tensors.
    torch.save({key: getattr(cache, key).cpu() for key in
                ("hidden", "logp_rows", "logp_other", "target_nll", "target_row", "owners", "rows")},
               output / "head_cache.pt")
    emit({"phase": "solve_retention_metric", "elapsed_seconds": time.perf_counter()-started})
    solver = RetainMetricSolver(cache, include_development=bool(protocol))
    emit({"phase": "retention_metric_ready", **solver.diagnostics})
    zero = torch.zeros(len(rows), cache.hidden.shape[1], device=cache.hidden.device)
    base_stats = cached_measure(cache, zero)
    initial = summarize(base_stats, config)
    best, best_delta = None, None
    diagnostic, diagnostic_delta, history = None, None, []
    def selection_key(item):
        if not protocol:
            return tuple(item["score"])
        if item["training_forgetting"]["target_met"]:
            return (0, item["delta_norm"], *item["score"])
        return (1, *item["score"], item["delta_norm"])
    for tau in args.taus:
        for ridge in args.ridges:
            direction = solver.solve(tau, ridge)
            for strength in args.strengths:
                delta = direction * strength
                stats = cached_measure(cache, delta)
                if not all(math.isfinite(r[k]) for r in stats for k in ("nll", "kl")):
                    emit({"phase": "candidate_rejected", "tau": tau, "ridge": ridge,
                          "strength": strength, "reason": "nonfinite_statistics"})
                    continue
                summary = summarize(stats, config)
                item = {"candidate": len(history)+1, "tau": tau, "ridge": ridge, "strength": strength,
                        "delta_norm": float(delta.norm()), **summary}
                # A valid baseline or worsening edit must not be called a successful edit.
                improved = tuple(summary["score"]) < tuple(initial["score"])
                item["eligible"] = summary["eligible"] and improved
                if diagnostic is None or tuple(item["score"]) < tuple(diagnostic["score"]):
                    diagnostic, diagnostic_delta = item, delta.cpu().clone()
                if item["eligible"] and (best is None or selection_key(item) < selection_key(best)):
                    best, best_delta = item, delta.cpu().clone()
                history.append(item)
                with (output / "candidates.jsonl").open("a") as stream:
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                emit({"phase": "cached_candidate", "candidate": item["candidate"],
                      "tau": tau, "ridge": ridge, "strength": strength, "eligible": item["eligible"],
                      "mean_training_probability": summary["training_forgetting"]["mean_token_probability"],
                      "max_training_probability": summary["training_forgetting"]["max_token_probability"],
                      ("development_max_nll_increase" if protocol else "validation_max_nll_increase"):
                          summary["development_protection" if protocol else "validation_protection"]["max_retained_nll_increase"],
                      ("development_max_kl" if protocol else "validation_max_kl"):
                          summary["development_protection" if protocol else "validation_protection"]["max_retained_kl"],
                      "selected_candidate": best["candidate"] if best else None})

    torch.save({"rows": cache.rows.cpu(), "delta": diagnostic_delta, "candidate": diagnostic},
               output / "best_training_only_delta.pt")
    selection = {"selected_candidate": best["candidate"] if best else None,
                 "eligible_candidates": sum(r["eligible"] for r in history),
                 "score_fields": ["max_training_token_probability", "mean_training_token_probability"],
                 "validation_used_in_solve": False, "validation_forget_used_for_selection": False,
                 "official_evaluation_used_for_selection": False,
                 "validation_retention_used_for_selection": not bool(protocol),
                 "development_retention_used_for_selection": bool(protocol),
                 "reclassified_validation_retention_used_in_solve": bool(protocol),
                 "final_test_used_for_selection": False}
    report = {"method": ARCHITECTURE, "head_preparation": separation,
              "initial": initial, "checkpoint_selection": selection,
              "selected": best, "best_training_only_candidate": diagnostic,
              "solver": solver.diagnostics, "history": history,
              "elapsed_seconds": time.perf_counter()-started,
              "official_eff_gen_measured": False, "native_checkpoint_created": False}
    if protocol:
        report["development_protocol"] = protocol
        selection["rule"] = protocol["selection_rule"]
        selection.pop("validation_used_in_solve")
        selection.pop("validation_retention_used_for_selection")
        selection["development_used_in_solve"] = True
    write_json(output / "training_report.json", report)
    if best is None:
        emit({"status": "no_valid_edit", "report": str(output / "training_report.json"),
              "best_training_only_candidate": diagnostic})
        return 2

    # Factor a dense answer-row update EXACTLY using an identity factor. This
    # increases head capacity to row rank without changing other vocabulary rows.
    editor = StaticEditor(model, [], rows, {}, rank=len(rows))
    editor.rows["head"].A.copy_(torch.eye(len(rows), device=args.device))
    editor.rows["head"].B.copy_(best_delta.T.to(args.device))
    settings = {"architecture": {"rank": len(rows), "blocks": 0, "channels_per_block": 0},
                "max_length": args.max_length, "abstention": "I don't know.",
                "export_atol": 1e-4, "export_rtol": 1e-5, "training": asdict(config)}
    manifest = {"architecture": ARCHITECTURE, "settings": settings, "training_config": asdict(config),
                "model_path": args.model_path, "model_config": model.config.to_dict(),
                "source_model_config": source_model_config, "head_preparation": separation,
                "training_dtype": "float32", "deployment_dtype": "float32",
                "input_rows": [], "output_rows": rows, "shared_endpoints": False,
                "selected_channels": {}, "training_bundle_sha256": bundle_hash,
                "training_bundle_path": str(bundle_file.resolve()),
                "source_bundle_sha256": source_hash, "fresh_start": True,
                "forget_associations": [f for f in facts.values() if f["role"] == "forget"],
                "training_text_fingerprints": text_fingerprints(bundle),
                "grid": {"taus": args.taus, "ridges": args.ridges, "strengths": args.strengths},
                "solver": solver.diagnostics, "runtime_router": False, "runtime_guard": False}
    if protocol:
        manifest["development_protocol"] = protocol
        manifest["development_protocol_sha256"] = hashlib.sha256(Path(args.development_protocol).read_bytes()).hexdigest()
        manifest["development_protocol_path"] = str(Path(args.development_protocol).resolve())
        manifest["active_dataset"] = {"path": str((output / "development_examples.json").resolve()),
            "sha256": hashlib.sha256((output / "development_examples.json").read_bytes()).hexdigest(),
            "source_bundle_scope": "immutable source archive; split reclassification and exclusions defined by active_dataset"}
    write_json(output / "manifest.json", manifest)
    torch.save(editor.artifact(), output / "training_factors.pt")
    emit({"phase": "verify_selected_on_real_model", "selected_candidate": best["candidate"]})
    actual = measure(editor, examples)
    predicted = cached_measure(cache, best_delta.to(args.device))
    errors = {key: max(abs(a[key]-b[key]) for a, b in zip(actual, predicted)) for key in ("base_nll", "nll", "kl")}
    parity = all(math.isclose(a[key], b[key], abs_tol=1e-4, rel_tol=1e-5)
                 for a, b in zip(actual, predicted) for key in errors)
    actual_summary = summarize(actual, config)
    actual_base = summarize([{**r, "nll": r["base_nll"], "nll_increase": 0.0, "kl": 0.0}
                             for r in actual], config)
    actual_improved = tuple(actual_summary["score"]) < tuple(actual_base["score"])
    report.update(selected_actual=actual_summary, cache_model_parity={"passed": parity, "max_abs_errors": errors},
                  actual_training_improved=actual_improved,
                  accepted_steps=int(parity and actual_summary["eligible"] and actual_improved),
                  training_forget=[r for r in actual if r["split"] == "train" and r["role"] == "forget"])
    report["development" if protocol else "validation"] = [r for r in actual
        if r["split"] == ("development" if protocol else "validation")]
    write_json(output / "training_report.json", report)
    emit({"phase": "real_model_verification", "cache_parity": report["cache_model_parity"], **actual_summary})
    if not report["accepted_steps"]:
        emit({"status": "selected_edit_failed_real_model_checks", "report": str(output / "training_report.json")})
        return 2
    if args.training_only:
        emit({"status": "valid_factors_saved", "run": str(output), "official_eff_gen_measured": False})
        return 0
    def reload_model(path):
        return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
                  local_files_only=True, attn_implementation="eager").to(args.device).eval()
    emit({"phase": "export_merge_reload_verification"})
    exported = export_verified(editor, tokenizer, examples, config, output / "checkpoint", torch.float32,
                               reload_model, atol=1e-4, rtol=1e-5, manifest=manifest)
    report.update(native_checkpoint_created=True, elapsed_seconds=time.perf_counter()-started)
    write_json(output / "training_report.json", report)
    emit({"status": "verified_native_checkpoint", "checkpoint": str(output / "checkpoint"),
          "training_target_met": actual_summary["training_forgetting"]["target_met"],
          "all_bundle_forgetting_target_met": exported["forgetting_target_met"],
          "official_eff_gen_measured": False, "elapsed_seconds": report["elapsed_seconds"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
