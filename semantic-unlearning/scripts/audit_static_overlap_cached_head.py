#!/usr/bin/env python3
"""Audit saved head directions without re-extracting features or exporting edits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import torch

from static_overlap_cached_head import (HeadCache, RetainMetricSolver, cached_measure,
                                       feasible_strength, prepare_independent_head, summarize)
from static_overlap_core import StaticEditor
from static_overlap_data import encode_bundle, endpoint_rows, load_bundle
from static_overlap_training import TrainConfig, measure, sha256_file


@torch.no_grad()
def verify_training_candidate(run, model_path, cache, config, local_files_only):
    """Verify a rejected diagnostic edit without saving any native checkpoint."""
    from transformers import AutoModelForCausalLM

    saved = torch.load(run / "best_training_only_delta.pt", map_location="cpu", weights_only=True)
    delta = saved["delta"]
    if (not torch.equal(saved["rows"], cache.rows.cpu()) or not isinstance(delta, torch.Tensor)
            or delta.shape != (len(cache.rows), cache.hidden.shape[1]) or not torch.isfinite(delta).all()):
        raise ValueError("Training-only diagnostic delta differs from cached dimensions/rows")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32,
              local_files_only=local_files_only, attn_implementation="eager").to(cache.hidden.device).eval()
    preparation = json.loads((run / "head_preparation.json").read_text())
    prepared = prepare_independent_head(model, next(e for e in cache.examples if e.split == "train"),
                                        allow_untie=preparation["applied"])
    if prepared["source_shared_endpoints"] != preparation["source_shared_endpoints"]:
        raise ValueError("Original model sharing differs from the diagnostic run")
    editor = StaticEditor(model, [], cache.rows.cpu().tolist(), {}, len(cache.rows))
    editor.rows["head"].A.copy_(torch.eye(len(cache.rows), device=cache.hidden.device))
    editor.rows["head"].B.copy_(delta.T.to(cache.hidden.device))
    actual = measure(editor, cache.examples)
    expected = cached_measure(cache, delta.to(cache.hidden.device))
    errors = {key: max(abs(a[key]-b[key]) for a, b in zip(actual, expected))
              for key in ("base_nll", "nll", "kl")}
    parity = all(math.isfinite(a[key]) and math.isclose(a[key], b[key], abs_tol=1e-4, rel_tol=1e-5)
                 for a, b in zip(actual, expected) for key in errors)
    result = {"candidate": saved["candidate"]["candidate"], "cache_model_parity_passed": parity,
              "max_abs_errors": errors, "actual_summary": summarize(actual, config),
              "actual_rows": actual, "native_checkpoint_created": False,
              "scope": "diagnostic training-optimal delta, not an approved checkpoint"}
    del editor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_saved_cache(run, tokenizer, device="cpu", max_length=512):
    bundle, facts, bundle_hash = load_bundle(run / "training_bundle.json")
    examples = encode_bundle(bundle, tokenizer, max_length, "I don't know.")
    data = torch.load(run / "head_cache.pt", map_location="cpu", weights_only=True)
    expected_keys = {"hidden", "logp_rows", "logp_other", "target_nll", "target_row", "owners", "rows"}
    if set(data) != expected_keys or any(not isinstance(v, torch.Tensor) for v in data.values()):
        raise ValueError("Invalid cache tensor schema")
    _, rows = endpoint_rows(facts, examples, tokenizer, abstention_enabled=False)
    if data["rows"].dtype != torch.long or data["rows"].tolist() != rows:
        raise ValueError("Saved head rows differ from the supplied tokenizer/bundle")
    # Legacy caches have no tokenization manifest. Reconstruct and compare their
    # precise owner order and target-row map before interpreting any statistics.
    groups = defaultdict(list)
    for owner, e in enumerate(examples):
        groups[tuple(e.input_ids)].append((owner, e))
    owners, targets = [], []
    lookup = {token: i for i, token in enumerate(rows)}
    for entries in groups.values():
        for owner, e in entries:
            for label in e.labels[1:]:
                if label != -100:
                    owners.append(owner)
                    targets.append(lookup.get(label, -1))
    for key, expected in (("owners", owners), ("target_row", targets)):
        if data[key].dtype != torch.long or data[key].tolist() != expected:
            raise ValueError(f"Saved {key} differs from tokenizer/bundle reconstruction")
    n = len(owners)
    for key in ("hidden", "logp_rows", "logp_other", "target_nll"):
        value = data[key]
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite or non-floating cache values: {key}")
    if (data["hidden"].ndim != 2 or data["hidden"].shape[0] != n
            or data["logp_rows"].shape != (n, len(rows))
            or data["logp_other"].shape != (n,) or data["target_nll"].shape != (n,)):
        raise ValueError("Saved cache tensor dimensions differ from examples")
    mass = torch.logaddexp(data["logp_other"], data["logp_rows"].logsumexp(-1))
    if mass.abs().max() > 1e-8 or data["target_nll"].min() < -1e-8:
        raise ValueError("Saved cache does not describe normalized baseline probabilities")
    cache = HeadCache(examples, **{k: v.to(device) for k, v in data.items()})
    return cache, bundle_hash


@torch.no_grad()
def main(argv=None):
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-run", required=True)
    p.add_argument("--model-path", required=True, help="Original tokenizer path; no model weights are loaded")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--verify-best-training", action="store_true",
                   help="Also load the original model to verify the strongest training-only delta; never export it")
    args = p.parse_args(argv)
    for key in ("training_run", "model_path", "out"):
        if not getattr(args, key).strip():
            p.error(f"--{key.replace('_', '-')} cannot be empty")
    out, run = Path(args.out), Path(args.training_run)
    if out.exists():
        p.error("--out already exists; use a new file")
    if args.iterations < 1 or args.max_length < 2:
        p.error("Invalid iteration count or max length")
    source = json.loads((run / "training_report.json").read_text())
    if source.get("development_protocol"):
        raise ValueError("This legacy validation-ray audit is not applicable to reclassified development runs")
    if source.get("method") != "static_overlap_cached_head_v1":
        raise ValueError("Expected a cached-head training report")
    prior = source["initial"]
    fit, val = prior["training_protection"], prior["validation_protection"]
    # Preserve the recorded scientific and internal thresholds. Never infer a
    # bigger budget from an observed violation or optimize those thresholds.
    if val["nominal_retain_nll_budget"] != .05 or val["nominal_retain_kl_budget"] != .01:
        raise ValueError("This audit requires the unchanged 0.05/0.01 scientific limits")
    config = TrainConfig(target_probability=prior["training_forgetting"]["target_probability"],
        retain_nll_safety_margin=.05-fit["applied_nll_budget"],
        retain_kl_safety_margin=.01-fit["applied_kl_budget"])
    config.validate()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,
                                              local_files_only=args.local_files_only)
    cache, bundle_hash = load_saved_cache(run, tokenizer, args.device, args.max_length)
    zero = torch.zeros(len(cache.rows), cache.hidden.shape[1], device=args.device, dtype=cache.hidden.dtype)
    baseline = summarize(cached_measure(cache, zero), config)
    if any(not math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-8)
           for a, b in zip(baseline["score"], prior["score"])):
        raise ValueError("Reconstructed cache baseline differs from the original report")
    print(json.dumps({"phase": "audit_saved_cache", "model_weights_loaded": False,
                      "transformer_passes": 0}), flush=True)
    solver = RetainMetricSolver(cache)
    families = {}
    for item in source["history"]:
        pair = (item["tau"], item["ridge"])
        families[pair] = max(families.get(pair, 0), item["strength"])
    results = []
    for (tau, ridge), upper in families.items():
        boundary = feasible_strength(cache, solver.solve(tau, ridge), config, upper, args.iterations)
        item = {"tau": tau, "ridge": ridge, **boundary}
        results.append(item)
        print(json.dumps({"phase": "ray_retention_boundary", **item}, allow_nan=False), flush=True)
    best = min(results, key=lambda x: x["summary"]["score"])
    report = {"scope": "cached retention boundaries on the existing fitted directions only",
              "native_model_verified": False, "native_checkpoint_created": False,
              "official_eff_gen_measured": False, "validation_used_in_solve": False,
              "validation_retention_used_for_boundary_search": True,
              "validation_forget_used_for_selection": False,
              "source_training_run": str(run.resolve()), "training_bundle_sha256": bundle_hash,
              "cache_sha256": sha256_file(run / "head_cache.pt"),
              "source_report_sha256": sha256_file(run / "training_report.json"),
              "baseline": baseline, "families": results, "best_feasible_on_searched_rays": best,
              "elapsed_seconds": time.perf_counter()-started}
    if args.verify_best_training:
        print(json.dumps({"phase": "verify_training_only_candidate", "native_export_enabled": False}), flush=True)
        report["training_only_verification"] = verify_training_candidate(run, args.model_path, cache, config,
                                                                         args.local_files_only)
        report["elapsed_seconds"] = time.perf_counter()-started
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": str(out), "best_feasible_on_searched_rays": best,
                      "native_checkpoint_created": False}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
