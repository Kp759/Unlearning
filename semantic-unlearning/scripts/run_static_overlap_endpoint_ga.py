#!/usr/bin/env python3
"""One bounded forgetting-first experiment on the original tied endpoint mask."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from run_static_overlap_mlp_pilot import References, development_gate, emit, encode_pilot, fitting_batch, measure_pilot
from static_overlap_core import model_logits
from static_overlap_endpoint_ga import EndpointEditor, endpoint_step, locality_hashes, verify_locality
from static_overlap_endpoint_protocol import METHOD, load_pilot
from static_overlap_mlp_protocol import write_new
from static_overlap_training import TrainConfig, export_verified, sha256_file


def baseline_rows(examples, references):
    return [{"id": e.id, "split": e.split, "role": e.role, "base_nll": references.nll[e.id],
             "nll": references.nll[e.id], "nll_increase": 0., "kl": 0.} for e in examples]


def compact_gate(gate):
    """Keep complete IDs in JSON artifacts; terminal shows counts and extrema."""
    result = json.loads(json.dumps(gate))
    for split in ("train", "development"):
        preservation = result[split]["preservation"]
        preservation["violating_anchors"] = len(preservation.pop("violating_anchor_ids", []))
    return result


def fit(editor, examples, references, config, plan, output, *, source=None, data=None):
    train_f = [e for e in examples if e.split == "train" and e.role == "forget"]
    train_r = [e for e in examples if e.split == "train" and e.role in ("retain", "language")]
    rng = random.Random(plan["seed"])
    rng.shuffle(train_f)
    rng.shuffle(train_r)
    initial = baseline_rows(examples, references)
    baseline = development_gate(initial, config)
    write_new(output / "baseline_metrics.json", {"gate": baseline, "rows": initial})
    emit(phase="endpoint_baseline", **compact_gate(baseline))
    optimizer = torch.optim.Adam(editor.parameters, lr=plan["learning_rate"])
    started = time.monotonic()
    hard_f, hard_r, seen_f, seen_r = [], [], set(), set()
    cursor_f = cursor_r = rejected = stale = 0
    previous_mean = sum(references.nll[e.id] for e in train_f) / len(train_f)
    history, gates, selected, stop = [], [], None, "step_budget"
    last_gate = baseline
    step = 0

    def check_gate(at_step):
        nonlocal hard_f, hard_r, previous_mean, stale, last_gate, selected, stop
        emit(phase="endpoint_development_gate_start", step=at_step, examples=len(examples))
        rows = measure_pilot(editor.model, examples, references)
        last_gate = development_gate(rows, config)
        mean = sum(r["nll"] for r in rows if r["split"] == "train" and r["role"] == "forget") / len(train_f)
        gain = mean - previous_mean
        stale = stale + 1 if gain < plan["min_gate_nll_gain"] else 0
        previous_mean = mean
        gates.append({"step": at_step, "gate": last_gate, "training_mean_nll_gain_since_gate": gain,
                      "elapsed_seconds": time.monotonic()-started})
        (output / "last_metrics.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
        torch.save(editor.artifact(), output / "last_endpoint_delta.pt")
        emit(phase="endpoint_development_gate", step=at_step, training_mean_nll_gain_since_gate=gain,
             elapsed_seconds=time.monotonic()-started, **compact_gate(last_gate))
        if last_gate["passed"]:
            selected, stop = at_step, "development_gate_passed"
            torch.save(editor.artifact(), output / "training_factors.pt")
        # Development is used solely for this gate; all replay comes from fitting.
        hard_f = [r["id"] for r in sorted((r for r in rows if r["split"] == "train" and r["role"] == "forget"),
                  key=lambda r: r["nll"])[:plan["forget_batch"] // 2]]
        hard_r = [r["id"] for r in sorted((r for r in rows if r["split"] == "train" and r["role"] != "forget"),
                  key=lambda r: max(r["nll_increase"]/.04, r["kl"]/.008), reverse=True)[:plan["retain_batch"] // 2]]

    def report():
        return {"method": METHOD, "exploratory": True, "stop_reason": stop, "selected_step": selected,
            "last_gate": last_gate, "baseline_gate": baseline, "history": history, "gates": gates,
            "elapsed_seconds": time.monotonic()-started,
            "fitting_forget_seen": len(seen_f), "fitting_forget_total": len(train_f),
            "fitting_preservation_seen": len(seen_r), "fitting_preservation_total": len(train_r),
            "native_checkpoint_created": False, "final_evaluation_started": False,
            "development_used_for_gradients": False, "minibatch_checks_are_not_full_retention_guarantees": True}

    for step in range(1, plan["steps"] + 1):
        if time.monotonic()-started >= plan["max_training_seconds"]:
            stop = "wall_time_budget"
            step -= 1
            break
        batch_f = fitting_batch(train_f, step, plan["forget_batch"], hard_f, offset=cursor_f)
        batch_r = fitting_batch(train_r, step, plan["retain_batch"], hard_r, offset=cursor_r)
        cursor_f += len(batch_f) - min(len(hard_f), plan["forget_batch"] // 2)
        cursor_r += len(batch_r) - min(len(hard_r), plan["retain_batch"] // 2)
        result = endpoint_step(editor, optimizer, batch_f, batch_r, references, config, plan)
        result.update(step=step, elapsed_seconds=time.monotonic()-started)
        history.append(result)
        seen_f.update(e.id for e in batch_f)
        seen_r.update(e.id for e in batch_r)
        rejected = 0 if result["accepted"] else rejected + 1
        emit(phase="endpoint_step", **result)
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(result, allow_nan=False)+"\n")
        if step % plan["check_every"] == 0 or step == plan["steps"]:
            check_gate(step)
            (output / "training_report.json").write_text(json.dumps(report(), indent=2, allow_nan=False)+"\n")
            if selected is not None:
                break
            if stale >= plan["stalled_gates"]:
                stop = "insufficient_training_forgetting_progress"
                break
        if rejected >= plan["max_stalled_steps"]:
            stop = "consecutive_rejected_steps"
            break
    if not gates or gates[-1]["step"] != step:
        check_gate(step)
    return report()


def main(argv=None, *, protocol_loader=load_pilot, method=METHOD, fit_function=fit,
         editor_factory=None, prepare_base=None, hash_function=None, verify_function=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    p = protocol_loader(args.pilot_protocol)
    if Path(args.model_path).resolve() != Path(p["base_model_path"]).resolve():
        raise ValueError("Start from the original base model")
    output = Path(args.pilot_protocol).resolve().parent
    write_new(output / "training_started.json", {"model": str(Path(args.model_path).resolve()),
        "pilot_protocol_sha256": sha256_file(args.pilot_protocol)})
    plan = p["plan"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(plan["seed"])
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=args.local_files_only)
    source = json.loads(Path(p["source_bundle"]["path"]).read_text())
    data = json.loads(Path(p["data"]["path"]).read_text())
    examples = encode_pilot(source, data, tokenizer, plan["max_length"])
    mask = json.loads(Path(p["overlap_manifest"]["path"]).read_text())
    rows = sorted(set(mask["input_rows"]) | set(mask["output_rows"]))
    if set(rows) & set(tokenizer.all_special_ids):
        raise ValueError("Original overlap mask unexpectedly includes special tokens")
    if ({f["id"] for f in source["facts"] if f["role"] == "forget"}
            != {f["id"] for f in mask["forget_associations"]}):
        raise ValueError("Overlap mask and development data target different forget facts")
    emit(phase="load_original_base", model=args.model_path, method=method, examples=len(examples))
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32,
        local_files_only=args.local_files_only, attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    preparation = prepare_base(model, examples) if prepare_base else None
    if preparation is not None:
        write_new(output / "endpoint_separation.json", preparation)
        emit(phase="endpoint_separation", **preparation)
    write_new(output / "encoded_development_examples.json", [asdict(e) for e in examples])
    emit(phase="fingerprint_frozen_weights", editable_rows=len(rows))
    original = hash_function(model, mask) if hash_function else locality_hashes(model, rows)
    write_new(output / "original_frozen_weight_hashes.json", original)
    write_new(output / "endpoint_mask.json", {"input_rows": mask["input_rows"], "output_rows": mask["output_rows"],
        "source_row_union": rows, "source_manifest_sha256": p["overlap_manifest"]["sha256"]})
    references = References(output / "base_references")
    references.build(model, examples)
    editor = (EndpointEditor(model, mask["input_rows"], mask["output_rows"])
              if editor_factory is None else editor_factory(model, mask))
    with torch.no_grad():
        actual = model_logits(model, examples[0])
        with editor.base():
            expected = model_logits(model, examples[0])
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise ValueError("Zero endpoint delta failed exact base parity")
        del actual, expected
    emit(phase="endpoint_preparation", editable_rows=len(rows), trainable_parameters=sum(p.numel() for p in editor.parameters),
         shared_endpoints=editor.shared, original_tying_preserved=editor.shared, rank_restriction=False, original_overlap_mask_exact=True,
         transformer_trainable=False, base_logits_exact=True)
    config = TrainConfig(target_probability=plan["target_probability"], retain_nll_budget=.05, retain_kl_budget=.01,
        retain_nll_safety_margin=plan["fitting_nll_margin"], retain_kl_safety_margin=plan["fitting_kl_margin"])
    report = fit_function(editor, examples, references, config, plan, output, source=source, data=data)
    report["pilot_protocol_sha256"] = sha256_file(args.pilot_protocol)
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if report["selected_step"] is None:
        emit(status="no_development_valid_edit", stop_reason=report["stop_reason"],
             report=str(output / "training_report.json"), final_tests_touched=False)
        return 2
    manifest = {"method": method, "exploratory": True, "model_path": str(Path(args.model_path).resolve()),
        "exploratory_protocol_path": str(Path(args.pilot_protocol).resolve()),
        "exploratory_protocol_sha256": sha256_file(args.pilot_protocol),
        "forget_associations": [f for f in source["facts"] if f["role"] == "forget"],
        "training_text_fingerprints": data["training_text_fingerprints"],
        "settings": {"abstention": "I don't know.", **plan}, "endpoint_mask": json.loads((output / "endpoint_mask.json").read_text()),
        "endpoint_separation": preparation}
    write_new(output / "training_manifest.json", manifest)
    locality = {}
    def reload_verified(path):
        loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
            local_files_only=True, attn_implementation="eager").to(args.device).eval()
        locality.update(verify_function(loaded, mask, original) if verify_function else verify_locality(loaded, rows, original))
        return loaded
    emit(phase="merge_reload_strict_verification", selected_step=report["selected_step"])
    export_verified(editor, tokenizer, examples, config, output / "checkpoint", torch.float32,
        reload_verified, atol=1e-4, rtol=1e-5, manifest=manifest, numeric_slack=0., require_forgetting=True)
    report.update(native_checkpoint_created=True, locality=locality)
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    emit(status="verified_endpoint_checkpoint", checkpoint=str(output / "checkpoint"), locality=locality,
         development_gate=compact_gate(report["last_gate"]), official_eff_gen_measured=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
