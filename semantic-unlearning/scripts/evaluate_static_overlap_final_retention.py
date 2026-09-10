#!/usr/bin/env python3
"""One final preservation test of a fixed native checkpoint, without selection.

Scores both the frozen independent factual set and preservation anchors in the
separate evaluation bundle. Uses full-vocabulary KL(base||edit), original-base
answer NLL increases, and exactly 0.05/0.01 limits without numeric slack.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import torch

from evaluate_static_overlap_edit import verify_checkpoint
from freeze_static_overlap_development import load_protocol, write_new
from static_overlap_core import model_logits, selected_logits
from static_overlap_data import encode_bundle, load_bundle, text_fingerprints
from static_overlap_training import TrainConfig, sha256_file, training_protection


def claim_final(protocol_path, checkpoint):
    marker = Path(protocol_path).parent / "final_evaluation_started.json"
    identity = {"protocol_sha256": sha256_file(protocol_path),
                "checkpoint_export_sha256": sha256_file(Path(checkpoint) / "static_edit_export.json")}
    if marker.exists():
        previous = json.loads(marker.read_text())
        if any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("The final test is already bound to a different checkpoint; no reselection")
    else:
        write_new(marker, {**identity, "checkpoint": str(Path(checkpoint).resolve()),
                          "started_utc": datetime.now(timezone.utc).isoformat()})
    return identity


def tokenizer_signature(tokenizer):
    value = json.loads(tokenizer.backend_tokenizer.to_str())
    value.pop("padding", None)
    value.pop("truncation", None)
    return value


@torch.no_grad()
def score_final(checkpoint, base_path, groups, device, max_length, local_files_only, emit=print):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=True, local_files_only=local_files_only)
    edited_tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, local_files_only=True)
    if tokenizer_signature(base_tokenizer) != tokenizer_signature(edited_tokenizer):
        raise ValueError("Final base/edit tokenizer pipelines differ")
    examples, names = [], []
    for name, bundle in groups.items():
        group = [e for e in encode_bundle(bundle, base_tokenizer, max_length, "I don't know.")
                 if e.role in ("retain", "language")]
        if not group or any(e.split != "test" for e in group):
            raise ValueError("Final preservation requires nonempty test-only anchors")
        examples.extend(group)
        names.extend([name] * len(group))
    # Only one model resides on the GPU. Full-vocabulary base probabilities are
    # streamed to temporary disk rather than approximating KL with selected rows.
    with tempfile.TemporaryDirectory(prefix="final-retention-") as directory:
        directory = Path(directory)
        base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32,
            local_files_only=local_files_only, attn_implementation="eager").to(device).eval()
        for i, e in enumerate(examples):
            logits = model_logits(base, e)
            values, labels = selected_logits(logits, e)
            torch.save({"logp": values.double().log_softmax(-1).cpu(), "labels": labels.cpu()},
                       directory / f"{i}.pt")
            if i == 0 or (i + 1) % 100 == 0:
                emit(json.dumps({"phase": "final_base_references", "anchors": i + 1, "total": len(examples)}))
        del logits, values, labels, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32,
            local_files_only=True, attn_implementation="eager").to(device).eval()
        rows = []
        for i, (name, e) in enumerate(zip(names, examples)):
            reference = torch.load(directory / f"{i}.pt", weights_only=True)
            values, labels = selected_logits(model_logits(model, e), e)
            if not torch.equal(labels.cpu(), reference["labels"]):
                raise ValueError("Final test labels differ")
            lp = reference["logp"].to(device)
            lq = values.double().log_softmax(-1)
            base_nll = -lp.gather(1, labels[:, None]).mean().item()
            nll = -lq.gather(1, labels[:, None]).mean().item()
            rows.append({"id": e.id, "set": name, "split": "test", "role": e.role,
                         "base_nll": base_nll, "nll": nll, "nll_increase": nll-base_nll,
                         "kl": (lp.exp() * (lp-lq)).sum(-1).mean().clamp_min(0).item()})
            if i == 0 or (i + 1) % 100 == 0:
                emit(json.dumps({"phase": "final_edited_scores", "anchors": i + 1, "total": len(examples)}))
        del model
    config = TrainConfig(retain_nll_budget=.05, retain_kl_budget=.01)
    summaries = {name: training_protection([r for r in rows if r["set"] == name], config)[1]
                 for name in groups}
    return {"sets": summaries, "retention_passed": all(s["retention_passed"] for s in summaries.values()),
            "rows": rows, "numeric_budget_slack": 0.0, "official_eff_gen_measured": False,
            "used_for_selection": False, "original_base_reference": str(Path(base_path).resolve())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)
    protocol = load_protocol(args.protocol)
    export = verify_checkpoint(args.checkpoint)
    if export["deployment_dtype"] != "torch.float32":
        raise ValueError("Frozen experiment requires FP32 final evaluation")
    manifest = json.loads((Path(args.checkpoint) / "training_manifest.json").read_text())
    if manifest.get("development_protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("Checkpoint does not belong to this frozen protocol")
    if Path(manifest["model_path"]).resolve() != Path(args.model_path).resolve():
        raise ValueError("Use the recorded original base model")
    groups = {}
    for name, purpose in (("final_retention", "preservation_test"), ("evaluation_bundle", "evaluation")):
        groups[name], _, _ = load_bundle(protocol["files"][name]["path"], purpose)
        if set(manifest["training_text_fingerprints"]) & set(text_fingerprints(groups[name])):
            raise ValueError("Frozen final prompts overlap development inputs")
    identity = claim_final(args.protocol, args.checkpoint)
    out = Path(args.protocol).parent / "final_retention_results.json"
    if out.exists():
        saved = json.loads(out.read_text())
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("Final report identity differs")
        print(json.dumps({"status": "already_evaluated", "retention_passed": saved["retention_passed"],
                          "report": str(out)}), flush=True)
        return 0
    report = score_final(args.checkpoint, args.model_path, groups, args.device,
                         protocol["experiment"]["max_length"], args.local_files_only,
                         lambda s: print(s, flush=True))
    write_new(out, {**identity, **report})
    print(json.dumps({"status": "final_preservation_evaluated", "retention_passed": report["retention_passed"],
                      "sets": report["sets"], "report": str(out)}), flush=True)
    # Completed measurement is not selection. Save failures and still allow the
    # separate official evaluation to run on this SAME fixed checkpoint.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
