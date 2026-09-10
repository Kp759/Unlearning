#!/usr/bin/env python3
"""Recover a native checkpoint from a completed training run, without retraining."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from static_overlap_core import StaticEditor
from static_overlap_data import encode_bundle, endpoint_rows, load_bundle
from static_overlap_training import TrainConfig, export_verified, measure, sha256_file, within_budgets


def verify_recovered_statistics(editor, examples, training_report):
    """Reject wrong base/tokenizer/factors by reproducing saved per-example metrics.

    Older runs have no base-weight hashes. This is a numerical reproduction
    check, not a cryptographic identity claim about their source model.
    """
    saved = training_report["validation"] + training_report.get("training_forget", [])
    expected = {row["id"]: row for row in saved}
    selected = [e for e in examples if e.id in expected]
    if not saved or len(expected) != len(saved) or len(selected) != len(expected):
        raise ValueError("Saved report examples do not match the recovery bundle")
    maximum = {key: 0.0 for key in ("base_nll", "nll", "kl")}
    for row in measure(editor, selected):
        previous = expected[row["id"]]
        if any(row[key] != previous[key] for key in ("role", "split")):
            raise ValueError(f"Recovered role/split differs: {row['id']}")
        for key in maximum:
            a, b = row[key], previous[key]
            if not (math.isfinite(a) and math.isfinite(b)
                    and math.isclose(a, b, abs_tol=1e-4, rel_tol=1e-5)):
                raise ValueError(f"Recovered {key} differs for {row['id']}: saved={b}, recovered={a}. "
                                 "Use the original base model, tokenizer and training factors.")
            maximum[key] = max(maximum[key], abs(a - b))
    return {"matched_examples": len(expected), "max_abs_differences": maximum,
            "atol": 1e-4, "rtol": 1e-5}


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--output-dir", help="New checkpoint directory; defaults to <run>/checkpoint_<dtype>")
    parser.add_argument("--model-path", help="Original base model; defaults to the saved manifest path")
    parser.add_argument("--training-bundle", help="Original bundle; defaults to the saved manifest path")
    parser.add_argument("--deployment-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    run = Path(args.training_run)
    manifest = json.loads((run / "manifest.json").read_text())
    training_report = json.loads((run / "training_report.json").read_text())
    cached_head = manifest["architecture"] == "static_overlap_cached_head_v1"
    if manifest["architecture"] not in ("static_overlap_constrained_embedding_mlp_head_v1", "static_overlap_cached_head_v1"):
        raise ValueError("Unsupported training artifact architecture")
    settings = manifest["settings"]
    config = TrainConfig(**manifest["training_config"])
    config.validate()
    if not training_report["accepted_steps"] or not within_budgets(training_report["validation"], config)[0]:
        raise ValueError("The saved run has no accepted edit with valid validation retention")
    bundle_path = args.training_bundle or manifest.get("training_bundle_path")
    if not bundle_path:
        parser.error("This older manifest requires --training-bundle")
    bundle, facts, bundle_hash = load_bundle(bundle_path)
    if bundle_hash != manifest["training_bundle_sha256"]:
        raise ValueError("Recovery training bundle hash differs from the saved run")
    output = Path(args.output_dir) if args.output_dir else run / f"checkpoint_{args.deployment_dtype}"
    if output.exists():
        raise FileExistsError(f"Use a new --output-dir; refusing to overwrite {output}")
    model_path = args.model_path or manifest["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=args.local_files_only)
    examples = encode_bundle(bundle, tokenizer, settings["max_length"],
                             settings["abstention"] if config.lambda_abstain else "")
    inputs, outputs = endpoint_rows(facts, examples, tokenizer,
                                   False if cached_head else bool(config.lambda_abstain))
    if cached_head:
        inputs = []
        if (manifest["shared_endpoints"] or manifest["selected_channels"]
                or settings["architecture"]["rank"] != len(outputs)):
            raise ValueError("Invalid cached-head artifact support/rank")
    if inputs != manifest["input_rows"] or outputs != manifest["output_rows"]:
        raise ValueError("Recovery tokenizer produces different editable token rows")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=getattr(torch, manifest["training_dtype"]),
        local_files_only=args.local_files_only, attn_implementation="eager").to(args.device).eval()
    for key in ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "tie_word_embeddings", "rope_theta", "rope_scaling"):
        if model.config.to_dict().get(key) != manifest["model_config"].get(key):
            raise ValueError(f"Recovery base model configuration differs: {key}")
    channels = {int(k): v for k, v in manifest["selected_channels"].items()}
    editor = StaticEditor(model, inputs, outputs, channels, settings["architecture"]["rank"])
    if editor.shared != manifest["shared_endpoints"]:
        raise ValueError("Recovery endpoint sharing differs from the manifest")
    factor_path = run / "training_factors.pt"
    editor.load_artifact(torch.load(factor_path, map_location="cpu", weights_only=True))
    reproduction = verify_recovered_statistics(editor, examples, training_report)
    print(json.dumps({"recovered_training": reproduction}, indent=2), flush=True)
    recovered_manifest = deepcopy(manifest)
    recovered_manifest["deployment_dtype"] = args.deployment_dtype
    recovered_manifest["recovery"] = {
        "training_run": str(run.resolve()), "model_path": str(model_path),
        "training_bundle_path": str(Path(bundle_path).resolve()),
        "factors_sha256": sha256_file(factor_path), "manifest_sha256": sha256_file(run / "manifest.json"),
        "training_report_sha256": sha256_file(run / "training_report.json"),
        "reproduction": reproduction, "optimizer_steps": 0}
    dtype = getattr(torch, args.deployment_dtype)

    def reload_model(path):
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, local_files_only=True, attn_implementation="eager").to(args.device)

    report = export_verified(editor, tokenizer, examples, config, output, dtype, reload_model,
                             atol=settings["export_atol"], rtol=settings["export_rtol"],
                             manifest=recovered_manifest)
    print(f"Verified recovered checkpoint: {output}", flush=True)
    print(f"Finite-anchor forgetting target met: {report['forgetting_target_met']}; "
          "official Eff/Gen still require evaluation", flush=True)


if __name__ == "__main__":
    main()
