#!/usr/bin/env python3
"""Localize, jointly train, and export static embedding–MLP–head edits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from static_overlap_core import StaticEditor, localize, model_logits
from static_overlap_data import encode_bundle, endpoint_rows, load_bundle, text_fingerprints
from static_overlap_training import TrainConfig, export_verified, train, within_budgets


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/static_overlap_edit.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--deployment-dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--steps", type=int, help="Override the config step count for a coverage pilot")
    return parser.parse_args(argv)


def load_config(path):
    raw = json.loads(Path(path).read_text())
    if set(raw) != {"architecture", "training", "max_length", "abstention", "export_atol", "export_rtol"}:
        raise ValueError("Unknown/missing configuration fields")
    architecture = raw["architecture"]
    if set(architecture) != {"rank", "blocks", "channels_per_block"}:
        raise ValueError("Architecture requires rank, blocks, channels_per_block")
    if any(type(v) is not int or v <= 0 for v in architecture.values()):
        raise ValueError("Architecture sizes must be positive integers")
    config = TrainConfig(**raw["training"])
    config.validate()
    if type(raw["max_length"]) is not int or raw["max_length"] < 2:
        raise ValueError("Invalid maximum sequence length")
    if not isinstance(raw["abstention"], str) or (config.lambda_abstain and not raw["abstention"].strip()):
        raise ValueError("Abstention objective requires ordinary nonempty response text")
    for key in ("export_atol", "export_rtol"):
        if not isinstance(raw[key], (int, float)) or not 0 <= raw[key] < float("inf"):
            raise ValueError("Export tolerances must be finite and nonnegative")
    return raw, config


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args(argv)
    settings, config = load_config(args.config)
    if args.steps is not None:
        config.steps = args.steps
        config.validate()
        settings["training"]["steps"] = args.steps
    bundle, facts, bundle_hash = load_bundle(args.training_bundle)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,
                                             local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only, attn_implementation="eager",
    ).to(args.device).eval()
    examples = encode_bundle(bundle, tokenizer, settings["max_length"],
                             settings["abstention"] if config.lambda_abstain else "")
    train_forget = [e for e in examples if e.split == "train" and e.role == "forget"]
    train_retain = [e for e in examples if e.split == "train" and e.role == "retain"]
    input_rows, output_rows = endpoint_rows(facts, examples, tokenizer, bool(config.lambda_abstain))
    architecture = settings["architecture"]
    channels, scores = localize(model, train_forget, train_retain,
                                architecture["blocks"], architecture["channels_per_block"])
    editor = StaticEditor(model, input_rows, output_rows, channels, architecture["rank"])
    # Exact zero-delta computation check before fitting, including full prompts.
    with torch.no_grad():
        for example in examples:
            with editor.base():
                base = model_logits(model, example)
            if not torch.equal(base, model_logits(model, example)):
                raise RuntimeError("Zero initialization failed to reproduce base logits")
    manifest = {"architecture": "static_overlap_constrained_embedding_mlp_head_v1",
                "settings": settings, "training_config": asdict(config),
                "model_path": args.model_path, "model_commit": getattr(model.config, "_commit_hash", None),
                "model_config": model.config.to_dict(), "training_dtype": args.dtype,
                "deployment_dtype": args.deployment_dtype,
                "training_bundle_sha256": bundle_hash,
                "training_bundle_path": str(Path(args.training_bundle).resolve()),
                "input_rows": input_rows, "output_rows": output_rows,
                "shared_endpoints": editor.shared, "selected_channels": channels,
                "localization": scores, "trainable_parameters": sum(p.numel() for p in editor.parameters),
                "runtime_router": False, "runtime_guard": False,
                "forget_associations": [f for f in facts.values() if f["role"] == "forget"],
                "training_text_fingerprints": text_fingerprints(bundle)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = train(editor, examples, config, output / "training.jsonl")
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(editor.artifact(), output / "training_factors.pt")
    passed, protection = within_budgets(report["validation"], config)
    if not report["accepted_steps"] or not passed:
        raise RuntimeError(f"No validated edited checkpoint to export; inspect training_report.json: {protection}")

    def reload_model(path):
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=getattr(torch, args.deployment_dtype),
            local_files_only=True, attn_implementation="eager",
        ).to(args.device)

    exported = export_verified(editor, tokenizer, examples, config, output / "checkpoint",
                    getattr(torch, args.deployment_dtype), reload_model,
                    atol=settings["export_atol"], rtol=settings["export_rtol"], manifest=manifest)
    print(f"Verified native checkpoint: {output / 'checkpoint'}", flush=True)
    print(f"Finite-anchor forgetting target met: {exported['forgetting_target_met']}; "
          "official held-out Eff/Gen still require evaluation", flush=True)


if __name__ == "__main__":
    main()
