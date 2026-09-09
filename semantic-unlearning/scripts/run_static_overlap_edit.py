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
from static_overlap_training import TrainConfig, export_verified, sha256_file, train, within_budgets


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
    parser.add_argument("--resume-training-run", help="Continue saved factors on the original base, with fresh Adam state")
    parser.add_argument("--training-only", action="store_true",
                        help="Save factors and diagnostics without creating or verifying a native checkpoint")
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


def resume_factors(editor, examples, parent_run, parent_manifest):
    """Reproduce the saved state before allowing any additional optimizer step."""
    from export_static_overlap_edit import verify_recovered_statistics

    report = json.loads((parent_run / "training_report.json").read_text())
    parent_accepted = report.get("accepted_steps_total", report["accepted_steps"])
    if not parent_accepted:
        raise ValueError("The parent run has no accepted training steps")
    for key in ("model_type", "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                "num_attention_heads", "num_key_value_heads", "tie_word_embeddings", "rope_theta", "rope_scaling"):
        if editor.model.config.to_dict().get(key) != parent_manifest["model_config"].get(key):
            raise ValueError(f"Continuation base model configuration differs: {key}")
    if editor.shared != parent_manifest["shared_endpoints"]:
        raise ValueError("Continuation endpoint sharing differs from the manifest")
    factors = parent_run / "training_factors.pt"
    editor.load_artifact(torch.load(factors, map_location="cpu", weights_only=True))
    reproduction = verify_recovered_statistics(editor, examples, report)
    return {"training_run": str(parent_run.resolve()), "parent_accepted_steps": parent_accepted,
            "factors_sha256": sha256_file(factors), "manifest_sha256": sha256_file(parent_run / "manifest.json"),
            "training_report_sha256": sha256_file(parent_run / "training_report.json"),
            "reproduction": reproduction, "optimizer_state": "reset",
            "retention_reference": "original_base_model"}


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args(argv)
    settings, config = load_config(args.config)
    if args.steps is not None:
        config.steps = args.steps
        config.validate()
        settings["training"]["steps"] = args.steps
    bundle, facts, bundle_hash = load_bundle(args.training_bundle)
    parent_run = Path(args.resume_training_run) if args.resume_training_run else None
    parent_manifest = json.loads((parent_run / "manifest.json").read_text()) if parent_run else None
    if parent_manifest is not None:
        if (parent_manifest["architecture"] != "static_overlap_constrained_embedding_mlp_head_v1"
                or parent_manifest["training_bundle_sha256"] != bundle_hash
                or parent_manifest["settings"]["architecture"] != settings["architecture"]
                or parent_manifest["training_dtype"] != args.dtype
                or parent_manifest["settings"]["max_length"] != settings["max_length"]
                or parent_manifest["settings"]["abstention"] != settings["abstention"]):
            raise ValueError("Continuation requires the original bundle, architecture, tokenization settings and training dtype")
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
    if parent_manifest is None:
        channels, scores = localize(model, train_forget, train_retain,
                                    architecture["blocks"], architecture["channels_per_block"])
    else:
        if input_rows != parent_manifest["input_rows"] or output_rows != parent_manifest["output_rows"]:
            raise ValueError("Continuation tokenizer produces different editable token rows")
        channels = {int(k): v for k, v in parent_manifest["selected_channels"].items()}
        scores = parent_manifest["localization"]
    editor = StaticEditor(model, input_rows, output_rows, channels, architecture["rank"])
    # Exact zero-delta computation check before fitting, including full prompts.
    continuation = None
    if parent_manifest is not None:
        continuation = resume_factors(editor, examples, parent_run, parent_manifest)
        print(json.dumps({"continuation": continuation}, indent=2), flush=True)
    else:
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
                "training_only": args.training_only,
                "training_bundle_sha256": bundle_hash,
                "training_bundle_path": str(Path(args.training_bundle).resolve()),
                "input_rows": input_rows, "output_rows": output_rows,
                "shared_endpoints": editor.shared, "selected_channels": channels,
                "localization": scores, "trainable_parameters": sum(p.numel() for p in editor.parameters),
                "runtime_router": False, "runtime_guard": False,
                "forget_associations": [f for f in facts.values() if f["role"] == "forget"],
                "training_text_fingerprints": text_fingerprints(bundle)}
    if continuation is not None:
        manifest["continuation"] = continuation
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = train(editor, examples, config, output / "training.jsonl", resume=parent_run is not None)
    report["accepted_steps_total"] = report["accepted_steps"] + (continuation["parent_accepted_steps"] if continuation else 0)
    report["native_export_requested"] = not args.training_only
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    torch.save(editor.artifact(), output / "training_factors.pt")
    if args.training_only:
        print(json.dumps({"training_only": True, "native_checkpoint_created": False,
                          "accepted_steps": report["accepted_steps"], "stop_reason": report["stop_reason"],
                          "initial_training_forget_loss": report["initial_training_forget_loss"],
                          "training_forget_loss": report["training_forget_loss"],
                          "initial_training_forgetting": report["initial_training_forgetting"],
                          "training_forgetting": report["training_forgetting"],
                          "training_protection": report["training_protection"],
                          "validation_protection": report["validation_protection"]}, indent=2), flush=True)
        return
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
