#!/usr/bin/env python3
"""Train the input-only extended association-token experiment."""
import argparse
import json
from pathlib import Path

import torch

from run_static_overlap_mlp_pilot import emit
from static_overlap_extended_tokens import run
from static_overlap_extended_tokens_protocol import METHOD, load_pilot
from static_overlap_mlp_protocol import write_new
from static_overlap_training import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    protocol = load_pilot(args.pilot_protocol)
    if Path(args.model_path).resolve() != Path(protocol["base_model_path"]).resolve():
        raise ValueError("Start from the registered original base model")
    output = Path(args.pilot_protocol).resolve().parent
    write_new(output / "training_started.json", {
        "model": str(Path(args.model_path).resolve()),
        "pilot_protocol_sha256": sha256_file(args.pilot_protocol),
    })
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(protocol["plan"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, local_files_only=args.local_files_only)
    source = json.loads(Path(protocol["source_bundle"]["path"]).read_text())
    data = json.loads(Path(protocol["data"]["path"]).read_text())
    emit(phase="load_original_base", model=args.model_path, method=METHOD)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32,
        local_files_only=args.local_files_only, attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    run(model, tokenizer, source, data, protocol["plan"], output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
