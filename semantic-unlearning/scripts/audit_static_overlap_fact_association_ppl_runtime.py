#!/usr/bin/env python3
"""Read-only runtime-aligned PPL audit for saved association checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)
from static_overlap_fact_association_embeddings import load_artifact_into_model
from static_overlap_fact_association_v2_gate import (
    load_relation_prototype_artifact,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_path = Path(manifest["model_path"]).resolve()
    text = load_official_ppl_text(args.wikidata_dir)
    if text is None:
        raise FileNotFoundError(args.wikidata_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = dtype_from_str(args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base.requires_grad_(False)
    device = next(base.parameters()).device

    base_legacy = official_perplexity(
        base, tok, text, device, max_input_length=100
    )
    base_runtime = runtime_aligned_perplexity(
        base, tok, text, device, max_input_length=100
    )

    architecture = str(artifact.get("architecture", ""))
    if architecture == "relation_prototype_fact_association_bank_v2":
        edited, bank = load_relation_prototype_artifact(base, artifact)
    else:
        edited, bank = load_artifact_into_model(base, artifact)
    edited.eval()

    edited_legacy = official_perplexity(
        edited, tok, text, device, max_input_length=100
    )
    counters_after_legacy = bank.counters()
    edited_runtime = runtime_aligned_perplexity(
        edited, tok, text, device, max_input_length=100
    )
    counters_after_runtime = bank.counters()
    runtime_route_delta = {
        "hook_calls": (
            counters_after_runtime["hook_calls"]
            - counters_after_legacy["hook_calls"]
        ),
        "active_batch_rows": (
            counters_after_runtime["active_batch_rows"]
            - counters_after_legacy["active_batch_rows"]
        ),
        "active_token_positions": (
            counters_after_runtime["active_token_positions"]
            - counters_after_legacy["active_token_positions"]
        ),
        "active_fact_counts": [
            after - before
            for after, before in zip(
                counters_after_runtime["active_fact_counts"],
                counters_after_legacy["active_fact_counts"],
            )
        ],
    }
    result = {
        "kind": "fact_association_runtime_aligned_ppl_audit_v1",
        "run_dir": str(run_dir),
        "dtype": args.dtype,
        "legacy": {
            "base_ppl": base_legacy,
            "edited_ppl": edited_legacy,
            "delta": edited_legacy - base_legacy,
            "status": (
                "historical whole-sequence result; structurally blind to a "
                "last-position-only intervention on scored logits"
            ),
        },
        "runtime_aligned": {
            "base": base_runtime,
            "edited": edited_runtime,
            "delta_ppl": edited_runtime["ppl"] - base_runtime["ppl"],
            "ratio": edited_runtime["ppl"] / base_runtime["ppl"],
            "route_activity_during_runtime_aligned_scoring": runtime_route_delta,
            "utility_interpretation": (
                "stressful_for_the_intervention"
                if runtime_route_delta["active_batch_rows"] > 0
                else "no_association_route_fired_on_this_raw_text"
            ),
        },
        "runtime_counters_total": bank.counters(),
        "official_mcf_prompt_fields_read": False,
    }
    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / f"runtime_aligned_ppl_audit_{args.dtype}.json"
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite PPL audit: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
