#!/usr/bin/env python3
"""Reload and verify a compositional-marker checkpoint without held-out probes.

This is deliberately a fresh process.  It verifies that the ordinary sparse
embedding and LM-head weights survive serialization and still satisfy every
training-safe positive under the exact official-compatible MCF NLL arithmetic.
It opens neither the original MultiCounterFact file nor any official
paraphrase, neighborhood, retain, or PPL input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_compare as gagd
import mcf_compositional_marker_write_read as method
import sure_canonical_core as canonical


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--context-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--forget-margin", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    value = p.parse_args(list(argv) if argv is not None else None)
    if int(value.batch_size) <= 0:
        p.error("--batch-size must be positive")
    return value


def acceptance_payload(
    margins: torch.Tensor,
    direct_flags: Sequence[bool],
    *,
    forget_margin: float,
) -> Dict[str, Any]:
    if margins.ndim != 1 or margins.shape[0] != len(direct_flags):
        raise ValueError("margin/direct-flag lengths do not match")
    threshold = float(forget_margin) - 1e-6
    failures = margins < threshold
    direct_failures = sum(
        int(bool(direct_flags[index]) and bool(failures[index]))
        for index in range(len(direct_flags))
    )
    positive_failures = int(failures.sum())
    return {
        "schema_version": 1,
        "kind": "mcf_compositional_marker_post_reload_acceptance",
        "checkpoint_was_reloaded": True,
        "official_compatible_nll_arithmetic": True,
        "thresholds": {
            "forget_margin": float(forget_margin),
            "max_direct_failures": 0,
            "max_training_safe_positive_failures": 0,
        },
        "observed": {
            "direct_prompt_instances": int(sum(bool(x) for x in direct_flags)),
            "training_safe_positive_instances": len(direct_flags),
            "direct_failures": direct_failures,
            "training_safe_positive_failures": positive_failures,
            "minimum_margin": float(margins.min()) if margins.numel() else float("nan"),
        },
        "data_access": {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
        "passed": direct_failures == 0 and positive_failures == 0,
    }


def _load_object(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.context_manifest).resolve()
    raw_records = json.loads(visible_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise RuntimeError("training-visible artifact must be a JSON list")
    locked_split.assert_direct_only_training_view(raw_records)
    records = method._record_views(raw_records)

    context_manifest = _load_object(manifest_path)
    if context_manifest.get("protocol") != method.PROTOCOL:
        raise RuntimeError("context-manifest protocol mismatch")
    if context_manifest.get("data_access") != {
        "official_paraphrases_seen": 0,
        "official_neighborhoods_seen": 0,
        "benchmark_retain_seen": 0,
        "official_ppl_seen": False,
    }:
        raise RuntimeError("context manifest does not certify the training-data firewall")
    context_rows = context_manifest.get("records")
    if not isinstance(context_rows, list) or len(context_rows) != len(records):
        raise RuntimeError("context manifest does not match the locked forget records")
    contexts: Dict[int, Mapping[str, Any]] = {}
    for position, (record, row) in enumerate(zip(records, context_rows)):
        case_id = int(record["case_id"])
        if int(row.get("case_id", -1)) != case_id:
            raise RuntimeError(f"context case mismatch at position {position}")
        contexts[case_id] = row

    instances, _owners, direct_flags = method.build_prompt_instances(records, contexts)
    ns = argparse.Namespace(
        model_path=str(Path(a.model_dir).resolve()),
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    margins = method.evaluate_instance_margins(
        model,
        tok,
        instances,
        device,
        llama_like=canonical.is_llama_like(model, tok),
        batch_size=int(a.batch_size),
    )
    payload = acceptance_payload(
        margins,
        direct_flags,
        forget_margin=float(a.forget_margin),
    )
    output = Path(a.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"post-reload acceptance: {output}")
    if not payload["passed"]:
        raise SystemExit("reloaded checkpoint failed training-safe acceptance")


if __name__ == "__main__":
    main()
