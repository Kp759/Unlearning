#!/usr/bin/env python3
"""Tokenizer-only preflight for MCF private-vocabulary rewiring V1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from transformers import AutoTokenizer

import mcf_private_vocab_rewiring_v1_core as core


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    protocol_dir = Path(args.protocol_dir)
    forget = json.loads(
        (protocol_dir / "training_visible_forget_direct.json").read_text(encoding="utf-8")
    )
    base = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    subjects = core.unique_subjects(forget)
    mapping = core.build_subject_slot_mapping(base, subjects)

    target = Path(args.output_dir) / "tokenizer_preflight"
    if target.exists():
        shutil.rmtree(target)
    base.save_pretrained(target)
    core.patch_saved_tokenizer_reserved_slots(target, mapping)
    private = AutoTokenizer.from_pretrained(target, use_fast=True)
    if len(private) != len(base):
        raise RuntimeError("private tokenizer changed vocabulary size")
    core.validate_subject_routing(private, mapping)

    report = {
        "unique_forget_subjects": len(subjects),
        "reserved_slots_assigned": len(mapping),
        "vocab_size_before": len(base),
        "vocab_size_after": len(private),
        "vocab_size_unchanged": len(base) == len(private),
        "sample": mapping[:5],
    }
    (Path(args.output_dir) / "method").mkdir(exist_ok=True)
    (Path(args.output_dir) / "method" / "tokenizer_preflight.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
