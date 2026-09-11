#!/usr/bin/env python3
"""Train a native no-router sparse-MLP writer edit on natural MCF prompts.

The forget set matches the official ZeroUnlearn split.  The exact 1,000
official-retain records are reserved and never used for fitting.  Training
preservation uses different first-half MCF records.  Official paraphrase and
neighborhood fields never enter optimization.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from mcf_sampling import sample_official_mcf_records
from run_static_overlap_mlp_pilot import References, emit, fit, select_layer
from static_overlap_core import StaticEditor, model_logits
from static_overlap_natural_writer import (
    METHOD,
    PLAN,
    encode_natural_views,
    mcf_facts,
    select_writer_channels,
    training_text_fingerprints,
)
from static_overlap_training import TrainConfig, export_verified


OFFICIAL_RETAIN_NUM = 1000


def _record_identity(record):
    return int(record["case_id"])


def sample_preservation_records(records, *, count, seed, reserved):
    """Sample fitting-only retain records disjoint from official retain."""
    half = len(records) // 2
    reserved_ids = {_record_identity(record) for record in reserved}
    pool = [
        record
        for record in records[:half]
        if _record_identity(record) not in reserved_ids
    ]
    if len(pool) < count:
        raise ValueError(
            f"Need {count} fitting-only retain records after reserving official retain; "
            f"only {len(pool)} remain"
        )
    return random.Random(seed + 104729).sample(pool, k=count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=PLAN["retain_num"])
    parser.add_argument("--seed", type=int, default=PLAN["seed"])
    args = parser.parse_args(argv)

    if args.forget_num != 50:
        raise ValueError("The registered natural-writer baseline uses forget_num=50")
    if args.seed != PLAN["seed"]:
        raise ValueError(f"The registered natural-writer baseline uses seed={PLAN['seed']}")
    if args.retain_num <= 0:
        raise ValueError("retain-num must be positive")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    if not model_path.is_dir() or not mcf_path.is_file():
        raise FileNotFoundError("Model directory or MCF JSON is missing")

    records = json.loads(mcf_path.read_text())
    forget_records, official_retain_records = sample_official_mcf_records(
        records,
        forget_num=args.forget_num,
        retain_num=OFFICIAL_RETAIN_NUM,
        seed=args.seed,
        strict=True,
    )
    preservation_records = sample_preservation_records(
        records,
        count=args.retain_num,
        seed=args.seed,
        reserved=official_retain_records,
    )
    forget_facts = mcf_facts(forget_records, "forget")
    retain_facts = mcf_facts(preservation_records, "retain")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    examples = encode_natural_views(
        forget_facts,
        retain_facts,
        tokenizer,
        PLAN["max_length"],
    )
    (output / "natural_writer_examples.json").write_text(
        json.dumps([asdict(example) for example in examples], indent=2) + "\n"
    )
    split_summary = {
        f"{split}/{role}": sum(
            e.split == split and e.role == role for e in examples
        )
        for split in ("train", "development")
        for role in ("forget", "retain")
    }
    emit(
        phase="natural_writer_data_ready",
        method=METHOD,
        forget_facts=len(forget_facts),
        fitting_retain_facts=len(retain_facts),
        official_retain_records_reserved=len(official_retain_records),
        examples=split_summary,
        private_tokens=False,
        runtime_router=False,
        official_paraphrase_fields_used=False,
        official_neighborhood_fields_used=False,
    )

    layer, layer_report = select_layer(
        model,
        examples,
        PLAN["candidate_layers"],
        PLAN["localization_examples"],
        PLAN["seed"],
    )
    (output / "writer_layer_localization.json").write_text(
        json.dumps(layer_report, indent=2) + "\n"
    )
    channels, channel_report = select_writer_channels(
        model,
        examples,
        layer,
        PLAN["localization_examples"],
        PLAN["writer_channels"],
        PLAN["seed"],
    )
    (output / "writer_channel_localization.json").write_text(
        json.dumps(channel_report, indent=2) + "\n"
    )
    emit(
        phase="natural_writer_localized",
        selected_layer=layer,
        selected_channels=len(channels),
        development_used_for_layer_selection=True,
        development_used_for_channel_selection=False,
    )

    references = References(output / "base_references")
    references.build(model, examples)
    editor = StaticEditor(model, [], [], {layer: channels}, rank=PLAN["rank"])
    if editor.rows or set(editor.downs) != {layer}:
        raise ValueError("Natural writer must edit exactly one MLP down-projection")

    # StaticEditor is zero initialized; assert ordinary natural-prompt logits are
    # bit exact before the first update.
    with torch.no_grad():
        edited = model_logits(editor.model, examples[0])
        with editor.base():
            base = model_logits(editor.model, examples[0])
        if not torch.equal(edited, base):
            raise ValueError("Zero-initialized natural writer changed base logits")
    emit(
        phase="natural_writer_preparation",
        layer=layer,
        channels=len(channels),
        rank=PLAN["rank"],
        base_logits_exact=True,
        tokenizer_extended=False,
        runtime_router=False,
    )

    config = TrainConfig(
        target_probability=PLAN["target_probability"],
        retain_nll_budget=0.05,
        retain_kl_budget=0.01,
        retain_nll_safety_margin=PLAN["fitting_nll_margin"],
        retain_kl_safety_margin=PLAN["fitting_kl_margin"],
    )
    report = fit(editor, examples, references, PLAN, config, output, method=METHOD)
    report.update({
        "method": METHOD,
        "architecture": "native_sparse_single_mlp_writer_low_rank",
        "layer_localization": layer_report,
        "channel_localization": channel_report,
        "private_tokens": False,
        "runtime_router": False,
        "official_eff_prompt_training_visible": True,
        "official_gen_paraphrases_training_visible": False,
        "official_retain_records_training_visible": False,
    })
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    if report["selected_step"] is None:
        emit(
            status="no_development_valid_natural_writer",
            report=str(output / "training_report.json"),
            official_evaluation_started=False,
        )
        return 2

    forget_manifest = [
        {
            key: fact[key]
            for key in (
                "id",
                "role",
                "subject",
                "relation",
                "object",
                "aliases",
                "answer_aliases",
            )
        }
        for fact in forget_facts
    ]
    manifest = {
        "method": METHOD,
        "architecture": "native_sparse_single_mlp_writer_low_rank",
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "forget_associations": forget_manifest,
        "forget_case_ids": [fact["case_id"] for fact in forget_facts],
        "training_text_fingerprints": training_text_fingerprints(examples),
        "settings": {
            "abstention": "",
            **PLAN,
            "selected_layer": layer,
            "selected_channels": channels,
            "fitting_retain_num": args.retain_num,
            "official_retain_reserved": OFFICIAL_RETAIN_NUM,
        },
        "runtime_router": False,
        "runtime_guard": False,
        "requires_fact_id_token_injection": False,
        "tokenizer_extended": False,
        "official_eff_prompt_training_visible": True,
        "official_gen_paraphrases_training_visible": False,
        "official_retain_records_training_visible": False,
    }

    def reload_native(path):
        return AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            local_files_only=True,
            attn_implementation="eager",
        ).to(args.device).eval()

    checkpoint = output / "checkpoint"
    export_report = export_verified(
        editor,
        tokenizer,
        examples,
        config,
        checkpoint,
        torch.float32,
        reload_native,
        atol=1e-4,
        rtol=1e-5,
        manifest=manifest,
        numeric_slack=0.0,
        require_forgetting=True,
    )
    report.update({
        "native_checkpoint_created": True,
        "checkpoint": str(checkpoint),
        "export": export_report,
        "official_evaluation_started": False,
    })
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    emit(
        status="verified_natural_writer_checkpoint",
        checkpoint=str(checkpoint),
        selected_layer=layer,
        selected_channels=len(channels),
        training_gate=report["last_gate"],
        runtime_router=False,
        private_tokens=False,
        official_evaluation_started=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
