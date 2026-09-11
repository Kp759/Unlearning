#!/usr/bin/env python3
"""Run Extended Association Tokens v2 from base Llama and MCF only.

The experiment uses independent private input rows, eight authored training
views per fact, four disjoint development views, worst-view answer suppression,
and maximum-first checkpoint selection.  It never reads official MCF
paraphrase or neighborhood prompts and never starts official evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES
from run_static_overlap_extended_tokens_standalone import (
    assert_natural_parity,
    mcf_forget_facts,
    natural_logit_snapshots,
)
from mcf_sampling import sample_official_mcf_records
from run_static_overlap_mlp_pilot import emit
from static_overlap_data import Example, _encode
from static_overlap_extended_tokens import (
    association_token_specs,
    build_routed_examples,
    initialize_rows,
)
from static_overlap_extended_tokens_v2 import (
    RowWiseExtendedTokenEditor,
    routed_metrics,
    train_row_wise,
)


METHOD = "static_overlap_extended_tokens_standalone_v2"

# These banks were authored independently of MultiCounterFact's official
# paraphrase_prompts and neighborhood_prompts.  Development strings are never
# included in an optimizer loss.
TRAIN_SCAFFOLDS = (
    "What is the {relation} of {subject}? Answer:",
    "Which {relation} is associated with {subject}? Answer:",
    "For {subject}, identify the {relation}:",
    "The {relation} recorded for {subject} is",
    "Name the {relation} linked to {subject}:",
    "In reference to {subject}, sources list the {relation} as",
    "What value is given for {subject}'s {relation}? Response:",
    "Complete this fact about {subject}: the {relation} is",
)
DEVELOPMENT_SCAFFOLDS = (
    "Supply the {relation} attributed to {subject}:",
    "Looking up {subject}, which {relation} is listed?",
    "The requested {relation} entry concerning {subject} reads",
    "State what is documented as the {relation} of {subject}:",
)

PLAN = {
    "steps": 1500,
    "check_every": 50,
    "learning_rate": 0.05,
    "backtracks": 8,
    "max_training_seconds": 3600,
    "max_stalled_steps": 100,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "association_tokens": 50,
    "training_views_per_fact": len(TRAIN_SCAFFOLDS),
    "development_views_per_fact": len(DEVELOPMENT_SCAFFOLDS),
    "unknown_completion": " I don't know.",
    "unknown_weight": 1.0,
    # Evaluated in order.  A row below target remains trainable for abstention,
    # but every accepted proposal must keep its worst answer view below target.
    "radius_schedule": (
        (1e-3, 1.0),
        (1e-5, 0.25),
        (1e-6, 0.05),
        (0.0, 0.01),
    ),
}


def encode_v2_authored_views(facts, tokenizer, max_length):
    """Build disjoint 8/4 train/development prompt families from relation names."""
    if set(TRAIN_SCAFFOLDS) & set(DEVELOPMENT_SCAFFOLDS):
        raise ValueError("Training and development scaffold banks overlap")
    examples = []
    for fact in facts:
        relation_id = fact["relation"]
        if relation_id not in RELATION_NOUN_PHRASES:
            raise ValueError(f"No authored relation noun for {relation_id}")
        relation = RELATION_NOUN_PHRASES[relation_id]
        if relation_id == "P1412":
            relation = "language spoken or written"
        for split, scaffolds in (
            ("train", TRAIN_SCAFFOLDS),
            ("development", DEVELOPMENT_SCAFFOLDS),
        ):
            for family, scaffold in enumerate(scaffolds):
                prompt = scaffold.format(subject=fact["subject"], relation=relation)
                completion = " " + fact["object"]
                full = prompt + completion
                ids, offsets = _encode(tokenizer, full, max_length)
                start = len(prompt) + 1
                positions = [
                    index for index, (left, right) in enumerate(offsets)
                    if right > start and left < len(full) and right > left
                ]
                if not positions or 0 in positions:
                    raise ValueError(f"No answer tokens for {fact['id']} / {split}_{family}")
                labels = [token if index in positions else -100 for index, token in enumerate(ids)]
                examples.append(Example(
                    id=f"{fact['id']}:{split}:authored_{family}",
                    split=split,
                    role="forget",
                    fact_id=fact["id"],
                    input_ids=ids,
                    labels=labels,
                    prompt=prompt,
                    completion=completion,
                    group=f"{split}_authored_{family}",
                ))
    for fact in facts:
        for split, expected in (
            ("train", len(TRAIN_SCAFFOLDS)),
            ("development", len(DEVELOPMENT_SCAFFOLDS)),
        ):
            actual = sum(
                example.fact_id == fact["id"] and example.split == split
                for example in examples
            )
            if actual != expected:
                raise ValueError(
                    f"Expected {expected} {split} views for {fact['id']}; got {actual}"
                )
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    if args.forget_num != 50 or args.seed != 1:
        raise ValueError("This registered standalone ablation is fixed to forget_num=50, seed=1.")

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    if not model_path.is_dir() or not mcf_path.is_file():
        raise FileNotFoundError("Model directory or MCF JSON is missing.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    records = json.loads(mcf_path.read_text())
    forget_records, _ = sample_official_mcf_records(
        records, forget_num=args.forget_num, retain_num=0, seed=args.seed, strict=True
    )
    facts_with_case_ids = mcf_forget_facts(forget_records)
    facts = [
        {key: value for key, value in fact.items() if key != "case_id"}
        for fact in facts_with_case_ids
    ]
    examples = encode_v2_authored_views(facts, tokenizer, PLAN["max_length"])
    emit(
        phase="standalone_mcf_v2_ready",
        method=METHOD,
        forget_facts=len(facts),
        train_views=sum(example.split == "train" for example in examples),
        development_views=sum(example.split == "development" for example in examples),
        official_paraphrase_fields_used=False,
        official_neighborhood_fields_used=False,
    )

    original_vocab_size = model.get_input_embeddings().num_embeddings
    specs = association_token_specs(facts)
    if len(specs) != PLAN["association_tokens"]:
        raise ValueError("Expected exactly 50 private association tokens.")
    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [spec["token"] for spec in specs]
    })
    if added != len(specs) or len(tokenizer) != original_vocab_size + len(specs):
        raise ValueError("Tokenizer extension is not a clean 50-row append.")
    if model.get_output_embeddings().weight.shape[0] != original_vocab_size:
        raise ValueError("Output vocabulary must remain unchanged.")

    facts_by_id = {fact["id"]: fact for fact in facts}
    initial_rows = initialize_rows(model, tokenizer, facts_by_id, specs)
    natural_before = natural_logit_snapshots(model, examples)
    editor = RowWiseExtendedTokenEditor(model, initial_rows)
    assert_natural_parity(editor.model, examples, natural_before)

    routed_answer, routed_unknown = build_routed_examples(
        examples,
        tokenizer,
        specs,
        original_vocab_size,
        PLAN["unknown_completion"],
    )
    fact_to_row = {spec["fact_id"]: index for index, spec in enumerate(specs)}
    manifest = {
        "method": METHOD,
        "architecture": "input_only_extended_association_tokens_row_wise_v2",
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "sampling": {
            "forget_num": 50,
            "retain_num": 0,
            "seed": 1,
            "convention": "ZeroUnlearn/official MCF: forget sampled from half two",
        },
        "plan": {**PLAN, "radius_schedule": [list(item) for item in PLAN["radius_schedule"]]},
        "association_tokens": specs,
        "forget_case_ids": [fact["case_id"] for fact in facts_with_case_ids],
        "original_vocab_size": original_vocab_size,
        "extended_input_vocab_size": len(tokenizer),
        "output_vocab_size": model.get_output_embeddings().weight.shape[0],
        "base_parameters_trainable": 0,
        "independent_trainable_rows": len(editor.parameters),
        "extended_input_parameters": sum(row.numel() for row in editor.parameters),
        "training_scaffolds": list(TRAIN_SCAFFOLDS),
        "development_scaffolds": list(DEVELOPMENT_SCAFFOLDS),
        "training_prompt_source": "eight independently authored relation-noun templates",
        "development_prompt_source": "four disjoint independently authored relation-noun templates",
        "optimization": (
            "one Adam optimizer per fact row; worst answer view plus mean abstention NLL"
        ),
        "acceptance": (
            "worst answer probability monotonic until target; then target constrained "
            "abstention improvement"
        ),
        "checkpoint_selection": (
            "global train/development maximum answer probability then mean abstention NLL"
        ),
        "official_paraphrase_fields_used": False,
        "official_neighborhood_fields_used": False,
        "development_views_used_for_gradients": False,
        "natural_prompt_logits_exact_base": True,
        "requires_fact_id_token_injection": True,
        "official_evaluation_eligible": False,
        "final_tests_touched": False,
    }
    (output / "standalone_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "standalone_examples.json").write_text(
        json.dumps([asdict(example) for example in examples], indent=2) + "\n"
    )
    tokenizer.save_pretrained(output / "extended_tokenizer")

    report = train_row_wise(
        editor,
        examples,
        routed_answer,
        routed_unknown,
        fact_to_row,
        PLAN,
        output,
    )
    assert_natural_parity(editor.model, examples, natural_before)
    final_metrics = routed_metrics(
        editor.model, routed_answer, routed_unknown, PLAN["target_probability"]
    )
    report.update({
        "method": METHOD,
        "manifest": manifest,
        "final_metrics": final_metrics,
        "natural_prompt_logits_exact_base_after_training": True,
        "official_evaluation_started": False,
        "final_tests_touched": False,
    })
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(editor.artifact(), output / "extended_input_rows.pt")
    emit(
        status="standalone_extended_token_v2_oracle_ablation_complete",
        stop_reason=report["stop_reason"],
        best_step=report["best_step"],
        final_metrics=final_metrics,
        output=str(output),
        natural_prompt_logits_exact_base=True,
        official_evaluation_started=False,
        final_tests_touched=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
