#!/usr/bin/env python3
"""Standalone oracle-routed extended-token ablation from base Llama + MCF only.

This intentionally does NOT read MCF paraphrase_prompts/neighborhood_prompts and
does NOT depend on any previous static-overlap/head/MLP experiment artifacts.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from freeze_static_overlap_development import rewrite
from mcf_sampling import sample_official_mcf_records
from run_static_overlap_mlp_pilot import emit
from static_overlap_data import Example, _encode
from static_overlap_extended_tokens import (
    ExtendedTokenEditor,
    association_token_specs,
    build_routed_examples,
    initialize_rows,
    routed_forgetting,
    routed_unknown_completion,
    train_extended_tokens,
)
from static_overlap_mlp_protocol import authored_prompts
from static_overlap_training import TrainConfig


METHOD = "static_overlap_extended_tokens_standalone_v1"
PLAN = {
    "steps": 300,
    "check_every": 10,
    "learning_rate": .05,
    "forget_batch": 8,
    "step_radius": 1.,
    "backtracks": 8,
    "max_training_seconds": 1800,
    "max_stalled_steps": 20,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "association_tokens": 50,
    "unknown_completion": " I don't know.",
    "unknown_weight": 1.,
}


def mcf_forget_facts(records):
    facts = []
    for record in records:
        rr = rewrite(record)
        target = rr["target_true"]
        answer = str(target["str"] if isinstance(target, dict) else target).strip()
        subject = str(rr["subject"]).strip()
        relation = str(rr["relation_id"]).strip()
        if not subject or not relation or not answer:
            raise ValueError(f"Malformed MCF forget record: {record.get('case_id')}")
        facts.append({
            "id": f"mcf_forget_{int(record['case_id'])}",
            "role": "forget",
            "subject": subject,
            "relation": relation,
            "object": answer,
            "aliases": [],
            "answer_aliases": [],
            "case_id": int(record["case_id"]),
        })
    return facts


def encode_authored_views(facts, tokenizer, max_length):
    """Use authored relation templates only; never benchmark paraphrase fields."""
    examples = []
    for fact in facts:
        for split in ("train", "development"):
            for family, prompt in authored_prompts(fact, split):
                completion = " " + fact["object"]
                full = prompt + completion
                ids, offsets = _encode(tokenizer, full, max_length)
                start = len(prompt) + 1
                positions = [
                    i for i, (a, b) in enumerate(offsets)
                    if b > start and a < len(full) and b > a
                ]
                if not positions or 0 in positions:
                    raise ValueError(f"No answer tokens for {fact['id']} / {family}")
                labels = [
                    token if i in positions else -100
                    for i, token in enumerate(ids)
                ]
                examples.append(Example(
                    id=f"{fact['id']}:{split}:{family}",
                    split=split,
                    role="forget",
                    fact_id=fact["id"],
                    input_ids=ids,
                    labels=labels,
                    prompt=prompt,
                    completion=completion,
                    group=family,
                ))
    for split in ("train", "development"):
        missing = [
            fact["id"] for fact in facts
            if not any(e.fact_id == fact["id"] and e.split == split for e in examples)
        ]
        if missing:
            raise ValueError(f"Missing {split} authored views: {missing[:3]}")
    return examples


@torch.no_grad()
def natural_logit_snapshots(model, examples, per_split=4):
    chosen = []
    for split in ("train", "development"):
        chosen.extend([e for e in examples if e.split == split][:per_split])
    return {
        e.id: model(
            input_ids=torch.tensor([e.input_ids], device=next(model.parameters()).device),
            use_cache=False,
        ).logits.detach().cpu()
        for e in chosen
    }


@torch.no_grad()
def assert_natural_parity(model, examples, snapshots):
    by_id = {e.id: e for e in examples}
    for eid, expected in snapshots.items():
        e = by_id[eid]
        actual = model(
            input_ids=torch.tensor([e.input_ids], device=next(model.parameters()).device),
            use_cache=False,
        ).logits.detach().cpu()
        if not torch.equal(expected, actual):
            raise ValueError(f"Natural-prompt full-logit parity failed: {eid}")


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
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    data = json.loads(mcf_path.read_text())
    forget_records, _ = sample_official_mcf_records(
        data, forget_num=args.forget_num, retain_num=0, seed=args.seed, strict=True
    )
    facts = mcf_forget_facts(forget_records)
    source_facts = [{k: v for k, v in fact.items() if k != "case_id"} for fact in facts]

    # Deliberately construct prompts only from requested_rewrite subject/relation/target_true.
    # No official paraphrase_prompts or neighborhood_prompts are read.
    examples = encode_authored_views(source_facts, tokenizer, PLAN["max_length"])
    emit(
        phase="standalone_mcf_ready",
        method=METHOD,
        forget_facts=len(source_facts),
        train_views=sum(e.split == "train" for e in examples),
        development_views=sum(e.split == "development" for e in examples),
        official_paraphrase_fields_used=False,
        official_neighborhood_fields_used=False,
    )

    original_vocab_size = model.get_input_embeddings().num_embeddings
    specs = association_token_specs(source_facts)
    if len(specs) != PLAN["association_tokens"]:
        raise ValueError("Expected exactly 50 private association tokens.")

    added = tokenizer.add_special_tokens({
        "additional_special_tokens": [row["token"] for row in specs]
    })
    if added != len(specs) or len(tokenizer) != original_vocab_size + len(specs):
        raise ValueError("Tokenizer extension is not a clean 50-row append.")
    if model.get_output_embeddings().weight.shape[0] != original_vocab_size:
        raise ValueError("Output vocabulary must remain unchanged.")

    facts_by_id = {fact["id"]: fact for fact in source_facts}
    initial_rows = initialize_rows(model, tokenizer, facts_by_id, specs)

    config = TrainConfig(
        target_probability=PLAN["target_probability"],
        retain_nll_budget=.05,
        retain_kl_budget=.01,
        retain_nll_safety_margin=0.,
        retain_kl_safety_margin=0.,
    )

    from static_overlap_core import answer_nll, model_logits
    base_nll = {
        e.id: float(answer_nll(model_logits(model, e), e))
        for e in examples
    }
    natural_before = natural_logit_snapshots(model, examples)

    editor = ExtendedTokenEditor(model, initial_rows)
    assert_natural_parity(editor.model, examples, natural_before)

    routed_answer, routed_unknown = build_routed_examples(
        examples,
        tokenizer,
        specs,
        original_vocab_size,
        PLAN["unknown_completion"],
    )

    manifest = {
        "method": METHOD,
        "architecture": "input_only_extended_association_tokens_standalone_v1",
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "sampling": {"forget_num": 50, "retain_num": 0, "seed": 1,
                     "convention": "ZeroUnlearn/official MCF: forget sampled from half two"},
        "association_tokens": specs,
        "forget_case_ids": [fact["case_id"] for fact in facts],
        "original_vocab_size": original_vocab_size,
        "extended_input_vocab_size": len(tokenizer),
        "output_vocab_size": model.get_output_embeddings().weight.shape[0],
        "base_parameters_trainable": 0,
        "extended_input_parameters": editor.embedding.extra.numel(),
        "unknown_completion": PLAN["unknown_completion"],
        "training_prompt_source": "authored relation templates only",
        "official_paraphrase_fields_used": False,
        "official_neighborhood_fields_used": False,
        "natural_prompt_logits_exact_base": True,
        "requires_fact_id_token_injection": True,
        "official_evaluation_eligible": False,
        "final_tests_touched": False,
    }
    (output / "standalone_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "standalone_examples.json").write_text(
        json.dumps([asdict(e) for e in examples], indent=2) + "\n"
    )
    tokenizer.save_pretrained(output / "extended_tokenizer")

    report = train_extended_tokens(
        editor, examples, routed_answer, routed_unknown,
        base_nll, config, PLAN, output
    )

    # The scientific guarantee of this ablation: ordinary prompts remain exact.
    assert_natural_parity(editor.model, examples, natural_before)
    routed_final = routed_forgetting(editor.model, routed_answer, base_nll, config)
    unknown_final = routed_unknown_completion(editor.model, routed_unknown)

    report.update({
        "method": METHOD,
        "manifest": manifest,
        "final_routed_forgetting": routed_final,
        "final_routed_unknown_completion": unknown_final,
        "natural_prompt_logits_exact_base_after_training": True,
        "official_evaluation_started": False,
        "final_tests_touched": False,
    })
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(editor.artifact(), output / "extended_input_rows.pt")
    emit(
        status="standalone_extended_token_oracle_ablation_complete",
        stop_reason=report["stop_reason"],
        output=str(output),
        natural_prompt_logits_exact_base=True,
        official_evaluation_started=False,
        final_tests_touched=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
