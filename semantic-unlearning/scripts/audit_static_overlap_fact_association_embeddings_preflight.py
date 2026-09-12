#!/usr/bin/env python3
"""Training-safe preflight for a saved fact-association embedding checkpoint.

This script deliberately does NOT open MCF benchmark paraphrases, neighborhoods,
or official retain records.  It checks:
  1) prompt-prefix routing invariance to appended candidate suffixes,
  2) same-subject/different-relation collateral effects,
  3) authored train/development forgetting after reload in the requested dtype,
  4) teacher-forced all-token top-1 correctness on authored prompts.

It is an audit, not a checkpoint selector.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from mcf_zero_unlearn_official_eval import dtype_from_str
from static_overlap_data import Example
from static_overlap_extended_tokens_v2 import routed_metrics
from static_overlap_fact_association_embeddings import (
    load_artifact_into_model,
    make_unknown_examples,
    relation_negative_prompts,
)


def load_examples(path):
    rows = json.loads(Path(path).read_text())
    return [Example(**row) for row in rows]


@torch.no_grad()
def prompt_distributions(model, tokenizer, prompts, batch_size=8, bank=None):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    log_probs = []
    top1 = []
    routes = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = prompts[start:start + int(batch_size)]
        encoded = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        logits = model(**encoded, use_cache=False).logits.float()
        mask = encoded["attention_mask"].bool()
        positions = (
            torch.arange(mask.shape[1], device=device)[None, :]
            .expand_as(mask)
            .masked_fill(~mask, -1)
            .max(dim=1)
            .values
        )
        if bool((positions < 0).any()):
            raise ValueError("Empty prompt in collateral audit")
        final = logits[
            torch.arange(logits.shape[0], device=device),
            positions,
        ]
        lp = final.log_softmax(-1)
        log_probs.extend(row.cpu() for row in lp)
        top1.extend(int(x) for x in final.argmax(-1).cpu().tolist())
        if bank is not None:
            current = list(bank.last_active_fact_indices)
            if len(current) != len(batch):
                raise RuntimeError("Route capture length mismatch")
            routes.extend(current)
    return log_probs, top1, routes


@torch.no_grad()
def authored_top1_summary(model, examples, batch_size=16):
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start:start + int(batch_size)]
        maximum = max(len(e.input_ids) for e in batch)
        ids = torch.zeros((len(batch), maximum), dtype=torch.long, device=device)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for index, example in enumerate(batch):
            length = len(example.input_ids)
            ids[index, :length] = torch.tensor(example.input_ids, device=device)
            attention[index, :length] = 1
            labels[index, :length] = torch.tensor(example.labels, device=device)
        prefix_lengths = []
        for row in labels:
            positions = (row != -100).nonzero(as_tuple=False).reshape(-1)
            if not int(positions.numel()):
                raise ValueError("Authored example has no answer labels")
            prefix_lengths.append(int(positions.min().item()))
        if hasattr(model, "set_association_prefix_lengths"):
            model.set_association_prefix_lengths(prefix_lengths)
        logits = model(
            input_ids=ids,
            attention_mask=attention,
            use_cache=False,
        ).logits.float()
        predictions = logits[:, :-1].argmax(-1)
        shifted = labels[:, 1:]
        for index, example in enumerate(batch):
            mask = shifted[index] != -100
            if not bool(mask.any()):
                raise ValueError("No shifted answer labels")
            correct = bool(
                (predictions[index][mask] == shifted[index][mask]).all()
            )
            rows.append({
                "id": example.id,
                "fact_id": example.fact_id,
                "split": example.split,
                "all_answer_tokens_top1_correct": correct,
            })

    result = {}
    for split in ("train", "development"):
        current = [row for row in rows if row["split"] == split]
        by_fact = {}
        for row in current:
            by_fact.setdefault(row["fact_id"], []).append(row)
        failing_facts = sorted(
            fact_id
            for fact_id, values in by_fact.items()
            if any(row["all_answer_tokens_top1_correct"] for row in values)
        )
        correct_prompts = sum(
            row["all_answer_tokens_top1_correct"] for row in current
        )
        result[split] = {
            "prompt_count": len(current),
            "prompts_still_all_token_top1_correct": correct_prompts,
            "fraction_still_all_token_top1_correct": (
                correct_prompts / len(current) if current else None
            ),
            "facts_with_any_all_token_top1_correct_prompt": len(failing_facts),
            "facts_fully_not_all_token_top1_correct": (
                len(by_fact) - len(failing_facts)
            ),
            "facts_total": len(by_fact),
            "failing_fact_ids": failing_facts,
        }
    return result


@torch.no_grad()
def prefix_invariance_audit(model, bank, tokenizer, facts):
    prompt = "A neutral preface discussing arithmetic and weather."
    suffix_a = " " + str(facts[0]["subject"])
    suffix_b = " an ordinary continuation with no named entity."
    a = tokenizer(
        prompt + suffix_a,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    b = tokenizer(
        prompt + suffix_b,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    ids_a = a["input_ids"][0].tolist()
    ids_b = b["input_ids"][0].tolist()
    common = 0
    for left, right in zip(ids_a, ids_b):
        if left != right:
            break
        common += 1
    if common <= 1:
        raise RuntimeError("Could not construct a stable common prompt-token prefix")

    device = next(model.parameters()).device
    ta = a.to(device)
    tb = b.to(device)
    model.set_association_prefix_lengths([common])
    la = model(**ta, use_cache=False).logits[:, common - 1].float().cpu()
    route_a = list(bank.last_active_fact_indices)
    model.set_association_prefix_lengths([common])
    lb = model(**tb, use_cache=False).logits[:, common - 1].float().cpu()
    route_b = list(bank.last_active_fact_indices)
    exact = torch.equal(la, lb)
    return {
        "common_prompt_token_count": common,
        "suffix_a_contains_forgotten_subject": str(facts[0]["subject"]),
        "route_a": route_a,
        "route_b": route_b,
        "route_equal": route_a == route_b,
        "first_answer_logits_bit_exact": exact,
        "max_abs_first_answer_logit_difference": float((la - lb).abs().max()),
        "passed": route_a == route_b and exact,
    }



def compare_prompt_group(
    name,
    prompts,
    base_lp,
    base_top1,
    edited_lp,
    edited_top1,
    routes,
    expected_owners=None,
):
    kl = []
    for p, q in zip(base_lp, edited_lp):
        p = p.float()
        q = q.float()
        kl.append(float((p.exp() * (p - q)).sum().clamp_min(0)))
    top1_changed = sum(a != b for a, b in zip(base_top1, edited_top1))
    result = {
        "name": name,
        "prompt_count": len(prompts),
        "runtime_any_route_fraction": (
            sum(bool(route) for route in routes) / len(routes)
            if routes else None
        ),
        "next_token_top1_changed_fraction": (
            top1_changed / len(prompts) if prompts else None
        ),
        "base_to_edited_next_token_kl_mean": (
            sum(kl) / len(kl) if kl else None
        ),
        "base_to_edited_next_token_kl_max": max(kl) if kl else None,
        "worst_kl": [
            {
                "prompt": prompts[index],
                "kl": kl[index],
                "base_top1": base_top1[index],
                "edited_top1": edited_top1[index],
                "route": routes[index],
            }
            for index in sorted(
                range(len(kl)), key=lambda i: kl[i], reverse=True
            )[:20]
        ],
    }
    if expected_owners is not None:
        result["runtime_expected_owner_route_fraction"] = (
            sum(
                owner in route
                for owner, route in zip(expected_owners, routes)
            ) / len(routes)
        )
        result["runtime_wrong_owner_route_fraction"] = (
            sum(
                any(index != owner for index in route)
                for owner, route in zip(expected_owners, routes)
            ) / len(routes)
        )
    return result

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest_path = run_dir / "association_manifest.json"
    artifact_path = run_dir / "fact_association_embeddings.pt"
    examples_path = run_dir / "association_examples.json"
    report_path = run_dir / "training_report.json"
    for path in (manifest_path, artifact_path, examples_path, report_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text())
    parent_report = json.loads(report_path.read_text())
    examples = load_examples(examples_path)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    facts = artifact["facts"]
    model_path = Path(manifest["model_path"]).resolve()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = dtype_from_str(args.dtype)
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base.requires_grad_(False)

    same_subject_prompts = []
    same_subject_owner = []
    count = int(manifest["plan"].get("relation_negative_count", 8))
    for index, fact in enumerate(facts):
        prompts = relation_negative_prompts(fact, facts, count)
        same_subject_prompts.extend(prompts)
        same_subject_owner.extend([index] * len(prompts))

    incidental_subject_prompts = [
        (
            f"Background context briefly mentions {fact['subject']}. "
            "Continue with a neutral observation about careful documentation:"
        )
        for fact in facts
    ]
    incidental_owner = list(range(len(facts)))
    object_only_prompts = [
        (
            f"Use the phrase {fact['object']} in a neutral, unrelated sentence "
            "about writing style:"
        )
        for fact in facts
    ]

    base_same_lp, base_same_top1, _ = prompt_distributions(
        base,
        tokenizer,
        same_subject_prompts,
        batch_size=args.batch_size,
    )
    base_incidental_lp, base_incidental_top1, _ = prompt_distributions(
        base,
        tokenizer,
        incidental_subject_prompts,
        batch_size=args.batch_size,
    )
    base_object_lp, base_object_top1, _ = prompt_distributions(
        base,
        tokenizer,
        object_only_prompts,
        batch_size=args.batch_size,
    )

    edited, bank = load_artifact_into_model(base, artifact)
    edited.eval()
    edited_same_lp, edited_same_top1, same_routes = prompt_distributions(
        edited,
        tokenizer,
        same_subject_prompts,
        batch_size=args.batch_size,
        bank=bank,
    )
    edited_incidental_lp, edited_incidental_top1, incidental_routes = (
        prompt_distributions(
            edited,
            tokenizer,
            incidental_subject_prompts,
            batch_size=args.batch_size,
            bank=bank,
        )
    )
    edited_object_lp, edited_object_top1, object_routes = prompt_distributions(
        edited,
        tokenizer,
        object_only_prompts,
        batch_size=args.batch_size,
        bank=bank,
    )

    same_subject = compare_prompt_group(
        "same_subject_different_relation",
        same_subject_prompts,
        base_same_lp,
        base_same_top1,
        edited_same_lp,
        edited_same_top1,
        same_routes,
        expected_owners=same_subject_owner,
    )
    same_subject["prompt_source"] = (
        "training-safe same-subject/different-relation authored controls; "
        "no official MCF paraphrases/neighborhoods/retain sample"
    )
    incidental_subject = compare_prompt_group(
        "incidental_subject_mention",
        incidental_subject_prompts,
        base_incidental_lp,
        base_incidental_top1,
        edited_incidental_lp,
        edited_incidental_top1,
        incidental_routes,
        expected_owners=incidental_owner,
    )
    incidental_subject["prompt_source"] = "synthetic training-safe incidental mentions"
    object_only = compare_prompt_group(
        "forgotten_object_without_intended_subject",
        object_only_prompts,
        base_object_lp,
        base_object_top1,
        edited_object_lp,
        edited_object_top1,
        object_routes,
    )
    object_only["prompt_source"] = "synthetic training-safe object-only mentions"

    prefix_invariance = prefix_invariance_audit(
        edited, bank, tokenizer, facts
    )

    answer_map = {example.id: example for example in examples}
    unknown_map = make_unknown_examples(
        examples,
        tokenizer,
        int(manifest["plan"]["max_length"]),
        str(manifest["plan"]["unknown_completion"]),
    )
    authored_metrics = routed_metrics(
        edited,
        answer_map,
        unknown_map,
        float(manifest["plan"]["target_probability"]),
    )
    authored_top1 = authored_top1_summary(edited, examples)

    result = {
        "kind": "fact_association_embedding_training_safe_preflight_v1",
        "run_dir": str(run_dir),
        "dtype": str(args.dtype),
        "official_evaluation_opened": False,
        "prefix_invariance": prefix_invariance,
        "same_subject_different_relation": same_subject,
        "incidental_subject_mention": incidental_subject,
        "forgotten_object_only": object_only,
        "authored_reload_metrics": authored_metrics,
        "authored_all_token_top1": authored_top1,
        "parent_fp32_final_metrics": parent_report.get("final_metrics"),
        "notes": {
            "historical_token_probability_definition": (
                "exp(-mean teacher-forced answer-token NLL), i.e. geometric mean"
            ),
            "generation_audit_included": False,
        },
    }
    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / f"training_safe_preflight_{args.dtype}.json"
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "prefix_invariance": prefix_invariance,
        "same_subject": {
            key: value
            for key, value in same_subject.items()
            if key != "worst_kl"
        },
        "incidental_subject": {
            key: value
            for key, value in incidental_subject.items()
            if key != "worst_kl"
        },
        "object_only": {
            key: value
            for key, value in object_only.items()
            if key != "worst_kl"
        },
        "authored_train": authored_metrics["train"],
        "authored_development": authored_metrics["development"],
        "authored_top1": authored_top1,
        "out": str(out),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
