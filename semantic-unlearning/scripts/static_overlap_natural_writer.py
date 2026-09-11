"""Natural-prompt sparse MLP writer localization for no-router unlearning.

This module intentionally uses ordinary tokenizer IDs only.  It selects a
single MLP writer layer and a sparse set of down-projection input channels whose
answer-NLL gradients are strong on forget prompts and comparatively quiet on
retain prompts.  The selected channels are later edited through StaticEditor and
merged into a native checkpoint; there is no runtime router or private token.
"""
from __future__ import annotations

from collections import defaultdict
import math
import random

import torch

from freeze_static_overlap_development import rewrite
from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES
from run_static_overlap_extended_tokens_standalone_v2 import (
    DEVELOPMENT_SCAFFOLDS,
    TRAIN_SCAFFOLDS,
)
from run_static_overlap_mlp_pilot import balanced_subset
from static_overlap_core import answer_nll, model_logits
from static_overlap_data import Example, _encode


METHOD = "static_overlap_natural_writer_v1"

PLAN = {
    "candidate_layers": [7, 11, 15, 19, 23, 27],
    "localization_examples": 24,
    "writer_channels": 512,
    "rank": 16,
    "steps": 300,
    "check_every": 20,
    "learning_rate": 0.003,
    "forget_batch": 8,
    "retain_batch": 16,
    "kl_weight": 10.0,
    "nll_weight": 10.0,
    "relative_delta_cap": 0.01,
    "max_training_seconds": 3600,
    "target_probability": 1e-6,
    "max_length": 512,
    "seed": 1,
    "retain_num": 300,
    "fitting_nll_margin": 0.01,
    "fitting_kl_margin": 0.002,
}

# Forgetting gets broad authored coverage. Retention uses a smaller prompt bank
# so the protection set remains tractable while still checking prompt transfer.
FORGET_TRAIN_SCAFFOLDS = TRAIN_SCAFFOLDS
FORGET_DEVELOPMENT_SCAFFOLDS = DEVELOPMENT_SCAFFOLDS
RETAIN_TRAIN_SCAFFOLDS = TRAIN_SCAFFOLDS[:2]
RETAIN_DEVELOPMENT_SCAFFOLDS = DEVELOPMENT_SCAFFOLDS[:2]


def mcf_facts(records, role):
    """Extract training-visible facts and the canonical rewrite prompt."""
    if role not in ("forget", "retain"):
        raise ValueError("role must be forget or retain")
    facts = []
    for record in records:
        rr = rewrite(record)
        target = rr["target_true"]
        answer = str(target["str"] if isinstance(target, dict) else target).strip()
        subject = str(rr["subject"]).strip()
        relation = str(rr["relation_id"]).strip()
        template = str(rr["prompt"])
        canonical_prompt = template.format(subject).strip()
        if not subject or not relation or not answer or not canonical_prompt:
            raise ValueError(f"Malformed MCF record: {record.get('case_id')}")
        facts.append({
            "id": f"mcf_{role}_{int(record['case_id'])}",
            "role": role,
            "subject": subject,
            "relation": relation,
            "object": answer,
            "aliases": [],
            "answer_aliases": [],
            "case_id": int(record["case_id"]),
            "canonical_prompt": canonical_prompt,
        })
    return facts


def _relation_noun(relation_id):
    if relation_id not in RELATION_NOUN_PHRASES:
        raise ValueError(f"No authored relation noun for {relation_id}")
    if relation_id == "P1412":
        return "language spoken or written"
    return RELATION_NOUN_PHRASES[relation_id]


def _encode_answer_example(fact, split, family, prompt, tokenizer, max_length):
    completion = " " + fact["object"]
    full = prompt + completion
    ids, offsets = _encode(tokenizer, full, max_length)
    start = len(prompt) + 1
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < len(full) and right > left
    ]
    if not positions or 0 in positions:
        raise ValueError(f"No answer tokens for {fact['id']} / {split} / {family}")
    for index in positions:
        left, right = offsets[index]
        if left < start and full[left:start].strip():
            raise ValueError(
                f"Tokenizer crosses prompt/answer boundary for {fact['id']} / {family}"
            )
        if right > len(full):
            raise ValueError("Tokenizer offset exceeds source text")
    labels = [token if index in positions else -100 for index, token in enumerate(ids)]
    return Example(
        id=f"{fact['id']}:{split}:{family}",
        split=split,
        role=fact["role"],
        fact_id=fact["id"],
        input_ids=ids,
        labels=labels,
        prompt=prompt,
        completion=completion,
        group=family,
    )


def encode_natural_views(forget_facts, retain_facts, tokenizer, max_length):
    """Build natural train/development examples without any private token."""
    examples = []
    for fact in [*forget_facts, *retain_facts]:
        relation = _relation_noun(fact["relation"])
        if fact["role"] == "forget":
            train_scaffolds = FORGET_TRAIN_SCAFFOLDS
            dev_scaffolds = FORGET_DEVELOPMENT_SCAFFOLDS
        else:
            train_scaffolds = RETAIN_TRAIN_SCAFFOLDS
            dev_scaffolds = RETAIN_DEVELOPMENT_SCAFFOLDS

        # The canonical requested_rewrite prompt is training-visible by design.
        # This closes the Eff input mismatch while official paraphrase prompts
        # remain unopened and therefore still test generalization (Gen).
        examples.append(
            _encode_answer_example(
                fact,
                "train",
                "canonical_rewrite",
                fact["canonical_prompt"],
                tokenizer,
                max_length,
            )
        )
        for split, scaffolds in (
            ("train", train_scaffolds),
            ("development", dev_scaffolds),
        ):
            for family, scaffold in enumerate(scaffolds):
                prompt = scaffold.format(subject=fact["subject"], relation=relation)
                examples.append(
                    _encode_answer_example(
                        fact,
                        split,
                        f"authored_{family}",
                        prompt,
                        tokenizer,
                        max_length,
                    )
                )

    # Tokenized duplicates can make a development gate look better than it is.
    # Deduplicate exact input/label pairs within the same role/split, but fail if
    # the same tokenized example crosses a split or role boundary.
    seen = {}
    result = []
    for example in examples:
        key = (tuple(example.input_ids), tuple(example.labels))
        previous = seen.get(key)
        if previous is not None:
            if previous != (example.split, example.role):
                raise ValueError(
                    f"Tokenized example crosses split/role boundaries: {example.id}"
                )
            continue
        seen[key] = (example.split, example.role)
        result.append(example)

    for split in ("train", "development"):
        for role in ("forget", "retain"):
            if not any(e.split == split and e.role == role for e in result):
                raise ValueError(f"Empty natural-writer stratum {split}/{role}")
    return result


def _mean_channel_gradient(model, examples, layer):
    """Return per-input-channel norm of the mean down_proj weight gradient."""
    if not examples:
        raise ValueError("Channel localization requires nonempty examples")
    down = model.model.layers[layer].mlp.down_proj
    model.requires_grad_(False)
    down.weight.requires_grad_(True)
    try:
        model.zero_grad(set_to_none=True)
        for example in examples:
            (answer_nll(model_logits(model, example), example) / len(examples)).backward()
        if down.weight.grad is None or not torch.isfinite(down.weight.grad).all():
            raise ValueError("Non-finite or missing writer-localization gradient")
        return down.weight.grad.detach().float().square().sum(0).sqrt().cpu()
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)


def select_writer_channels(model, examples, layer, example_count, channel_count, seed):
    """Select forget-sensitive, retain-quiet MLP input channels on fitting data."""
    train_forget = balanced_subset(
        [e for e in examples if e.split == "train" and e.role == "forget"],
        example_count,
        seed,
    )
    train_retain = balanced_subset(
        [e for e in examples if e.split == "train" and e.role == "retain"],
        example_count,
        seed + 1,
    )
    forget = _mean_channel_gradient(model, train_forget, layer)
    retain = _mean_channel_gradient(model, train_retain, layer)
    if forget.shape != retain.shape:
        raise ValueError("Forget/retain channel gradients differ in shape")
    if not 0 < int(channel_count) <= int(forget.numel()):
        raise ValueError("writer_channels is outside the selected MLP width")

    # A small data-derived denominator floor avoids selecting channels that are
    # numerically tiny for both roles solely because retain sensitivity is ~0.
    nonzero = retain[retain > 0]
    floor = (
        float(nonzero.median()) * 0.01
        if int(nonzero.numel())
        else max(float(forget.max()) * 1e-6, 1e-12)
    )
    score = forget / (retain + floor)
    selected = torch.topk(score, k=int(channel_count), largest=True).indices.tolist()
    selected = sorted(int(index) for index in selected)
    top = torch.topk(score, k=min(20, int(channel_count)), largest=True).indices.tolist()

    diagnostics = {
        "layer": int(layer),
        "score": "forget_gradient_l2 / (retain_gradient_l2 + 1pct_retain_median_floor)",
        "denominator_floor": floor,
        "selected_channel_count": len(selected),
        "selected_channels": selected,
        "top_channels": [
            {
                "channel": int(index),
                "score": float(score[index]),
                "forget_gradient_norm": float(forget[index]),
                "retain_gradient_norm": float(retain[index]),
            }
            for index in top
        ],
        "forget_example_ids": [e.id for e in train_forget],
        "retain_example_ids": [e.id for e in train_retain],
        "development_used_for_channel_selection": False,
    }
    return selected, diagnostics


def training_text_fingerprints(examples):
    return sorted({
        " ".join(text.casefold().split())
        for example in examples
        for text in (example.prompt, example.prompt + example.completion)
    })
