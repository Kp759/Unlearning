#!/usr/bin/env python3
"""Compatibility evaluator that adds released-table MCF correctness metrics.

The existing :mod:`mcf_zero_unlearn_official_eval` evaluator is retained as the
source of split construction, summary statistics, PPL, and serialized result
shape.  This module changes only the per-prompt prediction primitive so that a
single forward pass records BOTH:

1. the legacy target_new/target_true token-average NLLs; and
2. teacher-forced whole-target top-1 correctness for target_true.

For MCF, ``target_true_correct`` is True iff every token in the sensitive
``target_true`` completion is the model's top-1 token under teacher forcing.
This is the discrete accuracy quantity used by the released CounterFact
ZeroUnlearn evaluation path for table-style Eff/Gen/Spe.

No model parameters, ZeroUnlearn algorithm code, hparams, splits, or PPL logic
are changed here.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import mcf_zero_unlearn_official_eval as legacy


@torch.no_grad()
def official_test_batch_prediction_dual(
    model: Any,
    tok: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    device: torch.device,
    llama_like: bool = True,
) -> list[dict[str, Any]]:
    """Return NLL pairs plus exact teacher-forced whole-target correctness.

    The tokenization/indexing mirrors the existing legacy evaluator.  Both
    completions are kept in the batch so the NLL values are directly comparable
    with earlier runs.  Correctness is recorded for both completions, although
    the released MCF table-style metrics consume ``target_true_correct``.
    """
    if len(prefixes) == 0:
        return []

    prefix_lens = [len(x) for x in tok(list(prefixes))["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in (target_new, target_true)
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (tok(f" {x}")["input_ids"] for x in (target_new, target_true))
    if llama_like:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [x - 1 for x in prefix_lens]

    logits = model(**prompt_tok).logits
    if llama_like:
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    correct = np.ones((logits.size(0),), dtype=np.bool_)

    for i in range(logits.size(0)):
        cur_tokens = a_tok if i % 2 == 0 else b_tok
        cur_len = len(cur_tokens)
        for j, cur_tok in enumerate(cur_tokens):
            pos = prefix_lens[i // 2] + j - 1
            token_logits = logits[i, pos, :]
            probs[i] += -torch.nn.functional.log_softmax(
                token_logits, dim=0
            )[cur_tok].item()
            if int(token_logits.argmax().item()) != int(cur_tok):
                correct[i] = False
        probs[i] /= max(1, cur_len)

    return [
        {
            "target_new": probs[i].item(),
            "target_true": probs[i + 1].item(),
            "target_new_correct": bool(correct[i]),
            "target_true_correct": bool(correct[i + 1]),
        }
        for i in range(0, len(probs), 2)
    ]


@torch.no_grad()
def official_compute_rewrite_quality_counterfact_dual(
    model: Any,
    tok: Any,
    record: Mapping[str, Any],
    device: torch.device,
    llama_like: bool = True,
) -> dict[str, Any]:
    """Return the legacy NLL fields plus ``*_prompts_correct`` arrays."""
    rr = record["requested_rewrite"]
    subject = rr["subject"]
    target_new = rr["target_new"]["str"]
    target_true = rr["target_true"]["str"]

    prompt_groups = [
        [rr["prompt"].format(subject)],
        list(record.get("paraphrase_prompts", [])),
        list(record.get("neighborhood_prompts", [])),
    ]
    flat_prompts = [prompt for group in prompt_groups for prompt in group]
    predictions = official_test_batch_prediction_dual(
        model=model,
        tok=tok,
        prefixes=flat_prompts,
        target_new=target_new,
        target_true=target_true,
        device=device,
        llama_like=llama_like,
    )

    cutoffs = [0] + np.cumsum([len(x) for x in prompt_groups]).tolist()
    grouped = [
        predictions[cutoffs[i - 1] : cutoffs[i]]
        for i in range(1, len(cutoffs))
    ]

    names = ("rewrite", "paraphrase", "neighborhood")
    out: dict[str, Any] = {}
    for name, rows in zip(names, grouped):
        out[f"{name}_prompts_probs"] = [
            {
                "target_new": float(row["target_new"]),
                "target_true": float(row["target_true"]),
            }
            for row in rows
        ]
        out[f"{name}_prompts_correct"] = [
            bool(row["target_true_correct"]) for row in rows
        ]
    return out


def table_style_accuracy_from_raw(
    metric_data: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Case-macro released-table accuracy percentages from evaluator raw rows."""
    key_map = {
        "Eff": "rewrite_prompts_correct",
        "Gen": "paraphrase_prompts_correct",
        "Spe": "neighborhood_prompts_correct",
    }
    result: dict[str, float | None] = {}
    for metric, key in key_map.items():
        case_values: list[float] = []
        for row in metric_data or []:
            post = row.get("post", {}) if isinstance(row, Mapping) else {}
            values = post.get(key, []) if isinstance(post, Mapping) else []
            if values:
                case_values.append(float(np.mean([bool(x) for x in values])))
        result[metric] = (
            100.0 * float(np.mean(case_values)) if case_values else None
        )
    return result


@contextmanager
def _patched_legacy_record_evaluator():
    """Temporarily route legacy evaluation through the dual-metric primitive."""
    original = legacy.official_compute_rewrite_quality_counterfact
    legacy.official_compute_rewrite_quality_counterfact = (
        official_compute_rewrite_quality_counterfact_dual
    )
    try:
        yield
    finally:
        legacy.official_compute_rewrite_quality_counterfact = original


def evaluate_loaded_model_official(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Drop-in replacement for the legacy loaded-model MCF evaluator."""
    with _patched_legacy_record_evaluator():
        result = legacy.evaluate_loaded_model_official(*args, **kwargs)

    for split in ("forget", "retain"):
        raw = result.get(f"{split}_raw")
        if isinstance(raw, Sequence):
            table = table_style_accuracy_from_raw(raw)
            result[f"{split}_released_table_style_accuracy"] = table
    result["dual_metric_evaluator"] = {
        "enabled": True,
        "legacy_nll_path_preserved": True,
        "teacher_forced_target_true_whole_sequence_top1_recorded": True,
        "model_or_algorithm_changed": False,
    }
    return result
