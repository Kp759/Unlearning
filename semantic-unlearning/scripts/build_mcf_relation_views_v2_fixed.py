#!/usr/bin/env python3
"""Compatibility-fixed entrypoint for build_mcf_relation_views_v2.

Fixes two runtime/calibration issues without changing the base builder's data
contract:
1) normalize tokenizer/apply_chat_template outputs to a real rank-2 input_ids
   tensor before generation;
2) score the SAME-vs-DIFFERENT semantic-equivalence verifier in the model's
   instruction/chat format instead of as a raw plain-text prompt.

Official paraphrases/neighborhoods remain unread by the builder. Target answers
are not supplied to generation or semantic verification.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import os

import torch
import torch.nn.functional as F

import build_mcf_relation_views_v2 as builder


def _extract_input_ids(value: Any, *, label: str) -> torch.Tensor:
    if torch.is_tensor(value):
        input_ids = value
    elif isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RuntimeError(f"{label} encoding mapping has no input_ids")
        input_ids = value["input_ids"]
    elif hasattr(value, "input_ids"):
        input_ids = value.input_ids
    else:
        raise RuntimeError(f"unsupported {label} encoding type: {type(value).__name__}")

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError(
            f"{label} input_ids must be rank-2 tensor, got "
            f"{type(input_ids).__name__} shape={getattr(input_ids, 'shape', None)}"
        )
    return input_ids


def _chat_prefix_ids(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    value: Any = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            value = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            value = None
    if value is None:
        value = tokenizer(text, return_tensors="pt")
    return _extract_input_ids(value, label="chat/generation").to(device)


def _fixed_encode_generation_prompt(
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> torch.Tensor:
    return _chat_prefix_ids(tokenizer, text, device)


def _continuation_ids(tokenizer: Any, text: str) -> list[int]:
    value = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    ids = [int(x) for x in value]
    if not ids:
        raise RuntimeError(f"empty continuation tokenization for {text!r}")
    return ids


@torch.no_grad()
def _chat_continuation_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> float:
    prefix = _chat_prefix_ids(tokenizer, prompt, device)
    if prefix.shape[0] != 1:
        raise RuntimeError("semantic verifier expects a single prompt")
    cids = _continuation_ids(tokenizer, continuation)
    cont = torch.tensor([cids], dtype=prefix.dtype, device=device)
    ids = torch.cat([prefix, cont], dim=1)
    logits = model(input_ids=ids, use_cache=False).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    start = int(prefix.shape[1])
    positions = torch.arange(start - 1, start - 1 + len(cids), device=device)
    tokens = torch.tensor(cids, dtype=torch.long, device=device)
    return float(logp[0, positions, tokens].mean().item())


_LOW_MARGIN_PRINTS = 0
_LOW_MARGIN_LIMIT = 40


@torch.no_grad()
def _fixed_equivalence_margin(
    model: Any,
    tokenizer: Any,
    canonical: str,
    candidate: str,
    device: torch.device,
) -> float:
    global _LOW_MARGIN_PRINTS
    prompt = builder.verifier_prompt(canonical, candidate)
    same = _chat_continuation_logprob(model, tokenizer, prompt, "SAME", device)
    different = _chat_continuation_logprob(model, tokenizer, prompt, "DIFFERENT", device)
    margin = float(same - different)

    # Diagnostic only. If another family fails, surface the candidate/margin so
    # we do not have to infer whether failure is generation or verification.
    threshold = float(os.environ.get("RELATION_EQ_MARGIN", "0.5"))
    if margin < threshold and _LOW_MARGIN_PRINTS < _LOW_MARGIN_LIMIT:
        print(
            "[relation-v2 verifier reject] "
            f"margin={margin:.4f} threshold={threshold:.4f} "
            f"canonical={canonical!r} candidate={candidate!r}",
            flush=True,
        )
        _LOW_MARGIN_PRINTS += 1
    return margin


builder.encode_generation_prompt = _fixed_encode_generation_prompt
builder.equivalence_margin = _fixed_equivalence_margin


if __name__ == "__main__":
    builder.main()
