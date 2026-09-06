#!/usr/bin/env python3
"""Compatibility-fixed entrypoint for build_mcf_relation_views_v2.

Normalizes tokenizer/apply_chat_template outputs to a real input_ids tensor
before generation. This fixes Transformers versions where apply_chat_template
returns a BatchEncoding-like object even when return_tensors='pt'.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

import build_mcf_relation_views_v2 as builder


def _fixed_encode_generation_prompt(
    tokenizer: Any,
    text: str,
    device: torch.device,
) -> torch.Tensor:
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

    # Transformers versions differ here: value may be a Tensor,
    # BatchEncoding/Mapping, or occasionally an object exposing input_ids.
    if torch.is_tensor(value):
        input_ids = value
    elif isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RuntimeError("generation encoding mapping has no input_ids")
        input_ids = value["input_ids"]
    elif hasattr(value, "input_ids"):
        input_ids = value.input_ids
    else:
        raise RuntimeError(
            f"unsupported generation encoding type: {type(value).__name__}"
        )

    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise RuntimeError(
            f"generation input_ids must be rank-2 tensor, got {type(input_ids).__name__} "
            f"shape={getattr(input_ids, 'shape', None)}"
        )
    return input_ids.to(device)


builder.encode_generation_prompt = _fixed_encode_generation_prompt


if __name__ == "__main__":
    builder.main()
