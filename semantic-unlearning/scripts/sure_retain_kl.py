#!/usr/bin/env python3
"""Shared retain-preservation utilities for SURE-LM.

The retain objective is benchmark independent.  Each retain record contributes
one direct prompt context.  The frozen Base model supplies the full next-token
teacher distribution and SURE minimizes KL(Base || current) on those contexts.
No retain answer label, paraphrase/locality prompt, or benchmark-specific target
is required.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import sure_canonical_core as core


def retain_prompt_cases(records: Sequence[Mapping[str, Any]]) -> List[core.SensitivePredictionCase]:
    cases: List[core.SensitivePredictionCase] = []
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"Retain record {position} lacks requested_rewrite")
        prompt_template = str(rr.get("prompt", ""))
        subject = str(rr.get("subject", ""))
        if not prompt_template:
            raise ValueError(f"Retain record {position} lacks prompt")
        try:
            prompt = prompt_template.format(subject)
        except Exception as exc:
            raise ValueError(f"Could not format retain prompt at {position}") from exc
        cases.append(
            core.SensitivePredictionCase(
                case_id=int(record.get("case_id", position)),
                record_position=position,
                token_index=0,
                prompt=prompt,
                target_text="",
            )
        )
    return cases


def full_distribution_kl(current_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    """KL(Base || current) over the complete vocabulary."""
    cur = current_logits.float()
    ref = base_logits.to(device=cur.device, dtype=torch.float32)
    if cur.ndim != 2 or cur.shape != ref.shape:
        raise ValueError("current/base retain logits must have equal [batch,vocab] shape")
    cur_logp = F.log_softmax(cur, dim=-1)
    ref_logp = F.log_softmax(ref, dim=-1)
    ref_p = ref_logp.exp()
    return (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()
