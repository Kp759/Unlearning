#!/usr/bin/env python3
"""Core primitives for V1.4 relation-scoped private-null routing.

V1.4 keeps the V1.1 position-preserving private subject rows, but it does not
route every occurrence of a registered subject.  A small per-subject relation
gate is trained on frozen-Base prompt features.  Only prompts classified as the
registered forgotten relation are rewritten to private subject ids; other
relations stay on the exact Base token path.

The private path is trained toward a natural abstention string and away from the
sensitive true object.  The internal reserved ids are never exposed as output
labels.  This is behavioral unlearning/routing, not a latent-erasure claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "mcf_private_vocab_rewiring_v1_4_relation_null_routing"
ABSTENTION_TEXT = "I don't know"


@dataclass(frozen=True)
class GateExample:
    case_id: int
    subject: str
    relation_id: str
    text: str
    label: int
    source: str


def relation_template_bank(
    protection_fit: Sequence[Mapping[str, Any]],
) -> list[tuple[str, str]]:
    """Return unique training-visible (relation_id, prompt-template) pairs."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for record in protection_fit:
        rr = record["requested_rewrite"]
        relation_id = str(rr["relation_id"])
        prompt = str(rr["prompt"])
        if prompt.count("{}") != 1:
            continue
        key = (relation_id, prompt)
        if key not in seen:
            seen.add(key)
            out.append(key)
    out.sort()
    return out


def build_gate_examples(
    forget_records: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    protection_fit: Sequence[Mapping[str, Any]],
    *,
    negatives_per_case: int = 16,
    seed: int = 14141,
) -> list[GateExample]:
    """Build leakage-safe positive/negative examples for the relation gate.

    Positives are the already-locked V1.3 five-view forget prompts.  Negatives
    use only sanitized protection-fit prompt templates whose relation_id differs
    from that case's forgotten relation.  Targets/held-out prompt groups are not
    used.
    """
    bank = relation_template_bank(protection_fit)
    examples: list[GateExample] = []
    for record in forget_records:
        cid = int(record["case_id"])
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        forgotten_relation = str(rr["relation_id"])
        views = list(view_map.get(cid, []))
        if len(views) != 5:
            raise RuntimeError(f"V1.4 requires exactly five locked views for case {cid}")
        for template in views:
            if str(template).count("{}") != 1:
                raise RuntimeError(f"case {cid} has invalid positive view template")
            examples.append(
                GateExample(
                    case_id=cid,
                    subject=subject,
                    relation_id=forgotten_relation,
                    text=str(template).format(subject),
                    label=1,
                    source="locked_v1_3_positive_view",
                )
            )

        candidates = [item for item in bank if item[0] != forgotten_relation]
        rng = random.Random(int(seed) + cid)
        take = min(int(negatives_per_case), len(candidates))
        chosen = rng.sample(candidates, take) if take else []
        if not chosen:
            raise RuntimeError(f"case {cid} has no different-relation gate negatives")
        for relation_id, template in chosen:
            examples.append(
                GateExample(
                    case_id=cid,
                    subject=subject,
                    relation_id=str(relation_id),
                    text=str(template).format(subject),
                    label=0,
                    source="training_visible_different_relation_negative",
                )
            )
    return examples


@torch.no_grad()
def extract_last_token_features(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract normalized frozen-Base final-token hidden states."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(texts), int(batch_size)):
        batch = list(texts[start : start + int(batch_size)])
        encoded = tokenizer(batch, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1].float()
        attention = encoded["attention_mask"]
        last = attention.sum(dim=1).long() - 1
        index = torch.arange(hidden.shape[0], device=device)
        feature = hidden[index, last]
        rows.append(F.normalize(feature, dim=-1).cpu())
    return torch.cat(rows, dim=0)


class PerSubjectRelationGate(nn.Module):
    """One small linear relation head per registered forget subject/case."""

    def __init__(self, case_ids: Sequence[int], hidden_size: int):
        super().__init__()
        ordered = [int(value) for value in case_ids]
        if len(set(ordered)) != len(ordered):
            raise ValueError("gate case ids must be unique")
        self.case_ids = ordered
        self.case_to_index = {case_id: i for i, case_id in enumerate(ordered)}
        self.weight = nn.Parameter(torch.zeros(len(ordered), int(hidden_size)))
        self.bias = nn.Parameter(torch.zeros(len(ordered)))

    def case_indices(self, case_ids: Sequence[int], *, device: torch.device) -> torch.Tensor:
        try:
            values = [self.case_to_index[int(case_id)] for case_id in case_ids]
        except KeyError as exc:
            raise RuntimeError(f"unknown relation-gate case id {exc.args[0]}") from exc
        return torch.tensor(values, dtype=torch.long, device=device)

    def forward(self, features: torch.Tensor, case_indices: torch.Tensor) -> torch.Tensor:
        rows = self.weight.index_select(0, case_indices)
        bias = self.bias.index_select(0, case_indices)
        return (features * rows).sum(dim=-1) + bias


def calibrate_case_thresholds(
    logits: torch.Tensor,
    labels: torch.Tensor,
    case_ids: Sequence[int],
) -> tuple[Dict[int, float], Dict[str, Any]]:
    """Calibrate one hard threshold per case and require train-set separation."""
    if logits.ndim != 1 or labels.ndim != 1 or logits.shape != labels.shape:
        raise ValueError("gate logits/labels must be aligned rank-1 tensors")
    if len(case_ids) != int(logits.numel()):
        raise ValueError("gate case ids do not align with logits")

    thresholds: Dict[int, float] = {}
    per_case: Dict[str, Any] = {}
    all_correct = 0
    for cid in sorted(set(int(value) for value in case_ids)):
        idx = [i for i, value in enumerate(case_ids) if int(value) == cid]
        local_logits = logits[idx]
        local_labels = labels[idx]
        pos = local_logits[local_labels > 0.5]
        neg = local_logits[local_labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            raise RuntimeError(f"case {cid} lacks positive or negative gate examples")
        min_pos = float(pos.min().item())
        max_neg = float(neg.max().item())
        separable = min_pos > max_neg
        threshold = 0.5 * (min_pos + max_neg)
        thresholds[cid] = threshold
        predictions = (local_logits >= threshold).to(local_labels.dtype)
        correct = int(predictions.eq(local_labels).sum().item())
        all_correct += correct
        per_case[str(cid)] = {
            "positive_count": int(pos.numel()),
            "negative_count": int(neg.numel()),
            "minimum_positive_logit": min_pos,
            "maximum_negative_logit": max_neg,
            "separation_gap": min_pos - max_neg,
            "threshold": threshold,
            "perfect_training_separation": bool(separable and correct == len(idx)),
        }
    metrics = {
        "examples": int(logits.numel()),
        "correct": int(all_correct),
        "accuracy": float(all_correct / max(1, int(logits.numel()))),
        "all_cases_perfectly_separable": all(
            bool(value["perfect_training_separation"]) for value in per_case.values()
        ),
        "per_case": per_case,
    }
    return thresholds, metrics


def gate_predictions(
    logits: torch.Tensor,
    case_ids: Sequence[int],
    thresholds: Mapping[int, float],
) -> torch.Tensor:
    values = torch.tensor(
        [float(thresholds[int(case_id)]) for case_id in case_ids],
        dtype=logits.dtype,
        device=logits.device,
    )
    return logits >= values


def serialize_gate_state(
    gate: PerSubjectRelationGate,
    thresholds: Mapping[int, float],
) -> Dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "case_ids": list(gate.case_ids),
        "hidden_size": int(gate.weight.shape[1]),
        "thresholds": {str(int(k)): float(v) for k, v in thresholds.items()},
        "feature": "frozen_base_final_prompt_token_hidden_state_l2_normalized",
        "gate": "per_subject_linear_binary_relation_classifier",
    }
