#!/usr/bin/env python3
"""Abstention-anchored erasure directions for RSNR-V2 Stage 0.

RSNR-V1A's frozen spec sets ``target_new_used: False`` -- never training or
steering toward the CounterFact counterfactual is what separates the method
from the ROME/MEMIT/AlphaEdit editing lineage.  The existing directional
Emb+LM machinery (``mcf_sure_directional_emb_lm_stage1.py``) violates that: it
builds ``d = h_true - h_new`` with a ``w_true - w_new`` fallback, both of which
require ``target_new``.

This module re-anchors the same construction on RSNR's own abstention string,
so the erasure direction becomes

    d = h_true - h_IDK          (teacher-forced hidden contrast)
    d = w_true - w_IDK          (decoder discriminant, first answer token)

and ``target_new`` is never read.

Note on which branch actually fires.  ``expand_answer_field_cases`` builds the
teacher-forced prompt for answer token ``j`` as ``prompt + decode(answer[:j])``.
At ``j == 0`` that prefix is empty, so the sensitive and abstention prompts are
character-identical and the hidden contrast is *exactly* zero by construction --
not a numerical accident.  Single-token MCF answers therefore always resolve
through the decoder discriminant, which is the hidden-space gradient of the
``logit(true) - logit(IDK)`` gap and is the right object to erase anyway.
Multi-token answers additionally contribute genuine hidden contrasts at
``j > 0``, where the two prefixes really do differ.

``summarize_direction_sources`` reports that split so a run whose directions
silently collapsed onto the degenerate third fallback is visible in provenance
rather than buried.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Sequence

# RSNR-V1A trains this exact abstention string; Stage 0 must anchor on the same
# behaviour it will later route to, or the two stages pull in different
# directions.
ABSTENTION_TEXT = "I don't know."

# Synthetic requested_rewrite field carrying the abstention answer. Named so it
# cannot collide with a benchmark field, and asserted absent before it is set.
ABSTENTION_FIELD = "rsnr_abstention_target"

# contrast_direction()'s third fallback is not a contrast at all -- it returns
# the raw sensitive hidden state when both the hidden contrast and the decoder
# discriminant vanish. Erasing along it is not the intended geometry.
DEGENERATE_SOURCE = "sensitive_hidden_fallback"

HIDDEN_SOURCE = "hidden_sensitive_minus_reference"
DISCRIMINANT_SOURCE = "decoder_row_sensitive_minus_reference_fallback"


def attach_abstention_reference(
    records: Sequence[Mapping[str, Any]],
    *,
    abstention_text: str = ABSTENTION_TEXT,
    field: str = ABSTENTION_FIELD,
) -> List[Dict[str, Any]]:
    """Return copies of ``records`` carrying a constant abstention answer field.

    The input records are never mutated: Stage 0 shares its record list with
    the locked-split validation and with RSNR's own membership rows, and an
    in-place edit there would silently change what those see.
    """
    if not isinstance(abstention_text, str) or not abstention_text.strip():
        raise ValueError("abstention_text must be a non-empty string")

    out: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"Record {position} lacks requested_rewrite")
        if field in rr:
            raise ValueError(
                f"Record {position} already carries {field!r}; refusing to "
                "overwrite a field that may not be the abstention anchor"
            )
        target_true = rr.get("target_true")
        if not isinstance(target_true, Mapping) or not target_true.get("str"):
            raise ValueError(f"Record {position} lacks target_true.str")

        copied = deepcopy(dict(record))
        copied["requested_rewrite"] = dict(copied["requested_rewrite"])
        copied["requested_rewrite"][field] = {"str": abstention_text}
        out.append(copied)
    return out


def abstention_margin_view(
    records: Sequence[Mapping[str, Any]],
    *,
    abstention_text: str = ABSTENTION_TEXT,
) -> List[Dict[str, Any]]:
    """Throwaway records whose *reference answer slot* holds the abstention text.

    The margin machinery (``sure_stage2_sparse_repair.mcf_instances``) builds an
    ``MCFPromptInstance`` with exactly two answer strings and reads the
    reference one from ``requested_rewrite.target_new``; its ``reference_field``
    argument only chooses between those two slots, so it cannot be pointed at a
    third field.  To keep the RSNR contract the reference slot is therefore
    *filled with the abstention string*, making the direct margin

        NLL(target_true) - NLL(IDK)

    instead of the CounterFact ``NLL(target_true) - NLL(target_new)``.

    The result is a computation-only view.  It must never be persisted or fed
    to anything that reports ``target_new``, because in these records that field
    no longer holds the benchmark counterfactual.
    """
    if not isinstance(abstention_text, str) or not abstention_text.strip():
        raise ValueError("abstention_text must be a non-empty string")

    out: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"Record {position} lacks requested_rewrite")
        copied = deepcopy(dict(record))
        copied["requested_rewrite"] = dict(copied["requested_rewrite"])
        copied["requested_rewrite"]["target_new"] = {"str": abstention_text}
        copied["requested_rewrite"]["_reference_slot_holds"] = "abstention"
        out.append(copied)
    return out


def assert_target_new_unused(direction_reports: Sequence[Mapping[str, Any]]) -> None:
    """Fail if any direction was built from a target_new-anchored source.

    ``build_row_specific_contrast_bases`` labels every direction with the
    branch that produced it.  Under abstention anchoring the reference cases
    are the abstention answer, so no report may claim otherwise; this is the
    machine-checkable form of ``target_new_used: False``.
    """
    for report in direction_reports:
        sources = report.get("direction_sources", {})
        if not isinstance(sources, Mapping):
            raise ValueError("direction report lacks a direction_sources mapping")
        unknown = set(sources) - {HIDDEN_SOURCE, DISCRIMINANT_SOURCE, DEGENERATE_SOURCE}
        if unknown:
            raise RuntimeError(
                f"token {report.get('token_id')} has unrecognized direction "
                f"sources {sorted(unknown)}; the abstention anchor cannot be "
                "verified"
            )


def summarize_direction_sources(
    direction_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate which contrast branch produced each direction."""
    totals: Dict[str, int] = {
        HIDDEN_SOURCE: 0,
        DISCRIMINANT_SOURCE: 0,
        DEGENERATE_SOURCE: 0,
    }
    degenerate_tokens: List[int] = []
    for report in direction_reports:
        sources = report.get("direction_sources", {})
        for name, count in sources.items():
            totals[name] = int(totals.get(name, 0)) + int(count)
        if int(sources.get(DEGENERATE_SOURCE, 0)) > 0:
            degenerate_tokens.append(int(report.get("token_id", -1)))

    total = sum(totals.values())
    return {
        "anchor": "abstention",
        "abstention_text": ABSTENTION_TEXT,
        "target_new_used": False,
        "counts": totals,
        "total_directions": total,
        "hidden_contrast_fraction": (totals[HIDDEN_SOURCE] / total) if total else 0.0,
        "decoder_discriminant_fraction": (
            (totals[DISCRIMINANT_SOURCE] / total) if total else 0.0
        ),
        # Expected to be zero. Non-zero means both the hidden contrast and the
        # decoder discriminant vanished, i.e. target_true and the abstention
        # string share that answer token, and the "direction" is just the raw
        # sensitive hidden state.
        "degenerate_fallback_count": totals[DEGENERATE_SOURCE],
        "degenerate_fallback_token_ids": sorted(degenerate_tokens),
    }


def check_abstention_separability(
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    llama_like: bool,
    abstention_text: str = ABSTENTION_TEXT,
) -> Dict[str, Any]:
    """Report target_true answers whose first token equals the abstention's.

    When they coincide the decoder discriminant ``w_true - w_IDK`` is exactly
    zero at the first answer token, so that row falls through to the degenerate
    branch.  Callers surface this before spending GPU time.
    """
    import sure_canonical_core as core

    abstention_ids = core.answer_token_ids(tok, abstention_text, llama_like=llama_like)
    if not abstention_ids:
        raise RuntimeError(f"abstention text {abstention_text!r} tokenized to nothing")
    abstention_first = int(abstention_ids[0])

    collisions: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        answer = str(rr["target_true"]["str"])
        ids = core.answer_token_ids(tok, answer, llama_like=llama_like)
        if ids and int(ids[0]) == abstention_first:
            collisions.append(
                {
                    "record_position": position,
                    "case_id": int(record.get("case_id", position)),
                    "target_true": answer,
                    "shared_first_token_id": abstention_first,
                }
            )
    return {
        "abstention_first_token_id": abstention_first,
        "abstention_token_count": len(abstention_ids),
        "first_token_collisions": collisions,
        "separable": not collisions,
    }
