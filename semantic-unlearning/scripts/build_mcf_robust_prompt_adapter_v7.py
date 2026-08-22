#!/usr/bin/env python3
"""High-precision v7 MCF robust-prompt dataset adapter.

v7 preserves the v6 pipeline and leak-free contract, but inserts an
answer-blind deterministic high-precision guard around the relation-slot LLM:

  direct -> generation -> high-precision family/type/slot guard
         -> LLM relation classifier -> semantic equivalence judge -> keep 3--8

The deterministic guard is authoritative only for rejection: an LLM cannot
rescue a family/type/slot mismatch.  If fewer than 3 candidates survive, the
record becomes direct_only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import build_mcf_robust_prompt_adapter_v6 as v6
import mcf_dataset_adapter_high_precision as hp


ARTIFACT_PROTOCOL = "mcf_direct_only_robust_prompt_adapter_v7"
ADAPTER_PROTOCOL = "mcf_relation_slot_answer_type_semantic_high_precision_adapter_v2"


def _patch_relation_adapter() -> tuple[Any, Any]:
    relation = v6.relation_adapter
    original_direct = relation.profile_direct
    original_candidates = relation.profile_candidates

    def profile_direct(model, tok, *, subject: str, direct_prompt: str, max_new_tokens: int = 128):
        llm = original_direct(
            model, tok,
            subject=subject,
            direct_prompt=direct_prompt,
            max_new_tokens=max_new_tokens,
        )
        guard = hp.direct_guard(subject, direct_prompt)
        llm["high_precision_guard"] = guard
        # High-precision direct family/types become the adapter contract.  The
        # LLM profile is retained in parsed/raw fields for auditing.
        llm["safe_to_augment"] = bool(llm.get("safe_to_augment", False) and guard["safe_to_augment"])
        llm["relation_label"] = str(guard["family"])
        llm["relation_description"] = (
            f"high-precision MCF slot family: {guard['family']}"
        )
        llm["answer_types"] = list(guard["answer_types"])
        if not guard["safe_to_augment"]:
            llm["ambiguity"] = "high"
        return llm

    def profile_candidates(
        model,
        tok,
        *,
        subject: str,
        direct_prompt: str,
        direct_profile,
        candidates,
        batch_size: int = 8,
        max_new_tokens: int = 128,
    ):
        results = original_candidates(
            model, tok,
            subject=subject,
            direct_prompt=direct_prompt,
            direct_profile=direct_profile,
            candidates=candidates,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )
        direct_guard = direct_profile.get("high_precision_guard") or hp.direct_guard(
            subject, direct_prompt
        )
        direct_family = str(direct_guard.get("family", "unknown"))
        for result in results:
            guard = hp.candidate_guard(
                subject=subject,
                direct_prompt=direct_prompt,
                direct_family=direct_family,
                candidate=str(result["candidate"]),
            )
            result["high_precision_guard"] = guard
            # LLM gates remain necessary, but are no longer sufficient.
            result["relation_pass"] = bool(result.get("relation_pass", False) and guard["accepted"])
            result["answer_type_pass"] = bool(result.get("answer_type_pass", False) and guard["accepted"])
            result["presemantic_pass"] = bool(
                result["relation_pass"] and result["answer_type_pass"]
            )
        return results

    relation.profile_direct = profile_direct
    relation.profile_candidates = profile_candidates
    return original_direct, original_candidates


def _restore_relation_adapter(originals: tuple[Any, Any]) -> None:
    v6.relation_adapter.profile_direct = originals[0]
    v6.relation_adapter.profile_candidates = originals[1]


def _postprocess(out: Path) -> None:
    data = json.loads(out.read_text(encoding="utf-8"))
    data["protocol"] = ARTIFACT_PROTOCOL
    data["adapter_protocol"] = ADAPTER_PROTOCOL
    data["high_precision_guard"] = {
        "enabled": True,
        "protocol": hp.PROTOCOL,
        "authoritative_for_rejection": True,
        "uses_target_true": False,
        "uses_target_new": False,
        "uses_official_paraphrases": False,
        "requirements": [
            "same_coarse_relation_slot_family",
            "same_deterministic_answer_type_set",
            "no_new_named_content",
            "no_new_numeric_content",
            "no_new_answer_head_modifier",
            "clear_open_slot_or_factual_question",
        ],
        "fallback": "direct_only_when_fewer_than_min_surrogates_survive",
    }
    for row in data.get("records", []):
        rel = row.setdefault("relation_slot", {})
        rel["high_precision_family"] = rel.get("relation_label")
        rel["high_precision_guard_protocol"] = hp.PROTOCOL
    out.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", required=True)
    known, _ = p.parse_known_args(argv)
    out = Path(known.output).resolve()

    old_protocol = v6.ARTIFACT_PROTOCOL
    old_adapter_protocol = v6.ADAPTER_PROTOCOL
    originals = _patch_relation_adapter()
    try:
        # Make the artifact self-identify as v7 during construction too.
        v6.ARTIFACT_PROTOCOL = ARTIFACT_PROTOCOL
        v6.ADAPTER_PROTOCOL = ADAPTER_PROTOCOL
        v6.main(argv)
    finally:
        v6.ARTIFACT_PROTOCOL = old_protocol
        v6.ADAPTER_PROTOCOL = old_adapter_protocol
        _restore_relation_adapter(originals)

    _postprocess(out)
    print(f"v7 high-precision MCF adapter finalized: {out}", flush=True)
    print(f"high-precision guard: {hp.PROTOCOL}", flush=True)


if __name__ == "__main__":
    main()
