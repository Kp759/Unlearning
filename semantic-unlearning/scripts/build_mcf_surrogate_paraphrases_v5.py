#!/usr/bin/env python3
"""v5 MCF semantic-surrogate builder using boolean-consensus validation.

This reuses the held-out-safe v3 generation pipeline but swaps in the v3
relation-slot validator whose structured criterion booleans are authoritative
and whose free-form verdict strings are audit-only.  This fixes a calibration
failure observed with the local 3B judge: all required booleans were correct
while the verdict string still said NOT_EQUIVALENT.

The builder still fails closed unless every locked fact obtains exactly K
accepted semantic surrogates.  On failure, an answer-blind diagnostic is written
with every semantic judge call made before the failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import build_mcf_surrogate_paraphrases_v3 as v3
import mcf_surrogate_semantic_validator_v3 as semantic_v3


BUILDER_PROTOCOL = "mcf_locked_direct_only_boolean_consensus_surrogates_v5"


def _postprocess_artifact(out: Path) -> None:
    data = json.loads(out.read_text(encoding="utf-8"))
    data["builder_protocol"] = BUILDER_PROTOCOL
    sem = data.setdefault("semantic_validation", {})
    sem["protocol"] = semantic_v3.VALIDATOR_PROTOCOL
    sem["validator_kind"] = "relation_slot_boolean_consensus_plus_adversarial_boolean_consensus"
    sem["completion_fragments_explicitly_allowed"] = True
    sem["structured_booleans_authoritative"] = True
    sem["free_form_verdict_is_audit_only"] = True
    sem["criteria"] = [
        "same_slot_relation",
        "same_answer_type",
        "no_added_factual_constraints",
        "semantically_coherent",
        "adversarial_no_relation_shift",
        "adversarial_no_answer_type_shift",
        "adversarial_no_added_constraint_or_claim",
        "adversarial_no_semantic_malformation",
        "not_generic_wrapper",
    ]
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _diagnostic_path(out: Path) -> Path:
    return out.with_name(out.stem + "_v5_failure_diagnostic.json")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", required=True)
    known, _ = p.parse_known_args(argv)
    out = Path(known.output).resolve()

    v3.semantic = semantic_v3
    v3.BUILDER_PROTOCOL = BUILDER_PROTOCOL

    judge_calls: List[Dict[str, Any]] = []
    original_validate = semantic_v3.validate_candidates

    def validate_with_diagnostics(model, tok, **kwargs):
        results = original_validate(model, tok, **kwargs)
        judge_calls.append({
            "subject": kwargs.get("subject"),
            "direct_prompt": kwargs.get("direct_prompt"),
            "candidates": [str(x) for x in kwargs.get("candidates", [])],
            "results": results,
        })
        return results

    semantic_v3.validate_candidates = validate_with_diagnostics
    try:
        v3.main(argv)
    except Exception as exc:
        diagnostic: Dict[str, Any] = {
            "schema_version": 1,
            "builder_protocol": BUILDER_PROTOCOL,
            "validator_protocol": semantic_v3.VALIDATOR_PROTOCOL,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_requested": str(out),
            "semantic_judge_calls": judge_calls,
            "note": (
                "No training artifact should be used from this failed run. "
                "semantic_judge_calls contains answer-blind subject/direct/candidate "
                "inputs and raw/parsed judge outputs."
            ),
        }
        path = _diagnostic_path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"v5 failure diagnostic: {path}", flush=True)
        raise
    finally:
        semantic_v3.validate_candidates = original_validate

    _postprocess_artifact(out)
    print(f"v5 boolean-consensus semantic artifact finalized: {out}", flush=True)
    print(f"validator protocol: {semantic_v3.VALIDATOR_PROTOCOL}", flush=True)


if __name__ == "__main__":
    main()
