#!/usr/bin/env python3
"""Calibrated v4 semantic-surrogate builder for locked MCF forget facts.

v4 reuses the held-out-safe v3 generation pipeline but replaces the overly
sentence-oriented semantic rubric with relation-slot equivalence validation.
MCF completion fragments are explicitly valid. On success the emitted artifact
is marked v4; on failure a diagnostic JSON is still written with the candidates
that reached semantic judging and their raw/parsed judge outputs.

No target_true/target_new is supplied to the generator or semantic judge. Known
answer strings remain post-generation rejection guards only. Official MCF
paraphrase/neighborhood/retain/PPL data are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import build_mcf_surrogate_paraphrases_v3 as v3
import mcf_surrogate_semantic_validator_v2 as semantic_v2


BUILDER_PROTOCOL = "mcf_locked_direct_only_relation_slot_surrogates_v4"


def _postprocess_artifact(out: Path) -> None:
    data = json.loads(out.read_text(encoding="utf-8"))
    data["builder_protocol"] = BUILDER_PROTOCOL
    sem = data.setdefault("semantic_validation", {})
    sem["protocol"] = semantic_v2.VALIDATOR_PROTOCOL
    sem["validator_kind"] = "relation_slot_equivalence_plus_adversarial_critic"
    sem["completion_fragments_explicitly_allowed"] = True
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
    return out.with_name(out.stem + "_v4_failure_diagnostic.json")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", required=True)
    known, _ = p.parse_known_args(argv)
    out = Path(known.output).resolve()

    # v3 resolves semantic helpers from its module global at runtime.
    v3.semantic = semantic_v2
    v3.BUILDER_PROTOCOL = BUILDER_PROTOCOL

    judge_calls: List[Dict[str, Any]] = []
    original_validate = semantic_v2.validate_candidates

    def validate_with_diagnostics(model, tok, **kwargs):
        results = original_validate(model, tok, **kwargs)
        judge_calls.append({
            "subject": kwargs.get("subject"),
            "direct_prompt": kwargs.get("direct_prompt"),
            "candidates": [str(x) for x in kwargs.get("candidates", [])],
            "results": results,
        })
        return results

    semantic_v2.validate_candidates = validate_with_diagnostics
    try:
        v3.main(argv)
    except Exception as exc:
        diagnostic: Dict[str, Any] = {
            "schema_version": 1,
            "builder_protocol": BUILDER_PROTOCOL,
            "validator_protocol": semantic_v2.VALIDATOR_PROTOCOL,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_requested": str(out),
            "semantic_judge_calls": judge_calls,
            "note": (
                "No training artifact should be used from this failed run. "
                "semantic_judge_calls contains only answer-blind subject/direct/candidate "
                "inputs and judge outputs."
            ),
        }
        path = _diagnostic_path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"v4 failure diagnostic: {path}", flush=True)
        raise
    finally:
        semantic_v2.validate_candidates = original_validate

    _postprocess_artifact(out)
    print(f"v4 calibrated semantic artifact finalized: {out}", flush=True)
    print(f"validator protocol: {semantic_v2.VALIDATOR_PROTOCOL}", flush=True)


if __name__ == "__main__":
    main()
