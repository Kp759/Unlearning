#!/usr/bin/env python3
"""Calibrated v4 semantic-surrogate builder for locked MCF forget facts.

v4 reuses the held-out-safe v3 generation pipeline but replaces the overly
sentence-oriented semantic rubric with relation-slot equivalence validation.
MCF completion fragments are explicitly valid.  On success the emitted artifact
is marked v4; on failure a diagnostic JSON is still written so generated
candidates and judge decisions are inspectable.

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


def _patch_v3() -> None:
    # v3 resolves semantic helpers from its module global at runtime.
    v3.semantic = semantic_v2
    v3.BUILDER_PROTOCOL = BUILDER_PROTOCOL


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
    # Parse only enough to know the requested output path. v3 performs the full
    # authoritative argument and split validation.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--output", required=True)
    known, _ = p.parse_known_args(argv)
    out = Path(known.output).resolve()

    _patch_v3()
    try:
        v3.main(argv)
    except Exception as exc:
        # v3 may fail closed before writing its normal receipt. Record the exact
        # exception and invocation metadata so this failure remains auditable.
        diagnostic: Dict[str, Any] = {
            "schema_version": 1,
            "builder_protocol": BUILDER_PROTOCOL,
            "validator_protocol": semantic_v2.VALIDATOR_PROTOCOL,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_requested": str(out),
            "note": (
                "No training artifact should be used from this failed run. "
                "If a semantic receipt exists beside the output, inspect it as well."
            ),
        }
        path = _diagnostic_path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
        print(f"v4 failure diagnostic: {path}", flush=True)
        raise

    _postprocess_artifact(out)
    print(f"v4 calibrated semantic artifact finalized: {out}", flush=True)
    print(f"validator protocol: {semantic_v2.VALIDATOR_PROTOCOL}", flush=True)


if __name__ == "__main__":
    main()
