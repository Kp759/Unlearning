#!/usr/bin/env python3
"""Utilities for exact ZeroUnlearn MCF Eff/Gen parity.

ZeroUnlearn's MCF summarizer reports:

    Eff = post_rewrite_success
    Gen = post_paraphrase_success

where success is the fraction of prompts for which
NLL(target_true) > NLL(target_new), i.e. target_new is preferred.

Some later experiments in this repository temporarily relabeled the complement
(`post_*_sensitive_pref`) as Eff/Gen.  That is useful as an additional
sensitive-answer diagnostic but is not table-compatible with ZeroUnlearn.
This module restores the historical ZeroUnlearn labels while preserving the
complement under explicit SensitivePref_* names.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


ZERO_UNLEARN_EFF_DEFINITION = (
    "100 * mean[NLL(target_true) > NLL(target_new)] on rewrite prompts"
)
ZERO_UNLEARN_GEN_DEFINITION = (
    "100 * mean[NLL(target_true) > NLL(target_new)] on paraphrase prompts"
)


def _first(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (list, tuple)) and value:
        return None if value[0] is None else float(value[0])
    if value is None:
        return None
    return float(value)


def apply_zero_unlearn_eff_gen(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with Eff/Gen mapped exactly as ZeroUnlearn does.

    The function intentionally does not change Spe or any NLL/probability
    statistics.  `SensitivePref_Eff`/`SensitivePref_Gen` expose the complement
    diagnostic when `post_*_sensitive_pref` exists.
    """
    out = deepcopy(dict(summary))
    rewrite_success = _first(out, "post_rewrite_success")
    paraphrase_success = _first(out, "post_paraphrase_success")
    if rewrite_success is None or paraphrase_success is None:
        raise KeyError(
            "ZeroUnlearn parity requires post_rewrite_success and "
            "post_paraphrase_success in the summary"
        )

    sensitive_eff = _first(out, "post_rewrite_sensitive_pref")
    sensitive_gen = _first(out, "post_paraphrase_sensitive_pref")
    if sensitive_eff is not None:
        out["SensitivePref_Eff"] = sensitive_eff
    if sensitive_gen is not None:
        out["SensitivePref_Gen"] = sensitive_gen

    out["Eff"] = rewrite_success
    out["Gen"] = paraphrase_success
    out["Eff_definition"] = ZERO_UNLEARN_EFF_DEFINITION
    out["Gen_definition"] = ZERO_UNLEARN_GEN_DEFINITION
    out["Eff_Gen_source"] = "ZeroUnlearn post_*_success exact parity"
    return out


def patch_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Patch an already-computed MCF result without rerunning the model."""
    out = deepcopy(dict(payload))

    for key in ("forget", "retain"):
        if isinstance(out.get(key), Mapping):
            out[key] = apply_zero_unlearn_eff_gen(out[key])

    legacy = out.get("legacy_counterfact")
    if isinstance(legacy, Mapping):
        legacy = deepcopy(dict(legacy))
        for key in ("forget", "retain"):
            if isinstance(legacy.get(key), Mapping):
                legacy[key] = apply_zero_unlearn_eff_gen(legacy[key])
        legacy["Eff_Gen_definition"] = {
            "Eff": ZERO_UNLEARN_EFF_DEFINITION,
            "Gen": ZERO_UNLEARN_GEN_DEFINITION,
        }
        out["legacy_counterfact"] = legacy

    # Older one-model result shape.
    if isinstance(out.get("summary"), Mapping):
        out["summary"] = apply_zero_unlearn_eff_gen(out["summary"])
        out["Eff"] = out["summary"]["Eff"]
        out["Gen"] = out["summary"]["Gen"]
    elif isinstance(out.get("forget"), Mapping):
        # Keep convenient top-level aliases when they already exist or when the
        # result is a one-model evaluator output.
        if "Eff" in out:
            out["Eff"] = out["forget"]["Eff"]
        if "Gen" in out:
            out["Gen"] = out["forget"]["Gen"]

    out["zero_unlearn_eff_gen_parity"] = {
        "applied": True,
        "Eff": ZERO_UNLEARN_EFF_DEFINITION,
        "Gen": ZERO_UNLEARN_GEN_DEFINITION,
        "sensitive_preference_kept_separately": True,
    }
    return out
