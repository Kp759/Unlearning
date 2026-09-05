#!/usr/bin/env python3
"""ZeroUnlearn paper-facing MCF Eff/Gen metrics.

IMPORTANT: ZeroUnlearn's paper (Eq. 16) defines Efficacy as the average
likelihood assigned to the ORIGINAL forget target y_f, with lower being better:

    Eff = 100 * mean P_theta'(target_true | rewrite_prompt)

Generalization uses the same residual-sensitive likelihood on paraphrased
queries.  This is distinct from the CounterFact/ROME editing statistic
``post_*_success`` that asks whether target_new is preferred to target_true.

The repository historically conflated those two notions.  This module keeps the
CounterFact pairwise statistic under explicit ``CF_EditSuccess_*`` names and
uses only raw per-prompt target_true NLLs to compute paper-facing Eff/Gen.

Our evaluator stores an answer's token-AVERAGED negative log likelihood.  Thus
``exp(-target_true_nll)`` is the answer likelihood proxy used here (geometric
mean token probability), expressed as a percentage to match ZeroUnlearn tables.
"""
from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping, Sequence


ZERO_UNLEARN_EFF_DEFINITION = (
    "100 * mean_case(mean_prompt(exp(-NLL(target_true)))) on rewrite prompts; "
    "ZeroUnlearn paper Eq. (16), lower is better"
)
ZERO_UNLEARN_GEN_DEFINITION = (
    "100 * mean_case(mean_prompt(exp(-NLL(target_true)))) on paraphrase prompts; "
    "lower is better"
)
COUNTERFACT_PAIRWISE_DEFINITION = (
    "100 * mean[NLL(target_true) > NLL(target_new)]; CounterFact edit-success "
    "diagnostic, NOT ZeroUnlearn paper Eff/Gen"
)


def _first(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (list, tuple)) and value:
        return None if value[0] is None else float(value[0])
    if value is None:
        return None
    return float(value)


def _paper_residual_probability(
    metric_data: Sequence[Mapping[str, Any]],
    prompt_key: str,
) -> float | None:
    """Return case-macro residual target_true likelihood in percent.

    ZeroUnlearn summarizes records case-wise.  We therefore average all prompt
    instances belonging to one record first and then average across records.
    For MCF rewrites there is one prompt per case; paraphrases normally have the
    same count per case, so this is also equal to a prompt-micro mean.
    """
    case_values: list[float] = []
    for row in metric_data or []:
        post = row.get("post", {}) if isinstance(row, Mapping) else {}
        xs = post.get(prompt_key, []) if isinstance(post, Mapping) else []
        probabilities: list[float] = []
        for item in xs or []:
            if not isinstance(item, Mapping) or item.get("target_true") is None:
                continue
            nll = float(item["target_true"])
            if not math.isfinite(nll):
                continue
            probabilities.append(math.exp(-nll))
        if probabilities:
            case_values.append(sum(probabilities) / len(probabilities))
    if not case_values:
        return None
    return 100.0 * sum(case_values) / len(case_values)


def compute_zero_unlearn_paper_eff_gen(
    metric_data: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    """Compute paper-style residual-sensitive Eff/Gen from raw evaluator rows."""
    return {
        "Eff": _paper_residual_probability(metric_data, "rewrite_prompts_probs"),
        "Gen": _paper_residual_probability(metric_data, "paraphrase_prompts_probs"),
    }


def apply_zero_unlearn_eff_gen(
    summary: Mapping[str, Any],
    metric_data: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return summary with ZeroUnlearn PAPER Eff/Gen semantics.

    Raw per-prompt rows are required.  Computing ``exp(-mean NLL)`` from an
    already-aggregated NLL would not equal ``mean(exp(-NLL))`` and is therefore
    deliberately rejected rather than silently approximated.
    """
    if metric_data is None:
        raise ValueError(
            "ZeroUnlearn paper Eff/Gen require raw metric_data with per-prompt "
            "target_true NLLs; aggregated summaries are insufficient"
        )

    out = deepcopy(dict(summary))
    paper = compute_zero_unlearn_paper_eff_gen(metric_data)
    if paper["Eff"] is None or paper["Gen"] is None:
        raise KeyError(
            "ZeroUnlearn paper Eff/Gen require rewrite/paraphrase target_true NLLs"
        )

    # Preserve CounterFact pairwise quantities under names that cannot be
    # confused with the unlearning paper metric.
    cf_eff = _first(out, "post_rewrite_success")
    cf_gen = _first(out, "post_paraphrase_success")
    if cf_eff is not None:
        out["CF_EditSuccess_Eff"] = cf_eff
    if cf_gen is not None:
        out["CF_EditSuccess_Gen"] = cf_gen

    sensitive_eff = _first(out, "post_rewrite_sensitive_pref")
    sensitive_gen = _first(out, "post_paraphrase_sensitive_pref")
    if sensitive_eff is not None:
        out["SensitivePref_Eff"] = sensitive_eff
    if sensitive_gen is not None:
        out["SensitivePref_Gen"] = sensitive_gen

    # The existing ``Spe`` in our legacy evaluator is a probability-difference
    # statistic.  ZeroUnlearn's paper defines Spe as neighborhood accuracy, so
    # do not claim exact Spe parity here.  Keep the old value explicitly named.
    if out.get("Spe") is not None:
        out["Legacy_Spe_ProbabilityDiff"] = out["Spe"]
    out["Spe_paper_parity_available"] = False
    out["Spe_paper_note"] = (
        "ZeroUnlearn paper Spe is neighborhood accuracy; rerun the paper-parity "
        "evaluator with argmax correctness to obtain it exactly"
    )

    out["Eff"] = round(float(paper["Eff"]), 6)
    out["Gen"] = round(float(paper["Gen"]), 6)
    out["Eff_definition"] = ZERO_UNLEARN_EFF_DEFINITION
    out["Gen_definition"] = ZERO_UNLEARN_GEN_DEFINITION
    out["Eff_Gen_source"] = "ZeroUnlearn paper Eq. (16) residual target_true likelihood"
    out["CF_EditSuccess_definition"] = COUNTERFACT_PAIRWISE_DEFINITION
    return out


def patch_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Patch a saved evaluator payload using its stored raw rows; no GPU needed."""
    out = deepcopy(dict(payload))

    for split in ("forget", "retain"):
        summary = out.get(split)
        raw = out.get(f"{split}_raw")
        if isinstance(summary, Mapping) and isinstance(raw, Sequence):
            out[split] = apply_zero_unlearn_eff_gen(summary, raw)

    legacy = out.get("legacy_counterfact")
    if isinstance(legacy, Mapping):
        legacy = deepcopy(dict(legacy))
        for split in ("forget", "retain"):
            summary = legacy.get(split)
            raw = out.get(f"{split}_raw")
            if isinstance(summary, Mapping) and isinstance(raw, Sequence):
                legacy[split] = apply_zero_unlearn_eff_gen(summary, raw)
        legacy["Eff_Gen_definition"] = {
            "Eff": ZERO_UNLEARN_EFF_DEFINITION,
            "Gen": ZERO_UNLEARN_GEN_DEFINITION,
        }
        legacy["pairwise_counterfact_is_not_eff_gen"] = True
        out["legacy_counterfact"] = legacy

    if isinstance(out.get("summary"), Mapping):
        raw = out.get("forget_raw")
        if not isinstance(raw, Sequence):
            raise ValueError("saved one-model result lacks forget_raw required for paper Eff/Gen")
        out["summary"] = apply_zero_unlearn_eff_gen(out["summary"], raw)
        out["Eff"] = out["summary"]["Eff"]
        out["Gen"] = out["summary"]["Gen"]
    elif isinstance(out.get("forget"), Mapping):
        if "Eff" in out:
            out["Eff"] = out["forget"]["Eff"]
        if "Gen" in out:
            out["Gen"] = out["forget"]["Gen"]

    out["zero_unlearn_eff_gen_parity"] = {
        "applied": True,
        "source": "ZeroUnlearn paper Eq. (16), not CounterFact post_*_success",
        "Eff": ZERO_UNLEARN_EFF_DEFINITION,
        "Gen": ZERO_UNLEARN_GEN_DEFINITION,
        "counterfact_pairwise_kept_separately": True,
        "paper_spe_requires_argmax_rerun": True,
    }
    return out
