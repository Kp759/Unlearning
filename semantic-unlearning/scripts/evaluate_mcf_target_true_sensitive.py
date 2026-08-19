#!/usr/bin/env python3
"""Build paper-facing MCF metrics when original target_true is sensitive.

Inputs are two raw JSON files produced by ``mcf_zero_unlearn_official_eval.py``
on the ORIGINAL UNSWAPPED MCF source: one for the frozen base model and one for
the post-unlearning model.  This script never reinterprets the raw evaluator's
``Eff``/``Gen`` aliases.  Instead it derives an explicit metric schema:

  Eff-Pref       = 100 E[ NLL(sensitive target_true) < NLL(reference target_new) ]
  Gen-Pref       = same strict preference test on paraphrases
  Sensitive NLL  = E[NLL(target_true)]
  Delta Sens NLL = E[NLL_post(target_true) - NLL_base(target_true)]
  Separation     = E[NLL(target_true) - NLL(target_new)]

Exact NLL ties are False, matching the strict-comparison convention used by the
ROME/MEMIT/ZeroUnlearn CounterFact summarizers.  Lower preference rates and
higher sensitive NLL / separation indicate stronger suppression.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

METRIC_SCHEMA = "mcf_target_true_sensitive_v2"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mean(xs: Sequence[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def _pstdev(xs: Sequence[float]) -> float | None:
    if not xs:
        return None
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _metric(mean: float | None, sd: float | None, *, scale: float = 1.0) -> Dict[str, float | None]:
    return {
        "mean": None if mean is None else float(mean * scale),
        "population_sd": None if sd is None else float(sd * scale),
    }


def _assert_same_records(base_rows: Sequence[Mapping[str, Any]], post_rows: Sequence[Mapping[str, Any]], name: str) -> None:
    if len(base_rows) != len(post_rows):
        raise RuntimeError(f"{name}: base/post record counts differ")
    for i, (b, p) in enumerate(zip(base_rows, post_rows)):
        if b.get("requested_rewrite") != p.get("requested_rewrite"):
            raise RuntimeError(f"{name}: base/post requested_rewrite mismatch at position {i}")


def _record_stats(rows: Sequence[Mapping[str, Any]], prompt_key: str) -> Dict[str, Any]:
    pref_per_record: List[float] = []
    sens_nll_per_record: List[float] = []
    ref_nll_per_record: List[float] = []
    sep_per_record: List[float] = []
    prompt_instances = 0
    sensitive_wins = 0
    reference_wins = 0
    ties = 0

    for row in rows:
        xs = row.get("post", {}).get(prompt_key, [])
        if not xs:
            continue
        prefs: List[float] = []
        sens_vals: List[float] = []
        ref_vals: List[float] = []
        seps: List[float] = []
        for x in xs:
            sens = float(x["target_true"])
            ref = float(x["target_new"])
            prompt_instances += 1
            if sens < ref:
                pref = 1.0
                sensitive_wins += 1
            else:
                # Strict comparison: exact ties are failures, not half credit.
                pref = 0.0
                if sens > ref:
                    reference_wins += 1
                else:
                    ties += 1
            prefs.append(pref)
            sens_vals.append(sens)
            ref_vals.append(ref)
            seps.append(sens - ref)
        pref_per_record.append(float(statistics.mean(prefs)))
        sens_nll_per_record.append(float(statistics.mean(sens_vals)))
        ref_nll_per_record.append(float(statistics.mean(ref_vals)))
        sep_per_record.append(float(statistics.mean(seps)))

    return {
        "preference_per_record": pref_per_record,
        "sensitive_nll_per_record": sens_nll_per_record,
        "reference_nll_per_record": ref_nll_per_record,
        "separation_per_record": sep_per_record,
        "record_count": len(pref_per_record),
        "prompt_instance_count": prompt_instances,
        "sensitive_preferred_prompt_instances": sensitive_wins,
        "reference_preferred_prompt_instances": reference_wins,
        "exact_nll_ties": ties,
    }


def _paired_delta(post_values: Sequence[float], base_values: Sequence[float], label: str) -> List[float]:
    if len(post_values) != len(base_values):
        raise RuntimeError(f"{label}: paired lengths differ")
    return [float(p - b) for p, b in zip(post_values, base_values)]


def _summarize_pair(base_rows: Sequence[Mapping[str, Any]], post_rows: Sequence[Mapping[str, Any]], prompt_key: str) -> Dict[str, Any]:
    base = _record_stats(base_rows, prompt_key)
    post = _record_stats(post_rows, prompt_key)
    if base["record_count"] != post["record_count"]:
        raise RuntimeError(f"{prompt_key}: base/post evaluated record counts differ")

    delta_sens = _paired_delta(
        post["sensitive_nll_per_record"],
        base["sensitive_nll_per_record"],
        f"{prompt_key} sensitive NLL",
    )
    delta_ref = _paired_delta(
        post["reference_nll_per_record"],
        base["reference_nll_per_record"],
        f"{prompt_key} reference NLL",
    )
    delta_sep = _paired_delta(
        post["separation_per_record"],
        base["separation_per_record"],
        f"{prompt_key} separation",
    )

    return {
        "base_sensitive_preference": _metric(
            _mean(base["preference_per_record"]),
            _pstdev(base["preference_per_record"]),
            scale=100.0,
        ),
        "post_sensitive_preference": _metric(
            _mean(post["preference_per_record"]),
            _pstdev(post["preference_per_record"]),
            scale=100.0,
        ),
        "base_sensitive_nll": _metric(
            _mean(base["sensitive_nll_per_record"]),
            _pstdev(base["sensitive_nll_per_record"]),
        ),
        "post_sensitive_nll": _metric(
            _mean(post["sensitive_nll_per_record"]),
            _pstdev(post["sensitive_nll_per_record"]),
        ),
        "delta_sensitive_nll": _metric(_mean(delta_sens), _pstdev(delta_sens)),
        "base_reference_nll": _metric(
            _mean(base["reference_nll_per_record"]),
            _pstdev(base["reference_nll_per_record"]),
        ),
        "post_reference_nll": _metric(
            _mean(post["reference_nll_per_record"]),
            _pstdev(post["reference_nll_per_record"]),
        ),
        "delta_reference_nll": _metric(_mean(delta_ref), _pstdev(delta_ref)),
        "base_nll_separation_sensitive_minus_reference": _metric(
            _mean(base["separation_per_record"]),
            _pstdev(base["separation_per_record"]),
        ),
        "post_nll_separation_sensitive_minus_reference": _metric(
            _mean(post["separation_per_record"]),
            _pstdev(post["separation_per_record"]),
        ),
        "delta_nll_separation": _metric(_mean(delta_sep), _pstdev(delta_sep)),
        "post_counts": {
            "record_count": post["record_count"],
            "prompt_instance_count": post["prompt_instance_count"],
            "sensitive_preferred_prompt_instances": post["sensitive_preferred_prompt_instances"],
            "reference_preferred_prompt_instances": post["reference_preferred_prompt_instances"],
            "exact_nll_ties": post["exact_nll_ties"],
        },
    }


def _scalar_pair(value: Any) -> List[float | None]:
    if isinstance(value, list) and len(value) >= 2:
        return [None if value[0] is None else float(value[0]), None if value[1] is None else float(value[1])]
    return [None, None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-eval-json", required=True)
    p.add_argument("--post-eval-json", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    base_path = Path(a.base_eval_json).resolve()
    post_path = Path(a.post_eval_json).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    post = json.loads(post_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in ("seed", "unlearn_num", "retain_num", "sample_mode"):
        if base.get(key) != post.get(key):
            raise RuntimeError(f"base/post evaluation mismatch for {key}")
    if int(manifest.get("seed", -1)) != int(post["seed"]):
        raise RuntimeError("split manifest seed does not match evaluations")
    semantics = manifest.get("target_semantics", {})
    if semantics.get("original_sensitive_field") != "target_true":
        raise RuntimeError("split manifest does not declare target_true sensitive")

    base_forget = base.get("forget_raw", [])
    post_forget = post.get("forget_raw", [])
    _assert_same_records(base_forget, post_forget, "forget")

    direct = _summarize_pair(base_forget, post_forget, "rewrite_prompts_probs")
    paraphrase = _summarize_pair(base_forget, post_forget, "paraphrase_prompts_probs")

    post_forget_summary = post["forget"]
    base_forget_summary = base["forget"]
    base_ppl = base.get("forget_PPL")
    post_ppl = post.get("forget_PPL")

    report = {
        "schema_version": 2,
        "metric_schema": METRIC_SCHEMA,
        "method": "SURE-LM-canonical-target-true-sensitive",
        "dataset": "MCF",
        "seed": int(post["seed"]),
        "target_semantics": {
            "sensitive": "original requested_rewrite.target_true",
            "counterfactual_reference": "original requested_rewrite.target_new",
            "final_evaluation": "original unswapped MCF",
        },
        "comparison_contract": {
            "same_forget_split": True,
            "same_record_order": True,
            "strict_nll_ties_count_as_sensitive_preference_false": True,
            "preference_macro_averaged_over_records": True,
        },
        "metrics": {
            "Eff_Pref": direct["post_sensitive_preference"],
            "Gen_Pref": paraphrase["post_sensitive_preference"],
            "Sensitive_NLL_direct": direct["post_sensitive_nll"],
            "Delta_Sensitive_NLL_direct": direct["delta_sensitive_nll"],
            "NLL_Separation_direct": direct["post_nll_separation_sensitive_minus_reference"],
            "Delta_NLL_Separation_direct": direct["delta_nll_separation"],
            "Sensitive_NLL_paraphrase": paraphrase["post_sensitive_nll"],
            "Delta_Sensitive_NLL_paraphrase": paraphrase["delta_sensitive_nll"],
            "NLL_Separation_paraphrase": paraphrase["post_nll_separation_sensitive_minus_reference"],
            "Delta_NLL_Separation_paraphrase": paraphrase["delta_nll_separation"],
            "Spe_margin": {
                "mean": _scalar_pair(post_forget_summary.get("post_neighborhood_diff"))[0],
                "population_sd": _scalar_pair(post_forget_summary.get("post_neighborhood_diff"))[1],
            },
            "Spe_success": {
                "mean": _scalar_pair(post_forget_summary.get("post_neighborhood_success"))[0],
                "population_sd": _scalar_pair(post_forget_summary.get("post_neighborhood_success"))[1],
            },
            "PPL": None if post_ppl is None else float(post_ppl),
            "Delta_PPL": (
                None
                if base_ppl is None or post_ppl is None
                else float(post_ppl) - float(base_ppl)
            ),
        },
        "base_reference": {
            "Eff_Pref": direct["base_sensitive_preference"],
            "Gen_Pref": paraphrase["base_sensitive_preference"],
            "Sensitive_NLL_direct": direct["base_sensitive_nll"],
            "Sensitive_NLL_paraphrase": paraphrase["base_sensitive_nll"],
            "NLL_Separation_direct": direct["base_nll_separation_sensitive_minus_reference"],
            "NLL_Separation_paraphrase": paraphrase["base_nll_separation_sensitive_minus_reference"],
            "Spe_margin": _scalar_pair(base_forget_summary.get("post_neighborhood_diff"))[0],
            "Spe_success": _scalar_pair(base_forget_summary.get("post_neighborhood_success"))[0],
            "PPL": None if base_ppl is None else float(base_ppl),
        },
        "diagnostics": {
            "direct": direct,
            "paraphrase": paraphrase,
        },
        "directions": {
            "Eff_Pref": "lower_is_better",
            "Gen_Pref": "lower_is_better",
            "Sensitive_NLL_direct": "higher_is_stronger_suppression",
            "Delta_Sensitive_NLL_direct": "positive_is_stronger_suppression",
            "NLL_Separation_direct": "higher_is_stronger_suppression",
            "Delta_NLL_Separation_direct": "positive_is_stronger_suppression",
            "Sensitive_NLL_paraphrase": "higher_is_stronger_suppression",
            "Delta_Sensitive_NLL_paraphrase": "positive_is_stronger_suppression",
            "NLL_Separation_paraphrase": "higher_is_stronger_suppression",
            "Delta_NLL_Separation_paraphrase": "positive_is_stronger_suppression",
            "Spe_margin": "higher_is_better",
            "Spe_success": "higher_is_better",
            "PPL": "lower_or_stable_is_better",
        },
        "provenance": {
            "base_eval_json": str(base_path),
            "post_eval_json": str(post_path),
            "split_manifest": str(manifest_path),
            "source_mcf_sha256": manifest.get("source_sha256"),
            "training_visible_sha256": manifest.get("training_visible_sha256"),
            "forget_case_ids": manifest.get("sampling", {}).get("forget_case_ids", []),
            "retain_case_ids": manifest.get("sampling", {}).get("retain_case_ids", []),
        },
    }

    out = Path(a.out).resolve()
    write_json(out, report)

    m = report["metrics"]
    print("\n================================================================")
    print("MCF CANONICAL SURE — ORIGINAL target_true IS SENSITIVE")
    print("================================================================")
    print(f"{'Metric':<32}{'Value':>12}{'Direction':>14}")
    print("-" * 58)
    print(f"{'Eff-Pref':<32}{m['Eff_Pref']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Gen-Pref':<32}{m['Gen_Pref']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Delta Sensitive NLL direct':<32}{m['Delta_Sensitive_NLL_direct']['mean']:>12.4f}{'↑':>14}")
    print(f"{'NLL Separation direct':<32}{m['NLL_Separation_direct']['mean']:>12.4f}{'↑':>14}")
    print(f"{'Spe-Margin':<32}{m['Spe_margin']['mean']:>12.2f}{'↑':>14}")
    print(f"{'Spe-Success':<32}{m['Spe_success']['mean']:>12.2f}{'↑':>14}")
    ppl_text = "null" if m["PPL"] is None else f"{m['PPL']:.4f}"
    print(f"{'PPL':<32}{ppl_text:>12}{'↓':>14}")
    print("================================================================")
    print(
        "Direct sensitive preference: "
        f"{direct['post_counts']['sensitive_preferred_prompt_instances']}/"
        f"{direct['post_counts']['prompt_instance_count']}"
    )
    print(
        "Paraphrase sensitive preference: "
        f"{paraphrase['post_counts']['sensitive_preferred_prompt_instances']}/"
        f"{paraphrase['post_counts']['prompt_instance_count']}"
    )
    print("Exact ties are strict failures, not 0.5 credit.")
    print("Wrote:", out)


if __name__ == "__main__":
    main()
