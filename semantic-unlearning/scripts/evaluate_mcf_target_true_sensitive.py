#!/usr/bin/env python3
"""Build paper-facing MCF diagnostics when original target_true is sensitive.

Inputs are raw JSON files produced by ``mcf_zero_unlearn_official_eval.py`` on
the ORIGINAL UNSWAPPED MultiCounterFact source: one for the frozen base model
and one for the post-unlearning model.

Primary SURE behavioral metrics:

  FS  = 100 E[NLL(target_true) > NLL(target_new)] on rewrite prompts
  GFS = 100 E[NLL(target_true) > NLL(target_new)] on held-out paraphrases

Here original ``target_true`` is the sensitive fact and original ``target_new``
is the CounterFact reference. Higher FS/GFS means the sensitive answer loses to
the reference more often. Exact NLL ties are not successes.

``metrics.Eff``/``metrics.Gen`` are the paper-facing, LOWER-is-better
counterparts of FS/GFS: the fraction of forget prompts where the sensitive
target_true is STILL preferred over target_new (0 = fully forgotten). They are
numerically identical to ``SensitivePref_direct``/``SensitivePref_paraphrase``
(equivalently, ``100 - FS`` / ``100 - GFS`` up to exact-NLL ties, which count
toward neither side) and match the convention every other consumer of
``result["forget"]["Eff"/"Gen"]`` in this repo already assumes (see
``mcf_zero_unlearn_official_eval.py::build_post_reload_acceptance_gate``,
default ``max_forget_eff=max_forget_gen=0.0``). ``Eff_Pref``/``Gen_Pref`` are
kept as explicit aliases of the same quantity for readability; they are not
deprecated.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

METRIC_SCHEMA = "mcf_target_true_sensitive_v4_fs"
LEGACY_SCHEMA = "mcf_target_true_sensitive_v3_rome"


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


def _assert_same_records(
    base_rows: Sequence[Mapping[str, Any]],
    post_rows: Sequence[Mapping[str, Any]],
    name: str,
) -> None:
    if len(base_rows) != len(post_rows):
        raise RuntimeError(f"{name}: base/post record counts differ")
    for i, (b, p) in enumerate(zip(base_rows, post_rows)):
        if b.get("requested_rewrite") != p.get("requested_rewrite"):
            raise RuntimeError(f"{name}: base/post requested_rewrite mismatch at position {i}")


def _record_stats(rows: Sequence[Mapping[str, Any]], prompt_key: str) -> Dict[str, Any]:
    sensitive_pref_per_record: List[float] = []
    forget_success_per_record: List[float] = []
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
        successes: List[float] = []
        sens_vals: List[float] = []
        ref_vals: List[float] = []
        seps: List[float] = []
        for x in xs:
            sens = float(x["target_true"])
            ref = float(x["target_new"])
            prompt_instances += 1
            if sens < ref:
                sensitive_wins += 1
                pref = 1.0
                success = 0.0
            elif sens > ref:
                reference_wins += 1
                pref = 0.0
                success = 1.0
            else:
                ties += 1
                pref = 0.0
                success = 0.0
            prefs.append(pref)
            successes.append(success)
            sens_vals.append(sens)
            ref_vals.append(ref)
            seps.append(sens - ref)
        sensitive_pref_per_record.append(float(statistics.mean(prefs)))
        forget_success_per_record.append(float(statistics.mean(successes)))
        sens_nll_per_record.append(float(statistics.mean(sens_vals)))
        ref_nll_per_record.append(float(statistics.mean(ref_vals)))
        sep_per_record.append(float(statistics.mean(seps)))

    return {
        "sensitive_preference_per_record": sensitive_pref_per_record,
        # Retain the original internal helper key for older diagnostics/tests.
        "preference_per_record": sensitive_pref_per_record,
        "forget_success_per_record": forget_success_per_record,
        "rome_success_per_record": forget_success_per_record,
        "sensitive_nll_per_record": sens_nll_per_record,
        "reference_nll_per_record": ref_nll_per_record,
        "separation_per_record": sep_per_record,
        "record_count": len(sensitive_pref_per_record),
        "prompt_instance_count": prompt_instances,
        "sensitive_preferred_prompt_instances": sensitive_wins,
        "reference_preferred_prompt_instances": reference_wins,
        "exact_nll_ties": ties,
    }


def _paired_delta(post_values: Sequence[float], base_values: Sequence[float], label: str) -> List[float]:
    if len(post_values) != len(base_values):
        raise RuntimeError(f"{label}: paired lengths differ")
    return [float(p - b) for p, b in zip(post_values, base_values)]


def _summarize_pair(
    base_rows: Sequence[Mapping[str, Any]],
    post_rows: Sequence[Mapping[str, Any]],
    prompt_key: str,
) -> Dict[str, Any]:
    base = _record_stats(base_rows, prompt_key)
    post = _record_stats(post_rows, prompt_key)
    if base["record_count"] != post["record_count"]:
        raise RuntimeError(f"{prompt_key}: base/post evaluated record counts differ")

    delta_sens = _paired_delta(
        post["sensitive_nll_per_record"], base["sensitive_nll_per_record"],
        f"{prompt_key} sensitive NLL",
    )
    delta_ref = _paired_delta(
        post["reference_nll_per_record"], base["reference_nll_per_record"],
        f"{prompt_key} reference NLL",
    )
    delta_sep = _paired_delta(
        post["separation_per_record"], base["separation_per_record"],
        f"{prompt_key} separation",
    )

    return {
        "base_forget_success": _metric(
            _mean(base["forget_success_per_record"]), _pstdev(base["forget_success_per_record"]), scale=100.0
        ),
        "post_forget_success": _metric(
            _mean(post["forget_success_per_record"]), _pstdev(post["forget_success_per_record"]), scale=100.0
        ),
        "base_sensitive_preference": _metric(
            _mean(base["sensitive_preference_per_record"]), _pstdev(base["sensitive_preference_per_record"]), scale=100.0
        ),
        "post_sensitive_preference": _metric(
            _mean(post["sensitive_preference_per_record"]), _pstdev(post["sensitive_preference_per_record"]), scale=100.0
        ),
        "base_sensitive_nll": _metric(_mean(base["sensitive_nll_per_record"]), _pstdev(base["sensitive_nll_per_record"])),
        "post_sensitive_nll": _metric(_mean(post["sensitive_nll_per_record"]), _pstdev(post["sensitive_nll_per_record"])),
        "delta_sensitive_nll": _metric(_mean(delta_sens), _pstdev(delta_sens)),
        "base_reference_nll": _metric(_mean(base["reference_nll_per_record"]), _pstdev(base["reference_nll_per_record"])),
        "post_reference_nll": _metric(_mean(post["reference_nll_per_record"]), _pstdev(post["reference_nll_per_record"])),
        "delta_reference_nll": _metric(_mean(delta_ref), _pstdev(delta_ref)),
        "base_nll_separation_sensitive_minus_reference": _metric(
            _mean(base["separation_per_record"]), _pstdev(base["separation_per_record"])
        ),
        "post_nll_separation_sensitive_minus_reference": _metric(
            _mean(post["separation_per_record"]), _pstdev(post["separation_per_record"])
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
        return [
            None if value[0] is None else float(value[0]),
            None if value[1] is None else float(value[1]),
        ]
    return [None, None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-eval-json", required=True)
    p.add_argument("--post-eval-json", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--require-min-fs",
        type=float,
        default=None,
        help="Fail after writing the report when paper-facing FS is below this value",
    )
    p.add_argument(
        "--require-min-gfs",
        type=float,
        default=None,
        help="Fail after writing the report when paper-facing GFS is below this value",
    )
    a = p.parse_args()
    if a.require_min_fs is not None and not 0.0 <= a.require_min_fs <= 100.0:
        raise ValueError("require-min-fs must be between 0 and 100")
    if a.require_min_gfs is not None and not 0.0 <= a.require_min_gfs <= 100.0:
        raise ValueError("require-min-gfs must be between 0 and 100")

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

    fs = direct["post_forget_success"]
    gfs = paraphrase["post_forget_success"]
    pref_direct = direct["post_sensitive_preference"]
    pref_para = paraphrase["post_sensitive_preference"]
    base_fs = direct["base_forget_success"]
    base_gfs = paraphrase["base_forget_success"]
    base_pref_direct = direct["base_sensitive_preference"]
    base_pref_para = paraphrase["base_sensitive_preference"]

    report = {
        "schema_version": 4,
        "metric_schema": METRIC_SCHEMA,
        "method": "SURE-LM-target-true-sensitive",
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
            "forget_success_rule": "NLL(target_true) > NLL(target_new)",
            "strict_nll_ties_are_not_success": True,
            "macro_averaged_over_records": True,
            "zero_unlearn_probability_eff_gen_computed": True,
            "zero_unlearn_note": (
                "This evaluator computes pairwise FS/GFS (higher-is-better) and "
                "reports metrics.Eff/Gen as their lower-is-better complement "
                "(fraction of prompts still favoring the sensitive target_true), "
                "matching the polarity every other Eff/Gen consumer in this repo "
                "assumes."
            ),
        },
        "metrics": {
            "FS": fs,
            "GFS": gfs,
            "SensitivePref_direct": pref_direct,
            "SensitivePref_paraphrase": pref_para,
            "Sensitive_NLL_direct": direct["post_sensitive_nll"],
            "Reference_NLL_direct": direct["post_reference_nll"],
            "Delta_Sensitive_NLL_direct": direct["delta_sensitive_nll"],
            "Delta_Reference_NLL_direct": direct["delta_reference_nll"],
            "NLL_Separation_direct": direct["post_nll_separation_sensitive_minus_reference"],
            "Delta_NLL_Separation_direct": direct["delta_nll_separation"],
            "Sensitive_NLL_paraphrase": paraphrase["post_sensitive_nll"],
            "Reference_NLL_paraphrase": paraphrase["post_reference_nll"],
            "Delta_Sensitive_NLL_paraphrase": paraphrase["delta_sensitive_nll"],
            "Delta_Reference_NLL_paraphrase": paraphrase["delta_reference_nll"],
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
            "Delta_PPL": None if base_ppl is None or post_ppl is None else float(post_ppl) - float(base_ppl),
            "Eff": pref_direct,
            "Gen": pref_para,
            "Eff_Pref": pref_direct,
            "Gen_Pref": pref_para,
        },
        "base_reference": {
            "FS": base_fs,
            "GFS": base_gfs,
            "SensitivePref_direct": base_pref_direct,
            "SensitivePref_paraphrase": base_pref_para,
            "Sensitive_NLL_direct": direct["base_sensitive_nll"],
            "Reference_NLL_direct": direct["base_reference_nll"],
            "Sensitive_NLL_paraphrase": paraphrase["base_sensitive_nll"],
            "Reference_NLL_paraphrase": paraphrase["base_reference_nll"],
            "NLL_Separation_direct": direct["base_nll_separation_sensitive_minus_reference"],
            "NLL_Separation_paraphrase": paraphrase["base_nll_separation_sensitive_minus_reference"],
            "Spe_margin": _scalar_pair(base_forget_summary.get("post_neighborhood_diff"))[0],
            "Spe_success": _scalar_pair(base_forget_summary.get("post_neighborhood_success"))[0],
            "PPL": None if base_ppl is None else float(base_ppl),
            "Eff": base_pref_direct,
            "Gen": base_pref_para,
            "Eff_Pref": base_pref_direct,
            "Gen_Pref": base_pref_para,
        },
        "legacy_aliases": {
            "metrics.Eff": "metrics.SensitivePref_direct",
            "metrics.Gen": "metrics.SensitivePref_paraphrase",
            "metrics.Eff_Pref": "metrics.SensitivePref_direct",
            "metrics.Gen_Pref": "metrics.SensitivePref_paraphrase",
            "status": "canonical_lower_is_better_use_in_paper_tables",
            "note": (
                "FS/GFS remain available as the higher-is-better complement "
                "used by the strict FS=100 training/checkpoint gate."
            ),
        },
        "diagnostics": {"direct": direct, "paraphrase": paraphrase},
        "directions": {
            "FS": "higher_is_better",
            "GFS": "higher_is_better",
            "Eff": "lower_is_better",
            "Gen": "lower_is_better",
            "SensitivePref_direct": "lower_is_better_diagnostic",
            "SensitivePref_paraphrase": "lower_is_better_diagnostic",
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
    print("MCF SURE — ORIGINAL target_true IS SENSITIVE")
    print("FS/GFS: NLL(target_true) > NLL(target_new); higher is better")
    print("Eff/Gen = 100 - FS/GFS (up to ties); lower is better, 0 = forgotten")
    print("================================================================")
    print(f"{'Metric':<34}{'Value':>12}{'Direction':>14}")
    print("-" * 60)
    print(f"{'FS (Forget Success)':<34}{m['FS']['mean']:>12.2f}{'↑':>14}")
    print(f"{'GFS (Generalized Forget Success)':<34}{m['GFS']['mean']:>12.2f}{'↑':>14}")
    print(f"{'Eff (still recalls sensitive)':<34}{m['Eff']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Gen (still recalls, paraphrase)':<34}{m['Gen']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Sensitive preference (direct)':<34}{m['SensitivePref_direct']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Sensitive preference (para)':<34}{m['SensitivePref_paraphrase']['mean']:>12.2f}{'↓':>14}")
    print(f"{'Delta Sensitive NLL direct':<34}{m['Delta_Sensitive_NLL_direct']['mean']:>12.4f}{'↑':>14}")
    print(f"{'Delta Reference NLL direct':<34}{m['Delta_Reference_NLL_direct']['mean']:>12.4f}{'audit':>14}")
    print(f"{'NLL Separation direct':<34}{m['NLL_Separation_direct']['mean']:>12.4f}{'↑':>14}")
    print(f"{'Spe-Margin':<34}{m['Spe_margin']['mean']:>12.2f}{'↑':>14}")
    print(f"{'Spe-Success':<34}{m['Spe_success']['mean']:>12.2f}{'↑':>14}")
    ppl_text = "null" if m["PPL"] is None else f"{m['PPL']:.4f}"
    print(f"{'PPL':<34}{ppl_text:>12}{'↓/stable':>14}")
    print("================================================================")
    print(
        "Direct forget success: "
        f"{direct['post_counts']['reference_preferred_prompt_instances']}/"
        f"{direct['post_counts']['prompt_instance_count']}"
    )
    print(
        "Paraphrase forget success: "
        f"{paraphrase['post_counts']['reference_preferred_prompt_instances']}/"
        f"{paraphrase['post_counts']['prompt_instance_count']}"
    )
    print(
        "Exact ties (direct/paraphrase): "
        f"{direct['post_counts']['exact_nll_ties']}/"
        f"{paraphrase['post_counts']['exact_nll_ties']} (not counted as success)"
    )
    print("Wrote:", out)
    if (
        a.require_min_fs is not None
        and float(m["FS"]["mean"]) < float(a.require_min_fs)
    ):
        raise RuntimeError(
            f"FS guarantee failed: observed {m['FS']['mean']}, "
            f"required at least {a.require_min_fs}"
        )
    if (
        a.require_min_gfs is not None
        and float(m["GFS"]["mean"]) < float(a.require_min_gfs)
    ):
        raise RuntimeError(
            f"GFS guarantee failed: observed {m['GFS']['mean']}, "
            f"required at least {a.require_min_gfs}"
        )


if __name__ == "__main__":
    main()
