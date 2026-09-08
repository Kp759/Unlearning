#!/usr/bin/env python3
"""Fix5p: matched Seed-1 end-to-end old-router vs Fix5o-router atomic evaluation.

Purpose
-------
Measure whether the Fix5o router's increased coverage of previously Base-disclosing
paraphrases produces an actual end-to-end improvement when the downstream suppression
mechanism is frozen.

This experiment changes ONLY the router checkpoint/eta between the two integrated
arms. It freezes:
  * Llama model/tokenizer;
  * official Seed-1 forget50 query bank;
  * exact-name target-local selector;
  * frozen Fix5l per-binding sensitive-answer token supports;
  * fixed penalty magnitude (expected 12.0);
  * quotient disabled;
  * deterministic generation settings;
  * canonical disclosure metric.

Conditions for each atomic query:
  1. base: no router, no correction;
  2. old_integrated: Fix5k exact-name router + frozen fixed token penalty;
  3. fix5o_integrated: Fix5o augmented exact-name router + same frozen penalty.

Both teacher-forced preference/NLL metrics and deterministic generated canonical
answer disclosure are reported. No mixed-query controller claim is made here; Fix5n-v3
remains a separately frozen output-position result.

Optional --fix5m-records provides a strict saved-output identity audit for the Base
and old-integrated generated outputs on the same direct/paraphrase query bank. No
training, threshold tuning, quotient, new support construction, or model editing.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_fixed_penalty_integration_fix5l_seed1 as fix5l
import mcf_target_local_generation_mixed_eval_fix5m_seed1 as fix5m

base = fix5l.base
SEED = 1


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            out.append(value)
    return out


def load_router_specs(
    fix5l_dir: Path,
    fix5k_dir: Path,
    fix5o_dir: Path,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    fix5l_report = load_json(fix5l_dir / "mcf_target_local_fixed_penalty_integration_fix5l.json")
    old_eta = float(fix5l_report["frozen_router"]["eta"])
    if fix5l_report["frozen_router"]["recognition_snapshot"].get("pilot_pass") is not True:
        raise RuntimeError("Fix5p requires the historical Fix5k/Fix5l router pilot_pass=true")

    fix5o_report = load_json(fix5o_dir / "mcf_target_local_augmented_relation_router_fix5o.json")
    new_result = fix5o_report["results"]["augmented_exact_name"]
    if new_result.get("pilot_pass_preservation_and_original_validation") is not True:
        raise RuntimeError("Fix5p requires Fix5o augmented router preservation/original-validation pilot_pass=true")
    new_eta = float(new_result["eta"])

    old_head_path = fix5k_dir / "exact_name_target_local_linear_head.pt"
    new_head_path = fix5o_dir / "augmented_exact_name_linear_head.pt"
    old_head, old_classes = fix5l.load_head(old_head_path, device)
    new_head, new_classes = fix5l.load_head(new_head_path, device)
    if old_classes != new_classes:
        raise RuntimeError("old and Fix5o class inventories differ; refusing unmatched comparison")

    return {
        "old": {
            "head": old_head,
            "classes": old_classes,
            "eta": old_eta,
            "head_path": str(old_head_path),
        },
        "fix5o": {
            "head": new_head,
            "classes": new_classes,
            "eta": new_eta,
            "head_path": str(new_head_path),
        },
        "reports": {
            "fix5l": fix5l_report,
            "fix5o": fix5o_report,
        },
    }


def route_one(
    query: str,
    spec: Mapping[str, Any],
    model: Any,
    tok: Any,
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    device: torch.device,
    encode_batch_size: int,
) -> fix5l.RouteDecision:
    return fix5l.route_query(
        query,
        model,
        tok,
        spec["head"],
        spec["classes"],
        float(spec["eta"]),
        support_map,
        device,
        encode_batch_size,
    )


def generate_with_ids(
    model: Any,
    tok: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    token_ids: Sequence[int],
    penalty: float,
) -> dict[str, Any]:
    processor = None
    if token_ids:
        processor = fix5l.FixedTokenPenaltyLogitsProcessor(token_ids, penalty=penalty)
    return fix5m.generate_from_inputs(
        model,
        tok,
        input_ids,
        attention_mask,
        max_new_tokens,
        processor=processor,
    )


def score_two_routes(
    model: Any,
    tok: Any,
    query: str,
    target_new: str,
    target_true: str,
    old_ids: Sequence[int],
    new_ids: Sequence[int],
    penalty: float,
    device: torch.device,
    llama_like: bool,
) -> dict[str, Any]:
    old = fix5l.score_choice_pair(
        model, tok, query, target_new, target_true,
        old_ids, penalty, device, llama_like,
    )
    new = fix5l.score_choice_pair(
        model, tok, query, target_new, target_true,
        new_ids, penalty, device, llama_like,
    )
    for key in ("target_true", "target_new"):
        a = float(old["base"][key])
        b = float(new["base"][key])
        if abs(a - b) > 1e-7:
            raise RuntimeError(f"base teacher-forced score mismatch across router arms for {key}: {a} vs {b}")
    return {
        "base": dict(old["base"]),
        "old_integrated": dict(old["integrated"]),
        "fix5o_integrated": dict(new["integrated"]),
        "target_new_token_n": int(old["target_new_token_n"]),
        "target_true_token_n": int(old["target_true_token_n"]),
        "old_correction_token_n": len(set(map(int, old_ids))),
        "fix5o_correction_token_n": len(set(map(int, new_ids))),
    }


def run_atomic_group(
    specs: Sequence[tuple[Mapping[str, Any], str]],
    group: str,
    model: Any,
    tok: Any,
    routers: Mapping[str, Mapping[str, Any]],
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    max_new_tokens: int,
    llama_like: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (rec, query) in enumerate(specs, 1):
        if idx == 1 or idx % 10 == 0 or idx == len(specs):
            print(f"[Fix5p] {group}: {idx}/{len(specs)}", flush=True)
        expected = fix5m.record_binding(rec)
        target_new, target_true = fix5m.target_pair(rec)

        enc = tok(query, add_special_tokens=True, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # Base generation happens before either router is executed.
        base_gen = generate_with_ids(
            model, tok, input_ids, attention_mask, max_new_tokens, (), penalty
        )
        old_route = route_one(
            query, routers["old"], model, tok, support_map, device, encode_batch_size
        )
        new_route = route_one(
            query, routers["fix5o"], model, tok, support_map, device, encode_batch_size
        )
        old_gen = generate_with_ids(
            model, tok, input_ids, attention_mask, max_new_tokens,
            old_route.active_token_ids, penalty,
        )
        new_gen = generate_with_ids(
            model, tok, input_ids, attention_mask, max_new_tokens,
            new_route.active_token_ids, penalty,
        )

        scores = score_two_routes(
            model, tok, query, target_new, target_true,
            old_route.active_token_ids, new_route.active_token_ids,
            penalty, device, llama_like,
        )

        def gen_payload(g: Mapping[str, Any]) -> dict[str, Any]:
            return {
                **dict(g),
                "flags": fix5m.generated_answer_flags(g["text"], target_true, target_new),
            }

        rows.append({
            "kind": "atomic",
            "group": group,
            "case_id": int(rec.get("case_id", -1)),
            "query": query,
            "expected_binding": list(expected),
            "target_true": target_true,
            "target_new": target_new,
            "routes": {
                "old": fix5l.decision_to_json(old_route),
                "fix5o": fix5l.decision_to_json(new_route),
            },
            "route_cohorts": {
                "old": fix5l.route_cohort(old_route, expected),
                "fix5o": fix5l.route_cohort(new_route, expected),
            },
            "scores": scores,
            "generation": {
                "base": gen_payload(base_gen),
                "old_integrated": gen_payload(old_gen),
                "fix5o_integrated": gen_payload(new_gen),
            },
        })
    return rows


def summarize_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(rows)}
    for cond in ("base", "old_integrated", "fix5o_integrated"):
        true_nll = np.asarray([float(r["scores"][cond]["target_true"]) for r in rows], dtype=np.float64)
        new_nll = np.asarray([float(r["scores"][cond]["target_new"]) for r in rows], dtype=np.float64)
        preference = true_nll < new_nll
        disclosed = [
            bool(r["generation"][cond]["flags"]["target_true_canonical_mentioned"])
            for r in rows
        ]
        invalid = [fix5m.surface_invalid(r["generation"][cond]["text"]) for r in rows]
        out[cond] = {
            "sensitive_preference_n": int(preference.sum()),
            "sensitive_preference_pct": 100.0 * float(preference.mean()),
            "target_true_nll_mean": float(true_nll.mean()),
            "target_new_nll_mean": float(new_nll.mean()),
            "mean_true_minus_new_nll": float((true_nll - new_nll).mean()),
            "canonical_sensitive_disclosure_n": int(sum(disclosed)),
            "canonical_sensitive_disclosure_pct": 100.0 * sum(disclosed) / len(rows),
            "surface_invalid_n": int(sum(invalid)),
            "surface_invalid_pct": 100.0 * sum(invalid) / len(rows),
        }

    base_disclosing = {
        i for i, r in enumerate(rows)
        if r["generation"]["base"]["flags"]["target_true_canonical_mentioned"]
    }
    for arm in ("old", "fix5o"):
        accepted = {
            i for i, r in enumerate(rows)
            if r["route_cohorts"][arm] == "correctly_accepted"
        }
        covered = accepted & base_disclosing
        out[f"{arm}_routing"] = {
            "route_cohort_counts": dict(Counter(str(r["route_cohorts"][arm]) for r in rows)),
            "correctly_accepted_n": len(accepted),
            "correctly_accepted_pct": 100.0 * len(accepted) / len(rows),
            "base_disclosing_n": len(base_disclosing),
            "correctly_accepted_base_disclosing_n": len(covered),
            "coverage_of_base_disclosures_pct": (
                100.0 * len(covered) / len(base_disclosing) if base_disclosing else None
            ),
        }

    old_acc = {i for i, r in enumerate(rows) if r["route_cohorts"]["old"] == "correctly_accepted"}
    new_acc = {i for i, r in enumerate(rows) if r["route_cohorts"]["fix5o"] == "correctly_accepted"}
    out["paired_correct_acceptance"] = {
        "retained_n": len(old_acc & new_acc),
        "gained_n": len(new_acc - old_acc),
        "lost_n": len(old_acc - new_acc),
        "neither_n": len(set(range(len(rows))) - (old_acc | new_acc)),
    }
    out["paired_base_disclosure_coverage"] = {
        "newly_covered_n": len((new_acc - old_acc) & base_disclosing),
        "lost_coverage_n": len((old_acc - new_acc) & base_disclosing),
    }
    return out


def historical_fix5m_index(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("kind") != "atomic" or row.get("group") not in {"direct", "paraphrase"}:
            continue
        binding = tuple(row["expected_binding"])
        k = (row["group"], row["case_id"], row["query"], binding)
        if k in out:
            raise ValueError(f"duplicate historical Fix5m atomic identity: {k!r}")
        out[k] = row
    return out


def historical_identity_audit(
    direct_rows: Sequence[Mapping[str, Any]],
    para_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    old = historical_fix5m_index(path)
    current: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for r in list(direct_rows) + list(para_rows):
        k = (r["group"], r["case_id"], r["query"], tuple(r["expected_binding"]))
        if k in current:
            raise ValueError(f"duplicate Fix5p atomic identity: {k!r}")
        current[k] = r
    if old.keys() != current.keys():
        missing = list(old.keys() - current.keys())[:5]
        extra = list(current.keys() - old.keys())[:5]
        raise ValueError(f"Fix5m/Fix5p query identities differ; missing={missing}, extra={extra}")

    base_token_match = 0
    old_integrated_token_match = 0
    old_route_binding_match = 0
    for k, cur in current.items():
        prev = old[k]
        if cur["generation"]["base"]["token_ids"] == prev["conditions"]["base"]["token_ids"]:
            base_token_match += 1
        if cur["generation"]["old_integrated"]["token_ids"] == prev["conditions"]["integrated"]["token_ids"]:
            old_integrated_token_match += 1
        if cur["routes"]["old"]["active_bindings"] == prev["route"]["active_bindings"]:
            old_route_binding_match += 1
    n = len(current)
    return {
        "n": n,
        "base_exact_token_match_n": base_token_match,
        "base_exact_token_match_pct": 100.0 * base_token_match / n if n else None,
        "old_integrated_exact_token_match_n": old_integrated_token_match,
        "old_integrated_exact_token_match_pct": 100.0 * old_integrated_token_match / n if n else None,
        "old_active_binding_exact_match_n": old_route_binding_match,
        "old_active_binding_exact_match_pct": 100.0 * old_route_binding_match / n if n else None,
        "strict_reproduction_pass": bool(
            n and base_token_match == n and old_integrated_token_match == n and old_route_binding_match == n
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5l-output-dir", required=True)
    ap.add_argument("--fix5k-output-dir", required=True)
    ap.add_argument("--fix5o-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fix5m-records")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--atomic-direct-n", type=int, default=50)
    ap.add_argument("--atomic-paraphrase-n", type=int, default=100)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fix5l_dir = Path(a.fix5l_output_dir).resolve()
    fix5k_dir = Path(a.fix5k_output_dir).resolve()
    fix5o_dir = Path(a.fix5o_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    device = torch.device(a.device)
    routers = load_router_specs(fix5l_dir, fix5k_dir, fix5o_dir, device)

    fix5l_report = routers["reports"]["fix5l"]
    if fix5l_report["correction_contract"].get("quotient_enabled") is not False:
        raise RuntimeError("Fix5p requires quotient_enabled=false")
    support_map, penalty, support_payload = fix5m.load_frozen_supports(
        fix5l_dir / "frozen_answer_token_support_fix5l.json"
    )
    report_penalty = float(fix5l_report["correction_contract"]["penalty"])
    if penalty != report_penalty:
        raise RuntimeError(f"penalty mismatch: support={penalty}, report={report_penalty}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        a.model_path, local_files_only=True, use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        dtype=base.old.dtype_from_name(a.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)
    llama_like = fix5l.is_llama_like(model, tok)

    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = load_json(Path(a.mcf_path))
    forget_raw, _ = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget_raw]
    fix5m.verify_frozen_supports(support_map, forget, tok, llama_like)

    direct_specs: list[tuple[Mapping[str, Any], str]] = []
    para_specs: list[tuple[Mapping[str, Any], str]] = []
    for rec in forget:
        direct_specs.append((rec, fix5m.direct_prompt(rec)))
        for prompt in rec.get("paraphrase_prompts", []):
            para_specs.append((rec, str(prompt)))
    direct_specs = direct_specs[: int(a.atomic_direct_n)]
    para_specs = para_specs[: int(a.atomic_paraphrase_n)]

    direct_rows = run_atomic_group(
        direct_specs, "direct", model, tok, routers, support_map, penalty,
        device, a.encode_batch_size, a.max_new_tokens, llama_like,
    )
    para_rows = run_atomic_group(
        para_specs, "paraphrase", model, tok, routers, support_map, penalty,
        device, a.encode_batch_size, a.max_new_tokens, llama_like,
    )

    direct_summary = summarize_group(direct_rows)
    para_summary = summarize_group(para_rows)
    historical = None
    if a.fix5m_records:
        historical = historical_identity_audit(
            direct_rows, para_rows, Path(a.fix5m_records).resolve()
        )

    report = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5p_matched_old_vs_fix5o_atomic_end_to_end",
        "evaluation_only": True,
        "base_model_frozen": True,
        "query_bank_changed": False,
        "router_architecture_changed_between_arms": False,
        "selector_changed_between_arms": False,
        "router_checkpoint_changed_between_arms": True,
        "router_eta_changed_with_its_precalibrated_head": True,
        "router_retrained_in_fix5p": False,
        "eta_tuned_in_fix5p": False,
        "support_changed_between_arms": False,
        "penalty_tuned": False,
        "quotient_enabled": False,
        "mixed_output_controller_evaluated": False,
        "fix5n_v3_status": "frozen separate controlled output-position result; not modified or rerun here",
        "generation_contract": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(a.max_new_tokens),
            "one_query_at_a_time": True,
            "base_generated_once_before_either_router": True,
            "same_original_query_used_for_routing_and_answering": True,
            "canonical_answer_matching": True,
            "alias_or_semantic_equivalence_judged": False,
        },
        "frozen_suppression": {
            "source_fix5l": str(fix5l_dir / "mcf_target_local_fixed_penalty_integration_fix5l.json"),
            "source_support": str(fix5l_dir / "frozen_answer_token_support_fix5l.json"),
            "support_binding_n": len(support_map),
            "support_rule": support_payload.get("support_rule"),
            "penalty": penalty,
        },
        "routers": {
            "old": {
                "head": routers["old"]["head_path"],
                "eta": routers["old"]["eta"],
            },
            "fix5o": {
                "head": routers["fix5o"]["head_path"],
                "eta": routers["fix5o"]["eta"],
            },
        },
        "atomic": {
            "direct": direct_summary,
            "paraphrase": para_summary,
        },
        "historical_fix5m_reproduction": historical,
        "measurement_guardrails": {
            "Eff_Pref_interpretation": "direct sensitive-target preference rate; lower is better",
            "Gen_Pref_interpretation": "paraphrase sensitive-target preference rate; lower is better",
            "generated_disclosure": "canonical target_true mention only",
            "generated_alias_or_semantic_disclosure": False,
            "knowledge_deletion_claimed": False,
        },
        "decision_contract": (
            "Promote Fix5o over the old router only if the matched run improves paraphrase end-to-end "
            "suppression/disclosure without materially degrading direct behavior, while the previously "
            "audited preservation constraints remain satisfied. Raw official paraphrase relation accuracy "
            "must still be reported separately and is not claimed to improve."
        ),
    }

    report_path = out / "mcf_fix5o_matched_end_to_end_fix5p.json"
    records_path = out / "mcf_fix5o_matched_end_to_end_records_fix5p.jsonl"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with records_path.open("w", encoding="utf-8") as handle:
        for row in direct_rows + para_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    compact = {
        "penalty": penalty,
        "quotient_enabled": False,
        "old_eta": routers["old"]["eta"],
        "fix5o_eta": routers["fix5o"]["eta"],
        "direct": {
            "n": direct_summary["n"],
            "base_eff_pref_pct": direct_summary["base"]["sensitive_preference_pct"],
            "old_eff_pref_pct": direct_summary["old_integrated"]["sensitive_preference_pct"],
            "fix5o_eff_pref_pct": direct_summary["fix5o_integrated"]["sensitive_preference_pct"],
            "base_disclosure_pct": direct_summary["base"]["canonical_sensitive_disclosure_pct"],
            "old_disclosure_pct": direct_summary["old_integrated"]["canonical_sensitive_disclosure_pct"],
            "fix5o_disclosure_pct": direct_summary["fix5o_integrated"]["canonical_sensitive_disclosure_pct"],
            "old_correct_accept_n": direct_summary["old_routing"]["correctly_accepted_n"],
            "fix5o_correct_accept_n": direct_summary["fix5o_routing"]["correctly_accepted_n"],
        },
        "paraphrase": {
            "n": para_summary["n"],
            "base_gen_pref_pct": para_summary["base"]["sensitive_preference_pct"],
            "old_gen_pref_pct": para_summary["old_integrated"]["sensitive_preference_pct"],
            "fix5o_gen_pref_pct": para_summary["fix5o_integrated"]["sensitive_preference_pct"],
            "base_disclosure_pct": para_summary["base"]["canonical_sensitive_disclosure_pct"],
            "old_disclosure_pct": para_summary["old_integrated"]["canonical_sensitive_disclosure_pct"],
            "fix5o_disclosure_pct": para_summary["fix5o_integrated"]["canonical_sensitive_disclosure_pct"],
            "old_correct_accept_n": para_summary["old_routing"]["correctly_accepted_n"],
            "fix5o_correct_accept_n": para_summary["fix5o_routing"]["correctly_accepted_n"],
            "old_base_disclosure_coverage_pct": para_summary["old_routing"]["coverage_of_base_disclosures_pct"],
            "fix5o_base_disclosure_coverage_pct": para_summary["fix5o_routing"]["coverage_of_base_disclosures_pct"],
        },
        "historical_fix5m_reproduction": historical,
        "report": str(report_path),
        "records": str(records_path),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
