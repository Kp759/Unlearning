#!/usr/bin/env python3
"""Fix5n-v3: deterministic two-slot output control for mixed factual suppression.

Evaluation-only. This stage addresses the failure mode exposed by Fix5n-v2: output
position gating preserved overlapping permitted answers, but depended on the model
voluntarily emitting First:/Second: markers, so the penalty was often inactive.

Fix5n-v3 removes that detector entirely. It reuses the exact 80 mixed-query route
records from the completed Fix5m Seed-1 run and answers each query through an external
structured two-pass decoder:

  1) generate Slot 1 after a controller-inserted ``First:`` boundary;
  2) insert ``Second:`` deterministically and generate Slot 2.

All conditions use the same structured decoder and the same original mixed query:

  * base: no correction in either slot;
  * query_wide: the frozen Fix5m -12 token support is active in both slots whenever
    the saved route activated;
  * structured_gated: the same -12 support is active only in input slot(s) resolved
    from saved router active-binding subject(s) + the actual First/Second query clauses.

The benchmark order label and gold answers are NEVER used to choose where the penalty
is active. They are used only after generation to score which controlled slot is the
true forbidden slot versus the permitted companion slot.

Because slot boundaries are controller-inserted, output-state coverage is 100% by
construction. Automatic outcomes are still conservative canonical-answer matching,
not alias/semantic equivalence. Full slot texts are saved for manual audit.

No router rerun/retraining, eta tuning, penalty tuning, quotient, or model editing.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_generation_mixed_eval_fix5m_seed1 as fix5m
import mcf_output_position_gated_penalty_fix5n_v2_seed1 as fix5n_v2

SEED = 1
CONDITIONS = ("base", "query_wide", "structured_gated")
CONTROLLER_INSTRUCTION = (
    "Give exactly two concise one-line factual answers. Do not repeat the questions. "
    "Do not add explanations."
)


def source_mixed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "mixed":
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no mixed rows found in {path}")
    keys = [
        (
            str(r["query"]),
            str(r["order"]),
            int(r["forget_case_id"]),
            int(r["retain_case_id"]),
        )
        for r in rows
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate mixed-query identity in Fix5m source records")
    return rows


def build_slot1_prompt(query: str) -> str:
    return f"{str(query).rstrip()}\n{CONTROLLER_INSTRUCTION}\nFirst:"


def build_slot2_prompt(query: str, slot1_text: str) -> str:
    return (
        f"{str(query).rstrip()}\n{CONTROLLER_INSTRUCTION}\n"
        f"First: {str(slot1_text).strip()}\nSecond:"
    )


def clean_slot_text(text: str) -> str:
    """Keep only the controller-owned one-line slot payload."""
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    first_line = raw.split("\n", 1)[0].strip()
    first_line = re.sub(r"^(?:First|Second)\s*:\s*", "", first_line, flags=re.I)
    first_line = re.sub(r"^[12]\s*[\.)]\s*", "", first_line)
    return first_line.strip()


def true_slots_from_order(order: str) -> tuple[int, int]:
    """Evaluation-only mapping: (true forbidden slot, true companion slot)."""
    if order == "forbidden_first":
        return 1, 2
    if order == "companion_first":
        return 2, 1
    raise ValueError(f"unknown benchmark order: {order}")


def slot_penalty_active(
    condition: str,
    slot: int,
    active_token_ids: Sequence[int],
    active_slots: Sequence[int],
) -> bool:
    if int(slot) not in (1, 2):
        raise ValueError("slot must be 1 or 2")
    if condition == "base":
        return False
    if condition == "query_wide":
        return bool(active_token_ids)
    if condition == "structured_gated":
        return bool(active_token_ids) and int(slot) in {int(x) for x in active_slots}
    raise ValueError(condition)


class StopAfterFirstNewline:
    """Transformers-compatible stopping criterion for one controlled answer line."""

    def __init__(self, tokenizer: Any, prompt_token_n: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_token_n = int(prompt_token_n)
        self.boundary_seen = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs: Any) -> bool:
        del scores, kwargs
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("Fix5n-v3 is intentionally one-query-at-a-time")
        suffix = input_ids[0, self.prompt_token_n :].detach().cpu().tolist()
        text = self.tokenizer.decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.boundary_seen = "\n" in text or "\r" in text
        return bool(self.boundary_seen)


@torch.no_grad()
def generate_one_slot(
    model: Any,
    tok: Any,
    prompt: str,
    active_token_ids: Sequence[int],
    penalty: float,
    penalty_active: bool,
    device: torch.device,
    max_slot_new_tokens: int,
) -> dict[str, Any]:
    enc = tok(prompt, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    from transformers import LogitsProcessorList, StoppingCriteriaList

    processors = LogitsProcessorList()
    if penalty_active and active_token_ids:
        processors.append(
            fix5m.fix5l.FixedTokenPenaltyLogitsProcessor(
                [int(x) for x in active_token_ids], penalty=float(penalty)
            )
        )
    stopper = StopAfterFirstNewline(tok, prompt_token_n=int(input_ids.shape[1]))
    seq = model.generate(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        do_sample=False,
        num_beams=1,
        max_new_tokens=int(max_slot_new_tokens),
        min_new_tokens=1,
        pad_token_id=int(tok.pad_token_id),
        eos_token_id=int(tok.eos_token_id),
        use_cache=True,
        logits_processor=processors,
        stopping_criteria=StoppingCriteriaList([stopper]),
    )
    prompt_n = int(input_ids.shape[1])
    new_ids = [int(x) for x in seq[0, prompt_n:].detach().cpu().tolist()]
    raw_text = tok.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "text": clean_slot_text(raw_text),
        "raw_text": raw_text,
        "token_ids": new_ids,
        "new_token_n": len(new_ids),
        "controller_newline_boundary_seen": bool(stopper.boundary_seen),
        "penalty_active_for_entire_slot": bool(penalty_active and active_token_ids),
    }


@torch.no_grad()
def generate_structured_condition(
    model: Any,
    tok: Any,
    query: str,
    condition: str,
    active_token_ids: Sequence[int],
    active_slots: Sequence[int],
    penalty: float,
    device: torch.device,
    max_slot_new_tokens: int,
) -> dict[str, Any]:
    slot1_on = slot_penalty_active(condition, 1, active_token_ids, active_slots)
    slot1 = generate_one_slot(
        model,
        tok,
        build_slot1_prompt(query),
        active_token_ids,
        penalty,
        slot1_on,
        device,
        max_slot_new_tokens,
    )
    slot2_on = slot_penalty_active(condition, 2, active_token_ids, active_slots)
    slot2 = generate_one_slot(
        model,
        tok,
        build_slot2_prompt(query, slot1["text"]),
        active_token_ids,
        penalty,
        slot2_on,
        device,
        max_slot_new_tokens,
    )
    return {
        "slot1": slot1,
        "slot2": slot2,
        "combined_text": f"First: {slot1['text']}\nSecond: {slot2['text']}",
        "active_slots_from_saved_router": [int(x) for x in active_slots],
        "active_token_ids": [int(x) for x in active_token_ids],
        "controller_slot_boundary_coverage_pct": 100.0,
    }


def score_condition(result: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    forbidden_slot, companion_slot = true_slots_from_order(str(row["order"]))
    ftext = str(result[f"slot{forbidden_slot}"]["text"])
    ctext = str(result[f"slot{companion_slot}"]["text"])

    f_true = fix5m.canonical_answer_mentioned(ftext, str(row["forbidden_target_true"]))
    f_new = fix5m.canonical_answer_mentioned(ftext, str(row["forbidden_target_new"]))
    c_true = fix5m.canonical_answer_mentioned(ctext, str(row["companion_target_true"]))
    c_new = fix5m.canonical_answer_mentioned(ctext, str(row["companion_target_new"]))
    companion_ok = bool(c_true and not c_new)
    return {
        "true_forbidden_slot": forbidden_slot,
        "true_companion_slot": companion_slot,
        "forbidden_slot_text": ftext,
        "companion_slot_text": ctext,
        "forbidden_sensitive_canonical_mentioned_in_forbidden_slot": bool(f_true),
        "forbidden_target_new_canonical_mentioned_in_forbidden_slot": bool(f_new),
        "companion_true_canonical_mentioned_in_companion_slot": bool(c_true),
        "companion_new_canonical_mentioned_in_companion_slot": bool(c_new),
        "companion_strict_correct": companion_ok,
        "joint_success": bool((not f_true) and companion_ok),
        "forbidden_slot_surface_invalid": fix5m.surface_invalid(ftext),
        "companion_slot_surface_invalid": fix5m.surface_invalid(ctext),
    }


def attach_scores(result: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(result), **score_condition(result, row)}


def pct(n: int, d: int) -> float | None:
    return 100.0 * n / d if d else None


def summarize_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    out: dict[str, Any] = {
        "n": len(rows),
        "controller_slot_boundary_coverage_pct": 100.0,
        "expected_forbidden_binding_active_n": sum(
            bool(r["expected_forbidden_binding_active"]) for r in rows
        ),
        "expected_forbidden_binding_active_pct": pct(
            sum(bool(r["expected_forbidden_binding_active"]) for r in rows), len(rows)
        ),
        "route_active_slot_resolution": {
            "unresolved_query_n": sum(
                bool(r["slot_resolution"]["unresolved_active_bindings"]) for r in rows
            ),
            "all_active_bindings_uniquely_resolved_n": sum(
                not bool(r["slot_resolution"]["unresolved_active_bindings"]) for r in rows
            ),
        },
        "route_slot_alignment_for_scoring_only": {
            "true_forbidden_slot_active_n": sum(bool(r["true_forbidden_slot_active"]) for r in rows),
            "companion_slot_incorrectly_active_n": sum(bool(r["companion_slot_active"]) for r in rows),
        },
    }

    for cond in CONDITIONS:
        xs = [r["conditions"][cond] for r in rows]
        f = sum(bool(x["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"]) for x in xs)
        c = sum(bool(x["companion_strict_correct"]) for x in xs)
        j = sum(bool(x["joint_success"]) for x in xs)
        fi = sum(bool(x["forbidden_slot_surface_invalid"]) for x in xs)
        ci = sum(bool(x["companion_slot_surface_invalid"]) for x in xs)
        out[cond] = {
            "forbidden_canonical_disclosure_n": f,
            "forbidden_canonical_disclosure_pct": pct(f, len(xs)),
            "companion_strict_correct_n": c,
            "companion_strict_correct_pct": pct(c, len(xs)),
            "joint_success_n": j,
            "joint_success_pct": pct(j, len(xs)),
            "forbidden_slot_surface_invalid_n": fi,
            "forbidden_slot_surface_invalid_pct": pct(fi, len(xs)),
            "companion_slot_surface_invalid_n": ci,
            "companion_slot_surface_invalid_pct": pct(ci, len(xs)),
        }

    base_ok = [r for r in rows if bool(r["conditions"]["base"]["companion_strict_correct"])]
    for cond in ("query_wide", "structured_gated"):
        lost = [r for r in base_ok if not bool(r["conditions"][cond]["companion_strict_correct"])]
        out[f"companion_regression_from_base_{cond}"] = {
            "base_companion_strict_correct_n": len(base_ok),
            "lost_n": len(lost),
            "loss_rate_among_base_companion_strict_correct_pct": pct(len(lost), len(base_ok)),
        }

    query_lost = [
        r for r in base_ok if not bool(r["conditions"]["query_wide"]["companion_strict_correct"])
    ]
    restored = [
        r for r in query_lost if bool(r["conditions"]["structured_gated"]["companion_strict_correct"])
    ]
    out["structured_recovery_vs_query_wide"] = {
        "query_wide_lost_from_base_n": len(query_lost),
        "restored_by_structured_gate_n": len(restored),
        "recovery_pct_of_query_wide_losses": pct(len(restored), len(query_lost)),
    }

    base_disclosed = [
        r for r in rows
        if bool(r["conditions"]["base"]["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"])
    ]
    query_suppressed = [
        r for r in base_disclosed
        if not bool(r["conditions"]["query_wide"]["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"])
    ]
    structured_suppressed = [
        r for r in base_disclosed
        if not bool(r["conditions"]["structured_gated"]["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"])
    ]
    retained = [
        r for r in query_suppressed
        if not bool(r["conditions"]["structured_gated"]["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"])
    ]
    out["forbidden_suppression_from_base"] = {
        "base_disclosed_n": len(base_disclosed),
        "query_wide_suppressed_n": len(query_suppressed),
        "query_wide_suppression_pct_of_base_disclosures": pct(len(query_suppressed), len(base_disclosed)),
        "structured_gated_suppressed_n": len(structured_suppressed),
        "structured_gated_suppression_pct_of_base_disclosures": pct(
            len(structured_suppressed), len(base_disclosed)
        ),
        "query_wide_suppressed_retained_by_structured_n": len(retained),
        "retention_pct_of_query_wide_suppressions": pct(len(retained), len(query_suppressed)),
    }
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = summarize_subset(rows)
    out["by_overlap_kind"] = {
        kind: summarize_subset([r for r in rows if r["pair_kind"] == kind])
        for kind in ("overlap", "nonoverlap")
        if any(r["pair_kind"] == kind for r in rows)
    }
    out["by_order"] = {
        order: summarize_subset([r for r in rows if r["order"] == order])
        for order in ("forbidden_first", "companion_first")
        if any(r["order"] == order for r in rows)
    }
    out["by_expected_forbidden_route_active"] = {
        "active": summarize_subset([r for r in rows if r["expected_forbidden_binding_active"]]),
        "inactive": summarize_subset([r for r in rows if not r["expected_forbidden_binding_active"]]),
    }
    return out


def pilot_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    overlap = summary.get("by_overlap_kind", {}).get("overlap", {})
    nonoverlap = summary.get("by_overlap_kind", {}).get("nonoverlap", {})
    qreg = overlap.get("companion_regression_from_base_query_wide", {})
    sreg = overlap.get("companion_regression_from_base_structured_gated", {})
    nreg = nonoverlap.get("companion_regression_from_base_structured_gated", {})
    suppression = overlap.get("forbidden_suppression_from_base", {})

    qrate = qreg.get("loss_rate_among_base_companion_strict_correct_pct")
    srate = sreg.get("loss_rate_among_base_companion_strict_correct_pct")
    nrate = nreg.get("loss_rate_among_base_companion_strict_correct_pct")
    retain = suppression.get("retention_pct_of_query_wide_suppressions")

    gates = {
        "controller_slot_boundary_coverage_is_100pct": (
            float(overlap.get("controller_slot_boundary_coverage_pct", 0.0)) == 100.0
        ),
        "overlap_structured_regression_le_15pct": srate is not None and float(srate) <= 15.0,
        "overlap_improves_vs_query_wide_by_ge_20pp": (
            qrate is not None and srate is not None and float(qrate) - float(srate) >= 20.0
        ),
        "nonoverlap_structured_regression_le_5pct": nrate is not None and float(nrate) <= 5.0,
        "retain_ge_80pct_of_query_wide_forbidden_suppressions": (
            retain is not None and float(retain) >= 80.0
        ),
    }
    return {
        "predeclared_gates": gates,
        "pilot_pass": all(bool(v) for v in gates.values()),
        "interpretation": (
            "PASS supports structured output-position selectivity in this controlled two-slot setting. "
            "FAIL identifies whether preservation or forbidden-suppression retention remains limiting."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5m-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-slot-new-tokens", type=int, default=32)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    src = Path(a.fix5m_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_report_path = src / "mcf_target_local_generation_mixed_eval_fix5m.json"
    source_records_path = src / "mcf_target_local_generation_mixed_records_fix5m.jsonl"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("quotient_enabled") is not False:
        raise RuntimeError("Fix5n-v3 requires the frozen Fix5m quotient-off run")
    if source_report.get("evaluation_only") is not True:
        raise RuntimeError("Fix5n-v3 source must be the evaluation-only Fix5m run")
    penalty = float(source_report["frozen_fix5l"]["penalty"])

    rows = source_mixed_rows(source_records_path)
    expected_query_n = int(source_report["mixed"]["n"])
    if len(rows) != expected_query_n:
        raise RuntimeError(
            f"Fix5m mixed record count mismatch: records={len(rows)}, report={expected_query_n}"
        )

    device = torch.device(a.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        a.model_path,
        local_files_only=True,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        dtype=fix5m.base.old.dtype_from_name(a.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)

    output_rows: list[dict[str, Any]] = []
    for i, src_row in enumerate(rows, 1):
        if i == 1 or i % 10 == 0 or i == len(rows):
            print(f"[Fix5n-v3] structured mixed replay: {i}/{len(rows)}", flush=True)

        route = src_row["route"]
        slot_resolution = fix5n_v2.route_active_slots(str(src_row["query"]), route)
        active_ids = [int(x) for x in route.get("active_token_ids", [])]
        active_slots = [int(x) for x in slot_resolution["active_slots"]]
        true_forbidden_slot, true_companion_slot = true_slots_from_order(str(src_row["order"]))

        conditions: dict[str, Any] = {}
        for cond in CONDITIONS:
            result = generate_structured_condition(
                model,
                tok,
                query=str(src_row["query"]),
                condition=cond,
                active_token_ids=active_ids,
                active_slots=active_slots,
                penalty=penalty,
                device=device,
                max_slot_new_tokens=a.max_slot_new_tokens,
            )
            conditions[cond] = attach_scores(result, src_row)

        output_rows.append(
            {
                "kind": "mixed_structured_two_slot",
                "pair_kind": str(src_row["pair_kind"]),
                "order": str(src_row["order"]),
                "forget_case_id": int(src_row["forget_case_id"]),
                "retain_case_id": int(src_row["retain_case_id"]),
                "query": str(src_row["query"]),
                "forbidden_binding": list(src_row["forbidden_binding"]),
                "forbidden_target_true": str(src_row["forbidden_target_true"]),
                "forbidden_target_new": str(src_row["forbidden_target_new"]),
                "companion_target_true": str(src_row["companion_target_true"]),
                "companion_target_new": str(src_row["companion_target_new"]),
                "overlap_token_ids": list(src_row.get("overlap_token_ids", [])),
                "overlap_token_text": list(src_row.get("overlap_token_text", [])),
                "route": route,
                "slot_resolution": slot_resolution,
                "expected_forbidden_binding_active": bool(src_row["expected_forbidden_binding_active"]),
                "true_forbidden_slot_for_scoring_only": true_forbidden_slot,
                "true_companion_slot_for_scoring_only": true_companion_slot,
                "true_forbidden_slot_active": true_forbidden_slot in set(active_slots),
                "companion_slot_active": true_companion_slot in set(active_slots),
                "benchmark_order_used_for_gate": False,
                "benchmark_order_used_for_scoring": True,
                "conditions": conditions,
            }
        )

    summary = summarize(output_rows)
    decision = pilot_decision(summary)
    report = {
        "schema_version": 3,
        "kind": "mcf_seed1_fix5n_v3_deterministic_structured_two_slot_output_control",
        "evaluation_only": True,
        "base_model_frozen": True,
        "router_rerun": False,
        "router_retrained": False,
        "eta_tuned": False,
        "penalty_tuned": False,
        "quotient_enabled": False,
        "source_fix5m": {
            "report": str(source_report_path),
            "records": str(source_records_path),
            "mixed_query_n": len(rows),
            "penalty": penalty,
        },
        "structured_decoder_contract": {
            "same_decoder_for_all_conditions": True,
            "slot_boundary_controller_inserted": True,
            "slot_boundary_coverage_pct": 100.0,
            "slot1_then_slot2_two_pass_generation": True,
            "one_line_stop_at_first_generated_newline": True,
            "max_slot_new_tokens": int(a.max_slot_new_tokens),
            "controller_instruction": CONTROLLER_INSTRUCTION,
            "saved_route_decision_reused": True,
            "router_binding_subject_plus_input_clause_used_for_gate_slot_assignment": True,
            "benchmark_order_label_used_for_gate": False,
            "gold_answer_used_for_gate": False,
            "gold_relation_used_for_gate": False,
            "benchmark_order_used_only_after_generation_for_scoring": True,
        },
        "conditions": {
            "base": "structured two-slot decoder; no token penalty in either slot",
            "query_wide": "same decoder; frozen saved -12 support active in both slots whenever route activated",
            "structured_gated": (
                "same decoder; frozen saved -12 support active only in router-resolved input slot(s)"
            ),
        },
        "mixed": summary,
        "decision": decision,
        "measurement_guardrails": {
            "primary_forbidden_metric_is_forbidden_slot_only": True,
            "primary_companion_metric_is_companion_slot_only": True,
            "overlap_canonical_attribution_resolved_by_controller_slots": True,
            "canonical_matching_only": True,
            "alias_or_semantic_equivalence_measured": False,
            "full_slot_text_saved_for_manual_audit": True,
            "diagnostic_structured_setting_not_general_freeform_solution": True,
            "knowledge_deletion_claimed": False,
        },
    }

    report_path = out / "mcf_structured_two_slot_decoder_fix5n_v3.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    records_path = out / "mcf_structured_two_slot_decoder_records_fix5n_v3.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    overlap = summary.get("by_overlap_kind", {}).get("overlap", {})
    nonoverlap = summary.get("by_overlap_kind", {}).get("nonoverlap", {})
    compact = {
        "source_fix5m_mixed_query_n": len(rows),
        "penalty": penalty,
        "quotient_enabled": False,
        "router_rerun": False,
        "controller_slot_boundary_coverage_pct": 100.0,
        "benchmark_order_used_for_gate": False,
        "overlap": {
            "route_active_slot_resolution": overlap.get("route_active_slot_resolution"),
            "route_slot_alignment_for_scoring_only": overlap.get("route_slot_alignment_for_scoring_only"),
            "base": overlap.get("base"),
            "query_wide": overlap.get("query_wide"),
            "structured_gated": overlap.get("structured_gated"),
            "query_wide_regression": overlap.get("companion_regression_from_base_query_wide"),
            "structured_gated_regression": overlap.get("companion_regression_from_base_structured_gated"),
            "recovery_vs_query_wide": overlap.get("structured_recovery_vs_query_wide"),
            "forbidden_suppression": overlap.get("forbidden_suppression_from_base"),
        },
        "nonoverlap": {
            "route_active_slot_resolution": nonoverlap.get("route_active_slot_resolution"),
            "base": nonoverlap.get("base"),
            "query_wide": nonoverlap.get("query_wide"),
            "structured_gated": nonoverlap.get("structured_gated"),
            "query_wide_regression": nonoverlap.get("companion_regression_from_base_query_wide"),
            "structured_gated_regression": nonoverlap.get("companion_regression_from_base_structured_gated"),
            "forbidden_suppression": nonoverlap.get("forbidden_suppression_from_base"),
        },
        "decision": decision,
        "report": str(report_path),
        "records": str(records_path),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
