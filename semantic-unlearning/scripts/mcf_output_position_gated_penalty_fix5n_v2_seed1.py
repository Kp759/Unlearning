#!/usr/bin/env python3
"""Fix5n-v2: saved-route output-position-gated penalty replay (Seed 1).

Evaluation-only and leakage-safe with respect to output-slot assignment. This stage
reuses the exact 80 mixed-query records from the completed Fix5m run. It does not
rerun/retrain/recalibrate the router and does not change eta, support tokens, penalty,
model weights, or quotient state.

Conditions:
  * base: saved Fix5m Base generation
  * query_wide: saved Fix5m integrated query-wide -12 generation
  * position_gated: one new generation using the same saved active token IDs, but the
    penalty is applied only while the generated answer-state parser is in a slot that
    the frozen router itself mapped to an active forbidden binding.

Crucially, the position gate does NOT use the benchmark's forbidden_first /
companion_first order label to decide where to intervene. The active output slot is
derived from (1) the saved router active binding subject(s) and (2) the actual First:/
Second: input clauses. If an active subject cannot be mapped uniquely to one input
clause, that binding is unresolved and contributes no position-gated penalty.

Generated output state is recognized only from the generated suffix using explicit
First:/Second: labels or line-leading 1./2. labels. Before a usable marker is seen the
penalty is OFF. This is a structured-output diagnostic, not a claim that free-form
output-position binding is solved.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
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

SEED = 1
CONDITIONS = ("base", "query_wide", "position_gated")

_FIRST_PATTERNS = (
    re.compile(r"(?im)(?:^|\n)\s*first\s*:\s*"),
    re.compile(r"(?m)(?:^|\n)\s*1\s*[\.)]\s*"),
)
_SECOND_PATTERNS = (
    re.compile(r"(?im)(?:^|\n)\s*second\s*:\s*"),
    re.compile(r"(?m)(?:^|\n)\s*2\s*[\.)]\s*"),
)


def _last_match(patterns: Sequence[re.Pattern[str]], text: str) -> int | None:
    positions = [m.start() for p in patterns for m in p.finditer(str(text))]
    return max(positions) if positions else None


def detect_generated_slot(text: str) -> int | None:
    first = _last_match(_FIRST_PATTERNS, text)
    second = _last_match(_SECOND_PATTERNS, text)
    if first is None and second is None:
        return None
    if second is None:
        return 1
    if first is None:
        return 2
    return 1 if first > second else 2


def marker_diagnostics(text: str) -> dict[str, Any]:
    first = _last_match(_FIRST_PATTERNS, text)
    second = _last_match(_SECOND_PATTERNS, text)
    return {
        "first_marker_seen": first is not None,
        "second_marker_seen": second is not None,
        "final_detected_slot": detect_generated_slot(text),
    }


def parse_mixed_query_slots(query: str) -> dict[int, str]:
    """Parse the exact Fix5m First/Second question structure from input text only."""
    text = str(query)
    first = re.search(r"(?m)^First:\s*", text)
    second = re.search(r"(?m)^Second:\s*", text)
    tail = re.search(r"(?m)^Answer both questions in order\.\s*$", text)
    if first is None or second is None or tail is None:
        raise RuntimeError("Fix5n-v2 requires the saved Fix5m First/Second mixed-query format")
    if not (first.end() <= second.start() < tail.start()):
        raise RuntimeError("malformed First/Second mixed-query ordering")
    return {
        1: text[first.end() : second.start()].strip(),
        2: text[second.end() : tail.start()].strip(),
    }


def subject_occurs(text: str, subject: str) -> bool:
    return bool(
        re.search(
            r"(?<!\w)" + re.escape(str(subject)) + r"(?!\w)",
            str(text),
            flags=re.IGNORECASE,
        )
    )


def route_active_slots(query: str, route: Mapping[str, Any]) -> dict[str, Any]:
    """Map saved active router bindings to input slots without using benchmark order."""
    clauses = parse_mixed_query_slots(query)
    resolved: set[int] = set()
    unresolved: list[dict[str, Any]] = []
    for binding_raw in route.get("active_bindings", []):
        binding = list(binding_raw)
        if len(binding) < 2:
            unresolved.append({"binding": binding, "reason": "malformed_binding"})
            continue
        subject = str(binding[0])
        hits = [slot for slot, clause in clauses.items() if subject_occurs(clause, subject)]
        if len(hits) == 1:
            resolved.add(int(hits[0]))
        else:
            unresolved.append(
                {
                    "binding": [str(binding[0]), str(binding[1])],
                    "subject_slot_hits": hits,
                    "reason": "subject_not_uniquely_mapped_to_input_slot",
                }
            )
    return {
        "active_slots": sorted(resolved),
        "unresolved_active_bindings": unresolved,
        "active_binding_n": len(route.get("active_bindings", [])),
        "resolved_active_slot_n": len(resolved),
        "input_clauses": clauses,
        "uses_benchmark_order_label": False,
        "uses_gold_answer": False,
        "uses_gold_relation": False,
    }


class PositionGatedTokenPenaltyLogitsProcessor:
    def __init__(
        self,
        tokenizer: Any,
        prompt_token_n: int,
        token_ids: Sequence[int],
        penalty: float,
        active_slots: Sequence[int],
    ) -> None:
        slots = tuple(sorted({int(x) for x in active_slots}))
        if any(x not in (1, 2) for x in slots):
            raise ValueError("active_slots may contain only 1 and/or 2")
        self.tokenizer = tokenizer
        self.prompt_token_n = int(prompt_token_n)
        self.token_ids = tuple(sorted({int(x) for x in token_ids}))
        self.penalty = float(penalty)
        self.active_slots = slots
        self.total_step_n = 0
        self.active_step_n = 0
        self.slot_step_n: Counter[int | None] = Counter()
        self.first_marker_seen = False
        self.second_marker_seen = False

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise RuntimeError("Fix5n-v2 is intentionally one-query-at-a-time")
        suffix = input_ids[0, self.prompt_token_n :].detach().cpu().tolist()
        text = self.tokenizer.decode(
            suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        diag = marker_diagnostics(text)
        self.first_marker_seen = self.first_marker_seen or bool(diag["first_marker_seen"])
        self.second_marker_seen = self.second_marker_seen or bool(diag["second_marker_seen"])
        slot = diag["final_detected_slot"]
        self.total_step_n += 1
        self.slot_step_n[slot] += 1
        if slot in self.active_slots and self.token_ids:
            self.active_step_n += 1
            return fix5m.fix5l.apply_fixed_penalty(scores, self.token_ids, self.penalty)
        return scores

    def snapshot(self) -> dict[str, Any]:
        usable = any(
            (slot == 1 and self.first_marker_seen) or (slot == 2 and self.second_marker_seen)
            for slot in self.active_slots
        )
        return {
            "active_slots_from_router_query": list(self.active_slots),
            "total_generation_step_n": self.total_step_n,
            "penalty_active_step_n": self.active_step_n,
            "penalty_active_step_pct": (
                100.0 * self.active_step_n / self.total_step_n
                if self.total_step_n
                else None
            ),
            "pre_marker_or_unknown_step_n": int(self.slot_step_n.get(None, 0)),
            "slot1_step_n": int(self.slot_step_n.get(1, 0)),
            "slot2_step_n": int(self.slot_step_n.get(2, 0)),
            "first_marker_seen": self.first_marker_seen,
            "second_marker_seen": self.second_marker_seen,
            "usable_active_slot_marker_seen": bool(usable),
        }


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
        (str(r["query"]), str(r["order"]), int(r["forget_case_id"]), int(r["retain_case_id"]))
        for r in rows
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate mixed-query identity in Fix5m source records")
    return rows


@torch.no_grad()
def generate_position_gated(
    model: Any,
    tok: Any,
    query: str,
    active_token_ids: Sequence[int],
    penalty: float,
    active_slots: Sequence[int],
    device: torch.device,
    max_new_tokens: int,
) -> dict[str, Any]:
    enc = tok(query, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    processor = PositionGatedTokenPenaltyLogitsProcessor(
        tok,
        prompt_token_n=int(input_ids.shape[1]),
        token_ids=active_token_ids,
        penalty=penalty,
        active_slots=active_slots,
    )
    gen = fix5m.generate_from_inputs(
        model,
        tok,
        input_ids,
        attention_mask,
        max_new_tokens=max_new_tokens,
        processor=processor,
    )
    return {**gen, "position_gate": processor.snapshot()}


def condition_flags(text: str, row: Mapping[str, Any]) -> dict[str, Any]:
    f_disclose = fix5m.canonical_answer_mentioned(text, str(row["forbidden_target_true"]))
    c_true = fix5m.canonical_answer_mentioned(text, str(row["companion_target_true"]))
    c_new = fix5m.canonical_answer_mentioned(text, str(row["companion_target_new"]))
    return {
        "forbidden_sensitive_canonical_mentioned": f_disclose,
        "forbidden_target_new_canonical_mentioned": fix5m.canonical_answer_mentioned(
            text, str(row["forbidden_target_new"])
        ),
        "companion_true_canonical_mentioned": c_true,
        "companion_new_canonical_mentioned": c_new,
        "companion_strict_correct": bool(c_true and not c_new),
        "joint_success": bool((not f_disclose) and c_true and not c_new),
        "surface_invalid": fix5m.surface_invalid(text),
    }


def normalize_source_condition(row: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    if source_name not in row["conditions"]:
        raise RuntimeError(f"source Fix5m row missing condition {source_name}")
    x = dict(row["conditions"][source_name])
    required = {
        "forbidden_sensitive_canonical_mentioned",
        "companion_true_canonical_mentioned",
        "companion_new_canonical_mentioned",
        "companion_strict_correct",
        "joint_success",
        "surface_invalid",
    }
    if not required.issubset(x):
        x.update(condition_flags(str(x.get("text", "")), row))
    return x


def attribution_ambiguous(row: Mapping[str, Any]) -> bool:
    if "canonical_answer_attribution_ambiguous" in row:
        return bool(row["canonical_answer_attribution_ambiguous"])
    a = fix5m.canonical_normalize(str(row["forbidden_target_true"]))
    b = fix5m.canonical_normalize(str(row["companion_target_true"]))
    if not a or not b:
        return True
    return f" {a} " in f" {b} " or f" {b} " in f" {a} "


def summarize_subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": len(rows),
        "expected_forbidden_binding_active_n": sum(
            bool(r["expected_forbidden_binding_active"]) for r in rows
        ),
        "expected_forbidden_binding_active_pct": 100.0
        * sum(bool(r["expected_forbidden_binding_active"]) for r in rows)
        / len(rows),
        "route_active_slot_resolution": {
            "all_active_bindings_uniquely_resolved_n": sum(
                not bool(r["slot_resolution"]["unresolved_active_bindings"]) for r in rows
            ),
            "unresolved_query_n": sum(
                bool(r["slot_resolution"]["unresolved_active_bindings"]) for r in rows
            ),
        },
    }
    for cond in CONDITIONS:
        xs = [r["conditions"][cond] for r in rows]
        forbidden = sum(bool(x["forbidden_sensitive_canonical_mentioned"]) for x in xs)
        companion = sum(bool(x["companion_strict_correct"]) for x in xs)
        invalid = sum(bool(x["surface_invalid"]) for x in xs)
        out[cond] = {
            "forbidden_canonical_disclosure_n": forbidden,
            "forbidden_canonical_disclosure_pct": 100.0 * forbidden / len(xs),
            "companion_strict_correct_n": companion,
            "companion_strict_correct_pct": 100.0 * companion / len(xs),
            "joint_success_n": sum(bool(x["joint_success"]) for x in xs),
            "joint_success_pct": 100.0 * sum(bool(x["joint_success"]) for x in xs) / len(xs),
            "surface_invalid_n": invalid,
            "surface_invalid_pct": 100.0 * invalid / len(xs),
        }

    base_ok = [r for r in rows if bool(r["conditions"]["base"]["companion_strict_correct"])]
    for cond in ("query_wide", "position_gated"):
        lost = [r for r in base_ok if not bool(r["conditions"][cond]["companion_strict_correct"])]
        out[f"companion_regression_from_base_{cond}"] = {
            "base_companion_strict_correct_n": len(base_ok),
            "lost_n": len(lost),
            "loss_rate_among_base_companion_strict_correct_pct": (
                100.0 * len(lost) / len(base_ok) if base_ok else None
            ),
        }

    query_lost = [
        r for r in base_ok if not bool(r["conditions"]["query_wide"]["companion_strict_correct"])
    ]
    restored = [r for r in query_lost if bool(r["conditions"]["position_gated"]["companion_strict_correct"])]
    out["position_gate_recovery_vs_query_wide"] = {
        "query_wide_lost_from_base_n": len(query_lost),
        "restored_by_position_gate_n": len(restored),
        "recovery_pct_of_query_wide_losses": (
            100.0 * len(restored) / len(query_lost) if query_lost else None
        ),
    }

    ambiguous = [r for r in rows if attribution_ambiguous(r)]
    safe = [r for r in rows if not attribution_ambiguous(r)]
    out["canonical_answer_attribution"] = {
        "ambiguous_n": len(ambiguous),
        "unambiguous_n": len(safe),
        "companion_metrics_use_all_rows": True,
    }
    if safe:
        safe_metrics: dict[str, Any] = {"n": len(safe)}
        for cond in CONDITIONS:
            xs = [r["conditions"][cond] for r in safe]
            safe_metrics[cond] = {
                "forbidden_canonical_disclosure_n": sum(
                    bool(x["forbidden_sensitive_canonical_mentioned"]) for x in xs
                ),
                "forbidden_canonical_disclosure_pct": 100.0
                * sum(bool(x["forbidden_sensitive_canonical_mentioned"]) for x in xs)
                / len(xs),
                "joint_success_n": sum(bool(x["joint_success"]) for x in xs),
                "joint_success_pct": 100.0 * sum(bool(x["joint_success"]) for x in xs) / len(xs),
            }
        out["attribution_safe_forbidden_and_joint"] = safe_metrics
    else:
        out["attribution_safe_forbidden_and_joint"] = {
            "n": 0,
            "base": None,
            "query_wide": None,
            "position_gated": None,
        }

    gate_rows = [r["conditions"]["position_gated"]["position_gate"] for r in rows]
    out["position_gate_diagnostics"] = {
        "usable_active_slot_marker_seen_n": sum(
            bool(x["usable_active_slot_marker_seen"]) for x in gate_rows
        ),
        "usable_active_slot_marker_seen_pct": 100.0
        * sum(bool(x["usable_active_slot_marker_seen"]) for x in gate_rows)
        / len(gate_rows),
        "first_marker_seen_n": sum(bool(x["first_marker_seen"]) for x in gate_rows),
        "second_marker_seen_n": sum(bool(x["second_marker_seen"]) for x in gate_rows),
        "mean_penalty_active_step_pct": float(
            np.mean([float(x["penalty_active_step_pct"] or 0.0) for x in gate_rows])
        ),
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
    out["by_overlap_and_order"] = {}
    for kind in ("overlap", "nonoverlap"):
        for order in ("forbidden_first", "companion_first"):
            sub = [r for r in rows if r["pair_kind"] == kind and r["order"] == order]
            if sub:
                out["by_overlap_and_order"][f"{kind}:{order}"] = summarize_subset(sub)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5m-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=64)
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
        raise RuntimeError("Fix5n-v2 requires the frozen Fix5m quotient-off run")
    if source_report.get("evaluation_only") is not True:
        raise RuntimeError("Fix5n-v2 source must be the evaluation-only Fix5m run")
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
            print(f"[Fix5n-v2] position-gated mixed replay: {i}/{len(rows)}", flush=True)
        route = src_row["route"]
        slot_resolution = route_active_slots(str(src_row["query"]), route)
        active_ids = [int(x) for x in route.get("active_token_ids", [])]
        pos = generate_position_gated(
            model,
            tok,
            query=str(src_row["query"]),
            active_token_ids=active_ids,
            penalty=penalty,
            active_slots=slot_resolution["active_slots"],
            device=device,
            max_new_tokens=a.max_new_tokens,
        )
        pos_condition = {**pos, **condition_flags(str(pos["text"]), src_row)}
        row = {
            "kind": "mixed",
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
            "canonical_answer_attribution_ambiguous": attribution_ambiguous(src_row),
            "conditions": {
                "base": normalize_source_condition(src_row, "base"),
                "query_wide": normalize_source_condition(src_row, "integrated"),
                "position_gated": pos_condition,
            },
        }
        output_rows.append(row)

    summary = summarize(output_rows)
    report = {
        "schema_version": 2,
        "kind": "mcf_seed1_fix5n_v2_saved_route_output_position_gated_penalty_replay",
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
            "atomic_snapshot_carried_forward_not_rerun": source_report.get("atomic", {}),
        },
        "conditions": {
            "base": "saved deterministic Base generation from Fix5m",
            "query_wide": "saved frozen query-wide -12 generation from Fix5m",
            "position_gated": (
                "new generation using exact saved active token IDs; active input slots are "
                "derived from saved router active-binding subjects plus actual First/Second "
                "query clauses; penalty applies only when generated output state is in one "
                "of those resolved active slots"
            ),
        },
        "position_gate_contract": {
            "input_query_changed_from_fix5m": False,
            "saved_route_decision_reused": True,
            "benchmark_order_label_used_to_choose_active_slot": False,
            "gold_answer_used_for_gate": False,
            "gold_relation_used_for_gate": False,
            "gate_reads_generated_suffix_only_for_output_state": True,
            "router_binding_subject_plus_input_clause_used_for_slot_assignment": True,
            "unresolved_active_binding_behavior": "no position-gated penalty for unresolved binding",
            "before_first_usable_output_marker_penalty_active": False,
            "recognized_output_markers": ["First:", "Second:", "line-leading 1.", "line-leading 2."],
            "diagnostic_not_general_freeform_solution": True,
        },
        "mixed": summary,
        "measurement_guardrails": {
            "canonical_disclosure_only": True,
            "alias_or_semantic_disclosure_measured": False,
            "overlap_forbidden_attribution_ambiguous_rows_excluded_from_attribution_safe_metrics": True,
            "companion_preservation_uses_all_rows": True,
        },
        "decision_rule": (
            "Continue output-position control only if position_gated materially reduces "
            "overlap companion regression versus query_wide while retaining comparable "
            "attribution-safe forbidden suppression on rows with resolved active slots and "
            "usable output markers. If output-marker coverage is poor, improve output-state "
            "detection before changing penalty strength or the router."
        ),
    }

    report_path = out / "mcf_output_position_gated_penalty_fix5n_v2.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records_path = out / "mcf_output_position_gated_penalty_records_fix5n_v2.jsonl"
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
        "benchmark_order_used_for_gate": False,
        "overlap": {
            "route_active_slot_resolution": overlap.get("route_active_slot_resolution"),
            "query_wide_regression": overlap.get("companion_regression_from_base_query_wide"),
            "position_gated_regression": overlap.get("companion_regression_from_base_position_gated"),
            "recovery_vs_query_wide": overlap.get("position_gate_recovery_vs_query_wide"),
            "position_gate_diagnostics": overlap.get("position_gate_diagnostics"),
            "attribution_safe_forbidden_and_joint": overlap.get("attribution_safe_forbidden_and_joint"),
        },
        "nonoverlap": {
            "route_active_slot_resolution": nonoverlap.get("route_active_slot_resolution"),
            "query_wide_regression": nonoverlap.get("companion_regression_from_base_query_wide"),
            "position_gated_regression": nonoverlap.get("companion_regression_from_base_position_gated"),
            "position_gate_diagnostics": nonoverlap.get("position_gate_diagnostics"),
            "attribution_safe_forbidden_and_joint": nonoverlap.get("attribution_safe_forbidden_and_joint"),
        },
        "by_order": summary.get("by_order"),
        "report": str(report_path),
        "records": str(records_path),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
