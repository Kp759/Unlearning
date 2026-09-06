#!/usr/bin/env python3
"""Fix5l: Seed-1 exact-name target-local router + fixed query-locked -12 integration.

This is an end-to-end *conditional suppression integration baseline*, not a new router.
It freezes the successful Fix5k exact-name target-local relation head and selector and
compares three conditions on the original answering prompt:

  1. base:        no routing and no output change
  2. router_only: execute the frozen router, but leave answer logits unchanged
  3. integrated:  execute the same router and subtract a fixed penalty from the
                  selected vocabulary support whenever a forbidden binding activates

The quotient is disabled. The cropped/marked target-local text is used only by the
recognition branch; the answering model always receives the original query.

Correction support contract
---------------------------
The older private-vocabulary artifact used in prior Method4/5 runs is not assumed to
be available here. Fix5l therefore states its support rule explicitly: for each
registered forbidden (subject, relation) binding, freeze the tokenizer IDs of that
fact's sensitive target_true answer. If one or more bindings activate, subtract the
same fixed penalty (default 12.0) from the UNION of those IDs at every answer-token
prediction step. There is no gold answer-position gating and no relation/answer/case
metadata is passed into the router. The same apply_fixed_penalty() primitive is used
for teacher-forced scoring and the provided generation logits processor.

Primary outputs are preference/NLL metrics, not generated leakage. No generation
claim is made unless a caller separately uses the shared processor in generation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_typed_masking_ablation_fix5k_seed1 as fix5k

local = fix5k.local
base = fix5k.base
NONE = base.NONE
SEED = 1


@dataclass(frozen=True)
class BindingSupport:
    subject: str
    relation: str
    target_true: str
    target_new: str
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class RouteDecision:
    query: str
    candidates: tuple[str, ...]
    active_bindings: tuple[tuple[str, str], ...]
    active_token_ids: tuple[int, ...]
    routes: tuple[dict[str, Any], ...]


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def target_string(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("str", ""))
    return str(value)


def is_llama_like(model: Any, tok: Any) -> bool:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    name = str(getattr(model.config, "_name_or_path", "")).lower()
    tok_cls = tok.__class__.__name__.lower()
    return "llama" in model_type or "llama" in name or "llama" in tok_cls


def answer_token_ids(tok: Any, text: str, llama_like: bool) -> list[int]:
    ids = list(tok(" " + str(text), add_special_tokens=True)["input_ids"])
    if llama_like and ids:
        ids = ids[1:]
    return [int(x) for x in ids]


def apply_fixed_penalty(scores: torch.Tensor, token_ids: Sequence[int], penalty: float) -> torch.Tensor:
    """Return scores with -penalty on the union support; input is never mutated."""
    out = scores.clone()
    ids = sorted({int(x) for x in token_ids})
    if ids:
        idx = torch.tensor(ids, dtype=torch.long, device=out.device)
        if out.ndim == 1:
            out[idx] -= float(penalty)
        elif out.ndim == 2:
            out[:, idx] -= float(penalty)
        else:
            raise ValueError(f"scores must be rank 1 or 2, got {tuple(out.shape)}")
    return out


class FixedTokenPenaltyLogitsProcessor:
    """HF-compatible generation processor sharing Fix5l correction semantics."""

    def __init__(self, token_ids: Sequence[int], penalty: float = 12.0):
        self.token_ids = tuple(sorted({int(x) for x in token_ids}))
        self.penalty = float(penalty)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        del input_ids
        return apply_fixed_penalty(scores, self.token_ids, self.penalty)


def load_head(path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    classes = list(payload["classes"])
    state = payload["state_dict"]
    weight = state.get("fc.weight")
    if weight is None or weight.ndim != 2:
        raise RuntimeError("saved exact-name head lacks linear fc.weight")
    head = base.Linear(int(weight.shape[1]), len(classes)).to(device)
    head.load_state_dict(state)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, classes


def build_support_map(
    forget_records: Sequence[Mapping[str, Any]],
    tok: Any,
    llama_like: bool,
) -> dict[tuple[str, str], BindingSupport]:
    out: dict[tuple[str, str], BindingSupport] = {}
    for rec in forget_records:
        rr0 = rec["requested_rewrite"]
        rr = rr0[0] if isinstance(rr0, list) else rr0
        subject = str(rr["subject"])
        info = base.old.rr(rec)
        relation = str(info["relation_id"])
        true = target_string(rr["target_true"])
        new = target_string(rr["target_new"])
        ids = tuple(sorted(set(answer_token_ids(tok, true, llama_like))))
        if not ids:
            raise RuntimeError(f"empty sensitive answer support for {(subject, relation)}")
        key = (subject, relation)
        if key in out:
            raise RuntimeError(f"duplicate forbidden binding in Seed-1 sample: {key}")
        out[key] = BindingSupport(subject, relation, true, new, ids)
    return out


@torch.no_grad()
def route_query(
    query: str,
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], BindingSupport],
    device: torch.device,
    encode_batch_size: int,
) -> RouteDecision:
    """Run the frozen router from query text only; no evaluation metadata is accepted."""
    bank = set(support_map)
    bank_subjects = sorted({s for s, _ in bank}, key=len, reverse=True)
    candidates = local.registered_subject_candidates(query, bank_subjects)
    if not candidates:
        return RouteDecision(str(query), tuple(), tuple(), tuple(), tuple())

    views = [local.routing_view(query, subject, bank_subjects) for subject in candidates]
    features, inv, _ = local.encode_unique(
        model, tok, [v.selected_text for v in views], device, encode_batch_size
    )
    with torch.no_grad():
        unique_logits = head(features.to(device)).cpu()
    logits = unique_logits[torch.tensor(inv, dtype=torch.long)]
    pred, margin = base.margin(logits)
    routes: list[dict[str, Any]] = []
    active: list[tuple[str, str]] = []
    none_idx = classes.index(NONE)
    for i, (subject, view) in enumerate(zip(candidates, views)):
        pred_idx = int(pred[i].item())
        relation = str(classes[pred_idx])
        relation_accepted = bool(
            view.scope_supported
            and pred_idx != none_idx
            and float(margin[i].item()) >= float(eta)
        )
        binding = (subject, relation)
        activates = bool(relation_accepted and binding in bank)
        if activates:
            active.append(binding)
        routes.append({
            "subject": subject,
            "enumerated_from_query": True,
            "selected_text": view.selected_text,
            "selection_status": view.selection_status,
            "scope_supported": bool(view.scope_supported),
            "predicted_relation": relation,
            "margin": float(margin[i].item()),
            "relation_accepted": relation_accepted,
            "forbidden_bank_lookup": bool(binding in bank),
            "activates": activates,
        })

    active = sorted(set(active))
    token_ids = sorted({tid for b in active for tid in support_map[b].token_ids})
    return RouteDecision(
        query=str(query),
        candidates=tuple(candidates),
        active_bindings=tuple(active),
        active_token_ids=tuple(token_ids),
        routes=tuple(routes),
    )


def route_cohort(
    decision: RouteDecision,
    expected_binding: tuple[str, str],
) -> str:
    active = set(decision.active_bindings)
    wrong = active - {expected_binding}
    if wrong:
        return "wrong_binding_accepted"
    if expected_binding in active:
        return "correctly_accepted"
    target_routes = [r for r in decision.routes if r["subject"] == expected_binding[0]]
    if target_routes and any(r["predicted_relation"] == expected_binding[1] for r in target_routes):
        return "correctly_classified_but_rejected_or_unsupported"
    return "misclassified"


@torch.no_grad()
def score_choice_pair(
    model: Any,
    tok: Any,
    prefix: str,
    target_new: str,
    target_true: str,
    correction_ids: Sequence[int],
    penalty: float,
    device: torch.device,
    llama_like: bool,
) -> dict[str, Any]:
    """Teacher-force true/new with one query-locked correction decision."""
    prefix_ids = tok([prefix], add_special_tokens=True)["input_ids"][0]
    prefix_len = len(prefix_ids)
    inputs = tok(
        [f"{prefix} {target_new}", f"{prefix} {target_true}"],
        padding=True,
        return_tensors="pt",
    ).to(device)
    logits = model(**inputs, use_cache=False, return_dict=True).logits.float()
    new_ids = answer_token_ids(tok, target_new, llama_like)
    true_ids = answer_token_ids(tok, target_true, llama_like)
    if llama_like:
        logits = logits[:, 1:, :]
        prefix_len -= 1

    def one(row: int, ids: Sequence[int]) -> tuple[float, float, int]:
        base_nll = 0.0
        integrated_nll = 0.0
        for j, tid in enumerate(ids):
            pos = prefix_len + j - 1
            if pos < 0 or pos >= logits.shape[1]:
                raise RuntimeError(
                    f"invalid teacher-forcing position {pos} for prefix={prefix!r} answer={ids}"
                )
            base_scores = logits[row, pos, :]
            corrected = apply_fixed_penalty(base_scores, correction_ids, penalty)
            base_nll += -torch.log_softmax(base_scores, dim=0)[int(tid)].item()
            integrated_nll += -torch.log_softmax(corrected, dim=0)[int(tid)].item()
        n = max(1, len(ids))
        return base_nll / n, integrated_nll / n, len(ids)

    new_base, new_integrated, new_len = one(0, new_ids)
    true_base, true_integrated, true_len = one(1, true_ids)
    # Router-only identity uses the exact same unmodified answer logits as base.
    router_new = new_base
    router_true = true_base
    if router_new != new_base or router_true != true_base:
        raise AssertionError("router-only identity condition differs from base")
    return {
        "base": {"target_new": new_base, "target_true": true_base},
        "router_only": {"target_new": router_new, "target_true": router_true},
        "integrated": {"target_new": new_integrated, "target_true": true_integrated},
        "target_new_token_n": new_len,
        "target_true_token_n": true_len,
        "correction_token_n": len(set(map(int, correction_ids))),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(rows)}
    for condition in ("base", "router_only", "integrated"):
        t = np.array([float(r["scores"][condition]["target_true"]) for r in rows], dtype=np.float64)
        n = np.array([float(r["scores"][condition]["target_new"]) for r in rows], dtype=np.float64)
        pref = t < n
        out[condition] = {
            "sensitive_preference_pct": 100.0 * float(pref.mean()),
            "target_true_nll_mean": float(t.mean()),
            "target_new_nll_mean": float(n.mean()),
            "mean_true_minus_new_nll": float((t - n).mean()),
        }
    base_true = np.array([float(r["scores"]["base"]["target_true"]) for r in rows])
    int_true = np.array([float(r["scores"]["integrated"]["target_true"]) for r in rows])
    out["integrated_minus_base_sensitive_nll_mean"] = float((int_true - base_true).mean())
    cohorts = Counter(str(r["route_cohort"]) for r in rows)
    out["route_cohort_counts"] = dict(cohorts)
    out["active_route_pct"] = 100.0 * sum(bool(r["route"]["active_bindings"]) for r in rows) / len(rows)
    out["identity_control_max_abs_nll_diff"] = max(
        abs(float(r["scores"]["router_only"][k]) - float(r["scores"]["base"][k]))
        for r in rows for k in ("target_true", "target_new")
    )
    by_cohort: dict[str, Any] = {}
    for cohort in sorted(cohorts):
        sub = [r for r in rows if r["route_cohort"] == cohort]
        bt = np.array([r["scores"]["base"]["target_true"] for r in sub], dtype=np.float64)
        it = np.array([r["scores"]["integrated"]["target_true"] for r in sub], dtype=np.float64)
        by_cohort[cohort] = {
            "n": len(sub),
            "integrated_minus_base_sensitive_nll_mean": float((it - bt).mean()),
            "base_sensitive_preference_pct": 100.0 * float(np.mean([
                r["scores"]["base"]["target_true"] < r["scores"]["base"]["target_new"] for r in sub
            ])),
            "integrated_sensitive_preference_pct": 100.0 * float(np.mean([
                r["scores"]["integrated"]["target_true"] < r["scores"]["integrated"]["target_new"] for r in sub
            ])),
        }
    out["by_route_cohort"] = by_cohort
    return out


def decision_to_json(d: RouteDecision) -> dict[str, Any]:
    return {
        "query": d.query,
        "candidates": list(d.candidates),
        "active_bindings": [list(x) for x in d.active_bindings],
        "active_token_ids": list(d.active_token_ids),
        "routes": list(d.routes),
    }


def record_relation_binding(record: Mapping[str, Any]) -> tuple[str, str]:
    rr0 = record["requested_rewrite"]
    rr = rr0[0] if isinstance(rr0, list) else rr0
    return str(rr["subject"]), str(base.old.rr(record)["relation_id"])


def direct_prompt(record: Mapping[str, Any]) -> str:
    rr0 = record["requested_rewrite"]
    rr = rr0[0] if isinstance(rr0, list) else rr0
    return str(rr["prompt"]).format(str(rr["subject"]))


def target_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    rr0 = record["requested_rewrite"]
    rr = rr0[0] if isinstance(rr0, list) else rr0
    return target_string(rr["target_new"]), target_string(rr["target_true"])


@torch.no_grad()
def evaluate_forget(
    forget_records: Sequence[Mapping[str, Any]],
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    llama_like: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_rows: list[dict[str, Any]] = []
    para_rows: list[dict[str, Any]] = []
    route_cache: dict[str, RouteDecision] = {}

    def route(prefix: str) -> RouteDecision:
        key = norm_text(prefix)
        if key not in route_cache:
            route_cache[key] = route_query(
                prefix, model, tok, head, classes, eta, support_map, device, encode_batch_size
            )
        return route_cache[key]

    for rec in forget_records:
        case_id = int(rec.get("case_id", -1))
        expected = record_relation_binding(rec)
        new, true = target_pair(rec)
        prompts = [("direct", direct_prompt(rec))]
        prompts.extend(("paraphrase", str(x)) for x in rec.get("paraphrase_prompts", []))
        for group, prefix in prompts:
            d = route(prefix)
            scores = score_choice_pair(
                model, tok, prefix, new, true, d.active_token_ids, penalty, device, llama_like
            )
            row = {
                "case_id": case_id,
                "group": group,
                "query": prefix,
                "expected_binding": list(expected),
                "route_cohort": route_cohort(d, expected),
                "route": decision_to_json(d),
                "target_true": true,
                "target_new": new,
                "scores": scores,
            }
            (direct_rows if group == "direct" else para_rows).append(row)
    return direct_rows, para_rows


@torch.no_grad()
def evaluate_retain_direct(
    retain_records: Sequence[Mapping[str, Any]],
    limit: int,
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    llama_like: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for rec in list(retain_records)[: int(limit)]:
        prefix = direct_prompt(rec)
        new, true = target_pair(rec)
        d = route_query(prefix, model, tok, head, classes, eta, support_map, device, encode_batch_size)
        scores = score_choice_pair(
            model, tok, prefix, new, true, d.active_token_ids, penalty, device, llama_like
        )
        base_correct = scores["base"]["target_true"] < scores["base"]["target_new"]
        int_correct = scores["integrated"]["target_true"] < scores["integrated"]["target_new"]
        rows.append({
            "case_id": int(rec.get("case_id", -1)),
            "query": prefix,
            "route": decision_to_json(d),
            "target_true": true,
            "target_new": new,
            "base_correct": bool(base_correct),
            "integrated_correct": bool(int_correct),
            "base_correct_to_integrated_incorrect": bool(base_correct and not int_correct),
            "scores": scores,
        })
    if not rows:
        return {"n": 0}, rows
    report = {
        "n": len(rows),
        "active_correction_n": sum(bool(r["route"]["active_bindings"]) for r in rows),
        "active_correction_pct": 100.0 * sum(bool(r["route"]["active_bindings"]) for r in rows) / len(rows),
        "base_correct_n": sum(r["base_correct"] for r in rows),
        "integrated_correct_n": sum(r["integrated_correct"] for r in rows),
        "base_correct_to_integrated_incorrect_n": sum(r["base_correct_to_integrated_incorrect"] for r in rows),
        "target_true_nll_delta_mean": float(np.mean([
            r["scores"]["integrated"]["target_true"] - r["scores"]["base"]["target_true"] for r in rows
        ])),
        "router_only_identity_max_abs_nll_diff": max(
            abs(r["scores"]["router_only"][k] - r["scores"]["base"][k])
            for r in rows for k in ("target_true", "target_new")
        ),
    }
    return report, rows


def overlap_candidates(
    retain_records: Sequence[Mapping[str, Any]],
    support_map: Mapping[tuple[str, str], BindingSupport],
    tok: Any,
    llama_like: bool,
    limit: int,
) -> list[tuple[Mapping[str, Any], tuple[str, str], list[int]]]:
    out: list[tuple[Mapping[str, Any], tuple[str, str], list[int]]] = []
    supports = {b: set(v.token_ids) for b, v in support_map.items()}
    for rec in retain_records:
        _, true = target_pair(rec)
        rids = set(answer_token_ids(tok, true, llama_like))
        for binding, ids in supports.items():
            overlap = sorted(rids & ids)
            if overlap:
                out.append((rec, binding, overlap))
                break
        if len(out) >= int(limit):
            break
    return out


@torch.no_grad()
def evaluate_forced_overlap_stress(
    retain_records: Sequence[Mapping[str, Any]],
    support_map: Mapping[tuple[str, str], BindingSupport],
    tok: Any,
    model: Any,
    penalty: float,
    device: torch.device,
    llama_like: bool,
    limit: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for rec, binding, overlap in overlap_candidates(retain_records, support_map, tok, llama_like, limit):
        prefix = direct_prompt(rec)
        new, true = target_pair(rec)
        forced_ids = support_map[binding].token_ids
        scores = score_choice_pair(
            model, tok, prefix, new, true, forced_ids, penalty, device, llama_like
        )
        base_correct = scores["base"]["target_true"] < scores["base"]["target_new"]
        forced_correct = scores["integrated"]["target_true"] < scores["integrated"]["target_new"]
        rows.append({
            "retain_case_id": int(rec.get("case_id", -1)),
            "retain_query": prefix,
            "retain_target_true": true,
            "forced_forbidden_binding": list(binding),
            "forced_sensitive_answer": support_map[binding].target_true,
            "overlap_token_ids": overlap,
            "overlap_token_text": [tok.decode([x]) for x in overlap],
            "base_correct": bool(base_correct),
            "forced_correct": bool(forced_correct),
            "base_correct_to_forced_incorrect": bool(base_correct and not forced_correct),
            "scores": scores,
        })
    if not rows:
        return {"n": 0, "note": "No retained target_true answer overlapped a frozen forbidden support token."}, rows
    return {
        "n": len(rows),
        "base_correct_to_forced_incorrect_n": sum(r["base_correct_to_forced_incorrect"] for r in rows),
        "base_correct_to_forced_incorrect_pct": 100.0 * sum(r["base_correct_to_forced_incorrect"] for r in rows) / len(rows),
        "retain_true_nll_delta_mean_under_forced_active_support": float(np.mean([
            r["scores"]["integrated"]["target_true"] - r["scores"]["base"]["target_true"] for r in rows
        ])),
        "interpretation": (
            "Forced-active vocabulary-overlap stress only. The retained atomic query itself did not activate the router; "
            "this estimates output-token collateral if an overlapping forbidden support is active in a mixed response."
        ),
    }, rows


def compact_saved_recognition(report: Mapping[str, Any]) -> dict[str, Any]:
    arm = report["results"]["exact_name_target_local"]
    vp = arm["validation_policy"]
    dev = arm["development_only_official_seed1"]["paraphrase"]
    return {
        "pilot_pass": bool(arm["pilot_pass"]),
        "eta": float(arm["eta"]),
        "validation_relation_accuracy_pct": arm["semantic"]["validation"]["accuracy_pct"],
        "validation_correct_forbidden_accept_pct": vp["correct_forbidden_binding_accept_pct"],
        "validation_route_permitted_fpr_pct": vp["permitted_false_activation_pct"],
        "validation_query_permitted_fpr_pct": vp["whole_query"]["permitted_query_false_activation_pct"],
        "validation_mixed_companion_fpr_pct": vp["whole_query"]["mixed_query_permitted_companion_false_activation_pct"],
        "official_para_accuracy_pct": dev["semantic"]["accuracy_pct"],
        "official_para_correct_forbidden_accept_pct": dev["policy"]["correct_forbidden_binding_accept_pct"],
        "validation_selected_text_fit_overlap": arm.get("validation_selected_text_fit_overlap"),
        "official_para_selected_text_fit_overlap": dev.get("selected_text_fit_overlap"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5k-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--penalty", type=float, default=12.0)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--retain-eval-n", type=int, default=100)
    ap.add_argument("--overlap-stress-n", type=int, default=50)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    src = Path(a.fix5k_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_report_path = src / "target_local_typed_masking_ablation_fix5k.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    recognition = compact_saved_recognition(source_report)
    if not recognition["pilot_pass"]:
        raise RuntimeError("Fix5l requires the saved exact-name Fix5k arm to have pilot_pass=true")
    eta = float(recognition["eta"])

    device = torch.device(a.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        dtype=base.old.dtype_from_name(a.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    llama_like = is_llama_like(model, tok)

    head_path = src / "exact_name_target_local_linear_head.pt"
    if not head_path.is_file():
        raise RuntimeError(f"missing frozen Fix5k head: {head_path}")
    head, classes = load_head(head_path, device)

    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget_raw, retain_raw = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget_raw]
    retain = [off.normalize_record(x) for x in retain_raw]
    support_map = build_support_map(forget, tok, llama_like)
    if len(support_map) != 50:
        raise RuntimeError(f"expected 50 frozen forbidden bindings, got {len(support_map)}")
    unknown_rel = sorted({rel for _, rel in support_map if rel not in classes})
    if unknown_rel:
        raise RuntimeError(f"Fix5k head class list does not cover frozen bank relations: {unknown_rel}")

    (out / "frozen_answer_token_support_fix5l.json").write_text(
        json.dumps({
            "schema_version": 1,
            "support_rule": "unique tokenizer IDs of leading-space sensitive target_true answer for each frozen Seed-1 (subject, relation) binding",
            "penalty": float(a.penalty),
            "query_locked": True,
            "gold_answer_position_used": False,
            "quotient_enabled": False,
            "bindings": [
                {
                    "subject": v.subject,
                    "relation": v.relation,
                    "target_true": v.target_true,
                    "target_new": v.target_new,
                    "token_ids": list(v.token_ids),
                    "tokens": [tok.decode([x]) for x in v.token_ids],
                }
                for _, v in sorted(support_map.items())
            ],
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    direct_rows, para_rows = evaluate_forget(
        forget, model, tok, head, classes, eta, support_map, a.penalty,
        device, a.encode_batch_size, llama_like
    )
    direct_summary = summarize_rows(direct_rows)
    para_summary = summarize_rows(para_rows)

    retain_summary, retain_rows = evaluate_retain_direct(
        retain, a.retain_eval_n, model, tok, head, classes, eta, support_map,
        a.penalty, device, a.encode_batch_size, llama_like
    )
    overlap_summary, overlap_rows = evaluate_forced_overlap_stress(
        retain, support_map, tok, model, a.penalty, device, llama_like, a.overlap_stress_n
    )

    report = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5l_exact_name_target_local_query_locked_fixed_penalty_integration",
        "approximate_behavioral_unlearning": True,
        "conditions": {
            "base": {"router_executed": False, "output_changed": False},
            "router_only": {"router_executed": True, "output_changed": False},
            "integrated": {"router_executed": True, "output_changed_when_active": True},
        },
        "frozen_router": {
            "source_fix5k_report": str(source_report_path),
            "head": str(head_path),
            "representation": "exact_name_target_local",
            "eta": eta,
            "selector": "Fix5j/Fix5i target-local query-only selector imported through Fix5k",
            "recognition_snapshot": recognition,
            "retrained_in_fix5l": False,
        },
        "correction_contract": {
            "penalty": float(a.penalty),
            "sign": "subtract from selected logits",
            "support": "per-binding unique target_true tokenizer IDs; union across active bindings",
            "support_artifact": str(out / "frozen_answer_token_support_fix5l.json"),
            "query_locked": True,
            "route_computed_before_answer_suffix": True,
            "same_route_for_target_true_and_target_new_scoring": True,
            "applied_at_every_answer_prediction_step": True,
            "gold_answer_position_used": False,
            "quotient_enabled": False,
            "answering_model_receives_original_query": True,
            "recognizer_receives_target_local_marked_copy": True,
            "old_private_vocab_artifact_claimed_identical": False,
        },
        "forget": {
            "direct": direct_summary,
            "paraphrase": para_summary,
            "Eff_Pref_base_pct": direct_summary["base"]["sensitive_preference_pct"],
            "Eff_Pref_integrated_pct": direct_summary["integrated"]["sensitive_preference_pct"],
            "Gen_Pref_base_pct": para_summary["base"]["sensitive_preference_pct"],
            "Gen_Pref_integrated_pct": para_summary["integrated"]["sensitive_preference_pct"],
        },
        "preservation": {
            "fresh_retain_direct": retain_summary,
            "forced_active_answer_token_overlap_stress": overlap_summary,
            "saved_router_validation": recognition,
            "corpus_ppl_measured": False,
            "generated_disclosure_measured": False,
        },
        "identity_control": {
            "direct_max_abs_nll_diff": direct_summary["identity_control_max_abs_nll_diff"],
            "paraphrase_max_abs_nll_diff": para_summary["identity_control_max_abs_nll_diff"],
            "retain_max_abs_nll_diff": retain_summary.get("router_only_identity_max_abs_nll_diff"),
            "passes_exact_identity": bool(
                direct_summary["identity_control_max_abs_nll_diff"] == 0.0
                and para_summary["identity_control_max_abs_nll_diff"] == 0.0
                and (retain_summary.get("router_only_identity_max_abs_nll_diff") in (None, 0.0))
            ),
        },
        "interpretation_guardrail": (
            "Fix5l tests conditional output suppression after a frozen recognition pilot. Eff_Pref/Gen_Pref are preference rates, not generated leakage. "
            "No knowledge-deletion, generated-disclosure, corpus-PPL, or mixed-response preservation claim follows from this scoring-only integration."
        ),
    }
    report_path = out / "mcf_target_local_fixed_penalty_integration_fix5l.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    detail_path = out / "mcf_target_local_fixed_penalty_records_fix5l.jsonl"
    with detail_path.open("w", encoding="utf-8") as f:
        for r in direct_rows + para_rows:
            f.write(json.dumps({"cohort": "forget", **r}, ensure_ascii=False) + "\n")
        for r in retain_rows:
            f.write(json.dumps({"cohort": "fresh_retain_direct", **r}, ensure_ascii=False) + "\n")
        for r in overlap_rows:
            f.write(json.dumps({"cohort": "forced_active_overlap_stress", **r}, ensure_ascii=False) + "\n")

    compact = {
        "router": recognition,
        "penalty": float(a.penalty),
        "quotient_enabled": False,
        "identity_control_pass": report["identity_control"]["passes_exact_identity"],
        "Eff_Pref": {
            "base_pct": report["forget"]["Eff_Pref_base_pct"],
            "integrated_pct": report["forget"]["Eff_Pref_integrated_pct"],
        },
        "Gen_Pref": {
            "base_pct": report["forget"]["Gen_Pref_base_pct"],
            "integrated_pct": report["forget"]["Gen_Pref_integrated_pct"],
        },
        "direct_sensitive_nll_delta": direct_summary["integrated_minus_base_sensitive_nll_mean"],
        "paraphrase_sensitive_nll_delta": para_summary["integrated_minus_base_sensitive_nll_mean"],
        "direct_route_cohorts": direct_summary["route_cohort_counts"],
        "paraphrase_route_cohorts": para_summary["route_cohort_counts"],
        "fresh_retain": retain_summary,
        "forced_overlap_stress": overlap_summary,
        "generated_disclosure_measured": False,
        "corpus_ppl_measured": False,
        "report": str(report_path),
        "records": str(detail_path),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
