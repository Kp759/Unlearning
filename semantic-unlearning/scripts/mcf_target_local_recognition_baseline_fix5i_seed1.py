#!/usr/bin/env python3
"""Fix5i: full recognition-only target-local routing baseline with frozen marked head.

This experiment keeps the Fix5f target-marked linear head frozen and compares two
input pipelines on the same calibration/validation manifests:

  full_marked: existing [TARGET]name[/TARGET] full-query representation
  target_local: query-only conservative routing view for each registered subject

The target-local selector never receives a gold relation or answer. It enumerates
registered subject candidates from the original query text, isolates an explicit
First:/Second: clause when the designated candidate occurs in exactly one clause,
passes through ordinary atomic single-subject requests, and marks ambiguous scopes as
unsupported. Unsupported scopes remain in the denominator and are conservatively
ineligible for suppression activation.

The saved classifier is not retrained. Each input pipeline is separately calibrated
on calibration examples processed by that same pipeline under route-level,
whole-query, and per-family preservation constraints. Validation is evaluated once
with the frozen selector/head/threshold. Output correction and quotient stay off.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_relation_classifier_fix5_seed1 as base
import mcf_target_representation_compare_fix5e_seed1 as rep

Row = base.Row
NONE = base.NONE
OTHER = base.OTHER


@dataclass(frozen=True)
class RoutingView:
    selected_text: str
    selection_status: str
    selected_character_offsets: tuple[int, int] | None
    scope_supported: bool
    enumerated_subjects: tuple[str, ...]


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def registered_subject_candidates(text: str, bank_subjects: Sequence[str]) -> list[str]:
    found = []
    for subject in bank_subjects:
        if base.old.subject_regex(subject).search(str(text)):
            found.append(subject)
    return found


def split_first_second_with_offsets(text: str) -> tuple[tuple[str, tuple[int, int]], tuple[str, tuple[int, int]]] | None:
    m = re.match(r"^\s*First\s*:\s*(.*?)\s*Second\s*:\s*(.*?)\s*$", str(text), flags=re.I | re.S)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if not a or not b:
        return None
    a0, a1 = m.span(1)
    b0, b1 = m.span(2)
    while a0 < a1 and text[a0].isspace():
        a0 += 1
    while a1 > a0 and text[a1 - 1].isspace():
        a1 -= 1
    while b0 < b1 and text[b0].isspace():
        b0 += 1
    while b1 > b0 and text[b1 - 1].isspace():
        b1 -= 1
    return ((text[a0:a1], (a0, a1)), (text[b0:b1], (b0, b1)))


def routing_view(query: str, target_subject: str, bank_subjects: Sequence[str]) -> RoutingView:
    query = str(query)
    candidates = registered_subject_candidates(query, bank_subjects)
    candidate_tuple = tuple(candidates)
    if target_subject not in candidates:
        marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
        return RoutingView(
            selected_text=marked if found else norm_text(query),
            selection_status="UNSUPPORTED_TARGET_NOT_ENUMERATED",
            selected_character_offsets=(0, len(query)),
            scope_supported=False,
            enumerated_subjects=candidate_tuple,
        )

    structured = split_first_second_with_offsets(query)
    if structured is not None:
        hits = [bool(base.old.subject_regex(target_subject).search(clause)) for clause, _ in structured]
        if sum(hits) == 1:
            clause, offsets = structured[hits.index(True)]
            marked, found = rep.target_preserving_text(clause, target_subject, bank_subjects)
            if found:
                return RoutingView(
                    selected_text=marked,
                    selection_status="SELECTED_EXPLICIT_CLAUSE",
                    selected_character_offsets=offsets,
                    scope_supported=True,
                    enumerated_subjects=candidate_tuple,
                )
        marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
        return RoutingView(
            selected_text=marked if found else norm_text(query),
            selection_status="UNSUPPORTED_EXPLICIT_SCOPE_AMBIGUOUS",
            selected_character_offsets=(0, len(query)),
            scope_supported=False,
            enumerated_subjects=candidate_tuple,
        )

    target_hits = len(base.old.subject_regex(target_subject).findall(query))
    if target_hits != 1:
        marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
        return RoutingView(
            selected_text=marked if found else norm_text(query),
            selection_status="UNSUPPORTED_REPEATED_TARGET",
            selected_character_offsets=(0, len(query)),
            scope_supported=False,
            enumerated_subjects=candidate_tuple,
        )

    if len(candidates) == 1:
        marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
        if not found:
            return RoutingView(
                selected_text=norm_text(query),
                selection_status="UNSUPPORTED_TARGET_MARKING_FAILED",
                selected_character_offsets=(0, len(query)),
                scope_supported=False,
                enumerated_subjects=candidate_tuple,
            )
        return RoutingView(
            selected_text=marked,
            selection_status="ATOMIC_PASSTHROUGH",
            selected_character_offsets=(0, len(query)),
            scope_supported=True,
            enumerated_subjects=candidate_tuple,
        )

    marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
    return RoutingView(
        selected_text=marked if found else norm_text(query),
        selection_status="UNSUPPORTED_UNSTRUCTURED_MULTI_SUBJECT",
        selected_character_offsets=(0, len(query)),
        scope_supported=False,
        enumerated_subjects=candidate_tuple,
    )


def full_marked_view(query: str, target_subject: str, bank_subjects: Sequence[str]) -> RoutingView:
    marked, found = rep.target_preserving_text(query, target_subject, bank_subjects)
    candidates = tuple(registered_subject_candidates(query, bank_subjects))
    return RoutingView(
        selected_text=marked if found else norm_text(query),
        selection_status="FULL_MARKED_BASELINE" if found else "FULL_MARKING_FAILED",
        selected_character_offsets=(0, len(query)),
        scope_supported=bool(found and target_subject in candidates),
        enumerated_subjects=candidates,
    )


def extract_clause_for_subject(query: str, subject: str) -> str | None:
    structured = split_first_second_with_offsets(query)
    if structured is None:
        return None
    hits = [bool(base.old.subject_regex(subject).search(clause)) for clause, _ in structured]
    if sum(hits) != 1:
        return None
    return structured[hits.index(True)][0]


def build_mixed_queries(
    policy_rows: Sequence[Row],
    bank_subjects: Sequence[str],
    limit: int,
    phase: str,
) -> list[Row]:
    """Build training-safe one-forbidden + one-permitted independent-distractor queries.

    Uses only existing phase-local policy rows. The distractor relation comes from a
    crossed-binding permitted route; no relation label is used to select a clause.
    """
    positives = [
        r for r in policy_rows
        if r.forbidden and split_first_second_with_offsets(r.text) is None
    ]
    distractors = [
        r for r in policy_rows
        if (not r.forbidden) and base.bucket(r.kind) == "crossed_binding"
    ]
    out: list[Row] = []
    seen_queries: set[str] = set()
    if not positives or not distractors:
        return out
    dpos = 0
    for pos in positives:
        if len(seen_queries) >= int(limit):
            break
        chosen: tuple[Row, str] | None = None
        for off in range(len(distractors)):
            d = distractors[(dpos + off) % len(distractors)]
            if d.subject.casefold() == pos.subject.casefold():
                continue
            clause = extract_clause_for_subject(d.text, d.subject)
            if clause is None:
                continue
            chosen = (d, clause)
            dpos = (dpos + off + 1) % len(distractors)
            break
        if chosen is None:
            continue
        d, clause = chosen
        query = f"First: {norm_text(pos.text)} Second: {norm_text(clause)}"
        key = query.casefold()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        out.append(Row(
            text=query,
            subject=pos.subject,
            relation=pos.relation,
            forbidden=True,
            kind=f"mixed_forbidden_distractor_{phase}",
            family="mixed_forbidden_distractor",
            case_id=pos.case_id,
            masked="",
            candidate=True,
        ))
        out.append(Row(
            text=query,
            subject=d.subject,
            relation=d.relation,
            forbidden=False,
            kind=f"mixed_forbidden_distractor_{phase}",
            family="mixed_forbidden_distractor",
            case_id=pos.case_id,
            masked="",
            candidate=True,
        ))
    return out


def load_head(path: Path, input_dim: int, classes: Sequence[str], device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if list(payload["classes"]) != list(classes):
        raise RuntimeError("saved class ordering mismatch")
    head = base.Linear(input_dim, len(classes)).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def encode_unique(model: Any, tok: Any, texts: Sequence[str], device: torch.device, batch: int) -> tuple[torch.Tensor, list[int], list[str]]:
    unique: list[str] = []
    lookup: dict[str, int] = {}
    inv: list[int] = []
    for text in texts:
        key = str(text)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(key)
        inv.append(lookup[key])
    feats = base.encode(model, tok, unique, device, batch)
    return feats, inv, unique


def score_views(
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    device: torch.device,
    batch: int,
) -> torch.Tensor:
    features, inv, _ = encode_unique(model, tok, [v.selected_text for v in views], device, batch)
    with torch.no_grad():
        u = head(features.to(device)).cpu()
    return u[torch.tensor(inv, dtype=torch.long)]


def raw_semantic_report(rows: Sequence[Row], logits: torch.Tensor, classes: Sequence[str]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "accuracy_pct": None, "by_relation": {}, "by_family": {}}
    c2i = {c: i for i, c in enumerate(classes)}
    return base.sem_report(rows, logits, c2i, list(classes))


def decision_tensors(
    rows: Sequence[Row],
    logits: torch.Tensor,
    views: Sequence[RoutingView],
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    pred, margin = base.margin(logits)
    labels = [classes[int(i)] for i in pred]
    scope = torch.tensor([v.scope_supported for v in views], dtype=torch.bool)
    accepted_relation = scope & (pred != int(none_idx)) & (margin >= float(eta))
    binding = torch.tensor([(r.subject, labels[i]) in bank for i, r in enumerate(rows)], dtype=torch.bool)
    activates = accepted_relation & binding
    correct = torch.tensor([labels[i] == r.relation for i, r in enumerate(rows)], dtype=torch.bool)
    forbidden = torch.tensor([r.forbidden for r in rows], dtype=torch.bool)
    candidate = torch.tensor([r.candidate for r in rows], dtype=torch.bool)
    return {
        "pred": pred,
        "margin": margin,
        "labels": labels,
        "scope": scope,
        "accepted_relation": accepted_relation,
        "binding": binding,
        "activates": activates,
        "correct": correct,
        "forbidden": forbidden,
        "candidate": candidate,
    }


def rate_pct(mask: torch.Tensor) -> float | None:
    if mask.numel() == 0:
        return None
    return 100.0 * float(mask.float().mean())


def whole_query_report(rows: Sequence[Row], d: Mapping[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[(norm_text(row.text).casefold(), base.bucket(row.kind))].append(i)

    permitted_groups = {k: idx for k, idx in groups.items() if not any(rows[i].forbidden for i in idx)}
    mixed_groups = {
        k: idx for k, idx in groups.items()
        if any(rows[i].forbidden for i in idx) and any(not rows[i].forbidden for i in idx)
    }

    def permitted_summary(selected: Mapping[tuple[str, str], Sequence[int]]) -> tuple[int, int, float | None]:
        if not selected:
            return 0, 0, None
        err = sum(bool(d["activates"][list(idx)].any().item()) for idx in selected.values())
        return len(selected), err, 100.0 * err / len(selected)

    n, e, pct = permitted_summary(permitted_groups)
    by_family = {}
    for fam in sorted({k[1] for k in permitted_groups}):
        sub = {k: idx for k, idx in permitted_groups.items() if k[1] == fam}
        fn, fe, fpct = permitted_summary(sub)
        by_family[fam] = {"query_n": fn, "false_activation_n": fe, "false_activation_pct": fpct}

    mixed_correct = 0
    mixed_companion_error = 0
    mixed_success = 0
    for idx in mixed_groups.values():
        forb = [i for i in idx if rows[i].forbidden]
        perm = [i for i in idx if not rows[i].forbidden]
        correct_forbidden = any(bool((d["activates"][i] & d["correct"][i]).item()) for i in forb)
        companion_error = any(bool(d["activates"][i].item()) for i in perm)
        mixed_correct += int(correct_forbidden)
        mixed_companion_error += int(companion_error)
        mixed_success += int(correct_forbidden and not companion_error)

    mn = len(mixed_groups)
    return {
        "permitted_query_n": n,
        "permitted_query_false_activation_n": e,
        "permitted_query_false_activation_pct": pct,
        "permitted_query_family": by_family,
        "mixed_query_n": mn,
        "mixed_query_correct_forbidden_activation_n": mixed_correct,
        "mixed_query_correct_forbidden_activation_pct": (100.0 * mixed_correct / mn) if mn else None,
        "mixed_query_permitted_companion_false_activation_n": mixed_companion_error,
        "mixed_query_permitted_companion_false_activation_pct": (100.0 * mixed_companion_error / mn) if mn else None,
        "mixed_query_joint_success_n": mixed_success,
        "mixed_query_joint_success_pct": (100.0 * mixed_success / mn) if mn else None,
    }


def policy_report(
    rows: Sequence[Row],
    logits: torch.Tensor,
    views: Sequence[RoutingView],
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    d = decision_tensors(rows, logits, views, eta, classes, none_idx, bank)
    forb_idx = torch.where(d["forbidden"])[0]
    perm_idx = torch.where(~d["forbidden"])[0]
    cp_idx = torch.where((~d["forbidden"]) & d["candidate"])[0]

    fam = {}
    for f in sorted({base.bucket(r.kind) for r in rows if not r.forbidden}):
        ids = torch.tensor([i for i, r in enumerate(rows) if (not r.forbidden) and base.bucket(r.kind) == f], dtype=torch.long)
        fam[f] = {
            "n": len(ids),
            "candidate_present_n": int(d["candidate"][ids].sum()) if len(ids) else 0,
            "false_activation_pct": rate_pct(d["activates"][ids]) if len(ids) else None,
        }

    unsupported = torch.tensor([not v.scope_supported for v in views], dtype=torch.bool)
    selection_counts: dict[str, int] = defaultdict(int)
    for v in views:
        selection_counts[v.selection_status] += 1

    out = {
        "n": len(rows),
        "relation_accuracy_pct": rate_pct(d["correct"]),
        "forbidden_n": int(d["forbidden"].sum()),
        "permitted_n": int((~d["forbidden"]).sum()),
        "correct_forbidden_binding_accept_pct": rate_pct((d["activates"] & d["correct"])[forb_idx]) if len(forb_idx) else None,
        "wrong_forbidden_binding_accept_pct": rate_pct((d["activates"] & ~d["correct"])[forb_idx]) if len(forb_idx) else None,
        "permitted_false_activation_pct": rate_pct(d["activates"][perm_idx]) if len(perm_idx) else None,
        "candidate_present_permitted_n": len(cp_idx),
        "candidate_present_permitted_false_activation_pct": rate_pct(d["activates"][cp_idx]) if len(cp_idx) else None,
        "permitted_negative_families": fam,
        "unsupported_route_n": int(unsupported.sum()),
        "unsupported_route_pct": rate_pct(unsupported),
        "selection_status_counts": dict(selection_counts),
        "whole_query": whole_query_report(rows, d),
    }
    return out


def calibrate(
    rows: Sequence[Row],
    logits: torch.Tensor,
    views: Sequence[RoutingView],
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
    eps: float,
    eps_wrong: float,
    min_accept: float,
) -> tuple[float, dict[str, Any]]:
    _, margins = base.margin(logits)
    candidates = sorted(set([0.0] + [float(x) for x in margins.tolist()]))
    candidates.append(max(candidates) + 1.0)
    best = None
    for eta in candidates:
        r = policy_report(rows, logits, views, eta, classes, none_idx, bank)
        wrong = (r["wrong_forbidden_binding_accept_pct"] or 0.0) / 100.0
        route = (r["permitted_false_activation_pct"] or 0.0) / 100.0
        cp = (r["candidate_present_permitted_false_activation_pct"] or 0.0) / 100.0
        route_fam = {
            k: (v["false_activation_pct"] or 0.0) / 100.0
            for k, v in r["permitted_negative_families"].items()
        }
        wq = r["whole_query"]
        query = (wq["permitted_query_false_activation_pct"] or 0.0) / 100.0
        query_fam = {
            k: (v["false_activation_pct"] or 0.0) / 100.0
            for k, v in wq["permitted_query_family"].items()
        }
        mixed_companion = (wq["mixed_query_permitted_companion_false_activation_pct"] or 0.0) / 100.0
        if (
            wrong > eps_wrong + 1e-12
            or route > eps + 1e-12
            or cp > eps + 1e-12
            or query > eps + 1e-12
            or mixed_companion > eps + 1e-12
            or any(v > eps + 1e-12 for v in route_fam.values())
            or any(v > eps + 1e-12 for v in query_fam.values())
        ):
            continue
        correct = (r["correct_forbidden_binding_accept_pct"] or 0.0) / 100.0
        mixed_correct = (wq["mixed_query_correct_forbidden_activation_pct"] or 0.0) / 100.0
        key = (-correct, -mixed_correct, max(query_fam.values(), default=0.0), query, cp, route, wrong, -eta)
        item = {
            "eta": float(eta),
            "correct_forbidden_binding_accept_rate": correct,
            "mixed_query_correct_forbidden_activation_rate": mixed_correct,
            "wrong_forbidden_binding_accept_rate": wrong,
            "route_permitted_false_activation_rate": route,
            "candidate_present_permitted_false_activation_rate": cp,
            "route_family_false_activation_rates": route_fam,
            "query_permitted_false_activation_rate": query,
            "query_family_false_activation_rates": query_fam,
            "mixed_query_permitted_companion_false_activation_rate": mixed_companion,
        }
        if best is None or key < best[0]:
            best = (key, item)
    if best is None:
        raise RuntimeError("no target-local calibration threshold satisfies preservation constraints")
    result = best[1]
    result["status"] = (
        "ACCEPTABLE_OPERATING_POINT"
        if result["correct_forbidden_binding_accept_rate"] >= float(min_accept)
        else "NO_ACCEPTABLE_OPERATING_POINT"
    )
    result["selection_rule"] = (
        "maximize correct forbidden-binding acceptance under route, candidate-present, per-family, "
        "whole-query, query-family, and mixed-companion preservation budgets"
    )
    return float(result["eta"]), result


def exact_fit_overlap(texts: Sequence[str], fit_texts: set[str]) -> dict[str, Any]:
    flags = [norm_text(t).casefold() in fit_texts for t in texts]
    n = len(flags)
    hit = sum(flags)
    return {"n": n, "overlap_n": hit, "overlap_pct": (100.0 * hit / n) if n else None}


def route_records(
    split: str,
    group: str,
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
    fit_texts: set[str],
) -> list[dict[str, Any]]:
    d = decision_tensors(rows, logits, views, eta, classes, none_idx, bank)
    records = []
    for i, (row, view) in enumerate(zip(rows, views)):
        records.append({
            "split": split,
            "group": group,
            "case_id": row.case_id,
            "original_query": row.text,
            "designated_subject": row.subject,
            "enumerated_subjects": list(view.enumerated_subjects),
            "selected_text": view.selected_text,
            "selected_character_offsets": list(view.selected_character_offsets) if view.selected_character_offsets else None,
            "selection_status": view.selection_status,
            "scope_supported": view.scope_supported,
            "expected_relation": row.relation,
            "forbidden": row.forbidden,
            "candidate_present": row.candidate,
            "predicted_relation": d["labels"][i],
            "margin": float(d["margin"][i]),
            "relation_correct": bool(d["correct"][i]),
            "accepted_relation": bool(d["accepted_relation"][i]),
            "forbidden_bank_lookup": bool(d["binding"][i]),
            "activates": bool(d["activates"][i]),
            "exact_selected_text_fit_overlap": norm_text(view.selected_text).casefold() in fit_texts,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5f-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--epsilon-retain", type=float, default=0.02)
    ap.add_argument("--epsilon-wrong", type=float, default=0.02)
    ap.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    ap.add_argument("--min-validation-relation-accuracy", type=float, default=0.70)
    ap.add_argument("--mixed-queries-per-phase", type=int, default=50)
    a = ap.parse_args()

    src = Path(a.fix5f_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)

    cache = torch.load(src / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    classes = list(cache["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[NONE]
    marked_policy = {
        phase: rows_from_dicts(cache["policy_rows"]["target_marked"][phase])
        for phase in ("fit", "calib", "validation")
    }
    marked_semantic = {
        phase: rows_from_dicts(cache["semantic_rows"]["target_marked"][phase])
        for phase in ("fit", "calib", "validation")
    }
    all_policy = marked_policy["fit"] + marked_policy["calib"] + marked_policy["validation"]
    bank = {(r.subject, r.relation) for r in all_policy if r.forbidden}
    bank_subjects = sorted({s for s, _ in bank}, key=len, reverse=True)
    if not bank:
        raise RuntimeError("forbidden bank reconstructed from saved policy manifest is empty")

    mixed = {
        phase: build_mixed_queries(marked_policy[phase], bank_subjects, a.mixed_queries_per_phase, phase)
        for phase in ("calib", "validation")
    }
    policy_eval = {
        "calib": marked_policy["calib"] + mixed["calib"],
        "validation": marked_policy["validation"] + mixed["validation"],
    }

    # Rebuild development-only official direct/paraphrase rows. They remain passthrough
    # unless the query itself contains an explicitly supported multi-clause structure.
    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, _ = sample_official_mcf_records(data, 50, 1000, 1, strict=True)
    forget = [off.normalize_record(x) for x in forget]
    dr, pr = base.dev_rows(forget)
    dev = {"direct": dr, "paraphrase": pr}

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

    # Infer feature dimension from the saved marked cache and load the existing head.
    input_dim = int(cache["arms"]["target_marked"]["features"].shape[1])
    head = load_head(src / "target_marked_linear_head.pt", input_dim, classes, device)

    pipelines = {
        "full_marked": lambda row: full_marked_view(row.text, row.subject, bank_subjects),
        "target_local": lambda row: routing_view(row.text, row.subject, bank_subjects),
    }

    fit_views_local = [routing_view(r.text, r.subject, bank_subjects) for r in marked_semantic["fit"]]
    fit_texts_local = {norm_text(v.selected_text).casefold() for v in fit_views_local}
    fit_views_full = [full_marked_view(r.text, r.subject, bank_subjects) for r in marked_semantic["fit"]]
    fit_texts_full = {norm_text(v.selected_text).casefold() for v in fit_views_full}

    results: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for name, view_fn in pipelines.items():
        fit_texts = fit_texts_local if name == "target_local" else fit_texts_full
        sem_logits: dict[str, torch.Tensor] = {}
        sem_views: dict[str, list[RoutingView]] = {}
        for phase in ("fit", "calib", "validation"):
            views = [view_fn(r) for r in marked_semantic[phase]]
            sem_views[phase] = views
            sem_logits[phase] = score_views(marked_semantic[phase], views, model, tok, head, device, a.encode_batch_size)

        policy_logits: dict[str, torch.Tensor] = {}
        policy_views: dict[str, list[RoutingView]] = {}
        for phase in ("calib", "validation"):
            views = [view_fn(r) for r in policy_eval[phase]]
            policy_views[phase] = views
            policy_logits[phase] = score_views(policy_eval[phase], views, model, tok, head, device, a.encode_batch_size)

        eta, cal = calibrate(
            policy_eval["calib"],
            policy_logits["calib"],
            policy_views["calib"],
            classes,
            none_idx,
            bank,
            a.epsilon_retain,
            a.epsilon_wrong,
            a.min_calib_correct_accept,
        )
        vp = policy_report(
            policy_eval["validation"],
            policy_logits["validation"],
            policy_views["validation"],
            eta,
            classes,
            none_idx,
            bank,
        )
        sem = {
            phase: raw_semantic_report(marked_semantic[phase], sem_logits[phase], classes)
            for phase in ("fit", "calib", "validation")
        }

        dev_results = {}
        for group in ("direct", "paraphrase"):
            rows = dev[group]
            views = [view_fn(r) for r in rows]
            logits = score_views(rows, views, model, tok, head, device, a.encode_batch_size)
            dev_results[group] = {
                "semantic": raw_semantic_report(rows, logits, classes),
                "policy": policy_report(rows, logits, views, eta, classes, none_idx, bank),
                "selected_text_fit_overlap": exact_fit_overlap([v.selected_text for v in views], fit_texts),
            }
            all_records.extend(route_records(
                "development_only_seed1", group, rows, views, logits, eta, classes, none_idx, bank, fit_texts
            ))

        fam_ok = all(
            (v["false_activation_pct"] or 0.0) <= 100 * a.epsilon_retain + 1e-9
            for v in vp["permitted_negative_families"].values()
        )
        qfam_ok = all(
            (v["false_activation_pct"] or 0.0) <= 100 * a.epsilon_retain + 1e-9
            for v in vp["whole_query"]["permitted_query_family"].values()
        )
        mixed_companion = vp["whole_query"]["mixed_query_permitted_companion_false_activation_pct"] or 0.0
        pilot = (
            cal["status"] == "ACCEPTABLE_OPERATING_POINT"
            and (sem["validation"]["accuracy_pct"] or 0.0) >= 100 * a.min_validation_relation_accuracy
            and (vp["correct_forbidden_binding_accept_pct"] or 0.0) >= 100 * a.min_calib_correct_accept
            and (vp["wrong_forbidden_binding_accept_pct"] or 0.0) <= 100 * a.epsilon_wrong + 1e-9
            and (vp["permitted_false_activation_pct"] or 0.0) <= 100 * a.epsilon_retain + 1e-9
            and (vp["candidate_present_permitted_false_activation_pct"] or 0.0) <= 100 * a.epsilon_retain + 1e-9
            and (vp["whole_query"]["permitted_query_false_activation_pct"] or 0.0) <= 100 * a.epsilon_retain + 1e-9
            and mixed_companion <= 100 * a.epsilon_retain + 1e-9
            and fam_ok and qfam_ok
        )

        results[name] = {
            "eta": eta,
            "calibration": cal,
            "semantic": sem,
            "validation_policy": vp,
            "development_only_official_seed1": dev_results,
            "pilot_pass": bool(pilot),
            "fit_selected_text_unique_n": len(fit_texts),
            "validation_selected_text_fit_overlap": exact_fit_overlap(
                [v.selected_text for v in sem_views["validation"]], fit_texts
            ),
            "policy_validation_selected_text_fit_overlap": exact_fit_overlap(
                [v.selected_text for v in policy_views["validation"]], fit_texts
            ),
        }
        all_records.extend(route_records(
            "validation", "semantic", marked_semantic["validation"], sem_views["validation"],
            sem_logits["validation"], eta, classes, none_idx, bank, fit_texts
        ))
        all_records.extend(route_records(
            "validation", "policy_plus_mixed", policy_eval["validation"], policy_views["validation"],
            policy_logits["validation"], eta, classes, none_idx, bank, fit_texts
        ))

    comparison = {
        "validation_relation_accuracy_delta_local_minus_full":
            (results["target_local"]["semantic"]["validation"]["accuracy_pct"] or 0.0)
            - (results["full_marked"]["semantic"]["validation"]["accuracy_pct"] or 0.0),
        "validation_correct_forbidden_accept_delta_local_minus_full":
            (results["target_local"]["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0)
            - (results["full_marked"]["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0),
        "validation_query_fpr": {
            name: results[name]["validation_policy"]["whole_query"]["permitted_query_false_activation_pct"]
            for name in pipelines
        },
        "mixed_query_joint_success_pct": {
            name: results[name]["validation_policy"]["whole_query"]["mixed_query_joint_success_pct"]
            for name in pipelines
        },
        "official_para_accuracy_pct": {
            name: results[name]["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"]
            for name in pipelines
        },
        "official_para_correct_forbidden_accept_pct": {
            name: results[name]["development_only_official_seed1"]["paraphrase"]["policy"]["correct_forbidden_binding_accept_pct"]
            for name in pipelines
        },
    }

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5i_frozen_marked_head_target_local_recognition_baseline",
        "recognition_only": True,
        "head_retrained": False,
        "base_model_frozen": True,
        "output_correction_enabled": False,
        "quotient_enabled": False,
        "selector_contract": {
            "subject_candidates_from_query_and_registered_bank": True,
            "gold_relation_used_for_selection": False,
            "gold_answer_used_for_selection": False,
            "explicit_first_second_clause_selection": True,
            "atomic_single_subject_passthrough": True,
            "unsupported_scope_is_conservatively_ineligible_for_activation": True,
            "language_model_answering_input_rewritten": False,
        },
        "mixed_query_contract": {
            "per_phase_limit": int(a.mixed_queries_per_phase),
            "calibration_route_n": len(mixed["calib"]),
            "validation_route_n": len(mixed["validation"]),
            "each_query_has_one_forbidden_and_one_permitted_companion": True,
            "official_paraphrases_used_to_construct_mixed_queries": False,
        },
        "results": results,
        "comparison": comparison,
        "pilot_criteria": {
            "validation_relation_accuracy_min_pct": 100 * a.min_validation_relation_accuracy,
            "correct_forbidden_accept_min_pct": 100 * a.min_calib_correct_accept,
            "wrong_forbidden_accept_max_pct": 100 * a.epsilon_wrong,
            "route_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "candidate_present_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_route_family_max_pct": 100 * a.epsilon_retain,
            "whole_query_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_whole_query_family_max_pct": 100 * a.epsilon_retain,
            "mixed_query_permitted_companion_false_activation_max_pct": 100 * a.epsilon_retain,
        },
        "interpretation_guardrail": (
            "A target-local gain supports scope selection as useful for this frozen recognizer. It does not establish end-to-end unlearning, Gen leakage reduction, utility preservation, or architectural novelty."
        ),
    }
    report_path = out / "target_local_recognition_baseline_fix5i.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    records_path = out / "target_local_route_records_fix5i.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    compact = {
        name: {
            "eta": results[name]["eta"],
            "calibration_status": results[name]["calibration"]["status"],
            "validation_relation_accuracy_pct": results[name]["semantic"]["validation"]["accuracy_pct"],
            "validation_correct_forbidden_accept_pct": results[name]["validation_policy"]["correct_forbidden_binding_accept_pct"],
            "validation_route_permitted_fpr_pct": results[name]["validation_policy"]["permitted_false_activation_pct"],
            "validation_query_permitted_fpr_pct": results[name]["validation_policy"]["whole_query"]["permitted_query_false_activation_pct"],
            "validation_mixed_correct_forbidden_pct": results[name]["validation_policy"]["whole_query"]["mixed_query_correct_forbidden_activation_pct"],
            "validation_mixed_companion_fpr_pct": results[name]["validation_policy"]["whole_query"]["mixed_query_permitted_companion_false_activation_pct"],
            "validation_mixed_joint_success_pct": results[name]["validation_policy"]["whole_query"]["mixed_query_joint_success_pct"],
            "unsupported_route_pct": results[name]["validation_policy"]["unsupported_route_pct"],
            "official_para_accuracy_pct": results[name]["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"],
            "official_para_correct_forbidden_accept_pct": results[name]["development_only_official_seed1"]["paraphrase"]["policy"]["correct_forbidden_binding_accept_pct"],
            "pilot_pass": results[name]["pilot_pass"],
        }
        for name in pipelines
    }
    compact["comparison"] = comparison
    compact["report"] = str(report_path)
    compact["route_records"] = str(records_path)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
