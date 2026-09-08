#!/usr/bin/env python3
"""Semantic cross-encoder rescue router for selective factual suppression (Seed 1).

This is a single stronger router family rather than another template-augmentation
variant.  The already validated Fix5o exact-name target-local router remains the
high-precision primary route.  A frozen-Llama semantic verifier is allowed to RESCUE
only routes that Fix5o did not activate.

For every registered subject candidate, the verifier compares the target-local query
against every modeled relation using the relation's authored label, definition and
natural relation-specific examples.  The subject is replaced by TARGET_ENTITY inside
the verifier prompt so relation scoring cannot memorize entity identity.  A candidate
relation receives a yes-vs-no score.  Semantic confidence is

    min(top yes-minus-no score, top-vs-runner-up gap).

A single rescue threshold is calibrated WITHOUT official MCF paraphrases.  The
threshold maximizes semantic-calibration coverage while preserving the same <=2%
route/query/family/mixed-companion false-activation budgets.  Fix5o activations are
never removed by the rescue path.

If the frozen validation preservation gate passes, the script evaluates the hybrid
router end-to-end on the exact 50 direct + 100 paraphrase atomic query bank already
saved by Fix5p.  It reuses the frozen Fix5l sensitive-token supports, fixed -12
penalty, deterministic generation, and quotient-off contract.  Base and Fix5o
conditions are read from the strictly reproduced Fix5p records; only the hybrid
condition performs new generation/inference.

This measures behavioral suppression, not knowledge deletion.  Canonical generated
answer matching does not measure aliases or semantic disclosures.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_augmented_relation_router_fix5o_seed1 as fix5o
import mcf_target_local_fixed_penalty_integration_fix5l_seed1 as fix5l
import mcf_target_local_generation_mixed_eval_fix5m_seed1 as fix5m
import mcf_fix5o_matched_end_to_end_fix5p_seed1 as fix5p

local = fix5o.local
base = fix5o.base
Row = fix5o.Row
RoutingView = fix5o.RoutingView
NONE = base.NONE
SEED = 1

# A-priori semantic neighborhoods from the modeled relation ontology.  These are not
# derived from official paraphrase errors.  They tell the verifier what distinctions
# matter without supplying any query-specific answer.
RELATION_GROUPS = (
    ("P103", "P1412", "P37", "P364"),                    # language semantics
    ("P106", "P101", "P39", "P937"),                    # work/role semantics
    ("P30", "P276", "P19", "P740", "P36", "P495"),   # location/origin semantics
    ("P641", "P413", "P1303"),                           # sport/instrument semantics
    ("P176", "P495"),                                     # manufacture/origin
    ("P463", "P138"),                                     # membership vs naming
)


@dataclass(frozen=True)
class RelationProfile:
    relation: str
    label: str
    meaning: str
    examples: tuple[str, ...]
    confusable_labels: tuple[str, ...]


@dataclass(frozen=True)
class SemanticDecision:
    relation: str
    top_score: float
    runner_up_score: float
    gap: float
    confidence: float


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
                raise ValueError(f"{path}:{line_no}: expected object")
            out.append(value)
    return out


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def relation_neighbors(relation: str, labels: Mapping[str, str]) -> tuple[str, ...]:
    out: list[str] = []
    for group in RELATION_GROUPS:
        if relation not in group:
            continue
        for rid in group:
            if rid != relation and rid in labels and labels[rid] not in out:
                out.append(labels[rid])
    return tuple(out)


def collect_examples(contract: Mapping[str, Any], limit: int = 6) -> tuple[str, ...]:
    preferred = (
        "wh_question", "possessive_question", "alternative_question",
        "imperative_identify", "nominalized_question", "conversational_question",
    )
    examples: list[str] = []

    def add_families(families: Mapping[str, Any]) -> None:
        for family in preferred:
            values = families.get(family, [])
            if not isinstance(values, list):
                continue
            for value in values:
                if not isinstance(value, str):
                    continue
                rendered = norm_text(value.replace("{}", "TARGET_ENTITY"))
                if rendered and rendered not in examples:
                    examples.append(rendered)
                if len(examples) >= limit:
                    return

    fams = contract.get("families", {})
    if isinstance(fams, Mapping):
        add_families(fams)
    variants = contract.get("variants", {})
    if isinstance(variants, Mapping):
        for variant in variants.values():
            if len(examples) >= limit:
                break
            if isinstance(variant, Mapping) and isinstance(variant.get("families"), Mapping):
                add_families(variant["families"])
    return tuple(examples[:limit])


def load_profiles(path: Path, modeled: Sequence[str]) -> list[RelationProfile]:
    payload = load_json(path)
    relations = payload.get("relations", {})
    if not isinstance(relations, Mapping):
        raise ValueError("relation contract lacks relations mapping")
    labels: dict[str, str] = {}
    meanings: dict[str, str] = {}
    for rid in modeled:
        c = relations.get(rid)
        if not isinstance(c, Mapping):
            raise ValueError(f"missing relation contract for {rid}")
        label = str(c.get("label", "")).strip()
        meaning = str(c.get("meaning", "")).strip()
        if not label or not meaning:
            raise ValueError(f"relation {rid} lacks label/meaning")
        labels[rid] = label
        meanings[rid] = meaning
    profiles = []
    for rid in modeled:
        c = relations[rid]
        examples = collect_examples(c)
        if not examples:
            raise ValueError(f"relation {rid} has no usable authored examples")
        profiles.append(RelationProfile(
            relation=rid,
            label=labels[rid],
            meaning=meanings[rid],
            examples=examples,
            confusable_labels=relation_neighbors(rid, labels),
        ))
    return profiles


def semantic_query_text(view: RoutingView, subject: str) -> str:
    exact = f"[TARGET]{subject}[/TARGET]"
    text = str(view.selected_text)
    if exact in text:
        return norm_text(text.replace(exact, "TARGET_ENTITY", 1))
    # Fail-safe normalization for any legacy marker formatting.
    pat = re.compile(r"\[TARGET\].*?\[/TARGET\]", flags=re.I | re.S)
    if pat.search(text):
        return norm_text(pat.sub("TARGET_ENTITY", text, count=1))
    return norm_text(text)


def verifier_prompt(query: str, profile: RelationProfile, tok: Any) -> str:
    examples = "\n".join(f"- {x}" for x in profile.examples)
    confusions = (
        ", ".join(profile.confusable_labels)
        if profile.confusable_labels else "none specifically listed"
    )
    user = (
        "You are a factual-relation verifier. Decide whether the REQUEST asks for "
        "the exact CANDIDATE RELATION about TARGET_ENTITY. Judge the semantic predicate, "
        "not entity identity or keyword overlap. A related but different relation is No.\n\n"
        f"REQUEST:\n{query}\n\n"
        f"CANDIDATE RELATION: {profile.label}\n"
        f"DEFINITION: {profile.meaning}\n"
        f"EXAMPLE REQUESTS FOR THIS RELATION:\n{examples}\n"
        f"POTENTIALLY CONFUSABLE RELATIONS: {confusions}\n\n"
        "Does the REQUEST ask for this exact CANDIDATE RELATION? Answer only Yes or No."
    )
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user + "\nAnswer:"


def yes_no_token_ids(tok: Any) -> tuple[int, int, str]:
    for prefix in ("", " "):
        yes = tok(prefix + "Yes", add_special_tokens=False)["input_ids"]
        no = tok(prefix + "No", add_special_tokens=False)["input_ids"]
        if len(yes) == 1 and len(no) == 1 and int(yes[0]) != int(no[0]):
            return int(yes[0]), int(no[0]), prefix
    raise RuntimeError("semantic verifier requires a common single-token Yes/No spelling")


@torch.no_grad()
def score_semantic_texts(
    model: Any,
    tok: Any,
    texts: Sequence[str],
    profiles: Sequence[RelationProfile],
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    unique: list[str] = []
    lookup: dict[str, int] = {}
    inv: list[int] = []
    for text in texts:
        key = str(text)
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(key)
        inv.append(lookup[key])

    yes_id, no_id, option_prefix = yes_no_token_ids(tok)
    pairs: list[tuple[int, int, str]] = []
    for ti, text in enumerate(unique):
        for ri, profile in enumerate(profiles):
            pairs.append((ti, ri, verifier_prompt(text, profile, tok)))

    matrix = torch.empty((len(unique), len(profiles)), dtype=torch.float32)
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        total = len(pairs)
        for st in range(0, total, int(batch_size)):
            chunk = pairs[st:st + int(batch_size)]
            enc = tok(
                [x[2] for x in chunk], padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            logits = model(**enc, use_cache=False, return_dict=True).logits.float()
            last = enc["attention_mask"].sum(1).long() - 1
            row_ids = torch.arange(len(chunk), device=device)
            next_logits = logits[row_ids, last]
            scores = next_logits[:, yes_id] - next_logits[:, no_id]
            for j, (ti, ri, _) in enumerate(chunk):
                matrix[ti, ri] = float(scores[j].item())
            done = min(st + len(chunk), total)
            if st == 0 or done == total or done % max(1, int(batch_size) * 20) == 0:
                print(f"[semantic-rescue] verifier pairs: {done}/{total}", flush=True)
    finally:
        tok.padding_side = old_side

    expanded = matrix[torch.tensor(inv, dtype=torch.long)]
    return expanded, {
        "unique_query_text_n": len(unique),
        "relation_n": len(profiles),
        "verifier_pair_n": len(pairs),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "option_prefix": option_prefix,
    }


def semantic_decisions(scores: torch.Tensor, profiles: Sequence[RelationProfile]) -> list[SemanticDecision]:
    if scores.ndim != 2 or scores.shape[1] != len(profiles):
        raise ValueError("semantic score shape mismatch")
    vals, ids = torch.topk(scores, k=2, dim=1)
    out = []
    for i in range(scores.shape[0]):
        top = float(vals[i, 0].item())
        runner = float(vals[i, 1].item())
        gap = top - runner
        confidence = min(top, gap)
        out.append(SemanticDecision(
            relation=profiles[int(ids[i, 0].item())].relation,
            top_score=top,
            runner_up_score=runner,
            gap=gap,
            confidence=confidence,
        ))
    return out


def fix5o_row_decisions(
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    bank: set[tuple[str, str]],
    device: torch.device,
    encode_batch_size: int,
) -> list[dict[str, Any]]:
    features, inv, _ = local.encode_unique(
        model, tok, [v.selected_text for v in views], device, encode_batch_size
    )
    with torch.no_grad():
        unique_logits = head(features.to(device)).cpu()
    logits = unique_logits[torch.tensor(inv, dtype=torch.long)]
    none_idx = list(classes).index(NONE)
    d = local.decision_tensors(rows, logits, views, eta, classes, none_idx, bank)
    out = []
    for i, row in enumerate(rows):
        relation = str(d["labels"][i])
        out.append({
            "relation": relation,
            "relation_correct": bool(d["correct"][i]),
            "accepted_relation": bool(d["accepted_relation"][i]),
            "activates": bool(d["activates"][i]),
            "binding": (row.subject, relation),
            "margin": float(d["margin"][i]),
        })
    return out


def active_relations_for_row(
    row: Row,
    view: RoutingView,
    fix: Mapping[str, Any],
    sem: SemanticDecision,
    eta_semantic: float,
    bank: set[tuple[str, str]],
) -> set[str]:
    active: set[str] = set()
    if bool(fix["activates"]):
        active.add(str(fix["relation"]))
    # Rescue only when Fix5o did not already activate this candidate subject.
    if (
        not active
        and view.scope_supported
        and sem.confidence >= float(eta_semantic)
        and (row.subject, sem.relation) in bank
    ):
        active.add(sem.relation)
    return active


def policy_metrics(
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    fix_decisions: Sequence[Mapping[str, Any]],
    sem_decisions: Sequence[SemanticDecision],
    eta_semantic: float,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    if not (len(rows) == len(views) == len(fix_decisions) == len(sem_decisions)):
        raise ValueError("policy metric length mismatch")
    per_row: list[dict[str, Any]] = []
    for row, view, fix, sem in zip(rows, views, fix_decisions, sem_decisions):
        active_rel = active_relations_for_row(row, view, fix, sem, eta_semantic, bank)
        correct = bool(row.forbidden and row.relation in active_rel)
        wrong = bool(row.forbidden and any(r != row.relation for r in active_rel))
        false_activation = bool((not row.forbidden) and active_rel)
        per_row.append({
            "row": row,
            "active_relations": active_rel,
            "correct": correct,
            "wrong": wrong,
            "false_activation": false_activation,
        })

    forbidden = [x for x in per_row if x["row"].forbidden]
    permitted = [x for x in per_row if not x["row"].forbidden]
    candidate_permitted = [x for x in permitted if bool(x["row"].candidate)]

    fam: dict[str, dict[str, Any]] = {}
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in permitted:
        by_family[base.bucket(x["row"].kind)].append(x)
    for name, xs in sorted(by_family.items()):
        fam[name] = {
            "n": len(xs),
            "false_activation_n": sum(x["false_activation"] for x in xs),
            "false_activation_pct": 100.0 * sum(x["false_activation"] for x in xs) / len(xs),
        }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for x in per_row:
        grouped[norm_text(x["row"].text)].append(x)
    permitted_queries = [xs for xs in grouped.values() if not any(x["row"].forbidden for x in xs)]
    permitted_query_false = sum(any(x["false_activation"] for x in xs) for xs in permitted_queries)
    mixed_groups = [
        xs for xs in grouped.values()
        if any(base.bucket(x["row"].kind) == "mixed_forbidden_distractor" for x in xs)
        and any(x["row"].forbidden for x in xs)
        and any(not x["row"].forbidden for x in xs)
    ]
    mixed_companion_false = sum(
        any(x["false_activation"] for x in xs if not x["row"].forbidden)
        for xs in mixed_groups
    )
    mixed_correct = sum(
        any(x["correct"] for x in xs if x["row"].forbidden)
        for xs in mixed_groups
    )

    def pct(num: int, den: int) -> float | None:
        return 100.0 * num / den if den else None

    return {
        "n": len(rows),
        "forbidden_n": len(forbidden),
        "correct_forbidden_binding_accept_n": sum(x["correct"] for x in forbidden),
        "correct_forbidden_binding_accept_pct": pct(sum(x["correct"] for x in forbidden), len(forbidden)),
        "wrong_forbidden_binding_accept_n": sum(x["wrong"] for x in forbidden),
        "wrong_forbidden_binding_accept_pct": pct(sum(x["wrong"] for x in forbidden), len(forbidden)),
        "permitted_n": len(permitted),
        "permitted_false_activation_n": sum(x["false_activation"] for x in permitted),
        "permitted_false_activation_pct": pct(sum(x["false_activation"] for x in permitted), len(permitted)),
        "candidate_present_permitted_n": len(candidate_permitted),
        "candidate_present_permitted_false_activation_n": sum(x["false_activation"] for x in candidate_permitted),
        "candidate_present_permitted_false_activation_pct": pct(sum(x["false_activation"] for x in candidate_permitted), len(candidate_permitted)),
        "permitted_negative_families": fam,
        "permitted_query_n": len(permitted_queries),
        "permitted_query_false_activation_n": permitted_query_false,
        "permitted_query_false_activation_pct": pct(permitted_query_false, len(permitted_queries)),
        "mixed_query_n": len(mixed_groups),
        "mixed_query_correct_forbidden_activation_n": mixed_correct,
        "mixed_query_correct_forbidden_activation_pct": pct(mixed_correct, len(mixed_groups)),
        "mixed_query_permitted_companion_false_activation_n": mixed_companion_false,
        "mixed_query_permitted_companion_false_activation_pct": pct(mixed_companion_false, len(mixed_groups)),
    }


def preservation_ok(metrics: Mapping[str, Any], epsilon: float = 0.02) -> bool:
    limit = 100.0 * float(epsilon) + 1e-9
    scalars = (
        metrics.get("wrong_forbidden_binding_accept_pct"),
        metrics.get("permitted_false_activation_pct"),
        metrics.get("candidate_present_permitted_false_activation_pct"),
        metrics.get("permitted_query_false_activation_pct"),
        metrics.get("mixed_query_permitted_companion_false_activation_pct"),
    )
    if any(x is not None and float(x) > limit for x in scalars):
        return False
    for item in metrics.get("permitted_negative_families", {}).values():
        x = item.get("false_activation_pct")
        if x is not None and float(x) > limit:
            return False
    return True


def semantic_accuracy(rows: Sequence[Row], decisions: Sequence[SemanticDecision]) -> dict[str, Any]:
    if len(rows) != len(decisions):
        raise ValueError("semantic accuracy length mismatch")
    correct = sum(r.relation == d.relation for r, d in zip(rows, decisions))
    return {"n": len(rows), "correct_n": correct, "accuracy_pct": 100.0 * correct / len(rows) if rows else None}


def semantic_correct_accept_rate(rows: Sequence[Row], decisions: Sequence[SemanticDecision], eta: float) -> float:
    if not rows:
        return 0.0
    good = sum(r.relation == d.relation and d.confidence >= eta for r, d in zip(rows, decisions))
    return good / len(rows)


def calibrate_rescue_eta(
    semantic_calib_rows: Sequence[Row],
    semantic_calib_decisions: Sequence[SemanticDecision],
    policy_rows: Sequence[Row],
    policy_views: Sequence[RoutingView],
    fix_decisions: Sequence[Mapping[str, Any]],
    policy_sem_decisions: Sequence[SemanticDecision],
    bank: set[tuple[str, str]],
    epsilon: float,
) -> tuple[float, dict[str, Any]]:
    positive = sorted({
        float(d.confidence) for d in list(semantic_calib_decisions) + list(policy_sem_decisions)
        if d.confidence >= 0.0 and math.isfinite(d.confidence)
    }, reverse=True)
    candidates = [math.inf] + positive + [0.0]
    best: tuple[Any, ...] | None = None
    best_payload: dict[str, Any] | None = None
    for eta in candidates:
        pm = policy_metrics(policy_rows, policy_views, fix_decisions, policy_sem_decisions, eta, bank)
        if not preservation_ok(pm, epsilon):
            continue
        sem_rate = semantic_correct_accept_rate(semantic_calib_rows, semantic_calib_decisions, eta)
        forbidden_rate = (pm.get("correct_forbidden_binding_accept_pct") or 0.0) / 100.0
        rescue_count = sum(
            (not bool(f["activates"]))
            and v.scope_supported
            and d.confidence >= eta
            and (r.subject, d.relation) in bank
            for r, v, f, d in zip(policy_rows, policy_views, fix_decisions, policy_sem_decisions)
        )
        key = (sem_rate, forbidden_rate, rescue_count, float(eta) if math.isfinite(eta) else 1e30)
        if best is None or key > best:
            best = key
            best_payload = {
                "eta": eta,
                "semantic_calibration_correct_accept_rate": sem_rate,
                "semantic_calibration_correct_accept_pct": 100.0 * sem_rate,
                "policy": pm,
                "policy_rescue_row_n": rescue_count,
                "selection_rule": "maximize semantic-calibration correct acceptance under unchanged preservation budgets; tie-break forbidden coverage, rescue count, higher threshold",
            }
    if best_payload is None:
        raise RuntimeError("NO_ACCEPTABLE_SEMANTIC_RESCUE_OPERATING_POINT")
    return float(best_payload["eta"]), best_payload


def relation_view_from_route_record(record: Mapping[str, Any], bank_subjects: Sequence[str]) -> tuple[Row, RoutingView]:
    row = Row(
        text=str(record["original_query"]),
        subject=str(record["designated_subject"]),
        relation=str(record["expected_relation"]),
        forbidden=bool(record.get("forbidden", True)),
        kind=str(record.get("group", "development")),
        family=str(record.get("group", "development")),
        case_id=record.get("case_id"),
        candidate=True,
    )
    return row, fix5o.exact_view(row, bank_subjects)


def build_semantic_route(
    query: str,
    fix5o_route: Mapping[str, Any],
    semantic_lookup: Mapping[str, SemanticDecision],
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    eta_semantic: float,
) -> fix5l.RouteDecision:
    bank = set(support_map)
    active = {tuple(x) for x in fix5o_route.get("active_bindings", [])}
    routes = [dict(x) for x in fix5o_route.get("routes", [])]
    for route in routes:
        subject = str(route["subject"])
        if any(b[0] == subject for b in active):
            route["semantic_rescue_attempted"] = False
            route["semantic_rescue_activates"] = False
            continue
        selected = str(route["selected_text"])
        view = RoutingView(
            selected_text=selected,
            selection_status=str(route.get("selection_status", "")),
            selected_character_offsets=None,
            scope_supported=bool(route.get("scope_supported", False)),
            enumerated_subjects=tuple(str(x) for x in fix5o_route.get("candidates", [])),
        )
        sem_text = semantic_query_text(view, subject)
        dec = semantic_lookup.get(sem_text)
        if dec is None:
            raise KeyError(f"semantic score missing for route text: {sem_text!r}")
        binding = (subject, dec.relation)
        activates = bool(view.scope_supported and dec.confidence >= eta_semantic and binding in bank)
        route.update({
            "semantic_rescue_attempted": True,
            "semantic_relation": dec.relation,
            "semantic_top_score": dec.top_score,
            "semantic_gap": dec.gap,
            "semantic_confidence": dec.confidence,
            "semantic_eta": eta_semantic,
            "semantic_rescue_activates": activates,
        })
        if activates:
            active.add(binding)
    active_sorted = sorted(active)
    token_ids = sorted({tid for b in active_sorted for tid in support_map[b].token_ids})
    return fix5l.RouteDecision(
        query=str(query),
        candidates=tuple(str(x) for x in fix5o_route.get("candidates", [])),
        active_bindings=tuple(active_sorted),
        active_token_ids=tuple(token_ids),
        routes=tuple(routes),
    )


def endpoint_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for cond in ("base", "fix5o_integrated", "hybrid_integrated"):
        t = np.asarray([float(r["scores"][cond]["target_true"]) for r in rows], dtype=np.float64)
        n = np.asarray([float(r["scores"][cond]["target_new"]) for r in rows], dtype=np.float64)
        pref = t < n
        disclosure = [bool(r["generation"][cond]["flags"]["target_true_canonical_mentioned"]) for r in rows]
        out[cond] = {
            "sensitive_preference_n": int(pref.sum()),
            "sensitive_preference_pct": 100.0 * float(pref.mean()),
            "canonical_sensitive_disclosure_n": int(sum(disclosure)),
            "canonical_sensitive_disclosure_pct": 100.0 * sum(disclosure) / len(rows),
            "target_true_nll_mean": float(t.mean()),
            "target_new_nll_mean": float(n.mean()),
        }
    base_disclosing = {i for i, r in enumerate(rows) if r["generation"]["base"]["flags"]["target_true_canonical_mentioned"]}
    hybrid_acc = {i for i, r in enumerate(rows) if r["hybrid_route_cohort"] == "correctly_accepted"}
    fix_acc = {i for i, r in enumerate(rows) if r["fix5o_route_cohort"] == "correctly_accepted"}
    out["routing"] = {
        "fix5o_correctly_accepted_n": len(fix_acc),
        "hybrid_correctly_accepted_n": len(hybrid_acc),
        "gained_correct_accept_n": len(hybrid_acc - fix_acc),
        "lost_correct_accept_n": len(fix_acc - hybrid_acc),
        "base_disclosing_n": len(base_disclosing),
        "fix5o_coverage_of_base_disclosures_pct": 100.0 * len(fix_acc & base_disclosing) / len(base_disclosing) if base_disclosing else None,
        "hybrid_coverage_of_base_disclosures_pct": 100.0 * len(hybrid_acc & base_disclosing) / len(base_disclosing) if base_disclosing else None,
        "newly_covered_base_disclosures_n": len((hybrid_acc - fix_acc) & base_disclosing),
        "lost_coverage_base_disclosures_n": len((fix_acc - hybrid_acc) & base_disclosing),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5f-output-dir", required=True)
    ap.add_argument("--fix5l-output-dir", required=True)
    ap.add_argument("--fix5o-output-dir", required=True)
    ap.add_argument("--fix5p-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--relation-contracts", default=str(SCRIPT_DIR / "mcf_relation_contracts_fix5.json"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--verifier-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--epsilon-retain", type=float, default=0.02)
    ap.add_argument("--mixed-queries-per-phase", type=int, default=50)
    a = ap.parse_args()

    fix5f_dir = Path(a.fix5f_output_dir).resolve()
    fix5l_dir = Path(a.fix5l_output_dir).resolve()
    fix5o_dir = Path(a.fix5o_output_dir).resolve()
    fix5p_dir = Path(a.fix5p_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)

    cache = torch.load(fix5f_dir / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    classes = list(cache["classes"])
    modeled = [c for c in classes if c != NONE]
    semantic = {
        phase: rows_from_dicts(cache["semantic_rows"]["target_marked"][phase])
        for phase in ("calib", "validation")
    }
    policy = {
        phase: rows_from_dicts(cache["policy_rows"]["target_marked"][phase])
        for phase in ("calib", "validation")
    }
    all_policy_raw = []
    for phase in ("fit", "calib", "validation"):
        all_policy_raw.extend(rows_from_dicts(cache["policy_rows"]["target_marked"][phase]))
    bank = {(r.subject, r.relation) for r in all_policy_raw if r.forbidden}
    bank_subjects = sorted({s for s, _ in bank}, key=len, reverse=True)

    mixed = {
        phase: local.build_mixed_queries(policy[phase], bank_subjects, a.mixed_queries_per_phase, phase)
        for phase in ("calib", "validation")
    }
    policy_eval = {phase: policy[phase] + mixed[phase] for phase in ("calib", "validation")}

    fix5o_report = load_json(fix5o_dir / "mcf_target_local_augmented_relation_router_fix5o.json")
    fix5o_result = fix5o_report["results"]["augmented_exact_name"]
    if fix5o_result.get("pilot_pass_preservation_and_original_validation") is not True:
        raise RuntimeError("Fix5o primary router must pass its saved preservation pilot")
    fix5o_eta = float(fix5o_result["eta"])
    head, head_classes = fix5l.load_head(fix5o_dir / "augmented_exact_name_linear_head.pt", device)
    if list(head_classes) != classes:
        raise RuntimeError("Fix5o head class inventory differs from Fix5f cache")

    fix5p_report = load_json(fix5p_dir / "mcf_fix5o_matched_end_to_end_fix5p.json")
    if fix5p_report.get("historical_fix5m_reproduction", {}).get("strict_reproduction_pass") is not True:
        raise RuntimeError("Fix5p source must have strict historical reproduction pass")
    fix5p_records = load_jsonl(fix5p_dir / "mcf_fix5o_matched_end_to_end_records_fix5p.jsonl")
    atomic_records = [r for r in fix5p_records if r.get("kind") == "atomic" and r.get("group") in {"direct", "paraphrase"}]
    if len(atomic_records) != 150:
        raise RuntimeError(f"expected 150 Fix5p atomic records, got {len(atomic_records)}")

    support_map, penalty, support_payload = fix5m.load_frozen_supports(
        fix5l_dir / "frozen_answer_token_support_fix5l.json"
    )
    if abs(float(penalty) - 12.0) > 1e-12:
        raise RuntimeError(f"semantic rescue experiment freezes penalty=12, got {penalty}")

    profiles = load_profiles(Path(a.relation_contracts), modeled)
    profile_relations = [p.relation for p in profiles]
    if profile_relations != modeled:
        raise RuntimeError("relation profile order mismatch")

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

    # Prepare calibration/validation views.
    views = {
        "semantic_calib": [fix5o.exact_view(r, bank_subjects) for r in semantic["calib"]],
        "semantic_validation": [fix5o.exact_view(r, bank_subjects) for r in semantic["validation"]],
        "policy_calib": [fix5o.exact_view(r, bank_subjects) for r in policy_eval["calib"]],
        "policy_validation": [fix5o.exact_view(r, bank_subjects) for r in policy_eval["validation"]],
    }

    # Official dev route rows were already saved by Fix5o; they are evaluation only.
    route_records = load_jsonl(fix5o_dir / "mcf_target_local_augmented_relation_router_records_fix5o.jsonl")
    dev_records = [
        r for r in route_records
        if r.get("arm") == "augmented_exact_name"
        and r.get("group") in {"official_seed1_direct_development_only", "official_seed1_paraphrase_development_only"}
    ]
    dev_rows: list[Row] = []
    dev_views: list[RoutingView] = []
    dev_groups: list[str] = []
    for rec in dev_records:
        row, view = relation_view_from_route_record(rec, bank_subjects)
        dev_rows.append(row)
        dev_views.append(view)
        dev_groups.append(str(rec["group"]))

    # Add all atomic Fix5p route texts so end-to-end routing can reuse the same score cache.
    endpoint_sem_texts: list[str] = []
    for r in atomic_records:
        route = r["routes"]["fix5o"]
        for item in route.get("routes", []):
            view = RoutingView(
                selected_text=str(item["selected_text"]),
                selection_status=str(item.get("selection_status", "")),
                selected_character_offsets=None,
                scope_supported=bool(item.get("scope_supported", False)),
                enumerated_subjects=tuple(str(x) for x in route.get("candidates", [])),
            )
            endpoint_sem_texts.append(semantic_query_text(view, str(item["subject"])))

    all_rows_and_views = [
        (semantic["calib"], views["semantic_calib"]),
        (semantic["validation"], views["semantic_validation"]),
        (policy_eval["calib"], views["policy_calib"]),
        (policy_eval["validation"], views["policy_validation"]),
        (dev_rows, dev_views),
    ]
    semantic_texts: list[str] = []
    slices: list[tuple[int, int]] = []
    for rows0, views0 in all_rows_and_views:
        start = len(semantic_texts)
        semantic_texts.extend(semantic_query_text(v, r.subject) for r, v in zip(rows0, views0))
        slices.append((start, len(semantic_texts)))
    endpoint_start = len(semantic_texts)
    semantic_texts.extend(endpoint_sem_texts)

    sem_scores, verifier_audit = score_semantic_texts(
        model, tok, semantic_texts, profiles, device, a.verifier_batch_size
    )
    sem_all = semantic_decisions(sem_scores, profiles)

    sem_cal = sem_all[slices[0][0]:slices[0][1]]
    sem_val = sem_all[slices[1][0]:slices[1][1]]
    policy_sem_cal = sem_all[slices[2][0]:slices[2][1]]
    policy_sem_val = sem_all[slices[3][0]:slices[3][1]]
    dev_sem = sem_all[slices[4][0]:slices[4][1]]
    endpoint_sem = sem_all[endpoint_start:]

    fix_policy_cal = fix5o_row_decisions(
        policy_eval["calib"], views["policy_calib"], model, tok, head, classes,
        fix5o_eta, bank, device, a.encode_batch_size,
    )
    fix_policy_val = fix5o_row_decisions(
        policy_eval["validation"], views["policy_validation"], model, tok, head, classes,
        fix5o_eta, bank, device, a.encode_batch_size,
    )

    eta_semantic, calibration = calibrate_rescue_eta(
        semantic["calib"], sem_cal,
        policy_eval["calib"], views["policy_calib"], fix_policy_cal, policy_sem_cal,
        bank, a.epsilon_retain,
    )
    validation_policy = policy_metrics(
        policy_eval["validation"], views["policy_validation"], fix_policy_val,
        policy_sem_val, eta_semantic, bank,
    )
    validation_preservation_pass = preservation_ok(validation_policy, a.epsilon_retain)

    # Development recognition report.  Official paraphrases do not influence eta.
    dev_by_group: dict[str, dict[str, Any]] = {}
    for group in sorted(set(dev_groups)):
        ids = [i for i, g in enumerate(dev_groups) if g == group]
        rows0 = [dev_rows[i] for i in ids]
        sem0 = [dev_sem[i] for i in ids]
        raw = semantic_accuracy(rows0, sem0)
        correct_accept = sum(
            rows0[j].relation == sem0[j].relation and sem0[j].confidence >= eta_semantic
            for j in range(len(rows0))
        )
        dev_by_group[group] = {
            "semantic_crossencoder": raw,
            "semantic_correct_accept_n": correct_accept,
            "semantic_correct_accept_pct": 100.0 * correct_accept / len(rows0) if rows0 else None,
        }

    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5o_plus_frozen_semantic_crossencoder_rescue",
        "base_model_frozen": True,
        "fix5o_primary_router_frozen": True,
        "semantic_verifier_trainable_parameters": 0,
        "official_paraphrases_used_for_training": False,
        "official_paraphrases_used_for_calibration": False,
        "official_paraphrases_used_for_threshold_selection": False,
        "subject_identity_removed_inside_semantic_verifier": True,
        "semantic_profile_source": str(Path(a.relation_contracts).resolve()),
        "semantic_profiles": [p.__dict__ for p in profiles],
        "verifier_audit": verifier_audit,
        "fix5o_eta": fix5o_eta,
        "semantic_rescue_eta": eta_semantic,
        "calibration": calibration,
        "semantic_calibration": semantic_accuracy(semantic["calib"], sem_cal),
        "semantic_validation": semantic_accuracy(semantic["validation"], sem_val),
        "validation_policy": validation_policy,
        "validation_preservation_pass": validation_preservation_pass,
        "development_only_official_seed1": dev_by_group,
        "penalty": penalty,
        "quotient_enabled": False,
        "end_to_end_ran": False,
    }

    # End-to-end only after preservation gate.
    endpoint_rows: list[dict[str, Any]] = []
    if validation_preservation_pass:
        semantic_lookup: dict[str, SemanticDecision] = {}
        for text, dec in zip(endpoint_sem_texts, endpoint_sem):
            prev = semantic_lookup.get(text)
            if prev is not None and prev != dec:
                raise RuntimeError("inconsistent semantic decision for duplicate endpoint text")
            semantic_lookup[text] = dec

        for idx, src in enumerate(atomic_records, 1):
            if idx == 1 or idx % 10 == 0 or idx == len(atomic_records):
                print(f"[semantic-rescue] endpoint: {idx}/{len(atomic_records)}", flush=True)
            query = str(src["query"])
            expected = tuple(src["expected_binding"])
            target_true = str(src["target_true"])
            target_new = str(src["target_new"])
            fix_route = src["routes"]["fix5o"]
            hybrid_route = build_semantic_route(
                query, fix_route, semantic_lookup, support_map, eta_semantic
            )

            enc = tok(query, add_special_tokens=True, return_tensors="pt").to(device)
            processor = None
            if hybrid_route.active_token_ids:
                processor = fix5l.FixedTokenPenaltyLogitsProcessor(
                    hybrid_route.active_token_ids, penalty=penalty
                )
            hybrid_gen = fix5m.generate_from_inputs(
                model, tok, enc["input_ids"], enc["attention_mask"],
                a.max_new_tokens, processor=processor,
            )
            hybrid_score = fix5l.score_choice_pair(
                model, tok, query, target_new, target_true,
                hybrid_route.active_token_ids, penalty, device, llama_like,
            )
            base_saved = src["scores"]["base"]
            fix_saved = src["scores"]["fix5o_integrated"]
            # Teacher-forced base must reproduce the saved Fix5p value.
            for key in ("target_true", "target_new"):
                if abs(float(hybrid_score["base"][key]) - float(base_saved[key])) > 1e-7:
                    raise RuntimeError(f"Fix5p base NLL mismatch for {query!r} / {key}")

            endpoint_rows.append({
                "kind": "atomic",
                "group": str(src["group"]),
                "case_id": src["case_id"],
                "query": query,
                "expected_binding": list(expected),
                "target_true": target_true,
                "target_new": target_new,
                "fix5o_route_cohort": str(src["route_cohorts"]["fix5o"]),
                "hybrid_route_cohort": fix5l.route_cohort(hybrid_route, expected),
                "routes": {
                    "fix5o": fix_route,
                    "hybrid": fix5l.decision_to_json(hybrid_route),
                },
                "scores": {
                    "base": dict(base_saved),
                    "fix5o_integrated": dict(fix_saved),
                    "hybrid_integrated": dict(hybrid_score["integrated"]),
                },
                "generation": {
                    "base": src["generation"]["base"],
                    "fix5o_integrated": src["generation"]["fix5o_integrated"],
                    "hybrid_integrated": {
                        **hybrid_gen,
                        "flags": fix5m.generated_answer_flags(hybrid_gen["text"], target_true, target_new),
                    },
                },
            })

        direct = [r for r in endpoint_rows if r["group"] == "direct"]
        para = [r for r in endpoint_rows if r["group"] == "paraphrase"]
        if len(direct) != 50 or len(para) != 100:
            raise RuntimeError(f"endpoint query count mismatch direct={len(direct)} para={len(para)}")
        report["end_to_end_ran"] = True
        report["endpoint"] = {
            "direct": endpoint_summary(direct),
            "paraphrase": endpoint_summary(para),
            "source_fix5p_strict_reproduction_pass": True,
            "measurement_guardrails": {
                "Eff_Pref": "direct sensitive-target preference rate; lower is better",
                "Gen_Pref": "paraphrase sensitive-target preference rate; lower is better",
                "generated_disclosure": "canonical target_true mention only",
                "aliases_or_semantic_disclosure_measured": False,
                "knowledge_deletion_claimed": False,
            },
        }

    report_path = out / "mcf_semantic_rescue_router_seed1.json"
    records_path = out / "mcf_semantic_rescue_endpoint_records_seed1.jsonl"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if endpoint_rows:
        with records_path.open("w", encoding="utf-8") as handle:
            for row in endpoint_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    compact: dict[str, Any] = {
        "semantic_rescue_eta": eta_semantic,
        "semantic_calibration_accuracy_pct": report["semantic_calibration"]["accuracy_pct"],
        "semantic_validation_accuracy_pct": report["semantic_validation"]["accuracy_pct"],
        "validation_preservation_pass": validation_preservation_pass,
        "validation_policy": {
            "correct_forbidden_binding_accept_pct": validation_policy.get("correct_forbidden_binding_accept_pct"),
            "wrong_forbidden_binding_accept_pct": validation_policy.get("wrong_forbidden_binding_accept_pct"),
            "permitted_false_activation_pct": validation_policy.get("permitted_false_activation_pct"),
            "candidate_present_permitted_false_activation_pct": validation_policy.get("candidate_present_permitted_false_activation_pct"),
            "permitted_query_false_activation_pct": validation_policy.get("permitted_query_false_activation_pct"),
            "mixed_query_permitted_companion_false_activation_pct": validation_policy.get("mixed_query_permitted_companion_false_activation_pct"),
        },
        "development_only_official_seed1": dev_by_group,
        "end_to_end_ran": report["end_to_end_ran"],
    }
    if report.get("endpoint"):
        d = report["endpoint"]["direct"]
        p = report["endpoint"]["paraphrase"]
        compact["direct"] = {
            "base_eff_pref_pct": d["base"]["sensitive_preference_pct"],
            "fix5o_eff_pref_pct": d["fix5o_integrated"]["sensitive_preference_pct"],
            "hybrid_eff_pref_pct": d["hybrid_integrated"]["sensitive_preference_pct"],
            "base_disclosure_pct": d["base"]["canonical_sensitive_disclosure_pct"],
            "fix5o_disclosure_pct": d["fix5o_integrated"]["canonical_sensitive_disclosure_pct"],
            "hybrid_disclosure_pct": d["hybrid_integrated"]["canonical_sensitive_disclosure_pct"],
            **d["routing"],
        }
        compact["paraphrase"] = {
            "base_gen_pref_pct": p["base"]["sensitive_preference_pct"],
            "fix5o_gen_pref_pct": p["fix5o_integrated"]["sensitive_preference_pct"],
            "hybrid_gen_pref_pct": p["hybrid_integrated"]["sensitive_preference_pct"],
            "base_disclosure_pct": p["base"]["canonical_sensitive_disclosure_pct"],
            "fix5o_disclosure_pct": p["fix5o_integrated"]["canonical_sensitive_disclosure_pct"],
            "hybrid_disclosure_pct": p["hybrid_integrated"]["canonical_sensitive_disclosure_pct"],
            **p["routing"],
        }
    compact["report"] = str(report_path)
    if endpoint_rows:
        compact["records"] = str(records_path)
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
