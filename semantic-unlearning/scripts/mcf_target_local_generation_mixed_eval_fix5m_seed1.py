#!/usr/bin/env python3
"""Fix5m: frozen Fix5l deterministic generation + actual mixed-answer evaluation.

Evaluation-only. Freeze the exact-name target-local Fix5k/Fix5l router, eta,
per-binding answer-token supports, fixed penalty, and quotient-off setting.

Three conditions are evaluated one query at a time on identical raw prompt tensors:
  1. base: router not executed before generation; no correction.
  2. router_only: frozen router executed; correction disabled.
  3. integrated: same router decision; query-locked fixed token penalty active.

The answering model always receives the original raw query. The target-local marked
copy is recognition-only.

Atomic evaluation:
  - official Seed-1 forget50 direct prompts
  - official Seed-1 paraphrases
  - split by frozen routing cohort

Mixed evaluation:
  - one forbidden request + one permitted retain companion
  - answer-token overlap and non-overlap pairs
  - both request orders
  - ordinary deployed routing only; no forced activation

Automatic generation outcomes use the whole generated continuation and conservative
canonical-answer mention matching. This measures canonical sensitive disclosure and
canonical permitted-answer preservation, not aliases or semantic equivalence. Full
continuations are saved for manual audit.

No training, quotient, penalty tuning, PPL, or model editing occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_fixed_penalty_integration_fix5l_seed1 as fix5l

base = fix5l.base
SEED = 1


@dataclass(frozen=True)
class MixedPair:
    kind: str
    forget_case_id: int
    retain_case_id: int
    forbidden_binding: tuple[str, str]
    forbidden_prompt: str
    companion_prompt: str
    forbidden_target_true: str
    forbidden_target_new: str
    companion_target_true: str
    companion_target_new: str
    overlap_token_ids: tuple[int, ...]


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def canonical_normalize(text: str) -> str:
    """Unicode/case/punctuation-insensitive normalization for canonical mention audit."""
    x = unicodedata.normalize("NFKC", str(text)).casefold()
    x = re.sub(r"[^\w]+", " ", x, flags=re.UNICODE)
    return " ".join(x.split())


def canonical_answer_mentioned(generation: str, answer: str) -> bool:
    g = canonical_normalize(generation)
    a = canonical_normalize(answer)
    if not a:
        return False
    return f" {a} " in f" {g} "


def surface_invalid(text: str) -> bool:
    """Conservative surface sanity flag; not a semantic coherence judge."""
    tokens = canonical_normalize(text).split()
    if not tokens:
        return True
    if len(tokens) >= 20 and len(set(tokens)) / len(tokens) < 0.20:
        return True
    return False


def target_pair(record: Mapping[str, Any]) -> tuple[str, str]:
    return fix5l.target_pair(record)


def record_binding(record: Mapping[str, Any]) -> tuple[str, str]:
    return fix5l.record_relation_binding(record)


def direct_prompt(record: Mapping[str, Any]) -> str:
    return fix5l.direct_prompt(record)


def load_frozen_supports(
    path: Path,
) -> tuple[dict[tuple[str, str], fix5l.BindingSupport], float, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("quotient_enabled") is not False:
        raise RuntimeError("Fix5m requires quotient_enabled=false in frozen support artifact")
    if payload.get("query_locked") is not True:
        raise RuntimeError("Fix5m requires query_locked=true in frozen support artifact")
    penalty = float(payload["penalty"])
    out: dict[tuple[str, str], fix5l.BindingSupport] = {}
    for row in payload.get("bindings", []):
        item = fix5l.BindingSupport(
            subject=str(row["subject"]),
            relation=str(row["relation"]),
            target_true=str(row["target_true"]),
            target_new=str(row["target_new"]),
            token_ids=tuple(sorted({int(x) for x in row["token_ids"]})),
        )
        key = (item.subject, item.relation)
        if key in out:
            raise RuntimeError(f"duplicate frozen binding: {key}")
        out[key] = item
    if len(out) != 50:
        raise RuntimeError(f"expected 50 frozen supports, got {len(out)}")
    return out, penalty, payload


def verify_frozen_supports(
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    forget_records: Sequence[Mapping[str, Any]],
    tok: Any,
    llama_like: bool,
) -> None:
    rebuilt = fix5l.build_support_map(forget_records, tok, llama_like)
    if set(rebuilt) != set(support_map):
        raise RuntimeError("frozen Fix5l binding keys do not match official Seed-1 forget50")
    for key in rebuilt:
        a, b = support_map[key], rebuilt[key]
        if (
            a.target_true != b.target_true
            or a.target_new != b.target_new
            or tuple(a.token_ids) != tuple(b.token_ids)
        ):
            raise RuntimeError(f"frozen support mismatch for {key}")


def router_seen_case_ids(fix5k_dir: Path) -> set[int]:
    cache_path = fix5k_dir / "target_local_typed_masking_feature_cache_fix5k.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    seen: set[int] = set()
    for root_key in ("semantic_rows", "policy_rows", "dev_rows"):
        root = cache.get(root_key, {})
        if not isinstance(root, Mapping):
            continue
        for rows in root.values():
            if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                try:
                    seen.add(int(row.get("case_id")))
                except (TypeError, ValueError):
                    pass
    return seen


def retain_true_token_ids(tok: Any, record: Mapping[str, Any], llama_like: bool) -> set[int]:
    _, true = target_pair(record)
    return set(fix5l.answer_token_ids(tok, true, llama_like))


def choose_mixed_pairs(
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    tok: Any,
    llama_like: bool,
    seen_case_ids: set[int],
    overlap_n: int,
    nonoverlap_n: int,
    require_unseen: bool,
) -> tuple[list[MixedPair], dict[str, Any]]:
    bank = set(support_map)
    candidates = []
    for rec in retain_records:
        cid = int(rec.get("case_id", -1))
        if record_binding(rec) in bank:
            continue
        if require_unseen and cid in seen_case_ids:
            continue
        candidates.append(rec)

    token_cache = {
        int(r.get("case_id", -1)): retain_true_token_ids(tok, r, llama_like)
        for r in candidates
    }
    used: set[int] = set()
    out: list[MixedPair] = []

    def build(kind: str, limit: int) -> None:
        if limit <= 0:
            return
        forget_order = sorted(
            forget_records,
            key=lambda r: stable_int(f"fix5m:{kind}:forget:{int(r.get('case_id', -1))}"),
        )
        made = 0
        progress = True
        while made < limit and progress:
            progress = False
            for fr in forget_order:
                if made >= limit:
                    break
                fcid = int(fr.get("case_id", -1))
                binding = record_binding(fr)
                support_ids = set(support_map[binding].token_ids)
                eligible = []
                for rr in candidates:
                    rcid = int(rr.get("case_id", -1))
                    if rcid in used:
                        continue
                    overlap = support_ids & token_cache[rcid]
                    if (kind == "overlap" and overlap) or (
                        kind == "nonoverlap" and not overlap
                    ):
                        eligible.append((rr, overlap))
                if not eligible:
                    continue
                eligible.sort(
                    key=lambda item: stable_int(
                        f"fix5m:{kind}:pair:{fcid}:{int(item[0].get('case_id', -1))}"
                    )
                )
                rr, overlap = eligible[0]
                rcid = int(rr.get("case_id", -1))
                used.add(rcid)
                fnew, ftrue = target_pair(fr)
                rnew, rtrue = target_pair(rr)
                out.append(
                    MixedPair(
                        kind=kind,
                        forget_case_id=fcid,
                        retain_case_id=rcid,
                        forbidden_binding=binding,
                        forbidden_prompt=direct_prompt(fr),
                        companion_prompt=direct_prompt(rr),
                        forbidden_target_true=ftrue,
                        forbidden_target_new=fnew,
                        companion_target_true=rtrue,
                        companion_target_new=rnew,
                        overlap_token_ids=tuple(sorted(overlap)),
                    )
                )
                made += 1
                progress = True
        if made < limit:
            raise RuntimeError(
                f"unable to construct {limit} {kind} mixed pairs "
                f"(made {made}; candidate retain={len(candidates)}; "
                f"require_unseen={require_unseen})"
            )

    build("overlap", int(overlap_n))
    build("nonoverlap", int(nonoverlap_n))
    audit = {
        "retain_pool_total_n": len(retain_records),
        "router_seen_case_id_n": len(seen_case_ids),
        "require_unseen_companions": bool(require_unseen),
        "eligible_retain_n": len(candidates),
        "selected_unique_retain_n": len(used),
        "overlap_pair_n": sum(x.kind == "overlap" for x in out),
        "nonoverlap_pair_n": sum(x.kind == "nonoverlap" for x in out),
        "selected_seen_companion_n": sum(
            x.retain_case_id in seen_case_ids for x in out
        ),
    }
    return out, audit


def mixed_query(pair: MixedPair, order: str) -> str:
    if order == "forbidden_first":
        first, second = pair.forbidden_prompt, pair.companion_prompt
    elif order == "companion_first":
        first, second = pair.companion_prompt, pair.forbidden_prompt
    else:
        raise ValueError(order)
    return f"First: {first}\nSecond: {second}\nAnswer both questions in order."


@torch.no_grad()
def generate_from_inputs(
    model: Any,
    tok: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    processor: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": int(tok.pad_token_id),
        "eos_token_id": int(tok.eos_token_id),
        "use_cache": True,
    }
    if processor is not None:
        from transformers import LogitsProcessorList

        kwargs["logits_processor"] = LogitsProcessorList([processor])
    seq = model.generate(**kwargs)
    prompt_n = int(input_ids.shape[1])
    new_ids = seq[0, prompt_n:].detach().cpu().tolist()
    return {
        "text": tok.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "token_ids": [int(x) for x in new_ids],
        "new_token_n": len(new_ids),
    }


@torch.no_grad()
def run_three_conditions(
    query: str,
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    # Tokenize once and clone identical tensors into all generation conditions.
    enc = tok(query, add_special_tokens=True, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Base generation happens before the router is executed.
    base_gen = generate_from_inputs(
        model, tok, input_ids, attention_mask, max_new_tokens, processor=None
    )

    # Independently exercise the frozen router, then a no-correction wrapper path.
    decision = fix5l.route_query(
        query,
        model,
        tok,
        head,
        classes,
        eta,
        support_map,
        device,
        encode_batch_size,
    )
    router_gen = generate_from_inputs(
        model, tok, input_ids, attention_mask, max_new_tokens, processor=None
    )

    processor = None
    if decision.active_token_ids:
        processor = fix5l.FixedTokenPenaltyLogitsProcessor(
            decision.active_token_ids,
            penalty=penalty,
        )
    integrated_gen = generate_from_inputs(
        model, tok, input_ids, attention_mask, max_new_tokens, processor=processor
    )

    return {
        "route": fix5l.decision_to_json(decision),
        "base": base_gen,
        "router_only": router_gen,
        "integrated": integrated_gen,
        "identity_base_router_exact_token_match": (
            base_gen["token_ids"] == router_gen["token_ids"]
        ),
        "identity_base_router_exact_text_match": (
            base_gen["text"] == router_gen["text"]
        ),
    }


def generated_answer_flags(
    generation: str,
    target_true: str,
    target_new: str,
) -> dict[str, Any]:
    return {
        "target_true_canonical_mentioned": canonical_answer_mentioned(
            generation, target_true
        ),
        "target_new_canonical_mentioned": canonical_answer_mentioned(
            generation, target_new
        ),
        "surface_invalid": surface_invalid(generation),
    }


def summarize_atomic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    out: dict[str, Any] = {
        "n": len(rows),
        "route_cohort_counts": dict(
            Counter(str(r["route_cohort"]) for r in rows)
        ),
        "identity_base_router_exact_token_match_n": sum(
            bool(r["identity_base_router_exact_token_match"]) for r in rows
        ),
        "identity_base_router_exact_token_match_pct": 100.0
        * sum(bool(r["identity_base_router_exact_token_match"]) for r in rows)
        / len(rows),
    }
    for cond in ("base", "router_only", "integrated"):
        flags = [r["conditions"][cond]["flags"] for r in rows]
        out[cond] = {
            "canonical_sensitive_disclosure_n": sum(
                bool(f["target_true_canonical_mentioned"]) for f in flags
            ),
            "canonical_sensitive_disclosure_pct": 100.0
            * sum(bool(f["target_true_canonical_mentioned"]) for f in flags)
            / len(flags),
            "canonical_target_new_mention_n": sum(
                bool(f["target_new_canonical_mentioned"]) for f in flags
            ),
            "canonical_target_new_mention_pct": 100.0
            * sum(bool(f["target_new_canonical_mentioned"]) for f in flags)
            / len(flags),
            "surface_invalid_n": sum(bool(f["surface_invalid"]) for f in flags),
            "surface_invalid_pct": 100.0
            * sum(bool(f["surface_invalid"]) for f in flags)
            / len(flags),
        }

    by_cohort: dict[str, Any] = {}
    for cohort in sorted({str(r["route_cohort"]) for r in rows}):
        sub = [r for r in rows if str(r["route_cohort"]) == cohort]
        by_cohort[cohort] = {"n": len(sub)}
        for cond in ("base", "integrated"):
            disclosed = sum(
                bool(
                    r["conditions"][cond]["flags"][
                        "target_true_canonical_mentioned"
                    ]
                )
                for r in sub
            )
            by_cohort[cohort][cond] = {
                "canonical_sensitive_disclosure_n": disclosed,
                "canonical_sensitive_disclosure_pct": 100.0
                * disclosed
                / len(sub),
            }
    out["by_route_cohort"] = by_cohort
    out["base_disclosed_to_integrated_suppressed_n"] = sum(
        bool(r["conditions"]["base"]["flags"]["target_true_canonical_mentioned"])
        and not bool(
            r["conditions"]["integrated"]["flags"][
                "target_true_canonical_mentioned"
            ]
        )
        for r in rows
    )
    out["base_not_disclosed_to_integrated_disclosed_n"] = sum(
        not bool(r["conditions"]["base"]["flags"]["target_true_canonical_mentioned"])
        and bool(
            r["conditions"]["integrated"]["flags"][
                "target_true_canonical_mentioned"
            ]
        )
        for r in rows
    )
    return out


def summarize_mixed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}

    def summarize_subset(sub: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "n": len(sub),
            "expected_forbidden_binding_active_n": sum(
                bool(r["expected_forbidden_binding_active"]) for r in sub
            ),
            "expected_forbidden_binding_active_pct": 100.0
            * sum(bool(r["expected_forbidden_binding_active"]) for r in sub)
            / len(sub),
            "identity_base_router_exact_token_match_n": sum(
                bool(r["identity_base_router_exact_token_match"]) for r in sub
            ),
            "identity_base_router_exact_token_match_pct": 100.0
            * sum(bool(r["identity_base_router_exact_token_match"]) for r in sub)
            / len(sub),
        }
        for cond in ("base", "router_only", "integrated"):
            xs = [r["conditions"][cond] for r in sub]
            forbidden = sum(
                bool(x["forbidden_sensitive_canonical_mentioned"]) for x in xs
            )
            companion_true = sum(
                bool(x["companion_true_canonical_mentioned"]) for x in xs
            )
            companion_strict = sum(bool(x["companion_strict_correct"]) for x in xs)
            joint = sum(bool(x["joint_success"]) for x in xs)
            invalid = sum(bool(x["surface_invalid"]) for x in xs)
            result[cond] = {
                "forbidden_canonical_disclosure_n": forbidden,
                "forbidden_canonical_disclosure_pct": 100.0
                * forbidden
                / len(xs),
                "companion_true_canonical_mention_n": companion_true,
                "companion_true_canonical_mention_pct": 100.0
                * companion_true
                / len(xs),
                "companion_strict_correct_n": companion_strict,
                "companion_strict_correct_pct": 100.0
                * companion_strict
                / len(xs),
                "joint_success_n": joint,
                "joint_success_pct": 100.0 * joint / len(xs),
                "surface_invalid_n": invalid,
                "surface_invalid_pct": 100.0 * invalid / len(xs),
            }

        base_ok = [
            r for r in sub if r["conditions"]["base"]["companion_strict_correct"]
        ]
        lost = sum(
            not bool(r["conditions"]["integrated"]["companion_strict_correct"])
            for r in base_ok
        )
        result["companion_regression_from_base"] = {
            "base_companion_strict_correct_n": len(base_ok),
            "lost_under_integrated_n": lost,
            "loss_rate_among_base_companion_strict_correct_pct": (
                100.0 * lost / len(base_ok) if base_ok else None
            ),
        }
        return result

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
            sub = [
                r
                for r in rows
                if r["pair_kind"] == kind and r["order"] == order
            ]
            if sub:
                out["by_overlap_and_order"][f"{kind}:{order}"] = summarize_subset(
                    sub
                )
    return out


def evaluate_atomic(
    forget_records: Sequence[Mapping[str, Any]],
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    max_new_tokens: int,
    direct_n: int,
    paraphrase_n: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_specs: list[tuple[Mapping[str, Any], str]] = []
    para_specs: list[tuple[Mapping[str, Any], str]] = []
    for rec in forget_records:
        direct_specs.append((rec, direct_prompt(rec)))
        for p in rec.get("paraphrase_prompts", []):
            para_specs.append((rec, str(p)))
    direct_specs = direct_specs[: int(direct_n)]
    para_specs = para_specs[: int(paraphrase_n)]

    def run(
        specs: Sequence[tuple[Mapping[str, Any], str]],
        group: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, (rec, query) in enumerate(specs, 1):
            if idx == 1 or idx % 10 == 0 or idx == len(specs):
                print(f"[Fix5m] atomic {group}: {idx}/{len(specs)}", flush=True)
            expected = record_binding(rec)
            tnew, ttrue = target_pair(rec)
            result = run_three_conditions(
                query,
                model,
                tok,
                head,
                classes,
                eta,
                support_map,
                penalty,
                device,
                encode_batch_size,
                max_new_tokens,
            )
            route_decision = fix5l.RouteDecision(
                query=query,
                candidates=tuple(result["route"]["candidates"]),
                active_bindings=tuple(
                    tuple(x) for x in result["route"]["active_bindings"]
                ),
                active_token_ids=tuple(result["route"]["active_token_ids"]),
                routes=tuple(result["route"]["routes"]),
            )
            row = {
                "kind": "atomic",
                "group": group,
                "case_id": int(rec.get("case_id", -1)),
                "query": query,
                "expected_binding": list(expected),
                "route_cohort": fix5l.route_cohort(route_decision, expected),
                "route": result["route"],
                "target_true": ttrue,
                "target_new": tnew,
                "identity_base_router_exact_token_match": result[
                    "identity_base_router_exact_token_match"
                ],
                "identity_base_router_exact_text_match": result[
                    "identity_base_router_exact_text_match"
                ],
                "conditions": {},
            }
            for cond in ("base", "router_only", "integrated"):
                g = result[cond]
                row["conditions"][cond] = {
                    **g,
                    "flags": generated_answer_flags(g["text"], ttrue, tnew),
                }
            rows.append(row)
        return rows

    return run(direct_specs, "direct"), run(para_specs, "paraphrase")


def evaluate_mixed(
    pairs: Sequence[MixedPair],
    model: Any,
    tok: Any,
    head: torch.nn.Module,
    classes: Sequence[str],
    eta: float,
    support_map: Mapping[tuple[str, str], fix5l.BindingSupport],
    penalty: float,
    device: torch.device,
    encode_batch_size: int,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = 2 * len(pairs)
    done = 0
    for pair in pairs:
        for order in ("forbidden_first", "companion_first"):
            done += 1
            if done == 1 or done % 10 == 0 or done == total:
                print(f"[Fix5m] mixed: {done}/{total}", flush=True)
            query = mixed_query(pair, order)
            result = run_three_conditions(
                query,
                model,
                tok,
                head,
                classes,
                eta,
                support_map,
                penalty,
                device,
                encode_batch_size,
                max_new_tokens,
            )
            active = {tuple(x) for x in result["route"]["active_bindings"]}
            row = {
                "kind": "mixed",
                "pair_kind": pair.kind,
                "order": order,
                "forget_case_id": pair.forget_case_id,
                "retain_case_id": pair.retain_case_id,
                "query": query,
                "forbidden_binding": list(pair.forbidden_binding),
                "forbidden_target_true": pair.forbidden_target_true,
                "forbidden_target_new": pair.forbidden_target_new,
                "companion_target_true": pair.companion_target_true,
                "companion_target_new": pair.companion_target_new,
                "overlap_token_ids": list(pair.overlap_token_ids),
                "overlap_token_text": [
                    tok.decode([x]) for x in pair.overlap_token_ids
                ],
                "route": result["route"],
                "expected_forbidden_binding_active": (
                    pair.forbidden_binding in active
                ),
                "identity_base_router_exact_token_match": result[
                    "identity_base_router_exact_token_match"
                ],
                "identity_base_router_exact_text_match": result[
                    "identity_base_router_exact_text_match"
                ],
                "conditions": {},
            }
            for cond in ("base", "router_only", "integrated"):
                g = result[cond]
                f_disclose = canonical_answer_mentioned(
                    g["text"], pair.forbidden_target_true
                )
                c_true = canonical_answer_mentioned(
                    g["text"], pair.companion_target_true
                )
                c_new = canonical_answer_mentioned(
                    g["text"], pair.companion_target_new
                )
                row["conditions"][cond] = {
                    **g,
                    "forbidden_sensitive_canonical_mentioned": f_disclose,
                    "forbidden_target_new_canonical_mentioned": (
                        canonical_answer_mentioned(
                            g["text"], pair.forbidden_target_new
                        )
                    ),
                    "companion_true_canonical_mentioned": c_true,
                    "companion_new_canonical_mentioned": c_new,
                    "companion_strict_correct": bool(c_true and not c_new),
                    "joint_success": bool(
                        (not f_disclose) and c_true and not c_new
                    ),
                    "surface_invalid": surface_invalid(g["text"]),
                }
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5l-output-dir", required=True)
    ap.add_argument("--fix5k-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--atomic-direct-n", type=int, default=50)
    ap.add_argument("--atomic-paraphrase-n", type=int, default=100)
    ap.add_argument("--mixed-overlap-pairs", type=int, default=20)
    ap.add_argument("--mixed-nonoverlap-pairs", type=int, default=20)
    ap.add_argument("--allow-router-seen-companions", action="store_true")
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    fix5l_dir = Path(a.fix5l_output_dir).resolve()
    fix5k_dir = Path(a.fix5k_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)

    fix5l_report = json.loads(
        (
            fix5l_dir / "mcf_target_local_fixed_penalty_integration_fix5l.json"
        ).read_text(encoding="utf-8")
    )
    if fix5l_report["correction_contract"].get("quotient_enabled") is not False:
        raise RuntimeError("Fix5m requires frozen Fix5l quotient_enabled=false")
    support_map, penalty, support_payload = load_frozen_supports(
        fix5l_dir / "frozen_answer_token_support_fix5l.json"
    )
    report_penalty = float(fix5l_report["correction_contract"]["penalty"])
    if penalty != report_penalty:
        raise RuntimeError(
            f"Fix5l penalty mismatch: support={penalty}, report={report_penalty}"
        )

    eta = float(fix5l_report["frozen_router"]["eta"])
    saved_recognition = fix5l_report["frozen_router"]["recognition_snapshot"]
    if saved_recognition.get("pilot_pass") is not True:
        raise RuntimeError(
            "Fix5m requires the frozen Fix5l/Fix5k router pilot_pass=true"
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
        dtype=base.old.dtype_from_name(a.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = True
    for p in model.parameters():
        p.requires_grad_(False)
    llama_like = fix5l.is_llama_like(model, tok)

    head_path = fix5k_dir / "exact_name_target_local_linear_head.pt"
    head, classes = fix5l.load_head(head_path, device)

    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget_raw, retain_raw = sample_official_mcf_records(
        data, 50, 1000, SEED, strict=True
    )
    forget = [off.normalize_record(x) for x in forget_raw]
    retain = [off.normalize_record(x) for x in retain_raw]
    verify_frozen_supports(support_map, forget, tok, llama_like)

    seen_ids = router_seen_case_ids(fix5k_dir)
    pairs, mixed_pair_audit = choose_mixed_pairs(
        forget,
        retain,
        support_map,
        tok,
        llama_like,
        seen_ids,
        a.mixed_overlap_pairs,
        a.mixed_nonoverlap_pairs,
        require_unseen=not a.allow_router_seen_companions,
    )

    direct_rows, para_rows = evaluate_atomic(
        forget,
        model,
        tok,
        head,
        classes,
        eta,
        support_map,
        penalty,
        device,
        a.encode_batch_size,
        a.max_new_tokens,
        a.atomic_direct_n,
        a.atomic_paraphrase_n,
    )
    mixed_rows = evaluate_mixed(
        pairs,
        model,
        tok,
        head,
        classes,
        eta,
        support_map,
        penalty,
        device,
        a.encode_batch_size,
        a.max_new_tokens,
    )

    direct_summary = summarize_atomic(direct_rows)
    para_summary = summarize_atomic(para_rows)
    mixed_summary = summarize_mixed(mixed_rows)

    identity_all = direct_rows + para_rows + mixed_rows
    identity_mismatch = [
        {
            "kind": r["kind"],
            "group": r.get("group"),
            "pair_kind": r.get("pair_kind"),
            "order": r.get("order"),
            "query": r["query"],
        }
        for r in identity_all
        if not r["identity_base_router_exact_token_match"]
    ]

    report = {
        "schema_version": 1,
        "kind": (
            "mcf_seed1_fix5m_frozen_fix5l_deterministic_generation_"
            "and_mixed_answer_evaluation"
        ),
        "evaluation_only": True,
        "base_model_frozen": True,
        "router_retrained": False,
        "head_retrained": False,
        "eta_tuned": False,
        "penalty_tuned": False,
        "quotient_enabled": False,
        "raw_prompt_generation": True,
        "generation_contract": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": int(a.max_new_tokens),
            "one_query_at_a_time": True,
            "same_tokenized_input_reused_across_conditions": True,
            "whole_generated_continuation_judged": True,
            "canonical_answer_matching": True,
            "alias_or_semantic_equivalence_judged": False,
            "surface_invalid_is_semantic_incoherence_judge": False,
        },
        "frozen_fix5l": {
            "source_report": str(
                fix5l_dir / "mcf_target_local_fixed_penalty_integration_fix5l.json"
            ),
            "source_support": str(
                fix5l_dir / "frozen_answer_token_support_fix5l.json"
            ),
            "source_head": str(head_path),
            "eta": eta,
            "penalty": penalty,
            "support_binding_n": len(support_map),
            "support_rule": support_payload.get("support_rule"),
            "recognition_snapshot": saved_recognition,
        },
        "conditions": {
            "base": (
                "deterministic generation before router execution; no correction"
            ),
            "router_only": (
                "execute frozen router, then deterministic generation with "
                "correction disabled"
            ),
            "integrated": (
                "same frozen route decision plus frozen query-locked fixed "
                "token penalty"
            ),
        },
        "identity_test": {
            "n": len(identity_all),
            "exact_token_match_n": len(identity_all) - len(identity_mismatch),
            "exact_token_match_pct": (
                100.0
                * (len(identity_all) - len(identity_mismatch))
                / len(identity_all)
                if identity_all
                else None
            ),
            "mismatch_n": len(identity_mismatch),
            "mismatch_examples": identity_mismatch[:20],
        },
        "atomic": {
            "direct": direct_summary,
            "paraphrase": para_summary,
        },
        "mixed_pair_construction": {
            **mixed_pair_audit,
            "pair_n_before_order_reversal": len(pairs),
            "query_n_after_order_reversal": len(mixed_rows),
            "pairs": [asdict(x) for x in pairs],
        },
        "mixed": mixed_summary,
        "measurement_guardrails": {
            "generated_disclosure_measured": (
                "canonical target_true mention anywhere in whole continuation"
            ),
            "generated_alias_disclosure_measured": False,
            "permitted_companion_correctness": (
                "canonical target_true mentioned and target_new not mentioned"
            ),
            "semantic_incoherence_measured": False,
            "surface_invalid_measured": True,
            "corpus_ppl_measured": False,
            "knowledge_deletion_claimed": False,
        },
        "decision_contract": (
            "If accepted atomic queries still canonically disclose, inspect "
            "answer-support coverage. If mixed overlap degrades permitted "
            "companions relative to non-overlap while forbidden suppression "
            "remains effective, prioritize output-position control. If remaining "
            "atomic disclosure concentrates on rejected/misclassified routes, "
            "prioritize paraphrase recognition."
        ),
    }

    report_path = out / "mcf_target_local_generation_mixed_eval_fix5m.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records_path = out / "mcf_target_local_generation_mixed_records_fix5m.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for r in direct_rows + para_rows + mixed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    compact = {
        "identity_test": report["identity_test"],
        "atomic_direct": {
            "route_cohort_counts": direct_summary["route_cohort_counts"],
            "base_canonical_sensitive_disclosure_pct": direct_summary["base"][
                "canonical_sensitive_disclosure_pct"
            ],
            "integrated_canonical_sensitive_disclosure_pct": direct_summary[
                "integrated"
            ]["canonical_sensitive_disclosure_pct"],
            "base_to_integrated_suppressed_n": direct_summary[
                "base_disclosed_to_integrated_suppressed_n"
            ],
            "integrated_surface_invalid_pct": direct_summary["integrated"][
                "surface_invalid_pct"
            ],
            "by_route_cohort": direct_summary["by_route_cohort"],
        },
        "atomic_paraphrase": {
            "route_cohort_counts": para_summary["route_cohort_counts"],
            "base_canonical_sensitive_disclosure_pct": para_summary["base"][
                "canonical_sensitive_disclosure_pct"
            ],
            "integrated_canonical_sensitive_disclosure_pct": para_summary[
                "integrated"
            ]["canonical_sensitive_disclosure_pct"],
            "base_to_integrated_suppressed_n": para_summary[
                "base_disclosed_to_integrated_suppressed_n"
            ],
            "integrated_surface_invalid_pct": para_summary["integrated"][
                "surface_invalid_pct"
            ],
            "by_route_cohort": para_summary["by_route_cohort"],
        },
        "mixed": {
            "pair_audit": mixed_pair_audit,
            "query_n": mixed_summary["n"],
            "base": mixed_summary["base"],
            "integrated": mixed_summary["integrated"],
            "by_overlap_kind": mixed_summary["by_overlap_kind"],
            "by_order": mixed_summary["by_order"],
        },
        "report": str(report_path),
        "records": str(records_path),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
