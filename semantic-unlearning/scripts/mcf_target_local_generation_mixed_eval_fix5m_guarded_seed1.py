#!/usr/bin/env python3
"""Attribution-aware entrypoint for Fix5m mixed generation evaluation.

The frozen Fix5m experiment is unchanged. The original guard was too strict for the
Seed-1 data: requiring token overlap *and* canonically distinct answers produced zero
overlap pairs. This wrapper therefore keeps real token-overlap pairs and separates two
questions:

1) permitted-companion preservation: valid for every mixed pair;
2) forbidden canonical disclosure / joint success: reported on the canonically
   attribution-unambiguous subset only.

Companions are still required to be router-unseen when possible. If the unseen pool
cannot supply the requested overlap pairs, only that pair family falls back to the
full retain pool and the fallback is explicitly audited.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_generation_mixed_eval_fix5m_seed1 as core

_ORIGINAL_EVALUATE_MIXED = core.evaluate_mixed
_ORIGINAL_SUMMARIZE_MIXED = core.summarize_mixed


def canonically_attribution_ambiguous(a: str, b: str) -> bool:
    aa = core.canonical_normalize(a)
    bb = core.canonical_normalize(b)
    if not aa or not bb:
        return True
    pa = f" {aa} "
    pb = f" {bb} "
    return pa in pb or pb in pa


def choose_mixed_pairs_attribution_aware(
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    support_map: Mapping[tuple[str, str], core.fix5l.BindingSupport],
    tok: Any,
    llama_like: bool,
    seen_case_ids: set[int],
    overlap_n: int,
    nonoverlap_n: int,
    require_unseen: bool,
) -> tuple[list[core.MixedPair], dict[str, Any]]:
    bank = set(support_map)
    all_candidates = [
        r for r in retain_records if core.record_binding(r) not in bank
    ]
    unseen_candidates = [
        r for r in all_candidates if int(r.get("case_id", -1)) not in seen_case_ids
    ]
    token_cache = {
        int(r.get("case_id", -1)): core.retain_true_token_ids(tok, r, llama_like)
        for r in all_candidates
    }
    used: set[int] = set()
    out: list[core.MixedPair] = []
    fallback_families: list[str] = []

    def fill(kind: str, limit: int, candidates: Sequence[Mapping[str, Any]]) -> int:
        if limit <= 0:
            return 0
        forget_order = sorted(
            forget_records,
            key=lambda r: core.stable_int(
                f"fix5m-attribution-aware:{kind}:forget:{int(r.get('case_id', -1))}"
            ),
        )
        made = 0
        progress = True
        while made < limit and progress:
            progress = False
            for fr in forget_order:
                if made >= limit:
                    break
                fcid = int(fr.get("case_id", -1))
                binding = core.record_binding(fr)
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
                    key=lambda item: core.stable_int(
                        f"fix5m-attribution-aware:{kind}:pair:{fcid}:"
                        f"{int(item[0].get('case_id', -1))}"
                    )
                )
                rr, overlap = eligible[0]
                rcid = int(rr.get("case_id", -1))
                used.add(rcid)
                fnew, ftrue = core.target_pair(fr)
                rnew, rtrue = core.target_pair(rr)
                out.append(
                    core.MixedPair(
                        kind=kind,
                        forget_case_id=fcid,
                        retain_case_id=rcid,
                        forbidden_binding=binding,
                        forbidden_prompt=core.direct_prompt(fr),
                        companion_prompt=core.direct_prompt(rr),
                        forbidden_target_true=ftrue,
                        forbidden_target_new=fnew,
                        companion_target_true=rtrue,
                        companion_target_new=rnew,
                        overlap_token_ids=tuple(sorted(overlap)),
                    )
                )
                made += 1
                progress = True
        return made

    def build(kind: str, limit: int) -> None:
        if limit <= 0:
            return
        primary = unseen_candidates if require_unseen else all_candidates
        made = fill(kind, limit, primary)
        if made < limit and require_unseen:
            fallback_families.append(kind)
            made += fill(kind, limit - made, all_candidates)
        if made < limit:
            raise RuntimeError(
                f"unable to construct {limit} {kind} mixed pairs "
                f"(made {made}; unseen retain={len(unseen_candidates)}; "
                f"all retain={len(all_candidates)})"
            )

    build("overlap", int(overlap_n))
    build("nonoverlap", int(nonoverlap_n))

    ambiguous_pairs = [
        p
        for p in out
        if canonically_attribution_ambiguous(
            p.forbidden_target_true, p.companion_target_true
        )
    ]
    audit = {
        "retain_pool_total_n": len(retain_records),
        "router_seen_case_id_n": len(seen_case_ids),
        "require_unseen_companions_requested": bool(require_unseen),
        "eligible_unseen_retain_n": len(unseen_candidates),
        "eligible_all_retain_n": len(all_candidates),
        "selected_unique_retain_n": len(used),
        "overlap_pair_n": sum(x.kind == "overlap" for x in out),
        "nonoverlap_pair_n": sum(x.kind == "nonoverlap" for x in out),
        "selected_seen_companion_n": sum(
            x.retain_case_id in seen_case_ids for x in out
        ),
        "unseen_requirement_fallback_families": fallback_families,
        "canonical_answer_attribution_ambiguous_pair_n": len(ambiguous_pairs),
        "canonical_answer_attribution_ambiguous_overlap_pair_n": sum(
            p.kind == "overlap" for p in ambiguous_pairs
        ),
        "attribution_policy": (
            "Companion-preservation metrics use all pairs. Forbidden canonical "
            "disclosure and joint-success attribution-safe metrics use only pairs "
            "whose forbidden and companion canonical answers are distinguishable."
        ),
    }
    return out, audit


def evaluate_mixed_attribution_aware(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    rows = _ORIGINAL_EVALUATE_MIXED(*args, **kwargs)
    for row in rows:
        row["canonical_answer_attribution_ambiguous"] = (
            canonically_attribution_ambiguous(
                str(row["forbidden_target_true"]),
                str(row["companion_target_true"]),
            )
        )
    return rows


def _annotate_subset(
    report: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    ambiguous = [
        r for r in rows if bool(r.get("canonical_answer_attribution_ambiguous"))
    ]
    safe = [
        r for r in rows if not bool(r.get("canonical_answer_attribution_ambiguous"))
    ]
    report["canonical_answer_attribution"] = {
        "ambiguous_n": len(ambiguous),
        "ambiguous_pct": 100.0 * len(ambiguous) / len(rows) if rows else None,
        "unambiguous_n": len(safe),
        "companion_metrics_use_all_rows": True,
        "forbidden_disclosure_and_joint_metrics_should_use_unambiguous_rows": True,
    }
    if not safe:
        report["attribution_safe_forbidden_and_joint"] = {
            "n": 0,
            "base": None,
            "router_only": None,
            "integrated": None,
        }
        return

    safe_report = _ORIGINAL_SUMMARIZE_MIXED(safe)
    attribution_safe: dict[str, Any] = {"n": len(safe)}
    for cond in ("base", "router_only", "integrated"):
        attribution_safe[cond] = {
            "forbidden_canonical_disclosure_n": safe_report[cond][
                "forbidden_canonical_disclosure_n"
            ],
            "forbidden_canonical_disclosure_pct": safe_report[cond][
                "forbidden_canonical_disclosure_pct"
            ],
            "joint_success_n": safe_report[cond]["joint_success_n"],
            "joint_success_pct": safe_report[cond]["joint_success_pct"],
        }
    report["attribution_safe_forbidden_and_joint"] = attribution_safe


def summarize_mixed_attribution_aware(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    out = _ORIGINAL_SUMMARIZE_MIXED(rows)
    if not rows:
        return out
    _annotate_subset(out, rows)
    for kind, report in out.get("by_overlap_kind", {}).items():
        _annotate_subset(
            report, [r for r in rows if str(r.get("pair_kind")) == kind]
        )
    for order, report in out.get("by_order", {}).items():
        _annotate_subset(report, [r for r in rows if str(r.get("order")) == order])
    for key, report in out.get("by_overlap_and_order", {}).items():
        kind, order = key.split(":", 1)
        _annotate_subset(
            report,
            [
                r
                for r in rows
                if str(r.get("pair_kind")) == kind
                and str(r.get("order")) == order
            ],
        )
    out["interpretation_guardrail"] = (
        "All companion-preservation metrics remain valid on every mixed row. "
        "When canonical_answer_attribution.ambiguous_n > 0, do not use the legacy "
        "all-row forbidden-disclosure or joint-success fields for causal attribution; "
        "use attribution_safe_forbidden_and_joint instead."
    )
    return out


def main() -> None:
    core.choose_mixed_pairs = choose_mixed_pairs_attribution_aware
    core.evaluate_mixed = evaluate_mixed_attribution_aware
    core.summarize_mixed = summarize_mixed_attribution_aware
    core.main()


if __name__ == "__main__":
    main()
