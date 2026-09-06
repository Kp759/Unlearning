#!/usr/bin/env python3
"""Guarded entrypoint for Fix5m mixed generation evaluation.

The scientific experiment is unchanged. This wrapper tightens mixed-pair construction
so token-overlap pairs cannot have canonical answers that are identical or contain one
another. Without that guard, a whole-continuation canonical string audit could not
attribute the shared answer text to the forbidden versus permitted request.
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


def canonically_attribution_ambiguous(a: str, b: str) -> bool:
    aa = core.canonical_normalize(a)
    bb = core.canonical_normalize(b)
    if not aa or not bb:
        return True
    pa = f" {aa} "
    pb = f" {bb} "
    return pa in pb or pb in pa


def choose_mixed_pairs_guarded(
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
    candidates = []
    for rec in retain_records:
        cid = int(rec.get("case_id", -1))
        if core.record_binding(rec) in bank:
            continue
        if require_unseen and cid in seen_case_ids:
            continue
        candidates.append(rec)

    token_cache = {
        int(r.get("case_id", -1)): core.retain_true_token_ids(tok, r, llama_like)
        for r in candidates
    }
    used: set[int] = set()
    out: list[core.MixedPair] = []
    ambiguity_rejected_n = 0

    def build(kind: str, limit: int) -> None:
        nonlocal ambiguity_rejected_n
        if limit <= 0:
            return
        forget_order = sorted(
            forget_records,
            key=lambda r: core.stable_int(
                f"fix5m-guarded:{kind}:forget:{int(r.get('case_id', -1))}"
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
                fnew, ftrue = core.target_pair(fr)
                eligible = []
                for rr in candidates:
                    rcid = int(rr.get("case_id", -1))
                    if rcid in used:
                        continue
                    rnew, rtrue = core.target_pair(rr)
                    if canonically_attribution_ambiguous(ftrue, rtrue):
                        ambiguity_rejected_n += 1
                        continue
                    overlap = support_ids & token_cache[rcid]
                    if (kind == "overlap" and overlap) or (
                        kind == "nonoverlap" and not overlap
                    ):
                        eligible.append((rr, overlap, rnew, rtrue))
                if not eligible:
                    continue
                eligible.sort(
                    key=lambda item: core.stable_int(
                        f"fix5m-guarded:{kind}:pair:{fcid}:"
                        f"{int(item[0].get('case_id', -1))}"
                    )
                )
                rr, overlap, rnew, rtrue = eligible[0]
                rcid = int(rr.get("case_id", -1))
                used.add(rcid)
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
        if made < limit:
            raise RuntimeError(
                f"unable to construct {limit} guarded {kind} mixed pairs "
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
        "canonical_answer_attribution_ambiguity_excluded": True,
        "ambiguity_candidate_rejections_during_search_n": ambiguity_rejected_n,
    }
    return out, audit


def main() -> None:
    # Core main resolves this module global at runtime, so install the stricter
    # pair builder before entering the otherwise unchanged Fix5m evaluation.
    core.choose_mixed_pairs = choose_mixed_pairs_guarded
    core.main()


if __name__ == "__main__":
    main()
