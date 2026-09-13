#!/usr/bin/env python3
"""Compare MQuAKE seed-1 retain metrics after stratifying atomic overlap.

CPU-only: reuses saved prediction rows from frozen base, Router V1, and Router
V2. This exposes the benchmark-level retain score separately for exact
forget-association overlap and genuinely association-disjoint retain facts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import mquake_zero_unlearn_official_eval as mquake
from mquake_fact_association_embeddings import association_key_from_record, normalized


def sr(record):
    rr = record["requested_rewrite"]
    return normalized(rr["subject"]), str(rr.get("relation_id"))


def subject(record):
    return normalized(record["requested_rewrite"]["subject"])


def classify(record, forget_keys, forget_sr, forget_subjects):
    key = association_key_from_record(record)
    pair = sr(record)
    subj = subject(record)
    if key in forget_keys:
        return "exact_forget_association"
    if pair in forget_sr:
        return "same_subject_relation_different_object"
    if subj in forget_subjects:
        return "same_subject_different_relation"
    return "subject_disjoint"


def route_summary(rows):
    out = {}
    for prompt_type in ("rewrite", "atomic_gen"):
        current = [r for r in rows if r.get("prompt_type") == prompt_type]
        if not current:
            out[prompt_type] = {
                "token_decisions": 0,
                "active_token_decisions": 0,
                "route_active_fraction": None,
            }
            continue
        has_route = all("association_route_active" in r for r in current)
        active = (
            sum(bool(r["association_route_active"]) for r in current)
            if has_route else None
        )
        out[prompt_type] = {
            "token_decisions": len(current),
            "active_token_decisions": active,
            "route_active_fraction": (
                None if active is None else active / len(current)
            ),
        }
    return out


def summarize_subset(name, records, rows):
    case_ids = {int(r["case_id"]) for r in records}
    selected_rows = [
        r for r in rows if int(r["case_id"]) in case_ids
    ]
    summary = mquake.summarize_atomic_split(name, records, selected_rows)
    return {
        "atomic_records": len(records),
        "Eff": summary.get("Eff"),
        "AtomicGen": summary.get("AtomicGen"),
        "Eff_micro": summary.get("Eff_micro"),
        "AtomicGen_micro": summary.get("AtomicGen_micro"),
        "routes": route_summary(selected_rows),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mquake-path", default="data/MQuAKE-CF-3k-v2.json")
    p.add_argument(
        "--base-eval",
        default="outputs/fact_association_seed1_frozen_base/mquake_base_seed1.json",
    )
    p.add_argument(
        "--v1-eval",
        default="outputs/mquake_fact_assoc_seed1_uniqueassoc_train/official_mquake_eval.json",
    )
    p.add_argument(
        "--v2-eval",
        default="outputs/mquake_fact_assoc_router_v2_seed1/official_mquake_eval.json",
    )
    p.add_argument(
        "--model-path",
        default=None,
        help="Optional tokenizer path; otherwise read from V2 association manifest.",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    v2_path = Path(args.v2_eval).resolve()
    v2_run = v2_path.parent
    manifest = json.loads((v2_run / "association_manifest.json").read_text())
    model_path = Path(args.model_path or manifest["model_path"]).resolve()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_records, retain_records = mquake.load_official_eval_records(
        Path(args.mquake_path).resolve(),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )
    forget_keys = {association_key_from_record(r) for r in forget_records}
    forget_sr = {sr(r) for r in forget_records}
    forget_subjects = {subject(r) for r in forget_records}

    category_by_case = {
        int(r["case_id"]): classify(r, forget_keys, forget_sr, forget_subjects)
        for r in retain_records
    }
    categories = {
        category: [
            r for r in retain_records
            if category_by_case[int(r["case_id"])] == category
        ]
        for category in (
            "exact_forget_association",
            "same_subject_relation_different_object",
            "same_subject_different_relation",
            "subject_disjoint",
        )
    }
    categories["association_disjoint"] = [
        r for r in retain_records
        if category_by_case[int(r["case_id"])] != "exact_forget_association"
    ]

    eval_paths = {
        "frozen_base": Path(args.base_eval).resolve(),
        "router_v1": Path(args.v1_eval).resolve(),
        "router_v2": v2_path,
    }
    methods = {}
    for method, path in eval_paths.items():
        if not path.is_file():
            methods[method] = {"missing": str(path)}
            continue
        payload = json.loads(path.read_text())
        rows = payload.get("retain_raw")
        if rows is None:
            methods[method] = {
                "missing_retain_raw": str(path),
            }
            continue
        methods[method] = {
            "source": str(path),
            "overall": summarize_subset(
                "retain_all", retain_records, rows
            ),
            "strata": {
                name: summarize_subset(
                    f"retain_{name}", records, rows
                )
                for name, records in categories.items()
                if records
            },
        }

    counts = Counter(category_by_case.values())
    result = {
        "dataset": "MQuAKE-CF-3k-v2",
        "seed": 1,
        "protocol_note": (
            "1000 retain source instances flatten to atomic records; exact "
            "association overlap is defined by normalized "
            "(subject, relation_id, target_true)."
        ),
        "forget_atomic_records": len(forget_records),
        "retain_atomic_records": len(retain_records),
        "retain_stratum_counts": dict(counts),
        "association_disjoint_atomic_records": sum(
            count for name, count in counts.items()
            if name != "exact_forget_association"
        ),
        "methods": methods,
        "interpretation": {
            "official_overall_retain": (
                "kept unchanged for benchmark comparability, even though it "
                "contains exact protected associations"
            ),
            "association_disjoint_retain": (
                "diagnostic locality metric excluding only exact protected "
                "associations; same-subject/different-relation facts remain "
                "included because a selective router should preserve them"
            ),
        },
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else v2_run / "mquake_seed1_overlap_stratified_comparison.json"
    )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
