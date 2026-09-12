#!/usr/bin/env python3
"""Analyze duplicate natural-address groups in a locked MQuAKE forget split."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def norm(text: str) -> str:
    return " ".join(str(text).casefold().split())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-visible", required=True)
    args = p.parse_args()

    records = json.loads(Path(args.training_visible).read_text())
    groups = defaultdict(list)

    for record in records:
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        relation = str(rr.get("relation_id"))
        prompt = str(rr["prompt"]).format(subject)
        obj = str(rr["target_true"]["str"])
        key = (norm(subject), relation, prompt)
        groups[key].append({
            "case_id": int(record["case_id"]),
            "mquake_case_id": int(record["mquake_case_id"]),
            "source_index": int(record["source_index"]),
            "rewrite_index": int(record["rewrite_index"]),
            "subject": subject,
            "relation": relation,
            "prompt": prompt,
            "object": obj,
        })

    duplicate_groups = {
        key: rows for key, rows in groups.items() if len(rows) > 1
    }
    conflicting_groups = {
        key: rows
        for key, rows in duplicate_groups.items()
        if len({norm(row["object"]) for row in rows}) > 1
    }
    exact_duplicate_groups = {
        key: rows
        for key, rows in duplicate_groups.items()
        if len({norm(row["object"]) for row in rows}) == 1
    }

    unique_address_count = len(groups)
    duplicate_records = sum(len(v) - 1 for v in duplicate_groups.values())

    payload = {
        "atomic_records": len(records),
        "unique_natural_addresses": unique_address_count,
        "duplicate_address_group_count": len(duplicate_groups),
        "duplicate_records_beyond_first": duplicate_records,
        "exact_duplicate_same_object_group_count": len(exact_duplicate_groups),
        "conflicting_same_address_different_object_group_count": len(conflicting_groups),
        "exact_duplicate_groups": [
            {
                "subject": rows[0]["subject"],
                "relation": rows[0]["relation"],
                "prompt": rows[0]["prompt"],
                "objects": sorted({row["object"] for row in rows}),
                "count": len(rows),
                "case_ids": [row["case_id"] for row in rows],
            }
            for _, rows in sorted(
                exact_duplicate_groups.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            )
        ],
        "conflicting_groups": [
            {
                "subject": rows[0]["subject"],
                "relation": rows[0]["relation"],
                "prompt": rows[0]["prompt"],
                "objects": sorted({row["object"] for row in rows}),
                "count": len(rows),
                "case_ids": [row["case_id"] for row in rows],
                "rows": rows,
            }
            for _, rows in sorted(
                conflicting_groups.items(),
                key=lambda kv: (-len(kv[1]), kv[0]),
            )
        ],
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
