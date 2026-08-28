#!/usr/bin/env python3
"""Attribute held-out MCF Gen failures to training-safe surrogate coverage.

This is a post-hoc diagnostic. It reads a frozen official-evaluation artifact
and the already frozen semantic-surrogate receipt, then reports whether unseen
paraphrase failures concentrate in records that had no accepted surrogates.
Nothing produced here may be used for checkpoint selection or training.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-eval", required=True)
    parser.add_argument("--surrogate-artifact", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _key(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def analyze(
    official: Mapping[str, Any],
    surrogate: Mapping[str, Any],
) -> Dict[str, Any]:
    raw = official.get("forget_raw")
    surrogate_rows = surrogate.get("records")
    if not isinstance(raw, list) or not isinstance(surrogate_rows, list):
        raise RuntimeError("official evaluation or surrogate artifact lacks records")
    by_subject: Dict[str, Mapping[str, Any]] = {}
    for row in surrogate_rows:
        subject_key = _key(row.get("subject", ""))
        if not subject_key or subject_key in by_subject:
            raise RuntimeError("surrogate subjects must be non-empty and unique")
        by_subject[subject_key] = row

    groups: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "records": 0,
            "paraphrase_prompts": 0,
            "sensitive_preference_prompts": 0,
            "nonpositive_margin_prompts": 0,
            "records_with_sensitive_preference": 0,
        }
    )
    per_record = []
    for position, item in enumerate(raw):
        rewrite = item.get("requested_rewrite", {})
        subject = str(rewrite.get("subject", ""))
        source = by_subject.get(_key(subject))
        if source is None:
            raise RuntimeError(f"no surrogate row matches official subject {subject!r}")
        prompts = item.get("post", {}).get("paraphrase_prompts_probs", [])
        if not isinstance(prompts, list):
            raise RuntimeError(f"invalid paraphrase probabilities at position {position}")
        margins = [
            float(values["target_true"]) - float(values["target_new"])
            for values in prompts
        ]
        sensitive = sum(value < 0.0 for value in margins)
        nonpositive = sum(value <= 0.0 for value in margins)
        surrogates = source.get("surrogate_prompts", [])
        if not isinstance(surrogates, list):
            raise RuntimeError(f"invalid surrogate prompts for {subject!r}")
        status = str(source.get("augmentation_status") or "unknown")
        group = groups[status]
        group["records"] += 1
        group["paraphrase_prompts"] += len(margins)
        group["sensitive_preference_prompts"] += sensitive
        group["nonpositive_margin_prompts"] += nonpositive
        group["records_with_sensitive_preference"] += int(sensitive > 0)
        per_record.append(
            {
                "position": position,
                "case_id": int(source.get("case_id", -1)),
                "subject": subject,
                "augmentation_status": status,
                "accepted_surrogates": len(surrogates),
                "paraphrase_prompts": len(margins),
                "sensitive_preference_prompts": sensitive,
                "nonpositive_margin_prompts": nonpositive,
                "minimum_margin": min(margins) if margins else None,
            }
        )

    total_prompts = sum(row["paraphrase_prompts"] for row in groups.values())
    total_sensitive = sum(
        row["sensitive_preference_prompts"] for row in groups.values()
    )
    group_payload: Dict[str, Dict[str, Any]] = {}
    for name, row in sorted(groups.items()):
        prompts = int(row["paraphrase_prompts"])
        records = int(row["records"])
        group_payload[name] = {
            **row,
            "sensitive_preference_percent": (
                100.0 * int(row["sensitive_preference_prompts"]) / prompts
                if prompts
                else 0.0
            ),
            "records_with_sensitive_preference_percent": (
                100.0 * int(row["records_with_sensitive_preference"]) / records
                if records
                else 0.0
            ),
        }
    return {
        "schema_version": 1,
        "kind": "mcf_compositional_marker_gen_failure_attribution",
        "official_gen_recomputed": (
            100.0 * total_sensitive / total_prompts if total_prompts else 0.0
        ),
        "groups": group_payload,
        "failed_records": [
            row for row in per_record if row["sensitive_preference_prompts"] > 0
        ],
        "per_record": per_record,
        "diagnostic_only": True,
        "used_for_training_or_checkpoint_selection": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    official_path = Path(args.official_eval).resolve()
    surrogate_path = Path(args.surrogate_artifact).resolve()
    official = json.loads(official_path.read_text(encoding="utf-8"))
    surrogate = json.loads(surrogate_path.read_text(encoding="utf-8"))
    report = analyze(official, surrogate)
    report["official_eval"] = str(official_path)
    report["surrogate_artifact"] = str(surrogate_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"recomputed Gen={report['official_gen_recomputed']:.2f}; "
        f"wrote {out}"
    )


if __name__ == "__main__":
    main()
