#!/usr/bin/env python3
"""Read-only audit of saved Fix5o results. Standard library only; no model loads.

Reports paired routing changes, a literal-relation-label probe, saved policy metrics,
and (optionally) coverage of canonical disclosures in the earlier Fix5m base outputs.
No training, threshold tuning, new inference, or semantic disclosure judging occurs.
Input files are never modified. --out uses exclusive creation and never overwrites.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any

ARMS = ("baseline_exact_name", "augmented_exact_name")
GROUP = "official_seed1_paraphrase_development_only"


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(value)
    return rows


def flag(row: dict[str, Any], name: str) -> bool:
    value = row[name]
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be an explicit JSON boolean")
    return value


def key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["case_id"],
        row["original_query"],
        row["designated_subject"],
        row["expected_relation"],
    )


def index_arm(
    rows: list[dict[str, Any]], arm: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in rows:
        if row.get("arm") != arm or row.get("group") != GROUP:
            continue
        k = key(row)
        if k in result:
            raise ValueError(f"Duplicate route identity for {arm}: {k!r}")
        if flag(row, "relation_correct") != (
            row["predicted_relation"] == row["expected_relation"]
        ):
            raise ValueError(f"Inconsistent relation_correct for {k!r}")
        result[k] = row
    if not result:
        raise ValueError(f"No {GROUP} rows for {arm}")
    return result


def accepted_correct(row: dict[str, Any]) -> bool:
    return flag(row, "relation_correct") and flag(row, "activates")


def paired_audit(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict]]:
    arms = {a: index_arm(rows, a) for a in ARMS}
    b, a = (arms[name] for name in ARMS)
    if b.keys() != a.keys():
        raise ValueError(
            "Baseline/augmented identities differ; refusing an unpaired audit"
        )
    transitions = Counter()
    acceptance = Counter()
    details = []
    summaries = {}
    for name, values in arms.items():
        vals = list(values.values())
        ncorrect = sum(flag(r, "relation_correct") for r in vals)
        naccepted = sum(accepted_correct(r) for r in vals)
        summaries[name] = {
            "n": len(vals),
            "relation_correct_n": ncorrect,
            "correctly_accepted_n": naccepted,
            "correct_but_not_activated_n": ncorrect - naccepted,
            "misclassified_n": len(vals) - ncorrect,
            "wrong_binding_activation_n": sum(
                flag(r, "activates") and not flag(r, "relation_correct")
                for r in vals
            ),
            "selected_text_exact_fit_overlap_n": sum(
                flag(r, "exact_selected_text_fit_overlap") for r in vals
            ),
            "confusions": dict(
                Counter(
                    f'{r["expected_relation"]} -> {r["predicted_relation"]}'
                    for r in vals
                    if not flag(r, "relation_correct")
                )
            ),
        }
    for k, rb in b.items():
        ra = a[k]
        cb, ca = flag(rb, "relation_correct"), flag(ra, "relation_correct")
        ab, aa = accepted_correct(rb), accepted_correct(ra)
        semantic = (
            f'{"correct" if cb else "wrong"}_to_'
            f'{"correct" if ca else "wrong"}'
        )
        accepted = (
            "retained"
            if ab and aa
            else "lost"
            if ab
            else "gained"
            if aa
            else "neither"
        )
        transitions[semantic] += 1
        acceptance[accepted] += 1
        details.append(
            {
                "case_id": k[0],
                "query": k[1],
                "subject": k[2],
                "expected_relation": k[3],
                "semantic_transition": semantic,
                "correct_acceptance_transition": accepted,
                "baseline_prediction": rb["predicted_relation"],
                "augmented_prediction": ra["predicted_relation"],
                "baseline_margin": rb["margin"],
                "augmented_margin": ra["margin"],
            }
        )
    return {
        "status": "PAIRED_SAVED_PREDICTIONS_ONLY",
        "arms": summaries,
        "semantic_transition_counts": dict(transitions),
        "correct_acceptance_transition_counts": dict(acceptance),
        "per_prompt": details,
    }, arms


def literal_label_probe(
    rows: list[dict[str, Any]], labels: dict[str, str]
) -> dict[str, Any]:
    """Uses only supplied label strings; longest exact label wins, ties abstain.

    Remove the subject first to prevent entity names from triggering relation labels.
    This is a shortcut diagnostic, not a general relation classifier.
    """
    counts = Counter()
    skeletons = set()
    by_relation: dict[str, Counter] = {}
    for row in rows:
        text = re.sub(
            r"(?<!\w)" + re.escape(row["subject"]) + r"(?!\w)",
            " ENTITY ",
            row["text"],
            flags=re.I,
        )
        text = " ".join(text.casefold().split())
        skeletons.add(text)
        hits = [
            (rid, label)
            for rid, label in labels.items()
            if re.search(
                r"(?<!\w)" + re.escape(label.casefold()) + r"(?!\w)", text
            )
        ]
        expected = row["relation"]
        counts["n"] += 1
        counts["true_literal_label_present_n"] += int(
            any(rid == expected for rid, _ in hits)
        )
        prediction = None
        if hits:
            length = max(len(label) for _, label in hits)
            winners = [rid for rid, label in hits if len(label) == length]
            if len(winners) == 1:
                prediction = winners[0]
        counts["literal_rule_correct_n"] += int(prediction == expected)
        counts["literal_rule_abstain_n"] += int(prediction is None)
        by_relation.setdefault(expected, Counter())["n"] += 1
        by_relation[expected]["literal_rule_correct_n"] += int(
            prediction == expected
        )
    n = counts["n"]
    return {
        "description": (
            "Longest whole-phrase catalog label after subject removal; "
            "no model, no training"
        ),
        **dict(counts),
        "literal_rule_accuracy_pct": (
            100.0 * counts["literal_rule_correct_n"] / n if n else None
        ),
        "unique_subject_removed_text_n": len(skeletons),
        "by_relation": {k: dict(v) for k, v in by_relation.items()},
        "interpretation": (
            "High accuracy here shows the probe permits literal-label decoding; "
            "it does not establish natural paraphrase transfer."
        ),
    }


def disclosure_coverage(path: Path, arms: dict[str, dict]) -> dict[str, Any]:
    old = {}
    for row in load_jsonl(path):
        if row.get("kind") != "atomic" or row.get("group") != "paraphrase":
            continue
        binding = row["expected_binding"]
        k = (row["case_id"], row["query"], binding[0], binding[1])
        if k in old:
            raise ValueError(f"Duplicate Fix5m identity: {k!r}")
        old[k] = row
    base = arms[ARMS[0]]
    if old.keys() != base.keys():
        raise ValueError(
            "Fix5m and Fix5o paraphrase identities differ; "
            "refusing partial coverage estimate"
        )
    leaking = {
        k
        for k, row in old.items()
        if flag(
            row["conditions"]["base"]["flags"],
            "target_true_canonical_mentioned",
        )
    }
    result = {
        "n": len(old),
        "saved_base_canonical_disclosure_n": len(leaking),
        "new_generation_performed": False,
        "note": (
            "Coverage of saved BASE canonical mentions only, "
            "not new integrated leakage."
        ),
    }
    selected = {}
    for arm in ARMS:
        selected[arm] = {
            k for k, row in arms[arm].items() if accepted_correct(row)
        }
        covered = selected[arm] & leaking
        result[arm] = {
            "correctly_accepted_base_disclosing_n": len(covered),
            "coverage_of_base_disclosures_pct": (
                100.0 * len(covered) / len(leaking) if leaking else None
            ),
        }
    result["newly_covered_base_disclosures_n"] = len(
        (selected[ARMS[1]] - selected[ARMS[0]]) & leaking
    )
    result["lost_coverage_base_disclosures_n"] = len(
        (selected[ARMS[0]] - selected[ARMS[1]]) & leaking
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix5o-dir", type=Path, required=True)
    parser.add_argument("--fix5m-records", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    root = args.fix5o_dir.resolve()
    if args.out and args.out.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.out}")
    report = load_json(
        root / "mcf_target_local_augmented_relation_router_fix5o.json"
    )
    routes = load_jsonl(
        root / "mcf_target_local_augmented_relation_router_records_fix5o.jsonl"
    )
    paired, arms = paired_audit(routes)
    heldout = load_json(root / "augmentation_heldout_rows_fix5o.json")
    result = {
        "source_directory": str(root),
        "read_only_inputs": True,
        "new_model_inference": False,
        "thresholds_changed": False,
        "development_evidence_only": True,
        "paired_official_paraphrase": paired,
        "authored_probe_literal_label_audit": literal_label_probe(
            heldout, report["relation_contracts"]["labels"]
        ),
        "saved_policy_results": {
            arm: {
                "eta": report["results"][arm]["eta"],
                "fit_row_n": report["results"][arm]["fit_row_n"],
                "calibration": report["results"][arm]["calibration"],
                "validation_policy": report["results"][arm]["validation_policy"],
                "authored_probe_policy_status": report["results"][arm][
                    "augmentation_heldout_probe"
                ]["policy"],
                "authored_probe_semantic": report["results"][arm][
                    "augmentation_heldout_probe"
                ]["semantic"],
                "pilot_pass": report["results"][arm][
                    "pilot_pass_preservation_and_original_validation"
                ],
            }
            for arm in ARMS
        },
    }
    if args.fix5m_records:
        result["saved_base_disclosure_coverage"] = disclosure_coverage(
            args.fix5m_records, arms
        )
    if args.out:
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    compact = dict(result)
    compact["paired_official_paraphrase"] = {
        k: v for k, v in paired.items() if k != "per_prompt"
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    if args.out:
        print(f"\nNew audit written: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
