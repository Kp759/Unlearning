#!/usr/bin/env python3
"""Add retention-only training contexts without copying validation or MCF probes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from static_overlap_data import load_bundle, validate_bundle


# Fixed independently of validation failures and official evaluation prompts.
PREFIXES = ("Provide the missing factual information.\n", "Reference question: ")
VERSION = "static_retention_contexts_v1"


def retention_views(row, facts, abstention):
    retained = [deepcopy(s) for s in row["spans"] if facts[s["fact_id"]]["role"] == "retain"]
    if not retained:
        return []
    views = [(row["completion"], retained)]
    if not abstention or all(facts[s["fact_id"]]["role"] == "retain" for s in row["spans"]):
        return views
    # Preserve companion answers after both true and neutral forget completions.
    # Only RETAIN spans are labeled; no added forget/abstention supervision.
    text, spans, cursor = "", [], 0
    for span in row["spans"]:
        text += row["completion"][cursor:span["start"]]
        if facts[span["fact_id"]]["role"] == "forget":
            text += abstention
            if span["end"] < len(row["completion"]) and not row["completion"][span["end"]].isspace():
                text += " "
        else:
            start = len(text)
            text += row["completion"][span["start"]:span["end"]]
            spans.append({"start": start, "end": len(text), "fact_id": span["fact_id"]})
        cursor = span["end"]
    text += row["completion"][cursor:]
    views.append((text, spans))
    return views


def augment_retention(bundle, abstention="I don't know."):
    facts = validate_bundle(bundle, "training")
    if any(row["id"].startswith("train_retention_aug:") for row in bundle["examples"]):
        raise ValueError("Bundle already contains retention augmentation; use the original training bundle")
    result = deepcopy(bundle)
    provenance = []
    for row in bundle["examples"]:
        if row["split"] != "train" or row.get("role") == "language":
            continue
        for view, (completion, spans) in enumerate(retention_views(row, facts, abstention)):
            for prefix_index, prefix in enumerate(PREFIXES):
                rid = f"train_retention_aug:{row['id']}:{view}:{prefix_index}"
                result["examples"].append({"id": rid, "split": "train", "prompt": prefix + row["prompt"],
                                           "completion": completion, "spans": deepcopy(spans)})
                provenance.append({"id": rid, "source_training_id": row["id"], "prefix_index": prefix_index,
                                   "context": "original_completion" if view == 0 else "neutral_completion"})
    if not provenance:
        raise ValueError("No retention training rows available to augment")
    # Includes prompt/text disjointness checks against untouched validation.
    validate_bundle(result, "training")
    return result, {"version": VERSION, "prefixes": list(PREFIXES), "abstention": abstention,
                    "added_rows": len(provenance), "sources": provenance,
                    "validation_unchanged": True, "official_evaluation_read": False,
                    "added_forget_supervision": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--abstention", default="I don't know.")
    args = parser.parse_args(argv)
    if not args.training_bundle.strip() or not args.out.strip():
        parser.error("Input and output paths must be nonempty")
    output = Path(args.out)
    sidecar = output.with_suffix(".augmentation.json")
    if output.exists() or sidecar.exists():
        parser.error("Output bundle or augmentation report already exists; use a new path")
    bundle, _, source_hash = load_bundle(args.training_bundle)
    augmented, report = augment_retention(bundle, args.abstention)
    raw = (json.dumps(augmented, indent=2) + "\n").encode()
    report.update(source_bundle_sha256=source_hash, output_bundle_sha256=hashlib.sha256(raw).hexdigest(),
                  source_bundle_path=str(Path(args.training_bundle).resolve()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    with sidecar.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"training_bundle": str(output), "report": str(sidecar), "added_retention_rows": report["added_rows"]}, indent=2))


if __name__ == "__main__":
    main()
