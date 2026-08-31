#!/usr/bin/env python3
"""Build the shared MCF relation-suffix grammar while reserving seeds 1-10."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import mcf_exact_subject_target_sidecar_v5_core as core
from mcf_sampling import sample_official_mcf_records


def derive_lexicon(
    data: Sequence[Mapping[str, Any]],
    *,
    source_mcf_sha256: str,
    first_reserved_seed: int = 1,
    last_reserved_seed: int = 10,
) -> Dict[str, Any]:
    positions = {id(row): index for index, row in enumerate(data)}
    reserved: set[int] = set()
    split_counts = {}
    for seed in range(int(first_reserved_seed), int(last_reserved_seed) + 1):
        forget, retain = sample_official_mcf_records(data, 50, 1000, seed)
        reserved.update(positions[id(row)] for row in (*forget, *retain))
        split_counts[str(seed)] = {"forget": 50, "retain": 1000}

    lexicon: dict[str, set[str]] = defaultdict(set)
    development_records = 0
    for index, row in enumerate(data):
        if index in reserved:
            continue
        rewrite = row["requested_rewrite"]
        subject = str(rewrite["subject"])
        relation = str(rewrite["relation_id"])
        prompts = [
            str(rewrite["prompt"]).format(subject),
            *[str(value) for value in row.get("paraphrase_prompts", [])],
        ]
        for prompt in prompts:
            remainder = core.complete_subject_remainder(prompt, subject)
            if remainder is None:
                continue
            suffix = core.normalize_relation_suffix(remainder)
            if suffix:
                lexicon[suffix].add(relation)
        development_records += 1
    mapping = {
        suffix: sorted(relations)
        for suffix, relations in sorted(lexicon.items())
    }
    result = {
        "schema_version": 1,
        "kind": "mcf_frozen_relation_suffix_lexicon_v1",
        "derivation": {
            "source_mcf_sha256": str(source_mcf_sha256),
            "source_records": len(data),
            "reserved_seed_range": [
                int(first_reserved_seed),
                int(last_reserved_seed),
            ],
            "reserved_split_rule": "union_of_official_50_forget_plus_1000_retain_per_seed",
            "reserved_unique_records": len(reserved),
            "development_records_used": development_records,
            "prompt_sources": ["requested_rewrite", "paraphrase_prompts"],
            "neighborhood_prompts_used": False,
            "target_answers_used": False,
            "normalization": "NFKC_lowercase_collapse_whitespace_post_complete_subject",
            "seed_split_counts": split_counts,
        },
        "matching_rule": "longest_boundary_complete_prefix_relation_membership",
        "relation_count": len(
            {
                relation
                for relations in mapping.values()
                for relation in relations
            }
        ),
        "suffix_count": len(mapping),
        "ambiguous_suffix_count": sum(
            len(relations) > 1 for relations in mapping.values()
        ),
        "suffix_to_relation_ids": mapping,
    }
    result["lexicon_sha256"] = core.relation_lexicon_sha256(result)
    core.validate_relation_lexicon(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    source = Path(args.mcf_path).resolve()
    output = Path(args.output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text(encoding="utf-8"))
    value = derive_lexicon(
        data,
        source_mcf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
