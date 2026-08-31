#!/usr/bin/env python3
"""Build V6's frozen two-sided MCF relation grammar.

Seeds 1 and 2 are already consumed and are explicitly development data.
Seeds 3--10 remain reserved: none of their forget or retain records contributes
prompt text to this grammar.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import mcf_normalization_preserving_sidecar_v6_core as core
from mcf_sampling import sample_official_mcf_records


def derive_lexicon(
    data: Sequence[Mapping[str, Any]],
    *,
    source_mcf_sha256: str,
    first_reserved_seed: int = 3,
    last_reserved_seed: int = 10,
) -> Dict[str, Any]:
    positions = {id(row): index for index, row in enumerate(data)}
    reserved: set[int] = set()
    split_counts: dict[str, dict[str, int]] = {}
    for seed in range(int(first_reserved_seed), int(last_reserved_seed) + 1):
        forget, retain = sample_official_mcf_records(data, 50, 1000, seed)
        reserved.update(positions[id(row)] for row in (*forget, *retain))
        split_counts[str(seed)] = {"forget": 50, "retain": 1000}

    frames: dict[str, set[str]] = defaultdict(set)
    development_records = 0
    prompt_instances = 0
    for index, row in enumerate(data):
        if index in reserved:
            continue
        rewrite = row["requested_rewrite"]
        subject = str(rewrite["subject"])
        relation = str(rewrite["relation_id"])
        prompts = [
            str(rewrite["prompt"]).format(subject),
            *[str(item) for item in row.get("paraphrase_prompts", [])],
        ]
        for prompt in prompts:
            for left, right in core.subject_frame_parts(prompt, subject):
                if not right:
                    continue
                frames[core.frame_key(left, right)].add(relation)
                prompt_instances += 1
        development_records += 1

    mapping = {
        key: sorted(relations) for key, relations in sorted(frames.items())
    }
    result: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "mcf_frozen_two_sided_relation_frame_lexicon_v1",
        "derivation": {
            "source_mcf_sha256": str(source_mcf_sha256),
            "source_records": len(data),
            "consumed_development_seeds": [1, 2],
            "consumed_development_status": "not_blind_not_official",
            "reserved_seed_range": [
                int(first_reserved_seed),
                int(last_reserved_seed),
            ],
            "reserved_split_rule": "union_of_official_50_forget_plus_1000_retain_per_seed",
            "reserved_unique_records": len(reserved),
            "development_records_used": development_records,
            "development_prompt_instances_used": prompt_instances,
            "prompt_sources": ["requested_rewrite", "paraphrase_prompts"],
            "neighborhood_prompts_used": False,
            "target_answers_used": False,
            "normalization": "NFKC_lowercase_collapse_whitespace_left_current_clause_plus_right_remainder",
            "reserved_seed_split_counts": split_counts,
        },
        "matching_rule": "exact_left_clause_and_longest_boundary_complete_right_prefix",
        "frame_separator": core.FRAME_SEPARATOR,
        "relation_count": len(
            {
                relation
                for relations in mapping.values()
                for relation in relations
            }
        ),
        "frame_count": len(mapping),
        "ambiguous_frame_count": sum(
            len(relations) > 1 for relations in mapping.values()
        ),
        "frame_to_relation_ids": mapping,
    }
    result["lexicon_sha256"] = core.frame_lexicon_sha256(result)
    core.validate_frame_lexicon(result)
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
    if not isinstance(data, list):
        raise RuntimeError("MCF source must be a JSON list")
    artifact = derive_lexicon(
        data,
        source_mcf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
