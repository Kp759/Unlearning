#!/usr/bin/env python3
"""Build an MCF locked training view that mirrors the registered best-run pipeline.

The original source is NEVER modified and is used for final evaluation.
The training-visible full-size copy preserves record order and requested-rewrite
prompts, locks paraphrase/neighborhood/generation probes, and swaps only answer
semantics so the historical best-run objective suppresses ORIGINAL target_true:

    training target_new  <- ORIGINAL target_true   (sensitive)
    training target_true <- ORIGINAL target_new    (reference)

Because record order is preserved, official MCF sampling selects exactly the
same forget facts as the original source for every seed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from mcf_sampling import sample_official_mcf_records

LOCKED_FIELDS = ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts")
PROTOCOL = "mcf_best_run_target_true_sensitive_locked_v1"


def sha256_bytes(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def normalize_rr(record: Mapping[str, Any]) -> Dict[str, Any]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        if len(rr) != 1:
            raise ValueError("Expected exactly one requested_rewrite")
        rr = rr[0]
    if not isinstance(rr, Mapping):
        raise ValueError("requested_rewrite must be a mapping")
    return copy.deepcopy(dict(rr))


def swap_and_lock(record: Mapping[str, Any], index: int) -> Dict[str, Any]:
    out = copy.deepcopy(dict(record))
    rr = normalize_rr(record)
    old_new = copy.deepcopy(rr.get("target_new"))
    old_true = copy.deepcopy(rr.get("target_true"))
    if not isinstance(old_new, Mapping) or not old_new.get("str"):
        raise ValueError(f"record {index} lacks target_new.str")
    if not isinstance(old_true, Mapping) or not old_true.get("str"):
        raise ValueError(f"record {index} lacks target_true.str")
    rr["target_new"] = old_true
    rr["target_true"] = old_new
    for field in LOCKED_FIELDS:
        rr[field] = [] if field in rr else rr.get(field, [])
        out[field] = []
    out["requested_rewrite"] = rr
    return out


def selected_indices(data: Sequence[Dict[str, Any]], forget_num: int, retain_num: int, seed: int) -> Tuple[list[int], list[int]]:
    idx = {id(r): i for i, r in enumerate(data)}
    forget, retain = sample_official_mcf_records(
        data, forget_num=forget_num, retain_num=retain_num, seed=seed, strict=True
    )
    return [idx[id(r)] for r in forget], [idx[id(r)] for r in retain]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    a = p.parse_args()
    if a.forget_num <= 0 or a.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    src = Path(a.mcf_path).resolve()
    src_bytes = src.read_bytes()
    raw = json.loads(src_bytes)
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    visible = [swap_and_lock(r, i) for i, r in enumerate(raw)]
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    visible_path = out / "repair_visible_mcf_target_true_sensitive.json"
    text = json.dumps(visible, ensure_ascii=False, indent=2) + "\n"
    visible_path.write_text(text, encoding="utf-8")

    per_seed = []
    for seed in a.seeds:
        sf, sr = selected_indices(raw, a.forget_num, a.retain_num, seed)
        vf, vr = selected_indices(visible, a.forget_num, a.retain_num, seed)
        if sf != vf or sr != vr:
            raise AssertionError(f"swapped view changed official sampling for seed {seed}")
        per_seed.append({
            "seed": int(seed),
            "forget_record_indices": sf,
            "retain_record_indices": sr,
            "forget_case_ids": [raw[i].get("case_id", i) for i in sf],
            "retain_case_ids": [raw[i].get("case_id", i) for i in sr],
        })

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "source_dataset": str(src),
        "training_visible_dataset": str(visible_path),
        "source_sha256": sha256_bytes(src_bytes),
        "training_visible_sha256": sha256_bytes(text.encode("utf-8")),
        "dataset_size": len(raw),
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(a.forget_num),
            "retain_num": int(a.retain_num),
            "seeds": [int(x) for x in a.seeds],
        },
        "target_semantics": {
            "sensitive": "ORIGINAL target_true",
            "reference": "ORIGINAL target_new",
            "training_target_new": "ORIGINAL target_true",
            "training_target_true": "ORIGINAL target_new",
            "final_evaluation_uses_original_unswapped_source": True,
        },
        "data_roles": {
            "training_visible": ["requested_rewrite prompt", "sensitive answer", "reference answer"],
            "evaluation_only": list(LOCKED_FIELDS),
            "same_underlying_forget_facts_at_final_eval": True,
            "prompt_level_holdout": True,
        },
        "per_seed": per_seed,
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("target-true-sensitive best-run training view:", visible_path)
    print("split manifest:", manifest_path)
    print("mapping: training target_new=ORIGINAL target_true (sensitive)")
    print("mapping: training target_true=ORIGINAL target_new (reference)")


if __name__ == "__main__":
    main()
