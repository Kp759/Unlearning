#!/usr/bin/env python3
"""Build a disjoint direct-only MCF calibration set for context-protected repair.

The calibration set is sampled only from records excluded from BOTH the locked
forget split and the final retain-evaluation split recorded in a seed manifest.
All official paraphrase, neighborhood, and generation prompts are removed.

The calibration records keep ORIGINAL CounterFact semantics unchanged:
target_true is the ordinary factual answer and target_new is the counterfactual
alternative. They are used only to estimate ordinary factual hidden-state
geometry / logit drift, never to optimize forget labels or evaluate metrics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping

LOCKED_FIELDS = ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts")
PROTOCOL = "mcf_disjoint_direct_context_calibration_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_rr(record: Mapping[str, Any], position: int) -> Dict[str, Any]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        if len(rr) != 1:
            raise ValueError(f"record {position}: expected one requested_rewrite")
        rr = rr[0]
    if not isinstance(rr, Mapping):
        raise ValueError(f"record {position}: requested_rewrite must be a mapping")
    return copy.deepcopy(dict(rr))


def direct_only(record: Mapping[str, Any], position: int) -> Dict[str, Any]:
    out = copy.deepcopy(dict(record))
    rr = _normalize_rr(record, position)
    tn = rr.get("target_new")
    tt = rr.get("target_true")
    if not isinstance(tn, Mapping) or not tn.get("str"):
        raise ValueError(f"record {position} lacks target_new.str")
    if not isinstance(tt, Mapping) or not tt.get("str"):
        raise ValueError(f"record {position} lacks target_true.str")
    for field in LOCKED_FIELDS:
        out[field] = []
        if field in rr:
            rr[field] = []
    out["requested_rewrite"] = rr
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--seed-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--calibration-num", type=int, default=128)
    p.add_argument("--calibration-seed-offset", type=int, default=100003)
    a = p.parse_args()

    if a.calibration_num <= 0:
        raise ValueError("--calibration-num must be positive")

    mcf_path = Path(a.mcf_path).resolve()
    manifest_path = Path(a.seed_manifest).resolve()
    raw = json.loads(mcf_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    expected_sha = manifest.get("source_sha256")
    actual_sha = _sha256(mcf_path)
    if expected_sha and str(expected_sha) != actual_sha:
        raise RuntimeError("seed manifest source_sha256 does not match --mcf-path")

    sampling = manifest.get("sampling", {})
    forget_indices = {int(x) for x in sampling.get("forget_record_indices", [])}
    retain_indices = {int(x) for x in sampling.get("retain_record_indices", [])}
    if not forget_indices:
        raise RuntimeError("seed manifest does not contain forget_record_indices")
    if not retain_indices:
        raise RuntimeError("seed manifest does not contain retain_record_indices")
    if forget_indices & retain_indices:
        raise RuntimeError("seed manifest forget/retain sets unexpectedly overlap")

    excluded = forget_indices | retain_indices
    candidates = [i for i in range(len(raw)) if i not in excluded]
    if a.calibration_num > len(candidates):
        raise ValueError(
            f"requested {a.calibration_num} calibration records but only "
            f"{len(candidates)} records remain after forget+retain exclusion"
        )

    seed = int(manifest.get("seed", 0))
    rng_seed = seed + int(a.calibration_seed_offset)
    selected = sorted(random.Random(rng_seed).sample(candidates, k=a.calibration_num))
    if set(selected) & excluded:
        raise AssertionError("calibration sampling is not disjoint")

    calibration = [direct_only(raw[i], i) for i in selected]
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": seed,
        "rng_seed": rng_seed,
        "source_dataset": str(mcf_path),
        "source_sha256": actual_sha,
        "seed_manifest": str(manifest_path),
        "calibration_json": str(out),
        "calibration_num": len(selected),
        "calibration_record_indices": selected,
        "calibration_case_ids": [raw[i].get("case_id", i) for i in selected],
        "excluded_forget_record_indices": sorted(forget_indices),
        "excluded_final_retain_record_indices": sorted(retain_indices),
        "disjoint_from_forget": True,
        "disjoint_from_final_retain_eval": True,
        "target_semantics": {
            "calibration_is_original_unswapped_mcf": True,
            "target_true": "ordinary factual answer",
            "target_new": "CounterFact alternative",
        },
        "data_roles": {
            "used_for": [
                "retain-aware hidden geometry",
                "selected-sensitive-row logit-drift penalty",
            ],
            "not_used_for": [
                "forget labels",
                "official paraphrase evaluation",
                "official neighborhood evaluation",
                "official generation evaluation",
                "final retain evaluation",
            ],
            "evaluation_fields_removed": list(LOCKED_FIELDS),
        },
    }
    manifest_out = out.with_suffix(out.suffix + ".manifest.json")
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("Calibration JSON:", out)
    print("Calibration manifest:", manifest_out)
    print("Records:", len(selected))
    print("Disjoint from forget and final retain evaluation: yes")


if __name__ == "__main__":
    main()
