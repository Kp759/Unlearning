#!/usr/bin/env python3
"""Build the canonical SURE-LM locked MQuAKE adapter split.

MQuAKE is sampled at the instance level (50 forget / 1000 retain by default),
then requested_rewrite facts are flattened only after sampling.  The
training-visible artifact exposes only direct prompt/subject/target_true fields.
It contains no target_new/Unknown target, atomic natural-language question,
multi-hop question, retain example, or PPL input.

The canonical shared SURE engine consumes one record per direct fact.  Therefore
this manifest records both the official MQuAKE source-instance count and the
flattened direct-fact count.  Compatibility keys ``sampling.forget_num`` and
``sampling.forget_case_ids`` deliberately refer to flattened training-visible
facts, while ``forget_num_instances`` remains the official sampling unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import mquake_zero_unlearn_official_eval as mquake


PROTOCOL = "mquake_sure_canonical_locked_direct_only"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def direct_only_atomic_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    rewrite = record.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("flattened MQuAKE record lacks requested_rewrite")
    target_true = rewrite.get("target_true")
    if not isinstance(target_true, Mapping) or not str(target_true.get("str", "")):
        raise ValueError("MQuAKE target_true is missing")
    return {
        "case_id": int(record["case_id"]),
        "mquake_case_id": int(record["mquake_case_id"]),
        "source_index": int(record["source_index"]),
        "rewrite_index": int(record["rewrite_index"]),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "relation_id": rewrite.get("relation_id"),
            "target_true": {"str": str(target_true["str"])},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def assert_locked(records: List[Dict[str, Any]]) -> None:
    forbidden_top = {
        "atomic_gen_prompt",
        "multihop_questions",
        "multihop_answer",
        "multihop_answer_alias",
        "multihop_new_answer",
        "multihop_new_answer_alias",
        "questions",
        "question",
    }
    for i, record in enumerate(records):
        leaked = forbidden_top.intersection(record)
        if leaked:
            raise AssertionError(f"training-visible record {i} leaked {sorted(leaked)}")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise AssertionError(f"training-visible record {i} lacks requested_rewrite")
        if "target_new" in rr or "mquake_target_new" in rr or "question" in rr:
            raise AssertionError(f"training-visible record {i} leaked held-out/replacement fields")
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise AssertionError(f"training-visible record {i} exposes held-out probes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50, help="official MQuAKE source instances")
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--mquake-url", default=mquake.MQUAKE_URL)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    source_path = mquake.download_mquake(Path(a.mquake_path).resolve(), url=a.mquake_url)
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or len(raw) != 3000 or not all(isinstance(x, dict) for x in raw):
        raise ValueError("expected the pinned 3000-instance MQuAKE-CF-3k-v2 JSON list")

    forget_pairs, retain_pairs = mquake.sample_zerounlearn_instances(
        raw, forget_num=a.forget_num, retain_num=a.retain_num, seed=a.seed, strict=True
    )
    flattened = mquake.flatten_sampled_instances(forget_pairs)
    visible = [direct_only_atomic_record(record) for record in flattened]
    assert_locked(visible)

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    visible_path = out / "training_visible_forget.json"
    visible_text = json.dumps(visible, ensure_ascii=False, indent=2) + "\n"
    visible_path.write_text(visible_text, encoding="utf-8")

    half = len(raw) // 2
    forget_indices = [int(i) for i, _ in forget_pairs]
    retain_indices = [int(i) for i, _ in retain_pairs]
    if set(forget_indices) & set(retain_indices):
        raise AssertionError("forget/retain source instances overlap")
    if any(i < half for i in forget_indices) or any(i >= half for i in retain_indices):
        raise AssertionError("MQuAKE sample escaped the ZeroUnlearn half-pool rule")
    visible_sources = sorted({int(r["source_index"]) for r in visible})
    if visible_sources != sorted(forget_indices):
        raise AssertionError("flattened direct facts do not match sampled forget instances")

    atomic_ids = [int(r["case_id"]) for r in visible]
    manifest = {
        "schema_version": 2,
        "protocol": PROTOCOL,
        "canonical_engine": "SURE-LM shared MCF/ZsRE architecture",
        "canonical_adapter": "target_true-sensitive/top1-margin (same semantics as ZsRE)",
        "benchmark_dataset": "mquake",
        "seed": int(a.seed),
        "source_dataset": str(source_path),
        "source_revision": mquake.MQUAKE_REV,
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_forget": str(visible_path),
        "training_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
        "dataset_size_instances": len(raw),
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {"start": half, "stop_exclusive": len(raw), "size": len(raw) - half},
        },
        "sampling": {
            "implementation": "sample_zerounlearn_instances",
            "unit": "MQuAKE instance before requested_rewrite flattening",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num_instances": int(a.forget_num),
            "retain_num_instances": int(a.retain_num),
            "forget_source_indices": forget_indices,
            "retain_source_indices": retain_indices,
            "forget_atomic_fact_count": len(visible),
            "forget_atomic_case_ids": atomic_ids,
            # Shared Stage1/2 compatibility: the engine sees flattened direct facts.
            "forget_num": len(visible),
            "forget_case_ids": atomic_ids,
        },
        "data_roles": {
            "stage1_visible": ["sampled forget requested_rewrite prompt/subject/target_true"],
            "stage2_visible": ["same direct facts only"],
            "evaluation_only": [
                "1000 sampled retain instances",
                "atomic natural-language questions / AtomicGen",
                "instance-level multi-hop questions",
                "benchmark counterfactual target_new",
                "PPL corpus",
            ],
            "target_new_visible_to_training": False,
            "same_underlying_forget_instances_at_final_eff_eval": True,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("canonical MQuAKE training-visible direct facts:", visible_path)
    print("split manifest:", manifest_path)
    print(
        f"seed={a.seed}: source forget instances={a.forget_num}, "
        f"flattened direct facts={len(visible)}, retain evaluation instances={a.retain_num}"
    )


if __name__ == "__main__":
    main()
