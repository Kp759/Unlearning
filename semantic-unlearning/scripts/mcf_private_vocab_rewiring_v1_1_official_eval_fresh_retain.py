#!/usr/bin/env python3
"""Official MCF eval for V1.1 with a fresh, fully disjoint 1000-retain set.

This wrapper preserves the repository's existing ZeroUnlearn-compatible MCF
metrics and the exact seed-1 forget sample, but changes two things required by
V1.1:

1. Load the saved position-preserving private-subject tokenizer route rather
   than silently falling back to plain AutoTokenizer.
2. Sample a fresh first-half retain set after excluding every V1.1 training-
   visible protection record and the previously reserved 1000 official retain
   IDs recorded in the run's split manifest.

The selected fresh retain IDs are serialized next to the result for provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

from transformers import AutoTokenizer as HFAutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_private_vocab_rewiring_v1_1_core as pvcore
from mcf_sampling import sample_official_mcf_records


class _EvalRoutingTokenizer:
    """Forward tokenizer attributes while applying V1.1's subject-ID rewrite."""

    def __init__(self, routed: pvcore.PositionPreservingSubjectTokenizer):
        object.__setattr__(self, "_routed", routed)

    @property
    def base_tokenizer(self):
        return self._routed.base_tokenizer

    @property
    def pad_token(self):
        return self.base_tokenizer.pad_token

    @pad_token.setter
    def pad_token(self, value):
        self.base_tokenizer.pad_token = value

    @property
    def eos_token(self):
        return self.base_tokenizer.eos_token

    @eos_token.setter
    def eos_token(self, value):
        self.base_tokenizer.eos_token = value

    def __len__(self):
        return len(self._routed)

    def __call__(self, *args, **kwargs):
        return self._routed(*args, **kwargs)

    def __getattr__(self, name):
        routed = object.__getattribute__(self, "_routed")
        if hasattr(routed, name):
            return getattr(routed, name)
        return getattr(routed.base_tokenizer, name)


class _RoutingAwareAutoTokenizer:
    @staticmethod
    def from_pretrained(path, *args, **kwargs):
        model_dir = Path(path)
        routing = model_dir / "private_subject_routing.json"
        if routing.is_file():
            routed = pvcore.load_position_preserving_tokenizer(model_dir, HFAutoTokenizer)
            return _EvalRoutingTokenizer(routed)
        return HFAutoTokenizer.from_pretrained(path, *args, **kwargs)


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "protocol" / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("seed", -1)) != 1:
        raise RuntimeError("V1.1 official eval expects the locked seed-1 split")
    return payload


def _excluded_retain_ids(manifest: Mapping[str, Any]) -> set[int]:
    excluded: set[int] = {
        int(value) for value in manifest.get("official_retain_case_ids_only", [])
    }
    case_ids = manifest.get("case_ids", {})
    if not isinstance(case_ids, Mapping):
        raise RuntimeError("split manifest lacks case_ids mapping")
    for values in case_ids.values():
        excluded.update(int(value) for value in values)
    return excluded


def _expected_forget_ids(manifest: Mapping[str, Any]) -> list[int]:
    values = manifest.get("case_ids", {}).get("forget", [])
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", required=True)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--fresh-retain-seed",
        type=int,
        default=700002,
        help="Independent RNG seed for the fresh retain sample.",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    p.add_argument("--skip-ppl", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        raise SystemExit("This official V1.1 evaluation is locked to seed=1, forget=50, retain=1000")

    run_dir = Path(args.run_dir).resolve()
    model_dir = run_dir / "model"
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not (model_dir / "private_subject_routing.json").is_file():
        raise RuntimeError("saved V1.1 checkpoint is missing private_subject_routing.json")

    manifest = _load_manifest(run_dir)
    excluded = _excluded_retain_ids(manifest)
    expected_forget = _expected_forget_ids(manifest)
    if len(expected_forget) != 50:
        raise RuntimeError(f"expected 50 locked forget IDs, found {len(expected_forget)}")

    selected: Dict[str, Any] = {}

    def fresh_official_split(data, unlearn_num, retain_num, seed):
        # Reproduce the exact original forget sample. sample_official_mcf_records
        # samples forget first, so retain_num=0 leaves the forget sequence intact.
        forget_records, _ = sample_official_mcf_records(
            data, unlearn_num, 0, seed, strict=True
        )
        forget_ids = [int(row["case_id"]) for row in forget_records]
        if forget_ids != expected_forget:
            raise RuntimeError(
                "official evaluator forget sample does not match the V1.1 training split"
            )

        half = len(data) // 2
        candidates = [
            row for row in data[:half]
            if int(row["case_id"]) not in excluded
        ]
        if len(candidates) < retain_num:
            raise RuntimeError(
                f"fresh retain pool has only {len(candidates)} records after exclusions; "
                f"need {retain_num}"
            )
        rng = random.Random(int(args.fresh_retain_seed))
        retain_records = rng.sample(candidates, k=retain_num)
        retain_ids = [int(row["case_id"]) for row in retain_records]
        if excluded.intersection(retain_ids):
            raise AssertionError("fresh retain sample overlaps an excluded V1.1 record")
        if len(set(retain_ids)) != retain_num:
            raise AssertionError("fresh retain sample contains duplicate case IDs")

        selected.update(
            {
                "schema_version": 1,
                "protocol": "mcf_v1_1_official_eval_fresh_disjoint_retain",
                "forget_seed": int(seed),
                "fresh_retain_seed": int(args.fresh_retain_seed),
                "forget_case_ids": forget_ids,
                "fresh_retain_case_ids": retain_ids,
                "fresh_retain_count": len(retain_ids),
                "excluded_case_id_count": len(excluded),
                "excluded_previous_official_retain_count": len(
                    manifest.get("official_retain_case_ids_only", [])
                ),
                "excluded_training_visible_counts": {
                    key: len(values)
                    for key, values in manifest.get("case_ids", {}).items()
                },
                "fresh_retain_disjoint_from_all_excluded": True,
            }
        )
        return (
            [official.normalize_record(row) for row in forget_records],
            [official.normalize_record(row) for row in retain_records],
        )

    # Keep all metric code identical; only tokenizer loading and official split
    # sampling are swapped for this V1.1 evaluation.
    official.AutoTokenizer = _RoutingAwareAutoTokenizer
    official.sample_official_split = fresh_official_split

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = official.evaluate_model_dir_official(
        method="private_vocab_rewiring_v1_1_fresh_retain",
        model_dir=model_dir,
        mcf_path=args.mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=out,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode="official",
        dtype=args.dtype,
        device_map=args.device_map,
        skip_ppl=args.skip_ppl,
    )

    if not selected:
        raise RuntimeError("fresh retain sampler was not invoked")
    selection_path = out.with_name(out.stem + "_fresh_retain_manifest.json")
    selection_path.write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )

    summary = official.result_to_comparison_row(result)
    print(json.dumps(summary, indent=2))
    print(f"Official result: {out}")
    print(f"Fresh retain manifest: {selection_path}")


if __name__ == "__main__":
    main()
