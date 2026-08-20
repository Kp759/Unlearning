#!/usr/bin/env python3
"""Build the canonical direct-only split for guarded two-stage SURE-LM.

MCF and ZsRE use the same data roles:

* training sees exactly the sampled direct forget prompts and sensitive answers;
* no replacement/reference answer is exposed;
* no benchmark retain example is exposed to either training stage;
* paraphrases, locality/neighborhood probes, generation prompts, and PPL text
  remain evaluation-only;
* a prompt-only copy of the official retain sample is written for an optional
  post-training exact-KL audit.  The learner has no argument that can read it.

For MCF, original ``target_true`` is copied into the canonical SURE
``target_new`` sensitive slot.  Original MCF ``target_new`` is never written to
the training-visible file.  ZsRE keeps original ``target_true`` in place.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from mcf_sampling import sample_official_mcf_records
import zsre_zero_unlearn_official_eval as zsre


PROTOCOL = "sure_exact_constrained_residual_stage2_v5"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_mcf_rewrite(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rewrite = record.get("requested_rewrite")
    if isinstance(rewrite, list):
        if len(rewrite) != 1:
            raise ValueError("Expected exactly one MCF requested_rewrite entry")
        rewrite = rewrite[0]
    if not isinstance(rewrite, Mapping):
        raise ValueError("MCF requested_rewrite must be a mapping")
    return rewrite


def mcf_direct_sensitive_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    rewrite = normalize_mcf_rewrite(raw)
    sensitive = rewrite.get("target_true", {}).get("str")
    if not sensitive:
        raise ValueError(f"MCF record {case_id} lacks original target_true")
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            # The shared SURE adapter treats target_new as MCF's sensitive slot.
            "target_new": {"str": str(sensitive)},
        },
        "semantic_adapter": {
            "original_sensitive_field": "target_true",
            "canonical_sensitive_slot": "target_new",
            "original_reference_field": "target_new",
            "reference_answer_exposed_to_training": False,
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def zsre_direct_sensitive_record(
    raw: Mapping[str, Any], case_id: int
) -> Dict[str, Any]:
    required = ("src", "subject", "answers")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"ZsRE record {case_id} is missing fields: {missing}")
    answers = raw["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"ZsRE record {case_id} has no original answer")
    subject = str(raw["subject"])
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(raw["src"]).replace(subject, "{}"),
            "subject": subject,
            "target_true": {"str": str(answers[0])},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def prompt_only_retain_record(
    dataset: str, raw: Mapping[str, Any], case_id: int
) -> Dict[str, Any]:
    if dataset == "mcf":
        rewrite = normalize_mcf_rewrite(raw)
        prompt = str(rewrite["prompt"])
        subject = str(rewrite["subject"])
    elif dataset == "zsre":
        subject = str(raw["subject"])
        prompt = str(raw["src"]).replace(subject, "{}")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return {
        "case_id": int(case_id),
        "requested_rewrite": {"prompt": prompt, "subject": subject},
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
        "data_role": "post_training_exact_kl_audit_only",
    }


def assert_training_view_locked(
    dataset: str, records: Sequence[Mapping[str, Any]]
) -> None:
    sensitive_field = "target_new" if dataset == "mcf" else "target_true"
    forbidden_field = "target_true" if dataset == "mcf" else "target_new"
    for position, record in enumerate(records):
        for field in (
            "paraphrase_prompts",
            "neighborhood_prompts",
            "attribute_prompts",
            "generation_prompts",
        ):
            if record.get(field):
                raise AssertionError(
                    f"training-visible record {position} exposes {field}"
                )
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise AssertionError(
                f"training-visible record {position} lacks requested_rewrite"
            )
        if not rewrite.get(sensitive_field, {}).get("str"):
            raise AssertionError(
                f"training-visible record {position} lacks {sensitive_field}"
            )
        if forbidden_field in rewrite:
            raise AssertionError(
                f"training-visible record {position} exposes {forbidden_field}"
            )


def sample_records(
    dataset: str,
    raw: Sequence[Mapping[str, Any]],
    *,
    forget_num: int,
    retain_eval_num: int,
    seed: int,
) -> Tuple[List[Tuple[int, Mapping[str, Any]]], List[Tuple[int, Mapping[str, Any]]]]:
    if dataset == "mcf":
        identity = {id(record): index for index, record in enumerate(raw)}
        forget, retain = sample_official_mcf_records(
            raw,
            forget_num=forget_num,
            retain_num=retain_eval_num,
            seed=seed,
            strict=True,
        )
        return (
            [(identity[id(record)], record) for record in forget],
            [(identity[id(record)], record) for record in retain],
        )
    if dataset == "zsre":
        forget, retain = zsre.sample_official_zsre_raw_records(
            raw,
            forget_num=forget_num,
            retain_num=retain_eval_num,
            seed=seed,
            strict=True,
        )
        return list(forget), list(retain)
    raise ValueError(f"Unsupported dataset: {dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-eval-num", type=int, default=1000)
    parser.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.retain_eval_num <= 0:
        raise ValueError("forget-num and retain-eval-num must be positive")

    source = Path(args.dataset_path).resolve()
    if args.dataset == "zsre":
        source = zsre.download_zsre(source, url=args.zsre_url)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("Source dataset must be a JSON list of objects")

    forget_pairs, retain_pairs = sample_records(
        args.dataset,
        raw,
        forget_num=args.forget_num,
        retain_eval_num=args.retain_eval_num,
        seed=args.seed,
    )
    forget_ids = [int(index) for index, _ in forget_pairs]
    retain_ids = [int(index) for index, _ in retain_pairs]
    half = len(raw) // 2
    if set(forget_ids) & set(retain_ids):
        raise AssertionError("official forget and retain samples overlap")
    if any(index < half for index in forget_ids):
        raise AssertionError("forget sample escaped the official second-half pool")
    if any(index >= half for index in retain_ids):
        raise AssertionError("retain sample escaped the official first-half pool")

    converter = (
        mcf_direct_sensitive_record
        if args.dataset == "mcf"
        else zsre_direct_sensitive_record
    )
    forget_visible = [converter(record, index) for index, record in forget_pairs]
    retain_audit = [
        prompt_only_retain_record(args.dataset, record, index)
        for index, record in retain_pairs
    ]
    assert_training_view_locked(args.dataset, forget_visible)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    forget_path = output_dir / "training_visible_forget.json"
    retain_audit_path = output_dir / "evaluation_only_retain_prompts.json"
    forget_text = json.dumps(forget_visible, indent=2, ensure_ascii=False) + "\n"
    retain_text = json.dumps(retain_audit, indent=2, ensure_ascii=False) + "\n"
    forget_path.write_text(forget_text, encoding="utf-8")
    retain_audit_path.write_text(retain_text, encoding="utf-8")

    if args.dataset == "mcf":
        target_semantics = {
            "original_sensitive_field": "target_true",
            "training_sensitive_slot": "target_new",
            "forbidden_training_answer_fields": ["target_true"],
            "original_reference_field": "target_new",
            "reference_answer_exposed_to_training": False,
            "final_evaluation_uses_original_unswapped_fields": True,
        }
    else:
        target_semantics = {
            "original_sensitive_field": "target_true",
            "training_sensitive_slot": "target_true",
            "forbidden_training_answer_fields": ["target_new"],
            "neutral_or_replacement_target_visible": False,
        }

    forget_sha256 = sha256_bytes(forget_text.encode("utf-8"))
    manifest = {
        "schema_version": 3,
        "protocol": PROTOCOL,
        "metric_schema": (
            "mcf_target_true_sensitive_v2" if args.dataset == "mcf" else None
        ),
        "dataset": args.dataset,
        "seed": int(args.seed),
        "source_dataset": str(source),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_forget": str(forget_path),
        "training_visible_forget_sha256": forget_sha256,
        "training_visible_sha256": forget_sha256,
        "evaluation_only_retain_prompts": str(retain_audit_path),
        "evaluation_only_retain_prompts_sha256": sha256_bytes(
            retain_text.encode("utf-8")
        ),
        "dataset_size": len(raw),
        "target_semantics": target_semantics,
        "learner_adapter_contract": {
            "schema_version": 1,
            "sensitive_answer_field": target_semantics["training_sensitive_slot"],
            "forbidden_answer_fields": target_semantics[
                "forbidden_training_answer_fields"
            ],
            "direct_prompt_field": "requested_rewrite.prompt",
            "subject_field": "requested_rewrite.subject",
            "architecture_parameters_are_dataset_independent": True,
        },
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {
                "start": half,
                "stop_exclusive": len(raw),
                "size": len(raw) - half,
            },
        },
        "sampling": {
            "implementation": (
                "sample_official_mcf_records"
                if args.dataset == "mcf"
                else "sample_official_zsre_raw_records"
            ),
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(args.forget_num),
            "benchmark_retain_train_num": 0,
            "retain_eval_num": int(args.retain_eval_num),
            "forget_case_ids": forget_ids,
            "retain_eval_case_ids": retain_ids,
            "retain_case_ids": retain_ids,
            "forget_retain_overlap": 0,
        },
        "data_roles": {
            "stage1_visible": ["direct forget prompt", "sensitive answer"],
            "stage2_visible": [
                "all direct forget prompts already visible to Stage 1",
                "same sensitive answers",
                "Stage-1 failure/success partition computed without held-out data",
            ],
            "benchmark_retain_examples_visible_to_training": 0,
            "replacement_or_reference_answers_visible_to_training": False,
            "evaluation_only": [
                "forget paraphrases/rephrases",
                "forget locality/neighborhood prompts",
                "forget generation prompts",
                "official retain sample",
                "PPL text",
            ],
            "evaluation_only_retain_prompt_file_purpose": (
                "post-training exact sparse-row KL audit only"
            ),
            "heldout_probes_visible_during_training": False,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("guarded SURE training file:", forget_path)
    print("post-training retain audit file:", retain_audit_path)
    print("split manifest:", manifest_path)
    print(
        f"dataset={args.dataset} seed={args.seed}: "
        f"train={len(forget_visible)} forget + 0 benchmark retain; "
        f"post-training retain audit={len(retain_audit)}"
    )


if __name__ == "__main__":
    main()
