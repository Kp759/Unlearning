#!/usr/bin/env python3
"""Build MCF robust prompt sets through an explicit dataset-adapter pipeline.

Pipeline per locked training-visible fact:

    direct prompt
       -> candidate generation
       -> relation-slot classifier
       -> answer-type compatibility filter
       -> semantic equivalence judge
       -> keep 3--8 safe surrogates only

If fewer than ``--min-surrogates`` survive, that fact becomes ``direct_only``;
we never pad with weak or semantically shifted prompts.  The resulting artifact
is an adapter output consumed by the common Gen-aware SURE Stage 2.

Data-access contract:
  * generator/classifiers/judges see only subject + locked direct prompt +
    generated candidate text;
  * target_true/target_new are used only after generation for literal
    answer-occurrence rejection and are never placed in an LLM prompt;
  * official MCF paraphrases, neighborhoods, retain examples, and PPL text are
    never read.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import build_mcf_surrogate_paraphrases as generation
import mcf_dataset_adapter_relation_slot as relation_adapter
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_surrogate_answer_guard as answer_guard
import mcf_surrogate_semantic_validator_v3 as semantic
import sure_stage2_sparse_repair as stage2


ARTIFACT_PROTOCOL = "mcf_direct_only_robust_prompt_adapter_v6"
ADAPTER_PROTOCOL = "mcf_relation_slot_answer_type_semantic_adapter_v1"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--min-surrogates", type=int, default=3)
    p.add_argument("--max-surrogates", type=int, default=8)
    p.add_argument("--generator-model-path", required=True)
    p.add_argument("--adapter-model-path", default=None,
                   help="Defaults to generator model; profiles relation/answer slots")
    p.add_argument("--semantic-model-path", default=None,
                   help="Defaults to adapter model; final semantic judge")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--generation-rounds", type=int, default=10)
    p.add_argument("--generation-oversample", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--classifier-batch-size", type=int, default=8)
    p.add_argument("--classifier-max-new-tokens", type=int, default=128)
    p.add_argument("--judge-batch-size", type=int, default=8)
    p.add_argument("--judge-max-new-tokens", type=int, default=96)
    a = p.parse_args(argv)
    if a.forget_num <= 0:
        p.error("forget-num must be positive")
    if a.min_surrogates <= 0 or a.max_surrogates <= 0:
        p.error("min/max-surrogates must be positive")
    if a.min_surrogates > a.max_surrogates:
        p.error("min-surrogates must be <= max-surrogates")
    if a.generation_rounds <= 0 or a.generation_oversample <= 0:
        p.error("generation-rounds/oversample must be positive")
    if a.max_new_tokens <= 0 or a.classifier_batch_size <= 0 or a.judge_batch_size <= 0:
        p.error("token limits and batch sizes must be positive")
    if a.classifier_max_new_tokens <= 0 or a.judge_max_new_tokens <= 0:
        p.error("classifier/judge max-new-tokens must be positive")
    if not math.isfinite(a.temperature) or a.temperature <= 0:
        p.error("temperature must be finite and positive")
    if not math.isfinite(a.top_p) or not (0 < a.top_p <= 1):
        p.error("top-p must be in (0,1]")
    return a


def _dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[str(name)]


def _load_model(path: str, dtype: str, device_map: str):
    kwargs: Dict[str, Any] = {"torch_dtype": _dtype(dtype)}
    if device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-map single requires CUDA")
        model = model.to("cuda")
    model.eval()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _same_path(a: str, b: str) -> bool:
    return str(Path(a).resolve()) == str(Path(b).resolve())


def _pre_filter_candidates(
    raw_candidates: Sequence[str],
    *,
    subject: str,
    direct_prompt: str,
    answers: Sequence[str],
    seen: set[str],
) -> tuple[List[str], List[Dict[str, Any]]]:
    """Cheap lexical/structural gate before any classifier call."""
    audit: List[Dict[str, Any]] = []
    values = generation._validated_unique(
        list(raw_candidates),
        subject=subject,
        direct_prompt=direct_prompt,
        answers=answers,
        limit=max(1, len(raw_candidates) + 1),
    )
    out: List[str] = []
    for candidate in values:
        key = generation._normalize_cmp(candidate)
        if key in seen:
            audit.append({"candidate": candidate, "stage": "prefilter", "accepted": False, "reason": "duplicate_across_rounds"})
            continue
        seen.add(key)
        structural = semantic.structural_rejection_reason(direct_prompt, candidate)
        if structural is not None:
            audit.append({"candidate": candidate, "stage": "prefilter", "accepted": False, "reason": structural})
            continue
        out.append(candidate)
    return out, audit


def _semantic_reason(result: Dict[str, Any]) -> str:
    return semantic.rejection_reason(result)


def _hist(audit: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(str(x.get("reason", "unknown")) for x in audit if not x.get("accepted", False)))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    a = parse_args(argv)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)

    adapter_path = a.adapter_model_path or a.generator_model_path
    semantic_path = a.semantic_model_path or adapter_path

    generator_model, generator_tok = _load_model(a.generator_model_path, a.dtype, a.device_map)
    if _same_path(adapter_path, a.generator_model_path):
        adapter_model, adapter_tok = generator_model, generator_tok
    else:
        adapter_model, adapter_tok = _load_model(adapter_path, a.dtype, a.device_map)
    if _same_path(semantic_path, adapter_path):
        semantic_model, semantic_tok = adapter_model, adapter_tok
    elif _same_path(semantic_path, a.generator_model_path):
        semantic_model, semantic_tok = generator_model, generator_tok
    else:
        semantic_model, semantic_tok = _load_model(semantic_path, a.dtype, a.device_map)

    artifact_records: List[Dict[str, Any]] = []
    receipt_records: List[Dict[str, Any]] = []
    augmented_count = 0
    direct_only_count = 0

    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        answers = [str(rr["target_true"]["str"]), str(rr["target_new"]["str"])]

        direct_profile = relation_adapter.profile_direct(
            adapter_model,
            adapter_tok,
            subject=subject,
            direct_prompt=direct_prompt,
            max_new_tokens=int(a.classifier_max_new_tokens),
        )

        accepted: List[str] = []
        accepted_keys: set[str] = set()
        seen: set[str] = {generation._normalize_cmp(direct_prompt)}
        audit: List[Dict[str, Any]] = []
        raw_rounds: List[Dict[str, Any]] = []

        if direct_profile.get("safe_to_augment", False):
            for round_idx in range(int(a.generation_rounds)):
                if len(accepted) >= int(a.max_surrogates):
                    break
                remaining = int(a.max_surrogates) - len(accepted)
                requested = max(
                    int(a.max_surrogates),
                    remaining * int(a.generation_oversample),
                )
                seed = (
                    int(a.seed) * 1000003
                    + int(position) * 9973
                    + int(round_idx) * 104729
                    + 17
                )
                generated = generation.generate_local_surrogates(
                    generator_model,
                    generator_tok,
                    subject=subject,
                    direct_prompt=direct_prompt,
                    count=requested,
                    seed=seed,
                    max_new_tokens=int(a.max_new_tokens),
                    temperature=float(a.temperature),
                    top_p=float(a.top_p),
                )
                raw_rounds.append({
                    "round": int(round_idx),
                    "seed": int(seed),
                    "raw_generations": generated["raw"],
                })
                candidates, pre_audit = _pre_filter_candidates(
                    generated["accepted"],
                    subject=subject,
                    direct_prompt=direct_prompt,
                    answers=answers,
                    seen=seen,
                )
                audit.extend(pre_audit)
                if not candidates:
                    continue

                relation_results = relation_adapter.profile_candidates(
                    adapter_model,
                    adapter_tok,
                    subject=subject,
                    direct_prompt=direct_prompt,
                    direct_profile=direct_profile,
                    candidates=candidates,
                    batch_size=int(a.classifier_batch_size),
                    max_new_tokens=int(a.classifier_max_new_tokens),
                )

                semantic_candidates: List[str] = []
                relation_by_candidate: Dict[str, Dict[str, Any]] = {}
                for result in relation_results:
                    candidate = str(result["candidate"])
                    relation_by_candidate[candidate] = result
                    if not result["relation_pass"]:
                        audit.append({
                            "candidate": candidate,
                            "stage": "relation_slot_classifier",
                            "accepted": False,
                            "reason": "relation_slot_mismatch_or_constraint",
                            "detail": result,
                        })
                        continue
                    if not result["answer_type_pass"]:
                        audit.append({
                            "candidate": candidate,
                            "stage": "answer_type_filter",
                            "accepted": False,
                            "reason": "answer_type_incompatible",
                            "detail": result,
                        })
                        continue
                    semantic_candidates.append(candidate)

                if not semantic_candidates:
                    continue

                semantic_results = semantic.validate_candidates(
                    semantic_model,
                    semantic_tok,
                    subject=subject,
                    direct_prompt=direct_prompt,
                    candidates=semantic_candidates,
                    batch_size=int(a.judge_batch_size),
                    max_new_tokens=int(a.judge_max_new_tokens),
                )
                for result in semantic_results:
                    candidate = str(result["candidate"])
                    if not result.get("accepted", False):
                        audit.append({
                            "candidate": candidate,
                            "stage": "semantic_equivalence_judge",
                            "accepted": False,
                            "reason": _semantic_reason(result),
                            "relation_gate": relation_by_candidate.get(candidate),
                            "semantic": result,
                        })
                        continue
                    key = generation._normalize_cmp(candidate)
                    if key in accepted_keys:
                        continue
                    accepted_keys.add(key)
                    accepted.append(candidate)
                    audit.append({
                        "candidate": candidate,
                        "stage": "semantic_equivalence_judge",
                        "accepted": True,
                        "reason": "accepted",
                        "relation_gate": relation_by_candidate.get(candidate),
                        "semantic": result,
                    })
                    if len(accepted) >= int(a.max_surrogates):
                        break
        else:
            audit.append({
                "candidate": None,
                "stage": "direct_relation_profile",
                "accepted": False,
                "reason": "direct_prompt_not_safely_augmentable",
                "detail": direct_profile,
            })

        if len(accepted) >= int(a.min_surrogates):
            kept = accepted[: int(a.max_surrogates)]
            status = "robust_prompt_set"
            augmented_count += 1
        else:
            kept = []
            status = "direct_only"
            direct_only_count += 1

        artifact_records.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "subject": subject,
            "direct_prompt": direct_prompt,
            "augmentation_status": status,
            "surrogate_count": len(kept),
            "surrogate_prompts": kept,
            "relation_slot": {
                "relation_label": direct_profile.get("relation_label"),
                "relation_description": direct_profile.get("relation_description"),
                "answer_types": direct_profile.get("answer_types", []),
                "ambiguity": direct_profile.get("ambiguity"),
                "safe_to_augment": bool(direct_profile.get("safe_to_augment", False)),
            },
        })
        receipt_records.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "subject": subject,
            "direct_prompt": direct_prompt,
            "direct_profile": direct_profile,
            "augmentation_status": status,
            "accepted_before_minimum_gate": accepted,
            "kept_surrogates": kept,
            "candidate_audit": audit,
            "generation_rounds": raw_rounds,
            "rejection_histogram": _hist(audit),
            "baseline_answer_occurrences": {
                answer: answer_guard.answer_occurrence_count(direct_prompt, answer)
                for answer in answers
            },
        })
        print(
            f"record {position:02d}: {status}; kept={len(kept)}; "
            f"safe_candidates={len(accepted)}; rounds={len(raw_rounds)}; "
            f"relation={direct_profile.get('relation_label')}; "
            f"types={direct_profile.get('answer_types')}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "protocol": ARTIFACT_PROTOCOL,
        "adapter_protocol": ADAPTER_PROTOCOL,
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "min_surrogates": int(a.min_surrogates),
        "max_surrogates": int(a.max_surrogates),
        "source_training_visible_path": str(visible_path),
        "source_split_manifest": str(manifest_path),
        "adapter_summary": {
            "records_total": len(artifact_records),
            "records_with_robust_prompt_sets": int(augmented_count),
            "records_direct_only": int(direct_only_count),
            "surrogate_prompts_total": int(sum(x["surrogate_count"] for x in artifact_records)),
        },
        "pipeline": [
            "candidate_generation",
            "relation_slot_classifier",
            "answer_type_compatibility_filter",
            "semantic_equivalence_judge",
            "keep_3_to_8_or_direct_only",
        ],
        "generator": {
            "model_path": str(a.generator_model_path),
            "generation_rounds": int(a.generation_rounds),
            "generation_oversample": int(a.generation_oversample),
            "temperature": float(a.temperature),
            "top_p": float(a.top_p),
            "max_new_tokens": int(a.max_new_tokens),
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "deterministic_wrapper_fallback_used": False,
        },
        "relation_slot_adapter": {
            "protocol": relation_adapter.ADAPTER_PROTOCOL,
            "model_path": str(adapter_path),
            "answer_type_vocabulary": list(relation_adapter.ANSWER_TYPES),
            "classifier_received_target_true": False,
            "classifier_received_target_new": False,
            "direct_ambiguous_prompts_may_be_direct_only": True,
        },
        "semantic_validation": {
            "protocol": semantic.VALIDATOR_PROTOCOL,
            "model_path": str(semantic_path),
            "structured_booleans_authoritative": True,
            "validator_received_target_true": False,
            "validator_received_target_new": False,
            "validator_received_official_paraphrases": False,
        },
        "data_access": {
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "official_PPL_seen": False,
        },
        "records": artifact_records,
    }

    out = Path(a.output).resolve()
    receipt = out.with_name(out.stem + "_adapter_receipt.json")
    _write_json(out, payload)
    _write_json(receipt, {
        "schema_version": 1,
        "protocol": ARTIFACT_PROTOCOL + "_receipt",
        "adapter_protocol": ADAPTER_PROTOCOL,
        "records": receipt_records,
    })
    print(f"MCF robust-prompt adapter artifact: {out}")
    print(f"MCF robust-prompt adapter receipt: {receipt}")
    print(
        f"Augmented {augmented_count}/{len(artifact_records)} records; "
        f"direct-only {direct_only_count}; total kept surrogates "
        f"{sum(x['surrogate_count'] for x in artifact_records)}."
    )
    print("Official MCF paraphrase/neighborhood/retain/PPL data were NOT read.")


if __name__ == "__main__":
    main()
