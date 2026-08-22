#!/usr/bin/env python3
"""Build semantically validated, held-out-safe MCF surrogate paraphrases.

Pipeline per locked forget fact:

  direct prompt only
      -> repeated local-LLM paraphrase generation
      -> subject/duplicate/answer-occurrence guards
      -> deterministic weak-wrapper rejection
      -> strict semantic-equivalence judge
      -> adversarial semantic critic
      -> exactly K approved surrogates

The generator and semantic validator never receive target_true, target_new,
official MCF paraphrases, neighborhoods, retain examples, or official PPL text.
The known target strings are used only AFTER generation as an answer-occurrence
rejection guard; they are never provided to either LLM prompt.

No generic deterministic wrappers are admitted as fallbacks in v3. If semantic
generation cannot produce K validated paraphrases, the builder fails closed and
writes no training artifact.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import build_mcf_surrogate_paraphrases as base
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_surrogate_answer_guard as answer_guard
import mcf_surrogate_semantic_validator as semantic
import sure_stage2_sparse_repair as stage2


BUILDER_PROTOCOL = "mcf_locked_direct_only_semantic_surrogates_v3"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--surrogates-per-record", type=int, default=8)
    p.add_argument("--generator-model-path", required=True)
    p.add_argument(
        "--validator-model-path",
        default=None,
        help="Defaults to --generator-model-path; same path reuses one loaded model",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--generation-rounds", type=int, default=6)
    p.add_argument("--generation-oversample", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--judge-batch-size", type=int, default=8)
    p.add_argument("--judge-max-new-tokens", type=int, default=96)
    a = p.parse_args(argv)
    if a.forget_num <= 0 or a.surrogates_per_record <= 0:
        p.error("forget-num and surrogates-per-record must be positive")
    if a.generation_rounds <= 0 or a.generation_oversample <= 0:
        p.error("generation-rounds and generation-oversample must be positive")
    if a.max_new_tokens <= 0 or a.judge_batch_size <= 0 or a.judge_max_new_tokens <= 0:
        p.error("generation/judge token and batch sizes must be positive")
    if not math.isfinite(a.temperature) or a.temperature <= 0:
        p.error("temperature must be finite and positive")
    if not math.isfinite(a.top_p) or not (0 < a.top_p <= 1):
        p.error("top-p must be in (0,1]")
    return a


def _load_model(path: str, dtype: str, device_map: str):
    kwargs: Dict[str, Any] = {"torch_dtype": base._dtype(dtype)}
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


def _prefilter(
    candidates: List[str],
    *,
    subject: str,
    direct_prompt: str,
    answers: List[str],
    already_seen: set[str],
) -> List[str]:
    """Apply answer-blind structural rules plus post-generation answer guard."""
    # Reuse the v1 subject/length/dedup and baseline-aware answer guard. Give it
    # a very large limit because semantic validation, not this function, selects K.
    values = base._validated_unique(
        candidates,
        subject=subject,
        direct_prompt=direct_prompt,
        answers=answers,
        limit=max(1, len(candidates) + 1),
    )
    out: List[str] = []
    for value in values:
        key = base._normalize_cmp(value)
        if key in already_seen:
            continue
        if semantic.structural_rejection_reason(direct_prompt, value) is not None:
            # Keep structural rejections in the audit by passing them to the
            # validator later only if desired; here we can cheaply discard them.
            already_seen.add(key)
            continue
        already_seen.add(key)
        out.append(value)
    return out


def _rejection_hist(records: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(semantic.rejection_reason(x) for x in records if not x.get("accepted")))


def main(argv=None) -> None:
    a = parse_args(argv)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)

    validator_path = a.validator_model_path or a.generator_model_path
    generator_model, generator_tok = _load_model(
        a.generator_model_path, a.dtype, a.device_map
    )
    if str(Path(validator_path).resolve()) == str(Path(a.generator_model_path).resolve()):
        validator_model, validator_tok = generator_model, generator_tok
        shared_model = True
    else:
        validator_model, validator_tok = _load_model(
            validator_path, a.dtype, a.device_map
        )
        shared_model = False

    artifact_records: List[Dict[str, Any]] = []
    audit_records: List[Dict[str, Any]] = []

    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        answers = [
            str(rr["target_true"]["str"]),
            str(rr["target_new"]["str"]),
        ]
        accepted: List[str] = []
        accepted_keys: set[str] = set()
        seen_candidates: set[str] = {base._normalize_cmp(direct_prompt)}
        validation_audit: List[Dict[str, Any]] = []
        raw_rounds: List[Dict[str, Any]] = []

        for round_idx in range(int(a.generation_rounds)):
            remaining = int(a.surrogates_per_record) - len(accepted)
            if remaining <= 0:
                break
            requested = max(
                int(a.surrogates_per_record),
                remaining * int(a.generation_oversample),
            )
            seed = (
                int(a.seed) * 1000003
                + int(position) * 9973
                + int(round_idx) * 104729
                + 17
            )
            generated = base.generate_local_surrogates(
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
            candidates = _prefilter(
                list(generated["accepted"]),
                subject=subject,
                direct_prompt=direct_prompt,
                answers=answers,
                already_seen=seen_candidates,
            )
            if not candidates:
                continue
            judged = semantic.validate_candidates(
                validator_model,
                validator_tok,
                subject=subject,
                direct_prompt=direct_prompt,
                candidates=candidates,
                batch_size=int(a.judge_batch_size),
                max_new_tokens=int(a.judge_max_new_tokens),
            )
            validation_audit.extend(judged)
            for result in judged:
                if not result.get("accepted"):
                    continue
                candidate = str(result["candidate"])
                key = base._normalize_cmp(candidate)
                if key in accepted_keys:
                    continue
                accepted_keys.add(key)
                accepted.append(candidate)
                if len(accepted) >= int(a.surrogates_per_record):
                    break

        if len(accepted) < int(a.surrogates_per_record):
            baseline_counts = {
                answer: answer_guard.answer_occurrence_count(direct_prompt, answer)
                for answer in answers
            }
            raise RuntimeError(
                f"Record {position} produced only {len(accepted)} semantically "
                f"validated surrogates; requested {a.surrogates_per_record}; "
                f"generation_rounds={a.generation_rounds}; "
                f"rejections={_rejection_hist(validation_audit)}; "
                f"baseline_answer_occurrences={baseline_counts}"
            )

        accepted = accepted[: int(a.surrogates_per_record)]
        artifact_records.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "subject": subject,
            "direct_prompt": direct_prompt,
            "surrogate_prompts": accepted,
        })
        audit_records.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "subject": subject,
            "direct_prompt": direct_prompt,
            "accepted_surrogates": accepted,
            "candidate_judgments": validation_audit,
            "generation_rounds": raw_rounds,
            "rejection_histogram": _rejection_hist(validation_audit),
        })
        print(
            f"record {position:02d}: accepted {len(accepted)}/"
            f"{a.surrogates_per_record} after "
            f"{len(raw_rounds)} generation round(s)",
            flush=True,
        )

    payload = {
        "schema_version": base.SCHEMA_VERSION,
        # Keep the v1 artifact protocol so existing field-level validators can
        # parse it; semantic_validation below adds the stronger v3 contract.
        "protocol": base.PROTOCOL,
        "builder_protocol": BUILDER_PROTOCOL,
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "surrogates_per_record": int(a.surrogates_per_record),
        "source_training_visible_path": str(visible_path),
        "source_split_manifest": str(manifest_path),
        "generator": {
            "mode": "local_llm_semantic_only",
            "model_path": str(a.generator_model_path),
            "temperature": float(a.temperature),
            "top_p": float(a.top_p),
            "max_new_tokens": int(a.max_new_tokens),
            "generation_rounds": int(a.generation_rounds),
            "generation_oversample": int(a.generation_oversample),
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "post_generation_answer_rejection_filter": True,
            "answer_rejection_policy": "reject_new_occurrences_beyond_direct_prompt_baseline",
            "deterministic_wrapper_fallback_used": False,
        },
        "semantic_validation": {
            "enabled": True,
            "protocol": semantic.VALIDATOR_PROTOCOL,
            "validator_model_path": str(validator_path),
            "shared_generator_validator_model": bool(shared_model),
            "dual_pass_consensus": True,
            "required_for_every_surrogate": True,
            "validator_received_subject": True,
            "validator_received_direct_prompt": True,
            "validator_received_candidate": True,
            "validator_received_target_true": False,
            "validator_received_target_new": False,
            "validator_received_official_paraphrases": False,
            "criteria": [
                "same_relation",
                "same_answer_type",
                "no_added_factual_claims",
                "grammatical",
                "completion_compatible",
                "adversarial_no_relation_shift",
                "adversarial_no_answer_type_shift",
                "adversarial_no_added_factual_claim",
                "adversarial_not_malformed",
                "not_generic_wrapper",
            ],
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    receipt = out.with_name(out.stem + "_semantic_validation_receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "builder_protocol": BUILDER_PROTOCOL,
                "validator_protocol": semantic.VALIDATOR_PROTOCOL,
                "records": audit_records,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Semantic surrogate artifact: {out}")
    print(f"Semantic validation receipt: {receipt}")
    print(
        f"Built {len(artifact_records)} records x "
        f"{a.surrogates_per_record} semantically validated surrogates."
    )
    print("Official MCF paraphrase/neighborhood/retain/PPL data were NOT read.")


if __name__ == "__main__":
    main()
