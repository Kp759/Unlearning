#!/usr/bin/env python3
"""Leakage-safe V1.3 training-view generator, revision 2.

This preserves the original V1.3 leakage contract and semantic gates.  The only
behavioral correction is answer-leak filtering for cases where an answer string
is already part of the mandatory literal subject (for example subject ``BMW M5``
and target_true ``BMW``).  We first replace the exactly-once subject occurrence
with ``{}``, then scan the remaining template for true/new answer leakage.

Therefore an answer occurrence inside the required subject is permitted, while
any answer occurrence outside the subject is still rejected.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import build_mcf_private_vocab_rewiring_v1_3_training_views as base


def answer_leak_outside_subject(
    candidate: str,
    subject: str,
    true_answer: str,
    new_answer: str,
) -> tuple[bool, str | None]:
    """Check answer leakage only after removing the mandatory subject span."""
    template = base.subject_to_template(candidate, subject)
    if template is None:
        return False, None
    return base.literal_answer_leak(template, true_answer, new_answer), template


def main() -> None:
    args = base.parse_args()
    source = Path(args.forget_direct).resolve()
    if source.name != "training_visible_forget_direct.json":
        raise RuntimeError(
            "V1.3 generator accepts only training_visible_forget_direct.json; "
            f"got {source.name}"
        )
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise RuntimeError("sanitized forget source must be a JSON list")
    base.validate_sanitized_forget(raw)

    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=base.dtype_from_name(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    cases: list[Dict[str, Any]] = []
    for index, record in enumerate(raw):
        rr = record["requested_rewrite"]
        case_id = int(record["case_id"])
        subject = str(rr["subject"])
        relation_id = str(rr["relation_id"])
        canonical_template = str(rr["prompt"])
        canonical = base.normalize_space(canonical_template.format(subject))
        true_answer = str(rr["target_true"]["str"])
        new_answer = str(rr["target_new"]["str"])

        canonical_true_lp = base.sequence_logprob(
            model, tokenizer, canonical, true_answer, device=device
        )
        canonical_new_lp = base.sequence_logprob(
            model, tokenizer, canonical, new_answer, device=device
        )
        canonical_margin = canonical_true_lp - canonical_new_lp
        minimum_candidate_margin = canonical_margin - float(args.max_margin_degradation)

        accepted: list[Dict[str, Any]] = [
            {
                "template": canonical_template,
                "source": "canonical_requested_rewrite",
                "base_true_logprob": canonical_true_lp,
                "base_new_logprob": canonical_new_lp,
                "base_true_minus_new_margin": canonical_margin,
            }
        ]
        seen = {base.normalize_space(canonical).casefold()}
        rejected_counts: Dict[str, int] = {}

        for attempt in range(int(args.max_attempts)):
            if len(accepted) >= int(args.views_per_case):
                break
            instruction = base.generation_instruction(
                subject, canonical, int(args.candidates_per_attempt)
            )
            input_ids = base.encode_generation_prompt(tokenizer, instruction, device)
            local_seed = int(args.seed) + case_id * 101 + attempt
            torch.manual_seed(local_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(local_seed)
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    pad_token_id=(
                        tokenizer.eos_token_id
                        if tokenizer.pad_token_id is None
                        else tokenizer.pad_token_id
                    ),
                )
            generated = tokenizer.decode(
                output[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            for candidate in base.clean_generated_lines(generated):
                if len(accepted) >= int(args.views_per_case):
                    break
                key = candidate.casefold()
                if key in seen:
                    rejected_counts["duplicate"] = rejected_counts.get("duplicate", 0) + 1
                    continue
                if len(candidate) < 8 or len(candidate) > 300:
                    rejected_counts["length"] = rejected_counts.get("length", 0) + 1
                    continue

                leak, template = answer_leak_outside_subject(
                    candidate, subject, true_answer, new_answer
                )
                if template is None:
                    rejected_counts["subject_not_exactly_once"] = rejected_counts.get(
                        "subject_not_exactly_once", 0
                    ) + 1
                    continue
                if leak:
                    rejected_counts["literal_answer_leak_outside_subject"] = rejected_counts.get(
                        "literal_answer_leak_outside_subject", 0
                    ) + 1
                    continue

                true_lp = base.sequence_logprob(
                    model, tokenizer, candidate, true_answer, device=device
                )
                new_lp = base.sequence_logprob(
                    model, tokenizer, candidate, new_answer, device=device
                )
                margin = true_lp - new_lp
                if true_lp < canonical_true_lp - float(args.max_true_logprob_drop):
                    rejected_counts["base_true_logprob_too_low"] = rejected_counts.get(
                        "base_true_logprob_too_low", 0
                    ) + 1
                    continue
                if margin < minimum_candidate_margin:
                    rejected_counts["base_semantic_margin_too_low"] = rejected_counts.get(
                        "base_semantic_margin_too_low", 0
                    ) + 1
                    continue

                seen.add(key)
                accepted.append(
                    {
                        "template": template,
                        "source": "local_base_model_synthetic_paraphrase",
                        "generation_attempt": attempt,
                        "generation_seed": local_seed,
                        "base_true_logprob": true_lp,
                        "base_new_logprob": new_lp,
                        "base_true_minus_new_margin": margin,
                    }
                )

        if len(accepted) < int(args.views_per_case):
            raise RuntimeError(
                f"case {case_id} ({subject!r}, {relation_id}) produced only "
                f"{len(accepted)}/{args.views_per_case} accepted training views; "
                f"rejections={rejected_counts}. No held-out fallback is permitted."
            )

        cases.append(
            {
                "case_id": case_id,
                "subject": subject,
                "relation_id": relation_id,
                "views": accepted[: int(args.views_per_case)],
                "rejected_counts": rejected_counts,
            }
        )
        print(
            f"case {index + 1:2d}/{len(raw)} id={case_id}: "
            f"accepted {len(accepted[: int(args.views_per_case)])} views",
            flush=True,
        )

    payload = {
        "schema_version": 2,
        "protocol": base.PROTOCOL,
        "leakage_contract": {
            "input_file_required": "training_visible_forget_direct.json",
            "full_mcf_path_accepted": False,
            "official_paraphrase_prompts_read": False,
            "official_neighborhood_prompts_read": False,
            "official_generation_prompts_read": False,
            "official_retain_records_read": False,
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "target_strings_used_only_for_post_generation_filtering": True,
            "mandatory_subject_span_removed_before_literal_answer_leak_scan": True,
            "answer_occurrence_outside_subject_rejected": True,
        },
        "source_sha256": base.sha256_bytes(source_bytes),
        "model_path": str(Path(args.model_path).resolve()),
        "seed": int(args.seed),
        "views_per_case": int(args.views_per_case),
        "synthetic_views_per_case": int(args.views_per_case) - 1,
        "semantic_filter": {
            "max_true_logprob_drop": float(args.max_true_logprob_drop),
            "max_margin_degradation": float(args.max_margin_degradation),
            "literal_true_or_new_answer_outside_subject_rejected": True,
            "literal_subject_exactly_once_required": True,
        },
        "cases": cases,
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "views_per_case": int(args.views_per_case),
                "heldout_probe_text_read": False,
                "output": str(out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
