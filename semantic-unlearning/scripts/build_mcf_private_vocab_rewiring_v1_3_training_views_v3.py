#!/usr/bin/env python3
"""Leakage-safe V1.3 training-view generator, revision 3.

Build canonical + four training-only views without reading any held-out MCF probes.
Priority: (1) same-relation templates from sanitized protection_fit, then
(2) constrained minimal paraphrases from the frozen Base model.  The existing
Base semantic/logprob gates and subject-aware answer-leak filter remain intact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import build_mcf_private_vocab_rewiring_v1_3_training_views as base
import build_mcf_private_vocab_rewiring_v1_3_training_views_v2 as v2


def validate_direct(records: Sequence[Mapping[str, Any]], role: str) -> None:
    if not records:
        raise RuntimeError(f"empty {role} set")
    for i, record in enumerate(records):
        if set(record).intersection(base.FORBIDDEN_PROBE_FIELDS):
            raise RuntimeError(f"held-out probe leaked into {role} record {i}")
        if set(record) != base.EXPECTED_RECORD_KEYS:
            raise RuntimeError(f"unexpected keys in {role} record {i}")
        if str(record.get("data_role")) != role:
            raise RuntimeError(f"wrong role in {role} record {i}")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or set(rr) != base.EXPECTED_REWRITE_KEYS:
            raise RuntimeError(f"invalid requested_rewrite in {role} record {i}")
        if str(rr["prompt"]).count("{}") != 1:
            raise RuntimeError(f"invalid direct prompt in {role} record {i}")


def minimal_instruction(subject: str, canonical: str, count: int, mode: int) -> str:
    styles = [
        "Make only small syntactic changes. Keep all relation-bearing content words whenever possible.",
        "Convert the prompt into natural question-style variants without changing the requested relation.",
        "Create declarative completion variants with the same relation and no factual answer.",
        "Reorder or minimally rephrase the syntax while preserving the exact semantic slot being requested.",
    ]
    return f"""Create {count} TRAINING-ONLY paraphrases of this factual prompt.

Hard rules:
- Keep the literal subject exactly once: {subject}
- Preserve exactly the SAME semantic relation as the original prompt.
- Do NOT broaden, narrow, or switch the relation.
- Do NOT answer the fact and do NOT include any candidate factual value.
- Each output must still be completable by the same short answer type.
- {styles[mode % len(styles)]}
- Output exactly {count} lines and nothing else.

Original prompt:
{canonical}

Paraphrases:"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--forget-direct", required=True)
    p.add_argument("--protection-fit-direct", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--views-per-case", type=int, default=5)
    p.add_argument("--candidates-per-attempt", type=int, default=12)
    p.add_argument("--max-attempts", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=13131)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--max-true-logprob-drop", type=float, default=3.0)
    p.add_argument("--max-margin-degradation", type=float, default=1.0)
    a = p.parse_args()
    if a.views_per_case < 2 or a.max_attempts <= 0:
        p.error("invalid view-generation settings")
    return a


def main() -> None:
    args = parse_args()
    forget_path = Path(args.forget_direct).resolve()
    fit_path = Path(args.protection_fit_direct).resolve()
    if forget_path.name != "training_visible_forget_direct.json":
        raise RuntimeError("V1.3 v3 accepts only training_visible_forget_direct.json")
    if fit_path.name != "training_visible_protection_fit_direct.json":
        raise RuntimeError("V1.3 v3 accepts only training_visible_protection_fit_direct.json")
    forget_bytes, fit_bytes = forget_path.read_bytes(), fit_path.read_bytes()
    forget, fit = json.loads(forget_bytes), json.loads(fit_bytes)
    validate_direct(forget, "forget")
    validate_direct(fit, "protection_fit")

    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    relation_bank: Dict[str, list[str]] = {}
    for record in fit:
        rr = record["requested_rewrite"]
        relation_bank.setdefault(str(rr["relation_id"]), []).append(str(rr["prompt"]))
    relation_bank = {
        rid: list(dict.fromkeys(templates)) for rid, templates in relation_bank.items()
    }

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
    for index, record in enumerate(forget):
        rr = record["requested_rewrite"]
        case_id, subject = int(record["case_id"]), str(rr["subject"])
        relation_id, canonical_template = str(rr["relation_id"]), str(rr["prompt"])
        true_answer, new_answer = str(rr["target_true"]["str"]), str(rr["target_new"]["str"])
        canonical = base.normalize_space(canonical_template.format(subject))
        canonical_true_lp = base.sequence_logprob(model, tokenizer, canonical, true_answer, device=device)
        canonical_new_lp = base.sequence_logprob(model, tokenizer, canonical, new_answer, device=device)
        canonical_margin = canonical_true_lp - canonical_new_lp
        min_margin = canonical_margin - float(args.max_margin_degradation)

        accepted: list[Dict[str, Any]] = [{
            "template": canonical_template,
            "source": "canonical_requested_rewrite",
            "base_true_logprob": canonical_true_lp,
            "base_new_logprob": canonical_new_lp,
            "base_true_minus_new_margin": canonical_margin,
        }]
        seen = {canonical.casefold()}
        rejected: Dict[str, int] = {}

        def admit(candidate: str, template: str, source: str, meta: Dict[str, Any]) -> bool:
            key = base.normalize_space(candidate).casefold()
            if key in seen:
                rejected["duplicate"] = rejected.get("duplicate", 0) + 1
                return False
            if len(candidate) < 8 or len(candidate) > 300:
                rejected["length"] = rejected.get("length", 0) + 1
                return False
            if base.literal_answer_leak(template, true_answer, new_answer):
                rejected["literal_answer_leak_outside_subject"] = rejected.get("literal_answer_leak_outside_subject", 0) + 1
                return False
            true_lp = base.sequence_logprob(model, tokenizer, candidate, true_answer, device=device)
            new_lp = base.sequence_logprob(model, tokenizer, candidate, new_answer, device=device)
            margin = true_lp - new_lp
            if true_lp < canonical_true_lp - float(args.max_true_logprob_drop):
                rejected["base_true_logprob_too_low"] = rejected.get("base_true_logprob_too_low", 0) + 1
                return False
            if margin < min_margin:
                rejected["base_semantic_margin_too_low"] = rejected.get("base_semantic_margin_too_low", 0) + 1
                return False
            seen.add(key)
            accepted.append({
                "template": template,
                "source": source,
                **meta,
                "base_true_logprob": true_lp,
                "base_new_logprob": new_lp,
                "base_true_minus_new_margin": margin,
            })
            return True

        # First use only direct templates from training-visible protection_fit with the same relation_id.
        for template in relation_bank.get(relation_id, []):
            if len(accepted) >= int(args.views_per_case):
                break
            if template == canonical_template or template.count("{}") != 1:
                continue
            candidate = base.normalize_space(template.format(subject))
            admit(candidate, template, "training_visible_same_relation_template", {})

        # Fill remaining slots using constrained local Base paraphrases only.
        for attempt in range(int(args.max_attempts)):
            if len(accepted) >= int(args.views_per_case):
                break
            instruction = minimal_instruction(subject, canonical, int(args.candidates_per_attempt), attempt)
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
                    pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
                )
            generated = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            for candidate in base.clean_generated_lines(generated):
                if len(accepted) >= int(args.views_per_case):
                    break
                leak, template = v2.answer_leak_outside_subject(candidate, subject, true_answer, new_answer)
                if template is None:
                    rejected["subject_not_exactly_once"] = rejected.get("subject_not_exactly_once", 0) + 1
                    continue
                if leak:
                    rejected["literal_answer_leak_outside_subject"] = rejected.get("literal_answer_leak_outside_subject", 0) + 1
                    continue
                admit(candidate, template, "local_base_model_minimal_paraphrase", {
                    "generation_attempt": attempt,
                    "generation_seed": local_seed,
                    "generation_mode": attempt % 4,
                })

        if len(accepted) < int(args.views_per_case):
            raise RuntimeError(
                f"case {case_id} ({subject!r}, {relation_id}) produced only {len(accepted)}/{args.views_per_case} "
                f"accepted training views; relation_templates_available={len(relation_bank.get(relation_id, []))}; "
                f"rejections={rejected}. No held-out fallback is permitted."
            )
        cases.append({
            "case_id": case_id,
            "subject": subject,
            "relation_id": relation_id,
            "views": accepted[: int(args.views_per_case)],
            "rejected_counts": rejected,
            "same_relation_templates_available": len(relation_bank.get(relation_id, [])),
        })
        print(f"case {index+1:2d}/{len(forget)} id={case_id}: accepted {args.views_per_case} views", flush=True)

    payload = {
        "schema_version": 3,
        "protocol": base.PROTOCOL,
        "leakage_contract": {
            "full_mcf_path_accepted": False,
            "official_paraphrase_prompts_read": False,
            "official_neighborhood_prompts_read": False,
            "official_generation_prompts_read": False,
            "official_retain_records_read": False,
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "training_visible_protection_fit_direct_used": True,
            "only_same_relation_id_protection_templates_used_for_forget_views": True,
            "target_strings_used_only_for_post_generation_filtering": True,
            "mandatory_subject_span_removed_before_literal_answer_leak_scan": True,
        },
        "forget_source_sha256": base.sha256_bytes(forget_bytes),
        "protection_fit_source_sha256": base.sha256_bytes(fit_bytes),
        "model_path": str(Path(args.model_path).resolve()),
        "seed": int(args.seed),
        "views_per_case": int(args.views_per_case),
        "synthetic_views_per_case": int(args.views_per_case)-1,
        "semantic_filter": {
            "max_true_logprob_drop": float(args.max_true_logprob_drop),
            "max_margin_degradation": float(args.max_margin_degradation),
            "thresholds_unchanged_from_v1_3_v2": True,
        },
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "views_per_case": args.views_per_case, "heldout_probe_text_read": False, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
