#!/usr/bin/env python3
"""Build an auditable surrogate-paraphrase artifact for locked MCF forget records.

The generator is intentionally isolated from official MCF paraphrase/neighborhood
probes. It reads only the locked training-visible records and split manifest. The
text-generation prompt receives only requested_rewrite.subject and the formatted
requested_rewrite.prompt. Neither target_true nor target_new is supplied to the
generator. After generation, the known training-visible answers are used only as
rejection filters so a surrogate cannot accidentally introduce either answer.

Important: if an answer string already occurs in the locked direct prompt, that
baseline occurrence is not treated as leakage. A surrogate is rejected only if it
introduces additional whole-token answer occurrences beyond the direct prompt.

Modes:
  * local_llm (recommended): local instruction-tuned causal LM semantic rewrites.
  * deterministic: structural variants for tests/debugging and fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_frozen_head_representation_repair as contract_helpers
import mcf_surrogate_answer_guard as answer_guard
import sure_stage2_sparse_repair as stage2


SCHEMA_VERSION = 1
PROTOCOL = "mcf_locked_direct_only_surrogate_paraphrases_v1"


def _dtype(name: str):
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _normalize_cmp(text: str) -> str:
    return _normalize_space(text).strip(" \t\n\r\"'`.,:;!?").casefold()


def _clean_candidate(text: str) -> str:
    text = str(text).strip()
    if text.startswith('"') and text.endswith('"'):
        try:
            value = json.loads(text)
            if isinstance(value, str):
                text = value
        except Exception:
            pass
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        text = lines[0]
    text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", text)
    return _normalize_space(text.strip().strip('"').strip("'").strip())


def _contains_answer(candidate: str, answers: Sequence[str]) -> bool:
    """Backward-compatible helper used by tests and diagnostics."""
    return answer_guard.contains_answer(candidate, answers)


def deterministic_surrogates(subject: str, direct_prompt: str) -> List[str]:
    """Deterministic fallback variants built only from subject + direct prompt."""
    subject = _normalize_space(subject)
    direct = _normalize_space(direct_prompt)
    candidates = [
        f"Complete this factual statement about {subject}: {direct}",
        f"Regarding {subject}, complete the fact: {direct}",
        f"What completes the following fact about {subject}? {direct}",
        f"Fill in the missing factual continuation for {subject}: {direct}",
        f"Provide the missing completion for this statement about {subject}: {direct}",
        f"Fact about {subject}. Complete the statement: {direct}",
        f"For {subject}, supply the factual completion: {direct}",
        f"Answer only with the completion of this fact about {subject}: {direct}",
        f"Finish this factual relation concerning {subject}: {direct}",
        f"Complete the following relation for {subject}: {direct}",
        f"Which completion makes this statement about {subject} factually complete? {direct}",
        f"Give the missing object for this fact about {subject}: {direct}",
    ]
    seen = {_normalize_cmp(direct)}
    out: List[str] = []
    for value in candidates:
        value = _clean_candidate(value)
        key = _normalize_cmp(value)
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _instruction(subject: str, direct_prompt: str) -> str:
    return (
        "You are creating training-only paraphrases for a factual completion prompt.\n"
        "Rewrite the prompt so it asks for exactly the same missing factual answer, "
        "but use substantially different wording and syntax.\n"
        "Rules:\n"
        f"1. Keep the subject exactly as written: {subject}\n"
        "2. Do NOT answer the fact and do NOT guess the missing answer.\n"
        "3. End immediately where the missing answer should begin.\n"
        "4. Do not add explanations, labels, numbering, quotation marks, or multiple variants.\n"
        "5. Preserve the factual relation; change only the wording.\n\n"
        f"Original prompt: {direct_prompt}\n"
        "Paraphrase:"
    )


def _model_input(tok, subject: str, direct_prompt: str) -> str:
    instruction = _instruction(subject, direct_prompt)
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


@torch.no_grad()
def generate_local_surrogates(
    model,
    tok,
    *,
    subject: str,
    direct_prompt: str,
    count: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    if count <= 0:
        return {"accepted": [], "raw": []}
    device = next(model.parameters()).device
    text = _model_input(tok, subject, direct_prompt)
    enc = tok(text, return_tensors="pt").to(device)
    n = max(int(count) * 2, int(count) + 3)
    # torch.Generator support differs across generate backends; use the global
    # torch RNG after an explicit seed so behavior remains reproducible.
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    seq = model.generate(
        **enc,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        num_return_sequences=n,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    prefix = int(enc["input_ids"].shape[1])
    raw = [tok.decode(row[prefix:], skip_special_tokens=True) for row in seq]
    return {"accepted": [_clean_candidate(x) for x in raw], "raw": raw}


def _validated_unique(
    candidates: Sequence[str],
    *,
    subject: str,
    direct_prompt: str,
    answers: Sequence[str],
    limit: int,
) -> List[str]:
    direct_key = _normalize_cmp(direct_prompt)
    subject_key = _normalize_cmp(subject)
    seen = {direct_key}
    out: List[str] = []
    for candidate in candidates:
        candidate = _clean_candidate(candidate)
        key = _normalize_cmp(candidate)
        if not candidate or not key or key in seen:
            continue
        if subject_key and subject_key not in key:
            continue
        if len(candidate) < max(8, len(subject) + 2) or len(candidate) > 320:
            continue
        # Baseline-aware leakage guard. If an answer string already appears in
        # the direct prompt, preserving that occurrence is allowed; introducing
        # extra whole-token answer mentions is not.
        if answer_guard.introduced_answer_occurrences(
            candidate, direct_prompt, answers
        ):
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= int(limit):
            break
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--surrogates-per-record", type=int, default=8)
    p.add_argument("--mode", choices=("local_llm", "deterministic"), default="local_llm")
    p.add_argument("--generator-model-path", default=None)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    a = p.parse_args(list(argv) if argv is not None else None)
    if a.forget_num <= 0 or a.surrogates_per_record <= 0:
        p.error("forget-num and surrogates-per-record must be positive")
    if a.max_new_tokens <= 0:
        p.error("max-new-tokens must be positive")
    if not math.isfinite(a.temperature) or a.temperature <= 0:
        p.error("temperature must be finite and positive")
    if not math.isfinite(a.top_p) or not (0 < a.top_p <= 1):
        p.error("top-p must be in (0,1]")
    if a.mode == "local_llm" and not a.generator_model_path:
        p.error("--generator-model-path is required for --mode local_llm")
    return a


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)

    model = None
    tok = None
    if a.mode == "local_llm":
        kwargs: Dict[str, Any] = {"torch_dtype": _dtype(a.dtype)}
        if a.device_map == "auto":
            kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(a.generator_model_path, **kwargs)
        if a.device_map == "single":
            if not torch.cuda.is_available():
                raise RuntimeError("--device-map single requires CUDA")
            model = model.to("cuda")
        model.eval()
        tok = AutoTokenizer.from_pretrained(a.generator_model_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    artifact_records: List[Dict[str, Any]] = []
    raw_receipt: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        answers = [str(rr["target_true"]["str"]), str(rr["target_new"]["str"])]

        raw: List[str] = []
        generated: List[str] = []
        if a.mode == "local_llm":
            result = generate_local_surrogates(
                model,
                tok,
                subject=subject,
                direct_prompt=direct_prompt,
                count=int(a.surrogates_per_record),
                seed=int(a.seed) * 1000003 + int(position) * 9973 + 17,
                max_new_tokens=int(a.max_new_tokens),
                temperature=float(a.temperature),
                top_p=float(a.top_p),
            )
            generated.extend(result["accepted"])
            raw.extend(result["raw"])

        generated.extend(deterministic_surrogates(subject, direct_prompt))
        accepted = _validated_unique(
            generated,
            subject=subject,
            direct_prompt=direct_prompt,
            answers=answers,
            limit=int(a.surrogates_per_record),
        )
        if len(accepted) < int(a.surrogates_per_record):
            baseline_counts = {
                answer: answer_guard.answer_occurrence_count(direct_prompt, answer)
                for answer in answers
            }
            raise RuntimeError(
                f"Record {position} produced only {len(accepted)} valid surrogates; "
                f"requested {a.surrogates_per_record}; "
                f"baseline_answer_occurrences={baseline_counts}"
            )

        artifact_records.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "subject": subject,
            "direct_prompt": direct_prompt,
            "surrogate_prompts": accepted,
        })
        raw_receipt.append({
            "case_id": int(record.get("case_id", position)),
            "sampled_position": int(position),
            "raw_generations": raw,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "surrogates_per_record": int(a.surrogates_per_record),
        "source_training_visible_path": str(visible_path),
        "source_split_manifest": str(manifest_path),
        "generator": {
            "mode": a.mode,
            "model_path": None if a.mode != "local_llm" else str(a.generator_model_path),
            "temperature": None if a.mode != "local_llm" else float(a.temperature),
            "top_p": None if a.mode != "local_llm" else float(a.top_p),
            "max_new_tokens": None if a.mode != "local_llm" else int(a.max_new_tokens),
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "post_generation_answer_rejection_filter": True,
            "answer_rejection_policy": "reject_new_occurrences_beyond_direct_prompt_baseline",
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
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt_path = out.with_name(out.stem + "_raw_generation_receipt.json")
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol": PROTOCOL + "_raw_generation_receipt",
                "records": raw_receipt,
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Surrogate artifact: {out}")
    print(f"Raw generation receipt: {receipt_path}")
    print(f"Built {len(artifact_records)} records x {a.surrogates_per_record} surrogates.")
    print("Official MCF paraphrase/neighborhood/retain/PPL data were NOT read.")


if __name__ == "__main__":
    main()
