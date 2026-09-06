#!/usr/bin/env python3
"""Build a training-only, family-structured relation-view corpus for Seed-1 MCF.

The builder accepts ONLY the sanitized ``training_visible_forget_direct.json``.
It never reads official paraphrase/neighborhood/generation probes.  The frozen
Base model is shown only a subject and the canonical requested-rewrite prompt;
true/new answers are never included in generation or semantic verification.

Unlike the older free-form five-view corpus, this builder requests one explicit
linguistic family at a time and rejects vague/drifted candidates with a separate
SAME-vs-DIFFERENT relation-equivalence check.  The goal is better training data
for relation recognition, not a new unlearning method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROTOCOL = "mcf_relation_view_corpus_v2"
SEED = 24291
FAMILY_SPECS = (
    ("wh_question", "Write a direct WH-question (What/Which/Where/Who) asking for the same relation."),
    ("possessive_question", "Write a natural possessive question using the entity name possessively when possible."),
    ("relation_fronted_question", "Put the relation concept first, then ask which value applies to the entity."),
    ("imperative_identify", "Write an explicit request beginning with Identify, Name, or State that asks for the same relation."),
    ("alternative_question", "Write a different natural question form, substantially different from the canonical syntax."),
    ("reordered_cloze", "Write a completion/cloze prompt with the relation phrase before or around the entity, not the canonical word order."),
    ("nominalized_question", "Ask using a clear relation noun phrase such as profession, affiliation, continent, language, producer, location, or the appropriate relation noun for this case."),
    ("conversational_question", "Write a concise conversational question that still explicitly asks for exactly the same relation."),
)
EXPECTED_KEYS = {"case_id", "requested_rewrite", "data_role"}
EXPECTED_RR_KEYS = {"prompt", "subject", "relation_id", "target_true", "target_new"}
FORBIDDEN_FIELDS = {"paraphrase_prompts", "neighborhood_prompts", "attribute_prompts", "generation_prompts"}
VAGUE_PATTERNS = (
    r"^.+\s+from\s*$",
    r"^.+\s+is\s+in\s*$",
    r"^.+\s+was\s+in\s*$",
    r"^.+\s+is\s+a\s+professional\s*[?.]?$",
    r"^.+\s+is\s+a\s+part\s+of\s+the\s*$",
    r"^.+\s+belongs\s+to\s+the\s*$",
    r"^.+\s+plays\s*$",
)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def validate_source(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("sanitized forget set is empty")
    seen: set[int] = set()
    for i, row in enumerate(rows):
        if set(row).intersection(FORBIDDEN_FIELDS):
            raise RuntimeError(f"held-out probe field leaked into training row {i}")
        if set(row) != EXPECTED_KEYS:
            raise RuntimeError(f"unexpected keys in row {i}: {sorted(row)}")
        if row.get("data_role") != "forget":
            raise RuntimeError(f"row {i} is not forget role")
        cid = int(row["case_id"])
        if cid in seen:
            raise RuntimeError(f"duplicate case_id {cid}")
        seen.add(cid)
        rr = row.get("requested_rewrite")
        if not isinstance(rr, Mapping) or set(rr) != EXPECTED_RR_KEYS:
            raise RuntimeError(f"invalid requested_rewrite for case {cid}")
        if str(rr["prompt"]).count("{}") != 1:
            raise RuntimeError(f"case {cid}: canonical prompt must have one {{}} slot")


def generation_prompt(subject: str, canonical: str, family: str, instruction: str, count: int) -> str:
    return f"""Create {count} TRAINING-ONLY rewrites of an incomplete factual prompt.

Entity: {subject}
Canonical prompt: {canonical}
Requested linguistic family: {family}
Family instruction: {instruction}

STRICT RULES:
- Every rewrite must ask for EXACTLY the same semantic relation/attribute as the canonical prompt.
- Keep the literal entity string exactly as written: {subject}
- Mention that literal entity exactly once.
- Do not answer the fact and do not add any factual value.
- Make the relation explicit; never use vague forms like 'is in', 'from', 'plays', 'works as', or 'is a professional' unless the missing relation itself is explicitly named.
- Each rewrite must stand alone and be grammatical.
- Make the syntax materially different from the canonical prompt.
- Output only the rewrites, one per line, with no numbering or explanation.
"""


def verifier_prompt(canonical: str, candidate: str) -> str:
    return f"""Compare two incomplete factual prompts about the same entity.
Decide whether they ask for EXACTLY THE SAME relation/attribute, not merely a related fact.
Examples of DIFFERENT: continent vs country, employer vs occupation, native language vs nationality, producer vs country of origin.
Answer only SAME or DIFFERENT.

Prompt A: {canonical}
Prompt B: {candidate}
Decision:"""


def encode_generation_prompt(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt"
            )
            return ids.to(device)
        except Exception:
            pass
    return tokenizer(text, return_tensors="pt").input_ids.to(device)


def continuation_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(" " + text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if not ids:
        raise RuntimeError(f"empty continuation tokenization for {text!r}")
    return [int(x) for x in ids]


@torch.no_grad()
def continuation_logprob(model: Any, tokenizer: Any, prompt: str, continuation: str, device: torch.device) -> float:
    pids = tokenizer(prompt, add_special_tokens=True, return_attention_mask=False)["input_ids"]
    cids = continuation_ids(tokenizer, continuation)
    ids = torch.tensor([list(map(int, pids)) + cids], device=device)
    logits = model(input_ids=ids).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    start = len(pids)
    pos = torch.arange(start - 1, start - 1 + len(cids), device=device)
    tok = torch.tensor(cids, device=device)
    return float(logp[0, pos, tok].mean().item())


@torch.no_grad()
def equivalence_margin(model: Any, tokenizer: Any, canonical: str, candidate: str, device: torch.device) -> float:
    prompt = verifier_prompt(canonical, candidate)
    same = continuation_logprob(model, tokenizer, prompt, "SAME", device)
    diff = continuation_logprob(model, tokenizer, prompt, "DIFFERENT", device)
    return same - diff


def clean_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in str(text).splitlines():
        line = normalize_space(raw)
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line)
        line = line.strip('"` ')
        if line:
            out.append(line)
    return out


def subject_to_template(candidate: str, subject: str) -> str | None:
    if candidate.count(subject) != 1:
        return None
    value = candidate.replace(subject, "{}", 1)
    return value if value.count("{}") == 1 else None


def too_vague(candidate: str, subject: str) -> bool:
    rendered = normalize_space(candidate)
    if len(rendered.split()) < 5:
        return True
    if len(rendered) > 320:
        return True
    low = rendered.casefold()
    for pattern in VAGUE_PATTERNS:
        if re.match(pattern, low, flags=re.IGNORECASE):
            return True
    # After removing the entity, require enough lexical material to name a relation.
    remainder = normalize_space(rendered.replace(subject, " "))
    lexical = re.findall(r"[A-Za-z][A-Za-z'-]+", remainder)
    return len(lexical) < 4


def token_jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", a.casefold()))
    tb = set(re.findall(r"[a-z0-9]+", b.casefold()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--forget-direct", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--candidates-per-attempt", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=180)
    p.add_argument("--temperature", type=float, default=0.75)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-equivalence-margin", type=float, default=0.5)
    p.add_argument("--max-jaccard-to-canonical", type=float, default=0.82)
    args = p.parse_args()
    if args.candidates_per_attempt <= 0 or args.max_attempts <= 0:
        p.error("candidate/attempt counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    source = Path(args.forget_direct).resolve()
    if source.name != "training_visible_forget_direct.json":
        raise RuntimeError("builder accepts only training_visible_forget_direct.json")
    raw_bytes = source.read_bytes()
    rows = json.loads(raw_bytes)
    if not isinstance(rows, list) or not all(isinstance(x, dict) for x in rows):
        raise RuntimeError("sanitized forget source must be a JSON list")
    validate_source(rows)

    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype_from_name(args.dtype), local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval(); model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    cases: list[dict[str, Any]] = []
    global_rejects: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        rr = row["requested_rewrite"]
        cid = int(row["case_id"])
        subject = str(rr["subject"])
        canonical_template = str(rr["prompt"])
        canonical = normalize_space(canonical_template.format(subject))
        views: list[dict[str, Any]] = [{
            "family": "canonical_cloze",
            "template": canonical_template,
            "source": "canonical_requested_rewrite",
            "equivalence_margin": None,
        }]
        accepted_texts = [canonical]
        case_rejects: dict[str, int] = {}

        for family_index, (family, instruction) in enumerate(FAMILY_SPECS):
            chosen: dict[str, Any] | None = None
            for attempt in range(int(args.max_attempts)):
                gp = generation_prompt(subject, canonical, family, instruction, int(args.candidates_per_attempt))
                input_ids = encode_generation_prompt(tokenizer, gp, device)
                local_seed = int(args.seed) + cid * 101 + family_index * 1009 + attempt
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
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
                for candidate in clean_lines(generated):
                    template = subject_to_template(candidate, subject)
                    if template is None:
                        reason = "subject_not_exactly_once"
                    elif too_vague(candidate, subject):
                        reason = "vague_or_too_short"
                    elif any(normalize_space(candidate).casefold() == x.casefold() for x in accepted_texts):
                        reason = "duplicate"
                    elif token_jaccard(candidate, canonical) > float(args.max_jaccard_to_canonical):
                        reason = "too_lexically_close_to_canonical"
                    else:
                        margin = equivalence_margin(model, tokenizer, canonical, candidate, device)
                        if margin < float(args.min_equivalence_margin):
                            reason = "relation_equivalence_rejected"
                        else:
                            chosen = {
                                "family": family,
                                "template": template,
                                "source": "family_conditioned_generation",
                                "equivalence_margin": float(margin),
                                "attempt": attempt,
                            }
                            accepted_texts.append(normalize_space(candidate))
                            break
                    case_rejects[reason] = case_rejects.get(reason, 0) + 1
                    global_rejects[reason] = global_rejects.get(reason, 0) + 1
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(
                    f"case_id={cid}: could not produce accepted family={family}; rejects={case_rejects}"
                )
            views.append(chosen)

        cases.append({
            "case_id": cid,
            "relation_id": str(rr["relation_id"]),
            "subject": subject,
            "views": views,
            "rejected_counts": case_rejects,
        })
        print(json.dumps({
            "case": row_index + 1, "of": len(rows), "case_id": cid,
            "families": [v["family"] for v in views], "rejects": case_rejects,
        }), flush=True)

    payload = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "source_sha256": sha256_bytes(raw_bytes),
        "cases": cases,
        "views_per_case": 1 + len(FAMILY_SPECS),
        "families": ["canonical_cloze"] + [x[0] for x in FAMILY_SPECS],
        "family_split_recommendation": {
            "fit": ["canonical_cloze", "wh_question", "possessive_question", "imperative_identify", "alternative_question"],
            "calibration": ["relation_fronted_question", "reordered_cloze"],
            "validation": ["nominalized_question", "conversational_question"],
        },
        "leakage_contract": {
            "full_mcf_path_accepted": False,
            "official_paraphrase_prompts_read": False,
            "official_neighborhood_prompts_read": False,
            "official_generation_prompts_read": False,
            "official_retain_records_read": False,
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "verifier_received_target_true": False,
            "verifier_received_target_new": False,
        },
        "quality_controls": {
            "family_conditioned_generation": True,
            "exact_subject_once": True,
            "vague_form_filter": True,
            "minimum_equivalence_margin": float(args.min_equivalence_margin),
            "max_jaccard_to_canonical": float(args.max_jaccard_to_canonical),
            "semantic_verifier": "frozen Base SAME-vs-DIFFERENT relation-equivalence logprob margin",
            "global_rejected_counts": global_rejects,
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "cases": len(cases), "views_per_case": payload["views_per_case"]}, indent=2))


if __name__ == "__main__":
    main()
