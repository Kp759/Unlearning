#!/usr/bin/env python3
"""Generate TRAINING-ONLY multi-view forget prompts for private-vocab rewiring V1.3.

Leakage contract
================
This process accepts ONLY the sanitized ``training_visible_forget_direct.json``
produced by the locked split builder.  It has no ``--mcf-path`` argument and
must never read official paraphrases, neighborhood prompts, generation probes,
or any evaluation retain records.

The local frozen Base model paraphrases each canonical direct prompt.  The
model is shown only the subject and canonical requested-rewrite prompt; target
answers are NOT included in the generation instruction.  Target strings from
the sanitized training record are used only after generation to reject answer
leakage and to perform a Base-semantic consistency filter.

The output is a reproducible training corpus keyed by case_id.  Official held-
out probes remain completely absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


FORBIDDEN_PROBE_FIELDS = {
    "paraphrase_prompts",
    "neighborhood_prompts",
    "attribute_prompts",
    "generation_prompts",
}
EXPECTED_RECORD_KEYS = {"case_id", "requested_rewrite", "data_role"}
EXPECTED_REWRITE_KEYS = {
    "prompt",
    "subject",
    "relation_id",
    "target_true",
    "target_new",
}
PROTOCOL = "mcf_private_vocab_rewiring_v1_3_training_multiview_corpus"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def normalize_space(text: str) -> str:
    return " ".join(str(text).strip().split())


def validate_sanitized_forget(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError("sanitized forget set is empty")
    seen: set[int] = set()
    for position, record in enumerate(records):
        if set(record).intersection(FORBIDDEN_PROBE_FIELDS):
            raise RuntimeError(f"held-out probe leaked into training record {position}")
        if set(record) != EXPECTED_RECORD_KEYS:
            raise RuntimeError(
                f"training record {position} has unexpected keys: {sorted(record)}"
            )
        if str(record.get("data_role")) != "forget":
            raise RuntimeError(f"training record {position} is not forget role")
        case_id = int(record["case_id"])
        if case_id in seen:
            raise RuntimeError(f"duplicate forget case_id {case_id}")
        seen.add(case_id)
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or set(rr) != EXPECTED_REWRITE_KEYS:
            raise RuntimeError(f"invalid requested_rewrite at case {case_id}")
        prompt = str(rr["prompt"])
        if "{}" not in prompt:
            raise RuntimeError(f"canonical prompt lacks subject placeholder at case {case_id}")
        for target_key in ("target_true", "target_new"):
            target = rr.get(target_key)
            if not isinstance(target, Mapping) or set(target) != {"str"}:
                raise RuntimeError(f"invalid {target_key} at case {case_id}")


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    value = str(answer).strip()
    ids = tokenizer(
        " " + value,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not ids:
        raise RuntimeError(f"answer tokenizes empty: {answer!r}")
    return [int(v) for v in ids]


@torch.no_grad()
def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    device: torch.device,
) -> float:
    pids = tokenizer(
        prompt, add_special_tokens=False, return_attention_mask=False
    )["input_ids"]
    pids = [int(v) for v in pids]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        pids = [int(bos)] + pids
    aids = answer_ids(tokenizer, answer)
    ids = torch.tensor([pids + aids], device=device, dtype=torch.long)
    logits = model(input_ids=ids).logits.float()
    logp = F.log_softmax(logits, dim=-1)
    start = len(pids)
    positions = torch.arange(start - 1, start - 1 + len(aids), device=device)
    target = torch.tensor(aids, device=device, dtype=torch.long)
    return float(logp[0, positions, target].mean().item())


def generation_instruction(subject: str, canonical: str, count: int) -> str:
    return f"""You are creating TRAINING-ONLY paraphrases of a factual completion prompt.

Rules:
- Preserve exactly the SAME subject and SAME semantic relation.
- Keep the literal subject string exactly as written: {subject}
- Do NOT answer the fact.
- Do NOT add any factual value after the relation.
- Each line must remain a prompt that can be completed by a short factual answer.
- Use substantially different natural wording and syntax across lines.
- Do not explain anything.
- Output exactly {count} lines, one paraphrase per line.

Original prompt:
{canonical}

Paraphrases:"""


def encode_generation_prompt(tokenizer: Any, instruction: str, device: torch.device) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            return ids.to(device)
        except Exception:
            pass
    return tokenizer(instruction, return_tensors="pt")["input_ids"].to(device)


def clean_generated_lines(text: str) -> list[str]:
    out: list[str] = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        line = line.strip("`\"'")
        if line:
            out.append(normalize_space(line))
    return out


def literal_answer_leak(candidate: str, true_answer: str, new_answer: str) -> bool:
    low = candidate.casefold()
    for answer in (true_answer, new_answer):
        value = normalize_space(answer).casefold()
        if value and value in low:
            return True
    return False


def subject_to_template(candidate: str, subject: str) -> str | None:
    if candidate.count(subject) != 1:
        return None
    templated = candidate.replace(subject, "{}", 1)
    if templated.count("{}") != 1:
        return None
    return templated


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--forget-direct", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--views-per-case", type=int, default=5)
    p.add_argument("--candidates-per-attempt", type=int, default=10)
    p.add_argument("--max-attempts", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=13131)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument(
        "--max-true-logprob-drop",
        type=float,
        default=3.0,
        help="Reject a candidate if Base target_true logprob falls this far below canonical.",
    )
    p.add_argument(
        "--max-margin-degradation",
        type=float,
        default=1.0,
        help="Candidate Base (true-new) margin may be at most this much worse than canonical.",
    )
    args = p.parse_args()
    if args.views_per_case < 2:
        p.error("views-per-case must include canonical + at least one synthetic view")
    if args.candidates_per_attempt < args.views_per_case - 1:
        p.error("candidates-per-attempt is too small")
    if args.max_attempts <= 0:
        p.error("max-attempts must be positive")
    return args


def main() -> None:
    args = parse_args()
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
    validate_sanitized_forget(raw)

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
        torch_dtype=dtype_from_name(args.dtype),
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
        canonical = normalize_space(canonical_template.format(subject))
        true_answer = str(rr["target_true"]["str"])
        new_answer = str(rr["target_new"]["str"])

        canonical_true_lp = sequence_logprob(
            model, tokenizer, canonical, true_answer, device=device
        )
        canonical_new_lp = sequence_logprob(
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
        seen = {normalize_space(canonical).casefold()}
        rejected_counts: Dict[str, int] = {}

        for attempt in range(int(args.max_attempts)):
            if len(accepted) >= int(args.views_per_case):
                break
            instruction = generation_instruction(
                subject, canonical, int(args.candidates_per_attempt)
            )
            input_ids = encode_generation_prompt(tokenizer, instruction, device)
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
            for candidate in clean_generated_lines(generated):
                if len(accepted) >= int(args.views_per_case):
                    break
                key = candidate.casefold()
                if key in seen:
                    rejected_counts["duplicate"] = rejected_counts.get("duplicate", 0) + 1
                    continue
                if len(candidate) < 8 or len(candidate) > 300:
                    rejected_counts["length"] = rejected_counts.get("length", 0) + 1
                    continue
                template = subject_to_template(candidate, subject)
                if template is None:
                    rejected_counts["subject_not_exactly_once"] = rejected_counts.get(
                        "subject_not_exactly_once", 0
                    ) + 1
                    continue
                if literal_answer_leak(candidate, true_answer, new_answer):
                    rejected_counts["literal_answer_leak"] = rejected_counts.get(
                        "literal_answer_leak", 0
                    ) + 1
                    continue

                true_lp = sequence_logprob(
                    model, tokenizer, candidate, true_answer, device=device
                )
                new_lp = sequence_logprob(
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
        "schema_version": 1,
        "protocol": PROTOCOL,
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
        },
        "source_sha256": sha256_bytes(source_bytes),
        "model_path": str(Path(args.model_path).resolve()),
        "seed": int(args.seed),
        "views_per_case": int(args.views_per_case),
        "synthetic_views_per_case": int(args.views_per_case) - 1,
        "semantic_filter": {
            "max_true_logprob_drop": float(args.max_true_logprob_drop),
            "max_margin_degradation": float(args.max_margin_degradation),
            "literal_true_or_new_answer_rejected": True,
            "literal_subject_exactly_once_required": True,
        },
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
