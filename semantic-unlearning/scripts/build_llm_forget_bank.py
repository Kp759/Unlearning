"""
scripts/build_llm_forget_bank.py
--------------------------------

Build an LLM-extracted forget knowledge bank from TOFU forget QA pairs.

This script does three things:

1. Loads every QA pair from the forget split.
2. Uses an instruction LLM to extract structured forget facts:
   - subject
   - subject_type
   - flexible key/value facts
   - match_strings
   - erase_strings
3. Converts erase_strings into tokenizer token IDs and writes:
   - outputs/forget_knowledge_bank_llm_<split>.json
   - outputs/semantic_tokens_raw_llm_bank.json or outputs/semantic_tokens.json

Important design choice:
- match_strings can include question words and relation keys.
- erase_strings are restricted to subject + answer/fact values.
- Generic keys like genre, award, author, book, city, country are NOT erased by default.

Example run:

python scripts/build_llm_forget_bank.py \
  --config config/config_3b_instruct_forget05.yaml \
  --forget-split forget05 \
  --target-model outputs/finetuned_model_3B_instruct \
  --extractor-model Qwen/Qwen2.5-7B-Instruct \
  --extractor-dtype float16 \
  --merge-existing-freq \
  --out-bank outputs/forget_knowledge_bank_llm_forget05_3b_instruct.json \
  --out-semantic-tokens outputs/semantic_tokens_raw_llm_bank.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_WORDS = {
    "what", "who", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "did", "does", "do",
    "the", "a", "an", "of", "in", "on", "for", "to", "by",
    "and", "or", "with", "from", "as", "at", "into", "about",
    "known", "name", "answer", "question", "tell", "give",
    "this", "that", "these", "those", "it", "its", "his", "her",
}

# These are useful for matching but dangerous for static erasure.
# We do not erase these as standalone values.
GENERIC_RELATION_KEYS = {
    "genre",
    "award",
    "prize",
    "author",
    "book",
    "novel",
    "work",
    "birth",
    "birth_place",
    "birthplace",
    "birth_date",
    "born",
    "city",
    "country",
    "nationality",
    "occupation",
    "profession",
    "education",
    "publisher",
    "language",
    "year",
    "date",
    "full_name",
    "name",
    "notable_work",
}


SYSTEM_PROMPT = """
You are an information extraction system for machine unlearning.

Given one QA pair, extract the private or forgettable knowledge as structured JSON.

Rules:
- Return JSON only.
- Do not add explanations.
- Do not hallucinate facts not supported by the QA pair.
- Infer useful keys automatically.
- Use snake_case keys.
- The facts field must be a list of key-value objects.
- The subject should be the main entity/person/work being asked about.
- match_strings should include strings useful for matching future questions.
- erase_strings should include only specific subject names and specific answer/fact values.
- Do not put generic relation words alone in erase_strings, such as genre, award, author, book, city, country.
"""


def build_user_prompt(question: str, answer: str) -> str:
    return f"""
Extract structured forget knowledge from this QA pair.

Question:
{question}

Answer:
{answer}

Return JSON with this exact schema:

{{
  "subject": "main entity name or null",
  "subject_type": "person | fictional_author | book | place | organization | fact | unknown",
  "facts": [
    {{
      "key": "auto_inferred_key",
      "value": "answer_value",
      "confidence": 0.0
    }}
  ],
  "match_strings": [
    "strings useful for matching future questions"
  ],
  "erase_strings": [
    "specific strings whose embeddings should be erased"
  ]
}}
""".strip()


def normalize_str(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def normalize_key(x: Any) -> str:
    x = normalize_str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from model output.
    Handles markdown fences and extra text.
    """
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    candidate = text[start:end + 1]
    return json.loads(candidate)


def heuristic_subject_from_question(question: str) -> Optional[str]:
    """
    Fallback subject extractor if LLM fails.
    This is intentionally simple and conservative.
    """
    q = normalize_str(question)

    patterns = [
        r"genre is (.+?) known for",
        r"award did (.+?) receive",
        r"prize did (.+?) receive",
        r"where was (.+?) born",
        r"when was (.+?) born",
        r"was (.+?) born",
        r"book written by (.+?)[\?\.]?$",
        r"novel written by (.+?)[\?\.]?$",
        r"written by (.+?)[\?\.]?$",
        r"who is (.+?)[\?\.]?$",
        r"what is (.+?) known for",
        r"of (.+?)[\?\.]?$",
        r"by (.+?)[\?\.]?$",
    ]

    for pat in patterns:
        m = re.search(pat, q, flags=re.IGNORECASE)
        if m:
            candidate = normalize_str(m.group(1)).strip(" ?.")
            if len(candidate.split()) >= 2:
                return candidate

    # fallback: longest capitalized phrase
    caps = re.findall(r"(?:[A-Z][A-Za-z'\-]+(?:\s+|$)){2,}", q)
    caps = [normalize_str(c).strip(" ?.") for c in caps]
    caps = [c for c in caps if len(c.split()) >= 2]

    if caps:
        caps.sort(key=len, reverse=True)
        return caps[0]

    return None


def split_specific_values(value: str) -> List[str]:
    """
    Add useful subphrases for answer values.
    Example:
      "Kuwait City, Kuwait" -> ["Kuwait City, Kuwait", "Kuwait City", "Kuwait"]
    """
    value = normalize_str(value)
    if not value:
        return []

    parts = [value]

    for sep in [",", ";", " / ", " and "]:
        if sep in value:
            for p in value.split(sep):
                p = normalize_str(p)
                if p:
                    parts.append(p)

    # Deduplicate.
    return list(dict.fromkeys(parts))


def is_generic_erase_string(s: str) -> bool:
    """
    Protect standalone generic relation words from static erasure.
    """
    s_norm = normalize_key(s)

    if not s_norm:
        return True

    if s_norm in GENERIC_RELATION_KEYS:
        return True

    # Single very short words are usually unsafe.
    if len(s_norm) <= 2:
        return True

    return False


def validate_extraction(obj: Dict[str, Any], question: str, answer: str) -> Dict[str, Any]:
    """
    Safer validation:
    - match_strings can contain question, keys, subject, values.
    - erase_strings contains ONLY subject + answer/fact values.
    - LLM-provided erase_strings are not trusted directly because they may contain
      generic keys or full questions.
    """

    question = normalize_str(question)
    answer = normalize_str(answer)

    subject = normalize_str(obj.get("subject"))
    if not subject or subject.lower() in {"null", "none", "unknown"}:
        subject = heuristic_subject_from_question(question) or ""

    subject_type = normalize_key(obj.get("subject_type")) or "unknown"

    raw_facts = obj.get("facts", [])
    if not isinstance(raw_facts, list):
        raw_facts = []

    clean_facts = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue

        key = normalize_key(f.get("key"))
        value = normalize_str(f.get("value"))
        confidence = f.get("confidence", 0.8)

        if not key or not value:
            continue

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.8

        clean_facts.append({
            "key": key,
            "value": value,
            "confidence": confidence,
        })

    # Fallback: answer is always a forgettable value.
    if not clean_facts and answer:
        clean_facts.append({
            "key": "answer",
            "value": answer,
            "confidence": 0.7,
        })

    raw_match_strings = obj.get("match_strings", [])
    if not isinstance(raw_match_strings, list):
        raw_match_strings = []

    match_strings = [normalize_str(x) for x in raw_match_strings if normalize_str(x)]

    # Always match on original question.
    match_strings.append(question)

    erase_strings = []

    # Always erase the answer value and useful answer subphrases.
    for v in split_specific_values(answer):
        if not is_generic_erase_string(v):
            erase_strings.append(v)

    # Erase the subject if present.
    if subject and not is_generic_erase_string(subject):
        match_strings.append(subject)
        erase_strings.append(subject)

    # Facts:
    # - key goes to match_strings only.
    # - value goes to match_strings and erase_strings.
    for f in clean_facts:
        key = f["key"]
        value = f["value"]

        if key:
            match_strings.append(key)

        if value:
            match_strings.append(value)
            for v in split_specific_values(value):
                if not is_generic_erase_string(v):
                    erase_strings.append(v)

    # Deduplicate.
    match_strings = list(dict.fromkeys([s for s in match_strings if normalize_str(s)]))
    erase_strings = list(dict.fromkeys([s for s in erase_strings if normalize_str(s)]))

    return {
        "subject": subject if subject else None,
        "subject_type": subject_type,
        "facts": clean_facts,
        "match_strings": match_strings,
        "erase_strings": erase_strings,
    }


def clean_token_text(token: str) -> str:
    token = token.replace("Ġ", "").replace("▁", "").strip()
    token = re.sub(r"[^A-Za-z0-9'\-]", "", token)
    return token.lower()


def tokenize_erase_strings(target_tokenizer, erase_strings: List[str]) -> List[Dict[str, Any]]:
    """
    Tokenize erase strings using target model tokenizer.

    We tokenize both raw and leading-space variants because LLaMA/Qwen-style
    tokenizers often encode word-in-context differently.
    """
    token_entries = []
    seen = set()

    for text in erase_strings:
        text = normalize_str(text)
        if not text:
            continue

        variants = list(dict.fromkeys([
            text,
            " " + text,
            text.lower(),
            " " + text.lower(),
        ]))

        for variant in variants:
            ids = target_tokenizer.encode(variant, add_special_tokens=False)
            toks = target_tokenizer.convert_ids_to_tokens(ids)

            for tid, raw_tok in zip(ids, toks):
                tid = int(tid)
                cleaned = clean_token_text(raw_tok)

                if not cleaned:
                    continue
                if cleaned in STOP_WORDS:
                    continue
                if len(cleaned) < 2:
                    continue
                if tid in seen:
                    continue

                seen.add(tid)
                token_entries.append({
                    "id": tid,
                    "text": target_tokenizer.decode([tid]),
                    "raw_token": raw_tok,
                    "source_string": text,
                })

    return token_entries


def resolve_torch_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()

    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16

    if dtype in {"fp16", "float16", "half"}:
        return torch.float16

    return torch.float32


def get_input_device(model) -> torch.device:
    """
    Works for normal and device_map='auto' models.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_llm(model_name: str, dtype: str):
    torch_dtype = resolve_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def run_llm_extract(
    model,
    tokenizer,
    question: str,
    answer: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_user_prompt(question, answer)},
    ]

    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            SYSTEM_PROMPT.strip()
            + "\n\n"
            + build_user_prompt(question, answer)
            + "\n\nJSON:"
        )

    input_device = get_input_device(model)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    return extract_json(text)


def load_existing_freq_tokens(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        print(f"[Merge] Existing frequency file not found: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = data.get("semantic_tokens", [])
    print(f"[Merge] Loaded existing frequency tokens: {len(tokens)} from {path}")
    return tokens


def convert_bank_to_semantic_tokens(
    token_bank: List[Dict[str, Any]],
    tokenizer,
) -> List[Dict[str, Any]]:
    semantic_tokens = []

    for x in token_bank:
        tid = int(x["token_id"])
        semantic_tokens.append({
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "freq_forget": int(x.get("record_count", 0)),
            "freq_retain": 0,
            "retain_ratio": 0.0,
            "differential": float(x.get("record_count", 0)),
            "mean_forget_score": 0.0,
            "mean_retain_score": 0.0,
            "best_layer": -1,
            "source": "llm_record_bank_subject_value",
        })

    return semantic_tokens


def merge_with_existing_frequency(
    llm_tokens: List[Dict[str, Any]],
    existing_tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {}

    for t in existing_tokens:
        tid = int(t["token_id"])
        t = dict(t)
        t["token_id"] = tid
        by_id[tid] = t

    for t in llm_tokens:
        tid = int(t["token_id"])

        if tid in by_id:
            old = by_id[tid]
            old_source = old.get("source", "frequency")
            old["source"] = f"{old_source}+llm_record_bank_subject_value"
            old["freq_forget"] = max(
                int(old.get("freq_forget", 0)),
                int(t.get("freq_forget", 0)),
            )
            old["differential"] = max(
                float(old.get("differential", 0.0)),
                float(t.get("differential", 0.0)),
            )
        else:
            by_id[tid] = t

    merged = list(by_id.values())

    merged.sort(
        key=lambda x: (
            0 if "llm_record_bank" in x.get("source", "") else 1,
            -int(x.get("freq_forget", 0)),
            int(x.get("token_id", 0)),
        )
    )

    return merged


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--forget-split", default=None)

    # Target model = model being unlearned. Its tokenizer determines token IDs.
    parser.add_argument("--target-model", default=None)

    # Extractor model = instruction LLM used to produce structured JSON.
    parser.add_argument("--extractor-model", required=True)
    parser.add_argument("--extractor-dtype", default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--out-bank", default=None)
    parser.add_argument("--out-semantic-tokens", default=None)

    parser.add_argument(
        "--merge-existing-freq",
        action="store_true",
        help="Merge LLM bank tokens with existing outputs/semantic_tokens.json from frequency analysis.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit. If unset, process full forget split.",
    )

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    target_model = args.target_model or cfg["model"]["name"]

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_bank = (
        Path(args.out_bank)
        if args.out_bank
        else out_dir / f"forget_knowledge_bank_llm_{forget_split}.json"
    )

    out_semantic = (
        Path(args.out_semantic_tokens)
        if args.out_semantic_tokens
        else out_dir / "semantic_tokens.json"
    )

    print("=" * 80)
    print("[Build LLM Forget Bank]")
    print("=" * 80)
    print(f"[Forget split]     {forget_split}")
    print(f"[Target model]     {target_model}")
    print(f"[Extractor model]  {args.extractor_model}")
    print(f"[Output bank]      {out_bank}")
    print(f"[Output tokens]    {out_semantic}")
    print("=" * 80)

    print(f"\n[Load] Target tokenizer: {target_model}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model)

    print(f"\n[Load] Extractor LLM: {args.extractor_model}")
    extractor_model, extractor_tokenizer = load_llm(
        args.extractor_model,
        args.extractor_dtype,
    )

    print(f"\n[Load] TOFU forget split: {forget_split}")
    ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"[Debug] Using only max_samples={len(ds)}")

    records = []
    token_counter = Counter()
    token_to_records = defaultdict(list)
    token_metadata = {}

    failed = 0

    for idx, sample in enumerate(tqdm(ds, desc="Building LLM forget bank")):
        question = normalize_str(sample["question"])
        answer = normalize_str(sample["answer"])
        record_id = f"{forget_split}_{idx:06d}"

        try:
            raw = run_llm_extract(
                extractor_model,
                extractor_tokenizer,
                question,
                answer,
                max_new_tokens=args.max_new_tokens,
            )
            extracted = validate_extraction(raw, question, answer)

        except Exception as e:
            failed += 1
            print(f"\n[WARN] LLM extraction failed for idx={idx}, record_id={record_id}: {e}")

            extracted = validate_extraction(
                {
                    "subject": heuristic_subject_from_question(question),
                    "subject_type": "unknown",
                    "facts": [
                        {
                            "key": "answer",
                            "value": answer,
                            "confidence": 0.5,
                        }
                    ],
                    "match_strings": [question],
                    "erase_strings": [answer],
                },
                question,
                answer,
            )

        tokens = tokenize_erase_strings(
            target_tokenizer,
            extracted["erase_strings"],
        )

        token_ids = sorted({int(t["id"]) for t in tokens})

        record = {
            "record_id": record_id,
            "split": forget_split,
            "question": question,
            "answer": answer,
            **extracted,
            "token_ids": token_ids,
            "tokens": tokens,
        }

        records.append(record)

        for t in tokens:
            tid = int(t["id"])
            token_counter[tid] += 1
            token_to_records[tid].append(record_id)
            token_metadata[tid] = t

    token_bank = []
    for tid, count in token_counter.most_common():
        tid = int(tid)
        token_bank.append({
            "token_id": tid,
            "token_str": target_tokenizer.decode([tid]),
            "record_count": int(count),
            "record_ids": token_to_records[tid],
            "example_source_string": token_metadata.get(tid, {}).get("source_string", ""),
        })

    bank_output = {
        "method": "llm_structured_forget_knowledge_bank_subject_value_only",
        "forget_split": forget_split,
        "target_model": target_model,
        "extractor_model": args.extractor_model,
        "n_records": len(records),
        "n_failed_extractions": failed,
        "n_unique_tokens": len(token_bank),
        "records": records,
        "token_bank": token_bank,
        "token_ids": [int(x["token_id"]) for x in token_bank],
    }

    with open(out_bank, "w", encoding="utf-8") as f:
        json.dump(bank_output, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Saved LLM forget bank: {out_bank}")
    print(f"[✓] Records: {len(records)}")
    print(f"[✓] Failed extractions: {failed}")
    print(f"[✓] Unique LLM-bank tokens: {len(token_bank)}")

    llm_semantic_tokens = convert_bank_to_semantic_tokens(
        token_bank,
        target_tokenizer,
    )

    if args.merge_existing_freq:
        existing_path = out_dir / "semantic_tokens.json"
        existing_freq_tokens = load_existing_freq_tokens(existing_path)
        semantic_tokens = merge_with_existing_frequency(
            llm_semantic_tokens,
            existing_freq_tokens,
        )
        method_name = "frequency_plus_llm_record_bank_subject_value_only"
    else:
        semantic_tokens = llm_semantic_tokens
        method_name = "llm_record_bank_subject_value_only"

    semantic_output = {
        "method": method_name,
        "forget_split": forget_split,
        "target_model": target_model,
        "extractor_model": args.extractor_model,
        "n_semantic_tokens": len(semantic_tokens),
        "n_llm_record_tokens": sum(
            1 for x in semantic_tokens
            if "llm_record_bank" in x.get("source", "")
        ),
        "n_frequency_tokens": sum(
            1 for x in semantic_tokens
            if "frequency" in x.get("source", "")
        ),
        "token_ids": [int(t["token_id"]) for t in semantic_tokens],
        "token_strings": [t["token_str"] for t in semantic_tokens],
        "semantic_tokens": semantic_tokens,
    }

    with open(out_semantic, "w", encoding="utf-8") as f:
        json.dump(semantic_output, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Saved semantic token file: {out_semantic}")
    print(f"[✓] Total candidate erase tokens: {len(semantic_tokens)}")

    print("\nTop 40 candidate erase tokens:")
    for t in semantic_tokens[:40]:
        print(
            f"  {int(t['token_id']):>8} | "
            f"{repr(t['token_str'])} | "
            f"freq_forget={t.get('freq_forget')} | "
            f"source={t.get('source')}"
        )

    print("\nNext recommended step:")
    print("  Run retain-aware filter before erasure:")
    print("  python scripts/filter_forget_tokens_retain_tfidf.py \\")
    print("    --config <same_config> \\")
    print("    --tokens-json outputs/semantic_tokens_raw_llm_bank.json \\")
    print("    --out outputs/semantic_tokens.json")
    print("\nThen run:")
    print("  python scripts/erase_embeddings.py --config <same_config> --method zero --skip-eval")


if __name__ == "__main__":
    main()