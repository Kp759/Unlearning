import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STOP_WORDS = {
    "what", "who", "when", "where", "which", "why", "how",
    "is", "are", "was", "were", "did", "does", "do",
    "the", "a", "an", "of", "in", "on", "for", "to", "by",
    "known", "name", "author", "book", "written", "receive",
    "born", "answer", "question"
}


SYSTEM_PROMPT = """
You are an information extraction system for machine unlearning.

Your task:
Given one QA pair, extract the private / forgettable knowledge as structured JSON.

Rules:
- Return JSON only.
- Do not add explanations.
- Do not hallucinate facts not supported by the QA pair.
- Infer useful keys automatically.
- Use snake_case keys.
- The "facts" field must be a list of key-value objects.
- The "subject" should be the main entity/person/work being asked about.
- The "erase_strings" should include all strings whose token embeddings should be erased.
- Include both subject tokens and answer/value tokens.
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
    "strings whose embeddings should be erased"
  ]
}}
"""


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    # Remove markdown fences if present.
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    candidate = text[start:end + 1]
    return json.loads(candidate)


def normalize_str(x: Any) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x).strip())


def validate_extraction(obj: Dict[str, Any], question: str, answer: str) -> Dict[str, Any]:
    subject = normalize_str(obj.get("subject"))
    subject_type = normalize_str(obj.get("subject_type")) or "unknown"

    facts = obj.get("facts", [])
    if not isinstance(facts, list):
        facts = []

    clean_facts = []
    for f in facts:
        if not isinstance(f, dict):
            continue

        key = normalize_str(f.get("key")).lower().replace(" ", "_")
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

    # Fallback: always keep answer as a fact.
    if not clean_facts:
        clean_facts.append({
            "key": "answer",
            "value": normalize_str(answer),
            "confidence": 0.7,
        })

    match_strings = obj.get("match_strings", [])
    erase_strings = obj.get("erase_strings", [])

    if not isinstance(match_strings, list):
        match_strings = []
    if not isinstance(erase_strings, list):
        erase_strings = []

    match_strings = [normalize_str(x) for x in match_strings if normalize_str(x)]
    erase_strings = [normalize_str(x) for x in erase_strings if normalize_str(x)]

    # Force important strings into the bank.
    match_strings.append(normalize_str(question))
    erase_strings.append(normalize_str(answer))

    if subject:
        match_strings.append(subject)
        erase_strings.append(subject)

    for f in clean_facts:
        erase_strings.append(f["key"])
        erase_strings.append(f["value"])
        match_strings.append(f["key"])
        match_strings.append(f["value"])

    # Deduplicate while preserving order.
    match_strings = list(dict.fromkeys(match_strings))
    erase_strings = list(dict.fromkeys(erase_strings))

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
    token_entries = []
    seen = set()

    for text in erase_strings:
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
                cleaned = clean_token_text(raw_tok)

                if not cleaned:
                    continue
                if cleaned in STOP_WORDS:
                    continue
                if len(cleaned) < 2:
                    continue
                if int(tid) in seen:
                    continue

                seen.add(int(tid))
                token_entries.append({
                    "id": int(tid),
                    "text": target_tokenizer.decode([int(tid)]),
                    "raw_token": raw_tok,
                    "source_string": text,
                })

    return token_entries


def load_llm(model_name: str, dtype: str):
    if dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

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
def run_llm_extract(model, tokenizer, question: str, answer: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": build_user_prompt(question, answer).strip()},
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
            + build_user_prompt(question, answer).strip()
            + "\n\nJSON:"
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("semantic_tokens", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--forget-split", default=None)

    # Target tokenizer/model = model being unlearned.
    parser.add_argument("--target-model", default=None)

    # Extractor LLM = can be same model or stronger instruction model.
    parser.add_argument("--extractor-model", required=True)

    parser.add_argument("--extractor-dtype", default="float16")
    parser.add_argument("--out-bank", default=None)

    # This writes outputs/semantic_tokens.json so erase_embeddings.py works unchanged.
    parser.add_argument("--out-semantic-tokens", default=None)

    # Merge with existing frequency output.
    parser.add_argument("--merge-existing-freq", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    target_model = args.target_model or cfg["model"]["name"]

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_bank = Path(args.out_bank) if args.out_bank else out_dir / f"forget_knowledge_bank_llm_{forget_split}.json"
    out_semantic = Path(args.out_semantic_tokens) if args.out_semantic_tokens else out_dir / "semantic_tokens.json"

    print(f"[Target tokenizer] {target_model}")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model)

    print(f"[Extractor LLM] {args.extractor_model}")
    extractor_model, extractor_tokenizer = load_llm(args.extractor_model, args.extractor_dtype)

    print(f"[Dataset] locuslab/TOFU | split={forget_split}")
    ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    records = []
    token_counter = Counter()
    token_to_records = defaultdict(list)
    token_metadata = {}

    for idx, sample in enumerate(tqdm(ds, desc="Building LLM forget bank")):
        question = sample["question"]
        answer = sample["answer"]

        try:
            raw = run_llm_extract(
                extractor_model,
                extractor_tokenizer,
                question,
                answer,
            )
            extracted = validate_extraction(raw, question, answer)

        except Exception as e:
            print(f"[WARN] LLM extraction failed for idx={idx}: {e}")
            extracted = validate_extraction(
                {
                    "subject": None,
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

        tokens = tokenize_erase_strings(target_tokenizer, extracted["erase_strings"])
        token_ids = sorted({t["id"] for t in tokens})

        record_id = f"{forget_split}_{idx:06d}"

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
        token_bank.append({
            "token_id": int(tid),
            "token_str": target_tokenizer.decode([int(tid)]),
            "record_count": int(count),
            "record_ids": token_to_records[int(tid)],
        })

    bank_output = {
        "method": "llm_structured_forget_knowledge_bank",
        "forget_split": forget_split,
        "target_model": target_model,
        "extractor_model": args.extractor_model,
        "n_records": len(records),
        "n_unique_tokens": len(token_bank),
        "records": records,
        "token_bank": token_bank,
        "token_ids": [x["token_id"] for x in token_bank],
    }

    with open(out_bank, "w", encoding="utf-8") as f:
        json.dump(bank_output, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved LLM forget bank: {out_bank}")

    # Convert to existing semantic_tokens.json format.
    semantic_tokens = []

    for x in token_bank:
        tid = int(x["token_id"])
        semantic_tokens.append({
            "token_id": tid,
            "token_str": x["token_str"],
            "freq_forget": x["record_count"],
            "freq_retain": 0,
            "retain_ratio": 0.0,
            "differential": float(x["record_count"]),
            "mean_forget_score": 0.0,
            "mean_retain_score": 0.0,
            "best_layer": -1,
            "source": "llm_record_bank",
        })

    if args.merge_existing_freq:
        existing_path = out_dir / "semantic_tokens.json"
        existing = load_existing_freq_tokens(existing_path)

        by_id = {int(t["token_id"]): t for t in existing}

        for t in semantic_tokens:
            tid = int(t["token_id"])
            if tid in by_id:
                old = by_id[tid]
                old["source"] = "frequency+llm_record_bank"
                old["freq_forget"] = max(old.get("freq_forget", 0), t.get("freq_forget", 0))
            else:
                by_id[tid] = t

        semantic_tokens = list(by_id.values())

    semantic_tokens.sort(
        key=lambda x: (
            0 if "llm_record_bank" in x["source"] else 1,
            -x.get("freq_forget", 0),
        )
    )

    semantic_output = {
        "method": "frequency_plus_llm_record_bank" if args.merge_existing_freq else "llm_record_bank_only",
        "forget_split": forget_split,
        "n_semantic_tokens": len(semantic_tokens),
        "n_frequency_tokens": sum(1 for x in semantic_tokens if "frequency" in x["source"]),
        "n_llm_record_tokens": sum(1 for x in semantic_tokens if "llm_record_bank" in x["source"]),
        "token_ids": [int(t["token_id"]) for t in semantic_tokens],
        "token_strings": [t["token_str"] for t in semantic_tokens],
        "semantic_tokens": semantic_tokens,
    }

    with open(out_semantic, "w", encoding="utf-8") as f:
        json.dump(semantic_output, f, indent=2, ensure_ascii=False)

    print(f"[✓] Saved semantic token file for static erasure: {out_semantic}")
    print(f"[✓] Total erase tokens: {len(semantic_tokens)}")

    print("\nTop 30 tokens:")
    for t in semantic_tokens[:30]:
        print(f"  {t['token_id']:>8} | {repr(t['token_str'])} | {t['source']}")


if __name__ == "__main__":
    main()