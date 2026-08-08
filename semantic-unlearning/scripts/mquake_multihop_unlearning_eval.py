#!/usr/bin/env python3
"""Post-selection MQuAKE multi-hop leakage evaluation for unlearning.

This script uses the source-locked official MQuAKE few-shot prompt files and
all three multi-hop questions per sampled forget instance.  It is intentionally
separate from training/repair so these questions cannot affect checkpoint
selection.

MQuAKE is natively a knowledge-editing benchmark: official editing success is
counted when any of the three questions yields the post-edit ``new_answer`` or
an alias.  For unlearning we report the symmetric leakage diagnostic:

    MH-Leak@Any = fraction of instances for which any of the three questions
                  still yields the pre-edit/original ``answer`` or an alias.

Lower is better.  We additionally report the native-style post-edit answer
accuracy as a diagnostic, but it is not an objective of neutral unlearning.
"""

from __future__ import annotations

import argparse
import json
import re
import string
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import dtype_from_str
from mquake_zero_unlearn_official_eval import (
    MQUAKE_FILENAME,
    MQUAKE_REV,
    MQUAKE_URL,
    download_mquake,
    file_sha256,
    load_mquake_raw,
)


STANDARD_PROMPT_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/"
    f"{MQUAKE_REV}/prompts/multihop-prompts.txt"
)
COT_PROMPT_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/"
    f"{MQUAKE_REV}/prompts/multihop-cot-prompts.txt"
)


def download_text(path: Path, url: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    # Exact-match normalization follows common QA practice while preserving
    # alphanumeric content.  Raw generated text is retained in the JSONL.
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def answer_set(answer: str, aliases: Sequence[str]) -> set[str]:
    return {
        norm
        for norm in [normalize_answer(answer), *[normalize_answer(a) for a in aliases]]
        if norm
    }


def parse_standard_generation(text: str) -> str:
    # The official standard prompt is Q:/A: demonstrations.  Use only the first
    # generated answer line and strip accidental continuation into a new Q:.
    text = text.strip()
    if "\nQ:" in text:
        text = text.split("\nQ:", 1)[0]
    if "\nQuestion:" in text:
        text = text.split("\nQuestion:", 1)[0]
    return text.splitlines()[0].strip() if text.splitlines() else text


def parse_cot_generation(text: str) -> str:
    text = text.strip()
    # Prefer an explicit final Answer: span, which matches the official CoT
    # demonstrations.  Otherwise use the final non-empty line.
    matches = list(re.finditer(r"(?i)\banswer\s*:\s*", text))
    if matches:
        answer = text[matches[-1].end() :].strip()
        return answer.splitlines()[0].strip() if answer else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def contains_alias(generated: str, answers: set[str]) -> bool:
    norm = normalize_answer(generated)
    if not norm:
        return False
    return any(value == norm or f" {value} " in f" {norm} " for value in answers)


def exact_alias(generated: str, answers: set[str]) -> bool:
    return normalize_answer(generated) in answers


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_forget_instances(
    mquake_path: Path,
    split_manifest: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = load_mquake_raw(mquake_path)
    manifest = json.loads(Path(split_manifest).read_text(encoding="utf-8"))
    if manifest.get("official_revision") != MQUAKE_REV:
        raise ValueError(
            "Split manifest revision does not match evaluator revision: "
            f"{manifest.get('official_revision')} != {MQUAKE_REV}"
        )
    expected_sha = manifest.get("dataset_sha256")
    actual_sha = file_sha256(mquake_path)
    if expected_sha and expected_sha != actual_sha:
        raise ValueError(
            f"MQuAKE SHA mismatch: manifest={expected_sha}, actual={actual_sha}"
        )
    indices = [int(value) for value in manifest["forget"]["instance_source_indices"]]
    return [raw[index] for index in indices], manifest


def build_prompts(
    instances: Sequence[Mapping[str, Any]],
    task_prompt: str,
    mode: str,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    prompts: List[str] = []
    identities: List[Tuple[int, int]] = []
    for instance_index, record in enumerate(instances):
        questions = record.get("questions", [])
        if not isinstance(questions, list) or len(questions) != 3:
            raise ValueError(
                f"Expected exactly three MQuAKE questions, got {len(questions)}"
            )
        for question_index, question in enumerate(questions):
            if mode == "standard":
                prompt = task_prompt.rstrip() + f"\nQ: {question} A:"
            elif mode == "cot":
                prompt = task_prompt.rstrip() + f"\n\nQuestion: {question}\nThoughts:"
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            prompts.append(prompt)
            identities.append((instance_index, question_index))
    return prompts, identities


@torch.no_grad()
def generate_answers(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    *,
    batch_size: int,
    max_new_tokens: int,
) -> List[str]:
    device = next(model.parameters()).device
    answers: List[str] = []
    for batch in tqdm(list(_chunks(list(prompts), batch_size)), desc="MQuAKE generation"):
        encoded = tok(
            list(batch),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        input_width = int(encoded["input_ids"].shape[1])
        output = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
        continuations = output[:, input_width:]
        answers.extend(tok.batch_decode(continuations, skip_special_tokens=True))
    return answers


def hop_count(record: Mapping[str, Any]) -> int:
    single_hops = record.get("single_hops", [])
    return len(single_hops) if isinstance(single_hops, list) else 0


def evaluate_mode(
    *,
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[Mapping[str, Any]],
    task_prompt: str,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompts, identities = build_prompts(instances, task_prompt, mode)
    generated = generate_answers(
        model,
        tok,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if len(generated) != len(identities):
        raise RuntimeError("Generation count does not match MQuAKE questions")

    rows: List[Dict[str, Any]] = []
    per_instance: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for text, (instance_index, question_index) in zip(generated, identities):
        record = instances[instance_index]
        parser = parse_standard_generation if mode == "standard" else parse_cot_generation
        parsed = parser(text)
        old_answers = answer_set(
            str(record.get("answer", "")),
            [str(value) for value in record.get("answer_alias", [])],
        )
        new_answers = answer_set(
            str(record.get("new_answer", "")),
            [str(value) for value in record.get("new_answer_alias", [])],
        )
        row = {
            "source_instance_index": instance_index,
            "mquake_case_id": int(record.get("case_id", instance_index)),
            "hop_count": hop_count(record),
            "question_index": question_index,
            "question": str(record["questions"][question_index]),
            "raw_generation": text,
            "parsed_answer": parsed,
            "old_answer_exact": exact_alias(parsed, old_answers),
            "old_answer_contains": contains_alias(parsed, old_answers),
            "new_answer_exact": exact_alias(parsed, new_answers),
            "new_answer_contains": contains_alias(parsed, new_answers),
        }
        rows.append(row)
        per_instance[instance_index].append(row)

    instance_rows: List[Dict[str, Any]] = []
    for instance_index, qrows in sorted(per_instance.items()):
        instance_rows.append(
            {
                "instance_index": instance_index,
                "mquake_case_id": qrows[0]["mquake_case_id"],
                "hop_count": qrows[0]["hop_count"],
                "old_exact_any": any(row["old_answer_exact"] for row in qrows),
                "old_contains_any": any(row["old_answer_contains"] for row in qrows),
                "new_exact_any": any(row["new_answer_exact"] for row in qrows),
                "new_contains_any": any(row["new_answer_contains"] for row in qrows),
            }
        )

    def pct(values: Sequence[bool]) -> float:
        return round(100.0 * sum(bool(v) for v in values) / max(1, len(values)), 6)

    by_hop: Dict[str, Dict[str, Any]] = {}
    for hop in sorted({int(row["hop_count"]) for row in instance_rows}):
        selected = [row for row in instance_rows if int(row["hop_count"]) == hop]
        by_hop[str(hop)] = {
            "num_instances": len(selected),
            "MHLeak_exact_any": pct([row["old_exact_any"] for row in selected]),
            "MHLeak_contains_any": pct([row["old_contains_any"] for row in selected]),
            "PostEditAcc_exact_any": pct([row["new_exact_any"] for row in selected]),
        }

    summary = {
        "mode": mode,
        "num_instances": len(instance_rows),
        "num_questions": len(rows),
        "MHLeak_exact_any": pct([row["old_exact_any"] for row in instance_rows]),
        "MHLeak_contains_any": pct([row["old_contains_any"] for row in instance_rows]),
        "PostEditAcc_exact_any": pct([row["new_exact_any"] for row in instance_rows]),
        "PostEditAcc_contains_any": pct([row["new_contains_any"] for row in instance_rows]),
        "by_hop": by_hop,
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mquake-path", default=f"data/{MQUAKE_FILENAME}")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--prompt-dir", default="data/mquake_prompts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["standard", "cot", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--standard-max-new-tokens", type=int, default=32)
    parser.add_argument("--cot-max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    args = parser.parse_args()

    mquake_path = download_mquake(Path(args.mquake_path), url=MQUAKE_URL)
    instances, manifest = load_forget_instances(mquake_path, Path(args.split_manifest))

    prompt_dir = Path(args.prompt_dir)
    standard_path = download_text(
        prompt_dir / "multihop-prompts.txt",
        STANDARD_PROMPT_URL,
    )
    cot_path = download_text(
        prompt_dir / "multihop-cot-prompts.txt",
        COT_PROMPT_URL,
    )

    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype_from_str(args.dtype)}
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    if args.device_map != "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Decoder-only batch generation is safest with left padding.
    tok.padding_side = "left"

    modes = [args.mode] if args.mode != "both" else ["standard", "cot"]
    results: Dict[str, Any] = {}
    raw_by_mode: Dict[str, Any] = {}
    for mode in modes:
        task_prompt = (
            standard_path.read_text(encoding="utf-8")
            if mode == "standard"
            else cot_path.read_text(encoding="utf-8")
        )
        max_new = (
            args.standard_max_new_tokens
            if mode == "standard"
            else args.cot_max_new_tokens
        )
        summary, rows = evaluate_mode(
            model=model,
            tok=tok,
            instances=instances,
            task_prompt=task_prompt,
            mode=mode,
            batch_size=args.batch_size,
            max_new_tokens=max_new,
        )
        results[mode] = summary
        raw_by_mode[mode] = rows

    output = {
        "dataset": MQUAKE_FILENAME,
        "dataset_revision": MQUAKE_REV,
        "dataset_sha256": file_sha256(mquake_path),
        "model_dir": str(args.model_dir),
        "split_manifest": str(args.split_manifest),
        "seed": manifest.get("seed"),
        "protocol": {
            "prompt_source": "official source-locked MQuAKE multihop prompt files",
            "question_rule": "all three questions per forget instance",
            "unlearning_metric": "old/pre-edit answer leakage; any-of-three is a failure",
            "native_editing_diagnostic": "new/post-edit answer; any-of-three is success",
            "checkpoint_selection": "strictly post-selection; no multi-hop question is repair data",
        },
        "results": results,
        "raw": raw_by_mode,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for mode, summary in results.items():
        print(
            f"{mode}: MHLeak_exact_any={summary['MHLeak_exact_any']}%, "
            f"MHLeak_contains_any={summary['MHLeak_contains_any']}%, "
            f"PostEditAcc_exact_any={summary['PostEditAcc_exact_any']}%"
        )


if __name__ == "__main__":
    main()
