#!/usr/bin/env python3
"""ZeroUnlearn-compatible evaluation for the ZsRE MEND benchmark.

The vendored ZeroUnlearn adapter turns each raw ZsRE row into:

* one rewrite prompt containing the original answer to forget;
* one paraphrase prompt containing the same answer;
* token-prefix neighborhood prompts containing an unrelated answer to retain.

Unlike CounterFact, ZsRE is evaluated with token-level greedy accuracy.  Eff
and Gen are therefore lower-is-better original-answer accuracies, while Spe is
higher-is-better neighborhood-answer accuracy.  This module deliberately
mirrors ``ZeroUnlearn/dsets/zsre.py`` and
``ZeroUnlearn/experiments/py/eval_utils_zsre.py`` while supporting bounded
evaluation batches and CPU unit tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    is_llama_like,
    load_official_ppl_text,
    official_perplexity,
)


ZSRE_URL = "https://memit.baulab.info/data/dsets/zsre_mend_eval.json"
UPSTREAM_NEUTRAL_TARGET = "<|endoftext|>"


@dataclass(frozen=True)
class PredictionCase:
    """One exact next-token decision made by the official ZsRE evaluator."""

    case_id: int
    prompt_type: str
    prompt_index: int
    token_index: int
    prompt: str
    target_text: str

    @property
    def identity(self) -> Tuple[int, str, int, int]:
        return (
            self.case_id,
            self.prompt_type,
            self.prompt_index,
            self.token_index,
        )


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _input_ids(tokenized: Any) -> Any:
    if isinstance(tokenized, Mapping):
        return tokenized["input_ids"]
    return tokenized.input_ids


def _flat_ids(tok: Any, text: str) -> List[int]:
    ids = _input_ids(tok(text))
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized sequence")
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def download_zsre(path: Path, url: str = ZSRE_URL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading ZsRE to {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_zsre_raw(path: Path, url: str = ZSRE_URL) -> List[Dict[str, Any]]:
    path = download_zsre(path, url=url)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("ZsRE JSON must contain a list of records")
    return raw


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_official_zsre_raw_records(
    raw: Sequence[Dict[str, Any]],
    forget_num: int,
    retain_num: int,
    seed: int,
    *,
    strict: bool = False,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
    """Match ZeroUnlearn: sample forget from half two, then retain from half one."""

    indexed = list(enumerate(raw))
    half = len(indexed) // 2
    retain_pool = indexed[:half]
    forget_pool = indexed[half:]
    if strict and (
        len(forget_pool) < forget_num or len(retain_pool) < retain_num
    ):
        raise ValueError(
            "Official ZsRE split is too small: "
            f"forget_pool={len(forget_pool)}, retain_pool={len(retain_pool)}, "
            f"requested forget={forget_num}, retain={retain_num}"
        )
    forget_num = min(forget_num, len(forget_pool))
    retain_num = min(retain_num, len(retain_pool))
    rng = random.Random(seed)
    forget = rng.sample(forget_pool, k=forget_num)
    retain = rng.sample(retain_pool, k=retain_num)
    return forget, retain


def build_zsre_record(
    raw_record: Mapping[str, Any],
    case_id: int,
    tok: Any,
) -> Dict[str, Any]:
    """Mirror ``MENDQADataset`` without importing the vendored package."""

    required = (
        "src",
        "subject",
        "answers",
        "rephrase",
        "loc",
        "loc_ans",
    )
    missing = [key for key in required if key not in raw_record]
    if missing:
        raise ValueError(f"ZsRE record {case_id} is missing fields: {missing}")
    if "nq question: " not in str(raw_record["loc"]):
        raise ValueError(
            f"ZsRE record {case_id} neighborhood prompt lacks 'nq question: '"
        )
    answers = raw_record["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"ZsRE record {case_id} has no original answer")

    location_answer_ids = _flat_ids(tok, " " + str(raw_record["loc_ans"]))
    neighborhood_prompts = [
        {
            "prompt": (
                str(raw_record["loc"])
                + "?"
                + tok.decode(location_answer_ids[:token_index])
            ),
            "target": tok.decode([location_answer_ids[token_index]]),
        }
        for token_index in range(len(location_answer_ids))
    ]
    subject = str(raw_record["subject"])
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(raw_record["src"]).replace(subject, "{}"),
            "subject": subject,
            "target_new": {"str": UPSTREAM_NEUTRAL_TARGET},
            "target_true": {"str": str(answers[0])},
        },
        "paraphrase_prompts": [str(raw_record["rephrase"])],
        "neighborhood_prompts": neighborhood_prompts,
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def load_official_eval_records(
    zsre_path: Path,
    tok: Any,
    forget_num: int,
    retain_num: int,
    seed: int,
    *,
    zsre_url: str = ZSRE_URL,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw = load_zsre_raw(zsre_path, url=zsre_url)
    forget_raw, retain_raw = sample_official_zsre_raw_records(
        raw,
        forget_num=forget_num,
        retain_num=retain_num,
        seed=seed,
        strict=True,
    )
    forget = [
        build_zsre_record(record, case_id, tok)
        for case_id, record in forget_raw
    ]
    retain = [
        build_zsre_record(record, case_id, tok)
        for case_id, record in retain_raw
    ]
    return forget, retain


def original_answer_token_ids(
    tok: Any,
    answer: str,
    *,
    llama_like: bool,
) -> List[int]:
    target_ids = _flat_ids(tok, " " + answer)
    if llama_like:
        if not target_ids:
            raise ValueError("Llama-style target tokenization returned no BOS token")
        target_ids = target_ids[1:]
    if not target_ids:
        raise ValueError(f"Original answer tokenized to no evaluated tokens: {answer!r}")
    return target_ids


def expand_prediction_cases(
    record: Mapping[str, Any],
    tok: Any,
    *,
    llama_like: bool,
    prompt_types: Sequence[str] = ("rewrite", "paraphrase", "neighborhood"),
) -> List[PredictionCase]:
    """Expand one adapted record into the official next-token decisions."""

    case_id = int(record["case_id"])
    rewrite = record["requested_rewrite"]
    subject = str(rewrite["subject"])
    target_true = str(rewrite["target_true"]["str"])
    target_ids = original_answer_token_ids(
        tok,
        target_true,
        llama_like=llama_like,
    )

    cases: List[PredictionCase] = []
    prompt_groups = {
        "rewrite": [str(rewrite["prompt"]).format(subject)],
        "paraphrase": [str(value) for value in record["paraphrase_prompts"]],
    }
    for prompt_type in ("rewrite", "paraphrase"):
        if prompt_type not in prompt_types:
            continue
        for prompt_index, prompt in enumerate(prompt_groups[prompt_type]):
            for token_index, token_id in enumerate(target_ids):
                decoded_prefix = tok.decode(target_ids[:token_index])
                if llama_like and token_index > 0:
                    evaluated_prompt = prompt + " " + decoded_prefix
                else:
                    evaluated_prompt = prompt + decoded_prefix
                cases.append(
                    PredictionCase(
                        case_id=case_id,
                        prompt_type=prompt_type,
                        prompt_index=prompt_index,
                        token_index=token_index,
                        prompt=evaluated_prompt,
                        target_text=tok.decode([token_id]),
                    )
                )

    if "neighborhood" in prompt_types:
        for prompt_index, item in enumerate(record["neighborhood_prompts"]):
            cases.append(
                PredictionCase(
                    case_id=case_id,
                    prompt_type="neighborhood",
                    prompt_index=prompt_index,
                    token_index=prompt_index,
                    prompt=str(item["prompt"]),
                    target_text=str(item["target"]),
                )
            )
    return cases


def official_target_ids(
    tok: Any,
    target_texts: Sequence[str],
    *,
    llama_like: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = tok(
        list(target_texts),
        padding=True,
        return_tensors="pt",
    )
    input_ids = _input_ids(encoded)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    column = 1 if llama_like else 0
    if input_ids.shape[1] <= column:
        raise ValueError("Official target tokenization lacks the expected token column")
    return input_ids[:, column].to(device)


@torch.no_grad()
def predict_cases(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[PredictionCase],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int = 8,
    desc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run the same last-non-padding-token argmax used upstream."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    rows: List[Dict[str, Any]] = []
    batches = list(_chunks(list(cases), batch_size))
    iterator = tqdm(batches, desc=desc, leave=False) if desc else batches
    for batch in iterator:
        encoded = tok(
            [case.prompt for case in batch],
            padding=True,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded, use_cache=False)
        attention = encoded["attention_mask"]
        last_non_masked = attention.sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        final_logits = output.logits[batch_indices, last_non_masked, :]
        predicted_ids = final_logits.argmax(dim=-1)
        target_ids = official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        for case, predicted_id, target_id in zip(
            batch,
            predicted_ids.detach().cpu().tolist(),
            target_ids.detach().cpu().tolist(),
        ):
            rows.append(
                {
                    **asdict(case),
                    "target_token_id": int(target_id),
                    "predicted_token_id": int(predicted_id),
                    "correct": bool(predicted_id == target_id),
                }
            )
    return rows


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def official_summarize(
    split_name: str,
    metric_data: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Match ZeroUnlearn's per-record macro averaging and 0-100 scaling."""

    output: Dict[str, Any] = {
        "split_name": split_name,
        "num_cases": len(metric_data),
    }
    key_map = {
        "rewrite": "rewrite_prompts_correct",
        "paraphrase": "paraphrase_prompts_correct",
        "neighborhood": "neighborhood_prompts_correct",
    }
    for display, key in key_map.items():
        per_record = [
            float(np.mean(item["post"][key]))
            for item in metric_data
            if item["post"].get(key)
        ]
        all_tokens = [
            bool(value)
            for item in metric_data
            for value in item["post"].get(key, [])
        ]
        macro = _mean_or_none(per_record)
        micro = _mean_or_none([float(value) for value in all_tokens])
        output[f"post_{display}_acc"] = [
            None if macro is None else round(100.0 * macro, 6),
            None
            if not per_record
            else round(100.0 * float(np.std(per_record)), 6),
        ]
        output[f"post_{display}_micro_acc"] = (
            None if micro is None else round(100.0 * micro, 6)
        )
        output[f"post_{display}_correct_tokens"] = int(sum(all_tokens))
        output[f"post_{display}_total_tokens"] = len(all_tokens)

    output["Eff"] = output["post_rewrite_acc"][0]
    output["Gen"] = output["post_paraphrase_acc"][0]
    output["Spe"] = output["post_neighborhood_acc"][0]
    return output


@torch.no_grad()
def evaluate_record_split(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    split_name: str,
    batch_size: int = 8,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    cases = [
        case
        for record in records
        for case in expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    predicted = predict_cases(
        model,
        tok,
        cases,
        device,
        llama_like=llama_like,
        batch_size=batch_size,
        desc=f"ZsRE {split_name}",
    )
    by_record: Dict[int, Dict[str, List[bool]]] = {
        int(record["case_id"]): {
            "rewrite": [],
            "paraphrase": [],
            "neighborhood": [],
        }
        for record in records
    }
    for row in predicted:
        by_record[int(row["case_id"])][str(row["prompt_type"])].append(
            bool(row["correct"])
        )

    metric_data: List[Dict[str, Any]] = []
    for record in records:
        grouped = by_record[int(record["case_id"])]
        metric_data.append(
            {
                "case_id": int(record["case_id"]),
                "requested_rewrite": record["requested_rewrite"],
                "post": {
                    "rewrite_prompts_correct": grouped["rewrite"],
                    "paraphrase_prompts_correct": grouped["paraphrase"],
                    "neighborhood_prompts_correct": grouped["neighborhood"],
                },
            }
        )
    return official_summarize(split_name, metric_data), metric_data


def evaluate_loaded_model_official(
    *,
    method: str,
    model: torch.nn.Module,
    tok: Any,
    model_dir: Any,
    zsre_path: Path,
    wikidata_dir: Path,
    out_path: Optional[Path] = None,
    forget_num: int = 50,
    retain_num: int = 1000,
    seed: int = 1,
    batch_size: int = 8,
    skip_ppl: bool = False,
    zsre_url: str = ZSRE_URL,
    records: Optional[
        Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    llama_like = is_llama_like(model, tok)
    if records is None:
        forget_records, retain_records = load_official_eval_records(
            zsre_path,
            tok,
            forget_num=forget_num,
            retain_num=retain_num,
            seed=seed,
            zsre_url=zsre_url,
        )
    else:
        forget_records, retain_records = records

    forget_summary, forget_raw = evaluate_record_split(
        model,
        tok,
        forget_records,
        device,
        llama_like=llama_like,
        split_name="forget",
        batch_size=batch_size,
    )
    retain_summary, retain_raw = evaluate_record_split(
        model,
        tok,
        retain_records,
        device,
        llama_like=llama_like,
        split_name="retain",
        batch_size=batch_size,
    )

    ppl = None
    if not skip_ppl:
        ppl_text = load_official_ppl_text(wikidata_dir)
        if ppl_text is None:
            print(f"[warning] wikidata dir {wikidata_dir} not found; PPL is null")
        else:
            ppl = official_perplexity(
                model,
                tok,
                ppl_text,
                device,
                max_input_length=100,
            )

    result = {
        "method": method,
        "model_dir": str(model_dir),
        "dataset": "ZsRE",
        "protocol": "ZeroUnlearn official-compatible token accuracy",
        "seed": int(seed),
        "unlearn_num": int(forget_num),
        "retain_num": int(retain_num),
        "llama_like": bool(llama_like),
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def evaluate_model_dir_official(
    *,
    method: str,
    model_dir: Path,
    zsre_path: Path,
    wikidata_dir: Path,
    out_path: Path,
    forget_num: int,
    retain_num: int,
    seed: int,
    dtype: str,
    device_map: str,
    batch_size: int,
    skip_ppl: bool,
    zsre_url: str,
) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype_from_str(dtype)}
    if device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    if device_map != "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required unless --device-map auto is usable")
        model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(model_dir)
    return evaluate_loaded_model_official(
        method=method,
        model=model,
        tok=tok,
        model_dir=model_dir,
        zsre_path=zsre_path,
        wikidata_dir=wikidata_dir,
        out_path=out_path,
        forget_num=forget_num,
        retain_num=retain_num,
        seed=seed,
        batch_size=batch_size,
        skip_ppl=skip_ppl,
        zsre_url=zsre_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--zsre-url", default=ZSRE_URL)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", default="checkpoint")
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args()

    result = evaluate_model_dir_official(
        method=args.method,
        model_dir=Path(args.model_dir),
        zsre_path=Path(args.zsre_path),
        wikidata_dir=Path(args.wikidata_dir),
        out_path=Path(args.out),
        forget_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
    )
    print(
        "ZsRE official result: "
        f"Eff={result['forget']['Eff']}, "
        f"Gen={result['forget']['Gen']}, "
        f"Spe={result['forget']['Spe']}, "
        f"PPL={result['forget_PPL']}"
    )


if __name__ == "__main__":
    main()
