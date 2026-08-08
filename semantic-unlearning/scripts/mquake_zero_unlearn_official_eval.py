#!/usr/bin/env python3
"""Source-locked MQuAKE-CF-3k-v2 evaluation for semantic unlearning.

This module intentionally exposes two layers of evaluation:

1. ``Eff``: a ZeroUnlearn-code-compatible atomic erasure metric.  MQuAKE
   instances are sampled at the instance level exactly like ZeroUnlearn
   (first half retain, second half forget), then each ``requested_rewrite`` is
   flattened into an atomic fact.  The original ``target_true`` answer is
   teacher-forced token by token and scored by next-token argmax accuracy.
   Lower is better; 0.00 means no evaluated sensitive token remains argmax.

2. Extensions that are useful for a paper but are *not* native ZeroUnlearn
   MQuAKE columns:
      * ``AtomicGen``: the same sensitive-answer token accuracy on MQuAKE's
        natural-language single-hop ``question`` for each rewrite.
      * retain atomic rewrite/question accuracy, for utility auditing.
      * a split/source manifest that records instance IDs and flattened facts.

The official MQuAKE multi-hop generation protocol is deliberately kept out of
this file.  It should be evaluated only after checkpoint selection so the
three official multi-hop questions never become repair/tuning data.

Dataset source is pinned to the MQuAKE September-2024 v2 update.
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


MQUAKE_REV = "fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd"
MQUAKE_FILENAME = "MQuAKE-CF-3k-v2.json"
MQUAKE_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/"
    f"{MQUAKE_REV}/datasets/{MQUAKE_FILENAME}"
)
NEUTRAL_TARGET = "Unknown"


@dataclass(frozen=True)
class PredictionCase:
    """One teacher-forced next-token decision."""

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


def resolve_neutral_target_token_id(
    tok: Any,
    target: str = NEUTRAL_TARGET,
) -> int:
    """Require the active-repair neutral answer to be exactly one token."""

    ids = _input_ids(tok(target, add_special_tokens=False))
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one neutral-target sequence")
        ids = ids[0]
    token_ids = [int(value) for value in ids]
    if len(token_ids) != 1:
        raise ValueError(
            f"Neutral target {target!r} must tokenize to exactly one token; "
            f"got {token_ids}"
        )
    return token_ids[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_mquake(path: Path, url: str = MQUAKE_URL) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading source-locked MQuAKE to {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_mquake_raw(path: Path, url: str = MQUAKE_URL) -> List[Dict[str, Any]]:
    path = download_mquake(path, url=url)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("MQuAKE JSON must contain a list")
    if len(raw) != 3000:
        raise ValueError(
            f"Expected 3000 MQuAKE-CF-3k-v2 instances, found {len(raw)}"
        )
    return raw


def sample_zerounlearn_instances(
    raw: Sequence[Dict[str, Any]],
    forget_num: int,
    retain_num: int,
    seed: int,
    *,
    strict: bool = True,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
    """Mirror ZeroUnlearn's first-half retain / second-half forget sampling."""

    indexed = list(enumerate(raw))
    half = len(indexed) // 2
    retain_pool = indexed[:half]
    forget_pool = indexed[half:]
    if strict and (
        forget_num > len(forget_pool) or retain_num > len(retain_pool)
    ):
        raise ValueError(
            "Requested split exceeds MQuAKE pools: "
            f"forget={forget_num}/{len(forget_pool)}, "
            f"retain={retain_num}/{len(retain_pool)}"
        )
    forget_num = min(forget_num, len(forget_pool))
    retain_num = min(retain_num, len(retain_pool))

    # ZeroUnlearn samples forget first and retain second from one seeded RNG.
    rng = random.Random(seed)
    forget = rng.sample(forget_pool, k=forget_num)
    retain = rng.sample(retain_pool, k=retain_num)
    return forget, retain


def _atomic_case_id(source_index: int, rewrite_index: int) -> int:
    # MQuAKE chains contain only a handful of rewrites.  A width of 100 leaves
    # ample room while keeping IDs deterministic and human-decodable.
    if not 0 <= rewrite_index < 100:
        raise ValueError(f"rewrite_index out of range: {rewrite_index}")
    return int(source_index) * 100 + int(rewrite_index)


def build_atomic_records(
    raw_record: Mapping[str, Any],
    source_index: int,
) -> List[Dict[str, Any]]:
    """Flatten one MQuAKE instance after instance-level sampling.

    ``target_true`` is the sensitive/original fact to forget.  The original
    counterfactual ``target_new`` is retained only as provenance; it is *not*
    the desired unlearning answer.  The method-facing desired answer is the
    one-token neutral target ``Unknown``.
    """

    rewrites = raw_record.get("requested_rewrite")
    if not isinstance(rewrites, list) or not rewrites:
        raise ValueError(
            f"MQuAKE instance {source_index} has no requested_rewrite list"
        )
    multihop_questions = raw_record.get("questions", [])
    if not isinstance(multihop_questions, list):
        raise ValueError(f"MQuAKE instance {source_index} questions is not a list")

    mquake_case_id = int(raw_record.get("case_id", source_index))
    records: List[Dict[str, Any]] = []
    for rewrite_index, rewrite in enumerate(rewrites):
        if not isinstance(rewrite, Mapping):
            raise ValueError(
                f"MQuAKE instance {source_index} rewrite {rewrite_index} is invalid"
            )
        required = ("prompt", "subject", "target_true", "target_new", "question")
        missing = [key for key in required if key not in rewrite]
        if missing:
            raise ValueError(
                f"MQuAKE instance {source_index} rewrite {rewrite_index} "
                f"missing {missing}"
            )
        target_true = rewrite["target_true"]
        target_new = rewrite["target_new"]
        if not isinstance(target_true, Mapping) or not str(target_true.get("str", "")):
            raise ValueError("MQuAKE target_true must contain a non-empty str")
        if not isinstance(target_new, Mapping) or not str(target_new.get("str", "")):
            raise ValueError("MQuAKE target_new must contain a non-empty str")

        records.append(
            {
                "case_id": _atomic_case_id(source_index, rewrite_index),
                "mquake_case_id": mquake_case_id,
                "source_index": int(source_index),
                "rewrite_index": int(rewrite_index),
                "requested_rewrite": {
                    "prompt": str(rewrite["prompt"]),
                    "subject": str(rewrite["subject"]),
                    "relation_id": rewrite.get("relation_id"),
                    # Method-facing semantics: sensitive original -> neutral.
                    "target_true": {"str": str(target_true["str"])},
                    "target_new": {"str": NEUTRAL_TARGET},
                    # Preserve the benchmark's counterfactual target verbatim.
                    "mquake_target_new": dict(target_new),
                    "question": str(rewrite["question"]),
                },
                # Held out from Setting 5e and active repair.
                "atomic_gen_prompt": str(rewrite["question"]),
                "multihop_questions": [str(q) for q in multihop_questions],
                "multihop_answer": str(raw_record.get("answer", "")),
                "multihop_answer_alias": [
                    str(value) for value in raw_record.get("answer_alias", [])
                ],
                "multihop_new_answer": str(raw_record.get("new_answer", "")),
                "multihop_new_answer_alias": [
                    str(value) for value in raw_record.get("new_answer_alias", [])
                ],
                # Kept empty on purpose: no MQuAKE test question is repair data.
                "paraphrase_prompts": [],
                "neighborhood_prompts": [],
                "attribute_prompts": [],
                "generation_prompts": [],
            }
        )
    return records


def flatten_sampled_instances(
    sampled: Sequence[Tuple[int, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    return [
        record
        for source_index, raw_record in sampled
        for record in build_atomic_records(raw_record, source_index)
    ]


def load_official_eval_records(
    mquake_path: Path,
    tok: Any,
    forget_num: int,
    retain_num: int,
    seed: int,
    *,
    mquake_url: str = MQUAKE_URL,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Tokenizer argument is accepted to match the existing benchmark adapters.
    resolve_neutral_target_token_id(tok)
    raw = load_mquake_raw(mquake_path, url=mquake_url)
    forget_instances, retain_instances = sample_zerounlearn_instances(
        raw,
        forget_num=forget_num,
        retain_num=retain_num,
        seed=seed,
        strict=True,
    )
    return (
        flatten_sampled_instances(forget_instances),
        flatten_sampled_instances(retain_instances),
    )


def original_answer_token_ids(
    tok: Any,
    answer: str,
    *,
    llama_like: bool,
) -> List[int]:
    """Mirror ZeroUnlearn's MQuAKE teacher-forced target tokenization."""

    target_ids = _flat_ids(tok, " " + answer)
    if llama_like:
        if not target_ids:
            raise ValueError("Llama-style target tokenization returned no BOS token")
        target_ids = target_ids[1:]
    if not target_ids:
        raise ValueError(f"Sensitive answer has no evaluated tokens: {answer!r}")
    return target_ids


def expand_prediction_cases(
    record: Mapping[str, Any],
    tok: Any,
    *,
    llama_like: bool,
    prompt_types: Sequence[str] = ("rewrite", "atomic_gen"),
) -> List[PredictionCase]:
    """Expand an atomic record into exact token-prefix decisions."""

    rewrite = record["requested_rewrite"]
    subject = str(rewrite["subject"])
    sensitive = str(rewrite["target_true"]["str"])
    target_ids = original_answer_token_ids(tok, sensitive, llama_like=llama_like)
    prompt_groups = {
        "rewrite": [str(rewrite["prompt"]).format(subject)],
        "atomic_gen": [str(record["atomic_gen_prompt"])],
    }

    cases: List[PredictionCase] = []
    for prompt_type in prompt_types:
        if prompt_type not in prompt_groups:
            raise ValueError(f"Unsupported MQuAKE prompt type: {prompt_type}")
        for prompt_index, prompt in enumerate(prompt_groups[prompt_type]):
            for token_index, token_id in enumerate(target_ids):
                decoded_prefix = tok.decode(target_ids[:token_index])
                if llama_like and token_index > 0:
                    evaluated_prompt = prompt + " " + decoded_prefix
                else:
                    evaluated_prompt = prompt + decoded_prefix
                cases.append(
                    PredictionCase(
                        case_id=int(record["case_id"]),
                        prompt_type=prompt_type,
                        prompt_index=prompt_index,
                        token_index=token_index,
                        prompt=evaluated_prompt,
                        target_text=tok.decode([token_id]),
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
    encoded = tok(list(target_texts), padding=True, return_tensors="pt")
    input_ids = _input_ids(encoded)
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    column = 1 if llama_like else 0
    if input_ids.shape[1] <= column:
        raise ValueError("Target tokenization lacks the expected token column")
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
    """Use the same last-non-padding-token argmax convention as ZeroUnlearn."""

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
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
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


def summarize_atomic_split(
    split_name: str,
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_atomic: Dict[int, Dict[str, List[bool]]] = {
        int(record["case_id"]): {"rewrite": [], "atomic_gen": []}
        for record in records
    }
    record_by_atomic = {int(record["case_id"]): record for record in records}
    for row in rows:
        by_atomic[int(row["case_id"])][str(row["prompt_type"])].append(
            bool(row["correct"])
        )

    output: Dict[str, Any] = {
        "split_name": split_name,
        "num_atomic_facts": len(records),
        "num_instances": len({int(r["source_index"]) for r in records}),
    }
    for prompt_type, display in (("rewrite", "Eff"), ("atomic_gen", "AtomicGen")):
        per_atomic = [
            float(np.mean(by_atomic[int(record["case_id"])][prompt_type]))
            for record in records
            if by_atomic[int(record["case_id"])][prompt_type]
        ]
        all_tokens = [
            bool(value)
            for grouped in by_atomic.values()
            for value in grouped[prompt_type]
        ]
        atomic_macro = _mean_or_none(per_atomic)
        micro = _mean_or_none([float(value) for value in all_tokens])

        by_instance: Dict[int, List[float]] = {}
        for record in records:
            values = by_atomic[int(record["case_id"])][prompt_type]
            if not values:
                continue
            by_instance.setdefault(int(record["source_index"]), []).append(
                float(np.mean(values))
            )
        instance_values = [float(np.mean(values)) for values in by_instance.values()]
        instance_macro = _mean_or_none(instance_values)

        output[display] = None if atomic_macro is None else round(100.0 * atomic_macro, 6)
        output[f"{display}_micro"] = None if micro is None else round(100.0 * micro, 6)
        output[f"{display}_instance_macro"] = (
            None if instance_macro is None else round(100.0 * instance_macro, 6)
        )
        output[f"{display}_correct_tokens"] = int(sum(all_tokens))
        output[f"{display}_total_tokens"] = len(all_tokens)

    # Keep provenance compact but sufficient to reproduce the split exactly.
    output["source_indices"] = sorted({int(r["source_index"]) for r in records})
    output["mquake_case_ids"] = sorted({int(r["mquake_case_id"]) for r in records})
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
    include_atomic_gen: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt_types = ("rewrite", "atomic_gen") if include_atomic_gen else ("rewrite",)
    cases = [
        case
        for record in records
        for case in expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=prompt_types,
        )
    ]
    predicted = predict_cases(
        model,
        tok,
        cases,
        device,
        llama_like=llama_like,
        batch_size=batch_size,
        desc=f"MQuAKE {split_name}",
    )
    summary = summarize_atomic_split(split_name, records, predicted)
    return summary, predicted


def write_split_manifest(
    path: Path,
    *,
    mquake_path: Path,
    seed: int,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "dataset": MQUAKE_FILENAME,
        "official_repository": "princeton-nlp/MQuAKE",
        "official_revision": MQUAKE_REV,
        "dataset_sha256": file_sha256(mquake_path),
        "seed": int(seed),
        "sampling": "first-half retain pool / second-half forget pool; sample forget first then retain",
        "flattening": "sample instances first, then flatten requested_rewrite",
        "forget": {
            "instance_source_indices": sorted(
                {int(r["source_index"]) for r in forget_records}
            ),
            "atomic_case_ids": [int(r["case_id"]) for r in forget_records],
        },
        "retain": {
            "instance_source_indices": sorted(
                {int(r["source_index"]) for r in retain_records}
            ),
            "atomic_case_ids": [int(r["case_id"]) for r in retain_records],
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_loaded_model_official(
    *,
    method: str,
    model: torch.nn.Module,
    tok: Any,
    model_dir: Any,
    mquake_path: Path,
    wikidata_dir: Path,
    out_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    forget_num: int = 1000,
    retain_num: int = 1000,
    seed: int = 0,
    batch_size: int = 8,
    skip_ppl: bool = False,
    include_atomic_gen: bool = True,
    mquake_url: str = MQUAKE_URL,
    records: Optional[
        Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]
    ] = None,
) -> Dict[str, Any]:
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    resolve_neutral_target_token_id(tok)
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    llama_like = is_llama_like(model, tok)

    mquake_path = download_mquake(Path(mquake_path), url=mquake_url)
    if records is None:
        forget_records, retain_records = load_official_eval_records(
            mquake_path,
            tok,
            forget_num=forget_num,
            retain_num=retain_num,
            seed=seed,
            mquake_url=mquake_url,
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
        include_atomic_gen=include_atomic_gen,
    )
    retain_summary, retain_raw = evaluate_record_split(
        model,
        tok,
        retain_records,
        device,
        llama_like=llama_like,
        split_name="retain",
        batch_size=batch_size,
        include_atomic_gen=include_atomic_gen,
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
        "dataset": MQUAKE_FILENAME,
        "dataset_revision": MQUAKE_REV,
        "dataset_sha256": file_sha256(mquake_path),
        "protocol": {
            "main": "ZeroUnlearn-compatible atomic MQuAKE Eff token accuracy",
            "Eff": "teacher-forced original target_true next-token argmax accuracy; lower is better",
            "AtomicGen": "held-out atomic question accuracy; extension, not a native ZeroUnlearn MQuAKE column",
            "sampling": "1000-instance default: first-half retain / second-half forget, seeded random sampling",
            "flattening": "requested_rewrite is flattened only after instance sampling",
            "neutral_target": NEUTRAL_TARGET,
        },
        "seed": int(seed),
        "unlearn_num_instances": int(forget_num),
        "retain_num_instances": int(retain_num),
        "llama_like": bool(llama_like),
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
    }

    if manifest_path is not None:
        write_split_manifest(
            manifest_path,
            mquake_path=mquake_path,
            seed=seed,
            forget_records=forget_records,
            retain_records=retain_records,
        )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result


def evaluate_model_dir_official(
    *,
    method: str,
    model_dir: Path,
    mquake_path: Path,
    wikidata_dir: Path,
    out_path: Path,
    manifest_path: Optional[Path],
    forget_num: int,
    retain_num: int,
    seed: int,
    dtype: str,
    device_map: str,
    batch_size: int,
    skip_ppl: bool,
    include_atomic_gen: bool,
    mquake_url: str,
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
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=out_path,
        manifest_path=manifest_path,
        forget_num=forget_num,
        retain_num=retain_num,
        seed=seed,
        batch_size=batch_size,
        skip_ppl=skip_ppl,
        include_atomic_gen=include_atomic_gen,
        mquake_url=mquake_url,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mquake-path", default=f"data/{MQUAKE_FILENAME}")
    parser.add_argument("--mquake-url", default=MQUAKE_URL)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--method", default="checkpoint")
    parser.add_argument("--unlearn-num", type=int, default=1000)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument(
        "--skip-atomic-gen",
        action="store_true",
        help="Evaluate only ZeroUnlearn-compatible Eff and PPL.",
    )
    args = parser.parse_args()

    result = evaluate_model_dir_official(
        method=args.method,
        model_dir=Path(args.model_dir),
        mquake_path=Path(args.mquake_path),
        wikidata_dir=Path(args.wikidata_dir),
        out_path=Path(args.out),
        manifest_path=(
            None if args.split_manifest is None else Path(args.split_manifest)
        ),
        forget_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        dtype=args.dtype,
        device_map=args.device_map,
        batch_size=args.batch_size,
        skip_ppl=args.skip_ppl,
        include_atomic_gen=not args.skip_atomic_gen,
        mquake_url=args.mquake_url,
    )
    print(
        "MQuAKE result: "
        f"Eff={result['forget']['Eff']}, "
        f"AtomicGen={result['forget'].get('AtomicGen')}, "
        f"RetainEff={result['retain']['Eff']}, "
        f"PPL={result['forget_PPL']}"
    )


if __name__ == "__main__":
    main()
