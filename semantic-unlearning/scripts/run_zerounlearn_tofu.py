#!/usr/bin/env python3
"""Run original closed-form ZeroUnlearn on the framework's TOFU protocol.

The vendored upstream implementation has adapters for MCF, CounterFact, and
ZsRE, but not TOFU. This runner leaves the algorithm unchanged and supplies a
reviewable request bridge:

* each full TOFU question is the ZeroUnlearn subject;
* the prompt template is exactly the chat-formatted prompt scored by
  ``tofu_eval.py``;
* the clean TOFU answer is the sensitive ``target_true``;
* tokenizer EOS is the neutral ``target_new`` for forget requests;
* paired retain answers remain ordinary retention requests;
* the edited model is evaluated in memory by the same TOFU evaluator used for
  the framework's existing result table.

The fixed comparison protocol is seed 42, ``forget05``/``retain95``, all 200
forget examples, 1,000 sampled retain examples, BF16 model loading, and the
reviewed original Llama-3.2-3B-Instruct ZeroUnlearn hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset

import run_zerounlearn_official_mcf as mcf_zero
import tofu_gagd_neighborhood_confidence as tofu_protocol
import tofu_gagd_results


METHOD = "Original ZeroUnlearn"
METHOD_KEY = "original_zerounlearn"
DATASET = "TOFU"
FORGET_SPLIT = "forget05"
RETAIN_SPLIT = "retain95"
SEED = 42
FORGET_NUM = 200
RETAIN_NUM = 1000
DTYPE_NAME = "bfloat16"
MAX_LENGTH = 256
EDIT_LAYER_NUMS = 3
ADD_RETAIN = False
USE_H = False
CASE_ID_RETAIN_OFFSET = 10_000_000
PAIRED_RETAIN_SPLITS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}

SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_MODEL_PATH = SEMANTIC_ROOT / "outputs" / "finetuned_model_3B_instruct"
DEFAULT_ZERO_ROOT = REPOSITORY_ROOT / "ZeroUnlearn"
DEFAULT_HPARAMS = (
    DEFAULT_ZERO_ROOT
    / "hparams"
    / "ZeroUnlearn"
    / "Llama-3.2-3B-Instruct.json"
)
DEFAULT_WIKIDATA = SEMANTIC_ROOT / "data" / "wikidata"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "outputs" / "zerounlearn_tofu"


@dataclass(frozen=True)
class SampledTOFURow:
    split: str
    source_index: int
    sampled_position: int
    row: Dict[str, Any]

    @property
    def question(self) -> str:
        return str(self.row["question"])

    @property
    def answer(self) -> str:
        return str(self.row["answer"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--zero-unlearn-root", default=str(DEFAULT_ZERO_ROOT))
    parser.add_argument("--hparams-path", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--wikidata-dir", default=str(DEFAULT_WIKIDATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--forget-split", default=FORGET_SPLIT)
    parser.add_argument("--retain-split", default=RETAIN_SPLIT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--forget-num", type=int, default=FORGET_NUM)
    parser.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    parser.add_argument("--dtype", choices=[DTYPE_NAME], default=DTYPE_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--n-real-authors-eval", type=int, default=None)
    parser.add_argument("--n-world-facts-eval", type=int, default=None)
    parser.add_argument("--n-perturbed-eval", type=int, default=None)
    parser.add_argument(
        "--reference-truth-ratios",
        required=True,
        help=(
            "Retain-only-oracle forget truth-ratio JSON emitted while "
            "evaluating the framework Base row."
        ),
    )
    parser.add_argument(
        "--base-summary",
        default=None,
        help=(
            "Optional existing Base TOFU summary. When supplied, the runner "
            "writes a Base-versus-ZeroUnlearn comparison."
        ),
    )
    parser.add_argument(
        "--framework-eval-dir",
        default=None,
        help=(
            "Optional existing framework evaluation directory. Available "
            "method summaries are included with ZeroUnlearn in a new TOFU "
            "comparison table."
        ),
    )
    parser.add_argument(
        "--max-forget-answer-probability",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--min-retain-probability-ratio",
        type=float,
        default=0.9999998,
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally save the edited BF16 checkpoint. Evaluation always "
            "runs in memory and does not require this."
        ),
    )
    return parser


def validate_protocol_args(args: argparse.Namespace) -> None:
    required = {
        "--forget-split": (args.forget_split, FORGET_SPLIT),
        "--retain-split": (args.retain_split, RETAIN_SPLIT),
        "--seed": (args.seed, SEED),
        "--forget-num": (args.forget_num, FORGET_NUM),
        "--retain-num": (args.retain_num, RETAIN_NUM),
        "--dtype": (args.dtype, DTYPE_NAME),
        "--max-length": (args.max_length, MAX_LENGTH),
    }
    mismatches = [
        f"{name} must be {expected!r}, got {actual!r}"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "The TOFU comparison is intentionally restricted to the "
            "framework's fixed protocol:\n- " + "\n- ".join(mismatches)
        )
    if PAIRED_RETAIN_SPLITS.get(args.forget_split) != args.retain_split:
        raise ValueError(
            f"{args.forget_split} must be paired with "
            f"{PAIRED_RETAIN_SPLITS.get(args.forget_split)!r}"
        )
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    for name in (
        "n_real_authors_eval",
        "n_world_facts_eval",
        "n_perturbed_eval",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.max_forget_answer_probability < 1.0:
        raise ValueError(
            "--max-forget-answer-probability must lie in (0,1)"
        )
    if not 0.0 < args.min_retain_probability_ratio <= 1.0:
        raise ValueError(
            "--min-retain-probability-ratio must lie in (0,1]"
        )


def checkpoint_weight_files(model_path: Path) -> List[Path]:
    single = model_path / "model.safetensors"
    if single.is_file():
        return [single]
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return []
    payload = mcf_zero.read_json(index_path)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError(f"Invalid model weight index: {index_path}")
    return sorted({model_path / str(name) for name in weight_map.values()})


def require_runtime_files(
    model_path: Path,
    zero_root: Path,
    hparams_path: Path,
    wikidata_dir: Path,
    reference_truth_ratios: Path,
) -> None:
    errors: List[str] = []
    if not model_path.is_dir():
        errors.append(f"full-TOFU model directory missing: {model_path}")
    else:
        weight_files = checkpoint_weight_files(model_path)
        if not weight_files:
            errors.append(f"model weights missing from: {model_path}")
        else:
            errors.extend(
                f"model weight shard missing: {path}"
                for path in weight_files
                if not path.is_file()
            )
    if not zero_root.is_dir():
        errors.append(f"vendored ZeroUnlearn root missing: {zero_root}")
    if not hparams_path.is_file():
        errors.append(f"ZeroUnlearn hparams missing: {hparams_path}")
    if not wikidata_dir.is_dir():
        errors.append(f"ZeroUnlearn Wikidata moments corpus missing: {wikidata_dir}")
    if not reference_truth_ratios.is_file():
        errors.append(
            "retain-only reference truth-ratio JSON missing: "
            f"{reference_truth_ratios}"
        )
    if errors:
        raise FileNotFoundError(
            "Required TOFU ZeroUnlearn inputs are unavailable:\n- "
            + "\n- ".join(errors)
        )


def hash_zero_inputs(
    hparams_path: Path,
    zero_root: Path,
) -> Dict[str, str]:
    paths = [hparams_path]
    paths.extend(
        zero_root / relative
        for relative in mcf_zero.HASHED_ZERO_SOURCE_RELATIVE_PATHS
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ZeroUnlearn source files are missing:\n- "
            + "\n- ".join(str(path) for path in missing)
        )
    return {
        str(path.resolve()): mcf_zero.sha256_file(path)
        for path in paths
    }


def validate_reviewed_zero_inputs(
    hashes: Mapping[str, str],
    hparams_path: Path,
    zero_root: Path,
) -> None:
    expected = {
        str(hparams_path.resolve()): mcf_zero.EXPECTED_HPARAMS_SHA256,
        **{
            str((zero_root / relative).resolve()): digest
            for relative, digest in mcf_zero.EXPECTED_ZERO_SOURCE_SHA256.items()
        },
    }
    errors = [
        f"{path}: expected {digest}, got {hashes.get(path)}"
        for path, digest in expected.items()
        if hashes.get(path) != digest
    ]
    if errors:
        raise RuntimeError(
            "ZeroUnlearn hparams or original source differs from the reviewed "
            "MCF comparison implementation:\n- " + "\n- ".join(errors)
        )


def deterministic_sampled_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    sample_size: int,
    seed: int,
) -> List[SampledTOFURow]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if len(rows) < sample_size:
        raise ValueError(
            f"TOFU {split} contains {len(rows)} rows, need {sample_size}"
        )
    if len(rows) == sample_size:
        indices = list(range(len(rows)))
    else:
        indices = random.Random(seed).sample(range(len(rows)), sample_size)
    sampled: List[SampledTOFURow] = []
    for sampled_position, source_index in enumerate(indices):
        row = dict(rows[source_index])
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"TOFU {split} source row {source_index} has no question"
            )
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"TOFU {split} source row {source_index} has no answer"
            )
        sampled.append(
            SampledTOFURow(
                split=split,
                source_index=source_index,
                sampled_position=sampled_position,
                row=row,
            )
        )
    return sampled


def load_protocol_rows(
    *,
    forget_split: str,
    retain_split: str,
    forget_num: int,
    retain_num: int,
    seed: int,
) -> Tuple[
    List[SampledTOFURow],
    List[SampledTOFURow],
    Dict[str, Any],
]:
    forget_dataset = load_dataset(
        "locuslab/TOFU",
        name=forget_split,
        split="train",
    )
    retain_dataset = load_dataset(
        "locuslab/TOFU",
        name=retain_split,
        split="train",
    )
    forget = deterministic_sampled_rows(
        list(forget_dataset),
        split=forget_split,
        sample_size=forget_num,
        seed=seed,
    )
    retain = deterministic_sampled_rows(
        list(retain_dataset),
        split=retain_split,
        sample_size=retain_num,
        seed=seed,
    )
    metadata = {
        "repository": "locuslab/TOFU",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "forget_fingerprint": getattr(forget_dataset, "_fingerprint", None),
        "retain_fingerprint": getattr(retain_dataset, "_fingerprint", None),
        "forget_source_indices": [sample.source_index for sample in forget],
        "retain_source_indices": [sample.source_index for sample in retain],
    }
    return forget, retain, metadata


def prompt_template_for_question(tokenizer: Any, question: str) -> str:
    """Build one ``str.format`` slot that reproduces the evaluator prompt."""
    sentinel = "__ZERO_UNLEARN_TOFU_SUBJECT_7F30C4A9__"
    formatted = tofu_protocol.format_question_prompt(tokenizer, sentinel)
    if formatted.count(sentinel) != 1:
        raise RuntimeError(
            "Tokenizer chat template did not preserve the TOFU question "
            "sentinel exactly once"
        )
    escaped = formatted.replace("{", "{{").replace("}", "}}")
    template = escaped.replace(sentinel, "{}")
    expected = tofu_protocol.format_question_prompt(tokenizer, question)
    actual = template.format(question)
    if actual != expected:
        raise RuntimeError(
            "ZeroUnlearn prompt template does not reproduce the exact TOFU "
            "evaluation prompt"
        )
    return template


def sampled_rows_to_requests(
    samples: Sequence[SampledTOFURow],
    tokenizer: Any,
    *,
    neutral_target: Optional[str],
    case_id_offset: int,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for sample in samples:
        target_new = sample.answer if neutral_target is None else neutral_target
        requests.append(
            {
                "case_id": case_id_offset + sample.source_index,
                "prompt": prompt_template_for_question(
                    tokenizer,
                    sample.question,
                ),
                "subject": sample.question,
                "target_true": {"str": sample.answer},
                "target_new": {"str": target_new},
            }
        )
    return requests


def validate_requests(
    samples: Sequence[SampledTOFURow],
    requests: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    neutral_target: Optional[str],
    case_id_offset: int,
) -> None:
    if len(samples) != len(requests):
        raise RuntimeError("TOFU request conversion changed the record count")
    errors: List[str] = []
    for sample, request in zip(samples, requests):
        case_id = case_id_offset + sample.source_index
        if request.get("case_id") != case_id:
            errors.append(
                f"source index {sample.source_index}: case_id changed"
            )
        if request.get("subject") != sample.question:
            errors.append(
                f"source index {sample.source_index}: subject changed"
            )
        prompt = request.get("prompt")
        if (
            not isinstance(prompt, str)
            or prompt.format(sample.question)
            != tofu_protocol.format_question_prompt(
                tokenizer,
                sample.question,
            )
        ):
            errors.append(
                f"source index {sample.source_index}: prompt mismatch"
            )
        if request.get("target_true") != {"str": sample.answer}:
            errors.append(
                f"source index {sample.source_index}: sensitive answer changed"
            )
        expected_new = (
            sample.answer if neutral_target is None else neutral_target
        )
        if request.get("target_new") != {"str": expected_new}:
            errors.append(
                f"source index {sample.source_index}: target_new mismatch"
            )
    if errors:
        raise RuntimeError(
            "Invalid ZeroUnlearn TOFU requests:\n- " + "\n- ".join(errors)
        )


def sampled_rows_sha256(
    forget: Sequence[SampledTOFURow],
    retain: Sequence[SampledTOFURow],
) -> str:
    payload = {
        "forget": [
            {
                "source_index": sample.source_index,
                "question": sample.question,
                "answer": sample.answer,
            }
            for sample in forget
        ],
        "retain": [
            {
                "source_index": sample.source_index,
                "question": sample.question,
                "answer": sample.answer,
            }
            for sample in retain
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_reference_truth_ratios(path: Path) -> List[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "Reference truth-ratio JSON must contain a non-empty list"
        )
    values = [float(value) for value in payload]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Reference truth ratios must all be finite")
    return values


def validate_reference_truth_ratio_count(
    values: Sequence[float],
    *,
    forget_split: str,
    n_perturbed: Optional[int],
) -> int:
    dataset = load_dataset(
        "locuslab/TOFU",
        name=f"{forget_split}_perturbed",
        split="train",
    )
    expected = (
        len(dataset)
        if n_perturbed is None
        else min(len(dataset), n_perturbed)
    )
    if len(values) != expected:
        raise ValueError(
            "Retain-only reference truth-ratio count does not match the "
            f"TOFU evaluation subset: expected {expected}, got {len(values)}"
        )
    return expected


def validate_base_summary(
    summary: Mapping[str, Any],
    *,
    seed: int,
    forget_split: str,
    retain_split: str,
    forget_num: int,
    retain_num: int,
) -> None:
    expected = {
        "seed": seed,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "n_forget_eval": forget_num,
        "n_retain_eval": retain_num,
    }
    errors = [
        f"{key}: expected {value!r}, got {summary.get(key)!r}"
        for key, value in expected.items()
        if summary.get(key) != value
    ]
    for key in ("forget_answer_prob", "retain_answer_prob"):
        try:
            value = float(summary[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"{key}: missing or non-numeric")
        else:
            if not math.isfinite(value):
                errors.append(f"{key}: must be finite")
    if errors:
        raise ValueError(
            "Base TOFU summary does not match the ZeroUnlearn protocol:\n- "
            + "\n- ".join(errors)
        )


def available_framework_summaries(
    framework_eval_dir: Optional[Path],
    base_summary: Optional[Path],
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    if framework_eval_dir is not None:
        for display_name in tofu_gagd_results.METHOD_ORDER:
            key = tofu_gagd_results.METHOD_KEYS[display_name]
            if key == METHOD_KEY:
                continue
            candidate = framework_eval_dir / f"{key}_summary.json"
            if candidate.is_file():
                paths[key] = candidate
    if base_summary is not None:
        paths["base"] = base_summary
    return paths


def write_comparison(
    output_dir: Path,
    *,
    zero_summary_path: Path,
    base_summary_path: Path,
    framework_eval_dir: Optional[Path],
    max_forget_answer_probability: float,
    min_retain_probability_ratio: float,
) -> None:
    result_paths = available_framework_summaries(
        framework_eval_dir,
        base_summary_path,
    )
    result_paths[METHOD_KEY] = zero_summary_path
    rows: List[Dict[str, Any]] = []
    for display_name in tofu_gagd_results.METHOD_ORDER:
        key = tofu_gagd_results.METHOD_KEYS[display_name]
        path = result_paths.get(key)
        if path is None:
            continue
        summary = mcf_zero.read_json(path)
        rows.append(
            tofu_gagd_results.row_from_summary(
                display_name,
                path.resolve(),
                summary,
                max_forget_answer_probability,
            )
        )
    protocol = tofu_gagd_results.verify_protocol(rows)
    tofu_gagd_results.add_base_differences(
        rows,
        min_retain_probability_ratio,
    )
    tofu_gagd_results.write_outputs(
        output_dir,
        rows,
        protocol,
        max_forget_answer_probability,
        min_retain_probability_ratio,
    )


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_protocol_args(args)
    model_path = Path(args.model_path).expanduser().resolve()
    zero_root = Path(args.zero_unlearn_root).expanduser().resolve()
    hparams_path = Path(args.hparams_path).expanduser().resolve()
    wikidata_dir = Path(args.wikidata_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reference_path = Path(args.reference_truth_ratios).expanduser().resolve()
    base_summary_path = (
        Path(args.base_summary).expanduser().resolve()
        if args.base_summary
        else None
    )
    framework_eval_dir = (
        Path(args.framework_eval_dir).expanduser().resolve()
        if args.framework_eval_dir
        else None
    )
    require_runtime_files(
        model_path,
        zero_root,
        hparams_path,
        wikidata_dir,
        reference_path,
    )
    if base_summary_path is not None and not base_summary_path.is_file():
        raise FileNotFoundError(f"Base summary missing: {base_summary_path}")
    if framework_eval_dir is not None and not framework_eval_dir.is_dir():
        raise FileNotFoundError(
            f"Framework evaluation directory missing: {framework_eval_dir}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "One CUDA GPU is required for original ZeroUnlearn; CUDA is "
            "unavailable"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    source_hashes_before = hash_zero_inputs(hparams_path, zero_root)
    validate_reviewed_zero_inputs(
        source_hashes_before,
        hparams_path,
        zero_root,
    )
    reference_values = load_reference_truth_ratios(reference_path)
    reference_count = validate_reference_truth_ratio_count(
        reference_values,
        forget_split=args.forget_split,
        n_perturbed=args.n_perturbed_eval,
    )
    forget_samples, retain_samples, dataset_metadata = load_protocol_rows(
        forget_split=args.forget_split,
        retain_split=args.retain_split,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_all_seeds(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import tofu_eval

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    neutral_target, neutral_target_id = mcf_zero.resolve_eos_neutral_target(
        tokenizer
    )
    forget_requests = sampled_rows_to_requests(
        forget_samples,
        tokenizer,
        neutral_target=neutral_target,
        case_id_offset=0,
    )
    retain_requests = sampled_rows_to_requests(
        retain_samples,
        tokenizer,
        neutral_target=None,
        case_id_offset=CASE_ID_RETAIN_OFFSET,
    )
    validate_requests(
        forget_samples,
        forget_requests,
        tokenizer,
        neutral_target=neutral_target,
        case_id_offset=0,
    )
    validate_requests(
        retain_samples,
        retain_requests,
        tokenizer,
        neutral_target=None,
        case_id_offset=CASE_ID_RETAIN_OFFSET,
    )

    params_class, apply_unl_to_model = mcf_zero.import_original_zerounlearn(
        zero_root
    )
    hparams = params_class.from_json(hparams_path)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(
            "Original hparams changed unexpectedly: expected layers "
            f"[16, 17, 18], got {hparams.layers}"
        )

    print(f"Loading full-TOFU BF16 checkpoint: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    zero_summary_path = evaluation_dir / f"{METHOD_KEY}_summary.json"
    provenance_path = output_dir / "zerounlearn_tofu_provenance.json"
    checkpoint_dir = output_dir / "checkpoint"
    run_status = "running"
    runtime: Dict[str, Any] = {}
    provenance: Dict[str, Any] = {
        "status": run_status,
        "method": METHOD,
        "dataset": DATASET,
        "algorithm_entrypoint": (
            "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model"
        ),
        "upstream_repository": mcf_zero.UPSTREAM_REPOSITORY,
        "vendored_source_revision": mcf_zero.git_source_revision(
            REPOSITORY_ROOT,
            zero_root / "ZeroUnlearn" / "ZeroUnlearn_main.py",
        ),
        "model_path": str(model_path),
        "model_metadata": (
            mcf_zero.read_json(model_path / "finetune_metadata.json")
            if (model_path / "finetune_metadata.json").is_file()
            else None
        ),
        "dtype": args.dtype,
        "zero_unlearn_compute_dtype": "float32",
        "seed": args.seed,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "dataset_metadata": dataset_metadata,
        "sampled_rows_sha256": sampled_rows_sha256(
            forget_samples,
            retain_samples,
        ),
        "request_adapter": {
            "subject_source": "TOFU question",
            "prompt_source": (
                "tofu_eval-compatible chat-formatted Question/Answer prompt"
            ),
            "sensitive_target_source": "TOFU clean answer",
            "sensitive_request_field": "target_true.str",
            "neutral_request_field": "target_new.str",
            "neutral_target_source": "tokenizer.eos_token",
            "neutral_target": neutral_target,
            "neutral_target_id": neutral_target_id,
            "retain_target_true_source": "TOFU clean answer",
            "retain_target_new_source": "TOFU clean answer",
            "evaluation_records_modified": False,
        },
        "hparams_path": str(hparams_path),
        "hparams": mcf_zero.read_json(hparams_path),
        "edit_layer_nums": EDIT_LAYER_NUMS,
        "add_retain": ADD_RETAIN,
        "use_h": USE_H,
        "reference_truth_ratios": str(reference_path),
        "reference_truth_ratios_sha256": mcf_zero.sha256_file(reference_path),
        "reference_truth_ratio_count": reference_count,
        "official_evaluation_path": str(zero_summary_path),
        "checkpoint_saved": bool(args.save_model),
        "checkpoint_path": str(checkpoint_dir) if args.save_model else None,
        "cuda_device_index": device.index,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "source_hashes_before": source_hashes_before,
        "exact_command": [
            sys.executable,
            str(SCRIPT_PATH),
            *sys.argv[1:],
        ],
    }
    mcf_zero.write_json(provenance_path, provenance)

    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        apply_started = time.perf_counter()
        print(
            "Upcasting the exact BF16 full-TOFU starting weights to FP32 for "
            "the original closed-form matrix solve"
        )
        model.float()
        print(
            f"Applying original ZeroUnlearn to {len(forget_requests)} TOFU "
            f"forget requests with {len(retain_requests)} paired retain requests"
        )
        with mcf_zero.working_directory(SEMANTIC_ROOT):
            edited_model, original_weights = apply_unl_to_model(
                model=model,
                tok=tokenizer,
                retain_requests=deepcopy(retain_requests),
                unlearn_requests=deepcopy(forget_requests),
                hparams=hparams,
                copy=False,
                return_orig_weights=False,
                cache_template=None,
                save_path=None,
                add_retain=ADD_RETAIN,
                edit_layer_nums=EDIT_LAYER_NUMS,
                use_h=USE_H,
            )
        edited_model.to(dtype=torch.bfloat16)
        edited_model.eval()
        torch.cuda.synchronize(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        runtime = {
            "apply_seconds": time.perf_counter() - apply_started,
            "peak_cuda_memory_allocated_bytes": int(peak_allocated),
            "peak_cuda_memory_allocated_gib": peak_allocated / (1024**3),
            "peak_cuda_memory_reserved_bytes": int(peak_reserved),
            "peak_cuda_memory_reserved_gib": peak_reserved / (1024**3),
            "peak_additional_cuda_memory_bytes": int(
                max(0, peak_allocated - memory_before)
            ),
        }
        del original_weights
        gc.collect()
        torch.cuda.empty_cache()

        evaluation_started = time.perf_counter()
        evaluator = tofu_eval.Evaluator.from_loaded(
            edited_model,
            tokenizer,
            device="cuda:0",
            max_length=args.max_length,
        )
        results, details, artifacts = tofu_eval.evaluate_with_evaluator(
            evaluator,
            method=METHOD_KEY,
            model_dir=f"in-memory:{METHOD}",
            forget_split=args.forget_split,
            retain_split=args.retain_split,
            seed=args.seed,
            n_forget=args.forget_num,
            n_retain=args.retain_num,
            n_real_authors=args.n_real_authors_eval,
            n_world_facts=args.n_world_facts_eval,
            n_perturbed=args.n_perturbed_eval,
            max_new_tokens=args.max_new_tokens,
            base_model=False,
            reference_values=reference_values,
        )
        runtime["official_evaluation_seconds"] = (
            time.perf_counter() - evaluation_started
        )
        runtime["total_seconds"] = (
            runtime["apply_seconds"]
            + runtime["official_evaluation_seconds"]
        )
        zero_metadata = {
            "display_name": METHOD,
            "dataset": DATASET,
            "model_path": str(model_path),
            "zero_unlearn_runtime": runtime,
            "sampled_rows_sha256": provenance["sampled_rows_sha256"],
            "request_adapter": provenance["request_adapter"],
        }
        results.update(zero_metadata)
        details["zero_unlearn"] = zero_metadata
        paths = tofu_eval.write_evaluation_outputs(
            evaluation_dir,
            method=METHOD_KEY,
            forget_split=args.forget_split,
            retain_split=args.retain_split,
            results=results,
            details=details,
            artifacts=artifacts,
        )
        if args.save_model:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            edited_model.save_pretrained(
                checkpoint_dir,
                safe_serialization=True,
            )
            tokenizer.save_pretrained(checkpoint_dir)
        run_status = "completed"
    except Exception:
        run_status = "failed"
        raise
    finally:
        source_hashes_after = hash_zero_inputs(hparams_path, zero_root)
        unchanged = source_hashes_before == source_hashes_after
        provenance.update(
            {
                "status": run_status,
                "runtime": runtime or None,
                "source_hashes_after": source_hashes_after,
                "source_hashes_unchanged": unchanged,
            }
        )
        mcf_zero.write_json(provenance_path, provenance)
        if not unchanged:
            raise RuntimeError(
                "ZeroUnlearn hparams or source changed during execution"
            )

    if base_summary_path is not None:
        base_summary = mcf_zero.read_json(base_summary_path)
        validate_base_summary(
            base_summary,
            seed=args.seed,
            forget_split=args.forget_split,
            retain_split=args.retain_split,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
        )
        write_comparison(
            output_dir / "comparison",
            zero_summary_path=paths["summary"],
            base_summary_path=base_summary_path,
            framework_eval_dir=framework_eval_dir,
            max_forget_answer_probability=(
                args.max_forget_answer_probability
            ),
            min_retain_probability_ratio=(
                args.min_retain_probability_ratio
            ),
        )

    tofu_eval.print_result_summary(results, paths)
    print(
        f"ZeroUnlearn runtime: {runtime['apply_seconds']:.3f}s apply, "
        f"{runtime['peak_cuda_memory_allocated_gib']:.3f} GiB peak CUDA"
    )
    del evaluator
    del edited_model
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "result": results,
        "paths": paths,
        "provenance_path": provenance_path,
        "runtime": runtime,
    }


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
