#!/usr/bin/env python3
"""Run original closed-form ZeroUnlearn and compare it with seed-0 results.

This runner deliberately separates the method from the evaluator:

* the edit is performed by ``ZeroUnlearn.apply_unl_to_model`` from the
  vendored original ZeroUnlearn implementation;
* forget requests use the tokenizer EOS token as ZeroUnlearn's neutral
  ``target_new`` state, while the source MCF records remain unchanged;
* the split comes from the shared official MCF sampler;
* base and edited models are measured by
  ``mcf_zero_unlearn_official_eval.evaluate_loaded_model_official``.

Only seed 0, 50 forget records, 1000 retain records, BF16, official sampling,
and the original Llama-3.2-3B-Instruct ZeroUnlearn hyperparameters are allowed.
The script does not save a model checkpoint and never invokes ZeroUnlearn-GD.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from mcf_sampling import sample_official_mcf_records
from mcf_zero_unlearn_official_eval import evaluate_loaded_model_official


METHOD = "ZeroUnlearn"
UPSTREAM_REPOSITORY = "https://github.com/XMUDeepLIT/ZeroUnlearn"
MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
DEFAULT_MODEL_PATH = (
    "/scratch/yl258/kp759/"
    "hf/models--meta-llama--Llama-3.2-3B-Instruct/"
    f"snapshots/{MODEL_REVISION}"
)

SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_ZERO_ROOT = REPOSITORY_ROOT / "ZeroUnlearn"
DEFAULT_HPARAMS = (
    DEFAULT_ZERO_ROOT / "hparams" / "ZeroUnlearn" / "Llama-3.2-3B-Instruct.json"
)
DEFAULT_MCF = SEMANTIC_ROOT / "data" / "multi_counterfact.json"
DEFAULT_WIKIDATA = SEMANTIC_ROOT / "data" / "wikidata"
DEFAULT_OUTPUT = SEMANTIC_ROOT / "outputs" / "zerounlearn_vs_seed0"

SEED = 0
FORGET_NUM = 50
RETAIN_NUM = 1000
SAMPLE_MODE = "official"
DTYPE_NAME = "bfloat16"
EDIT_LAYER_NUMS = 3
ADD_RETAIN = False
USE_H = False


@dataclass(frozen=True)
class MethodSpec:
    key: str
    display_name: str
    expected_eff: float
    expected_gen: float
    expected_spe: float
    expected_ppl: float
    preferred_paths: Tuple[str, ...]
    path_hints: Tuple[str, ...]

    @property
    def expected_metrics(self) -> Dict[str, float]:
        return {
            "Eff": self.expected_eff,
            "Gen": self.expected_gen,
            "Spe": self.expected_spe,
            "PPL": self.expected_ppl,
        }


EXISTING_METHOD_SPECS: Tuple[MethodSpec, ...] = (
    MethodSpec(
        "base",
        "Base",
        6.0,
        6.0,
        10.89,
        11.0625,
        ("outputs/official_eval_base_seed0_spefix.json",),
        ("base", "seed0", "spefix"),
    ),
    MethodSpec(
        "json_lmhead_restore",
        "JSON-LMHead Restore",
        0.0,
        0.0,
        27.67,
        11.0625,
        ("outputs/official_eval_lmhead_zero_true_restore150_seed0_spefix.json",),
        ("lmhead", "true_restore150", "seed0", "spefix"),
    ),
    MethodSpec(
        "emb_lm_all_tokens",
        "Emb/LM all tokens",
        0.0,
        6.0,
        33.78,
        31.125,
        (
            "outputs/gagd_vs_json_lmhead/seed0/official_eval/"
            "emb_lm_all_tokens_official_eval.json",
            "outputs/gagd_compare/mcf/official_eval/"
            "emb_lm_all_tokens_official_eval.json",
        ),
        ("emb_lm_all_tokens", "official_eval"),
    ),
    MethodSpec(
        "emb_lm_selective_tokens",
        "Emb/LM selective",
        2.0,
        5.0,
        22.64,
        21.0,
        (
            "outputs/gagd_vs_json_lmhead/seed0/official_eval/"
            "emb_lm_selective_tokens_official_eval.json",
            "outputs/gagd_compare/mcf/official_eval/"
            "emb_lm_selective_tokens_official_eval.json",
        ),
        ("emb_lm_selective_tokens", "official_eval"),
    ),
    MethodSpec(
        "full_all_tokens",
        "Full all tokens",
        6.0,
        8.0,
        13.10,
        11.4375,
        (
            "outputs/gagd_vs_json_lmhead/seed0/official_eval/"
            "full_all_tokens_official_eval.json",
            "outputs/gagd_compare/mcf/official_eval/"
            "full_all_tokens_official_eval.json",
        ),
        ("full_all_tokens", "official_eval"),
    ),
    MethodSpec(
        "full_selective_tokens",
        "Full selective",
        6.0,
        6.0,
        13.28,
        10.9375,
        (
            "outputs/gagd_vs_json_lmhead/seed0/official_eval/"
            "full_selective_tokens_official_eval.json",
            "outputs/gagd_compare/mcf/official_eval/"
            "full_selective_tokens_official_eval.json",
        ),
        ("full_selective_tokens", "official_eval"),
    ),
    MethodSpec(
        "setting_5e",
        "Setting 5e",
        2.0,
        2.0,
        13.32,
        11.0625,
        (
            "outputs/gagd_vs_json_lmhead/seed0/official_eval/"
            "emb_lm_all_restore_post_training_true_official_eval.json",
            "outputs/gagd_compare/mcf/official_eval/"
            "emb_lm_all_restore_post_training_true_official_eval.json",
        ),
        (
            "emb_lm_all_restore_post_training_true",
            "official_eval",
        ),
    ),
    MethodSpec(
        "protected_lm_head_repair",
        "Setting 5e + protected LM-head repair",
        0.0,
        0.0,
        13.74,
        11.0625,
        (),
        ("active_case_repair", "protected", "official_eval"),
    ),
    MethodSpec(
        "neighborhood_confidence_repair",
        "Neighborhood-confidence repair",
        0.0,
        0.0,
        20.01,
        11.0625,
        (),
        ("neighborhood", "confidence", "repair", "official_eval"),
    ),
)

DISPLAY_ORDER = (
    "Base",
    "Original ZeroUnlearn",
    "JSON-LMHead Restore",
    "Emb/LM all tokens",
    "Emb/LM selective",
    "Full all tokens",
    "Full selective",
    "Setting 5e",
    "Setting 5e + protected LM-head repair",
    "Neighborhood-confidence repair",
)

PAIRWISE_TARGETS = (
    "JSON-LMHead Restore",
    "Setting 5e",
    "Setting 5e + protected LM-head repair",
    "Neighborhood-confidence repair",
)

HASHED_ZERO_SOURCE_RELATIVE_PATHS = (
    "ZeroUnlearn/ZeroUnlearn_main.py",
    "ZeroUnlearn/ZeroUnlearn_hparams.py",
    "ZeroUnlearn/compute_z.py",
    "ZeroUnlearn/compute_ks.py",
    "ZeroUnlearn/compute_vs.py",
    "ZeroUnlearn/compute_kvs.py",
    "rome/layer_stats_retain.py",
    "rome/tok_dataset.py",
    "util/runningstats.py",
)

EXPECTED_MCF_SHA256 = "977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"
EXPECTED_HPARAMS_SHA256 = (
    "2e2640f29f24112b0838db1cf03dc6773762fed203ad22a6588891c52f8dc5b3"
)
EXPECTED_ZERO_SOURCE_SHA256 = {
    "ZeroUnlearn/ZeroUnlearn_main.py": (
        "961d98651b02b563835566829ca9957b37e699a11e9954d28b552123d957b713"
    ),
    "ZeroUnlearn/ZeroUnlearn_hparams.py": (
        "61787e298d10478c7349abae8000631d2f174e5783713c27e4c852d68ead86f2"
    ),
    "ZeroUnlearn/compute_z.py": (
        "1905939df80a0b62b4b46a9ad2f3bc3afa08b69f5e1ae104c1b0c549d49f4f86"
    ),
    "ZeroUnlearn/compute_ks.py": (
        "f634c32c5e91dcbbb1d9d774fc27bf795f1ca095b5df3851bafe5b578e4e685c"
    ),
    "ZeroUnlearn/compute_vs.py": (
        "54d8ff83e20ca6eec5dd99121ecd9b2fd57a1f08b0656e9a09f0e42da2d71261"
    ),
    "ZeroUnlearn/compute_kvs.py": (
        "6675c2920cc5c427c6176480ecb64d53ecfafb80973d98a5754c520d7d02c1e9"
    ),
    "rome/layer_stats_retain.py": (
        "c81ce70c35994a1732bee34cb4ecaacf87175f73d8beccfc3a1e03fde50c2eb4"
    ),
    "rome/tok_dataset.py": (
        "3f12d20f6d50b2dd09fa0b9fc3f465c458fc8795fe6b5b6986ad3e3f58a63a52"
    ),
    "util/runningstats.py": (
        "d6afca05b88b71e095223f2b9c7ec312e3b7acfcfbd8aa78384095cc1f7dd4e3"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--zero-unlearn-root", default=str(DEFAULT_ZERO_ROOT))
    parser.add_argument("--hparams-path", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--mcf-path", default=str(DEFAULT_MCF))
    parser.add_argument("--wikidata-dir", default=str(DEFAULT_WIKIDATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--forget-num", type=int, default=FORGET_NUM)
    parser.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    parser.add_argument("--sample-mode", choices=["official"], default=SAMPLE_MODE)
    parser.add_argument("--dtype", choices=["bfloat16"], default=DTYPE_NAME)
    parser.add_argument(
        "--existing-result",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help=(
            "Override auto-discovery for an existing result. Valid keys: "
            + ", ".join(spec.key for spec in EXISTING_METHOD_SPECS)
        ),
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=0.02,
        help="Tolerance for rounded Eff/Gen/Spe reference validation.",
    )
    parser.add_argument(
        "--ppl-tolerance",
        type=float,
        default=0.01,
        help="Tolerance for PPL reference and base validation.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help=(
            "Validate existing seed-0 results and write the manifest without "
            "loading a model or running ZeroUnlearn."
        ),
    )
    return parser


def validate_protocol_args(args: argparse.Namespace) -> None:
    required = {
        "--seed": (args.seed, SEED),
        "--forget-num": (args.forget_num, FORGET_NUM),
        "--retain-num": (args.retain_num, RETAIN_NUM),
        "--sample-mode": (args.sample_mode, SAMPLE_MODE),
        "--dtype": (args.dtype, DTYPE_NAME),
    }
    mismatches = [
        f"{name} must be {expected!r}, got {actual!r}"
        for name, (actual, expected) in required.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "This runner is intentionally restricted to the fair seed-0 "
            "protocol:\n- " + "\n- ".join(mismatches)
        )
    if args.metric_tolerance < 0 or args.ppl_tolerance < 0:
        raise ValueError("Metric tolerances must be non-negative")
    revision = Path(args.model_path).name
    if revision != MODEL_REVISION:
        raise ValueError(
            "The comparison requires the exact model snapshot revision "
            f"{MODEL_REVISION}, got path ending in {revision!r}"
        )


def parse_result_overrides(
    values: Sequence[str],
    semantic_root: Path,
) -> Dict[str, Path]:
    valid = {spec.key for spec in EXISTING_METHOD_SPECS}
    parsed: Dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or not key or not raw_path:
            raise ValueError(f"Invalid --existing-result {value!r}; expected KEY=PATH")
        if key not in valid:
            raise ValueError(
                f"Unknown existing-result key {key!r}; valid keys: {sorted(valid)}"
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            alternatives = (
                Path.cwd() / path,
                REPOSITORY_ROOT / path,
                semantic_root / path,
            )
            path = next(
                (candidate for candidate in alternatives if candidate.exists()),
                semantic_root / path,
            )
        parsed[key] = path.resolve()
    return parsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_protocol_inputs(
    mcf_path: Path,
    hparams_path: Path,
    zero_root: Path,
) -> Dict[str, str]:
    paths = [mcf_path, hparams_path]
    paths.extend(zero_root / relative for relative in HASHED_ZERO_SOURCE_RELATIVE_PATHS)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required protocol/source files are missing:\n- "
            + "\n- ".join(str(path) for path in missing)
        )
    return {str(path.resolve()): sha256_file(path) for path in paths}


def validate_expected_protocol_hashes(
    hashes: Mapping[str, str],
    mcf_path: Path,
    hparams_path: Path,
    zero_root: Path,
) -> None:
    expected = {
        str(mcf_path.resolve()): EXPECTED_MCF_SHA256,
        str(hparams_path.resolve()): EXPECTED_HPARAMS_SHA256,
        **{
            str((zero_root / relative).resolve()): digest
            for relative, digest in EXPECTED_ZERO_SOURCE_SHA256.items()
        },
    }
    errors = [
        f"{path}: expected {digest}, got {hashes.get(path)}"
        for path, digest in expected.items()
        if hashes.get(path) != digest
    ]
    if errors:
        raise RuntimeError(
            "MCF, ZeroUnlearn hparams, or original source differs from the "
            "reviewed comparison inputs:\n- " + "\n- ".join(errors)
        )


def case_ids(records: Sequence[Mapping[str, Any]]) -> List[int]:
    ids: List[int] = []
    for position, record in enumerate(records):
        if "case_id" not in record:
            raise ValueError(
                f"MCF record at sampled position {position} has no case_id"
            )
        ids.append(int(record["case_id"]))
    return ids


def normalize_rewrite(record: Mapping[str, Any]) -> Dict[str, Any]:
    requested = record.get("requested_rewrite")
    if isinstance(requested, list):
        if len(requested) != 1 or not isinstance(requested[0], dict):
            raise ValueError(
                f"case_id={record.get('case_id')} has unsupported requested_rewrite"
            )
        requested = requested[0]
    if not isinstance(requested, dict):
        raise ValueError(
            f"case_id={record.get('case_id')} has no requested_rewrite mapping"
        )
    return dict(requested)


def records_to_zero_unlearn_requests(
    records: Sequence[Mapping[str, Any]],
    *,
    neutral_target: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert MCF records to independent ZeroUnlearn request dictionaries.

    ZeroUnlearn's closed-form forget objective uses ``target_new`` as the
    neutral state M_n. For forget requests, callers pass the tokenizer EOS
    token through ``neutral_target``. Retain requests omit it and preserve the
    original MCF target. Deep copies ensure neither conversion can mutate the
    records later consumed by the official evaluator.
    """
    if neutral_target is not None and not neutral_target:
        raise ValueError("neutral_target must be a non-empty token string")

    requests: List[Dict[str, Any]] = []
    for record in records:
        request = {
            "case_id": int(record["case_id"]),
            **deepcopy(normalize_rewrite(record)),
        }
        if neutral_target is not None:
            target_new = request.get("target_new")
            if not isinstance(target_new, dict):
                raise ValueError(
                    f"case_id={record.get('case_id')} has no target_new mapping"
                )
            request["target_new"] = {"str": neutral_target}
        requests.append(request)
    return requests


def resolve_eos_neutral_target(tokenizer: Any) -> Tuple[str, int]:
    """Resolve and validate the single-token EOS neutral state."""
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RuntimeError("Tokenizer has no usable eos_token for ZeroUnlearn")
    if eos_token_id is None:
        raise RuntimeError("Tokenizer has no eos_token_id for ZeroUnlearn")

    encoded = tokenizer(eos_token, add_special_tokens=False)
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if not isinstance(input_ids, list) or input_ids != [int(eos_token_id)]:
        raise RuntimeError(
            "Tokenizer EOS string must encode to exactly eos_token_id without "
            f"special-token insertion; token={eos_token!r}, "
            f"eos_token_id={eos_token_id!r}, input_ids={input_ids!r}"
        )
    return eos_token, int(eos_token_id)


def validate_neutral_forget_requests(
    source_records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    neutral_target: str,
) -> None:
    """Guard against training on MCF counterfactual targets by accident."""
    if len(source_records) != len(requests):
        raise RuntimeError(
            "Neutral forget-request conversion changed the request count"
        )

    errors: List[str] = []
    for position, (record, request) in enumerate(zip(source_records, requests)):
        source = normalize_rewrite(record)
        case_id = int(record["case_id"])
        if int(request.get("case_id", -1)) != case_id:
            errors.append(
                f"position {position}: case_id changed from {case_id} to "
                f"{request.get('case_id')!r}"
            )
        request_target_new = request.get("target_new")
        if (
            not isinstance(request_target_new, Mapping)
            or request_target_new.get("str") != neutral_target
        ):
            errors.append(
                f"case_id={case_id}: target_new is not the EOS neutral target"
            )
        if request.get("target_true") != source.get("target_true"):
            errors.append(f"case_id={case_id}: target_true was modified")
        for key in ("prompt", "subject"):
            if request.get(key) != source.get(key):
                errors.append(f"case_id={case_id}: {key} was modified")

    if errors:
        raise RuntimeError(
            "Invalid ZeroUnlearn neutral forget requests:\n- " + "\n- ".join(errors)
        )


def load_seed0_split(
    mcf_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    with mcf_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("MCF JSON must contain a list")
    forget, retain = sample_official_mcf_records(
        data,
        FORGET_NUM,
        RETAIN_NUM,
        SEED,
        strict=True,
    )
    if len(forget) != FORGET_NUM or len(retain) != RETAIN_NUM:
        raise RuntimeError("Official MCF sampler returned the wrong split sizes")
    return data, forget, retain


def _number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Result metric {name} is missing or non-finite: {value!r}")
    return float(value)


def extract_result_metrics(payload: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    if isinstance(payload.get("forget"), dict):
        summary = payload["forget"]
        ppl = payload.get("forget_PPL", payload.get("PPL"))
    elif all(key in payload for key in ("Eff", "Gen", "Spe")):
        summary = payload
        ppl = payload.get("PPL", payload.get("forget_PPL"))
    elif isinstance(payload.get("final_official"), dict):
        final = payload["final_official"]
        summary = final.get("metrics", final)
        ppl = final.get("PPL", summary.get("PPL"))
    else:
        raise ValueError("JSON does not contain official Eff/Gen/Spe metrics")
    spe_success = summary.get("Spe_success")
    nested_summary = payload.get("summary")
    if spe_success is None and isinstance(nested_summary, dict):
        spe_success = nested_summary.get("Spe_success")
    return {
        "Eff": _number(summary.get("Eff"), "Eff"),
        "Gen": _number(summary.get("Gen"), "Gen"),
        "Spe": _number(summary.get("Spe"), "Spe"),
        "Spe_success": (
            _number(spe_success, "Spe_success") if spe_success is not None else None
        ),
        "PPL": _number(ppl, "PPL"),
    }


def validate_reference_metrics(
    spec: MethodSpec,
    metrics: Mapping[str, Optional[float]],
    metric_tolerance: float,
    ppl_tolerance: float,
) -> None:
    errors: List[str] = []
    for name, expected in spec.expected_metrics.items():
        actual = metrics[name]
        tolerance = ppl_tolerance if name == "PPL" else metric_tolerance
        if actual is None or abs(float(actual) - expected) > tolerance:
            errors.append(
                f"{name}: expected {expected}, got {actual} " f"(tolerance {tolerance})"
            )
    if errors:
        raise ValueError(
            f"{spec.display_name} does not match the seed-0 reference table:\n- "
            + "\n- ".join(errors)
        )


def protocol_metadata(payload: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": payload.get("seed"),
        "dataset": payload.get("dataset"),
        "sample_mode": payload.get(
            "sample_mode",
            payload.get("mcf_sample_mode", payload.get("split_mode")),
        ),
        "forget_num": payload.get(
            "unlearn_num",
            payload.get("forget_num", payload.get("forget_n")),
        ),
        "retain_num": payload.get("retain_num", payload.get("retain_n")),
    }


def validate_result_protocol(
    display_name: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = protocol_metadata(payload)
    aliases = {
        "dataset": {"MCF", "mcf"},
        "sample_mode": {
            SAMPLE_MODE,
            "official_zero_unlearn",
        },
    }
    errors: List[str] = []
    if metadata["seed"] is None or int(metadata["seed"]) != SEED:
        errors.append(f"seed must be 0, got {metadata['seed']!r}")
    if metadata["dataset"] not in aliases["dataset"]:
        errors.append(f"dataset must be MCF, got {metadata['dataset']!r}")
    if metadata["sample_mode"] not in aliases["sample_mode"]:
        errors.append(f"sample mode must be official, got {metadata['sample_mode']!r}")
    if metadata["forget_num"] is None or int(metadata["forget_num"]) != FORGET_NUM:
        errors.append(
            f"forget count must be {FORGET_NUM}, got {metadata['forget_num']!r}"
        )
    if metadata["retain_num"] is None or int(metadata["retain_num"]) != RETAIN_NUM:
        errors.append(
            f"retain count must be {RETAIN_NUM}, got {metadata['retain_num']!r}"
        )
    if errors:
        raise ValueError(
            f"{display_name} has incompatible protocol metadata:\n- "
            + "\n- ".join(errors)
        )
    return {
        "seed": SEED,
        "dataset": "MCF",
        "sample_mode": SAMPLE_MODE,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
    }


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def metrics_match_spec(
    path: Path,
    spec: MethodSpec,
    metric_tolerance: float,
    ppl_tolerance: float,
) -> bool:
    try:
        metrics = extract_result_metrics(read_json(path))
        validate_reference_metrics(
            spec,
            metrics,
            metric_tolerance,
            ppl_tolerance,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def official_result_json_candidates(outputs_root: Path) -> Iterable[Path]:
    seen: set[Path] = set()
    patterns = (
        "official_eval*.json",
        "*_official_eval.json",
        "official_eval.json",
    )
    for pattern in patterns:
        for path in outputs_root.rglob(pattern):
            resolved = path.resolve()
            if resolved not in seen and path.is_file():
                seen.add(resolved)
                yield resolved


def discovery_score(path: Path, payload: Mapping[str, Any], spec: MethodSpec) -> int:
    searchable = f"{path} {payload.get('method', '')}".lower()
    return sum(hint.lower() in searchable for hint in spec.path_hints)


def locate_existing_result(
    spec: MethodSpec,
    semantic_root: Path,
    overrides: Mapping[str, Path],
    metric_tolerance: float,
    ppl_tolerance: float,
) -> Path:
    if spec.key in overrides:
        path = overrides[spec.key]
        if not path.is_file():
            raise FileNotFoundError(
                f"Explicit result for {spec.display_name} does not exist: {path}"
            )
        payload = read_json(path)
        validate_reference_metrics(
            spec,
            extract_result_metrics(payload),
            metric_tolerance,
            ppl_tolerance,
        )
        return path

    for relative in spec.preferred_paths:
        path = semantic_root / relative
        if path.is_file():
            payload = read_json(path)
            validate_reference_metrics(
                spec,
                extract_result_metrics(payload),
                metric_tolerance,
                ppl_tolerance,
            )
            return path.resolve()

    outputs_root = semantic_root / "outputs"
    matches = [
        path
        for path in official_result_json_candidates(outputs_root)
        if metrics_match_spec(
            path,
            spec,
            metric_tolerance,
            ppl_tolerance,
        )
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not locate the exact seed-0 result JSON for "
            f"{spec.display_name}. Pass --existing-result {spec.key}=PATH."
        )
    scored = [(discovery_score(path, read_json(path), spec), path) for path in matches]
    best_score = max(score for score, _ in scored)
    best = sorted(path for score, path in scored if score == best_score)
    if best_score == 0 or len(best) != 1:
        raise RuntimeError(
            f"Result discovery for {spec.display_name} is ambiguous. Candidates:\n- "
            + "\n- ".join(str(path) for path in best)
            + f"\nPass --existing-result {spec.key}=PATH."
        )
    return best[0]


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)


def related_provenance_jsons(
    result_path: Path,
    payload: Mapping[str, Any],
) -> List[Path]:
    names = {
        "config_used.json",
        "repair_summary.json",
        "repair_experiment_config.json",
        "mcf_json_lmhead_ablation_summary.json",
        "config.json",
    }
    candidates: set[Path] = {result_path}
    current = result_path.parent
    for _ in range(4):
        if current.parent == current:
            break
        for name in names:
            path = current / name
            if path.is_file():
                candidates.add(path.resolve())
        candidates.update(
            path.resolve()
            for path in current.glob("*summary.json")
            if path.is_file() and path.stat().st_size <= 20 * 1024 * 1024
        )
        current = current.parent

    model_dir = payload.get("model_dir")
    if isinstance(model_dir, str) and not model_dir.startswith("in-memory:"):
        model_path = Path(model_dir)
        if not model_path.is_absolute():
            model_path = SEMANTIC_ROOT / model_path
        if model_path.is_dir():
            for name in names:
                path = model_path / name
                if path.is_file():
                    candidates.add(path.resolve())
    return sorted(candidates)


def find_model_evidence(
    result_path: Path,
    payload: Mapping[str, Any],
    model_path: Path,
) -> Tuple[bool, List[str]]:
    exact = str(model_path.resolve())
    revision = model_path.name
    evidence: List[str] = []
    for path in related_provenance_jsons(result_path, payload):
        try:
            provenance = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        strings = list(_walk_strings(provenance))
        if exact in strings or any(
            exact in value or revision in value for value in strings
        ):
            evidence.append(str(path))
    return bool(evidence), evidence


def extract_stored_case_ids(
    payload: Mapping[str, Any],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    forget = payload.get("forget_case_ids")
    retain = payload.get("retain_case_ids")
    if isinstance(forget, list):
        forget = [int(value) for value in forget]
    else:
        forget = None
    if isinstance(retain, list):
        retain = [int(value) for value in retain]
    else:
        retain = None
    return forget, retain


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def checkpoint_path_from_result(
    result_path: Path,
    payload: Mapping[str, Any],
) -> Optional[str]:
    model_dir = payload.get("model_dir")
    if isinstance(model_dir, str) and not model_dir.startswith("in-memory:"):
        path = Path(model_dir)
        if not path.is_absolute():
            path = SEMANTIC_ROOT / path
        if path.is_dir() and path.resolve() != Path(DEFAULT_MODEL_PATH).resolve():
            return display_path(path)
    current = result_path.parent
    for _ in range(4):
        checkpoint = current / "checkpoint"
        if checkpoint.is_dir():
            return display_path(checkpoint)
        current = current.parent
    return None


def build_existing_manifest(
    semantic_root: Path,
    model_path: Path,
    forget_ids: Sequence[int],
    retain_ids: Sequence[int],
    overrides: Mapping[str, Path],
    metric_tolerance: float,
    ppl_tolerance: float,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    expected_forget = list(forget_ids)
    expected_retain = list(retain_ids)
    for spec in EXISTING_METHOD_SPECS:
        result_path = locate_existing_result(
            spec,
            semantic_root,
            overrides,
            metric_tolerance,
            ppl_tolerance,
        )
        payload = read_json(result_path)
        metrics = extract_result_metrics(payload)
        validate_reference_metrics(
            spec,
            metrics,
            metric_tolerance,
            ppl_tolerance,
        )
        metadata = validate_result_protocol(spec.display_name, payload)
        stored_forget, stored_retain = extract_stored_case_ids(payload)
        if stored_forget is not None and stored_forget != expected_forget:
            raise RuntimeError(
                f"{spec.display_name} stored forget case IDs differ from the "
                "official seed-0 split"
            )
        if stored_retain is not None and stored_retain != expected_retain:
            raise RuntimeError(
                f"{spec.display_name} stored retain case IDs differ from the "
                "official seed-0 split"
            )
        model_verified, model_evidence = find_model_evidence(
            result_path,
            payload,
            model_path,
        )
        if not model_verified:
            raise RuntimeError(
                f"Could not verify that {spec.display_name} derives from model "
                f"revision {MODEL_REVISION}. Add saved provenance beside "
                f"{result_path} rather than assuming model equivalence."
            )
        summary = payload.get("forget", payload.get("summary", payload))
        evaluator_verified = (
            "official_eval" in result_path.name
            and isinstance(summary, dict)
            and "Spe" in summary
            and ("Spe_success" in summary or "post_neighborhood_success" in summary)
        )
        if not evaluator_verified:
            raise RuntimeError(
                f"{spec.display_name} does not have verifiable official-evaluator "
                f"provenance: {result_path}"
            )
        entries.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "source_json_path": display_path(result_path),
                "checkpoint_path": (
                    None
                    if spec.key == "base"
                    else checkpoint_path_from_result(
                        result_path,
                        payload,
                    )
                ),
                **metadata,
                "model_path": str(model_path),
                "model_revision": MODEL_REVISION,
                "model_provenance_evidence": [
                    display_path(Path(path)) for path in model_evidence
                ],
                "forget_case_ids": expected_forget,
                "retain_case_ids": expected_retain,
                "case_ids_source": (
                    "stored_in_result"
                    if stored_forget is not None and stored_retain is not None
                    else "reconstructed_from_official_sampler_seed0"
                ),
                **metrics,
                "same_split_verified": True,
                "same_model_verified": True,
                "same_evaluator_verified": True,
            }
        )
    return entries


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def validate_base_metrics(
    result: Mapping[str, Any],
    metric_tolerance: float,
    ppl_tolerance: float,
) -> None:
    metrics = extract_result_metrics(result)
    base_spec = EXISTING_METHOD_SPECS[0]
    validate_reference_metrics(
        base_spec,
        metrics,
        metric_tolerance,
        ppl_tolerance,
    )


@contextmanager
def working_directory(path: Path) -> Iterable[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def import_original_zerounlearn(zero_root: Path) -> Tuple[Any, Any]:
    zero_root_string = str(zero_root.resolve())
    if zero_root_string not in sys.path:
        sys.path.insert(0, zero_root_string)
    try:
        with working_directory(zero_root):
            from ZeroUnlearn import ZeroUnlearnHyperParams, apply_unl_to_model
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the vendored original ZeroUnlearn implementation. "
            "Use the environment specified by ZeroUnlearn/requirments.txt "
            "(notably its PyTorch, Transformers, and scientific-Python "
            "versions); no fallback algorithm will be substituted."
        ) from exc

    module_name = getattr(apply_unl_to_model, "__module__", "")
    if module_name != "ZeroUnlearn.ZeroUnlearn_main":
        raise RuntimeError(
            "Resolved the wrong ZeroUnlearn implementation: "
            f"apply_unl_to_model came from {module_name!r}"
        )
    return ZeroUnlearnHyperParams, apply_unl_to_model


def git_source_revision(repository_root: Path, path: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                [
                    "git",
                    "log",
                    "-1",
                    "--format=%H",
                    "--",
                    str(path.relative_to(repository_root)),
                ],
                cwd=repository_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment_official_result(
    result: Dict[str, Any],
    *,
    model_path: Path,
    forget_ids: Sequence[int],
    retain_ids: Sequence[int],
    runtime: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    result.update(
        {
            "method": METHOD if runtime is not None else "Base",
            "model_path": str(model_path),
            "model_revision": MODEL_REVISION,
            "dtype": DTYPE_NAME,
            "forget_case_ids": list(forget_ids),
            "retain_case_ids": list(retain_ids),
            "case_ids_source": "official_sampler_seed0",
            "zero_unlearn_runtime": dict(runtime) if runtime is not None else None,
        }
    )
    return result


def comparison_row_from_manifest(entry: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "Method": entry["display_name"],
        "Eff ↓": entry["Eff"],
        "Gen ↓": entry["Gen"],
        "Spe ↑": entry["Spe"],
        "PPL ↓": entry["PPL"],
        "Source result path": entry["source_json_path"],
        "Same split verified": bool(entry["same_split_verified"]),
        "Same model verified": bool(entry["same_model_verified"]),
        "Same evaluator verified": bool(entry["same_evaluator_verified"]),
    }


def zero_unlearn_manifest_entry(
    result_path: Path,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = extract_result_metrics(result)
    return {
        "key": "original_zerounlearn",
        "display_name": "Original ZeroUnlearn",
        "source_json_path": display_path(result_path),
        "checkpoint_path": None,
        "seed": SEED,
        "dataset": "MCF",
        "sample_mode": SAMPLE_MODE,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "model_path": result["model_path"],
        "model_revision": result["model_revision"],
        "forget_case_ids": result["forget_case_ids"],
        "retain_case_ids": result["retain_case_ids"],
        "case_ids_source": result["case_ids_source"],
        **metrics,
        "same_split_verified": True,
        "same_model_verified": True,
        "same_evaluator_verified": True,
    }


def pairwise_differences(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    indexed = {str(row["Method"]): row for row in rows}
    zero = indexed["Original ZeroUnlearn"]
    differences: List[Dict[str, Any]] = []
    for target_name in PAIRWISE_TARGETS:
        target = indexed[target_name]
        differences.append(
            {
                "comparison": f"Original ZeroUnlearn minus {target_name}",
                "reference_method": target_name,
                "Eff difference": float(zero["Eff ↓"]) - float(target["Eff ↓"]),
                "Gen difference": float(zero["Gen ↓"]) - float(target["Gen ↓"]),
                "Spe difference": float(zero["Spe ↑"]) - float(target["Spe ↑"]),
                "PPL difference": float(zero["PPL ↓"]) - float(target["PPL ↓"]),
            }
        )
    return differences


def write_comparison_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    differences: Sequence[Mapping[str, Any]],
) -> None:
    ordered = sorted(rows, key=lambda row: DISPLAY_ORDER.index(str(row["Method"])))
    json_path = output_dir / "comparison_seed0.json"
    write_json(
        json_path,
        {
            "protocol": {
                "dataset": "MCF",
                "seed": SEED,
                "sample_mode": SAMPLE_MODE,
                "forget_num": FORGET_NUM,
                "retain_num": RETAIN_NUM,
                "model_revision": MODEL_REVISION,
                "dtype": DTYPE_NAME,
                "evaluator": (
                    "semantic-unlearning/scripts/" "mcf_zero_unlearn_official_eval.py"
                ),
            },
            "rows": list(ordered),
            "pairwise_differences": list(differences),
        },
    )

    fieldnames = list(ordered[0].keys())
    with (output_dir / "comparison_seed0.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# Original ZeroUnlearn versus seed-0 methods",
        "",
        "Eff and Gen: lower is better. Spe: higher is better. PPL: lower is better.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in ordered:
        lines.append("| " + " | ".join(str(row[field]) for field in fieldnames) + " |")
    lines.extend(
        [
            "",
            "## Pairwise metric differences",
            "",
            "Each value is `Original ZeroUnlearn - comparison method`; metrics "
            "are not combined into a single score.",
            "",
            "| Comparison | Eff difference | Gen difference | Spe difference | "
            "PPL difference |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for difference in differences:
        lines.append(
            f"| {difference['comparison']} | "
            f"{difference['Eff difference']:.6g} | "
            f"{difference['Gen difference']:.6g} | "
            f"{difference['Spe difference']:.6g} | "
            f"{difference['PPL difference']:.6g} |"
        )
    (output_dir / "comparison_seed0.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def require_runtime_files(
    model_path: Path,
    mcf_path: Path,
    wikidata_dir: Path,
    hparams_path: Path,
    zero_root: Path,
) -> None:
    errors: List[str] = []
    if not model_path.is_dir():
        errors.append(f"model snapshot directory missing: {model_path}")
    if not mcf_path.is_file():
        errors.append(f"MCF JSON missing: {mcf_path}")
    if not wikidata_dir.is_dir():
        errors.append(f"official Wikidata PPL corpus missing: {wikidata_dir}")
    if not hparams_path.is_file():
        errors.append(f"ZeroUnlearn hparams missing: {hparams_path}")
    if not zero_root.is_dir():
        errors.append(f"ZeroUnlearn source root missing: {zero_root}")
    if errors:
        raise FileNotFoundError(
            "Required comparison inputs are unavailable:\n- " + "\n- ".join(errors)
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_protocol_args(args)

    model_path = Path(args.model_path).expanduser().resolve()
    zero_root = Path(args.zero_unlearn_root).expanduser().resolve()
    hparams_path = Path(args.hparams_path).expanduser().resolve()
    mcf_path = Path(args.mcf_path).expanduser().resolve()
    wikidata_dir = Path(args.wikidata_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    require_runtime_files(
        model_path,
        mcf_path,
        wikidata_dir,
        hparams_path,
        zero_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_hashes_before = hash_protocol_inputs(
        mcf_path,
        hparams_path,
        zero_root,
    )
    validate_expected_protocol_hashes(
        source_hashes_before,
        mcf_path,
        hparams_path,
        zero_root,
    )
    _, forget_records, retain_records = load_seed0_split(mcf_path)
    forget_ids = case_ids(forget_records)
    retain_ids = case_ids(retain_records)
    overrides = parse_result_overrides(args.existing_result, SEMANTIC_ROOT)
    manifest_entries = build_existing_manifest(
        SEMANTIC_ROOT,
        model_path,
        forget_ids,
        retain_ids,
        overrides,
        args.metric_tolerance,
        args.ppl_tolerance,
    )
    manifest_payload = {
        "protocol": {
            "seed": SEED,
            "dataset": "MCF",
            "sample_mode": SAMPLE_MODE,
            "forget_num": FORGET_NUM,
            "retain_num": RETAIN_NUM,
            "model_path": str(model_path),
            "model_revision": MODEL_REVISION,
            "dtype": DTYPE_NAME,
        },
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
        "case_ids_reconstructed": True,
        "case_id_reconstruction": (
            "MCF first half retain pool; second half forget pool; one "
            "random.Random(0) samples forget then retain"
        ),
        "methods": manifest_entries,
    }
    manifest_path = output_dir / "existing_results_manifest.json"
    write_json(manifest_path, manifest_payload)
    if args.manifest_only:
        print(f"Validated existing results manifest: {manifest_path}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "One CUDA GPU is required for original ZeroUnlearn; CUDA is unavailable"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_all_seeds(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading exact BF16 base snapshot: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    neutral_target, neutral_target_id = resolve_eos_neutral_target(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("Evaluating and validating the unedited base model")
    base_result = evaluate_loaded_model_official(
        method="Base",
        model=model,
        tok=tokenizer,
        model_dir=model_path,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=None,
        unlearn_num=FORGET_NUM,
        retain_num=RETAIN_NUM,
        seed=SEED,
        sample_mode=SAMPLE_MODE,
        skip_ppl=False,
    )
    augment_official_result(
        base_result,
        model_path=model_path,
        forget_ids=forget_ids,
        retain_ids=retain_ids,
        runtime=None,
    )
    base_result_path = output_dir / "base_seed0_official_eval.json"
    write_json(base_result_path, base_result)
    validate_base_metrics(
        base_result,
        args.metric_tolerance,
        args.ppl_tolerance,
    )
    print("Base validation passed")

    params_class, apply_unl_to_model = import_original_zerounlearn(zero_root)
    hparams = params_class.from_json(hparams_path)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(
            "Original hparams changed unexpectedly: expected layers [16,17,18], "
            f"got {hparams.layers}"
        )
    retain_requests = records_to_zero_unlearn_requests(retain_records)
    forget_requests = records_to_zero_unlearn_requests(
        forget_records,
        neutral_target=neutral_target,
    )
    validate_neutral_forget_requests(
        forget_records,
        forget_requests,
        neutral_target,
    )
    print(
        "Using tokenizer EOS as ZeroUnlearn's neutral forget target: "
        f"{neutral_target!r} (token id {neutral_target_id}); official "
        "evaluation records remain unchanged"
    )

    run_status = "running"
    runtime: Dict[str, Any] = {}
    provenance_path = output_dir / "zerounlearn_seed0_provenance.json"
    provenance: Dict[str, Any] = {
        "status": run_status,
        "method": METHOD,
        "algorithm_entrypoint": ("ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model"),
        "upstream_repository": UPSTREAM_REPOSITORY,
        "vendored_source_revision": git_source_revision(
            REPOSITORY_ROOT,
            zero_root / "ZeroUnlearn" / "ZeroUnlearn_main.py",
        ),
        "model_path": str(model_path),
        "model_revision": MODEL_REVISION,
        "dtype": DTYPE_NAME,
        "zero_unlearn_compute_dtype": "float32",
        "zero_unlearn_compute_dtype_reason": (
            "The original implementation combines FP32 moment matrices with "
            "model weights and requires matching FP32 matmul operands. The "
            "exact BF16-loaded starting values are upcast without reloading, "
            "edited by the original entrypoint, then cast back to BF16 for "
            "official evaluation."
        ),
        "seed": SEED,
        "dataset": "MCF",
        "sample_mode": SAMPLE_MODE,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
        "hparams_path": str(hparams_path),
        "hparams": read_json(hparams_path),
        "edit_layer_nums": EDIT_LAYER_NUMS,
        "add_retain": ADD_RETAIN,
        "use_h": USE_H,
        "neutral_target": {
            "role": "ZeroUnlearn M_n for forget-training requests only",
            "source": "tokenizer.eos_token",
            "token": neutral_target,
            "token_id": neutral_target_id,
            "forget_request_field": "target_new.str",
            "forget_request_count": len(forget_requests),
            "retain_requests_modified": False,
            "official_evaluation_records_modified": False,
            "source_mcf_modified": False,
        },
        "checkpoint_saved": False,
        "cuda_device_index": device.index,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "multi_gpu_device_map_used": False,
        "zero_unlearn_working_directory": str(SEMANTIC_ROOT),
        "source_hashes_before": source_hashes_before,
        "exact_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
    }
    write_json(provenance_path, provenance)
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        apply_started = time.perf_counter()
        print(
            "Upcasting the exact BF16-loaded starting weights to FP32 for the "
            "original closed-form matrix solve"
        )
        model.float()
        print("Applying original closed-form ZeroUnlearn in memory")
        with working_directory(SEMANTIC_ROOT):
            edited_model, original_weights = apply_unl_to_model(
                model=model,
                tok=tokenizer,
                retain_requests=retain_requests,
                unlearn_requests=forget_requests,
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
        apply_seconds = time.perf_counter() - apply_started
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        runtime = {
            "apply_seconds": apply_seconds,
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
        zero_result = evaluate_loaded_model_official(
            method=METHOD,
            model=edited_model,
            tok=tokenizer,
            model_dir=f"in-memory:{METHOD}",
            mcf_path=mcf_path,
            wikidata_dir=wikidata_dir,
            out_path=None,
            unlearn_num=FORGET_NUM,
            retain_num=RETAIN_NUM,
            seed=SEED,
            sample_mode=SAMPLE_MODE,
            skip_ppl=False,
        )
        runtime["official_evaluation_seconds"] = (
            time.perf_counter() - evaluation_started
        )
        runtime["total_seconds"] = (
            runtime["apply_seconds"] + runtime["official_evaluation_seconds"]
        )
        augment_official_result(
            zero_result,
            model_path=model_path,
            forget_ids=forget_ids,
            retain_ids=retain_ids,
            runtime=runtime,
        )
        zero_result_path = output_dir / "zerounlearn_seed0_official_eval.json"
        write_json(zero_result_path, zero_result)
        run_status = "completed"
    except Exception:
        run_status = "failed"
        raise
    finally:
        source_hashes_after = hash_protocol_inputs(
            mcf_path,
            hparams_path,
            zero_root,
        )
        hashes_unchanged = source_hashes_before == source_hashes_after
        provenance.update(
            {
                "status": run_status,
                "runtime": runtime or None,
                "source_hashes_after": source_hashes_after,
                "source_hashes_unchanged": hashes_unchanged,
            }
        )
        write_json(provenance_path, provenance)
        if not hashes_unchanged:
            raise RuntimeError(
                "MCF, hparams, or original ZeroUnlearn source hashes changed "
                "during execution"
            )

    zero_entry = zero_unlearn_manifest_entry(zero_result_path, zero_result)
    base_entry = {
        **next(entry for entry in manifest_entries if entry["key"] == "base"),
        "source_json_path": display_path(base_result_path),
        **extract_result_metrics(base_result),
    }
    comparison_entries = [
        base_entry,
        zero_entry,
        *[entry for entry in manifest_entries if entry["key"] != "base"],
    ]
    rows = [comparison_row_from_manifest(entry) for entry in comparison_entries]
    differences = pairwise_differences(rows)
    write_comparison_outputs(output_dir, rows, differences)
    write_json(
        output_dir / "complete_results_manifest.json",
        {
            **manifest_payload,
            "methods": comparison_entries,
            "zero_unlearn_runtime": runtime,
            "source_hashes_unchanged": True,
        },
    )

    metrics = extract_result_metrics(zero_result)
    print(
        "ZeroUnlearn result: "
        f"Eff={metrics['Eff']}, Gen={metrics['Gen']}, "
        f"Spe={metrics['Spe']}, Spe_success={metrics['Spe_success']}, "
        f"PPL={metrics['PPL']}"
    )
    print(
        f"Runtime={runtime['apply_seconds']:.3f}s; "
        f"peak CUDA={runtime['peak_cuda_memory_allocated_gib']:.3f} GiB"
    )
    print(f"Comparison outputs: {output_dir}")


if __name__ == "__main__":
    main()
