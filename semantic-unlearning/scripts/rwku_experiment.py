#!/usr/bin/env python3
"""Run the six-method RWKU target experiment for one seed.

Seed ``s`` maps to the ``s``-th published RWKU target, preserving RWKU's
single-target semantics.  The model sees only the deterministic calibration
half of level-1/level-2 probes during unlearning and repair.  Headline direct
and paraphrase metrics use the disjoint held-out level-2 half.

Methods:
  * Base model
  * Original ZeroUnlearn
  * Setting 5e without repair
  * Setting 5e + protected LM-head repair
  * Repair-only control
  * Protected representation unlearning
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_compare as gagd
from mcf_sampling import sample_official_mcf_records
from rwku_data import (
    DEFAULT_DATA_ROOT,
    RWKU_CODE_REVISION,
    RWKU_DATASET_REVISION,
    TARGETS_BY_SEED,
    build_split_manifest,
    ensure_positive_training_data,
    ensure_target_data,
    partition_records,
    target_for_seed,
)
from rwku_eval import (
    FrozenHeadProbe,
    build_frozen_head_probe,
    evaluate_rwku,
    format_qa_prompt,
    load_wikidata_text,
    write_json,
)
from rwku_repair import (
    RepairConfig,
    run_protected_lm_head_repair,
    validate_config as validate_repair_config,
)
from rwku_representation import (
    RepresentationConfig,
    run_representation_unlearning,
    validate_config as validate_representation_config,
)
from run_zerounlearn_official_mcf import (
    DEFAULT_HPARAMS,
    DEFAULT_ZERO_ROOT,
    import_original_zerounlearn,
    records_to_zero_unlearn_requests,
    resolve_eos_neutral_target,
    working_directory,
)


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_ROOT = SEMANTIC_ROOT / "outputs" / "rwku"
DEFAULT_MCF_PATH = SEMANTIC_ROOT / "data" / "multi_counterfact.json"
DEFAULT_WIKIDATA_DIR = SEMANTIC_ROOT / "data" / "wikidata"
DEFAULT_MODEL_PATH = gagd.DEFAULT_MODEL_PATH
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE

METHOD_BASE = "Base model"
METHOD_ZERO = "Original ZeroUnlearn"
METHOD_SETTING5 = "Setting 5e without repair"
METHOD_REPAIRED = "Setting 5e + protected LM-head repair"
METHOD_REPAIR_ONLY = "Repair-only control"
METHOD_REPRESENTATION = "Protected representation unlearning"
METHOD_ORDER = (
    METHOD_BASE,
    METHOD_ZERO,
    METHOD_SETTING5,
    METHOD_REPAIRED,
    METHOD_REPAIR_ONLY,
    METHOD_REPRESENTATION,
)


def parse_candidate_scales(value: str) -> Tuple[float, ...]:
    scales = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if not scales:
        raise argparse.ArgumentTypeError("candidate scale list is empty")
    if any(not 0.0 <= scale <= 1.0 for scale in scales):
        raise argparse.ArgumentTypeError("candidate scales must be in [0,1]")
    if 0.0 not in scales:
        scales += (0.0,)
    return tuple(sorted(set(scales), reverse=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=range(10))
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mcf-path", type=Path, default=DEFAULT_MCF_PATH)
    parser.add_argument("--wikidata-dir", type=Path, default=DEFAULT_WIKIDATA_DIR)
    parser.add_argument("--zero-root", type=Path, default=DEFAULT_ZERO_ROOT)
    parser.add_argument("--zero-hparams", type=Path, default=DEFAULT_HPARAMS)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--repair-retain-num", type=int, default=128)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated keys: base,zero,setting5,repaired,repair-only,"
            "representation. "
            "The base pass is always run because all probability ratios and "
            "frozen-head probes require it."
        ),
    )

    # Established Setting 5e values used for MCF/ZsRE.
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=4)
    parser.add_argument("--emb-lm-lr", type=float, default=1e-4)
    parser.add_argument("--forget-weight", type=float, default=2.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--emb-lm-optimizer",
        choices=["sgd", "adam", "adamw", "adamw8bit"],
        default="adamw",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["epoch", "with_replacement"],
        default="epoch",
    )
    parser.add_argument("--post-training-new-true-alpha", type=float, default=0.75)
    parser.add_argument("--post-training-new-retain-alpha", type=float, default=0.50)
    parser.add_argument(
        "--post-training-new-true-retain-alpha",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Protected sparse active-pair LM-head repair.
    parser.add_argument("--repair-steps", type=int, default=800)
    parser.add_argument("--repair-lr", type=float, default=5e-3)
    parser.add_argument("--repair-active-margin", type=float, default=0.25)
    parser.add_argument("--repair-selection-margin", type=float, default=0.05)
    parser.add_argument("--repair-l2-lambda", type=float, default=1e-6)
    parser.add_argument("--repair-protected-logit-lambda", type=float, default=1.0)
    parser.add_argument("--repair-max-delta-norm", type=float, default=None)
    parser.add_argument(
        "--project-away-protected-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--repair-protected-projection-rank",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--repair-protected-contexts-per-example",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--repair-exclude-protected-answer-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--repair-min-protected-probability-ratio",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--repair-max-protected-logit-drift",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--repair-max-protected-top1-changes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--repair-stop-when-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--repair-candidate-scales",
        type=parse_candidate_scales,
        default=parse_candidate_scales(
            "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,"
            ".0625,.046875,.03125,.015625,.0078125,0"
        ),
    )

    # Base-initialized representation unlearning. The output head and input
    # embeddings remain frozen; only low-rank late-transformer deltas train.
    parser.add_argument("--representation-steps", type=int, default=1250)
    parser.add_argument("--representation-lr", type=float, default=2e-4)
    parser.add_argument("--representation-rank", type=int, default=16)
    parser.add_argument("--representation-alpha", type=float, default=32.0)
    parser.add_argument("--representation-last-n-layers", type=int, default=8)
    parser.add_argument("--representation-max-length", type=int, default=512)
    parser.add_argument("--representation-retain-top-k", type=int, default=64)
    parser.add_argument(
        "--representation-retain-examples-per-step",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--representation-external-mc-retain-limit",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--representation-external-mc-gate-limit",
        type=int,
        default=64,
    )
    parser.add_argument("--representation-positive-max-rows", type=int, default=64)
    parser.add_argument(
        "--representation-positive-subject-task-max-rows",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--representation-matched-positive-max-rows",
        type=int,
        default=72,
    )
    parser.add_argument(
        "--representation-positive-gate-fraction",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--representation-positive-tokens-per-row",
        type=int,
        default=512,
    )
    parser.add_argument("--representation-answer-margin", type=float, default=8.0)
    parser.add_argument(
        "--representation-answer-probability-target",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--representation-frozen-head-logit-spread-tolerance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--representation-mc-logit-spread-tolerance",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--representation-answer-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-answer-probability-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-frozen-head-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--representation-mc-weight", type=float, default=1.0)
    parser.add_argument(
        "--representation-positive-proxy-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--representation-matched-positive-retain-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-concept-erasure-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-concept-orthogonal-retain-weight",
        type=float,
        default=0.5,
    )
    parser.add_argument("--representation-concept-rank", type=int, default=8)
    parser.add_argument(
        "--representation-retain-kl-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--representation-retain-answer-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--representation-retain-hidden-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--representation-max-retain-kl",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--representation-max-retain-p95-kl",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--representation-min-retain-probability-ratio",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--representation-max-retain-probability-ratio",
        type=float,
        default=1.005,
    )
    parser.add_argument(
        "--representation-min-retain-p05-probability-ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--representation-max-retain-p95-probability-ratio",
        type=float,
        default=1.05,
    )
    parser.add_argument(
        "--representation-min-retain-top1-agreement",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--representation-min-retain-hidden-cosine",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--representation-min-retain-p05-hidden-cosine",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--representation-max-retain-hidden-relative-l2",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--representation-max-retain-p95-hidden-relative-l2",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--representation-max-proxy-mia-advantage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--representation-max-matched-positive-feature-drift",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--representation-max-calibration-generation-recovery",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--representation-max-frozen-head-chance-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-grad-clip",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--representation-gate-retain-limit",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--representation-selection-calibration-limit",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--representation-selection-generation-limit",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--representation-selection-generation-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--representation-selection-generation-max-new-tokens",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--representation-checkpoint-interval",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--representation-checkpoint-funnel-count",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--representation-checkpoint-funnel-retain-limit",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--representation-checkpoint-funnel-calibration-limit",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--representation-checkpoint-funnel-scales",
        type=parse_candidate_scales,
        default=parse_candidate_scales(
            "1,.5,.25,.125,.0625,.015625,0"
        ),
    )
    parser.add_argument(
        "--representation-checkpoint-scale-neighborhood",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--representation-candidate-scales",
        type=parse_candidate_scales,
        default=parse_candidate_scales(
            "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,"
            ".0625,.03125,.015625,0"
        ),
    )

    # Bounded smoke-evaluation controls. Omitted means the full benchmark.
    parser.add_argument("--forget-eval-limit", type=int, default=None)
    parser.add_argument("--adversarial-eval-limit", type=int, default=None)
    parser.add_argument("--mia-eval-limit", type=int, default=None)
    parser.add_argument("--neighbor-eval-limit", type=int, default=None)
    parser.add_argument("--utility-eval-limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, splits, paths, and configuration without a model.",
    )

    # Compatibility fields consumed by gagd_compare.train_mode.
    parser.set_defaults(
        dataset="mcf",
        mode=SETTING5_MODE,
        lr=1e-5,
        full_lr=None,
        optimizer=None,
        full_optimizer=None,
        forget_loss_type="mcf_margin",
        kl_retain_weight=0.0,
        save_model=False,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be strictly between 0 and 1")
    for name in (
        "steps",
        "batch_size",
        "retain_batch_size",
        "eval_batch_size",
        "retain_num",
        "repair_retain_num",
        "repair_steps",
        "repair_protected_contexts_per_example",
        "representation_steps",
        "representation_rank",
        "representation_last_n_layers",
        "representation_max_length",
        "representation_retain_top_k",
        "representation_retain_examples_per_step",
        "representation_external_mc_retain_limit",
        "representation_external_mc_gate_limit",
        "representation_positive_max_rows",
        "representation_positive_subject_task_max_rows",
        "representation_matched_positive_max_rows",
        "representation_positive_tokens_per_row",
        "representation_concept_rank",
        "representation_gate_retain_limit",
        "representation_selection_calibration_limit",
        "representation_selection_generation_batch_size",
        "representation_selection_generation_max_new_tokens",
        "representation_checkpoint_interval",
        "representation_checkpoint_funnel_count",
        "representation_checkpoint_funnel_retain_limit",
        "representation_checkpoint_funnel_calibration_limit",
        "representation_checkpoint_scale_neighborhood",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.representation_selection_generation_limit < 0:
        raise ValueError(
            "--representation-selection-generation-limit must be non-negative"
        )
    for name in (
        "forget_eval_limit",
        "adversarial_eval_limit",
        "mia_eval_limit",
        "neighbor_eval_limit",
        "utility_eval_limit",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.repair_protected_projection_rank < 0:
        raise ValueError("--repair-protected-projection-rank must be non-negative")
    if args.repair_max_protected_top1_changes < 0:
        raise ValueError("--repair-max-protected-top1-changes must be non-negative")
    if not 0.0 < args.repair_min_protected_probability_ratio <= 1.0:
        raise ValueError(
            "--repair-min-protected-probability-ratio must be in (0,1]"
        )
    if args.repair_max_protected_logit_drift < 0:
        raise ValueError("--repair-max-protected-logit-drift must be non-negative")
    if args.repair_protected_logit_lambda < 0:
        raise ValueError("--repair-protected-logit-lambda must be non-negative")
    for name in (
        "post_training_new_true_alpha",
        "post_training_new_retain_alpha",
        "post_training_new_true_retain_alpha",
    ):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            raise ValueError(f"{name} must be in [0,1]")
    for name in ("emb_lm_lr",):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "forget_weight",
        "retain_weight",
        "forget_margin",
        "weight_decay",
        "grad_clip",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    validate_repair_config(repair_config(args))
    validate_representation_config(representation_config(args))
    if args.dtype in {"bf16", "fp16"} and args.dry_run:
        return
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError(
            "The RWKU methods require a CUDA GPU. Use --dry-run to validate "
            "the pinned data/protocol on a CPU-only machine."
        )


def selected_methods(value: str) -> Tuple[str, ...]:
    mapping = {
        "base": METHOD_BASE,
        "zero": METHOD_ZERO,
        "setting5": METHOD_SETTING5,
        "repaired": METHOD_REPAIRED,
        "repair-only": METHOD_REPAIR_ONLY,
        "representation": METHOD_REPRESENTATION,
    }
    if value.strip().lower() == "all":
        return METHOD_ORDER
    keys = [key.strip().lower() for key in value.split(",") if key.strip()]
    unknown = sorted(set(keys) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown method key(s): {unknown}")
    methods = [METHOD_BASE]
    methods.extend(mapping[key] for key in keys if mapping[key] != METHOD_BASE)
    return tuple(dict.fromkeys(methods))


def dtype_from_name(value: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[value]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    model_path: str,
    *,
    dtype: torch.dtype,
    for_training: bool,
    gradient_checkpointing: bool,
) -> Tuple[nn.Module, Any]:
    path = Path(model_path)
    if path.is_absolute() and not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.config.use_cache = not for_training
    if for_training and gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def release_model(model: Optional[nn.Module]) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_model_for_evaluation(model: nn.Module) -> None:
    """Restore generation-efficient state after a training-only load."""

    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if hasattr(model, "config"):
        model.config.use_cache = True
    model.eval()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative names and bytes for a small local evaluation corpus."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Directory contains no files: {root}")
    for item in files:
        digest.update(str(item.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def local_model_identity(
    model_path: str,
    *,
    allow_missing: bool = False,
) -> Dict[str, Any]:
    path = Path(model_path)
    if path.is_absolute() and not path.exists():
        if allow_missing:
            return {
                "requested": str(model_path),
                "kind": "missing_local_path",
                "exists": False,
            }
        raise FileNotFoundError(f"Model path does not exist: {path}")
    identity: Dict[str, Any] = {"requested": str(model_path)}
    if not path.is_dir():
        identity["kind"] = "hub_or_relative_identifier"
        return identity
    resolved = path.resolve()
    identity.update(
        {
            "kind": "local_directory",
            "resolved_path": str(resolved),
        }
    )
    # Hugging Face cache snapshots are content-addressed directories of the
    # form ``.../snapshots/<revision>``.  Record that revision explicitly so a
    # ten-target aggregate cannot silently mix snapshots that happen to expose
    # the same filenames and sizes.
    snapshot_indices = [
        index
        for index, component in enumerate(resolved.parts[:-1])
        if component == "snapshots"
    ]
    if snapshot_indices:
        snapshot_index = snapshot_indices[-1]
        if snapshot_index + 1 < len(resolved.parts):
            identity["snapshot_revision"] = resolved.parts[snapshot_index + 1]
    metadata_names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    identity["metadata_sha256"] = {
        name: file_sha256(resolved / name)
        for name in metadata_names
        if (resolved / name).is_file()
    }
    weight_patterns = ("*.safetensors", "pytorch_model*.bin")
    weight_files = sorted(
        {
            item
            for pattern in weight_patterns
            for item in resolved.glob(pattern)
            if item.is_file()
        }
    )
    weight_identities: List[Dict[str, Any]] = []
    for item in weight_files:
        stat = item.stat()
        weight_identity: Dict[str, Any] = {
            "name": item.name,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if item.is_symlink():
            # In the HF cache the symlink target basename is the immutable blob
            # digest.  Capturing it is a cheap content identity and avoids
            # hashing multi-gigabyte shards once per seed.
            weight_identity["symlink_target"] = str(item.readlink())
            weight_identity["resolved_target_name"] = item.resolve().name
        weight_identities.append(weight_identity)
    identity["weight_files"] = weight_identities
    return identity


def implementation_identity() -> Dict[str, str]:
    paths = (
        SCRIPT_PATH,
        SEMANTIC_ROOT / "scripts" / "rwku_data.py",
        SEMANTIC_ROOT / "scripts" / "rwku_eval.py",
        SEMANTIC_ROOT / "scripts" / "rwku_repair.py",
        SEMANTIC_ROOT / "scripts" / "rwku_representation.py",
        SEMANTIC_ROOT / "scripts" / "aggregate_rwku_results.py",
        SEMANTIC_ROOT / "scripts" / "gagd_compare.py",
        SEMANTIC_ROOT / "scripts" / "gagd_active_case_repair.py",
        SEMANTIC_ROOT / "scripts" / "mcf_sampling.py",
        SEMANTIC_ROOT / "scripts" / "mcf_zero_unlearn_official_eval.py",
        SEMANTIC_ROOT / "scripts" / "run_zerounlearn_official_mcf.py",
        SEMANTIC_ROOT / "scripts" / "run_rwku_experiment.sh",
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path)
        for path in paths
    }


def zero_unlearn_identity(args: argparse.Namespace) -> Dict[str, Any]:
    entrypoint = Path(args.zero_root) / "ZeroUnlearn" / "ZeroUnlearn_main.py"
    hparams_impl = (
        Path(args.zero_root) / "ZeroUnlearn" / "ZeroUnlearn_hparams.py"
    )
    for path in (entrypoint, hparams_impl, Path(args.zero_hparams)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing ZeroUnlearn protocol file: {path}")
    return {
        "root": str(Path(args.zero_root).resolve()),
        "entrypoint_sha256": file_sha256(entrypoint),
        "hparams_implementation_sha256": file_sha256(hparams_impl),
        "hparams_path": str(Path(args.zero_hparams).resolve()),
        "hparams_sha256": file_sha256(Path(args.zero_hparams)),
    }


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def example_content_sha256(example: gagd.Example) -> str:
    return mapping_sha256(
        {
            "prompt": str(example.prompt),
            "answer": str(example.answer),
        }
    )


def load_mcf_retain(
    path: Path,
    *,
    seed: int,
    retain_num: int,
) -> Tuple[List[Dict[str, Any]], List[gagd.Example]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("MCF data must contain a JSON list")
    _, retain_records = sample_official_mcf_records(
        raw,
        forget_num=1,
        retain_num=retain_num,
        seed=seed,
        strict=True,
    )
    examples: List[gagd.Example] = []
    for record in retain_records:
        rewrite, paraphrases = gagd.extract_mcf_rewrite(record)
        subject = str(rewrite["subject"])
        target_new = gagd.normalize_answer(str(rewrite["target_new"]["str"]))
        target_true_mapping = rewrite.get("target_true")
        if not isinstance(target_true_mapping, Mapping):
            raise ValueError("MCF retain record is missing target_true")
        target_true = gagd.normalize_answer(str(target_true_mapping["str"]))
        examples.append(
            gagd.Example(
                prompt=gagd.format_mcf_prompt(str(rewrite["prompt"]), subject),
                answer=target_new,
                subject=subject,
                target_new=target_new,
                target_true=target_true,
                paraphrase_prompts=[
                    gagd.format_mcf_prompt(value, subject)
                    for value in paraphrases
                ],
                source="mcf_retain",
            )
        )
    return retain_records, examples


def load_matched_positive_training_rows(
    data_root: Path,
    *,
    seed: int,
    allow_download: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Interleave non-target RWKU positive-training rows by subject.

    These are structurally matched nonmember proxies for the MIA objective.
    No level-3, official MIA, neighbor, or utility record is loaded here.
    """

    banks: List[List[Dict[str, Any]]] = []
    hashes: Dict[str, str] = {}
    for offset in range(1, len(TARGETS_BY_SEED)):
        other_seed = (seed + offset) % len(TARGETS_BY_SEED)
        other_target, positive_rows, digest = ensure_positive_training_data(
            data_root,
            other_seed,
            allow_download=allow_download,
        )
        banks.append(list(positive_rows))
        hashes[other_target.directory] = digest
    if not banks:
        raise RuntimeError("RWKU matched-positive bank is empty")
    rows: List[Dict[str, Any]] = []
    for row_index in range(max(len(bank) for bank in banks)):
        for bank in banks:
            if row_index < len(bank):
                rows.append(dict(bank[row_index]))
    return rows, hashes


def setting5_examples(
    tokenizer: Any,
    calibration_rows: Sequence[Mapping[str, Any]],
) -> List[gagd.Example]:
    if not tokenizer.eos_token:
        raise ValueError("Setting 5e requires tokenizer.eos_token")
    examples: List[gagd.Example] = []
    for row in calibration_rows:
        sensitive = gagd.normalize_answer(str(row["answer"]))
        examples.append(
            gagd.Example(
                prompt=format_qa_prompt(tokenizer, row),
                answer=sensitive,
                subject=str(row["subject"]),
                target_new=sensitive,
                target_true=str(tokenizer.eos_token),
                paraphrase_prompts=[],
                source="rwku_calibration",
            )
        )
    return examples


def _placeholder_prompt(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    subject: str,
) -> str:
    adapted = dict(row)
    query = str(row["query"])
    placeholder_query, replacements = re.subn(
        re.escape(subject),
        "{}",
        query,
        count=1,
        flags=re.IGNORECASE,
    )
    if replacements == 0:
        placeholder_query = "Regarding {}, " + query
    adapted["query"] = placeholder_query
    prompt = format_qa_prompt(tokenizer, adapted)
    if prompt.count("{}") != 1:
        raise ValueError("ZeroUnlearn prompt must contain exactly one subject placeholder")
    return prompt


def zerounlearn_forget_requests(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    subject: str,
    seed: int,
) -> List[Dict[str, Any]]:
    neutral, _ = resolve_eos_neutral_target(tokenizer)
    requests: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        requests.append(
            {
                "case_id": seed * 100_000 + index,
                "prompt": _placeholder_prompt(
                    tokenizer,
                    row,
                    subject=subject,
                ),
                "subject": subject,
                "target_true": {"str": str(row["answer"])},
                "target_new": {"str": neutral},
            }
        )
    return requests


def repair_config(args: argparse.Namespace) -> RepairConfig:
    return RepairConfig(
        steps=args.repair_steps,
        learning_rate=args.repair_lr,
        active_margin=args.repair_active_margin,
        selection_margin=args.repair_selection_margin,
        l2_lambda=args.repair_l2_lambda,
        protected_logit_lambda=args.repair_protected_logit_lambda,
        max_delta_norm=args.repair_max_delta_norm,
        project_away_protected=args.project_away_protected_hidden,
        protected_projection_rank=args.repair_protected_projection_rank,
        protected_contexts_per_example=(
            args.repair_protected_contexts_per_example
        ),
        exclude_protected_answer_rows=(
            args.repair_exclude_protected_answer_rows
        ),
        min_protected_probability_ratio=(
            args.repair_min_protected_probability_ratio
        ),
        max_protected_logit_drift=(
            args.repair_max_protected_logit_drift
        ),
        max_protected_top1_changes=(
            args.repair_max_protected_top1_changes
        ),
        stop_when_satisfied=args.repair_stop_when_satisfied,
        candidate_scales=tuple(args.repair_candidate_scales),
    )


def representation_config(args: argparse.Namespace) -> RepresentationConfig:
    """Resolve the fixed, Base-initialized representation protocol."""

    return RepresentationConfig(
        steps=args.representation_steps,
        learning_rate=args.representation_lr,
        rank=args.representation_rank,
        alpha=args.representation_alpha,
        last_n_layers=args.representation_last_n_layers,
        target_modules=(
            "q_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_length=args.representation_max_length,
        retain_top_k=args.representation_retain_top_k,
        retain_examples_per_step=(
            args.representation_retain_examples_per_step
        ),
        external_mc_retain_limit=(
            args.representation_external_mc_retain_limit
        ),
        external_mc_gate_limit=(
            args.representation_external_mc_gate_limit
        ),
        positive_max_rows=args.representation_positive_max_rows,
        positive_subject_task_max_rows=(
            args.representation_positive_subject_task_max_rows
        ),
        matched_positive_max_rows=(
            args.representation_matched_positive_max_rows
        ),
        positive_gate_fraction=args.representation_positive_gate_fraction,
        positive_tokens_per_row=(
            args.representation_positive_tokens_per_row
        ),
        answer_demotion_margin=args.representation_answer_margin,
        answer_probability_target=(
            args.representation_answer_probability_target
        ),
        frozen_head_logit_spread_tolerance=(
            args.representation_frozen_head_logit_spread_tolerance
        ),
        mc_logit_spread_tolerance=(
            args.representation_mc_logit_spread_tolerance
        ),
        answer_demotion_weight=args.representation_answer_weight,
        answer_probability_weight=(
            args.representation_answer_probability_weight
        ),
        frozen_head_weight=args.representation_frozen_head_weight,
        mc_weight=args.representation_mc_weight,
        positive_proxy_weight=args.representation_positive_proxy_weight,
        matched_positive_retain_weight=(
            args.representation_matched_positive_retain_weight
        ),
        concept_erasure_weight=(
            args.representation_concept_erasure_weight
        ),
        concept_orthogonal_retain_weight=(
            args.representation_concept_orthogonal_retain_weight
        ),
        concept_rank=args.representation_concept_rank,
        retain_kl_weight=args.representation_retain_kl_weight,
        retain_answer_weight=args.representation_retain_answer_weight,
        retain_hidden_weight=args.representation_retain_hidden_weight,
        candidate_scales=tuple(args.representation_candidate_scales),
        max_retain_kl=args.representation_max_retain_kl,
        max_retain_p95_kl=args.representation_max_retain_p95_kl,
        min_retain_answer_probability_ratio=(
            args.representation_min_retain_probability_ratio
        ),
        max_retain_answer_probability_ratio=(
            args.representation_max_retain_probability_ratio
        ),
        min_retain_p05_probability_ratio=(
            args.representation_min_retain_p05_probability_ratio
        ),
        max_retain_p95_probability_ratio=(
            args.representation_max_retain_p95_probability_ratio
        ),
        min_retain_top1_agreement=(
            args.representation_min_retain_top1_agreement
        ),
        min_retain_hidden_cosine=(
            args.representation_min_retain_hidden_cosine
        ),
        min_retain_p05_hidden_cosine=(
            args.representation_min_retain_p05_hidden_cosine
        ),
        max_retain_hidden_relative_l2=(
            args.representation_max_retain_hidden_relative_l2
        ),
        max_retain_p95_hidden_relative_l2=(
            args.representation_max_retain_p95_hidden_relative_l2
        ),
        max_proxy_mia_advantage=(
            args.representation_max_proxy_mia_advantage
        ),
        max_matched_positive_base_feature_drift=(
            args.representation_max_matched_positive_feature_drift
        ),
        max_calibration_generation_recovery=(
            args.representation_max_calibration_generation_recovery
        ),
        max_calibration_frozen_head_chance_ratio=(
            args.representation_max_frozen_head_chance_ratio
        ),
        grad_clip=args.representation_grad_clip,
        seed=args.seed,
        gate_retain_limit=args.representation_gate_retain_limit,
        selection_calibration_limit=(
            args.representation_selection_calibration_limit
        ),
        selection_generation_limit=(
            args.representation_selection_generation_limit
        ),
        selection_generation_batch_size=(
            args.representation_selection_generation_batch_size
        ),
        selection_generation_max_new_tokens=(
            args.representation_selection_generation_max_new_tokens
        ),
        checkpoint_interval=args.representation_checkpoint_interval,
        checkpoint_funnel_count=(
            args.representation_checkpoint_funnel_count
        ),
        checkpoint_funnel_retain_limit=(
            args.representation_checkpoint_funnel_retain_limit
        ),
        checkpoint_funnel_calibration_limit=(
            args.representation_checkpoint_funnel_calibration_limit
        ),
        checkpoint_funnel_scales=tuple(
            scale
            for scale in args.representation_checkpoint_funnel_scales
            if scale > 0.0
        ),
        checkpoint_scale_neighborhood=(
            args.representation_checkpoint_scale_neighborhood
        ),
    )


def evaluation_limits(args: argparse.Namespace) -> Dict[str, int]:
    values = {
        "forget": args.forget_eval_limit,
        "adversarial": args.adversarial_eval_limit,
        "mia": args.mia_eval_limit,
        "neighbor": args.neighbor_eval_limit,
        "utility": args.utility_eval_limit,
    }
    return {key: int(value) for key, value in values.items() if value is not None}


def evaluate_method(
    *,
    method: str,
    model: nn.Module,
    tokenizer: Any,
    target_subject: str,
    held_out_cloze: Sequence[Mapping[str, Any]],
    held_out_direct: Sequence[Mapping[str, Any]],
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    args: argparse.Namespace,
    base_retain_mean_logprobs: Optional[Mapping[str, float]],
    frozen_probe: FrozenHeadProbe,
) -> Dict[str, Any]:
    started = time.perf_counter()
    result = evaluate_rwku(
        method=method,
        model=model,
        tokenizer=tokenizer,
        subject=target_subject,
        held_out_cloze=held_out_cloze,
        held_out_direct=held_out_direct,
        datasets=datasets,
        wikidata_dir=args.wikidata_dir,
        batch_size=args.eval_batch_size,
        base_retain_mean_logprobs=base_retain_mean_logprobs,
        frozen_head_probe=frozen_probe,
        limits=evaluation_limits(args),
        skip_ppl=args.skip_ppl,
    )
    result["runtime"] = {
        "evaluation_seconds": time.perf_counter() - started,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        ),
    }
    return result


def save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    path: Path,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def run_original_zero(
    *,
    args: argparse.Namespace,
    tokenizer: Any,
    model: nn.Module,
    calibration_rows: Sequence[Mapping[str, Any]],
    target_subject: str,
    retain_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    parameters_class, apply_unlearning = import_original_zerounlearn(
        args.zero_root
    )
    hparams = parameters_class.from_json(args.zero_hparams)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(
            "Expected original ZeroUnlearn layers [16,17,18], got "
            f"{list(hparams.layers)}"
        )
    retain_requests = records_to_zero_unlearn_requests(retain_records)
    forget_requests = zerounlearn_forget_requests(
        tokenizer,
        calibration_rows,
        subject=target_subject,
        seed=args.seed,
    )
    started = time.perf_counter()
    model.float()
    with working_directory(SEMANTIC_ROOT):
        edited_model, original_weights = apply_unlearning(
            model=model,
            tok=tokenizer,
            retain_requests=retain_requests,
            unlearn_requests=forget_requests,
            hparams=hparams,
            copy=False,
            return_orig_weights=False,
            cache_template=None,
            save_path=None,
            add_retain=False,
            edit_layer_nums=3,
            use_h=False,
        )
    del original_weights
    edited_model.to(dtype=dtype_from_name(args.dtype))
    edited_model.eval()
    return {
        "model": edited_model,
        "provenance": {
            "algorithm_entrypoint": (
                "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model"
            ),
            "hparams_path": str(args.zero_hparams),
            "hparams_sha256": file_sha256(args.zero_hparams),
            "calibration_request_count": len(forget_requests),
            "retain_request_count": len(retain_requests),
            "sensitive_field": "target_true",
            "neutral_field": "target_new",
            "neutral_source": "tokenizer.eos_token",
            "compute_dtype": "float32",
            "output_dtype": args.dtype,
            "apply_seconds": time.perf_counter() - started,
        },
    }


def config_payload(
    args: argparse.Namespace,
    *,
    methods: Sequence[str],
    target: Any,
    calibration_rows: Sequence[Mapping[str, Any]],
    held_out_direct: Sequence[Mapping[str, Any]],
    file_hashes: Mapping[str, str],
    split_manifests: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "dataset": "RWKU",
        "rwku_code_revision": RWKU_CODE_REVISION,
        "rwku_dataset_revision": RWKU_DATASET_REVISION,
        "seed": args.seed,
        "target": asdict(target),
        "single_target_run": True,
        "methods": list(methods),
        "model_path": str(args.model_path),
        "model_identity": local_model_identity(
            str(args.model_path),
            allow_missing=args.dry_run,
        ),
        "dtype": args.dtype,
        "implementation_file_sha256": implementation_identity(),
        "calibration_fraction": args.calibration_fraction,
        "calibration_count": len(calibration_rows),
        "held_out_direct_count": len(held_out_direct),
        "split_manifests": split_manifests,
        "data_file_sha256": dict(file_hashes),
        "setting5": {
            "mode": SETTING5_MODE,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "retain_batch_size": args.retain_batch_size,
            "learning_rate": args.emb_lm_lr,
            "optimizer": args.emb_lm_optimizer,
            "forget_weight": args.forget_weight,
            "retain_weight": args.retain_weight,
            "forget_margin": args.forget_margin,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
            "sampling_strategy": args.sampling_strategy,
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "post_training_overlap_alphas": [
                args.post_training_new_true_alpha,
                args.post_training_new_retain_alpha,
                args.post_training_new_true_retain_alpha,
            ],
        },
        "external_retain_partitions": {
            "optimization_count": args.retain_num,
            "checkpoint_gate_count": args.repair_retain_num,
            "sampling": (
                "one deterministic MCF sample split into disjoint optimization "
                "and external checkpoint-gate partitions"
            ),
        },
        "repair": asdict(repair_config(args)),
        "representation": {
            **asdict(representation_config(args)),
            "initialization": "fresh_base_model",
            "target_training_source": "pinned positive.json",
            "mia_proxy_reference": (
                "round-robin positive.json rows from the other nine targets"
            ),
            "official_evaluation_records_used_for_selection": False,
        },
        "evaluation_limits": evaluation_limits(args),
        "skip_ppl": bool(args.skip_ppl),
        "eval_batch_size": int(args.eval_batch_size),
        "dry_run": args.dry_run,
        "exact_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    methods = selected_methods(args.methods)
    target = target_for_seed(args.seed)
    output_dir = Path(args.output_root) / f"seed{args.seed:02d}_{target.directory}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        # Invalidate a prior successful artifact before any preflight step can
        # fail, so strict aggregation can never pick up stale results.
        write_json(
            output_dir / "results.json",
            {
                "status": "preflight",
                "seed": args.seed,
                "target": asdict(target),
            },
        )

    target, datasets, file_hashes = ensure_target_data(
        args.data_root,
        args.seed,
        allow_download=not args.no_download,
    )
    calibration_level1, held_out_level1 = partition_records(
        datasets["forget_level1.json"],
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    calibration_level2, held_out_level2 = partition_records(
        datasets["forget_level2.json"],
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    calibration_rows = calibration_level1 + calibration_level2
    split_manifests = {
        "level1": build_split_manifest(
            calibration_level1,
            held_out_level1,
        ),
        "level2": build_split_manifest(
            calibration_level2,
            held_out_level2,
        ),
    }
    payload = config_payload(
        args,
        methods=methods,
        target=target,
        calibration_rows=calibration_rows,
        held_out_direct=held_out_level2,
        file_hashes=file_hashes,
        split_manifests=split_manifests,
    )
    write_json(output_dir / "config_used.json", payload)
    if methods != (METHOD_BASE,) and not args.mcf_path.is_file():
        raise FileNotFoundError(f"Missing local MCF retain corpus: {args.mcf_path}")
    if METHOD_ZERO in methods and not args.zero_hparams.is_file():
        raise FileNotFoundError(
            f"Missing original ZeroUnlearn hparams: {args.zero_hparams}"
        )
    if METHOD_ZERO in methods and not args.zero_root.is_dir():
        raise FileNotFoundError(
            f"Missing vendored ZeroUnlearn directory: {args.zero_root}"
        )
    if METHOD_ZERO in methods:
        payload["zero_unlearn"] = zero_unlearn_identity(args)
    else:
        payload["zero_unlearn"] = None
    if args.skip_ppl:
        payload["wikidata_corpus"] = None
    else:
        wikidata_text = load_wikidata_text(args.wikidata_dir)
        if not wikidata_text:
            raise FileNotFoundError(
                "A readable local Wikidata corpus is required unless "
                "--skip-ppl is used: "
                f"{args.wikidata_dir}"
            )
        payload["wikidata_corpus"] = {
            "path": str(Path(args.wikidata_dir).resolve()),
            "directory_sha256": directory_sha256(args.wikidata_dir),
            "loaded_text_sha256": hashlib.sha256(
                wikidata_text.encode("utf-8")
            ).hexdigest(),
        }
    retain_records: List[Dict[str, Any]] = []
    retain_examples: List[gagd.Example] = []
    protected_examples: List[gagd.Example] = []
    if methods != (METHOD_BASE,):
        all_retain_records, all_retain_examples = load_mcf_retain(
            args.mcf_path,
            seed=args.seed,
            retain_num=args.retain_num + args.repair_retain_num,
        )
        retain_records = all_retain_records[: args.retain_num]
        retain_examples = all_retain_examples[: args.retain_num]
        protected_examples = all_retain_examples[args.retain_num :]
        if len(protected_examples) != args.repair_retain_num:
            raise RuntimeError("MCF checkpoint-gate partition has the wrong size")
        payload["mcf_retain_provenance"] = {
            "file_sha256": file_sha256(args.mcf_path),
            "optimization_record_sha256": [
                mapping_sha256(record) for record in retain_records
            ],
            "checkpoint_gate_record_sha256": [
                mapping_sha256(record)
                for record in all_retain_records[args.retain_num :]
            ],
            "optimization_example_sha256": [
                example_content_sha256(example)
                for example in retain_examples
            ],
            "checkpoint_gate_example_sha256": [
                example_content_sha256(example)
                for example in protected_examples
            ],
            "partitions_disjoint": not (
                {
                    mapping_sha256(record) for record in retain_records
                }
                & {
                    mapping_sha256(record)
                    for record in all_retain_records[args.retain_num :]
                }
            ),
        }
        if not payload["mcf_retain_provenance"]["partitions_disjoint"]:
            raise RuntimeError(
                "MCF optimization and checkpoint-gate partitions overlap"
            )
        optimization_example_hashes = set(
            payload["mcf_retain_provenance"][
                "optimization_example_sha256"
            ]
        )
        checkpoint_example_hashes = set(
            payload["mcf_retain_provenance"][
                "checkpoint_gate_example_sha256"
            ]
        )
        if optimization_example_hashes & checkpoint_example_hashes:
            raise RuntimeError(
                "MCF optimization and checkpoint-gate examples overlap by "
                "prompt/answer content"
            )
        payload["mcf_retain_provenance"][
            "example_content_partitions_disjoint"
        ] = True
    matched_positive_rows: List[Dict[str, Any]] = []
    matched_positive_file_hashes: Dict[str, str] = {}
    if METHOD_REPRESENTATION in methods:
        (
            matched_positive_rows,
            matched_positive_file_hashes,
        ) = load_matched_positive_training_rows(
            args.data_root,
            seed=args.seed,
            allow_download=not args.no_download,
        )
        payload["matched_positive_file_sha256_by_target"] = (
            matched_positive_file_hashes
        )
    write_json(output_dir / "config_used.json", payload)
    if args.dry_run:
        print(
            f"RWKU dry run validated seed {args.seed}: {target.subject}; "
            f"calibration={len(calibration_rows)}, "
            f"held-out direct={len(held_out_level2)}, "
            f"MCF optimization/gate={len(retain_examples)}/"
            f"{len(protected_examples)}, "
            f"matched positives={len(matched_positive_rows)}; "
            f"output={output_dir}"
        )
        return
    # Invalidate any stale successful artifact before the expensive run.
    write_json(
        output_dir / "results.json",
        {
            **payload,
            "status": "running",
            "method_order": list(METHOD_ORDER),
            "methods_run": [],
            "results": {},
        },
    )
    dtype = dtype_from_name(args.dtype)
    set_all_seeds(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    results: Dict[str, Any] = {}
    method_provenance: Dict[str, Any] = {}

    print(f"Loading base model for RWKU seed {args.seed} ({target.subject})")
    base_model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    all_answers = [
        str(row["answer"])
        for filename in (
            "forget_level1.json",
            "forget_level2.json",
            "forget_level3.json",
        )
        for row in datasets[filename]
    ]
    frozen_probe = build_frozen_head_probe(
        base_model,
        tokenizer,
        held_out_level2,
        additional_answers=all_answers,
    )
    print("Evaluating base model and capturing retain-reference probabilities")
    base_result = evaluate_method(
        method=METHOD_BASE,
        model=base_model,
        tokenizer=tokenizer,
        target_subject=target.subject,
        held_out_cloze=held_out_level1,
        held_out_direct=held_out_level2,
        datasets=datasets,
        args=args,
        base_retain_mean_logprobs=None,
        frozen_probe=frozen_probe,
    )
    results[METHOD_BASE] = base_result
    write_json(output_dir / "base_model.json", base_result)
    base_retain = base_result["retain_reference_mean_logprobs"]

    if METHOD_REPAIR_ONLY in methods:
        print("Running repair-only control from the untouched base model")
        repair_only_report = run_protected_lm_head_repair(
            base_model,
            tokenizer,
            calibration_rows=calibration_rows,
            protected_examples=protected_examples,
            config=repair_config(args),
            output_dir=output_dir / "repair_only",
        )
        repair_only_result = evaluate_method(
            method=METHOD_REPAIR_ONLY,
            model=base_model,
            tokenizer=tokenizer,
            target_subject=target.subject,
            held_out_cloze=held_out_level1,
            held_out_direct=held_out_level2,
            datasets=datasets,
            args=args,
            base_retain_mean_logprobs=base_retain,
            frozen_probe=frozen_probe,
        )
        repair_only_result["repair"] = repair_only_report
        results[METHOD_REPAIR_ONLY] = repair_only_result
        write_json(output_dir / "repair_only.json", repair_only_result)
        if args.save_checkpoints:
            save_checkpoint(
                base_model,
                tokenizer,
                output_dir / "repair_only" / "checkpoint",
            )
    release_model(base_model)
    base_model = None

    if METHOD_ZERO in methods:
        print("Applying the vendored original ZeroUnlearn implementation")
        zero_model, zero_tokenizer = load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            for_training=False,
            gradient_checkpointing=False,
        )
        zero_artifact = run_original_zero(
            args=args,
            tokenizer=zero_tokenizer,
            model=zero_model,
            calibration_rows=calibration_rows,
            target_subject=target.subject,
            retain_records=retain_records,
        )
        zero_model = zero_artifact.pop("model")
        zero_result = evaluate_method(
            method=METHOD_ZERO,
            model=zero_model,
            tokenizer=zero_tokenizer,
            target_subject=target.subject,
            held_out_cloze=held_out_level1,
            held_out_direct=held_out_level2,
            datasets=datasets,
            args=args,
            base_retain_mean_logprobs=base_retain,
            frozen_probe=frozen_probe,
        )
        zero_result["unlearning"] = zero_artifact["provenance"]
        results[METHOD_ZERO] = zero_result
        method_provenance[METHOD_ZERO] = zero_artifact["provenance"]
        write_json(output_dir / "original_zerounlearn.json", zero_result)
        if args.save_checkpoints:
            save_checkpoint(
                zero_model,
                zero_tokenizer,
                output_dir / "original_zerounlearn" / "checkpoint",
            )
        release_model(zero_model)
        del zero_model, zero_tokenizer, zero_artifact
        gc.collect()
        torch.cuda.empty_cache()

    if METHOD_SETTING5 in methods or METHOD_REPAIRED in methods:
        print("Training Setting 5e on calibration probes and unrelated retain facts")
        set_all_seeds(args.seed)
        setting5_model, setting5_tokenizer = load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            for_training=True,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        forget_examples = setting5_examples(
            setting5_tokenizer,
            calibration_rows,
        )
        requested_save = args.save_model
        args.save_model = False
        training_started = time.perf_counter()
        train_summary = gagd.train_mode(
            setting5_model,
            setting5_tokenizer,
            forget_examples,
            retain_examples,
            selected_ids=[],
            mode=SETTING5_MODE,
            args=args,
            mode_dir=output_dir / "setting5_training",
        )
        args.save_model = requested_save
        training_provenance = {
            "trainable": asdict(train_summary),
            "training_seconds": time.perf_counter() - training_started,
            "calibration_example_count": len(forget_examples),
            "retain_example_count": len(retain_examples),
        }
        prepare_model_for_evaluation(setting5_model)
        method_provenance[METHOD_SETTING5] = training_provenance
        if METHOD_SETTING5 in methods:
            setting5_result = evaluate_method(
                method=METHOD_SETTING5,
                model=setting5_model,
                tokenizer=setting5_tokenizer,
                target_subject=target.subject,
                held_out_cloze=held_out_level1,
                held_out_direct=held_out_level2,
                datasets=datasets,
                args=args,
                base_retain_mean_logprobs=base_retain,
                frozen_probe=frozen_probe,
            )
            setting5_result["unlearning"] = training_provenance
            results[METHOD_SETTING5] = setting5_result
            write_json(
                output_dir / "setting5_without_repair.json",
                setting5_result,
            )
        if args.save_checkpoints:
            save_checkpoint(
                setting5_model,
                setting5_tokenizer,
                output_dir / "setting5_training" / "checkpoint",
            )

        if METHOD_REPAIRED in methods:
            print(
                "Applying protected sparse active-pair LM-head repair "
                "to Setting 5e"
            )
            repaired_report = run_protected_lm_head_repair(
                setting5_model,
                setting5_tokenizer,
                calibration_rows=calibration_rows,
                protected_examples=protected_examples,
                config=repair_config(args),
                output_dir=output_dir / "setting5_repaired",
            )
            repaired_result = evaluate_method(
                method=METHOD_REPAIRED,
                model=setting5_model,
                tokenizer=setting5_tokenizer,
                target_subject=target.subject,
                held_out_cloze=held_out_level1,
                held_out_direct=held_out_level2,
                datasets=datasets,
                args=args,
                base_retain_mean_logprobs=base_retain,
                frozen_probe=frozen_probe,
            )
            repaired_result["unlearning"] = training_provenance
            repaired_result["repair"] = repaired_report
            results[METHOD_REPAIRED] = repaired_result
            method_provenance[METHOD_REPAIRED] = {
                "training": training_provenance,
                "repair": repaired_report,
            }
            write_json(
                output_dir / "setting5_protected_repair.json",
                repaired_result,
            )
            if args.save_checkpoints:
                save_checkpoint(
                    setting5_model,
                    setting5_tokenizer,
                    output_dir / "setting5_repaired" / "checkpoint",
                )
        release_model(setting5_model)
        del setting5_model, setting5_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    if METHOD_REPRESENTATION in methods:
        print(
            "Training Base-initialized protected representation unlearning "
            "with disjoint external retain gates"
        )
        set_all_seeds(args.seed)
        representation_model, representation_tokenizer = (
            load_model_and_tokenizer(
                args.model_path,
                dtype=dtype,
                for_training=True,
                gradient_checkpointing=args.gradient_checkpointing,
            )
        )
        representation_started = time.perf_counter()
        representation_report = run_representation_unlearning(
            representation_model,
            representation_tokenizer,
            calibration_rows=calibration_rows,
            retain_examples=retain_examples,
            protected_examples=protected_examples,
            positive_rows=datasets["positive.json"],
            matched_positive_rows=matched_positive_rows,
            config=representation_config(args),
            output_dir=output_dir / "protected_representation",
        )
        prepare_model_for_evaluation(representation_model)
        representation_report["training_seconds"] = (
            time.perf_counter() - representation_started
        )
        representation_report["data"][
            "matched_positive_file_sha256_by_target"
        ] = matched_positive_file_hashes
        representation_result = evaluate_method(
            method=METHOD_REPRESENTATION,
            model=representation_model,
            tokenizer=representation_tokenizer,
            target_subject=target.subject,
            held_out_cloze=held_out_level1,
            held_out_direct=held_out_level2,
            datasets=datasets,
            args=args,
            base_retain_mean_logprobs=base_retain,
            frozen_probe=frozen_probe,
        )
        representation_result["unlearning"] = representation_report
        results[METHOD_REPRESENTATION] = representation_result
        method_provenance[METHOD_REPRESENTATION] = representation_report
        write_json(
            output_dir / "protected_representation.json",
            representation_result,
        )
        if args.save_checkpoints:
            save_checkpoint(
                representation_model,
                representation_tokenizer,
                output_dir / "protected_representation" / "checkpoint",
            )
        release_model(representation_model)
        del representation_model, representation_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    combined = {
        **payload,
        "status": "complete",
        "method_order": list(METHOD_ORDER),
        "methods_run": [
            method for method in METHOD_ORDER if method in results
        ],
        "results": results,
        "method_provenance": method_provenance,
    }
    write_json(output_dir / "results.json", combined)
    print(f"Completed RWKU seed {args.seed}; results: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
