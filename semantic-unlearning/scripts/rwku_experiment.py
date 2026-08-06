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
    evaluate_qa_rows,
    evaluate_rwku,
    format_qa_prompt,
    load_wikidata_text,
    rwku_success_contract,
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
from build_rwku_entity_facts import (
    build_probe_artifacts,
    export_mcf_shaped_training_requests,
    official_locked_descriptor,
)
from build_rwku_matched_protection import (
    FORBIDDEN_PATH_MARKERS as PROTECTION_FORBIDDEN_PATH_MARKERS,
    validate_key_provenance,
)
from rwku_artifact_access import (
    LEGACY_PROTOCOL_STATUS,
    PROBE_PROTOCOL_LABEL,
    PROBE_PROTOCOL_STATUS,
    TARGET_ONLY_PROTOCOL_LABEL,
    TARGET_ONLY_PROTOCOL_STATUS,
    make_artifact,
    read_artifact,
    sha256_file as artifact_file_sha256,
    write_artifact,
)
from rwku_checkpoint_receipt import (
    assert_model_modification_allowed,
    create_checkpoint_receipt,
    load_receipt,
    mark_evaluation_complete,
    open_official_evaluation,
)
from rwku_fact_sampler import (
    build_fact_cycle_plan,
    exposure_report,
    plan_sha256,
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
METHOD_REPRESENTATION = "Protected representation unlearning v2"
METHOD_ORDER = (
    METHOD_BASE,
    METHOD_ZERO,
    METHOD_SETTING5,
    METHOD_REPAIRED,
    METHOD_REPAIR_ONLY,
    METHOD_REPRESENTATION,
)

TRAINING_SOURCE_PROBE = "probe_assisted_entity_fact"
TRAINING_SOURCE_TARGET_ONLY = "target_only_generated_entity_corpus"


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
    parser.add_argument(
        "--stage",
        choices=("prepare", "train", "evaluate", "all"),
        default="all",
        help="Explicit protocol stage; omitted preserves the legacy RWKU command.",
    )
    parser.add_argument(
        "--training-source",
        choices=(TRAINING_SOURCE_PROBE, TRAINING_SOURCE_TARGET_ONLY),
        default=None,
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-revision", default="local_pinned_snapshot")
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
    parser.add_argument("--entity-fact-bundle", type=Path)
    parser.add_argument("--generated-entity-fact-bundle", type=Path)
    parser.add_argument("--generator-receipt", type=Path)
    parser.add_argument("--matched-protection-train", type=Path)
    parser.add_argument("--matched-protection-gate", type=Path)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--fact-overrides", type=Path)
    parser.add_argument("--fact-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--prompt-holdout-per-seen-fact", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--legacy-row-split",
        action="store_true",
        help="Use the historical independent row-hash split (prompt-held-out only).",
    )
    parser.add_argument(
        "--strict-fact-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-relation-conditioned-reverse-prompts",
        action="store_true",
        help="Boundary-expanding ablation; disabled in the primary result.",
    )
    parser.add_argument("--export-mcf-shaped-training-requests", type=Path)
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
    parser.add_argument("--representation-steps", type=int, default=1800)
    parser.add_argument("--representation-lr", type=float, default=2e-4)
    parser.add_argument("--representation-rank", type=int, default=24)
    parser.add_argument("--representation-alpha", type=float, default=48.0)
    parser.add_argument("--representation-last-n-layers", type=int, default=12)
    parser.add_argument("--representation-max-length", type=int, default=512)
    parser.add_argument("--representation-retain-top-k", type=int, default=64)
    parser.add_argument(
        "--representation-retain-examples-per-step",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--representation-qa-tasks-per-step",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--representation-positive-tasks-per-phase",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--representation-constraint-polish-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--representation-constraint-polish-limit",
        type=int,
        default=96,
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
        "--representation-frozen-head-demotion-margin",
        type=float,
        default=1.0,
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
        default=2.0,
    )
    parser.add_argument(
        "--representation-frozen-head-weight",
        type=float,
        default=2.0,
    )
    parser.add_argument("--representation-mc-weight", type=float, default=1.0)
    parser.add_argument(
        "--representation-positive-proxy-weight",
        type=float,
        default=1.0,
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
        "--representation-layerwise-concept-erasure-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--representation-layerwise-concept-orthogonal-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--representation-concept-anchor-layer-stride",
        type=int,
        default=2,
    )
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
        "--representation-min-frozen-head-normalized-rank",
        type=float,
        default=0.90,
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
        default=192,
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
    if args.training_source is not None and not args.experiment_id:
        raise ValueError("Staged RWKU tracks require --experiment-id")
    if args.legacy_row_split and args.training_source is not None:
        raise ValueError(
            "--legacy-row-split cannot be combined with an entity-fact training source"
        )
    if args.confirmatory and args.training_source != TRAINING_SOURCE_TARGET_ONLY:
        raise ValueError(
            "--confirmatory is reserved for target-only generated-corpus runs"
        )
    if (
        args.confirmatory
        and args.training_source == TRAINING_SOURCE_TARGET_ONLY
        and args.stage == "all"
    ):
        raise ValueError(
            "Confirmatory target-only runs reject --stage all; train and evaluate "
            "must be separate processes"
        )
    if args.confirmatory and args.stage == "evaluate":
        if args.skip_ppl:
            raise ValueError("Confirmatory evaluation cannot use --skip-ppl")
        bounded = [
            name
            for name in (
                "forget_eval_limit",
                "adversarial_eval_limit",
                "mia_eval_limit",
                "neighbor_eval_limit",
                "utility_eval_limit",
            )
            if getattr(args, name) is not None
        ]
        if bounded:
            raise ValueError(
                "Confirmatory evaluation cannot use bounded smoke limits: "
                + ", ".join(bounded)
            )
    if not 0.0 <= args.fact_holdout_fraction <= 1.0:
        raise ValueError("--fact-holdout-fraction must be in [0,1]")
    if args.prompt_holdout_per_seen_fact <= 0:
        raise ValueError("--prompt-holdout-per-seen-fact must be positive")
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
        "representation_qa_tasks_per_step",
        "representation_positive_tasks_per_phase",
        "representation_constraint_polish_limit",
        "representation_external_mc_retain_limit",
        "representation_external_mc_gate_limit",
        "representation_positive_max_rows",
        "representation_positive_subject_task_max_rows",
        "representation_matched_positive_max_rows",
        "representation_positive_tokens_per_row",
        "representation_concept_rank",
        "representation_concept_anchor_layer_stride",
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
    if args.training_source is not None and args.stage == "prepare":
        return
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
        SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
        SEMANTIC_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
        SEMANTIC_ROOT / "scripts" / "rwku_fact_sampler.py",
        SEMANTIC_ROOT / "scripts" / "build_rwku_entity_facts.py",
        SEMANTIC_ROOT / "scripts" / "build_rwku_generated_entity_corpus.py",
        SEMANTIC_ROOT / "scripts" / "build_rwku_matched_protection.py",
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
        qa_tasks_per_step=args.representation_qa_tasks_per_step,
        positive_tasks_per_phase=(
            args.representation_positive_tasks_per_phase
        ),
        constraint_polish_fraction=(
            args.representation_constraint_polish_fraction
        ),
        constraint_polish_limit=(
            args.representation_constraint_polish_limit
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
        frozen_head_demotion_margin=(
            args.representation_frozen_head_demotion_margin
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
        layerwise_concept_erasure_weight=(
            args.representation_layerwise_concept_erasure_weight
        ),
        layerwise_concept_orthogonal_weight=(
            args.representation_layerwise_concept_orthogonal_weight
        ),
        concept_anchor_layer_stride=(
            args.representation_concept_anchor_layer_stride
        ),
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
        min_calibration_frozen_head_normalized_rank=(
            args.representation_min_frozen_head_normalized_rank
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


def _staged_output_dir(args: argparse.Namespace) -> Path:
    if not args.experiment_id:
        raise ValueError("Staged RWKU protocol requires --experiment-id")
    return Path(args.output_root) / args.experiment_id


def _state_path(args: argparse.Namespace) -> Path:
    return _staged_output_dir(args) / "experiment_state.json"


def _read_state(args: argparse.Namespace) -> Dict[str, Any]:
    path = _state_path(args)
    if not path.is_file():
        raise ValueError(
            f"Experiment {args.experiment_id!r} is not PREPARED; missing {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("experiment_id") != args.experiment_id:
        raise ValueError("Experiment state ID mismatch")
    return value


def _write_state(args: argparse.Namespace, state: str, **extra: Any) -> None:
    existing: Dict[str, Any] = {}
    if _state_path(args).is_file():
        existing = _read_state(args)
    order = {
        "PREPARED": 0,
        "TRAINING": 1,
        "CHECKPOINT_FROZEN": 2,
        "OFFICIAL_EVALUATION_OPENED": 3,
        "EVALUATION_COMPLETE": 4,
    }
    old = existing.get("state")
    if old is not None and order[state] < order[old]:
        raise ValueError(f"Backward RWKU state transition is forbidden: {old} -> {state}")
    write_json(
        _state_path(args),
        {
            **existing,
            "schema_version": "rwku_experiment_state_v1",
            "experiment_id": args.experiment_id,
            "training_source": args.training_source,
            "state": state,
            **extra,
        },
    )


def _default_fact_overrides(args: argparse.Namespace) -> Path:
    return (
        Path(args.fact_overrides)
        if args.fact_overrides is not None
        else SEMANTIC_ROOT / "config" / "rwku" / "fact_overrides" / f"seed{args.seed}.json"
    )


def _prepare_staged(args: argparse.Namespace) -> None:
    output_dir = _staged_output_dir(args)
    if _state_path(args).is_file():
        current = _read_state(args)
        if current.get("state") not in {"PREPARED"}:
            raise ValueError(
                "Preparation cannot replace an experiment that has entered training"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = target_for_seed(args.seed)
    if args.training_source == TRAINING_SOURCE_PROBE:
        override_path = _default_fact_overrides(args)
        if not override_path.is_file():
            raise FileNotFoundError(
                f"Probe-assisted strict fact assignment requires: {override_path}"
            )
        result = build_probe_artifacts(
            data_root=args.data_root,
            seed=args.seed,
            output_dir=output_dir,
            fact_overrides_path=override_path,
            unseen_fact_fraction=args.fact_holdout_fraction,
            prompt_holdout_per_seen_fact=args.prompt_holdout_per_seen_fact,
            split_seed=args.split_seed,
            strict=args.strict_fact_audit,
            allow_download=not args.no_download,
        )
        if args.export_mcf_shaped_training_requests:
            export_mcf_shaped_training_requests(
                result["artifacts"]["training_bundle.json"],
                destination=args.export_mcf_shaped_training_requests,
            )
        protocol_label = PROBE_PROTOCOL_LABEL
        protocol_status = PROBE_PROTOCOL_STATUS
        audit = result["audit"]
        prepared_training_bundle_path = output_dir / "training_bundle.json"
        prepared_training_bundle_sha256 = artifact_file_sha256(
            prepared_training_bundle_path
        )
        prepared_generator_receipt_sha256 = None
    else:
        if args.generated_entity_fact_bundle is None:
            raise ValueError(
                "Target-only preparation requires --generated-entity-fact-bundle; "
                "build it independently before this stage"
            )
        if args.generator_receipt is None:
            raise ValueError("Target-only preparation requires --generator-receipt")
        # Do not parse the training bundle here: its immutable role permits it
        # only during train. File identities are recorded without opening any
        # official RWKU evaluation record.
        if not args.generated_entity_fact_bundle.is_file():
            raise FileNotFoundError(args.generated_entity_fact_bundle)
        if not args.generator_receipt.is_file():
            raise FileNotFoundError(args.generator_receipt)
        metadata = {
            "seed": args.seed,
            "entity_id": f"rwku:{target.directory}",
            "subject": target.subject,
            "generated_training_bundle_path": str(args.generated_entity_fact_bundle.resolve()),
            "generated_training_bundle_file_sha256": artifact_file_sha256(args.generated_entity_fact_bundle),
            "generator_receipt_path": str(args.generator_receipt.resolve()),
            "generator_receipt_file_sha256": artifact_file_sha256(args.generator_receipt),
        }
        locked = make_artifact(
            "official_locked_eval",
            official_locked_descriptor(args.seed, include_level12=True),
            protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
            protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
            metadata=metadata,
        )
        write_artifact(output_dir / "official_locked_eval.json", locked)
        protocol_label = TARGET_ONLY_PROTOCOL_LABEL
        protocol_status = TARGET_ONLY_PROTOCOL_STATUS
        audit = {
            "official_level1_level2_opened": False,
            "generated_training_bundle_file_sha256": metadata[
                "generated_training_bundle_file_sha256"
            ],
            "generator_receipt_file_sha256": metadata[
                "generator_receipt_file_sha256"
            ],
        }
        prepared_training_bundle_path = args.generated_entity_fact_bundle
        prepared_training_bundle_sha256 = metadata[
            "generated_training_bundle_file_sha256"
        ]
        prepared_generator_receipt_sha256 = metadata[
            "generator_receipt_file_sha256"
        ]
    _write_state(
        args,
        "PREPARED",
        protocol_label=protocol_label,
        protocol_status=protocol_status,
        confirmatory=bool(args.confirmatory),
        target={"seed": args.seed, "directory": target.directory, "subject": target.subject},
        prepare_audit=audit,
        prepared_training_bundle_path=str(
            Path(prepared_training_bundle_path).resolve()
        ),
        prepared_training_bundle_file_sha256=prepared_training_bundle_sha256,
        prepared_generator_receipt_file_sha256=prepared_generator_receipt_sha256,
        official_evaluation_opened=False,
    )
    print(
        f"Prepared {args.experiment_id}: {protocol_label}; "
        f"official evaluation remains locked; output={output_dir}"
    )


def setting5_entity_fact_examples(
    tokenizer: Any,
    views: Sequence[Mapping[str, Any]],
    *,
    include_reverse: bool,
) -> Tuple[List[gagd.Example], Dict[str, gagd.Example]]:
    """Compile method-visible fact views without changing Setting 5e direction."""

    if not tokenizer.eos_token:
        raise ValueError("Setting 5e requires tokenizer.eos_token")
    examples: List[gagd.Example] = []
    by_view: Dict[str, gagd.Example] = {}
    for view in views:
        if view.get("training_allowed") is not True:
            raise ValueError("Entity-fact compiler received a non-training view")
        boundary_expanding = bool(view.get("boundary_expanding", False))
        if boundary_expanding and not include_reverse:
            raise ValueError(
                "Boundary-expanding reverse view is disabled; pass the explicit "
                "relation-conditioned reverse ablation flag"
            )
        if boundary_expanding and view.get("prompt_style") != "relation-conditioned reverse":
            raise ValueError("Reverse views must be relation-conditioned and explicitly labeled")
        sensitive = gagd.normalize_answer(
            str(view.get("sensitive_answer_alias") or view["canonical_sensitive_answer"])
        )
        row = {
            "query": str(view["query"]),
            "answer": sensitive,
            "subject": str(view["subject"]),
            "level": str(view.get("level", "generated")),
            "type": str(view.get("query_type", view.get("prompt_style", ""))),
        }
        example = gagd.Example(
            prompt=format_qa_prompt(tokenizer, row),
            answer=sensitive,
            subject=str(view["subject"]),
            target_new=sensitive,
            target_true=str(tokenizer.eos_token),
            paraphrase_prompts=[],
            source=str(view["fact_id"]),
        )
        view_id = str(view["view_id"])
        if view_id in by_view:
            raise ValueError(f"Duplicate entity-fact view ID: {view_id}")
        by_view[view_id] = example
        examples.append(example)
    return examples, by_view


def _protection_examples(
    artifact: Mapping[str, Any],
    tokenizer: Any,
) -> List[gagd.Example]:
    examples: List[gagd.Example] = []
    for wrapped in artifact["payload"].get("records", []):
        row = wrapped.get("record", wrapped)
        prompt = row.get("prompt") or row.get("query") or row.get("text")
        answer = row.get("answer") or row.get("target_true") or row.get("target")
        if isinstance(answer, Mapping):
            answer = answer.get("str")
        if not prompt or not answer:
            raise ValueError(
                "Matched-protection rows require a prompt/query/text and answer"
            )
        normalized = gagd.normalize_answer(str(answer))
        examples.append(
            gagd.Example(
                prompt=str(prompt),
                answer=normalized,
                subject=str(row.get("subject", "independent protection")),
                target_new=normalized,
                target_true=str(tokenizer.eos_token),
                paraphrase_prompts=[],
                source=f"matched_protection:{wrapped.get('content_sha256', '')}",
            )
        )
    return examples


def _validate_training_bundle_sources(
    training: Mapping[str, Any],
    *,
    training_source: str,
) -> None:
    allowed_probe_files = {"forget_level1.json", "forget_level2.json"}
    for view in training["payload"].get("views", []):
        if view.get("training_allowed") is not True:
            raise ValueError("Training bundle contains a non-training view")
        source_file = str(view.get("source_file", ""))
        if training_source == TRAINING_SOURCE_PROBE:
            if source_file not in allowed_probe_files:
                raise ValueError(
                    f"Probe-assisted training has forbidden source: {source_file!r}"
                )
        elif source_file != "generated_raw_corpus.json" or str(
            view.get("level", "")
        ) != "generated":
            raise ValueError(
                "Target-only training accepts only independently generated corpus views"
            )


def _validate_matched_protection_artifact(
    artifact: Mapping[str, Any],
    *,
    target_subject: str,
) -> None:
    keys = artifact["payload"].get("keys")
    if not isinstance(keys, list) or not keys:
        raise ValueError("Matched-protection artifact requires provenance keys")
    for key in keys:
        validate_key_provenance(key)
    normalized_subject = " ".join(target_subject.casefold().split())
    for wrapped in artifact["payload"].get("records", []):
        source_path = str(wrapped.get("source_path", "")).casefold()
        if any(marker in source_path for marker in PROTECTION_FORBIDDEN_PATH_MARKERS):
            raise ValueError("Matched protection points to an RWKU evaluation source")
        content = " ".join(
            str(wrapped.get(field, ""))
            for field in ("content", "normalized_content")
        ).casefold()
        if normalized_subject and normalized_subject in content:
            raise ValueError(
                "Matched protection must be target-independent, not target-entity data"
            )


def _write_mcf_partition_manifests(
    output_dir: Path,
    retain_records: Sequence[Mapping[str, Any]],
    gate_records: Sequence[Mapping[str, Any]],
) -> Tuple[Path, Path]:
    optimization = output_dir / "mcf_retain_optimization_manifest.json"
    gate = output_dir / "mcf_repair_gate_manifest.json"
    write_json(
        optimization,
        {
            "role": "optimization_protection",
            "record_sha256": [mapping_sha256(row) for row in retain_records],
        },
    )
    write_json(
        gate,
        {
            "role": "repair_selection_gate",
            "gradient_allowed": False,
            "record_sha256": [mapping_sha256(row) for row in gate_records],
        },
    )
    return optimization, gate


def _train_staged(args: argparse.Namespace) -> None:
    state = _read_state(args)
    existing_receipt = (
        args.checkpoint_receipt
        or _staged_output_dir(args) / "checkpoint_receipt.json"
    )
    if Path(existing_receipt).is_file():
        assert_model_modification_allowed(
            Path(existing_receipt), experiment_id=args.experiment_id
        )
    if state.get("state") != "PREPARED":
        raise ValueError(f"Training requires PREPARED, got {state.get('state')}")
    if args.batch_size != 1:
        raise ValueError("Entity-fact balanced cycles require --batch-size 1")
    output_dir = _staged_output_dir(args)
    if args.training_source == TRAINING_SOURCE_PROBE:
        bundle_path = args.entity_fact_bundle or output_dir / "training_bundle.json"
        generator_receipt_path = None
    else:
        if args.generated_entity_fact_bundle is None:
            raise ValueError("Target-only training requires --generated-entity-fact-bundle")
        bundle_path = args.generated_entity_fact_bundle
        generator_receipt_path = args.generator_receipt
        if generator_receipt_path is None:
            raise ValueError("Target-only training requires --generator-receipt")
    training = read_artifact(
        bundle_path,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    if (
        artifact_file_sha256(Path(bundle_path))
        != state.get("prepared_training_bundle_file_sha256")
    ):
        raise ValueError("Training bundle changed after PREPARED state")
    if generator_receipt_path is not None and (
        artifact_file_sha256(Path(generator_receipt_path))
        != state.get("prepared_generator_receipt_file_sha256")
    ):
        raise ValueError("Generator receipt changed after PREPARED state")
    if generator_receipt_path is not None:
        generator_receipt_artifact = read_artifact(
            Path(generator_receipt_path),
            stage="train",
            expected_role="generator_receipt",
        )
        generator_payload = generator_receipt_artifact["payload"]
        if generator_payload.get("status") != "complete":
            raise ValueError("Target-only training requires a complete generator receipt")
        if (
            generator_payload.get("final_entity_fact_bundle_sha256")
            != training["sha256"]
        ):
            raise ValueError(
                "Generator receipt does not identify the generated training bundle"
            )
    expected_label = (
        PROBE_PROTOCOL_LABEL
        if args.training_source == TRAINING_SOURCE_PROBE
        else TARGET_ONLY_PROTOCOL_LABEL
    )
    if training["protocol_label"] != expected_label:
        raise ValueError("Training bundle belongs to a different RWKU protocol")
    _validate_training_bundle_sources(
        training,
        training_source=args.training_source,
    )
    if args.matched_protection_train is None or args.matched_protection_gate is None:
        raise ValueError(
            "Staged Setting 5e requires disjoint --matched-protection-train and "
            "--matched-protection-gate artifacts"
        )
    matched_train = read_artifact(
        args.matched_protection_train,
        stage="train",
        gradient=True,
        expected_role="optimization_protection",
    )
    matched_gate = read_artifact(
        args.matched_protection_gate,
        stage="train",
        selection=True,
        expected_role="repair_selection_gate",
    )
    target = target_for_seed(args.seed)
    for artifact in (matched_train, matched_gate):
        if artifact["protocol_label"] != expected_label:
            raise ValueError("Matched-protection artifact belongs to another protocol")
        _validate_matched_protection_artifact(
            artifact,
            target_subject=target.subject,
        )
    matched_train_hashes = {
        str(row.get("content_sha256"))
        for row in matched_train["payload"].get("records", [])
    }
    matched_gate_hashes = {
        str(row.get("content_sha256"))
        for row in matched_gate["payload"].get("records", [])
    }
    overlap = (matched_train_hashes & matched_gate_hashes) - {"None"}
    if overlap:
        raise ValueError("Matched-protection train/gate content hashes overlap")
    _write_state(args, "TRAINING", training_bundle=str(Path(bundle_path).resolve()))
    dtype = dtype_from_name(args.dtype)
    set_all_seeds(args.seed)
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    views = list(training["payload"].get("views", []))
    forget_examples, examples_by_view = setting5_entity_fact_examples(
        tokenizer,
        views,
        include_reverse=args.include_relation_conditioned_reverse_prompts,
    )
    views_by_fact: Dict[str, List[Mapping[str, Any]]] = {}
    for view in views:
        views_by_fact.setdefault(str(view["fact_id"]), []).append(view)
    plan = build_fact_cycle_plan(views_by_fact, steps=args.steps, seed=args.seed)
    args.precomputed_forget_batches = [
        [examples_by_view[str(entry["view_id"])]] for entry in plan
    ]
    exposures = exposure_report(plan, seed=args.seed, tokenizer=tokenizer)
    exposures["plan_sha256"] = plan_sha256(plan)
    write_json(output_dir / "fact_exposure_report.json", exposures)

    all_mcf_records, all_mcf_examples = load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.retain_num + args.repair_retain_num,
    )
    retain_records = all_mcf_records[: args.retain_num]
    gate_records = all_mcf_records[args.retain_num :]
    retain_examples = all_mcf_examples[: args.retain_num]
    mcf_gate_examples = all_mcf_examples[args.retain_num :]
    optimization_matched = _protection_examples(matched_train, tokenizer)
    selection_matched = _protection_examples(matched_gate, tokenizer)
    retain_examples = [*retain_examples, *optimization_matched]
    protected_gate_examples = [*mcf_gate_examples, *selection_matched]
    optimization_protection_examples = [
        *all_mcf_examples[: min(len(all_mcf_examples), args.repair_retain_num)],
        *optimization_matched,
    ]
    mcf_optimization_manifest, mcf_gate_manifest = _write_mcf_partition_manifests(
        output_dir, retain_records, gate_records
    )

    requested_save = args.save_model
    args.save_model = False
    started = time.perf_counter()
    train_summary = gagd.train_mode(
        model,
        tokenizer,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=SETTING5_MODE,
        args=args,
        mode_dir=output_dir / "setting5_training",
    )
    args.save_model = requested_save
    prepare_model_for_evaluation(model)
    setting5_checkpoint = output_dir / "setting5_training" / "checkpoint"
    save_checkpoint(model, tokenizer, setting5_checkpoint)
    repair_report = run_protected_lm_head_repair(
        model,
        tokenizer,
        calibration_rows=[
            {
                "query": view["query"],
                "answer": view.get("sensitive_answer_alias") or view["canonical_sensitive_answer"],
                "subject": view["subject"],
                "fact_id": view["fact_id"],
                "view_id": view["view_id"],
                "prompt_style": view.get("prompt_style"),
                "boundary_expanding": bool(view.get("boundary_expanding", False)),
            }
            for view in views
        ],
        protected_examples=protected_gate_examples,
        optimization_protection_examples=optimization_protection_examples,
        config=repair_config(args),
        output_dir=output_dir / "setting5_repaired",
    )
    repaired_checkpoint = output_dir / "setting5_repaired" / "checkpoint"
    save_checkpoint(model, tokenizer, repaired_checkpoint)
    training_report = {
        "protocol_label": expected_label,
        "protocol_status": training["protocol_status"],
        "method": METHOD_REPAIRED,
        "representation_method_merged": False,
        "setting5": {
            "trainable": asdict(train_summary),
            "steps": args.steps,
            "training_seconds": time.perf_counter() - started,
            "field_mapping": {
                "answer": "sensitive answer",
                "target_new": "sensitive answer",
                "target_true": "tokenizer.eos_token resolved at runtime",
            },
        },
        "balanced_fact_cycle": exposures,
        "reverse_direction_ablation": {
            "enabled": bool(args.include_relation_conditioned_reverse_prompts),
            "view_count": sum(
                bool(view.get("boundary_expanding", False)) for view in views
            ),
            "fact_ids": sorted(
                {
                    str(view["fact_id"])
                    for view in views
                    if view.get("boundary_expanding", False)
                }
            ),
            "reported_separately_from_primary": True,
        },
        "repair": repair_report,
        "final_evaluation_used_for_training_or_selection": False,
    }
    training_report_path = output_dir / "training_report.json"
    write_json(training_report_path, training_report)
    method_config = {
        "setting5_mode": SETTING5_MODE,
        "steps": args.steps,
        "optimizer": args.emb_lm_optimizer,
        "learning_rate": args.emb_lm_lr,
        "forget_weight": args.forget_weight,
        "retain_weight": args.retain_weight,
        "forget_margin": args.forget_margin,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "repair": asdict(repair_config(args)),
        "reverse_prompts": bool(args.include_relation_conditioned_reverse_prompts),
        "training_report_path": str(training_report_path.resolve()),
        "training_report_sha256": file_sha256(training_report_path),
    }
    receipt_path = args.checkpoint_receipt or output_dir / "checkpoint_receipt.json"
    receipt = create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=args.experiment_id,
        protocol_label=expected_label,
        protocol_status=training["protocol_status"],
        target_entity=target.subject,
        target_entity_id=f"rwku:{target.directory}",
        base_model_identity=local_model_identity(args.model_path),
        base_model_revision=args.model_revision,
        tokenizer_identity={
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
        },
        checkpoint_paths=[setting5_checkpoint, repaired_checkpoint],
        training_bundle_path=Path(bundle_path),
        optimization_protection_path=args.matched_protection_train,
        mcf_retain_optimization_paths=[mcf_optimization_manifest],
        mcf_repair_gate_paths=[mcf_gate_manifest],
        matched_protection_train_path=args.matched_protection_train,
        matched_protection_gate_path=args.matched_protection_gate,
        method_configuration=method_config,
        implementation_files=[
            SCRIPT_PATH,
            SEMANTIC_ROOT / "scripts" / "gagd_compare.py",
            SEMANTIC_ROOT / "scripts" / "rwku_repair.py",
            SEMANTIC_ROOT / "scripts" / "rwku_fact_sampler.py",
            SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
        ],
        sampler_provenance=exposures,
        generator_receipt_path=generator_receipt_path,
        official_locked_eval_path=output_dir / "official_locked_eval.json",
        confirmatory=args.confirmatory,
    )
    _write_state(
        args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(Path(receipt_path).resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        official_evaluation_opened=False,
    )
    print(f"Checkpoint frozen for {args.experiment_id}; receipt={receipt_path}")


def _rows_from_eval_artifact(artifact: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for view in artifact["payload"].get("views", []):
        rows.append(
            {
                "query": view["query"],
                "answer": view.get("sensitive_answer_alias") or view["canonical_sensitive_answer"],
                "subject": view["subject"],
                "level": str(view.get("level", "")),
                "type": str(view.get("query_type", view.get("prompt_style", ""))),
                "fact_id": view["fact_id"],
                "view_id": view["view_id"],
            }
        )
    return rows


def _wilson_interval(successes: int, total: int) -> Optional[List[float]]:
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * ((proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) ** 0.5) / denominator
    return [100.0 * max(0.0, center - radius), 100.0 * min(1.0, center + radius)]


def _recovery_metric(
    details: Sequence[Mapping[str, Any]],
    *,
    target_count: int = 1,
    success_key: str = "recovery_success",
    confidence_interval_meaningful: bool = True,
) -> Dict[str, Any]:
    numerator = sum(bool(row.get(success_key)) for row in details)
    denominator = len(details)
    facts = {row.get("fact_id") for row in details if row.get("fact_id")}
    return {
        "numerator": int(numerator),
        "denominator": denominator,
        "percentage": 100.0 * numerator / denominator if denominator else None,
        "independent_fact_group_count": len(facts) if facts else None,
        "prompt_count": denominator,
        "target_count": target_count,
        "confidence_interval_95_wilson": (
            _wilson_interval(int(numerator), denominator)
            if confidence_interval_meaningful
            else None
        ),
    }


def _score_metric(value: Any, denominator: int, *, target_count: int = 1) -> Dict[str, Any]:
    return {
        "numerator": None,
        "denominator": denominator,
        "score": value,
        "independent_fact_group_count": None,
        "prompt_count": denominator,
        "target_count": target_count,
        "confidence_interval": None,
    }


def _required_evaluation_sections(
    native: Mapping[str, Any],
    *,
    calibration_report: Mapping[str, Any],
    seen_details: Sequence[Mapping[str, Any]],
    unseen_details: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    details = native["details"]
    metrics = native["metrics"]
    summary = native["summary"]
    level1_details = details["cloze"]
    level2_details = details["direct"]
    adversarial = details["adversarial"]
    level3_by_type: Dict[str, Any] = {}
    for attack_type in sorted({str(row.get("type", "")) for row in adversarial}):
        level3_by_type[attack_type] = _recovery_metric(
            [row for row in adversarial if str(row.get("type", "")) == attack_type]
        )
    repair = calibration_report.get("repair", {})
    calibration_after = repair.get("active_prompt_instances_after")
    calibration_before = repair.get("active_prompt_instances_before")
    calibration = {
        "numerator": calibration_after,
        "denominator": calibration_before,
        "percentage": (
            100.0 * calibration_after / calibration_before
            if isinstance(calibration_after, int) and calibration_before
            else None
        ),
        "source": "pre-freeze calibration repair report; no evaluation artifact reopened",
        "independent_fact_group_count": None,
        "prompt_count": calibration_before,
        "target_count": 1,
        "confidence_interval": None,
    }
    return {
        "calibration_recovery": calibration,
        "seen_fact_unseen_prompt_recovery": _recovery_metric(seen_details),
        "unseen_fact_recovery": _recovery_metric(unseen_details),
        "level1_recovery": _recovery_metric(level1_details),
        "level2_recovery": _recovery_metric(level2_details),
        "level3_adversarial_recovery": _recovery_metric(adversarial),
        "level3_by_attack_type": level3_by_type,
        "target_answer_probability": _score_metric(summary["forget"]["target_answer_token_probability"], len(level2_details)),
        "full_answer_probability": _score_metric(summary["controls"]["full_answer_geometric_probability"], len(level2_details)),
        "forced_prefix_recovery": _recovery_metric(details["forced_prefix"]),
        "answer_alias_recovery": _recovery_metric(details["answer_alias"], success_key="alias_recovery_success"),
        "multiple_choice_recovery": _recovery_metric(details["multiple_choice_balanced"], success_key="correct", confidence_interval_meaningful=False),
        "open_ended_recovery": _recovery_metric(level2_details),
        "frozen_head_recovery": _recovery_metric(details["frozen_base_head_probe"], success_key="correct"),
        "membership_inference": _score_metric(summary["forget"]["membership_inference_attack_advantage"], len(details["membership_inference_forget"]) + len(details["membership_inference_retain"])),
        "neighbor_utility": _score_metric(summary["retain"]["neighboring_entity_accuracy"], len(details["neighbor"])),
        "downstream_utility": _score_metric(summary["retain"]["general_utility"], sum(len(details[key]) for key in ("mmlu", "bbh", "truthfulqa", "triviaqa"))),
        "fluency": _score_metric(metrics["fluency"], len(details["fluency"])),
        "perplexity": _score_metric(summary["retain"]["perplexity"], 1),
        "full_retain_probability_ratio": _score_metric(summary["retain"]["full_retain_probability_ratio"], 1),
    }


def _evaluate_staged(args: argparse.Namespace) -> None:
    state = _read_state(args)
    if state.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError(
            f"Evaluation requires CHECKPOINT_FROZEN, got {state.get('state')}"
        )
    receipt_path = args.checkpoint_receipt or _staged_output_dir(args) / "checkpoint_receipt.json"
    pre_open_receipt = load_receipt(receipt_path)
    expected_protocol_label = (
        PROBE_PROTOCOL_LABEL
        if args.training_source == TRAINING_SOURCE_PROBE
        else TARGET_ONLY_PROTOCOL_LABEL
    )
    if pre_open_receipt["protocol_label"] != expected_protocol_label:
        raise ValueError("Checkpoint receipt belongs to a different RWKU track")
    if mapping_sha256(local_model_identity(args.model_path)) != mapping_sha256(
        pre_open_receipt["base_model_identity"]
    ):
        raise ValueError("Evaluation base model identity differs from the frozen receipt")
    receipt = open_official_evaluation(
        receipt_path,
        experiment_id=args.experiment_id,
    )
    _write_state(
        args,
        "OFFICIAL_EVALUATION_OPENED",
        official_evaluation_opened=True,
        official_evaluation_opened_at_utc=receipt["official_evaluation_opened_at_utc"],
    )
    locked = read_artifact(
        _staged_output_dir(args) / "official_locked_eval.json",
        stage="evaluate",
        evaluation=True,
        expected_role="official_locked_eval",
    )
    # Only after the receipt transition and locked-descriptor integrity check
    # may this process open official record files.
    target, datasets, file_hashes = ensure_target_data(
        args.data_root,
        args.seed,
        allow_download=not args.no_download,
    )
    for filename, descriptor in locked["payload"]["files"].items():
        if file_hashes[filename] != descriptor["sha256"]:
            raise ValueError(f"Official locked evaluation file changed: {filename}")

    seen_rows: List[Dict[str, Any]] = []
    unseen_rows: List[Dict[str, Any]] = []
    if args.training_source == TRAINING_SOURCE_PROBE:
        seen_artifact = read_artifact(
            _staged_output_dir(args) / "seen_fact_unseen_prompt_eval.json",
            stage="evaluate",
            evaluation=True,
            expected_role="seen_fact_unseen_prompt_eval",
        )
        unseen_artifact = read_artifact(
            _staged_output_dir(args) / "unseen_fact_eval.json",
            stage="evaluate",
            evaluation=True,
            expected_role="unseen_fact_eval",
        )
        seen_rows = _rows_from_eval_artifact(seen_artifact)
        unseen_rows = _rows_from_eval_artifact(unseen_artifact)
        held_out = [*seen_rows, *unseen_rows]
        held_out_level1 = [row for row in held_out if row["level"] == "1"]
        held_out_level2 = [row for row in held_out if row["level"] == "2"]
    else:
        held_out_level1 = list(datasets["forget_level1.json"])
        held_out_level2 = list(datasets["forget_level2.json"])
    if not held_out_level1 or not held_out_level2:
        raise ValueError("Final RWKU evaluation requires non-empty Level 1 and Level 2 sets")

    training_report_path = Path(receipt["method_configuration"]["training_report_path"])
    if file_sha256(training_report_path) != receipt["method_configuration"]["training_report_sha256"]:
        raise ValueError("Training report changed after checkpoint freeze")
    with training_report_path.open("r", encoding="utf-8") as handle:
        calibration_report = json.load(handle)

    dtype = dtype_from_name(args.dtype)
    base_model, tokenizer = load_model_and_tokenizer(
        args.model_path, dtype=dtype, for_training=False, gradient_checkpointing=False
    )
    all_answers = [
        str(row["answer"])
        for filename in ("forget_level1.json", "forget_level2.json", "forget_level3.json")
        for row in datasets[filename]
    ]
    frozen_probe = build_frozen_head_probe(
        base_model, tokenizer, held_out_level2, additional_answers=all_answers
    )
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
    base_retain = base_result["retain_reference_mean_logprobs"]
    release_model(base_model)
    del base_model
    checkpoint_path = Path(receipt["checkpoint_paths"][-1]["path"])
    candidate, candidate_tokenizer = load_model_and_tokenizer(
        str(checkpoint_path), dtype=dtype, for_training=False, gradient_checkpointing=False
    )
    candidate_result = evaluate_method(
        method=METHOD_REPAIRED,
        model=candidate,
        tokenizer=candidate_tokenizer,
        target_subject=target.subject,
        held_out_cloze=held_out_level1,
        held_out_direct=held_out_level2,
        datasets=datasets,
        args=args,
        base_retain_mean_logprobs=base_retain,
        frozen_probe=frozen_probe,
    )
    for detail, row in zip(candidate_result["details"]["cloze"], held_out_level1):
        if row.get("fact_id"):
            detail["fact_id"] = row["fact_id"]
    for detail, row in zip(candidate_result["details"]["direct"], held_out_level2):
        if row.get("fact_id"):
            detail["fact_id"] = row["fact_id"]
    seen_details: List[Dict[str, Any]] = []
    unseen_details: List[Dict[str, Any]] = []
    if seen_rows:
        _, seen_details = evaluate_qa_rows(
            candidate, candidate_tokenizer, seen_rows, batch_size=args.eval_batch_size
        )
        for detail, row in zip(seen_details, seen_rows):
            detail["fact_id"] = row["fact_id"]
    if unseen_rows:
        _, unseen_details = evaluate_qa_rows(
            candidate, candidate_tokenizer, unseen_rows, batch_size=args.eval_batch_size
        )
        for detail, row in zip(unseen_details, unseen_rows):
            detail["fact_id"] = row["fact_id"]
    candidate_result["protocol_label"] = receipt["protocol_label"]
    candidate_result["protocol_status"] = receipt["protocol_status"]
    candidate_result["evaluation_sections"] = _required_evaluation_sections(
        candidate_result,
        calibration_report=calibration_report,
        seen_details=seen_details,
        unseen_details=unseen_details,
    )
    candidate_result["interpretation_axes"] = {
        "calibration_efficacy": "reported separately",
        "prompt_generalization": "seen-fact/unseen-prompt generalization",
        "unseen_fact_transfer": "unseen-fact entity transfer",
        "adversarial_resistance": "official Level 3",
        "decoder_suppression": "normal LM-head recovery controls",
        "frozen_head_representation_recovery": "not representation erasure",
        "utility_preservation": "neighbors, downstream utility, fluency, PPL, retain ratio",
    }
    result_path = _staged_output_dir(args) / "official_evaluation.json"
    write_json(result_path, {"base": base_result, "unlearned": candidate_result})
    release_model(candidate)
    complete = mark_evaluation_complete(receipt_path, experiment_id=args.experiment_id)
    _write_state(
        args,
        "EVALUATION_COMPLETE",
        official_evaluation_opened=True,
        evaluation_completed_at_utc=complete["evaluation_completed_at_utc"],
        result_path=str(result_path.resolve()),
    )
    print(f"RWKU official evaluation complete: {result_path}")


def run_staged_protocol(args: argparse.Namespace) -> None:
    if args.legacy_row_split:
        raise ValueError("Legacy row splitting is not an entity-fact staged protocol")
    if args.stage == "prepare":
        _prepare_staged(args)
        return
    if args.stage == "train":
        _train_staged(args)
        return
    if args.stage == "evaluate":
        _evaluate_staged(args)
        return
    if args.training_source == TRAINING_SOURCE_TARGET_ONLY and args.confirmatory:
        raise ValueError("Confirmatory target-only runs cannot use --stage all")
    _prepare_staged(args)
    if args.dry_run:
        return
    _train_staged(args)
    _evaluate_staged(args)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.training_source is not None:
        run_staged_protocol(args)
        return
    if args.stage != "all":
        raise ValueError(
            "--stage prepare/train/evaluate requires --training-source; the "
            "legacy command remains an all-in-one prompt-held-out experiment"
        )
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
    payload["protocol_label"] = "rwku_legacy_independent_row_hash_split"
    payload["protocol_status"] = LEGACY_PROTOCOL_STATUS
    payload["split_interpretation"] = (
        "prompt-held-out only; not a relation-aware unseen-fact split"
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

    artifact_by_method = {
        METHOD_BASE: output_dir / "base_model.json",
        METHOD_ZERO: output_dir / "original_zerounlearn.json",
        METHOD_SETTING5: output_dir / "setting5_without_repair.json",
        METHOD_REPAIRED: output_dir / "setting5_protected_repair.json",
        METHOD_REPAIR_ONLY: output_dir / "repair_only.json",
        METHOD_REPRESENTATION: output_dir / "protected_representation.json",
    }
    for method, result in results.items():
        if method != METHOD_BASE:
            contract = rwku_success_contract(
                base_result["summary"],
                result["summary"],
            )
            if method == METHOD_REPRESENTATION:
                selection = result["unlearning"]["selection"]
                internal_pass = bool(
                    selection["accepted_all_calibration_efficacy_targets"]
                    and selection["accepted_all_non_target_protection_gates"]
                )
                contract["internal_representation_selection"] = {
                    "required": True,
                    "pass": internal_pass,
                    "calibration_efficacy_gates_pass": bool(
                        selection[
                            "accepted_all_calibration_efficacy_targets"
                        ]
                    ),
                    "non_target_protection_gates_pass": bool(
                        selection[
                            "accepted_all_non_target_protection_gates"
                        ]
                    ),
                }
                if not internal_pass:
                    contract["failed_criteria"].append(
                        "internal_representation_selection"
                    )
                    contract["failed_criteria"] = sorted(
                        set(contract["failed_criteria"])
                    )
                    contract["passed"] = False
            result["success_contract"] = contract
        write_json(artifact_by_method[method], result)

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
