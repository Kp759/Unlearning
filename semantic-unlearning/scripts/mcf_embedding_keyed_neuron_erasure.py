#!/usr/bin/env python3
"""Embedding-keyed sparse-neuron conditional suppression for locked MCF.

Architecture
------------

    ordinary sparse subject embedding-row deltas (frozen Stage 1)
        -> frozen early Transformer layers
        -> record-specific contextual activation code
        -> frozen sparse SwiGLU detector features
        -> registered internal response threshold
        -> isolated sparse residual actuator
        -> frozen remaining Transformer
        -> exactly unchanged LM head

For each edit, a disjoint group of existing low-activation MLP features was
selected using only training-safe writer-on versus writer-off activations.
V3.6.1 deterministically replays the exact V3.5.4 detector and fixed 0.20/0.25
boundary, then
retains the canonical prompt-role labels introduced by V3.5.3. Exact duplicate
prompts share one canonical hidden state and one multi-label target: every
record for which that prompt is registered positive stays active even when the
same prompt appears as another record's relative negative. Every other
writer-on label and every writer-off label remains inactive. V3.5.4 removed
V3.5.3's complete-update global tails from optimization while retaining the
equal-record mean plus per-record worst-two objectives that repaired writer-off
isolation in V3.5.2. It also records a non-optimizing component-gradient audit
before the first update. A fresh all-cell
multi-label certificate and exact V3.5.4 tensor hashes must pass. V3.6.1 then
hash-binds V3.5.5's discarded 4/8/16 width sweep, reconstructs its exact
selected width-16 actuator bank, and trains that separate bank from exact zero.
Detector and actuator features remain disjoint and the ordinary Base MLP
contribution stays untouched. A criterion-stopped positive warm start is
followed by globally balanced training-safe preservation optimization. Unlike
V3.6, the behavioral preservation bank applies a preregistered exact-prompt
rule: a prompt registered for forgetting cannot simultaneously act as another
record's preservation negative. All merely lexical or subtoken overlaps remain
in scope. Negative-context optimization compares float32 current NLLs with
float32 baselines and targets zero drift, while the scientific acceptance
ceiling remains 0.05 under the exact native-dtype scorer. A
candidate state is saved only after every locked training-only acceptance gate
passes; official evaluation remains unavailable to this learner.

Data firewall
-------------

This learner has deliberately no ``--mcf-path`` argument.  It accepts only a
locked direct-forget training view, an audited training-safe context manifest,
and a frozen Stage-1 writer whose stored manifest hash must match exactly.
Official paraphrases, neighborhoods, reserved retain prompts, PPL text,
aliases, and adversarial attacks are unavailable to this process. A separate
9,438-record development-retain overlap audit is registered only as consumed
architecture-motivation evidence; none of its records or statistics enters
selection, optimization, acceptance, or retry. Official probes may be opened
only by a separately preregistered post-feasibility process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional_method
import mcf_compositional_marker_core as compositional_core
import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import sure_canonical_core as canonical


METHOD = "Embedding-Keyed Sparse Contextual Residual Suppression"
PROTOCOL = neuron_core.PROTOCOL
FROZEN_DETECTOR_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_2"
FROZEN_V3_3_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_3"
FROZEN_V3_4_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_4"
FROZEN_V3_5_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5"
FROZEN_V3_5_1_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_1"
FROZEN_V3_5_2_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_2"
FROZEN_V3_5_3_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_3"
FROZEN_V3_5_4_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_4"
FROZEN_V3_5_5_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_5"
FROZEN_V3_6_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_6"
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "MCF_PATH",
    "OFFICIAL_MCF_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
)
COUNT_DERIVED_FRACTION_ABS_TOLERANCE = 1e-7
FROZEN_DETECTOR_REPLAY_ABS_TOLERANCE = 1e-5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--stage1-report", required=True)
    parser.add_argument("--stage1-writer-log", required=True)
    parser.add_argument("--clean-stage1-portability-preflight", required=True)
    parser.add_argument("--clean-stage1-acceptance", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--experiment-label", default="primary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument(
        "--writer-mode",
        choices=("embedding_keyed", "none"),
        default="embedding_keyed",
        help=(
            "embedding_keyed trains the proposed conjunction with the frozen "
            "sparse embedding writer. none is the independently optimized, "
            "matched-capacity sparse-MLP control and materializes no embedding edit."
        ),
    )

    parser.add_argument("--neuron-layer", type=int, default=27)
    parser.add_argument("--neurons-per-record", type=int, default=4)
    parser.add_argument("--dormant-fraction", type=float, default=0.20)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "writer_contrastive",
            "base_context_contrastive",
            "dormant_random",
        ),
        default="writer_contrastive",
    )
    parser.add_argument("--selection-stability-weight", type=float, default=1.0)
    parser.add_argument("--selection-positive-contexts", type=int, default=8)
    parser.add_argument("--selection-negative-contexts", type=int, default=8)
    parser.add_argument("--selection-protected-prompts", type=int, default=8192)
    parser.add_argument(
        "--selection-min-corpus-prompts",
        type=int,
        default=4096,
        help=(
            "Reserve this many disjoint corpus prompts inside the protected-neuron "
            "profile. Training prompts cannot consume this quota."
        ),
    )
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=5000)
    parser.add_argument("--corpus-protection-prompts", type=int, default=8192)
    parser.add_argument(
        "--writer-preflight-amplitude-threshold", type=float, default=4.5
    )
    parser.add_argument(
        "--writer-preflight-min-global-fraction", type=float, default=0.95
    )
    parser.add_argument(
        "--writer-preflight-min-record-fraction", type=float, default=0.80
    )

    parser.add_argument("--detector-steps", type=int, default=1000)
    parser.add_argument("--detector-lr", type=float, default=1e-3)
    parser.add_argument(
        "--detector-initialization",
        choices=("frozen_v3_2", "train"),
        default="train",
        help=(
            "V3.5.4 imports the exact passed V3.2 gate/up tensors as the repair "
            "initialization."
        ),
    )
    parser.add_argument(
        "--frozen-v3-2-run-dir",
        help=(
            "Rejected V3.2 output root containing method/embedding_keyed_neuron_state.pt "
            "and its training-only detector receipts."
        ),
    )
    parser.add_argument(
        "--frozen-v3-3-run-dir",
        help=(
            "Rejected V3.3 output root whose detector-pass/actuator-feasibility "
            "failure is a historical V3.4 input only; V3.5 imports V3.4 directly."
        ),
    )
    parser.add_argument(
        "--frozen-v3-4-run-dir",
        help=(
            "Rejected V3.4 output root whose cap-1.5 reachability and structural "
            "selectivity failure are hash-bound into the V3.5 architecture test."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-run-dir",
        help=(
            "Rejected V3.5 output root whose single unresolved writer-off gate "
            "collision is hash-bound as historical evidence."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-1-run-dir",
        help=(
            "Completed V3.5.1 forensic output identifying the exact non-owner "
            "writer-off collision that licenses the global-scope repair."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-2-run-dir",
        help=(
            "Rejected V3.5.2 output whose exact duplicate-prompt positive/negative "
            "constraint contradiction licenses multi-label prompt semantics."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-3-run-dir",
        help=(
            "Rejected V3.5.3 output whose 29/50 positive collapse under "
            "complete-update global tails licenses the balanced V3.5.4 repair."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-4-run-dir",
        help=(
            "Rejected V3.5.4 output whose exact 50/50 detector tensor hashes and "
            "cap-1.50 four-feature actuator failure license the width sweep."
        ),
    )
    parser.add_argument(
        "--frozen-v3-5-5-run-dir",
        help=(
            "Completed V3.5.5 output whose exact detector replay, nested actuator "
            "selection, native width-16 reachability pass, and discarded-fit "
            "receipt license V3.6/V3.6.1 full preservation training."
        ),
    )
    parser.add_argument(
        "--frozen-v3-6-run-dir",
        help=(
            "Rejected V3.6 output root whose successful width-16 warm start and "
            "official-compatible negative-locality miss license V3.6.1's "
            "coherent float32 negative-preservation objective."
        ),
    )
    parser.add_argument(
        "--detector-record-batch",
        type=int,
        default=4,
        help=(
            "Record-microbatch capacity for detector gradient accumulation. "
            "Every optimizer update still covers every record."
        ),
    )
    parser.add_argument(
        "--detector-positive-contexts",
        choices=("all",),
        default="all",
        help="V3.2 locks detector training to every training-safe positive context.",
    )
    parser.add_argument(
        "--detector-negative-contexts",
        choices=("all",),
        default="all",
        help="V3.2 locks detector training to every training-safe negative context.",
    )
    parser.add_argument("--detector-tail-k", type=int, default=2)
    parser.add_argument(
        "--detector-global-tail-weight",
        type=float,
        default=0.0,
        help=(
            "Historical complete-update worst-two weight. V3.5.4 locks this to "
            "zero; global extrema remain audited but are not optimized."
        ),
    )
    parser.add_argument(
        "--detector-response-mode",
        choices=("absolute_signed_group_activation",),
        default="absolute_signed_group_activation",
    )
    parser.add_argument(
        "--detector-positive-floor",
        type=float,
        default=0.25,
        help="Locked positive acceptance floor; this is not the training target.",
    )
    parser.add_argument(
        "--detector-training-positive-floor",
        type=float,
        default=0.30,
        help="Positive optimization target with preregistered certificate headroom.",
    )
    parser.add_argument("--detector-negative-weight", type=float, default=5.0)
    parser.add_argument("--detector-cross-weight", type=float, default=2.0)
    parser.add_argument("--detector-writer-off-weight", type=float, default=10.0)
    parser.add_argument("--detector-consistency-weight", type=float, default=1.0)
    parser.add_argument("--detector-l2", type=float, default=1e-5)
    parser.add_argument("--detector-relative-cap", type=float, default=1.0)
    parser.add_argument(
        "--detector-off-abs-max",
        type=float,
        default=0.20,
        help="Locked negative/writer-off acceptance ceiling.",
    )
    parser.add_argument(
        "--detector-training-off-abs-max",
        type=float,
        default=0.15,
        help="Negative/writer-off optimization ceiling with certificate headroom.",
    )
    parser.add_argument(
        "--detector-certificate-abs-tolerance",
        type=float,
        default=1e-7,
        help="Preregistered numerical comparison tolerance for the detector gate.",
    )

    parser.add_argument("--actuator-feasibility-steps", type=int, default=100)
    parser.add_argument(
        "--training-only-multilabel-detector-repair-feasibility",
        "--training-only-actuator-width-sweep",
        "--training-only-global-writer-off-repair-feasibility",
        "--training-only-isolated-threshold-feasibility",
        "--training-only-actuator-cap-sweep",
        dest="training_only_actuator_width_sweep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replay and hash-bind the V3.5.4 balanced multi-label detector, then "
            "run discarded separate-actuator width feasibility fits."
        ),
    )
    parser.add_argument(
        "--actuator-feasibility-caps",
        nargs="+",
        type=float,
        default=[1.50],
        help="V3.5.5 keeps the native per-column cap fixed at 1.50.",
    )
    parser.add_argument(
        "--training-only-full-preservation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "V3.6.1 hash-binds V3.5.5 and the rejected V3.6 preservation run, "
            "reconstructs the exact width-16 separate actuator bank, runs a "
            "criterion-stopped positive warm start, and then trains the locked "
            "coherent float32 full-preservation objective without official evaluation "
            "access."
        ),
    )
    parser.add_argument(
        "--actuator-feasibility-widths",
        nargs="+",
        type=int,
        default=[4, 8, 16],
        help="Nested per-record actuator-bank widths for V3.5.5.",
    )
    parser.add_argument(
        "--actuator-architecture",
        choices=(
            "replace_selected_neuron_contribution",
            "isolated_thresholded_residual",
            "separate_threshold_gated_actuator_bank",
        ),
        default="separate_threshold_gated_actuator_bank",
        help=(
            "V3.5.5 leaves the ordinary MLP output untouched and fits only a "
            "separate frozen-feature threshold-gated residual bank."
        ),
    )
    parser.add_argument(
        "--threshold-gate-numerical-guard",
        type=float,
        default=1e-6,
        help=(
            "Moves the runtime off boundary above 0.20 and the on boundary below "
            "0.25 so the registered detector comparison tolerance maps to exact "
            "clipped gate endpoints."
        ),
    )
    parser.add_argument(
        "--detector-selectivity-ratio-epsilon",
        type=float,
        default=1e-8,
        help="Numerical denominator floor for diagnostic on/off activation ratios.",
    )
    parser.add_argument(
        "--detector-selectivity-warning-ratio",
        type=float,
        default=100.0,
        help=(
            "Preregistered heuristic warning level for the p10 on/off activation "
            "ratio; diagnostic only, not a scientific acceptance gate."
        ),
    )
    parser.add_argument("--actuator-steps", type=int, default=100)
    parser.add_argument("--actuator-lr", type=float, default=5e-4)
    parser.add_argument(
        "--actuator-batch-size",
        type=int,
        default=4,
        help="Context microbatch capacity inside an all-record optimizer update.",
    )
    parser.add_argument(
        "--actuator-protected-batch",
        type=int,
        default=80,
        help="Deterministic protected-prompt sample per global optimizer update.",
    )
    parser.add_argument("--actuator-positive-contexts", choices=("all",), default="all")
    parser.add_argument("--actuator-negative-contexts", choices=("all",), default="all")
    parser.add_argument("--actuator-tail-k", type=int, default=2)
    parser.add_argument("--actuator-writer-off-every", type=int, default=1)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=20.0)
    parser.add_argument("--reference-nll-weight", type=float, default=50.0)
    parser.add_argument("--reference-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--protected-kl-weight", type=float, default=20.0)
    parser.add_argument(
        "--negative-preservation-weight",
        type=float,
        default=50.0,
        help="V3.6.1 weight for all-record negative-context preservation.",
    )
    parser.add_argument("--negative-nll-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--negative-training-nll-tolerance",
        type=float,
        default=0.0,
        help=(
            "Float32 optimization target for negative NLL drift. V3.6.1 fixes "
            "this at zero while retaining the 0.05 native-dtype acceptance gate."
        ),
    )
    parser.add_argument("--protected-kl-mean-tolerance", type=float, default=0.05)
    parser.add_argument("--protected-kl-max-tolerance", type=float, default=0.50)
    parser.add_argument("--writer-off-nll-weight", type=float, default=50.0)
    parser.add_argument("--actuator-writer-off-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--actuator-l2", type=float, default=1e-4)
    parser.add_argument("--actuator-relative-cap", type=float, default=0.50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--kl-topk", type=int, default=64)
    parser.add_argument("--hook-materialization-tolerance", type=float, default=0.05)

    parser.add_argument(
        "--min-writer-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--min-decoder-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--require-writer-necessity",
        "--require-within-checkpoint-writer-dependence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the within-checkpoint writer-off intervention to break the "
            "joint solution. Disabled only for the preregistered no-writer control."
        ),
    )
    parser.add_argument(
        "--gate-policy",
        choices=("strict", "report"),
        default="strict",
        help="Strict requires the training-only detector certificate before saving.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--save-rejected-checkpoint",
        action="store_true",
        help=(
            "Freeze a fixed-budget rejected checkpoint for evaluation. Allowed "
            "only for the independently trained no-writer control."
        ),
    )
    value = parser.parse_args(list(argv) if argv is not None else None)

    if int(value.forget_num) <= 0:
        parser.error("--forget-num must be positive")
    if int(value.neuron_layer) < 0:
        parser.error("--neuron-layer must be non-negative")
    if int(value.frequency_doc_start) < 20:
        parser.error(
            "--frequency-doc-start must be >= 20; documents 0:20 are reserved "
            "for held-out official PPL"
        )
    if int(value.neurons_per_record) <= 0:
        parser.error("--neurons-per-record must be positive")
    if not 0.0 < float(value.dormant_fraction) <= 1.0:
        parser.error("--dormant-fraction must lie in (0, 1]")
    if int(value.selection_protected_prompts) <= 0:
        parser.error("--selection-protected-prompts must be positive")
    if (
        min(
            int(value.selection_positive_contexts),
            int(value.selection_negative_contexts),
        )
        <= 0
    ):
        parser.error("selection positive/negative context counts must be positive")
    if (
        not 0
        <= int(value.selection_min_corpus_prompts)
        <= int(value.selection_protected_prompts)
    ):
        parser.error(
            "--selection-min-corpus-prompts must lie between zero and the "
            "protected-prompt budget"
        )
    if int(value.corpus_protection_prompts) < int(value.selection_min_corpus_prompts):
        parser.error(
            "--corpus-protection-prompts must cover the reserved selection corpus quota"
        )
    if int(value.detector_steps) < 0:
        parser.error("--detector-steps must be non-negative")
    if value.detector_initialization == "frozen_v3_2":
        if value.writer_mode != "embedding_keyed":
            parser.error("V3.2 detector initialization requires the writer")
        if not str(value.frozen_v3_2_run_dir or "").strip():
            parser.error("--frozen-v3-2-run-dir is required for V3.2 initialization")
    elif value.frozen_v3_2_run_dir:
        parser.error("--frozen-v3-2-run-dir is only valid with V3.2 initialization")
    if int(value.detector_record_batch) <= 0:
        parser.error("--detector-record-batch must be positive")
    if int(value.detector_tail_k) <= 0:
        parser.error("--detector-tail-k must be positive")
    if (
        not math.isfinite(float(value.detector_global_tail_weight))
        or float(value.detector_global_tail_weight) < 0
    ):
        parser.error("--detector-global-tail-weight must be finite and non-negative")
    if int(value.actuator_feasibility_steps) <= 0:
        parser.error("--actuator-feasibility-steps must be positive")
    caps = [float(cap) for cap in value.actuator_feasibility_caps]
    if not caps or any(not math.isfinite(cap) or cap <= 0.0 for cap in caps):
        parser.error("--actuator-feasibility-caps must contain positive finite caps")
    if caps != sorted(set(caps)):
        parser.error("--actuator-feasibility-caps must be strictly increasing")
    widths = [int(width) for width in value.actuator_feasibility_widths]
    if not widths or any(width <= 0 for width in widths):
        parser.error("--actuator-feasibility-widths must contain positive integers")
    if widths != sorted(set(widths)):
        parser.error("--actuator-feasibility-widths must be strictly increasing")
    if (
        not math.isfinite(float(value.detector_selectivity_ratio_epsilon))
        or float(value.detector_selectivity_ratio_epsilon) <= 0.0
    ):
        parser.error("--detector-selectivity-ratio-epsilon must be finite and positive")
    if (
        not math.isfinite(float(value.detector_selectivity_warning_ratio))
        or float(value.detector_selectivity_warning_ratio) <= 0.0
    ):
        parser.error("--detector-selectivity-warning-ratio must be finite and positive")
    if value.training_only_actuator_width_sweep and value.training_only_full_preservation:
        parser.error(
            "V3.5.5 width feasibility and V3.6.1 full preservation are mutually exclusive"
        )
    if value.training_only_actuator_width_sweep:
        if int(value.detector_steps) != 100:
            parser.error(
                "V3.5.5 requires exactly 100 deterministic detector replay updates"
            )
        if float(value.detector_global_tail_weight) != 0.0:
            parser.error(
                "V3.5.5 requires --detector-global-tail-weight 0 so complete-update "
                "extrema are diagnostic only"
            )
        if int(value.actuator_steps) != 0:
            parser.error("V3.5.5 feasibility requires --actuator-steps 0")
        if value.actuator_architecture != "separate_threshold_gated_actuator_bank":
            parser.error(
                "V3.5.5 feasibility requires the separate threshold-gated "
                "actuator-bank architecture"
            )
        if caps != [1.5]:
            parser.error(
                "V3.5.5 retains --actuator-feasibility-caps at exactly 1.50"
            )
        if widths != [4, 8, 16]:
            parser.error("V3.5.5 requires actuator widths exactly 4 8 16")
    elif value.training_only_full_preservation:
        if int(value.detector_steps) != 100:
            parser.error(
                "V3.6.1 requires exactly 100 deterministic detector replay updates"
            )
        if float(value.detector_global_tail_weight) != 0.0:
            parser.error("V3.6.1 requires --detector-global-tail-weight 0")
        if int(value.actuator_steps) != 200:
            parser.error("V3.6.1 requires exactly 200 full preservation updates")
        if value.actuator_architecture != "separate_threshold_gated_actuator_bank":
            parser.error(
                "V3.6.1 requires the separate threshold-gated actuator-bank architecture"
            )
        if caps != [1.5]:
            parser.error("V3.6.1 retains the native per-column cap at exactly 1.50")
        if widths != [4, 8, 16]:
            parser.error(
                "V3.6.1 reconstructs the exact nested V3.5.5 widths 4 8 16"
            )
        if not math.isclose(
            float(value.actuator_relative_cap), 1.5, abs_tol=1e-12
        ):
            parser.error("V3.6.1 requires --actuator-relative-cap 1.50")
        if not math.isclose(
            float(value.negative_preservation_weight), 50.0, abs_tol=1e-12
        ):
            parser.error("V3.6.1 requires --negative-preservation-weight 50")
        if not math.isclose(
            float(value.negative_training_nll_tolerance), 0.0, abs_tol=1e-12
        ):
            parser.error("V3.6.1 requires --negative-training-nll-tolerance 0")
    elif int(value.actuator_steps) <= 0:
        parser.error("--actuator-steps must be positive outside feasibility mode")
    if int(value.actuator_batch_size) <= 0:
        parser.error("--actuator-batch-size must be positive")
    if int(value.actuator_protected_batch) <= 0:
        parser.error("--actuator-protected-batch must be positive")
    if int(value.actuator_tail_k) <= 0:
        parser.error("--actuator-tail-k must be positive")
    if float(value.actuator_writer_off_nll_tolerance) < 0:
        parser.error("--actuator-writer-off-nll-tolerance must be non-negative")
    if float(value.negative_nll_tolerance) < 0:
        parser.error("--negative-nll-tolerance must be non-negative")
    if (
        not math.isfinite(float(value.negative_training_nll_tolerance))
        or not 0.0
        <= float(value.negative_training_nll_tolerance)
        <= float(value.negative_nll_tolerance)
    ):
        parser.error(
            "--negative-training-nll-tolerance must lie between zero and the "
            "locked --negative-nll-tolerance"
        )
    if (
        not math.isfinite(float(value.negative_preservation_weight))
        or float(value.negative_preservation_weight) < 0.0
    ):
        parser.error("--negative-preservation-weight must be finite and non-negative")
    if float(value.protected_kl_mean_tolerance) < 0:
        parser.error("--protected-kl-mean-tolerance must be non-negative")
    if float(value.protected_kl_max_tolerance) < float(
        value.protected_kl_mean_tolerance
    ):
        parser.error(
            "--protected-kl-max-tolerance must be at least the mean tolerance"
        )
    if float(value.detector_positive_floor) < 0:
        parser.error("--detector-positive-floor must be non-negative")
    if float(value.detector_training_positive_floor) < float(
        value.detector_positive_floor
    ):
        parser.error(
            "--detector-training-positive-floor must be at least the locked "
            "--detector-positive-floor"
        )
    if float(value.detector_off_abs_max) < 0:
        parser.error("--detector-off-abs-max must be non-negative")
    if (
        not 0
        <= float(value.detector_training_off_abs_max)
        <= float(value.detector_off_abs_max)
    ):
        parser.error(
            "--detector-training-off-abs-max must lie between zero and the "
            "locked --detector-off-abs-max"
        )
    if not 0 <= float(value.detector_certificate_abs_tolerance) <= 1e-6:
        parser.error("--detector-certificate-abs-tolerance must lie in [0, 1e-6]")
    for name in (
        "detector_negative_weight",
        "detector_cross_weight",
        "detector_writer_off_weight",
        "detector_consistency_weight",
        "detector_l2",
    ):
        if float(getattr(value, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if float(value.writer_preflight_amplitude_threshold) < 0:
        parser.error("--writer-preflight-amplitude-threshold must be non-negative")
    for name in (
        "writer_preflight_min_global_fraction",
        "writer_preflight_min_record_fraction",
    ):
        if not 0.0 <= float(getattr(value, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if value.writer_mode == "embedding_keyed":
        if value.selection_mode == "base_context_contrastive":
            parser.error(
                "embedding_keyed mode must use writer_contrastive or dormant_random selection"
            )
        if not bool(value.require_writer_necessity):
            parser.error(
                "embedding_keyed mode must retain the within-checkpoint writer-dependence gate"
            )
    else:
        if value.selection_mode == "writer_contrastive":
            parser.error("no-writer control cannot use writer_contrastive selection")
        if bool(value.require_writer_necessity):
            parser.error("no-writer control must disable --require-writer-necessity")
    if bool(value.save_rejected_checkpoint) and value.writer_mode != "none":
        parser.error(
            "--save-rejected-checkpoint is restricted to the no-writer control"
        )
    if value.training_only_actuator_width_sweep or value.training_only_full_preservation:
        revision = "V3.5.5" if value.training_only_actuator_width_sweep else "V3.6.1"
        if value.experiment_label != "primary":
            parser.error(f"{revision} is restricted to the primary run")
        if value.writer_mode != "embedding_keyed":
            parser.error(f"{revision} requires the embedding writer")
        if value.detector_initialization != "frozen_v3_2":
            parser.error(f"{revision} replay requires the frozen V3.2 detector")
        if not str(value.frozen_v3_4_run_dir or "").strip():
            parser.error(f"--frozen-v3-4-run-dir is required for {revision}")
        if not str(value.frozen_v3_5_run_dir or "").strip():
            parser.error(f"--frozen-v3-5-run-dir is required for {revision}")
        if not str(value.frozen_v3_5_1_run_dir or "").strip():
            parser.error(f"--frozen-v3-5-1-run-dir is required for {revision}")
        if not str(value.frozen_v3_5_2_run_dir or "").strip():
            parser.error(f"--frozen-v3-5-2-run-dir is required for {revision}")
        if not str(value.frozen_v3_5_3_run_dir or "").strip():
            parser.error(f"--frozen-v3-5-3-run-dir is required for {revision}")
        if not str(value.frozen_v3_5_4_run_dir or "").strip():
            parser.error(f"--frozen-v3-5-4-run-dir is required for {revision}")
        if value.frozen_v3_3_run_dir:
            parser.error(
                f"{revision} imports V3.4 through V3.5.4, not V3.3 directly"
            )
        if value.training_only_actuator_width_sweep and value.frozen_v3_5_5_run_dir:
            parser.error("V3.5.5 cannot import itself")
        if value.training_only_full_preservation and not str(
            value.frozen_v3_5_5_run_dir or ""
        ).strip():
            parser.error("--frozen-v3-5-5-run-dir is required for V3.6.1")
        if value.training_only_full_preservation and not str(
            value.frozen_v3_6_run_dir or ""
        ).strip():
            parser.error("--frozen-v3-6-run-dir is required for V3.6.1")
        if value.training_only_actuator_width_sweep and value.frozen_v3_6_run_dir:
            parser.error("V3.5.5 cannot import V3.6")
        if value.training_only_actuator_width_sweep and (
            value.save_checkpoint or value.save_rejected_checkpoint
        ):
            parser.error("V3.5.5 training-only feasibility cannot save checkpoints")
        if value.training_only_full_preservation:
            if not value.save_checkpoint:
                parser.error(
                    "V3.6.1 requires --save-checkpoint for a passing candidate"
                )
            if value.save_rejected_checkpoint:
                parser.error("V3.6.1 never saves rejected checkpoints")
    elif (
        value.frozen_v3_3_run_dir
        or value.frozen_v3_4_run_dir
        or value.frozen_v3_5_run_dir
        or value.frozen_v3_5_1_run_dir
        or value.frozen_v3_5_2_run_dir
        or value.frozen_v3_5_3_run_dir
        or value.frozen_v3_5_4_run_dir
        or value.frozen_v3_5_5_run_dir
        or value.frozen_v3_6_run_dir
    ):
        parser.error(
            "frozen V3.3 through V3.6 run directories require a registered "
            "V3.5.5/V3.6.1 training-only mode"
        )
    guard = float(value.threshold_gate_numerical_guard)
    if not math.isfinite(guard) or guard < 0.0:
        parser.error("--threshold-gate-numerical-guard must be finite and non-negative")
    if (
        float(value.detector_positive_floor) - guard
        <= float(value.detector_off_abs_max) + guard
    ):
        parser.error("threshold-gate guard closes the registered detector gap")
    for name in ("detector_relative_cap", "actuator_relative_cap"):
        if not 0.0 < float(getattr(value, name)) <= 2.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0, 2]")
    if float(value.hook_materialization_tolerance) < 0:
        parser.error("--hook-materialization-tolerance must be non-negative")
    if value.device_map == "auto":
        parser.error("training sparse neurons requires --device-map single")
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into learner environment: "
            + ", ".join(sorted(exposed))
        )


def _validate_experiment_registry(
    registry: Mapping[str, Any], args: argparse.Namespace
) -> None:
    if registry.get("protocol") != PROTOCOL:
        raise RuntimeError("experiment registry protocol mismatch")
    writer_prerequisite = registry.get("stage1_writer_prerequisite")
    if not isinstance(writer_prerequisite, Mapping):
        raise RuntimeError("registry lacks the clean Stage-1 writer prerequisite")
    expected_template_hash = compositional_method.sha256_json(
        synthetic.RELATION_ALTERNATE_TEMPLATES
    )
    expected_writer_prerequisite = {
        "protocol": compositional_core.PROTOCOL,
        "positive_context_policy": compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY,
        "relation_template_bank_sha256": expected_template_hash,
        "training_origin": "Base model with no resumed Stage-1 state",
        "writer_steps": 1200,
        "writer_steps_semantics": (
            "optimizer updates after full-record gradient accumulation"
        ),
        "writer_record_batch": 3,
        "writer_record_batch_semantics": ("gradient_accumulation_microbatch_capacity"),
        "writer_update_coverage": "all_records_accumulated",
        "writer_records_per_optimizer_update": 50,
        "writer_microbatches_per_optimizer_update": 17,
        "writer_optimizer_updates": 1200,
        "writer_record_exposures": 60000,
        "writer_record_local_reference_exposures": 3600,
        "writer_record_exposure_multiplier_vs_v6_1": 50 / 3,
        "writer_gradient_normalization": (
            "equal_record_mean_plus_global_prompt_mean_kl"
        ),
        "writer_kl_evaluation": (
            "exact_registered_topk_rows_without_full_vocabulary_materialization"
        ),
        "writer_gradient_conflict_audit_phases": ["initial", "final"],
        "writer_gradient_conflict_audit_objectives": [
            "positive_write",
            "full_writer",
        ],
        "writer_gradient_conflict_audit_hash_bound": True,
        "writer_positive_context_mode": "all",
        "writer_positive_context_batch": 7,
        "writer_positive_tail_k": 2,
        "writer_negative_context_batch": 5,
        "writer_objective": "mean_plus_worst_k_squared_shortfall",
        "cross_record_parameter_sharing_audit_required": True,
    }
    for key, expected_value in expected_writer_prerequisite.items():
        if writer_prerequisite.get(key) != expected_value:
            raise RuntimeError(f"registry clean-writer prerequisite mismatch: {key}")
    exposure_audit = registry.get("development_retain_shared_row_exposure_audit")
    expected_exposure_audit = {
        "data_role": "consumed_development_evidence_not_blind_evaluation",
        "source_writer_protocol": compositional_core.PROTOCOL,
        "forget_records": 50,
        "reserved_official_retain_records_excluded": 1000,
        "development_retain_records_consumed": 9438,
        "forget_records_with_subject_word_overlap": 43,
        "unique_forget_subject_words": 110,
        "unique_forget_subject_words_reused": 65,
        "development_prompts_with_forget_subject_word": 4298,
        "forget_records_with_subject_subtoken_overlap": 50,
        "unique_forget_subject_token_ids": 236,
        "unique_forget_subject_token_ids_reused": 199,
        "development_prompts_with_forget_subject_subtoken": 5763,
        "actual_edited_embedding_rows": 234,
        "actual_edited_embedding_rows_reused": 198,
        "development_prompts_with_actual_edited_row": 5762,
        "forget_subjects_with_reused_actual_edited_row": 49,
        "official_evaluation_prompts_seen": 0,
    }
    if not isinstance(exposure_audit, Mapping):
        raise RuntimeError("registry lacks the consumed development exposure audit")
    for key, expected_value in expected_exposure_audit.items():
        if exposure_audit.get(key) != expected_value:
            raise RuntimeError(f"registry development exposure audit mismatch: {key}")
    if "not_blind_evaluation" not in str(exposure_audit.get("data_role", "")):
        raise RuntimeError(
            "development-retain exposure audit lost its consumed-data label"
        )
    forensic_scope = registry.get("v3_5_1_scope")
    expected_forensic_scope = {
        "training_only": True,
        "forensic_replay_only": True,
        "source_v3_5_rejection_hash_bound": True,
        "expected_writer_off_gate_cells": 17300,
        "expected_nonzero_writer_off_gate_cells": 1,
        "expected_source_case_id": 10803,
        "raw_signed_response_matrix_required": True,
        "source_context_index_and_provenance_required": True,
        "detector_group_and_owner_status_required": True,
        "threshold_calibration_prohibited": True,
        "detector_optimizer_construction_prohibited": True,
        "actuator_optimizer_construction_prohibited": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
        "ordinary_existing_weight_materialization_claimed": False,
        "downstream_architecture_change_checkpoint_controls_and_official_audits": (
            "deferred until the collision is identified and a separate fix is "
            "preregistered"
        ),
    }
    if not isinstance(forensic_scope, Mapping):
        raise RuntimeError("registry lacks the V3.5.1 forensic-only scope")
    for key, expected_value in expected_forensic_scope.items():
        if forensic_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5.1 forensic scope mismatch: {key}")
    repair_scope = registry.get("v3_5_2_scope")
    expected_repair_scope = {
        "training_only": True,
        "source_v3_5_1_forensics_hash_bound": True,
        "diagnosed_source_case_id": 10803,
        "diagnosed_source_context_index": 4,
        "diagnosed_detector_case_id": 17353,
        "diagnosed_detector_group_index": 30,
        "diagnosed_owner_group": False,
        "repair_targets_collision_identity_directly": False,
        "all_writer_off_groups_per_context": 50,
        "detector_repair_optimizer_updates": 100,
        "global_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }
    if not isinstance(repair_scope, Mapping):
        raise RuntimeError("registry lacks the V3.5.2 repair scope")
    for key, expected_value in expected_repair_scope.items():
        if repair_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5.2 repair scope mismatch: {key}")
    multilabel_scope = registry.get("v3_5_3_scope")
    expected_multilabel_scope = {
        "training_only": True,
        "source_v3_5_2_rejection_hash_bound": True,
        "duplicate_prompt_sha256": (
            "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
        ),
        "positive_source_case_id": 10472,
        "positive_source_context_index": 1,
        "negative_source_case_id": 19763,
        "negative_source_context_index": 4,
        "shared_detector_case_id": 10472,
        "repair_targets_prompt_identity_directly": False,
        "same_record_positive_negative_conflicts_allowed": False,
        "canonical_hidden_reuse_required": True,
        "global_worst_two_weight": 1.0,
        "detector_repair_optimizer_updates": 100,
        "multilabel_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }
    if not isinstance(multilabel_scope, Mapping):
        raise RuntimeError("registry lacks the V3.5.3 multi-label repair scope")
    for key, expected_value in expected_multilabel_scope.items():
        if multilabel_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5.3 repair scope mismatch: {key}")
    balanced_scope = registry.get("v3_5_4_scope")
    expected_balanced_scope = {
        "training_only": True,
        "source_v3_5_3_rejection_hash_bound": True,
        "source_v3_5_3_owner_gate_passed_records": 29,
        "source_v3_5_3_owner_gate_total_records": 50,
        "source_v3_5_3_positive_failures": 21,
        "source_v3_5_3_negative_failures": 0,
        "source_v3_5_3_writer_off_failures": 0,
        "canonical_hidden_reuse_required": True,
        "canonical_multilabel_semantics_unchanged": True,
        "complete_update_global_tail_optimization_weight": 0.0,
        "per_record_worst_two_retained": True,
        "all_writer_off_groups_per_context": 50,
        "first_update_component_gradient_audit_required": True,
        "detector_repair_optimizer_updates": 100,
        "multilabel_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }
    if not isinstance(balanced_scope, Mapping):
        raise RuntimeError("registry lacks the V3.5.4 balanced repair scope")
    for key, expected_value in expected_balanced_scope.items():
        if balanced_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5.4 repair scope mismatch: {key}")
    width_scope = registry.get("v3_5_5_scope")
    expected_width_scope = {
        "training_only": True,
        "source_v3_5_4_rejection_hash_bound": True,
        "source_v3_5_4_detector_passed_records": 50,
        "source_v3_5_4_detector_total_records": 50,
        "source_v3_5_4_actuator_cap": 1.5,
        "source_v3_5_4_actuator_columns": 200,
        "source_v3_5_4_saturated_columns": 147,
        "source_v3_5_4_positive_reachability_passed": False,
        "exact_detector_tensor_replay_required": True,
        "detector_neurons_per_record": 4,
        "actuator_widths_per_record": [4, 8, 16],
        "detector_actuator_neurons_disjoint": True,
        "nested_actuator_prefixes": True,
        "native_per_column_relative_cap": 1.5,
        "matched_width4_group_budget_controls": [8, 16],
        "matched_controls_used_for_width_selection": False,
        "optimizer_updates_per_arm": 100,
        "smallest_native_passing_width_selected": True,
        "every_fitted_actuator_discarded": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }
    if not isinstance(width_scope, Mapping):
        raise RuntimeError("registry lacks the V3.5.5 actuator-width scope")
    for key, expected_value in expected_width_scope.items():
        if width_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5.5 width scope mismatch: {key}")
    full_scope = registry.get("v3_6_scope")
    expected_full_scope = {
        "training_only": True,
        "source_v3_5_5_success_hash_bound": True,
        "exact_v3_5_4_detector_tensor_replay_required": True,
        "exact_v3_5_5_width16_selection_replay_required": True,
        "detector_neurons_per_record": 4,
        "actuator_neurons_per_record": 16,
        "detector_actuator_neurons_disjoint": True,
        "neuron_layer": 27,
        "native_per_column_relative_cap": 1.5,
        "positive_warm_start_max_updates": 100,
        "positive_warm_start_check_every": 5,
        "positive_warm_start_first_passing_audit_selected": True,
        "warm_start_optimizer_state_retained": True,
        "full_preservation_optimizer_updates": 200,
        "all_records_and_contexts_per_update": True,
        "complete_protected_bank_final_audit_required": True,
        "candidate_checkpoint_requires_every_training_gate": True,
        "rejected_checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited_in_learner": True,
    }
    if not isinstance(full_scope, Mapping):
        raise RuntimeError("registry lacks the V3.6 full-preservation scope")
    for key, expected_value in expected_full_scope.items():
        if full_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.6 full scope mismatch: {key}")
    coherent_scope = registry.get("v3_6_1_scope")
    expected_coherent_scope = {
        "training_only": True,
        "source_v3_6_rejection_hash_bound": True,
        "source_v3_6_positive_warm_start_passed": True,
        "source_v3_6_full_preservation_updates": 200,
        "source_v3_6_candidate_checkpoint_saved": False,
        "source_v3_6_official_evaluation_prompts_seen": 0,
        "source_v3_6_negative_nll_abs_max": 0.125,
        "source_v3_6_negative_nll_acceptance_ceiling": 0.05,
        "exact_v3_5_4_detector_tensor_replay_required": True,
        "exact_v3_5_5_width16_selection_replay_required": True,
        "detector_neurons_per_record": 4,
        "actuator_neurons_per_record": 16,
        "detector_actuator_neurons_disjoint": True,
        "neuron_layer": 27,
        "native_per_column_relative_cap": 1.5,
        "positive_warm_start_max_updates": 100,
        "positive_warm_start_check_every": 5,
        "positive_warm_start_first_passing_audit_selected": True,
        "warm_start_optimizer_state_retained": True,
        "full_preservation_optimizer_updates": 200,
        "negative_training_nll_tolerance": 0.0,
        "negative_acceptance_nll_tolerance": 0.05,
        "negative_preservation_weight": 50.0,
        "raw_negative_occurrences": 465,
        "coherent_preservation_negative_occurrences": 464,
        "excluded_multi_role_negative_occurrences": 1,
        "negative_prompt_precedence": (
            "forget_positive_exact_prompt_precedes_preservation_negative"
        ),
        "negative_precedence_matching_scope": "byte_exact_full_prompt_only",
        "excluded_negative_prompt_sha256": (
            "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
        ),
        "lexical_or_subtoken_overlap_exclusion_prohibited": True,
        "all_records_and_contexts_per_update": True,
        "complete_protected_bank_final_audit_required": True,
        "candidate_checkpoint_requires_every_training_gate": True,
        "rejected_checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited_in_learner": True,
    }
    if not isinstance(coherent_scope, Mapping):
        raise RuntimeError("registry lacks the V3.6.1 coherent-preservation scope")
    for key, expected_value in expected_coherent_scope.items():
        if coherent_scope.get(key) != expected_value:
            raise RuntimeError(f"registry V3.6.1 scope mismatch: {key}")
    detector_revision = registry.get("detector_training_revision")
    expected_detector_revision = {
        "version": "v3.5.4_canonical_multilabel_balanced_tail_repair",
        "mode": "training_only_canonical_prompt_multilabel_balanced_repair",
        "primary_initialization": "frozen_v3_2",
        "source_v3_5_protocol": FROZEN_V3_5_PROTOCOL,
        "source_v3_5_1_protocol": FROZEN_V3_5_1_PROTOCOL,
        "source_v3_5_2_protocol": FROZEN_V3_5_2_PROTOCOL,
        "source_v3_5_3_protocol": FROZEN_V3_5_3_PROTOCOL,
        "source_v3_5_3_owner_gate_passed_records": 29,
        "source_v3_5_3_owner_gate_total_records": 50,
        "source_v3_5_3_positive_failures": 21,
        "source_protocol": FROZEN_DETECTOR_PROTOCOL,
        "source_training_revision": "v3.2",
        "source_gate_passed_records": 50,
        "source_gate_total_records": 50,
        "source_gate_passed": True,
        "source_optimizer_updates": 1000,
        "optimizer_constructed_this_run": True,
        "optimizer_updates_this_run": 100,
        "imported_tensors": ["gate_delta", "up_delta"],
        "source_down_delta_imported": False,
        "current_down_delta_reset_to_zero": True,
        "exact_source_artifact_hash_receipt_required": True,
        "fresh_full_context_source_replay_required_before_repair": True,
        "source_replay_abs_tolerance": FROZEN_DETECTOR_REPLAY_ABS_TOLERANCE,
        "controls_initialization": "train",
        "control_optimizer_updates": 1000,
        "training_input": (
            "canonical_exact_prompt_cached_selected_layer_mlp_input_hidden_states"
        ),
        "cache_dtype": "float32",
        "cache_device": "cpu",
        "cache_scope": [
            "writer_on_positive",
            "writer_on_negative",
            "writer_off_positive",
        ],
        "duplicate_prompt_policy": (
            "one bit-identical hidden-state representative reused across every "
            "source-role occurrence"
        ),
        "prompt_label_semantics": (
            "active groups are every record for which the exact prompt is "
            "registered positive; source-relative negative roles leave those "
            "labels active"
        ),
        "update_coverage": "all_records_accumulated",
        "record_microbatch_argument": "detector_record_batch",
        "records_per_optimizer_update": 50,
        "optimizer_updates_total_per_detector": 100,
        "record_exposures": 5000,
        "positive_context_mode": "all",
        "negative_context_mode": "all",
        "tail_k": 2,
        "positive_objective": (
            "active_label_equal_record_mean_plus_worst_k_squared_shortfall"
        ),
        "negative_objective": (
            "source_owner_equal_record_mean_plus_worst_k_squared_excess"
        ),
        "cross_objective": (
            "inactive_label_equal_record_mean_plus_worst_k_squared_excess"
        ),
        "writer_off_objective": (
            "all_detector_groups_equal_source_record_mean_plus_worst_k_squared_excess"
        ),
        "writer_off_groups_per_context": 50,
        "global_tail_weight": 0.0,
        "global_tail_terms": "diagnostic_only_not_optimized",
        "training_positive_floor": 0.30,
        "training_off_abs_max": 0.15,
        "certificate_positive_floor": 0.25,
        "certificate_off_abs_max": 0.20,
        "certificate_abs_tolerance": 1e-7,
        "certificate_thresholds_unchanged_from_v3_1": True,
        "gradient_normalization": "equal_record_mean_plus_per_record_worst_k",
        "gradient_balance_audit": {
            "optimizer_step": 1,
            "parameter_grad_buffers_mutated": False,
            "components": [
                "positive_write",
                "source_negative",
                "inactive_cross",
                "writer_off_all_groups",
                "positive_consistency",
                "parameter_l2",
                "total",
            ],
        },
        "gradient_clip_frequency": "once_per_optimizer_update",
        "norm_projection_frequency": "once_per_optimizer_update",
        "endpoint_audit_phases": [
            "pre_update",
            "post_adam",
            "post_projection",
            "final_fresh_full_context_certificate",
        ],
        "complete_training_log_required": True,
        "official_evaluation_prompts_seen": 0,
    }
    if not isinstance(detector_revision, Mapping):
        raise RuntimeError("registry lacks the V3.5 detector lineage revision")
    for key, expected_value in expected_detector_revision.items():
        if detector_revision.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5 detector revision mismatch: {key}")

    actuator_revision = registry.get("actuator_training_revision")
    expected_actuator_revision = {
        "version": "v3.6.1",
        "mode": "training_only_width16_coherent_float32_negative_preservation",
        "source_v3_6_protocol": FROZEN_V3_6_PROTOCOL,
        "source_v3_6_rejection_hash_receipt_required": True,
        "source_v3_6_result": {
            "positive_warm_start_passed": True,
            "positive_warm_start_first_passing_step": 35,
            "full_preservation_optimizer_updates": 200,
            "final_logged_positive_failures": 0,
            "final_logged_writer_off_nll_abs_max": 0.0,
            "final_native_negative_nll_abs_max": 0.125,
            "negative_nll_acceptance_ceiling": 0.05,
            "candidate_checkpoint_saved": False,
            "official_evaluation_prompts_seen": 0,
        },
        "source_v3_5_5_protocol": FROZEN_V3_5_5_PROTOCOL,
        "source_v3_5_5_success_hash_receipt_required": True,
        "source_v3_5_5_detector_tensor_replay_required": True,
        "source_v3_5_5_width16_selection_replay_required": True,
        "frozen_training_contexts": {
            "writer_on_positive": 346,
            "writer_on_negative": 465,
            "writer_off_positive": 346,
        },
        "source_v3_5_5_success": {
            "selected_smallest_positive_reachable_width": 16,
            "selected_smallest_mechanism_ready_width": 16,
            "native_per_column_relative_cap": 1.5,
            "native_width16_positive_reachable": True,
            "native_width16_writer_off_structural_selectivity_passed": True,
            "native_width16_actuator_columns": 800,
            "native_width16_saturated_columns": 34,
            "matched_width4_group_budget_width16_positive_reachable": False,
            "every_fitted_actuator_discarded": True,
            "full_preservation_objective_started": False,
            "official_evaluation_prompts_seen": 0,
        },
        "architecture": {
            "base_mlp_path": "bit_exact_untouched",
            "frozen_detector_features_per_record": 4,
            "actuator_features_per_record": 16,
            "detector_actuator_feature_sets_disjoint": True,
            "actuator_selection": "exact_v3_5_5_nested_width16_prefix",
            "actuator_gate_up_rows": "frozen_unmodified_Base_rows",
            "residual_formula": (
                "BaseMLP(h) + threshold_gate_r(h) * BaseActuatorFeatures_r(h) @ "
                "down_delta_r.T"
            ),
            "off_boundary": 0.200001,
            "on_boundary": 0.249999,
            "threshold_gate": (
                "clip((response - off_boundary) / (on_boundary - off_boundary), 0, 1)"
            ),
            "down_delta_exact_zero_is_algebraic_identity": True,
            "original_base_down_columns_modified": False,
            "ordinary_existing_weight_materialization": False,
        },
        "positive_warm_start": {
            "initial_down_delta": "bit_exact_zero",
            "optimizer_state": "fresh_adamw_retained_into_full_preservation",
            "maximum_optimizer_updates": 100,
            "audit_every": 5,
            "stopping_rule": (
                "first fresh full-context audit with direct_failures == 0 and "
                "positive_failures == 0"
            ),
            "learning_rate": 5e-4,
            "records_per_optimizer_update": 50,
            "positive_contexts_per_optimizer_update": 346,
            "context_microbatch_capacity": 4,
            "tail_k": 2,
            "objective": (
                "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
            ),
            "forget_margin": 1.0,
            "gradient_clip_frequency": "once_per_optimizer_update",
            "norm_projection_frequency": "once_per_optimizer_update",
            "complete_incremental_training_log_required": True,
            "fresh_full_context_audit_required": True,
            "zero_actuator_identity_abs_tolerance": 1e-6,
            "native_per_column_relative_cap": 1.5,
        },
        "full_preservation": {
            "optimizer_updates": 200,
            "optimizer_state_reinitialized_after_warm_start": False,
            "records_per_optimizer_update": 50,
            "positive_contexts_per_optimizer_update": 346,
            "raw_negative_occurrences": 465,
            "negative_contexts_per_optimizer_update": 464,
            "excluded_multi_role_negative_occurrences": 1,
            "negative_prompt_precedence": (
                "forget_positive_exact_prompt_precedes_preservation_negative"
            ),
            "negative_precedence_matching_scope": "byte_exact_full_prompt_only",
            "excluded_negative_prompt_sha256": (
                "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
            ),
            "lexical_or_subtoken_overlap_exclusion_prohibited": True,
            "writer_off_contexts_per_optimizer_update": 346,
            "protected_contexts_per_optimizer_update": 80,
            "protected_prompt_bank": 8192,
            "protected_sampling_seed_offset": 78103,
            "positive_margin_weight": 20.0,
            "positive_reference_nll_weight": 50.0,
            "negative_preservation_weight": 50.0,
            "negative_training_nll_tolerance": 0.0,
            "negative_acceptance_nll_tolerance": 0.05,
            "negative_preservation_objective": (
                "on the coherent 464-occurrence preservation bank, equal-record "
                "mean-plus-worst-two squared absolute float32-current minus "
                "float32-baseline NLL drift with a zero training target; exact "
                "native-dtype acceptance remains at 0.05"
            ),
            "negative_audit_numerics": (
                "report differentiable float32 and exact official-compatible "
                "native-dtype drift separately"
            ),
            "writer_off_weight": 50.0,
            "protected_kl_weight": 20.0,
            "actuator_l2": 1e-4,
            "forget_margin": 1.0,
            "reference_nll_regression_max": 0.05,
            "negative_nll_abs_max": 0.05,
            "writer_off_nll_abs_max": 0.05,
            "protected_kl_mean_max": 0.05,
            "protected_kl_absolute_max": 0.5,
            "native_per_column_relative_cap": 1.5,
            "candidate_saved_only_if_all_training_gates_pass": True,
            "rejected_checkpoint_saved": False,
            "official_evaluation_allowed": False,
        },
        "official_evaluation_prompts_seen": 0,
    }
    if not isinstance(actuator_revision, Mapping):
        raise RuntimeError("registry lacks the V3.6.1 actuator-training revision")
    for key, expected_value in expected_actuator_revision.items():
        if actuator_revision.get(key) != expected_value:
            raise RuntimeError(f"registry V3.6.1 actuator revision mismatch: {key}")
    ownership_binding = registry.get("selected_neuron_ownership_binding")
    expected_ownership_binding = {
        "scope": "primary_embedding_keyed_configuration",
        "source_runs": [
            "v3",
            "v3.1",
            "v3.2",
            "v3.3",
            "v3.4",
            "v3.5",
            "v3.5.1",
            "v3.5.2",
            "v3.5.3",
            "v3.5.4",
        ],
        "jq_projection": "[.ownership[].selected_neurons]",
        "jq_compact_sha256": (
            "acc3cc05868483f6c40a8909fca064b59c4ec4d000a76cf1ece6c3e818c750d1"
        ),
        "writer_configuration": {
            "row_norm_cap": 8.0,
            "row_norm_cap_frequency_alpha": 0.15,
            "max_subject_token_frequency": 1000000000,
        },
        "required": True,
    }
    if not isinstance(ownership_binding, Mapping):
        raise RuntimeError("registry lacks the V3.5 selected-neuron binding")
    for key, expected_value in expected_ownership_binding.items():
        if ownership_binding.get(key) != expected_value:
            raise RuntimeError(f"registry V3.5 neuron binding mismatch: {key}")
    label = str(args.experiment_label)
    if label == "primary":
        expected = registry.get("primary_configuration")
        if not isinstance(expected, Mapping):
            raise RuntimeError("registry lacks primary configuration")
    else:
        changes = None
        rows = registry.get("controlled_training_ablations", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and str(row.get("name")) == label:
                    changes = row.get("change_only")
                    break
        if changes is None and label.startswith("neurons_per_record_"):
            changes = {"neurons_per_record": int(label.rsplit("_", 1)[-1])}
        if changes is None and label.startswith("layer_"):
            changes = {"neuron_layer": int(label.rsplit("_", 1)[-1])}
        if not isinstance(changes, Mapping):
            raise RuntimeError(f"experiment label is not preregistered: {label!r}")
        primary = registry.get("primary_configuration")
        common = registry.get("ablation_common_configuration", {})
        if not isinstance(primary, Mapping) or not isinstance(common, Mapping):
            raise RuntimeError("registry lacks primary/ablation-common configuration")
        expected = {**primary, **common, **changes}
    for key, registered in expected.items():
        if not hasattr(args, str(key)):
            raise RuntimeError(f"registry key has no learner argument: {key!r}")
        observed = getattr(args, str(key))
        if isinstance(registered, float):
            matches = math.isclose(float(observed), float(registered), abs_tol=1e-12)
        else:
            matches = observed == registered
        if not matches:
            raise RuntimeError(
                f"experiment {label!r} diverges from registry: {key}="
                f"{observed!r}, expected {registered!r}"
            )


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    return compositional_method.distribution([float(value) for value in values])


def _tensor_digest(tensor: torch.Tensor, *, row_chunk: int = 256) -> str:
    digest = hashlib.sha256()
    detached = tensor.detach()
    for start in range(0, int(detached.shape[0]), int(row_chunk)):
        block = detached[start : start + int(row_chunk)].contiguous().cpu()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def selected_neuron_ownership_jq_compact_sha256(
    ownership: Sequence[Sequence[int]],
) -> str:
    """Match ``jq -c '[.ownership[].selected_neurons]' | sha256sum``."""

    normalized = [[int(neuron) for neuron in group] for group in ownership]
    compact_with_newline = json.dumps(normalized, separators=(",", ":")) + "\n"
    return hashlib.sha256(compact_with_newline.encode("utf-8")).hexdigest()


def compare_detector_gate_replays(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    abs_tolerance: float,
) -> Dict[str, Any]:
    """Compare two fresh evaluations of the same full detector certificate."""

    if float(abs_tolerance) < 0:
        raise ValueError("abs_tolerance must be non-negative")
    first_rows = first.get("per_record")
    second_rows = second.get("per_record")
    if not isinstance(first_rows, list) or not isinstance(second_rows, list):
        raise ValueError("detector gate replays must contain per_record lists")
    if len(first_rows) != len(second_rows):
        return {
            "record_count_match": False,
            "record_binding_match": False,
            "decisions_match": False,
            "metric_abs_max": float("inf"),
            "passed": False,
        }
    metric_abs_max = 0.0
    record_binding_match = True
    decisions_match = True
    metric_keys = ("positive_min", "negative_abs_max", "writer_off_abs_max")
    decision_keys = (
        "positive_passed",
        "negative_passed",
        "writer_off_passed",
        "passed",
    )
    for first_row, second_row in zip(first_rows, second_rows):
        if not isinstance(first_row, Mapping) or not isinstance(second_row, Mapping):
            raise ValueError("detector gate replay rows must be mappings")
        record_binding_match = bool(
            record_binding_match
            and first_row.get("record_index") == second_row.get("record_index")
            and first_row.get("case_id") == second_row.get("case_id")
        )
        metric_abs_max = max(
            metric_abs_max,
            *(
                abs(float(first_row[key]) - float(second_row[key]))
                for key in metric_keys
            ),
        )
        decisions_match = bool(
            decisions_match
            and all(
                bool(first_row.get(key)) == bool(second_row.get(key))
                for key in decision_keys
            )
        )
    return {
        "record_count_match": True,
        "record_binding_match": record_binding_match,
        "decisions_match": decisions_match,
        "metric_abs_max": metric_abs_max,
        "passed": bool(
            record_binding_match
            and decisions_match
            and metric_abs_max <= float(abs_tolerance)
        ),
    }


def compare_actuator_audits(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    abs_tolerance: float,
) -> Dict[str, Any]:
    """Compare two full-context audits of the same actuator parameter state."""

    if float(abs_tolerance) < 0:
        raise ValueError("abs_tolerance must be non-negative")
    count_keys = ("direct_failures", "positive_failures", "positive_contexts")
    metric_keys = [
        "minimum_margin",
        "reference_nll_regression_max",
        "writer_off_nll_abs_max",
    ]
    for optional_key in (
        "negative_nll_abs_max",
        "negative_float32_nll_abs_max",
    ):
        first_has = optional_key in first
        second_has = optional_key in second
        if first_has != second_has:
            return {
                "counts_match": False,
                "decisions_match": False,
                "metric_abs_max": float("inf"),
                "passed": False,
            }
        if first_has:
            metric_keys.append(optional_key)
    counts_match = all(first.get(key) == second.get(key) for key in count_keys)
    metric_abs_max = 0.0
    metrics_present = True
    for key in metric_keys:
        if key not in first or key not in second:
            metrics_present = False
            metric_abs_max = float("inf")
            break
        metric_abs_max = max(
            metric_abs_max, abs(float(first[key]) - float(second[key]))
        )

    first_rows = first.get("per_record")
    second_rows = second.get("per_record")
    per_record_metric_keys = (
        "direct_margin",
        "positive_min",
        "reference_nll_regression_max",
        "writer_off_nll_abs_max",
    )
    record_binding_match = bool(
        isinstance(first_rows, list)
        and isinstance(second_rows, list)
        and len(first_rows) == len(second_rows)
        and all(
            int(left.get("record_index", -1)) == int(right.get("record_index", -2))
            and int(left.get("case_id", -1)) == int(right.get("case_id", -2))
            for left, right in zip(first_rows, second_rows)
        )
    )
    per_record_metrics_match = record_binding_match
    if record_binding_match:
        for left, right in zip(first_rows, second_rows):
            if left.get("positive_contexts") != right.get(
                "positive_contexts"
            ) or left.get("positive_failures") != right.get("positive_failures"):
                per_record_metrics_match = False
                break
            for key in per_record_metric_keys:
                if key not in left or key not in right:
                    per_record_metrics_match = False
                    metric_abs_max = float("inf")
                    break
                metric_abs_max = max(
                    metric_abs_max, abs(float(left[key]) - float(right[key]))
                )
            if not per_record_metrics_match:
                break
    passed = bool(
        counts_match
        and metrics_present
        and record_binding_match
        and per_record_metrics_match
        and metric_abs_max <= float(abs_tolerance)
    )
    return {
        "counts_match": counts_match,
        "record_binding_match": record_binding_match,
        "per_record_metrics_match": per_record_metrics_match,
        "metrics_present": metrics_present,
        "metric_abs_max": metric_abs_max,
        "abs_tolerance": float(abs_tolerance),
        "passed": passed,
    }


def import_frozen_v3_2_detector(
    run_dir: Path,
    *,
    stage1_path: Path,
    case_ids: Sequence[int],
    layer: int,
    ownership: Sequence[Sequence[int]],
    selected_neurons: Sequence[int],
    flat_signs: torch.Tensor,
    output_head_sha256: str,
    editor: neuron_core.SparseSwiGLUNeuronEditor,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load the exact V3.2 gate/up tensors while discarding its failed actuator."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "state": method_dir / "embedding_keyed_neuron_state.pt",
        "gate": method_dir / "detector_gate_report.json",
        "endpoint_audit": method_dir / "detector_endpoint_audit.json",
        "selection": method_dir / "neuron_selection_report.json",
        "summary": method_dir / "embedding_keyed_neuron_summary.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.2 detector lacks {name}: {path}")

    state = torch.load(paths["state"], map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("frozen V3.2 detector state must be a mapping")
    gate = _load_json(paths["gate"])
    endpoint = _load_json(paths["endpoint_audit"])
    selection = _load_json(paths["selection"])
    summary = _load_json(paths["summary"])
    firewall = _load_json(paths["firewall"])

    expected_ownership = [list(map(int, group)) for group in ownership]
    expected_selected = [int(neuron) for neuron in selected_neurons]
    expected_hash = selected_neuron_ownership_jq_compact_sha256(expected_ownership)
    gate_criterion = gate.get("criterion", {})
    checks = {
        "state_protocol": state.get("protocol") == FROZEN_DETECTOR_PROTOCOL,
        "summary_protocol": summary.get("protocol") == FROZEN_DETECTOR_PROTOCOL,
        "gate_protocol": gate.get("protocol") == FROZEN_DETECTOR_PROTOCOL,
        "detector_revision": state.get("detector_training_revision") == "v3.2",
        "case_ids": [int(value) for value in state.get("case_ids", [])]
        == [int(value) for value in case_ids],
        "layer": int(state.get("layer", -1)) == int(layer),
        "selected_neurons": [int(value) for value in state.get("selected_neurons", [])]
        == expected_selected,
        "ownership": state.get("ownership") == expected_ownership,
        "ownership_hash": str(
            state.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == expected_hash,
        "selection_hash": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == expected_hash,
        "source_writer": str(state.get("source_stage1_state_sha256") or "")
        == compositional_method.sha256_file(stage1_path),
        "output_head": str(state.get("output_head_sha256") or "")
        == str(output_head_sha256),
        "detector_gate": bool(gate.get("passed"))
        and int(gate.get("passed_records", -1)) == len(case_ids)
        and int(gate.get("total_records", -1)) == len(case_ids),
        "detector_gate_criterion": bool(
            isinstance(gate_criterion, Mapping)
            and math.isclose(
                float(gate_criterion.get("positive_floor", float("nan"))),
                0.25,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(gate_criterion.get("negative_abs_max", float("nan"))),
                0.20,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(gate_criterion.get("writer_off_abs_max", float("nan"))),
                0.20,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(gate_criterion.get("comparison_abs_tolerance", float("nan"))),
                1e-7,
                abs_tol=1e-12,
            )
            and gate_criterion.get("writer_off_required") is True
        ),
        "endpoint_audit": bool(endpoint.get("complete")),
        "summary_detector_gate": bool(
            summary.get("acceptance", {}).get("detector_gate_passed")
        ),
        "source_run_rejected": summary.get("acceptance", {}).get("passed") is False,
        "source_checkpoint_not_saved": (
            summary.get("acceptance", {}).get("checkpoint_saved") is False
        ),
        "official_gate_prompts_zero": gate.get("official_evaluation_prompts_seen") == 0,
        "official_endpoint_prompts_zero": endpoint.get(
            "official_evaluation_prompts_seen"
        )
        == 0,
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    source_signs = state.get("flat_signs")
    checks["flat_signs"] = bool(
        isinstance(source_signs, torch.Tensor)
        and source_signs.shape == flat_signs.shape
        and torch.equal(
            source_signs.detach().float().cpu(), flat_signs.detach().float().cpu()
        )
    )
    gate_delta = state.get("gate_delta")
    up_delta = state.get("up_delta")
    source_base = state.get("base_neuron_weights")
    checks["base_neuron_weights"] = bool(
        isinstance(source_base, Mapping)
        and all(
            isinstance(source_base.get(name), torch.Tensor)
            and source_base[name].shape == expected.shape
            and torch.equal(
                source_base[name].detach().float().cpu(),
                expected.detach().float().cpu(),
            )
            for name, expected in (
                ("gate_rows", editor.base_gate_rows),
                ("up_rows", editor.base_up_rows),
                ("down_columns", editor.base_down_columns),
            )
        )
    )
    checks["gate_delta"] = bool(
        isinstance(gate_delta, torch.Tensor)
        and gate_delta.shape == editor.gate_delta.shape
        and torch.isfinite(gate_delta.float()).all()
    )
    checks["up_delta"] = bool(
        isinstance(up_delta, torch.Tensor)
        and up_delta.shape == editor.up_delta.shape
        and torch.isfinite(up_delta.float()).all()
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.2 detector lineage failed: " + ", ".join(sorted(failed))
        )

    with torch.no_grad():
        editor.gate_delta.copy_(gate_delta.to(editor.gate_delta.device).float())
        editor.up_delta.copy_(up_delta.to(editor.up_delta.device).float())
        editor.down_delta.zero_()

    receipt = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_2_detector_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_DETECTOR_PROTOCOL,
        "target_protocol": PROTOCOL,
        "case_ids": [int(value) for value in case_ids],
        "layer": int(layer),
        "selected_neuron_ownership_jq_compact_sha256": expected_hash,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "checks": checks,
        "down_delta_imported": False,
        "down_delta_reset_to_zero": True,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }
    return receipt, gate


def validate_frozen_v3_3_rejection(
    run_dir: Path,
    *,
    case_ids: Sequence[int],
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Hash-bind the exact V3.3 training-only actuator rejection."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "gate": method_dir / "detector_gate_report.json",
        "endpoint_audit": method_dir / "detector_endpoint_audit.json",
        "selection": method_dir / "neuron_selection_report.json",
        "feasibility": method_dir / "actuator_positive_only_feasibility.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.3 rejection lacks {name}: {path}")

    gate = _load_json(paths["gate"])
    endpoint = _load_json(paths["endpoint_audit"])
    selection = _load_json(paths["selection"])
    feasibility = _load_json(paths["feasibility"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    audit = feasibility.get("final_audit", {})
    norms = audit.get("norms", {}) if isinstance(audit, Mapping) else {}
    log = feasibility.get("complete_training_log")
    saturation_step = None
    if isinstance(log, list):
        for row in log:
            if (
                isinstance(row, Mapping)
                and float(row.get("down_max_relative_norm", float("-inf")))
                >= 0.50 - 1e-6
            ):
                saturation_step = int(row.get("step", -1))
                break
    gate_rows = gate.get("per_record")
    checks = {
        "gate_protocol": gate.get("protocol") == FROZEN_V3_3_PROTOCOL,
        "endpoint_protocol": endpoint.get("protocol") == FROZEN_V3_3_PROTOCOL,
        "feasibility_protocol": feasibility.get("protocol") == FROZEN_V3_3_PROTOCOL,
        "gate_passed": bool(gate.get("passed"))
        and int(gate.get("passed_records", -1)) == len(case_ids)
        and int(gate.get("total_records", -1)) == len(case_ids),
        "case_ids": bool(
            isinstance(gate_rows, list)
            and [int(row.get("case_id", -1)) for row in gate_rows]
            == [int(value) for value in case_ids]
        ),
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "feasibility_rejected": feasibility.get("passed") is False,
        "relative_norm_cap": math.isclose(
            float(feasibility.get("relative_norm_cap", float("nan"))),
            0.50,
            abs_tol=1e-12,
        ),
        "optimizer_steps": int(feasibility.get("optimizer_steps_recorded", -1)) == 100,
        "complete_log": isinstance(log, list) and len(log) == 100,
        "cap_saturated_by_step": bool(
            saturation_step is not None and saturation_step <= 15
        ),
        "direct_failures": int(audit.get("direct_failures", -1)) == 40,
        "positive_failures": int(audit.get("positive_failures", -1)) == 300,
        "positive_contexts": int(audit.get("positive_contexts", -1)) == 346,
        "minimum_margin": math.isclose(
            float(audit.get("minimum_margin", float("nan"))),
            -8.75,
            abs_tol=1e-12,
        ),
        "down_cap_achieved": math.isclose(
            float(norms.get("down_max_relative_norm", float("nan"))),
            0.50,
            abs_tol=1e-6,
        ),
        "reference_regression": math.isclose(
            float(audit.get("reference_nll_regression_max", float("nan"))),
            2.375,
            abs_tol=1e-12,
        ),
        "writer_off_drift": math.isclose(
            float(audit.get("writer_off_nll_abs_max", float("nan"))),
            4.8125,
            abs_tol=1e-12,
        ),
        "full_training_not_started": rejection.get("full_actuator_training_started")
        is False,
        "checkpoint_not_saved": rejection.get("checkpoint_saved") is False,
        "official_refused": rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (gate, endpoint, feasibility, rejection)
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.3 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_3_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_3_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "observed": {
            "detector_passed_records": int(gate["passed_records"]),
            "detector_total_records": int(gate["total_records"]),
            "cap_saturation_step": saturation_step,
            "direct_failures": int(audit["direct_failures"]),
            "positive_failures": int(audit["positive_failures"]),
            "positive_contexts": int(audit["positive_contexts"]),
            "minimum_margin": float(audit["minimum_margin"]),
            "reference_nll_regression_max": float(
                audit["reference_nll_regression_max"]
            ),
            "writer_off_nll_abs_max": float(audit["writer_off_nll_abs_max"]),
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_4_rejection(
    run_dir: Path,
    *,
    case_ids: Sequence[int],
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Hash-bind the V3.4 reachability-pass/selectivity-fail result."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "gate": method_dir / "detector_gate_report.json",
        "endpoint_audit": method_dir / "detector_endpoint_audit.json",
        "selection": method_dir / "neuron_selection_report.json",
        "sweep": method_dir / "actuator_norm_cap_reachability_sweep.json",
        "selectivity": method_dir / "frozen_detector_selectivity_audit.json",
        "zero_actuator": (
            method_dir / "frozen_detector_zero_actuator_behavior_audit.json"
        ),
        "completion": method_dir / "training_only_sweep_completion.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.4 rejection lacks {name}: {path}")

    gate = _load_json(paths["gate"])
    endpoint = _load_json(paths["endpoint_audit"])
    selection = _load_json(paths["selection"])
    sweep = _load_json(paths["sweep"])
    selectivity = _load_json(paths["selectivity"])
    zero_actuator = _load_json(paths["zero_actuator"])
    completion = _load_json(paths["completion"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    gate_rows = gate.get("per_record")
    response_ratio = (
        selectivity.get("aggregate", {})
        .get("owned_signed_group_response", {})
        .get("writer_on_to_off_ratio", {})
    )
    cap_rows = sweep.get("cap_artifacts")
    cap_150 = (
        next(
            (
                row
                for row in cap_rows
                if isinstance(row, Mapping)
                and math.isclose(float(row.get("cap", float("nan"))), 1.5)
            ),
            None,
        )
        if isinstance(cap_rows, list)
        else None
    )
    checks = {
        "gate_protocol": gate.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "endpoint_protocol": endpoint.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "sweep_protocol": sweep.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "selectivity_protocol": selectivity.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "zero_actuator_protocol": zero_actuator.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "completion_protocol": completion.get("protocol") == FROZEN_V3_4_PROTOCOL,
        "gate_passed": bool(gate.get("passed"))
        and int(gate.get("passed_records", -1)) == len(case_ids)
        and int(gate.get("total_records", -1)) == len(case_ids),
        "case_ids": bool(
            isinstance(gate_rows, list)
            and [int(row.get("case_id", -1)) for row in gate_rows]
            == [int(value) for value in case_ids]
        ),
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "registered_caps": sweep.get("caps_preregistered")
        == [0.5, 0.75, 1.0, 1.5, 2.0],
        "completed_caps": sweep.get("caps_completed") == [0.5, 0.75, 1.0, 1.5, 2.0],
        "positive_reachability": sweep.get("positive_reachability_passed") is True
        and math.isclose(
            float(sweep.get("selected_smallest_positive_reachable_cap", float("nan"))),
            1.5,
            abs_tol=1e-12,
        )
        and isinstance(cap_150, Mapping)
        and cap_150.get("positive_reachable") is True
        and int(cap_150.get("down_saturated_columns", -1)) == 34,
        "structural_rejection": sweep.get("mechanism_readiness_passed") is False
        and sweep.get("structural_selectivity_diagnostic", {}).get("passed") is False
        and sweep.get("conclusion")
        == "positive_reachability_found_but_structural_writer_off_selectivity_failed",
        "zero_actuator_drift": math.isclose(
            float(zero_actuator.get("writer_off_nll_abs_max", float("nan"))),
            0.25,
            abs_tol=1e-12,
        )
        and zero_actuator.get("writer_off_preservation_passed") is False,
        "weak_response_ratio": bool(
            isinstance(response_ratio, Mapping)
            and round(float(response_ratio.get("p10", float("nan"))), 4) == 2.1545
            and round(float(response_ratio.get("median", float("nan"))), 4) == 3.2468
        ),
        "all_weights_discarded": sweep.get("all_fitted_weights_discarded") is True,
        "detector_unchanged": sweep.get("frozen_detector_tensors_unchanged") is True,
        "full_training_not_started": sweep.get("full_preservation_objective_started")
        is False,
        "checkpoint_not_saved": sweep.get("checkpoint_saved") is False,
        "completion_matches": completion.get("positive_reachability_passed") is True
        and completion.get("mechanism_readiness_passed") is False
        and math.isclose(
            float(
                completion.get("selected_smallest_positive_reachable_cap", float("nan"))
            ),
            1.5,
            abs_tol=1e-12,
        ),
        "rejection_reason": rejection.get("stage")
        == "actuator_norm_cap_reachability_sweep"
        and rejection.get("reason")
        == "positive_reachability_without_structural_writer_off_selectivity",
        "official_refused": sweep.get("official_evaluation_allowed") is False
        and completion.get("official_evaluation_allowed") is False
        and rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (
                gate,
                endpoint,
                sweep,
                selectivity,
                zero_actuator,
                completion,
                rejection,
            )
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.4 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_4_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_4_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "observed": {
            "detector_passed_records": int(gate["passed_records"]),
            "detector_total_records": int(gate["total_records"]),
            "response_ratio_p10": float(response_ratio["p10"]),
            "response_ratio_median": float(response_ratio["median"]),
            "zero_actuator_writer_off_nll_abs_max": float(
                zero_actuator["writer_off_nll_abs_max"]
            ),
            "selected_smallest_positive_reachable_cap": float(
                sweep["selected_smallest_positive_reachable_cap"]
            ),
            "mechanism_readiness_passed": bool(sweep["mechanism_readiness_passed"]),
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_rejection(
    run_dir: Path,
    *,
    case_ids: Sequence[int],
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Hash-bind the single-cell global writer-off rejection from V3.5."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "gate": method_dir / "detector_gate_report.json",
        "endpoint_audit": method_dir / "detector_endpoint_audit.json",
        "selection": method_dir / "neuron_selection_report.json",
        "threshold_gate": method_dir / "isolated_threshold_gate_report.json",
        "selectivity": method_dir / "frozen_detector_selectivity_audit.json",
        "v3_2_import": method_dir / "frozen_v3_2_detector_import.json",
        "v3_4_import": method_dir / "frozen_v3_4_rejection_import.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5 rejection lacks {name}: {path}")

    gate = _load_json(paths["gate"])
    endpoint = _load_json(paths["endpoint_audit"])
    selection = _load_json(paths["selection"])
    threshold = _load_json(paths["threshold_gate"])
    selectivity = _load_json(paths["selectivity"])
    v3_2_import = _load_json(paths["v3_2_import"])
    v3_4_import = _load_json(paths["v3_4_import"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    gate_rows = gate.get("per_record")
    checks_payload = threshold.get("checks", {})
    aggregate = threshold.get("aggregate", {})
    writer_off = aggregate.get("writer_off_gate", {})
    positive_owner = aggregate.get("positive_owner_gate", {})
    positive_cross = aggregate.get("positive_cross_gate", {})
    negative = aggregate.get("negative_gate", {})
    threshold_rows = threshold.get("per_record")
    failing_writer_off_rows = (
        [
            row
            for row in threshold_rows
            if isinstance(row, Mapping)
            and float(row.get("writer_off_gate_abs_max", 0.0)) > 1e-6
        ]
        if isinstance(threshold_rows, list)
        else []
    )
    writer_off_count = int(writer_off.get("n", -1))
    writer_off_max = float(writer_off.get("max", float("nan")))
    writer_off_sum = float(writer_off.get("mean", float("nan"))) * writer_off_count
    checks = {
        "gate_protocol": gate.get("protocol") == FROZEN_V3_5_PROTOCOL,
        "endpoint_protocol": endpoint.get("protocol") == FROZEN_V3_5_PROTOCOL,
        "threshold_protocol": threshold.get("protocol") == FROZEN_V3_5_PROTOCOL,
        "selectivity_protocol": selectivity.get("protocol") == FROZEN_V3_5_PROTOCOL,
        "gate_passed": bool(gate.get("passed"))
        and int(gate.get("passed_records", -1)) == len(case_ids)
        and int(gate.get("total_records", -1)) == len(case_ids),
        "case_ids": bool(
            isinstance(gate_rows, list)
            and [int(row.get("case_id", -1)) for row in gate_rows]
            == [int(value) for value in case_ids]
        ),
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "threshold_boundaries": math.isclose(
            float(
                threshold.get("boundaries", {}).get(
                    "runtime_off_boundary", float("nan")
                )
            ),
            0.200001,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(
                threshold.get("boundaries", {}).get("runtime_on_boundary", float("nan"))
            ),
            0.249999,
            abs_tol=1e-12,
        ),
        "only_writer_off_check_failed": checks_payload
        == {
            "all_positive_owner_gates_one": True,
            "all_positive_cross_gates_zero": True,
            "all_negative_gates_zero": True,
            "all_writer_off_gates_zero": False,
        },
        "positive_owner_all_one": int(positive_owner.get("n", -1)) == 346
        and math.isclose(float(positive_owner.get("min", float("nan"))), 1.0),
        "positive_cross_all_zero": int(positive_cross.get("n", -1)) == 16954
        and math.isclose(float(positive_cross.get("max", float("nan"))), 0.0),
        "negative_all_zero": int(negative.get("n", -1)) == 23250
        and math.isclose(float(negative.get("max", float("nan"))), 0.0),
        "writer_off_single_cell": writer_off_count == 17300
        and math.isclose(
            writer_off_max,
            0.5735465884208679,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            writer_off_sum,
            writer_off_max,
            rel_tol=0.0,
            abs_tol=1e-10,
        ),
        "single_source_record": len(failing_writer_off_rows) == 1
        and int(failing_writer_off_rows[0].get("case_id", -1)) == 10803,
        "threshold_rejected": threshold.get("passed") is False,
        "v3_2_import": v3_2_import.get("passed") is True,
        "v3_4_import": v3_4_import.get("passed") is True,
        "rejection_reason": rejection.get("stage") == "isolated_threshold_gate"
        and rejection.get("reason")
        == "frozen_detector_gap_did_not_map_to_exact_branch_gate",
        "actuator_not_started": rejection.get("actuator_training_started") is False,
        "checkpoint_not_saved": rejection.get("checkpoint_saved") is False,
        "official_refused": rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (
                gate,
                endpoint,
                threshold,
                selectivity,
                v3_2_import,
                v3_4_import,
                rejection,
            )
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "observed": {
            "detector_passed_records": int(gate["passed_records"]),
            "writer_off_gate_cells": writer_off_count,
            "nonzero_writer_off_gate_cells": 1,
            "writer_off_gate_max": writer_off_max,
            "writer_off_source_case_id": 10803,
            "actuator_training_started": False,
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_1_forensics(
    run_dir: Path,
    *,
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Hash-bind the exact V3.5.1 non-owner collision diagnosis."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "selection": method_dir / "neuron_selection_report.json",
        "global_gate": method_dir / "isolated_threshold_gate_report.json",
        "forensics": method_dir / "v3_5_collision_forensics.json",
        "completion": method_dir / "training_only_v3_5_1_completion.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5.1 forensics lacks {name}: {path}")

    selection = _load_json(paths["selection"])
    global_gate = _load_json(paths["global_gate"])
    forensics = _load_json(paths["forensics"])
    completion = _load_json(paths["completion"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    collision = forensics.get("single_writer_off_collision", {})
    gate_counts = global_gate.get("gate_endpoint_violation_counts", {})
    expected_counts = {
        "positive_owner": 0,
        "positive_cross": 0,
        "negative": 0,
        "writer_off": 1,
    }
    exact_identity = {
        "source_case_id": 10803,
        "source_context_index": 4,
        "detector_case_id": 17353,
        "detector_group_index": 30,
        "owner_group": False,
    }
    checks = {
        "protocol": completion.get("protocol") == FROZEN_V3_5_1_PROTOCOL
        and forensics.get("protocol") == FROZEN_V3_5_1_PROTOCOL
        and global_gate.get("protocol") == FROZEN_V3_5_1_PROTOCOL,
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "single_gate_violation": gate_counts == expected_counts,
        "completion_result": completion.get("complete") is True
        and completion.get("result") == "single_v3_5_writer_off_collision_identified"
        and completion.get("diagnosis")
        == "cross_record_detector_fired_without_embedding_writer",
        "collision_identity": all(
            completion.get(key) == expected and collision.get(key) == expected
            for key, expected in exact_identity.items()
        ),
        "collision_value": math.isclose(
            float(completion.get("raw_signed_response", float("nan"))),
            0.2286771833896637,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(completion.get("runtime_gate", float("nan"))),
            0.5735465884208679,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "ownerwise_consistent": forensics.get("ownerwise_certificate_consistent")
        is True,
        "no_optimizers": completion.get("detector_optimizer_constructed") is False
        and completion.get("actuator_optimizer_constructed") is False,
        "threshold_unchanged": completion.get("threshold_changed") is False,
        "tensors_unchanged": completion.get("detector_tensors_changed") is False
        and completion.get("actuator_tensors_changed") is False,
        "checkpoint_not_saved": completion.get("checkpoint_saved") is False,
        "rejection_preserved": rejection.get("stage") == "v3.5_collision_forensics"
        and rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (global_gate, forensics, completion, rejection)
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5.1 forensic lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_1_forensics_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_1_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "diagnosis": "cross_record_detector_fired_without_embedding_writer",
        "collision": {
            **exact_identity,
            "raw_signed_response": 0.2286771833896637,
            "runtime_gate": 0.5735465884208679,
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_2_rejection(
    run_dir: Path,
    *,
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Bind the exact duplicate-prompt contradiction exposed by V3.5.2."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "selection": method_dir / "neuron_selection_report.json",
        "owner_gate": method_dir / "detector_gate_report.json",
        "global_gate": (
            method_dir / "detector_step_100_post_projection_global_isolation_gate.json"
        ),
        "training_log": method_dir / "detector_training_log.json",
        "endpoint": method_dir / "detector_endpoint_audit.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5.2 rejection lacks {name}: {path}")

    selection = _load_json(paths["selection"])
    owner_gate = _load_json(paths["owner_gate"])
    global_gate = _load_json(paths["global_gate"])
    training_log = _load_json(paths["training_log"])
    endpoint = _load_json(paths["endpoint"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    positive_cells = global_gate.get("violating_cells", {}).get("positive_owner", [])
    negative_cells = global_gate.get("violating_cells", {}).get("negative", [])
    writer_off_cells = global_gate.get("violating_cells", {}).get("writer_off", [])
    duplicate_digest = (
        "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
    )
    expected_positive = {
        "source_record_index": 16,
        "source_case_id": 10472,
        "source_context_index": 1,
        "detector_group_index": 16,
        "detector_case_id": 10472,
        "owner_group": True,
        "source_prompt_sha256": duplicate_digest,
    }
    expected_negative = {
        "source_record_index": 1,
        "source_case_id": 19763,
        "source_context_index": 4,
        "detector_group_index": 16,
        "detector_case_id": 10472,
        "owner_group": False,
        "source_prompt_sha256": duplicate_digest,
    }
    checks = {
        "protocol": owner_gate.get("protocol") == FROZEN_V3_5_2_PROTOCOL
        and global_gate.get("protocol") == FROZEN_V3_5_2_PROTOCOL
        and training_log.get("protocol") == FROZEN_V3_5_2_PROTOCOL,
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "owner_gate_49_of_50": owner_gate.get("passed_records") == 49
        and owner_gate.get("total_records") == 50
        and owner_gate.get("failure_counts")
        == {"positive": 1, "negative": 0, "writer_off": 0},
        "global_violation_counts": global_gate.get("gate_endpoint_violation_counts")
        == {
            "positive_owner": 1,
            "positive_cross": 0,
            "negative": 1,
            "writer_off": 0,
        },
        "single_positive_cell": len(positive_cells) == 1
        and all(
            positive_cells[0].get(key) == expected
            for key, expected in expected_positive.items()
        ),
        "single_negative_cell": len(negative_cells) == 1
        and all(
            negative_cells[0].get(key) == expected
            for key, expected in expected_negative.items()
        ),
        "same_exact_prompt": len(positive_cells) == len(negative_cells) == 1
        and positive_cells[0].get("source_prompt_sha256")
        == negative_cells[0].get("source_prompt_sha256")
        == duplicate_digest,
        "exact_responses": len(positive_cells) == len(negative_cells) == 1
        and math.isclose(
            float(positive_cells[0].get("response", float("nan"))),
            0.22039906680583954,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(negative_cells[0].get("response", float("nan"))),
            0.21156451106071472,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "writer_off_repair_succeeded": not writer_off_cells
        and global_gate.get("response_certificate_violation_counts", {}).get(
            "writer_off"
        )
        == 0
        and float(
            global_gate.get("aggregate", {})
            .get("writer_off_response", {})
            .get("max", float("inf"))
        )
        <= 0.15
        and float(
            global_gate.get("aggregate", {})
            .get("writer_off_response", {})
            .get("min", float("-inf"))
        )
        >= -0.15,
        "training_complete": training_log.get("optimizer_steps_expected") == 100
        and training_log.get("optimizer_steps_recorded") == 100
        and training_log.get("complete") is True,
        "endpoint_complete": endpoint.get("complete") is True,
        "rejection_preserved": rejection.get("stage") == "sparse_context_detector"
        and rejection.get("actuator_training_started") is False
        and rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (owner_gate, global_gate, training_log, endpoint, rejection)
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen") == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5.2 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_2_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_2_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "diagnosis": "same_exact_prompt_has_positive_and_record_relative_negative_roles",
        "duplicate_prompt_sha256": duplicate_digest,
        "positive_cell": dict(positive_cells[0]),
        "negative_cell": dict(negative_cells[0]),
        "writer_off_repair_succeeded": True,
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_3_rejection(
    run_dir: Path,
    *,
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Bind V3.5.3's positive collapse under complete-update global tails."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "selection": method_dir / "neuron_selection_report.json",
        "prompt_manifest": method_dir / "multilabel_prompt_manifest.json",
        "owner_gate": method_dir / "detector_gate_report.json",
        "global_gate": (
            method_dir / "detector_step_100_post_projection_global_isolation_gate.json"
        ),
        "training_log": method_dir / "detector_training_log.json",
        "endpoint": method_dir / "detector_endpoint_audit.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5.3 rejection lacks {name}: {path}")

    selection = _load_json(paths["selection"])
    prompt_manifest = _load_json(paths["prompt_manifest"])
    owner_gate = _load_json(paths["owner_gate"])
    global_gate = _load_json(paths["global_gate"])
    training_log = _load_json(paths["training_log"])
    endpoint = _load_json(paths["endpoint"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    global_counts = global_gate.get("response_certificate_violation_counts", {})
    checks = {
        "protocol": all(
            payload.get("protocol") == FROZEN_V3_5_3_PROTOCOL
            for payload in (
                prompt_manifest,
                owner_gate,
                global_gate,
                training_log,
                endpoint,
            )
        ),
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "owner_gate_29_of_50": owner_gate.get("passed_records") == 29
        and owner_gate.get("total_records") == 50
        and owner_gate.get("passed") is False
        and owner_gate.get("failure_counts")
        == {"positive": 21, "negative": 0, "writer_off": 0},
        "canonical_multilabel_semantics": prompt_manifest.get(
            "frozen_v3_5_2_duplicate_prompt_reproduced"
        )
        is True
        and prompt_manifest.get("same_record_positive_negative_conflicts") == 0,
        "negative_and_writer_off_certificate_clean": all(
            int(global_counts.get(name, -1)) == 0
            for name in (
                "source_negative_owner",
                "writer_off",
            )
        ),
        "positive_certificate_failed": int(
            global_counts.get("positive_owner", 0)
        )
        > 0
        or int(global_counts.get("writer_on_active", 0)) > 0,
        "global_tail_optimization_enabled": training_log.get("optimization", {}).get(
            "global_tail_weight"
        )
        == 1.0,
        "training_complete": training_log.get("optimizer_steps_expected") == 100
        and training_log.get("optimizer_steps_recorded") == 100
        and training_log.get("complete") is True,
        "endpoint_complete": endpoint.get("complete") is True,
        "rejection_preserved": rejection.get("stage") == "sparse_context_detector"
        and rejection.get("actuator_training_started") is False
        and rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (
                prompt_manifest,
                owner_gate,
                global_gate,
                training_log,
                endpoint,
                rejection,
            )
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen")
                == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5.3 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_3_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_3_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "diagnosis": "global_tail_weighting_collapsed_valid_positive_responses",
        "owner_gate_passed_records": 29,
        "owner_gate_total_records": 50,
        "negative_and_writer_off_certificate_clean": True,
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_4_rejection(
    run_dir: Path,
    *,
    ownership_sha256: str,
) -> Dict[str, Any]:
    """Bind V3.5.4's exact detector pass and isolated width-four rejection."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "selection": method_dir / "neuron_selection_report.json",
        "detector_gate": method_dir / "detector_gate_report.json",
        "threshold_gate": method_dir / "isolated_threshold_gate_report.json",
        "cap_fit": method_dir / "actuator_cap_1p50_feasibility.json",
        "sweep": method_dir / "v3_5_4_multilabel_actuator_feasibility.json",
        "completion": method_dir / "training_only_v3_5_4_completion.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5.4 rejection lacks {name}: {path}")

    selection = _load_json(paths["selection"])
    detector_gate = _load_json(paths["detector_gate"])
    threshold_gate = _load_json(paths["threshold_gate"])
    cap_fit = _load_json(paths["cap_fit"])
    sweep = _load_json(paths["sweep"])
    completion = _load_json(paths["completion"])
    rejection = _load_json(paths["rejection"])
    firewall = _load_json(paths["firewall"])
    tensor_receipt = cap_fit.get("frozen_detector_tensors", {})
    gate_sha256 = str(tensor_receipt.get("gate_delta_sha256") or "")
    up_sha256 = str(tensor_receipt.get("up_delta_sha256") or "")
    final_audit = cap_fit.get("final_audit", {})
    geometry = cap_fit.get("down_norm_geometry", {})
    threshold_checks = threshold_gate.get("checks", {})
    required_threshold_checks = (
        "all_positive_source_owners_active",
        "all_writer_on_active_labels_one",
        "all_writer_on_inactive_labels_zero",
        "all_source_negative_owners_zero",
        "all_writer_off_labels_zero",
    )
    checks = {
        "protocol": all(
            payload.get("protocol") == FROZEN_V3_5_4_PROTOCOL
            for payload in (
                detector_gate,
                threshold_gate,
                cap_fit,
                sweep,
                completion,
            )
        ),
        "ownership": str(
            selection.get("selected_neuron_ownership_jq_compact_sha256") or ""
        )
        == str(ownership_sha256),
        "detector_50_of_50": detector_gate.get("passed_records") == 50
        and detector_gate.get("total_records") == 50
        and detector_gate.get("passed") is True,
        "threshold_gate_schema": threshold_gate.get("schema_version") == 3
        and threshold_gate.get("kind")
        == "mcf_embedding_keyed_neuron_multilabel_global_isolation_gate",
        "threshold_gate_perfect": threshold_gate.get("passed") is True
        and all(
            threshold_checks.get(name) is True
            for name in required_threshold_checks
        ),
        "detector_hashes_recorded": len(gate_sha256) == 64
        and len(up_sha256) == 64
        and tensor_receipt.get("unchanged") is True,
        "cap_1_5_rejected": math.isclose(
            float(cap_fit.get("cap", float("nan"))), 1.5, abs_tol=1e-12
        )
        and cap_fit.get("positive_reachable") is False
        and int(final_audit.get("positive_failures", 0)) > 0,
        "cap_geometry_matches_observed_run": int(
            geometry.get("selected_columns", -1)
        )
        == 200
        and int(geometry.get("saturated_columns", -1)) == 147,
        "weights_discarded": cap_fit.get("fitted_weights_discarded") is True
        and cap_fit.get("down_delta_after_discard") == "bit_exact_zero"
        and sweep.get("all_fitted_weights_discarded") is True,
        "completion_rejected": completion.get("positive_reachability_passed")
        is False
        and completion.get("mechanism_readiness_passed") is False
        and completion.get("conclusion")
        == "isolated_threshold_branch_not_positive_reachable_at_registered_cap",
        "no_full_training_or_checkpoint": completion.get(
            "full_preservation_objective_started"
        )
        is False
        and completion.get("checkpoint_saved") is False
        and rejection.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (threshold_gate, cap_fit, sweep, completion, rejection)
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen")
                == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5.4 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_4_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_4_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "detector_gate_delta_sha256": gate_sha256,
        "detector_up_delta_sha256": up_sha256,
        "detector_passed_records": 50,
        "detector_total_records": 50,
        "actuator_width": 4,
        "actuator_cap": 1.5,
        "actuator_saturated_columns": 147,
        "positive_reachability_passed": False,
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_5_5_success(run_dir: Path) -> Dict[str, Any]:
    """Bind V3.5.5's exact discarded width-16 mechanism-readiness pass."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "selection": method_dir / "actuator_neuron_selection_report.json",
        "detector_replay": method_dir / "exact_v3_5_4_detector_replay.json",
        "sweep": method_dir / "v3_5_5_actuator_width_feasibility.json",
        "completion": method_dir / "training_only_v3_5_5_completion.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.5.5 success lacks {name}: {path}")

    selection = _load_json(paths["selection"])
    detector_replay = _load_json(paths["detector_replay"])
    sweep = _load_json(paths["sweep"])
    completion = _load_json(paths["completion"])
    firewall = _load_json(paths["firewall"])
    width16_rows = selection.get("ownership_by_width", {}).get("16", [])
    width16_ownership = [
        [int(value) for value in row.get("selected_neurons", [])]
        for row in width16_rows
        if isinstance(row, Mapping)
    ]
    flat_width16 = [value for group in width16_ownership for value in group]
    width16_ownership_sha256 = compositional_method.sha256_json(width16_ownership)
    arms = {
        (int(row.get("actuator_width", -1)), str(row.get("budget_regime") or "")): row
        for row in sweep.get("arm_artifacts", [])
        if isinstance(row, Mapping)
    }
    native4 = arms.get((4, "native_per_column_cap"), {})
    native8 = arms.get((8, "native_per_column_cap"), {})
    native16 = arms.get((16, "native_per_column_cap"), {})
    matched8 = arms.get((8, "matched_width4_group_budget_control"), {})
    matched16 = arms.get((16, "matched_width4_group_budget_control"), {})
    gate_sha256 = str(detector_replay.get("observed_gate_delta_sha256") or "")
    up_sha256 = str(detector_replay.get("observed_up_delta_sha256") or "")
    checks = {
        "protocol": all(
            payload.get("protocol") == FROZEN_V3_5_5_PROTOCOL
            for payload in (selection, detector_replay, sweep, completion)
        ),
        "detector_replay": detector_replay.get("passed") is True
        and detector_replay.get("gate_delta_bit_exact") is True
        and detector_replay.get("up_delta_bit_exact") is True
        and detector_replay.get("all_cell_gate_passed") is True
        and len(gate_sha256) == 64
        and len(up_sha256) == 64,
        "selection": selection.get("actuator_widths") == [4, 8, 16]
        and selection.get("maximum_actuator_neurons_per_record") == 16
        and selection.get("maximum_actuator_neuron_count") == 800
        and selection.get("nested_prefixes") is True
        and selection.get("detector_actuator_disjoint") is True
        and len(width16_ownership) == 50
        and all(len(group) == 16 for group in width16_ownership)
        and len(flat_width16) == len(set(flat_width16)) == 800,
        "native_width_outcomes": native4.get("positive_reachable") is False
        and native8.get("positive_reachable") is False
        and native16.get("positive_reachable") is True
        and native16.get("writer_off_structural_selectivity_passed") is True
        and native16.get("actuator_columns") == 800
        and native16.get("saturated_columns") == 34
        and math.isclose(
            float(sweep.get("native_per_column_cap", -1.0)),
            1.5,
            abs_tol=1e-12,
        ),
        "matched_budget_controls": matched8.get("positive_reachable") is False
        and matched16.get("positive_reachable") is False,
        "selection_rule": sweep.get("selected_smallest_positive_reachable_width")
        == 16
        and sweep.get("selected_smallest_mechanism_ready_width") == 16
        and sweep.get("positive_reachability_passed") is True
        and sweep.get("structural_selectivity_passed") is True
        and sweep.get("mechanism_readiness_passed") is True
        and sweep.get("conclusion") == "separate_actuator_width_sweep_passed",
        "fits_discarded": sweep.get("all_fitted_weights_discarded") is True
        and sweep.get("frozen_detector_tensors_unchanged") is True,
        "completion": completion.get("selected_smallest_mechanism_ready_width")
        == 16
        and completion.get("mechanism_readiness_passed") is True
        and completion.get("full_preservation_objective_started") is False
        and completion.get("checkpoint_saved") is False
        and completion.get("official_evaluation_allowed") is False,
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for payload in (selection, detector_replay, sweep, completion)
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen")
                == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.5.5 success lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_5_5_success_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_5_5_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "detector_gate_delta_sha256": gate_sha256,
        "detector_up_delta_sha256": up_sha256,
        "selected_actuator_width": 16,
        "actuator_cap": 1.5,
        "width16_ownership_sha256": width16_ownership_sha256,
        "width16_ownership": width16_ownership,
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_frozen_v3_6_rejection(run_dir: Path) -> Dict[str, Any]:
    """Bind V3.6's complete no-checkpoint negative-locality rejection."""

    root = Path(run_dir).resolve()
    method_dir = root / "method" if (root / "method").is_dir() else root
    paths = {
        "warm_start": method_dir / "v3_6_positive_warm_start.json",
        "full_training": method_dir / "v3_6_full_preservation_training.json",
        "final_audit": method_dir / "v3_6_final_full_context_audit.json",
        "protected_kl": method_dir / "v3_6_protected_kl_audit.json",
        "endpoint": method_dir / "v3_6_actuator_endpoint_audit.json",
        "causal": method_dir / "v3_6_causal_component_audit.json",
        "completion": method_dir / "training_only_v3_6_completion.json",
        "rejection": method_dir / "training_rejection.json",
        "firewall": method_dir / "training_firewall_receipt.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"frozen V3.6 rejection lacks {name}: {path}")

    payloads = {name: _load_json(path) for name, path in paths.items()}
    warm = payloads["warm_start"]
    full = payloads["full_training"]
    final = payloads["final_audit"]
    completion = payloads["completion"]
    rejection = payloads["rejection"]
    firewall = payloads["firewall"]
    negative_abs_max = float(final.get("negative_nll_abs_max", float("nan")))
    checks = {
        "protocol": all(
            payload.get("protocol") == FROZEN_V3_6_PROTOCOL
            for name, payload in payloads.items()
            if name != "firewall"
        ),
        "warm_start": warm.get("passed") is True
        and warm.get("first_passing_step") == 35
        and warm.get("used_to_initialize_full_preservation") is True
        and warm.get("final_audit", {}).get("positive_passed") is True,
        "full_training_complete": full.get("full_optimizer_steps_expected") == 200
        and full.get("full_optimizer_steps_recorded") == 200
        and full.get("complete_training_log") is True,
        "positive_preserved": final.get("positive_passed") is True
        and final.get("direct_failures") == 0
        and final.get("positive_failures") == 0,
        "reference_preserved": final.get("reference_preservation_passed") is True,
        "writer_off_preserved": final.get("writer_off_preservation_passed") is True,
        "negative_locality_rejected": final.get("negative_preservation_passed")
        is False
        and math.isclose(negative_abs_max, 0.125, abs_tol=1e-12)
        and negative_abs_max > 0.05,
        "overall_rejected": full.get("passed") is False
        and completion.get("full_preservation_objective_started") is True
        and completion.get("full_preservation_passed") is False
        and completion.get("candidate_checkpoint_saved") is False
        and completion.get("eligible_for_separate_official_evaluation") is False
        and completion.get("official_evaluation_allowed_in_this_process") is False
        and rejection.get("checkpoint_saved") is False
        and rejection.get("official_evaluation_allowed") is False,
        "no_candidate": not (method_dir / "v3_6_candidate_state.pt").exists(),
        "official_prompts_zero": all(
            payload.get("official_evaluation_prompts_seen") == 0
            for name, payload in payloads.items()
            if name != "firewall"
        ),
        "firewall_official_prompts_zero": all(
            (
                firewall.get("data_access", {}).get("official_paraphrases_seen") == 0,
                firewall.get("data_access", {}).get("official_neighborhoods_seen")
                == 0,
                firewall.get("data_access", {}).get("benchmark_retain_seen") == 0,
                firewall.get("data_access", {}).get("official_ppl_seen") is False,
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            "frozen V3.6 rejection lineage failed: " + ", ".join(sorted(failed))
        )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_frozen_v3_6_rejection_import",
        "source_run_dir": str(root),
        "source_method_dir": str(method_dir),
        "source_protocol": FROZEN_V3_6_PROTOCOL,
        "target_protocol": PROTOCOL,
        "source_artifacts": {
            name: {"path": str(path), "sha256": compositional_method.sha256_file(path)}
            for name, path in paths.items()
        },
        "positive_warm_start_first_passing_step": 35,
        "full_preservation_optimizer_updates": 200,
        "final_negative_nll_abs_max": negative_abs_max,
        "negative_nll_acceptance_ceiling": 0.05,
        "candidate_checkpoint_saved": False,
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def _resolve_swiglu_mlp(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")
    if int(layer_index) >= len(layers):
        raise ValueError(
            f"neuron layer {layer_index} is outside model with {len(layers)} layers"
        )
    mlp = getattr(layers[int(layer_index)], "mlp", None)
    if mlp is None:
        raise RuntimeError(f"layer {layer_index} has no MLP")
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise RuntimeError(
                f"layer {layer_index} MLP is not a supported SwiGLU module: missing {name}"
            )
    if mlp.gate_proj.weight.shape != mlp.up_proj.weight.shape:
        raise RuntimeError("gate_proj and up_proj shapes differ")
    if int(mlp.down_proj.weight.shape[1]) != int(mlp.gate_proj.weight.shape[0]):
        raise RuntimeError("down_proj input does not match SwiGLU intermediate size")
    return mlp


def _validate_firewall(
    context_manifest: Mapping[str, Any], stage1_state: Mapping[str, Any]
) -> None:
    data_access = context_manifest.get("data_access")
    if not isinstance(data_access, Mapping):
        raise RuntimeError("context manifest lacks data-access receipt")
    expected_zero = {
        "official_paraphrases_seen": 0,
        "official_neighborhoods_seen": 0,
        "benchmark_retain_seen": 0,
        "official_ppl_seen": False,
    }
    for key, expected in expected_zero.items():
        if data_access.get(key) != expected:
            raise RuntimeError(
                f"data firewall violation: {key}={data_access.get(key)!r}, "
                f"expected {expected!r}"
            )
    forbidden = {
        "paraphrase_prompts",
        "neighborhood_prompts",
        "retain_prompts",
        "official_ppl_text",
        "adversarial_prompts",
        "alias_prompts",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(str(key) for key in value)
            if overlap:
                raise RuntimeError(
                    "training context artifact contains evaluation-only fields: "
                    + ", ".join(sorted(overlap))
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(context_manifest)
    stored_hash = str(stage1_state.get("context_manifest_sha256") or "")
    if not stored_hash:
        raise RuntimeError("Stage-1 state lacks its exact context-manifest hash")


def _validate_clean_stage1_lineage(
    context_manifest: Mapping[str, Any],
    stage1_state: Mapping[str, Any],
    stage1_report: Mapping[str, Any],
    context_path: Path,
    stage1_log_path: Path,
) -> Dict[str, Any]:
    """Require a fresh, relation-preserving Stage-1 writer artifact."""

    policy = context_manifest.get("positive_context_policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("context manifest lacks its positive-context policy")
    expected_policy = compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY
    if str(policy.get("name") or "") != expected_policy:
        raise RuntimeError(
            "neuron suppression requires the clean relation-templates-only "
            f"writer policy {expected_policy!r}"
        )
    if bool(policy.get("free_form_generated_surrogates_allowed")):
        raise RuntimeError(
            "clean writer policy unexpectedly allows free-form surrogates"
        )
    expected_template_hash = compositional_method.sha256_json(
        synthetic.RELATION_ALTERNATE_TEMPLATES
    )
    if str(policy.get("relation_template_bank_sha256") or "") != expected_template_hash:
        raise RuntimeError("clean writer relation-template bank hash is stale")
    counts = policy.get("source_prompt_counts")
    if (
        not isinstance(counts, Mapping)
        or int(counts.get("external_free_form_surrogate", -1)) != 0
    ):
        raise RuntimeError("clean writer contains external free-form surrogate prompts")
    if context_manifest.get("surrogate_receipt") is not None:
        raise RuntimeError("clean writer context manifest contains a surrogate receipt")
    coverage = context_manifest.get("synthetic_coverage")
    if (
        not isinstance(coverage, Mapping)
        or int(coverage.get("generic_fallback_records", -1)) != 0
    ):
        raise RuntimeError(
            "clean writer lacks explicit hand-authored relation coverage for every record"
        )
    rows = context_manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("clean writer context manifest has no records")
    sharing = context_manifest.get("cross_record_parameter_sharing")
    selected_embedding_rows = context_manifest.get("selected_embedding_rows")
    if (
        not isinstance(sharing, Mapping)
        or not isinstance(selected_embedding_rows, list)
        or int(sharing.get("case_count", -1)) != len(rows)
        or int(sharing.get("selected_row_count", -1)) != len(selected_embedding_rows)
        or int(sharing.get("positive_prompt_count", -1))
        != sum(
            len(row.get("positive_prompts", []))
            for row in rows
            if isinstance(row, Mapping)
        )
    ):
        raise RuntimeError(
            "clean writer lacks its cross-record parameter-sharing audit"
        )
    allowed_sources = {
        "canonical_direct",
        "hand_authored_relation_template_or_corpus_prefix",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("clean writer context record is invalid")
        positives = row.get("positive_prompts")
        provenance = row.get("positive_prompt_provenance")
        if (
            not isinstance(positives, list)
            or not isinstance(provenance, list)
            or len(provenance) != len(positives)
        ):
            raise RuntimeError("clean writer prompt provenance is incomplete")
        for index, (prompt, source_row) in enumerate(zip(positives, provenance)):
            if (
                not isinstance(source_row, Mapping)
                or str(source_row.get("prompt") or "") != str(prompt)
                or str(source_row.get("source") or "") not in allowed_sources
            ):
                raise RuntimeError("clean writer prompt provenance is invalid")
            if index == 0 and source_row.get("source") != "canonical_direct":
                raise RuntimeError("clean writer direct prompt is not first")

    if str(stage1_state.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("Stage-1 writer protocol is stale or incompatible")
    lineage = stage1_state.get("training_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("Stage-1 state lacks a training-lineage receipt")
    if (
        not bool(lineage.get("from_scratch"))
        or str(lineage.get("mode") or "") != "from_scratch"
        or lineage.get("resumed_from") is not None
        or int(lineage.get("writer_steps", 0)) != 1200
    ):
        raise RuntimeError(
            "Stage-1 writer must be trained from Base for exactly 1,200 steps"
        )
    expected_optimization = {
        "record_batch": 3,
        "record_batch_semantics": "gradient_accumulation_microbatch_capacity",
        "update_coverage": "all_records_accumulated",
        "records_per_optimizer_update": 50,
        "microbatches_per_optimizer_update": 17,
        "optimizer_updates": 1200,
        "record_exposures": 60000,
        "record_local_reference_exposures": 3600,
        "record_exposure_multiplier_vs_record_local": 50 / 3,
        "gradient_normalization": ("equal_record_mean_plus_global_prompt_mean_kl"),
        "kl_evaluation": (
            "exact_registered_topk_rows_without_full_vocabulary_materialization"
        ),
        "gradient_conflict_audit_phases": ["initial", "final"],
        "gradient_conflict_audit_objectives": ["positive_write", "full_writer"],
        "positive_context_mode": "all",
        "positive_context_batch": 7,
        "positive_tail_k": 2,
        "negative_context_batch": 5,
        "objective": "mean_plus_worst_k_squared_shortfall",
    }
    observed_optimization = lineage.get("writer_optimization")
    if (
        not isinstance(observed_optimization, Mapping)
        or dict(observed_optimization) != expected_optimization
    ):
        raise RuntimeError(
            "Stage-1 writer was not trained with the registered V6.2 "
            "all-record accumulated objective"
        )
    if dict(stage1_state.get("writer_optimization", {})) != expected_optimization:
        raise RuntimeError("Stage-1 state writer-optimization receipt mismatch")
    context_hash = compositional_method.sha256_file(context_path)
    if str(lineage.get("current_context_manifest_sha256") or "") != context_hash:
        raise RuntimeError(
            "Stage-1 lineage is not bound to the current context manifest"
        )
    if str(lineage.get("positive_context_policy") or "") != expected_policy:
        raise RuntimeError("Stage-1 lineage records a different context policy")
    base_model_path = str(lineage.get("base_model_path") or "")
    base_transformer_fingerprint = float(
        lineage.get("base_transformer_fingerprint", float("nan"))
    )
    base_selected_rows_sha256 = str(
        lineage.get("base_selected_embedding_rows_sha256") or ""
    )
    if (
        not base_model_path
        or not math.isfinite(base_transformer_fingerprint)
        or len(base_selected_rows_sha256) != 64
    ):
        raise RuntimeError("Stage-1 lineage lacks its Base-model binding")

    if str(stage1_report.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("Stage-1 writer report protocol mismatch")
    if str(stage1_report.get("context_manifest_sha256") or "") != context_hash:
        raise RuntimeError("Stage-1 report is not bound to the context manifest")
    if str(stage1_report.get("positive_context_policy") or "") != expected_policy:
        raise RuntimeError("Stage-1 report records a different context policy")
    writer_configuration = stage1_report.get("writer_configuration")
    if not isinstance(writer_configuration, Mapping) or any(
        writer_configuration.get(key) != value
        for key, value in expected_optimization.items()
    ):
        raise RuntimeError("Stage-1 report writer-optimization receipt mismatch")
    report_lineage = stage1_report.get("training_lineage")
    if not isinstance(report_lineage, Mapping) or dict(report_lineage) != dict(lineage):
        raise RuntimeError("Stage-1 state/report lineage receipts differ")

    state_gradient_audit = stage1_state.get("gradient_conflict_audit")
    report_gradient_audit = stage1_report.get("gradient_conflict_audit")
    gradient_audit_sha256 = str(
        stage1_state.get("gradient_conflict_audit_sha256") or ""
    )
    gradient_audit_path = stage1_log_path.with_name(
        "stage1_gradient_conflict_audit.json"
    )
    gradient_audit_file_sha256 = (
        compositional_method.sha256_file(gradient_audit_path)
        if gradient_audit_path.is_file()
        else ""
    )
    if (
        not isinstance(state_gradient_audit, Mapping)
        or not isinstance(report_gradient_audit, Mapping)
        or dict(state_gradient_audit) != dict(report_gradient_audit)
        or gradient_audit_sha256
        != compositional_method.sha256_json(state_gradient_audit)
        or str(stage1_report.get("gradient_conflict_audit_sha256") or "")
        != gradient_audit_sha256
        or str(lineage.get("gradient_conflict_audit_sha256") or "")
        != gradient_audit_sha256
        or not gradient_audit_file_sha256
        or str(stage1_state.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or str(stage1_report.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or str(lineage.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or state_gradient_audit.get("official_evaluation_opened") is not False
    ):
        raise RuntimeError("Stage-1 gradient-conflict audit binding is invalid")
    expected_case_ids = [int(row["case_id"]) for row in rows]
    for phase in ("initial", "final"):
        phase_report = state_gradient_audit.get(phase)
        if not isinstance(phase_report, Mapping) or phase_report.get("phase") != phase:
            raise RuntimeError("Stage-1 gradient-conflict audit phase is incomplete")
        for objective in ("positive_write", "full_writer"):
            objective_report = phase_report.get(objective)
            if (
                not isinstance(objective_report, Mapping)
                or [int(value) for value in objective_report.get("case_ids", [])]
                != expected_case_ids
            ):
                raise RuntimeError(
                    "Stage-1 gradient-conflict audit objective is incomplete"
                )
    for objective in ("positive_write", "full_writer"):
        initial_report = state_gradient_audit["initial"][objective]
        if (
            int(initial_report.get("nonzero_gradient_records", -1)) != len(rows)
            or int(initial_report.get("valid_pair_count", -1))
            != len(rows) * (len(rows) - 1) // 2
        ):
            raise RuntimeError(
                "Stage-1 initial gradient-conflict audit has zero gradients"
            )

    if not stage1_log_path.is_file():
        raise RuntimeError("Stage-1 writer log is missing")
    log_sha256 = compositional_method.sha256_file(stage1_log_path)
    expected_log_sha256 = str(stage1_state.get("writer_log_sha256") or "")
    if not expected_log_sha256 or log_sha256 != expected_log_sha256:
        raise RuntimeError("Stage-1 writer log hash does not match its state")
    if str(stage1_report.get("writer_log_sha256") or "") != log_sha256:
        raise RuntimeError("Stage-1 writer log hash does not match its report")
    with stage1_log_path.open("r", encoding="utf-8") as log_handle:
        log_events = sum(1 for line in log_handle if line.strip())
    expected_events = int(stage1_state.get("writer_log_event_count", -1))
    if log_events <= 0 or log_events != expected_events:
        raise RuntimeError("Stage-1 writer log is empty or has an invalid event count")
    if int(stage1_report.get("writer_log_event_count", -1)) != log_events:
        raise RuntimeError("Stage-1 writer report log-event count mismatch")
    return {
        "positive_context_policy": expected_policy,
        "protocol": compositional_core.PROTOCOL,
        "from_scratch": True,
        "writer_steps": int(lineage["writer_steps"]),
        "context_manifest_sha256": context_hash,
        "writer_log_sha256": log_sha256,
        "writer_log_event_count": log_events,
        "gradient_conflict_audit_sha256": gradient_audit_sha256,
        "gradient_conflict_audit_file_sha256": gradient_audit_file_sha256,
        "writer_optimization": expected_optimization,
        "base_model_path": base_model_path,
        "base_transformer_fingerprint": base_transformer_fingerprint,
        "base_selected_embedding_rows_sha256": base_selected_rows_sha256,
    }


def _context_sets_by_case(
    context_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    rows = context_manifest.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("context manifest lacks record contexts")
    by_case: Dict[int, Dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise RuntimeError("invalid context-manifest record")
        case_id = int(value["case_id"])
        positives = value.get("positive_prompts")
        negatives = value.get("negative_contexts")
        if not isinstance(positives, list) or not positives:
            raise RuntimeError(f"case {case_id} has no training positives")
        if not isinstance(negatives, list) or not negatives:
            raise RuntimeError(f"case {case_id} has no training negatives")
        by_case[case_id] = dict(value)
    expected = [int(record["case_id"]) for record in records]
    if set(by_case) != set(expected):
        raise RuntimeError("context-manifest cases do not match locked training view")
    for record in records:
        case_id = int(record["case_id"])
        direct = str(record["direct_prompt"])
        if str(by_case[case_id]["positive_prompts"][0]) != direct:
            raise RuntimeError(
                f"case {case_id} direct prompt changed in context manifest"
            )
    return by_case


def _validate_clean_stage1_acceptance(
    receipt: Mapping[str, Any],
    *,
    seed: int,
    case_ids: Sequence[int],
    expected_artifacts: Mapping[str, str],
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    """Require the integrity *and* replayed portability conjunction."""

    if receipt.get("kind") != "mcf_clean_stage1_writer_acceptance":
        raise RuntimeError("clean Stage-1 acceptance receipt has the wrong kind")
    if receipt.get("passed") is not True:
        raise RuntimeError("clean Stage-1 acceptance receipt did not pass")
    if str(receipt.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("clean Stage-1 acceptance protocol mismatch")
    if int(receipt.get("seed", -1)) != int(seed):
        raise RuntimeError("clean Stage-1 acceptance seed mismatch")
    if [int(value) for value in receipt.get("case_ids", [])] != [
        int(value) for value in case_ids
    ]:
        raise RuntimeError("clean Stage-1 acceptance case IDs mismatch")
    checks = receipt.get("checks")
    required_checks = ("artifact_integrity", "training_safe_portability")
    if not isinstance(checks, Mapping) or not all(
        checks.get(key) is True for key in required_checks
    ):
        raise RuntimeError("clean Stage-1 acceptance conjunction is incomplete")
    if bool(receipt.get("official_evaluation_opened")):
        raise RuntimeError("clean Stage-1 acceptance crossed the evaluation firewall")
    artifacts = receipt.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or not str(artifacts.get("stage1_gradient_conflict_audit_sha256") or "")
        or any(
            str(artifacts.get(key) or "") != value
            for key, value in expected_artifacts.items()
        )
    ):
        raise RuntimeError("clean Stage-1 acceptance artifact binding mismatch")

    preflight = receipt.get("training_safe_portability")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        raise RuntimeError("clean Stage-1 portability preflight did not pass")
    if preflight.get("kind") != "mcf_clean_stage1_training_safe_portability_preflight":
        raise RuntimeError("clean Stage-1 portability preflight has the wrong kind")
    validate_writer_preflight_summary(
        preflight,
        amplitude_threshold=float(amplitude_threshold),
        minimum_global_fraction=float(minimum_global_fraction),
        minimum_record_fraction=float(minimum_record_fraction),
    )
    return {
        "kind": str(receipt["kind"]),
        "passed": True,
        "training_safe_portability": {
            "prompt_count": int(preflight["prompt_count"]),
            "complete_count": int(preflight["complete_count"]),
            "global_complete_fraction": float(preflight["global_complete_fraction"]),
            "minimum_record_complete_fraction": float(
                preflight["minimum_record_complete_fraction"]
            ),
            "amplitude_threshold": float(preflight["amplitude_threshold"]),
        },
    }


def _record_views(locked_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return compositional_method._record_views(locked_records)


def _unique_prompts(values: Sequence[str]) -> List[str]:
    return compositional_core.ordered_unique([str(value) for value in values])


def _round_robin_unique_prompts(
    groups: Sequence[Sequence[str]],
    *,
    limit: int,
    excluded: Sequence[str] = (),
) -> List[str]:
    """Select prompts without letting early records consume a fixed budget."""

    if int(limit) <= 0:
        return []
    normalized_groups = [_unique_prompts(group) for group in groups]
    positions = [0 for _ in normalized_groups]
    seen = {str(value) for value in excluded}
    selected: List[str] = []
    while len(selected) < int(limit):
        progressed = False
        for group_index, group in enumerate(normalized_groups):
            while positions[group_index] < len(group):
                prompt = str(group[positions[group_index]])
                positions[group_index] += 1
                if prompt in seen:
                    continue
                seen.add(prompt)
                selected.append(prompt)
                progressed = True
                break
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def build_selection_protected_prompts(
    training_groups: Sequence[Sequence[str]],
    corpus_prompts: Sequence[str],
    *,
    total_limit: int,
    minimum_corpus: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build an auditable, stratified neuron-profile prompt bank.

    Corpus prompts are reserved first, then the training quota is filled in
    round-robin record order.  Spare capacity is backfilled from either source.
    This prevents the historical ``all_training + corpus`` truncation from
    silently deleting the broad-corpus profile or all prompts from later cases.
    """

    total_limit = int(total_limit)
    minimum_corpus = int(minimum_corpus)
    if total_limit <= 0 or not 0 <= minimum_corpus <= total_limit:
        raise ValueError("invalid protected-prompt quotas")
    corpus = _unique_prompts(corpus_prompts)
    if len(corpus) < minimum_corpus:
        raise RuntimeError(
            f"only {len(corpus)} unique corpus prompts are available; "
            f"{minimum_corpus} are required for broad neuron profiling"
        )
    reserved_corpus = corpus[:minimum_corpus]
    training_quota = total_limit - len(reserved_corpus)
    training = _round_robin_unique_prompts(
        training_groups,
        limit=training_quota,
        excluded=reserved_corpus,
    )
    selected = [*reserved_corpus, *training]
    if len(selected) < total_limit:
        selected_set = set(selected)
        selected.extend(
            prompt for prompt in corpus[minimum_corpus:] if prompt not in selected_set
        )
        selected = selected[:total_limit]
    if len(selected) < total_limit:
        selected.extend(
            _round_robin_unique_prompts(
                training_groups,
                limit=total_limit - len(selected),
                excluded=selected,
            )
        )
    selected = _unique_prompts(selected)[:total_limit]
    corpus_set = set(corpus)
    training_set = {
        prompt for group in training_groups for prompt in _unique_prompts(group)
    }
    selected_set = set(selected)
    source_counts = {
        "corpus": sum(prompt in corpus_set for prompt in selected),
        "training_only": sum(
            prompt in training_set and prompt not in corpus_set for prompt in selected
        ),
        "unclassified": sum(
            prompt not in training_set and prompt not in corpus_set
            for prompt in selected
        ),
    }
    represented_groups = sum(
        any(prompt in selected_set for prompt in _unique_prompts(group))
        for group in training_groups
    )
    return selected, {
        "total_limit": total_limit,
        "minimum_corpus_required": minimum_corpus,
        "selected_total": len(selected),
        "full_budget_passed": len(selected) == total_limit,
        "source_counts": source_counts,
        "training_groups_total": len(training_groups),
        "training_groups_represented": represented_groups,
        "corpus_quota_passed": source_counts["corpus"] >= minimum_corpus,
        "all_training_groups_represented": represented_groups == len(training_groups),
    }


def activation_tail_profile(
    activations: torch.Tensor,
    neuron_ids: Sequence[int],
    *,
    activation_threshold: float,
    down_column_norms: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """Summarize how much selected neurons already function on a prompt bank."""

    values = activations.detach().float().cpu()
    if values.ndim != 2 or values.shape[1] != len(neuron_ids):
        raise ValueError("activation profile shape does not match selected neurons")
    if values.shape[0] == 0:
        raise ValueError("activation profile needs at least one prompt")
    if not torch.isfinite(values).all():
        raise ValueError("activation profile contains non-finite values")
    if float(activation_threshold) < 0:
        raise ValueError("activation threshold must be non-negative")
    if down_column_norms is not None:
        down_norms = down_column_norms.detach().float().cpu().reshape(-1)
        if down_norms.numel() != len(neuron_ids):
            raise ValueError("down-column norms do not match selected neurons")
    else:
        down_norms = None

    absolute = values.abs()
    quantile_levels = torch.tensor([0.99, 0.999], dtype=torch.float32)
    quantiles = torch.quantile(absolute, quantile_levels, dim=0)
    rms = values.square().mean(dim=0).sqrt()
    per_neuron: List[Dict[str, Any]] = []
    for column, neuron_id in enumerate(neuron_ids):
        row: Dict[str, Any] = {
            "neuron_id": int(neuron_id),
            "activation_rms": float(rms[column]),
            "activation_abs_p99": float(quantiles[0, column]),
            "activation_abs_p999": float(quantiles[1, column]),
            "activation_abs_max": float(absolute[:, column].max()),
            "activation_threshold_exceedance_fraction": float(
                (absolute[:, column] > float(activation_threshold)).float().mean()
            ),
        }
        if down_norms is not None:
            norm = float(down_norms[column])
            row.update(
                {
                    "base_down_column_norm": norm,
                    "base_residual_contribution_rms_bound": norm
                    * row["activation_rms"],
                    "base_residual_contribution_abs_p999_bound": norm
                    * row["activation_abs_p999"],
                }
            )
        per_neuron.append(row)

    return {
        "prompt_count": int(values.shape[0]),
        "neuron_count": int(values.shape[1]),
        "activation_threshold": float(activation_threshold),
        "activation_rms": _distribution([row["activation_rms"] for row in per_neuron]),
        "activation_abs_p999": _distribution(
            [row["activation_abs_p999"] for row in per_neuron]
        ),
        "activation_abs_max": _distribution(
            [row["activation_abs_max"] for row in per_neuron]
        ),
        "per_neuron": per_neuron,
    }


def summarize_writer_preflight(
    amplitude_groups: Sequence[torch.Tensor],
    *,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    if not amplitude_groups or any(group.numel() == 0 for group in amplitude_groups):
        raise ValueError("writer preflight needs non-empty amplitudes per record")
    groups = [group.detach().float().reshape(-1).cpu() for group in amplitude_groups]
    if not all(torch.isfinite(group).all() for group in groups):
        raise ValueError("writer preflight contains non-finite amplitudes")
    per_record = []
    for group in groups:
        prompt_count = int(group.numel())
        complete_count = int((group >= float(amplitude_threshold)).sum())
        per_record.append(
            {
                "prompt_count": prompt_count,
                "complete_count": complete_count,
                "complete_fraction": complete_count / prompt_count,
                "amplitude": _distribution([float(value) for value in group]),
            }
        )
    total = sum(row["prompt_count"] for row in per_record)
    complete = sum(row["complete_count"] for row in per_record)
    global_fraction = complete / total
    checks = {
        "global_complete_fraction": global_fraction >= float(minimum_global_fraction),
        "minimum_record_complete_fraction": min(
            row["complete_fraction"] for row in per_record
        )
        >= float(minimum_record_fraction),
    }
    return {
        "amplitude_threshold": float(amplitude_threshold),
        "prompt_count": total,
        "complete_count": complete,
        "global_complete_fraction": global_fraction,
        "minimum_record_complete_fraction": min(
            row["complete_fraction"] for row in per_record
        ),
        "criterion": {
            "minimum_global_fraction": float(minimum_global_fraction),
            "minimum_record_fraction": float(minimum_record_fraction),
        },
        "per_record": per_record,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_writer_preflight_summary(
    summary: Mapping[str, Any],
    *,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> None:
    """Recompute the portability decision from its per-record counts."""

    rows = summary.get("per_record")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("writer preflight has no per-record rows")
    prompt_count = 0
    complete_count = 0
    fractions: List[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("writer preflight per-record row is invalid")
        prompts = int(row.get("prompt_count", 0))
        complete = int(row.get("complete_count", -1))
        if prompts <= 0 or complete < 0 or complete > prompts:
            raise RuntimeError("writer preflight per-record counts are invalid")
        fraction = complete / prompts
        if not math.isclose(
            float(row.get("complete_fraction", float("nan"))),
            fraction,
            rel_tol=0.0,
            abs_tol=COUNT_DERIVED_FRACTION_ABS_TOLERANCE,
        ):
            raise RuntimeError("writer preflight per-record fraction is inconsistent")
        prompt_count += prompts
        complete_count += complete
        fractions.append(fraction)
    global_fraction = complete_count / prompt_count
    minimum_observed = min(fractions)
    for observed, expected, label in (
        (summary.get("prompt_count"), prompt_count, "prompt count"),
        (summary.get("complete_count"), complete_count, "complete count"),
    ):
        if int(observed if observed is not None else -1) != expected:
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    for observed, expected, label in (
        (
            float(summary.get("amplitude_threshold", float("nan"))),
            float(amplitude_threshold),
            "amplitude threshold",
        ),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    for observed, expected, label in (
        (
            float(summary.get("global_complete_fraction", float("nan"))),
            global_fraction,
            "global fraction",
        ),
        (
            float(summary.get("minimum_record_complete_fraction", float("nan"))),
            minimum_observed,
            "minimum-record fraction",
        ),
    ):
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=COUNT_DERIVED_FRACTION_ABS_TOLERANCE,
        ):
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    criterion = summary.get("criterion")
    if (
        not isinstance(criterion, Mapping)
        or not math.isclose(
            float(criterion.get("minimum_global_fraction", float("nan"))),
            float(minimum_global_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(criterion.get("minimum_record_fraction", float("nan"))),
            float(minimum_record_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("writer preflight registered fractions are inconsistent")
    expected_checks = {
        "global_complete_fraction": global_fraction >= float(minimum_global_fraction),
        "minimum_record_complete_fraction": minimum_observed
        >= float(minimum_record_fraction),
    }
    observed_checks = summary.get("checks")
    if not isinstance(observed_checks, Mapping) or any(
        observed_checks.get(key) is not value for key, value in expected_checks.items()
    ):
        raise RuntimeError("writer preflight Boolean checks are inconsistent")
    if summary.get("passed") is not all(expected_checks.values()):
        raise RuntimeError("writer preflight pass flag is inconsistent")


@torch.no_grad()
def measure_training_safe_writer_preflight(
    model: torch.nn.Module,
    tok: Any,
    embedding_writer: Any,
    positive_prompts_by_record: Sequence[Sequence[str]],
    marker_map: Mapping[Any, Any],
    case_ids: Sequence[int],
    device: torch.device,
    *,
    batch_size: int,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    """Replay the exact pre-decoder writer gate on training-safe positives."""

    if len(positive_prompts_by_record) != len(case_ids):
        raise ValueError("writer preflight prompt groups/case IDs differ")
    previous_enabled = bool(embedding_writer.enabled)
    amplitude_groups: List[torch.Tensor] = []
    try:
        for position, prompts in enumerate(positive_prompts_by_record):
            case_id = int(case_ids[position])
            marker = marker_map.get(case_id, marker_map.get(str(case_id)))
            if not isinstance(marker, torch.Tensor) or marker.ndim != 1:
                raise RuntimeError(f"Stage-1 marker missing for case {case_id}")
            embedding_writer.enabled = False
            writer_off_hidden = compositional_method.batched_last_hidden_only(
                model,
                tok,
                list(prompts),
                device,
                batch_size=int(batch_size),
            )
            embedding_writer.enabled = True
            writer_on_hidden = compositional_method.batched_last_hidden_only(
                model,
                tok,
                list(prompts),
                device,
                batch_size=int(batch_size),
            )
            amplitude_groups.append(
                (writer_on_hidden - writer_off_hidden) @ marker.detach().float().cpu()
            )
    finally:
        embedding_writer.enabled = previous_enabled

    result = summarize_writer_preflight(
        amplitude_groups,
        amplitude_threshold=float(amplitude_threshold),
        minimum_global_fraction=float(minimum_global_fraction),
        minimum_record_fraction=float(minimum_record_fraction),
    )
    for index, row in enumerate(result["per_record"]):
        row["case_id"] = int(case_ids[index])
    result.update(
        {
            "kind": "frozen_writer_training_safe_pre_decoder_preflight",
            "applicable": True,
            "official_evaluation_prompts_seen": 0,
            "used_for_decoder_hyperparameter_selection": False,
        }
    )
    return result


@torch.no_grad()
def capture_base_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Capture the ordinary SwiGLU activation entering ``down_proj``."""
    rows: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("activation capture requires a separate model backbone")
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            captured.clear()
            backbone(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one down-projection activation, captured {len(captured)}"
                )
            activation = captured[0]
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            rows.append(activation[batch_rows, positions, :].detach().float().cpu())
    finally:
        handle.remove()
        tok.padding_side = old_side
    if not rows:
        return torch.empty((0, int(mlp.gate_proj.weight.shape[0])))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def capture_mlp_input_last_token_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Cache the exact frozen input used by selected detector gate/up rows.

    During detector training the sparse down projection is disabled, so edits
    to this MLP cannot affect the tensor entering it.  Caching that tensor once
    therefore removes repeated 3B-backbone execution without approximating the
    learned detector computation.  Values are stored as CPU float32 because
    ``SparseSwiGLUNeuronEditor.selected_activations`` applies the same float32
    conversion during an ordinary model forward.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.register_forward_pre_hook(pre_hook)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("detector cache requires a separate model backbone")
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            captured.clear()
            backbone(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(f"expected one MLP input, captured {len(captured)}")
            hidden = captured[0]
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            rows.append(hidden[batch_rows, positions, :].detach().float().cpu())
    finally:
        handle.remove()
        tok.padding_side = old_side
    if not rows:
        return torch.empty((0, int(mlp.gate_proj.weight.shape[1])))
    return torch.cat(rows, dim=0)


def capture_grouped_mlp_input_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompt_groups: Sequence[Sequence[str]],
    device: torch.device,
    *,
    batch_size: int,
) -> List[torch.Tensor]:
    """Capture a flattened prompt bank once and restore record grouping."""
    sizes = [len(group) for group in prompt_groups]
    prompts = [str(prompt) for group in prompt_groups for prompt in group]
    hidden = capture_mlp_input_last_token_hidden_states(
        model, tok, mlp, prompts, device, batch_size=int(batch_size)
    )
    if int(hidden.shape[0]) != sum(sizes):
        raise RuntimeError("detector cache row count does not match prompt bank")
    result: List[torch.Tensor] = []
    start = 0
    for size in sizes:
        result.append(hidden[start : start + size])
        start += size
    return result


def build_multilabel_prompt_manifest(
    positive_prompt_groups: Sequence[Sequence[str]],
    negative_prompt_groups: Sequence[Sequence[str]],
    case_ids: Sequence[int],
) -> Dict[str, Any]:
    """Build exact-prompt active detector labels across all source roles.

    A record-relative negative is not globally negative when the same exact
    prompt is registered as a positive for another record.  The returned
    manifest preserves every occurrence while assigning one canonical prompt
    index and the complete set of positive detector labels.
    """
    records = len(case_ids)
    if not (len(positive_prompt_groups) == len(negative_prompt_groups) == records):
        raise ValueError("prompt groups and case_ids must have equal lengths")
    entries: List[Dict[str, Any]] = []
    prompt_to_index: Dict[str, int] = {}
    digest_to_prompt: Dict[str, str] = {}

    def prompt_index(prompt_value: str) -> int:
        prompt = str(prompt_value)
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        prior = digest_to_prompt.get(digest)
        if prior is not None and prior != prompt:
            raise RuntimeError("SHA-256 collision in canonical detector prompts")
        digest_to_prompt[digest] = prompt
        if prompt not in prompt_to_index:
            prompt_to_index[prompt] = len(entries)
            entries.append(
                {
                    "canonical_prompt_index": len(entries),
                    "prompt": prompt,
                    "prompt_sha256": digest,
                    "active_group_indices": set(),
                    "positive_occurrences": [],
                    "negative_occurrences": [],
                }
            )
        return prompt_to_index[prompt]

    positive_indices: List[List[int]] = [[] for _ in range(records)]
    negative_indices: List[List[int]] = [[] for _ in range(records)]
    for record_index, prompts in enumerate(positive_prompt_groups):
        for context_index, prompt in enumerate(prompts):
            index = prompt_index(str(prompt))
            positive_indices[record_index].append(index)
            entries[index]["active_group_indices"].add(record_index)
            entries[index]["positive_occurrences"].append(
                {
                    "source_record_index": record_index,
                    "source_case_id": int(case_ids[record_index]),
                    "source_context_index": context_index,
                }
            )
    for record_index, prompts in enumerate(negative_prompt_groups):
        for context_index, prompt in enumerate(prompts):
            index = prompt_index(str(prompt))
            negative_indices[record_index].append(index)
            entries[index]["negative_occurrences"].append(
                {
                    "source_record_index": record_index,
                    "source_case_id": int(case_ids[record_index]),
                    "source_context_index": context_index,
                }
            )

    direct_conflicts: List[Dict[str, Any]] = []
    active_mask = torch.zeros((len(entries), records), dtype=torch.bool)
    public_entries: List[Dict[str, Any]] = []
    for entry in entries:
        active = sorted(int(value) for value in entry["active_group_indices"])
        active_mask[int(entry["canonical_prompt_index"]), active] = True
        negative_sources = {
            int(row["source_record_index"]) for row in entry["negative_occurrences"]
        }
        overlap = sorted(set(active).intersection(negative_sources))
        if overlap:
            direct_conflicts.append(
                {
                    "prompt_sha256": entry["prompt_sha256"],
                    "record_indices": overlap,
                }
            )
        public_entries.append(
            {
                "canonical_prompt_index": int(entry["canonical_prompt_index"]),
                "prompt_sha256": str(entry["prompt_sha256"]),
                "active_group_indices": active,
                "active_case_ids": [int(case_ids[index]) for index in active],
                "positive_occurrences": list(entry["positive_occurrences"]),
                "negative_occurrences": list(entry["negative_occurrences"]),
                "appears_in_both_roles": bool(
                    entry["positive_occurrences"] and entry["negative_occurrences"]
                ),
            }
        )
    if direct_conflicts:
        raise RuntimeError(
            "a prompt is both positive and source-relative negative for the same record"
        )

    both_roles = [row for row in public_entries if row["appears_in_both_roles"]]
    return {
        "canonical_prompts": [str(entry["prompt"]) for entry in entries],
        "prompt_to_index": prompt_to_index,
        "active_mask": active_mask,
        "positive_indices_by_record": positive_indices,
        "negative_indices_by_record": negative_indices,
        "report": {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_multilabel_prompt_manifest",
            "definition": (
                "active groups are every record for which the exact prompt is a "
                "registered positive; record-relative negative roles do not erase "
                "those positive labels"
            ),
            "record_count": records,
            "positive_occurrences": sum(map(len, positive_indices)),
            "negative_occurrences": sum(map(len, negative_indices)),
            "unique_prompts": len(entries),
            "prompts_in_both_positive_and_negative_roles": len(both_roles),
            "same_record_positive_negative_conflicts": 0,
            "entries": public_entries,
            "role_overlap_entries": both_roles,
            "official_evaluation_prompts_seen": 0,
        },
    }


def canonicalize_grouped_hidden_states(
    prompt_groups: Sequence[Sequence[str]],
    hidden_groups: Sequence[torch.Tensor],
    *,
    canonical_prompts: Sequence[str],
    prompt_to_index: Mapping[str, int],
) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
    """Reuse one bit-identical cached hidden row for each exact prompt."""
    if len(prompt_groups) != len(hidden_groups):
        raise ValueError("prompt and hidden groups must have equal lengths")
    representatives: List[torch.Tensor | None] = [None] * len(canonical_prompts)
    occurrence_counts = [0] * len(canonical_prompts)
    duplicate_max_abs_deltas: List[float] = []
    for prompts, hidden in zip(prompt_groups, hidden_groups):
        if hidden.ndim != 2 or int(hidden.shape[0]) != len(prompts):
            raise ValueError("hidden group rows must match prompt occurrences")
        for context_index, prompt in enumerate(prompts):
            index = int(prompt_to_index[str(prompt)])
            row = hidden[context_index]
            occurrence_counts[index] += 1
            if representatives[index] is None:
                representatives[index] = row.detach().clone()
            else:
                duplicate_max_abs_deltas.append(
                    float((row - representatives[index]).abs().max())
                )
    missing = [index for index, row in enumerate(representatives) if row is None]
    if missing:
        raise RuntimeError(f"canonical hidden cache lacks prompt indices: {missing}")
    resolved = [row for row in representatives if row is not None]
    canonical_groups: List[torch.Tensor] = []
    for prompts in prompt_groups:
        canonical_groups.append(
            torch.stack(
                [resolved[int(prompt_to_index[str(prompt)])] for prompt in prompts]
            )
        )
    return canonical_groups, {
        "representative_policy": (
            "first occurrence in positive-record order, then negative-record order"
        ),
        "unique_prompts": len(canonical_prompts),
        "occurrences": sum(occurrence_counts),
        "duplicate_occurrences_reused": sum(
            max(0, count - 1) for count in occurrence_counts
        ),
        "precanonicalization_duplicate_hidden_abs_max": (
            max(duplicate_max_abs_deltas) if duplicate_max_abs_deltas else 0.0
        ),
        "postcanonicalization_duplicate_hidden_abs_max": 0.0,
        "bit_identical_reuse_required": True,
    }


def capture_editor_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    editor: neuron_core.SparseSwiGLUNeuronEditor,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    editor.capture_activations = True
    try:
        encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("detector training requires a separate model backbone")
        backbone(**encoded, use_cache=False, return_dict=True)
        activation = editor.last_edited_activations
        if activation is None:
            raise RuntimeError("sparse neuron editor did not capture activations")
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(prompts), device=device)
        return activation[rows, positions, :]
    finally:
        editor.capture_activations = False
        tok.padding_side = old_side


def _negative_prompt_instances(
    records: Sequence[Mapping[str, Any]],
    context_sets: Mapping[int, Mapping[str, Any]],
) -> List[mcf_repair.MCFPromptInstance]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        case_id = int(record["case_id"])
        negatives = context_sets[case_id]["negative_contexts"]
        for prompt_index, row in enumerate(negatives):
            instances.append(
                mcf_repair.MCFPromptInstance(
                    record_index=case_id,
                    sampled_position=position,
                    prompt_type=str(row.get("kind") or "training_safe_negative"),
                    prompt_index=prompt_index,
                    prompt=str(row["prompt"]),
                    target_new=str(record["reference"]),
                    target_true=str(record["answer"]),
                )
            )
    return instances


def apply_forget_positive_precedence_to_negative_instances(
    negative_instances: Sequence[mcf_repair.MCFPromptInstance],
    prompt_labels: Mapping[str, Any],
    case_ids: Sequence[int],
) -> Tuple[
    List[mcf_repair.MCFPromptInstance],
    List[List[int]],
    Dict[str, Any],
]:
    """Remove exact forget-positive prompts from the preservation-negative bank.

    Canonical detector labels already use multi-label semantics: an exact prompt
    registered as a positive for any record keeps that active label even when it
    also occurs as another record's source-relative negative.  The behavioral
    preservation objective must use the same precedence rule.  Otherwise one
    deterministic prompt prefix is simultaneously required to activate a forget
    branch and to remain unchanged.

    This does not relabel ordinary negatives that merely share words or token
    rows with a forget prompt.  Only byte-exact prompt matches with at least one
    registered positive label are excluded.
    """

    prompt_to_index = prompt_labels.get("prompt_to_index")
    active_mask = prompt_labels.get("active_mask")
    if not isinstance(prompt_to_index, Mapping) or not isinstance(
        active_mask, torch.Tensor
    ):
        raise RuntimeError("canonical prompt labels lack precedence inputs")
    if active_mask.ndim != 2 or int(active_mask.shape[1]) != len(case_ids):
        raise RuntimeError("canonical active-label mask has incompatible shape")

    coherent: List[mcf_repair.MCFPromptInstance] = []
    indices_by_record: List[List[int]] = [[] for _ in case_ids]
    excluded: List[Dict[str, Any]] = []
    for raw_index, instance in enumerate(negative_instances):
        prompt = str(instance.prompt)
        canonical_index = prompt_to_index.get(prompt)
        if canonical_index is None:
            raise RuntimeError("negative prompt is absent from canonical manifest")
        active_groups = (
            active_mask[int(canonical_index)]
            .nonzero(as_tuple=False)
            .reshape(-1)
            .tolist()
        )
        if active_groups:
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            excluded.append(
                {
                    "raw_negative_occurrence_index": int(raw_index),
                    "canonical_prompt_index": int(canonical_index),
                    "prompt_sha256": digest,
                    "negative_source_record_index": int(
                        instance.sampled_position
                    ),
                    "negative_source_case_id": int(instance.record_index),
                    "negative_source_context_index": int(instance.prompt_index),
                    "negative_source_prompt_type": str(instance.prompt_type),
                    "positive_active_group_indices": [
                        int(value) for value in active_groups
                    ],
                    "positive_active_case_ids": [
                        int(case_ids[int(value)]) for value in active_groups
                    ],
                    "reason": "exact_prompt_is_registered_for_forgetting",
                }
            )
            continue
        coherent_index = len(coherent)
        coherent.append(instance)
        source_record = int(instance.sampled_position)
        if not 0 <= source_record < len(case_ids):
            raise RuntimeError("negative occurrence has invalid source record")
        indices_by_record[source_record].append(coherent_index)

    if any(not indices for indices in indices_by_record):
        raise RuntimeError(
            "forget-positive precedence emptied a record's preservation bank"
        )
    report = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_negative_preservation_precedence",
        "protocol": PROTOCOL,
        "rule": "forget_positive_exact_prompt_precedes_preservation_negative",
        "matching_scope": "byte_exact_full_prompt_only",
        "lexical_or_subtoken_overlap_is_not_excluded": True,
        "raw_negative_occurrences": len(negative_instances),
        "coherent_preservation_negative_occurrences": len(coherent),
        "excluded_multi_role_negative_occurrences": len(excluded),
        "excluded_occurrences": excluded,
        "official_evaluation_prompts_seen": 0,
    }
    return coherent, indices_by_record, report


def _failure_counts(
    margins: torch.Tensor,
    direct_flags: Sequence[bool],
    margin: float,
) -> Tuple[int, int]:
    threshold = float(margin) - 1e-6
    direct = sum(
        int(bool(direct_flags[index]) and float(value) < threshold)
        for index, value in enumerate(margins)
    )
    return direct, int((margins < threshold).sum())


@torch.no_grad()
def evaluate_instance_nlls_float32(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the exact differentiable float32 NLL definition without grads."""

    new_rows: List[torch.Tensor] = []
    true_rows: List[torch.Tensor] = []
    for start in range(0, len(instances), int(batch_size)):
        current_new, current_true = (
            compositional_method.differentiable_instance_nlls(
                model,
                tok,
                instances[start : start + int(batch_size)],
                device,
                llama_like=llama_like,
            )
        )
        new_rows.append(current_new.detach().cpu())
        true_rows.append(current_true.detach().cpu())
    if not new_rows:
        return torch.empty(0), torch.empty(0)
    return torch.cat(new_rows), torch.cat(true_rows)


def _topk_kl(
    logits: torch.Tensor,
    prompts: Sequence[str],
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    terms = _topk_kl_terms(logits, prompts, cache, device)
    return terms.mean() if int(terms.numel()) else logits.sum() * 0.0


def _topk_kl_terms(
    logits: torch.Tensor,
    prompts: Sequence[str],
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    """Return one frozen-writer top-k KL term per protected prompt."""

    terms: List[torch.Tensor] = []
    for row, prompt in enumerate(prompts):
        ids = cache[str(prompt)]["top_ids"].to(device)
        target = cache[str(prompt)]["top_log_probs"].to(device)
        observed = torch.log_softmax(logits[row].float()[ids], dim=-1)
        terms.append(F.kl_div(observed, target, log_target=True, reduction="sum"))
    return torch.stack(terms) if terms else logits.new_empty((0,), dtype=torch.float32)


def _replace_embedding_rows(
    layer: torch.nn.Module, token_ids: Sequence[int], values: torch.Tensor
) -> None:
    ids = torch.tensor(
        [int(value) for value in token_ids],
        dtype=torch.long,
        device=layer.weight.device,
    )
    with torch.no_grad():
        layer.weight.index_copy_(
            0, ids, values.to(device=layer.weight.device, dtype=layer.weight.dtype)
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    width_sweep_mode = bool(args.training_only_actuator_width_sweep)
    full_preservation_mode = bool(args.training_only_full_preservation)
    detector_replay_mode = bool(width_sweep_mode or full_preservation_mode)
    if not detector_replay_mode:
        raise RuntimeError(
            "this checkout is restricted to registered V3.5.5 width feasibility "
            "or V3.6.1 training-only full preservation; official evaluation remains "
            "a separate process"
        )
    _validate_environment_firewall()
    gagd.set_seed(int(args.seed))
    gagd.require_cuda_if_needed(args.device_map)
    actuator_rng_seed = int(args.seed) + 78103
    actuator_rng = random.Random(actuator_rng_seed)
    out_dir = gagd.resolve_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visible_path = Path(args.training_visible_path).resolve()
    split_path = Path(args.split_manifest).resolve()
    context_path = Path(args.context_manifest).resolve()
    stage1_path = Path(args.stage1_state).resolve()
    stage1_report_path = Path(args.stage1_report).resolve()
    stage1_log_path = Path(args.stage1_writer_log).resolve()
    stage1_gradient_audit_path = stage1_log_path.with_name(
        "stage1_gradient_conflict_audit.json"
    )
    stage1_portability_path = Path(args.clean_stage1_portability_preflight).resolve()
    stage1_acceptance_path = Path(args.clean_stage1_acceptance).resolve()
    registry_path = Path(args.experiment_registry).resolve()
    for path in (
        visible_path,
        split_path,
        context_path,
        stage1_path,
        stage1_report_path,
        stage1_log_path,
        stage1_gradient_audit_path,
        stage1_portability_path,
        stage1_acceptance_path,
        registry_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    experiment_registry = _load_json(registry_path)
    _validate_experiment_registry(experiment_registry, args)

    locked_records, split_manifest = directional.validate_locked(
        visible_path, split_path, int(args.seed), int(args.forget_num)
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("neuron suppression requires the locked direct-only split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = _record_views(locked_records)
    context_manifest = _load_json(context_path)
    stage1_state = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("Stage-1 writer state must be a mapping")
    stage1_report = _load_json(stage1_report_path)
    stage1_portability = _load_json(stage1_portability_path)
    stage1_acceptance = _load_json(stage1_acceptance_path)
    _validate_firewall(context_manifest, stage1_state)
    if compositional_method.sha256_file(context_path) != str(
        stage1_state["context_manifest_sha256"]
    ):
        raise RuntimeError(
            "Stage-1 writer and training-safe context manifest hashes differ"
        )
    clean_writer_receipt = _validate_clean_stage1_lineage(
        context_manifest,
        stage1_state,
        stage1_report,
        context_path,
        stage1_log_path,
    )
    if str(Path(args.model_path).resolve()) != str(
        clean_writer_receipt["base_model_path"]
    ):
        raise RuntimeError("decoder Base-model path differs from Stage-1 lineage")
    visible_sha256 = compositional_method.sha256_file(visible_path)
    split_sha256 = compositional_method.sha256_file(split_path)
    if (
        str(context_manifest.get("source_training_visible_sha256") or "")
        != visible_sha256
    ):
        raise RuntimeError(
            "training-visible file is not the exact file bound into the Stage-1 "
            "context manifest"
        )
    if str(context_manifest.get("source_split_manifest_sha256") or "") != split_sha256:
        raise RuntimeError(
            "split manifest is not the exact manifest bound into the Stage-1 "
            "context manifest"
        )
    case_ids = [int(record["case_id"]) for record in records]
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("Stage-1 cases do not match the locked training view")
    stored_preflight_threshold = stage1_state.get("writer_positive_amplitude_threshold")
    if stored_preflight_threshold is not None and not math.isclose(
        float(stored_preflight_threshold),
        float(args.writer_preflight_amplitude_threshold),
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "registered writer preflight threshold differs from the frozen writer"
        )
    if dict(stage1_acceptance.get("training_safe_portability", {})) != dict(
        stage1_portability
    ):
        raise RuntimeError(
            "clean Stage-1 acceptance embeds a different portability preflight"
        )
    clean_acceptance_receipt = _validate_clean_stage1_acceptance(
        stage1_acceptance,
        seed=int(args.seed),
        case_ids=case_ids,
        expected_artifacts={
            "training_visible_sha256": visible_sha256,
            "split_manifest_sha256": split_sha256,
            "context_manifest_sha256": compositional_method.sha256_file(context_path),
            "stage1_state_sha256": compositional_method.sha256_file(stage1_path),
            "stage1_report_sha256": compositional_method.sha256_file(
                stage1_report_path
            ),
            "stage1_writer_log_sha256": compositional_method.sha256_file(
                stage1_log_path
            ),
            "stage1_gradient_conflict_audit_sha256": (
                compositional_method.sha256_file(stage1_gradient_audit_path)
            ),
            "training_safe_portability_sha256": compositional_method.sha256_file(
                stage1_portability_path
            ),
        },
        amplitude_threshold=float(args.writer_preflight_amplitude_threshold),
        minimum_global_fraction=float(args.writer_preflight_min_global_fraction),
        minimum_record_fraction=float(args.writer_preflight_min_record_fraction),
    )
    context_sets = _context_sets_by_case(context_manifest, records)

    firewall_receipt = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "training_visible_path": str(visible_path),
        "training_visible_sha256": visible_sha256,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_sha256,
        "context_manifest_path": str(context_path),
        "context_manifest_sha256": compositional_method.sha256_file(context_path),
        "stage1_state_path": str(stage1_path),
        "stage1_state_sha256": compositional_method.sha256_file(stage1_path),
        "stage1_report_path": str(stage1_report_path),
        "stage1_report_sha256": compositional_method.sha256_file(stage1_report_path),
        "stage1_writer_log_path": str(stage1_log_path),
        "stage1_writer_log_sha256": compositional_method.sha256_file(stage1_log_path),
        "stage1_gradient_conflict_audit_path": str(stage1_gradient_audit_path),
        "stage1_gradient_conflict_audit_sha256": (
            compositional_method.sha256_file(stage1_gradient_audit_path)
        ),
        "clean_stage1_portability_path": str(stage1_portability_path),
        "clean_stage1_portability_sha256": compositional_method.sha256_file(
            stage1_portability_path
        ),
        "clean_stage1_acceptance_path": str(stage1_acceptance_path),
        "clean_stage1_acceptance_sha256": compositional_method.sha256_file(
            stage1_acceptance_path
        ),
        "clean_stage1_writer": clean_writer_receipt,
        "clean_stage1_acceptance": clean_acceptance_receipt,
        "experiment_label": str(args.experiment_label),
        "detector_initialization": str(args.detector_initialization),
        "frozen_v3_2_run_dir": (
            str(Path(args.frozen_v3_2_run_dir).resolve())
            if args.frozen_v3_2_run_dir
            else None
        ),
        "frozen_v3_4_run_dir": (
            str(Path(args.frozen_v3_4_run_dir).resolve())
            if args.frozen_v3_4_run_dir
            else None
        ),
        "frozen_v3_5_run_dir": (
            str(Path(args.frozen_v3_5_run_dir).resolve())
            if args.frozen_v3_5_run_dir
            else None
        ),
        "frozen_v3_5_1_run_dir": (
            str(Path(args.frozen_v3_5_1_run_dir).resolve())
            if args.frozen_v3_5_1_run_dir
            else None
        ),
        "frozen_v3_5_2_run_dir": (
            str(Path(args.frozen_v3_5_2_run_dir).resolve())
            if args.frozen_v3_5_2_run_dir
            else None
        ),
        "frozen_v3_5_3_run_dir": (
            str(Path(args.frozen_v3_5_3_run_dir).resolve())
            if args.frozen_v3_5_3_run_dir
            else None
        ),
        "frozen_v3_5_4_run_dir": (
            str(Path(args.frozen_v3_5_4_run_dir).resolve())
            if args.frozen_v3_5_4_run_dir
            else None
        ),
        "frozen_v3_5_5_run_dir": (
            str(Path(args.frozen_v3_5_5_run_dir).resolve())
            if args.frozen_v3_5_5_run_dir
            else None
        ),
        "frozen_v3_6_run_dir": (
            str(Path(args.frozen_v3_6_run_dir).resolve())
            if args.frozen_v3_6_run_dir
            else None
        ),
        "development_retain_shared_row_exposure": {
            "registered_as_consumed_architecture_motivation": True,
            "development_records": 9438,
            "official_retain_records_excluded": 1000,
            "audit_file_or_raw_development_records_available_to_learner": False,
            "used_for_selection_optimization_acceptance_or_retry": False,
        },
        "experiment_registry_path": str(registry_path),
        "experiment_registry_sha256": compositional_method.sha256_file(registry_path),
        "official_evaluation_file_argument_exists": False,
        "forbidden_evaluation_environment_variables_present": [],
        "data_access": dict(context_manifest["data_access"]),
        "evaluation_only_unavailable_during_training": [
            "official paraphrases",
            "official neighborhoods",
            "benchmark retain prompts",
            "official PPL text",
            "aliases",
            "descriptions",
            "adversarial prompts",
            "relearning attacks",
        ],
        "used_for_checkpoint_selection": [
            "locked direct forget prompts",
            "training-safe hand-authored relation-template and corpus-prefix positives",
            "training-visible compositional negatives",
            "disjoint Wikipedia protection prefixes",
            *(
                [
                    "hash-bound passed V3.2 training-only detector tensors",
                    "hash-bound V3.5 single-cell global writer-off rejection",
                    "hash-bound V3.5.1 non-owner collision diagnosis",
                    "hash-bound V3.5.2 exact duplicate-prompt contradiction",
                    "hash-bound V3.5.3 balanced-tail rejection",
                    "hash-bound V3.5.4 perfect detector and width-four rejection",
                    *(
                        [
                            "hash-bound V3.5.5 discarded width-16 mechanism-readiness pass",
                            "hash-bound V3.6 negative-locality rejection with no checkpoint",
                        ]
                        if full_preservation_mode
                        else []
                    ),
                ]
                if str(args.detector_initialization) == "frozen_v3_2"
                else []
            ),
        ],
        "ppl_corpus_partition": {
            "official_evaluation_document_interval": [0, 20],
            "training_protection_document_interval": [
                int(args.frequency_doc_start),
                int(args.frequency_doc_start) + int(args.frequency_docs),
            ],
            "overlap": 0,
        },
    }
    gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embeddings and LM head remain tied")
    device = gagd.first_device(model)
    llama_like = canonical.is_llama_like(model, tok)
    mlp = _resolve_swiglu_mlp(model, int(args.neuron_layer))
    output_head_digest_before = _tensor_digest(output_layer.weight)

    selected_embedding_rows = [
        int(value) for value in stage1_state.get("selected_embedding_rows", [])
    ]
    embedding_delta = stage1_state.get("embedding_delta")
    if not selected_embedding_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("Stage-1 state lacks sparse embedding rows/delta")
    if embedding_delta.shape != (
        len(selected_embedding_rows),
        int(input_layer.weight.shape[1]),
    ):
        raise RuntimeError("Stage-1 embedding delta has incompatible shape")
    observed_transformer_fingerprint = (
        compositional_method.frozen_transformer_fingerprint(model)
    )
    if not math.isclose(
        observed_transformer_fingerprint,
        float(clean_writer_receipt["base_transformer_fingerprint"]),
        rel_tol=1e-12,
        abs_tol=1e-3,
    ):
        raise RuntimeError("decoder Transformer differs from Stage-1 Base model")
    selected_row_index = torch.tensor(
        selected_embedding_rows,
        dtype=torch.long,
        device=input_layer.weight.device,
    )
    observed_selected_rows_sha256 = compositional_method.tensor_sha256(
        input_layer.weight.index_select(0, selected_row_index)
    )
    if observed_selected_rows_sha256 != str(
        clean_writer_receipt["base_selected_embedding_rows_sha256"]
    ):
        raise RuntimeError(
            "decoder Base embedding rows differ from Stage-1 writer lineage"
        )
    writer_present = str(args.writer_mode) == "embedding_keyed"
    applied_embedding_delta = (
        embedding_delta if writer_present else torch.zeros_like(embedding_delta)
    )
    embedding_writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer, selected_embedding_rows, applied_embedding_delta
    )

    (
        positive_instances,
        positive_owners,
        direct_flags,
    ) = compositional_method.build_prompt_instances(records, context_sets)
    negative_instances = _negative_prompt_instances(records, context_sets)
    positive_indices_by_record = [list() for _ in records]
    negative_indices_by_record = [list() for _ in records]
    for index, owner in enumerate(positive_owners):
        positive_indices_by_record[int(owner)].append(index)
    for index, instance in enumerate(negative_instances):
        negative_indices_by_record[int(instance.sampled_position)].append(index)
    if any(not indices for indices in positive_indices_by_record):
        raise RuntimeError("every record needs positive actuator contexts")
    if any(not indices for indices in negative_indices_by_record):
        raise RuntimeError("every record needs negative actuator contexts")
    registered_actuator = experiment_registry.get("actuator_training_revision", {})
    registered_contexts = (
        registered_actuator.get("frozen_training_contexts", {})
        if isinstance(registered_actuator, Mapping)
        else {}
    )
    registered_full_actuator = (
        registered_actuator.get("full_preservation", {})
        if isinstance(registered_actuator, Mapping)
        else {}
    )
    expected_positive_contexts = int(
        registered_contexts.get(
            "writer_on_positive",
            registered_full_actuator.get("positive_contexts_per_optimizer_update", -1),
        )
    )
    expected_negative_contexts = int(
        registered_contexts.get(
            "writer_on_negative",
            registered_full_actuator.get("negative_contexts_per_optimizer_update", -1),
        )
    )
    if len(positive_instances) != expected_positive_contexts:
        raise RuntimeError(
            "training-safe positive count differs from the actuator registry: "
            f"{len(positive_instances)} != {expected_positive_contexts}"
        )
    if len(negative_instances) != expected_negative_contexts:
        raise RuntimeError(
            "training-safe negative count differs from the actuator registry: "
            f"{len(negative_instances)} != {expected_negative_contexts}"
        )
    training_prompt_groups: List[List[str]] = []
    positive_prompts_by_record: List[List[str]] = []
    negative_prompts_by_record: List[List[str]] = []
    for record in records:
        context = context_sets[int(record["case_id"])]
        positives = [str(prompt) for prompt in context["positive_prompts"]]
        negatives = [str(row["prompt"]) for row in context["negative_contexts"]]
        positive_prompts_by_record.append(positives)
        negative_prompts_by_record.append(negatives)
        training_prompt_groups.append([*positives, *negatives])

    if writer_present:
        marker_map = stage1_state.get("markers")
        if not isinstance(marker_map, Mapping):
            raise RuntimeError("Stage-1 state lacks marker directions for preflight")
        writer_preflight = measure_training_safe_writer_preflight(
            model,
            tok,
            embedding_writer,
            positive_prompts_by_record,
            marker_map,
            case_ids,
            device,
            batch_size=int(args.cache_batch_size),
            amplitude_threshold=float(args.writer_preflight_amplitude_threshold),
            minimum_global_fraction=float(args.writer_preflight_min_global_fraction),
            minimum_record_fraction=float(args.writer_preflight_min_record_fraction),
        )
    else:
        writer_preflight = {
            "kind": "frozen_writer_training_safe_pre_decoder_preflight",
            "applicable": False,
            "writer_mode": "none",
            "passed": True,
            "official_evaluation_prompts_seen": 0,
        }
    gagd.write_json(out_dir / "writer_preflight_report.json", writer_preflight)
    if not bool(writer_preflight["passed"]):
        raise SystemExit(
            "frozen writer failed its training-safe portability preflight; "
            "decoder construction is refused"
        )

    documents = subject_writer.load_frequency_documents(
        args.wikidata_dir, int(args.frequency_doc_start), int(args.frequency_docs)
    )
    if int(args.frequency_docs) > 0 and not documents:
        raise RuntimeError(
            f"no disjoint protection documents loaded from {args.wikidata_dir!r}"
        )
    corpus_prompts = synthetic.corpus_context_prefixes(
        documents,
        count=int(args.corpus_protection_prompts),
        seed=int(args.seed) + 1907,
    )
    protected_prompts, protected_prompt_report = build_selection_protected_prompts(
        training_prompt_groups,
        corpus_prompts,
        total_limit=int(args.selection_protected_prompts),
        minimum_corpus=int(args.selection_min_corpus_prompts),
    )
    if not protected_prompt_report["corpus_quota_passed"]:
        raise RuntimeError("broad-corpus neuron-selection quota was not met")
    if not protected_prompt_report["full_budget_passed"]:
        raise RuntimeError("neuron-selection protected prompt budget was not filled")
    if not protected_prompt_report["all_training_groups_represented"]:
        raise RuntimeError("neuron-selection bank omitted one or more record groups")

    print(f"\nStage 0: {args.writer_mode} sparse-neuron selection")
    selection_start = time.time()
    embedding_writer.enabled = False
    protected_off = capture_base_last_token_activations(
        model,
        tok,
        mlp,
        protected_prompts,
        device,
        batch_size=int(args.cache_batch_size),
    )
    record_writer_on: List[torch.Tensor] = []
    record_writer_off: List[torch.Tensor] = []
    record_context_negative: List[torch.Tensor] = []
    for position, prompts in enumerate(positive_prompts_by_record):
        selected = prompts[: int(args.selection_positive_contexts)]
        selected_negative = negative_prompts_by_record[position][
            : int(args.selection_negative_contexts)
        ]
        embedding_writer.enabled = False
        off = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        embedding_writer.enabled = writer_present
        on = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        record_writer_off.append(off)
        record_writer_on.append(on)
        embedding_writer.enabled = False
        context_negative = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected_negative,
            device,
            batch_size=int(args.cache_batch_size),
        )
        record_context_negative.append(context_negative)
        displacement_label = (
            "writer activation displacement"
            if writer_present
            else "base positive/negative mean separation"
        )
        separation = (
            float((on - off).abs().max())
            if writer_present
            else float((on.mean(dim=0) - context_negative.mean(dim=0)).abs().max())
        )
        print(
            f"  case {case_ids[position]}: selection contexts={len(selected)}, "
            f"max {displacement_label}={separation:.4f}"
        )
    ownership, sign_groups, selection_reports = neuron_core.select_record_owned_neurons(
        record_writer_on,
        record_writer_off,
        protected_off,
        neurons_per_record=int(args.neurons_per_record),
        dormant_fraction=float(args.dormant_fraction),
        stability_weight=float(args.selection_stability_weight),
        selection_mode=str(args.selection_mode),
        context_negative_activations=record_context_negative,
        generator=torch.Generator(device="cpu").manual_seed(int(args.seed) + 4109),
    )
    # The actuator is multiplied by the absolute SwiGLU activation at runtime;
    # orient the audit/training statistic by the writer-on (or Base-positive for
    # the no-writer control) activation itself, not by an unavailable paired
    # Base displacement.
    sign_groups = []
    for record_index, neurons in enumerate(ownership):
        index = torch.tensor(neurons, dtype=torch.long)
        means = record_writer_on[record_index].index_select(1, index).mean(dim=0)
        sign_groups.append(
            torch.where(means >= 0, torch.ones_like(means), -torch.ones_like(means))
        )
    selected_neurons, flat_signs_cpu, local_groups = neuron_core.flatten_ownership(
        ownership, sign_groups
    )
    ownership_sha256 = selected_neuron_ownership_jq_compact_sha256(ownership)
    ownership_binding = experiment_registry["selected_neuron_ownership_binding"]
    bound_writer_configuration = ownership_binding["writer_configuration"]
    writer_configuration_matches_binding = bool(
        math.isclose(
            float(stage1_state.get("row_norm_cap")),
            float(bound_writer_configuration["row_norm_cap"]),
            abs_tol=1e-12,
        )
        and math.isclose(
            float(stage1_state.get("row_norm_cap_frequency_alpha")),
            float(bound_writer_configuration["row_norm_cap_frequency_alpha"]),
            abs_tol=1e-12,
        )
        and int(stage1_state.get("max_subject_token_frequency"))
        == int(bound_writer_configuration["max_subject_token_frequency"])
    )
    ownership_binding_required = bool(
        writer_present
        and str(args.experiment_label) == "primary"
        and writer_configuration_matches_binding
        and ownership_binding.get("required")
    )
    expected_ownership_sha256 = str(ownership_binding["jq_compact_sha256"])
    ownership_binding_passed = bool(
        not ownership_binding_required or ownership_sha256 == expected_ownership_sha256
    )
    selected_index = torch.tensor(selected_neurons, dtype=torch.long)
    selected_protected = protected_off.index_select(1, selected_index)
    base_down_norms = (
        mlp.down_proj.weight.detach()
        .float()
        .cpu()
        .index_select(1, selected_index)
        .norm(dim=0)
    )
    corpus_prompt_set = set(corpus_prompts)
    corpus_mask = torch.tensor(
        [prompt in corpus_prompt_set for prompt in protected_prompts],
        dtype=torch.bool,
    )
    if int(corpus_mask.sum()) < int(args.selection_min_corpus_prompts):
        raise RuntimeError("selected activation profile lost its corpus quota")
    selected_activation_profile = {
        "all_protected_prompts": activation_tail_profile(
            selected_protected,
            selected_neurons,
            activation_threshold=float(args.detector_off_abs_max),
            down_column_norms=base_down_norms,
        ),
        "broad_corpus_prompts": activation_tail_profile(
            selected_protected[corpus_mask],
            selected_neurons,
            activation_threshold=float(args.detector_off_abs_max),
            down_column_norms=base_down_norms,
        ),
        "interpretation": (
            "Baseline activation and residual-contribution bounds for every "
            "candidate ultimately edited. Quietness on the training prompts alone "
            "is not treated as evidence that a neuron lacks an original function."
        ),
    }
    selection_report = {
        "layer": int(args.neuron_layer),
        "zero_based_layer_index": True,
        "neurons_per_record": int(args.neurons_per_record),
        "selected_neuron_count": len(selected_neurons),
        "selected_neuron_ownership_jq_projection": ("[.ownership[].selected_neurons]"),
        "selected_neuron_ownership_jq_compact_sha256": ownership_sha256,
        "registered_selected_neuron_ownership_jq_compact_sha256": (
            expected_ownership_sha256
        ),
        "selected_neuron_ownership_binding_required": ownership_binding_required,
        "selected_neuron_ownership_binding_passed": ownership_binding_passed,
        "writer_configuration_matches_neuron_ownership_binding": (
            writer_configuration_matches_binding
        ),
        "intermediate_size": int(mlp.gate_proj.weight.shape[0]),
        "dormant_fraction": float(args.dormant_fraction),
        "writer_mode": str(args.writer_mode),
        "selection_mode": str(args.selection_mode),
        "detector_response_mode": str(args.detector_response_mode),
        "protected_prompt_count": len(protected_prompts),
        "protected_prompt_bank": protected_prompt_report,
        "protected_activation_rms": _distribution(
            [float(value) for value in protected_off.square().mean(dim=0).sqrt()]
        ),
        "selected_baseline_activation_profile": selected_activation_profile,
        "ownership": [
            {"case_id": case_ids[index], **row}
            for index, row in enumerate(selection_reports)
        ],
        "selection_wall_time_seconds": time.time() - selection_start,
    }
    gagd.write_json(out_dir / "neuron_selection_report.json", selection_report)
    if not ownership_binding_passed:
        raise RuntimeError(
            "V3.5.4 primary neuron ownership differs from the registered V3 through "
            "V3.5.3 "
            f"selection: observed {ownership_sha256}, expected "
            f"{expected_ownership_sha256}"
        )
    print(
        f"  selected {len(selected_neurons)} disjoint existing neurons at layer "
        f"{args.neuron_layer} ({args.neurons_per_record} per record)"
    )

    editor = neuron_core.SparseSwiGLUNeuronEditor(mlp, selected_neurons)
    if str(args.actuator_architecture) in {
        "isolated_thresholded_residual",
        "separate_threshold_gated_actuator_bank",
    }:
        guard = float(args.threshold_gate_numerical_guard)
        editor.configure_isolated_threshold_residual(
            local_groups,
            flat_signs_cpu,
            off_boundary=float(args.detector_off_abs_max) + guard,
            on_boundary=float(args.detector_positive_floor) - guard,
        )
    editor.install(mlp)
    flat_signs = flat_signs_cpu.to(device)
    embedding_writer.enabled = writer_present
    frozen_detector_import: Dict[str, Any] | None = None
    frozen_detector_source_gate: Dict[str, Any] | None = None
    frozen_v3_4_rejection: Dict[str, Any] | None = None
    frozen_v3_5_rejection: Dict[str, Any] | None = None
    frozen_v3_5_1_forensics: Dict[str, Any] | None = None
    frozen_v3_5_2_rejection: Dict[str, Any] | None = None
    frozen_v3_5_3_rejection: Dict[str, Any] | None = None
    frozen_v3_5_4_rejection: Dict[str, Any] | None = None
    frozen_v3_5_5_success: Dict[str, Any] | None = None
    frozen_v3_6_rejection: Dict[str, Any] | None = None
    if str(args.detector_initialization) == "frozen_v3_2":
        (
            frozen_detector_import,
            frozen_detector_source_gate,
        ) = import_frozen_v3_2_detector(
            Path(args.frozen_v3_2_run_dir),
            stage1_path=stage1_path,
            case_ids=case_ids,
            layer=int(args.neuron_layer),
            ownership=ownership,
            selected_neurons=selected_neurons,
            flat_signs=flat_signs_cpu,
            output_head_sha256=output_head_digest_before,
            editor=editor,
        )
        gagd.write_json(
            out_dir / "frozen_v3_2_detector_import.json", frozen_detector_import
        )
        firewall_receipt["frozen_v3_2_detector_import"] = frozen_detector_import
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print(
            "  imported exact V3.2 detector gate/up tensors; down_delta reset to zero"
        )
    if detector_replay_mode:
        frozen_v3_4_rejection = validate_frozen_v3_4_rejection(
            Path(args.frozen_v3_4_run_dir),
            case_ids=case_ids,
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_4_rejection_import.json", frozen_v3_4_rejection
        )
        firewall_receipt["frozen_v3_4_rejection_import"] = frozen_v3_4_rejection
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print(
            "  hash-bound preserved V3.4 reachability-pass/selectivity-reject outcome"
        )
        frozen_v3_5_rejection = validate_frozen_v3_5_rejection(
            Path(args.frozen_v3_5_run_dir),
            case_ids=case_ids,
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_5_rejection_import.json", frozen_v3_5_rejection
        )
        firewall_receipt["frozen_v3_5_rejection_import"] = frozen_v3_5_rejection
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print("  hash-bound preserved V3.5 single-cell writer-off gate rejection")
        frozen_v3_5_1_forensics = validate_frozen_v3_5_1_forensics(
            Path(args.frozen_v3_5_1_run_dir),
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_5_1_forensics_import.json",
            frozen_v3_5_1_forensics,
        )
        firewall_receipt["frozen_v3_5_1_forensics_import"] = frozen_v3_5_1_forensics
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        collision = frozen_v3_5_1_forensics["collision"]
        print(
            "  hash-bound V3.5.1 diagnosis: source case "
            f"{collision['source_case_id']} context "
            f"{collision['source_context_index']} -> detector case "
            f"{collision['detector_case_id']} (non-owner)"
        )
        frozen_v3_5_2_rejection = validate_frozen_v3_5_2_rejection(
            Path(args.frozen_v3_5_2_run_dir),
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_5_2_rejection_import.json",
            frozen_v3_5_2_rejection,
        )
        firewall_receipt["frozen_v3_5_2_rejection_import"] = frozen_v3_5_2_rejection
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print(
            "  hash-bound V3.5.2 contradiction: one exact prompt is positive for "
            "case 10472 and record-relative negative for case 19763"
        )
        frozen_v3_5_3_rejection = validate_frozen_v3_5_3_rejection(
            Path(args.frozen_v3_5_3_run_dir),
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_5_3_rejection_import.json",
            frozen_v3_5_3_rejection,
        )
        firewall_receipt["frozen_v3_5_3_rejection_import"] = (
            frozen_v3_5_3_rejection
        )
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print(
            "  hash-bound V3.5.3 rejection: 29/50 owner positives passed while "
            "negative and writer-off certificates stayed clean"
        )
        frozen_v3_5_4_rejection = validate_frozen_v3_5_4_rejection(
            Path(args.frozen_v3_5_4_run_dir),
            ownership_sha256=ownership_sha256,
        )
        gagd.write_json(
            out_dir / "frozen_v3_5_4_rejection_import.json",
            frozen_v3_5_4_rejection,
        )
        firewall_receipt["frozen_v3_5_4_rejection_import"] = (
            frozen_v3_5_4_rejection
        )
        gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
        print(
            "  hash-bound V3.5.4 result: detector 50/50 PASS; separate isolated "
            "width-four actuator cap 1.50 REJECTED with 147/200 columns saturated"
        )
        if full_preservation_mode:
            frozen_v3_5_5_success = validate_frozen_v3_5_5_success(
                Path(args.frozen_v3_5_5_run_dir)
            )
            gagd.write_json(
                out_dir / "frozen_v3_5_5_success_import.json",
                frozen_v3_5_5_success,
            )
            firewall_receipt["frozen_v3_5_5_success_import"] = (
                frozen_v3_5_5_success
            )
            gagd.write_json(
                out_dir / "training_firewall_receipt.json", firewall_receipt
            )
            print(
                "  hash-bound V3.5.5 result: native width 16 at cap 1.50 "
                "mechanism-ready; every feasibility fit discarded"
            )
            frozen_v3_6_rejection = validate_frozen_v3_6_rejection(
                Path(args.frozen_v3_6_run_dir)
            )
            gagd.write_json(
                out_dir / "frozen_v3_6_rejection_import.json",
                frozen_v3_6_rejection,
            )
            firewall_receipt["frozen_v3_6_rejection_import"] = (
                frozen_v3_6_rejection
            )
            gagd.write_json(
                out_dir / "training_firewall_receipt.json", firewall_receipt
            )
            print(
                "  hash-bound V3.6 result: width-16 reachability retained, "
                "native negative NLL drift 0.125 > 0.05; no checkpoint saved"
            )

    print("\nStage 1: build canonical multi-label prompt semantics")
    prompt_labels = build_multilabel_prompt_manifest(
        positive_prompts_by_record,
        negative_prompts_by_record,
        case_ids,
    )
    prompt_label_report = prompt_labels["report"]
    duplicate_digest = (
        frozen_v3_5_2_rejection["duplicate_prompt_sha256"]
        if frozen_v3_5_2_rejection is not None
        else ""
    )
    diagnosed_overlap = [
        row
        for row in prompt_label_report["role_overlap_entries"]
        if row["prompt_sha256"] == duplicate_digest
    ]
    diagnosed_overlap_passed = bool(
        len(diagnosed_overlap) == 1
        and 16 in diagnosed_overlap[0]["active_group_indices"]
        and 10472 in diagnosed_overlap[0]["active_case_ids"]
        and any(
            int(row["source_case_id"]) == 19763
            and int(row["source_context_index"]) == 4
            for row in diagnosed_overlap[0]["negative_occurrences"]
        )
    )
    if not diagnosed_overlap_passed:
        raise RuntimeError(
            "canonical prompt labels do not reproduce the frozen V3.5.2 "
            "positive/negative role overlap"
        )
    prompt_label_report.update(
        {
            "protocol": PROTOCOL,
            "frozen_v3_5_2_duplicate_prompt_reproduced": True,
            "diagnosed_duplicate_prompt_sha256": duplicate_digest,
            "diagnosed_duplicate_active_case_id": 10472,
            "diagnosed_duplicate_negative_source_case_id": 19763,
        }
    )
    prompt_label_path = out_dir / "multilabel_prompt_manifest.json"
    gagd.write_json(prompt_label_path, prompt_label_report)
    firewall_receipt["multilabel_prompt_manifest"] = {
        "path": str(prompt_label_path),
        "sha256": compositional_method.sha256_file(prompt_label_path),
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
    print(
        "  canonical prompts: "
        f"{prompt_label_report['unique_prompts']} unique across "
        f"{prompt_label_report['positive_occurrences']} positive and "
        f"{prompt_label_report['negative_occurrences']} negative occurrences; "
        f"{prompt_label_report['prompts_in_both_positive_and_negative_roles']} "
        "multi-role prompts"
    )

    (
        preservation_negative_instances,
        preservation_negative_indices_by_record,
        negative_precedence_report,
    ) = apply_forget_positive_precedence_to_negative_instances(
        negative_instances,
        prompt_labels,
        case_ids,
    )
    registered_precedence = registered_full_actuator.get(
        "negative_prompt_precedence"
    )
    expected_preservation_negatives = int(
        registered_full_actuator.get("negative_contexts_per_optimizer_update", -1)
    )
    expected_excluded_negatives = int(
        registered_full_actuator.get(
            "excluded_multi_role_negative_occurrences", -1
        )
    )
    expected_excluded_digest = str(
        registered_full_actuator.get("excluded_negative_prompt_sha256", "")
    )
    excluded_rows = negative_precedence_report["excluded_occurrences"]
    precedence_registered = bool(
        registered_precedence
        == "forget_positive_exact_prompt_precedes_preservation_negative"
        and len(preservation_negative_instances)
        == expected_preservation_negatives
        and len(excluded_rows) == expected_excluded_negatives
        and len(excluded_rows) == 1
        and excluded_rows[0]["prompt_sha256"] == expected_excluded_digest
        and excluded_rows[0]["prompt_sha256"] == duplicate_digest
        and excluded_rows[0]["negative_source_case_id"] == 19763
        and excluded_rows[0]["negative_source_context_index"] == 4
        and excluded_rows[0]["positive_active_case_ids"] == [10472]
    )
    if not precedence_registered:
        raise RuntimeError(
            "negative preservation precedence differs from the registered "
            "single exact-prompt coherence repair"
        )
    negative_precedence_report.update(
        {
            "registered": True,
            "source_v3_5_2_contradiction_hash_bound": True,
            "raw_detector_negative_occurrences_retained": len(
                negative_instances
            ),
        }
    )
    negative_precedence_path = (
        out_dir / "negative_preservation_precedence_report.json"
    )
    gagd.write_json(negative_precedence_path, negative_precedence_report)
    firewall_receipt["negative_preservation_precedence"] = {
        "path": str(negative_precedence_path),
        "sha256": compositional_method.sha256_file(negative_precedence_path),
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)
    print(
        "  preservation precedence: "
        f"{len(preservation_negative_instances)}/{len(negative_instances)} "
        "coherent negatives retained; the one exact forget-positive collision "
        "is excluded"
    )

    print("\nStage 1: cache and canonicalize frozen layer-input states")
    editor.write_enabled = False
    detector_trainable = bool(
        str(args.detector_initialization) == "train"
        or detector_replay_mode
    )
    editor.gate_delta.requires_grad_(detector_trainable)
    editor.up_delta.requires_grad_(detector_trainable)
    editor.down_delta.requires_grad_(False)
    embedding_writer.enabled = writer_present
    positive_hidden_cache = capture_grouped_mlp_input_hidden_states(
        model,
        tok,
        mlp,
        positive_prompts_by_record,
        device,
        batch_size=int(args.cache_batch_size),
    )
    negative_hidden_cache = capture_grouped_mlp_input_hidden_states(
        model,
        tok,
        mlp,
        negative_prompts_by_record,
        device,
        batch_size=int(args.cache_batch_size),
    )
    if writer_present:
        embedding_writer.enabled = False
        writer_off_hidden_cache = capture_grouped_mlp_input_hidden_states(
            model,
            tok,
            mlp,
            positive_prompts_by_record,
            device,
            batch_size=int(args.cache_batch_size),
        )
    else:
        writer_off_hidden_cache = [
            torch.empty((0, int(mlp.gate_proj.weight.shape[1]))) for _ in records
        ]
    embedding_writer.enabled = writer_present
    legacy_positive_hidden_cache = positive_hidden_cache
    legacy_negative_hidden_cache = negative_hidden_cache
    legacy_writer_off_hidden_cache = writer_off_hidden_cache

    combined_prompt_groups = [
        *positive_prompts_by_record,
        *negative_prompts_by_record,
    ]
    combined_hidden_groups = [
        *positive_hidden_cache,
        *negative_hidden_cache,
    ]
    canonical_combined, writer_on_canonicalization = canonicalize_grouped_hidden_states(
        combined_prompt_groups,
        combined_hidden_groups,
        canonical_prompts=prompt_labels["canonical_prompts"],
        prompt_to_index=prompt_labels["prompt_to_index"],
    )
    positive_hidden_cache = canonical_combined[: len(records)]
    negative_hidden_cache = canonical_combined[len(records) :]

    positive_canonical_prompts: List[str] = []
    positive_prompt_to_index: Dict[str, int] = {}
    for prompts in positive_prompts_by_record:
        for prompt in prompts:
            prompt = str(prompt)
            if prompt not in positive_prompt_to_index:
                positive_prompt_to_index[prompt] = len(positive_canonical_prompts)
                positive_canonical_prompts.append(prompt)
    (
        writer_off_hidden_cache,
        writer_off_canonicalization,
    ) = canonicalize_grouped_hidden_states(
        positive_prompts_by_record,
        writer_off_hidden_cache,
        canonical_prompts=positive_canonical_prompts,
        prompt_to_index=positive_prompt_to_index,
    )

    canonical_active_mask = prompt_labels["active_mask"]
    positive_active_mask_cache = [
        canonical_active_mask.index_select(0, torch.tensor(indices, dtype=torch.long))
        for indices in prompt_labels["positive_indices_by_record"]
    ]
    negative_active_mask_cache = [
        canonical_active_mask.index_select(0, torch.tensor(indices, dtype=torch.long))
        for indices in prompt_labels["negative_indices_by_record"]
    ]
    if any(int(rows.shape[0]) == 0 for rows in positive_hidden_cache):
        raise RuntimeError("every record needs at least one cached positive context")
    if any(int(rows.shape[0]) == 0 for rows in negative_hidden_cache):
        raise RuntimeError("every record needs at least one cached negative context")
    hidden_width = int(mlp.gate_proj.weight.shape[1])
    all_cache_rows = [
        *positive_hidden_cache,
        *negative_hidden_cache,
        *writer_off_hidden_cache,
    ]
    if any(
        rows.ndim != 2
        or int(rows.shape[1]) != hidden_width
        or rows.device.type != "cpu"
        or rows.dtype != torch.float32
        or not bool(torch.isfinite(rows).all())
        for rows in all_cache_rows
    ):
        raise RuntimeError("detector hidden-state cache failed shape/dtype validation")
    detector_cache_report = {
        "schema_version": 1,
        "kind": "training_only_frozen_mlp_input_detector_cache",
        "layer": int(args.neuron_layer),
        "hidden_width": hidden_width,
        "storage": "in_memory_cpu_float32",
        "detector_computation": (
            "cached_h -> selected gate_proj rows -> SiLU -> selected up_proj "
            "rows -> signed record-group activation"
        ),
        "cache_exactness": (
            "the selected MLP down projection is disabled during detector training; "
            "the cached MLP input is independent of learned gate/up rows; every "
            "exact duplicate prompt reuses one bit-identical canonical row"
        ),
        "canonical_prompt_labels": {
            "path": str(prompt_label_path),
            "sha256": compositional_method.sha256_file(prompt_label_path),
        },
        "writer_on_canonicalization": writer_on_canonicalization,
        "writer_off_canonicalization": writer_off_canonicalization,
        "records": len(records),
        "writer_on_positive_contexts": sum(
            int(rows.shape[0]) for rows in positive_hidden_cache
        ),
        "writer_on_negative_contexts": sum(
            int(rows.shape[0]) for rows in negative_hidden_cache
        ),
        "writer_off_positive_contexts": sum(
            int(rows.shape[0]) for rows in writer_off_hidden_cache
        ),
        "positive_context_mode": str(args.detector_positive_contexts),
        "negative_context_mode": str(args.detector_negative_contexts),
        "per_record": [
            {
                "record_index": index,
                "case_id": int(case_ids[index]),
                "writer_on_positive_contexts": int(
                    positive_hidden_cache[index].shape[0]
                ),
                "writer_on_negative_contexts": int(
                    negative_hidden_cache[index].shape[0]
                ),
                "writer_off_positive_contexts": int(
                    writer_off_hidden_cache[index].shape[0]
                ),
            }
            for index in range(len(records))
        ],
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(
        out_dir / "detector_hidden_cache_report.json", detector_cache_report
    )
    print(
        "  cached "
        f"{detector_cache_report['writer_on_positive_contexts']} writer-on positives, "
        f"{detector_cache_report['writer_on_negative_contexts']} writer-on negatives, "
        f"and {detector_cache_report['writer_off_positive_contexts']} writer-off "
        "positives"
    )

    def cached_detector_responses(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or int(hidden.shape[1]) != hidden_width:
            raise ValueError("cached detector hidden states have the wrong shape")
        edited_activation = editor.edited_selected_activations(
            hidden.to(device=device, non_blocking=True)
        )
        return neuron_core.signed_group_activations(
            edited_activation, local_groups, flat_signs
        )

    detector_microbatches = math.ceil(
        len(records) / max(1, int(args.detector_record_batch))
    )
    detector_optimizer_steps_this_run = (
        int(args.detector_steps)
        if (
            str(args.detector_initialization) == "train"
            or detector_replay_mode
        )
        else 0
    )

    def detector_optimization_metadata() -> Dict[str, Any]:
        initialization = str(args.detector_initialization)
        if (
            initialization == "frozen_v3_2"
            and detector_replay_mode
        ):
            revision = "v3.5.4_canonical_multilabel_balanced_tail_repair"
        elif initialization == "frozen_v3_2":
            revision = "frozen_v3.2_replay"
        else:
            revision = "control_training"
        return {
            "revision": revision,
            "initialization": initialization,
            "update_coverage": "all_records_accumulated",
            "records_per_optimizer_update": len(records),
            "record_microbatch_capacity": int(args.detector_record_batch),
            "microbatches_per_optimizer_update": detector_microbatches,
            "positive_context_mode": str(args.detector_positive_contexts),
            "negative_context_mode": str(args.detector_negative_contexts),
            "tail_k": int(args.detector_tail_k),
            "prompt_label_semantics": "canonical_exact_prompt_multilabel",
            "duplicate_hidden_state_policy": "bit_identical_canonical_reuse",
            "positive_objective": (
                "active_label_equal_record_mean_plus_worst_k"
            ),
            "negative_objective": (
                "source_owner_equal_record_mean_plus_worst_k"
            ),
            "cross_objective": (
                "inactive_label_equal_record_mean_plus_worst_k"
            ),
            "writer_off_objective": (
                "all_detector_groups_equal_source_record_mean_plus_worst_k"
                if detector_replay_mode
                else "historical_source_owner_only_objective"
            ),
            "writer_off_groups_per_context": (
                len(records) if detector_replay_mode else 1
            ),
            "global_tail_weight": float(args.detector_global_tail_weight),
            "global_tail_terms": "diagnostic_only_not_optimized",
            "forensic_replay_only": False,
            "optimizer_constructed": detector_optimizer_steps_this_run > 0,
            "training_positive_floor": float(args.detector_training_positive_floor),
            "training_off_abs_max": float(args.detector_training_off_abs_max),
            "certificate_positive_floor": float(args.detector_positive_floor),
            "certificate_off_abs_max": float(args.detector_off_abs_max),
            "certificate_abs_tolerance": float(args.detector_certificate_abs_tolerance),
            "certificate_thresholds_unchanged_from_v3_1": True,
            "gradient_normalization": (
                "equal_record_mean_plus_per_record_worst_k"
            ),
            "source_optimizer_steps": 1000 if initialization == "frozen_v3_2" else 0,
            "optimizer_steps_this_run": detector_optimizer_steps_this_run,
            "record_exposures_this_run": detector_optimizer_steps_this_run
            * len(records),
            "gradient_clip_frequency": "once_per_optimizer_update",
            "norm_projection_frequency": "once_per_optimizer_update",
            "endpoint_audit_phases": [
                "pre_update",
                "post_adam",
                "post_projection",
                "final_fresh_full_context_certificate",
            ],
            "complete_training_log_required": True,
            "selected_neuron_ownership_jq_compact_sha256": ownership_sha256,
            "cached_mlp_inputs": True,
        }

    @torch.no_grad()
    def record_cached_detector_responses(
        hidden_groups: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for record_index, hidden in enumerate(hidden_groups):
            if int(hidden.shape[0]) == 0:
                result.append(torch.empty(0))
                continue
            response = cached_detector_responses(hidden)
            result.append(response[:, record_index].detach().cpu())
        return result

    @torch.no_grad()
    def record_cached_selected_activations(
        hidden_groups: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for hidden in hidden_groups:
            if int(hidden.shape[0]) == 0:
                result.append(torch.empty((0, len(selected_neurons))))
                continue
            result.append(
                editor.edited_selected_activations(
                    hidden.to(device=device, non_blocking=True)
                )
                .detach()
                .float()
                .cpu()
            )
        return result

    def paired_ratio_summary(
        writer_on: torch.Tensor,
        writer_off: torch.Tensor,
    ) -> Dict[str, Any]:
        ratios = neuron_core.paired_on_off_ratios(
            writer_on,
            writer_off,
            epsilon=float(args.detector_selectivity_ratio_epsilon),
        )
        return {
            key: _distribution(value.reshape(-1).tolist())
            for key, value in ratios.items()
        }

    @torch.no_grad()
    def detector_selectivity_audit() -> Dict[str, Any]:
        if not writer_present:
            raise RuntimeError("detector selectivity audit requires a writer")
        on_responses = record_cached_detector_responses(positive_hidden_cache)
        off_responses = record_cached_detector_responses(writer_off_hidden_cache)
        on_activations = record_cached_selected_activations(positive_hidden_cache)
        off_activations = record_cached_selected_activations(writer_off_hidden_cache)
        aggregate: Dict[str, Dict[str, List[torch.Tensor]]] = {
            name: {"writer_on": [], "writer_off": []}
            for name in (
                "owned_signed_group_response",
                "owned_neuron_abs_activation",
                "owned_activation_vector_l2",
                "all_selected_activation_vector_l2",
            )
        }
        per_record: List[Dict[str, Any]] = []
        for record_index in range(len(records)):
            on_response = on_responses[record_index]
            off_response = off_responses[record_index]
            on_all = on_activations[record_index]
            off_all = off_activations[record_index]
            if on_response.shape != off_response.shape or on_all.shape != off_all.shape:
                raise RuntimeError("writer-on/off detector audit lost prompt pairing")
            group_index = torch.tensor(local_groups[record_index], dtype=torch.long)
            on_owned = on_all.index_select(1, group_index)
            off_owned = off_all.index_select(1, group_index)
            values = {
                "owned_signed_group_response": (on_response, off_response),
                "owned_neuron_abs_activation": (on_owned.abs(), off_owned.abs()),
                "owned_activation_vector_l2": (
                    on_owned.norm(dim=1),
                    off_owned.norm(dim=1),
                ),
                "all_selected_activation_vector_l2": (
                    on_all.norm(dim=1),
                    off_all.norm(dim=1),
                ),
            }
            summaries = {
                name: paired_ratio_summary(on_value, off_value)
                for name, (on_value, off_value) in values.items()
            }
            response_ratios = neuron_core.paired_on_off_ratios(
                on_response,
                off_response,
                epsilon=float(args.detector_selectivity_ratio_epsilon),
            )
            owned_l2_ratios = neuron_core.paired_on_off_ratios(
                on_owned.norm(dim=1),
                off_owned.norm(dim=1),
                epsilon=float(args.detector_selectivity_ratio_epsilon),
            )
            all_l2_ratios = neuron_core.paired_on_off_ratios(
                on_all.norm(dim=1),
                off_all.norm(dim=1),
                epsilon=float(args.detector_selectivity_ratio_epsilon),
            )
            for name, (on_value, off_value) in values.items():
                aggregate[name]["writer_on"].append(on_value.reshape(-1))
                aggregate[name]["writer_off"].append(off_value.reshape(-1))
            per_record.append(
                {
                    "record_index": record_index,
                    "case_id": int(case_ids[record_index]),
                    "positive_contexts": int(on_response.numel()),
                    "metrics": summaries,
                    "paired_context_rows": [
                        {
                            "context_index": context_index,
                            "writer_on_owned_group_response": float(
                                on_response[context_index]
                            ),
                            "writer_off_owned_group_response": float(
                                off_response[context_index]
                            ),
                            "owned_group_on_off_ratio": float(
                                response_ratios["writer_on_to_off_ratio"][context_index]
                            ),
                            "writer_on_owned_activations": [
                                float(value) for value in on_owned[context_index]
                            ],
                            "writer_off_owned_activations": [
                                float(value) for value in off_owned[context_index]
                            ],
                            "owned_activation_l2_on_off_ratio": float(
                                owned_l2_ratios["writer_on_to_off_ratio"][context_index]
                            ),
                            "all_selected_activation_l2_on_off_ratio": float(
                                all_l2_ratios["writer_on_to_off_ratio"][context_index]
                            ),
                        }
                        for context_index in range(int(on_response.numel()))
                    ],
                }
            )

        aggregate_summaries: Dict[str, Any] = {}
        for name, parts in aggregate.items():
            aggregate_summaries[name] = paired_ratio_summary(
                torch.cat(parts["writer_on"]), torch.cat(parts["writer_off"])
            )
        warning_ratio = float(args.detector_selectivity_warning_ratio)
        warning_flags = {
            name: bool(
                float(summary["writer_on_to_off_ratio"].get("p10", 0.0)) < warning_ratio
            )
            for name, summary in aggregate_summaries.items()
        }
        return {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_repaired_detector_selectivity_audit",
            "protocol": PROTOCOL,
            "parameter_state": (
                "V3.2-initialized gate/up after the registered V3.5.4 balanced "
                "canonical multi-label repair; exact-zero down_delta"
            ),
            "paired_contexts": len(positive_instances),
            "record_count": len(records),
            "ratio_denominator_epsilon": float(args.detector_selectivity_ratio_epsilon),
            "heuristic_warning_ratio": warning_ratio,
            "heuristic_is_acceptance_gate": False,
            "interpretation": (
                "The ratio warning is a mechanism diagnostic, not a proof of NLL "
                "gain or a substitute for the direct writer-off behavior audit."
            ),
            "aggregate": aggregate_summaries,
            "p10_below_heuristic_warning": warning_flags,
            "any_p10_below_heuristic_warning": any(warning_flags.values()),
            "per_record": per_record,
            "official_evaluation_prompts_seen": 0,
        }

    @torch.no_grad()
    def isolated_threshold_gate_report(
        *, phase: str, optimizer_step: int
    ) -> Dict[str, Any]:
        """Certify response bounds and runtime gates over every detector cell."""

        if editor.residual_mode != "isolated_thresholded_residual":
            raise RuntimeError(
                "threshold gate report requires the isolated architecture"
            )

        def gates_from_responses(responses: torch.Tensor) -> torch.Tensor:
            width = float(editor.threshold_gate_on_boundary) - float(
                editor.threshold_gate_off_boundary
            )
            return (
                (responses - float(editor.threshold_gate_off_boundary)) / width
            ).clamp(min=0.0, max=1.0)

        endpoint_tolerance = 1e-6
        certificate_tolerance = float(args.detector_certificate_abs_tolerance)
        off_limit = float(args.detector_off_abs_max) + certificate_tolerance
        positive_limit = float(args.detector_positive_floor) - certificate_tolerance
        per_record: List[Dict[str, Any]] = []
        response_parts: Dict[str, List[torch.Tensor]] = {
            "positive_owner": [],
            "positive_cross": [],
            "negative": [],
            "writer_off": [],
        }
        gate_parts: Dict[str, List[torch.Tensor]] = {
            name: [] for name in response_parts
        }
        violations: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name in response_parts
        }
        raw_response_matrices: List[Dict[str, Any]] = []

        def prompt_digest(category: str, source: int, context: int) -> str:
            prompts = (
                negative_prompts_by_record
                if category == "negative"
                else positive_prompts_by_record
            )
            return hashlib.sha256(
                str(prompts[source][context]).encode("utf-8")
            ).hexdigest()

        def context_provenance(
            category: str, source: int, context: int
        ) -> Dict[str, Any]:
            record_context = context_sets[int(case_ids[source])]
            if category == "negative":
                rows = record_context.get("negative_contexts", [])
            else:
                rows = record_context.get("positive_prompt_provenance", [])
            if not isinstance(rows, list) or context >= len(rows):
                return {"available": False}
            row = rows[context]
            if not isinstance(row, Mapping):
                return {"available": False}
            return {
                "available": True,
                **{
                    str(key): value
                    for key, value in row.items()
                    if str(key) != "prompt"
                },
            }

        def append_off_violations(
            *,
            category: str,
            source_record: int,
            responses: torch.Tensor,
            gates: torch.Tensor,
            detector_indices: Sequence[int],
        ) -> None:
            response_bad = responses.abs().gt(off_limit)
            gate_bad = gates.abs().gt(endpoint_tolerance)
            bad = response_bad | gate_bad
            for context_index, local_group_index in torch.nonzero(
                bad, as_tuple=False
            ).tolist():
                detector_group = int(detector_indices[local_group_index])
                violations[category].append(
                    {
                        "source_record_index": int(source_record),
                        "source_case_id": int(case_ids[source_record]),
                        "source_context_index": int(context_index),
                        "source_prompt_sha256": prompt_digest(
                            category, source_record, context_index
                        ),
                        "source_context_provenance": context_provenance(
                            category, source_record, context_index
                        ),
                        "detector_group_index": detector_group,
                        "detector_case_id": int(case_ids[detector_group]),
                        "owner_group": detector_group == int(source_record),
                        "response": float(responses[context_index, local_group_index]),
                        "response_abs": float(
                            responses[context_index, local_group_index].abs()
                        ),
                        "gate": float(gates[context_index, local_group_index]),
                        "response_certificate_violation": bool(
                            response_bad[context_index, local_group_index]
                        ),
                        "gate_endpoint_violation": bool(
                            gate_bad[context_index, local_group_index]
                        ),
                    }
                )

        for record_index in range(len(records)):
            positive_response = (
                cached_detector_responses(positive_hidden_cache[record_index])
                .detach()
                .cpu()
            )
            negative_response = (
                cached_detector_responses(negative_hidden_cache[record_index])
                .detach()
                .cpu()
            )
            off_response = (
                cached_detector_responses(writer_off_hidden_cache[record_index])
                .detach()
                .cpu()
            )
            positive_gate = gates_from_responses(positive_response)
            negative_gate = gates_from_responses(negative_response)
            off_gate = gates_from_responses(off_response)
            owner_positive_response = positive_response[:, record_index]
            owner_positive_gate = positive_gate[:, record_index]
            nonowner_indices = [
                index for index in range(len(records)) if index != record_index
            ]
            nonowner_index = torch.tensor(nonowner_indices, dtype=torch.long)
            positive_cross_response = positive_response.index_select(1, nonowner_index)
            positive_cross_gate = positive_gate.index_select(1, nonowner_index)
            response_parts["positive_owner"].append(owner_positive_response.reshape(-1))
            response_parts["positive_cross"].append(positive_cross_response.reshape(-1))
            response_parts["negative"].append(negative_response.reshape(-1))
            response_parts["writer_off"].append(off_response.reshape(-1))
            gate_parts["positive_owner"].append(owner_positive_gate.reshape(-1))
            gate_parts["positive_cross"].append(positive_cross_gate.reshape(-1))
            gate_parts["negative"].append(negative_gate.reshape(-1))
            gate_parts["writer_off"].append(off_gate.reshape(-1))

            owner_bad = owner_positive_response.lt(
                positive_limit
            ) | owner_positive_gate.lt(1.0 - endpoint_tolerance)
            for context_index in (
                torch.nonzero(owner_bad, as_tuple=False).reshape(-1).tolist()
            ):
                violations["positive_owner"].append(
                    {
                        "source_record_index": record_index,
                        "source_case_id": int(case_ids[record_index]),
                        "source_context_index": int(context_index),
                        "source_prompt_sha256": prompt_digest(
                            "positive_owner", record_index, context_index
                        ),
                        "source_context_provenance": context_provenance(
                            "positive_owner", record_index, context_index
                        ),
                        "detector_group_index": record_index,
                        "detector_case_id": int(case_ids[record_index]),
                        "owner_group": True,
                        "response": float(owner_positive_response[context_index]),
                        "response_abs": float(
                            owner_positive_response[context_index].abs()
                        ),
                        "gate": float(owner_positive_gate[context_index]),
                        "response_certificate_violation": bool(
                            owner_positive_response[context_index] < positive_limit
                        ),
                        "gate_endpoint_violation": bool(
                            owner_positive_gate[context_index]
                            < 1.0 - endpoint_tolerance
                        ),
                    }
                )
            append_off_violations(
                category="positive_cross",
                source_record=record_index,
                responses=positive_cross_response,
                gates=positive_cross_gate,
                detector_indices=nonowner_indices,
            )
            append_off_violations(
                category="negative",
                source_record=record_index,
                responses=negative_response,
                gates=negative_gate,
                detector_indices=list(range(len(records))),
            )
            append_off_violations(
                category="writer_off",
                source_record=record_index,
                responses=off_response,
                gates=off_gate,
                detector_indices=list(range(len(records))),
            )
            raw_response_matrices.append(
                {
                    "source_record_index": record_index,
                    "source_case_id": int(case_ids[record_index]),
                    "detector_group_case_ids": [int(value) for value in case_ids],
                    "positive": positive_response.tolist(),
                    "negative": negative_response.tolist(),
                    "writer_off": off_response.tolist(),
                }
            )
            per_record.append(
                {
                    "record_index": record_index,
                    "case_id": int(case_ids[record_index]),
                    "positive_contexts": int(owner_positive_gate.numel()),
                    "positive_owner_response_min": float(owner_positive_response.min()),
                    "positive_owner_gate_min": float(owner_positive_gate.min()),
                    "positive_owner_gate_median": float(owner_positive_gate.median()),
                    "positive_cross_response_abs_max": float(
                        positive_cross_response.abs().max()
                    ),
                    "positive_cross_gate_abs_max": float(
                        positive_cross_gate.abs().max()
                    ),
                    "negative_response_abs_max": float(negative_response.abs().max()),
                    "negative_gate_abs_max": float(negative_gate.abs().max()),
                    "writer_off_response_abs_max": float(off_response.abs().max()),
                    "writer_off_gate_abs_max": float(off_gate.abs().max()),
                }
            )

        responses = {name: torch.cat(parts) for name, parts in response_parts.items()}
        gates = {name: torch.cat(parts) for name, parts in gate_parts.items()}
        response_violation_counts = {
            name: sum(bool(row["response_certificate_violation"]) for row in rows)
            for name, rows in violations.items()
        }
        gate_violation_counts = {
            name: sum(bool(row["gate_endpoint_violation"]) for row in rows)
            for name, rows in violations.items()
        }
        writer_off_gate_violations = [
            row
            for row in violations["writer_off"]
            if bool(row["gate_endpoint_violation"])
        ]
        writer_off_gate_argmax = (
            max(writer_off_gate_violations, key=lambda row: abs(float(row["gate"])))
            if writer_off_gate_violations
            else None
        )
        checks = {
            "all_positive_owner_responses_certified": not violations["positive_owner"],
            "all_positive_cross_responses_certified": not violations["positive_cross"],
            "all_negative_responses_certified": not violations["negative"],
            "all_writer_off_responses_certified": not violations["writer_off"],
            "all_positive_owner_gates_one": bool(
                float(gates["positive_owner"].min()) >= 1.0 - endpoint_tolerance
            ),
            "all_positive_cross_gates_zero": bool(
                float(gates["positive_cross"].abs().max()) <= endpoint_tolerance
            ),
            "all_negative_gates_zero": bool(
                float(gates["negative"].abs().max()) <= endpoint_tolerance
            ),
            "all_writer_off_gates_zero": bool(
                float(gates["writer_off"].abs().max()) <= endpoint_tolerance
            ),
        }
        return {
            "schema_version": 2,
            "kind": "mcf_embedding_keyed_neuron_global_isolation_gate",
            "protocol": PROTOCOL,
            "phase": str(phase),
            "optimizer_step": int(optimizer_step),
            "architecture": str(args.actuator_architecture),
            "definition": (
                "clip((signed_group_response - off_boundary) / "
                "(on_boundary - off_boundary), 0, 1)"
            ),
            "certificate_scope": (
                "owner positive plus every non-owner positive, every negative group, "
                "and every writer-off group"
            ),
            "threshold_calibration": (
                "none; global 0.20/0.25 boundaries inherited unchanged from V3.5"
            ),
            "boundaries": {
                "registered_detector_off_abs_max": float(args.detector_off_abs_max),
                "registered_detector_positive_floor": float(
                    args.detector_positive_floor
                ),
                "certificate_abs_tolerance": certificate_tolerance,
                "numerical_guard": float(args.threshold_gate_numerical_guard),
                "runtime_off_boundary": float(editor.threshold_gate_off_boundary),
                "runtime_on_boundary": float(editor.threshold_gate_on_boundary),
                "endpoint_tolerance": endpoint_tolerance,
            },
            "aggregate": {
                **{
                    f"{name}_response": _distribution(value.tolist())
                    for name, value in responses.items()
                },
                **{
                    f"{name}_gate": _distribution(value.tolist())
                    for name, value in gates.items()
                },
            },
            "violation_counts": {name: len(rows) for name, rows in violations.items()},
            "response_certificate_violation_counts": response_violation_counts,
            "gate_endpoint_violation_counts": gate_violation_counts,
            "violating_cells": violations,
            "writer_off_gate_argmax_cell": writer_off_gate_argmax,
            "raw_signed_response_matrices_by_source_record": raw_response_matrices,
            "checks": checks,
            "passed": all(checks.values()),
            "per_record": per_record,
            "official_evaluation_prompts_seen": 0,
        }

    @torch.no_grad()
    def multilabel_threshold_gate_report(
        *, phase: str, optimizer_step: int
    ) -> Dict[str, Any]:
        """Certify canonical active/inactive labels over every detector cell."""

        if editor.residual_mode != "isolated_thresholded_residual":
            raise RuntimeError("multi-label gate requires the isolated architecture")

        def gates_from_responses(responses: torch.Tensor) -> torch.Tensor:
            width = float(editor.threshold_gate_on_boundary) - float(
                editor.threshold_gate_off_boundary
            )
            return (
                (responses - float(editor.threshold_gate_off_boundary)) / width
            ).clamp(min=0.0, max=1.0)

        endpoint_tolerance = 1e-6
        certificate_tolerance = float(args.detector_certificate_abs_tolerance)
        off_limit = float(args.detector_off_abs_max) + certificate_tolerance
        positive_limit = float(args.detector_positive_floor) - certificate_tolerance
        categories = (
            "positive_owner",
            "writer_on_active",
            "writer_on_inactive",
            "source_negative_owner",
            "writer_off",
        )
        response_parts: Dict[str, List[torch.Tensor]] = {
            name: [] for name in categories
        }
        gate_parts: Dict[str, List[torch.Tensor]] = {name: [] for name in categories}
        violations: Dict[str, List[Dict[str, Any]]] = {name: [] for name in categories}
        raw_response_matrices: List[Dict[str, Any]] = []

        def provenance(role: str, source: int, context: int) -> Dict[str, Any]:
            record_context = context_sets[int(case_ids[source])]
            rows = (
                record_context.get("negative_contexts", [])
                if role == "negative"
                else record_context.get("positive_prompt_provenance", [])
            )
            if not isinstance(rows, list) or context >= len(rows):
                return {"available": False}
            row = rows[context]
            if not isinstance(row, Mapping):
                return {"available": False}
            return {
                "available": True,
                **{
                    str(key): value
                    for key, value in row.items()
                    if str(key) != "prompt"
                },
            }

        def cell(
            *,
            role: str,
            source: int,
            context: int,
            detector_group: int,
            response: torch.Tensor,
            gate: torch.Tensor,
            active_label: bool,
        ) -> Dict[str, Any]:
            prompts = (
                negative_prompts_by_record
                if role == "negative"
                else positive_prompts_by_record
            )
            prompt = str(prompts[source][context])
            return {
                "source_role": role,
                "source_record_index": int(source),
                "source_case_id": int(case_ids[source]),
                "source_context_index": int(context),
                "source_prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "source_context_provenance": provenance(role, source, context),
                "detector_group_index": int(detector_group),
                "detector_case_id": int(case_ids[detector_group]),
                "source_owner_group": detector_group == int(source),
                "active_label": bool(active_label),
                "response": float(response),
                "response_abs": float(response.abs()),
                "gate": float(gate),
            }

        for source in range(len(records)):
            source_matrices: Dict[str, Any] = {
                "source_record_index": source,
                "source_case_id": int(case_ids[source]),
                "detector_group_case_ids": [int(value) for value in case_ids],
            }
            for role, cache, active_cache in (
                (
                    "positive",
                    positive_hidden_cache,
                    positive_active_mask_cache,
                ),
                (
                    "negative",
                    negative_hidden_cache,
                    negative_active_mask_cache,
                ),
            ):
                responses = cached_detector_responses(cache[source]).detach().cpu()
                gates = gates_from_responses(responses)
                active = active_cache[source]
                if active.shape != responses.shape:
                    raise RuntimeError("multi-label mask lost response alignment")
                response_parts["writer_on_active"].append(responses[active])
                gate_parts["writer_on_active"].append(gates[active])
                response_parts["writer_on_inactive"].append(responses[~active])
                gate_parts["writer_on_inactive"].append(gates[~active])

                active_response_bad = responses.lt(positive_limit) & active
                active_gate_bad = gates.lt(1.0 - endpoint_tolerance) & active
                for context, detector_group in torch.nonzero(
                    active_response_bad | active_gate_bad, as_tuple=False
                ).tolist():
                    row = cell(
                        role=role,
                        source=source,
                        context=context,
                        detector_group=detector_group,
                        response=responses[context, detector_group],
                        gate=gates[context, detector_group],
                        active_label=True,
                    )
                    row.update(
                        {
                            "response_certificate_violation": bool(
                                active_response_bad[context, detector_group]
                            ),
                            "gate_endpoint_violation": bool(
                                active_gate_bad[context, detector_group]
                            ),
                        }
                    )
                    violations["writer_on_active"].append(row)

                inactive_response_bad = responses.abs().gt(off_limit) & ~active
                inactive_gate_bad = gates.abs().gt(endpoint_tolerance) & ~active
                for context, detector_group in torch.nonzero(
                    inactive_response_bad | inactive_gate_bad, as_tuple=False
                ).tolist():
                    row = cell(
                        role=role,
                        source=source,
                        context=context,
                        detector_group=detector_group,
                        response=responses[context, detector_group],
                        gate=gates[context, detector_group],
                        active_label=False,
                    )
                    row.update(
                        {
                            "response_certificate_violation": bool(
                                inactive_response_bad[context, detector_group]
                            ),
                            "gate_endpoint_violation": bool(
                                inactive_gate_bad[context, detector_group]
                            ),
                        }
                    )
                    violations["writer_on_inactive"].append(row)

                if role == "positive":
                    owner_response = responses[:, source]
                    owner_gate = gates[:, source]
                    response_parts["positive_owner"].append(owner_response)
                    gate_parts["positive_owner"].append(owner_gate)
                    owner_bad_response = owner_response.lt(positive_limit)
                    owner_bad_gate = owner_gate.lt(1.0 - endpoint_tolerance)
                    for context in (
                        torch.nonzero(
                            owner_bad_response | owner_bad_gate, as_tuple=False
                        )
                        .reshape(-1)
                        .tolist()
                    ):
                        row = cell(
                            role=role,
                            source=source,
                            context=context,
                            detector_group=source,
                            response=owner_response[context],
                            gate=owner_gate[context],
                            active_label=True,
                        )
                        row.update(
                            {
                                "response_certificate_violation": bool(
                                    owner_bad_response[context]
                                ),
                                "gate_endpoint_violation": bool(
                                    owner_bad_gate[context]
                                ),
                            }
                        )
                        violations["positive_owner"].append(row)
                else:
                    owner_response = responses[:, source]
                    owner_gate = gates[:, source]
                    response_parts["source_negative_owner"].append(owner_response)
                    gate_parts["source_negative_owner"].append(owner_gate)
                    owner_bad_response = owner_response.abs().gt(off_limit)
                    owner_bad_gate = owner_gate.abs().gt(endpoint_tolerance)
                    for context in (
                        torch.nonzero(
                            owner_bad_response | owner_bad_gate, as_tuple=False
                        )
                        .reshape(-1)
                        .tolist()
                    ):
                        row = cell(
                            role=role,
                            source=source,
                            context=context,
                            detector_group=source,
                            response=owner_response[context],
                            gate=owner_gate[context],
                            active_label=False,
                        )
                        row.update(
                            {
                                "response_certificate_violation": bool(
                                    owner_bad_response[context]
                                ),
                                "gate_endpoint_violation": bool(
                                    owner_bad_gate[context]
                                ),
                            }
                        )
                        violations["source_negative_owner"].append(row)
                source_matrices[role] = responses.tolist()

            writer_off_response = (
                cached_detector_responses(writer_off_hidden_cache[source])
                .detach()
                .cpu()
            )
            writer_off_gate = gates_from_responses(writer_off_response)
            response_parts["writer_off"].append(writer_off_response.reshape(-1))
            gate_parts["writer_off"].append(writer_off_gate.reshape(-1))
            off_response_bad = writer_off_response.abs().gt(off_limit)
            off_gate_bad = writer_off_gate.abs().gt(endpoint_tolerance)
            for context, detector_group in torch.nonzero(
                off_response_bad | off_gate_bad, as_tuple=False
            ).tolist():
                row = cell(
                    role="positive",
                    source=source,
                    context=context,
                    detector_group=detector_group,
                    response=writer_off_response[context, detector_group],
                    gate=writer_off_gate[context, detector_group],
                    active_label=False,
                )
                row["source_role"] = "writer_off_positive"
                row.update(
                    {
                        "response_certificate_violation": bool(
                            off_response_bad[context, detector_group]
                        ),
                        "gate_endpoint_violation": bool(
                            off_gate_bad[context, detector_group]
                        ),
                    }
                )
                violations["writer_off"].append(row)
            source_matrices["writer_off"] = writer_off_response.tolist()
            raw_response_matrices.append(source_matrices)

        responses = {
            name: torch.cat(parts).reshape(-1) for name, parts in response_parts.items()
        }
        gates = {
            name: torch.cat(parts).reshape(-1) for name, parts in gate_parts.items()
        }
        response_violation_counts = {
            name: sum(bool(row["response_certificate_violation"]) for row in rows)
            for name, rows in violations.items()
        }
        gate_violation_counts = {
            name: sum(bool(row["gate_endpoint_violation"]) for row in rows)
            for name, rows in violations.items()
        }
        checks = {
            "all_positive_source_owners_active": not violations["positive_owner"],
            "all_writer_on_active_labels_one": not violations["writer_on_active"],
            "all_writer_on_inactive_labels_zero": not violations["writer_on_inactive"],
            "all_source_negative_owners_zero": not violations["source_negative_owner"],
            "all_writer_off_labels_zero": not violations["writer_off"],
        }
        writer_off_gate_violations = [
            row
            for row in violations["writer_off"]
            if bool(row["gate_endpoint_violation"])
        ]
        return {
            "schema_version": 3,
            "kind": "mcf_embedding_keyed_neuron_multilabel_global_isolation_gate",
            "protocol": PROTOCOL,
            "phase": str(phase),
            "optimizer_step": int(optimizer_step),
            "architecture": str(args.actuator_architecture),
            "definition": (
                "exact-prompt active labels use the 0.25 on certificate; every "
                "inactive writer-on label and every writer-off label uses the "
                "absolute 0.20 off certificate"
            ),
            "prompt_label_manifest": {
                "path": str(prompt_label_path),
                "sha256": compositional_method.sha256_file(prompt_label_path),
            },
            "threshold_calibration": "none; 0.20/0.25 boundaries unchanged",
            "boundaries": {
                "registered_detector_off_abs_max": float(args.detector_off_abs_max),
                "registered_detector_positive_floor": float(
                    args.detector_positive_floor
                ),
                "certificate_abs_tolerance": certificate_tolerance,
                "runtime_off_boundary": float(editor.threshold_gate_off_boundary),
                "runtime_on_boundary": float(editor.threshold_gate_on_boundary),
                "endpoint_tolerance": endpoint_tolerance,
            },
            "aggregate": {
                **{
                    f"{name}_response": _distribution(value.tolist())
                    for name, value in responses.items()
                },
                **{
                    f"{name}_gate": _distribution(value.tolist())
                    for name, value in gates.items()
                },
            },
            "response_certificate_violation_counts": response_violation_counts,
            "gate_endpoint_violation_counts": gate_violation_counts,
            "violating_cells": violations,
            "writer_off_gate_argmax_cell": (
                max(
                    writer_off_gate_violations,
                    key=lambda row: abs(float(row["gate"])),
                )
                if writer_off_gate_violations
                else None
            ),
            "raw_signed_response_matrices_by_source_record": raw_response_matrices,
            "checks": checks,
            "passed": all(checks.values()),
            "official_evaluation_prompts_seen": 0,
        }

    @torch.no_grad()
    def full_cached_detector_gate(
        *,
        phase: str,
        optimizer_step: int,
        positive_cache: Sequence[torch.Tensor] | None = None,
        negative_cache: Sequence[torch.Tensor] | None = None,
        writer_off_cache: Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, Any]:
        phase_contracts = {
            "pre_update": (
                "after final-step gradient accumulation and before gradient "
                "clipping or Adam"
            ),
            "post_adam": (
                "after final-step gradient clipping and Adam and before "
                "relative-norm projection"
            ),
            "post_projection": ("after final-step Adam and relative-norm projection"),
            "final_fresh_full_context_certificate": (
                "fresh replay after detector optimization is complete"
            ),
        }
        if phase not in phase_contracts:
            raise ValueError(f"unsupported detector gate audit phase: {phase!r}")
        selected_positive_cache = (
            positive_hidden_cache if positive_cache is None else positive_cache
        )
        selected_negative_cache = (
            negative_hidden_cache if negative_cache is None else negative_cache
        )
        selected_writer_off_cache = (
            writer_off_hidden_cache if writer_off_cache is None else writer_off_cache
        )
        positive_detector = record_cached_detector_responses(selected_positive_cache)
        negative_detector = record_cached_detector_responses(selected_negative_cache)
        writer_off_detector = (
            record_cached_detector_responses(selected_writer_off_cache)
            if writer_present
            else [torch.empty(0) for _ in positive_prompts_by_record]
        )
        gate = neuron_core.detector_gate_report(
            positive_detector,
            negative_detector,
            writer_off_detector,
            positive_floor=float(args.detector_positive_floor),
            off_abs_max=float(args.detector_off_abs_max),
            require_writer_off=writer_present,
            comparison_abs_tolerance=float(args.detector_certificate_abs_tolerance),
        )
        gate.update(
            {
                "schema_version": 2,
                "kind": (
                    "training_only_embedding_code_detector_gate"
                    if writer_present
                    else "training_only_base_context_sparse_mlp_detector_gate"
                ),
                "protocol": PROTOCOL,
                "writer_mode": str(args.writer_mode),
                "phase": str(phase),
                "parameter_state": phase_contracts[phase],
                "optimizer_step": int(optimizer_step),
                "optimization": detector_optimization_metadata(),
                "record_index_binding": "locked training-visible record order",
                "official_evaluation_prompts_seen": 0,
            }
        )
        if len(gate["per_record"]) != len(case_ids):
            raise RuntimeError("detector gate lost the locked record-index binding")
        for row, case_id in zip(gate["per_record"], case_ids):
            row["case_id"] = int(case_id)
        return gate

    print(
        "\nStage 1: "
        + (
            "repair imported detector with canonical multi-label isolation"
            if detector_replay_mode
            else "certify imported frozen sparse contextual-code detector"
            if str(args.detector_initialization) == "frozen_v3_2"
            else "train globally balanced sparse contextual-code detector"
        )
    )
    initial_frozen_detector_replay: Dict[str, Any] | None = None
    initial_global_isolation_gate: Dict[str, Any] | None = None
    initial_global_isolation_gate_path: Path | None = None
    if frozen_detector_source_gate is not None:
        initial_detector_gate = full_cached_detector_gate(
            phase="final_fresh_full_context_certificate",
            optimizer_step=0,
            positive_cache=legacy_positive_hidden_cache,
            negative_cache=legacy_negative_hidden_cache,
            writer_off_cache=legacy_writer_off_hidden_cache,
        )
        initial_frozen_detector_replay = compare_detector_gate_replays(
            frozen_detector_source_gate,
            initial_detector_gate,
            abs_tolerance=FROZEN_DETECTOR_REPLAY_ABS_TOLERANCE,
        )
        if not initial_frozen_detector_replay["passed"]:
            raise RuntimeError(
                "V3.5.4 legacy-cache initial detector state differs from frozen "
                "V3.2 source"
            )
        initial_detector_gate[
            "frozen_v3_2_source_replay"
        ] = initial_frozen_detector_replay
        initial_detector_gate_path = out_dir / "detector_initial_import_gate.json"
        gagd.write_json(initial_detector_gate_path, initial_detector_gate)
        initial_global_isolation_gate = multilabel_threshold_gate_report(
            phase="v3.5.4_pre_repair_multilabel_isolation",
            optimizer_step=0,
        )
        initial_global_isolation_gate_path = (
            out_dir / "isolated_threshold_gate_initial_report.json"
        )
        initial_collision = initial_global_isolation_gate.get(
            "writer_off_gate_argmax_cell"
        )
        expected_collision = (
            frozen_v3_5_1_forensics.get("collision", {})
            if frozen_v3_5_1_forensics is not None
            else {}
        )
        initial_collision_replayed = bool(
            isinstance(initial_collision, Mapping)
            and int(initial_collision.get("source_case_id", -1))
            == int(expected_collision.get("source_case_id", -2))
            and int(initial_collision.get("source_context_index", -1))
            == int(expected_collision.get("source_context_index", -2))
            and int(initial_collision.get("detector_case_id", -1))
            == int(expected_collision.get("detector_case_id", -2))
            and int(initial_collision.get("detector_group_index", -1))
            == int(expected_collision.get("detector_group_index", -2))
            and initial_collision.get("source_owner_group") is False
        )
        initial_global_isolation_gate["lineage"] = {
            "frozen_v3_5_1_collision_artifact_hash_bound": bool(
                frozen_v3_5_1_forensics is not None
                and frozen_v3_5_1_forensics.get("passed") is True
            ),
            "same_collision_is_argmax_after_canonicalization": (
                initial_collision_replayed
            ),
            "canonicalized_initial_state_is_acceptance_gate": False,
        }
        gagd.write_json(
            initial_global_isolation_gate_path,
            initial_global_isolation_gate,
        )

    detector_optimizer = torch.optim.AdamW(
        [editor.gate_delta, editor.up_delta],
        lr=float(args.detector_lr),
        weight_decay=0.0,
    )
    detector_log: List[Dict[str, Any]] = []
    detector_gradient_balance_audit: Dict[str, Any] | None = None
    endpoint_gate_reports: Dict[str, Dict[str, Any]] = {}
    endpoint_gate_paths: Dict[str, Path] = {}
    global_post_projection_gate: Dict[str, Any] | None = None
    global_post_projection_gate_path: Path | None = None
    for step in range(1, detector_optimizer_steps_this_run + 1):
        detector_optimizer.zero_grad(set_to_none=True)
        response_parts: List[torch.Tensor] = []
        source_owners: List[int] = []
        positive_occurrence_flags: List[bool] = []
        active_mask_parts: List[torch.Tensor] = []
        writer_off_response_parts: List[torch.Tensor] = []
        writer_off_owners: List[int] = []
        for record_start in range(0, len(records), int(args.detector_record_batch)):
            record_indices = list(
                range(
                    record_start,
                    min(
                        len(records),
                        record_start + int(args.detector_record_batch),
                    ),
                )
            )
            hidden_rows: List[torch.Tensor] = []
            active_rows: List[torch.Tensor] = []
            off_hidden_rows: List[torch.Tensor] = []
            for record_index in record_indices:
                positive_hidden = positive_hidden_cache[record_index]
                negative_hidden = negative_hidden_cache[record_index]
                hidden_rows.extend((positive_hidden, negative_hidden))
                active_rows.extend(
                    (
                        positive_active_mask_cache[record_index],
                        negative_active_mask_cache[record_index],
                    )
                )
                source_owners.extend(
                    [record_index]
                    * (int(positive_hidden.shape[0]) + int(negative_hidden.shape[0]))
                )
                positive_occurrence_flags.extend(
                    [True] * int(positive_hidden.shape[0])
                    + [False] * int(negative_hidden.shape[0])
                )
                if writer_present:
                    off_hidden = writer_off_hidden_cache[record_index]
                    off_hidden_rows.append(off_hidden)
                    writer_off_owners.extend([record_index] * int(off_hidden.shape[0]))
            response_parts.append(
                cached_detector_responses(torch.cat(hidden_rows, dim=0))
            )
            active_mask_parts.append(torch.cat(active_rows, dim=0))
            if writer_present:
                writer_off_response_parts.append(
                    cached_detector_responses(torch.cat(off_hidden_rows, dim=0))
                )

        responses = torch.cat(response_parts, dim=0)
        owner_tensor = torch.tensor(source_owners, dtype=torch.long, device=device)
        positive_tensor = torch.tensor(
            positive_occurrence_flags, dtype=torch.bool, device=device
        )
        active_tensor = torch.cat(active_mask_parts, dim=0).to(
            device=device, non_blocking=True
        )
        detector_loss, pieces = neuron_core.detector_multilabel_objective(
            responses,
            owner_tensor,
            positive_tensor,
            active_tensor,
            positive_target=float(args.detector_training_positive_floor),
            off_target_abs_max=float(args.detector_training_off_abs_max),
            tail_k=int(args.detector_tail_k),
            negative_weight=float(args.detector_negative_weight),
            cross_weight=float(args.detector_cross_weight),
            global_tail_weight=float(args.detector_global_tail_weight),
        )
        writer_off_loss = responses.sum() * 0.0
        writer_off_pieces = {
            "writer_off_mean": writer_off_loss,
            "writer_off_tail": writer_off_loss,
            "writer_off_global_tail": writer_off_loss,
        }
        if writer_present:
            writer_off_response = torch.cat(writer_off_response_parts, dim=0)
            writer_off_owner_tensor = torch.tensor(
                writer_off_owners, dtype=torch.long, device=device
            )
            (
                writer_off_loss,
                writer_off_pieces,
            ) = neuron_core.detector_global_writer_off_objective(
                writer_off_response,
                writer_off_owner_tensor,
                off_target_abs_max=float(args.detector_training_off_abs_max),
                tail_k=int(args.detector_tail_k),
                global_tail_weight=float(args.detector_global_tail_weight),
            )
        l2 = editor.gate_delta.square().mean() + editor.up_delta.square().mean()
        total_loss = (
            detector_loss
            + float(args.detector_consistency_weight) * pieces["consistency"]
            + float(args.detector_writer_off_weight) * writer_off_loss
            + float(args.detector_l2) * l2
        )
        if step == 1 and detector_replay_mode:
            parameters = (editor.gate_delta, editor.up_delta)
            optimizer_state_entries_before = len(detector_optimizer.state)
            if optimizer_state_entries_before != 0:
                raise RuntimeError(
                    "first-update gradient audit expected a fresh optimizer state"
                )
            if any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError(
                    "gradient-balance audit requires empty parameter grad buffers"
                )

            def gradient_norm_and_alignment(
                objective: torch.Tensor,
                total_gradients: Sequence[torch.Tensor | None],
                total_norm: float,
            ) -> Tuple[float, float | None]:
                gradients = torch.autograd.grad(
                    objective,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                norm_square = 0.0
                dot = 0.0
                for gradient, total_gradient in zip(gradients, total_gradients):
                    if gradient is None:
                        continue
                    gradient_float = gradient.detach().float()
                    norm_square += float(gradient_float.square().sum())
                    if total_gradient is not None:
                        dot += float(
                            (gradient_float * total_gradient.detach().float()).sum()
                        )
                norm = math.sqrt(max(0.0, norm_square))
                cosine = (
                    dot / (norm * total_norm)
                    if norm > 0.0 and total_norm > 0.0
                    else None
                )
                return norm, cosine

            total_gradients = torch.autograd.grad(
                total_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            total_norm = math.sqrt(
                sum(
                    float(gradient.detach().float().square().sum())
                    for gradient in total_gradients
                    if gradient is not None
                )
            )
            component_specs = {
                "positive_write": (pieces["write"], 1.0),
                "source_negative": (
                    pieces["negative"],
                    float(args.detector_negative_weight),
                ),
                "inactive_cross": (
                    pieces["cross"],
                    float(args.detector_cross_weight),
                ),
                "writer_off_all_groups": (
                    writer_off_loss,
                    float(args.detector_writer_off_weight),
                ),
                "positive_consistency": (
                    pieces["consistency"],
                    float(args.detector_consistency_weight),
                ),
                "parameter_l2": (l2, float(args.detector_l2)),
            }
            component_rows: Dict[str, Any] = {}
            weighted_off_norm_sum = 0.0
            for name, (objective, coefficient) in component_specs.items():
                raw_norm, cosine = gradient_norm_and_alignment(
                    objective,
                    total_gradients,
                    total_norm,
                )
                weighted_norm = abs(float(coefficient)) * raw_norm
                component_rows[name] = {
                    "unweighted_objective_value": float(objective.detach()),
                    "optimization_coefficient": float(coefficient),
                    "weighted_objective_value": float(
                        (float(coefficient) * objective).detach()
                    ),
                    "unweighted_gradient_l2_norm": raw_norm,
                    "weighted_gradient_l2_norm": weighted_norm,
                    "unweighted_gradient_cosine_with_total": cosine,
                }
                if name in {
                    "source_negative",
                    "inactive_cross",
                    "writer_off_all_groups",
                }:
                    weighted_off_norm_sum += weighted_norm
            component_rows["total"] = {
                "unweighted_objective_value": float(total_loss.detach()),
                "optimization_coefficient": 1.0,
                "weighted_objective_value": float(total_loss.detach()),
                "unweighted_gradient_l2_norm": total_norm,
                "weighted_gradient_l2_norm": total_norm,
                "unweighted_gradient_cosine_with_total": (
                    1.0 if total_norm > 0.0 else None
                ),
            }
            positive_norm = float(
                component_rows["positive_write"]["weighted_gradient_l2_norm"]
            )
            optimizer_state_entries_after = len(detector_optimizer.state)
            detector_gradient_balance_audit = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_detector_gradient_balance_audit",
                "protocol": PROTOCOL,
                "optimizer_step": 1,
                "measurement_phase": "before_total_backward_gradient_clip_and_adam",
                "global_tail_optimization_weight": float(
                    args.detector_global_tail_weight
                ),
                "global_tail_terms": "diagnostic_only_not_optimized",
                "total_gradient_l2_norm": total_norm,
                "components": component_rows,
                "weighted_off_component_norm_sum_upper_bound": (
                    weighted_off_norm_sum
                ),
                "positive_to_weighted_off_norm_sum_ratio": (
                    positive_norm / weighted_off_norm_sum
                    if weighted_off_norm_sum > 0.0
                    else None
                ),
                "parameter_grad_buffers_mutated": any(
                    parameter.grad is not None for parameter in parameters
                ),
                "optimizer_state_entries_before": optimizer_state_entries_before,
                "optimizer_state_entries_after": optimizer_state_entries_after,
                "optimizer_state_mutated": (
                    optimizer_state_entries_before != optimizer_state_entries_after
                ),
                "used_for_optimization": False,
                "official_evaluation_prompts_seen": 0,
            }
            if detector_gradient_balance_audit["parameter_grad_buffers_mutated"]:
                raise RuntimeError("gradient-balance audit mutated parameter gradients")
            if detector_gradient_balance_audit["optimizer_state_mutated"]:
                raise RuntimeError("gradient-balance audit mutated optimizer state")
        total_loss.backward()
        accumulated = {name: float(value.detach()) for name, value in pieces.items()}
        accumulated["writer_off"] = float(writer_off_loss.detach())
        accumulated.update(
            {name: float(value.detach()) for name, value in writer_off_pieces.items()}
        )
        total_value = float(total_loss.detach())
        if not math.isfinite(total_value):
            raise FloatingPointError(f"non-finite detector loss at step {step}")
        if step == detector_optimizer_steps_this_run:
            phase = "pre_update"
            endpoint_gate_reports[phase] = full_cached_detector_gate(
                phase=phase, optimizer_step=step
            )
            endpoint_gate_paths[phase] = (
                out_dir / f"detector_step_{step}_pre_update_gate.json"
            )
            gagd.write_json(endpoint_gate_paths[phase], endpoint_gate_reports[phase])
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [editor.gate_delta, editor.up_delta], float(args.grad_clip)
            )
        detector_optimizer.step()
        if step == detector_optimizer_steps_this_run:
            phase = "post_adam"
            endpoint_gate_reports[phase] = full_cached_detector_gate(
                phase=phase, optimizer_step=step
            )
            endpoint_gate_paths[phase] = (
                out_dir / f"detector_step_{step}_post_adam_gate.json"
            )
            gagd.write_json(endpoint_gate_paths[phase], endpoint_gate_reports[phase])
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == detector_optimizer_steps_this_run:
            phase = "post_projection"
            endpoint_gate_reports[phase] = full_cached_detector_gate(
                phase=phase, optimizer_step=step
            )
            endpoint_gate_paths[phase] = (
                out_dir / f"detector_step_{step}_post_projection_gate.json"
            )
            gagd.write_json(endpoint_gate_paths[phase], endpoint_gate_reports[phase])
            global_post_projection_gate = multilabel_threshold_gate_report(
                phase="post_projection",
                optimizer_step=step,
            )
            global_post_projection_gate_path = (
                out_dir
                / f"detector_step_{step}_post_projection_global_isolation_gate.json"
            )
            gagd.write_json(
                global_post_projection_gate_path,
                global_post_projection_gate,
            )
        row = {
            "step": step,
            "loss": total_value,
            **accumulated,
            "l2": float(l2.detach()),
            "records_per_optimizer_update": len(records),
            "microbatches_per_optimizer_update": detector_microbatches,
            "gate_max_relative_norm": cap["gate_max_relative_norm"],
            "up_max_relative_norm": cap["up_max_relative_norm"],
            "loss_measurement_phase": "pre_update",
            "norm_measurement_phase": "post_projection",
        }
        detector_log.append(row)
        if step == 1 or step % 25 == 0 or step == detector_optimizer_steps_this_run:
            print(
                f"  step {step:>4}: loss {row['loss']:.4f}, "
                f"write {row['write']:.4f}, off {row['writer_off']:.4f}, "
                f"cross {row['cross']:.4f}"
            )
    if len(detector_log) != detector_optimizer_steps_this_run:
        raise RuntimeError("complete detector training log lost optimizer steps")
    if (
        detector_replay_mode
        and detector_gradient_balance_audit is None
    ):
        raise RuntimeError("V3.5.4 requires the first-update gradient-balance audit")
    detector_gradient_balance_path = out_dir / "detector_gradient_balance_audit.json"
    if detector_gradient_balance_audit is not None:
        gagd.write_json(
            detector_gradient_balance_path,
            detector_gradient_balance_audit,
        )
    detector_training_log_report = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_complete_detector_training_log",
        "protocol": PROTOCOL,
        "revision": detector_optimization_metadata()["revision"],
        "optimizer_steps_expected": detector_optimizer_steps_this_run,
        "optimizer_steps_recorded": len(detector_log),
        "complete": len(detector_log) == detector_optimizer_steps_this_run,
        "optimization": detector_optimization_metadata(),
        "gradient_balance_audit": (
            {
                "path": str(detector_gradient_balance_path),
                "sha256": compositional_method.sha256_file(
                    detector_gradient_balance_path
                ),
                "optimizer_step": 1,
                "used_for_optimization": False,
            }
            if detector_gradient_balance_audit is not None
            else None
        ),
        "records": detector_log,
        "official_evaluation_prompts_seen": 0,
    }
    detector_training_log_path = out_dir / "detector_training_log.json"
    gagd.write_json(detector_training_log_path, detector_training_log_report)
    del detector_optimizer

    detector_gate = full_cached_detector_gate(
        phase="final_fresh_full_context_certificate",
        optimizer_step=detector_optimizer_steps_this_run,
    )
    frozen_detector_replay = initial_frozen_detector_replay
    source_to_repaired_detector_delta: Dict[str, Any] | None = None
    if frozen_detector_source_gate is not None:
        source_to_repaired_detector_delta = compare_detector_gate_replays(
            frozen_detector_source_gate,
            detector_gate,
            abs_tolerance=FROZEN_DETECTOR_REPLAY_ABS_TOLERANCE,
        )
        detector_gate["initial_frozen_v3_2_source_replay"] = frozen_detector_replay
        detector_gate[
            "source_to_repaired_detector_delta"
        ] = source_to_repaired_detector_delta
    detector_gate_path = out_dir / "detector_gate_report.json"
    gagd.write_json(detector_gate_path, detector_gate)
    endpoint_phases = ("pre_update", "post_adam", "post_projection")
    endpoint_complete = bool(
        detector_optimizer_steps_this_run == 0
        or all(phase in endpoint_gate_reports for phase in endpoint_phases)
    )
    endpoint_replay = (
        compare_detector_gate_replays(
            endpoint_gate_reports["post_projection"],
            detector_gate,
            abs_tolerance=float(args.detector_certificate_abs_tolerance),
        )
        if detector_optimizer_steps_this_run > 0
        else {
            "record_count_match": True,
            "record_binding_match": True,
            "decisions_match": True,
            "metric_abs_max": 0.0,
            "passed": True,
        }
    )
    post_projection_matches_final = bool(endpoint_replay["passed"])
    endpoint_audit = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_detector_endpoint_audit",
        "protocol": PROTOCOL,
        "optimizer_step": detector_optimizer_steps_this_run,
        "phases_required": list(endpoint_phases),
        "phase_audits_not_applicable_for_frozen_import": bool(
            detector_optimizer_steps_this_run == 0
        ),
        "artifacts": {
            phase: {
                "path": str(endpoint_gate_paths[phase]),
                "sha256": compositional_method.sha256_file(endpoint_gate_paths[phase]),
                "passed_records": int(endpoint_gate_reports[phase]["passed_records"]),
                "total_records": int(endpoint_gate_reports[phase]["total_records"]),
                "passed": bool(endpoint_gate_reports[phase]["passed"]),
            }
            for phase in endpoint_phases
            if phase in endpoint_gate_reports
        },
        "final_fresh_full_context_certificate": {
            "path": str(detector_gate_path),
            "sha256": compositional_method.sha256_file(detector_gate_path),
            "passed_records": int(detector_gate["passed_records"]),
            "total_records": int(detector_gate["total_records"]),
            "passed": bool(detector_gate["passed"]),
        },
        "complete_training_log": {
            "path": str(detector_training_log_path),
            "sha256": compositional_method.sha256_file(detector_training_log_path),
            "optimizer_steps_recorded": len(detector_log),
            "complete": bool(detector_training_log_report["complete"]),
        },
        "gradient_balance_audit": (
            {
                "path": str(detector_gradient_balance_path),
                "sha256": compositional_method.sha256_file(
                    detector_gradient_balance_path
                ),
                "optimizer_step": 1,
                "parameter_grad_buffers_mutated": False,
                "optimizer_state_mutated": False,
                "used_for_optimization": False,
                "complete": True,
            }
            if detector_gradient_balance_audit is not None
            else {
                "required": bool(detector_replay_mode),
                "complete": not bool(detector_replay_mode),
            }
        ),
        "post_projection_matches_final_fresh_certificate": (
            post_projection_matches_final
        ),
        "post_projection_final_replay": endpoint_replay,
        "initial_frozen_v3_2_source_replay": frozen_detector_replay,
        "source_to_repaired_detector_delta": source_to_repaired_detector_delta,
        "frozen_v3_2_import": frozen_detector_import,
        "complete": bool(
            endpoint_complete
            and post_projection_matches_final
            and (
                detector_gradient_balance_audit is not None
                or not detector_replay_mode
            )
        ),
        "official_evaluation_prompts_seen": 0,
    }
    if not endpoint_audit["complete"]:
        raise RuntimeError("detector endpoint audit is incomplete or inconsistent")
    gagd.write_json(out_dir / "detector_endpoint_audit.json", endpoint_audit)
    diagnostic_lines = [
        "record_index\tcase_id\tpositive_min\tnegative_abs_max\t"
        "writer_off_abs_max\tpassed"
    ]
    diagnostic_lines.extend(
        "\t".join(
            (
                str(row["record_index"]),
                str(row["case_id"]),
                f"{float(row['positive_min']):+.8f}",
                f"{float(row['negative_abs_max']):.8f}",
                f"{float(row['writer_off_abs_max']):.8f}",
                str(bool(row["passed"])).lower(),
            )
        )
        for row in detector_gate["per_record"]
    )
    (out_dir / "detector_gate_case_report.tsv").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )
    print(
        f"  detector gate: {detector_gate['passed_records']}/"
        f"{detector_gate['total_records']} records"
    )
    if writer_present and args.gate_policy == "strict" and not detector_gate["passed"]:
        rejection = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
            "stage": "sparse_context_detector",
            "reason": "detector_gate_failed",
            "writer_mode": str(args.writer_mode),
            "detector_gate": detector_gate,
            "detector_training_log_path": str(detector_training_log_path),
            "detector_endpoint_audit_path": str(
                out_dir / "detector_endpoint_audit.json"
            ),
            "actuator_training_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
            "next_action": (
                "Redesign or retrain the frozen writer/detector under a newly "
                "registered training-only configuration; do not tune on official probes."
            ),
        }
        gagd.write_json(out_dir / "training_rejection.json", rejection)
        editor.remove()
        embedding_writer.remove()
        raise SystemExit(
            "embedding-keyed detector failed its locked training-only gate; "
            "actuator training and official evaluation are refused"
        )

    detector_selectivity: Dict[str, Any] | None = None
    threshold_gate: Dict[str, Any] | None = None
    if detector_replay_mode:
        detector_selectivity = detector_selectivity_audit()
        gagd.write_json(
            out_dir / "repaired_detector_selectivity_audit.json",
            detector_selectivity,
        )
        response_ratio = detector_selectivity["aggregate"][
            "owned_signed_group_response"
        ]["writer_on_to_off_ratio"]
        print(
            "  repaired detector owned-response on/off ratio: "
            f"p10={float(response_ratio['p10']):.4f}, "
            f"median={float(response_ratio['median']):.4f}"
        )
        threshold_gate = multilabel_threshold_gate_report(
            phase="v3.5.5_exact_v3.5.4_detector_replay_certificate",
            optimizer_step=detector_optimizer_steps_this_run,
        )
        gagd.write_json(out_dir / "isolated_threshold_gate_report.json", threshold_gate)
        if (
            initial_global_isolation_gate is None
            or initial_global_isolation_gate_path is None
            or global_post_projection_gate is None
            or global_post_projection_gate_path is None
        ):
            raise RuntimeError("V3.5.4 multi-label endpoint audit is incomplete")

        def global_gate_state_digest(report: Mapping[str, Any]) -> str:
            return compositional_method.sha256_json(
                {
                    "aggregate": report.get("aggregate"),
                    "response_certificate_violation_counts": report.get(
                        "response_certificate_violation_counts"
                    ),
                    "gate_endpoint_violation_counts": report.get(
                        "gate_endpoint_violation_counts"
                    ),
                    "checks": report.get("checks"),
                    "passed": report.get("passed"),
                    "raw_signed_response_matrices_by_source_record": report.get(
                        "raw_signed_response_matrices_by_source_record"
                    ),
                }
            )

        post_projection_global_digest = global_gate_state_digest(
            global_post_projection_gate
        )
        final_global_digest = global_gate_state_digest(threshold_gate)
        global_endpoint_replay_passed = bool(
            post_projection_global_digest == final_global_digest
        )
        endpoint_audit["global_isolation_endpoint_audit"] = {
            "initial_pre_repair": {
                "path": str(initial_global_isolation_gate_path),
                "sha256": compositional_method.sha256_file(
                    initial_global_isolation_gate_path
                ),
                "gate_endpoint_violation_counts": initial_global_isolation_gate[
                    "gate_endpoint_violation_counts"
                ],
                "passed": bool(initial_global_isolation_gate["passed"]),
            },
            "post_projection": {
                "path": str(global_post_projection_gate_path),
                "sha256": compositional_method.sha256_file(
                    global_post_projection_gate_path
                ),
                "state_digest": post_projection_global_digest,
                "passed": bool(global_post_projection_gate["passed"]),
            },
            "final_fresh": {
                "path": str(out_dir / "isolated_threshold_gate_report.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "isolated_threshold_gate_report.json"
                ),
                "state_digest": final_global_digest,
                "passed": bool(threshold_gate["passed"]),
            },
            "post_projection_matches_final_fresh": global_endpoint_replay_passed,
            "complete": global_endpoint_replay_passed,
        }
        endpoint_audit["complete"] = bool(
            endpoint_audit["complete"] and global_endpoint_replay_passed
        )
        gagd.write_json(out_dir / "detector_endpoint_audit.json", endpoint_audit)
        if not endpoint_audit["complete"]:
            raise RuntimeError(
                "V3.5.4 post-projection multi-label gate differs from final fresh "
                "replay"
            )
        print(
            "  multi-label isolated threshold gate: "
            f"positive_min={float(threshold_gate['aggregate']['positive_owner_gate']['min']):.6f}, "
            f"writer_off_max={float(threshold_gate['aggregate']['writer_off_gate']['max']):.6f}, "
            f"passed={bool(threshold_gate['passed'])}"
        )
        if not threshold_gate["passed"]:
            gagd.write_json(
                out_dir / "training_rejection.json",
                {
                    "schema_version": 1,
                    "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                    "protocol": PROTOCOL,
                    "stage": "isolated_threshold_gate",
                    "reason": "repaired_detector_failed_multilabel_all_cell_gate",
                    "threshold_gate": threshold_gate,
                    "actuator_training_started": False,
                    "checkpoint_saved": False,
                    "official_evaluation_allowed": False,
                    "official_evaluation_prompts_seen": 0,
                },
            )
            editor.remove()
            embedding_writer.remove()
            raise SystemExit(
                "repaired detector failed the multi-label all-cell gate before "
                "actuator training; "
                "official evaluation remains refused"
            )
        if frozen_v3_5_4_rejection is None:
            raise RuntimeError("detector replay lacks the frozen V3.5.4 tensor receipt")
        observed_gate_sha256 = _tensor_digest(editor.gate_delta)
        observed_up_sha256 = _tensor_digest(editor.up_delta)
        detector_replay_receipt = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_5_4_exact_detector_replay",
            "protocol": PROTOCOL,
            "source_protocol": FROZEN_V3_5_4_PROTOCOL,
            "optimizer_updates_replayed": detector_optimizer_steps_this_run,
            "expected_gate_delta_sha256": frozen_v3_5_4_rejection[
                "detector_gate_delta_sha256"
            ],
            "observed_gate_delta_sha256": observed_gate_sha256,
            "expected_up_delta_sha256": frozen_v3_5_4_rejection[
                "detector_up_delta_sha256"
            ],
            "observed_up_delta_sha256": observed_up_sha256,
            "gate_delta_bit_exact": bool(
                observed_gate_sha256
                == frozen_v3_5_4_rejection["detector_gate_delta_sha256"]
            ),
            "up_delta_bit_exact": bool(
                observed_up_sha256
                == frozen_v3_5_4_rejection["detector_up_delta_sha256"]
            ),
            "all_cell_gate_passed": bool(threshold_gate["passed"]),
            "official_evaluation_prompts_seen": 0,
        }
        detector_replay_receipt["passed"] = bool(
            detector_replay_receipt["gate_delta_bit_exact"]
            and detector_replay_receipt["up_delta_bit_exact"]
            and detector_replay_receipt["all_cell_gate_passed"]
        )
        if full_preservation_mode:
            if frozen_v3_5_5_success is None:
                raise RuntimeError("V3.6.1 lacks frozen V3.5.5 detector hashes")
            detector_replay_receipt["frozen_v3_5_5_gate_delta_sha256"] = (
                frozen_v3_5_5_success["detector_gate_delta_sha256"]
            )
            detector_replay_receipt["frozen_v3_5_5_up_delta_sha256"] = (
                frozen_v3_5_5_success["detector_up_delta_sha256"]
            )
            detector_replay_receipt["matches_frozen_v3_5_5"] = bool(
                observed_gate_sha256
                == frozen_v3_5_5_success["detector_gate_delta_sha256"]
                and observed_up_sha256
                == frozen_v3_5_5_success["detector_up_delta_sha256"]
            )
            detector_replay_receipt["passed"] = bool(
                detector_replay_receipt["passed"]
                and detector_replay_receipt["matches_frozen_v3_5_5"]
            )
        detector_replay_path = out_dir / "exact_v3_5_4_detector_replay.json"
        gagd.write_json(detector_replay_path, detector_replay_receipt)
        if not detector_replay_receipt["passed"]:
            gagd.write_json(
                out_dir / "training_rejection.json",
                {
                    "schema_version": 1,
                    "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                    "protocol": PROTOCOL,
                    "stage": "exact_v3_5_4_detector_replay",
                    "reason": "replayed_detector_tensors_differ_from_frozen_v3_5_4",
                    "detector_replay_receipt": detector_replay_receipt,
                    "actuator_training_started": False,
                    "checkpoint_saved": False,
                    "official_evaluation_allowed": False,
                    "official_evaluation_prompts_seen": 0,
                },
            )
            editor.remove()
            embedding_writer.remove()
            raise SystemExit(
                "V3.5.4 detector replay was not bit-exact; actuator construction "
                "and official evaluation remain refused"
            )
        print("  exact V3.5.4 detector tensor hashes reproduced bit-for-bit")

    print(
        "\nStage 2: "
        + (
            "training-only actuator reachability diagnostics; LM head frozen"
            if width_sweep_mode
            else "training-only width-16 full preservation; LM head frozen"
        )
    )
    # Every Stage-2 preservation target is the writer-only model.  The frozen
    # detector is a read-only branch input in V3.5.5/V3.6.1 and therefore may not be
    # smuggled into the reference baseline.
    editor.enabled = False
    editor.gate_delta.requires_grad_(False)
    editor.up_delta.requires_grad_(False)
    editor.down_delta.requires_grad_(True)
    embedding_writer.enabled = writer_present
    pre_target_new, pre_target_true = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    if full_preservation_mode:
        (
            pre_target_float32_new,
            pre_target_float32_true,
        ) = evaluate_instance_nlls_float32(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        (
            pre_negative_new,
            pre_negative_true,
        ) = compositional_method.evaluate_instance_nlls(
            model,
            tok,
            preservation_negative_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        (
            pre_negative_float32_new,
            pre_negative_float32_true,
        ) = evaluate_instance_nlls_float32(
            model,
            tok,
            preservation_negative_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        registered_protected_bank = int(
            registered_full_actuator.get("protected_prompt_bank", -1)
        )
        if registered_protected_bank <= 0:
            raise RuntimeError("registry lacks a positive protected prompt-bank size")
        protected_for_kl = _unique_prompts(
            [
                *[
                    instance.prompt
                    for instance in preservation_negative_instances
                ],
                *corpus_prompts,
            ]
        )[:registered_protected_bank]
        registered_protected_per_update = int(
            registered_full_actuator.get("protected_contexts_per_optimizer_update", -1)
        )
        if len(protected_for_kl) != registered_protected_bank:
            raise RuntimeError(
                "protected prompt bank differs from the registered size: "
                f"{len(protected_for_kl)} != {registered_protected_bank}"
            )
        if len(protected_for_kl) < registered_protected_per_update:
            raise RuntimeError(
                "protected prompt bank cannot supply the registered actuator "
                f"batch: {len(protected_for_kl)} < {registered_protected_per_update}"
            )
        writer_only_cache = compositional_method.cache_prompt_baselines(
            model,
            tok,
            protected_for_kl,
            device,
            batch_size=int(args.cache_batch_size),
            topk=int(args.kl_topk),
        )
    editor.enabled = True
    editor.write_enabled = True

    # Writer-off preservation is measured against a frozen Base-without-writer
    # baseline once.  V3.2 recomputed this baseline every stochastic step and
    # then drove all drift to exactly zero; V3.3 uses the registered 0.05 band.
    embedding_writer.enabled = False
    editor.write_enabled = False
    (
        pre_writer_off_new,
        pre_writer_off_true,
    ) = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    if full_preservation_mode:
        (
            pre_writer_off_float32_new,
            pre_writer_off_float32_true,
        ) = evaluate_instance_nlls_float32(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        nll_numerics_receipt = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_6_1_nll_numerics_receipt",
            "protocol": PROTOCOL,
            "training_nll_definition": (
                "float32 log_softmax current minus float32 log_softmax baseline"
            ),
            "acceptance_nll_definition": (
                "official-compatible native-dtype log_softmax current minus "
                "native-dtype baseline"
            ),
            "mixed_dtype_subtraction_prohibited": True,
            "negative_preservation_precedence": {
                "rule": negative_precedence_report["rule"],
                "matching_scope": negative_precedence_report["matching_scope"],
                "raw_negative_occurrences": negative_precedence_report[
                    "raw_negative_occurrences"
                ],
                "coherent_preservation_negative_occurrences": (
                    negative_precedence_report[
                        "coherent_preservation_negative_occurrences"
                    ]
                ),
                "excluded_multi_role_negative_occurrences": (
                    negative_precedence_report[
                        "excluded_multi_role_negative_occurrences"
                    ]
                ),
                "report_sha256": compositional_method.sha256_file(
                    negative_precedence_path
                ),
            },
            "negative_training_nll_tolerance": float(
                args.negative_training_nll_tolerance
            ),
            "negative_acceptance_nll_tolerance": float(
                args.negative_nll_tolerance
            ),
            "baseline_native_vs_float32_abs": {
                "positive": _distribution(
                    torch.cat(
                        (
                            (pre_target_new - pre_target_float32_new).abs(),
                            (pre_target_true - pre_target_float32_true).abs(),
                        )
                    ).tolist()
                ),
                "negative": _distribution(
                    torch.cat(
                        (
                            (pre_negative_new - pre_negative_float32_new).abs(),
                            (pre_negative_true - pre_negative_float32_true).abs(),
                        )
                    ).tolist()
                ),
                "writer_off": _distribution(
                    torch.cat(
                        (
                            (
                                pre_writer_off_new
                                - pre_writer_off_float32_new
                            ).abs(),
                            (
                                pre_writer_off_true
                                - pre_writer_off_float32_true
                            ).abs(),
                        )
                    ).tolist()
                ),
            },
            "official_evaluation_prompts_seen": 0,
        }
        gagd.write_json(
            out_dir / "v3_6_1_nll_numerics_receipt.json",
            nll_numerics_receipt,
        )
    selected_ids_device = torch.tensor(
        selected_neurons,
        dtype=torch.long,
        device=mlp.down_proj.weight.device,
    )
    base_selected_down = (
        mlp.down_proj.weight.detach().index_select(1, selected_ids_device).clone()
    )
    base_selected_down_sha256 = _tensor_digest(base_selected_down)
    editor.enabled = False
    try:
        with torch.no_grad():
            mlp.down_proj.weight.index_fill_(1, selected_ids_device, 0.0)
        zeroed_down_new, zeroed_down_true = compositional_method.evaluate_instance_nlls(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
    finally:
        with torch.no_grad():
            mlp.down_proj.weight.index_copy_(
                1,
                selected_ids_device,
                base_selected_down.to(mlp.down_proj.weight.dtype),
            )
        editor.enabled = True
    restored_selected_down = mlp.down_proj.weight.detach().index_select(
        1, selected_ids_device
    )
    if _tensor_digest(restored_selected_down) != base_selected_down_sha256:
        raise RuntimeError("selected Base down columns were not restored bit-exactly")
    zeroing_new_drift = (zeroed_down_new - pre_writer_off_new).abs()
    zeroing_true_drift = (zeroed_down_true - pre_writer_off_true).abs()
    zeroing_per_record: List[Dict[str, Any]] = []
    for record_index, indices in enumerate(positive_indices_by_record):
        record_drift = torch.cat(
            (zeroing_new_drift[indices], zeroing_true_drift[indices])
        )
        zeroing_per_record.append(
            {
                "record_index": record_index,
                "case_id": int(case_ids[record_index]),
                "positive_contexts": len(indices),
                "nll_abs_max": float(record_drift.max()),
                "nll_abs_median": float(record_drift.median()),
            }
        )
    base_down_zeroing_audit = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_selected_base_down_zeroing_audit",
        "protocol": PROTOCOL,
        "parameter_state": (
            "writer off; Base gate/up unchanged; selected original down columns "
            "temporarily zero; isolated residual disabled"
        ),
        "selected_columns": len(selected_neurons),
        "base_selected_down_sha256_before": base_selected_down_sha256,
        "base_selected_down_sha256_after_restore": _tensor_digest(
            restored_selected_down
        ),
        "restored_bit_exact": True,
        "nll_abs_max": max(
            float(zeroing_new_drift.max()), float(zeroing_true_drift.max())
        ),
        "new_target_nll_abs": _distribution(zeroing_new_drift.tolist()),
        "true_target_nll_abs": _distribution(zeroing_true_drift.tolist()),
        "per_record": zeroing_per_record,
        "used_for_hyperparameter_selection": False,
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(
        out_dir / "selected_base_down_zeroing_audit.json",
        base_down_zeroing_audit,
    )
    print(
        "  selected-Base-down zeroing audit: "
        f"max |ΔNLL|={float(base_down_zeroing_audit['nll_abs_max']):.4f}"
    )
    editor.write_enabled = True
    embedding_writer.enabled = writer_present

    def differentiable_group_nlls(
        instances: Sequence[mcf_repair.MCFPromptInstance],
        indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_rows: List[torch.Tensor] = []
        true_rows: List[torch.Tensor] = []
        for start in range(0, len(indices), int(args.actuator_batch_size)):
            chunk = list(indices[start : start + int(args.actuator_batch_size)])
            (
                current_new,
                current_true,
            ) = compositional_method.differentiable_instance_nlls(
                model,
                tok,
                [instances[index] for index in chunk],
                device,
                llama_like=llama_like,
            )
            new_rows.append(current_new)
            true_rows.append(current_true)
        return torch.cat(new_rows), torch.cat(true_rows)

    def actuator_optimization_metadata() -> Dict[str, Any]:
        width_sweep = bool(width_sweep_mode)
        return {
            "revision": (
                "v3.5.5_exact_detector_replay_separate_actuator_width_sweep"
                if width_sweep
                else "v3.6.1_width16_coherent_float32_negative_preservation"
            ),
            "initial_down_delta": "exact_zero",
            "feasibility_optimizer_steps": int(args.actuator_feasibility_steps),
            "full_optimizer_steps": int(args.actuator_steps),
            "update_coverage": (
                "all_records_all_positive_contexts_accumulated"
                if width_sweep
                else "all_records_all_contexts_accumulated"
            ),
            "records_per_optimizer_update": len(records),
            "positive_contexts_per_optimizer_update": len(positive_instances),
            "negative_contexts_per_optimizer_update": (
                0 if width_sweep else len(preservation_negative_instances)
            ),
            "writer_off_contexts_per_optimizer_update": (
                0
                if width_sweep
                else (len(positive_instances) if writer_present else 0)
            ),
            "protected_contexts_per_optimizer_update": (
                0
                if width_sweep
                else min(int(args.actuator_protected_batch), len(protected_for_kl))
            ),
            "context_microbatch_capacity": int(args.actuator_batch_size),
            "tail_k": int(args.actuator_tail_k),
            "gradient_normalization": (
                "equal_record_mean_plus_per_record_worst_k"
                if width_sweep
                else "equal_record_mean_plus_global_protected_prompt_mean"
            ),
            "forget_margin": float(args.forget_margin),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "negative_preservation_weight": float(
                args.negative_preservation_weight
            ),
            "negative_training_nll_tolerance": float(
                args.negative_training_nll_tolerance
            ),
            "negative_acceptance_nll_tolerance": float(
                args.negative_nll_tolerance
            ),
            "writer_off_nll_tolerance": float(args.actuator_writer_off_nll_tolerance),
            "relative_norm_cap": float(args.actuator_relative_cap),
            "protected_sampling_seed": actuator_rng_seed,
            "gradient_clip_frequency": "once_per_optimizer_update",
            "norm_projection_frequency": "once_per_optimizer_update",
        }

    @torch.no_grad()
    def full_actuator_audit(*, phase: str, optimizer_step: int) -> Dict[str, Any]:
        phase_contracts = {
            "norm_cap_sweep_initial": (
                "fresh audit at exact-zero down_delta before any cap-specific fit"
            ),
            "positive_only_feasibility_final": (
                "after the last feasibility Adam step and norm projection"
            ),
            "norm_cap_sweep_final": (
                "fresh audit after the cap-specific final Adam step and projection"
            ),
            "training_check": "after the current Adam step and norm projection",
            "pre_update": (
                "after final-step gradient accumulation and before clipping or Adam"
            ),
            "post_adam": (
                "after final-step clipping and Adam and before norm projection"
            ),
            "post_projection": "after final-step Adam and norm projection",
            "final_fresh_full_context_audit": (
                "fresh replay after full actuator optimization is complete"
            ),
        }
        editor.enabled = True
        editor.write_enabled = True
        embedding_writer.enabled = writer_present
        current_new, current_true = compositional_method.evaluate_instance_nlls(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        margins = current_true - current_new
        direct_failure, positive_failure = _failure_counts(
            margins, direct_flags, float(args.forget_margin)
        )
        writer_on_abs_max = max(
            float((current_new - pre_target_new).abs().max()),
            float((current_true - pre_target_true).abs().max()),
        )
        reference_regression_max = max(0.0, float((current_new - pre_target_new).max()))
        embedding_writer.enabled = False
        off_new, off_true = compositional_method.evaluate_instance_nlls(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )
        writer_off_abs_max = max(
            float((off_new - pre_writer_off_new).abs().max()),
            float((off_true - pre_writer_off_true).abs().max()),
        )
        embedding_writer.enabled = writer_present
        norms = editor.relative_norm_report()
        per_record: List[Dict[str, Any]] = []
        for record_index, indices in enumerate(positive_indices_by_record):
            record_margins = margins[indices]
            direct_indices = [index for index in indices if direct_flags[index]]
            if len(direct_indices) != 1:
                raise RuntimeError("every record must have exactly one direct context")
            reference_regression = current_new[indices] - pre_target_new[indices]
            writer_off_drift = torch.cat(
                (
                    (off_new[indices] - pre_writer_off_new[indices]).abs(),
                    (off_true[indices] - pre_writer_off_true[indices]).abs(),
                )
            )
            per_record.append(
                {
                    "record_index": record_index,
                    "case_id": int(case_ids[record_index]),
                    "positive_contexts": len(indices),
                    "direct_margin": float(margins[direct_indices[0]]),
                    "positive_min": float(record_margins.min()),
                    "positive_median": float(record_margins.median()),
                    "positive_failures": int(
                        (record_margins < float(args.forget_margin) - 1e-6).sum()
                    ),
                    "reference_nll_regression_max": max(
                        0.0, float(reference_regression.max())
                    ),
                    "writer_off_nll_abs_max": float(writer_off_drift.max()),
                }
            )
        reference_passed = bool(
            reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
        )
        writer_off_passed = bool(
            not writer_present
            or writer_off_abs_max
            <= float(args.actuator_writer_off_nll_tolerance) + 1e-6
        )
        return {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_training_only_actuator_audit",
            "protocol": PROTOCOL,
            "phase": str(phase),
            "parameter_state": phase_contracts.get(
                str(phase), "training-only full-context audit"
            ),
            "optimizer_step": int(optimizer_step),
            "criterion": {
                "forget_margin": float(args.forget_margin),
                "reference_nll_regression_max": float(args.reference_nll_tolerance),
                "writer_off_nll_abs_max": float(args.actuator_writer_off_nll_tolerance),
                "zero_actuator_identity_abs_max": 1e-6,
            },
            "direct_failures": int(direct_failure),
            "positive_failures": int(positive_failure),
            "positive_contexts": len(positive_instances),
            "minimum_margin": float(margins.min()),
            "median_margin": float(margins.median()),
            "reference_nll_regression_max": reference_regression_max,
            "writer_on_nll_abs_max": writer_on_abs_max,
            "writer_off_nll_abs_max": writer_off_abs_max,
            "positive_passed": bool(direct_failure == 0 and positive_failure == 0),
            "reference_preservation_passed": reference_passed,
            "writer_off_preservation_applicable": writer_present,
            "writer_off_preservation_passed": writer_off_passed,
            "passed": bool(
                direct_failure == 0
                and positive_failure == 0
                and reference_passed
                and writer_off_passed
            ),
            "per_record": per_record,
            "norms": norms,
            "official_evaluation_prompts_seen": 0,
        }

    @torch.no_grad()
    def actuator_residual_write_selectivity_audit() -> Dict[str, Any]:
        """Audit the exact layer-output residual contributed by ``down_delta``."""

        on_activations = record_cached_selected_activations(positive_hidden_cache)
        off_activations = record_cached_selected_activations(writer_off_hidden_cache)
        on_norm_parts: List[torch.Tensor] = []
        off_norm_parts: List[torch.Tensor] = []
        per_record: List[Dict[str, Any]] = []
        for record_index, (on_activation, off_activation) in enumerate(
            zip(on_activations, off_activations)
        ):
            if on_activation.shape != off_activation.shape:
                raise RuntimeError("actuator residual audit lost on/off prompt pairing")
            on_device = on_activation.to(device=device)
            off_device = off_activation.to(device=device)
            if editor.residual_mode == "isolated_thresholded_residual":
                (
                    on_device,
                    _on_response,
                    _on_gate,
                ) = editor.isolated_actuator_features_from_activations(on_device)
                (
                    off_device,
                    _off_response,
                    _off_gate,
                ) = editor.isolated_actuator_features_from_activations(off_device)
            on_write = F.linear(on_device, editor.down_delta).float().cpu()
            off_write = F.linear(off_device, editor.down_delta).float().cpu()
            on_norm = on_write.norm(dim=1)
            off_norm = off_write.norm(dim=1)
            on_norm_parts.append(on_norm)
            off_norm_parts.append(off_norm)
            per_record.append(
                {
                    "record_index": record_index,
                    "case_id": int(case_ids[record_index]),
                    "positive_contexts": int(on_norm.numel()),
                    "residual_write_l2": paired_ratio_summary(on_norm, off_norm),
                }
            )
        aggregate = paired_ratio_summary(
            torch.cat(on_norm_parts), torch.cat(off_norm_parts)
        )
        return {
            "kind": "mcf_embedding_keyed_neuron_actuator_residual_write_selectivity",
            "definition": (
                "L2 norm of the exact actuator feature @ down_delta.T at the "
                "selected layer output; V3.5 actuator features include the "
                "record-level clipped threshold gate"
            ),
            "aggregate": aggregate,
            "per_record": per_record,
            "official_evaluation_prompts_seen": 0,
        }

    def down_norm_geometry(
        cap: float, *, include_per_neuron: bool = True
    ) -> Dict[str, Any]:
        ratios = editor.down_relative_norms().float().cpu()
        saturation_tolerance = 1e-6
        owner_indices = [
            record_index for record_index, group in enumerate(ownership) for _ in group
        ]
        if len(owner_indices) != len(selected_neurons):
            raise RuntimeError("down-column ownership geometry is inconsistent")
        saturated = ratios >= float(cap) - saturation_tolerance
        result: Dict[str, Any] = {
            "relative_norm": _distribution(ratios.tolist()),
            "cap": float(cap),
            "saturation_abs_tolerance": saturation_tolerance,
            "saturated_columns": int(saturated.sum()),
            "selected_columns": int(ratios.numel()),
            "saturated_fraction": float(saturated.float().mean()),
        }
        if include_per_neuron:
            result["per_neuron"] = [
                {
                    "flat_index": index,
                    "neuron_id": int(selected_neurons[index]),
                    "record_index": int(owner_indices[index]),
                    "case_id": int(case_ids[owner_indices[index]]),
                    "relative_norm": float(ratios[index]),
                    "saturated": bool(saturated[index]),
                }
                for index in range(len(selected_neurons))
            ]
        return result

    if (
        detector_replay_mode
        and str(args.actuator_architecture)
        == "separate_threshold_gated_actuator_bank"
    ):
        if (
            detector_selectivity is None
            or threshold_gate is None
            or frozen_v3_5_4_rejection is None
        ):
            raise RuntimeError(
                "separate actuator stage lacks its perfect gate or frozen V3.5.4 lineage"
            )
        detector_replay_path = out_dir / "exact_v3_5_4_detector_replay.json"
        detector_replay_receipt = _load_json(detector_replay_path)
        if detector_replay_receipt.get("passed") is not True:
            raise RuntimeError("separate actuator bank lacks an exact detector replay")

        widths = [int(width) for width in args.actuator_feasibility_widths]
        cap = float(args.actuator_feasibility_caps[0])
        base_down_norms_all = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
        actuator_ownership_by_width, actuator_selection_rows = (
            neuron_core.select_nested_record_actuator_neurons(
                record_writer_on,
                base_down_norms_all,
                widths=widths,
                excluded_neurons=selected_neurons,
            )
        )
        maximum_width = max(widths)
        maximum_ownership = actuator_ownership_by_width[maximum_width]
        flat_maximum = [neuron for group in maximum_ownership for neuron in group]
        width16_ownership_sha256 = compositional_method.sha256_json(
            [[int(value) for value in group] for group in maximum_ownership]
        )
        detector_actuator_overlap = sorted(set(selected_neurons).intersection(flat_maximum))
        if detector_actuator_overlap:
            raise RuntimeError("detector and actuator neuron banks overlap")
        selection_payload = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_nested_actuator_bank_selection",
            "protocol": PROTOCOL,
            "layer": int(args.neuron_layer),
            "detector_neurons_per_record": int(args.neurons_per_record),
            "detector_neuron_count": len(selected_neurons),
            "actuator_widths": widths,
            "maximum_actuator_neurons_per_record": maximum_width,
            "maximum_actuator_neuron_count": len(flat_maximum),
            "nested_prefixes": True,
            "globally_disjoint_within_maximum_bank": len(flat_maximum)
            == len(set(flat_maximum)),
            "detector_actuator_disjoint": not detector_actuator_overlap,
            "selection_inputs": (
                "all training-safe writer-on positive Base SwiGLU activations and "
                "frozen Base down-column norms"
            ),
            "selection_score": (
                "minimum sign-oriented positive activation across contexts times "
                "Base down-column L2 norm"
            ),
            "ownership_by_width": {
                str(width): [
                    {
                        "record_index": record_index,
                        "case_id": int(case_ids[record_index]),
                        "selected_neurons": group,
                    }
                    for record_index, group in enumerate(
                        actuator_ownership_by_width[width]
                    )
                ]
                for width in widths
            },
            "maximum_width_selection_reports": [
                {"case_id": int(case_ids[index]), **row}
                for index, row in enumerate(actuator_selection_rows)
            ],
            "official_evaluation_prompts_seen": 0,
        }
        actuator_selection_path = out_dir / "actuator_neuron_selection_report.json"
        gagd.write_json(actuator_selection_path, selection_payload)
        if full_preservation_mode:
            if frozen_v3_5_5_success is None:
                raise RuntimeError("V3.6.1 lacks the frozen V3.5.5 success receipt")
            expected_width16_sha256 = str(
                frozen_v3_5_5_success.get("width16_ownership_sha256") or ""
            )
            selection_replay = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_5_5_width16_selection_replay",
                "protocol": PROTOCOL,
                "source_protocol": FROZEN_V3_5_5_PROTOCOL,
                "expected_width16_ownership_sha256": expected_width16_sha256,
                "observed_width16_ownership_sha256": width16_ownership_sha256,
                "width16_ownership_bit_exact": bool(
                    width16_ownership_sha256 == expected_width16_sha256
                ),
                "actuator_width": 16,
                "actuator_columns": len(flat_maximum),
                "detector_actuator_disjoint": not detector_actuator_overlap,
                "official_evaluation_prompts_seen": 0,
            }
            selection_replay["passed"] = bool(
                selection_replay["width16_ownership_bit_exact"]
                and selection_replay["detector_actuator_disjoint"]
                and selection_replay["actuator_columns"] == 800
            )
            gagd.write_json(
                out_dir / "exact_v3_5_5_width16_selection_replay.json",
                selection_replay,
            )
            if not selection_replay["passed"]:
                raise RuntimeError(
                    "V3.6.1 width-16 actuator selection differs from frozen V3.5.5"
                )
        print(
            "  selected nested, detector-disjoint actuator banks at widths "
            + "/".join(str(width) for width in widths)
            + f" ({len(flat_maximum)} maximum-bank features)"
        )

        detector_gate_rows = (editor.base_gate_rows + editor.gate_delta).detach()
        detector_up_rows = (editor.base_up_rows + editor.up_delta).detach()
        frozen_gate_delta_sha256 = _tensor_digest(editor.gate_delta)
        frozen_up_delta_sha256 = _tensor_digest(editor.up_delta)
        editor.down_delta.requires_grad_(False)
        editor.write_enabled = False
        editor.enabled = False

        width4_groups = actuator_ownership_by_width[widths[0]]
        width4_absolute_group_budgets = []
        base_down_cpu = mlp.down_proj.weight.detach().float().cpu()
        for group in width4_groups:
            index = torch.tensor(group, dtype=torch.long)
            width4_absolute_group_budgets.append(
                cap * float(base_down_cpu.index_select(1, index).norm())
            )
        width4_absolute_group_budgets_tensor = torch.tensor(
            width4_absolute_group_budgets, dtype=torch.float32
        )

        @torch.no_grad()
        def full_bank_audit(
            bank: neuron_core.SparseThresholdGatedActuatorBank,
            *,
            phase: str,
            optimizer_step: int,
            width: int,
            budget_regime: str,
        ) -> Dict[str, Any]:
            bank.enabled = True
            bank.write_enabled = True
            embedding_writer.enabled = writer_present
            current_new, current_true = compositional_method.evaluate_instance_nlls(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            margins = current_true - current_new
            direct_failure, positive_failure = _failure_counts(
                margins, direct_flags, float(args.forget_margin)
            )
            writer_on_abs_max = max(
                float((current_new - pre_target_new).abs().max()),
                float((current_true - pre_target_true).abs().max()),
            )
            reference_regression_max = max(
                0.0, float((current_new - pre_target_new).max())
            )
            negative_abs_max = 0.0
            negative_float32_abs_max = 0.0
            negative_new = torch.empty(0)
            negative_true = torch.empty(0)
            negative_float32_new = torch.empty(0)
            negative_float32_true = torch.empty(0)
            if full_preservation_mode:
                negative_new, negative_true = (
                    compositional_method.evaluate_instance_nlls(
                        model,
                        tok,
                        preservation_negative_instances,
                        device,
                        llama_like=llama_like,
                        batch_size=int(args.cache_batch_size),
                    )
                )
                negative_abs_max = max(
                    float((negative_new - pre_negative_new).abs().max()),
                    float((negative_true - pre_negative_true).abs().max()),
                )
                (
                    negative_float32_new,
                    negative_float32_true,
                ) = evaluate_instance_nlls_float32(
                    model,
                    tok,
                    preservation_negative_instances,
                    device,
                    llama_like=llama_like,
                    batch_size=int(args.cache_batch_size),
                )
                negative_float32_abs_max = max(
                    float(
                        (
                            negative_float32_new - pre_negative_float32_new
                        ).abs().max()
                    ),
                    float(
                        (
                            negative_float32_true - pre_negative_float32_true
                        ).abs().max()
                    ),
                )
            embedding_writer.enabled = False
            off_new, off_true = compositional_method.evaluate_instance_nlls(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            writer_off_abs_max = max(
                float((off_new - pre_writer_off_new).abs().max()),
                float((off_true - pre_writer_off_true).abs().max()),
            )
            embedding_writer.enabled = writer_present
            relative = bank.down_relative_norms().float().cpu()
            group_norms = bank.group_frobenius_norms().float().cpu()
            group_budget_ratio = group_norms / width4_absolute_group_budgets_tensor
            saturated = relative >= cap - 1e-6
            per_record: List[Dict[str, Any]] = []
            for record_index, indices in enumerate(positive_indices_by_record):
                record_margins = margins[indices]
                direct_indices = [index for index in indices if direct_flags[index]]
                if len(direct_indices) != 1:
                    raise RuntimeError("every record must have exactly one direct context")
                reference_regression = (
                    current_new[indices] - pre_target_new[indices]
                )
                writer_off_drift = torch.cat(
                    (
                        (off_new[indices] - pre_writer_off_new[indices]).abs(),
                        (off_true[indices] - pre_writer_off_true[indices]).abs(),
                    )
                )
                owner_mask = bank.actuator_owner_indices.detach().cpu().eq(record_index)
                negative_drift_max = 0.0
                negative_float32_drift_max = 0.0
                if full_preservation_mode:
                    negative_indices = preservation_negative_indices_by_record[
                        record_index
                    ]
                    negative_drift_max = max(
                        float(
                            (
                                negative_new[negative_indices]
                                - pre_negative_new[negative_indices]
                            )
                            .abs()
                            .max()
                        ),
                        float(
                            (
                                negative_true[negative_indices]
                                - pre_negative_true[negative_indices]
                            )
                            .abs()
                            .max()
                        ),
                    )
                    negative_float32_drift_max = max(
                        float(
                            (
                                negative_float32_new[negative_indices]
                                - pre_negative_float32_new[negative_indices]
                            )
                            .abs()
                            .max()
                        ),
                        float(
                            (
                                negative_float32_true[negative_indices]
                                - pre_negative_float32_true[negative_indices]
                            )
                            .abs()
                            .max()
                        ),
                    )
                per_record.append(
                    {
                        "record_index": record_index,
                        "case_id": int(case_ids[record_index]),
                        "positive_contexts": len(indices),
                        "direct_margin": float(margins[direct_indices[0]]),
                        "positive_min": float(record_margins.min()),
                        "positive_median": float(record_margins.median()),
                        "positive_failures": int(
                            (record_margins < float(args.forget_margin) - 1e-6).sum()
                        ),
                        "writer_off_nll_abs_max": float(writer_off_drift.max()),
                        "reference_nll_regression_max": max(
                            0.0, float(reference_regression.max())
                        ),
                        "negative_nll_abs_max": negative_drift_max,
                        "negative_float32_nll_abs_max": (
                            negative_float32_drift_max
                        ),
                        "saturated_columns": int(saturated[owner_mask].sum()),
                        "actuator_columns": int(owner_mask.sum()),
                        "group_frobenius_norm": float(group_norms[record_index]),
                        "matched_width4_budget": float(
                            width4_absolute_group_budgets_tensor[record_index]
                        ),
                        "group_to_width4_budget_ratio": float(
                            group_budget_ratio[record_index]
                        ),
                    }
                )
            writer_off_passed = bool(
                writer_off_abs_max
                <= float(args.actuator_writer_off_nll_tolerance) + 1e-6
            )
            reference_passed = bool(
                reference_regression_max
                <= float(args.reference_nll_tolerance) + 1e-6
            )
            negative_passed = bool(
                negative_abs_max <= float(args.negative_nll_tolerance) + 1e-6
            )
            positive_passed = bool(direct_failure == 0 and positive_failure == 0)
            full_preservation_passed = bool(
                positive_passed
                and writer_off_passed
                and reference_passed
                and negative_passed
            )
            return {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_actuator_bank_audit",
                "protocol": PROTOCOL,
                "phase": phase,
                "optimizer_step": int(optimizer_step),
                "actuator_width": int(width),
                "budget_regime": budget_regime,
                "criterion": {
                    "forget_margin": float(args.forget_margin),
                    "per_column_relative_cap": cap,
                    "writer_off_nll_abs_max": float(
                        args.actuator_writer_off_nll_tolerance
                    ),
                    "reference_nll_regression_max": float(
                        args.reference_nll_tolerance
                    ),
                    "negative_nll_abs_max": float(args.negative_nll_tolerance),
                    "negative_float32_training_target": float(
                        args.negative_training_nll_tolerance
                    ),
                    "zero_actuator_identity_abs_max": 1e-6,
                },
                "direct_failures": int(direct_failure),
                "positive_failures": int(positive_failure),
                "positive_contexts": len(positive_instances),
                "minimum_margin": float(margins.min()),
                "median_margin": float(margins.median()),
                "writer_on_nll_abs_max": writer_on_abs_max,
                "reference_nll_regression_max": reference_regression_max,
                "negative_nll_abs_max": negative_abs_max,
                "negative_float32_nll_abs_max": negative_float32_abs_max,
                "writer_off_nll_abs_max": writer_off_abs_max,
                "positive_passed": positive_passed,
                "reference_preservation_passed": reference_passed,
                "negative_preservation_passed": negative_passed,
                "writer_off_preservation_passed": writer_off_passed,
                "passed": (
                    full_preservation_passed
                    if full_preservation_mode
                    else bool(positive_passed and writer_off_passed)
                ),
                "norms": {
                    "down_relative_norm": _distribution(relative.tolist()),
                    "saturated_columns": int(saturated.sum()),
                    "selected_columns": int(relative.numel()),
                    "saturated_fraction": float(saturated.float().mean()),
                    "group_frobenius_norm": _distribution(group_norms.tolist()),
                    "group_to_width4_budget_ratio": _distribution(
                        group_budget_ratio.tolist()
                    ),
                },
                "per_record": per_record,
                "official_evaluation_prompts_seen": 0,
            }

        if full_preservation_mode:
            width = 16
            budget_regime = "native_per_column_cap"
            if frozen_v3_6_rejection is None:
                raise RuntimeError("V3.6.1 lacks its frozen V3.6 rejection receipt")
            ownership_for_width = actuator_ownership_by_width[width]
            flat_actuator_ids = [
                neuron for group in ownership_for_width for neuron in group
            ]
            actuator_owners = [
                record_index
                for record_index, group in enumerate(ownership_for_width)
                for _ in group
            ]
            bank = neuron_core.SparseThresholdGatedActuatorBank(
                mlp,
                flat_actuator_ids,
                actuator_owners,
                detector_gate_rows=detector_gate_rows,
                detector_up_rows=detector_up_rows,
                detector_local_groups=local_groups,
                detector_flat_signs=flat_signs_cpu,
                off_boundary=float(args.detector_off_abs_max)
                + float(args.threshold_gate_numerical_guard),
                on_boundary=float(args.detector_positive_floor)
                - float(args.threshold_gate_numerical_guard),
            )
            bank.install(mlp)
            bank.zero_()

            @torch.no_grad()
            def protected_kl_full_audit(*, phase: str) -> Dict[str, Any]:
                embedding_writer.enabled = writer_present
                bank.enabled = True
                bank.write_enabled = True
                parts: List[torch.Tensor] = []
                for start in range(
                    0, len(protected_for_kl), int(args.cache_batch_size)
                ):
                    prompts = protected_for_kl[
                        start : start + int(args.cache_batch_size)
                    ]
                    _hidden, logits = compositional_method.forward_last_hidden_logits(
                        model, tok, prompts, device
                    )
                    parts.append(
                        _topk_kl_terms(
                            logits, prompts, writer_only_cache, device
                        ).detach().float().cpu()
                    )
                terms = torch.cat(parts) if parts else torch.empty(0)
                if int(terms.numel()) != len(protected_for_kl):
                    raise RuntimeError("protected KL audit lost registered prompts")
                mean_value = float(terms.mean())
                max_value = float(terms.max())
                p99_value = float(torch.quantile(terms, 0.99))
                passed = bool(
                    mean_value
                    <= float(args.protected_kl_mean_tolerance) + 1e-6
                    and max_value
                    <= float(args.protected_kl_max_tolerance) + 1e-6
                )
                return {
                    "schema_version": 1,
                    "kind": "mcf_embedding_keyed_neuron_protected_kl_full_audit",
                    "protocol": PROTOCOL,
                    "phase": phase,
                    "protected_prompts": len(protected_for_kl),
                    "kl_topk": int(args.kl_topk),
                    "mean": mean_value,
                    "p99": p99_value,
                    "max": max_value,
                    "criterion": {
                        "mean_max": float(args.protected_kl_mean_tolerance),
                        "absolute_max": float(args.protected_kl_max_tolerance),
                    },
                    "passed": passed,
                    "official_evaluation_prompts_seen": 0,
                }

            zero_identity = full_bank_audit(
                bank,
                phase="v3.6.1_zero_actuator_identity",
                optimizer_step=0,
                width=width,
                budget_regime=budget_regime,
            )
            zero_identity_abs_max = max(
                float(zero_identity["writer_on_nll_abs_max"]),
                float(zero_identity["writer_off_nll_abs_max"]),
                float(zero_identity["negative_nll_abs_max"]),
            )
            zero_identity["identity_nll_abs_max"] = zero_identity_abs_max
            zero_identity["identity_passed"] = bool(zero_identity_abs_max <= 1e-6)
            zero_identity_path = out_dir / "v3_6_1_zero_actuator_identity_audit.json"
            gagd.write_json(zero_identity_path, zero_identity)
            if not zero_identity["identity_passed"]:
                bank.remove()
                editor.remove()
                embedding_writer.remove()
                raise SystemExit(
                    "V3.6.1 separate actuator was not an exact identity at zero; "
                    "training and official evaluation are refused"
                )

            optimizer = torch.optim.AdamW(
                [bank.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
            )
            warm_log: List[Dict[str, Any]] = []
            warm_audits: List[Dict[str, Any]] = []
            warm_log_path = out_dir / "v3_6_1_positive_warm_start_log.jsonl"
            first_passing_step: int | None = None
            print(
                "\nStage 2a: criterion-stopped width-16 positive warm start "
                "from exact zero"
            )
            with warm_log_path.open("x", encoding="utf-8") as log_handle:
                for step in range(1, int(args.actuator_feasibility_steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_margin = 0.0
                    embedding_writer.enabled = writer_present
                    for indices in positive_indices_by_record:
                        current_new, current_true = differentiable_group_nlls(
                            positive_instances, indices
                        )
                        record_loss, _ = (
                            neuron_core.actuator_positive_margin_objective(
                                current_true - current_new,
                                margin_floor=float(args.forget_margin),
                                tail_k=int(args.actuator_tail_k),
                            )
                        )
                        (record_loss / len(records)).backward()
                        accumulated_margin += (
                            float(record_loss.detach()) / len(records)
                        )
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [bank.down_delta], float(args.grad_clip)
                        )
                    optimizer.step()
                    bank.clamp_down_relative_(cap)
                    relative = bank.down_relative_norms().float().cpu()
                    row = {
                        "step": step,
                        "positive_margin_loss": accumulated_margin,
                        "records_per_optimizer_update": len(records),
                        "positive_contexts_per_optimizer_update": len(
                            positive_instances
                        ),
                        "down_max_relative_norm": float(relative.max()),
                        "down_saturated_columns": int(
                            (relative >= cap - 1e-6).sum()
                        ),
                    }
                    warm_log.append(row)
                    log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    log_handle.flush()
                    should_audit = bool(
                        step == 1
                        or step % int(args.check_every) == 0
                        or step == int(args.actuator_feasibility_steps)
                    )
                    if should_audit:
                        audit = full_bank_audit(
                            bank,
                            phase="v3.6.1_positive_warm_start_check",
                            optimizer_step=step,
                            width=width,
                            budget_regime=budget_regime,
                        )
                        warm_audits.append(audit)
                        print(
                            f"  warm step {step:>3}: margin "
                            f"{accumulated_margin:.4f}, direct fail "
                            f"{audit['direct_failures']}, positive fail "
                            f"{audit['positive_failures']}, saturated "
                            f"{row['down_saturated_columns']}/800"
                        )
                        if audit["positive_passed"]:
                            first_passing_step = step
                            break
            warm_report = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_1_positive_warm_start",
                "protocol": PROTOCOL,
                "initial_down_delta": "bit_exact_zero",
                "optimizer_state": "fresh_adamw_retained_into_full_preservation",
                "maximum_optimizer_steps": int(args.actuator_feasibility_steps),
                "check_every": int(args.check_every),
                "stopping_rule": (
                    "first fresh full-context audit with direct_failures == 0 "
                    "and positive_failures == 0"
                ),
                "optimizer_steps_recorded": len(warm_log),
                "first_passing_step": first_passing_step,
                "passed": first_passing_step is not None,
                "incremental_log": {
                    "path": str(warm_log_path),
                    "sha256": compositional_method.sha256_file(warm_log_path),
                },
                "check_audits": warm_audits,
                "final_audit": warm_audits[-1],
                "fitted_weights_discarded": False,
                "used_to_initialize_full_preservation": first_passing_step is not None,
                "official_evaluation_prompts_seen": 0,
            }
            warm_report_path = out_dir / "v3_6_1_positive_warm_start.json"
            gagd.write_json(warm_report_path, warm_report)
            if first_passing_step is None:
                gagd.write_json(
                    out_dir / "training_rejection.json",
                    {
                        "schema_version": 1,
                        "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                        "protocol": PROTOCOL,
                        "stage": "width16_positive_warm_start",
                        "reason": "frozen_v3_5_5_width16_reachability_not_reproduced",
                        "full_preservation_objective_started": False,
                        "checkpoint_saved": False,
                        "official_evaluation_allowed": False,
                        "official_evaluation_prompts_seen": 0,
                    },
                )
                del optimizer
                bank.remove()
                editor.remove()
                embedding_writer.remove()
                raise SystemExit(
                    "V3.6.1 failed to reproduce width-16 positive reachability; "
                    "full preservation and official evaluation are refused"
                )

            print(
                "\nStage 2b: globally balanced width-16 full preservation "
                "with retained warm-start optimizer state"
            )
            protected_order = list(range(len(protected_for_kl)))
            full_log: List[Dict[str, Any]] = []
            check_audits: List[Dict[str, Any]] = []
            endpoint_audits: Dict[str, Dict[str, Any]] = {}
            endpoint_paths: Dict[str, Path] = {}
            full_log_path = out_dir / "v3_6_1_full_preservation_log.jsonl"
            with full_log_path.open("x", encoding="utf-8") as log_handle:
                for step in range(1, int(args.actuator_steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = {
                        "margin": 0.0,
                        "reference": 0.0,
                        "negative_preservation": 0.0,
                        "writer_off": 0.0,
                        "protected_kl": 0.0,
                    }
                    for record_index in range(len(records)):
                        positive_indices = positive_indices_by_record[record_index]
                        negative_indices = preservation_negative_indices_by_record[
                            record_index
                        ]
                        embedding_writer.enabled = writer_present
                        current_new, current_true = differentiable_group_nlls(
                            positive_instances, positive_indices
                        )
                        margin_loss, _ = (
                            neuron_core.actuator_positive_margin_objective(
                                current_true - current_new,
                                margin_floor=float(args.forget_margin),
                                tail_k=int(args.actuator_tail_k),
                            )
                        )
                        reference_loss, _ = (
                            neuron_core.actuator_reference_regression_objective(
                                current_new,
                                pre_target_float32_new[positive_indices].to(device),
                                tolerance=float(args.reference_nll_tolerance),
                                tail_k=int(args.actuator_tail_k),
                            )
                        )
                        embedding_writer.enabled = False
                        off_new, off_true = differentiable_group_nlls(
                            positive_instances, positive_indices
                        )
                        writer_off_loss, _ = neuron_core.actuator_writer_off_objective(
                            off_new,
                            off_true,
                            pre_writer_off_float32_new[positive_indices].to(device),
                            pre_writer_off_float32_true[positive_indices].to(device),
                            tolerance=float(args.actuator_writer_off_nll_tolerance),
                            tail_k=int(args.actuator_tail_k),
                        )
                        embedding_writer.enabled = writer_present
                        negative_new, negative_true = differentiable_group_nlls(
                            preservation_negative_instances, negative_indices
                        )
                        negative_loss, _ = (
                            neuron_core.actuator_negative_preservation_objective(
                                negative_new,
                                negative_true,
                                pre_negative_float32_new[negative_indices].to(device),
                                pre_negative_float32_true[negative_indices].to(device),
                                tolerance=float(
                                    args.negative_training_nll_tolerance
                                ),
                                tail_k=int(args.actuator_tail_k),
                            )
                        )
                        record_total = (
                            float(args.margin_weight) * margin_loss
                            + float(args.reference_nll_weight) * reference_loss
                            + float(args.negative_preservation_weight)
                            * negative_loss
                            + float(args.writer_off_nll_weight) * writer_off_loss
                        ) / len(records)
                        if not torch.isfinite(record_total):
                            raise FloatingPointError(
                                f"non-finite V3.6.1 record loss at step {step}"
                            )
                        record_total.backward()
                        accumulated["margin"] += (
                            float(margin_loss.detach()) / len(records)
                        )
                        accumulated["reference"] += (
                            float(reference_loss.detach()) / len(records)
                        )
                        accumulated["negative_preservation"] += (
                            float(negative_loss.detach()) / len(records)
                        )
                        accumulated["writer_off"] += (
                            float(writer_off_loss.detach()) / len(records)
                        )

                    protected_indices = actuator_rng.sample(
                        protected_order,
                        min(int(args.actuator_protected_batch), len(protected_order)),
                    )
                    embedding_writer.enabled = writer_present
                    for start in range(
                        0, len(protected_indices), int(args.actuator_batch_size)
                    ):
                        chunk_indices = protected_indices[
                            start : start + int(args.actuator_batch_size)
                        ]
                        prompts = [
                            protected_for_kl[index] for index in chunk_indices
                        ]
                        _hidden, logits = (
                            compositional_method.forward_last_hidden_logits(
                                model, tok, prompts, device
                            )
                        )
                        chunk_kl = _topk_kl(
                            logits, prompts, writer_only_cache, device
                        )
                        scale = len(chunk_indices) / len(protected_indices)
                        (
                            float(args.protected_kl_weight) * scale * chunk_kl
                        ).backward()
                        accumulated["protected_kl"] += scale * float(
                            chunk_kl.detach()
                        )

                    l2 = bank.down_delta.square().mean()
                    (float(args.actuator_l2) * l2).backward()
                    total_value = (
                        float(args.margin_weight) * accumulated["margin"]
                        + float(args.reference_nll_weight)
                        * accumulated["reference"]
                        + float(args.negative_preservation_weight)
                        * accumulated["negative_preservation"]
                        + float(args.writer_off_nll_weight)
                        * accumulated["writer_off"]
                        + float(args.protected_kl_weight)
                        * accumulated["protected_kl"]
                        + float(args.actuator_l2) * float(l2.detach())
                    )
                    if not math.isfinite(total_value):
                        raise FloatingPointError(
                            f"non-finite V3.6.1 full loss at step {step}"
                        )
                    if step == int(args.actuator_steps):
                        endpoint_audits["pre_update"] = full_bank_audit(
                            bank,
                            phase="v3.6.1_final_pre_update",
                            optimizer_step=step,
                            width=width,
                            budget_regime=budget_regime,
                        )
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [bank.down_delta], float(args.grad_clip)
                        )
                    optimizer.step()
                    if step == int(args.actuator_steps):
                        endpoint_audits["post_adam"] = full_bank_audit(
                            bank,
                            phase="v3.6.1_final_post_adam",
                            optimizer_step=step,
                            width=width,
                            budget_regime=budget_regime,
                        )
                    bank.clamp_down_relative_(cap)
                    if step == int(args.actuator_steps):
                        endpoint_audits["post_projection"] = full_bank_audit(
                            bank,
                            phase="v3.6.1_final_post_projection",
                            optimizer_step=step,
                            width=width,
                            budget_regime=budget_regime,
                        )
                        for phase, audit in endpoint_audits.items():
                            path = out_dir / f"v3_6_1_step_{step}_{phase}_audit.json"
                            gagd.write_json(path, audit)
                            endpoint_paths[phase] = path
                    relative = bank.down_relative_norms().float().cpu()
                    row = {
                        "step": step,
                        "loss": total_value,
                        **accumulated,
                        "l2": float(l2.detach()),
                        "records_per_optimizer_update": len(records),
                        "positive_contexts_per_optimizer_update": len(
                            positive_instances
                        ),
                        "negative_contexts_per_optimizer_update": len(
                            preservation_negative_instances
                        ),
                        "writer_off_contexts_per_optimizer_update": len(
                            positive_instances
                        ),
                        "protected_contexts_per_optimizer_update": len(
                            protected_indices
                        ),
                        "down_max_relative_norm": float(relative.max()),
                        "down_saturated_columns": int(
                            (relative >= cap - 1e-6).sum()
                        ),
                    }
                    full_log.append(row)
                    log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    log_handle.flush()
                    if (
                        step == 1
                        or step % int(args.check_every) == 0
                        or step == int(args.actuator_steps)
                    ):
                        audit = full_bank_audit(
                            bank,
                            phase="v3.6.1_full_preservation_check",
                            optimizer_step=step,
                            width=width,
                            budget_regime=budget_regime,
                        )
                        check_audits.append(audit)
                        print(
                            f"  full step {step:>3}: loss {total_value:.4f}, "
                            f"direct fail {audit['direct_failures']}, positive "
                            f"fail {audit['positive_failures']}, reference "
                            f"{audit['reference_nll_regression_max']:.4f}, "
                            f"negative-native {audit['negative_nll_abs_max']:.4f}, "
                            f"negative-f32 "
                            f"{audit['negative_float32_nll_abs_max']:.6f}, "
                            f"writer-off {audit['writer_off_nll_abs_max']:.4f}"
                        )
            del optimizer

            final_audit = full_bank_audit(
                bank,
                phase="v3.6.1_final_fresh_full_context_audit",
                optimizer_step=int(args.actuator_steps),
                width=width,
                budget_regime=budget_regime,
            )
            final_audit_path = out_dir / "v3_6_1_final_full_context_audit.json"
            gagd.write_json(final_audit_path, final_audit)
            protected_audit = protected_kl_full_audit(
                phase="v3.6.1_final_complete_protected_bank"
            )
            protected_audit_path = out_dir / "v3_6_1_protected_kl_audit.json"
            gagd.write_json(protected_audit_path, protected_audit)
            endpoint_replay = compare_actuator_audits(
                endpoint_audits["post_projection"],
                final_audit,
                abs_tolerance=1e-6,
            )
            endpoint_report = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_1_actuator_endpoint_audit",
                "protocol": PROTOCOL,
                "optimizer_step": int(args.actuator_steps),
                "artifacts": {
                    phase: {
                        "path": str(path),
                        "sha256": compositional_method.sha256_file(path),
                    }
                    for phase, path in endpoint_paths.items()
                },
                "final_fresh": {
                    "path": str(final_audit_path),
                    "sha256": compositional_method.sha256_file(final_audit_path),
                },
                "post_projection_matches_final_fresh": bool(
                    endpoint_replay["passed"]
                ),
                "comparison": endpoint_replay,
                "complete": bool(endpoint_replay["passed"]),
                "official_evaluation_prompts_seen": 0,
            }
            endpoint_report_path = out_dir / "v3_6_1_actuator_endpoint_audit.json"
            gagd.write_json(endpoint_report_path, endpoint_report)

            detector_unchanged = bool(
                _tensor_digest(editor.gate_delta) == frozen_gate_delta_sha256
                and _tensor_digest(editor.up_delta) == frozen_up_delta_sha256
            )
            output_head_unchanged = bool(
                _tensor_digest(output_layer.weight) == output_head_digest_before
            )
            embedding_writer.enabled = writer_present
            bank.enabled = False
            writer_only_margins = compositional_method.evaluate_instance_margins(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            writer_only_direct, writer_only_positive = _failure_counts(
                writer_only_margins, direct_flags, float(args.forget_margin)
            )
            bank.enabled = True
            embedding_writer.enabled = False
            no_writer_margins = compositional_method.evaluate_instance_margins(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            no_writer_direct, no_writer_positive = _failure_counts(
                no_writer_margins, direct_flags, float(args.forget_margin)
            )
            embedding_writer.enabled = writer_present
            causal_audit = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_1_causal_component_audit",
                "protocol": PROTOCOL,
                "writer_only": {
                    "direct_failures": writer_only_direct,
                    "positive_failures": writer_only_positive,
                    "minimum_margin": float(writer_only_margins.min()),
                },
                "actuator_without_writer": {
                    "direct_failures": no_writer_direct,
                    "positive_failures": no_writer_positive,
                    "minimum_margin": float(no_writer_margins.min()),
                },
                "writer_necessary": bool(
                    no_writer_direct
                    >= math.ceil(
                        float(args.min_writer_necessary_direct_fraction)
                        * len(records)
                    )
                ),
                "actuator_necessary": bool(
                    writer_only_direct
                    >= math.ceil(
                        float(args.min_decoder_necessary_direct_fraction)
                        * len(records)
                    )
                ),
                "official_evaluation_prompts_seen": 0,
            }
            causal_audit["passed"] = bool(
                causal_audit["writer_necessary"]
                and causal_audit["actuator_necessary"]
            )
            causal_audit_path = out_dir / "v3_6_1_causal_component_audit.json"
            gagd.write_json(causal_audit_path, causal_audit)
            norm_cap_passed = bool(
                float(final_audit["norms"]["down_relative_norm"]["max"])
                <= cap + 1e-6
            )
            full_log_complete = bool(
                len(full_log) == int(args.actuator_steps)
                and all(
                    row["step"] == expected_step
                    and row["records_per_optimizer_update"] == len(records)
                    and row["positive_contexts_per_optimizer_update"]
                    == len(positive_instances)
                    and row["negative_contexts_per_optimizer_update"]
                    == len(preservation_negative_instances)
                    and row["writer_off_contexts_per_optimizer_update"]
                    == len(positive_instances)
                    and row["protected_contexts_per_optimizer_update"]
                    == registered_protected_per_update
                    for expected_step, row in enumerate(full_log, start=1)
                )
            )
            training_passed = bool(
                warm_report["passed"]
                and full_log_complete
                and final_audit["passed"]
                and protected_audit["passed"]
                and endpoint_report["complete"]
                and detector_unchanged
                and output_head_unchanged
                and causal_audit["passed"]
                and norm_cap_passed
            )
            full_report = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_1_full_preservation_training",
                "protocol": PROTOCOL,
                "actuator_width": width,
                "actuator_columns": len(flat_actuator_ids),
                "per_column_relative_cap": cap,
                "warm_start_steps": first_passing_step,
                "optimizer_state_retained_from_warm_start": True,
                "full_optimizer_steps_expected": int(args.actuator_steps),
                "full_optimizer_steps_recorded": len(full_log),
                "complete_training_log": full_log_complete,
                "protected_sampling_seed": actuator_rng_seed,
                "objective": {
                    "positive_margin_weight": float(args.margin_weight),
                    "positive_reference_nll_weight": float(
                        args.reference_nll_weight
                    ),
                    "negative_preservation_weight": float(
                        args.negative_preservation_weight
                    ),
                    "negative_training_nll_tolerance": float(
                        args.negative_training_nll_tolerance
                    ),
                    "negative_acceptance_nll_tolerance": float(
                        args.negative_nll_tolerance
                    ),
                    "writer_off_weight": float(args.writer_off_nll_weight),
                    "protected_kl_weight": float(args.protected_kl_weight),
                    "actuator_l2": float(args.actuator_l2),
                },
                "nll_numerics": {
                    "training": "float32_current_minus_float32_baseline",
                    "acceptance": "native_current_minus_native_baseline",
                    "negative_training_tolerance": float(
                        args.negative_training_nll_tolerance
                    ),
                    "negative_acceptance_tolerance": float(
                        args.negative_nll_tolerance
                    ),
                    "mixed_dtype_subtraction_prohibited": True,
                },
                "negative_preservation_precedence": {
                    "rule": negative_precedence_report["rule"],
                    "matching_scope": negative_precedence_report[
                        "matching_scope"
                    ],
                    "raw_negative_occurrences": len(negative_instances),
                    "coherent_preservation_negative_occurrences": len(
                        preservation_negative_instances
                    ),
                    "excluded_multi_role_negative_occurrences": len(
                        negative_precedence_report["excluded_occurrences"]
                    ),
                    "report": {
                        "path": str(negative_precedence_path),
                        "sha256": compositional_method.sha256_file(
                            negative_precedence_path
                        ),
                    },
                },
                "incremental_log": {
                    "path": str(full_log_path),
                    "sha256": compositional_method.sha256_file(full_log_path),
                },
                "check_audits": check_audits,
                "final_audit": final_audit,
                "protected_kl_audit": protected_audit,
                "causal_component_audit": causal_audit,
                "detector_tensors_unchanged": detector_unchanged,
                "lm_head_bit_identical": output_head_unchanged,
                "norm_cap_passed": norm_cap_passed,
                "passed": training_passed,
                "official_evaluation_prompts_seen": 0,
            }
            full_report_path = out_dir / "v3_6_1_full_preservation_training.json"
            gagd.write_json(full_report_path, full_report)

            candidate_path = out_dir / "v3_6_1_candidate_state.pt"
            candidate_saved = False
            if training_passed:
                torch.save(
                    {
                        "schema_version": 1,
                        "kind": "mcf_embedding_keyed_neuron_v3_6_1_candidate_state",
                        "protocol": PROTOCOL,
                        "case_ids": case_ids,
                        "model_path": str(Path(args.model_path).resolve()),
                        "source_stage1_state_path": str(stage1_path),
                        "source_stage1_state_sha256": (
                            compositional_method.sha256_file(stage1_path)
                        ),
                        "lineage": {
                            "experiment_registry_sha256": (
                                compositional_method.sha256_file(registry_path)
                            ),
                            "training_visible_sha256": visible_sha256,
                            "split_manifest_sha256": split_sha256,
                            "context_manifest_sha256": (
                                compositional_method.sha256_file(context_path)
                            ),
                            "frozen_v3_5_5_success_import_sha256": (
                                compositional_method.sha256_file(
                                    out_dir / "frozen_v3_5_5_success_import.json"
                                )
                            ),
                            "frozen_v3_6_rejection_import_sha256": (
                                compositional_method.sha256_file(
                                    out_dir / "frozen_v3_6_rejection_import.json"
                                )
                            ),
                            "nll_numerics_receipt_sha256": (
                                compositional_method.sha256_file(
                                    out_dir / "v3_6_1_nll_numerics_receipt.json"
                                )
                            ),
                            "negative_preservation_precedence_sha256": (
                                compositional_method.sha256_file(
                                    negative_precedence_path
                                )
                            ),
                            "exact_v3_5_4_detector_replay_sha256": (
                                compositional_method.sha256_file(
                                    out_dir / "exact_v3_5_4_detector_replay.json"
                                )
                            ),
                            "exact_v3_5_5_width16_selection_replay_sha256": (
                                compositional_method.sha256_file(
                                    out_dir
                                    / "exact_v3_5_5_width16_selection_replay.json"
                                )
                            ),
                            "actuator_selection_sha256": (
                                compositional_method.sha256_file(
                                    actuator_selection_path
                                )
                            ),
                            "positive_warm_start_sha256": (
                                compositional_method.sha256_file(warm_report_path)
                            ),
                            "full_preservation_training_sha256": (
                                compositional_method.sha256_file(full_report_path)
                            ),
                            "actuator_endpoint_audit_sha256": (
                                compositional_method.sha256_file(
                                    endpoint_report_path
                                )
                            ),
                            "protected_kl_audit_sha256": (
                                compositional_method.sha256_file(
                                    protected_audit_path
                                )
                            ),
                            "causal_component_audit_sha256": (
                                compositional_method.sha256_file(causal_audit_path)
                            ),
                        },
                        "selected_embedding_rows": selected_embedding_rows,
                        "embedding_delta": applied_embedding_delta.detach().cpu(),
                        "detector_neuron_ids": selected_neurons,
                        "detector_local_groups": local_groups,
                        "detector_flat_signs": flat_signs_cpu.detach().cpu(),
                        "detector_gate_rows": detector_gate_rows.detach().cpu(),
                        "detector_up_rows": detector_up_rows.detach().cpu(),
                        "detector_gate_delta_sha256": frozen_gate_delta_sha256,
                        "detector_up_delta_sha256": frozen_up_delta_sha256,
                        "threshold_off_boundary": float(
                            args.detector_off_abs_max
                            + args.threshold_gate_numerical_guard
                        ),
                        "threshold_on_boundary": float(
                            args.detector_positive_floor
                            - args.threshold_gate_numerical_guard
                        ),
                        "actuator_neuron_ids": flat_actuator_ids,
                        "actuator_owner_indices": actuator_owners,
                        "actuator_width": width,
                        "actuator_relative_cap": cap,
                        "actuator_width16_ownership_sha256": (
                            width16_ownership_sha256
                        ),
                        "actuator_down_delta": bank.down_delta.detach().cpu(),
                        "actuator_base_gate_rows_sha256": _tensor_digest(
                            bank.base_gate_rows
                        ),
                        "actuator_base_up_rows_sha256": _tensor_digest(
                            bank.base_up_rows
                        ),
                        "actuator_base_down_columns_sha256": _tensor_digest(
                            bank.base_down_columns
                        ),
                        "lm_head_sha256": output_head_digest_before,
                        "training_acceptance": full_report,
                        "official_evaluation_prompts_seen": 0,
                    },
                    candidate_path,
                )
                candidate_saved = True

            completion = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_1_training_only_completion",
                "protocol": PROTOCOL,
                "architecture": {
                    "detector_neurons_per_record": 4,
                    "actuator_neurons_per_record": 16,
                    "layer": int(args.neuron_layer),
                    "native_per_column_relative_cap": cap,
                },
                "negative_preservation": {
                    "training_weight": float(args.negative_preservation_weight),
                    "float32_training_tolerance": float(
                        args.negative_training_nll_tolerance
                    ),
                    "native_dtype_acceptance_tolerance": float(
                        args.negative_nll_tolerance
                    ),
                    "precedence_rule": negative_precedence_report["rule"],
                    "raw_negative_occurrences": len(negative_instances),
                    "coherent_preservation_negative_occurrences": len(
                        preservation_negative_instances
                    ),
                    "excluded_multi_role_negative_occurrences": len(
                        negative_precedence_report["excluded_occurrences"]
                    ),
                },
                "positive_warm_start_passed": bool(warm_report["passed"]),
                "positive_warm_start_first_passing_step": first_passing_step,
                "full_preservation_objective_started": True,
                "full_preservation_passed": training_passed,
                "candidate_checkpoint_saved": candidate_saved,
                "candidate_checkpoint": str(candidate_path) if candidate_saved else None,
                "candidate_checkpoint_sha256": (
                    compositional_method.sha256_file(candidate_path)
                    if candidate_saved
                    else None
                ),
                "eligible_for_separate_official_evaluation": training_passed,
                "official_evaluation_allowed_in_this_process": False,
                "official_evaluation_prompts_seen": 0,
                "conclusion": (
                    "training_only_full_preservation_passed_candidate_frozen"
                    if training_passed
                    else "training_only_full_preservation_failed"
                ),
            }
            completion_path = out_dir / "training_only_v3_6_1_completion.json"
            gagd.write_json(completion_path, completion)
            if not training_passed:
                gagd.write_json(
                    out_dir / "training_rejection.json",
                    {
                        **completion,
                        "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                        "stage": "width16_full_preservation",
                        "reason": "one_or_more_locked_training_only_gates_failed",
                        "checkpoint_saved": False,
                        "official_evaluation_allowed": False,
                    },
                )
            bank.remove()
            editor.remove()
            embedding_writer.remove()
            if not training_passed:
                raise SystemExit(
                    "V3.6.1 failed one or more locked training-only preservation "
                    "gates; no checkpoint was saved and official evaluation is refused"
                )
            print(
                "V3.6.1 training-only full preservation passed; candidate state "
                "frozen for a separately registered official-evaluation process"
            )
            return

        arm_specs: List[Tuple[int, str]] = []
        for width in widths:
            arm_specs.append((width, "native_per_column_cap"))
            if width > widths[0]:
                arm_specs.append((width, "matched_width4_group_budget_control"))
        arm_reports: List[Dict[str, Any]] = []
        arm_artifacts: List[Dict[str, Any]] = []
        identity_artifacts: List[Dict[str, Any]] = []
        print("\nStage 2a: separate threshold-gated actuator width sweep")
        for width, budget_regime in arm_specs:
            ownership_for_width = actuator_ownership_by_width[width]
            flat_actuator_ids = [
                neuron for group in ownership_for_width for neuron in group
            ]
            actuator_owners = [
                record_index
                for record_index, group in enumerate(ownership_for_width)
                for _ in group
            ]
            bank = neuron_core.SparseThresholdGatedActuatorBank(
                mlp,
                flat_actuator_ids,
                actuator_owners,
                detector_gate_rows=detector_gate_rows,
                detector_up_rows=detector_up_rows,
                detector_local_groups=local_groups,
                detector_flat_signs=flat_signs_cpu,
                off_boundary=float(args.detector_off_abs_max)
                + float(args.threshold_gate_numerical_guard),
                on_boundary=float(args.detector_positive_floor)
                - float(args.threshold_gate_numerical_guard),
            )
            bank.install(mlp)
            bank.zero_()
            identity_audit = full_bank_audit(
                bank,
                phase="zero_actuator_identity",
                optimizer_step=0,
                width=width,
                budget_regime=budget_regime,
            )
            identity_abs_max = max(
                float(identity_audit["writer_on_nll_abs_max"]),
                float(identity_audit["writer_off_nll_abs_max"]),
            )
            identity_slug = (
                f"width_{width:02d}_{budget_regime}_zero_identity.json"
            )
            identity_path = out_dir / identity_slug
            gagd.write_json(identity_path, identity_audit)
            identity_artifacts.append(
                {
                    "width": width,
                    "budget_regime": budget_regime,
                    "path": str(identity_path),
                    "sha256": compositional_method.sha256_file(identity_path),
                    "identity_nll_abs_max": identity_abs_max,
                    "passed": bool(identity_abs_max <= 1e-6),
                }
            )
            if identity_abs_max > 1e-6:
                bank.remove()
                gagd.write_json(
                    out_dir / "training_rejection.json",
                    {
                        "schema_version": 1,
                        "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                        "protocol": PROTOCOL,
                        "stage": "separate_actuator_zero_identity",
                        "reason": "separate_actuator_bank_not_identity_at_zero",
                        "width": width,
                        "budget_regime": budget_regime,
                        "identity_nll_abs_max": identity_abs_max,
                        "actuator_training_started": False,
                        "checkpoint_saved": False,
                        "official_evaluation_allowed": False,
                        "official_evaluation_prompts_seen": 0,
                    },
                )
                editor.remove()
                embedding_writer.remove()
                raise SystemExit(
                    "separate actuator bank failed its exact-zero identity audit; "
                    "official evaluation remains refused"
                )

            optimizer = torch.optim.AdamW(
                [bank.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
            )
            training_log: List[Dict[str, Any]] = []
            arm_slug = f"width_{width:02d}_{budget_regime}"
            log_path = out_dir / f"actuator_{arm_slug}_training_log.jsonl"
            with log_path.open("x", encoding="utf-8") as log_handle:
                for step in range(1, int(args.actuator_feasibility_steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_margin = 0.0
                    embedding_writer.enabled = writer_present
                    for indices in positive_indices_by_record:
                        current_new, current_true = differentiable_group_nlls(
                            positive_instances, indices
                        )
                        record_loss, _pieces = (
                            neuron_core.actuator_positive_margin_objective(
                                current_true - current_new,
                                margin_floor=float(args.forget_margin),
                                tail_k=int(args.actuator_tail_k),
                            )
                        )
                        (record_loss / len(records)).backward()
                        accumulated_margin += (
                            float(record_loss.detach()) / len(records)
                        )
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [bank.down_delta], float(args.grad_clip)
                        )
                    optimizer.step()
                    bank.clamp_down_relative_(cap)
                    if budget_regime == "matched_width4_group_budget_control":
                        bank.clamp_group_frobenius_(
                            width4_absolute_group_budgets_tensor
                        )
                    relative = bank.down_relative_norms().float().cpu()
                    group_norms = bank.group_frobenius_norms().float().cpu()
                    saturated = relative >= cap - 1e-6
                    row = {
                        "step": step,
                        "positive_margin_loss": accumulated_margin,
                        "actuator_width": width,
                        "budget_regime": budget_regime,
                        "records_per_optimizer_update": len(records),
                        "positive_contexts_per_optimizer_update": len(
                            positive_instances
                        ),
                        "down_max_relative_norm": float(relative.max()),
                        "down_median_relative_norm": float(relative.median()),
                        "down_mean_relative_norm": float(relative.mean()),
                        "down_saturated_columns": int(saturated.sum()),
                        "group_frobenius_max": float(group_norms.max()),
                        "group_to_width4_budget_max": float(
                            (
                                group_norms
                                / width4_absolute_group_budgets_tensor
                            ).max()
                        ),
                    }
                    training_log.append(row)
                    log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    log_handle.flush()
                    if (
                        step == 1
                        or step % max(1, int(args.check_every)) == 0
                        or step == int(args.actuator_feasibility_steps)
                    ):
                        print(
                            f"  width {width:>2} {budget_regime} step {step:>3}: "
                            f"margin {accumulated_margin:.4f}, down max "
                            f"{row['down_max_relative_norm']:.4f}, saturated "
                            f"{row['down_saturated_columns']}/{len(flat_actuator_ids)}"
                        )
            del optimizer
            final_audit = full_bank_audit(
                bank,
                phase="final_fresh_full_context_audit",
                optimizer_step=int(args.actuator_feasibility_steps),
                width=width,
                budget_regime=budget_regime,
            )
            detector_unchanged = bool(
                _tensor_digest(editor.gate_delta) == frozen_gate_delta_sha256
                and _tensor_digest(editor.up_delta) == frozen_up_delta_sha256
            )
            if not detector_unchanged:
                bank.remove()
                raise RuntimeError("actuator width sweep changed frozen detector tensors")
            positive_reachable = bool(final_audit["positive_passed"])
            writer_off_passed = bool(
                final_audit["writer_off_preservation_passed"]
            )
            fitted_digest = _tensor_digest(bank.down_delta)
            best_step = min(
                training_log, key=lambda row: row["positive_margin_loss"]
            )
            report = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_actuator_width_feasibility_fit",
                "protocol": PROTOCOL,
                "architecture": str(args.actuator_architecture),
                "actuator_width": width,
                "actuator_columns": len(flat_actuator_ids),
                "per_column_relative_cap": cap,
                "budget_regime": budget_regime,
                "selection_driving_arm": budget_regime == "native_per_column_cap",
                "initial_down_delta": "exact_zero",
                "optimizer_state": "fresh_independent_adamw",
                "optimizer_steps_expected": int(args.actuator_feasibility_steps),
                "optimizer_steps_recorded": len(training_log),
                "objective": (
                    "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
                ),
                "forget_margin": float(args.forget_margin),
                "learning_rate": float(args.actuator_lr),
                "best_logged_step": best_step,
                "incremental_training_log": {
                    "path": str(log_path),
                    "sha256": compositional_method.sha256_file(log_path),
                    "rows": len(training_log),
                },
                "final_audit": final_audit,
                "positive_reachable": positive_reachable,
                "writer_off_structural_selectivity_passed": writer_off_passed,
                "joint_positive_and_structural_selectivity_passed": bool(
                    positive_reachable and writer_off_passed
                ),
                "frozen_detector_tensors_unchanged": detector_unchanged,
                "fitted_down_delta_sha256_before_discard": fitted_digest,
                "full_preservation_objective_started": False,
                "checkpoint_saved": False,
                "official_evaluation_allowed": False,
                "official_evaluation_prompts_seen": 0,
            }
            bank.zero_()
            if int(torch.count_nonzero(bank.down_delta)) != 0:
                bank.remove()
                raise RuntimeError("actuator-bank fitted weights were not discarded")
            report["fitted_weights_discarded"] = True
            report["down_delta_after_discard"] = "bit_exact_zero"
            report_path = out_dir / f"actuator_{arm_slug}_feasibility.json"
            gagd.write_json(report_path, report)
            arm_reports.append(report)
            arm_artifacts.append(
                {
                    "actuator_width": width,
                    "actuator_columns": len(flat_actuator_ids),
                    "budget_regime": budget_regime,
                    "selection_driving_arm": report["selection_driving_arm"],
                    "path": str(report_path),
                    "sha256": compositional_method.sha256_file(report_path),
                    "positive_reachable": positive_reachable,
                    "writer_off_structural_selectivity_passed": writer_off_passed,
                    "direct_failures": int(final_audit["direct_failures"]),
                    "positive_failures": int(final_audit["positive_failures"]),
                    "minimum_margin": float(final_audit["minimum_margin"]),
                    "median_margin": float(final_audit["median_margin"]),
                    "writer_off_nll_abs_max": float(
                        final_audit["writer_off_nll_abs_max"]
                    ),
                    "saturated_columns": int(
                        final_audit["norms"]["saturated_columns"]
                    ),
                }
            )
            bank.remove()

        native_arms = sorted(
            (
                row
                for row in arm_artifacts
                if row["budget_regime"] == "native_per_column_cap"
            ),
            key=lambda row: int(row["actuator_width"]),
        )
        positive_widths = [
            int(row["actuator_width"])
            for row in native_arms
            if row["positive_reachable"]
        ]
        mechanism_widths = [
            int(row["actuator_width"])
            for row in native_arms
            if row["positive_reachable"]
            and row["writer_off_structural_selectivity_passed"]
        ]
        selected_positive_width = min(positive_widths) if positive_widths else None
        selected_mechanism_width = (
            min(mechanism_widths) if mechanism_widths else None
        )
        matched_controls = [
            row
            for row in arm_artifacts
            if row["budget_regime"] == "matched_width4_group_budget_control"
        ]
        conclusion = (
            "separate_actuator_width_sweep_passed"
            if selected_mechanism_width is not None
            else (
                "positive_reachability_without_writer_off_selectivity"
                if selected_positive_width is not None
                else "separate_actuator_width_16_not_positive_reachable_at_cap_1_5"
            )
        )
        sweep = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_5_5_actuator_width_feasibility",
            "protocol": PROTOCOL,
            "mode": "training_only",
            "architecture": str(args.actuator_architecture),
            "frozen_detector_protocol": FROZEN_V3_5_4_PROTOCOL,
            "frozen_detector_replay": {
                "path": str(detector_replay_path),
                "sha256": compositional_method.sha256_file(detector_replay_path),
                "passed": True,
            },
            "actuator_widths_preregistered": widths,
            "arms_preregistered": [
                {"actuator_width": width, "budget_regime": regime}
                for width, regime in arm_specs
            ],
            "arms_completed": [
                {
                    "actuator_width": int(row["actuator_width"]),
                    "budget_regime": str(row["budget_regime"]),
                }
                for row in arm_artifacts
            ],
            "complete": len(arm_artifacts) == len(arm_specs),
            "native_per_column_cap": cap,
            "native_selection_rule": (
                "smallest width with direct_failures == 0 and positive_failures "
                "== 0; matched-budget controls are diagnostic only"
            ),
            "selected_smallest_positive_reachable_width": selected_positive_width,
            "selected_smallest_mechanism_ready_width": selected_mechanism_width,
            "positive_reachability_passed": selected_positive_width is not None,
            "structural_selectivity_passed": selected_mechanism_width is not None,
            "mechanism_readiness_passed": selected_mechanism_width is not None,
            "matched_total_norm_control": {
                "reference": "width_4_native_per_column_cap_absolute_group_budget",
                "absolute_group_budgets": width4_absolute_group_budgets,
                "controls": matched_controls,
                "used_for_width_selection": False,
            },
            "actuator_selection": {
                "path": str(actuator_selection_path),
                "sha256": compositional_method.sha256_file(
                    actuator_selection_path
                ),
                "detector_actuator_disjoint": True,
                "nested_prefixes": True,
            },
            "zero_identity_artifacts": identity_artifacts,
            "arm_artifacts": arm_artifacts,
            "all_fitted_weights_discarded": all(
                bool(report["fitted_weights_discarded"])
                for report in arm_reports
            ),
            "frozen_detector_tensors_unchanged": all(
                bool(report["frozen_detector_tensors_unchanged"])
                for report in arm_reports
            ),
            "conclusion": conclusion,
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
        }
        sweep_path = out_dir / "v3_5_5_actuator_width_feasibility.json"
        gagd.write_json(sweep_path, sweep)
        completion = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_5_5_training_only_completion",
            "protocol": PROTOCOL,
            "sweep_path": str(sweep_path),
            "sweep_sha256": compositional_method.sha256_file(sweep_path),
            "selected_smallest_positive_reachable_width": selected_positive_width,
            "selected_smallest_mechanism_ready_width": selected_mechanism_width,
            "positive_reachability_passed": bool(
                sweep["positive_reachability_passed"]
            ),
            "structural_selectivity_passed": bool(
                sweep["structural_selectivity_passed"]
            ),
            "mechanism_readiness_passed": bool(
                sweep["mechanism_readiness_passed"]
            ),
            "conclusion": conclusion,
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
        }
        completion_path = out_dir / "training_only_v3_5_5_completion.json"
        gagd.write_json(completion_path, completion)
        if not completion["mechanism_readiness_passed"]:
            gagd.write_json(
                out_dir / "training_rejection.json",
                {
                    **completion,
                    "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                    "stage": "separate_actuator_width_feasibility",
                    "reason": conclusion,
                },
            )
        editor.remove()
        embedding_writer.remove()
        if selected_mechanism_width is None:
            print(
                "no separate actuator width through 16 passed positive plus "
                "writer-off feasibility at cap 1.50; full training and official "
                "evaluation remain refused"
            )
        else:
            print(
                f"smallest mechanism-ready actuator width: "
                f"{selected_mechanism_width}; all fitted weights discarded and "
                "official evaluation remains refused pending a separately "
                "registered full preservation run"
            )
        return

    if args.training_only_actuator_width_sweep:
        if (
            detector_selectivity is None
            or threshold_gate is None
            or frozen_v3_4_rejection is None
            or frozen_v3_5_1_forensics is None
            or frozen_v3_5_2_rejection is None
            or frozen_v3_5_3_rejection is None
        ):
            raise RuntimeError(
                "V3.5.4 feasibility lacks its gate and V3.4/V3.5.1/V3.5.2/"
                "V3.5.3 lineage audits"
            )
        caps = [float(cap) for cap in args.actuator_feasibility_caps]
        with torch.no_grad():
            editor.down_delta.zero_()
        initial_audit = full_actuator_audit(
            phase="norm_cap_sweep_initial", optimizer_step=0
        )
        initial_audit_path = out_dir / "isolated_zero_actuator_identity_audit.json"
        gagd.write_json(
            initial_audit_path,
            initial_audit,
        )
        identity_abs_max = max(
            float(initial_audit["writer_on_nll_abs_max"]),
            float(initial_audit["writer_off_nll_abs_max"]),
        )
        if identity_abs_max > 1e-6:
            gagd.write_json(
                out_dir / "training_rejection.json",
                {
                    "schema_version": 1,
                    "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                    "protocol": PROTOCOL,
                    "stage": "isolated_zero_actuator_identity",
                    "reason": "isolated_branch_was_not_behaviorally_identical_at_zero",
                    "identity_nll_abs_max": identity_abs_max,
                    "initial_audit": initial_audit,
                    "actuator_training_started": False,
                    "checkpoint_saved": False,
                    "official_evaluation_allowed": False,
                    "official_evaluation_prompts_seen": 0,
                },
            )
            editor.remove()
            embedding_writer.remove()
            raise SystemExit(
                "isolated residual failed its exact-zero identity audit; actuator "
                "training and official evaluation remain refused"
            )
        frozen_gate_delta_sha256 = _tensor_digest(editor.gate_delta)
        frozen_up_delta_sha256 = _tensor_digest(editor.up_delta)
        print("\nStage 2a: fixed-cap isolated threshold-residual feasibility")
        cap_reports: List[Dict[str, Any]] = []
        cap_artifacts: List[Dict[str, Any]] = []
        for cap in caps:
            with torch.no_grad():
                editor.down_delta.zero_()
            if int(torch.count_nonzero(editor.down_delta)) != 0:
                raise RuntimeError("cap-specific actuator did not start at exact zero")
            optimizer = torch.optim.AdamW(
                [editor.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
            )
            training_log: List[Dict[str, Any]] = []
            cap_slug = f"{cap:.2f}".replace(".", "p")
            log_path = out_dir / f"actuator_cap_{cap_slug}_training_log.jsonl"
            with log_path.open("x", encoding="utf-8") as log_handle:
                for step in range(1, int(args.actuator_feasibility_steps) + 1):
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_margin = 0.0
                    embedding_writer.enabled = writer_present
                    for indices in positive_indices_by_record:
                        current_new, current_true = differentiable_group_nlls(
                            positive_instances, indices
                        )
                        (
                            record_loss,
                            _pieces,
                        ) = neuron_core.actuator_positive_margin_objective(
                            current_true - current_new,
                            margin_floor=float(args.forget_margin),
                            tail_k=int(args.actuator_tail_k),
                        )
                        (record_loss / len(records)).backward()
                        accumulated_margin += float(record_loss.detach()) / len(records)
                    if float(args.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [editor.down_delta], float(args.grad_clip)
                        )
                    optimizer.step()
                    editor.clamp_down_relative_(cap)
                    geometry = down_norm_geometry(cap, include_per_neuron=False)
                    row = {
                        "step": step,
                        "positive_margin_loss": accumulated_margin,
                        "records_per_optimizer_update": len(records),
                        "positive_contexts_per_optimizer_update": len(
                            positive_instances
                        ),
                        "down_max_relative_norm": float(
                            geometry["relative_norm"]["max"]
                        ),
                        "down_median_relative_norm": float(
                            geometry["relative_norm"]["median"]
                        ),
                        "down_mean_relative_norm": float(
                            geometry["relative_norm"]["mean"]
                        ),
                        "down_saturated_columns": int(geometry["saturated_columns"]),
                    }
                    training_log.append(row)
                    log_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    log_handle.flush()
                    if (
                        step == 1
                        or step % max(1, int(args.check_every)) == 0
                        or step == int(args.actuator_feasibility_steps)
                    ):
                        print(
                            f"  cap {cap:.2f} step {step:>3}: "
                            f"margin {accumulated_margin:.4f}, "
                            f"down max {row['down_max_relative_norm']:.4f}, "
                            f"saturated {row['down_saturated_columns']}/"
                            f"{len(selected_neurons)}"
                        )
            del optimizer
            final_audit = full_actuator_audit(
                phase="norm_cap_sweep_final",
                optimizer_step=int(args.actuator_feasibility_steps),
            )
            geometry = down_norm_geometry(cap)
            residual_selectivity = actuator_residual_write_selectivity_audit()
            detector_tensors_unchanged = bool(
                _tensor_digest(editor.gate_delta) == frozen_gate_delta_sha256
                and _tensor_digest(editor.up_delta) == frozen_up_delta_sha256
            )
            if not detector_tensors_unchanged:
                raise RuntimeError("cap sweep changed the frozen detector tensors")
            positive_reachable = bool(
                final_audit["direct_failures"] == 0
                and final_audit["positive_failures"] == 0
                and float(geometry["relative_norm"]["max"]) <= cap + 1e-6
            )
            writer_off_structural_selectivity = bool(
                float(final_audit["writer_off_nll_abs_max"])
                <= float(args.actuator_writer_off_nll_tolerance) + 1e-6
            )
            fitted_digest = _tensor_digest(editor.down_delta)
            best_step = min(training_log, key=lambda row: row["positive_margin_loss"])
            report = {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_actuator_cap_feasibility_fit",
                "protocol": PROTOCOL,
                "architecture": str(args.actuator_architecture),
                "cap": cap,
                "initial_down_delta": "exact_zero",
                "optimizer_state": "fresh_independent_adamw",
                "optimizer_steps_expected": int(args.actuator_feasibility_steps),
                "optimizer_steps_recorded": len(training_log),
                "records_per_optimizer_update": len(records),
                "positive_contexts_per_optimizer_update": len(positive_instances),
                "objective": (
                    "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
                ),
                "forget_margin": float(args.forget_margin),
                "learning_rate": float(args.actuator_lr),
                "complete_training_log": training_log,
                "incremental_training_log": {
                    "path": str(log_path),
                    "sha256": compositional_method.sha256_file(log_path),
                    "rows": len(training_log),
                },
                "best_logged_step": best_step,
                "final_audit": final_audit,
                "down_norm_geometry": geometry,
                "actuator_residual_write_selectivity": residual_selectivity,
                "frozen_detector_tensors": {
                    "gate_delta_sha256": frozen_gate_delta_sha256,
                    "up_delta_sha256": frozen_up_delta_sha256,
                    "unchanged": detector_tensors_unchanged,
                },
                "positive_reachable": positive_reachable,
                "writer_off_structural_selectivity_passed": (
                    writer_off_structural_selectivity
                ),
                "joint_positive_and_structural_selectivity_passed": bool(
                    positive_reachable and writer_off_structural_selectivity
                ),
                "fitted_down_delta_sha256_before_discard": fitted_digest,
                "full_preservation_objective_started": False,
                "checkpoint_saved": False,
                "official_evaluation_allowed": False,
                "official_evaluation_prompts_seen": 0,
            }
            with torch.no_grad():
                editor.down_delta.zero_()
            if int(torch.count_nonzero(editor.down_delta)) != 0:
                raise RuntimeError("cap-specific fitted weights were not discarded")
            report["fitted_weights_discarded"] = True
            report["down_delta_after_discard"] = "bit_exact_zero"
            report_path = out_dir / f"actuator_cap_{cap_slug}_feasibility.json"
            gagd.write_json(report_path, report)
            cap_reports.append(report)
            cap_artifacts.append(
                {
                    "cap": cap,
                    "path": str(report_path),
                    "sha256": compositional_method.sha256_file(report_path),
                    "positive_reachable": positive_reachable,
                    "writer_off_structural_selectivity_passed": (
                        writer_off_structural_selectivity
                    ),
                    "direct_failures": int(final_audit["direct_failures"]),
                    "positive_failures": int(final_audit["positive_failures"]),
                    "minimum_margin": float(final_audit["minimum_margin"]),
                    "median_margin": float(final_audit["median_margin"]),
                    "writer_off_nll_abs_max": float(
                        final_audit["writer_off_nll_abs_max"]
                    ),
                    "down_max_relative_norm": float(geometry["relative_norm"]["max"]),
                    "down_median_relative_norm": float(
                        geometry["relative_norm"]["median"]
                    ),
                    "down_saturated_columns": int(geometry["saturated_columns"]),
                }
            )

        decision = neuron_core.actuator_cap_sweep_decision(cap_artifacts)
        selected_cap = decision["selected_smallest_positive_reachable_cap"]
        selected_structural_cap = decision[
            "smallest_jointly_structurally_selective_cap"
        ]
        sweep = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_5_4_balanced_multilabel_repair_feasibility",
            "protocol": PROTOCOL,
            "mode": "training_only",
            "architecture": str(args.actuator_architecture),
            "caps_preregistered": caps,
            "caps_completed": [float(row["cap"]) for row in cap_artifacts],
            "complete": len(cap_artifacts) == len(caps),
            "selection_rule": (
                "smallest tested cap with direct_failures == 0 and "
                "positive_failures == 0"
            ),
            "selected_smallest_positive_reachable_cap": selected_cap,
            "positive_reachability_passed": bool(
                decision["positive_reachability_passed"]
            ),
            "structural_selectivity_diagnostic": {
                "criterion": {
                    "positive_reachable": True,
                    "writer_off_nll_abs_max": float(
                        args.actuator_writer_off_nll_tolerance
                    ),
                },
                "smallest_jointly_passing_cap": selected_structural_cap,
                "passed": bool(decision["structural_selectivity_passed"]),
                "used_to_select_positive_reachability_cap": False,
            },
            "mechanism_readiness_passed": bool(
                decision["positive_reachability_passed"]
                and decision["structural_selectivity_passed"]
            ),
            "mechanism_readiness_rule": (
                "positive reachability plus exact writer-off NLL drift within "
                "0.05 behind the isolated threshold gate at cap 1.50"
            ),
            "isolated_threshold_gate": {
                "path": str(out_dir / "isolated_threshold_gate_report.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "isolated_threshold_gate_report.json"
                ),
                "passed": bool(threshold_gate["passed"]),
            },
            "multilabel_prompt_manifest": {
                "path": str(prompt_label_path),
                "sha256": compositional_method.sha256_file(prompt_label_path),
                "unique_prompts": int(prompt_label_report["unique_prompts"]),
                "role_overlap_prompts": int(
                    prompt_label_report["prompts_in_both_positive_and_negative_roles"]
                ),
                "same_record_conflicts": int(
                    prompt_label_report["same_record_positive_negative_conflicts"]
                ),
            },
            "detector_gradient_balance_audit": {
                "path": str(detector_gradient_balance_path),
                "sha256": compositional_method.sha256_file(
                    detector_gradient_balance_path
                ),
                "optimizer_step": int(
                    detector_gradient_balance_audit["optimizer_step"]
                ),
                "global_tail_optimization_weight": float(
                    detector_gradient_balance_audit[
                        "global_tail_optimization_weight"
                    ]
                ),
                "parameter_grad_buffers_mutated": bool(
                    detector_gradient_balance_audit[
                        "parameter_grad_buffers_mutated"
                    ]
                ),
                "optimizer_state_mutated": bool(
                    detector_gradient_balance_audit["optimizer_state_mutated"]
                ),
                "used_for_optimization": bool(
                    detector_gradient_balance_audit["used_for_optimization"]
                ),
            },
            "repaired_detector_selectivity_audit": {
                "path": str(out_dir / "repaired_detector_selectivity_audit.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "repaired_detector_selectivity_audit.json"
                ),
                "any_p10_below_heuristic_warning": bool(
                    detector_selectivity["any_p10_below_heuristic_warning"]
                ),
            },
            "isolated_zero_actuator_identity_audit": {
                "path": str(initial_audit_path),
                "sha256": compositional_method.sha256_file(initial_audit_path),
                "down_delta": "bit_exact_zero",
                "writer_on_nll_abs_max": float(initial_audit["writer_on_nll_abs_max"]),
                "writer_off_nll_abs_max": float(
                    initial_audit["writer_off_nll_abs_max"]
                ),
                "identity_nll_abs_max": identity_abs_max,
                "identity_passed": bool(identity_abs_max <= 1e-6),
                "writer_off_preservation_passed": bool(
                    initial_audit["writer_off_preservation_passed"]
                ),
            },
            "selected_base_down_zeroing_audit": {
                "path": str(out_dir / "selected_base_down_zeroing_audit.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "selected_base_down_zeroing_audit.json"
                ),
                "nll_abs_max": float(base_down_zeroing_audit["nll_abs_max"]),
                "restored_bit_exact": bool(
                    base_down_zeroing_audit["restored_bit_exact"]
                ),
            },
            "frozen_v3_4_rejection_import": {
                "path": str(out_dir / "frozen_v3_4_rejection_import.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "frozen_v3_4_rejection_import.json"
                ),
                "passed": bool(frozen_v3_4_rejection["passed"]),
            },
            "frozen_v3_5_1_forensics_import": {
                "path": str(out_dir / "frozen_v3_5_1_forensics_import.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "frozen_v3_5_1_forensics_import.json"
                ),
                "diagnosis": frozen_v3_5_1_forensics["diagnosis"],
                "collision": frozen_v3_5_1_forensics["collision"],
                "passed": bool(frozen_v3_5_1_forensics["passed"]),
            },
            "frozen_v3_5_2_rejection_import": {
                "path": str(out_dir / "frozen_v3_5_2_rejection_import.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "frozen_v3_5_2_rejection_import.json"
                ),
                "diagnosis": frozen_v3_5_2_rejection["diagnosis"],
                "duplicate_prompt_sha256": frozen_v3_5_2_rejection[
                    "duplicate_prompt_sha256"
                ],
                "writer_off_repair_succeeded": bool(
                    frozen_v3_5_2_rejection["writer_off_repair_succeeded"]
                ),
                "passed": bool(frozen_v3_5_2_rejection["passed"]),
            },
            "frozen_v3_5_3_rejection_import": {
                "path": str(out_dir / "frozen_v3_5_3_rejection_import.json"),
                "sha256": compositional_method.sha256_file(
                    out_dir / "frozen_v3_5_3_rejection_import.json"
                ),
                "diagnosis": frozen_v3_5_3_rejection["diagnosis"],
                "owner_gate_passed_records": int(
                    frozen_v3_5_3_rejection["owner_gate_passed_records"]
                ),
                "owner_gate_total_records": int(
                    frozen_v3_5_3_rejection["owner_gate_total_records"]
                ),
                "negative_and_writer_off_certificate_clean": bool(
                    frozen_v3_5_3_rejection[
                        "negative_and_writer_off_certificate_clean"
                    ]
                ),
                "passed": bool(frozen_v3_5_3_rejection["passed"]),
            },
            "initial_audit": initial_audit,
            "cap_artifacts": cap_artifacts,
            "conclusion": decision["conclusion"],
            "all_fitted_weights_discarded": all(
                bool(report["fitted_weights_discarded"]) for report in cap_reports
            ),
            "frozen_detector_tensors_unchanged": all(
                bool(report["frozen_detector_tensors"]["unchanged"])
                for report in cap_reports
            ),
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
        }
        sweep_path = out_dir / "v3_5_4_multilabel_actuator_feasibility.json"
        gagd.write_json(sweep_path, sweep)
        completion = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_5_4_training_only_completion",
            "protocol": PROTOCOL,
            "sweep_path": str(sweep_path),
            "sweep_sha256": compositional_method.sha256_file(sweep_path),
            "positive_reachability_passed": bool(sweep["positive_reachability_passed"]),
            "selected_smallest_positive_reachable_cap": selected_cap,
            "structural_selectivity_passed": bool(
                sweep["structural_selectivity_diagnostic"]["passed"]
            ),
            "mechanism_readiness_passed": bool(sweep["mechanism_readiness_passed"]),
            "conclusion": sweep["conclusion"],
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
        }
        gagd.write_json(out_dir / "training_only_v3_5_4_completion.json", completion)
        if not completion["mechanism_readiness_passed"]:
            gagd.write_json(
                out_dir / "training_rejection.json",
                {
                    **completion,
                    "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                    "stage": "isolated_threshold_actuator_feasibility",
                    "reason": (
                        "no_tested_cap_reached_all_positive_margins"
                        if selected_cap is None
                        else "positive_reachability_without_structural_writer_off_selectivity"
                    ),
                },
            )
        editor.remove()
        embedding_writer.remove()
        if selected_cap is None:
            print(
                "the isolated branch at cap 1.50 did not reach the positive certificate; "
                "full training and official evaluation remain refused"
            )
        elif selected_structural_cap is None:
            print(
                f"smallest positive-reachable cap: {selected_cap:.2f}, but no "
                "positive-reachable cap preserved structural writer-off behavior; "
                "full training and official evaluation remain refused"
            )
        else:
            print(
                f"isolated branch passed positive and writer-off feasibility at "
                f"cap {selected_cap:.2f}; all fitted weights discarded and official "
                "evaluation remains refused"
            )
        return

    # Fixed-budget positive-only reachability test under the unchanged cap.
    # Its learned down columns are always discarded before the full objective.
    with torch.no_grad():
        editor.down_delta.zero_()
    feasibility_optimizer = torch.optim.AdamW(
        [editor.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
    )
    feasibility_log: List[Dict[str, Any]] = []
    print("\nStage 2a: positive-only all-record actuator feasibility at unchanged cap")
    for step in range(1, int(args.actuator_feasibility_steps) + 1):
        feasibility_optimizer.zero_grad(set_to_none=True)
        accumulated_margin = 0.0
        embedding_writer.enabled = writer_present
        for indices in positive_indices_by_record:
            current_new, current_true = differentiable_group_nlls(
                positive_instances, indices
            )
            record_loss, _pieces = neuron_core.actuator_positive_margin_objective(
                current_true - current_new,
                margin_floor=float(args.forget_margin),
                tail_k=int(args.actuator_tail_k),
            )
            (record_loss / len(records)).backward()
            accumulated_margin += float(record_loss.detach()) / len(records)
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([editor.down_delta], float(args.grad_clip))
        feasibility_optimizer.step()
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        row = {
            "step": step,
            "positive_margin_loss": accumulated_margin,
            "records_per_optimizer_update": len(records),
            "positive_contexts_per_optimizer_update": len(positive_instances),
            "down_max_relative_norm": cap["down_max_relative_norm"],
        }
        feasibility_log.append(row)
        if (
            step == 1
            or step % max(1, int(args.check_every)) == 0
            or step == int(args.actuator_feasibility_steps)
        ):
            print(
                f"  feasibility step {step:>3}: margin {accumulated_margin:.4f}, "
                f"down cap {row['down_max_relative_norm']:.4f}"
            )
    del feasibility_optimizer
    feasibility_audit = full_actuator_audit(
        phase="positive_only_feasibility_final",
        optimizer_step=int(args.actuator_feasibility_steps),
    )
    feasibility_log_complete = len(feasibility_log) == int(
        args.actuator_feasibility_steps
    )
    feasibility_passed = bool(
        feasibility_log_complete
        and feasibility_audit["direct_failures"] == 0
        and feasibility_audit["positive_failures"] == 0
        and feasibility_audit["norms"]["down_max_relative_norm"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    feasibility_report = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_positive_only_actuator_feasibility",
        "protocol": PROTOCOL,
        "objective": ("equal_record_mean_plus_worst_two_squared_margin_shortfall_only"),
        "optimizer_steps_expected": int(args.actuator_feasibility_steps),
        "optimizer_steps_recorded": len(feasibility_log),
        "complete": feasibility_log_complete,
        "tail_k": int(args.actuator_tail_k),
        "relative_norm_cap": float(args.actuator_relative_cap),
        "initial_down_delta": "exact_zero",
        "optimization": actuator_optimization_metadata(),
        "learned_weights_used_to_initialize_full_training": False,
        "complete_training_log": feasibility_log,
        "final_audit": feasibility_audit,
        "passed": feasibility_passed,
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(
        out_dir / "actuator_positive_only_feasibility.json", feasibility_report
    )
    with torch.no_grad():
        editor.down_delta.zero_()
    if not feasibility_passed:
        rejection = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
            "stage": "positive_only_actuator_feasibility",
            "reason": "unchanged_cap_positive_only_feasibility_failed",
            "detector_gate": detector_gate,
            "actuator_positive_only_feasibility": feasibility_report,
            "full_actuator_training_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
            "next_action": (
                "Treat the unchanged-cap sparse down-column actuator as a failed "
                "fixed-budget feasibility test; do not tune on official probes."
            ),
        }
        gagd.write_json(out_dir / "training_rejection.json", rejection)
        editor.remove()
        embedding_writer.remove()
        raise SystemExit(
            "positive-only actuator feasibility failed at the unchanged cap; "
            "full actuator training and official evaluation are refused"
        )

    print("\nStage 2b: globally balanced sparse MLP suppression actuator")
    actuator_optimizer = torch.optim.AdamW(
        [editor.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
    )
    actuator_log: List[Dict[str, Any]] = []
    actuator_endpoint_reports: Dict[str, Dict[str, Any]] = {}
    actuator_endpoint_paths: Dict[str, Path] = {}
    protected_order = list(range(len(protected_for_kl)))
    for step in range(1, int(args.actuator_steps) + 1):
        actuator_optimizer.zero_grad(set_to_none=True)
        accumulated = {
            "margin": 0.0,
            "reference": 0.0,
            "negative_preservation": 0.0,
            "writer_off": 0.0,
            "protected_kl": 0.0,
        }
        for record_index in range(len(records)):
            positive_indices = positive_indices_by_record[record_index]
            negative_indices = negative_indices_by_record[record_index]
            embedding_writer.enabled = writer_present
            current_new, current_true = differentiable_group_nlls(
                positive_instances, positive_indices
            )
            margin_loss, _ = neuron_core.actuator_positive_margin_objective(
                current_true - current_new,
                margin_floor=float(args.forget_margin),
                tail_k=int(args.actuator_tail_k),
            )
            reference_loss, _ = neuron_core.actuator_reference_regression_objective(
                current_new,
                pre_target_new[positive_indices].to(device),
                tolerance=float(args.reference_nll_tolerance),
                tail_k=int(args.actuator_tail_k),
            )
            writer_off_loss = margin_loss * 0.0
            if (
                writer_present
                and int(args.actuator_writer_off_every) > 0
                and step % int(args.actuator_writer_off_every) == 0
            ):
                embedding_writer.enabled = False
                off_new, off_true = differentiable_group_nlls(
                    positive_instances, positive_indices
                )
                writer_off_loss, _ = neuron_core.actuator_writer_off_objective(
                    off_new,
                    off_true,
                    pre_writer_off_new[positive_indices].to(device),
                    pre_writer_off_true[positive_indices].to(device),
                    tolerance=float(args.actuator_writer_off_nll_tolerance),
                    tail_k=int(args.actuator_tail_k),
                )
                embedding_writer.enabled = writer_present
            negative_new, negative_true = differentiable_group_nlls(
                negative_instances, negative_indices
            )
            (
                negative_preservation,
                _,
            ) = neuron_core.actuator_negative_preservation_objective(
                negative_new,
                negative_true,
                pre_negative_new[negative_indices].to(device),
                pre_negative_true[negative_indices].to(device),
                tail_k=int(args.actuator_tail_k),
            )
            record_total = (
                float(args.margin_weight) * margin_loss
                + float(args.reference_nll_weight) * reference_loss
                + negative_preservation
                + float(args.writer_off_nll_weight) * writer_off_loss
            ) / len(records)
            if not torch.isfinite(record_total):
                raise FloatingPointError(
                    f"non-finite record actuator loss at step {step}"
                )
            record_total.backward()
            accumulated["margin"] += float(margin_loss.detach()) / len(records)
            accumulated["reference"] += float(reference_loss.detach()) / len(records)
            accumulated["negative_preservation"] += float(
                negative_preservation.detach()
            ) / len(records)
            accumulated["writer_off"] += float(writer_off_loss.detach()) / len(records)

        protected_indices = actuator_rng.sample(
            protected_order,
            min(int(args.actuator_protected_batch), len(protected_order)),
        )
        for start in range(0, len(protected_indices), int(args.actuator_batch_size)):
            chunk_indices = protected_indices[
                start : start + int(args.actuator_batch_size)
            ]
            protected_batch = [protected_for_kl[index] for index in chunk_indices]
            _hidden, protected_logits = compositional_method.forward_last_hidden_logits(
                model, tok, protected_batch, device
            )
            chunk_kl = _topk_kl(
                protected_logits, protected_batch, writer_only_cache, device
            )
            chunk_scale = len(chunk_indices) / len(protected_indices)
            (float(args.protected_kl_weight) * chunk_scale * chunk_kl).backward()
            accumulated["protected_kl"] += chunk_scale * float(chunk_kl.detach())

        l2 = editor.down_delta.square().mean()
        (float(args.actuator_l2) * l2).backward()
        total_value = (
            float(args.margin_weight) * accumulated["margin"]
            + float(args.reference_nll_weight) * accumulated["reference"]
            + accumulated["negative_preservation"]
            + float(args.writer_off_nll_weight) * accumulated["writer_off"]
            + float(args.protected_kl_weight) * accumulated["protected_kl"]
            + float(args.actuator_l2) * float(l2.detach())
        )
        if not math.isfinite(total_value):
            raise FloatingPointError(f"non-finite actuator loss at step {step}")
        if step == int(args.actuator_steps):
            phase = "pre_update"
            actuator_endpoint_reports[phase] = full_actuator_audit(
                phase=phase, optimizer_step=step
            )
            actuator_endpoint_paths[phase] = (
                out_dir / f"actuator_step_{step}_pre_update_audit.json"
            )
            gagd.write_json(
                actuator_endpoint_paths[phase], actuator_endpoint_reports[phase]
            )
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([editor.down_delta], float(args.grad_clip))
        actuator_optimizer.step()
        if step == int(args.actuator_steps):
            phase = "post_adam"
            actuator_endpoint_reports[phase] = full_actuator_audit(
                phase=phase, optimizer_step=step
            )
            actuator_endpoint_paths[phase] = (
                out_dir / f"actuator_step_{step}_post_adam_audit.json"
            )
            gagd.write_json(
                actuator_endpoint_paths[phase], actuator_endpoint_reports[phase]
            )
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == int(args.actuator_steps):
            phase = "post_projection"
            actuator_endpoint_reports[phase] = full_actuator_audit(
                phase=phase, optimizer_step=step
            )
            actuator_endpoint_paths[phase] = (
                out_dir / f"actuator_step_{step}_post_projection_audit.json"
            )
            gagd.write_json(
                actuator_endpoint_paths[phase], actuator_endpoint_reports[phase]
            )
        row = {
            "step": step,
            "loss": total_value,
            **accumulated,
            "l2": float(l2.detach()),
            "records_per_optimizer_update": len(records),
            "positive_contexts_per_optimizer_update": len(positive_instances),
            "negative_contexts_per_optimizer_update": len(negative_instances),
            "writer_off_contexts_per_optimizer_update": (
                len(positive_instances) if writer_present else 0
            ),
            "protected_contexts_per_optimizer_update": len(protected_indices),
            "down_max_relative_norm": cap["down_max_relative_norm"],
            "loss_measurement_phase": "pre_update",
            "norm_measurement_phase": "post_projection",
        }
        actuator_log.append(row)
        if (
            step == 1
            or step % max(1, int(args.check_every)) == 0
            or step == int(args.actuator_steps)
        ):
            audit = full_actuator_audit(phase="training_check", optimizer_step=step)
            print(
                f"  step {step:>3}: loss {row['loss']:.4f}, "
                f"margin {row['margin']:.4f}, off {row['writer_off']:.4f}; "
                f"direct fail {audit['direct_failures']}, "
                f"all-positive fail {audit['positive_failures']}, "
                f"min margin {audit['minimum_margin']:+.4f}"
            )
    del actuator_optimizer
    actuator_training_log_report = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_complete_actuator_training_log",
        "protocol": PROTOCOL,
        "revision": "v3.3",
        "protected_sampling_seed": actuator_rng_seed,
        "optimization": actuator_optimization_metadata(),
        "optimizer_steps_expected": int(args.actuator_steps),
        "optimizer_steps_recorded": len(actuator_log),
        "complete": len(actuator_log) == int(args.actuator_steps),
        "records": actuator_log,
        "official_evaluation_prompts_seen": 0,
    }
    actuator_training_log_path = out_dir / "actuator_training_log.json"
    gagd.write_json(actuator_training_log_path, actuator_training_log_report)
    final_actuator_audit = full_actuator_audit(
        phase="final_fresh_full_context_audit",
        optimizer_step=int(args.actuator_steps),
    )
    final_actuator_audit_path = out_dir / "actuator_final_audit.json"
    gagd.write_json(final_actuator_audit_path, final_actuator_audit)
    endpoint_replay = compare_actuator_audits(
        actuator_endpoint_reports["post_projection"],
        final_actuator_audit,
        abs_tolerance=1e-6,
    )
    actuator_endpoint_audit = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_actuator_endpoint_audit",
        "protocol": PROTOCOL,
        "optimizer_step": int(args.actuator_steps),
        "artifacts": {
            phase: {
                "path": str(actuator_endpoint_paths[phase]),
                "sha256": compositional_method.sha256_file(
                    actuator_endpoint_paths[phase]
                ),
            }
            for phase in ("pre_update", "post_adam", "post_projection")
        },
        "final_fresh_full_context_audit": {
            "path": str(final_actuator_audit_path),
            "sha256": compositional_method.sha256_file(final_actuator_audit_path),
        },
        "complete_training_log": {
            "path": str(actuator_training_log_path),
            "sha256": compositional_method.sha256_file(actuator_training_log_path),
            "complete": bool(actuator_training_log_report["complete"]),
        },
        "post_projection_matches_final": bool(endpoint_replay["passed"]),
        "post_projection_final_replay": endpoint_replay,
        "official_evaluation_prompts_seen": 0,
    }
    actuator_endpoint_audit["complete"] = bool(
        actuator_training_log_report["complete"]
        and actuator_endpoint_audit["post_projection_matches_final"]
    )
    if not actuator_endpoint_audit["complete"]:
        raise RuntimeError("actuator endpoint audit is incomplete or inconsistent")
    gagd.write_json(out_dir / "actuator_endpoint_audit.json", actuator_endpoint_audit)

    embedding_writer.enabled = writer_present
    editor.enabled = True
    editor.write_enabled = True

    # Training-only causal ablations.  Every configuration uses the same fixed
    # learned tensors; none is selected using official evaluation probes.
    def evaluate_ablation(writer: bool, decoder: bool) -> torch.Tensor:
        embedding_writer.enabled = bool(writer and writer_present)
        editor.enabled = bool(decoder)
        return compositional_method.evaluate_instance_margins(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )

    full_hook_margins = evaluate_ablation(writer_present, True)
    embedding_only_margins = evaluate_ablation(writer_present, False)
    neuron_only_margins = evaluate_ablation(False, True)
    reconstructed_base_margins = evaluate_ablation(False, False)
    embedding_writer.enabled = writer_present
    editor.enabled = True
    full_direct_failures, full_positive_failures = _failure_counts(
        full_hook_margins, direct_flags, float(args.forget_margin)
    )
    embedding_only_direct, embedding_only_positive = _failure_counts(
        embedding_only_margins, direct_flags, float(args.forget_margin)
    )
    neuron_only_direct, neuron_only_positive = _failure_counts(
        neuron_only_margins, direct_flags, float(args.forget_margin)
    )
    base_direct, base_positive = _failure_counts(
        reconstructed_base_margins, direct_flags, float(args.forget_margin)
    )
    causal_ablation = {
        "kind": "locked_training_only_within_checkpoint_component_intervention",
        "interpretation_boundary": (
            "Shows which components the fitted checkpoint relies on; it does not "
            "establish architectural necessity relative to an independently "
            "optimized no-writer sparse-MLP control."
        ),
        "writer_mode": str(args.writer_mode),
        "configurations": {
            "full_embedding_plus_neuron": {
                "direct_failures": full_direct_failures,
                "positive_failures": full_positive_failures,
                "minimum_margin": float(full_hook_margins.min()),
            },
            "embedding_only": {
                "direct_failures": embedding_only_direct,
                "positive_failures": embedding_only_positive,
                "minimum_margin": float(embedding_only_margins.min()),
            },
            "neuron_only": {
                "direct_failures": neuron_only_direct,
                "positive_failures": neuron_only_positive,
                "minimum_margin": float(neuron_only_margins.min()),
            },
            "reconstructed_base": {
                "direct_failures": base_direct,
                "positive_failures": base_positive,
                "minimum_margin": float(reconstructed_base_margins.min()),
            },
        },
        "writer_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - neuron_only_margins]
        ),
        "decoder_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - embedding_only_margins]
        ),
        "writer_necessity_applicable": writer_present,
        "writer_is_necessary": bool(
            writer_present
            and neuron_only_direct
            >= math.ceil(
                float(args.min_writer_necessary_direct_fraction) * len(records)
            )
        ),
        "decoder_is_necessary": bool(
            embedding_only_direct
            >= math.ceil(
                float(args.min_decoder_necessary_direct_fraction) * len(records)
            )
        ),
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "causal_component_ablation.json", causal_ablation)

    print("\nMaterialization: ordinary embedding and SwiGLU weights")
    editor.remove()
    embedding_writer.remove()
    input_index = torch.tensor(
        selected_embedding_rows, dtype=torch.long, device=input_layer.weight.device
    )
    base_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    directional.materialize_input_delta(
        input_layer, selected_embedding_rows, applied_embedding_delta
    )
    edited_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    base_neuron_weights = neuron_core.SparseNeuronWeights(
        editor.base_gate_rows.detach().to(mlp.gate_proj.weight.dtype),
        editor.base_up_rows.detach().to(mlp.up_proj.weight.dtype),
        editor.base_down_columns.detach().to(mlp.down_proj.weight.dtype),
    )
    edited_neuron_weights = editor.materialize(mlp)
    (
        materialized_target_new,
        materialized_target_true,
    ) = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    materialized_margins = materialized_target_true - materialized_target_new
    materialized_direct_failures, materialized_positive_failures = _failure_counts(
        materialized_margins, direct_flags, float(args.forget_margin)
    )
    hook_materialization_drift = float(
        (materialized_margins - full_hook_margins).abs().max()
    )
    reference_drift = materialized_target_new - pre_target_new
    reference_regression_max = max(0.0, float(reference_drift.max()))
    materialized_writer_off_nll_abs_max = 0.0
    if writer_present:
        _replace_embedding_rows(input_layer, selected_embedding_rows, base_input_rows)
        try:
            (
                materialized_writer_off_new,
                materialized_writer_off_true,
            ) = compositional_method.evaluate_instance_nlls(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
        finally:
            _replace_embedding_rows(
                input_layer, selected_embedding_rows, edited_input_rows
            )
        materialized_writer_off_nll_abs_max = max(
            float((materialized_writer_off_new - pre_writer_off_new).abs().max()),
            float((materialized_writer_off_true - pre_writer_off_true).abs().max()),
        )
    output_head_digest_after = _tensor_digest(output_layer.weight)
    output_head_unchanged = output_head_digest_before == output_head_digest_after
    if not output_head_unchanged:
        raise RuntimeError("LM head changed despite the architectural invariant")

    norm_report = editor.relative_norm_report()
    model_parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    embedding_edited_scalar_count = (
        int(edited_input_rows.numel()) if writer_present else 0
    )
    neuron_edited_scalar_count = int(
        edited_neuron_weights.gate_rows.numel()
        + edited_neuron_weights.up_rows.numel()
        + edited_neuron_weights.down_columns.numel()
    )
    total_edited_scalar_count = (
        embedding_edited_scalar_count + neuron_edited_scalar_count
    )
    norm_cap_passed = bool(
        norm_report["gate_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["up_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["down_max_relative_norm"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    detector_policy_passed = bool(
        detector_gate["passed"] or args.gate_policy == "report"
    )
    writer_requirement_passed = bool(
        not args.require_writer_necessity or causal_ablation["writer_is_necessary"]
    )
    writer_off_preservation_passed = bool(
        not writer_present
        or materialized_writer_off_nll_abs_max
        <= float(args.actuator_writer_off_nll_tolerance) + 1e-6
    )
    frozen_detector_lineage_passed = bool(
        str(args.detector_initialization) != "frozen_v3_2"
        or (
            frozen_detector_import is not None
            and frozen_detector_import.get("passed") is True
            and frozen_detector_replay is not None
            and frozen_detector_replay.get("passed") is True
        )
    )
    training_passed = bool(
        feasibility_passed
        and materialized_direct_failures == 0
        and materialized_positive_failures == 0
        and reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
        and writer_off_preservation_passed
        and norm_cap_passed
        and writer_requirement_passed
        and causal_ablation["decoder_is_necessary"]
        and detector_policy_passed
        and endpoint_audit["complete"]
        and actuator_endpoint_audit["complete"]
        and final_actuator_audit["passed"]
        and frozen_detector_lineage_passed
        and output_head_unchanged
        and hook_materialization_drift
        <= float(args.hook_materialization_tolerance) + 1e-6
    )

    checkpoint_path = out_dir / "checkpoint"
    rejected_control_checkpoint_saved = bool(
        args.save_checkpoint
        and args.save_rejected_checkpoint
        and not training_passed
        and not writer_present
    )
    checkpoint_saved = bool(
        args.save_checkpoint and (training_passed or rejected_control_checkpoint_saved)
    )
    if checkpoint_saved:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tok.save_pretrained(checkpoint_path)

    actual_input_delta = (
        edited_input_rows.detach().float().cpu()
        - base_input_rows.detach().float().cpu()
    )
    state_path = out_dir / "embedding_keyed_neuron_state.pt"
    torch.save(
        {
            "schema_version": 5,
            "method": METHOD,
            "protocol": PROTOCOL,
            "seed": int(args.seed),
            "experiment_label": str(args.experiment_label),
            "writer_mode": str(args.writer_mode),
            "case_ids": case_ids,
            "layer": int(args.neuron_layer),
            "selected_embedding_rows": selected_embedding_rows,
            "base_selected_embedding_rows": base_input_rows.detach().cpu(),
            "edited_selected_embedding_rows": edited_input_rows.detach().cpu(),
            "actual_embedding_delta": actual_input_delta,
            "selected_neurons": selected_neurons,
            "ownership": ownership,
            "selected_neuron_ownership_jq_compact_sha256": ownership_sha256,
            "flat_signs": flat_signs_cpu,
            "base_neuron_weights": {
                "gate_rows": base_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": base_neuron_weights.up_rows.detach().cpu(),
                "down_columns": base_neuron_weights.down_columns.detach().cpu(),
            },
            "edited_neuron_weights": {
                "gate_rows": edited_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": edited_neuron_weights.up_rows.detach().cpu(),
                "down_columns": edited_neuron_weights.down_columns.detach().cpu(),
            },
            "gate_delta": editor.gate_delta.detach().cpu(),
            "up_delta": editor.up_delta.detach().cpu(),
            "down_delta": editor.down_delta.detach().cpu(),
            "detector_positive_floor": float(args.detector_positive_floor),
            "detector_off_abs_max": float(args.detector_off_abs_max),
            "detector_training_positive_floor": float(
                args.detector_training_positive_floor
            ),
            "detector_training_off_abs_max": float(args.detector_training_off_abs_max),
            "detector_certificate_abs_tolerance": float(
                args.detector_certificate_abs_tolerance
            ),
            "detector_response_mode": str(args.detector_response_mode),
            "detector_training_revision": detector_optimization_metadata()["revision"],
            "detector_initialization": str(args.detector_initialization),
            "detector_optimizer_steps_this_run": detector_optimizer_steps_this_run,
            "frozen_v3_2_detector_import": frozen_detector_import,
            "detector_tail_k": int(args.detector_tail_k),
            "detector_update_coverage": "all_records_accumulated",
            "actuator_training_revision": "v3.3",
            "actuator_feasibility_steps": int(args.actuator_feasibility_steps),
            "actuator_steps": int(args.actuator_steps),
            "actuator_tail_k": int(args.actuator_tail_k),
            "actuator_update_coverage": "all_records_all_contexts_accumulated",
            "actuator_writer_off_nll_tolerance": float(
                args.actuator_writer_off_nll_tolerance
            ),
            "training_audit_artifacts": {
                "detector_gate_report_sha256": compositional_method.sha256_file(
                    detector_gate_path
                ),
                "detector_endpoint_audit_sha256": compositional_method.sha256_file(
                    out_dir / "detector_endpoint_audit.json"
                ),
                "actuator_positive_only_feasibility_sha256": (
                    compositional_method.sha256_file(
                        out_dir / "actuator_positive_only_feasibility.json"
                    )
                ),
                "actuator_training_log_sha256": compositional_method.sha256_file(
                    actuator_training_log_path
                ),
                "actuator_final_audit_sha256": compositional_method.sha256_file(
                    final_actuator_audit_path
                ),
                "actuator_endpoint_audit_sha256": compositional_method.sha256_file(
                    out_dir / "actuator_endpoint_audit.json"
                ),
            },
            "writer_preflight_amplitude_threshold": float(
                args.writer_preflight_amplitude_threshold
            ),
            "writer_configuration": {
                "row_norm_cap": (
                    float(stage1_state["row_norm_cap"])
                    if stage1_state.get("row_norm_cap") is not None
                    else None
                ),
                "row_norm_cap_frequency_alpha": (
                    float(stage1_state["row_norm_cap_frequency_alpha"])
                    if stage1_state.get("row_norm_cap_frequency_alpha") is not None
                    else None
                ),
                "max_subject_token_frequency": (
                    int(stage1_state["max_subject_token_frequency"])
                    if stage1_state.get("max_subject_token_frequency") is not None
                    else None
                ),
            },
            "source_stage1_state_sha256": compositional_method.sha256_file(stage1_path),
            "detector_relative_cap": float(args.detector_relative_cap),
            "actuator_relative_cap": float(args.actuator_relative_cap),
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "context_manifest_sha256": compositional_method.sha256_file(context_path),
            "training_visible_sha256": compositional_method.sha256_file(visible_path),
            "output_head_sha256": output_head_digest_after,
        },
        state_path,
    )

    summary = {
        "schema_version": 5,
        "method": (
            METHOD
            if writer_present
            else "Matched Retrained Sparse-MLP Conditional Suppression Control"
        ),
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "experiment_label": str(args.experiment_label),
        "writer_mode": str(args.writer_mode),
        "forget_num": len(records),
        "model_path": str(Path(args.model_path).resolve()),
        "architecture": {
            "tokenizer_expanded": False,
            "new_token_added": False,
            "runtime_string_matcher": False,
            "external_router": False,
            "retrieval_cache": False,
            "sidecar": False,
            "adapter": False,
            "lora": False,
            "logit_bias": False,
            "lm_head_edited": False,
            "lm_head_sha256_before": output_head_digest_before,
            "lm_head_sha256_after": output_head_digest_after,
            "input_embedding_rows_loaded_from_frozen_writer": len(
                selected_embedding_rows
            ),
            "input_embedding_rows_edited": (
                len(selected_embedding_rows) if writer_present else 0
            ),
            "mlp_layer": int(args.neuron_layer),
            "selected_existing_mlp_neurons": len(selected_neurons),
            "neurons_per_record": int(args.neurons_per_record),
            "base_model_parameter_count": model_parameter_count,
            "embedding_edited_scalar_count": embedding_edited_scalar_count,
            "mlp_edited_scalar_count": neuron_edited_scalar_count,
            "total_edited_scalar_count": total_edited_scalar_count,
            "edited_parameter_fraction": (
                total_edited_scalar_count / max(1, model_parameter_count)
            ),
            "edited_mlp_tensors": [
                "explicit_internal_detector.gate_delta",
                "explicit_internal_detector.up_delta",
                "explicit_internal_residual.down_delta",
            ],
            "writer_mode": str(args.writer_mode),
            "writer": (
                "ordinary global sparse subject embedding-row deltas"
                if writer_present
                else "none; selected embedding rows remain bit-identical to Base"
            ),
            "context_composer": "frozen Transformer layers before the selected MLP",
            "detector": "record-owned sparse existing SwiGLU gate/up rows",
            "actuator": (
                "threshold-gated explicit sparse residual; original Base down "
                "columns untouched"
            ),
            "output_projection": "unchanged base LM head",
        },
        "optimization_budget": {
            "detector_response_mode": str(args.detector_response_mode),
            "dormant_fraction": float(args.dormant_fraction),
            "selection_stability_weight": float(args.selection_stability_weight),
            "selection_positive_contexts": int(args.selection_positive_contexts),
            "selection_negative_contexts": int(args.selection_negative_contexts),
            "detector_steps": int(args.detector_steps),
            "detector_initialization": str(args.detector_initialization),
            "detector_optimizer_steps_this_run": detector_optimizer_steps_this_run,
            "detector_optimizer_steps_source": (
                1000 if str(args.detector_initialization) == "frozen_v3_2" else 0
            ),
            "detector_lr": float(args.detector_lr),
            "detector_record_batch": int(args.detector_record_batch),
            "detector_record_batch_semantics": (
                "gradient_accumulation_microbatch_capacity"
            ),
            "detector_update_coverage": "all_records_accumulated",
            "detector_records_per_optimizer_update": len(records),
            "detector_microbatches_per_optimizer_update": detector_microbatches,
            "detector_record_exposures": int(args.detector_steps) * len(records),
            "detector_positive_contexts": str(args.detector_positive_contexts),
            "detector_negative_contexts": str(args.detector_negative_contexts),
            "detector_tail_k": int(args.detector_tail_k),
            "detector_positive_objective": (
                "active_label_equal_record_mean_plus_worst_k"
            ),
            "detector_negative_objective": (
                "source_owner_equal_record_mean_plus_worst_k"
            ),
            "detector_cross_objective": (
                "inactive_label_equal_record_mean_plus_worst_k"
            ),
            "detector_writer_off_objective": (
                "all_groups_equal_record_mean_plus_worst_k"
            ),
            "detector_gradient_normalization": (
                "equal_record_mean_plus_per_record_worst_k"
            ),
            "detector_global_tail_weight": float(args.detector_global_tail_weight),
            "detector_global_tail_terms": "diagnostic_only_not_optimized",
            "detector_cached_mlp_inputs": True,
            "detector_cached_mlp_inputs_canonicalized_by_exact_prompt": True,
            "detector_prompt_label_semantics": "canonical_exact_prompt_multilabel",
            "detector_positive_floor": float(args.detector_positive_floor),
            "detector_off_abs_max": float(args.detector_off_abs_max),
            "detector_training_positive_floor": float(
                args.detector_training_positive_floor
            ),
            "detector_training_off_abs_max": float(args.detector_training_off_abs_max),
            "detector_certificate_abs_tolerance": float(
                args.detector_certificate_abs_tolerance
            ),
            "detector_negative_weight": float(args.detector_negative_weight),
            "detector_cross_weight": float(args.detector_cross_weight),
            "detector_consistency_weight": float(args.detector_consistency_weight),
            "detector_l2": float(args.detector_l2),
            "detector_relative_cap": float(args.detector_relative_cap),
            "actuator_training_revision": "v3.5.4",
            "actuator_feasibility_steps": int(args.actuator_feasibility_steps),
            "actuator_feasibility_initialization": "exact_zero_down_delta",
            "actuator_feasibility_objective": (
                "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
            ),
            "actuator_feasibility_positive_exposures": (
                int(args.actuator_feasibility_steps) * len(positive_instances)
            ),
            "actuator_steps": int(args.actuator_steps),
            "actuator_lr": float(args.actuator_lr),
            "actuator_batch_size": int(args.actuator_batch_size),
            "actuator_batch_size_semantics": (
                "context_microbatch_capacity_inside_global_update"
            ),
            "actuator_protected_batch": int(args.actuator_protected_batch),
            "actuator_protected_sampling_seed": actuator_rng_seed,
            "actuator_update_coverage": "all_records_all_contexts_accumulated",
            "actuator_records_per_optimizer_update": len(records),
            "actuator_positive_contexts": str(args.actuator_positive_contexts),
            "actuator_negative_contexts": str(args.actuator_negative_contexts),
            "actuator_positive_contexts_per_optimizer_update": len(positive_instances),
            "actuator_negative_contexts_per_optimizer_update": len(negative_instances),
            "actuator_writer_off_contexts_per_optimizer_update": (
                len(positive_instances) if writer_present else 0
            ),
            "actuator_positive_context_exposures": (
                int(args.actuator_steps) * len(positive_instances)
            ),
            "actuator_negative_context_exposures": (
                int(args.actuator_steps) * len(negative_instances)
            ),
            "actuator_writer_off_context_exposures": (
                int(args.actuator_steps) * len(positive_instances)
                if writer_present
                else 0
            ),
            "actuator_protected_context_exposures": (
                int(args.actuator_steps)
                * min(int(args.actuator_protected_batch), len(protected_for_kl))
            ),
            "actuator_tail_k": int(args.actuator_tail_k),
            "actuator_positive_objective": (
                "equal_record_mean_plus_worst_k_squared_margin_shortfall"
            ),
            "actuator_reference_objective": (
                "equal_record_mean_plus_worst_k_squared_excess_above_tolerance"
            ),
            "actuator_negative_objective": (
                "equal_record_mean_plus_worst_k_squared_nll_drift"
            ),
            "actuator_writer_off_objective": (
                "equal_record_mean_plus_worst_k_squared_abs_nll_drift_excess"
            ),
            "actuator_gradient_normalization": (
                "equal_record_mean_plus_global_protected_prompt_mean"
            ),
            "actuator_gradient_clip_frequency": "once_per_optimizer_update",
            "actuator_norm_projection_frequency": "once_per_optimizer_update",
            "actuator_complete_training_log_required": True,
            "actuator_endpoint_audit_phases": [
                "pre_update",
                "post_adam",
                "post_projection",
                "final_fresh_full_context_audit",
            ],
            "actuator_writer_off_every": int(args.actuator_writer_off_every),
            "actuator_writer_off_nll_tolerance": float(
                args.actuator_writer_off_nll_tolerance
            ),
            "actuator_relative_cap": float(args.actuator_relative_cap),
            "actuator_l2": float(args.actuator_l2),
            "neurons_per_record": int(args.neurons_per_record),
            "selected_existing_mlp_neurons": len(selected_neurons),
            "mlp_layer": int(args.neuron_layer),
            "protected_prompt_count": len(protected_for_kl),
            "protected_kl_weight": float(args.protected_kl_weight),
            "margin_weight": float(args.margin_weight),
            "reference_nll_weight": float(args.reference_nll_weight),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "forget_margin": float(args.forget_margin),
            "grad_clip": float(args.grad_clip),
            "kl_topk": int(args.kl_topk),
        },
        "data_firewall": firewall_receipt,
        "writer_preflight": writer_preflight,
        "writer_configuration": {
            "row_norm_cap": (
                float(stage1_state["row_norm_cap"])
                if stage1_state.get("row_norm_cap") is not None
                else None
            ),
            "row_norm_cap_frequency_alpha": (
                float(stage1_state["row_norm_cap_frequency_alpha"])
                if stage1_state.get("row_norm_cap_frequency_alpha") is not None
                else None
            ),
            "max_subject_token_frequency": (
                int(stage1_state["max_subject_token_frequency"])
                if stage1_state.get("max_subject_token_frequency") is not None
                else None
            ),
            "source_stage1_state_sha256": compositional_method.sha256_file(stage1_path),
        },
        "selection": selection_report,
        "detector": {
            "response_mode": str(args.detector_response_mode),
            "training_revision": detector_optimization_metadata()["revision"],
            "initialization": str(args.detector_initialization),
            "frozen_v3_2_import": frozen_detector_import,
            "frozen_v3_2_source_replay": frozen_detector_replay,
            "hidden_cache": detector_cache_report,
            "training_log": detector_log,
            "complete_training_log": detector_training_log_report,
            "endpoint_audit": endpoint_audit,
            "gate": detector_gate,
            "relative_norm_cap": float(args.detector_relative_cap),
        },
        "actuator": {
            "training_revision": "v3.3",
            "positive_only_feasibility": feasibility_report,
            "training_log": actuator_log,
            "complete_training_log": actuator_training_log_report,
            "endpoint_audit": actuator_endpoint_audit,
            "final_full_context_audit": final_actuator_audit,
            "relative_norm_cap": float(args.actuator_relative_cap),
            "norms": norm_report,
            "forget_margin": float(args.forget_margin),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "reference_nll_drift": _distribution(
                [float(value) for value in reference_drift]
            ),
            "reference_nll_regression_max": reference_regression_max,
            "writer_off_nll_tolerance": float(args.actuator_writer_off_nll_tolerance),
            "materialized_writer_off_nll_abs_max": (
                materialized_writer_off_nll_abs_max if writer_present else None
            ),
            "hook_to_materialized_margin_abs_max": hook_materialization_drift,
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "direct_failures": materialized_direct_failures,
            "training_safe_positive_failures": materialized_positive_failures,
            "minimum_margin": float(materialized_margins.min()),
        },
        "causal_component_ablation": causal_ablation,
        "acceptance": {
            "writer_preflight_passed": bool(writer_preflight["passed"]),
            "positive_only_actuator_feasibility_passed": feasibility_passed,
            "detector_endpoint_audit_complete": bool(endpoint_audit["complete"]),
            "actuator_endpoint_audit_complete": bool(
                actuator_endpoint_audit["complete"]
            ),
            "final_full_context_actuator_audit_passed": bool(
                final_actuator_audit["passed"]
            ),
            "frozen_detector_lineage_passed": frozen_detector_lineage_passed,
            "zero_direct_failures": materialized_direct_failures == 0,
            "zero_training_safe_positive_failures": materialized_positive_failures == 0,
            "reference_nll_regression_within_tolerance": bool(
                reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
            ),
            "writer_off_nll_preservation_applicable": writer_present,
            "writer_off_nll_preservation_within_tolerance": (
                writer_off_preservation_passed
            ),
            "detector_gate_passed": bool(detector_gate["passed"]),
            "detector_gate_policy_passed": detector_policy_passed,
            "writer_necessary": bool(causal_ablation["writer_is_necessary"]),
            "writer_necessity_required": bool(args.require_writer_necessity),
            "writer_requirement_passed": writer_requirement_passed,
            "decoder_necessary": bool(causal_ablation["decoder_is_necessary"]),
            "hard_relative_norm_caps_passed": norm_cap_passed,
            "lm_head_bit_identical": output_head_unchanged,
            "hook_materialization_within_tolerance": bool(
                hook_materialization_drift
                <= float(args.hook_materialization_tolerance) + 1e-6
            ),
            "checkpoint_saved": checkpoint_saved,
            "rejected_control_checkpoint_saved": rejected_control_checkpoint_saved,
            "passed": training_passed,
        },
        "checkpoint": str(checkpoint_path) if checkpoint_saved else None,
        "state": str(state_path),
        "claim_boundary": (
            "This checkpoint is a context-conditional suppression intervention, "
            "not evidence of weight-level knowledge removal. Training establishes "
            "only locked-context fit and, when applicable, within-checkpoint key "
            "dependence. Official paraphrase/locality tails and the independently "
            "retrained no-writer control remain mandatory post-freeze evidence."
        ),
    }
    gagd.write_json(out_dir / "embedding_keyed_neuron_summary.json", summary)

    print("\n" + "=" * 76)
    print(f"  materialized direct failures          : {materialized_direct_failures}")
    print(f"  materialized positive failures        : {materialized_positive_failures}")
    print(
        f"  minimum training margin               : {float(materialized_margins.min()):+.4f}"
    )
    print(f"  maximum reference NLL regression      : {reference_regression_max:.4f}")
    if writer_present:
        print(
            "  maximum writer-off NLL drift          : "
            f"{materialized_writer_off_nll_abs_max:.4f}"
        )
        print(
            "  within-checkpoint writer-off direct fails: "
            f"{neuron_only_direct}/{len(records)}"
        )
    else:
        print("  embedding writer                       : absent by construction")
    print(
        f"  without neuron decoder direct fails   : {embedding_only_direct}/{len(records)}"
    )
    print(f"  LM head bit-identical                 : {output_head_unchanged}")
    print(f"  selected existing neurons             : {len(selected_neurons)}")
    print("  evaluation probes opened              : 0")
    print("=" * 76)
    if not training_passed:
        gagd.write_json(
            out_dir / "training_rejection.json",
            {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
                "protocol": PROTOCOL,
                "stage": "materialized_actuator_acceptance",
                "reason": "locked_training_only_acceptance_failed",
                "detector_gate": detector_gate,
                "actuator_positive_only_feasibility": feasibility_report,
                "actuator_final_audit": final_actuator_audit,
                "actuator_endpoint_audit": actuator_endpoint_audit,
                "materialized": {
                    "direct_failures": materialized_direct_failures,
                    "positive_failures": materialized_positive_failures,
                    "minimum_margin": float(materialized_margins.min()),
                    "reference_nll_regression_max": reference_regression_max,
                    "writer_off_nll_abs_max": (
                        materialized_writer_off_nll_abs_max if writer_present else None
                    ),
                },
                "acceptance": summary["acceptance"],
                "checkpoint_saved": checkpoint_saved,
                "official_evaluation_allowed": rejected_control_checkpoint_saved,
                "official_evaluation_prompts_seen": 0,
            },
        )
        if rejected_control_checkpoint_saved:
            print(
                "fixed-budget no-writer control was rejected by training acceptance; "
                "its frozen checkpoint is retained for mandatory post-freeze evaluation"
            )
            print(f"checkpoint: {checkpoint_path}")
            print(f"summary: {out_dir / 'embedding_keyed_neuron_summary.json'}")
            return
        raise SystemExit(
            f"{args.writer_mode} sparse-neuron run failed its locked training-only "
            "acceptance; official evaluation is refused"
        )
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {out_dir / 'embedding_keyed_neuron_summary.json'}")


if __name__ == "__main__":
    main()
