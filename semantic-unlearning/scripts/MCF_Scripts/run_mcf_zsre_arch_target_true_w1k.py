#!/usr/bin/env python3
"""Run target-true-sensitive MCF with W1K utility-preserved Stage 1.

Locked development ablation:

* forget set: 50 MCF records (unchanged);
* target_true is sensitive; target_new is the Stage-2 reference;
* Stage 1 uses the canonical ZsRE-style sensitive GA + same-prompt KL;
* Stage 1 additionally preserves 1,000 external Wikipedia next-token
  distributions with KL(Base || current);
* Stage-1 LR is locked to 4e-5 for this W1K ablation;
* Stage 1 still restores Base everywhere except trained sensitive rows;
* Stage 2 is the materialization-safe MCF repair (solver margin 0.25, retry
  0.5 by default, final acceptance margin 0.05).

Official MCF paraphrases, neighborhoods, benchmark-retain examples and PPL text
remain evaluation-only.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mcf_zsre_arch_target_true as base  # noqa: E402
import run_mcf_zsre_arch_target_true_materialized as materialized  # noqa: E402
import sure_stage1_gagd_w1k as w1k_stage1  # noqa: E402


METHOD = "SURE-MCF-ZsRE-architecture-target_true-sensitive-W1K-LR4e-5"
PROTOCOL = "mcf_zsre_arch_target_true_sensitive_w1k_lr4e5_v1"
LOCKED_STAGE1_LR = 4e-5
LOCKED_UTILITY_SAMPLE_SIZE = 1_000


def _split_w1k_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--utility-wikipedia-dir", required=True)
    parser.add_argument(
        "--utility-sample-size", type=int, default=LOCKED_UTILITY_SAMPLE_SIZE
    )
    parser.add_argument("--utility-batch-size", type=int, default=4)
    parser.add_argument("--utility-cache-batch-size", type=int, default=8)
    parser.add_argument("--utility-max-length", type=int, default=128)
    parser.add_argument("--utility-seed", type=int, default=1)
    parser.add_argument("--utility-exclude-first", type=int, default=20)
    parser.add_argument("--utility-kl-weight", type=float, default=1.0)
    w1k, remaining = parser.parse_known_args(list(argv))
    if w1k.utility_sample_size != LOCKED_UTILITY_SAMPLE_SIZE:
        parser.error(
            f"this ablation locks --utility-sample-size to {LOCKED_UTILITY_SAMPLE_SIZE}"
        )
    if w1k.utility_batch_size <= 0 or w1k.utility_cache_batch_size <= 0:
        parser.error("utility batch sizes must be positive")
    if w1k.utility_max_length < 8:
        parser.error("utility-max-length must be at least 8")
    if w1k.utility_exclude_first < 0:
        parser.error("utility-exclude-first must be non-negative")
    if not math.isfinite(w1k.utility_kl_weight) or w1k.utility_kl_weight <= 0:
        parser.error("utility-kl-weight must be finite and positive")
    return w1k, remaining


def _patched_plan(
    original_plan,
    args: argparse.Namespace,
    paths: base.SeedPaths,
    seed: int,
    w1k: argparse.Namespace,
) -> List[base.Step]:
    steps = original_plan(args, paths, seed)
    if not steps:
        raise RuntimeError("canonical plan is empty")
    stage1 = steps[0]
    command = list(stage1.command)
    command[1] = str(SCRIPTS_DIR / "sure_stage1_gagd_w1k.py")
    command.extend(
        [
            "--utility-wikipedia-dir",
            str(Path(w1k.utility_wikipedia_dir).resolve()),
            "--utility-sample-size",
            str(w1k.utility_sample_size),
            "--utility-batch-size",
            str(w1k.utility_batch_size),
            "--utility-cache-batch-size",
            str(w1k.utility_cache_batch_size),
            "--utility-max-length",
            str(w1k.utility_max_length),
            "--utility-seed",
            str(w1k.utility_seed),
            "--utility-exclude-first",
            str(w1k.utility_exclude_first),
            "--utility-kl-weight",
            str(w1k.utility_kl_weight),
        ]
    )
    steps[0] = base.Step(
        "STAGE 1 — ZSRE GA/KL + W1K WIKIPEDIA UTILITY + TARGET_TRUE-ROW RESTORATION",
        command,
    )
    return steps


def _patched_architecture(
    original_effective_architecture,
    args: argparse.Namespace,
    w1k: argparse.Namespace,
) -> Dict[str, Any]:
    architecture = copy.deepcopy(original_effective_architecture(args))
    architecture["stage1"].update(
        {
            "learning_rate": LOCKED_STAGE1_LR,
            "external_utility": {
                "protocol": w1k_stage1.UTILITY_PROTOCOL,
                "source": str(Path(w1k.utility_wikipedia_dir).resolve()),
                "sample_size": int(w1k.utility_sample_size),
                "batch_size": int(w1k.utility_batch_size),
                "cache_batch_size": int(w1k.utility_cache_batch_size),
                "max_length": int(w1k.utility_max_length),
                "utility_seed": int(w1k.utility_seed),
                "exclude_first": int(w1k.utility_exclude_first),
                "kl_weight": float(w1k.utility_kl_weight),
                "loss": "KL(Base || current) on external Wikipedia next-token distributions",
            },
        }
    )
    return architecture


def _patched_validate(original_validate, paths: base.SeedPaths) -> Dict[str, Any]:
    result = original_validate(paths)
    import json

    stage1 = json.loads(
        (paths.stage1_dir / "config_used.json").read_text(encoding="utf-8")
    )
    if stage1.get("method") != w1k_stage1.METHOD:
        raise RuntimeError("Stage 1 did not use the W1K utility-preserved learner")
    if int(stage1.get("external_utility_sample_size", -1)) != LOCKED_UTILITY_SAMPLE_SIZE:
        raise RuntimeError("Stage 1 did not use exactly W1K utility contexts")
    if abs(float(stage1.get("emb_lm_lr", 0.0)) - LOCKED_STAGE1_LR) > 1e-12:
        raise RuntimeError("Stage 1 LR is not the locked 4e-5 value")
    result.update(
        {
            "stage1_learning_rate": float(stage1["emb_lm_lr"]),
            "stage1_external_utility_sample_size": int(
                stage1["external_utility_sample_size"]
            ),
            "stage1_external_utility_kl": stage1.get(
                "external_utility_post_restoration_kl"
            ),
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    w1k, forwarded = _split_w1k_args(raw)

    parsed = base.parse_args(forwarded)
    # This variant intentionally fixes the medium Stage-1 LR.  Stage 2 keeps
    # its own repair LR unchanged.
    parsed.stage1_lr = LOCKED_STAGE1_LR

    original_parse = base.parse_args
    original_plan = base.seed_command_plan
    original_arch = base.effective_architecture
    original_validate = base.validate_stage_contract
    original_method = base.METHOD
    original_protocol = base.PROTOCOL

    try:
        base.METHOD = METHOD
        base.PROTOCOL = PROTOCOL
        base.parse_args = lambda _argv=None: parsed
        base.seed_command_plan = lambda args, paths, seed: _patched_plan(
            original_plan, args, paths, seed, w1k
        )
        base.effective_architecture = lambda args: _patched_architecture(
            original_arch, args, w1k
        )
        base.validate_stage_contract = lambda paths: _patched_validate(
            original_validate, paths
        )

        # materialized.main performs the Stage-2 robustness patch on top of the
        # Stage-1 plan above.  Materialization arguments, if supplied, remain in
        # ``forwarded`` and are consumed by that wrapper.
        materialized.main(forwarded)
    finally:
        base.parse_args = original_parse
        base.seed_command_plan = original_plan
        base.effective_architecture = original_arch
        base.validate_stage_contract = original_validate
        base.METHOD = original_method
        base.PROTOCOL = original_protocol


if __name__ == "__main__":
    main()
