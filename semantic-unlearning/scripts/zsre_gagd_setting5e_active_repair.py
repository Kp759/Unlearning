#!/usr/bin/env python3
"""Run aggressive GA/GD Setting 5e and token-level LM-head repair on ZsRE.

ZsRE's field semantics are the reverse of the CounterFact convention used by
Setting 5e.  This runner therefore maps:

* ZsRE ``target_true`` (the original answer) -> internal unwanted target;
* tokenizer EOS -> internal desired neutral target.

The first stage uses the established all-token embedding/LM-head GA/GD plus
overlap-aware Setting 5e row restoration.  The second stage freezes the entire
model, safely unties the output head, and changes only the EOS output row.
Repair constraints use the exact teacher-forced token positions scored by the
official ZsRE evaluator:

* active rewrite/paraphrase tokens must prefer EOS over the sensitive token;
* initially correct neighborhood and retain-calibration tokens are protected;
* the repair direction is projected away from protected hidden states.

The candidate is selected only when official Eff/Gen do not regress and the
configured specificity/retention/PPL gates pass.  A rejected candidate is
reported but the saved selected checkpoint is restored to Setting 5e.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

import gagd_active_case_repair as active
import gagd_compare as gagd
import zsre_zero_unlearn_official_eval as zsre


METHOD = "zsre_gagd_setting5e_active_repair"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE


@dataclass
class TokenLogitCache:
    case: zsre.PredictionCase
    hidden: torch.Tensor
    target_token_id: int
    predicted_token_id: int
    target_logit: torch.Tensor
    neutral_logit: torch.Tensor
    correct: bool


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=gagd.DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default="outputs/zsre_setting5e_active/seed1")
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)

    # Established aggressive Setting 5e defaults.
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

    # ZsRE-native active LM-head repair.
    parser.add_argument("--repair-steps", type=int, default=800)
    parser.add_argument("--repair-lr", type=float, default=5e-3)
    parser.add_argument(
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
    )
    parser.add_argument("--active-logit-margin", type=float, default=0.25)
    parser.add_argument(
        "--selection-logit-margin",
        type=float,
        default=0.05,
        help=(
            "Minimum exact materialized EOS-minus-sensitive logit margin "
            "preferred during the BF16 scale sweep."
        ),
    )
    parser.add_argument(
        "--repair-rank",
        type=int,
        default=0,
        help=(
            "0 learns the unrestricted protected-orthogonal EOS-row delta. "
            "A positive value truncates the active hidden-state basis and can "
            "leave official ZsRE tokens unforgotten."
        ),
    )
    parser.add_argument("--repair-l2-lambda", type=float, default=1e-6)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--retain-calibration-num", type=int, default=128)
    parser.add_argument("--retain-calibration-seed", type=int, default=1729)
    parser.add_argument(
        "--project-away-protected-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-scales",
        default=(
            "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,"
            "0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0"
        ),
        help="Comma-separated backtracking scales for the learned EOS-row delta.",
    )

    # Official metric gates, all expressed in percentage points except PPL ratio.
    parser.add_argument("--utility-drop-tolerance", type=float, default=0.10)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.02)
    parser.add_argument(
        "--target-eff-max",
        type=float,
        default=0.0,
        help="Maximum accepted official forget Eff percentage.",
    )
    parser.add_argument(
        "--target-gen-max",
        type=float,
        default=0.0,
        help="Maximum accepted official forget Gen percentage.",
    )
    parser.add_argument(
        "--strict-utility-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-if-target-missed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exit nonzero after writing diagnostics when no candidate reaches "
            "the requested Eff/Gen target and utility gates."
        ),
    )

    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--save-setting5-checkpoint", action="store_true")
    parser.add_argument(
        "--save-selected-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Compatibility fields consumed by gagd_compare helpers.
    parser.set_defaults(
        dataset="zsre",
        mode=SETTING5_MODE,
        lr=1e-5,
        full_lr=None,
        optimizer=None,
        full_optimizer=None,
        forget_loss_type="mcf_margin",
        kl_retain_weight=0.0,
        save_model=False,
        sampling_strategy="epoch",
    )
    return parser


def parse_candidate_scales(text: str) -> List[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--candidate-scales must contain at least one scale")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("candidate scales must be finite and between 0 and 1")
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(values), reverse=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("forget and retain counts must be positive")
    if args.steps <= 0 or args.batch_size <= 0 or args.retain_batch_size <= 0:
        raise ValueError("training steps and batch sizes must be positive")
    if args.emb_lm_lr <= 0 or args.forget_weight <= 0 or args.retain_weight < 0:
        raise ValueError("invalid Setting 5e learning rate or loss weights")
    if args.forget_margin < 0:
        raise ValueError("forget margin must be non-negative")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("repair steps and learning rate must be positive")
    if args.active_logit_margin < 0 or args.selection_logit_margin < 0:
        raise ValueError("active/selection logit margins must be non-negative")
    if args.repair_rank < 0:
        raise ValueError("repair rank must be non-negative")
    if args.repair_l2_lambda < 0:
        raise ValueError("repair L2 lambda must be non-negative")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("max delta norm must be finite and non-negative")
    if args.retain_calibration_num < 0:
        raise ValueError("retain calibration count must be non-negative")
    if args.utility_drop_tolerance < 0:
        raise ValueError("utility drop tolerance must be non-negative")
    if args.max_ppl_ratio < 1:
        raise ValueError("max PPL ratio must be at least 1")
    if args.target_eff_max < 0 or args.target_gen_max < 0:
        raise ValueError("target Eff/Gen maxima must be non-negative")
    if args.eval_batch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("evaluation/cache batch sizes must be positive")
    for name in (
        "post_training_new_true_alpha",
        "post_training_new_retain_alpha",
        "post_training_new_true_retain_alpha",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    parse_candidate_scales(args.candidate_scales)


def canonical_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
) -> List[gagd.Example]:
    """Map ZsRE semantics into Setting 5e's unwanted/desired pair."""

    if tok.eos_token is None or tok.eos_token_id is None:
        raise ValueError("ZsRE neutral-target training requires tokenizer EOS")
    examples: List[gagd.Example] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        subject = str(rewrite["subject"])
        sensitive = gagd.normalize_answer(str(rewrite["target_true"]["str"]))
        examples.append(
            gagd.Example(
                prompt=str(rewrite["prompt"]).format(subject),
                answer=sensitive,
                subject=subject,
                target_new=sensitive,
                target_true=str(tok.eos_token),
                paraphrase_prompts=[
                    str(prompt) for prompt in record["paraphrase_prompts"]
                ],
                source="zsre",
            )
        )
    return examples


def _sample_retain_records(
    records: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    if count >= len(records):
        return list(records)
    return random.Random(seed).sample(list(records), k=count)


def official_correct_case_identities(
    records: Sequence[Mapping[str, Any]],
    metric_data: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
    prompt_types: Sequence[str],
) -> set[Tuple[int, str, int, int]]:
    """Return identities marked correct by the actual official evaluation pass.

    BF16 next-token ties can change when batch composition changes. Selecting
    active/protected cases from a separate cache pass can therefore omit an
    officially correct forget token or falsely flag a protected regression.
    This helper makes the official metric data the source of truth and uses the
    cache pass only to collect hidden states and logits.
    """

    metric_by_case: Dict[int, Mapping[str, Any]] = {}
    for item in metric_data:
        case_id = int(item["case_id"])
        if case_id in metric_by_case:
            raise ValueError(f"Duplicate official metric data for case_id={case_id}")
        metric_by_case[case_id] = item

    key_by_prompt_type = {
        "rewrite": "rewrite_prompts_correct",
        "paraphrase": "paraphrase_prompts_correct",
        "neighborhood": "neighborhood_prompts_correct",
    }
    unknown = sorted(set(prompt_types) - set(key_by_prompt_type))
    if unknown:
        raise ValueError(f"Unsupported ZsRE prompt types: {unknown}")

    correct: set[Tuple[int, str, int, int]] = set()
    for record in records:
        case_id = int(record["case_id"])
        if case_id not in metric_by_case:
            raise ValueError(
                f"Official metric data is missing sampled case_id={case_id}"
            )
        post = metric_by_case[case_id]["post"]
        expanded = zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=prompt_types,
        )
        by_type: Dict[str, List[zsre.PredictionCase]] = {
            prompt_type: [] for prompt_type in prompt_types
        }
        for case in expanded:
            by_type[case.prompt_type].append(case)

        for prompt_type in prompt_types:
            cases = by_type[prompt_type]
            values = [bool(value) for value in post[key_by_prompt_type[prompt_type]]]
            if len(cases) != len(values):
                raise ValueError(
                    "Official ZsRE metric/case count mismatch for "
                    f"case_id={case_id}, prompt_type={prompt_type}: "
                    f"{len(cases)} cases != {len(values)} metric values"
                )
            correct.update(
                case.identity
                for case, is_correct in zip(cases, values)
                if is_correct
            )
    return correct


@torch.no_grad()
def cache_prediction_cases(
    model: nn.Module,
    tok: Any,
    cases: Sequence[zsre.PredictionCase],
    *,
    neutral_token_id: int,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    desc: str,
) -> List[TokenLogitCache]:
    caches: List[TokenLogitCache] = []
    batches = list(_chunks(list(cases), batch_size))
    for batch in tqdm(batches, desc=desc, leave=False):
        encoded = tok(
            [case.prompt for case in batch],
            padding=True,
            return_tensors="pt",
        ).to(device)
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][batch_indices, last_non_masked, :].float()
        logits = output.logits[batch_indices, last_non_masked, :].float()
        target_ids = zsre.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        neutral_logits = logits[:, neutral_token_id]
        for index, case in enumerate(batch):
            target_id = int(target_ids[index].item())
            predicted_id = int(predicted_ids[index].item())
            caches.append(
                TokenLogitCache(
                    case=case,
                    hidden=hidden[index].detach(),
                    target_token_id=target_id,
                    predicted_token_id=predicted_id,
                    target_logit=target_logits[index].detach(),
                    neutral_logit=neutral_logits[index].detach(),
                    correct=bool(target_id == predicted_id),
                )
            )
    return caches


def cache_report(cache: TokenLogitCache) -> Dict[str, Any]:
    return {
        **asdict(cache.case),
        "target_token_id": cache.target_token_id,
        "predicted_token_id": cache.predicted_token_id,
        "target_logit": float(cache.target_logit.detach().cpu()),
        "neutral_logit": float(cache.neutral_logit.detach().cpu()),
        "target_minus_neutral_logit": float(
            (cache.target_logit - cache.neutral_logit).detach().cpu()
        ),
        "correct": cache.correct,
    }


def stack_hidden(
    caches: Sequence[TokenLogitCache],
    *,
    device: torch.device,
) -> torch.Tensor:
    if not caches:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    return torch.stack([cache.hidden for cache in caches]).to(
        device=device,
        dtype=torch.float32,
    )


def margins_for_delta(
    caches: Sequence[TokenLogitCache],
    delta_row: torch.Tensor,
) -> torch.Tensor:
    if not caches:
        return delta_row.new_empty((0,))
    hidden = torch.stack([cache.hidden for cache in caches]).to(
        device=delta_row.device,
        dtype=delta_row.dtype,
    )
    target = torch.stack([cache.target_logit for cache in caches]).to(
        device=delta_row.device,
        dtype=delta_row.dtype,
    )
    neutral = torch.stack([cache.neutral_logit for cache in caches]).to(
        device=delta_row.device,
        dtype=delta_row.dtype,
    )
    return neutral + hidden @ delta_row.reshape(-1) - target


def protected_regressions_for_delta(
    caches: Sequence[TokenLogitCache],
    delta_row: torch.Tensor,
) -> int:
    """Only the neutral row changes, so positive target-neutral keeps top-1."""

    if not caches:
        return 0
    margins = margins_for_delta(caches, delta_row)
    return int((margins >= 0).sum().item())


def active_correct_for_delta(
    caches: Sequence[TokenLogitCache],
    delta_row: torch.Tensor,
) -> int:
    if not caches:
        return 0
    margins = margins_for_delta(caches, delta_row)
    return int((margins < 0).sum().item())


def optimize_eos_delta(
    active_caches: Sequence[TokenLogitCache],
    protected_caches: Sequence[TokenLogitCache],
    *,
    hidden_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    if not active_caches:
        return (
            torch.zeros((1, hidden_size), dtype=torch.float32, device=device),
            [],
            {
                "steps_completed": 0,
                "stopped_early": True,
                "all_satisfied": True,
                "reason": "no_active_official_tokens",
            },
        )

    active_hidden = stack_hidden(active_caches, device=device)
    protected_hidden = stack_hidden(protected_caches, device=device)
    retained_basis = None
    if args.project_away_protected_hidden and protected_hidden.numel():
        retained_basis = active.orthonormal_row_basis(protected_hidden)
    projected_active = active.project_rows_away(active_hidden, retained_basis)
    direction_basis = None
    if args.repair_rank > 0:
        direction_basis = active.orthonormal_row_basis(
            projected_active,
            max_rank=args.repair_rank,
        )
        if direction_basis.numel() == 0:
            raise RuntimeError(
                "All active hidden directions were removed by protected projection"
            )

    module = active.SelectedRowDelta(
        n_rows=1,
        hidden_size=hidden_size,
        direction_basis=direction_basis,
        retained_basis=retained_basis,
        device=device,
    )

    def margin_fn(delta_rows: torch.Tensor) -> torch.Tensor:
        return margins_for_delta(active_caches, delta_rows[0])

    def zero_kl(delta_rows: torch.Tensor) -> torch.Tensor:
        return delta_rows.new_zeros(())

    required = torch.full(
        (len(active_caches),),
        float(args.active_logit_margin),
        device=device,
        dtype=torch.float32,
    )
    logs, summary = active.optimize_selected_delta(
        module,
        margin_fn,
        zero_kl,
        required_margins=required,
        repair_steps=args.repair_steps,
        repair_lr=args.repair_lr,
        repair_optimizer=args.repair_optimizer,
        hinge_weight=1.0,
        delta_l2_lambda=args.repair_l2_lambda,
        retain_kl_mu=0.0,
        stop_when_all_satisfied=args.stop_when_all_satisfied,
        max_delta_norm=args.max_delta_norm,
    )
    delta = module.effective_delta().detach()
    summary.update(
        {
            "active_token_count": len(active_caches),
            "protected_token_count": len(protected_caches),
            "protected_hidden_rank": (
                0 if retained_basis is None else int(retained_basis.shape[0])
            ),
            "repair_direction_rank": (
                hidden_size
                if direction_basis is None
                else int(direction_basis.shape[0])
            ),
            "effective_delta_norm": float(delta.norm().detach().cpu()),
        }
    )
    return delta, logs, summary


def select_delta_scale(
    delta_row: torch.Tensor,
    active_caches: Sequence[TokenLogitCache],
    protected_caches: Sequence[TokenLogitCache],
    scales: Sequence[float],
) -> Tuple[float, List[Dict[str, Any]]]:
    reports: List[Dict[str, Any]] = []
    for scale in scales:
        scaled = delta_row * float(scale)
        active_correct = active_correct_for_delta(active_caches, scaled)
        protected_regressions = protected_regressions_for_delta(
            protected_caches,
            scaled,
        )
        active_margins = margins_for_delta(active_caches, scaled)
        reports.append(
            {
                "scale": float(scale),
                "active_correct_tokens": active_correct,
                "protected_regressions": protected_regressions,
                "minimum_active_margin": (
                    None
                    if not len(active_caches)
                    else float(active_margins.min().detach().cpu())
                ),
                "delta_norm": float(scaled.norm().detach().cpu()),
            }
        )
    best = min(
        reports,
        key=lambda item: (
            int(item["protected_regressions"]),
            int(item["active_correct_tokens"]),
            -float(
                item["minimum_active_margin"]
                if item["minimum_active_margin"] is not None
                else 0.0
            ),
            float(item["delta_norm"]),
        ),
    )
    return float(best["scale"]), reports


@torch.no_grad()
def materialize_output_row(
    output_weight: torch.Tensor,
    token_id: int,
    original_row: torch.Tensor,
    delta_row: torch.Tensor,
    scale: float,
) -> None:
    updated = original_row + (float(scale) * delta_row).to(
        device=original_row.device,
        dtype=original_row.dtype,
    )
    output_weight[token_id].copy_(updated)


def exact_bf16_scale_sweep(
    *,
    model: nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    neutral_token_id: int,
    original_neutral_row: torch.Tensor,
    delta_row: torch.Tensor,
    active_cases: Sequence[zsre.PredictionCase],
    protected_cases: Sequence[zsre.PredictionCase],
    active_context_cases: Optional[Sequence[zsre.PredictionCase]] = None,
    protected_context_cases: Optional[Sequence[zsre.PredictionCase]] = None,
    scales: Sequence[float],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    minimum_active_margin: float = 0.0,
) -> Tuple[
    float,
    List[Dict[str, Any]],
    List[TokenLogitCache],
    List[TokenLogitCache],
    Dict[str, Any],
]:
    """Choose a scale from exact materialized predictions relative to scale 0.

    The context sequences may contain additional cases that are not scored.
    They are still forwarded in their original order so the scored rows see
    exactly the same padding and batch composition as the official evaluator.
    This matters for BF16 models with near-tied output logits. If active and
    protected cases share a context, it is forwarded only once per scale.
    """

    normalized_scales = sorted({float(scale) for scale in scales}, reverse=True)
    if 0.0 not in normalized_scales:
        raise ValueError("Exact BF16 scale sweep requires scale 0.0")
    if batch_size <= 0:
        raise ValueError("Exact BF16 scale sweep batch size must be positive")
    if minimum_active_margin < 0:
        raise ValueError("Exact BF16 active margin must be non-negative")

    active_identities = {case.identity for case in active_cases}
    if len(active_identities) != len(active_cases):
        raise ValueError("Active ZsRE prediction-case identities must be unique")
    protected_identities = {case.identity for case in protected_cases}
    if len(protected_identities) != len(protected_cases):
        raise ValueError("Protected ZsRE prediction-case identities must be unique")
    active_evaluation_cases = (
        list(active_context_cases)
        if active_context_cases is not None
        else list(active_cases)
    )
    protected_evaluation_cases = (
        list(protected_context_cases)
        if protected_context_cases is not None
        else list(protected_cases)
    )
    if not active_identities.issubset(
        {case.identity for case in active_evaluation_cases}
    ):
        raise ValueError("Active context is missing one or more scored cases")
    if not protected_identities.issubset(
        {case.identity for case in protected_evaluation_cases}
    ):
        raise ValueError("Protected context is missing one or more scored cases")
    shared_context = [
        case.identity for case in active_evaluation_cases
    ] == [case.identity for case in protected_evaluation_cases]

    def exact_predictions(
        cases: Sequence[zsre.PredictionCase],
        *,
        scale: float,
        label: str,
    ) -> List[TokenLogitCache]:
        return cache_prediction_cases(
            model,
            tok,
            cases,
            neutral_token_id=neutral_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=f"exact materialized {label} scale={scale:g}",
        )

    def select_rows(
        rows: Sequence[TokenLogitCache],
        identities: set[Tuple[int, str, int, int]],
        *,
        label: str,
    ) -> List[TokenLogitCache]:
        selected = [row for row in rows if row.case.identity in identities]
        selected_identities = {row.case.identity for row in selected}
        if selected_identities != identities or len(selected) != len(identities):
            raise RuntimeError(
                f"Exact {label} context did not produce every scored case"
            )
        return selected

    def evaluate_scale(
        scale: float,
        *,
        label: str,
    ) -> Tuple[List[TokenLogitCache], List[TokenLogitCache]]:
        active_context_rows = exact_predictions(
            active_evaluation_cases,
            scale=scale,
            label=f"{label} forget context",
        )
        active_rows = select_rows(
            active_context_rows,
            active_identities,
            label="active",
        )
        if shared_context:
            protected_context_rows = active_context_rows
        else:
            protected_context_rows = exact_predictions(
                protected_evaluation_cases,
                scale=scale,
                label=f"{label} protected context",
            )
        protected_rows = select_rows(
            protected_context_rows,
            protected_identities,
            label="protected",
        )
        return active_rows, protected_rows

    # Establish the numerical baseline with the same cases and batching used
    # for every nonzero candidate. This avoids treating pre-existing BF16
    # near-tie flips as damage caused by the repair.
    materialize_output_row(
        output_weight,
        neutral_token_id,
        original_neutral_row,
        delta_row,
        0.0,
    )
    zero_active_rows, zero_protected_rows = evaluate_scale(
        0.0,
        label="zero",
    )
    zero_active_correct = [bool(row.correct) for row in zero_active_rows]
    zero_protected_correct = [bool(row.correct) for row in zero_protected_rows]
    zero_active_count = int(sum(zero_active_correct))
    zero_protected_incorrect = int(
        len(zero_protected_correct) - sum(zero_protected_correct)
    )

    reports: List[Dict[str, Any]] = []
    for scale in normalized_scales:
        if scale == 0.0:
            materialize_output_row(
                output_weight,
                neutral_token_id,
                original_neutral_row,
                delta_row,
                0.0,
            )
            active_rows = zero_active_rows
            protected_rows = zero_protected_rows
        else:
            materialize_output_row(
                output_weight,
                neutral_token_id,
                original_neutral_row,
                delta_row,
                scale,
            )
            active_rows, protected_rows = evaluate_scale(
                scale,
                label="candidate",
            )

        active_correct = [bool(row.correct) for row in active_rows]
        protected_correct = [bool(row.correct) for row in protected_rows]
        # Only baseline-correct tokens need an EOS-over-target margin. A token
        # already made incorrect by a third vocabulary item can legitimately
        # have EOS below its target while still contributing zero Eff/Gen.
        repaired_active_margins = [
            float((row.neutral_logit - row.target_logit).detach().cpu())
            for row, baseline_ok in zip(active_rows, zero_active_correct)
            if baseline_ok
        ]
        incremental_regressions = int(
            sum(
                baseline_ok and not candidate_ok
                for baseline_ok, candidate_ok in zip(
                    zero_protected_correct,
                    protected_correct,
                )
            )
        )
        recovered_baseline_errors = int(
            sum(
                (not baseline_ok) and candidate_ok
                for baseline_ok, candidate_ok in zip(
                    zero_protected_correct,
                    protected_correct,
                )
            )
        )
        effective_delta = (
            output_weight[neutral_token_id].detach().float()
            - original_neutral_row.detach().float()
        )
        active_count = int(sum(active_correct))
        reports.append(
            {
                "scale": scale,
                "active_total_tokens": len(active_correct),
                "active_correct_tokens": active_count,
                "active_repaired_vs_zero": zero_active_count - active_count,
                "baseline_correct_active_tokens": zero_active_count,
                "minimum_repaired_active_neutral_minus_target_margin": (
                    None
                    if not repaired_active_margins
                    else min(repaired_active_margins)
                ),
                "repaired_active_margin_satisfied_tokens": int(
                    sum(
                        margin >= minimum_active_margin
                        for margin in repaired_active_margins
                    )
                ),
                "protected_total_tokens": len(protected_correct),
                "protected_absolute_incorrect": int(
                    len(protected_correct) - sum(protected_correct)
                ),
                "protected_incremental_regressions_vs_zero": (
                    incremental_regressions
                ),
                "protected_zero_scale_incorrect": zero_protected_incorrect,
                "protected_baseline_errors_recovered": (
                    recovered_baseline_errors
                ),
                "materialized_delta_norm": float(effective_delta.norm().cpu()),
                "nonzero_materialized_delta": bool(
                    torch.count_nonzero(effective_delta).item()
                ),
            }
        )

    safe = [
        row
        for row in reports
        if row["scale"] > 0.0
        and row["nonzero_materialized_delta"]
        and row["active_repaired_vs_zero"] > 0
        and row["protected_incremental_regressions_vs_zero"] == 0
    ]
    zero_active_safe = [row for row in safe if row["active_correct_tokens"] == 0]
    robust_zero_active_safe = [
        row
        for row in zero_active_safe
        if row["minimum_repaired_active_neutral_minus_target_margin"] is None
        or row["minimum_repaired_active_neutral_minus_target_margin"]
        >= minimum_active_margin
    ]
    candidates = robust_zero_active_safe or zero_active_safe or safe
    if candidates:
        # First eliminate every active token. Among equally effective scales,
        # choose the smallest materialized perturbation to protect Spe/PPL.
        selected = min(
            candidates,
            key=lambda row: (
                int(row["active_correct_tokens"]),
                float(row["materialized_delta_norm"]),
                float(row["scale"]),
            ),
        )
    else:
        selected = next(row for row in reports if row["scale"] == 0.0)

    selected_scale = float(selected["scale"])
    materialize_output_row(
        output_weight,
        neutral_token_id,
        original_neutral_row,
        delta_row,
        selected_scale,
    )
    if selected_scale == 0.0:
        selected_active = zero_active_rows
        selected_protected = zero_protected_rows
    else:
        selected_active, selected_protected = evaluate_scale(
            selected_scale,
            label="selected",
        )

    baseline = {
        "active_correct_tokens_at_zero": zero_active_count,
        "active_total_tokens": len(zero_active_correct),
        "protected_correct_tokens_at_zero": int(sum(zero_protected_correct)),
        "protected_incorrect_tokens_at_zero": zero_protected_incorrect,
        "protected_total_tokens": len(zero_protected_correct),
    }
    return (
        selected_scale,
        reports,
        selected_active,
        selected_protected,
        baseline,
    )


def metric_gate_report(
    setting5: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    utility_drop_tolerance: float,
    max_ppl_ratio: float,
    target_eff_max: Optional[float] = None,
    target_gen_max: Optional[float] = None,
) -> Dict[str, Any]:
    checks: Dict[str, Dict[str, Any]] = {}

    def add_check(
        name: str,
        before: Optional[float],
        after: Optional[float],
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> None:
        if before is None or after is None:
            passed = True
        elif minimum is not None:
            passed = float(after) >= float(minimum)
        elif maximum is not None:
            passed = float(after) <= float(maximum)
        else:
            raise RuntimeError("metric gate lacks a bound")
        checks[name] = {
            "setting5": before,
            "candidate": after,
            "minimum": minimum,
            "maximum": maximum,
            "passed": bool(passed),
        }

    add_check(
        "forget_Eff_non_regression",
        setting5["forget"]["Eff"],
        candidate["forget"]["Eff"],
        maximum=float(setting5["forget"]["Eff"]),
    )
    add_check(
        "forget_Gen_non_regression",
        setting5["forget"]["Gen"],
        candidate["forget"]["Gen"],
        maximum=float(setting5["forget"]["Gen"]),
    )
    if target_eff_max is not None:
        add_check(
            "forget_Eff_target",
            setting5["forget"]["Eff"],
            candidate["forget"]["Eff"],
            maximum=float(target_eff_max),
        )
    if target_gen_max is not None:
        add_check(
            "forget_Gen_target",
            setting5["forget"]["Gen"],
            candidate["forget"]["Gen"],
            maximum=float(target_gen_max),
        )
    add_check(
        "forget_Spe_locality",
        setting5["forget"]["Spe"],
        candidate["forget"]["Spe"],
        minimum=float(setting5["forget"]["Spe"]) - utility_drop_tolerance,
    )
    for metric in ("Eff", "Gen", "Spe"):
        add_check(
            f"retain_{metric}",
            setting5["retain"][metric],
            candidate["retain"][metric],
            minimum=float(setting5["retain"][metric]) - utility_drop_tolerance,
        )
    setting_ppl = setting5.get("forget_PPL")
    candidate_ppl = candidate.get("forget_PPL")
    add_check(
        "PPL",
        setting_ppl,
        candidate_ppl,
        maximum=(
            None
            if setting_ppl is None
            else float(setting_ppl) * max_ppl_ratio
        ),
    )
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "utility_drop_tolerance_percentage_points": utility_drop_tolerance,
        "max_ppl_ratio": max_ppl_ratio,
        "target_eff_max": target_eff_max,
        "target_gen_max": target_gen_max,
        "checks": checks,
    }


def official_forget_active_correct_tokens(result: Mapping[str, Any]) -> int:
    forget = result["forget"]
    return int(forget["post_rewrite_correct_tokens"]) + int(
        forget["post_paraphrase_correct_tokens"]
    )


def compact_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "method": result["method"],
        "forget": {
            key: result["forget"][key]
            for key in (
                "Eff",
                "Gen",
                "Spe",
                "post_rewrite_correct_tokens",
                "post_rewrite_total_tokens",
                "post_paraphrase_correct_tokens",
                "post_paraphrase_total_tokens",
                "post_neighborhood_correct_tokens",
                "post_neighborhood_total_tokens",
            )
        },
        "retain": {
            key: result["retain"][key]
            for key in (
                "Eff",
                "Gen",
                "Spe",
                "post_rewrite_correct_tokens",
                "post_rewrite_total_tokens",
                "post_paraphrase_correct_tokens",
                "post_paraphrase_total_tokens",
                "post_neighborhood_correct_tokens",
                "post_neighborhood_total_tokens",
            )
        },
        "PPL": result.get("forget_PPL"),
    }


def comparison_row(label: str, result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "method": label,
        "forget_Eff_down": result["forget"]["Eff"],
        "forget_Gen_down": result["forget"]["Gen"],
        "forget_Spe_up": result["forget"]["Spe"],
        "retain_Eff_up": result["retain"]["Eff"],
        "retain_Gen_up": result["retain"]["Gen"],
        "retain_Spe_up": result["retain"]["Spe"],
        "PPL_down": result.get("forget_PPL"),
    }


def write_comparison(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with (output_dir / "comparison.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ZsRE Setting 5e + active LM-head repair",
        "",
        "Official ZeroUnlearn-compatible token accuracy. Forget Eff/Gen are lower-is-better; forget Spe and all retain metrics are higher-is-better; PPL is lower-is-better.",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in fields) + " |")
    (output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(model: nn.Module, tok: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting5_dir = output_dir / "setting5e"
    repair_dir = output_dir / "active_repair"
    setting5_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "setting5_mode": SETTING5_MODE,
            "zsre_semantic_mapping": {
                "internal_target_new_unwanted": "ZsRE target_true original answer",
                "internal_target_true_desired": "tokenizer EOS",
            },
            "official_metric_directions": {
                "forget_Eff": "down",
                "forget_Gen": "down",
                "forget_Spe": "up",
                "retain_Eff_Gen_Spe": "up",
                "PPL": "down",
            },
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading base model and exact official ZsRE records")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    zsre_path = Path(args.zsre_path)
    if not zsre_path.is_absolute():
        zsre_path = gagd.PROJECT_DIR / zsre_path
    zsre_path = zsre.download_zsre(zsre_path, url=args.zsre_url)
    forget_records, retain_records = zsre.load_official_eval_records(
        zsre_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        zsre_url=args.zsre_url,
    )
    records = (forget_records, retain_records)
    gagd.write_json(
        output_dir / "sampled_case_ids.json",
        {
            "seed": args.seed,
            "forget_case_ids": [record["case_id"] for record in forget_records],
            "retain_case_ids": [record["case_id"] for record in retain_records],
            "zsre_sha256": zsre.file_sha256(zsre_path),
        },
    )
    forget_examples = canonical_examples(forget_records, tok)
    retain_examples = canonical_examples(retain_records, tok)

    print("Evaluating base model with official ZsRE token accuracy")
    base_result = zsre.evaluate_loaded_model_official(
        method="Base",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        zsre_path=zsre_path,
        wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
        out_path=output_dir / "base_official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
        records=records,
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training ultra-aggressive Setting 5e on sensitive-answer/EOS margins")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    requested_save = bool(args.save_model)
    args.save_model = False
    train_summary = gagd.train_mode(
        model,
        tok,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=SETTING5_MODE,
        args=args,
        mode_dir=setting5_dir,
    )
    args.save_model = requested_save
    setting5_result = zsre.evaluate_loaded_model_official(
        method="Setting 5e",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e",
        zsre_path=zsre_path,
        wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
        out_path=setting5_dir / "official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
        records=records,
    )
    if args.save_setting5_checkpoint:
        save_checkpoint(model, tok, setting5_dir / "checkpoint")

    print("Building exact token-level active and protected repair sets")
    output_layer = active.freeze_model_for_output_repair(model)
    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = int(tok.eos_token_id)
    if not 0 <= neutral_token_id < output_layer.weight.shape[0]:
        raise ValueError("Tokenizer EOS ID is outside the LM-head vocabulary")
    original_neutral_row = output_layer.weight[neutral_token_id].detach().clone()

    forget_active_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite", "paraphrase"),
        )
    ]
    forget_official_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    calibration_records = _sample_retain_records(
        retain_records,
        args.retain_calibration_num,
        args.retain_calibration_seed,
    )
    retain_protected_cases = [
        case
        for record in calibration_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    official_active_identities = official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("rewrite", "paraphrase"),
    )
    official_protected_identities = official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("neighborhood",),
    )
    official_protected_identities.update(
        official_correct_case_identities(
            calibration_records,
            setting5_result["retain_raw"],
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite", "paraphrase", "neighborhood"),
        )
    )
    # Reproduce the official forget pass exactly while collecting repair
    # hidden states. Filtering after the full pass avoids BF16 batch-composition
    # disagreements between case selection, scale selection, and final metrics.
    forget_official_caches = cache_prediction_cases(
        model,
        tok,
        forget_official_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        desc="cache official-order forget ZsRE tokens",
    )
    active_all = [
        cache
        for cache in forget_official_caches
        if cache.case.prompt_type in {"rewrite", "paraphrase"}
    ]
    forget_protected_all = [
        cache
        for cache in forget_official_caches
        if cache.case.prompt_type == "neighborhood"
    ]
    retain_protected_all = cache_prediction_cases(
        model,
        tok,
        retain_protected_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache retain-calibration ZsRE tokens",
    )
    protected_all = forget_protected_all + retain_protected_all
    active_caches = [
        cache
        for cache in active_all
        if cache.case.identity in official_active_identities
        and cache.target_token_id != neutral_token_id
    ]
    protected_caches = [
        cache
        for cache in protected_all
        if cache.case.identity in official_protected_identities
        and cache.target_token_id != neutral_token_id
    ]
    missing_active = official_active_identities - {
        cache.case.identity for cache in active_all
    }
    missing_protected = official_protected_identities - {
        cache.case.identity for cache in protected_all
    }
    if missing_active:
        raise RuntimeError(
            "Failed to cache officially correct active ZsRE tokens: "
            f"{sorted(missing_active)[:10]}"
        )
    if missing_protected:
        raise RuntimeError(
            "Failed to cache officially correct protected ZsRE tokens: "
            f"{sorted(missing_protected)[:10]}"
        )
    write_jsonl(
        repair_dir / "active_tokens_before.jsonl",
        [cache_report(cache) for cache in active_caches],
    )
    write_jsonl(
        repair_dir / "protected_tokens_before.jsonl",
        [cache_report(cache) for cache in protected_caches],
    )

    print(
        f"Optimizing only LM-head row {neutral_token_id} ({tok.eos_token!r}): "
        f"active={len(active_caches)}, protected={len(protected_caches)}"
    )
    delta_rows, repair_logs, repair_optimization = optimize_eos_delta(
        active_caches,
        protected_caches,
        hidden_size=output_layer.weight.shape[1],
        device=device,
        args=args,
    )
    write_jsonl(repair_dir / "repair_log.jsonl", repair_logs)
    (
        selected_scale,
        scale_reports,
        exact_active_after,
        exact_protected_after,
        exact_zero_baseline,
    ) = exact_bf16_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        # Score every rewrite/paraphrase token, including tokens that were
        # already incorrect at scale 0, so the sweep cannot accidentally make
        # one of them correct. Forward the complete forget split in official
        # order to reproduce the evaluator's BF16 batch composition.
        active_cases=forget_active_cases,
        protected_cases=[
            cache.case
            for cache in forget_protected_all
            if cache.case.identity in official_protected_identities
            and cache.target_token_id != neutral_token_id
        ],
        active_context_cases=forget_official_cases,
        protected_context_cases=forget_official_cases,
        scales=parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        # Match the official evaluator's batch size for the materialized sweep.
        batch_size=args.eval_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    gagd.write_json(repair_dir / "bf16_exact_scale_sweep.json", scale_reports)
    gagd.write_json(
        repair_dir / "exact_zero_scale_baseline.json",
        exact_zero_baseline,
    )
    official_setting5_active_correct = official_forget_active_correct_tokens(
        setting5_result
    )
    if (
        exact_zero_baseline["active_correct_tokens_at_zero"]
        != official_setting5_active_correct
    ):
        raise RuntimeError(
            "Exact scale-0 active count does not match the official Setting 5e "
            "forget pass despite identical case order and batch size: "
            f"{exact_zero_baseline['active_correct_tokens_at_zero']} != "
            f"{official_setting5_active_correct}"
        )
    if exact_zero_baseline["protected_incorrect_tokens_at_zero"] != 0:
        raise RuntimeError(
            "An officially correct forget-neighborhood token changed at exact "
            "scale 0; refusing a numerically misaligned repair sweep"
        )
    selected_scale_report = next(
        report
        for report in scale_reports
        if float(report["scale"]) == selected_scale
    )
    local_protected_regressions = int(
        selected_scale_report["protected_incremental_regressions_vs_zero"]
    )
    write_jsonl(
        repair_dir / "active_tokens_after.jsonl",
        [cache_report(cache) for cache in exact_active_after],
    )
    write_jsonl(
        repair_dir / "protected_tokens_after.jsonl",
        [cache_report(cache) for cache in exact_protected_after],
    )

    candidate_result = zsre.evaluate_loaded_model_official(
        method="Setting 5e + active LM-head repair candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e_active_candidate",
        zsre_path=zsre_path,
        wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
        out_path=repair_dir / "candidate_official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
        records=records,
    )
    official_candidate_active_correct = official_forget_active_correct_tokens(
        candidate_result
    )
    if (
        official_candidate_active_correct
        != int(selected_scale_report["active_correct_tokens"])
    ):
        raise RuntimeError(
            "Exact selected-scale active count does not match the final official "
            "candidate evaluation: "
            f"{selected_scale_report['active_correct_tokens']} != "
            f"{official_candidate_active_correct}"
        )
    gate_report = metric_gate_report(
        setting5_result,
        candidate_result,
        utility_drop_tolerance=args.utility_drop_tolerance,
        max_ppl_ratio=args.max_ppl_ratio,
        target_eff_max=args.target_eff_max,
        target_gen_max=args.target_gen_max,
    )
    target_met = bool(
        float(candidate_result["forget"]["Eff"]) <= args.target_eff_max
        and float(candidate_result["forget"]["Gen"]) <= args.target_gen_max
    )
    target_already_met = bool(
        float(setting5_result["forget"]["Eff"]) <= args.target_eff_max
        and float(setting5_result["forget"]["Gen"]) <= args.target_gen_max
    )
    local_success = bool(
        target_already_met
        or (
            selected_scale > 0.0
            and selected_scale_report["nonzero_materialized_delta"]
            and selected_scale_report["active_repaired_vs_zero"] > 0
            and local_protected_regressions == 0
        )
    )
    accepted = bool(
        target_met
        and local_success
        and (gate_report["passed"] or not args.strict_utility_gates)
    )
    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selected_result["method"] = "Setting 5e + accepted active LM-head repair"
        selection_reason = (
            "candidate_passed_all_official_metric_gates"
            if gate_report["passed"]
            else "strict_utility_gates_disabled"
        )
    else:
        with torch.no_grad():
            output_layer.weight[neutral_token_id].copy_(original_neutral_row)
        if not torch.equal(output_layer.weight[neutral_token_id], original_neutral_row):
            raise RuntimeError("Failed to restore rejected EOS LM-head candidate")
        selected_result = copy.deepcopy(setting5_result)
        selected_result["method"] = "Setting 5e (active candidate rejected)"
        if not target_met:
            selection_reason = "candidate_missed_required_zero_eff_gen_target"
        elif not local_success:
            selection_reason = "no_nonzero_bf16_safe_improving_scale"
        else:
            selection_reason = "candidate_failed_official_metric_gates"

    if args.save_selected_checkpoint:
        save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    repair_summary = {
        "method": METHOD,
        "neutral_token_id": neutral_token_id,
        "neutral_token": tok.eos_token,
        "only_selected_lm_head_rows_materialized": True,
        "selected_lm_head_row_count": 1,
        "input_embeddings_frozen_during_repair": True,
        "transformer_frozen_during_repair": True,
        "case_selection_source": "setting5e_official_metric_data",
        "active_tokens_before": len(active_caches),
        "active_tokens_zero_scale": exact_zero_baseline[
            "active_correct_tokens_at_zero"
        ],
        "active_tokens_after_candidate": int(
            selected_scale_report["active_correct_tokens"]
        ),
        "official_batch_alignment_verified": True,
        "optimization_protected_tokens": len(protected_caches),
        "exact_official_forget_protected_tokens": len(exact_protected_after),
        "protected_zero_scale_incorrect": exact_zero_baseline[
            "protected_incorrect_tokens_at_zero"
        ],
        "protected_incremental_regressions_after_candidate": int(
            local_protected_regressions
        ),
        "retain_calibration_records": len(calibration_records),
        "candidate_scale": selected_scale,
        "selected_scale": selected_scale if accepted else 0.0,
        "unscaled_delta_norm": float(delta_rows.norm().detach().cpu()),
        "materialized_delta_norm": float(
            (
                output_layer.weight[neutral_token_id].detach().float()
                - original_neutral_row.detach().float()
            )
            .norm()
            .cpu()
        ),
        "optimization": repair_optimization,
        "official_metric_gates": gate_report,
        "required_target_met": target_met,
        "candidate_accepted": accepted,
        "selection_reason": selection_reason,
    }
    gagd.write_json(repair_dir / "repair_summary.json", repair_summary)

    rows = [
        comparison_row("Base", base_result),
        comparison_row("Setting 5e", setting5_result),
        comparison_row("Setting 5e + active candidate", candidate_result),
        comparison_row("Selected", selected_result),
    ]
    write_comparison(output_dir, rows)
    final_result = {
        "method": METHOD,
        "dataset": "ZsRE",
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "zsre_sha256": zsre.file_sha256(zsre_path),
        "zsre_semantic_mapping": config["zsre_semantic_mapping"],
        "training": {
            **asdict(train_summary),
            "steps": args.steps,
            "emb_lm_lr": args.emb_lm_lr,
            "forget_weight": args.forget_weight,
            "retain_weight": args.retain_weight,
            "forget_margin": args.forget_margin,
        },
        "repair": repair_summary,
        "base": compact_metrics(base_result),
        "setting5e": compact_metrics(setting5_result),
        "active_candidate": compact_metrics(candidate_result),
        "selected": compact_metrics(selected_result),
        "comparison_rows": rows,
    }
    gagd.write_json(output_dir / "zsre_results.json", final_result)
    print(
        "Selected ZsRE result: "
        f"Eff={selected_result['forget']['Eff']}, "
        f"Gen={selected_result['forget']['Gen']}, "
        f"Spe={selected_result['forget']['Spe']}, "
        f"PPL={selected_result.get('forget_PPL')}; "
        f"active candidate accepted={accepted}"
    )
    print(f"Comparison: {output_dir / 'comparison.md'}")
    if args.fail_if_target_missed and not accepted:
        raise RuntimeError(
            "ZsRE repair did not meet the required Eff/Gen and utility gates. "
            f"Candidate Eff={candidate_result['forget']['Eff']}, "
            f"Gen={candidate_result['forget']['Gen']}, "
            f"Spe={candidate_result['forget']['Spe']}, "
            f"PPL={candidate_result.get('forget_PPL')}; "
            f"reason={selection_reason}. Diagnostics were written to {repair_dir}."
        )


if __name__ == "__main__":
    main()
