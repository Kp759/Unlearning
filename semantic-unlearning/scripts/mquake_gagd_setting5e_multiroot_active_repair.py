#!/usr/bin/env python3
"""Experimental MQuAKE Setting 5e with protected multi-row active repair.

This is a method extension.  The reproducibility baseline in
``mquake_gagd_setting5e_active_repair.py`` is intentionally unchanged.

Only sampled ``requested_rewrite`` cloze facts are visible to Setting 5e and
repair.  Natural-language atomic questions, the three record-level questions,
answers/aliases, and counterfactual ``target_new`` values remain held out until
the checkpoint selection decision has been written.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as baseline
import mquake_multihop_unlearning_eval as multihop
import mquake_zero_unlearn_official_eval as mquake
import zsre_gagd_setting5e_active_repair as repair


METHOD = "mquake_gagd_setting5e_multiroot_active_repair"
METHOD_LABEL = "MQuAKE Setting 5e + protected multi-row LM-head active repair"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE
ACTIVE_SOURCE = "sampled_requested_rewrite_cloze_teacher_forced_prefixes"


def build_parser() -> argparse.ArgumentParser:
    parser = baseline.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="outputs/mquake_setting5e_multiroot_active/seed0",
        steps=600,
        batch_size=1,
        retain_batch_size=4,
        emb_lm_lr=1e-4,
        forget_weight=2.0,
        retain_weight=1.0,
        forget_margin=1.0,
        emb_lm_optimizer="adamw",
        sampling_strategy="epoch",
        repair_steps=600,
        repair_lr=2e-3,
        repair_optimizer="adamw",
        active_logit_margin=0.50,
        selection_logit_margin=0.10,
        repair_rank=0,
        repair_l2_lambda=1e-6,
        retain_calibration_num=1000,
        retain_calibration_seed=1729,
        project_away_protected_hidden=True,
        stop_when_all_satisfied=True,
        target_eff_max=0.0,
        utility_drop_tolerance=0.10,
        max_ppl_ratio=1.02,
        strict_utility_gates=True,
    )
    parser.add_argument(
        "--forget-sampling",
        choices=("instance_balanced", "atomic_epoch"),
        default="instance_balanced",
        help=(
            "Setting 5e forget sampling. instance_balanced samples an instance "
            "uniformly and then one of its requested_rewrite atoms uniformly."
        ),
    )
    parser.add_argument(
        "--protected-logit-drift-weight",
        type=float,
        default=1.0,
        help="Penalty on modified-row logit drift over protected retain states.",
    )
    parser.add_argument("--multihop-prompt-dir", default="data/mquake_prompts")
    parser.add_argument("--multihop-batch-size", type=int, default=4)
    parser.add_argument("--standard-max-new-tokens", type=int, default=32)
    parser.add_argument("--cot-max-new-tokens", type=int, default=128)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    repair.validate_args(args)
    if args.batch_size != 1:
        raise ValueError("The pinned Setting 5e protocol requires --batch-size 1")
    if args.protected_logit_drift_weight < 0:
        raise ValueError("protected logit-drift weight must be non-negative")
    if args.multihop_batch_size <= 0:
        raise ValueError("multi-hop batch size must be positive")


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_repair_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> List[mquake.PredictionCase]:
    """Build only cloze teacher-forced states; never read evaluation prompts."""

    cases: List[mquake.PredictionCase] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        expanded = mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
        if any(case.prompt_type != "rewrite" for case in expanded):
            raise RuntimeError("Repair cases must be requested_rewrite cloze cases")
        cases.extend(expanded)
    return cases


def instance_balanced_training_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    steps: int,
    seed: int,
) -> Tuple[List[gagd.Example], Dict[str, Any]]:
    """Pre-sample one instance and then one atom per Setting 5e step."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    by_instance: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[int(record["source_index"])].append(record)
    if not by_instance:
        raise ValueError("No MQuAKE forget instances were supplied")
    instance_ids = sorted(by_instance)
    for values in by_instance.values():
        values.sort(key=lambda row: int(row["rewrite_index"]))

    rng = random.Random(seed)
    sampled_records: List[Mapping[str, Any]] = []
    instance_counts: Counter[int] = Counter()
    atom_counts: Counter[int] = Counter()
    for _ in range(steps):
        instance_id = instance_ids[rng.randrange(len(instance_ids))]
        atoms = by_instance[instance_id]
        record = atoms[rng.randrange(len(atoms))]
        sampled_records.append(record)
        instance_counts[instance_id] += 1
        atom_counts[int(record["case_id"])] += 1
    examples = baseline.canonical_examples(sampled_records, tok)
    return examples, {
        "strategy": "instance_balanced",
        "seed": int(seed),
        "steps": int(steps),
        "instance_draw_counts": {str(k): v for k, v in sorted(instance_counts.items())},
        "atomic_draw_counts": {str(k): v for k, v in sorted(atom_counts.items())},
    }


def setting5_forget_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    strategy: str,
    steps: int,
    seed: int,
) -> Tuple[List[gagd.Example], Dict[str, Any]]:
    if strategy == "instance_balanced":
        return instance_balanced_training_examples(
            records, tok, steps=steps, seed=seed
        )
    if strategy == "atomic_epoch":
        examples = baseline.canonical_examples(records, tok)
        return examples, {
            "strategy": "atomic_epoch",
            "seed": int(seed),
            "steps": int(steps),
            "atomic_fact_count": len(examples),
        }
    raise ValueError(f"Unsupported forget sampling strategy: {strategy}")


def residual_active_caches(
    caches: Sequence[repair.TokenLogitCache],
    unknown_token_id: int,
) -> List[repair.TokenLogitCache]:
    return [
        cache
        for cache in caches
        if cache.correct and int(cache.target_token_id) != int(unknown_token_id)
    ]


def active_row_ids(
    caches: Sequence[repair.TokenLogitCache],
    unknown_token_id: int,
) -> List[int]:
    """Unknown plus unique residual sensitive rows; shared IDs stay shared."""

    sensitive = sorted(
        {
            int(cache.target_token_id)
            for cache in caches
            if cache.correct
            and int(cache.target_token_id) != int(unknown_token_id)
        }
    )
    return [int(unknown_token_id), *sensitive]


def constraints_per_sensitive_row(
    caches: Sequence[repair.TokenLogitCache],
) -> Dict[int, int]:
    return dict(sorted(Counter(int(cache.target_token_id) for cache in caches).items()))


def sample_retain_instances(
    records: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    """Sample instance identities, then retain every atom in each instance."""

    if count < 0:
        raise ValueError("retain calibration count must be non-negative")
    by_instance: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[int(record["source_index"])].append(record)
    instance_ids = sorted(by_instance)
    if count >= len(instance_ids):
        selected_ids = instance_ids
    else:
        selected_ids = sorted(random.Random(seed).sample(instance_ids, count))
    return [
        record
        for instance_id in selected_ids
        for record in sorted(
            by_instance[instance_id], key=lambda row: int(row["rewrite_index"])
        )
    ]


def freeze_model_for_multirow_repair(model: nn.Module) -> nn.Module:
    """Untie the output head without changing logits, then freeze the model."""

    return active.freeze_model_for_output_repair(model)


def multirow_margins(
    caches: Sequence[repair.TokenLogitCache],
    delta_rows: torch.Tensor,
    row_ids: Sequence[int],
    unknown_token_id: int,
) -> torch.Tensor:
    if not caches:
        return delta_rows.new_empty((0,))
    row_index = {int(token_id): index for index, token_id in enumerate(row_ids)}
    if unknown_token_id not in row_index:
        raise ValueError("Unknown token row is absent from the repair rows")
    unknown_delta = delta_rows[row_index[unknown_token_id]]
    values: List[torch.Tensor] = []
    for cache in caches:
        target_id = int(cache.target_token_id)
        if target_id not in row_index:
            raise ValueError(f"Residual sensitive row {target_id} is not trainable")
        hidden = cache.hidden.to(device=delta_rows.device, dtype=delta_rows.dtype)
        base_margin = (cache.neutral_logit - cache.target_logit).to(
            device=delta_rows.device, dtype=delta_rows.dtype
        )
        values.append(
            base_margin
            + hidden @ unknown_delta
            - hidden @ delta_rows[row_index[target_id]]
        )
    return torch.stack(values)


def protected_logit_drift_loss(
    delta_rows: torch.Tensor,
    protected_caches: Sequence[repair.TokenLogitCache],
) -> torch.Tensor:
    """Mean squared Setting-5e logit drift for every modified output row."""

    if not protected_caches:
        return delta_rows.new_zeros(())
    hidden = torch.stack([cache.hidden for cache in protected_caches]).to(
        device=delta_rows.device, dtype=delta_rows.dtype
    )
    drift = hidden @ delta_rows.transpose(0, 1)
    return drift.square().mean()


def optimize_multirow_delta(
    active_caches: Sequence[repair.TokenLogitCache],
    protected_caches: Sequence[repair.TokenLogitCache],
    *,
    row_ids: Sequence[int],
    unknown_token_id: int,
    hidden_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    if not active_caches:
        zeros = torch.zeros(
            (len(row_ids), hidden_size), dtype=torch.float32, device=device
        )
        return zeros, [], {
            "steps_completed": 0,
            "stopped_early": True,
            "all_satisfied": True,
            "reason": "no_residual_active_tokens",
        }

    protected_hidden = repair.stack_hidden(protected_caches, device=device)
    retained_basis = None
    if args.project_away_protected_hidden and protected_hidden.numel():
        retained_basis = active.orthonormal_row_basis(protected_hidden)

    active_hidden = repair.stack_hidden(active_caches, device=device)
    projected_active = active.project_rows_away(active_hidden, retained_basis)
    direction_basis = None
    if args.repair_rank > 0:
        direction_basis = active.orthonormal_row_basis(
            projected_active, max_rank=args.repair_rank
        )
        if direction_basis.numel() == 0:
            raise RuntimeError("Protected projection removed every active direction")

    module = active.SelectedRowDelta(
        n_rows=len(row_ids),
        hidden_size=hidden_size,
        direction_basis=direction_basis,
        retained_basis=retained_basis,
        device=device,
    )

    def margin_fn(delta: torch.Tensor) -> torch.Tensor:
        return multirow_margins(
            active_caches, delta, row_ids, unknown_token_id
        )

    def drift_fn(delta: torch.Tensor) -> torch.Tensor:
        return protected_logit_drift_loss(delta, protected_caches)

    required = torch.full(
        (len(active_caches),),
        float(args.active_logit_margin),
        dtype=torch.float32,
        device=device,
    )
    logs, summary = active.optimize_selected_delta(
        module,
        margin_fn,
        drift_fn,
        required_margins=required,
        repair_steps=args.repair_steps,
        repair_lr=args.repair_lr,
        repair_optimizer=args.repair_optimizer,
        hinge_weight=1.0,
        delta_l2_lambda=args.repair_l2_lambda,
        retain_kl_mu=args.protected_logit_drift_weight,
        stop_when_all_satisfied=args.stop_when_all_satisfied,
        max_delta_norm=args.max_delta_norm,
    )
    for row in logs:
        row["protected_logit_drift_loss"] = row.pop(
            "retain_kl_reference_to_repaired"
        )
        row["weighted_protected_logit_drift"] = row.pop("weighted_retain_kl")
    delta = module.effective_delta().detach()
    summary.update(
        {
            "modified_row_count": len(row_ids),
            "row_ids": [int(value) for value in row_ids],
            "active_constraint_count": len(active_caches),
            "protected_state_count": len(protected_caches),
            "protected_hidden_rank": (
                0 if retained_basis is None else int(retained_basis.shape[0])
            ),
            "protected_logit_drift_weight": float(
                args.protected_logit_drift_weight
            ),
        }
    )
    return delta, logs, summary


@torch.no_grad()
def materialize_multirow_scale(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
) -> None:
    if len(row_ids) != original_rows.shape[0] or len(row_ids) != delta_rows.shape[0]:
        raise ValueError("row IDs, originals, and deltas must have matching rows")
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    updated = original_rows + float(scale) * delta_rows.to(
        device=original_rows.device, dtype=original_rows.dtype
    )
    output_weight.index_copy_(0, ids, updated)
    if float(scale) == 0.0 and not torch.equal(
        output_weight.index_select(0, ids), original_rows
    ):
        raise RuntimeError("Scale 0 did not exactly restore Setting 5e rows")


def _select_caches(
    rows: Sequence[repair.TokenLogitCache],
    cases: Sequence[mquake.PredictionCase],
) -> List[repair.TokenLogitCache]:
    identities = {case.identity for case in cases}
    selected = [row for row in rows if row.case.identity in identities]
    if len(selected) != len(identities):
        raise RuntimeError("Exact materialization omitted scored token states")
    return selected


def scale_report_is_locally_safe(report: Mapping[str, Any]) -> bool:
    return bool(
        int(report["active_correct_tokens"]) == 0
        and int(report["active_margin_violations"]) == 0
        and int(report["protected_incremental_regressions_vs_zero"]) == 0
    )


@torch.no_grad()
def exact_bf16_multirow_scale_sweep(
    *,
    model: nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    unknown_token_id: int,
    active_cases: Sequence[mquake.PredictionCase],
    protected_cases: Sequence[mquake.PredictionCase],
    active_context_cases: Sequence[mquake.PredictionCase],
    protected_context_cases: Sequence[mquake.PredictionCase],
    scales: Sequence[float],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    minimum_active_margin: float,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    """Apply each scale jointly from immutable Setting-5e rows."""

    normalized = sorted({float(value) for value in scales}, reverse=True)
    if 0.0 not in normalized:
        raise ValueError("The exact scale sweep must include 0.0")

    def evaluate_context(
        context: Sequence[mquake.PredictionCase], label: str
    ) -> List[repair.TokenLogitCache]:
        return repair.cache_prediction_cases(
            model,
            tok,
            context,
            neutral_token_id=unknown_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=label,
        )

    materialize_multirow_scale(
        output_weight, row_ids, original_rows, delta_rows, 0.0
    )
    zero_active = _select_caches(
        evaluate_context(active_context_cases, "exact zero forget context"),
        active_cases,
    )
    zero_protected = _select_caches(
        evaluate_context(protected_context_cases, "exact zero retain context"),
        protected_cases,
    )
    zero_protected_correct = [bool(row.correct) for row in zero_protected]
    reports: List[Dict[str, Any]] = []
    for scale in normalized:
        materialize_multirow_scale(
            output_weight, row_ids, original_rows, delta_rows, scale
        )
        if scale == 0.0:
            active_rows, protected_rows = zero_active, zero_protected
        else:
            active_rows = _select_caches(
                evaluate_context(
                    active_context_cases, f"exact candidate forget scale={scale:g}"
                ),
                active_cases,
            )
            protected_rows = _select_caches(
                evaluate_context(
                    protected_context_cases,
                    f"exact candidate retain scale={scale:g}",
                ),
                protected_cases,
            )
        margins = [
            float((row.neutral_logit - row.target_logit).detach().cpu())
            for row in active_rows
        ]
        protected_correct = [bool(row.correct) for row in protected_rows]
        materialized = output_weight.index_select(
            0, torch.tensor(row_ids, device=output_weight.device)
        ).float() - original_rows.float()
        reports.append(
            {
                "scale": float(scale),
                "active_total_tokens": len(active_rows),
                "active_correct_tokens": int(sum(row.correct for row in active_rows)),
                "active_margin_violations": int(
                    sum(value < minimum_active_margin for value in margins)
                ),
                "minimum_active_unknown_minus_sensitive_margin": (
                    min(margins) if margins else None
                ),
                "protected_total_tokens": len(protected_rows),
                "protected_incremental_regressions_vs_zero": int(
                    sum(
                        before and not after
                        for before, after in zip(
                            zero_protected_correct, protected_correct
                        )
                    )
                ),
                "joint_materialized_delta_norm": float(materialized.norm().cpu()),
                "all_rows_applied_jointly": True,
            }
        )

    eligible = [
        row for row in reports if scale_report_is_locally_safe(row)
    ]
    selected = min(
        eligible or [row for row in reports if row["scale"] == 0.0],
        key=lambda row: (
            float(row["joint_materialized_delta_norm"]),
            float(row["scale"]),
        ),
    )
    selected_scale = float(selected["scale"])
    materialize_multirow_scale(
        output_weight, row_ids, original_rows, delta_rows, selected_scale
    )
    zero_baseline = {
        "active_correct_tokens_at_zero": int(sum(row.correct for row in zero_active)),
        "active_total_tokens": len(zero_active),
        "protected_correct_tokens_at_zero": int(
            sum(row.correct for row in zero_protected)
        ),
        "protected_total_tokens": len(zero_protected),
    }
    return selected_scale, reports, zero_baseline


def candidate_acceptance_report(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    scale_report: Mapping[str, Any],
    *,
    retain_tolerance: float = 0.10,
    max_ppl_ratio: float = 1.02,
) -> Dict[str, Any]:
    """Frozen selection rule; held-out AtomicGen/MH metrics are not inputs."""

    base_retain = base_result["retain"].get("Eff")
    candidate_retain = candidate_result["retain"].get("Eff")
    base_ppl = base_result.get("forget_PPL")
    candidate_ppl = candidate_result.get("forget_PPL")
    checks = {
        "forget_Eff_exactly_zero": float(candidate_result["forget"]["Eff"]) == 0.0,
        "retain_Eff_within_base_tolerance": (
            base_retain is not None
            and candidate_retain is not None
            and float(candidate_retain) >= float(base_retain) - retain_tolerance
        ),
        "PPL_within_base_ratio": (
            base_ppl is not None
            and candidate_ppl is not None
            and float(candidate_ppl) <= float(base_ppl) * max_ppl_ratio
        ),
        "no_incremental_protected_regression": int(
            scale_report["protected_incremental_regressions_vs_zero"]
        )
        == 0,
        "active_token_constraints_satisfied": scale_report_is_locally_safe(
            scale_report
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "forget_Eff": 0.0,
            "retain_Eff_minimum": (
                None if base_retain is None else float(base_retain) - retain_tolerance
            ),
            "PPL_maximum": (
                None if base_ppl is None else float(base_ppl) * max_ppl_ratio
            ),
            "protected_incremental_regressions": 0,
        },
    }


def select_candidate(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    scale_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Selection deliberately has no AtomicGen or multi-hop parameter."""

    return candidate_acceptance_report(base_result, candidate_result, scale_report)


def _row_reports(
    row_ids: Sequence[int],
    delta_rows: torch.Tensor,
    tok: Any,
    unknown_token_id: int,
    counts: Mapping[int, int],
) -> List[Dict[str, Any]]:
    return [
        {
            "token_id": int(token_id),
            "decoded_token": tok.decode([int(token_id)]),
            "role": "Unknown" if token_id == unknown_token_id else "residual_sensitive",
            "active_constraint_count": int(counts.get(int(token_id), 0)),
            "delta_norm": float(delta_rows[index].float().norm().cpu()),
        }
        for index, token_id in enumerate(row_ids)
    ]


def _evaluate_multihop_post_selection(
    *,
    model: nn.Module,
    tok: Any,
    mquake_path: Path,
    split_manifest: Path,
    prompt_dir: Path,
    args: argparse.Namespace,
    out_path: Path,
) -> Dict[str, Any]:
    instances, manifest = multihop.load_forget_instances(
        mquake_path, split_manifest
    )
    standard_path = multihop.download_text(
        prompt_dir / "multihop-prompts.txt", multihop.STANDARD_PROMPT_URL
    )
    cot_path = multihop.download_text(
        prompt_dir / "multihop-cot-prompts.txt", multihop.COT_PROMPT_URL
    )
    results: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    for mode, path, max_new in (
        ("standard", standard_path, args.standard_max_new_tokens),
        ("cot", cot_path, args.cot_max_new_tokens),
    ):
        summary, rows = multihop.evaluate_mode(
            model=model,
            tok=tok,
            instances=instances,
            task_prompt=path.read_text(encoding="utf-8"),
            mode=mode,
            batch_size=args.multihop_batch_size,
            max_new_tokens=max_new,
        )
        results[mode] = summary
        raw[mode] = rows
    payload = {
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": manifest.get("seed"),
        "checkpoint_selection": "completed before this evaluation",
        "results": results,
        "raw": raw,
    }
    gagd.write_json(out_path, payload)
    return payload


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting5_dir = output_dir / "setting5e"
    repair_dir = output_dir / "multirow_active_repair"
    setting5_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)
    mquake_path = Path(args.mquake_path)
    if not mquake_path.is_absolute():
        mquake_path = gagd.PROJECT_DIR / mquake_path
    mquake_path = mquake.download_mquake(mquake_path, url=args.mquake_url)
    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)

    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "dataset_revision": mquake.MQUAKE_REV,
            "training_source": "sampled requested_rewrite cloze facts only",
            "repair_source": ACTIVE_SOURCE,
            "evaluation_only_until_selection": [
                "requested_rewrite.question",
                "record questions[0:3]",
                "answer/new_answer and aliases",
                "AtomicGen",
                "standard and CoT multi-hop leakage",
            ],
            "counterfactual_target_new_is_training_target": False,
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading Base model and pinned MQuAKE split")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    forget_records, retain_records = mquake.load_official_eval_records(
        mquake_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        mquake_url=args.mquake_url,
    )
    records = (forget_records, retain_records)
    split_manifest = output_dir / "split_manifest.json"
    mquake.write_split_manifest(
        split_manifest,
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)
    base_result = baseline.evaluate_eff_only(
        method="Base",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "base_official_eval.json",
        args=args,
        records=records,
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training unchanged 600-step Setting 5e objective")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    if mquake.resolve_neutral_target_token_id(tok) != neutral_token_id:
        raise RuntimeError("Neutral token ID changed between model loads")
    forget_examples, sampling_report = setting5_forget_examples(
        forget_records,
        tok,
        strategy=args.forget_sampling,
        steps=args.steps,
        seed=args.seed,
    )
    retain_examples = baseline.canonical_examples(retain_records, tok)
    gagd.write_json(setting5_dir / "forget_sampling.json", sampling_report)
    args.post_training_excluded_token_ids = [neutral_token_id]
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
    setting5_result = baseline.evaluate_eff_only(
        method="Setting 5e",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=setting5_dir / "official_eval.json",
        args=args,
        records=records,
    )
    if args.save_setting5_checkpoint:
        baseline.save_checkpoint(model, tok, setting5_dir / "checkpoint")

    print("Caching cloze-only residual active and protected retain states")
    output_layer = freeze_model_for_multirow_repair(model)
    device = next(model.parameters()).device
    llama_like = mquake.is_llama_like(model, tok)
    forget_cases = build_repair_cases(forget_records, tok, llama_like=llama_like)
    forget_caches = repair.cache_prediction_cases(
        model,
        tok,
        forget_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache MQuAKE residual cloze tokens",
    )
    active_caches = residual_active_caches(forget_caches, neutral_token_id)
    row_ids = active_row_ids(active_caches, neutral_token_id)
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=output_layer.weight.device)
    original_rows = output_layer.weight.index_select(0, row_tensor).detach().clone()

    calibration_records = sample_retain_instances(
        retain_records,
        args.retain_calibration_num,
        args.retain_calibration_seed,
    )
    retain_cases = build_repair_cases(
        calibration_records, tok, llama_like=llama_like
    )
    retain_caches = repair.cache_prediction_cases(
        model,
        tok,
        retain_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache MQuAKE retain calibration tokens",
    )
    protected_caches = [cache for cache in retain_caches if cache.correct]
    repair.write_jsonl(
        repair_dir / "active_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in active_caches],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in protected_caches],
    )

    delta_rows, repair_logs, optimization = optimize_multirow_delta(
        active_caches,
        protected_caches,
        row_ids=row_ids,
        unknown_token_id=neutral_token_id,
        hidden_size=output_layer.weight.shape[1],
        device=device,
        args=args,
    )
    for row in repair_logs:
        row["active_source"] = ACTIVE_SOURCE
    repair.write_jsonl(repair_dir / "multirow_repair_log.jsonl", repair_logs)
    counts = constraints_per_sensitive_row(active_caches)
    counts[int(neutral_token_id)] = len(active_caches)
    rows_report = _row_reports(
        row_ids, delta_rows, tok, neutral_token_id, counts
    )
    gagd.write_json(repair_dir / "active_rows.json", rows_report)
    gagd.write_json(repair_dir / "row_delta_norms.json", rows_report)

    protected_cases = [cache.case for cache in protected_caches]
    selected_scale, scale_reports, zero_baseline = exact_bf16_multirow_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        row_ids=row_ids,
        original_rows=original_rows,
        delta_rows=delta_rows,
        unknown_token_id=neutral_token_id,
        active_cases=[cache.case for cache in active_caches],
        protected_cases=protected_cases,
        active_context_cases=forget_cases,
        protected_context_cases=retain_cases,
        scales=repair.parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    gagd.write_json(
        repair_dir / "bf16_exact_multirow_scale_sweep.json", scale_reports
    )
    selected_scale_report = next(
        row for row in scale_reports if float(row["scale"]) == selected_scale
    )

    candidate_result = baseline.evaluate_eff_only(
        method=METHOD_LABEL + " candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:multirow-candidate",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=repair_dir / "candidate_official_eval.json",
        args=args,
        records=records,
    )
    gate = select_candidate(base_result, candidate_result, selected_scale_report)
    accepted = bool(gate["accepted"])
    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selection_reason = "candidate_passed_exact_zero_eff_base_retain_ppl_and_protection_gates"
    else:
        materialize_multirow_scale(
            output_layer.weight, row_ids, original_rows, delta_rows, 0.0
        )
        selected_scale = 0.0
        selected_result = copy.deepcopy(setting5_result)
        selection_reason = "candidate_rejected_and_setting5e_exactly_restored"

    selection_commit = {
        "selection_irrevocable": True,
        "candidate_accepted": accepted,
        "selected_scale": float(selected_scale),
        "selection_reason": selection_reason,
        "held_out_metrics_observed": False,
        "atomic_gen_used_for_selection": False,
        "multihop_used_for_selection": False,
    }
    gagd.write_json(output_dir / "selection_commit.json", selection_commit)
    if args.save_selected_checkpoint:
        baseline.save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    # The checkpoint decision is now durable.  Only now open evaluation-only
    # atomic questions and record-level standard/CoT multi-hop questions.
    selected_extension = baseline.evaluate_extension(
        method=METHOD_LABEL + " post-selection AtomicGen",
        model=model,
        tok=tok,
        model_dir="in-memory:selected",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "selected_atomic_gen_eval.json",
        args=args,
        records=records,
    )
    multihop_result = _evaluate_multihop_post_selection(
        model=model,
        tok=tok,
        mquake_path=mquake_path,
        split_manifest=split_manifest,
        prompt_dir=gagd.resolve_output_path(args.multihop_prompt_dir),
        args=args,
        out_path=output_dir / "multihop_unlearning_eval.json",
    )

    final_materialized_delta = (
        output_layer.weight.index_select(0, row_tensor).detach().float()
        - original_rows.detach().float()
    )
    actual_modified_rows = int(
        (final_materialized_delta.abs().sum(dim=1) != 0).sum().item()
    )

    repair_summary = {
        "method": METHOD_LABEL,
        "active_source": ACTIVE_SOURCE,
        "number_of_modified_rows": actual_modified_rows,
        "candidate_modified_row_count": len(row_ids),
        "Unknown_row_ID": int(neutral_token_id),
        "sensitive_row_IDs": [int(value) for value in row_ids[1:]],
        "active_constraints_per_row": {str(k): v for k, v in counts.items()},
        "delta_norm_per_row": {
            str(row["token_id"]): row["delta_norm"] for row in rows_report
        },
        "selected_BF16_scale": float(selected_scale),
        "active_failures_before": len(active_caches),
        "active_failures_after": int(
            selected_scale_report["active_correct_tokens"] if accepted else len(active_caches)
        ),
        "protected_regressions_before": 0,
        "protected_regressions_after": int(
            selected_scale_report["protected_incremental_regressions_vs_zero"]
            if accepted
            else 0
        ),
        "Eff_before": setting5_result["forget"]["Eff"],
        "Eff_after": selected_result["forget"]["Eff"],
        "retain_Eff_before": setting5_result["retain"]["Eff"],
        "retain_Eff_after": selected_result["retain"]["Eff"],
        "PPL_before": setting5_result.get("forget_PPL"),
        "PPL_after": selected_result.get("forget_PPL"),
        "candidate_accepted": accepted,
        "candidate_rejected": not accepted,
        "candidate_reason": selection_reason,
        "gates": gate,
        "optimization": optimization,
        "zero_scale_baseline": zero_baseline,
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "official_atomic_questions_used_for_repair": False,
        "official_multihop_questions_used_for_repair": False,
    }
    gagd.write_json(repair_dir / "repair_summary.json", repair_summary)

    forget_extension = selected_extension["forget"]
    retain_extension = selected_extension["retain"]
    multihop_summaries = multihop_result["results"]
    reporting = {
        "Eff": selected_result["forget"].get("Eff"),
        "Eff_micro": selected_result["forget"].get("Eff_micro"),
        "Eff_instance_macro": selected_result["forget"].get(
            "Eff_instance_macro"
        ),
        "AtomicGen": forget_extension.get("AtomicGen"),
        "AtomicGen_micro": forget_extension.get("AtomicGen_micro"),
        "AtomicGen_instance_macro": forget_extension.get(
            "AtomicGen_instance_macro"
        ),
        "RetainEff": selected_result["retain"].get("Eff"),
        "RetainAtomicGen": retain_extension.get("AtomicGen"),
        "PPL": selected_result.get("forget_PPL"),
        "MHLeak_exact_any": {
            mode: summary.get("MHLeak_exact_any")
            for mode, summary in multihop_summaries.items()
        },
        "MHLeak_contains_any": {
            mode: summary.get("MHLeak_contains_any")
            for mode, summary in multihop_summaries.items()
        },
        "MHLeak_by_hop": {
            str(hop): {
                mode: summary.get("by_hop", {}).get(str(hop))
                for mode, summary in multihop_summaries.items()
            }
            for hop in (2, 3, 4)
        },
    }

    result = {
        "method": METHOD_LABEL,
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": int(args.seed),
        "training": {
            **asdict(train_summary),
            "forget_sampling": sampling_report,
            "steps": int(args.steps),
        },
        "repair": repair_summary,
        "base": baseline.compact_metrics(base_result),
        "setting5e": baseline.compact_metrics(setting5_result),
        "candidate": baseline.compact_metrics(candidate_result),
        "selected": baseline.compact_metrics(selected_result),
        "selected_extension": baseline.compact_metrics(selected_extension),
        "multihop": multihop_result["results"],
        "reporting": reporting,
        "selection": selection_commit,
    }
    gagd.write_json(output_dir / "mquake_results.json", result)
    print(
        f"Selected Eff={selected_result['forget']['Eff']}; "
        f"AtomicGen={selected_extension['forget'].get('AtomicGen')}; "
        f"RetainEff={selected_result['retain'].get('Eff')}; "
        f"PPL={selected_result.get('forget_PPL')}; accepted={accepted}"
    )
    if args.fail_if_target_missed and not accepted:
        raise RuntimeError("No multi-row candidate passed every fixed gate")
    if args.require_atomic_gen_zero:
        atomic_gen = selected_extension["forget"].get("AtomicGen")
        if atomic_gen is None or float(atomic_gen) > 0.0:
            raise RuntimeError(
                "Post-selection AtomicGen was not zero; the checkpoint remains "
                f"unchanged by this diagnostic (AtomicGen={atomic_gen})"
            )


if __name__ == "__main__":
    main()
