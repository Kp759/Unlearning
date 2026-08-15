#!/usr/bin/env python3
"""RWKU Batch-50: unrestricted hard forgetting + rank-64/128 nullspace restore.

This is a probe-assisted cross-benchmark development method.  It deliberately
separates the problem into two stages:

Stage 1 (hard forget)
  * start from the frozen Base model (no Setting-5 pretraining damage);
  * jointly target the same 50 Batch-50 forget probes;
  * learn an unrestricted selected-row LM-head delta ("rank 0" in the existing
    active-repair terminology);
  * increase the required answer-vs-EOS margin until post-materialization
    generation recovery on the same 50 training probes is exactly 0%.

Stage 2 (utility restore)
  * build B_F from every sensitive/neutral hidden state used by the 50 forget
    constraints;
  * project retain hidden states into B_F^perp and obtain an ordered restoration
    basis B_R;
  * independently optimize rank-64 and rank-128 row coefficients in span(B_R)
    against the 1000 deterministic MCF retain examples;
  * use the disjoint 128-example MCF gate only for snapshot selection;
  * reject/rollback any update that weakens the Stage-1 forget-margin floor;
  * only select snapshots whose exact same-50 generation recovery remains 0%.

The 1000 retain examples therefore play the same train+headline-eval role
requested for MCF/ZsRE-style comparison.  The 128 gate examples never receive
gradients.  No model checkpoint is written; only metrics/manifests/logs are
saved.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import rwku_batch50_experiment as BATCH
import rwku_experiment as EXP
import rwku_generated_s5e_rank2_active_repair as rank2
from rwku_batch50 import (
    REPAIR_RETAIN_NUM,
    RETAIN_EVAL_NUM,
    TOTAL_FORGET_TRAIN,
    materialize_batch_split,
)
from rwku_eval import evaluate_qa_rows, load_wikidata_text


SCRIPT_PATH = Path(__file__).resolve()
PROTOCOL_ID = "RWKU-Batch-50-Rank0-NullRestore-v1"
PROTOCOL_STATUS = "probe_assisted_cross_benchmark_development_method"
METHOD_BASE = "Base model"
METHOD_STAGE1 = "Rank-0 hard forget (Stage 1)"

DEFAULT_MARGIN_SCHEDULE = (0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_RESTORE_RANKS = (64, 128)


def parse_float_list(value: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive floats")
    if tuple(sorted(set(values))) != values:
        raise argparse.ArgumentTypeError("margin schedule must be strictly increasing and unique")
    return values


def parse_rank_list(value: str) -> Tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("restore ranks must be positive integers")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("restore ranks contain duplicates")
    return tuple(sorted(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=range(10))
    parser.add_argument("--model-path", default=EXP.DEFAULT_MODEL_PATH)
    parser.add_argument("--data-root", type=Path, default=EXP.DEFAULT_DATA_ROOT)
    parser.add_argument("--mcf-path", type=Path, default=EXP.DEFAULT_MCF_PATH)
    parser.add_argument("--wikidata-dir", type=Path, default=EXP.DEFAULT_WIKIDATA_DIR)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXP.SEMANTIC_ROOT / "outputs" / "rwku_batch50_rank0_nullrestore",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--retain-num", type=int, default=RETAIN_EVAL_NUM)
    parser.add_argument("--repair-retain-num", type=int, default=REPAIR_RETAIN_NUM)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--hard-margin-schedule",
        type=parse_float_list,
        default=DEFAULT_MARGIN_SCHEDULE,
        help="Increasing margin schedule used until same-50 generation recovery reaches 0%%.",
    )
    parser.add_argument("--hard-steps-per-margin", type=int, default=200)
    parser.add_argument("--hard-lr", type=float, default=5e-3)
    parser.add_argument("--hard-tail-weight", type=float, default=2.0)
    parser.add_argument("--hard-l2-lambda", type=float, default=1e-4)

    parser.add_argument(
        "--restore-ranks",
        type=parse_rank_list,
        default=DEFAULT_RESTORE_RANKS,
        help="Comma-separated nullspace restoration ranks (default: 64,128).",
    )
    parser.add_argument("--restore-steps", type=int, default=800)
    parser.add_argument("--restore-lr", type=float, default=5e-4)
    parser.add_argument("--restore-l2-lambda", type=float, default=1e-4)
    parser.add_argument("--restore-batch-size", type=int, default=64)
    parser.add_argument("--restore-snapshot-interval", type=int, default=100)
    parser.add_argument(
        "--restore-basis-retain-num",
        type=int,
        default=512,
        help="Number of the 1000 retain examples used only to construct B_R; all 1000 train the coefficients.",
    )
    parser.add_argument(
        "--margin-preservation-tolerance",
        type=float,
        default=1e-4,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.retain_num != RETAIN_EVAL_NUM:
        raise ValueError(f"{PROTOCOL_ID} fixes --retain-num={RETAIN_EVAL_NUM}")
    if args.repair_retain_num != REPAIR_RETAIN_NUM:
        raise ValueError(f"{PROTOCOL_ID} fixes --repair-retain-num={REPAIR_RETAIN_NUM}")
    for name in (
        "eval_batch_size",
        "hard_steps_per_margin",
        "restore_steps",
        "restore_batch_size",
        "restore_snapshot_interval",
        "restore_basis_retain_num",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.restore_basis_retain_num > args.retain_num:
        raise ValueError("--restore-basis-retain-num cannot exceed --retain-num")
    for name in ("hard_lr", "hard_tail_weight", "restore_lr"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("hard_l2_lambda", "restore_l2_lambda", "margin_preservation_tolerance"):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not args.dry_run and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; use --dry-run for CPU preflight")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    EXP.write_json(path, value)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _instances_from_examples(examples: Sequence[gagd.Example]) -> List[active.MCFPromptInstance]:
    instances: List[active.MCFPromptInstance] = []
    for index, example in enumerate(examples):
        instances.append(
            active.MCFPromptInstance(
                record_index=index,
                sampled_position=index,
                prompt_type="rwku_batch50_train",
                prompt_index=0,
                prompt=str(example.prompt),
                target_new=str(example.target_new),
                target_true=str(example.target_true),
            )
        )
    return instances


def _selected_output_rows(tokenizer: Any, instances: Sequence[active.MCFPromptInstance]) -> List[int]:
    selected: set[int] = set()
    for instance in instances:
        for value in (instance.target_new, instance.target_true):
            selected.update(
                gagd.token_ids_for_text(tokenizer, gagd.normalize_answer(str(value)))
            )
    selected -= gagd.special_token_ids(tokenizer)
    rows = sorted(selected)
    if not rows:
        raise ValueError("No non-special sensitive/neutral output rows were selected")
    return rows


def _materialize_delta(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    anchor_rows: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    ids = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    anchor = anchor_rows.to(device=output_weight.device, dtype=output_weight.dtype)
    update = delta.to(device=output_weight.device, dtype=output_weight.dtype)
    if anchor.shape != update.shape:
        raise ValueError("anchor/delta shape mismatch")
    with torch.no_grad():
        output_weight.index_copy_(0, ids, anchor + update)
        actual = output_weight.index_select(0, ids).float() - anchor_rows.to(
            device=output_weight.device, dtype=torch.float32
        )
    return actual


def _margin_summary(margins: torch.Tensor, required_margin: float) -> Dict[str, Any]:
    violations = int((margins < required_margin).sum().item()) if margins.numel() else 0
    return {
        "required_margin": float(required_margin),
        "violation_count": violations,
        "minimum_margin": float(margins.min().item()) if margins.numel() else math.inf,
        "mean_margin": float(margins.mean().item()) if margins.numel() else math.inf,
        "maximum_margin": float(margins.max().item()) if margins.numel() else math.inf,
    }


def project_delta_to_span(delta_rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    delta = delta_rows.float()
    basis = basis.float()
    if basis.numel() == 0:
        return torch.zeros_like(delta)
    return (delta @ basis.transpose(0, 1)) @ basis


def build_restore_basis(
    retain_hidden: torch.Tensor,
    forget_basis: torch.Tensor,
    *,
    max_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    null_hidden = active.project_rows_away(retain_hidden.float(), forget_basis.float())
    basis = active.orthonormal_row_basis(null_hidden, max_rank=max_rank)
    return null_hidden, basis


def _hard_tail_optimize(
    caches: Sequence[active.RewriteDeltaCache],
    initial_delta: torch.Tensor,
    *,
    required_margin: float,
    steps: int,
    learning_rate: float,
    hard_tail_weight: float,
    l2_lambda: float,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    parameter = nn.Parameter(initial_delta.detach().float().clone())
    optimizer = torch.optim.AdamW([parameter], lr=learning_rate)
    logs: List[Dict[str, Any]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        margins = active.margins_from_delta_caches(caches, parameter)
        violations = torch.relu(required_margin - margins)
        if violations.numel() == 0 or bool(torch.all(violations <= 0).item()):
            break
        hard_tail = violations.max().square()
        l2 = parameter.square().sum()
        loss = hard_tail_weight * hard_tail + l2_lambda * l2
        loss.backward()
        optimizer.step()
        after = active.margins_from_delta_caches(caches, parameter.detach())
        logs.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "hard_tail_loss": float(hard_tail.detach().cpu()),
                "delta_l2": float(l2.detach().cpu()),
                **_margin_summary(after, required_margin),
            }
        )
        if bool(torch.all(after >= required_margin).item()):
            break
    return parameter.detach().float(), logs


def _efficacy_gate(
    model: nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Dict[str, Any]:
    summary, _ = evaluate_qa_rows(
        model,
        tokenizer,
        rows,
        batch_size=batch_size,
        score_answers=False,
    )
    return {
        "count": int(summary["count"]),
        "recovery_accuracy": float(summary["recovery_accuracy"]),
        "rouge_l_recall": float(summary["rouge_l_recall"]),
    }


def _cyclic_batch(values: Sequence[Any], step: int, size: int) -> List[Any]:
    if not values:
        return []
    size = min(size, len(values))
    start = (step * size) % len(values)
    return [values[(start + offset) % len(values)] for offset in range(size)]


def _optimize_restore_rank(
    *,
    rank: int,
    full_restore_basis: torch.Tensor,
    stage1_delta: torch.Tensor,
    stage1_margins: torch.Tensor,
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    anchor_rows: torch.Tensor,
    forget_caches: Sequence[active.RewriteDeltaCache],
    retain_caches: Sequence[active.RetainKLCache],
    gate_caches: Sequence[active.RetainKLCache],
    model: nn.Module,
    tokenizer: Any,
    efficacy_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    actual_rank = min(rank, int(full_restore_basis.shape[0]))
    if actual_rank <= 0:
        raise RuntimeError("Nullspace restoration basis has rank zero")
    basis = full_restore_basis[:actual_rank]
    module = active.SelectedRowDelta(
        len(selected_ids),
        output_weight.shape[1],
        direction_basis=basis,
        device=output_weight.device,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.restore_lr)
    floor = torch.maximum(
        torch.full_like(stage1_margins, float(args.hard_margin_schedule[0])),
        stage1_margins - float(args.margin_preservation_tolerance),
    )
    logs: List[Dict[str, Any]] = []
    snapshots: List[Tuple[int, torch.Tensor, Dict[str, Any]]] = []

    for step in range(args.restore_steps):
        optimizer.zero_grad(set_to_none=True)
        restore = module.effective_delta()
        final_delta = stage1_delta + restore
        batch = _cyclic_batch(retain_caches, step, args.restore_batch_size)
        retain_kl = active.retain_kl_from_caches(batch, final_delta)
        l2 = restore.square().sum()
        loss = retain_kl + args.restore_l2_lambda * l2
        loss.backward()
        previous = module.coefficients.detach().clone()
        optimizer.step()

        with torch.no_grad():
            candidate_restore = module.effective_delta()
            candidate_delta = stage1_delta + candidate_restore
            margins = active.margins_from_delta_caches(forget_caches, candidate_delta)
            rolled_back = bool(torch.any(margins < floor).item())
            if rolled_back:
                module.coefficients.copy_(previous)
                for state in optimizer.state.values():
                    for value in state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()
                candidate_restore = module.effective_delta()
                candidate_delta = stage1_delta + candidate_restore
                margins = active.margins_from_delta_caches(forget_caches, candidate_delta)

        logs.append(
            {
                "step": step + 1,
                "batch_retain_kl": float(retain_kl.detach().cpu()),
                "restoration_l2": float(l2.detach().cpu()),
                "rolled_back_for_forgetting": rolled_back,
                "minimum_margin": float(margins.min().detach().cpu()),
            }
        )

        snapshot_due = (
            (step + 1) % args.restore_snapshot_interval == 0
            or step + 1 == args.restore_steps
        )
        if snapshot_due:
            candidate_cpu = candidate_delta.detach().cpu().float().clone()
            actual_delta = _materialize_delta(
                output_weight, selected_ids, anchor_rows, candidate_cpu
            )
            actual_margins = active.margins_from_delta_caches(forget_caches, actual_delta)
            efficacy = _efficacy_gate(
                model,
                tokenizer,
                efficacy_rows,
                batch_size=args.eval_batch_size,
            )
            gate_kl = active.retain_kl_from_caches(gate_caches, actual_delta)
            train_kl = active.retain_kl_from_caches(retain_caches, actual_delta)
            safe = (
                efficacy["recovery_accuracy"] == 0.0
                and bool(torch.all(actual_margins >= floor).item())
            )
            report = {
                "step": step + 1,
                "requested_rank": rank,
                "actual_rank": actual_rank,
                "same_50_efficacy": efficacy,
                "minimum_materialized_margin": float(actual_margins.min().detach().cpu()),
                "train_retain_kl_1000": float(train_kl.detach().cpu()),
                "selection_gate_kl_128": float(gate_kl.detach().cpu()),
                "safe_forget_gate": bool(safe),
                "restoration_norm": float((actual_delta - stage1_delta).norm().detach().cpu()),
            }
            snapshots.append((step + 1, actual_delta.detach().cpu().clone(), report))

    _write_jsonl(output_dir / f"restore_rank{rank}_training_log.jsonl", logs)
    safe_snapshots = [item for item in snapshots if item[2]["safe_forget_gate"]]
    if safe_snapshots:
        selected = min(
            safe_snapshots,
            key=lambda item: (
                float(item[2]["selection_gate_kl_128"]),
                float(item[2]["train_retain_kl_1000"]),
                int(item[0]),
            ),
        )
        selected_delta = selected[1]
        selected_report = dict(selected[2])
        selected_report["selection"] = "lowest disjoint-128 gate KL among perfect-forgetting snapshots"
    else:
        selected_delta = stage1_delta.detach().cpu().clone()
        selected_report = {
            "step": 0,
            "requested_rank": rank,
            "actual_rank": actual_rank,
            "safe_forget_gate": True,
            "selection": "fallback_to_stage1_no_safe_restoration_snapshot",
        }
    _write_json(
        output_dir / f"restore_rank{rank}_snapshot_sweep.json",
        {
            "requested_rank": rank,
            "actual_rank": actual_rank,
            "snapshots": [item[2] for item in snapshots],
            "selected": selected_report,
        },
    )
    return selected_delta.float(), selected_report


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_dir = Path(args.output_root) / f"batch_seed{args.seed:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split = materialize_batch_split(
        data_root=args.data_root,
        output_dir=output_dir / "split",
        batch_seed=args.seed,
        allow_download=not args.no_download,
    )
    if len(split["forget_train"]) != TOTAL_FORGET_TRAIN:
        raise RuntimeError("Batch-50 split must contain exactly 50 forget examples")

    config: Dict[str, Any] = {
        "status": "preflight" if args.dry_run else "running",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": PROTOCOL_STATUS,
        "batch_seed": args.seed,
        "target_seeds": split["manifest"]["target_seeds"],
        "subjects": [row["subject"] for row in split["manifest"]["targets"]],
        "stage1_source": "base_model",
        "forget_train_count": 50,
        "same_50_post_training_efficacy": True,
        "retain_train_and_headline_eval_count": args.retain_num,
        "disjoint_restore_selection_gate_count": args.repair_retain_num,
        "hard_margin_schedule": list(args.hard_margin_schedule),
        "hard_steps_per_margin": args.hard_steps_per_margin,
        "hard_lr": args.hard_lr,
        "hard_tail_weight": args.hard_tail_weight,
        "hard_l2_lambda": args.hard_l2_lambda,
        "restore_ranks": list(args.restore_ranks),
        "restore_steps": args.restore_steps,
        "restore_lr": args.restore_lr,
        "restore_l2_lambda": args.restore_l2_lambda,
        "restore_batch_size": args.restore_batch_size,
        "restore_basis_retain_num": args.restore_basis_retain_num,
        "checkpoint_policy": "no model checkpoints saved",
        "exact_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
    }
    _write_json(output_dir / "config_used.json", config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print(
            f"{PROTOCOL_ID} dry-run OK: seed={args.seed}, "
            f"targets={config['target_seeds']}, forget=50, retain=1000, ranks={args.restore_ranks}"
        )
        return

    all_retain_records, all_retain_examples = EXP.load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.retain_num + args.repair_retain_num,
    )
    retain_records = all_retain_records[: args.retain_num]
    gate_records = all_retain_records[args.retain_num :]
    retain_examples = all_retain_examples[: args.retain_num]
    gate_examples = all_retain_examples[args.retain_num :]
    if len(retain_examples) != RETAIN_EVAL_NUM or len(gate_examples) != REPAIR_RETAIN_NUM:
        raise RuntimeError("retain/gate partition counts differ from 1000/128")
    retain_hashes = {EXP.mapping_sha256(row) for row in retain_records}
    gate_hashes = {EXP.mapping_sha256(row) for row in gate_records}
    if retain_hashes & gate_hashes:
        raise RuntimeError("1000 retain and 128 selection-gate records overlap")

    dtype = EXP.dtype_from_name(args.dtype)
    wikidata_text = None if args.skip_ppl else load_wikidata_text(args.wikidata_dir)
    EXP.set_all_seeds(args.seed)
    model, tokenizer = EXP.load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    results: Dict[str, Any] = {}

    print("Evaluating untouched Base model")
    results[METHOD_BASE] = BATCH._evaluate_model(
        method=METHOD_BASE,
        model=model,
        tokenizer=tokenizer,
        split=split,
        retain_examples=retain_examples,
        args=args,
        wikidata_text=wikidata_text,
    )
    _write_json(output_dir / "base_model.json", results[METHOD_BASE])

    forget_examples = EXP.setting5_examples(tokenizer, split["forget_train"])
    instances = _instances_from_examples(forget_examples)
    selected_ids = _selected_output_rows(tokenizer, instances)
    output = active.freeze_model_for_output_repair(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("All model parameters must be frozen before rank-0 repair")
    device = output.weight.device
    ids = torch.tensor(selected_ids, dtype=torch.long, device=device)
    anchor_rows = output.weight.index_select(0, ids).detach().cpu().float().clone()
    llama_like = active.is_llama_like(model, tokenizer)
    forget_caches = active.build_prompt_instance_delta_caches(
        model,
        tokenizer,
        instances,
        selected_ids,
        device,
        args.eval_batch_size,
        llama_like,
    )

    print(
        f"Stage 1: unrestricted hard forgetting of all 50 probes; "
        f"selected LM-head rows={len(selected_ids)}"
    )
    delta = torch.zeros(
        (len(selected_ids), output.weight.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    stage1_log: List[Dict[str, Any]] = []
    stage1_gate: Optional[Dict[str, Any]] = None
    achieved_margin: Optional[float] = None
    for required_margin in args.hard_margin_schedule:
        delta, logs = _hard_tail_optimize(
            forget_caches,
            delta,
            required_margin=required_margin,
            steps=args.hard_steps_per_margin,
            learning_rate=args.hard_lr,
            hard_tail_weight=args.hard_tail_weight,
            l2_lambda=args.hard_l2_lambda,
        )
        for row in logs:
            row["margin_stage"] = float(required_margin)
        stage1_log.extend(logs)
        actual_delta = _materialize_delta(output.weight, selected_ids, anchor_rows, delta)
        actual_margins = active.margins_from_delta_caches(forget_caches, actual_delta)
        stage1_gate = _efficacy_gate(
            model,
            tokenizer,
            split["efficacy_forget"],
            batch_size=args.eval_batch_size,
        )
        print(
            f"  margin={required_margin:g}: same50 recovery="
            f"{stage1_gate['recovery_accuracy']:.2f}% min_margin="
            f"{float(actual_margins.min().detach().cpu()):.4f}"
        )
        delta = actual_delta.detach().float()
        if (
            stage1_gate["recovery_accuracy"] == 0.0
            and bool(torch.all(actual_margins >= required_margin).item())
        ):
            achieved_margin = float(required_margin)
            break
    _write_jsonl(output_dir / "stage1_hard_forget_log.jsonl", stage1_log)
    if stage1_gate is None or achieved_margin is None:
        raise RuntimeError(
            "Stage 1 failed to reach exact 0% same-50 generation recovery within the margin schedule"
        )
    stage1_delta = delta.detach().float()
    stage1_margins = active.margins_from_delta_caches(forget_caches, stage1_delta).detach()
    stage1_report = {
        "achieved_required_margin": achieved_margin,
        "same_50_efficacy": stage1_gate,
        "selected_output_row_count": len(selected_ids),
        "selected_output_rows": selected_ids,
        "delta_norm": float(stage1_delta.norm().detach().cpu()),
        **_margin_summary(stage1_margins, achieved_margin),
    }
    _write_json(output_dir / "stage1_hard_forget_summary.json", stage1_report)

    results[METHOD_STAGE1] = BATCH._evaluate_model(
        method=METHOD_STAGE1,
        model=model,
        tokenizer=tokenizer,
        split=split,
        retain_examples=retain_examples,
        args=args,
        wikidata_text=wikidata_text,
    )
    results[METHOD_STAGE1]["stage1"] = stage1_report
    _write_json(output_dir / "stage1_rank0_result.json", results[METHOD_STAGE1])

    # Build B_F from every sensitive and neutral answer hidden state.  Because
    # Stage 1 changes only the output layer, these hidden states remain valid.
    forget_hidden = torch.cat(
        [cache.target_new.hidden for cache in forget_caches]
        + [cache.target_true.hidden for cache in forget_caches],
        dim=0,
    ).float()
    forget_basis = active.orthonormal_row_basis(forget_hidden)

    # Retain caches are anchored at Base.  The 1000 examples receive gradients;
    # the disjoint 128 examples are selection-only.
    sampled_retain = rank2._mcf_sampled_records(retain_records, retain_examples)
    sampled_gate = rank2._mcf_sampled_records(gate_records, gate_examples)
    reference_weight, reference_bias = active.load_reference_output_layer(
        str(args.model_path), dtype
    )
    # Restore the exact Base rows while building caches so delta=0 means Base.
    _materialize_delta(
        output.weight,
        selected_ids,
        anchor_rows,
        torch.zeros_like(stage1_delta),
    )
    retain_caches = active.build_retain_kl_caches(
        model,
        reference_weight,
        reference_bias,
        tokenizer,
        sampled_retain,
        selected_ids,
        device,
    )
    gate_caches = active.build_retain_kl_caches(
        model,
        reference_weight,
        reference_bias,
        tokenizer,
        sampled_gate,
        selected_ids,
        device,
    )
    del reference_weight, reference_bias

    basis_caches = retain_caches[: args.restore_basis_retain_num]
    retain_hidden = torch.cat([cache.hidden for cache in basis_caches], dim=0).float()
    max_rank = max(args.restore_ranks)
    retain_null, full_restore_basis = build_restore_basis(
        retain_hidden,
        forget_basis,
        max_rank=max_rank,
    )
    overlap = (
        float(
            (full_restore_basis @ forget_basis.transpose(0, 1))
            .abs()
            .max()
            .detach()
            .cpu()
        )
        if full_restore_basis.numel() and forget_basis.numel()
        else 0.0
    )
    if overlap > 1e-4:
        raise RuntimeError(f"Restoration basis is not in B_F^perp: overlap={overlap}")
    _write_json(
        output_dir / "nullspace_bases.json",
        {
            "forget_hidden_shape": list(forget_hidden.shape),
            "forget_basis_rank": int(forget_basis.shape[0]),
            "retain_hidden_shape": list(retain_hidden.shape),
            "retain_null_shape": list(retain_null.shape),
            "requested_max_restore_rank": max_rank,
            "actual_restore_basis_rank": int(full_restore_basis.shape[0]),
            "max_forget_restore_basis_overlap": overlap,
            "basis_retain_example_count": args.restore_basis_retain_num,
            "restore_gradient_retain_example_count": len(retain_caches),
            "selection_gate_example_count": len(gate_caches),
        },
    )

    for rank in args.restore_ranks:
        print(f"Stage 2: optimizing nullspace restoration rank {rank}")
        _materialize_delta(output.weight, selected_ids, anchor_rows, stage1_delta)
        selected_delta, selection_report = _optimize_restore_rank(
            rank=rank,
            full_restore_basis=full_restore_basis,
            stage1_delta=stage1_delta,
            stage1_margins=stage1_margins,
            output_weight=output.weight,
            selected_ids=selected_ids,
            anchor_rows=anchor_rows,
            forget_caches=forget_caches,
            retain_caches=retain_caches,
            gate_caches=gate_caches,
            model=model,
            tokenizer=tokenizer,
            efficacy_rows=split["efficacy_forget"],
            args=args,
            output_dir=output_dir,
        )
        actual_delta = _materialize_delta(
            output.weight, selected_ids, anchor_rows, selected_delta
        )
        final_gate = _efficacy_gate(
            model,
            tokenizer,
            split["efficacy_forget"],
            batch_size=args.eval_batch_size,
        )
        if final_gate["recovery_accuracy"] != 0.0:
            raise RuntimeError(f"Rank-{rank} selected restoration violated exact same-50 forgetting")
        method = f"Rank-0 hard forget + NullRestore{rank}"
        result = BATCH._evaluate_model(
            method=method,
            model=model,
            tokenizer=tokenizer,
            split=split,
            retain_examples=retain_examples,
            args=args,
            wikidata_text=wikidata_text,
        )
        result["stage1"] = stage1_report
        result["restoration"] = {
            **selection_report,
            "selected_delta_norm": float(actual_delta.norm().detach().cpu()),
            "same_50_efficacy_after_full_materialization": final_gate,
        }
        results[method] = result
        _write_json(output_dir / f"rank0_nullrestore{rank}_result.json", result)

    config["status"] = "complete"
    config["stage1_achieved_margin"] = achieved_margin
    combined = {**config, "results": results}
    _write_json(output_dir / "config_used.json", config)
    _write_json(output_dir / "results.json", combined)
    print(f"{PROTOCOL_ID} seed {args.seed} complete: {output_dir / 'results.json'}")

    EXP.release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
