#!/usr/bin/env python3
"""Repair only hard MCF rewrite cases in an already-trained Setting 5 checkpoint.

This runner never reruns GA/GD.  The supplied checkpoint is treated as the
immutable starting point, the transformer and input embeddings remain frozen,
and only explicitly selected output ``lm_head`` vocabulary rows may change.

Three repair modes are available:

* ``true_scale`` sets active records' globally unique target-true output rows
  to a scaled base-model row.
* ``extrapolate_delta`` extrapolates active records' globally unique
  target-new output rows beyond the update already present in the input
  checkpoint.
* ``minimal_optimize`` learns a sparse output-row delta under a squared-hinge
  rewrite-margin constraint, an explicit delta norm, and an optional
  retain-prompt KL guard.

Official paraphrases are never used to select or optimize a repair.  They are
only touched when the optional final official evaluation is requested.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

import gagd_compare as gagd


METHOD = "gagd_active_case_repair"
ALPHA_KEYS = (
    "post_training_new_true_alpha",
    "post_training_new_retain_alpha",
    "post_training_new_true_retain_alpha",
)
GROUP_NAMES = (
    "unique_target_new",
    "unique_target_true",
    "target_new_true_overlap",
    "target_new_retain_overlap",
    "target_new_true_retain_overlap",
    "target_true_retain_overlap",
)


@dataclass(frozen=True)
class SampledMCFRecord:
    record_index: int
    sampled_position: int
    example: gagd.Example


@dataclass
class AnswerDeltaCache:
    base_token_nll: torch.Tensor
    hidden: torch.Tensor
    selected_probs: torch.Tensor
    target_selected_columns: torch.Tensor


@dataclass
class RewriteDeltaCache:
    target_new: AnswerDeltaCache
    target_true: AnswerDeltaCache


@dataclass
class RetainKLCache:
    hidden: torch.Tensor
    candidate_selected_probs: torch.Tensor
    reference_selected_probs: torch.Tensor
    baseline_kl: torch.Tensor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument(
        "--reference-model-path",
        default=None,
        help=(
            "Optional original 5e checkpoint for retain KL. It must share the "
            "same frozen transformer and input embeddings as --model-path; "
            "only its output layer is loaded."
        ),
    )
    parser.add_argument(
        "--experiment-config-path",
        default=None,
        help=(
            "Setting 5 experiment config. If omitted, config_used.json is "
            "discovered beside or above --model-path. The run fails rather "
            "than assuming an overlap-alpha tuple."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mcf-cache-path", required=True)
    parser.add_argument(
        "--sample-mode",
        choices=["official", "first", "shuffled"],
        default="official",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--repair-mode",
        choices=["true_scale", "extrapolate_delta", "minimal_optimize"],
        required=True,
    )
    parser.add_argument("--active-margin", type=float, default=0.1)
    parser.add_argument("--target-true-scale", type=float, default=1.50)
    parser.add_argument("--target-new-gamma", type=float, default=1.25)
    parser.add_argument("--repair-steps", type=int, default=50)
    parser.add_argument("--repair-lr", type=float, default=1e-2)
    parser.add_argument(
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
    )
    parser.add_argument("--hinge-weight", type=float, default=1.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    parser.add_argument("--retain-kl-mu", type=float, default=0.0)
    parser.add_argument("--retain-calibration-num", type=int, default=32)
    parser.add_argument("--retain-calibration-seed", type=int, default=1729)
    parser.add_argument(
        "--repair-rank",
        type=int,
        default=0,
        help=(
            "0 learns unrestricted selected-row deltas. A positive value "
            "restricts deltas to that rank in active rewrite hidden directions."
        ),
    )
    parser.add_argument(
        "--project-away-retain-hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--run-official-mcf-eval", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")

    # Compatible loading/evaluation controls.
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--margin-batch-size", type=int, default=4)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--mcf-url", default=gagd.MCF_URL)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("--forget-num and --retain-num must be positive")
    if args.active_margin < 0:
        raise ValueError("--active-margin must be non-negative")
    if args.target_true_scale <= 0:
        raise ValueError("--target-true-scale must be positive")
    if args.target_new_gamma < 0:
        raise ValueError("--target-new-gamma must be non-negative")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("--repair-steps and --repair-lr must be positive")
    if args.hinge_weight <= 0:
        raise ValueError("--hinge-weight must be positive")
    if args.delta_l2_lambda < 0 or args.retain_kl_mu < 0:
        raise ValueError("repair regularization weights must be non-negative")
    if args.retain_calibration_num < 0:
        raise ValueError("--retain-calibration-num must be non-negative")
    if args.repair_rank < 0:
        raise ValueError("--repair-rank must be non-negative")
    if args.margin_batch_size <= 0:
        raise ValueError("--margin-batch-size must be positive")
    if (
        args.repair_mode == "minimal_optimize"
        and (args.retain_kl_mu > 0 or args.project_away_retain_hidden)
        and args.retain_calibration_num <= 0
    ):
        raise ValueError(
            "retain calibration must be non-empty when KL or hidden projection is enabled"
        )
    if args.run_official_mcf_eval and args.sample_mode == "shuffled":
        raise ValueError(
            "Official evaluation supports only official or first sampling, not shuffled"
        )


def margin_from_nll(target_new_nll: Any, target_true_nll: Any) -> Any:
    """Positive means the model prefers target_true, as desired after unlearning."""
    return target_new_nll - target_true_nll


def squared_hinge_loss(
    margins: torch.Tensor,
    active_margin: float,
) -> torch.Tensor:
    return torch.relu(margins.new_tensor(active_margin) - margins).square().sum()


def all_margins_satisfied(margins: torch.Tensor, active_margin: float) -> bool:
    if margins.numel() == 0:
        return True
    return bool(torch.all(margins >= active_margin).item())


def select_active_positions(
    reports: Sequence[Dict[str, Any]],
    active_margin: float,
) -> List[int]:
    return [
        position
        for position, report in enumerate(reports)
        if float(report["margin"]) < active_margin
    ]


def sample_retain_calibration(
    records: Sequence[SampledMCFRecord],
    count: int,
    seed: int,
) -> List[SampledMCFRecord]:
    if count <= 0 or not records:
        return []
    count = min(count, len(records))
    indices = random.Random(seed).sample(range(len(records)), k=count)
    return [records[index] for index in indices]


def _config_candidates(model_path: str) -> List[Path]:
    path = Path(model_path)
    if not path.exists():
        return []
    candidates = [
        path / "repair_experiment_config.json",
        path / "config_used.json",
    ]
    current = path if path.is_dir() else path.parent
    for _ in range(4):
        candidates.append(current / "config_used.json")
        current = current.parent
    seen: set[Path] = set()
    return [
        candidate
        for candidate in candidates
        if not (candidate in seen or seen.add(candidate))
    ]


def _extract_alpha_tuple(config: Dict[str, Any]) -> Optional[Dict[str, float]]:
    preserved = config.get("preserved_5e_overlap_alphas")
    if isinstance(preserved, dict):
        mapping = {
            "post_training_new_true_alpha": preserved.get("target_new_true"),
            "post_training_new_retain_alpha": preserved.get("target_new_retain"),
            "post_training_new_true_retain_alpha": preserved.get(
                "target_new_true_retain"
            ),
        }
    elif all(key in config for key in ALPHA_KEYS):
        mapping = {key: config.get(key) for key in ALPHA_KEYS}
    else:
        overlap = config.get("post_training_overlap_alphas")
        if not isinstance(overlap, dict):
            return None
        mapping = {
            "post_training_new_true_alpha": overlap.get("target_new_true"),
            "post_training_new_retain_alpha": overlap.get("target_new_retain"),
            "post_training_new_true_retain_alpha": overlap.get(
                "target_new_true_retain"
            ),
        }
    if any(value is None for value in mapping.values()):
        return None
    resolved = {
        "target_new_true": float(mapping["post_training_new_true_alpha"]),
        "target_new_retain": float(mapping["post_training_new_retain_alpha"]),
        "target_new_true_retain": float(mapping["post_training_new_true_retain_alpha"]),
    }
    if any(not 0.0 <= value <= 1.0 for value in resolved.values()):
        raise ValueError(
            f"Recovered Setting 5 overlap alphas must remain in [0,1], got {resolved}"
        )
    return resolved


def recover_experiment_config(
    model_path: str,
    explicit_path: Optional[str],
) -> Tuple[Path, Dict[str, Any], Dict[str, float]]:
    candidates = (
        [Path(explicit_path)] if explicit_path else _config_candidates(model_path)
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        with candidate.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            continue
        canonical = config.get("source_experiment_config")
        if not isinstance(canonical, dict):
            canonical = config
        alphas = _extract_alpha_tuple(config) or _extract_alpha_tuple(canonical)
        if alphas is not None:
            return candidate.resolve(), canonical, alphas
    requested = (
        f"explicit config {explicit_path!r}"
        if explicit_path
        else f"a config beside or above checkpoint {model_path!r}"
    )
    raise FileNotFoundError(
        "Could not recover the Setting 5 overlap-alpha tuple from "
        f"{requested}. Pass --experiment-config-path; no 5e coefficients "
        "will be assumed."
    )


def validate_source_experiment_config(
    source_config: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    expected_values = {
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
    }
    for key, expected in expected_values.items():
        if key in source_config and int(source_config[key]) != int(expected):
            raise ValueError(
                f"Repair {key}={expected} does not match source Setting 5 "
                f"{key}={source_config[key]}"
            )
    source_sample_mode = source_config.get(
        "mcf_sample_mode", source_config.get("sample_mode")
    )
    if source_sample_mode is not None and str(source_sample_mode) != args.sample_mode:
        raise ValueError(
            f"Repair sample mode {args.sample_mode!r} does not match source "
            f"Setting 5 sample mode {source_sample_mode!r}"
        )
    source_dataset = source_config.get("dataset")
    if source_dataset is not None and str(source_dataset).lower() != "mcf":
        raise ValueError(
            f"Active-case repair requires an MCF source config, got {source_dataset!r}"
        )


def _convert_mcf_record(record: Dict[str, Any]) -> gagd.Example:
    rewrite, _ = gagd.extract_mcf_rewrite(record)
    subject = str(rewrite["subject"])
    target_new = gagd.normalize_answer(str(rewrite["target_new"]["str"]))
    target_true_block = rewrite.get("target_true")
    if not isinstance(target_true_block, dict) or not target_true_block.get("str"):
        raise ValueError(
            "Active-case repair requires target_true.str on every MCF record"
        )
    target_true = gagd.normalize_answer(str(target_true_block["str"]))
    prompt = gagd.format_mcf_prompt(str(rewrite["prompt"]), subject)
    return gagd.Example(
        prompt=prompt,
        answer=target_new,
        subject=subject,
        target_new=target_new,
        target_true=target_true,
        paraphrase_prompts=None,
        source="mcf",
    )


def load_sampled_mcf_records(
    args: argparse.Namespace,
) -> Tuple[List[SampledMCFRecord], List[SampledMCFRecord]]:
    cache_path = gagd.resolve_output_path(args.mcf_cache_path)
    gagd.download_if_missing(args.mcf_url, cache_path)
    with cache_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError("MCF JSON must contain a list")
    original_indices = {id(record): index for index, record in enumerate(raw)}
    forget_raw, retain_raw = gagd.sample_mcf_raw_records(
        raw,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )

    def convert(records: Sequence[Dict[str, Any]]) -> List[SampledMCFRecord]:
        converted: List[SampledMCFRecord] = []
        for sampled_position, record in enumerate(records):
            if id(record) not in original_indices:
                raise RuntimeError(
                    "Sampled MCF record lost its original dataset identity"
                )
            converted.append(
                SampledMCFRecord(
                    record_index=original_indices[id(record)],
                    sampled_position=sampled_position,
                    example=_convert_mcf_record(record),
                )
            )
        return converted

    return convert(forget_raw), convert(retain_raw)


def _group_sets(groups: gagd.PostTrainingTokenGroups) -> Dict[str, set[int]]:
    return {
        "target_new": set(groups.target_new),
        "target_true": set(groups.target_true),
        "retain": set(groups.retain),
        **{name: set(getattr(groups, name)) for name in GROUP_NAMES},
    }


def token_membership_report(
    tok: Any,
    text: str,
    field: str,
    groups: gagd.PostTrainingTokenGroups,
) -> List[Dict[str, Any]]:
    group_sets = _group_sets(groups)
    rows: List[Dict[str, Any]] = []
    for position, token_id in enumerate(gagd.token_ids_for_text(tok, text)):
        memberships = [
            group_name
            for group_name, token_ids in group_sets.items()
            if token_id in token_ids
        ]
        rows.append(
            {
                "position": position,
                "field": field,
                "token_id": int(token_id),
                "token": tok.decode([int(token_id)]),
                "groups": memberships,
            }
        )
    return rows


def build_target_batch(
    tok: Any,
    records: Sequence[SampledMCFRecord],
    field: str,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    examples = [
        gagd.Example(
            prompt=record.example.prompt,
            answer=getattr(record.example, field),
            source="mcf",
        )
        for record in records
    ]
    return gagd.build_batch(tok, examples, device, append_eos=False)


def answer_nll_per_example(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = output.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_nll = (
        -F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    )
    return (token_nll * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def rewrite_nll_tensors(
    model: nn.Module,
    tok: Any,
    records: Sequence[SampledMCFRecord],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    new_batch = build_target_batch(tok, records, "target_new", device)
    true_batch = build_target_batch(tok, records, "target_true", device)
    return (
        answer_nll_per_example(model, new_batch),
        answer_nll_per_example(model, true_batch),
    )


@torch.no_grad()
def evaluate_rewrite_margin_reports(
    model: nn.Module,
    tok: Any,
    records: Sequence[SampledMCFRecord],
    groups: gagd.PostTrainingTokenGroups,
    active_margin: float,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, Any]]:
    model.eval()
    new_values: List[float] = []
    true_values: List[float] = []
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        new_nll, true_nll = rewrite_nll_tensors(model, tok, batch_records, device)
        new_values.extend(float(value) for value in new_nll.detach().cpu())
        true_values.extend(float(value) for value in true_nll.detach().cpu())

    reports: List[Dict[str, Any]] = []
    for record, target_new_nll, target_true_nll in zip(
        records, new_values, true_values
    ):
        margin = float(margin_from_nll(target_new_nll, target_true_nll))
        reports.append(
            {
                "record_index": record.record_index,
                "sampled_position": record.sampled_position,
                "prompt": record.example.prompt,
                "target_new": record.example.target_new.strip(),
                "target_true": record.example.target_true.strip(),
                "target_new_nll": target_new_nll,
                "target_true_nll": target_true_nll,
                "margin": margin,
                "active_margin": active_margin,
                "is_active": margin < active_margin,
                "target_tokens": {
                    "target_new": token_membership_report(
                        tok, record.example.target_new, "target_new", groups
                    ),
                    "target_true": token_membership_report(
                        tok, record.example.target_true, "target_true", groups
                    ),
                },
            }
        )
    return reports


def active_report_payload(
    reports: Sequence[Dict[str, Any]],
    active_margin: float,
) -> Dict[str, Any]:
    cases = [report for report in reports if float(report["margin"]) < active_margin]
    return {
        "active_margin": active_margin,
        "count": len(cases),
        "cases": cases,
    }


def selected_rows_for_active_records(
    tok: Any,
    active_records: Sequence[SampledMCFRecord],
    groups: gagd.PostTrainingTokenGroups,
    repair_mode: str,
) -> List[int]:
    if repair_mode == "true_scale":
        allowed = set(groups.unique_target_true)
        fields = ("target_true",)
    elif repair_mode == "extrapolate_delta":
        allowed = set(groups.unique_target_new)
        fields = ("target_new",)
    else:
        allowed = None
        fields = ("target_new", "target_true")

    selected: set[int] = set()
    for record in active_records:
        for field in fields:
            selected.update(
                gagd.token_ids_for_text(tok, getattr(record.example, field))
            )
    selected -= gagd.special_token_ids(tok)
    if allowed is not None:
        selected &= allowed
    return sorted(selected)


def freeze_model_for_output_repair(model: nn.Module) -> nn.Module:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("Model must expose input and output embeddings")

    if input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr():
        if not isinstance(output_embeddings, nn.Linear):
            raise ValueError(
                "Tied output embeddings must be an nn.Linear so the lm_head can be untied"
            )
        replacement = nn.Linear(
            output_embeddings.in_features,
            output_embeddings.out_features,
            bias=output_embeddings.bias is not None,
            device=output_embeddings.weight.device,
            dtype=output_embeddings.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(output_embeddings.weight)
            if output_embeddings.bias is not None:
                replacement.bias.copy_(output_embeddings.bias)
        model.set_output_embeddings(replacement)
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        output_embeddings = replacement

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("Input embeddings were not frozen")
    if (
        model.get_input_embeddings().weight.data_ptr()
        == output_embeddings.weight.data_ptr()
    ):
        raise RuntimeError(
            "Output lm_head is still tied to the frozen input embeddings"
        )
    return output_embeddings


@torch.no_grad()
def apply_active_true_scale(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    base_rows: torch.Tensor,
    target_true_scale: float,
) -> None:
    if len(row_ids) != base_rows.shape[0]:
        raise ValueError("base row count does not match selected target-true rows")
    if not row_ids:
        return
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    final_rows = base_rows.to(
        device=output_weight.device, dtype=output_weight.dtype
    ).mul(target_true_scale)
    output_weight.index_copy_(0, ids, final_rows)


@torch.no_grad()
def apply_gamma_extrapolation(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    base_rows: torch.Tensor,
    input_checkpoint_rows: torch.Tensor,
    gamma: float,
) -> None:
    if (
        len(row_ids) != base_rows.shape[0]
        or base_rows.shape != input_checkpoint_rows.shape
    ):
        raise ValueError("row snapshots do not match selected target-new rows")
    if not row_ids:
        return
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    base = base_rows.to(device=output_weight.device, dtype=output_weight.dtype)
    checkpoint = input_checkpoint_rows.to(
        device=output_weight.device, dtype=output_weight.dtype
    )
    output_weight.index_copy_(0, ids, base + gamma * (checkpoint - base))


def load_base_output_rows(
    model_path: str,
    row_ids: Sequence[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not row_ids:
        return torch.empty((0, 0), dtype=dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    output_embeddings = base_model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Base model does not expose output embeddings")
    ids = torch.tensor(
        row_ids, dtype=torch.long, device=output_embeddings.weight.device
    )
    rows = output_embeddings.weight.index_select(0, ids).detach().cpu().clone()
    del base_model
    gc.collect()
    return rows


def _selected_column_lookup(selected_ids: Sequence[int]) -> Dict[int, int]:
    return {token_id: position for position, token_id in enumerate(selected_ids)}


@torch.no_grad()
def build_answer_delta_caches(
    model: nn.Module,
    tok: Any,
    records: Sequence[SampledMCFRecord],
    field: str,
    selected_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> List[AnswerDeltaCache]:
    caches: List[AnswerDeltaCache] = []
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=device)
    selected_columns = _selected_column_lookup(selected_ids)
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = build_target_batch(tok, chunk, field, device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        logits = output.logits[:, :-1, :].float()
        log_probs = F.log_softmax(logits, dim=-1)
        hidden = output.hidden_states[-1][:, :-1, :].float()
        labels = batch["labels"][:, 1:]
        mask = labels.ne(-100)
        for row in range(len(chunk)):
            row_mask = mask[row]
            target_ids = labels[row, row_mask].long()
            row_log_probs = log_probs[row, row_mask]
            base_token_nll = -row_log_probs.gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)
            selected_probs = row_log_probs.index_select(-1, selected_tensor).exp()
            target_columns = torch.tensor(
                [selected_columns.get(int(token_id), -1) for token_id in target_ids],
                dtype=torch.long,
                device=device,
            )
            caches.append(
                AnswerDeltaCache(
                    base_token_nll=base_token_nll.detach(),
                    hidden=hidden[row, row_mask].detach(),
                    selected_probs=selected_probs.detach(),
                    target_selected_columns=target_columns,
                )
            )
        del output, logits, log_probs, hidden
    return caches


def build_rewrite_delta_caches(
    model: nn.Module,
    tok: Any,
    records: Sequence[SampledMCFRecord],
    selected_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> List[RewriteDeltaCache]:
    new_caches = build_answer_delta_caches(
        model, tok, records, "target_new", selected_ids, device, batch_size
    )
    true_caches = build_answer_delta_caches(
        model, tok, records, "target_true", selected_ids, device, batch_size
    )
    return [
        RewriteDeltaCache(target_new=new_cache, target_true=true_cache)
        for new_cache, true_cache in zip(new_caches, true_caches)
    ]


def _log_partition_shift(
    selected_probs: torch.Tensor,
    corrections: torch.Tensor,
) -> torch.Tensor:
    selected_mass = selected_probs.sum(dim=-1).clamp(max=1.0)
    unchanged_mass = (1.0 - selected_mass).clamp_min(
        torch.finfo(selected_probs.dtype).tiny
    )
    unchanged_term = unchanged_mass.log().unsqueeze(-1)
    selected_terms = (
        selected_probs.clamp_min(torch.finfo(selected_probs.dtype).tiny).log()
        + corrections
    )
    return torch.logsumexp(
        torch.cat([unchanged_term, selected_terms], dim=-1),
        dim=-1,
    )


def answer_nll_from_delta_cache(
    cache: AnswerDeltaCache,
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    corrections = cache.hidden @ delta_rows.transpose(0, 1)
    log_shift = _log_partition_shift(cache.selected_probs, corrections)
    target_correction = corrections.new_zeros(corrections.shape[0])
    selected_mask = cache.target_selected_columns.ge(0)
    if selected_mask.any():
        target_correction[selected_mask] = corrections[
            selected_mask,
            cache.target_selected_columns[selected_mask],
        ]
    return (cache.base_token_nll + log_shift - target_correction).mean()


def margins_from_delta_caches(
    caches: Sequence[RewriteDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        [
            margin_from_nll(
                answer_nll_from_delta_cache(cache.target_new, delta_rows),
                answer_nll_from_delta_cache(cache.target_true, delta_rows),
            )
            for cache in caches
        ]
    )


@torch.no_grad()
def _prompt_hidden_and_log_probs(
    model: nn.Module,
    tok: Any,
    prompt: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tok(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    output = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
    )
    if output.logits.shape[1] < 2:
        return (
            output.hidden_states[-1].new_empty(
                (0, output.hidden_states[-1].shape[-1]),
                dtype=torch.float32,
            ),
            output.logits.new_empty(
                (0, output.logits.shape[-1]),
                dtype=torch.float32,
            ),
        )
    hidden = output.hidden_states[-1][:, :-1, :].squeeze(0).float()
    log_probs = F.log_softmax(output.logits[:, :-1, :].squeeze(0).float(), dim=-1)
    return hidden.detach(), log_probs.detach()


def load_reference_output_layer(
    model_path: str,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Reference model does not expose output embeddings")
    weight = output_embeddings.weight.detach().cpu().clone()
    bias = getattr(output_embeddings, "bias", None)
    bias_copy = bias.detach().cpu().clone() if bias is not None else None
    del model
    gc.collect()
    return weight, bias_copy


@torch.no_grad()
def build_retain_kl_caches(
    model: nn.Module,
    reference_output_weight: Optional[torch.Tensor],
    reference_output_bias: Optional[torch.Tensor],
    tok: Any,
    records: Sequence[SampledMCFRecord],
    selected_ids: Sequence[int],
    device: torch.device,
) -> List[RetainKLCache]:
    selected = torch.tensor(selected_ids, dtype=torch.long, device=device)
    caches: List[RetainKLCache] = []
    if reference_output_weight is not None:
        reference_output_weight = reference_output_weight.to(device=device)
    if reference_output_bias is not None:
        reference_output_bias = reference_output_bias.to(device=device)
    for record in records:
        hidden, candidate_log_probs = _prompt_hidden_and_log_probs(
            model, tok, record.example.prompt, device
        )
        if hidden.shape[0] == 0:
            continue
        candidate_selected_probs = candidate_log_probs.index_select(-1, selected).exp()
        if reference_output_weight is None:
            reference_selected_probs = candidate_selected_probs
            baseline_kl = candidate_log_probs.new_zeros(candidate_log_probs.shape[0])
        else:
            reference_logits = F.linear(
                hidden.to(dtype=reference_output_weight.dtype),
                reference_output_weight,
                reference_output_bias,
            ).float()
            reference_log_probs = F.log_softmax(reference_logits, dim=-1)
            if reference_log_probs.shape != candidate_log_probs.shape:
                raise ValueError(
                    "Reference and candidate produce incompatible retain logits"
                )
            reference_probs = reference_log_probs.exp()
            reference_selected_probs = reference_probs.index_select(-1, selected)
            baseline_kl = (
                reference_probs * (reference_log_probs - candidate_log_probs)
            ).sum(dim=-1)
        caches.append(
            RetainKLCache(
                hidden=hidden,
                candidate_selected_probs=candidate_selected_probs,
                reference_selected_probs=reference_selected_probs,
                baseline_kl=baseline_kl,
            )
        )
    return caches


def retain_kl_from_caches(
    caches: Sequence[RetainKLCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if not caches:
        return delta_rows.new_zeros(())
    values: List[torch.Tensor] = []
    for cache in caches:
        corrections = cache.hidden @ delta_rows.transpose(0, 1)
        log_shift = _log_partition_shift(cache.candidate_selected_probs, corrections)
        values.append(
            cache.baseline_kl
            + log_shift
            - (cache.reference_selected_probs * corrections).sum(dim=-1)
        )
    return torch.cat(values).mean()


def orthonormal_row_basis(
    rows: torch.Tensor,
    max_rank: Optional[int] = None,
) -> torch.Tensor:
    if rows.numel() == 0:
        return rows.new_empty((0, rows.shape[-1]), dtype=torch.float32)
    rows = rows.float()
    _, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
    tolerance = (
        max(rows.shape)
        * torch.finfo(rows.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    rank = int((singular_values > tolerance).sum().item())
    if max_rank is not None:
        rank = min(rank, max_rank)
    return right[:rank].contiguous()


def project_rows_away(
    rows: torch.Tensor,
    retained_basis: Optional[torch.Tensor],
) -> torch.Tensor:
    if retained_basis is None or retained_basis.numel() == 0:
        return rows
    return rows - (rows @ retained_basis.transpose(0, 1)) @ retained_basis


class SelectedRowDelta(nn.Module):
    """Small trainable delta whose optimizer state scales with selected rows."""

    def __init__(
        self,
        n_rows: int,
        hidden_size: int,
        *,
        direction_basis: Optional[torch.Tensor] = None,
        retained_basis: Optional[torch.Tensor] = None,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_rows = n_rows
        self.hidden_size = hidden_size
        if direction_basis is not None:
            self.register_buffer(
                "direction_basis",
                direction_basis.to(device=device, dtype=torch.float32),
            )
            self.coefficients = nn.Parameter(
                torch.zeros(
                    (n_rows, direction_basis.shape[0]),
                    device=device,
                    dtype=torch.float32,
                )
            )
            self.raw_delta = None
        else:
            self.direction_basis = None
            self.coefficients = None
            self.raw_delta = nn.Parameter(
                torch.zeros(
                    (n_rows, hidden_size),
                    device=device,
                    dtype=torch.float32,
                )
            )
        if retained_basis is not None:
            self.register_buffer(
                "retained_basis", retained_basis.to(device=device, dtype=torch.float32)
            )
        else:
            self.retained_basis = None

    def effective_delta(self) -> torch.Tensor:
        if self.coefficients is not None:
            delta = self.coefficients @ self.direction_basis
        else:
            if self.raw_delta is None:
                raise RuntimeError("SelectedRowDelta has no trainable parameter")
            delta = self.raw_delta
        return project_rows_away(delta, self.retained_basis)


def make_repair_optimizer(
    module: nn.Module,
    name: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = list(module.parameters())
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    raise ValueError(f"Unsupported repair optimizer: {name}")


def optimize_selected_delta(
    delta_module: SelectedRowDelta,
    margin_fn: Callable[[torch.Tensor], torch.Tensor],
    kl_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    active_margin: float,
    repair_steps: int,
    repair_lr: float,
    repair_optimizer: str,
    hinge_weight: float,
    delta_l2_lambda: float,
    retain_kl_mu: float,
    stop_when_all_satisfied: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    steps_completed = 0
    stopped_early = False

    with torch.no_grad():
        initial_margins = margin_fn(delta_module.effective_delta())
    if stop_when_all_satisfied and all_margins_satisfied(
        initial_margins, active_margin
    ):
        return logs, {
            "steps_completed": 0,
            "stopped_early": True,
            "all_satisfied": True,
        }

    optimizer = make_repair_optimizer(delta_module, repair_optimizer, repair_lr)
    for step in range(1, repair_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_rows = delta_module.effective_delta()
        margins = margin_fn(delta_rows)
        hinge = squared_hinge_loss(margins, active_margin)
        delta_l2 = delta_rows.square().sum()
        retain_kl = kl_fn(delta_rows)
        total = (
            hinge_weight * hinge + delta_l2_lambda * delta_l2 + retain_kl_mu * retain_kl
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite minimal repair loss at step {step}: "
                f"{float(total.detach().cpu())}"
            )
        total.backward()
        optimizer.step()
        steps_completed = step

        with torch.no_grad():
            updated_delta = delta_module.effective_delta()
            updated_margins = margin_fn(updated_delta)
            all_satisfied = all_margins_satisfied(updated_margins, active_margin)
            logs.append(
                {
                    "step": step,
                    "total_loss": float(total.detach().cpu()),
                    "squared_hinge": float(hinge.detach().cpu()),
                    "weighted_hinge": float((hinge_weight * hinge).detach().cpu()),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "weighted_delta_l2": float(
                        (delta_l2_lambda * delta_l2).detach().cpu()
                    ),
                    "retain_kl_reference_to_repaired": float(retain_kl.detach().cpu()),
                    "weighted_retain_kl": float(
                        (retain_kl_mu * retain_kl).detach().cpu()
                    ),
                    "minimum_margin_before_step": float(margins.min().detach().cpu()),
                    "minimum_margin_after_step": float(
                        updated_margins.min().detach().cpu()
                    ),
                    "unsatisfied_after_step": int(
                        (updated_margins < active_margin).sum().item()
                    ),
                    "all_training_rewrites_satisfied": all_satisfied,
                    "effective_delta_norm": float(updated_delta.norm().detach().cpu()),
                }
            )
        if stop_when_all_satisfied and all_satisfied:
            stopped_early = True
            break

    with torch.no_grad():
        final_margins = margin_fn(delta_module.effective_delta())
    return logs, {
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "all_satisfied": all_margins_satisfied(final_margins, active_margin),
    }


@torch.no_grad()
def materialize_selected_delta(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    delta_rows: torch.Tensor,
) -> None:
    if len(row_ids) != delta_rows.shape[0]:
        raise ValueError("delta row count does not match selected lm_head rows")
    if not row_ids:
        return
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    updated = output_weight.index_select(0, ids) + delta_rows.to(
        device=output_weight.device, dtype=output_weight.dtype
    )
    output_weight.index_copy_(0, ids, updated)


def _decoded_group_report(
    tok: Any,
    groups: gagd.PostTrainingTokenGroups,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for name, token_ids in asdict(groups).items():
        report[name] = {
            "count": len(token_ids),
            "token_ids": token_ids,
            "tokens": {
                str(token_id): tok.decode([int(token_id)]) for token_id in token_ids
            },
        }
    return report


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_repair_checkpoint(
    model: nn.Module,
    tok: Any,
    checkpoint_dir: Path,
    repair_config: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)
    if repair_config is not None:
        gagd.write_json(
            checkpoint_dir / "repair_experiment_config.json",
            repair_config,
        )


def _model_loading_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def _active_records(
    records: Sequence[SampledMCFRecord],
    active_positions: Sequence[int],
) -> List[SampledMCFRecord]:
    return [records[position] for position in active_positions]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, source_config, preserved_alphas = recover_experiment_config(
        args.model_path,
        args.experiment_config_path,
    )
    validate_source_experiment_config(source_config, args)
    config_used: Dict[str, Any] = {
        **vars(args),
        "method": METHOD,
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "repair_uses_official_paraphrases": False,
        "repair_parameter_scope": "selected_active_lm_head_rows_only",
        "reference_kl_hidden_source": "candidate_frozen_transformer",
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    print("Loading deterministic MCF forget/retain records")
    forget_records, retain_records = load_sampled_mcf_records(args)
    forget_examples = [record.example for record in forget_records]
    retain_examples = [record.example for record in retain_records]

    print(f"Loading input repair checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        _model_loading_args(args), for_training=False
    )
    output_embeddings = freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)

    groups = gagd.collect_post_training_token_groups(
        tok, forget_examples, retain_examples
    )
    before_reports = evaluate_rewrite_margin_reports(
        model,
        tok,
        forget_records,
        groups,
        args.active_margin,
        device,
        args.margin_batch_size,
    )
    active_positions = select_active_positions(before_reports, args.active_margin)
    active_records = _active_records(forget_records, active_positions)
    selected_ids = selected_rows_for_active_records(
        tok, active_records, groups, args.repair_mode
    )

    gagd.write_json(output_dir / "rewrite_margins_before.json", before_reports)
    gagd.write_json(
        output_dir / "active_cases_before.json",
        active_report_payload(before_reports, args.active_margin),
    )
    token_group_report = {
        "preserved_5e_overlap_alphas": preserved_alphas,
        "global_groups": _decoded_group_report(tok, groups),
        "repair_mode": args.repair_mode,
        "active_record_indices": [record.record_index for record in active_records],
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([token_id]) for token_id in selected_ids
        },
    }
    gagd.write_json(output_dir / "token_group_report.json", token_group_report)

    print(
        f"Active rewrites before repair: {len(active_positions)}/{len(forget_records)}; "
        f"selected lm_head rows: {len(selected_ids)}"
    )
    input_rows_before = model.get_input_embeddings().weight.detach()
    input_storage_pointer = input_rows_before.data_ptr()
    selected_before = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        )
        .detach()
        .clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    repair_logs: List[Dict[str, Any]] = []
    optimization_summary: Dict[str, Any] = {
        "steps_completed": 0,
        "stopped_early": False,
        "all_satisfied": len(active_positions) == 0,
    }
    actual_rank = 0
    retain_calibration_records: List[SampledMCFRecord] = []

    if selected_ids and args.repair_mode in {"true_scale", "extrapolate_delta"}:
        base_rows = load_base_output_rows(
            args.base_model_path,
            selected_ids,
            gagd.torch_dtype(args.dtype),
        )
        checkpoint_rows = selected_before.detach().cpu()
        if base_rows.shape != checkpoint_rows.shape:
            raise ValueError(
                "Base and input checkpoint lm_head rows have incompatible shapes"
            )
        if args.repair_mode == "true_scale":
            apply_active_true_scale(
                output_weight,
                selected_ids,
                base_rows,
                args.target_true_scale,
            )
        else:
            apply_gamma_extrapolation(
                output_weight,
                selected_ids,
                base_rows,
                checkpoint_rows,
                args.target_new_gamma,
            )
        repair_logs.append(
            {
                "step": 0,
                "repair_mode": args.repair_mode,
                "selected_lm_head_rows": len(selected_ids),
                "target_true_scale": (
                    args.target_true_scale if args.repair_mode == "true_scale" else None
                ),
                "target_new_gamma": (
                    args.target_new_gamma
                    if args.repair_mode == "extrapolate_delta"
                    else None
                ),
            }
        )

    elif selected_ids and args.repair_mode == "minimal_optimize":
        print("Caching exact sparse-delta rewrite objectives")
        rewrite_caches = build_rewrite_delta_caches(
            model,
            tok,
            forget_records,
            selected_ids,
            device,
            args.margin_batch_size,
        )
        retain_calibration_records = sample_retain_calibration(
            retain_records,
            args.retain_calibration_num,
            args.retain_calibration_seed,
        )
        reference_output_weight: Optional[torch.Tensor] = None
        reference_output_bias: Optional[torch.Tensor] = None
        if args.reference_model_path and args.retain_kl_mu > 0:
            print(
                "Loading optional frozen KL reference output layer: "
                f"{args.reference_model_path}"
            )
            (
                reference_output_weight,
                reference_output_bias,
            ) = load_reference_output_layer(
                args.reference_model_path,
                gagd.torch_dtype(args.dtype),
            )
        retain_caches = (
            build_retain_kl_caches(
                model,
                reference_output_weight,
                reference_output_bias,
                tok,
                retain_calibration_records,
                selected_ids,
                device,
            )
            if args.retain_kl_mu > 0 or args.project_away_retain_hidden
            else []
        )
        del reference_output_weight, reference_output_bias
        gc.collect()

        retained_basis: Optional[torch.Tensor] = None
        if args.project_away_retain_hidden:
            retained_hidden = torch.cat(
                [cache.hidden for cache in retain_caches], dim=0
            )
            retained_basis = orthonormal_row_basis(retained_hidden)
            print(
                f"Projecting repair away from {retained_basis.shape[0]} "
                "retain hidden directions"
            )

        direction_basis: Optional[torch.Tensor] = None
        if args.repair_rank > 0:
            active_hidden = torch.cat(
                [
                    answer_cache.hidden
                    for position in active_positions
                    for answer_cache in (
                        rewrite_caches[position].target_new,
                        rewrite_caches[position].target_true,
                    )
                ],
                dim=0,
            )
            active_hidden = project_rows_away(active_hidden, retained_basis)
            direction_basis = orthonormal_row_basis(
                active_hidden, max_rank=args.repair_rank
            )
            actual_rank = int(direction_basis.shape[0])
            if actual_rank == 0:
                raise ValueError(
                    "Active hidden directions vanished after retain projection"
                )
            print(f"Using rank-{actual_rank} active hidden-direction repair")

        delta_module = SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=retained_basis,
            device=output_weight.device,
        )
        margin_fn = lambda delta: margins_from_delta_caches(rewrite_caches, delta)
        kl_fn = lambda delta: retain_kl_from_caches(retain_caches, delta)
        repair_logs, optimization_summary = optimize_selected_delta(
            delta_module,
            margin_fn,
            kl_fn,
            active_margin=args.active_margin,
            repair_steps=args.repair_steps,
            repair_lr=args.repair_lr,
            repair_optimizer=args.repair_optimizer,
            hinge_weight=args.hinge_weight,
            delta_l2_lambda=args.delta_l2_lambda,
            retain_kl_mu=args.retain_kl_mu,
            stop_when_all_satisfied=args.stop_when_all_satisfied,
        )
        with torch.no_grad():
            materialize_selected_delta(
                output_weight,
                selected_ids,
                delta_module.effective_delta(),
            )

    write_jsonl(output_dir / "repair_log.jsonl", repair_logs)
    if model.get_input_embeddings().weight.data_ptr() != input_storage_pointer:
        raise RuntimeError("Input embedding storage changed during lm_head-only repair")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("Input embeddings unexpectedly became trainable")

    after_reports = evaluate_rewrite_margin_reports(
        model,
        tok,
        forget_records,
        groups,
        args.active_margin,
        device,
        args.margin_batch_size,
    )
    after_active_payload = active_report_payload(after_reports, args.active_margin)
    gagd.write_json(output_dir / "rewrite_margins_after.json", after_reports)
    gagd.write_json(output_dir / "active_cases_after.json", after_active_payload)

    selected_after = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        )
        .detach()
        .clone()
        if selected_ids
        else selected_before
    )
    selected_delta = selected_after.float() - selected_before.float()
    repair_summary = {
        "method": METHOD,
        "repair_mode": args.repair_mode,
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "reference_model_path": args.reference_model_path,
        "source_experiment_config_path": str(config_path),
        "preserved_5e_overlap_alphas": preserved_alphas,
        "forget_records": len(forget_records),
        "retain_records": len(retain_records),
        "active_margin": args.active_margin,
        "active_cases_before": len(active_positions),
        "active_cases_after": after_active_payload["count"],
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "changed_selected_lm_head_rows": int(
            selected_delta.norm(dim=1).gt(0).sum().item()
        )
        if selected_ids
        else 0,
        "selected_lm_head_delta_norm": float(selected_delta.norm().cpu()),
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "repair_rank_requested": args.repair_rank,
        "repair_rank_actual": actual_rank,
        "retain_calibration_record_indices": [
            record.record_index for record in retain_calibration_records
        ],
        "optimization": optimization_summary,
        "minimum_margin_before": min(
            (float(report["margin"]) for report in before_reports),
            default=None,
        ),
        "minimum_margin_after": min(
            (float(report["margin"]) for report in after_reports),
            default=None,
        ),
    }
    gagd.write_json(output_dir / "repair_summary.json", repair_summary)

    if args.save_model:
        checkpoint_dir = output_dir / "checkpoint"
        save_repair_checkpoint(
            model,
            tok,
            checkpoint_dir,
            repair_config=config_used,
        )
        print(f"Saved repaired checkpoint to {checkpoint_dir}")

    if args.run_official_mcf_eval:
        official_path = output_dir / "official_eval.json"
        official_result = gagd.evaluate_loaded_model_official(
            method=f"{METHOD}_{args.repair_mode}",
            model=model,
            tok=tok,
            model_dir=(
                output_dir / "checkpoint"
                if args.save_model
                else f"in-memory:{METHOD}_{args.repair_mode}"
            ),
            mcf_path=gagd.resolve_output_path(args.mcf_cache_path),
            wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
            out_path=official_path,
            unlearn_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            sample_mode=args.sample_mode,
            skip_ppl=args.skip_ppl,
        )
        print(
            "Official result: "
            f"Eff={official_result['forget']['Eff']}, "
            f"Gen={official_result['forget']['Gen']}, "
            f"Spe={official_result['forget']['Spe']}, "
            f"PPL={official_result.get('forget_PPL')}"
        )

    print(
        f"Done: active cases {len(active_positions)} -> "
        f"{after_active_payload['count']}; outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
