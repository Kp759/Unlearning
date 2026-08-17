#!/usr/bin/env python3
"""SURE-MQuAKE V7 Stage 1: sparse sensitive-row LM-head GA/GD.

This is the MQuAKE analogue of SURE-TOFU V7, aligned to the native
ZeroUnlearn-style MQuAKE token-level Eff metric.

Data firewall:
- use only the direct ``requested_rewrite`` prompts from the same 50 sampled
  forget instances;
- never load the 1,000 sampled retain instances;
- never load atomic natural-language questions, multi-hop questions, or the
  benchmark counterfactual target_new values.

Parameter firewall:
- transformer blocks stay frozen and exact Base;
- input embeddings stay frozen and exact Base;
- only LM-head rows that are sensitive target_true answer tokens are editable;
- every non-sensitive LM-head row remains exact Base by construction.

Objective:
- GA lowers the probability of each sensitive teacher-forced target token;
- GD preserves the Base distribution on the same prompt/position after removing
  that sensitive target token and renormalizing;
- L2 discourages unnecessary sparse-row displacement.

Checkpoint selection is forget-only.  The stage stops at the first/best sparse
candidate that makes every visible sensitive token lose by the configured
Stage-1 safety margin.  Retain/PPL are evaluation-only downstream.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7-sparse-sensitive-lmhead-GAGD"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


@dataclass
class TokenDeltaCache:
    case: mquake.PredictionCase
    hidden: torch.Tensor
    base_token_nll: torch.Tensor
    base_target_logit: torch.Tensor
    selected_probs: torch.Tensor
    base_selected_logits: torch.Tensor
    target_selected_column: int
    base_best_unselected_logit: torch.Tensor
    base_best_unselected_token_id: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Protected pretrained Base model")
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50, help="MQuAKE instance count")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--target-logit-margin",
        type=float,
        default=0.05,
        help="Stage-1 competitor-minus-sensitive margin required on every visible token.",
    )
    p.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def decoded_token(tok: Any, token_id: int) -> str:
    return tok.decode([int(token_id)])


def direct_rewrite_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> List[mquake.PredictionCase]:
    cases = [
        case
        for record in records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    identities = [case.identity for case in cases]
    if len(identities) != len(set(identities)):
        raise RuntimeError("MQuAKE direct rewrite token identities are not unique")
    if not cases:
        raise RuntimeError("MQuAKE locked forget records produced no rewrite token cases")
    return cases


@torch.no_grad()
def resolve_case_target_ids(
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    llama_like: bool,
    device: torch.device,
) -> List[int]:
    target_ids = mquake.official_target_ids(
        tok,
        [case.target_text for case in cases],
        llama_like=llama_like,
        device=device,
    )
    return [int(value) for value in target_ids.detach().cpu().tolist()]


@torch.no_grad()
def build_token_delta_caches(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    selected_ids: Sequence[int],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    desc: str,
) -> List[TokenDeltaCache]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    selected_lookup = {int(token_id): idx for idx, token_id in enumerate(selected_ids)}
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=device)
    caches: List[TokenDeltaCache] = []

    batches = list(_chunks(list(cases), batch_size))
    for batch in tqdm(batches, desc=desc, leave=False):
        encoded = tok(
            [case.prompt for case in batch],
            padding=True,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][batch_indices, last_non_masked, :].float()
        logits = output.logits[batch_indices, last_non_masked, :].float()
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )

        if selected_ids:
            base_selected_logits = logits.index_select(-1, selected_tensor)
            selected_probs = log_probs.index_select(-1, selected_tensor).exp()
        else:
            base_selected_logits = logits.new_empty((len(batch), 0))
            selected_probs = logits.new_empty((len(batch), 0))

        competitor_logits = logits.clone()
        if selected_ids:
            competitor_logits.index_fill_(1, selected_tensor, float("-inf"))
        competitor_logits.scatter_(1, target_ids[:, None], float("-inf"))
        best_unselected_logits, best_unselected_ids = competitor_logits.max(dim=-1)
        base_target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        base_target_nll = -log_probs.gather(1, target_ids[:, None]).squeeze(1)

        for index, case in enumerate(batch):
            target_id = int(target_ids[index].item())
            caches.append(
                TokenDeltaCache(
                    case=case,
                    hidden=hidden[index].detach(),
                    base_token_nll=base_target_nll[index].detach(),
                    base_target_logit=base_target_logits[index].detach(),
                    selected_probs=selected_probs[index].detach(),
                    base_selected_logits=base_selected_logits[index].detach(),
                    target_selected_column=int(selected_lookup.get(target_id, -1)),
                    base_best_unselected_logit=best_unselected_logits[index].detach(),
                    base_best_unselected_token_id=int(best_unselected_ids[index].item()),
                )
            )
        del output, hidden, logits, log_probs, competitor_logits
    return caches


def stack_cache_fields(
    caches: Sequence[TokenDeltaCache],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not caches:
        raise ValueError("cannot stack empty token cache")
    return {
        "hidden": torch.stack([cache.hidden for cache in caches]).to(device=device, dtype=torch.float32),
        "base_nll": torch.stack([cache.base_token_nll for cache in caches]).to(device=device, dtype=torch.float32),
        "base_target_logit": torch.stack([cache.base_target_logit for cache in caches]).to(device=device, dtype=torch.float32),
        "selected_probs": torch.stack([cache.selected_probs for cache in caches]).to(device=device, dtype=torch.float32),
        "base_selected_logits": torch.stack([cache.base_selected_logits for cache in caches]).to(device=device, dtype=torch.float32),
        "target_columns": torch.tensor(
            [cache.target_selected_column for cache in caches],
            dtype=torch.long,
            device=device,
        ),
        "best_unselected_logit": torch.stack(
            [cache.base_best_unselected_logit for cache in caches]
        ).to(device=device, dtype=torch.float32),
    }


def corrections_from_delta(stacked: Mapping[str, torch.Tensor], delta_rows: torch.Tensor) -> torch.Tensor:
    hidden = stacked["hidden"]
    if delta_rows.shape[0] == 0:
        return hidden.new_empty((hidden.shape[0], 0))
    return hidden @ delta_rows.transpose(0, 1)


def token_nlls_from_delta(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    corrections = corrections_from_delta(stacked, delta_rows)
    log_shift = active._log_partition_shift(stacked["selected_probs"], corrections)
    target_correction = stacked["base_nll"].new_zeros(stacked["base_nll"].shape)
    target_columns = stacked["target_columns"]
    selected_mask = target_columns.ge(0)
    if selected_mask.any():
        row_idx = selected_mask.nonzero(as_tuple=False).flatten()
        target_correction[selected_mask] = corrections[
            row_idx,
            target_columns[selected_mask],
        ]
    return stacked["base_nll"] + log_shift - target_correction


def competitor_minus_sensitive_margins(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    corrections = corrections_from_delta(stacked, delta_rows)
    target_columns = stacked["target_columns"]
    target_logits = stacked["base_target_logit"].clone()
    selected_mask = target_columns.ge(0)
    if selected_mask.any():
        row_idx = selected_mask.nonzero(as_tuple=False).flatten()
        target_logits[selected_mask] += corrections[
            row_idx,
            target_columns[selected_mask],
        ]

    best = stacked["best_unselected_logit"].clone()
    if corrections.shape[1] > 0:
        selected_logits = stacked["base_selected_logits"] + corrections
        if selected_mask.any():
            selected_logits = selected_logits.clone()
            row_idx = selected_mask.nonzero(as_tuple=False).flatten()
            selected_logits[row_idx, target_columns[selected_mask]] = float("-inf")
        best = torch.maximum(best, selected_logits.max(dim=-1).values)
    return best - target_logits


def same_prompt_non_target_kl(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(Base_non-target || Current_non-target) for sparse row changes."""
    corrections = corrections_from_delta(stacked, delta_rows)
    if corrections.shape[1] == 0:
        return delta_rows.new_zeros(())

    q_selected = stacked["selected_probs"].float()
    q_target = torch.exp(-stacked["base_nll"].float()).clamp(max=1.0 - 1e-7)
    target_columns = stacked["target_columns"]
    selected_mask = target_columns.ge(0)
    tiny = torch.finfo(torch.float32).tiny

    unchanged_mass = (1.0 - q_selected.sum(dim=-1)).clamp_min(0.0)
    unchanged_mass = torch.where(
        selected_mask,
        unchanged_mass,
        (unchanged_mass - q_target).clamp_min(0.0),
    )
    unchanged_log = unchanged_mass.clamp_min(tiny).log().unsqueeze(-1)
    selected_log_terms = q_selected.clamp_min(tiny).log() + corrections
    expected_c_num = (q_selected * corrections).sum(dim=-1)

    if selected_mask.any():
        selected_log_terms = selected_log_terms.clone()
        row_idx = selected_mask.nonzero(as_tuple=False).flatten()
        col_idx = target_columns[selected_mask]
        target_c = corrections[row_idx, col_idx]
        selected_log_terms[row_idx, col_idx] = float("-inf")
        expected_c_num[selected_mask] -= q_target[selected_mask] * target_c

    log_current_partition = torch.logsumexp(
        torch.cat([unchanged_log, selected_log_terms], dim=-1),
        dim=-1,
    )
    base_non_target_mass = (1.0 - q_target).clamp_min(tiny)
    kl = (
        log_current_partition
        - base_non_target_mass.log()
        - expected_c_num / base_non_target_mass
    )
    return kl.clamp_min(0.0).mean()


def metrics_from_delta(
    stacked: Mapping[str, torch.Tensor],
    delta_rows: torch.Tensor,
    *,
    target_margin: float,
    kl_value: float | None = None,
) -> Dict[str, Any]:
    nll = token_nlls_from_delta(stacked, delta_rows)
    margins = competitor_minus_sensitive_margins(stacked, delta_rows)
    if kl_value is None:
        kl_value = float(same_prompt_non_target_kl(stacked, delta_rows).detach().cpu())
    return {
        "official_active_sensitive_token_count": int((margins <= 0.0).sum().item()),
        "buffered_margin_unmet_token_count": int((margins < target_margin).sum().item()),
        "minimum_competitor_minus_sensitive_margin": float(margins.min().detach().cpu()),
        "mean_competitor_minus_sensitive_margin": float(margins.mean().detach().cpu()),
        "sensitive_token_probability_mean": float(torch.exp(-nll).mean().detach().cpu()),
        "sensitive_token_probability_max": float(torch.exp(-nll).max().detach().cpu()),
        "same_prompt_non_target_kl": float(kl_value),
        "selected_lm_head_delta_norm": float(delta_rows.norm().detach().cpu()),
    }


def priority(metrics: Mapping[str, Any]) -> Tuple[int, int, float, float, float]:
    return (
        int(metrics["official_active_sensitive_token_count"]),
        int(metrics["buffered_margin_unmet_token_count"]),
        float(metrics["same_prompt_non_target_kl"]),
        float(metrics["selected_lm_head_delta_norm"]),
        -float(metrics["minimum_competitor_minus_sensitive_margin"]),
    )


@torch.no_grad()
def set_selected_rows(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
    delta_rows: torch.Tensor,
) -> None:
    if not selected_ids:
        return
    ids = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    updated = base_rows.to(device=output_weight.device, dtype=torch.float32) + delta_rows.to(
        device=output_weight.device, dtype=torch.float32
    )
    output_weight.index_copy_(0, ids, updated.to(dtype=output_weight.dtype))


@torch.no_grad()
def exact_materialized_reports(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    target_margin: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for batch in _chunks(list(cases), batch_size):
        encoded = tok([case.prompt for case in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, use_cache=False)
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        logits = output.logits[batch_indices, last_non_masked, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        competitor_logits = logits.clone()
        competitor_logits.scatter_(1, target_ids[:, None], float("-inf"))
        best_logits, best_ids = competitor_logits.max(dim=-1)
        margins = best_logits - target_logits
        probs = F.softmax(logits, dim=-1).gather(1, target_ids[:, None]).squeeze(1)
        predicted = logits.argmax(dim=-1)
        for idx, case in enumerate(batch):
            reports.append(
                {
                    **asdict(case),
                    "target_token_id": int(target_ids[idx].item()),
                    "predicted_token_id": int(predicted[idx].item()),
                    "best_competitor_token_id": int(best_ids[idx].item()),
                    "competitor_minus_sensitive_margin": float(margins[idx].detach().cpu()),
                    "sensitive_probability": float(probs[idx].detach().cpu()),
                    "official_sensitive_token_still_argmax": bool(
                        int(predicted[idx].item()) == int(target_ids[idx].item())
                    ),
                    "target_margin_satisfied": bool(float(margins[idx].detach().cpu()) >= target_margin),
                }
            )
        del output, logits, competitor_logits

    margins_t = torch.tensor(
        [row["competitor_minus_sensitive_margin"] for row in reports], dtype=torch.float32
    )
    probs_t = torch.tensor([row["sensitive_probability"] for row in reports], dtype=torch.float32)
    summary = {
        "token_count": len(reports),
        "official_active_sensitive_token_count": sum(
            bool(row["official_sensitive_token_still_argmax"]) for row in reports
        ),
        "buffered_margin_unmet_token_count": sum(
            not bool(row["target_margin_satisfied"]) for row in reports
        ),
        "minimum_competitor_minus_sensitive_margin": float(margins_t.min().item()),
        "mean_competitor_minus_sensitive_margin": float(margins_t.mean().item()),
        "sensitive_token_probability_mean": float(probs_t.mean().item()),
        "sensitive_token_probability_max": float(probs_t.max().item()),
        "target_margin": float(target_margin),
    }
    return reports, summary


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps, and batch-size must be positive")
    if a.lr <= 0 or a.ga_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid Stage-1 learning rate/GA/GD weights")
    if a.delta_l2_lambda < 0 or a.grad_clip < 0 or a.target_logit_margin < 0:
        raise ValueError("invalid Stage-1 regularization/margin controls")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.repair_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, split_manifest = locked.load_locked_records(
        visible_path,
        manifest_path,
        a.forget_num,
        a.seed,
    )

    model_args = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)

    cases = direct_rewrite_cases(records, tok, llama_like=llama_like)
    target_ids = resolve_case_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    special_ids = gagd.special_token_ids(tok)
    sensitive_ids = sorted(set(target_ids) - special_ids)
    if not sensitive_ids:
        raise RuntimeError("MQuAKE visible sensitive tokens produced no editable LM-head rows")
    missing = sorted(set(target_ids) - set(sensitive_ids))
    if missing:
        raise RuntimeError(f"official sensitive target tokens unexpectedly include special ids: {missing}")

    selected_tensor = torch.tensor(sensitive_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()
    caches = build_token_delta_caches(
        model,
        tok,
        cases,
        sensitive_ids,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        desc="cache MQuAKE V7 Stage1 direct tokens",
    )
    stacked = stack_cache_fields(caches, device=device)

    zero = torch.zeros(
        (len(sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    base_kl = float(same_prompt_non_target_kl(stacked, zero).detach().cpu())
    base_metrics = metrics_from_delta(
        stacked,
        zero,
        target_margin=a.target_logit_margin,
        kl_value=base_kl,
    )

    module = active.SelectedRowDelta(
        len(sensitive_ids),
        int(output_weight.shape[1]),
        direction_basis=None,
        retained_basis=None,
        device=device,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=a.lr, weight_decay=0.0)

    best_delta = zero.detach().clone()
    best_metrics = dict(base_metrics)
    best_step = 0
    steps_completed = 0
    stopped_early = False
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        current_nll = token_nlls_from_delta(stacked, delta)
        ga_sensitive_logprob = -current_nll.mean()
        gd_non_target_kl = same_prompt_non_target_kl(stacked, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.ga_weight * ga_sensitive_logprob
            + a.gd_weight * gd_non_target_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite MQuAKE V7 Stage1 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage1 gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        steps_completed = step

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_kl = float(same_prompt_non_target_kl(stacked, candidate).detach().cpu())
            candidate_metrics = metrics_from_delta(
                stacked,
                candidate,
                target_margin=a.target_logit_margin,
                kl_value=candidate_kl,
            )
            if priority(candidate_metrics) < priority(best_metrics):
                best_delta = candidate.detach().clone()
                best_metrics = dict(candidate_metrics)
                best_step = step

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "ga_sensitive_logprob": float(ga_sensitive_logprob.detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_non_target_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **candidate_metrics,
            }
            logs.append(row)
            print(
                f"stage1-step={step} official_active={candidate_metrics['official_active_sensitive_token_count']} "
                f"buffered_unmet={candidate_metrics['buffered_margin_unmet_token_count']} "
                f"min_margin={candidate_metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={candidate_kl:.6g} norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
            )

        if (
            a.stop_when_all_satisfied
            and candidate_metrics["buffered_margin_unmet_token_count"] == 0
        ):
            best_delta = candidate.detach().clone()
            best_metrics = dict(candidate_metrics)
            best_step = step
            stopped_early = True
            break

    del optimizer
    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "train_log.jsonl", logs)

    set_selected_rows(output_weight, sensitive_ids, base_selected_rows, best_delta)
    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("MQuAKE V7 Stage1 modified input embeddings")

    exact_reports, exact_summary = exact_materialized_reports(
        model,
        tok,
        cases,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )
    write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", exact_reports)

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    write_json(
        root / "sensitive_lm_rows.json",
        {
            "sensitive_row_definition": "union of official target_true token ids on all direct requested_rewrite token positions from the 50 sampled forget instances",
            "sensitive_row_count": len(sensitive_ids),
            "sensitive_token_ids": sensitive_ids,
            "sensitive_tokens": {
                str(token_id): decoded_token(tok, token_id) for token_id in sensitive_ids
            },
            "non_sensitive_lm_rows_exact_base_by_construction": True,
            "input_embeddings_exact_base_by_construction": True,
            "transformer_frozen": True,
        },
    )

    status = (
        "PASS_STAGE1_MARGIN"
        if exact_summary["buffered_margin_unmet_token_count"] == 0
        else "PASS_STAGE1_WITH_RESIDUALS_FOR_STAGE2"
    )
    summary = {
        "status": status,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "sensitive_lm_head_row_count": len(sensitive_ids),
        "selected_lm_head_delta_norm": float(best_delta.norm().detach().cpu()),
        "cached_metrics": best_metrics,
        "materialized_bf16_metrics": exact_summary,
        "training_data_access": {
            "forget_instances": int(a.forget_num),
            "forget_atomic_facts": len(records),
            "prompt_types": ["requested_rewrite"],
            "benchmark_retain_instances": 0,
            "atomic_questions": 0,
            "multihop_questions": 0,
            "benchmark_counterfactual_targets": 0,
            "PPL": False,
        },
        "checkpoint_selection_uses_retain_or_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            **vars(a),
            "repair_visible_path_resolved": str(visible_path),
            "split_manifest_resolved": str(manifest_path),
            "split_sampling": split_manifest.get("sampling"),
            "parameter_scope": "sparse sensitive target_true LM-head rows only; transformer/input embeddings/non-sensitive LM rows exact Base",
            "ga_definition": "minimize mean sensitive target-token log probability on official direct rewrite token positions",
            "gd_definition": "same-prompt KL(Base_non-target || Current_non-target), current sensitive target token removed and renormalized",
            "selection_definition": "forget-only lexicographic priority; no retain/PPL/atomic/multihop data",
            "checkpoint": str(ckpt.resolve()),
        },
    )

    print("===== SURE-MQuAKE V7 STAGE1 =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"sensitive_rows={len(sensitive_ids)} best_step={best_step}"
    )
    print(
        f"BF16 official_active={exact_summary['official_active_sensitive_token_count']} "
        f"buffered_unmet={exact_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={exact_summary['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
