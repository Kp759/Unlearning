#!/usr/bin/env python3
"""Shared mechanics for the canonical SURE-LM MCF/ZsRE pipeline.

The benchmark adapter decides only (1) which answer is sensitive and (2) what
counts as a satisfied direct forget constraint.  Stage-1 GA/GD, vocabulary-row
restoration, sparse output-row parameterization, rank handling, and scale
selection are shared.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SensitivePredictionCase:
    case_id: int
    record_position: int
    token_index: int
    prompt: str
    target_text: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _input_ids(tokenized: Any) -> Any:
    if isinstance(tokenized, Mapping):
        return tokenized["input_ids"]
    return tokenized.input_ids


def flat_ids(tok: Any, text: str) -> List[int]:
    ids = _input_ids(tok(text))
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().cpu().tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Expected one tokenized sequence")
        ids = ids[0]
    return [int(x) for x in ids]


def sensitive_answer_field(dataset: str) -> str:
    dataset = str(dataset).lower()
    if dataset == "mcf":
        return "target_new"
    if dataset == "zsre":
        return "target_true"
    raise ValueError(f"Unsupported dataset: {dataset}")


def answer_token_ids(tok: Any, answer: str, *, llama_like: bool) -> List[int]:
    ids = flat_ids(tok, " " + str(answer))
    if llama_like:
        if not ids:
            raise ValueError("Llama-style answer tokenization returned no BOS token")
        ids = ids[1:]
    if not ids:
        raise ValueError(f"Sensitive answer tokenized to no evaluated tokens: {answer!r}")
    return ids


def expand_sensitive_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    dataset: str,
    llama_like: bool,
) -> List[SensitivePredictionCase]:
    """Expand direct forget answers into teacher-forced next-token decisions."""
    field = sensitive_answer_field(dataset)
    cases: List[SensitivePredictionCase] = []
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"Record {position} lacks requested_rewrite")
        subject = str(rr.get("subject", ""))
        prompt_template = str(rr.get("prompt", ""))
        target_block = rr.get(field)
        if not isinstance(target_block, Mapping) or not target_block.get("str"):
            raise ValueError(f"Record {position} lacks sensitive {field}.str")
        answer = str(target_block["str"])
        prompt = prompt_template.format(subject)
        tids = answer_token_ids(tok, answer, llama_like=llama_like)
        case_id = int(record.get("case_id", position))
        for token_index, token_id in enumerate(tids):
            decoded_prefix = tok.decode(tids[:token_index])
            if llama_like and token_index > 0:
                evaluated_prompt = prompt + " " + decoded_prefix
            else:
                evaluated_prompt = prompt + decoded_prefix
            cases.append(
                SensitivePredictionCase(
                    case_id=case_id,
                    record_position=position,
                    token_index=token_index,
                    prompt=evaluated_prompt,
                    target_text=tok.decode([token_id]),
                )
            )
    return cases


def official_target_ids(
    tok: Any,
    cases: Sequence[SensitivePredictionCase],
    *,
    llama_like: bool,
    device: torch.device,
) -> torch.Tensor:
    encoded = tok([c.target_text for c in cases], padding=True, return_tensors="pt")
    ids = _input_ids(encoded)
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
    column = 1 if llama_like else 0
    if ids.ndim != 2 or ids.shape[1] <= column:
        raise ValueError("Target tokenization lacks expected token column")
    return ids[:, column].to(device)


def forward_last_logits(
    model: nn.Module,
    tok: Any,
    cases: Sequence[SensitivePredictionCase],
    device: torch.device,
) -> torch.Tensor:
    encoded = tok([c.prompt for c in cases], padding=True, return_tensors="pt").to(device)
    output = model(**encoded, use_cache=False)
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(cases), device=device)
    return output.logits[rows, positions, :]


@torch.no_grad()
def forward_last_hidden(
    model: nn.Module,
    tok: Any,
    cases: Sequence[SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        chunks.append(output.hidden_states[-1][rows, positions, :].float().detach())
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32, device=device)
    return torch.cat(chunks, dim=0)


def ga_sensitive_logprob(logits: torch.Tensor, tids: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[rows, tids].mean()


def gd_non_sensitive_kl(
    current_logits: torch.Tensor,
    base_logits: torch.Tensor,
    tids: torch.Tensor,
) -> torch.Tensor:
    cur = current_logits.float()
    ref = base_logits.to(device=cur.device, dtype=torch.float32)
    if cur.shape != ref.shape or cur.ndim != 2:
        raise ValueError("current/base logits must have equal [batch,vocab] shape")
    bsz, vocab = cur.shape
    rows = torch.arange(bsz, device=cur.device)
    mask = torch.ones((bsz, vocab), dtype=torch.bool, device=cur.device)
    mask[rows, tids] = False
    cur_rest = cur[mask].view(bsz, vocab - 1)
    ref_rest = ref[mask].view(bsz, vocab - 1)
    cur_logp = F.log_softmax(cur_rest, dim=-1)
    ref_logp = F.log_softmax(ref_rest, dim=-1)
    ref_p = ref_logp.exp()
    return (ref_p * (ref_logp - cur_logp)).sum(dim=-1).mean()


@torch.no_grad()
def cache_base_logits(
    model: nn.Module,
    tok: Any,
    cases: Sequence[SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    cached: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(cases), batch_size):
        cached.append(
            forward_last_logits(model, tok, cases[start : start + batch_size], device)
            .detach()
            .float()
            .cpu()
        )
    return torch.cat(cached, dim=0)


class IndexSampler:
    def __init__(self, total: int, batch_size: int, seed: int) -> None:
        if total <= 0 or batch_size <= 0:
            raise ValueError("Sampler requires positive total and batch size")
        self.total = int(total)
        self.batch_size = min(int(batch_size), self.total)
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.cursor = 0

    def next(self) -> List[int]:
        result: List[int] = []
        while len(result) < self.batch_size:
            if self.cursor >= len(self.order):
                self.order = list(range(self.total))
                self.rng.shuffle(self.order)
                self.cursor = 0
            take = min(self.batch_size - len(result), len(self.order) - self.cursor)
            result.extend(self.order[self.cursor : self.cursor + take])
            self.cursor += take
        return result


@torch.no_grad()
def restore_sensitive_rows_only(
    tied_info: Dict[str, Any],
    base_rows: Dict[str, torch.Tensor],
    sensitive_ids: Sequence[int],
) -> Dict[str, Any]:
    in_w = tied_info["input_weight"]
    out_w = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    ids = torch.tensor(sorted(set(int(x) for x in sensitive_ids)), dtype=torch.long, device=in_w.device)
    if ids.numel() == 0:
        raise RuntimeError("No sensitive vocabulary rows found")

    trained_in = in_w.index_select(0, ids).detach().clone()
    trained_out = (
        trained_in
        if tied
        else out_w.index_select(0, ids.to(out_w.device)).detach().clone()
    )

    in_w.copy_(base_rows["input"].to(device=in_w.device, dtype=in_w.dtype))
    in_w.index_copy_(0, ids, trained_in)
    if not tied:
        out_ids = ids.to(out_w.device)
        out_w.copy_(base_rows["output"].to(device=out_w.device, dtype=out_w.dtype))
        out_w.index_copy_(0, out_ids, trained_out)

    return {
        "policy": "base_everywhere_plus_sensitive_trained_rows",
        "sensitive_row_count": int(ids.numel()),
        "sensitive_token_ids": [int(x) for x in ids.detach().cpu().tolist()],
        "all_non_sensitive_rows_restored_to_base": True,
        "protected_only_rows_restored_to_base": True,
        "factual_boost_applied": False,
        "tied_input_output": tied,
    }


def untie_and_freeze_output_head(model: nn.Module) -> nn.Module:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise ValueError("Model must expose input/output embeddings")
    if input_embeddings.weight.data_ptr() == output_embeddings.weight.data_ptr():
        if not isinstance(output_embeddings, nn.Linear):
            raise ValueError("Tied lm_head must be nn.Linear to clone and untie")
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
    if model.get_input_embeddings().weight.data_ptr() == output_embeddings.weight.data_ptr():
        raise RuntimeError("lm_head remains tied after clone")
    return output_embeddings


def orthonormal_row_basis(rows: torch.Tensor, max_rank: Optional[int]) -> torch.Tensor:
    if rows.numel() == 0:
        hidden = rows.shape[-1] if rows.ndim == 2 else 0
        return rows.new_empty((0, hidden), dtype=torch.float32)
    rows = rows.float()
    _, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
    tolerance = max(rows.shape) * torch.finfo(rows.dtype).eps * singular_values.max().clamp_min(1.0)
    rank = int((singular_values > tolerance).sum().item())
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return right[:rank].contiguous()


class SelectedRowDelta(nn.Module):
    """Shared sparse-row parameterization: fixed hidden basis or full row delta."""

    def __init__(
        self,
        n_rows: int,
        hidden_size: int,
        *,
        direction_basis: Optional[torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.n_rows = int(n_rows)
        self.hidden_size = int(hidden_size)
        if direction_basis is not None:
            if direction_basis.ndim != 2 or direction_basis.shape[1] != hidden_size:
                raise ValueError("direction basis has incompatible shape")
            self.register_buffer("direction_basis", direction_basis.to(device=device, dtype=torch.float32))
            self.coefficients = nn.Parameter(
                torch.zeros((n_rows, direction_basis.shape[0]), device=device, dtype=torch.float32)
            )
            self.raw_delta = None
        else:
            self.direction_basis = None
            self.coefficients = None
            self.raw_delta = nn.Parameter(
                torch.zeros((n_rows, hidden_size), device=device, dtype=torch.float32)
            )

    def effective_delta(self) -> torch.Tensor:
        if self.coefficients is not None:
            return self.coefficients @ self.direction_basis
        if self.raw_delta is None:
            raise RuntimeError("SelectedRowDelta has no parameter")
        return self.raw_delta

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())


def register_output_delta_hook(
    output_layer: nn.Module,
    row_ids: Sequence[int],
    delta_getter: Callable[[], torch.Tensor],
):
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=output_layer.weight.device)

    def hook(_module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
        if ids.numel() == 0:
            return output
        hidden = inputs[0]
        delta = delta_getter().to(device=hidden.device, dtype=torch.float32)
        correction = torch.matmul(hidden.float(), delta.transpose(0, 1))
        updated = output.clone()
        updated[..., ids] = updated[..., ids] + correction.to(dtype=updated.dtype)
        return updated

    return output_layer.register_forward_hook(hook)


@torch.no_grad()
def materialize_output_delta(
    output_layer: nn.Module,
    row_ids: Sequence[int],
    delta: torch.Tensor,
) -> None:
    if len(row_ids) != delta.shape[0]:
        raise ValueError("row id count does not match delta rows")
    if not row_ids:
        return
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=output_layer.weight.device)
    current = output_layer.weight.index_select(0, ids)
    output_layer.weight.index_copy_(
        0,
        ids,
        current + delta.to(device=current.device, dtype=current.dtype),
    )


def parse_rank_list(text: str) -> List[int]:
    ranks: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError("Candidate ranks must be >= 0; 0 means unrestricted")
        if value not in ranks:
            ranks.append(value)
    if not ranks:
        raise ValueError("No candidate repair ranks provided")
    return ranks


def parse_scales(text: str) -> List[float]:
    scales: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Candidate scales must be finite and non-negative")
        if value not in scales:
            scales.append(value)
    if not scales:
        raise ValueError("No candidate scales provided")
    return scales


def choose_scale(reports: Sequence[Dict[str, Any]]) -> float:
    zero = [float(r["scale"]) for r in reports if int(r["direct_failures"]) == 0]
    if zero:
        return min(zero)
    best = min(
        reports,
        key=lambda r: (int(r["direct_failures"]), float(r["scale"])),
    )
    return float(best["scale"])
