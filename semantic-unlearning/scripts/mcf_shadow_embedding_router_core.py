#!/usr/bin/env python3
"""Passive shadow-embedding routing with an isolated residual actuator.

This module implements the architectural successor to the rejected V3.6.2
embedding-keyed neuron candidate.  The ordinary model path always uses the
unaltered Base embedding table.  A second, no-gradient shadow forward applies
the frozen V6.2 sparse embedding delta and is used only to form a contextual
feature::

    d_l(x) = h_l(x; E + delta_E) - h_l(x; E)

An exact complete-subject key and a record-specific semantic readout of
``d_l`` jointly control a constant record-specific residual.  The semantic
readout has a fixed negative bias, so removing the shadow embedding delta
makes every semantic gate close by construction.  If either gate is closed,
the forward hook returns the original Base layer output object without an
arithmetic round trip.

The implementation intentionally contains no official-evaluation loader and
does not mutate model embeddings, the LM head, or any Transformer parameter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from scoped_span_edit import SpanGateRouter, find_decoder_layers


PROTOCOL = "mcf_passive_shadow_embedding_context_router_v4_0"
SCHEMA_VERSION = 1


def _hidden_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("decoder-layer output must be a Tensor or tuple beginning with one")


def _replace_layer_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (hidden, *output[1:])
    raise TypeError("decoder-layer output must be a Tensor or tuple beginning with one")


def causal_after_subject_mask(span_masks: torch.Tensor) -> torch.Tensor:
    """Return positions at or after the end of each complete subject span.

    ``span_masks`` is ``[batch, records, sequence]``.  A correction must never
    affect tokens preceding the full key, including the first pieces of a
    multi-token subject.  Duplicate-subject records remain separate columns.
    """

    if span_masks.ndim != 3 or span_masks.dtype is not torch.bool:
        raise ValueError("span masks must be a boolean [batch, records, sequence] tensor")
    if int(span_masks.shape[-1]) == 0:
        return span_masks.clone()
    next_is_subject = torch.zeros_like(span_masks)
    next_is_subject[..., :-1] = span_masks[..., 1:]
    end_positions = span_masks & ~next_is_subject
    return end_positions.to(torch.int64).cumsum(dim=-1).gt(0)


def fixed_bias_router_scores(
    features: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    fixed_bias: float = -1.0,
) -> torch.Tensor:
    """Apply record readouts to contextual shadow differences.

    ``features`` may be ``[..., hidden]`` and the result is ``[..., records]``.
    There is deliberately no learned bias: when the embedding delta is absent,
    the shadow feature is exact zero and all scores equal ``fixed_bias``.
    """

    if features.ndim < 2:
        raise ValueError("router features must end in a hidden dimension")
    if router_weight.ndim != 2:
        raise ValueError("router weight must be [records, hidden]")
    if int(features.shape[-1]) != int(router_weight.shape[-1]):
        raise ValueError("router feature and weight hidden sizes differ")
    return torch.nn.functional.linear(
        features.float(), router_weight.float(), bias=None
    ) + float(fixed_bias)


def record_balanced_router_hinge(
    scores: torch.Tensor,
    active_subjects: torch.Tensor,
    positive_labels: torch.Tensor,
    *,
    positive_floor: float = 1.0,
    negative_ceiling: float = -0.25,
    tail_k: int = 2,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Multi-label, record-balanced semantic router objective.

    Only exact-subject candidate cells participate.  Positive labels are
    pushed above ``positive_floor``; same-subject wrong-relation cells are
    pushed below ``negative_ceiling``.  Per-record means and worst tails avoid
    allowing records with more prompt variants to dominate optimization.
    """

    if scores.ndim != 2:
        raise ValueError("router scores must be [prompts, records]")
    if active_subjects.shape != scores.shape or positive_labels.shape != scores.shape:
        raise ValueError("router masks and labels must match scores")
    if active_subjects.dtype is not torch.bool or positive_labels.dtype is not torch.bool:
        raise ValueError("router masks and labels must be boolean")
    if bool((positive_labels & ~active_subjects).any()):
        raise ValueError("every positive router label must have a complete subject key")
    if int(tail_k) <= 0:
        raise ValueError("tail_k must be positive")

    positive_error = torch.relu(float(positive_floor) - scores).square()
    negative_error = torch.relu(scores - float(negative_ceiling)).square()
    per_record = []
    positive_cells = []
    negative_cells = []
    for record in range(int(scores.shape[1])):
        pos_mask = positive_labels[:, record]
        neg_mask = active_subjects[:, record] & ~pos_mask
        if not bool(pos_mask.any()):
            raise ValueError(f"router record {record} has no positive cells")
        pos = positive_error[:, record][pos_mask]
        neg = negative_error[:, record][neg_mask]
        k_pos = min(int(tail_k), int(pos.numel()))
        pos_term = pos.mean() + pos.topk(k_pos).values.mean()
        if int(neg.numel()):
            k_neg = min(int(tail_k), int(neg.numel()))
            neg_term = neg.mean() + neg.topk(k_neg).values.mean()
            negative_cells.append(neg)
        else:
            neg_term = scores[:, record].sum() * 0.0
        per_record.append(pos_term + neg_term)
        positive_cells.append(pos)
    loss = torch.stack(per_record).mean()
    all_positive = torch.cat(positive_cells)
    all_negative = (
        torch.cat(negative_cells)
        if negative_cells
        else scores.new_empty((0,), dtype=torch.float32)
    )
    return loss, {
        "positive_squared_hinge_mean": all_positive.mean(),
        "negative_squared_hinge_mean": (
            all_negative.mean()
            if int(all_negative.numel())
            else scores.sum() * 0.0
        ),
        "positive_violations": positive_error[positive_labels].gt(0).sum(),
        "negative_violations": negative_error[
            active_subjects & ~positive_labels
        ].gt(0).sum(),
    }


def router_certificate(
    scores: torch.Tensor,
    active_subjects: torch.Tensor,
    positive_labels: torch.Tensor,
    *,
    positive_floor: float = 0.25,
    negative_ceiling: float = -0.25,
) -> Dict[str, Any]:
    """Return an all-cell direct/calibration/held-out gate certificate."""

    if scores.shape != active_subjects.shape or scores.shape != positive_labels.shape:
        raise ValueError("certificate tensors must have identical shapes")
    positives = scores[positive_labels]
    negatives = scores[active_subjects & ~positive_labels]
    if not int(positives.numel()):
        raise ValueError("router certificate requires positive cells")
    positive_min = float(positives.min())
    negative_max = float(negatives.max()) if int(negatives.numel()) else float("-inf")
    positive_failures = int(positives.lt(float(positive_floor)).sum())
    negative_failures = int(negatives.gt(float(negative_ceiling)).sum())
    return {
        "positive_cells": int(positives.numel()),
        "negative_cells": int(negatives.numel()),
        "positive_min": positive_min,
        "positive_median": float(positives.median()),
        "negative_max": negative_max,
        "positive_floor": float(positive_floor),
        "negative_ceiling": float(negative_ceiling),
        "positive_failures": positive_failures,
        "negative_failures": negative_failures,
        "passed": positive_failures == 0 and negative_failures == 0,
    }


class PassiveShadowRouter(nn.Module):
    """Record-specific linear readout with a fixed writer-necessity bias."""

    def __init__(
        self,
        records: int,
        hidden_size: int,
        *,
        fixed_bias: float = -1.0,
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if int(records) <= 0 or int(hidden_size) <= 0:
            raise ValueError("router dimensions must be positive")
        if not float(fixed_bias) < float(threshold):
            raise ValueError("fixed writer-off bias must lie below the gate threshold")
        self.records = int(records)
        self.hidden_size = int(hidden_size)
        self.fixed_bias = float(fixed_bias)
        self.threshold = float(threshold)
        self.weight = nn.Parameter(torch.zeros(records, hidden_size, dtype=torch.float32))

    def scores(self, features: torch.Tensor) -> torch.Tensor:
        return fixed_bias_router_scores(
            features, self.weight, fixed_bias=self.fixed_bias
        )

    def gates(self, features: torch.Tensor, subject_candidates: torch.Tensor) -> torch.Tensor:
        scores = self.scores(features)
        if scores.shape != subject_candidates.shape:
            raise ValueError("subject candidates do not match router scores")
        return subject_candidates & scores.ge(self.threshold)

    @torch.no_grad()
    def clamp_row_norm_(self, maximum: float) -> None:
        if float(maximum) <= 0:
            raise ValueError("router row-norm cap must be positive")
        norms = self.weight.norm(dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(norms),
            torch.tensor(float(maximum), device=norms.device) / norms.clamp_min(1e-12),
        )
        self.weight.mul_(scale)


@dataclass
class ShadowRuntimeAudit:
    shadow_forwards: int = 0
    base_forwards: int = 0
    open_gate_cells: int = 0
    corrected_rows: int = 0
    closed_identity_rows: int = 0


class ShadowEmbeddingResidualBranch(nn.Module):
    """Dual-pass hook state and constant record-specific actuator residuals."""

    def __init__(
        self,
        layer: nn.Module,
        span_router: SpanGateRouter,
        semantic_router: PassiveShadowRouter,
        hidden_size: int,
        *,
        residual_reference_norms: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if semantic_router.hidden_size != int(hidden_size):
            raise ValueError("semantic router and residual hidden sizes differ")
        if span_router.n_records != semantic_router.records:
            raise ValueError("subject and semantic router record counts differ")
        self.span_router = span_router
        self.semantic_router = semantic_router
        self.hidden_size = int(hidden_size)
        self.enabled = True
        self.shadow_writer_enabled = True
        self.mode = "idle"
        self.shadow_hidden: Optional[torch.Tensor] = None
        self.last_scores: Optional[torch.Tensor] = None
        self.last_gates: Optional[torch.Tensor] = None
        self.audit = ShadowRuntimeAudit()
        self.residual = nn.Parameter(
            torch.zeros(semantic_router.records, hidden_size, dtype=torch.float32)
        )
        if residual_reference_norms is None:
            reference = torch.ones(semantic_router.records, dtype=torch.float32)
        else:
            reference = residual_reference_norms.detach().float().reshape(-1)
            if int(reference.numel()) != semantic_router.records:
                raise ValueError("residual reference norms must cover every record")
            if not bool(torch.isfinite(reference).all() & reference.gt(0).all()):
                raise ValueError("residual reference norms must be positive and finite")
        self.register_buffer("residual_reference_norms", reference)
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = _hidden_from_layer_output(output)
        if self.mode == "shadow_capture":
            self.shadow_hidden = hidden.detach()
            self.audit.shadow_forwards += 1
            return output
        if self.mode != "base_apply":
            return output
        self.audit.base_forwards += 1
        if not self.enabled:
            self.audit.closed_identity_rows += int(hidden.shape[0])
            return output
        if self.shadow_hidden is None or self.shadow_hidden.shape != hidden.shape:
            raise RuntimeError("shadow/Base layer states are absent or shape-incompatible")
        route = self.span_router.state
        if route is None:
            raise RuntimeError("subject router state was not populated by the Base pass")
        candidates = causal_after_subject_mask(route.span_masks).permute(0, 2, 1)
        candidates = candidates.to(device=hidden.device)
        features = self.shadow_hidden.to(hidden.device).float() - hidden.detach().float()
        scores = self.semantic_router.scores(features)
        instant = candidates & scores.ge(self.semantic_router.threshold)
        # Once the complete prompt establishes the relation, keep its branch
        # open while scoring answer tokens in that same causal sequence.
        gates = instant.to(torch.int64).cummax(dim=1).values.bool()
        self.last_scores = scores.detach()
        self.last_gates = gates.detach()
        open_cells = int(gates.sum())
        self.audit.open_gate_cells += open_cells
        row_open = gates.any(dim=(1, 2))
        self.audit.corrected_rows += int(row_open.sum())
        self.audit.closed_identity_rows += int((~row_open).sum())
        if not open_cells:
            return output
        correction = torch.einsum(
            "bsr,rh->bsh", gates.to(self.residual.dtype), self.residual
        )
        return _replace_layer_hidden(
            output, hidden + correction.to(device=hidden.device, dtype=hidden.dtype)
        )

    @torch.no_grad()
    def zero_residual_(self) -> None:
        self.residual.zero_()

    def relative_residual_norms(self) -> torch.Tensor:
        return self.residual.detach().norm(dim=1) / self.residual_reference_norms.clamp_min(
            1e-12
        )

    @torch.no_grad()
    def clamp_residual_relative_(self, cap: float) -> None:
        if float(cap) <= 0:
            raise ValueError("residual cap must be positive")
        norms = self.residual.norm(dim=1)
        limits = float(cap) * self.residual_reference_norms.to(norms.device)
        scale = torch.minimum(torch.ones_like(norms), limits / norms.clamp_min(1e-12))
        self.residual.mul_(scale.unsqueeze(1))

    def close(self) -> None:
        self._handle.remove()
        self.shadow_hidden = None
        self.mode = "closed"


class ShadowDualPathCausalLM(nn.Module):
    """Run a passive marked shadow before the untouched Base forward.

    ``embedding_writer`` must be a toggleable hook with an ``enabled`` field.
    Its delta is enabled only for the no-gradient shadow pass and disabled for
    the returned Base pass.  No model parameter is replaced or materialized.
    """

    def __init__(
        self,
        base_model: nn.Module,
        embedding_writer: Any,
        branch: ShadowEmbeddingResidualBranch,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.embedding_writer = embedding_writer
        self.branch = branch

    @property
    def config(self) -> Any:
        return self.base_model.config

    def get_input_embeddings(self) -> nn.Module:
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.base_model.get_output_embeddings()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("inputs_embeds") is not None:
            raise RuntimeError("shadow routing requires input_ids for exact subject keys")
        if kwargs.get("input_ids") is None and not args:
            raise RuntimeError("shadow routing requires input_ids")
        forwarded = dict(kwargs)
        forwarded["use_cache"] = False
        old_writer = bool(self.embedding_writer.enabled)
        self.branch.shadow_hidden = None
        try:
            self.embedding_writer.enabled = bool(self.branch.shadow_writer_enabled)
            self.branch.mode = "shadow_capture"
            with torch.no_grad():
                self.base_model(*args, **forwarded)
            self.embedding_writer.enabled = False
            self.branch.mode = "base_apply"
            return self.base_model(*args, **forwarded)
        finally:
            self.embedding_writer.enabled = old_writer
            self.branch.mode = "idle"
            self.branch.shadow_hidden = None


@dataclass
class InstalledShadowCandidate:
    """Handles for a sidecar installed without materializing any Base edit."""

    model: ShadowDualPathCausalLM
    embedding_writer: Any
    span_router: SpanGateRouter
    semantic_router: PassiveShadowRouter
    branch: ShadowEmbeddingResidualBranch

    def close(self) -> None:
        self.branch.close()
        self.span_router.close()
        self.embedding_writer.remove()


def install_shadow_candidate(
    base_model: nn.Module,
    state: Mapping[str, Any],
) -> InstalledShadowCandidate:
    """Install a frozen V4 sidecar around a freshly loaded Base model.

    The loader constructs hooks and external parameters only.  It never calls
    ``index_copy_`` or otherwise writes into the Base embedding, Transformer,
    or output head.  Base-model hash validation remains the evaluator's
    responsibility because the candidate is deliberately not a model clone.
    """

    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError("unsupported shadow candidate schema")
    if state.get("protocol") != PROTOCOL:
        raise RuntimeError("shadow candidate protocol mismatch")
    architecture = state.get("architecture")
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != expected
        for key, expected in {
            "main_embedding_path": "unaltered_base_embedding",
            "shadow_embedding_path": "frozen_v6_2_sparse_delta",
            "shadow_gradient_enabled": False,
            "base_embedding_mutated": False,
            "lm_head_mutated": False,
        }.items()
    ):
        raise RuntimeError("candidate does not preserve the passive shadow boundary")
    input_embedding = base_model.get_input_embeddings()
    if input_embedding is None or not hasattr(input_embedding, "weight"):
        raise RuntimeError("Base model has no input embedding table")
    layers = find_decoder_layers(base_model)
    layer_index = int(state["layer_index"])
    if not 0 <= layer_index < len(layers):
        raise RuntimeError("candidate layer is outside the Base model")
    case_ids = [int(value) for value in state.get("case_ids", [])]
    subjects = [str(value) for value in state.get("subjects", [])]
    patterns = state.get("subject_patterns")
    row_ids = [int(value) for value in state.get("embedding_row_ids", [])]
    delta = state.get("embedding_delta")
    router_weight = state.get("semantic_router_weight")
    residual = state.get("actuator_residual")
    reference = state.get("residual_reference_norms")
    records = len(case_ids)
    hidden = int(input_embedding.weight.shape[1])
    if (
        not records
        or len(subjects) != records
        or not isinstance(patterns, list)
        or len(patterns) != records
        or not row_ids
        or not isinstance(delta, torch.Tensor)
        or delta.shape != (len(row_ids), hidden)
        or not isinstance(router_weight, torch.Tensor)
        or router_weight.shape != (records, hidden)
        or not isinstance(residual, torch.Tensor)
        or residual.shape != (records, hidden)
        or not isinstance(reference, torch.Tensor)
        or reference.reshape(-1).shape != (records,)
    ):
        raise RuntimeError("shadow candidate tensor or record shapes are invalid")

    # Local import prevents the V3.6.2 mechanism module from depending on its
    # architectural successor.
    from mcf_embedding_keyed_neuron_core import ToggleableEmbeddingDelta

    writer = ToggleableEmbeddingDelta(input_embedding, row_ids, delta)
    writer.enabled = False
    span = SpanGateRouter(input_embedding, patterns, subjects=subjects, model=base_model)
    semantic = PassiveShadowRouter(records, hidden, fixed_bias=-1.0, threshold=0.0).to(
        input_embedding.weight.device
    )
    with torch.no_grad():
        semantic.weight.copy_(router_weight.to(semantic.weight.device))
    semantic.weight.requires_grad_(False)
    branch = ShadowEmbeddingResidualBranch(
        layers[layer_index],
        span,
        semantic,
        hidden,
        residual_reference_norms=reference.to(input_embedding.weight.device),
    ).to(input_embedding.weight.device)
    with torch.no_grad():
        branch.residual.copy_(residual.to(branch.residual.device))
    branch.residual.requires_grad_(False)
    wrapped = ShadowDualPathCausalLM(base_model, writer, branch)
    return InstalledShadowCandidate(wrapped, writer, span, semantic, branch)


def shadow_candidate_state(
    *,
    layer_index: int,
    case_ids: Sequence[int],
    subjects: Sequence[str],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    embedding_row_ids: Sequence[int],
    embedding_delta: torch.Tensor,
    semantic_router: PassiveShadowRouter,
    branch: ShadowEmbeddingResidualBranch,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    """Create the frozen sidecar payload without changing Base parameters."""

    if len(case_ids) != len(subjects) or len(case_ids) != semantic_router.records:
        raise ValueError("candidate record metadata is inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mcf_passive_shadow_embedding_context_router_candidate",
        "protocol": PROTOCOL,
        "architecture": {
            "main_embedding_path": "unaltered_base_embedding",
            "shadow_embedding_path": "frozen_v6_2_sparse_delta",
            "shadow_gradient_enabled": False,
            "router_feature": "shadow_minus_base_layer_state",
            "complete_subject_key_required": True,
            "semantic_gate_threshold": semantic_router.threshold,
            "semantic_fixed_writer_off_bias": semantic_router.fixed_bias,
            "actuator": "constant_record_specific_layer_residual",
            "closed_gate_identity": "returns_original_layer_output_object",
            "base_embedding_mutated": False,
            "lm_head_mutated": False,
        },
        "layer_index": int(layer_index),
        "case_ids": [int(value) for value in case_ids],
        "subjects": [str(value) for value in subjects],
        "subject_patterns": [
            [[int(token) for token in pattern] for pattern in record]
            for record in subject_patterns
        ],
        "embedding_row_ids": [int(value) for value in embedding_row_ids],
        "embedding_delta": embedding_delta.detach().cpu(),
        "semantic_router_weight": semantic_router.weight.detach().cpu(),
        "actuator_residual": branch.residual.detach().cpu(),
        "residual_reference_norms": branch.residual_reference_norms.detach().cpu(),
        "source_hashes": {str(key): str(value) for key, value in source_hashes.items()},
        "official_evaluation_prompts_seen": 0,
    }
