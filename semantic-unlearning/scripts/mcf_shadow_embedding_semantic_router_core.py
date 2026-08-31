#!/usr/bin/env python3
"""V4.1 factorized shadow-marker and Base-semantic contextual routing.

V4.0 asked one linear readout of ``shadow - Base`` to establish both marker
presence and relation identity.  Its first development held-out bank showed
that those roles do not transfer together.  V4.1 separates them:

* an exact complete-subject token key establishes record candidacy;
* a nonzero shadow/Base RMS difference establishes that the frozen V6.2
  embedding marker is present;
* a rank-limited shared Base-hidden subspace establishes relation semantics;
* only their conjunction can apply a constant record residual.

The embedding marker therefore remains causally necessary, while semantic
classification uses the frozen model representation designed to generalize
across surface forms.  Closed routes still return the original layer output
object without arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import mcf_shadow_embedding_router_core as v4
from scoped_span_edit import SpanGateRouter, find_decoder_layers


PROTOCOL = "mcf_shadow_marker_base_semantic_router_v4_1"
SCHEMA_VERSION = 1


def rms_normalize_hidden(hidden: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    if hidden.ndim < 2:
        raise ValueError("semantic hidden states must end in a hidden dimension")
    return hidden.float() * torch.rsqrt(
        hidden.float().square().mean(dim=-1, keepdim=True) + float(epsilon)
    )


def shadow_marker_rms(shadow_hidden: torch.Tensor, base_hidden: torch.Tensor) -> torch.Tensor:
    if shadow_hidden.shape != base_hidden.shape or shadow_hidden.ndim < 2:
        raise ValueError("shadow and Base hidden states must have equal shapes")
    return (
        shadow_hidden.float() - base_hidden.detach().float()
    ).square().mean(dim=-1).sqrt()


class BaseSemanticRouter(nn.Module):
    """Relation-tied readouts in a frozen shared low-rank Base subspace.

    V4.0's rejected readout had one independently trainable 3,072-dimensional
    vector per record.  With only a handful of positive/negative contexts per
    record, that parameterization could memorize the fit bank.  V4.1 derives
    one rank-limited basis from *all* fit records, freezes it, and trains only
    ``rank`` coefficients plus one bias per distinct relation.  Records that
    share a relation therefore share the exact same semantic classifier.
    """

    def __init__(
        self,
        records: int,
        hidden_size: int,
        shared_basis: torch.Tensor,
        record_relation_index: Sequence[int] | torch.Tensor,
        *,
        threshold: float = 0.0,
        initial_bias: float = -0.5,
    ) -> None:
        super().__init__()
        if int(records) <= 0 or int(hidden_size) <= 0:
            raise ValueError("semantic router dimensions must be positive")
        basis = shared_basis.detach().float()
        relation_index = torch.as_tensor(record_relation_index, dtype=torch.long)
        if basis.ndim != 2 or int(basis.shape[1]) != int(hidden_size):
            raise ValueError("shared semantic basis must be [rank, hidden]")
        if not 1 <= int(basis.shape[0]) <= 4:
            raise ValueError("shared semantic rank must lie in [1, 4]")
        if relation_index.shape != (int(records),) or bool(relation_index.lt(0).any()):
            raise ValueError("record relation indices must cover every record")
        relation_count = int(relation_index.max()) + 1
        if set(relation_index.tolist()) != set(range(relation_count)):
            raise ValueError("relation indices must be contiguous and used")
        gram = basis @ basis.T
        if not torch.allclose(
            gram, torch.eye(int(basis.shape[0])), atol=1e-5, rtol=1e-5
        ):
            raise ValueError("shared semantic basis rows must be orthonormal")
        self.records = int(records)
        self.hidden_size = int(hidden_size)
        self.rank = int(basis.shape[0])
        self.relation_count = relation_count
        self.threshold = float(threshold)
        self.register_buffer("shared_basis", basis)
        self.register_buffer("record_relation_index", relation_index)
        self.relation_coefficients = nn.Parameter(
            torch.zeros(relation_count, self.rank, dtype=torch.float32)
        )
        self.relation_bias = nn.Parameter(
            torch.full((relation_count,), float(initial_bias), dtype=torch.float32)
        )

    def scores(self, base_hidden: torch.Tensor) -> torch.Tensor:
        normalized = rms_normalize_hidden(base_hidden)
        shared = torch.nn.functional.linear(normalized, self.shared_basis.float())
        relation_scores = torch.nn.functional.linear(
            shared,
            self.relation_coefficients.float(),
            self.relation_bias.float(),
        )
        return relation_scores.index_select(-1, self.record_relation_index)

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.relation_coefficients.numel() + self.relation_bias.numel())

    @property
    def effective_record_weight(self) -> torch.Tensor:
        relation_weight = self.relation_coefficients.float() @ self.shared_basis.float()
        return relation_weight.index_select(0, self.record_relation_index)

    @torch.no_grad()
    def clamp_coefficient_norm_(self, maximum: float) -> None:
        if float(maximum) <= 0:
            raise ValueError("semantic coefficient norm cap must be positive")
        norms = self.relation_coefficients.norm(dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(norms),
            torch.tensor(float(maximum), device=norms.device)
            / norms.clamp_min(1e-12),
        )
        self.relation_coefficients.mul_(scale)


def fit_shared_contrast_basis(
    base_hidden: torch.Tensor,
    active_subjects: torch.Tensor,
    positive_labels: torch.Tensor,
    record_relation_index: Sequence[int] | torch.Tensor,
    *,
    rank: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Fit one frozen supervised contrast subspace from the fit split only.

    Each distinct relation contributes one normalized positive-minus-negative
    centroid contrast.  An SVD over those pooled contrasts yields at most four
    orthonormal directions.  No record-specific hidden-dimensional vector is
    optimized or stored.
    """

    if base_hidden.ndim != 2:
        raise ValueError("Base semantic states must be [prompts, hidden]")
    if active_subjects.shape != positive_labels.shape:
        raise ValueError("semantic masks must match")
    if int(active_subjects.shape[0]) != int(base_hidden.shape[0]):
        raise ValueError("semantic states and masks have different prompt counts")
    relation_index = torch.as_tensor(record_relation_index, dtype=torch.long)
    if relation_index.shape != (int(active_subjects.shape[1]),):
        raise ValueError("record relation indices do not match router groups")
    relation_count = int(relation_index.max()) + 1
    if not 1 <= int(rank) <= min(4, relation_count, int(base_hidden.shape[1])):
        raise ValueError("shared semantic rank is invalid")
    normalized = rms_normalize_hidden(base_hidden).cpu()
    active = active_subjects.cpu()
    labels = positive_labels.cpu()
    contrasts = []
    counts = []
    for relation in range(relation_count):
        groups = relation_index.eq(relation)
        pos_rows = labels[:, groups].any(dim=1)
        neg_rows = (active[:, groups] & ~labels[:, groups]).any(dim=1)
        if not bool(pos_rows.any()) or not bool(neg_rows.any()):
            raise ValueError(f"relation {relation} lacks positive or negative fit cells")
        contrast = normalized[pos_rows].mean(dim=0) - normalized[neg_rows].mean(dim=0)
        contrast_norm = contrast.norm().clamp_min(1e-12)
        contrasts.append(contrast / contrast_norm)
        counts.append(
            {
                "relation_index": relation,
                "positive_rows": int(pos_rows.sum()),
                "negative_rows": int(neg_rows.sum()),
                "raw_contrast_norm": float(contrast_norm),
            }
        )
    matrix = torch.stack(contrasts)
    _u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[: int(rank)].contiguous()
    explained = singular_values.square()
    report = {
        "method": "pooled_relation_centroid_contrast_svd",
        "rank": int(rank),
        "hidden_size": int(base_hidden.shape[1]),
        "relations": relation_count,
        "basis_trainable": False,
        "singular_values": [float(value) for value in singular_values[: int(rank)]],
        "explained_energy_fraction": float(
            explained[: int(rank)].sum() / explained.sum().clamp_min(1e-12)
        ),
        "relation_fit_counts": counts,
    }
    return basis, report


def marker_certificate(
    marker_rms: torch.Tensor,
    active_subjects: torch.Tensor,
    positive_labels: torch.Tensor,
    *,
    threshold: float,
) -> Dict[str, Any]:
    """Require marker presence for every exact-subject development cell."""

    if marker_rms.ndim != 1 or int(marker_rms.shape[0]) != int(active_subjects.shape[0]):
        raise ValueError("marker RMS must provide one value per prompt")
    if active_subjects.shape != positive_labels.shape:
        raise ValueError("marker certificate masks differ")
    relevant_rows = active_subjects.any(dim=1)
    positives = positive_labels.any(dim=1)
    if not bool(positives.any()):
        raise ValueError("marker certificate requires positive prompts")
    relevant = marker_rms[relevant_rows]
    positive_values = marker_rms[positives]
    failures = int(relevant.le(float(threshold)).sum())
    return {
        "threshold": float(threshold),
        "relevant_prompts": int(relevant.numel()),
        "positive_prompts": int(positive_values.numel()),
        "relevant_min": float(relevant.min()),
        "positive_min": float(positive_values.min()),
        "failures": failures,
        "writer_off_rms": 0.0,
        "writer_off_gate_open": False,
        "passed": failures == 0 and float(threshold) > 0.0,
    }


@dataclass
class FactorizedRuntimeAudit:
    shadow_forwards: int = 0
    base_forwards: int = 0
    marker_open_cells: int = 0
    semantic_open_cells: int = 0
    joint_open_cells: int = 0
    corrected_rows: int = 0
    closed_identity_rows: int = 0


class ShadowMarkerSemanticResidualBranch(nn.Module):
    """Conjunctive subject, marker-presence, and relation-semantic branch."""

    def __init__(
        self,
        layer: nn.Module,
        span_router: SpanGateRouter,
        semantic_router: BaseSemanticRouter,
        hidden_size: int,
        *,
        marker_rms_threshold: float,
        residual_reference_norms: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if span_router.n_records != semantic_router.records:
            raise ValueError("subject and semantic router record counts differ")
        if semantic_router.hidden_size != int(hidden_size):
            raise ValueError("semantic and actuator hidden sizes differ")
        if not float(marker_rms_threshold) > 0.0:
            raise ValueError("marker RMS threshold must be strictly positive")
        self.span_router = span_router
        self.semantic_router = semantic_router
        self.hidden_size = int(hidden_size)
        self.marker_rms_threshold = float(marker_rms_threshold)
        self.enabled = True
        self.shadow_writer_enabled = True
        self.mode = "idle"
        self.shadow_hidden: Optional[torch.Tensor] = None
        self.last_scores: Optional[torch.Tensor] = None
        self.last_marker_rms: Optional[torch.Tensor] = None
        self.last_marker_gates: Optional[torch.Tensor] = None
        self.last_gates: Optional[torch.Tensor] = None
        self.audit = FactorizedRuntimeAudit()
        self.residual = nn.Parameter(
            torch.zeros(semantic_router.records, hidden_size, dtype=torch.float32)
        )
        reference = (
            torch.ones(semantic_router.records, dtype=torch.float32)
            if residual_reference_norms is None
            else residual_reference_norms.detach().float().reshape(-1)
        )
        if int(reference.numel()) != semantic_router.records or not bool(
            torch.isfinite(reference).all() & reference.gt(0).all()
        ):
            raise ValueError("residual reference norms must cover every record")
        self.register_buffer("residual_reference_norms", reference)
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden = v4._hidden_from_layer_output(output)
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
            raise RuntimeError("factorized branch lacks its paired shadow state")
        route = self.span_router.state
        if route is None:
            raise RuntimeError("exact subject state was not populated")
        candidates = v4.causal_after_subject_mask(route.span_masks).permute(0, 2, 1)
        candidates = candidates.to(hidden.device)
        marker_values = shadow_marker_rms(self.shadow_hidden.to(hidden.device), hidden)
        marker_open = marker_values.gt(self.marker_rms_threshold).unsqueeze(-1)
        scores = self.semantic_router.scores(hidden.detach())
        semantic_open = scores.ge(self.semantic_router.threshold)
        instant = candidates & marker_open & semantic_open
        gates = instant.to(torch.int64).cummax(dim=1).values.bool()
        self.last_scores = scores.detach()
        self.last_marker_rms = marker_values.detach()
        self.last_marker_gates = marker_open.detach()
        self.last_gates = gates.detach()
        self.audit.marker_open_cells += int((candidates & marker_open).sum())
        self.audit.semantic_open_cells += int((candidates & semantic_open).sum())
        self.audit.joint_open_cells += int(gates.sum())
        row_open = gates.any(dim=(1, 2))
        self.audit.corrected_rows += int(row_open.sum())
        self.audit.closed_identity_rows += int((~row_open).sum())
        if not bool(gates.any()):
            return output
        correction = torch.einsum(
            "bsr,rh->bsh", gates.to(self.residual.dtype), self.residual
        )
        return v4._replace_layer_hidden(
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
        if not float(cap) > 0.0:
            raise ValueError("residual cap must be positive")
        norms = self.residual.norm(dim=1)
        limits = float(cap) * self.residual_reference_norms.to(norms.device)
        scale = torch.minimum(torch.ones_like(norms), limits / norms.clamp_min(1e-12))
        self.residual.mul_(scale.unsqueeze(1))

    def close(self) -> None:
        self._handle.remove()
        self.shadow_hidden = None
        self.mode = "closed"


@dataclass
class InstalledFactorizedCandidate:
    model: v4.ShadowDualPathCausalLM
    embedding_writer: Any
    span_router: SpanGateRouter
    semantic_router: BaseSemanticRouter
    branch: ShadowMarkerSemanticResidualBranch

    def close(self) -> None:
        self.branch.close()
        self.span_router.close()
        self.embedding_writer.remove()


def factorized_candidate_state(
    *,
    layer_index: int,
    case_ids: Sequence[int],
    subjects: Sequence[str],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    embedding_row_ids: Sequence[int],
    embedding_delta: torch.Tensor,
    relation_ids: Sequence[str],
    semantic_router: BaseSemanticRouter,
    branch: ShadowMarkerSemanticResidualBranch,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if len(case_ids) != len(subjects) or len(case_ids) != semantic_router.records:
        raise ValueError("factorized candidate metadata is inconsistent")
    if len(relation_ids) != semantic_router.relation_count:
        raise ValueError("factorized candidate relation metadata is inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mcf_shadow_marker_base_semantic_router_candidate",
        "protocol": PROTOCOL,
        "architecture": {
            "main_embedding_path": "unaltered_base_embedding",
            "shadow_embedding_path": "frozen_v6_2_sparse_delta",
            "shadow_gradient_enabled": False,
            "marker_gate": "shadow_minus_base_rms",
            "semantic_gate": "frozen_shared_rank_relation_router",
            "semantic_shared_rank": int(semantic_router.rank),
            "semantic_basis_trainable": False,
            "semantic_record_specific_hidden_vectors": False,
            "semantic_relation_tied_coefficients": True,
            "joint_gate": "exact_subject_and_marker_and_semantic",
            "actuator": "constant_record_specific_layer_residual",
            "closed_gate_identity": "returns_original_layer_output_object",
            "base_embedding_mutated": False,
            "lm_head_mutated": False,
        },
        "layer_index": int(layer_index),
        "marker_rms_threshold": float(branch.marker_rms_threshold),
        "case_ids": [int(value) for value in case_ids],
        "subjects": [str(value) for value in subjects],
        "subject_patterns": [
            [[int(token) for token in pattern] for pattern in record]
            for record in subject_patterns
        ],
        "embedding_row_ids": [int(value) for value in embedding_row_ids],
        "embedding_delta": embedding_delta.detach().cpu(),
        "relation_ids": [str(value) for value in relation_ids],
        "semantic_shared_basis": semantic_router.shared_basis.detach().cpu(),
        "semantic_record_relation_index": (
            semantic_router.record_relation_index.detach().cpu()
        ),
        "semantic_relation_coefficients": (
            semantic_router.relation_coefficients.detach().cpu()
        ),
        "semantic_relation_bias": semantic_router.relation_bias.detach().cpu(),
        "actuator_residual": branch.residual.detach().cpu(),
        "residual_reference_norms": branch.residual_reference_norms.detach().cpu(),
        "source_hashes": {str(key): str(value) for key, value in source_hashes.items()},
        "official_evaluation_prompts_seen": 0,
    }


def install_factorized_candidate(
    base_model: nn.Module, state: Mapping[str, Any]
) -> InstalledFactorizedCandidate:
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION or state.get(
        "protocol"
    ) != PROTOCOL:
        raise RuntimeError("factorized shadow candidate schema/protocol mismatch")
    architecture = state.get("architecture")
    required = {
        "main_embedding_path": "unaltered_base_embedding",
        "shadow_embedding_path": "frozen_v6_2_sparse_delta",
        "marker_gate": "shadow_minus_base_rms",
        "semantic_gate": "frozen_shared_rank_relation_router",
        "joint_gate": "exact_subject_and_marker_and_semantic",
        "semantic_basis_trainable": False,
        "semantic_record_specific_hidden_vectors": False,
        "semantic_relation_tied_coefficients": True,
        "base_embedding_mutated": False,
        "lm_head_mutated": False,
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("candidate violates the factorized passive boundary")
    input_embedding = base_model.get_input_embeddings()
    layers = find_decoder_layers(base_model)
    layer_index = int(state["layer_index"])
    if input_embedding is None or not 0 <= layer_index < len(layers):
        raise RuntimeError("factorized candidate is incompatible with Base model")
    records = len(state.get("case_ids", []))
    hidden = int(input_embedding.weight.shape[1])
    subjects = [str(value) for value in state.get("subjects", [])]
    patterns = state.get("subject_patterns")
    row_ids = [int(value) for value in state.get("embedding_row_ids", [])]
    delta = state.get("embedding_delta")
    relation_ids = state.get("relation_ids")
    basis = state.get("semantic_shared_basis")
    record_relation_index = state.get("semantic_record_relation_index")
    coefficients = state.get("semantic_relation_coefficients")
    bias = state.get("semantic_relation_bias")
    residual = state.get("actuator_residual")
    reference = state.get("residual_reference_norms")
    if (
        records <= 0
        or len(subjects) != records
        or not isinstance(patterns, list)
        or len(patterns) != records
        or not row_ids
        or not isinstance(delta, torch.Tensor)
        or delta.shape != (len(row_ids), hidden)
        or not isinstance(relation_ids, list)
        or not isinstance(basis, torch.Tensor)
        or basis.ndim != 2
        or basis.shape[1] != hidden
        or int(architecture.get("semantic_shared_rank", -1)) != basis.shape[0]
        or not isinstance(record_relation_index, torch.Tensor)
        or record_relation_index.shape != (records,)
        or not isinstance(coefficients, torch.Tensor)
        or coefficients.shape != (int(record_relation_index.max()) + 1, basis.shape[0])
        or len(relation_ids) != coefficients.shape[0]
        or not isinstance(bias, torch.Tensor)
        or bias.shape != (coefficients.shape[0],)
        or not isinstance(residual, torch.Tensor)
        or residual.shape != (records, hidden)
        or not isinstance(reference, torch.Tensor)
        or reference.reshape(-1).shape != (records,)
    ):
        raise RuntimeError("factorized candidate shapes are invalid")
    from mcf_embedding_keyed_neuron_core import ToggleableEmbeddingDelta

    writer = ToggleableEmbeddingDelta(input_embedding, row_ids, delta)
    writer.enabled = False
    span = SpanGateRouter(input_embedding, patterns, subjects=subjects, model=base_model)
    semantic = BaseSemanticRouter(
        records,
        hidden,
        basis,
        record_relation_index,
    ).to(input_embedding.weight.device)
    with torch.no_grad():
        semantic.relation_coefficients.copy_(
            coefficients.to(semantic.relation_coefficients.device)
        )
        semantic.relation_bias.copy_(bias.to(semantic.relation_bias.device))
    semantic.relation_coefficients.requires_grad_(False)
    semantic.relation_bias.requires_grad_(False)
    branch = ShadowMarkerSemanticResidualBranch(
        layers[layer_index],
        span,
        semantic,
        hidden,
        marker_rms_threshold=float(state["marker_rms_threshold"]),
        residual_reference_norms=reference.to(input_embedding.weight.device),
    ).to(input_embedding.weight.device)
    with torch.no_grad():
        branch.residual.copy_(residual.to(branch.residual.device))
    branch.residual.requires_grad_(False)
    wrapped = v4.ShadowDualPathCausalLM(base_model, writer, branch)
    return InstalledFactorizedCandidate(wrapped, writer, span, semantic, branch)
