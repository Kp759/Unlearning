#!/usr/bin/env python3
"""V6 outer routing around the frozen V3.6.2 shadow-marker actuator.

This module deliberately exposes two distinct arms:

``outer_and_detector``
    two-sided route AND the frozen learned four-neuron detector.  This retains
    the original writer/detector/actuator causal question while removing
    unrelated prompts before they reach the learned branch.

``outer_direct``
    the two-sided route directly gates the frozen width-16 actuator features.
    This measures how much of V3.6.2's Gen failure was detector recall.  The
    shadow writer is not claimed necessary for this arm; its writer-off
    ablation is mandatory and reported.

Neither arm mutates a Base parameter.  They are mechanistic diagnostics, not
newly trained or official candidates.
"""
from __future__ import annotations

from typing import Any, Sequence, Tuple

import torch
import torch.nn.functional as F

import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_normalization_preserving_sidecar_v6_core as v6_core


class RoutedShadowEmbeddingDelta:
    """Apply the frozen writer delta only in rows authorized by V6 routing."""

    def __init__(
        self,
        input_layer: torch.nn.Module,
        router: v6_core.TwoSidedEntityRelationRouter,
        row_ids: Sequence[int],
        delta: torch.Tensor,
    ) -> None:
        if not hasattr(input_layer, "weight"):
            raise ValueError("input embedding module must expose weight")
        if delta.ndim != 2 or int(delta.shape[0]) != len(row_ids):
            raise ValueError("shadow writer row ids and delta do not match")
        self.router = router
        self.enabled = True
        device = input_layer.weight.device
        vocabulary = int(input_layer.weight.shape[0])
        self.lookup = torch.full(
            (vocabulary,), -1, dtype=torch.long, device=device
        )
        self.ids = torch.tensor(
            [int(item) for item in row_ids], dtype=torch.long, device=device
        )
        if self.ids.numel():
            self.lookup[self.ids] = torch.arange(self.ids.numel(), device=device)
        self.delta = delta.detach().float().to(device)
        self.fired_rows = 0
        self.changed_token_positions = 0
        # Router pre-hook is registered before this forward hook by contract.
        self.handle = input_layer.register_forward_hook(self._hook)

    def _hook(
        self,
        _module: torch.nn.Module,
        inputs: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        state = self.router.state
        if (
            not self.enabled
            or self.ids.numel() == 0
            or state is None
            or not bool(state.active.any())
        ):
            return output
        token_ids = inputs[0].to(self.lookup.device)
        local = self.lookup[token_ids]
        authorized_rows = state.active.any(dim=1).to(local.device)
        mask = local.ge(0) & authorized_rows.unsqueeze(1)
        if not bool(mask.any()):
            return output
        safe = local.clamp_min(0)
        correction = self.delta.index_select(0, safe.reshape(-1)).reshape(
            *safe.shape, self.delta.shape[-1]
        )
        correction = correction * mask.unsqueeze(-1)
        self.fired_rows += int(mask.any(dim=1).sum())
        self.changed_token_positions += int(mask.sum())
        return output + correction.to(device=output.device, dtype=output.dtype)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class OuterRoutedThresholdGatedActuatorBank(
    neuron_core.SparseThresholdGatedActuatorBank
):
    """Frozen V3.6.2 actuator with a V6 sequence-level outer gate."""

    VALID_MODES = ("outer_and_detector", "outer_direct")

    def __init__(
        self,
        mlp: torch.nn.Module,
        actuator_neuron_ids: Sequence[int],
        actuator_owner_indices: Sequence[int],
        *,
        outer_router: v6_core.TwoSidedEntityRelationRouter,
        outer_gate_mode: str,
        detector_gate_rows: torch.Tensor,
        detector_up_rows: torch.Tensor,
        detector_local_groups: Sequence[Sequence[int]],
        detector_flat_signs: torch.Tensor,
        off_boundary: float,
        on_boundary: float,
    ) -> None:
        if str(outer_gate_mode) not in self.VALID_MODES:
            raise ValueError(f"unsupported hybrid outer-gate mode: {outer_gate_mode}")
        self.outer_router = outer_router
        self.outer_gate_mode = str(outer_gate_mode)
        self.outer_closed_calls = 0
        self.outer_open_calls = 0
        super().__init__(
            mlp,
            actuator_neuron_ids,
            actuator_owner_indices,
            detector_gate_rows=detector_gate_rows,
            detector_up_rows=detector_up_rows,
            detector_local_groups=detector_local_groups,
            detector_flat_signs=detector_flat_signs,
            off_boundary=off_boundary,
            on_boundary=on_boundary,
        )

    def _outer_gates(
        self,
        hidden: torch.Tensor,
        learned_gates: torch.Tensor,
    ) -> torch.Tensor:
        state = self.outer_router.state
        if state is None:
            return torch.zeros_like(learned_gates)
        active = state.active.to(device=learned_gates.device, dtype=learned_gates.dtype)
        leading = learned_gates.shape[:-1]
        if not leading or int(leading[0]) != int(active.shape[0]):
            raise RuntimeError("hybrid outer route and MLP batch are misaligned")
        view_shape = (int(active.shape[0]),) + (1,) * (len(leading) - 1) + (
            int(active.shape[1]),
        )
        outer = active.reshape(view_shape).expand(*leading, int(active.shape[1]))
        if self.outer_gate_mode == "outer_direct":
            return outer
        return learned_gates * outer

    def actuator_features(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = hidden.float()
        activation = self._act_fn(F.linear(x, self.base_gate_rows)) * F.linear(
            x, self.base_up_rows
        )
        responses, learned_gates = self.detector_responses_and_gates(x)
        effective_gates = self._outer_gates(hidden, learned_gates)
        expanded = effective_gates.index_select(
            -1, self.actuator_owner_indices.to(effective_gates.device)
        )
        return activation * expanded, responses, effective_gates

    def _hook(
        self,
        module: torch.nn.Module,
        inputs: Any,
        output: torch.Tensor,
    ) -> torch.Tensor:
        state = self.outer_router.state
        if (
            not self.enabled
            or not self.write_enabled
            or state is None
            or not bool(state.active.any())
        ):
            self.outer_closed_calls += 1
            return output
        self.outer_open_calls += 1
        return super()._hook(module, inputs, output)
