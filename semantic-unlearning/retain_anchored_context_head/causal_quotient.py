from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contextual_head import FactIndexedLogitCorrection
from .model_adapter import FrozenRandomProjector


class CausalDescriptorProjector(nn.Module):
    """Fuse Method-4 final-state context with an intervention-derived channel.

    ``channel_basis`` is a frozen orthonormal basis in one intermediate hidden
    layer. No Transformer parameter is trained. Query descriptors concatenate a
    fixed random projection of the final hidden state with coordinates in the
    causal channel, then L2-normalize the result so Method-4's radius remains on
    the same unit-sphere distance scale.
    """

    def __init__(
        self,
        *,
        random_projector: FrozenRandomProjector,
        channel_basis: Tensor,
        causal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if channel_basis.ndim != 2 or channel_basis.numel() == 0:
            raise ValueError("channel_basis must have shape [hidden, rank]")
        if channel_basis.shape[0] != random_projector.matrix.shape[0]:
            raise ValueError("channel basis hidden dimension does not match model hidden size")
        if causal_weight <= 0 or not math.isfinite(float(causal_weight)):
            raise ValueError("causal_weight must be finite and positive")
        self.random_projector = random_projector
        self.register_buffer(
            "channel_basis", channel_basis.detach().float().clone(), persistent=True
        )
        self.causal_weight = float(causal_weight)

    @property
    def output_dim(self) -> int:
        return int(self.random_projector.output_dim + self.channel_basis.shape[1])

    @property
    def channel_rank(self) -> int:
        return int(self.channel_basis.shape[1])

    def forward(self, final_hidden: Tensor, causal_hidden: Tensor) -> Tensor:
        if final_hidden.shape != causal_hidden.shape:
            raise ValueError("final_hidden and causal_hidden must have identical shapes")
        if final_hidden.shape[-1] != self.channel_basis.shape[0]:
            raise ValueError("hidden size does not match causal channel basis")

        context = self.random_projector(final_hidden).float()
        channel = causal_hidden.float() @ self.channel_basis
        channel = F.normalize(channel, dim=-1, eps=1e-8)
        fused = torch.cat([context, self.causal_weight * channel], dim=-1)
        return F.normalize(fused, dim=-1, eps=1e-8)


@dataclass(frozen=True)
class QuotientDiagnostics:
    num_facts: int
    hidden_size: int
    mean_effect_norm: float
    min_effect_norm: float
    max_effect_norm: float
    strength: float


class FactIndexedCausalQuotient(nn.Module):
    """Context-gated output-side quotient of intervention-validated effects.

    For fact i we store a unit downstream effect direction ``v_i`` and the
    final hidden state ``n_i`` produced by removing the discovered causal
    channel on the direct training anchor. Given contextual coordinates alpha,

        h' = h - strength * sum_i alpha_i ((h - n_i)^T v_i) v_i.

    At a cardinal forget anchor this removes the downstream component that was
    actually caused by the discovered channel. At protected anchors alpha is
    numerically zero, so the quotient is inactive.
    """

    def __init__(
        self,
        *,
        effect_directions: Tensor,
        neutral_final_hidden: Tensor,
        strength: float = 1.0,
    ) -> None:
        super().__init__()
        if effect_directions.ndim != 2 or neutral_final_hidden.ndim != 2:
            raise ValueError("effect and neutral tensors must have shape [facts, hidden]")
        if effect_directions.shape != neutral_final_hidden.shape:
            raise ValueError("effect_directions and neutral_final_hidden must match")
        if effect_directions.numel() == 0:
            raise ValueError("at least one causal effect direction is required")
        if strength < 0 or not math.isfinite(float(strength)):
            raise ValueError("strength must be finite and non-negative")

        raw_norm = effect_directions.float().norm(dim=-1)
        if torch.any(raw_norm <= 1e-8):
            raise ValueError("every causal effect direction must be non-zero")
        unit = effect_directions.float() / raw_norm.unsqueeze(-1)
        self.register_buffer("effect_directions", unit, persistent=True)
        self.register_buffer(
            "neutral_final_hidden", neutral_final_hidden.detach().float().clone(), persistent=True
        )
        self.register_buffer("raw_effect_norm", raw_norm.detach().clone(), persistent=True)
        self.strength = float(strength)

    @property
    def num_facts(self) -> int:
        return int(self.effect_directions.shape[0])

    def apply(self, hidden: Tensor, alpha: Tensor, fact_enabled: Tensor | None = None) -> Tensor:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [positions, hidden]")
        if alpha.ndim != 2 or alpha.shape[0] != hidden.shape[0]:
            raise ValueError("alpha must have shape [positions, facts]")
        if alpha.shape[1] != self.num_facts:
            raise ValueError("alpha fact dimension does not match quotient")
        if hidden.shape[1] != self.effect_directions.shape[1]:
            raise ValueError("hidden size does not match quotient effect directions")

        alpha_f = alpha.float()
        if fact_enabled is not None:
            if fact_enabled.shape != (self.num_facts,):
                raise ValueError("fact_enabled must have shape [facts]")
            alpha_f = alpha_f * fact_enabled.float().unsqueeze(0)

        h = hidden.float()
        v = self.effect_directions
        current_projection = h @ v.transpose(0, 1)
        neutral_projection = (self.neutral_final_hidden * v).sum(dim=-1)
        coefficient = alpha_f * (current_projection - neutral_projection.unsqueeze(0))
        delta = coefficient @ v
        result = h - self.strength * delta
        return result.to(dtype=hidden.dtype)

    @torch.no_grad()
    def diagnostics(self) -> QuotientDiagnostics:
        return QuotientDiagnostics(
            num_facts=self.num_facts,
            hidden_size=int(self.effect_directions.shape[1]),
            mean_effect_norm=float(self.raw_effect_norm.mean().item()),
            min_effect_norm=float(self.raw_effect_norm.min().item()),
            max_effect_norm=float(self.raw_effect_norm.max().item()),
            strength=self.strength,
        )


class ContextualCausalQuotientModel(nn.Module):
    """Method 5: retain-anchored contextual head + causal output quotient.

    ``quotient_fact_mask`` is fixed from direct training-visible intervention
    validation. A fact whose channel-removal intervention did not reduce the
    sensitive-answer log probability keeps the contextual logit correction but
    does not receive the quotient. This prevents calling an anti-causal effect a
    causal suppression direction.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        descriptor_projector: CausalDescriptorProjector,
        correction: FactIndexedLogitCorrection,
        quotient: FactIndexedCausalQuotient,
        causal_hidden_index: int,
        quotient_fact_mask: Tensor | None = None,
        alpha_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if alpha_chunk_size <= 0:
            raise ValueError("alpha_chunk_size must be positive")
        if quotient.num_facts != correction.num_facts:
            raise ValueError("quotient and correction must use the same fact count")
        self.base_model = base_model
        self.descriptor_projector = descriptor_projector
        self.correction = correction
        self.quotient = quotient
        self.causal_hidden_index = int(causal_hidden_index)
        self.alpha_chunk_size = int(alpha_chunk_size)

        if quotient_fact_mask is None:
            quotient_fact_mask = torch.ones(
                correction.num_facts,
                device=correction.feature_map.forget.device,
                dtype=correction.feature_map.forget.dtype,
            )
        if quotient_fact_mask.shape != (correction.num_facts,):
            raise ValueError("quotient_fact_mask must have shape [facts]")
        self.register_buffer(
            "quotient_fact_mask",
            quotient_fact_mask.detach().float().clone(),
            persistent=True,
        )

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    @property
    def config(self):
        return self.base_model.config

    def _alpha(self, descriptors: Tensor) -> Tensor:
        pieces = []
        for start in range(0, descriptors.shape[0], self.alpha_chunk_size):
            stop = min(descriptors.shape[0], start + self.alpha_chunk_size)
            pieces.append(self.correction.feature_map.alpha(descriptors[start:stop]))
        return torch.cat(pieces, dim=0)

    def forward(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.base_model(*args, **kwargs)
        if outputs.hidden_states is None:
            raise RuntimeError("base model did not return hidden states")
        if not 0 <= self.causal_hidden_index < len(outputs.hidden_states):
            raise RuntimeError("causal hidden index is outside returned hidden states")

        final_hidden = outputs.hidden_states[-1]
        causal_hidden = outputs.hidden_states[self.causal_hidden_index]
        batch, seq, dim = final_hidden.shape
        flat_final = final_hidden.reshape(batch * seq, dim)
        flat_causal = causal_hidden.reshape(batch * seq, dim)
        descriptors = self.descriptor_projector(flat_final, flat_causal)
        alpha = self._alpha(descriptors)
        enabled_alpha = alpha * self.correction.fact_enabled.float().unsqueeze(0)
        quotient_alpha = enabled_alpha * self.quotient_fact_mask.unsqueeze(0)

        quotiented = self.quotient.apply(
            flat_final,
            quotient_alpha,
        ).reshape(batch, seq, dim)

        output_head = self.base_model.get_output_embeddings()
        if output_head is None:
            raise RuntimeError("base model does not expose get_output_embeddings()")
        logits = output_head(quotiented)

        selected_delta = enabled_alpha @ self.correction.coefficients.transpose(0, 1)
        selected_delta = selected_delta.reshape(batch, seq, -1).to(dtype=logits.dtype)
        logits = logits.clone()
        logits[..., self.correction.selected_token_ids] += selected_delta
        outputs.logits = logits
        return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self
