from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

from .anchored_features import AnchoredFeatureMap


@dataclass(frozen=True)
class CorrectionDiagnostics:
    max_abs_retain_alpha: float
    max_abs_cardinal_error: float
    num_facts: int
    num_selected_tokens: int


class FactIndexedLogitCorrection(nn.Module):
    """Context-indexed output correction on top of a frozen base LM head.

    The module owns only a small coefficient matrix ``C`` over selected output
    token ids and fact-indexed contextual coordinates. It never overwrites the
    base model's ``lm_head.weight``.

    For base logits ``z0`` and contextual features ``alpha(x)``:

        z = z0 + S C alpha(x)

    where ``S`` scatters rows into the selected vocabulary token ids.
    """

    def __init__(
        self,
        *,
        feature_map: AnchoredFeatureMap,
        selected_token_ids: Sequence[int],
        vocab_size: int,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        token_ids = [int(t) for t in selected_token_ids]
        if not token_ids:
            raise ValueError("selected_token_ids must be non-empty")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("selected_token_ids must be unique")
        if min(token_ids) < 0 or max(token_ids) >= vocab_size:
            raise ValueError("selected token id outside vocabulary")

        self.feature_map = feature_map
        self.vocab_size = int(vocab_size)
        self.register_buffer(
            "selected_token_ids",
            torch.tensor(token_ids, dtype=torch.long, device=feature_map.forget.device),
            persistent=True,
        )
        coeff = torch.zeros(
            len(token_ids),
            feature_map.num_facts,
            device=feature_map.forget.device,
            dtype=feature_map.forget.dtype,
        )
        if init_scale != 0.0:
            coeff.normal_(mean=0.0, std=float(init_scale))
        self.coefficients = nn.Parameter(coeff)
        self.register_buffer(
            "fact_enabled",
            torch.ones(
                feature_map.num_facts,
                device=feature_map.forget.device,
                dtype=feature_map.forget.dtype,
            ),
            persistent=True,
        )

    @property
    def num_facts(self) -> int:
        return self.feature_map.num_facts

    def set_fact_enabled(self, fact_index: int, enabled: bool) -> None:
        if not 0 <= fact_index < self.num_facts:
            raise IndexError(f"fact_index {fact_index} outside [0, {self.num_facts})")
        with torch.no_grad():
            self.fact_enabled[fact_index] = 1.0 if enabled else 0.0

    def rollback_facts(self, fact_indices: Iterable[int]) -> None:
        for fact_index in fact_indices:
            self.set_fact_enabled(int(fact_index), False)

    def enable_all_facts(self) -> None:
        with torch.no_grad():
            self.fact_enabled.fill_(1.0)

    def selected_correction_for_descriptors(self, descriptors: Tensor) -> Tensor:
        """Return only the selected-token corrections, shape [batch, selected]."""
        alpha = self.feature_map.alpha(descriptors)
        alpha = alpha * self.fact_enabled.unsqueeze(0)
        return alpha @ self.coefficients.transpose(0, 1)

    def correction_for_descriptors(self, descriptors: Tensor) -> Tensor:
        """Return dense vocabulary-logit corrections for small diagnostic batches."""
        selected_delta = self.selected_correction_for_descriptors(descriptors)
        delta = torch.zeros(
            descriptors.shape[0],
            self.vocab_size,
            device=selected_delta.device,
            dtype=selected_delta.dtype,
        )
        delta.index_copy_(1, self.selected_token_ids, selected_delta)
        return delta

    def forward(self, base_logits: Tensor, descriptors: Tensor) -> Tensor:
        if base_logits.ndim != 2:
            raise ValueError(
                f"base_logits must have shape [batch, vocab], got {tuple(base_logits.shape)}"
            )
        if base_logits.shape[1] != self.vocab_size:
            raise ValueError(
                f"base_logits vocab {base_logits.shape[1]} != configured {self.vocab_size}"
            )
        if base_logits.shape[0] != descriptors.shape[0]:
            raise ValueError("base_logits and descriptors batch sizes must match")
        delta = self.correction_for_descriptors(descriptors)
        if delta.dtype != base_logits.dtype:
            delta = delta.to(dtype=base_logits.dtype)
        return base_logits + delta

    @torch.no_grad()
    def diagnostics(self) -> CorrectionDiagnostics:
        retain_residual = self.feature_map.retain_residual()
        cardinal_residual = self.feature_map.cardinal_residual()
        return CorrectionDiagnostics(
            max_abs_retain_alpha=float(retain_residual.abs().max().item()),
            max_abs_cardinal_error=float(cardinal_residual.abs().max().item()),
            num_facts=self.num_facts,
            num_selected_tokens=int(self.selected_token_ids.numel()),
        )


def sequence_margin_loss(
    corrected_logits: Tensor,
    *,
    sensitive_token_ids: Tensor,
    safe_token_ids: Tensor,
    margin: float,
    reduction: str = "mean",
) -> Tensor:
    """Hinge margin favoring a safe next token over a sensitive next token.

    Each row is one prediction event, so multi-token answers are handled by
    supplying one row per causal prefix/token position rather than globally
    banning any subword.
    """

    if corrected_logits.ndim != 2:
        raise ValueError("corrected_logits must have shape [events, vocab]")
    n = corrected_logits.shape[0]
    if sensitive_token_ids.shape != (n,) or safe_token_ids.shape != (n,):
        raise ValueError("token-id tensors must have shape [events]")

    row = torch.arange(n, device=corrected_logits.device)
    sensitive = corrected_logits[row, sensitive_token_ids]
    safe = corrected_logits[row, safe_token_ids]
    loss = torch.relu(float(margin) + sensitive - safe)

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"unsupported reduction: {reduction}")
