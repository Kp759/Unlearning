from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contextual_head import FactIndexedLogitCorrection


class FrozenRandomProjector(nn.Module):
    """Fixed random projection used as the channel-free descriptor baseline.

    The matrix is sampled once from a deterministic seed and stored as a buffer.
    No model or projection parameters are trained.  Inputs and projected outputs
    are L2-normalized so Euclidean distance corresponds monotonically to cosine
    distance on the unit sphere.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("projection dimensions must be positive")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        matrix = torch.randn(input_dim, output_dim, generator=generator, dtype=torch.float32)
        matrix = matrix / math.sqrt(float(output_dim))
        self.register_buffer("matrix", matrix.to(device=device, dtype=dtype), persistent=True)

    @property
    def output_dim(self) -> int:
        return int(self.matrix.shape[1])

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.shape[-1] != self.matrix.shape[0]:
            raise ValueError(
                f"hidden dim {hidden.shape[-1]} != projection input {self.matrix.shape[0]}"
            )
        x = F.normalize(hidden.float(), dim=-1)
        projected = x @ self.matrix.float()
        return F.normalize(projected, dim=-1)


class ContextualCorrectionModel(nn.Module):
    """Apply fact-indexed contextual logit corrections to a frozen causal LM.

    The base model remains unchanged.  The wrapper requests final hidden states,
    maps them through a fixed descriptor projector, evaluates the retain-anchored
    feature map, and adds only the selected-token logit corrections.

    This wrapper is intentionally sufficient for teacher-forced official MCF
    evaluation and perplexity.  It does not claim to implement generation APIs.
    """

    def __init__(
        self,
        *,
        base_model: nn.Module,
        projector: FrozenRandomProjector,
        correction: FactIndexedLogitCorrection,
        alpha_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if alpha_chunk_size <= 0:
            raise ValueError("alpha_chunk_size must be positive")
        self.base_model = base_model
        self.projector = projector
        self.correction = correction
        self.alpha_chunk_size = int(alpha_chunk_size)

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    @property
    def config(self):
        return self.base_model.config

    def _dense_delta(self, descriptors: Tensor) -> Tensor:
        chunks = []
        for start in range(0, descriptors.shape[0], self.alpha_chunk_size):
            stop = min(descriptors.shape[0], start + self.alpha_chunk_size)
            chunks.append(self.correction.correction_for_descriptors(descriptors[start:stop]))
        return torch.cat(chunks, dim=0)

    def forward(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.base_model(*args, **kwargs)
        if outputs.hidden_states is None:
            raise RuntimeError("base model did not return hidden states")

        hidden = outputs.hidden_states[-1]
        batch, seq, dim = hidden.shape
        descriptors = self.projector(hidden.reshape(batch * seq, dim))
        delta = self._dense_delta(descriptors).reshape(batch, seq, -1)
        corrected_logits = outputs.logits + delta.to(dtype=outputs.logits.dtype)
        outputs.logits = corrected_logits
        return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen language model must never enter training mode because this
        # prototype performs no stochastic base-model training.
        self.base_model.eval()
        return self
