from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contextual_head import FactIndexedLogitCorrection


class FrozenRandomProjector(nn.Module):
    """Fixed random projection used as the channel-free descriptor baseline."""

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
    """Apply sparse fact-indexed logit corrections to a frozen causal LM.

    This prototype is intentionally scoped to teacher-forced MCF evaluation and
    perplexity.  It does not claim to implement Hugging Face generation APIs.
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

    def forward(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.base_model(*args, **kwargs)
        if outputs.hidden_states is None:
            raise RuntimeError("base model did not return hidden states")

        hidden = outputs.hidden_states[-1]
        batch, seq, dim = hidden.shape
        flat_hidden = hidden.reshape(batch * seq, dim)
        flat_logits = outputs.logits.reshape(batch * seq, -1)
        token_ids = self.correction.selected_token_ids

        # Modify only selected columns.  This avoids allocating an additional
        # [batch*seq, vocab] dense correction tensor for a 128k-token vocabulary.
        for start in range(0, flat_hidden.shape[0], self.alpha_chunk_size):
            stop = min(flat_hidden.shape[0], start + self.alpha_chunk_size)
            descriptors = self.projector(flat_hidden[start:stop])
            selected_delta = self.correction.selected_correction_for_descriptors(descriptors)
            selected_delta = selected_delta.to(dtype=flat_logits.dtype)
            flat_logits[start:stop, token_ids] += selected_delta

        outputs.logits = flat_logits.reshape(batch, seq, -1)
        return outputs

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self
