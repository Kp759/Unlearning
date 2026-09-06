#!/usr/bin/env python3
"""Fix5f wrapper: robust marker-visibility audit for the Fix5e representation ablation.

Fix5e correctly compares TARGET_ENTITY against [TARGET]subject[/TARGET], but its
marker diagnostic searched for the standalone token-id sequence of each marker inside
contextual tokenization. Token boundaries can change with whitespace/punctuation, so
that can falsely report a visible marker as missing.

This wrapper changes only that diagnostic. It uses the fast tokenizer's character
offsets after right truncation and asks whether each literal marker occurs completely
inside the retained character prefix. Frozen features, matched rows, linear heads,
loss, calibration, policy accounting, coverage challenge, and all recognition-only
constraints remain unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_representation_compare_fix5e_seed1 as core


def marker_visibility_in_prefix(
    text: str,
    retained_char_end: int,
    required_markers: Sequence[str],
) -> dict[str, bool]:
    """Whether each literal marker is fully contained in the retained text prefix."""
    prefix = str(text)[: max(0, int(retained_char_end))]
    return {str(marker): str(marker) in prefix for marker in required_markers}


@torch.no_grad()
def encode_texts(
    model: Any,
    tok: Any,
    texts: Sequence[str],
    device: torch.device,
    batch: int,
    required_markers: Sequence[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Encode exactly as Fix5e, but audit markers with tokenizer character offsets."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("requires model.model")
    if not getattr(tok, "is_fast", False):
        raise RuntimeError("Fix5f marker audit requires a fast tokenizer with offset mappings")

    full_lengths = [
        len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]
    chunks: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for st in range(0, len(texts), int(batch)):
            bt = list(texts[st : st + int(batch)])
            enc = tok(
                bt,
                padding=True,
                truncation=True,
                max_length=core.base.MAX_LENGTH,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping").cpu()
            attention_cpu = enc["attention_mask"].cpu()
            enc = enc.to(device)

            h = backbone(**enc, use_cache=False, return_dict=True).last_hidden_state.float()
            m = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            chunks.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu())

            for j in range(len(bt)):
                active = attention_cpu[j].bool()
                active_offsets = offsets[j][active]
                retained_char_end = 0
                if active_offsets.numel():
                    retained_char_end = int(active_offsets[:, 1].max().item())
                pos = st + j
                vis = marker_visibility_in_prefix(
                    bt[j], retained_char_end, required_markers
                )
                diagnostics.append(
                    {
                        "text_index": pos,
                        "full_token_count": int(full_lengths[pos]),
                        "kept_token_count": int(active.sum().item()),
                        "truncated": bool(full_lengths[pos] > core.base.MAX_LENGTH),
                        "retained_char_end": retained_char_end,
                        "required_marker_visibility": vis,
                        "all_required_markers_visible_after_truncation": all(vis.values()),
                        "marker_visibility_method": "literal marker within retained character prefix from fast-tokenizer offsets",
                    }
                )
    finally:
        tok.padding_side = old_side

    return torch.cat(chunks, dim=0), diagnostics


# Patch only the visibility/encoding diagnostic implementation. The underlying hidden
# states and attention-mask mean pooling are mathematically the same as Fix5e.
core.encode_texts = encode_texts


if __name__ == "__main__":
    core.main()
