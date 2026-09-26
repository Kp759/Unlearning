"""Helpers shared by the MCF layer-sweep scripts (linear-classifier pipeline)."""
from __future__ import annotations

import torch

from static_overlap_fact_association_embeddings import block_output_capture


@torch.no_grad()
def boundary_norms(model, tokenizer, prompts, layers, batch_size=16):
    """L2 norm of each block's raw output at the final prompt token."""
    device = next(model.parameters()).device
    result = {int(layer): [] for layer in layers}
    for start in range(0, len(prompts), int(batch_size)):
        encoded = tokenizer(
            prompts[start:start + int(batch_size)],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        captures = [block_output_capture(model, layer) for layer in result]
        try:
            model(**encoded, use_cache=False)
        finally:
            for _, handle in captures:
                handle.remove()
        mask = encoded["attention_mask"].bool()
        positions = (
            torch.arange(mask.shape[1], device=device)[None, :]
            .expand_as(mask)
            .masked_fill(~mask, -1)
            .max(dim=1)
            .values
        )
        index = torch.arange(mask.shape[0], device=device)
        for layer, (captured, _) in zip(result, captures):
            hidden = captured["hidden"].float()[index, positions]
            result[layer].extend(hidden.norm(dim=-1).cpu().tolist())
    return {layer: torch.tensor(values) for layer, values in result.items()}


def resolve_norm_scale(value, norms, layer, reference_layer):
    if str(value).strip().lower() == "auto":
        return float(norms[layer].median() / norms[reference_layer].median())
    scale = float(value)
    if not scale > 0:
        raise ValueError("--norm-scale must be positive or 'auto'")
    return scale


def first_label_position(example):
    return next(i for i, label in enumerate(example.labels) if label != -100)


def oracle_route_map(tokenizer, example_maps, fact_to_row):
    """{prompt-prefix tokens: row} for every training-visible prompt.

    Two spellings per example: the teacher-forced prefix the trainer binds
    (tokens before the first labelled position) and the bare tokenized prompt
    the route audit uses. A prefix owned by two facts is an error.
    """
    mapping = {}
    for examples in example_maps:
        for example in examples.values():
            row = fact_to_row[example.fact_id]
            keys = (
                tuple(example.input_ids[:first_label_position(example)]),
                tuple(tokenizer(example.prompt)["input_ids"]),
            )
            for key in keys:
                if mapping.setdefault(key, row) != row:
                    raise ValueError(
                        f"Prompt prefix is shared by two facts: {example.prompt!r}"
                    )
    return mapping
