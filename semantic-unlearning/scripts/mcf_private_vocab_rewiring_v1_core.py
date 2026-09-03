#!/usr/bin/env python3
"""Core utilities for materialized private-vocabulary factual unlearning.

The method repurposes pre-existing reserved tokenizer slots as private lexical
clones of forget subjects. During training only a compact bank of clone vectors
is trainable. At materialization those vectors are copied into the selected
reserved input-embedding rows; every original lexical input row, the Transformer,
and the LM head remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


PROTOCOL = "mcf_private_vocab_rewiring_v1"
RESERVED_RE = re.compile(r"^<\|reserved_special_token_(\d+)\|>$")


def sha256_tensor(tensor: torch.Tensor, *, chunk_rows: int = 512) -> str:
    """Hash a tensor without materializing the full parameter on CPU at once."""
    value = tensor.detach()
    digest = hashlib.sha256()
    if value.ndim == 0:
        digest.update(value.cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    for start in range(0, int(value.shape[0]), int(chunk_rows)):
        chunk = value[start : start + int(chunk_rows)].cpu().contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def unique_subjects(records: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for record in records:
        subject = str(record["requested_rewrite"]["subject"])
        if subject not in seen:
            seen.add(subject)
            ordered.append(subject)
    return ordered


def discover_reserved_slots(tokenizer: Any, *, needed: int) -> list[tuple[str, int]]:
    """Return stable unused/reserved tokenizer slots suitable for cloning.

    We deliberately require explicitly named reserved slots rather than merely
    low-frequency lexical tokens. This avoids silently stealing utility from a
    real word. Slots used as BOS/EOS/PAD/UNK are excluded even if their textual
    names happen to match the reserved-token pattern.
    """
    protected = {
        int(value)
        for value in (
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        )
        if value is not None
    }
    candidates: list[tuple[str, int]] = []
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        if RESERVED_RE.match(str(token)) and int(token_id) not in protected:
            candidates.append((str(token), int(token_id)))
    candidates.sort(key=lambda item: item[1])
    if len(candidates) < int(needed):
        raise RuntimeError(
            f"need {needed} reserved tokenizer slots but found only {len(candidates)}"
        )
    return candidates[: int(needed)]


def build_subject_slot_mapping(
    tokenizer: Any, subjects: Sequence[str]
) -> list[Dict[str, Any]]:
    slots = discover_reserved_slots(tokenizer, needed=len(subjects))
    mapping: list[Dict[str, Any]] = []
    for subject, (reserved_token, token_id) in zip(subjects, slots):
        original_ids = tokenizer(
            str(subject), add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if not original_ids:
            raise RuntimeError(f"subject tokenizes to empty sequence: {subject!r}")
        mapping.append(
            {
                "subject": str(subject),
                "private_token_id": int(token_id),
                "original_reserved_token": str(reserved_token),
                "base_subject_token_ids": [int(value) for value in original_ids],
            }
        )
    return mapping


def _remove_selected_specials(value: Any, selected_tokens: set[str]) -> Any:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item in selected_tokens:
                continue
            if isinstance(item, dict) and str(item.get("content", "")) in selected_tokens:
                continue
            out.append(_remove_selected_specials(item, selected_tokens))
        return out
    if isinstance(value, dict):
        return {
            key: _remove_selected_specials(item, selected_tokens)
            for key, item in value.items()
        }
    return value


def patch_saved_tokenizer_reserved_slots(
    tokenizer_dir: Path, mapping: Sequence[Mapping[str, Any]]
) -> None:
    """Rename reserved AddedToken slots to literal subject strings in-place.

    This keeps token ids and vocabulary size fixed. The operation is intentionally
    strict and then expected to be validated by reloading the tokenizer.
    """
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise RuntimeError("private vocabulary rewiring requires a fast tokenizer.json")
    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    added = payload.get("added_tokens")
    if not isinstance(added, list):
        raise RuntimeError("tokenizer.json lacks an added_tokens list")

    by_id = {int(item["id"]): item for item in added if isinstance(item, dict) and "id" in item}
    selected_old: set[str] = set()
    for item in mapping:
        token_id = int(item["private_token_id"])
        old = str(item["original_reserved_token"])
        subject = str(item["subject"])
        if token_id not in by_id:
            raise RuntimeError(f"reserved id {token_id} is not an AddedToken entry")
        entry = by_id[token_id]
        if str(entry.get("content")) != old:
            raise RuntimeError(
                f"reserved id {token_id} changed unexpectedly: {entry.get('content')!r} != {old!r}"
            )
        selected_old.add(old)
        entry["content"] = subject
        entry["special"] = False
        entry["normalized"] = False
        entry["single_word"] = False
        entry["lstrip"] = False
        entry["rstrip"] = False
    tokenizer_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config_path = tokenizer_dir / "tokenizer_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        decoder = config.get("added_tokens_decoder")
        if isinstance(decoder, dict):
            for item in mapping:
                key = str(int(item["private_token_id"]))
                if key in decoder and isinstance(decoder[key], dict):
                    decoder[key]["content"] = str(item["subject"])
                    decoder[key]["special"] = False
                    decoder[key]["normalized"] = False
                    decoder[key]["single_word"] = False
                    decoder[key]["lstrip"] = False
                    decoder[key]["rstrip"] = False
        config = _remove_selected_specials(config, selected_old)
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    special_map = tokenizer_dir / "special_tokens_map.json"
    if special_map.is_file():
        special = json.loads(special_map.read_text(encoding="utf-8"))
        special = _remove_selected_specials(special, selected_old)
        special_map.write_text(
            json.dumps(special, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def validate_subject_routing(tokenizer: Any, mapping: Sequence[Mapping[str, Any]]) -> None:
    for item in mapping:
        subject = str(item["subject"])
        expected = int(item["private_token_id"])
        ids = tokenizer(
            subject, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if ids != [expected]:
            raise RuntimeError(
                f"subject routing failed for {subject!r}: expected {[expected]}, got {ids}"
            )
        probe = f"Tell me about {subject}."
        probe_ids = tokenizer(
            probe, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if expected not in probe_ids:
            raise RuntimeError(
                f"subject routing does not activate in sentence context for {subject!r}"
            )


def initialize_clone_rows(
    embedding_weight: torch.Tensor,
    mapping: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    rows = []
    weight = embedding_weight.detach()
    for item in mapping:
        ids = torch.tensor(
            [int(value) for value in item["base_subject_token_ids"]],
            device=weight.device,
            dtype=torch.long,
        )
        rows.append(weight.index_select(0, ids).float().mean(dim=0))
    return torch.stack(rows, dim=0)


class PrivateRowController(nn.Module):
    """Differentiably substitutes a compact trainable row bank at selected ids."""

    def __init__(self, private_token_ids: Sequence[int], initial_rows: torch.Tensor):
        super().__init__()
        ids = torch.tensor([int(v) for v in private_token_ids], dtype=torch.long)
        if initial_rows.ndim != 2 or initial_rows.shape[0] != ids.numel():
            raise ValueError("private row ids and initial rows are incompatible")
        self.register_buffer("private_token_ids", ids, persistent=True)
        self.rows = nn.Parameter(initial_rows.detach().clone().float())
        self.register_buffer("initial_rows", initial_rows.detach().clone().float(), persistent=True)
        self.enabled = True

    def apply(self, input_ids: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return output
        result = output
        ids = self.private_token_ids.to(input_ids.device)
        rows = self.rows.to(device=output.device, dtype=output.dtype)
        for index in range(ids.numel()):
            mask = input_ids.eq(ids[index]).unsqueeze(-1)
            if mask.any():
                result = torch.where(mask, rows[index].view(1, 1, -1), result)
        return result

    @torch.no_grad()
    def enforce_relative_cap(self, relative_cap: float) -> Dict[str, float]:
        if relative_cap <= 0:
            raise ValueError("relative cap must be positive")
        delta = self.rows - self.initial_rows
        delta_norm = delta.norm(dim=1)
        base_norm = self.initial_rows.norm(dim=1).clamp_min(1e-12)
        caps = float(relative_cap) * base_norm
        scale = torch.minimum(torch.ones_like(delta_norm), caps / delta_norm.clamp_min(1e-12))
        self.rows.copy_(self.initial_rows + delta * scale[:, None])
        final = (self.rows - self.initial_rows).norm(dim=1)
        return {
            "max_relative_delta": float((final / base_norm).max().item()),
            "mean_relative_delta": float((final / base_norm).mean().item()),
        }


@dataclass
class EmbeddingHook:
    controller: PrivateRowController
    handle: Any

    @classmethod
    def install(cls, embedding: nn.Module, controller: PrivateRowController) -> "EmbeddingHook":
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
            if not inputs:
                return output
            input_ids = inputs[0]
            if not torch.is_tensor(input_ids):
                return output
            return controller.apply(input_ids, output)

        return cls(controller=controller, handle=embedding.register_forward_hook(hook))

    def remove(self) -> None:
        self.handle.remove()


def materialize_private_rows(
    embedding_weight: torch.Tensor,
    controller: PrivateRowController,
) -> None:
    ids = controller.private_token_ids.to(embedding_weight.device)
    rows = controller.rows.detach().to(embedding_weight.device, dtype=embedding_weight.dtype)
    with torch.no_grad():
        embedding_weight.index_copy_(0, ids, rows)


def non_private_row_hash(
    weight: torch.Tensor, private_ids: Sequence[int], *, chunk_rows: int = 512
) -> str:
    """Hash all embedding rows except private slots without a full CPU copy."""
    excluded = {int(value) for value in private_ids}
    digest = hashlib.sha256()
    value = weight.detach()
    for start in range(0, int(value.shape[0]), int(chunk_rows)):
        stop = min(start + int(chunk_rows), int(value.shape[0]))
        keep = [i for i in range(start, stop) if i not in excluded]
        if not keep:
            continue
        ids = torch.tensor(keep, device=value.device, dtype=torch.long)
        chunk = value.index_select(0, ids).cpu().contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def generic_subject_contexts(subject: str) -> tuple[str, ...]:
    return (
        str(subject),
        f"Tell me about {subject}.",
        f"Information about {subject}:",
        f"The following statement concerns {subject}.",
    )
