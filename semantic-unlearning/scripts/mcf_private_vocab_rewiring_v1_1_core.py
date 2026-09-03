#!/usr/bin/env python3
"""Position-preserving private-vocabulary primitives for MCF factual unlearning.

V1.1 allocates one private reserved token for every token in every unique forget
subject.  A subject token sequence ``[t1, ..., tk]`` is deterministically
rewritten to a same-length private sequence ``[p1, ..., pk]``.  Private rows are
initialized by exact one-to-one copies ``E[pj] = E[tj]``.  Therefore token count,
following positions, RoPE indices, and the initial embedding sequence are
preserved exactly.

The lexical rewrite is subject-only and relation-agnostic.  It is not a
forget-query classifier: every occurrence of a registered subject sequence is
rewritten the same way in every context.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import mcf_private_vocab_rewiring_v1_core as v1


PROTOCOL = "mcf_private_vocab_rewiring_v1_1_position_preserving"


def build_position_preserving_mapping(
    tokenizer: Any, subjects: Sequence[str]
) -> list[Dict[str, Any]]:
    """Allocate one reserved private id for every original subject token id."""
    tokenized: list[tuple[str, list[int]]] = []
    needed = 0
    for subject in subjects:
        ids = tokenizer(
            str(subject), add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        ids = [int(value) for value in ids]
        if not ids:
            raise RuntimeError(f"subject tokenizes to empty sequence: {subject!r}")
        tokenized.append((str(subject), ids))
        needed += len(ids)

    slots = v1.discover_reserved_slots(tokenizer, needed=needed)
    cursor = 0
    mapping: list[Dict[str, Any]] = []
    for subject, base_ids in tokenized:
        local = slots[cursor : cursor + len(base_ids)]
        cursor += len(base_ids)
        private_tokens = [str(token) for token, _ in local]
        private_ids = [int(token_id) for _, token_id in local]
        if len(private_ids) != len(base_ids):
            raise AssertionError("position-preserving allocation changed subject length")
        mapping.append(
            {
                "subject": subject,
                "base_subject_token_ids": base_ids,
                "private_token_ids": private_ids,
                "private_reserved_tokens": private_tokens,
                "token_count": len(base_ids),
            }
        )
    if cursor != needed:
        raise AssertionError("reserved-slot allocation count mismatch")
    return mapping


def flatten_private_ids(mapping: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(token_id)
        for item in mapping
        for token_id in item["private_token_ids"]
    ]


def flatten_base_ids(mapping: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(token_id)
        for item in mapping
        for token_id in item["base_subject_token_ids"]
    ]


def initialize_exact_private_rows(
    embedding_weight: torch.Tensor,
    mapping: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Return one exact Base-row copy for every private row, in flat map order."""
    base_ids = torch.tensor(
        flatten_base_ids(mapping), device=embedding_weight.device, dtype=torch.long
    )
    return embedding_weight.detach().index_select(0, base_ids).float()


def _rewrite_ids(
    ids: Sequence[int], mapping: Sequence[Mapping[str, Any]]
) -> list[int]:
    """Longest-first deterministic subject-sequence rewrite preserving length."""
    source = [int(value) for value in ids]
    rules = sorted(
        [
            (
                [int(value) for value in item["base_subject_token_ids"]],
                [int(value) for value in item["private_token_ids"]],
                str(item["subject"]),
            )
            for item in mapping
        ],
        key=lambda row: (-len(row[0]), row[2]),
    )
    out: list[int] = []
    index = 0
    while index < len(source):
        matched = False
        for base_ids, private_ids, _subject in rules:
            stop = index + len(base_ids)
            if stop <= len(source) and source[index:stop] == base_ids:
                if len(base_ids) != len(private_ids):
                    raise RuntimeError("private rewrite is not position preserving")
                out.extend(private_ids)
                index = stop
                matched = True
                break
        if not matched:
            out.append(source[index])
            index += 1
    if len(out) != len(source):
        raise RuntimeError("position-preserving rewrite changed token count")
    return out


class PositionPreservingSubjectTokenizer:
    """Thin deterministic subject-sequence rewriter around the Base tokenizer.

    This wrapper changes token ids only after ordinary Base tokenization.  It
    never inspects relation text or model outputs.  Each registered subject
    sequence is replaced with its same-length private sequence in every context.
    """

    def __init__(self, base_tokenizer: Any, mapping: Sequence[Mapping[str, Any]]):
        self.base_tokenizer = base_tokenizer
        self.mapping = [dict(item) for item in mapping]
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        ):
            setattr(self, name, getattr(base_tokenizer, name, None))

    def __len__(self) -> int:
        return len(self.base_tokenizer)

    def __call__(self, text: Any, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        encoded = self.base_tokenizer(text, *args, **kwargs)
        ids = encoded["input_ids"]
        if ids and isinstance(ids[0], list):
            encoded["input_ids"] = [_rewrite_ids(row, self.mapping) for row in ids]
        else:
            encoded["input_ids"] = _rewrite_ids(ids, self.mapping)
        return encoded

    def save_pretrained(self, directory: str | Path) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.base_tokenizer.save_pretrained(path)
        (path / "private_subject_routing.json").write_text(
            json.dumps(
                {
                    "protocol": PROTOCOL,
                    "routing": "exact_longest_subject_token_sequence_same_length",
                    "relation_aware": False,
                    "mapping": self.mapping,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def load_position_preserving_tokenizer(directory: str | Path, auto_tokenizer: Any) -> PositionPreservingSubjectTokenizer:
    path = Path(directory)
    payload = json.loads((path / "private_subject_routing.json").read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("private subject routing protocol mismatch")
    base = auto_tokenizer.from_pretrained(path, use_fast=True)
    return PositionPreservingSubjectTokenizer(base, payload["mapping"])


def validate_position_preserving_routing(
    base_tokenizer: Any,
    private_tokenizer: PositionPreservingSubjectTokenizer,
    mapping: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    examples: list[Dict[str, Any]] = []
    for item in mapping:
        subject = str(item["subject"])
        base_expected = [int(value) for value in item["base_subject_token_ids"]]
        private_expected = [int(value) for value in item["private_token_ids"]]
        base_ids = base_tokenizer(
            subject, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        private_ids = private_tokenizer(
            subject, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if [int(v) for v in base_ids] != base_expected:
            raise RuntimeError(f"Base subject tokenization changed for {subject!r}")
        if [int(v) for v in private_ids] != private_expected:
            raise RuntimeError(
                f"private subject routing failed for {subject!r}: "
                f"expected {private_expected}, got {private_ids}"
            )
        if len(base_ids) != len(private_ids):
            raise RuntimeError(f"subject token count changed for {subject!r}")

        probe = f"Tell me about {subject}."
        probe_base = base_tokenizer(
            probe, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        probe_private = private_tokenizer(
            probe, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        if len(probe_base) != len(probe_private):
            raise RuntimeError(f"sentence token count changed for {subject!r}")
        examples.append(
            {
                "subject": subject,
                "base_ids": base_expected,
                "private_ids": private_expected,
                "token_count": len(base_expected),
            }
        )
    return {
        "subjects": len(mapping),
        "private_rows": len(flatten_private_ids(mapping)),
        "all_subject_lengths_preserved": True,
        "examples": examples[:5],
    }


def count_rewrites(
    base_tokenizer: Any,
    private_tokenizer: PositionPreservingSubjectTokenizer,
    text: str,
) -> int:
    base_ids = base_tokenizer(text, add_special_tokens=False)["input_ids"]
    private_ids = private_tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(base_ids) != len(private_ids):
        raise RuntimeError("rewrite changed token count")
    return sum(int(a) != int(b) for a, b in zip(base_ids, private_ids))
