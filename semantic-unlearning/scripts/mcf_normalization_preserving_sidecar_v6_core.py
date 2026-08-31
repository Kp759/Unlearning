#!/usr/bin/env python3
"""Two-sided exact-fact routing with normalization-preserving suppression.

V5 established perfect direct/paraphrase suppression, but a one-sided subject
router opened on nested entities (for example, ``Perth`` in ``HMAS Perth``).
Its additive logit bias also changed the softmax denominator, so even a route
whose registered answer tokens were unrelated to the scored neighborhood
answers could change specificity.

V6 changes exactly those two boundaries:

* a route must match a frozen relation frame on both sides of the complete
  subject; and
* an open route permutes existing logits among target-true-only tokens,
  target-new-only tokens, and reserved special-token slots.  The true-only
  tokens receive the smallest values and the new-only tokens receive the
  largest values.  No value is added or removed.  Every ordinary-vocabulary
  logit outside the two registered target sets is left untouched, and the
  full vocabulary logit multiset is preserved.

The Base embedding, Transformer, and LM head parameters are immutable.  A
closed route returns the exact LM-head Tensor object.  This remains contextual
behavioral suppression, not latent factual erasure.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence
import unicodedata

import torch
from torch import nn

import scoped_span_edit as scoped


PROTOCOL = "mcf_normalization_preserving_entity_sidecar_v6_0"
SCHEMA_VERSION = 1
FRAME_SEPARATOR = "\u241f"
RESERVED_TOKEN_PATTERN = re.compile(r"^<\|reserved_special_token_(\d+)\|>$")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value))


def normalize_frame_text(value: str) -> str:
    return " ".join(_normalize_text(value).lower().split())


def _subject_regex(subject: str) -> re.Pattern[str]:
    normalized = _normalize_text(subject)
    if not normalized.strip():
        raise ValueError("subjects must be non-empty")
    return re.compile(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)")


def normalized_left_clause(value: str) -> str:
    """Return the normalized clause fragment immediately before a subject."""

    normalized = _normalize_text(value)
    # Generated MCF prefixes commonly add an unrelated complete sentence.  It
    # must not become part of the fact grammar.  Question marks and semicolons
    # are also strong clause boundaries in the benchmark templates.
    fragment = re.split(r"[.!?;\n]+", normalized)[-1]
    return normalize_frame_text(fragment)


def subject_frame_parts(text: str, subject: str) -> list[tuple[str, str]]:
    normalized = _normalize_text(text)
    return [
        (
            normalized_left_clause(normalized[: match.start()]),
            normalize_frame_text(normalized[match.end() :]),
        )
        for match in _subject_regex(subject).finditer(normalized)
    ]


def frame_key(left_clause: str, right_relation: str) -> str:
    left = normalize_frame_text(left_clause)
    right = normalize_frame_text(right_relation)
    if FRAME_SEPARATOR in left or FRAME_SEPARATOR in right:
        raise ValueError("relation frame contains the reserved separator")
    return f"{left}{FRAME_SEPARATOR}{right}"


def split_frame_key(value: str) -> tuple[str, str]:
    parts = str(value).split(FRAME_SEPARATOR)
    if len(parts) != 2:
        raise ValueError("invalid two-sided relation-frame key")
    return parts[0], parts[1]


def frame_lexicon_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("lexicon_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frame_lexicon(value: Mapping[str, Any]) -> None:
    mapping = value.get("frame_to_relation_ids", {})
    if (
        int(value.get("schema_version", -1)) != 1
        or value.get("kind") != "mcf_frozen_two_sided_relation_frame_lexicon_v1"
        or value.get("matching_rule")
        != "exact_left_clause_and_longest_boundary_complete_right_prefix"
        or not isinstance(mapping, Mapping)
        or not mapping
        or int(value.get("frame_count", -1)) != len(mapping)
        or value.get("lexicon_sha256") != frame_lexicon_sha256(value)
    ):
        raise RuntimeError("frozen two-sided relation-frame lexicon is invalid")
    for key, relations in mapping.items():
        try:
            left, right = split_frame_key(str(key))
        except ValueError as exc:
            raise RuntimeError("relation-frame lexicon contains an invalid key") from exc
        if (
            normalize_frame_text(left) != left
            or not right
            or normalize_frame_text(right) != right
            or not isinstance(relations, list)
            or not relations
            or relations != sorted(set(str(item) for item in relations))
        ):
            raise RuntimeError("relation-frame lexicon contains an invalid row")


def relation_frame_membership(
    left_clause: str,
    remainder: str,
    frame_to_relation_ids: Mapping[str, Sequence[str]],
) -> tuple[set[str], Optional[str]]:
    """Return relations on the exact-left, longest-right matching frame."""

    left = normalized_left_clause(left_clause)
    right = normalize_frame_text(remainder)
    matches: list[tuple[int, str, set[str]]] = []
    for encoded_frame, relations in frame_to_relation_ids.items():
        candidate_left, candidate_right = split_frame_key(str(encoded_frame))
        if candidate_left != left or not right.startswith(candidate_right):
            continue
        if len(right) > len(candidate_right):
            next_character = right[len(candidate_right)]
            if candidate_right[-1].isalnum() and (
                next_character.isalnum() or next_character == "_"
            ):
                continue
        matches.append(
            (
                len(candidate_right),
                candidate_right,
                {str(item) for item in relations},
            )
        )
    if not matches:
        return set(), None
    _length, matched_right, relations = max(matches, key=lambda item: item[0])
    return relations, matched_right


def official_target_token_ids(
    tokenizer: Any,
    target: str,
    *,
    llama_like: bool,
) -> list[int]:
    encoded = tokenizer(f" {str(target)}")["input_ids"]
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("target tokenizer unexpectedly returned a batch")
        encoded = encoded[0]
    values = [int(value) for value in encoded]
    if llama_like:
        if not values:
            raise ValueError("Llama-like target encoding is empty")
        values = values[1:]
    if not values:
        raise ValueError("target encoding is empty after special-token removal")
    return values


def select_reserved_token_pool(
    tokenizer: Any,
    *,
    excluded_token_ids: Sequence[int],
    requested_size: int = 128,
) -> tuple[list[int], list[str]]:
    if int(requested_size) <= 0:
        raise ValueError("reserved-token pool size must be positive")
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, Mapping):
        raise RuntimeError("tokenizer does not expose a vocabulary mapping")
    excluded = {int(item) for item in excluded_token_ids}
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            excluded.add(int(value))
    rows = []
    for token, token_id in vocabulary.items():
        match = RESERVED_TOKEN_PATTERN.fullmatch(str(token))
        if match is None or int(token_id) in excluded:
            continue
        rows.append((int(match.group(1)), int(token_id), str(token)))
    rows.sort()
    if len(rows) < int(requested_size):
        raise RuntimeError(
            f"need {requested_size} unused reserved tokens, found {len(rows)}"
        )
    selected = rows[: int(requested_size)]
    return [row[1] for row in selected], [row[2] for row in selected]


def directional_target_token_banks(
    target_new_ids: Sequence[Sequence[int]],
    target_true_ids: Sequence[Sequence[int]],
) -> tuple[list[list[int]], list[list[int]], Dict[str, Any]]:
    if len(target_new_ids) != len(target_true_ids) or not target_new_ids:
        raise ValueError("target token banks must be non-empty and aligned")
    promoted_rows: list[list[int]] = []
    sensitive_rows: list[list[int]] = []
    report = []
    for record_index, (new_values, true_values) in enumerate(
        zip(target_new_ids, target_true_ids)
    ):
        new_set = {int(item) for item in new_values}
        true_set = {int(item) for item in true_values}
        promoted = sorted(new_set - true_set)
        sensitive = sorted(true_set - new_set)
        if (
            not new_set
            or not true_set
            or min(new_set | true_set) < 0
            or not promoted
            or not sensitive
        ):
            raise ValueError(
                "each target pair needs target_new- and target_true-exclusive tokens"
            )
        promoted_rows.append(promoted)
        sensitive_rows.append(sensitive)
        report.append(
            {
                "record_index": int(record_index),
                "promoted_token_ids": promoted,
                "sensitive_token_ids": sensitive,
                "shared_target_token_ids": sorted(new_set & true_set),
            }
        )
    return promoted_rows, sensitive_rows, {
        "records": len(sensitive_rows),
        "rule": (
            "permute_lowest_logits_to_target_true_only_and_highest_logits_to_"
            "target_new_only_using_reserved_special_slots"
        ),
        "arithmetic_logit_offsets": 0,
        "rows": report,
    }


class TwoSidedEntityRelationRouter(scoped.SpanGateRouter):
    """Complete subject AND frozen left/right relation-frame router."""

    def __init__(
        self,
        embedding: nn.Module,
        subject_patterns: Sequence[Sequence[Sequence[int]]],
        *,
        subjects: Sequence[str],
        relation_ids: Sequence[str],
        frame_to_relation_ids: Mapping[str, Sequence[str]],
        tokenizer: Any,
        model: Optional[nn.Module] = None,
    ) -> None:
        if len(relation_ids) != len(subjects):
            raise ValueError("fact router subjects and relations must align")
        self.tokenizer = tokenizer
        self.subjects = [str(item) for item in subjects]
        self._compiled_subjects = [_subject_regex(item) for item in subjects]
        self.relation_ids = [str(item) for item in relation_ids]
        self.frame_to_relation_ids = {
            str(key): [str(item) for item in relations]
            for key, relations in frame_to_relation_ids.items()
        }
        if not self.frame_to_relation_ids:
            raise ValueError("fact router relation-frame grammar is empty")
        super().__init__(
            embedding,
            subject_patterns,
            subjects=subjects,
            model=model,
        )

    @staticmethod
    def _starts(sequence: Sequence[int], pattern: Sequence[int]) -> list[int]:
        width = len(pattern)
        if width == 0 or width > len(sequence):
            return []
        pattern_list = list(pattern)
        return [
            start
            for start in range(len(sequence) - width + 1)
            if list(sequence[start : start + width]) == pattern_list
        ]

    def _fact_complete(
        self,
        sequence: Sequence[int],
        *,
        start: int,
        width: int,
        record_index: int,
    ) -> bool:
        left_text = self.tokenizer.decode(
            list(sequence[: int(start)]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        candidate_text = self.tokenizer.decode(
            list(sequence[int(start) :]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        normalized_candidate = _normalize_text(candidate_text)
        match = self._compiled_subjects[record_index].search(normalized_candidate)
        if match is None or normalized_candidate[: match.start()].strip():
            return False
        relations, _matched_right = relation_frame_membership(
            left_text,
            normalized_candidate[match.end() :],
            self.frame_to_relation_ids,
        )
        return self.relation_ids[record_index] in relations

    def route(self, input_ids: torch.Tensor) -> scoped.RouteState:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("V6 entity router expects [batch, sequence] input IDs")
        batch, sequence_length = input_ids.shape
        active = torch.zeros((batch, self.n_records), dtype=torch.bool)
        masks = torch.zeros(
            (batch, self.n_records, sequence_length), dtype=torch.bool
        )
        if not self.enabled:
            return scoped.RouteState(active=active, span_masks=masks)

        for batch_index, sequence in enumerate(input_ids.detach().cpu().tolist()):
            candidates: list[tuple[int, int, int, int]] = []
            for record_index, patterns in enumerate(self.subject_patterns):
                matches: list[tuple[int, int]] = []
                for pattern in patterns:
                    for start in self._starts(sequence, pattern):
                        if self._fact_complete(
                            sequence,
                            start=start,
                            width=len(pattern),
                            record_index=record_index,
                        ):
                            matches.append((len(pattern), start))
                if matches:
                    width, start = max(matches, key=lambda item: item[0])
                    candidates.append(
                        (
                            record_index,
                            int(self.subject_priorities[record_index]),
                            width,
                            start,
                        )
                    )
            accepted: list[tuple[int, int, int, int]] = []
            for candidate in sorted(
                candidates, key=lambda item: (-item[1], -item[2], item[3])
            ):
                _, _, width, start = candidate
                end = start + width
                conflict = False
                for _, _, kept_width, kept_start in accepted:
                    kept_end = kept_start + kept_width
                    overlaps = start < kept_end and kept_start < end
                    same_span = start == kept_start and end == kept_end
                    if overlaps and not same_span:
                        conflict = True
                        break
                if not conflict:
                    accepted.append(candidate)
            for record_index, _priority, width, start in accepted:
                active[batch_index, record_index] = True
                masks[batch_index, record_index, start : start + width] = True
        return scoped.RouteState(active=active, span_masks=masks)


def causal_after_subject_mask(span_masks: torch.Tensor) -> torch.Tensor:
    if span_masks.ndim != 3 or span_masks.dtype is not torch.bool:
        raise ValueError("span masks must be boolean [batch, records, sequence]")
    if int(span_masks.shape[-1]) == 0:
        return span_masks.clone()
    next_is_subject = torch.zeros_like(span_masks)
    next_is_subject[..., :-1] = span_masks[..., 1:]
    endpoints = span_masks & ~next_is_subject
    return endpoints.to(torch.int64).cumsum(dim=-1).gt(0)


class NormalizationPreservingSensitiveTokenSidecar:
    """Directionally permute routed target logits without changing their multiset."""

    def __init__(
        self,
        output_layer: nn.Module,
        router: TwoSidedEntityRelationRouter,
        promoted_token_ids: Sequence[Sequence[int]],
        sensitive_token_ids: Sequence[Sequence[int]],
        reserved_token_ids: Sequence[int],
    ) -> None:
        if (
            len(promoted_token_ids) != router.n_records
            or len(sensitive_token_ids) != router.n_records
        ):
            raise ValueError("sidecar records do not match the entity router")
        self.promoted_token_ids = [
            sorted(set(int(item) for item in row)) for row in promoted_token_ids
        ]
        self.sensitive_token_ids = [
            sorted(set(int(item) for item in row)) for row in sensitive_token_ids
        ]
        self.reserved_token_ids = [int(item) for item in reserved_token_ids]
        if (
            any(not row for row in self.promoted_token_ids)
            or any(not row for row in self.sensitive_token_ids)
            or not self.reserved_token_ids
            or len(set(self.reserved_token_ids)) != len(self.reserved_token_ids)
            or set(self.reserved_token_ids).intersection(
                item for row in self.sensitive_token_ids for item in row
            )
            or set(self.reserved_token_ids).intersection(
                item for row in self.promoted_token_ids for item in row
            )
            or any(
                set(promoted) & set(sensitive)
                for promoted, sensitive in zip(
                    self.promoted_token_ids, self.sensitive_token_ids
                )
            )
        ):
            raise ValueError("directional target and reserved token banks are invalid")
        self.router = router
        self.enabled = True
        self.calls = 0
        self.fired_rows = 0
        self.permuted_positions = 0
        self._handle = output_layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        self.calls += 1
        state = self.router.state
        if not self.enabled or state is None or not bool(state.active.any()):
            return output
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("LM-head output must be [batch, sequence, vocabulary]")
        maximum_id = max(
            [*self.reserved_token_ids]
            + [item for row in self.promoted_token_ids for item in row]
            + [item for row in self.sensitive_token_ids for item in row]
        )
        if maximum_id >= int(logits.shape[-1]):
            raise RuntimeError("V6 sidecar token ID exceeds model vocabulary")
        causal = causal_after_subject_mask(state.span_masks).to(logits.device)
        if (
            int(causal.shape[0]) != int(logits.shape[0])
            or int(causal.shape[2]) != int(logits.shape[1])
        ):
            raise RuntimeError("V6 route and LM-head logits are misaligned")
        updated = logits.clone()
        reserved = torch.tensor(
            self.reserved_token_ids, dtype=torch.long, device=logits.device
        )
        for batch_index, record_index in state.active.nonzero(
            as_tuple=False
        ).tolist():
            positions = causal[batch_index, record_index].nonzero(
                as_tuple=False
            ).flatten()
            if int(positions.numel()) == 0:
                continue
            sensitive = torch.tensor(
                self.sensitive_token_ids[record_index],
                dtype=torch.long,
                device=logits.device,
            )
            promoted = torch.tensor(
                self.promoted_token_ids[record_index],
                dtype=torch.long,
                device=logits.device,
            )
            source_ids = torch.cat((sensitive, reserved, promoted))
            values = updated[
                batch_index, positions[:, None], source_ids[None, :]
            ]
            sorted_values = values.sort(dim=-1).values
            sensitive_width = int(sensitive.numel())
            promoted_width = int(promoted.numel())
            updated[
                batch_index, positions[:, None], sensitive[None, :]
            ] = sorted_values[:, :sensitive_width]
            updated[
                batch_index, positions[:, None], reserved[None, :]
            ] = sorted_values[:, sensitive_width:-promoted_width]
            updated[
                batch_index, positions[:, None], promoted[None, :]
            ] = sorted_values[:, -promoted_width:]
            self.fired_rows += 1
            self.permuted_positions += int(positions.numel())
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    def close(self) -> None:
        self._handle.remove()


@dataclass
class NormalizationPreservingRuntime:
    model: nn.Module
    router: TwoSidedEntityRelationRouter
    sidecar: NormalizationPreservingSensitiveTokenSidecar
    metadata: Dict[str, Any]

    def close(self) -> None:
        self.sidecar.close()
        self.router.close()


def build_candidate_state(
    *,
    seed: int,
    case_ids: Sequence[int],
    subjects: Sequence[str],
    relation_ids: Sequence[str],
    frame_lexicon: Mapping[str, Any],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    target_new: Sequence[str],
    target_true: Sequence[str],
    target_new_ids: Sequence[Sequence[int]],
    target_true_ids: Sequence[Sequence[int]],
    reserved_token_ids: Sequence[int],
    reserved_token_strings: Sequence[str],
    llama_like: bool,
    base_embedding_sha256: str,
    base_lm_head_sha256: str,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if int(seed) <= 0:
        raise ValueError("V6 seed must be positive")
    records = len(case_ids)
    aligned = all(
        len(values) == records
        for values in (
            subjects,
            relation_ids,
            subject_patterns,
            target_new,
            target_true,
            target_new_ids,
            target_true_ids,
        )
    )
    if records <= 0 or not aligned or len(set(str(item) for item in subjects)) != records:
        raise ValueError("V6 requires aligned records with unique complete subjects")
    validate_frame_lexicon(frame_lexicon)
    available_relations = {
        str(relation)
        for relations in frame_lexicon["frame_to_relation_ids"].values()
        for relation in relations
    }
    missing = sorted(set(str(item) for item in relation_ids) - available_relations)
    if missing:
        raise ValueError(f"V6 relation-frame grammar lacks relations: {missing}")
    patterns = [
        [[int(token) for token in pattern] for pattern in record_patterns]
        for record_patterns in subject_patterns
    ]
    if any(not row for row in patterns):
        raise ValueError("every V6 record needs a subject token pattern")
    promoted, sensitive, directional_report = directional_target_token_banks(
        target_new_ids, target_true_ids
    )
    reserved_ids = [int(item) for item in reserved_token_ids]
    reserved_strings = [str(item) for item in reserved_token_strings]
    target_ids = {
        int(item)
        for bank in (target_new_ids, target_true_ids)
        for row in bank
        for item in row
    }
    if (
        len(reserved_ids) < 32
        or len(reserved_ids) != len(reserved_strings)
        or len(set(reserved_ids)) != len(reserved_ids)
        or set(reserved_ids) & target_ids
        or any(RESERVED_TOKEN_PATTERN.fullmatch(item) is None for item in reserved_strings)
    ):
        raise ValueError("V6 reserved special-token reservoir is invalid")
    valid_hash = re.compile(r"^[0-9a-f]{64}$")
    if not valid_hash.fullmatch(str(base_embedding_sha256)) or not valid_hash.fullmatch(
        str(base_lm_head_sha256)
    ):
        raise ValueError("V6 requires Base embedding and LM-head hashes")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mcf_normalization_preserving_entity_sidecar_candidate",
        "protocol": PROTOCOL,
        "seed": int(seed),
        "architecture": {
            "route": "two_sided_complete_entity_and_frozen_relation_frame_grammar",
            "intervention": "causal_directional_target_logit_permutation_with_reserved_special_slots",
            "normalization_preserving_by_construction": True,
            "ordinary_non_target_logits_unchanged_by_construction": True,
            "arithmetic_logit_offsets": 0,
            "embedding_parameters_mutated": False,
            "transformer_parameters_mutated": False,
            "lm_head_parameters_mutated": False,
            "learned_detector_parameters": 0,
            "learned_actuator_parameters": 0,
            "closed_route_returns_base_tensor_object": True,
            "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
        },
        "case_ids": [int(item) for item in case_ids],
        "subjects": [str(item) for item in subjects],
        "relation_ids": [str(item) for item in relation_ids],
        "frame_lexicon": dict(frame_lexicon),
        "frame_lexicon_sha256": str(frame_lexicon["lexicon_sha256"]),
        "subject_patterns": patterns,
        "target_new": [str(item) for item in target_new],
        "target_true": [str(item) for item in target_true],
        "target_new_ids": [[int(item) for item in row] for row in target_new_ids],
        "target_true_ids": [[int(item) for item in row] for row in target_true_ids],
        "promoted_token_ids": promoted,
        "sensitive_token_ids": sensitive,
        "reserved_token_ids": reserved_ids,
        "reserved_token_strings": reserved_strings,
        "directional_permutation_report": directional_report,
        "llama_like": bool(llama_like),
        "base_embedding_sha256": str(base_embedding_sha256),
        "base_lm_head_sha256": str(base_lm_head_sha256),
        "source_hashes": {str(key): str(value) for key, value in source_hashes.items()},
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
        "official_evaluation_prompts_seen": 0,
    }


def validate_candidate_state(state: Mapping[str, Any]) -> None:
    if (
        int(state.get("schema_version", -1)) != SCHEMA_VERSION
        or state.get("kind") != "mcf_normalization_preserving_entity_sidecar_candidate"
        or state.get("protocol") != PROTOCOL
        or int(state.get("seed", -1)) <= 0
    ):
        raise RuntimeError("unsupported V6 candidate state")
    required_architecture = {
        "route": "two_sided_complete_entity_and_frozen_relation_frame_grammar",
        "intervention": "causal_directional_target_logit_permutation_with_reserved_special_slots",
        "normalization_preserving_by_construction": True,
        "ordinary_non_target_logits_unchanged_by_construction": True,
        "arithmetic_logit_offsets": 0,
        "embedding_parameters_mutated": False,
        "transformer_parameters_mutated": False,
        "lm_head_parameters_mutated": False,
        "learned_detector_parameters": 0,
        "learned_actuator_parameters": 0,
        "closed_route_returns_base_tensor_object": True,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    architecture = state.get("architecture", {})
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value
        for key, value in required_architecture.items()
    ):
        raise RuntimeError("V6 candidate architecture metadata is invalid")
    records = len(state.get("case_ids", []))
    aligned_keys = (
        "subjects",
        "relation_ids",
        "subject_patterns",
        "target_new",
        "target_true",
        "target_new_ids",
        "target_true_ids",
        "promoted_token_ids",
        "sensitive_token_ids",
    )
    if records <= 0 or any(len(state.get(key, [])) != records for key in aligned_keys):
        raise RuntimeError("V6 candidate record tensors are not aligned")
    validate_frame_lexicon(state.get("frame_lexicon", {}))
    if state.get("frame_lexicon_sha256") != state["frame_lexicon"]["lexicon_sha256"]:
        raise RuntimeError("V6 candidate frame-lexicon hash differs")
    promoted, sensitive, _report = directional_target_token_banks(
        state["target_new_ids"], state["target_true_ids"]
    )
    if (
        promoted != state["promoted_token_ids"]
        or sensitive != state["sensitive_token_ids"]
    ):
        raise RuntimeError("V6 directional target token banks do not reproduce")
    reserved_ids = [int(item) for item in state.get("reserved_token_ids", [])]
    reserved_strings = [str(item) for item in state.get("reserved_token_strings", [])]
    target_ids = {
        int(item)
        for bank in (state["target_new_ids"], state["target_true_ids"])
        for row in bank
        for item in row
    }
    if (
        len(reserved_ids) < 32
        or len(reserved_ids) != len(reserved_strings)
        or len(set(reserved_ids)) != len(reserved_ids)
        or set(reserved_ids) & target_ids
        or any(RESERVED_TOKEN_PATTERN.fullmatch(item) is None for item in reserved_strings)
    ):
        raise RuntimeError("V6 candidate reserved-token reservoir is invalid")
    valid_hash = re.compile(r"^[0-9a-f]{64}$")
    if not valid_hash.fullmatch(str(state.get("base_embedding_sha256", ""))) or not (
        valid_hash.fullmatch(str(state.get("base_lm_head_sha256", "")))
    ):
        raise RuntimeError("V6 candidate Base parameter hashes are invalid")
    if state.get("optimizer_constructed") is not False or int(
        state.get("gradient_updates_performed", -1)
    ) != 0:
        raise RuntimeError("V6 candidate unexpectedly records optimization")


def install_candidate(
    model: nn.Module,
    tokenizer: Any,
    state_or_path: Mapping[str, Any] | str | Path,
) -> NormalizationPreservingRuntime:
    state = (
        torch.load(Path(state_or_path), map_location="cpu", weights_only=False)
        if isinstance(state_or_path, (str, Path))
        else dict(state_or_path)
    )
    if not isinstance(state, Mapping):
        raise RuntimeError("V6 candidate file is not a mapping")
    validate_candidate_state(state)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("model must expose input and output embeddings")
    if tensor_sha256(input_layer.weight) != state["base_embedding_sha256"]:
        raise RuntimeError("V6 candidate input embedding differs from Base binding")
    if tensor_sha256(output_layer.weight) != state["base_lm_head_sha256"]:
        raise RuntimeError("V6 candidate LM head differs from Base binding")
    if scoped.build_subject_patterns(tokenizer, state["subjects"]) != state[
        "subject_patterns"
    ]:
        raise RuntimeError("V6 candidate subject patterns do not reproduce")
    new_ids = [
        official_target_token_ids(
            tokenizer, item, llama_like=bool(state["llama_like"])
        )
        for item in state["target_new"]
    ]
    true_ids = [
        official_target_token_ids(
            tokenizer, item, llama_like=bool(state["llama_like"])
        )
        for item in state["target_true"]
    ]
    if new_ids != state["target_new_ids"] or true_ids != state["target_true_ids"]:
        raise RuntimeError("V6 candidate target token IDs do not reproduce")
    reproduced_reserved = tokenizer.convert_ids_to_tokens(state["reserved_token_ids"])
    if isinstance(reproduced_reserved, str):
        reproduced_reserved = [reproduced_reserved]
    if [str(item) for item in reproduced_reserved] != state["reserved_token_strings"]:
        raise RuntimeError("V6 reserved special tokens do not reproduce")
    router = TwoSidedEntityRelationRouter(
        input_layer,
        state["subject_patterns"],
        subjects=state["subjects"],
        relation_ids=state["relation_ids"],
        frame_to_relation_ids=state["frame_lexicon"]["frame_to_relation_ids"],
        tokenizer=tokenizer,
        model=model,
    )
    sidecar = NormalizationPreservingSensitiveTokenSidecar(
        output_layer,
        router,
        state["promoted_token_ids"],
        state["sensitive_token_ids"],
        state["reserved_token_ids"],
    )
    return NormalizationPreservingRuntime(
        model=model,
        router=router,
        sidecar=sidecar,
        metadata={
            "protocol": PROTOCOL,
            "seed": int(state["seed"]),
            "case_ids": [int(item) for item in state["case_ids"]],
            "subjects": [str(item) for item in state["subjects"]],
            "relation_ids": [str(item) for item in state["relation_ids"]],
            "claim_scope": state["architecture"]["claim_scope"],
        },
    )
