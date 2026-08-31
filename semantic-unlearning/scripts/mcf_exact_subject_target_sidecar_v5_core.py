#!/usr/bin/env python3
"""Exact-subject contextual target suppression without Base-model mutation.

V3.6.2 proved that the isolated actuator was strong enough to erase every
training context, but its learned detector missed official paraphrases.  V4.2
then proved that the sparse V6.2 marker itself can be absent on fresh positive
paraphrases.  This module removes both learned availability assumptions.

The route is the conjunction of a boundary-aware complete subject and a
frozen, zero-parameter relation-suffix grammar derived outside the reserved
seed-1-to-10 evaluation records.  For a routed row, a sparse output sidecar
promotes the registered reference (``target_new`` in CounterFact nomenclature)
and suppresses the registered sensitive answer (``target_true``).  The input
embedding, Transformer, and LM head parameters are never changed.  If the
subject-relation conjunction is absent, the LM-head hook returns its input
Tensor object unchanged, giving exact Base logits.

This is an explicit contextual behavioral-suppression mechanism.  It is not a
claim that the model's latent factual representation has been erased.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn

import scoped_span_edit as scoped


PROTOCOL = "mcf_exact_subject_target_logit_sidecar_v5_0"
SCHEMA_VERSION = 1
DEFAULT_LOGIT_BIAS = 256.0


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value))


def _subject_regex(subject: str) -> re.Pattern[str]:
    normalized = _normalize_text(subject)
    if not normalized.strip():
        raise ValueError("subjects must be non-empty")
    # Unicode word boundaries are intentional.  They close prefix collisions
    # such as BMW M5/BMW M54 and iPhone 1/iPhone 12 while retaining punctuation
    # and whitespace after a genuine complete subject.
    return re.compile(r"(?<!\w)" + re.escape(normalized) + r"(?!\w)")


def complete_subject_remainder(text: str, subject: str) -> Optional[str]:
    normalized = _normalize_text(text)
    match = _subject_regex(subject).search(normalized)
    return None if match is None else normalized[match.end() :]


def normalize_relation_suffix(value: str) -> str:
    return " ".join(_normalize_text(value).lower().split())


def relation_lexicon_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("lexicon_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_relation_lexicon(value: Mapping[str, Any]) -> None:
    mapping = value.get("suffix_to_relation_ids", {})
    if (
        int(value.get("schema_version", -1)) != 1
        or value.get("kind") != "mcf_frozen_relation_suffix_lexicon_v1"
        or not isinstance(mapping, Mapping)
        or not mapping
        or int(value.get("suffix_count", -1)) != len(mapping)
        or value.get("matching_rule")
        != "longest_boundary_complete_prefix_relation_membership"
        or value.get("lexicon_sha256") != relation_lexicon_sha256(value)
    ):
        raise RuntimeError("frozen relation suffix lexicon is invalid")
    for suffix, relations in mapping.items():
        if (
            not str(suffix)
            or normalize_relation_suffix(str(suffix)) != str(suffix)
            or not isinstance(relations, list)
            or not relations
            or relations != sorted(set(str(relation) for relation in relations))
        ):
            raise RuntimeError("relation suffix lexicon contains an invalid row")


def relation_suffix_membership(
    remainder: str,
    suffix_to_relation_ids: Mapping[str, Sequence[str]],
) -> tuple[set[str], Optional[str]]:
    """Return relation IDs on the longest boundary-complete suffix prefix."""

    normalized = normalize_relation_suffix(remainder)
    matches: list[tuple[int, str, set[str]]] = []
    for suffix, relations in suffix_to_relation_ids.items():
        key = str(suffix)
        if not normalized.startswith(key):
            continue
        if len(normalized) > len(key):
            next_character = normalized[len(key)]
            if key[-1].isalnum() and (next_character.isalnum() or next_character == "_"):
                continue
        matches.append((len(key), key, {str(value) for value in relations}))
    if not matches:
        return set(), None
    _length, suffix, relations = max(matches, key=lambda value: value[0])
    return relations, suffix


def official_target_token_ids(
    tokenizer: Any,
    target: str,
    *,
    llama_like: bool,
) -> list[int]:
    """Tokenize a target exactly as the repository's official MCF scorer."""

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


class BoundaryAwareSubjectRouter(scoped.SpanGateRouter):
    """Exact-subject AND frozen relation-grammar fact router."""

    def __init__(
        self,
        embedding: nn.Module,
        subject_patterns: Sequence[Sequence[Sequence[int]]],
        *,
        subjects: Sequence[str],
        relation_ids: Sequence[str],
        suffix_to_relation_ids: Mapping[str, Sequence[str]],
        tokenizer: Any,
        model: Optional[nn.Module] = None,
    ) -> None:
        if len(relation_ids) != len(subjects):
            raise ValueError("fact router subjects and relations must align")
        self.tokenizer = tokenizer
        self._compiled_subjects = [_subject_regex(value) for value in subjects]
        self.relation_ids = [str(value) for value in relation_ids]
        self.suffix_to_relation_ids = {
            str(suffix): [str(value) for value in relations]
            for suffix, relations in suffix_to_relation_ids.items()
        }
        if not self.suffix_to_relation_ids:
            raise ValueError("fact router relation suffix grammar is empty")
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
        target = list(pattern)
        return [
            start
            for start in range(len(sequence) - width + 1)
            if list(sequence[start : start + width]) == target
        ]

    def _fact_complete(
        self,
        sequence: Sequence[int],
        *,
        start: int,
        width: int,
        record_index: int,
    ) -> bool:
        # Decode from the matched token through the prompt completion.  The
        # lexical regex closes name-prefix collisions; the longest-prefix
        # grammar then identifies the relation phrase immediately following
        # the complete subject.
        left = int(start)
        right = len(sequence)
        text = self.tokenizer.decode(
            list(sequence[left:right]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        normalized = _normalize_text(text)
        match = self._compiled_subjects[record_index].search(normalized)
        if match is None:
            return False
        relations, _suffix = relation_suffix_membership(
            normalized[match.end() :], self.suffix_to_relation_ids
        )
        return self.relation_ids[record_index] in relations

    def route(self, input_ids: torch.Tensor) -> scoped.RouteState:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError("V5 subject router expects [batch, sequence] input IDs")
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
                record_matches: list[tuple[int, int]] = []
                for pattern in patterns:
                    for start in self._starts(sequence, pattern):
                        if self._fact_complete(
                            sequence,
                            start=start,
                            width=len(pattern),
                            record_index=record_index,
                        ):
                            record_matches.append((len(pattern), start))
                if record_matches:
                    width, start = max(record_matches, key=lambda value: value[0])
                    candidates.append(
                        (
                            record_index,
                            int(self.subject_priorities[record_index]),
                            width,
                            start,
                        )
                    )

            # Keep every non-overlapping subject.  Resolve only genuine span
            # overlap by preferring the longest complete name.  Equal spans
            # remain explicit for duplicate-name records (although V5's
            # candidate builder currently rejects such ambiguous bindings).
            accepted: list[tuple[int, int, int, int]] = []
            for candidate in sorted(
                candidates, key=lambda value: (-value[1], -value[2], value[3])
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
    """Positions whose next-token predictions occur after a complete key."""

    if span_masks.ndim != 3 or span_masks.dtype is not torch.bool:
        raise ValueError("span masks must be boolean [batch, records, sequence]")
    if int(span_masks.shape[-1]) == 0:
        return span_masks.clone()
    next_is_subject = torch.zeros_like(span_masks)
    next_is_subject[..., :-1] = span_masks[..., 1:]
    endpoints = span_masks & ~next_is_subject
    return endpoints.to(torch.int64).cumsum(dim=-1).gt(0)


def sparse_token_biases(
    target_new_ids: Sequence[Sequence[int]],
    target_true_ids: Sequence[Sequence[int]],
    *,
    logit_bias: float,
) -> tuple[list[list[int]], list[list[float]], Dict[str, Any]]:
    """Build non-cancelling target-token bias sets for each record."""

    if len(target_new_ids) != len(target_true_ids) or not target_new_ids:
        raise ValueError("target token banks must be non-empty and aligned")
    if not float(logit_bias) > 0.0 or not torch.isfinite(
        torch.tensor(float(logit_bias))
    ):
        raise ValueError("logit bias must be finite and positive")
    row_ids: list[list[int]] = []
    row_biases: list[list[float]] = []
    report_rows = []
    for record_index, (new_values, true_values) in enumerate(
        zip(target_new_ids, target_true_ids)
    ):
        new_set = {int(value) for value in new_values}
        true_set = {int(value) for value in true_values}
        if not new_set or not true_set or min(new_set | true_set) < 0:
            raise ValueError("target token IDs must be non-negative and non-empty")
        promote = sorted(new_set - true_set)
        suppress = sorted(true_set - new_set)
        if not promote or not suppress:
            raise ValueError(
                "each target pair needs at least one direction-specific token"
            )
        ids = [*promote, *suppress]
        biases = [float(logit_bias)] * len(promote) + [
            -float(logit_bias)
        ] * len(suppress)
        row_ids.append(ids)
        row_biases.append(biases)
        report_rows.append(
            {
                "record_index": int(record_index),
                "promoted_token_ids": promote,
                "suppressed_token_ids": suppress,
                "shared_token_ids": sorted(new_set & true_set),
            }
        )
    return row_ids, row_biases, {
        "records": len(row_ids),
        "logit_bias": float(logit_bias),
        "rule": "promote_target_new_only_and_suppress_target_true_only",
        "rows": report_rows,
    }


class ExactSubjectTargetLogitSidecar:
    """Sparse causal LM-head bias controlled by complete-subject identity."""

    def __init__(
        self,
        output_layer: nn.Module,
        router: BoundaryAwareSubjectRouter,
        token_ids: Sequence[Sequence[int]],
        token_biases: Sequence[Sequence[float]],
    ) -> None:
        if len(token_ids) != router.n_records or len(token_biases) != router.n_records:
            raise ValueError("sidecar records do not match the subject router")
        if any(
            not ids or len(ids) != len(biases)
            for ids, biases in zip(token_ids, token_biases)
        ):
            raise ValueError("every sidecar row needs aligned token IDs and biases")
        self.router = router
        self.token_ids = [[int(value) for value in row] for row in token_ids]
        self.token_biases = [
            [float(value) for value in row] for row in token_biases
        ]
        self.enabled = True
        self.calls = 0
        self.fired_rows = 0
        self.corrected_positions = 0
        self._handle = output_layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _inputs: Any, output: Any) -> Any:
        self.calls += 1
        state = self.router.state
        if not self.enabled or state is None or not bool(state.active.any()):
            # Structural identity: do not clone, cast, add zero, or reconstruct.
            return output
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise TypeError("LM-head output must be [batch, sequence, vocabulary]")
        causal = causal_after_subject_mask(state.span_masks).to(logits.device)
        if (
            int(causal.shape[0]) != int(logits.shape[0])
            or int(causal.shape[2]) != int(logits.shape[1])
        ):
            raise RuntimeError("sidecar route and LM-head logits are misaligned")
        updated = logits.clone()
        for batch_index, record_index in state.active.nonzero(
            as_tuple=False
        ).tolist():
            positions = causal[batch_index, record_index]
            count = int(positions.sum())
            if count == 0:
                continue
            for token_id, value in zip(
                self.token_ids[record_index], self.token_biases[record_index]
            ):
                if token_id >= int(updated.shape[-1]):
                    raise RuntimeError("sidecar token ID exceeds model vocabulary")
                updated[batch_index, positions, token_id] += float(value)
            self.fired_rows += 1
            self.corrected_positions += count
        if isinstance(output, tuple):
            return (updated, *output[1:])
        return updated

    def close(self) -> None:
        self._handle.remove()


@dataclass
class ExactSubjectTargetRuntime:
    model: nn.Module
    router: BoundaryAwareSubjectRouter
    sidecar: ExactSubjectTargetLogitSidecar
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
    relation_lexicon: Mapping[str, Any],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    target_new: Sequence[str],
    target_true: Sequence[str],
    target_new_ids: Sequence[Sequence[int]],
    target_true_ids: Sequence[Sequence[int]],
    llama_like: bool,
    logit_bias: float,
    base_embedding_sha256: str,
    base_lm_head_sha256: str,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    if int(seed) <= 0:
        raise ValueError("V5 seed must be positive")
    records = len(case_ids)
    aligned = (
        records > 0
        and len(subjects) == records
        and len(relation_ids) == records
        and len(subject_patterns) == records
        and len(target_new) == records
        and len(target_true) == records
        and len(target_new_ids) == records
        and len(target_true_ids) == records
    )
    valid_hash = re.compile(r"^[0-9a-f]{64}$")
    if not aligned or len(set(str(value) for value in subjects)) != records:
        raise ValueError("V5 requires aligned records with unique complete subjects")
    if not valid_hash.fullmatch(str(base_embedding_sha256)) or not valid_hash.fullmatch(
        str(base_lm_head_sha256)
    ):
        raise ValueError("V5 requires cryptographic Base embedding and LM-head hashes")
    validate_relation_lexicon(relation_lexicon)
    available_relations = {
        str(relation)
        for relations in relation_lexicon["suffix_to_relation_ids"].values()
        for relation in relations
    }
    missing_relations = sorted(set(str(value) for value in relation_ids) - available_relations)
    if missing_relations:
        raise ValueError(f"V5 relation grammar lacks registered relations: {missing_relations}")
    token_ids, token_biases, bias_report = sparse_token_biases(
        target_new_ids, target_true_ids, logit_bias=float(logit_bias)
    )
    patterns = [
        [[int(token) for token in pattern] for pattern in record_patterns]
        for record_patterns in subject_patterns
    ]
    if any(not record_patterns for record_patterns in patterns):
        raise ValueError("every V5 record needs a subject token pattern")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mcf_exact_subject_target_logit_sidecar_candidate",
        "protocol": PROTOCOL,
        "seed": int(seed),
        "architecture": {
            "route": "boundary_aware_exact_subject_and_frozen_relation_suffix_grammar",
            "intervention": "causal_sparse_target_logit_bias",
            "embedding_parameters_mutated": False,
            "transformer_parameters_mutated": False,
            "lm_head_parameters_mutated": False,
            "learned_detector_parameters": 0,
            "learned_actuator_parameters": 0,
            "closed_route_returns_base_tensor_object": True,
            "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
        },
        "case_ids": [int(value) for value in case_ids],
        "subjects": [str(value) for value in subjects],
        "relation_ids": [str(value) for value in relation_ids],
        "relation_lexicon": dict(relation_lexicon),
        "relation_lexicon_sha256": str(relation_lexicon["lexicon_sha256"]),
        "subject_patterns": patterns,
        "target_new": [str(value) for value in target_new],
        "target_true": [str(value) for value in target_true],
        "target_new_ids": [[int(value) for value in row] for row in target_new_ids],
        "target_true_ids": [[int(value) for value in row] for row in target_true_ids],
        "sidecar_token_ids": token_ids,
        "sidecar_token_biases": token_biases,
        "bias_report": bias_report,
        "llama_like": bool(llama_like),
        "logit_bias": float(logit_bias),
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
        or state.get("kind") != "mcf_exact_subject_target_logit_sidecar_candidate"
        or state.get("protocol") != PROTOCOL
        or int(state.get("seed", -1)) <= 0
    ):
        raise RuntimeError("unsupported V5 candidate state")
    architecture = state.get("architecture", {})
    required = {
        "embedding_parameters_mutated": False,
        "transformer_parameters_mutated": False,
        "lm_head_parameters_mutated": False,
        "learned_detector_parameters": 0,
        "learned_actuator_parameters": 0,
        "closed_route_returns_base_tensor_object": True,
        "claim_scope": "contextual_behavioral_suppression_not_latent_erasure",
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("V5 candidate architecture metadata is invalid")
    records = len(state.get("case_ids", []))
    aligned_keys = (
        "subjects",
        "relation_ids",
        "subject_patterns",
        "target_new",
        "target_true",
        "target_new_ids",
        "target_true_ids",
        "sidecar_token_ids",
        "sidecar_token_biases",
    )
    if records <= 0 or any(len(state.get(key, [])) != records for key in aligned_keys):
        raise RuntimeError("V5 candidate record tensors are not aligned")
    if len(set(str(value) for value in state["subjects"])) != records:
        raise RuntimeError("V5 candidate contains duplicate subjects")
    relation_lexicon = state.get("relation_lexicon", {})
    if not isinstance(relation_lexicon, Mapping):
        raise RuntimeError("V5 candidate relation lexicon is missing")
    validate_relation_lexicon(relation_lexicon)
    if state.get("relation_lexicon_sha256") != relation_lexicon["lexicon_sha256"]:
        raise RuntimeError("V5 candidate relation lexicon hash differs")
    expected_ids, expected_biases, _ = sparse_token_biases(
        state["target_new_ids"],
        state["target_true_ids"],
        logit_bias=float(state["logit_bias"]),
    )
    if expected_ids != state["sidecar_token_ids"] or expected_biases != state[
        "sidecar_token_biases"
    ]:
        raise RuntimeError("V5 sparse token biases do not reproduce")
    if state.get("optimizer_constructed") is not False or int(
        state.get("gradient_updates_performed", -1)
    ) != 0:
        raise RuntimeError("V5 candidate unexpectedly records optimization")
    valid_hash = re.compile(r"^[0-9a-f]{64}$")
    if not valid_hash.fullmatch(str(state.get("base_embedding_sha256", ""))) or not valid_hash.fullmatch(
        str(state.get("base_lm_head_sha256", ""))
    ):
        raise RuntimeError("V5 candidate Base-model hashes are invalid")


def install_candidate(
    model: nn.Module,
    tokenizer: Any,
    state_or_path: Mapping[str, Any] | str | Path,
) -> ExactSubjectTargetRuntime:
    state = (
        torch.load(Path(state_or_path), map_location="cpu", weights_only=False)
        if isinstance(state_or_path, (str, Path))
        else dict(state_or_path)
    )
    if not isinstance(state, Mapping):
        raise RuntimeError("V5 candidate file is not a mapping")
    validate_candidate_state(state)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("model must expose input and output embeddings")
    if tensor_sha256(input_layer.weight) != state["base_embedding_sha256"]:
        raise RuntimeError("V5 candidate input embedding differs from Base binding")
    if tensor_sha256(output_layer.weight) != state["base_lm_head_sha256"]:
        raise RuntimeError("V5 candidate LM head differs from Base binding")
    reproduced_patterns = scoped.build_subject_patterns(tokenizer, state["subjects"])
    if reproduced_patterns != state["subject_patterns"]:
        raise RuntimeError("V5 candidate tokenizer subject patterns do not reproduce")
    reproduced_new_ids = [
        official_target_token_ids(
            tokenizer, value, llama_like=bool(state["llama_like"])
        )
        for value in state["target_new"]
    ]
    reproduced_true_ids = [
        official_target_token_ids(
            tokenizer, value, llama_like=bool(state["llama_like"])
        )
        for value in state["target_true"]
    ]
    if reproduced_new_ids != state["target_new_ids"] or reproduced_true_ids != state[
        "target_true_ids"
    ]:
        raise RuntimeError("V5 candidate target token IDs do not reproduce")
    router = BoundaryAwareSubjectRouter(
        input_layer,
        state["subject_patterns"],
        subjects=state["subjects"],
        relation_ids=state["relation_ids"],
        suffix_to_relation_ids=state["relation_lexicon"]["suffix_to_relation_ids"],
        tokenizer=tokenizer,
        model=model,
    )
    sidecar = ExactSubjectTargetLogitSidecar(
        output_layer,
        router,
        state["sidecar_token_ids"],
        state["sidecar_token_biases"],
    )
    return ExactSubjectTargetRuntime(
        model=model,
        router=router,
        sidecar=sidecar,
        metadata={
            "protocol": PROTOCOL,
            "seed": int(state["seed"]),
            "case_ids": [int(value) for value in state["case_ids"]],
            "subjects": [str(value) for value in state["subjects"]],
            "relation_ids": [str(value) for value in state["relation_ids"]],
            "claim_scope": state["architecture"]["claim_scope"],
        },
    )
