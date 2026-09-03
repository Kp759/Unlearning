#!/usr/bin/env python3
"""Exact-subject ZsRE routing with normalization-preserving suppression.

ZsRE's official rewrite and paraphrase probes retain the complete subject,
while its locality probes are unrelated questions.  This module makes that
benchmark contract explicit: a frozen, parameter-free router opens on an
exact registered subject token sequence and a logit sidecar suppresses the
registered original-answer tokens by permuting existing logits only.

The permutation preserves the complete vocabulary-logit multiset at every
open position.  A closed route returns the exact Base LM-head tensor object.
No embedding, Transformer, or LM-head parameter is mutated.  The resulting
claim is contextual behavioral suppression, not latent factual erasure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import nn

import mcf_normalization_preserving_sidecar_v6_core as shared
import scoped_span_edit as scoped
import zsre_zero_unlearn_official_eval as official


PROTOCOL = "zsre_exact_subject_normalization_preserving_sidecar_v6_0"
SCHEMA_VERSION = 1
KIND = "zsre_exact_subject_normalization_preserving_sidecar_candidate"
CLAIM_SCOPE = "contextual_behavioral_suppression_not_latent_erasure"


def architecture() -> Dict[str, Any]:
    return {
        "route": "boundary_complete_exact_subject_token_sequence",
        "route_parameters": 0,
        "intervention": (
            "causal_directional_original_answer_logit_permutation_with_"
            "reserved_special_slots"
        ),
        "normalization_preserving_by_construction": True,
        "ordinary_non_target_logits_unchanged_by_construction": True,
        "arithmetic_logit_offsets": 0,
        "embedding_parameters_mutated": False,
        "transformer_parameters_mutated": False,
        "lm_head_parameters_mutated": False,
        "learned_detector_parameters": 0,
        "learned_actuator_parameters": 0,
        "closed_route_returns_base_tensor_object": True,
        "claim_scope": CLAIM_SCOPE,
    }


def _normalized_patterns(
    values: Sequence[Sequence[Sequence[int]]],
) -> list[list[list[int]]]:
    return [
        [[int(token) for token in pattern] for pattern in record_patterns]
        for record_patterns in values
    ]


def build_candidate_state(
    *,
    seed: int,
    case_ids: Sequence[int],
    subjects: Sequence[str],
    direct_prompts: Sequence[str],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    target_true: Sequence[str],
    target_true_ids: Sequence[Sequence[int]],
    neutral_target: str,
    neutral_target_id: int,
    reserved_token_ids: Sequence[int],
    reserved_token_strings: Sequence[str],
    llama_like: bool,
    base_embedding_sha256: str,
    base_lm_head_sha256: str,
    source_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    records = len(case_ids)
    aligned = all(
        len(values) == records
        for values in (
            subjects,
            direct_prompts,
            subject_patterns,
            target_true,
            target_true_ids,
        )
    )
    if int(seed) != 1:
        raise ValueError("ZsRE V6 development is frozen to consumed seed 1")
    if records != 50 or not aligned:
        raise ValueError("ZsRE V6 requires exactly 50 aligned forget records")
    if len(set(int(item) for item in case_ids)) != records:
        raise ValueError("ZsRE V6 case IDs must be unique")
    if len(set(str(item) for item in subjects)) != records:
        raise ValueError(
            "ZsRE V6 refuses duplicate subjects because record ownership would be ambiguous"
        )
    patterns = _normalized_patterns(subject_patterns)
    if any(not row or any(not pattern for pattern in row) for row in patterns):
        raise ValueError("every ZsRE V6 subject needs a non-empty token pattern")
    true_ids = [[int(item) for item in row] for row in target_true_ids]
    if any(not row or min(row) < 0 for row in true_ids):
        raise ValueError("every ZsRE V6 original answer needs evaluated token IDs")
    promoted, sensitive, directional_report = shared.directional_target_token_banks(
        [[int(neutral_target_id)] for _ in range(records)],
        true_ids,
    )
    reserved_ids = [int(item) for item in reserved_token_ids]
    reserved_strings = [str(item) for item in reserved_token_strings]
    excluded = {
        int(neutral_target_id),
        *(item for row in true_ids for item in row),
    }
    if (
        len(reserved_ids) != 128
        or len(reserved_ids) != len(reserved_strings)
        or len(set(reserved_ids)) != len(reserved_ids)
        or set(reserved_ids) & excluded
        or any(
            shared.RESERVED_TOKEN_PATTERN.fullmatch(item) is None
            for item in reserved_strings
        )
    ):
        raise ValueError("ZsRE V6 reserved-token reservoir is invalid")
    digest = re.compile(r"^[0-9a-f]{64}$")
    if not digest.fullmatch(str(base_embedding_sha256)) or not digest.fullmatch(
        str(base_lm_head_sha256)
    ):
        raise ValueError("ZsRE V6 requires Base embedding and LM-head hashes")
    if set(source_hashes) != {
        "training_visible",
        "split_manifest",
        "development_registry",
    } or any(not digest.fullmatch(str(value)) for value in source_hashes.values()):
        raise ValueError("ZsRE V6 source bindings must be SHA-256 digests")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "protocol": PROTOCOL,
        "seed": 1,
        "architecture": architecture(),
        "case_ids": [int(item) for item in case_ids],
        "subjects": [str(item) for item in subjects],
        "direct_prompts": [str(item) for item in direct_prompts],
        "subject_patterns": patterns,
        "target_true": [str(item) for item in target_true],
        "target_true_ids": true_ids,
        "neutral_target": str(neutral_target),
        "neutral_target_id": int(neutral_target_id),
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
        "evaluation_prompts_seen": 0,
    }


def validate_candidate_state(state: Mapping[str, Any]) -> None:
    if (
        int(state.get("schema_version", -1)) != SCHEMA_VERSION
        or state.get("kind") != KIND
        or state.get("protocol") != PROTOCOL
        or int(state.get("seed", -1)) != 1
        or state.get("architecture") != architecture()
    ):
        raise RuntimeError("unsupported ZsRE V6 candidate state")
    records = len(state.get("case_ids", []))
    aligned = (
        "subjects",
        "direct_prompts",
        "subject_patterns",
        "target_true",
        "target_true_ids",
        "promoted_token_ids",
        "sensitive_token_ids",
    )
    if records != 50 or any(len(state.get(key, [])) != records for key in aligned):
        raise RuntimeError("ZsRE V6 candidate record arrays are not aligned")
    if len(set(int(item) for item in state["case_ids"])) != records or len(
        set(str(item) for item in state["subjects"])
    ) != records:
        raise RuntimeError("ZsRE V6 candidate ownership is ambiguous")
    if _normalized_patterns(state["subject_patterns"]) != state["subject_patterns"]:
        raise RuntimeError("ZsRE V6 subject patterns are malformed")
    promoted, sensitive, _report = shared.directional_target_token_banks(
        [[int(state["neutral_target_id"])] for _ in range(records)],
        state["target_true_ids"],
    )
    if promoted != state["promoted_token_ids"] or sensitive != state[
        "sensitive_token_ids"
    ]:
        raise RuntimeError("ZsRE V6 directional token banks do not reproduce")
    reserved_ids = [int(item) for item in state.get("reserved_token_ids", [])]
    reserved_strings = [str(item) for item in state.get("reserved_token_strings", [])]
    target_ids = {
        int(state["neutral_target_id"]),
        *(int(item) for row in state["target_true_ids"] for item in row),
    }
    if (
        len(reserved_ids) != 128
        or len(reserved_ids) != len(reserved_strings)
        or len(set(reserved_ids)) != len(reserved_ids)
        or set(reserved_ids) & target_ids
        or any(
            shared.RESERVED_TOKEN_PATTERN.fullmatch(item) is None
            for item in reserved_strings
        )
    ):
        raise RuntimeError("ZsRE V6 reserved-token reservoir is invalid")
    digest = re.compile(r"^[0-9a-f]{64}$")
    if not digest.fullmatch(str(state.get("base_embedding_sha256", ""))) or not (
        digest.fullmatch(str(state.get("base_lm_head_sha256", "")))
    ):
        raise RuntimeError("ZsRE V6 Base parameter bindings are invalid")
    if set(state.get("source_hashes", {})) != {
        "training_visible",
        "split_manifest",
        "development_registry",
    } or any(
        not digest.fullmatch(str(value))
        for value in state.get("source_hashes", {}).values()
    ):
        raise RuntimeError("ZsRE V6 source hash binding is invalid")
    if (
        state.get("optimizer_constructed") is not False
        or int(state.get("gradient_updates_performed", -1)) != 0
        or int(state.get("evaluation_prompts_seen", -1)) != 0
    ):
        raise RuntimeError("ZsRE V6 candidate crossed its process firewall")


@dataclass
class Runtime:
    model: nn.Module
    router: scoped.SpanGateRouter
    sidecar: shared.NormalizationPreservingSensitiveTokenSidecar
    metadata: Dict[str, Any]

    def close(self) -> None:
        self.sidecar.close()
        self.router.close()


def install_candidate(
    model: nn.Module,
    tokenizer: Any,
    state_or_path: Mapping[str, Any] | str | Path,
) -> Runtime:
    state = (
        torch.load(Path(state_or_path), map_location="cpu", weights_only=False)
        if isinstance(state_or_path, (str, Path))
        else dict(state_or_path)
    )
    if not isinstance(state, Mapping):
        raise RuntimeError("ZsRE V6 candidate file is not a mapping")
    validate_candidate_state(state)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("model must expose input and output embeddings")
    if shared.tensor_sha256(input_layer.weight) != state["base_embedding_sha256"]:
        raise RuntimeError("ZsRE V6 input embedding differs from its Base binding")
    if shared.tensor_sha256(output_layer.weight) != state["base_lm_head_sha256"]:
        raise RuntimeError("ZsRE V6 LM head differs from its Base binding")
    reproduced_patterns = scoped.build_subject_patterns(tokenizer, state["subjects"])
    if reproduced_patterns != state["subject_patterns"]:
        raise RuntimeError("ZsRE V6 subject token patterns do not reproduce")
    reproduced_true = [
        official.original_answer_token_ids(
            tokenizer,
            answer,
            llama_like=bool(state["llama_like"]),
        )
        for answer in state["target_true"]
    ]
    if reproduced_true != state["target_true_ids"]:
        raise RuntimeError("ZsRE V6 original-answer token IDs do not reproduce")
    neutral_id = official.resolve_neutral_target_token_id(
        tokenizer, str(state["neutral_target"])
    )
    if int(neutral_id) != int(state["neutral_target_id"]):
        raise RuntimeError("ZsRE V6 neutral target token does not reproduce")
    reproduced_reserved = tokenizer.convert_ids_to_tokens(state["reserved_token_ids"])
    if isinstance(reproduced_reserved, str):
        reproduced_reserved = [reproduced_reserved]
    if [str(item) for item in reproduced_reserved] != state[
        "reserved_token_strings"
    ]:
        raise RuntimeError("ZsRE V6 reserved special tokens do not reproduce")
    router = scoped.SpanGateRouter(
        input_layer,
        state["subject_patterns"],
        subjects=state["subjects"],
        model=model,
    )
    sidecar = shared.NormalizationPreservingSensitiveTokenSidecar(
        output_layer,
        router,
        state["promoted_token_ids"],
        state["sensitive_token_ids"],
        state["reserved_token_ids"],
    )
    return Runtime(
        model=model,
        router=router,
        sidecar=sidecar,
        metadata={
            "protocol": PROTOCOL,
            "seed": 1,
            "case_ids": [int(item) for item in state["case_ids"]],
            "subjects": [str(item) for item in state["subjects"]],
            "claim_scope": CLAIM_SCOPE,
        },
    )
