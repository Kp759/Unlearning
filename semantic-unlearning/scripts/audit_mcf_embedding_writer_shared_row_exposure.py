#!/usr/bin/env python3
"""Audit shared-row exposure for the frozen MCF embedding writer.

This is a development-only diagnostic. It reads the original MCF source solely
to materialize the locked forget direct prompts and first-half records that are
not in the reserved official-retain sample. Every materialized development ID
is written to the report and is permanently consumed development data.

The report establishes parameter-sharing exposure; it does not measure or
claim causal retain degradation. It also correlates the actual per-row edit
norm with development prompt frequency. An optional historical state can add
the analogous LM-head-row analysis, explicitly labeled as an input-token
frequency proxy rather than output-behavior exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoTokenizer


PROTOCOL = "mcf_embedding_writer_shared_row_exposure_development_v2"
LOCKED_SPLIT_PROTOCOL = "sure_mcf_target_aware_direct_only_v8"
V6_2_WRITER_PROTOCOL = "mcf_context_composed_sparse_embedding_writer_v6_2"
WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--writer-state", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--historical-lm-head-state",
        help=(
            "Optional historical embedding+LM-head state containing selected_output_rows "
            "and output_delta (or paired Base/edited output rows)."
        ),
    )
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_rewrite(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rewrite = record["requested_rewrite"]
    if isinstance(rewrite, list):
        if len(rewrite) != 1:
            raise ValueError("MCF requested_rewrite list must contain one row")
        rewrite = rewrite[0]
    if not isinstance(rewrite, Mapping):
        raise ValueError("MCF requested_rewrite must be an object")
    return rewrite


def materialize_prompt(record: Mapping[str, Any]) -> str:
    rewrite = normalize_rewrite(record)
    prompt = str(rewrite["prompt"])
    subject = str(rewrite["subject"]).strip()
    if "{}" in prompt:
        prompt = prompt.format(subject)
    return prompt.strip()


def subject(record: Mapping[str, Any]) -> str:
    return str(normalize_rewrite(record)["subject"]).strip()


def words(value: str) -> set[str]:
    return {match.group(0).lower() for match in WORD_PATTERN.finditer(value)}


def token_ids(tokenizer: Any, value: str) -> List[int]:
    encoded = tokenizer(value, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("unexpected batched tokenizer result")
        ids = ids[0]
    return [int(item) for item in ids]


def tensor_from_state(
    state: Mapping[str, Any],
    *,
    direct_keys: Sequence[str],
    edited_key: str,
    base_key: str,
) -> Tuple[torch.Tensor, str]:
    for key in direct_keys:
        value = state.get(key)
        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu(), key
    edited = state.get(edited_key)
    base = state.get(base_key)
    if isinstance(edited, torch.Tensor) and isinstance(base, torch.Tensor):
        return (edited.detach().float().cpu() - base.detach().float().cpu()), (
            f"{edited_key}-{base_key}"
        )
    raise ValueError(
        "state lacks a direct delta and paired Base/edited tensors for the requested rows"
    )


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and float(values[order[end]]) == float(
            values[order[cursor]]
        ):
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mean_x = sum(float(value) for value in x) / len(x)
    mean_y = sum(float(value) for value in y) / len(y)
    centered_x = [float(value) - mean_x for value in x]
    centered_y = [float(value) - mean_y for value in y]
    denom = math.sqrt(
        sum(value * value for value in centered_x)
        * sum(value * value for value in centered_y)
    )
    if denom == 0.0:
        return None
    return sum(a * b for a, b in zip(centered_x, centered_y)) / denom


def correlation_report(
    frequencies: Sequence[int], norms: Sequence[float]
) -> Dict[str, Any]:
    raw_frequency = [float(value) for value in frequencies]
    log_frequency = [math.log1p(value) for value in raw_frequency]
    raw_norm = [float(value) for value in norms]
    return {
        "rows": len(raw_norm),
        "pearson_frequency_vs_norm": pearson(raw_frequency, raw_norm),
        "pearson_log1p_frequency_vs_norm": pearson(log_frequency, raw_norm),
        "spearman_frequency_vs_norm": pearson(
            average_ranks(raw_frequency), average_ranks(raw_norm)
        ),
        "causal_interpretation": (
            "descriptive association only; row overlap and correlation do not establish "
            "retain-behavior degradation"
        ),
    }


def distribution(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("distribution requires at least one value")

    def quantile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]

    return {
        "min": ordered[0],
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def decoded_token(tokenizer: Any, token_id: int) -> Dict[str, Any]:
    converted = tokenizer.convert_ids_to_tokens(int(token_id))
    return {
        "token_id": int(token_id),
        "token": str(converted),
        "decoded": tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
    }


def row_analysis(
    *,
    tokenizer: Any,
    row_ids: Sequence[int],
    delta: torch.Tensor,
    prompt_frequency: Mapping[int, int],
    top_k: int,
    frequency_label: str,
) -> Dict[str, Any]:
    if delta.ndim != 2 or int(delta.shape[0]) != len(row_ids):
        raise ValueError("row delta shape does not match selected row IDs")
    norms = delta.norm(dim=1).tolist()
    rows: List[Dict[str, Any]] = []
    for position, (row_id, norm) in enumerate(zip(row_ids, norms)):
        frequency = int(prompt_frequency.get(int(row_id), 0))
        rows.append(
            {
                **decoded_token(tokenizer, int(row_id)),
                "selected_row_index": position,
                "development_prompt_frequency": frequency,
                "delta_l2_norm": float(norm),
                "frequency_times_delta_norm": frequency * float(norm),
            }
        )
    frequencies = [int(row["development_prompt_frequency"]) for row in rows]
    return {
        "selected_rows": len(rows),
        "rows_seen_in_development_prompts": sum(value > 0 for value in frequencies),
        "frequency_definition": frequency_label,
        "delta_norm": distribution(norms),
        "development_prompt_frequency": distribution(frequencies),
        "correlation": correlation_report(frequencies, norms),
        "top_by_delta_norm": sorted(
            rows, key=lambda row: (-row["delta_l2_norm"], row["token_id"])
        )[:top_k],
        "top_by_development_frequency": sorted(
            rows,
            key=lambda row: (-row["development_prompt_frequency"], row["token_id"]),
        )[:top_k],
        "top_by_frequency_times_delta_norm": sorted(
            rows,
            key=lambda row: (-row["frequency_times_delta_norm"], row["token_id"]),
        )[:top_k],
        "per_row": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.top_k) <= 0:
        raise ValueError("--top-k must be positive")
    mcf_path = Path(args.mcf_path).resolve()
    split_path = Path(args.split_manifest).resolve()
    writer_state_path = Path(args.writer_state).resolve()
    output_path = Path(args.output).resolve()
    for path in (mcf_path, split_path, writer_state_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    raw = load_json(mcf_path)
    split = load_json(split_path)
    if not isinstance(raw, list) or not all(isinstance(row, Mapping) for row in raw):
        raise ValueError("MCF source must be a list of records")
    sampling = split.get("sampling", {})
    if split.get("protocol") != LOCKED_SPLIT_PROTOCOL:
        raise ValueError("split manifest is not the locked MCF direct-only protocol")
    if split.get("source_sha256") != sha256_file(mcf_path):
        raise ValueError("split manifest is not bound to the supplied MCF source")
    forget_ids = [int(value) for value in sampling.get("forget_case_ids", [])]
    official_retain_ids = [
        int(value)
        for value in sampling.get(
            "retain_eval_case_ids", sampling.get("retain_case_ids", [])
        )
    ]
    if len(forget_ids) != int(sampling.get("forget_num", len(forget_ids))):
        raise ValueError("forget IDs do not match split counts")
    if len(official_retain_ids) != int(
        sampling.get("retain_eval_num", len(official_retain_ids))
    ):
        raise ValueError("official retain IDs do not match split counts")
    if set(forget_ids).intersection(official_retain_ids):
        raise ValueError("forget and reserved official-retain IDs overlap")
    first_half = len(raw) // 2
    if any(value < first_half for value in forget_ids):
        raise ValueError("forget IDs escaped the locked second-half pool")
    if any(value >= first_half for value in official_retain_ids):
        raise ValueError("official retain IDs escaped the locked first-half pool")
    development_ids = sorted(set(range(first_half)) - set(official_retain_ids))
    if set(development_ids).intersection(official_retain_ids):
        raise AssertionError("development materialization includes official retain IDs")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        clean_up_tokenization_spaces=False,
    )
    forget_records = [raw[index] for index in forget_ids]
    development_records = [raw[index] for index in development_ids]
    forget_subjects = [subject(record) for record in forget_records]
    development_prompts = [materialize_prompt(record) for record in development_records]

    forget_subject_words = [words(value) for value in forget_subjects]
    development_word_sets = [words(value) for value in development_prompts]
    all_forget_words = set().union(*forget_subject_words)
    word_prompt_frequency: Counter[str] = Counter()
    for prompt_words in development_word_sets:
        word_prompt_frequency.update(prompt_words.intersection(all_forget_words))
    reused_words = {value for value in all_forget_words if word_prompt_frequency[value]}

    forget_subject_tokens = [
        set(token_ids(tokenizer, value)) for value in forget_subjects
    ]
    all_forget_tokens = set().union(*forget_subject_tokens)
    development_token_sets = [
        set(token_ids(tokenizer, prompt)) for prompt in development_prompts
    ]
    token_prompt_frequency: Counter[int] = Counter()
    for prompt_tokens in development_token_sets:
        token_prompt_frequency.update(prompt_tokens)
    reused_forget_tokens = {
        value for value in all_forget_tokens if token_prompt_frequency[value] > 0
    }

    writer_state = torch.load(writer_state_path, map_location="cpu", weights_only=False)
    if not isinstance(writer_state, Mapping):
        raise ValueError("writer state must be a mapping")
    if writer_state.get("protocol") != V6_2_WRITER_PROTOCOL:
        raise ValueError("writer state is not the registered clean V6.2 writer")
    if [int(value) for value in writer_state.get("case_ids", [])] != forget_ids:
        raise ValueError("writer state case IDs do not match the locked forget split")
    if int(writer_state.get("seed", -1)) != int(split.get("seed", -2)):
        raise ValueError("writer state seed does not match the locked split")
    selected_embedding_rows = [
        int(value) for value in writer_state.get("selected_embedding_rows", [])
    ]
    if not selected_embedding_rows:
        raise ValueError("writer selected_embedding_rows is empty")
    if len(selected_embedding_rows) != len(set(selected_embedding_rows)):
        raise ValueError("writer selected_embedding_rows contains duplicates")
    embedding_delta, embedding_delta_source = tensor_from_state(
        writer_state,
        direct_keys=("actual_embedding_delta", "embedding_delta"),
        edited_key="edited_selected_embedding_rows",
        base_key="base_selected_embedding_rows",
    )
    if int(embedding_delta.shape[0]) != len(selected_embedding_rows):
        raise ValueError("writer embedding delta does not match selected row IDs")
    selected_embedding_set = set(selected_embedding_rows)
    reused_selected_rows = {
        value for value in selected_embedding_set if token_prompt_frequency[value] > 0
    }

    subject_has_reused_actual_row = [
        bool(tokens.intersection(reused_selected_rows))
        for tokens in forget_subject_tokens
    ]
    development_with_forget_word = sum(
        bool(prompt_words.intersection(all_forget_words))
        for prompt_words in development_word_sets
    )
    development_with_forget_token = sum(
        bool(prompt_tokens.intersection(all_forget_tokens))
        for prompt_tokens in development_token_sets
    )
    development_with_selected_row = sum(
        bool(prompt_tokens.intersection(selected_embedding_set))
        for prompt_tokens in development_token_sets
    )

    embedding_analysis = row_analysis(
        tokenizer=tokenizer,
        row_ids=selected_embedding_rows,
        delta=embedding_delta,
        prompt_frequency=token_prompt_frequency,
        top_k=int(args.top_k),
        frequency_label=(
            "number of consumed development-retain direct prompts containing the token ID"
        ),
    )

    lm_head_analysis: Dict[str, Any] | None = None
    historical_path = None
    if args.historical_lm_head_state:
        historical_path = Path(args.historical_lm_head_state).resolve()
        historical_state = torch.load(
            historical_path, map_location="cpu", weights_only=False
        )
        if not isinstance(historical_state, Mapping):
            raise ValueError("historical LM-head state must be a mapping")
        selected_output_rows = [
            int(value) for value in historical_state.get("selected_output_rows", [])
        ]
        if not selected_output_rows or len(selected_output_rows) != len(
            set(selected_output_rows)
        ):
            raise ValueError("historical selected_output_rows is empty or duplicated")
        output_delta, output_delta_source = tensor_from_state(
            historical_state,
            direct_keys=("actual_output_delta", "output_delta"),
            edited_key="edited_selected_output_rows",
            base_key="base_selected_output_rows",
        )
        lm_head_analysis = {
            "delta_source": output_delta_source,
            "frequency_is_behavioral_exposure_measure": False,
            "interpretation": (
                "prompt occurrence of an output-row token is only a lexical proxy; "
                "LM-head rows affect the output distribution at every prediction position"
            ),
            **row_analysis(
                tokenizer=tokenizer,
                row_ids=selected_output_rows,
                delta=output_delta,
                prompt_frequency=token_prompt_frequency,
                top_k=int(args.top_k),
                frequency_label=(
                    "input-token occurrence proxy in consumed development-retain prompts"
                ),
            ),
        }

    report = {
        "schema_version": 2,
        "kind": "mcf_embedding_writer_shared_row_exposure_development_audit",
        "protocol": PROTOCOL,
        "data_role": "consumed_development_evidence_not_blind_evaluation",
        "sources": {
            "model_path": str(Path(args.model_path).resolve()),
            "mcf_path": str(mcf_path),
            "mcf_sha256": sha256_file(mcf_path),
            "split_manifest": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "writer_state": str(writer_state_path),
            "writer_state_sha256": sha256_file(writer_state_path),
            "writer_protocol": writer_state.get("protocol"),
            "historical_lm_head_state": str(historical_path)
            if historical_path
            else None,
            "historical_lm_head_state_sha256": (
                sha256_file(historical_path) if historical_path else None
            ),
        },
        "data_firewall": {
            "mcf_total_records": len(raw),
            "forget_records": len(forget_ids),
            "reserved_official_retain_records_excluded": len(official_retain_ids),
            "reserved_official_retain_case_ids_sha256": sha256_json(
                official_retain_ids
            ),
            "reserved_official_retain_prompts_materialized": 0,
            "development_retain_records_consumed": len(development_ids),
            "development_retain_case_ids": development_ids,
            "development_retain_case_ids_sha256": sha256_json(development_ids),
            "development_ids_must_never_be_described_as_blind_evaluation": True,
            "used_by_v3_5_selection_optimization_acceptance_or_retry": False,
            "official_evaluation_prompts_seen_by_learner": 0,
        },
        "subject_word_overlap": {
            "forget_records_with_overlap": sum(
                bool(values.intersection(reused_words))
                for values in forget_subject_words
            ),
            "forget_records": len(forget_records),
            "unique_forget_subject_words": len(all_forget_words),
            "unique_reused_forget_subject_words": len(reused_words),
            "development_prompts_with_overlap": development_with_forget_word,
            "development_prompts": len(development_prompts),
            "top_reused_words": [
                {"word": value, "development_prompt_frequency": int(count)}
                for value, count in word_prompt_frequency.most_common(int(args.top_k))
            ],
        },
        "subject_subtoken_overlap": {
            "forget_records_with_overlap": sum(
                bool(values.intersection(reused_forget_tokens))
                for values in forget_subject_tokens
            ),
            "forget_records": len(forget_records),
            "unique_forget_subject_token_ids": len(all_forget_tokens),
            "unique_reused_forget_subject_token_ids": len(reused_forget_tokens),
            "development_prompts_with_overlap": development_with_forget_token,
            "development_prompts": len(development_prompts),
        },
        "actual_embedding_row_overlap": {
            "checkpoint_row_id_path": "root.selected_embedding_rows",
            "embedding_delta_source": embedding_delta_source,
            "selected_rows": len(selected_embedding_rows),
            "selected_rows_reused": len(reused_selected_rows),
            "development_prompts_with_selected_row": development_with_selected_row,
            "development_prompts": len(development_prompts),
            "forget_subjects_with_reused_selected_row": sum(
                subject_has_reused_actual_row
            ),
            "forget_subjects": len(forget_subjects),
        },
        "embedding_row_norm_frequency_analysis": embedding_analysis,
        "historical_lm_head_row_norm_frequency_analysis": lm_head_analysis,
        "interpretation": {
            "established": "substantial shared-parameter exposure for token-row edits",
            "not_established": "that lexical overlap itself causes retain degradation",
            "architecture_implication": (
                "use sparse embedding edits as markers and condition the behavioral "
                "actuator on downstream context-composed hidden state"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "development_retain_records_consumed": len(development_ids),
                "reserved_official_retain_records_excluded": len(official_retain_ids),
                "actual_edited_embedding_rows_reused": len(reused_selected_rows),
                "actual_edited_embedding_rows": len(selected_embedding_rows),
                "development_prompts_with_actual_edited_row": development_with_selected_row,
                "official_evaluation_prompts_seen_by_learner": 0,
            },
            indent=2,
        )
    )
    print(
        "IMPORTANT: every development_retain_case_id in this report is consumed "
        "development data and must never be described as blind evaluation."
    )


if __name__ == "__main__":
    main()
