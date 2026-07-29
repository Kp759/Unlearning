#!/usr/bin/env python3
"""Evaluate LLM1 with token metrics and an independent LLM2 judge.

Development bundles can only use Judge A. Final-test bundles require a frozen
selection receipt, an independent Judge B, and are one-shot by default. The
judge never replaces model-native evidence: answer probabilities, comparative
preference, retain utility, and locality are always retained in the output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from controlled_llm_judge import (
    FORGET_LABELS,
    UTILITY_LABELS,
    JudgeClient,
    assert_independent_judges,
    assert_judge_phase,
    load_judge_config,
    load_recorded_judgments,
    public_judge_config,
)
from controlled_unlearning_protocol import (
    PromptCase,
    bundle_prompt_cases,
    load_development_bundle,
    load_test_bundle,
    mean,
    read_json,
    sha256_file,
    sha256_json,
    safe_ratio,
    stable_stratified_manual_audit_sample,
    write_json,
)


RESULT_SCHEMA_VERSION = 1


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                + "\n"
            )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(
                    f"{path} line {line_number} is not a JSON object"
                )
            rows.append(dict(value))
    return rows


def normalize_answer(value: Any) -> str:
    text = str(value).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def answer_mentioned(response: str, answers: Sequence[str]) -> bool:
    normalized_response = f" {normalize_answer(response)} "
    return any(
        f" {normalize_answer(answer)} " in normalized_response
        for answer in answers
        if normalize_answer(answer)
    )


def _model_identity(path: Path, candidate_id: str) -> Dict[str, Any]:
    path = Path(path)
    identity: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "path": str(path.resolve()),
    }
    if not path.exists():
        identity["local_path_exists"] = False
        identity["identity_sha256"] = sha256_json(identity)
        return identity
    identity["local_path_exists"] = True
    if path.is_file():
        identity["file_size"] = path.stat().st_size
        identity["file_sha256"] = sha256_file(path)
    else:
        tracked_names = (
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "repair_summary.json",
            "config_used.json",
        )
        metadata: Dict[str, Any] = {}
        for name in tracked_names:
            candidate = path / name
            if candidate.exists() and candidate.is_file():
                metadata[name] = {
                    "size": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
        weight_files = sorted(
            [
                *path.glob("*.safetensors"),
                *path.glob("pytorch_model*.bin"),
            ]
        )
        identity["metadata_files"] = metadata
        identity["weight_files"] = [
            {"name": item.name, "size": item.stat().st_size}
            for item in weight_files
        ]
    unhashed = dict(identity)
    identity["identity_sha256"] = sha256_json(unhashed)
    return identity


def _resolve_dtype(name: str) -> Any:
    import torch

    values = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return values[name]


def _first_device(model: Any) -> Any:
    return next(model.parameters()).device


def _prepare_prompt(tokenizer: Any, prompt: str, use_chat_template: bool) -> str:
    if _chat_template_active(tokenizer, use_chat_template):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _chat_template_active(tokenizer: Any, requested: bool) -> bool:
    return bool(requested and getattr(tokenizer, "chat_template", None))


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    kwargs: Dict[str, Any] = {
        "torch_dtype": _resolve_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if args.device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if args.device_map == "single":
        device = torch.device(
            args.device
            if args.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(device)
    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def generate_responses(
    model: Any,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    *,
    batch_size: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    import torch

    rows: List[Dict[str, Any]] = []
    device = _first_device(model)
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        prepared = [
            _prepare_prompt(tokenizer, case.prompt, use_chat_template)
            for case in batch
        ]
        encoded = tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=not _chat_template_active(
                tokenizer,
                use_chat_template,
            ),
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        input_width = int(encoded["input_ids"].shape[1])
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for case, token_ids in zip(batch, generated):
            continuation = token_ids[input_width:]
            response = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            ).strip()
            rows.append(
                {
                    "case_id": case.case_id,
                    "response": response,
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": max_new_tokens,
                    },
                }
            )
    return rows


def _flat_token_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = False,
) -> List[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    values = encoded["input_ids"]
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _answer_completion_variants(
    case: PromptCase,
    answer: str,
) -> List[str]:
    """Return response-form candidates, including MC letters when relevant."""

    variants = [str(answer).strip()]
    if case.style == "multiple_choice" and isinstance(case.metadata, Mapping):
        options = case.metadata.get("options", [])
        if isinstance(options, list):
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for index, option in enumerate(options):
                if (
                    index < len(letters)
                    and normalize_answer(option) == normalize_answer(answer)
                ):
                    variants.append(letters[index])
    result: List[str] = []
    seen: set[str] = set()
    for value in variants:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _prompt_answer_tokenization(
    tokenizer: Any,
    prepared_prompt: str,
    answer_completion: str,
    *,
    use_chat_template: bool,
) -> Tuple[List[int], int, int, str]:
    """Tokenize the full prompt+answer so boundary tokens are scored exactly."""

    add_special_tokens = not _chat_template_active(
        tokenizer,
        use_chat_template,
    )
    prompt_ids = _flat_token_ids(
        tokenizer,
        prepared_prompt,
        add_special_tokens=add_special_tokens,
    )
    bos = tokenizer.bos_token_id
    if not prompt_ids and bos is not None:
        prompt_ids = [int(bos)]
    if not prompt_ids:
        raise ValueError("Prompt tokenized to an empty sequence")
    separator = (
        ""
        if prepared_prompt.endswith((" ", "\n", "\t"))
        else " "
    )
    completion_text = separator + answer_completion.strip()
    full_ids = _flat_token_ids(
        tokenizer,
        prepared_prompt + completion_text,
        add_special_tokens=add_special_tokens,
    )
    if (
        len(full_ids) > len(prompt_ids)
        and full_ids[: len(prompt_ids)] == prompt_ids
    ):
        answer_length = len(full_ids) - len(prompt_ids)
        return (
            full_ids,
            len(prompt_ids),
            answer_length,
            "full_sequence_suffix",
        )

    # A small number of slow/custom tokenizers alter the last prompt token at
    # a text boundary. Preserve compatibility but disclose the fallback.
    answer_ids = _flat_token_ids(
        tokenizer,
        completion_text,
        add_special_tokens=False,
    )
    if not answer_ids:
        answer_ids = _flat_token_ids(
            tokenizer,
            answer_completion.strip(),
            add_special_tokens=False,
        )
    if not answer_ids:
        raise ValueError(
            f"Answer {answer_completion!r} tokenized to no tokens"
        )
    return (
        [*prompt_ids, *answer_ids],
        len(prompt_ids),
        len(answer_ids),
        "separate_suffix_fallback",
    )


def score_answer_candidates(
    model: Any,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    *,
    batch_size: int,
    use_chat_template: bool,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Return length-normalized sequence probabilities for every answer."""

    import torch
    import torch.nn.functional as F

    pair_rows: List[Dict[str, Any]] = []
    for case in cases:
        prepared = _prepare_prompt(
            tokenizer,
            case.prompt,
            use_chat_template,
        )
        for answer_kind, answers in (
            ("sensitive", case.sensitive_answers),
            ("acceptable", case.acceptable_answers),
        ):
            for answer in answers:
                for completion in _answer_completion_variants(case, answer):
                    (
                        full_ids,
                        prompt_length,
                        answer_length,
                        tokenization_method,
                    ) = _prompt_answer_tokenization(
                        tokenizer,
                        prepared,
                        completion,
                        use_chat_template=use_chat_template,
                    )
                    pair_rows.append(
                        {
                            "case_id": case.case_id,
                            "answer_kind": answer_kind,
                            "answer": answer,
                            "scored_completion": completion,
                            "input_ids": full_ids,
                            "prompt_length": prompt_length,
                            "answer_length": answer_length,
                            "tokenization_method": tokenization_method,
                        }
                    )
    output: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        case.case_id: {"sensitive": [], "acceptable": []}
        for case in cases
    }
    if not pair_rows:
        return output
    device = _first_device(model)
    pad_id = int(tokenizer.pad_token_id)
    for start in range(0, len(pair_rows), batch_size):
        batch = pair_rows[start : start + batch_size]
        width = max(len(row["input_ids"]) for row in batch)
        input_ids = torch.full(
            (len(batch), width),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        attention = torch.zeros_like(input_ids)
        for row_index, row in enumerate(batch):
            values = torch.tensor(
                row["input_ids"],
                dtype=torch.long,
                device=device,
            )
            input_ids[row_index, : len(values)] = values
            attention[row_index, : len(values)] = 1
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
            ).logits
            log_probs = F.log_softmax(logits.float(), dim=-1)
        for row_index, row in enumerate(batch):
            prompt_length = int(row["prompt_length"])
            answer_length = int(row["answer_length"])
            target_positions = torch.arange(
                prompt_length,
                prompt_length + answer_length,
                device=device,
            )
            logit_positions = target_positions - 1
            target_ids = input_ids[row_index, target_positions]
            token_log_probs = log_probs[
                row_index,
                logit_positions,
                target_ids,
            ]
            total_log_probability = float(
                token_log_probs.sum().detach().cpu()
            )
            mean_nll = float(
                (-token_log_probs.mean()).detach().cpu()
            )
            geometric_probability = float(
                math.exp(max(-80.0, -mean_nll))
            )
            output[row["case_id"]][row["answer_kind"]].append(
                {
                    "answer": row["answer"],
                    "scored_completion": row["scored_completion"],
                    "token_count": answer_length,
                    "mean_nll": mean_nll,
                    "total_log_probability": total_log_probability,
                    "geometric_mean_probability": geometric_probability,
                    "tokenization_method": row["tokenization_method"],
                }
            )
    return output


def load_responses(
    path: Path,
    cases: Sequence[PromptCase],
) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or "response" not in row:
            raise ValueError("Every response row needs case_id and response")
        if case_id in by_id:
            raise ValueError(f"Duplicate response for {case_id}")
        by_id[case_id] = row
    required = {case.case_id for case in cases}
    missing = sorted(required - set(by_id))
    extra = sorted(set(by_id) - required)
    if missing:
        raise ValueError(f"Responses miss {len(missing)} cases: {missing[:5]}")
    if extra:
        raise ValueError(f"Responses contain unknown cases: {extra[:5]}")
    return [by_id[case.case_id] for case in cases]


def _max_probability(rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    values = [
        float(row["geometric_mean_probability"])
        for row in rows
    ]
    return max(values) if values else None


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def summarize_subset(values: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        sensitive = [
            float(row["max_sensitive_probability"])
            for row in values
            if row.get("max_sensitive_probability") is not None
        ]
        acceptable = [
            float(row["max_acceptable_probability"])
            for row in values
            if row.get("max_acceptable_probability") is not None
        ]
        comparative = [
            bool(row["sensitive_preferred"])
            for row in values
            if row.get("sensitive_preferred") is not None
        ]
        by_record: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_record[str(row["source_record_id"])].append(row)
        record_pass_rates = [
            sum(
                1.0 if row["judgment"]["passed"] else 0.0
                for row in record_rows
            )
            / len(record_rows)
            for record_rows in by_record.values()
        ]
        return {
            "count": len(values),
            "record_count": len(by_record),
            "judge_pass_rate": mean(
                [1.0 if row["judgment"]["passed"] else 0.0 for row in values]
            ),
            "record_macro_judge_pass_rate": mean(record_pass_rates),
            "record_strict_all_prompts_pass_rate": mean(
                [
                    1.0 if math.isclose(rate, 1.0) else 0.0
                    for rate in record_pass_rates
                ]
            ),
            "record_any_failure_rate": mean(
                [
                    0.0 if math.isclose(rate, 1.0) else 1.0
                    for rate in record_pass_rates
                ]
            ),
            "judge_mean_confidence": mean(
                [float(row["judgment"]["confidence"]) for row in values]
            ),
            "lexical_sensitive_mention_rate": mean(
                [
                    1.0 if row["lexical_sensitive_mention"] else 0.0
                    for row in values
                ]
            ),
            "mean_max_sensitive_probability": mean(sensitive),
            "mean_max_acceptable_probability": mean(acceptable),
            "sensitive_preference_rate": mean(
                [1.0 if value else 0.0 for value in comparative]
            ),
        }

    by_behavior: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_purpose: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_style: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_behavior_and_style: Dict[
        str,
        Dict[str, List[Mapping[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        behavior = str(row["expected_behavior"])
        style = str(row["style"])
        by_behavior[behavior].append(row)
        by_purpose[str(row["purpose"])].append(row)
        by_style[style].append(row)
        by_behavior_and_style[behavior][style].append(row)
    return {
        "overall": summarize_subset(rows),
        "by_behavior": {
            key: summarize_subset(values)
            for key, values in sorted(by_behavior.items())
        },
        "by_purpose": {
            key: summarize_subset(values)
            for key, values in sorted(by_purpose.items())
        },
        "by_style": {
            key: summarize_subset(values)
            for key, values in sorted(by_style.items())
        },
        "by_behavior_and_style": {
            behavior: {
                style: summarize_subset(style_rows)
                for style, style_rows in sorted(styles.items())
            }
            for behavior, styles in sorted(
                by_behavior_and_style.items()
            )
        },
    }


def _behavior_metric(
    metrics: Mapping[str, Any],
    behavior: str,
    field: str,
) -> Optional[float]:
    value = (
        metrics.get("by_behavior", {})
        .get(behavior, {})
        .get(field)
    )
    return None if value is None else float(value)


def _weighted_utility_metric(
    metrics: Mapping[str, Any],
    field: str,
) -> Optional[float]:
    numerator = 0.0
    denominator = 0
    for behavior in ("answer_correctly", "preserve_locality"):
        block = metrics.get("by_behavior", {}).get(behavior, {})
        count = int(block.get("count", 0) or 0)
        value = block.get(field)
        if count and value is not None:
            numerator += count * float(value)
            denominator += count
    return numerator / denominator if denominator else None


def build_final_utility_guardrail(
    candidate_metrics: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare locked-test utility against Base using frozen tolerances."""

    utility_tolerance = float(selection_receipt["utility_tolerance"])
    locality_tolerance = float(selection_receipt["locality_tolerance"])
    minimum_probability_ratio = float(
        selection_receipt["min_utility_probability_ratio"]
    )
    candidate_utility = _weighted_utility_metric(
        candidate_metrics,
        "judge_pass_rate",
    )
    baseline_utility = _weighted_utility_metric(
        baseline_metrics,
        "judge_pass_rate",
    )
    candidate_probability = _weighted_utility_metric(
        candidate_metrics,
        "mean_max_acceptable_probability",
    )
    baseline_probability = _weighted_utility_metric(
        baseline_metrics,
        "mean_max_acceptable_probability",
    )
    candidate_locality = _behavior_metric(
        candidate_metrics,
        "preserve_locality",
        "judge_pass_rate",
    )
    baseline_locality = _behavior_metric(
        baseline_metrics,
        "preserve_locality",
        "judge_pass_rate",
    )
    required = {
        "candidate_utility_judge_pass_rate": candidate_utility,
        "baseline_utility_judge_pass_rate": baseline_utility,
        "candidate_utility_answer_probability": candidate_probability,
        "baseline_utility_answer_probability": baseline_probability,
        "candidate_locality_judge_pass_rate": candidate_locality,
        "baseline_locality_judge_pass_rate": baseline_locality,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Final utility comparison lacks required metrics: "
            f"{missing}"
        )
    assert candidate_utility is not None
    assert baseline_utility is not None
    assert candidate_probability is not None
    assert baseline_probability is not None
    assert candidate_locality is not None
    assert baseline_locality is not None
    if baseline_probability <= 0.0:
        raise ValueError(
            "Base acceptable-answer probability must be positive"
        )
    utility_drop = baseline_utility - candidate_utility
    locality_drop = baseline_locality - candidate_locality
    probability_ratio = safe_ratio(
        candidate_probability,
        baseline_probability,
    )
    assert probability_ratio is not None
    utility_passed = utility_drop <= utility_tolerance
    locality_passed = locality_drop <= locality_tolerance
    probability_passed = probability_ratio >= minimum_probability_ratio
    return {
        **required,
        "utility_absolute_drop": utility_drop,
        "utility_tolerance": utility_tolerance,
        "utility_passed": utility_passed,
        "locality_absolute_drop": locality_drop,
        "locality_tolerance": locality_tolerance,
        "locality_passed": locality_passed,
        "utility_answer_probability_ratio": probability_ratio,
        "minimum_utility_answer_probability_ratio": (
            minimum_probability_ratio
        ),
        "utility_answer_probability_passed": probability_passed,
        "passed": bool(
            utility_passed and locality_passed and probability_passed
        ),
    }


def _validate_selection_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    candidate_id: Optional[str],
) -> None:
    if receipt.get("kind") != "controlled_candidate_selection_receipt":
        raise ValueError("Invalid controlled candidate selection receipt")
    stored_hash = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Selection receipt hash mismatch")
    checks = {
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "test_bundle_sha256": bundle["bundle_sha256"],
    }
    if candidate_id is not None:
        checks["selected_candidate_id"] = candidate_id
    for field, expected in checks.items():
        if receipt.get(field) != expected:
            raise ValueError(
                f"Selection receipt {field}={receipt.get(field)!r} does not "
                f"match required {expected!r}"
            )
    if not bool(receipt.get("hyperparameters_frozen")):
        raise ValueError("Selection receipt does not freeze hyperparameters")
    if receipt.get("test_results_used_for_selection") is not False:
        raise ValueError("Selection receipt does not prove test isolation")


def _validate_base_reference_model(
    selection_receipt: Mapping[str, Any],
    model_path: Path,
) -> None:
    spec = selection_receipt.get("selected_candidate_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("Selection receipt lacks the frozen candidate spec")
    expected = Path(str(spec.get("base_model_path", ""))).resolve()
    if not str(spec.get("base_model_path", "")).strip():
        raise ValueError("Frozen candidate spec lacks base_model_path")
    if model_path.resolve() != expected:
        raise ValueError(
            "Base-reference evaluation must use the exact base_model_path "
            "from the frozen candidate spec"
        )


def _validate_baseline_summary(
    baseline: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    judge_public: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
) -> None:
    expected = {
        "kind": "controlled_unlearning_baseline_evaluation",
        "phase": "test",
        "partition": "utility_reference",
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "bundle_sha256": bundle["bundle_sha256"],
        "evaluation_role": "base_reference",
        "selection_receipt_sha256": selection_receipt["receipt_sha256"],
    }
    for field, value in expected.items():
        if baseline.get(field) != value:
            raise ValueError(
                f"Base-reference summary {field}={baseline.get(field)!r}; "
                f"expected {value!r}"
            )
    if sha256_json(baseline.get("judge", {})) != sha256_json(judge_public):
        raise ValueError(
            "Base-reference and candidate final evaluations used different "
            "Judge B configurations"
        )
    if not baseline.get("controls", {}).get(
        "token_probability_metrics_present"
    ):
        raise ValueError(
            "Base-reference summary lacks token-probability metrics"
        )


def _validate_application_receipt(
    receipt: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    selection_receipt: Mapping[str, Any],
    candidate_id: str,
    model_path: Path,
) -> None:
    if receipt.get("kind") != "controlled_model_application_receipt":
        raise ValueError("Invalid controlled model application receipt")
    stored_hash = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Application receipt hash mismatch")
    expected = {
        "phase": "final_apply",
        "stage": "final_apply",
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "final_apply_bundle_sha256": selection_receipt[
            "final_apply_bundle_sha256"
        ],
        "test_bundle_sha256": bundle["bundle_sha256"],
        "candidate_id": candidate_id,
        "selection_receipt_sha256": selection_receipt[
            "receipt_sha256"
        ],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(
                f"Application receipt {field}={receipt.get(field)!r}; "
                f"expected {value!r}"
            )
    selected_checkpoint = Path(
        str(receipt.get("selected_checkpoint", ""))
    ).resolve()
    if selected_checkpoint != model_path.resolve():
        raise ValueError(
            "Evaluated model path differs from the final-application receipt"
        )
    if receipt.get("dry_run"):
        raise ValueError("A dry-run receipt cannot unlock final evaluation")
    if receipt.get("started_from_fresh_base") is not True:
        raise ValueError("Application receipt does not prove a fresh-base run")
    if receipt.get("test_results_used_for_repair") is not False:
        raise ValueError("Application receipt permits test feedback")


def _manual_audit_status(
    queue: Sequence[Mapping[str, Any]],
    completed_path: Optional[Path],
    *,
    minimum_agreement_rate: float,
) -> Dict[str, Any]:
    if completed_path is None:
        return {
            "required": True,
            "complete": False,
            "reviewed_count": 0,
            "agreement_rate": None,
            "classification_agreement_rate": None,
            "judge_false_pass_count": None,
            "forget_false_pass_count": None,
            "minimum_agreement_rate": minimum_agreement_rate,
            "gate_passed": False,
        }
    completed = read_jsonl(completed_path)
    by_id = {str(row.get("case_id")): row for row in completed}
    required_ids = {str(row["case_id"]) for row in queue}
    if set(by_id) != required_ids:
        missing = sorted(required_ids - set(by_id))
        extra = sorted(set(by_id) - required_ids)
        raise ValueError(
            "Manual audit IDs do not match queue: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    agreements: List[float] = []
    classification_agreements: List[float] = []
    judge_false_pass_count = 0
    forget_false_pass_count = 0
    for queued in queue:
        reviewed = by_id[str(queued["case_id"])]
        if not str(reviewed.get("reviewer", "")).strip():
            raise ValueError("Manual audit row lacks reviewer")
        if not isinstance(reviewed.get("human_passed"), bool):
            raise ValueError("Manual audit human_passed must be boolean")
        human_classification = str(
            reviewed.get("human_classification", "")
        ).strip().upper()
        if not human_classification:
            raise ValueError("Manual audit lacks human_classification")
        allowed = (
            FORGET_LABELS
            if queued["expected_behavior"] == "avoid_sensitive"
            else UTILITY_LABELS
        )
        if human_classification not in allowed:
            raise ValueError(
                f"Manual classification {human_classification!r} is not "
                f"allowed for {queued['expected_behavior']}"
            )
        expected_human_pass = (
            human_classification == "CORRECT_ABSTENTION"
            if queued["expected_behavior"] == "avoid_sensitive"
            else human_classification == "CORRECT_RETAIN"
        )
        if bool(reviewed["human_passed"]) != expected_human_pass:
            raise ValueError(
                "Manual human_passed contradicts human_classification"
            )
        agreements.append(
            1.0
            if bool(reviewed["human_passed"])
            == bool(queued["judge_passed"])
            else 0.0
        )
        classification_agreements.append(
            1.0
            if human_classification
            == str(queued["judge_classification"]).upper()
            else 0.0
        )
        if bool(queued["judge_passed"]) and not bool(
            reviewed["human_passed"]
        ):
            judge_false_pass_count += 1
            if queued["expected_behavior"] == "avoid_sensitive":
                forget_false_pass_count += 1
    agreement_rate = mean(agreements)
    classification_agreement_rate = mean(classification_agreements)
    gate_passed = bool(
        agreement_rate is not None
        and agreement_rate >= minimum_agreement_rate
        and forget_false_pass_count == 0
    )
    return {
        "required": True,
        "complete": True,
        "reviewed_count": len(completed),
        "agreement_rate": agreement_rate,
        "classification_agreement_rate": classification_agreement_rate,
        "judge_false_pass_count": judge_false_pass_count,
        "forget_false_pass_count": forget_false_pass_count,
        "minimum_agreement_rate": minimum_agreement_rate,
        "gate_passed": gate_passed,
        "path": str(completed_path),
        "sha256": sha256_file(completed_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--phase",
        choices=["development", "test"],
        required=True,
    )
    parser.add_argument(
        "--partition",
        choices=["train", "validation", "all"],
        default=None,
    )
    parser.add_argument(
        "--evaluation-role",
        choices=["candidate", "base_reference"],
        default="candidate",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--responses-jsonl", default=None)
    parser.add_argument("--judge-config", required=True)
    parser.add_argument("--recorded-judgments", default=None)
    parser.add_argument("--selection-receipt", default=None)
    parser.add_argument("--application-receipt", default=None)
    parser.add_argument("--baseline-summary", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Defaults on for TOFU and off for MCF/ZsRE. Override only when "
            "the evaluated checkpoint used a different prompt protocol."
        ),
    )
    parser.add_argument("--manual-audit-count", type=int, default=30)
    parser.add_argument("--manual-audit-file", default=None)
    parser.add_argument(
        "--min-manual-judge-agreement",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--allow-final-test-rerun",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emergency recovery only. A rerun is recorded and makes the "
            "result non-preregistered."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.model_path is None and args.responses_jsonl is None:
        raise ValueError("Provide --model-path or --responses-jsonl")
    if args.responses_jsonl is not None and args.model_path is None:
        raise ValueError(
            "--model-path is still required for mandatory token-probability "
            "metrics, even when responses are precomputed"
        )
    for name in ("batch_size", "score_batch_size", "max_new_tokens"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.manual_audit_count <= 0:
        raise ValueError("--manual-audit-count must be positive")
    if not 0.0 <= args.min_manual_judge_agreement <= 1.0:
        raise ValueError(
            "--min-manual-judge-agreement must lie in [0,1]"
        )
    if args.evaluation_role == "base_reference" and args.phase != "test":
        raise ValueError("Base-reference evaluation is final-test-only")
    if args.phase == "test" and not args.selection_receipt:
        raise ValueError("Final test requires --selection-receipt")
    if (
        args.phase == "test"
        and args.evaluation_role == "candidate"
        and not args.application_receipt
    ):
        raise ValueError("Final test requires --application-receipt")
    if (
        args.phase == "test"
        and args.evaluation_role == "candidate"
        and not args.baseline_summary
    ):
        raise ValueError(
            "Final candidate evaluation requires --baseline-summary"
        )
    if args.evaluation_role == "base_reference" and args.application_receipt:
        raise ValueError(
            "Base-reference evaluation must not use an application receipt"
        )
    if args.evaluation_role == "base_reference" and args.baseline_summary:
        raise ValueError(
            "Base-reference evaluation must not use --baseline-summary"
        )
    if args.phase == "development" and args.selection_receipt:
        raise ValueError("Development evaluation must not use a test receipt")
    if args.phase == "development" and args.application_receipt:
        raise ValueError("Development evaluation must not use an application receipt")
    if args.phase == "development" and args.baseline_summary:
        raise ValueError(
            "Development evaluation must not use a final baseline summary"
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    bundle_path = Path(args.bundle).resolve()
    bundle = (
        load_development_bundle(bundle_path)
        if args.phase == "development"
        else load_test_bundle(bundle_path)
    )
    judge_config = load_judge_config(Path(args.judge_config))
    assert_judge_phase(judge_config, bundle_phase=args.phase)
    partition = args.partition
    if partition is None:
        partition = "validation" if args.phase == "development" else "all"
    if args.phase == "test" and partition != "all":
        raise ValueError("Final test must evaluate every locked prompt case")
    cases = bundle_prompt_cases(bundle)
    if partition != "all":
        cases = [case for case in cases if case.partition == partition]
    if args.evaluation_role == "base_reference":
        cases = [
            case
            for case in cases
            if case.expected_behavior
            in {"answer_correctly", "preserve_locality"}
        ]
        partition = "utility_reference"
    if not cases:
        raise ValueError("No prompt cases match the requested partition")
    use_chat_template = (
        bool(args.use_chat_template)
        if args.use_chat_template is not None
        else bundle["dataset"] == "tofu"
    )

    selection_receipt: Optional[Dict[str, Any]] = None
    application_receipt: Optional[Dict[str, Any]] = None
    baseline_summary: Optional[Dict[str, Any]] = None
    if args.phase == "test":
        selection_receipt = read_json(Path(args.selection_receipt))
        _validate_selection_receipt(
            selection_receipt,
            bundle=bundle,
            candidate_id=(
                args.candidate_id
                if args.evaluation_role == "candidate"
                else None
            ),
        )
        assert_independent_judges(
            selection_receipt["judge_a"],
            judge_config,
        )
        if args.evaluation_role == "base_reference":
            _validate_base_reference_model(
                selection_receipt,
                Path(args.model_path),
            )
        else:
            application_receipt = read_json(Path(args.application_receipt))
            _validate_application_receipt(
                application_receipt,
                bundle=bundle,
                selection_receipt=selection_receipt,
                candidate_id=args.candidate_id,
                model_path=Path(args.model_path),
            )
            baseline_summary = read_json(Path(args.baseline_summary))
            _validate_baseline_summary(
                baseline_summary,
                bundle=bundle,
                judge_public=public_judge_config(judge_config),
                selection_receipt=selection_receipt,
            )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = output_dir / "final_test_run_state.json"
    rerun = False
    if args.phase == "test" and run_state_path.exists():
        if not args.allow_final_test_rerun:
            raise RuntimeError(
                f"Final test was already started at {run_state_path}. "
                "Results must not feed back into repair. Use a new "
                "preregistered candidate; --allow-final-test-rerun is only "
                "for explicitly disclosed recovery."
            )
        rerun = True
    manual_audit_required = bool(
        args.phase == "test" and args.evaluation_role == "candidate"
    )
    audit_strata = {
        (case.expected_behavior, case.style)
        for case in cases
    }
    if (
        manual_audit_required
        and args.manual_audit_count < len(audit_strata)
    ):
        raise ValueError(
            "--manual-audit-count must cover every behavior/style stratum; "
            f"need at least {len(audit_strata)}"
        )
    audit_seed = int(bundle["seed"]) + int(bundle["fold"]) * 1009
    planned_audit_rows = (
        stable_stratified_manual_audit_sample(
            [
                {
                    "case_id": case.case_id,
                    "expected_behavior": case.expected_behavior,
                    "style": case.style,
                }
                for case in cases
            ],
            count=args.manual_audit_count,
            seed=audit_seed,
        )
        if manual_audit_required
        else []
    )
    planned_audit_ids = [
        str(row["case_id"]) for row in planned_audit_rows
    ]
    if manual_audit_required and not any(
        row["expected_behavior"] == "avoid_sensitive"
        for row in planned_audit_rows
    ):
        raise RuntimeError(
            "The preregistered manual audit contains no forget cases"
        )
    if args.phase == "test":
        write_json(
            run_state_path,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "started",
                "protocol_id": bundle["protocol_id"],
                "dataset": bundle["dataset"],
                "fold": bundle["fold"],
                "test_bundle_sha256": bundle["bundle_sha256"],
                "candidate_id": args.candidate_id,
                "evaluation_role": args.evaluation_role,
                "manual_audit_case_ids": planned_audit_ids,
                "manual_audit_selection_uses_outputs": False,
                "rerun": rerun,
            },
        )

    model_identity = _model_identity(
        Path(args.model_path),
        args.candidate_id,
    )
    model, tokenizer = load_model_and_tokenizer(args)
    if args.responses_jsonl:
        responses = load_responses(Path(args.responses_jsonl), cases)
    else:
        responses = generate_responses(
            model,
            tokenizer,
            cases,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=use_chat_template,
        )
        write_jsonl(output_dir / "responses.jsonl", responses)
    scores = score_answer_candidates(
        model,
        tokenizer,
        cases,
        batch_size=args.score_batch_size,
        use_chat_template=use_chat_template,
    )

    recorded = load_recorded_judgments(
        Path(args.recorded_judgments)
        if args.recorded_judgments
        else None
    )
    judge = JudgeClient(judge_config, recorded=recorded)
    response_by_id = {
        str(row["case_id"]): row
        for row in responses
    }
    result_rows: List[Dict[str, Any]] = []
    for position, case in enumerate(cases, 1):
        response_row = response_by_id[case.case_id]
        response_text = str(response_row["response"])
        judgment = judge.judge(case, response_text)
        candidate_scores = scores[case.case_id]
        sensitive_probability = _max_probability(
            candidate_scores["sensitive"]
        )
        acceptable_probability = _max_probability(
            candidate_scores["acceptable"]
        )
        sensitive_preferred: Optional[bool] = None
        if (
            sensitive_probability is not None
            and acceptable_probability is not None
        ):
            sensitive_preferred = (
                sensitive_probability > acceptable_probability
            )
        result_rows.append(
            {
                "case_id": case.case_id,
                "dataset": case.dataset,
                "fold": case.fold,
                "partition": case.partition,
                "purpose": case.purpose,
                "style": case.style,
                "source_record_id": case.source_record_id,
                "source_group_id": case.source_group_id,
                "expected_behavior": case.expected_behavior,
                "prompt": case.prompt,
                "response": response_text,
                "sensitive_answers": list(case.sensitive_answers),
                "acceptable_answers": list(case.acceptable_answers),
                "lexical_sensitive_mention": answer_mentioned(
                    response_text,
                    case.sensitive_answers,
                ),
                "lexical_acceptable_mention": answer_mentioned(
                    response_text,
                    case.acceptable_answers,
                ),
                "answer_probability": candidate_scores,
                "max_sensitive_probability": sensitive_probability,
                "max_acceptable_probability": acceptable_probability,
                "sensitive_preferred": sensitive_preferred,
                "judgment": judgment,
                "case_position": position,
            }
        )
        if position % 50 == 0:
            print(f"Judged {position}/{len(cases)} cases")
    write_jsonl(output_dir / "case_results.jsonl", result_rows)
    summary_metrics = summarize_rows(result_rows)
    utility_guardrail: Optional[Dict[str, Any]] = None
    if baseline_summary is not None and selection_receipt is not None:
        utility_guardrail = build_final_utility_guardrail(
            summary_metrics,
            baseline_summary["metrics"],
            selection_receipt,
        )

    result_by_id = {
        str(row["case_id"]): row
        for row in result_rows
    }
    audit_rows = [
        result_by_id[case_id]
        for case_id in planned_audit_ids
    ]
    audit_queue = [
        {
            "case_id": row["case_id"],
            "dataset": row["dataset"],
            "fold": row["fold"],
            "purpose": row["purpose"],
            "style": row["style"],
            "expected_behavior": row["expected_behavior"],
            "prompt": row["prompt"],
            "response": row["response"],
            "sensitive_answers": row["sensitive_answers"],
            "acceptable_answers": row["acceptable_answers"],
            "judge_classification": row["judgment"]["classification"],
            "judge_passed": row["judgment"]["passed"],
            "human_classification": None,
            "human_passed": None,
            "reviewer": None,
            "notes": None,
        }
        for row in audit_rows
    ]
    audit_queue_path: Optional[Path] = None
    if manual_audit_required:
        audit_queue_path = output_dir / "manual_audit_queue.jsonl"
        write_jsonl(audit_queue_path, audit_queue)
        manual_status = _manual_audit_status(
            audit_queue,
            Path(args.manual_audit_file)
            if args.manual_audit_file
            else None,
            minimum_agreement_rate=args.min_manual_judge_agreement,
        )
    else:
        manual_status = {
            "required": False,
            "complete": True,
            "reviewed_count": 0,
            "agreement_rate": None,
            "classification_agreement_rate": None,
            "judge_false_pass_count": None,
            "forget_false_pass_count": None,
            "minimum_agreement_rate": None,
            "gate_passed": True,
        }

    release_ready = bool(
        args.phase == "development"
        or (
            not rerun
            and (
                args.evaluation_role == "base_reference"
                or (
                    manual_status["complete"]
                    and manual_status["gate_passed"]
                    and utility_guardrail is not None
                    and utility_guardrail["passed"]
                )
            )
        )
    )

    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": (
            "controlled_unlearning_baseline_evaluation"
            if args.evaluation_role == "base_reference"
            else "controlled_unlearning_evaluation"
        ),
        "phase": args.phase,
        "partition": partition,
        "evaluation_role": args.evaluation_role,
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "bundle_sha256": bundle["bundle_sha256"],
        "candidate_id": args.candidate_id,
        "candidate_spec_sha256": (
            application_receipt.get("candidate_spec_sha256")
            if application_receipt is not None
            else None
        ),
        "selection_receipt_sha256": (
            selection_receipt.get("receipt_sha256")
            if selection_receipt is not None
            else None
        ),
        "application_receipt_sha256": (
            application_receipt.get("receipt_sha256")
            if application_receipt is not None
            else None
        ),
        "model_identity": model_identity,
        "judge": public_judge_config(judge_config),
        "case_count": len(result_rows),
        "metrics": summary_metrics,
        "utility_guardrail": utility_guardrail,
        "baseline_summary": (
            {
                "path": str(Path(args.baseline_summary).resolve()),
                "sha256": sha256_file(Path(args.baseline_summary).resolve()),
            }
            if args.baseline_summary
            else None
        ),
        "manual_audit": {
            **manual_status,
            "selection": "case_id_only_stratified_by_behavior_and_style",
            "queue_path": (
                str(audit_queue_path)
                if audit_queue_path is not None
                else None
            ),
            "queue_sha256": (
                sha256_file(audit_queue_path)
                if audit_queue_path is not None
                else None
            ),
        },
        "controls": {
            "judge_is_supplemental": True,
            "token_probability_metrics_present": True,
            "locality_reported_separately": (
                "preserve_locality" in summary_metrics["by_behavior"]
            ),
            "test_results_used_for_repair": False,
            "final_test_rerun": rerun,
            "use_chat_template": use_chat_template,
            "final_utility_guardrail_present": (
                utility_guardrail is not None
            ),
            "final_utility_guardrail_passed": (
                utility_guardrail["passed"]
                if utility_guardrail is not None
                else None
            ),
        },
        "release_ready": release_ready,
    }
    summary_path = output_dir / "evaluation_summary.json"
    write_json(summary_path, summary)
    if args.phase == "test":
        write_json(
            run_state_path,
            {
                **read_json(run_state_path),
                "status": "complete",
                "evaluation_summary": str(summary_path),
                "evaluation_summary_sha256": sha256_file(summary_path),
                "manual_audit_complete": manual_status["complete"],
                "manual_audit_gate_passed": manual_status["gate_passed"],
                "utility_guardrail_passed": (
                    utility_guardrail["passed"]
                    if utility_guardrail is not None
                    else None
                ),
                "release_ready": summary["release_ready"],
            },
        )
    print(f"Wrote {summary_path}")
    if manual_audit_required and not manual_status["complete"]:
        print(
            f"Manual audit is pending: complete {audit_queue_path}, then use "
            "finalize_controlled_manual_audit.py. Do not rerun final test."
        )


if __name__ == "__main__":
    main()
