#!/usr/bin/env python3
"""Run the existing ZsRE LLM1-only test with stop-on-Unknown decoding.

This wrapper leaves the original evaluator unchanged for reproducibility. It
patches the shared generation function at runtime so common single-token forms
of the abstention answer ``Unknown`` are treated as additional end-of-sequence
tokens. It also truncates the decoded response after the first textual Unknown
as a final safeguard for multi-token or casing variants.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

import evaluate_controlled_llm1_only as llm1_only
import evaluate_controlled_unlearning as controlled_eval
from controlled_unlearning_protocol import PromptCase


ABSTENTION_TEXT = "Unknown"
ABSTENTION_VARIANTS = (
    "Unknown",
    " Unknown",
    "\nUnknown",
    "\tUnknown",
    "unknown",
    " unknown",
    "\nunknown",
    "\tunknown",
)


def _flatten_token_ids(value: Any) -> List[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _append_eos_ids(values: List[int], candidate: Any) -> None:
    if isinstance(candidate, int):
        values.append(int(candidate))
    elif isinstance(candidate, (list, tuple)):
        values.extend(int(token_id) for token_id in candidate)


def _resolve_stop_ids(tokenizer: Any) -> Tuple[List[int], List[int]]:
    stop_ids: List[int] = []
    _append_eos_ids(stop_ids, getattr(tokenizer, "eos_token_id", None))
    _append_eos_ids(
        stop_ids,
        getattr(
            getattr(tokenizer, "generation_config", None),
            "eos_token_id",
            None,
        ),
    )

    unknown_token_ids: List[int] = []
    for variant in ABSTENTION_VARIANTS:
        encoded = tokenizer(
            variant,
            add_special_tokens=False,
        )
        token_ids = _flatten_token_ids(encoded["input_ids"])
        if len(token_ids) == 1:
            unknown_token_ids.append(token_ids[0])

    unknown_token_ids = list(dict.fromkeys(unknown_token_ids))
    if not unknown_token_ids:
        raise ValueError(
            "No configured Unknown variant tokenized to exactly one token. "
            "Decoded truncation cannot prevent wasted generation by itself."
        )

    stop_ids.extend(unknown_token_ids)
    stop_ids = list(dict.fromkeys(stop_ids))
    return stop_ids, unknown_token_ids


def _truncate_after_first_unknown(text: str) -> Tuple[str, bool]:
    """Keep all text through the first Unknown occurrence and drop the rest."""

    match = re.search(r"unknown", str(text), flags=re.IGNORECASE)
    if match is None:
        return str(text).strip(), False

    prefix = str(text)[: match.start()].rstrip()
    truncated = (
        f"{prefix} {ABSTENTION_TEXT}" if prefix else ABSTENTION_TEXT
    )
    return truncated.strip(), True


def generate_responses_stop_on_unknown(
    model: Any,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    *,
    batch_size: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    """Generate greedily and expose at most one textual Unknown."""

    import torch

    stop_ids, unknown_token_ids = _resolve_stop_ids(tokenizer)
    decoded_unknown_tokens = {
        token_id: tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )
        for token_id in unknown_token_ids
    }
    print(
        "Stop-on-Unknown decoding enabled: "
        f"unknown_token_ids={decoded_unknown_tokens}, "
        f"eos_token_ids={stop_ids}, "
        "decoded_truncation=True"
    )

    rows: List[Dict[str, Any]] = []
    device = controlled_eval._first_device(model)

    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        prepared = [
            controlled_eval._prepare_prompt(
                tokenizer,
                case.prompt,
                use_chat_template,
            )
            for case in batch
        ]
        encoded = tokenizer(
            prepared,
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=not controlled_eval._chat_template_active(
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
                eos_token_id=stop_ids,
            )

        for case, token_ids in zip(batch, generated):
            continuation = token_ids[input_width:]
            raw_response = tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            ).strip()
            response, text_truncated = _truncate_after_first_unknown(
                raw_response
            )
            rows.append(
                {
                    "case_id": case.case_id,
                    "response": response,
                    "generation": {
                        "do_sample": False,
                        "max_new_tokens": max_new_tokens,
                        "stop_on_unknown": True,
                        "unknown_text": ABSTENTION_TEXT,
                        "unknown_token_ids": unknown_token_ids,
                        "eos_token_ids": stop_ids,
                        "decoded_truncation": True,
                        "decoded_response_was_truncated": text_truncated,
                    },
                }
            )

    return rows


def main() -> None:
    # evaluate_controlled_llm1_only.evaluate_model resolves this global at
    # runtime, so patching both modules covers the LLM1-only path and any
    # direct controlled-evaluator use within the imported runner.
    controlled_eval.generate_responses = generate_responses_stop_on_unknown
    llm1_only.generate_responses = generate_responses_stop_on_unknown

    try:
        import run_zsre_llm1_test as zsre_runner
    except ModuleNotFoundError as error:
        raise SystemExit(
            "scripts/run_zsre_llm1_test.py is missing. Create or restore the "
            "existing ZsRE LLM1-only test runner before using this wrapper."
        ) from error

    zsre_runner.main()


if __name__ == "__main__":
    main()
