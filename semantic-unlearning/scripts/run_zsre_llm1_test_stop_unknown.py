#!/usr/bin/env python3
"""Run the existing ZsRE LLM1-only test with stop-on-Unknown decoding.

This wrapper leaves the original evaluator unchanged for reproducibility. It
patches the shared generation function at runtime so the single-token
abstention answer ``Unknown`` is treated as an additional end-of-sequence token.
The emitted Unknown token remains in the decoded response; generation stops
before a second copy can be produced.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import evaluate_controlled_llm1_only as llm1_only
import evaluate_controlled_unlearning as controlled_eval
from controlled_unlearning_protocol import PromptCase


ABSTENTION_TEXT = "Unknown"


def _flatten_token_ids(value: Any) -> List[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def _resolve_stop_ids(tokenizer: Any) -> Tuple[List[int], int]:
    encoded = tokenizer(
        ABSTENTION_TEXT,
        add_special_tokens=False,
    )
    unknown_ids = _flatten_token_ids(encoded["input_ids"])
    if len(unknown_ids) != 1:
        raise ValueError(
            f"{ABSTENTION_TEXT!r} must tokenize to exactly one token; "
            f"received {unknown_ids}"
        )
    unknown_token_id = unknown_ids[0]

    stop_ids: List[int] = []
    configured_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(configured_eos, int):
        stop_ids.append(int(configured_eos))
    elif isinstance(configured_eos, (list, tuple)):
        stop_ids.extend(int(token_id) for token_id in configured_eos)

    model_eos = getattr(
        getattr(tokenizer, "generation_config", None),
        "eos_token_id",
        None,
    )
    if isinstance(model_eos, int):
        stop_ids.append(int(model_eos))
    elif isinstance(model_eos, (list, tuple)):
        stop_ids.extend(int(token_id) for token_id in model_eos)

    stop_ids.append(unknown_token_id)
    stop_ids = list(dict.fromkeys(stop_ids))
    return stop_ids, unknown_token_id


def generate_responses_stop_on_unknown(
    model: Any,
    tokenizer: Any,
    cases: Sequence[PromptCase],
    *,
    batch_size: int,
    max_new_tokens: int,
    use_chat_template: bool,
) -> List[Dict[str, Any]]:
    """Generate greedily and stop after the first emitted Unknown token."""

    import torch

    stop_ids, unknown_token_id = _resolve_stop_ids(tokenizer)
    decoded_unknown = tokenizer.decode(
        [unknown_token_id],
        skip_special_tokens=False,
    )
    print(
        "Stop-on-Unknown decoding enabled: "
        f"token_id={unknown_token_id}, decoded={decoded_unknown!r}, "
        f"eos_token_ids={stop_ids}"
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
                        "stop_on_unknown": True,
                        "unknown_text": ABSTENTION_TEXT,
                        "unknown_token_id": unknown_token_id,
                        "eos_token_ids": stop_ids,
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
