#!/usr/bin/env python3
"""SURE-TOFU Stage1B v4: sparse LM-head forgetting with Base answer embeddings.

V4 intentionally changes only two controls relative to the progressive sparse
v3 path:

1. Every training-visible answer-token INPUT embedding row is restored exactly
   to the protected Full-TOFU Base before every restricted rank-0 attempt and
   again before final materialization.  Only sparse sensitive LM-head rows may
   carry the Stage1A/rank-0 forgetting displacement.
2. The direct-forget NLL buffer is fixed to zero.  Boundary bisection therefore
   seeks the minimum feasible edit satisfying the declared answer-probability
   target itself (default 3e-4), rather than a stronger buffered target.

All progressive row ranking/promotion, selected-row rank-0 optimization,
boundary bisection, locked data access, and fail-closed materialization checks
are inherited from v3/v2.  No retain, paraphrase, same-author holdout, PPL, or
other final-evaluation metric is used for optimization or selection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

import tofu_sure_rank0_forget_progressive_v3 as v3


METHOD = "SURE-TOFU-progressive-sparse-rank0-all-answer-embeddings-base-v4"

# Captured before monkey-patching.
_ORIGINAL_APPLY = v3.old.apply_answer_row_policy
_ORIGINAL_RESTORE_ERROR = v3.old.answer_row_restoration_error
_INPUT_WEIGHT_PTR: int | None = None


def apply_v4_row_policy(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    answer_ids: Sequence[int],
    sensitive_ids: Sequence[int],
    stage1a_input_rows: torch.Tensor,
    stage1a_output_rows: torch.Tensor,
    base_input_rows: torch.Tensor,
    base_output_rows: torch.Tensor,
) -> None:
    """Apply v3 LM policy, then force ALL visible answer embeddings to Base."""
    global _INPUT_WEIGHT_PTR
    _INPUT_WEIGHT_PTR = int(input_weight.data_ptr())

    _ORIGINAL_APPLY(
        input_weight,
        output_weight,
        answer_ids,
        sensitive_ids,
        stage1a_input_rows,
        stage1a_output_rows,
        base_input_rows,
        base_output_rows,
    )

    if not answer_ids:
        return
    ids = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    with torch.no_grad():
        input_weight.index_copy_(
            0,
            ids,
            base_input_rows.to(device=input_weight.device, dtype=input_weight.dtype),
        )


def v4_restoration_error(
    weight: torch.Tensor,
    answer_ids: Sequence[int],
    restored_ids: Sequence[int],
    reference_rows: torch.Tensor,
) -> float:
    """Audit all answer embeddings, but only non-sensitive LM-head rows.

    V3 already invokes this audit once for input embeddings and once for the LM
    head.  The input tensor pointer captured by apply_v4_row_policy lets v4
    strengthen only the input audit without changing the LM-head semantics.
    """
    if _INPUT_WEIGHT_PTR is not None and int(weight.data_ptr()) == _INPUT_WEIGHT_PTR:
        restored_ids = answer_ids
    return _ORIGINAL_RESTORE_ERROR(weight, answer_ids, restored_ids, reference_rows)


def _arg_value(name: str) -> str | None:
    for index, item in enumerate(sys.argv[1:], start=1):
        if item == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
        prefix = name + "="
        if item.startswith(prefix):
            return item[len(prefix):]
    return None


def _require_zero_buffer() -> None:
    value = _arg_value("--target-nll-buffer")
    if value is None:
        sys.argv.extend(["--target-nll-buffer", "0"])
        return
    if abs(float(value)) > 1e-12:
        raise ValueError(
            "SURE-TOFU v4 fixes --target-nll-buffer to 0; "
            f"received {value!r}"
        )


def _patch_json_outputs() -> None:
    output_dir = _arg_value("--output-dir")
    if output_dir is None:
        return
    root = Path(output_dir).expanduser().resolve()

    summary_path = root / "repair_summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["method"] = METHOD
        payload["input_embedding_policy"] = (
            "all training-visible answer-token input embedding rows exact Full-TOFU Base"
        )
        payload["lm_head_policy"] = (
            "non-sensitive answer rows exact Base; only progressively selected sensitive rows carry Stage1A plus rank0 displacement"
        )
        # v4_restoration_error strengthened the first v3 audit to all answer rows.
        payload["all_visible_answer_input_base_max_abs_error"] = payload.get(
            "non_sensitive_input_base_max_abs_error"
        )
        payload["target_nll_buffer"] = 0.0
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    restoration_path = root / "answer_row_restoration.json"
    if restoration_path.is_file():
        payload = json.loads(restoration_path.read_text(encoding="utf-8"))
        payload["policy"] = (
            "v4 progressive sparse LM-head sensitivity; ALL visible answer-token input embeddings exact Full-TOFU Base; non-sensitive LM-head rows exact Base"
        )
        payload["all_visible_answer_input_rows_exact_base"] = True
        payload["sensitive_input_rows_keep_stage1a"] = False
        restoration_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    config_path = root / "config_used.json"
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        payload["schema_version"] = max(int(payload.get("schema_version", 0)), 5)
        payload["method"] = METHOD
        payload["target_nll_buffer"] = 0.0
        payload["answer_embedding_restoration"] = (
            "all training-visible answer-token input rows exact Full-TOFU Base"
        )
        payload["answer_row_restoration"] = (
            "all visible answer input rows Base; non-sensitive visible answer LM-head rows Base"
        )
        config_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    _require_zero_buffer()
    v3.METHOD = METHOD
    v3.old.apply_answer_row_policy = apply_v4_row_policy
    v3.old.answer_row_restoration_error = v4_restoration_error
    v3.main()
    _patch_json_outputs()


if __name__ == "__main__":
    main()
