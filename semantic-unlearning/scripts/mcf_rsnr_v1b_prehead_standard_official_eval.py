#!/usr/bin/env python3
"""Official MCF evaluation for RSNR-V1B PreHead standard-unlearning adapter.

Reuses the validated RSNR-V1A PreHead evaluator while changing only:
  * expected artifact protocol;
  * paper-facing Eff/Gen labels to exact ZeroUnlearn post_*_success semantics;
  * method label in the serialized result.

All model scoring, routing, split validation, fresh-disjoint retain selection,
PPL, IDK diagnostics, true-answer suppression, greedy generation and alias
leakage audits are otherwise inherited unchanged.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import mcf_zero_unlearn_official_eval as official
from mcf_zero_unlearn_metric_parity import apply_zero_unlearn_eff_gen


_ORIGINAL_SUMMARIZE = official.official_summarize


def _zero_unlearn_parity_summarize(split_name, metric_data):
    return apply_zero_unlearn_eff_gen(_ORIGINAL_SUMMARIZE(split_name, metric_data))


official.official_summarize = _zero_unlearn_parity_summarize

import mcf_rsnr_v1a_prehead_official_eval as evaluator  # noqa: E402
import run_mcf_rsnr_v1b_prehead_standard_unlearn as trainer  # noqa: E402


evaluator.PROTOCOL = trainer.PROTOCOL


def _arg_value(flag: str) -> str:
    if flag not in sys.argv:
        raise RuntimeError(f"missing required argument {flag}")
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        raise RuntimeError(f"missing value for {flag}")
    return sys.argv[i + 1]


def main() -> None:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        evaluator.main()

    out = Path(_arg_value("--out")).resolve()
    payload = json.loads(out.read_text(encoding="utf-8"))
    payload["method"] = "rsnr_v1b_prehead_standard_unlearn"
    payload["training_objective"] = {
        "benchmark_aligned": True,
        "target_new_used_on_training_visible_views": True,
        "four_constraints": [
            "ZeroUnlearn-compatible target_new-vs-target_true margin",
            "IDK-vs-target_true margin",
            "Base-to-edited target_true logprob drop",
            "exact gate-off Base identity",
        ],
        "irreversible_latent_deletion_claimed": False,
    }
    legacy = payload.get("legacy_counterfact", {})
    if isinstance(legacy, dict):
        legacy["note"] = (
            "Eff/Gen are exact ZeroUnlearn post_rewrite_success/post_paraphrase_success labels. "
            "Unlike V1A, V1B intentionally uses target_new on training-visible five-view forget prompts."
        )
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    old_name = "rsnr_v1a_prehead_oracle_idk_aware"
    text = captured.getvalue().replace(old_name, payload["method"])
    if text.strip():
        print(text.rstrip())
    print(json.dumps({
        "method": payload["method"],
        "zero_unlearn_parity": True,
        "forget_Eff": payload["forget"]["Eff"],
        "forget_Gen": payload["forget"]["Gen"],
        "forget_Spe": payload["forget"]["Spe"],
        "forget_PPL": payload.get("forget_PPL"),
        "method_aligned_sensitive_answer": payload.get("method_aligned_sensitive_answer"),
        "generation_summary": {
            k: v for k, v in payload.get("rsnr_generation_audit", {}).items()
            if k != "per_prompt"
        },
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
