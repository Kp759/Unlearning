#!/usr/bin/env python3
"""Run a one-shot relaxed-gate MCF final application and LLM1-only test.

This is an explicitly exploratory protocol amendment. It preserves the strict
post-reload gate result as diagnostic evidence but does not require that gate to
pass before opening the locked test bundle. The selected candidate and its
validation evidence are frozen before final application begins.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

from controlled_unlearning_protocol import (
    bundle_prompt_cases,
    load_development_bundle,
    load_final_apply_bundle,
    load_test_bundle,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)
from evaluate_controlled_llm1_only import (
    compare_models,
    evaluate_model,
    summarize_model,
)
from mcf_zero_unlearn_official_eval import build_post_reload_acceptance_gate
from run_controlled_setting5e import (
    _public_model_identity,
    _run_mcf,
    _verify_materialized,
)


def _validate_hashed_receipt(receipt: Mapping[str, Any]) -> None:
    stored_hash = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Validation application receipt hash mismatch")


def _validate_validation_evidence(
    receipt: Mapping[str, Any],
    *,
    development: Mapping[str, Any],
    candidate_id: str,
) -> Mapping[str, Any]:
    _validate_hashed_receipt(receipt)
    expected = {
        "kind": "controlled_model_application_receipt",
        "phase": "development",
        "stage": "validation",
        "protocol_id": development["protocol_id"],
        "dataset": "mcf",
        "fold": development["fold"],
        "bundle_sha256": development["bundle_sha256"],
        "candidate_id": candidate_id,
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(
                f"Validation receipt {field}={receipt.get(field)!r}; "
                f"expected {value!r}"
            )
    gate = receipt.get("post_reload_acceptance")
    if not isinstance(gate, Mapping):
        raise ValueError("Validation receipt lacks post-reload MCF evidence")
    if gate.get("kind") != "mcf_post_reload_acceptance":
        raise ValueError("Validation receipt has invalid MCF gate evidence")
    if gate.get("checkpoint_was_reloaded") is not True:
        raise ValueError("Validation evidence was not measured after reload")
    return gate


def _resolve_committed_path(bundle_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = bundle_path.parent / path
    return path.resolve()


def _command_records(commands: list[list[str]]) -> list[dict[str, Any]]:
    return [
        {
            "argv": command,
            "shell_rendering_for_review": shlex.join(command),
        }
        for command in commands
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument("--candidate-spec", required=True)
    parser.add_argument("--validation-application-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default="single",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--confirm-open-locked-test",
        action="store_true",
        help=(
            "Required acknowledgement that this exploratory run consumes the "
            "locked test bundle for the relaxed protocol."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.confirm_open_locked_test:
        raise SystemExit(
            "Refusing to open the locked test bundle without "
            "--confirm-open-locked-test"
        )

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "Output directory must be new and empty; the relaxed test is "
            "one-shot."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    development_path = Path(args.development_bundle).resolve()
    development = load_development_bundle(development_path)
    if development.get("dataset") != "mcf":
        raise ValueError("This runner supports only MCF")

    spec_path = Path(args.candidate_spec).resolve()
    spec = read_json(spec_path)
    if not isinstance(spec, Mapping):
        raise ValueError("Candidate spec must be a JSON object")
    spec = dict(spec)
    candidate_id = str(spec.get("candidate_id", "")).strip()
    if not candidate_id:
        raise ValueError("Candidate spec lacks candidate_id")
    if spec.get("dataset") != "mcf":
        raise ValueError("Candidate spec does not target MCF")
    base_model_path = str(spec.get("base_model_path", "")).strip()
    if not base_model_path:
        raise ValueError("Candidate spec lacks base_model_path")

    validation_receipt_path = Path(
        args.validation_application_receipt
    ).resolve()
    validation_receipt = read_json(validation_receipt_path)
    if not isinstance(validation_receipt, Mapping):
        raise ValueError("Validation receipt must be a JSON object")
    validation_gate = _validate_validation_evidence(
        validation_receipt,
        development=development,
        candidate_id=candidate_id,
    )

    final_apply_path = _resolve_committed_path(
        development_path,
        str(development["final_apply_bundle_path"]),
    )
    test_path = _resolve_committed_path(
        development_path,
        str(development["test_bundle_path"]),
    )
    if sha256_file(final_apply_path) != development["final_apply_bundle_sha256"]:
        raise ValueError("Committed final-apply bundle hash mismatch")
    if sha256_file(test_path) != development["test_bundle_sha256"]:
        raise ValueError("Committed locked-test bundle hash mismatch")

    preregistration = {
        "schema_version": 1,
        "kind": "relaxed_mcf_llm1_test_preregistration",
        "protocol_mode": "relaxed_report_only",
        "protocol_id": development["protocol_id"],
        "dataset": "mcf",
        "fold": development["fold"],
        "candidate_id": candidate_id,
        "candidate_spec_path": str(spec_path),
        "candidate_spec_sha256": sha256_file(spec_path),
        "candidate_spec": spec,
        "validation_application_receipt": str(validation_receipt_path),
        "validation_application_receipt_sha256": sha256_file(
            validation_receipt_path
        ),
        "validation_strict_gate": dict(validation_gate),
        "strict_gate_required_for_advancement": False,
        "selection_rule": (
            "candidate frozen from the completed validation sweep before "
            "opening test; strict gate retained as report-only evidence"
        ),
        "final_apply_bundle_path": str(final_apply_path),
        "final_apply_bundle_sha256": development[
            "final_apply_bundle_sha256"
        ],
        "test_bundle_path": str(test_path),
        "test_bundle_sha256": development["test_bundle_sha256"],
        "hyperparameters_frozen": True,
        "test_results_may_change_repair": False,
        "judge_used": False,
    }
    preregistration_path = output_dir / "relaxed_preregistration.json"
    write_json(preregistration_path, preregistration)

    # The final-apply bundle is prompt-free. It is safe to load before test.
    final_apply = load_final_apply_bundle(final_apply_path)
    for field in ("protocol_id", "dataset", "fold"):
        if final_apply[field] != development[field]:
            raise ValueError(f"Final-apply {field} does not match development")
    if final_apply["bundle_sha256"] != development["final_apply_bundle_sha256"]:
        raise ValueError("Final-apply bundle commitment mismatch")

    materialized_metadata = final_apply["materialized_inputs"]
    materialized = _verify_materialized(
        final_apply_path,
        materialized_metadata,
    )
    commands: list[list[str]] = []
    application_dir = output_dir / "application"
    application_dir.mkdir(parents=True, exist_ok=True)

    # Passing stage="train" disables only the strict exit gate inside the
    # existing MCF runner. The records/counts are still the held-out
    # final-apply materialization from final_apply.json.
    checkpoint = _run_mcf(
        python=args.python,
        spec=spec,
        materialized=materialized,
        counts=materialized_metadata,
        seed=int(final_apply["seed"]),
        stage="train",
        output_dir=application_dir,
        dry_run=False,
        commands=commands,
    )

    official_path = (
        application_dir / "active_repair" / "official_eval_selected.json"
    )
    official = read_json(official_path)
    final_gate = build_post_reload_acceptance_gate(
        official,
        max_forget_eff=0.0,
        max_forget_gen=0.0,
        min_forget_margin=0.1,
    )
    official["post_reload_acceptance"] = final_gate
    write_json(official_path, official)

    application_receipt = {
        "schema_version": 1,
        "kind": "relaxed_controlled_model_application_receipt",
        "status": "accepted_for_relaxed_llm1_test",
        "protocol_mode": "relaxed_report_only",
        "protocol_id": final_apply["protocol_id"],
        "dataset": "mcf",
        "fold": final_apply["fold"],
        "phase": "final_apply",
        "stage": "final_apply",
        "final_apply_bundle_sha256": final_apply["bundle_sha256"],
        "test_bundle_sha256": development["test_bundle_sha256"],
        "candidate_id": candidate_id,
        "candidate_spec_sha256": sha256_json(spec),
        "base_model_identity": _public_model_identity(
            Path(base_model_path)
        ),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_identity": _public_model_identity(checkpoint),
        "post_reload_acceptance": final_gate,
        "strict_gate_passed": bool(final_gate["passed"]),
        "strict_gate_required_for_advancement": False,
        "started_from_fresh_base": True,
        "test_bundle_opened_during_application": False,
        "test_results_used_for_repair": False,
        "commands": _command_records(commands),
    }
    unhashed = dict(application_receipt)
    application_receipt["receipt_sha256"] = sha256_json(unhashed)
    application_receipt_path = application_dir / "application_receipt.json"
    write_json(application_receipt_path, application_receipt)

    # Only now open the locked test prompts.
    test_bundle = load_test_bundle(test_path)
    if test_bundle["bundle_sha256"] != development["test_bundle_sha256"]:
        raise ValueError("Locked test bundle commitment mismatch")
    for field in ("protocol_id", "dataset", "fold"):
        if test_bundle[field] != development[field]:
            raise ValueError(f"Locked test {field} does not match development")

    cases = bundle_prompt_cases(test_bundle)
    if not cases:
        raise ValueError("Locked test bundle contains no prompt cases")

    evaluation_dir = output_dir / "llm1_test"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    base_rows = evaluate_model(
        base_model_path,
        "base",
        cases,
        args,
        evaluation_dir,
    )
    unlearned_rows = evaluate_model(
        str(checkpoint),
        "unlearned",
        cases,
        args,
        evaluation_dir,
    )

    summary = {
        "schema_version": 1,
        "kind": "relaxed_mcf_llm1_test_evaluation",
        "protocol_mode": "relaxed_report_only",
        "phase": "test",
        "dataset": "mcf",
        "fold": test_bundle["fold"],
        "case_count": len(cases),
        "test_bundle_sha256": test_bundle["bundle_sha256"],
        "candidate_id": candidate_id,
        "base_model_path": str(Path(base_model_path).resolve()),
        "unlearned_model_path": str(checkpoint),
        "judge_used": False,
        "validation_strict_gate": dict(validation_gate),
        "final_application_strict_gate": final_gate,
        "strict_gate_required_for_advancement": False,
        "test_results_used_for_repair": False,
        "base": summarize_model(base_rows),
        "unlearned": summarize_model(unlearned_rows),
        "comparison": compare_models(base_rows, unlearned_rows),
        "application_receipt": str(application_receipt_path),
        "preregistration": str(preregistration_path),
    }
    summary_path = output_dir / "llm1_relaxed_test_summary.json"
    write_json(summary_path, summary)
    print(f"Wrote relaxed LLM1 test summary: {summary_path}")
    print(f"Final application checkpoint: {checkpoint}")
    print(
        "Final strict gate passed: "
        f"{final_gate['passed']} (reported, not enforced)"
    )


if __name__ == "__main__":
    main()
