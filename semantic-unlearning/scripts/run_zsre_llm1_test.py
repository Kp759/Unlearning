#!/usr/bin/env python3
"""Apply frozen ZsRE configuration to final_apply, then test LLM1 without a judge."""

from __future__ import annotations

import argparse
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
from run_controlled_setting5e import (
    _public_model_identity,
    _run_zsre,
    _verify_materialized,
)


def resolve_committed_path(bundle_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = bundle_path.parent / path
    return path.resolve()


def validate_validation_receipt(
    receipt: Mapping[str, Any],
    *,
    development: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> None:
    stored_hash = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)

    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Validation application receipt hash mismatch")

    expected = {
        "kind": "controlled_model_application_receipt",
        "status": "accepted",
        "phase": "development",
        "stage": "validation",
        "protocol_id": development["protocol_id"],
        "dataset": "zsre",
        "fold": development["fold"],
        "bundle_sha256": development["bundle_sha256"],
        "candidate_id": spec["candidate_id"],
        "candidate_spec_sha256": sha256_json(spec),
    }

    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(
                f"Validation receipt {field}={receipt.get(field)!r}; "
                f"expected {value!r}"
            )

    if receipt.get("started_from_fresh_base") is not True:
        raise ValueError("Validation receipt does not prove fresh-Base application")

    selected_checkpoint = Path(
        str(receipt.get("selected_checkpoint", ""))
    )
    if not selected_checkpoint.exists():
        raise FileNotFoundError(
            f"Validation checkpoint missing: {selected_checkpoint}"
        )


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
    parser.add_argument("--python", default=None)

    parser.add_argument(
        "--confirm-open-locked-test",
        action="store_true",
        help="Required acknowledgement that this consumes the locked test fold.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.confirm_open_locked_test:
        raise SystemExit(
            "Refusing to open test.json without "
            "--confirm-open-locked-test"
        )

    import sys

    python_executable = args.python or sys.executable

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory must be new and empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    development_path = Path(args.development_bundle).resolve()
    development = load_development_bundle(development_path)

    if development["dataset"] != "zsre":
        raise ValueError("This runner supports only ZsRE")

    spec_path = Path(args.candidate_spec).resolve()
    spec_value = read_json(spec_path)

    if not isinstance(spec_value, Mapping):
        raise ValueError("Candidate specification must be a JSON object")

    spec = dict(spec_value)

    if spec.get("dataset") != "zsre":
        raise ValueError("Candidate specification does not target ZsRE")

    if not str(spec.get("candidate_id", "")).strip():
        raise ValueError("Candidate specification lacks candidate_id")

    if not str(spec.get("base_model_path", "")).strip():
        raise ValueError("Candidate specification lacks base_model_path")

    if (
        spec.get("seed") is not None
        and int(spec["seed"]) != int(development["seed"])
    ):
        raise ValueError(
            f"Candidate seed {spec['seed']} does not match "
            f"fold seed {development['seed']}"
        )

    validation_receipt_path = Path(
        args.validation_application_receipt
    ).resolve()
    validation_receipt = read_json(validation_receipt_path)

    if not isinstance(validation_receipt, Mapping):
        raise ValueError("Validation receipt must be a JSON object")

    validate_validation_receipt(
        validation_receipt,
        development=development,
        spec=spec,
    )

    final_apply_path = resolve_committed_path(
        development_path,
        str(development["final_apply_bundle_path"]),
    )
    test_path = resolve_committed_path(
        development_path,
        str(development["test_bundle_path"]),
    )

    if not final_apply_path.is_file():
        raise FileNotFoundError(
            f"Missing final_apply bundle: {final_apply_path}"
        )

    if not test_path.is_file():
        raise FileNotFoundError(
            f"Missing locked test bundle: {test_path}"
        )

    preregistration = {
        "schema_version": 1,
        "kind": "zsre_llm1_only_test_preregistration",
        "protocol_mode": "llm1_only_no_external_judge",
        "protocol_id": development["protocol_id"],
        "dataset": "zsre",
        "fold": development["fold"],
        "candidate_id": spec["candidate_id"],
        "candidate_spec_path": str(spec_path),
        "candidate_spec_sha256": sha256_file(spec_path),
        "candidate_spec": spec,
        "validation_application_receipt": str(
            validation_receipt_path
        ),
        "validation_application_receipt_sha256": sha256_file(
            validation_receipt_path
        ),
        "final_apply_bundle_path": str(final_apply_path),
        "final_apply_bundle_sha256": development[
            "final_apply_bundle_sha256"
        ],
        "test_bundle_path": str(test_path),
        "test_bundle_sha256": development["test_bundle_sha256"],
        "hyperparameters_frozen": True,
        "judge_used": False,
        "test_results_may_change_method": False,
    }

    preregistration_path = output_dir / "preregistration.json"
    write_json(preregistration_path, preregistration)

    # Apply the frozen method to held-out deletion requests from fresh Base.
    # The final_apply bundle is prompt-free.
    final_apply = load_final_apply_bundle(final_apply_path)

    for field in ("protocol_id", "dataset", "fold"):
        if final_apply[field] != development[field]:
            raise ValueError(
                f"Final-apply {field} does not match development bundle"
            )

    if (
        final_apply["bundle_sha256"]
        != development["final_apply_bundle_sha256"]
    ):
        raise ValueError("Final-apply commitment mismatch")

    materialized_metadata = final_apply["materialized_inputs"]
    materialized = _verify_materialized(
        final_apply_path,
        materialized_metadata,
    )

    application_dir = output_dir / "application"
    application_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []

    checkpoint = _run_zsre(
        python=python_executable,
        spec=spec,
        materialized=materialized,
        counts=materialized_metadata,
        seed=int(final_apply["seed"]),
        stage="final_apply",
        output_dir=application_dir,
        dry_run=False,
        commands=commands,
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Final-application checkpoint missing: {checkpoint}"
        )

    official_results_path = application_dir / "zsre_results.json"
    official_results = read_json(official_results_path)

    application_receipt = {
        "schema_version": 1,
        "kind": "zsre_llm1_only_application_receipt",
        "status": "accepted",
        "protocol_mode": "llm1_only_no_external_judge",
        "phase": "final_apply",
        "stage": "final_apply",
        "protocol_id": final_apply["protocol_id"],
        "dataset": "zsre",
        "fold": final_apply["fold"],
        "final_apply_bundle_sha256": final_apply["bundle_sha256"],
        "test_bundle_sha256": development["test_bundle_sha256"],
        "candidate_id": spec["candidate_id"],
        "candidate_spec_sha256": sha256_json(spec),
        "base_model_identity": _public_model_identity(
            Path(str(spec["base_model_path"]))
        ),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_identity": _public_model_identity(
            checkpoint
        ),
        "official_zsre_result": str(official_results_path),
        "official_candidate_accepted": official_results[
            "repair"
        ]["candidate_accepted"],
        "started_from_fresh_base": True,
        "judge_used": False,
        "test_bundle_opened_during_application": False,
        "test_results_used_for_repair": False,
        "commands": [
            {
                "argv": command,
                "shell_rendering_for_review": " ".join(command),
            }
            for command in commands
        ],
    }

    unhashed = dict(application_receipt)
    application_receipt["receipt_sha256"] = sha256_json(unhashed)

    application_receipt_path = (
        application_dir / "application_receipt.json"
    )
    write_json(application_receipt_path, application_receipt)

    # Only after final application is complete do we open test.json.
    test_bundle = load_test_bundle(test_path)

    if (
        test_bundle["bundle_sha256"]
        != development["test_bundle_sha256"]
    ):
        raise ValueError("Locked test commitment mismatch")

    for field in ("protocol_id", "dataset", "fold"):
        if test_bundle[field] != development[field]:
            raise ValueError(
                f"Test bundle {field} does not match development bundle"
            )

    cases = bundle_prompt_cases(test_bundle)
    if not cases:
        raise ValueError("Test bundle contains no prompt cases")

    evaluation_dir = output_dir / "llm1_test"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = str(spec["base_model_path"])

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
        "kind": "zsre_llm1_only_test_evaluation",
        "protocol_mode": "llm1_only_no_external_judge",
        "phase": "test",
        "dataset": "zsre",
        "fold": test_bundle["fold"],
        "case_count": len(cases),
        "test_bundle_sha256": test_bundle["bundle_sha256"],
        "candidate_id": spec["candidate_id"],
        "base_model_path": str(Path(base_model_path).resolve()),
        "unlearned_model_path": str(checkpoint),
        "judge_used": False,
        "test_results_used_for_repair": False,
        "official_final_application": {
            "base": official_results["base"],
            "setting5e": official_results["setting5e"],
            "active_candidate": official_results[
                "active_candidate"
            ],
            "selected": official_results["selected"],
            "repair": official_results["repair"],
        },
        "base": summarize_model(base_rows),
        "unlearned": summarize_model(unlearned_rows),
        "comparison": compare_models(
            base_rows,
            unlearned_rows,
        ),
        "application_receipt": str(application_receipt_path),
        "preregistration": str(preregistration_path),
    }

    summary_path = output_dir / "llm1_test_summary.json"
    write_json(summary_path, summary)

    selected = official_results["selected"]

    print()
    print("Final application official ZsRE metrics")
    print(
        "Forget:",
        f'Eff={selected["forget"]["Eff"]}',
        f'Gen={selected["forget"]["Gen"]}',
        f'Spe={selected["forget"]["Spe"]}',
    )
    print(
        "Retain:",
        f'Eff={selected["retain"]["Eff"]}',
        f'Gen={selected["retain"]["Gen"]}',
        f'Spe={selected["retain"]["Spe"]}',
    )
    print(f"LLM1 test summary: {summary_path}")
    print(f"Final checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
