#!/usr/bin/env python3
"""One-shot recovery of the frozen V3.6.2 official evaluation.

The original official process opened the fixed MCF file and then stopped
before its first metric forward because it compared two different identity
namespaces: dataset-position IDs frozen by the training split builder versus
the raw records' embedded ``case_id`` values.  This evaluator accepts only
that exact preserved failure, binds the official split to the already frozen
training split manifest, and evaluates the same checkpoint, arms, metrics,
thresholds, and PPL source.

This is an evaluator-only recovery after official access, not a clean first
run and not permission to retrain, select, mutate, resume, or retry a model.
The failed output directory remains immutable evidence.  The recovery output
is itself single-use and cannot be resumed or retried.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import evaluate_mcf_embedding_keyed_neuron_v3_6_2_official as official_v1
from mcf_sampling import sample_official_mcf_records
from mcf_zero_unlearn_official_eval import (
    load_mcf,
    load_official_ppl_text,
    normalize_record,
)


PROTOCOL = (
    "mcf_embedding_keyed_sparse_neuron_suppression_"
    "v3_6_2_official_index_identity_recovery_v1"
)
ORIGINAL_PROTOCOL = official_v1.PROTOCOL
ORIGINAL_PROTOCOL_SHA256 = (
    "df280fa254b6f2bf55beace43103c601118874eb745d626211eb83581cf0b009"
)
ORIGINAL_EVALUATOR_SHA256 = (
    "73d4216d5aea60f1912faf19e5d89f7303078b796c1a323d61c1deaacc65e259"
)
EXPECTED_FAILURE = "official forget split differs from the frozen candidate"
SPLIT_PROTOCOL = "sure_mcf_target_aware_direct_only_v8"
EXPECTED_DATASET_SIZE = 20877


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-run-dir", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--original-protocol", required=True)
    parser.add_argument("--recovery-protocol", required=True)
    parser.add_argument("--failed-official-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def validate_recovery_protocol(
    value: Mapping[str, Any],
    *,
    source_path: Path,
    original_protocol_path: Path,
    original_evaluator_path: Path,
) -> None:
    require = official_v1.require
    require(value.get("schema_version") == 1, "recovery protocol schema mismatch")
    require(value.get("protocol") == PROTOCOL, "recovery protocol name mismatch")
    require(
        value.get("status") == "frozen_after_evaluator_failure_before_recovery",
        "recovery protocol is not frozen at the declared boundary",
    )
    require(
        value.get("training_protocol") == official_v1.TRAINING_PROTOCOL,
        "recovery protocol points to a different learner",
    )
    require(
        value.get("candidate_checkpoint_sha256")
        == official_v1.EXPECTED_CANDIDATE_SHA256,
        "recovery protocol candidate hash mismatch",
    )
    require(
        value.get("official_mcf_sha256") == official_v1.EXPECTED_MCF_SHA256,
        "recovery protocol MCF hash mismatch",
    )
    require(
        value.get("original_official_protocol") == ORIGINAL_PROTOCOL,
        "recovery protocol does not name the failed protocol",
    )
    require(
        value.get("original_official_protocol_sha256") == ORIGINAL_PROTOCOL_SHA256
        and official_v1.sha256_file(original_protocol_path)
        == ORIGINAL_PROTOCOL_SHA256,
        "original official protocol hash mismatch",
    )
    require(
        value.get("original_evaluator_source_sha256")
        == ORIGINAL_EVALUATOR_SHA256
        and official_v1.sha256_file(original_evaluator_path)
        == ORIGINAL_EVALUATOR_SHA256,
        "original evaluator source hash mismatch",
    )
    require(
        value.get("recovery_evaluator_source_sha256")
        == official_v1.sha256_file(source_path),
        "recovery evaluator differs from its frozen protocol",
    )
    require(
        value.get("forget_dataset_indices") == official_v1.EXPECTED_CASE_IDS,
        "recovery forget-index order mismatch",
    )
    require(
        value.get("evaluation") == official_v1.FIXED_EVALUATION,
        "recovery evaluation configuration changed",
    )
    require(
        value.get("fixed_arm_order") == official_v1.FIXED_ARM_ORDER,
        "recovery arm order changed",
    )
    require(
        value.get("candidate_acceptance_thresholds")
        == official_v1.FIXED_THRESHOLDS,
        "recovery metric thresholds changed",
    )
    diagnosed = value.get("diagnosed_failure")
    require(isinstance(diagnosed, Mapping), "recovery diagnosis is missing")
    require(
        diagnosed.get("error") == EXPECTED_FAILURE
        and diagnosed.get("official_dataset_opened") is True
        and diagnosed.get("official_metric_forwards_completed") == 0
        and diagnosed.get("candidate_or_training_changed") is False,
        "recovery diagnosis does not bind the observed pre-forward failure",
    )
    repair = value.get("repair_scope")
    require(isinstance(repair, Mapping), "recovery repair scope is missing")
    require(
        repair.get("old_identity") == "raw_record.case_id"
        and repair.get("correct_identity")
        == "split_manifest.sampling.dataset_position_indices"
        and repair.get("model_change") is False
        and repair.get("metric_change") is False
        and repair.get("threshold_change") is False
        and repair.get("split_change") is False,
        "recovery scope is not an evaluator-only identity repair",
    )
    policy = value.get("recovery_one_shot_policy")
    require(isinstance(policy, Mapping), "recovery one-shot policy is missing")
    for key in (
        "fresh_output_directory_required",
        "failed_output_preserved_immutable",
        "resume_prohibited",
        "retry_prohibited",
        "retraining_prohibited",
        "checkpoint_selection_prohibited",
        "gradient_updates_prohibited",
        "candidate_mutation_prohibited",
        "all_fixed_arms_reported_regardless_of_outcome",
    ):
        require(policy.get(key) is True, f"recovery policy weakened: {key}")


def validate_failed_official_attempt(path: Path) -> Dict[str, Any]:
    """Bind recovery to the exact V1 failure and prove no metric arm ran."""
    require = official_v1.require
    path = path.resolve()
    require(path.is_dir(), f"failed official output is missing: {path}")
    pre_path = path / "pre_open_firewall_receipt.json"
    opened_path = path / "official_evaluation_opened.json"
    terminal_path = path / "terminal_status.json"
    for required_path in (pre_path, opened_path, terminal_path):
        require(required_path.is_file(), f"failed-run receipt is missing: {required_path}")

    pre = official_v1.read_json(pre_path)
    opened = official_v1.read_json(opened_path)
    terminal = official_v1.read_json(terminal_path)
    require(pre.get("protocol") == ORIGINAL_PROTOCOL, "failed pre-open protocol mismatch")
    require(
        pre.get("official_evaluation_opened") is False
        and pre.get("all_training_and_lineage_checks_passed") is True
        and pre.get("candidate_checkpoint_sha256")
        == official_v1.EXPECTED_CANDIDATE_SHA256
        and pre.get("protocol_sha256") == ORIGINAL_PROTOCOL_SHA256
        and pre.get("evaluator_source_sha256") == ORIGINAL_EVALUATOR_SHA256
        and pre.get("optimizer_constructed") is False
        and pre.get("gradient_updates_performed") == 0,
        "failed pre-open receipt does not bind the frozen candidate/firewall",
    )
    require(
        opened.get("protocol") == ORIGINAL_PROTOCOL
        and opened.get("official_evaluation_opened") is True
        and opened.get("official_mcf_sha256") == official_v1.EXPECTED_MCF_SHA256
        and opened.get("expected_official_mcf_sha256")
        == official_v1.EXPECTED_MCF_SHA256
        and opened.get("candidate_checkpoint_sha256")
        == official_v1.EXPECTED_CANDIDATE_SHA256
        and opened.get("resume_or_retry_allowed") is False,
        "failed official-open receipt is not the registered attempt",
    )
    require(
        "forget_case_ids" not in opened,
        "failed attempt progressed beyond the registered identity check",
    )
    require(
        terminal.get("protocol") == ORIGINAL_PROTOCOL
        and terminal.get("status") == "failed_after_official_open"
        and terminal.get("official_evaluation_opened") is True
        and terminal.get("official_evaluation_completed") is False
        and terminal.get("error_type") == "RuntimeError"
        and terminal.get("error") == EXPECTED_FAILURE
        and terminal.get("partial_results_preserved") is True
        and terminal.get("retry_or_resume_permitted") is False,
        "failed terminal receipt is not the registered identity failure",
    )
    require(
        not (path / "official_evaluation.json").exists()
        and not (path / "artifact_manifest.json").exists(),
        "failed attempt contains a completed official result",
    )
    arm_files = sorted((path / "arms").glob("*.json")) if (path / "arms").exists() else []
    require(not arm_files, "failed attempt contains official arm metrics")

    artifacts = {
        "pre_open_firewall_receipt": {
            "path": str(pre_path),
            "sha256": official_v1.sha256_file(pre_path),
        },
        "official_evaluation_opened": {
            "path": str(opened_path),
            "sha256": official_v1.sha256_file(opened_path),
        },
        "terminal_status": {
            "path": str(terminal_path),
            "sha256": official_v1.sha256_file(terminal_path),
        },
    }
    return {
        "failed_official_run_dir": str(path),
        "artifacts": artifacts,
        "error": EXPECTED_FAILURE,
        "official_dataset_opened": True,
        "official_evaluation_completed": False,
        "official_arm_artifacts": 0,
        "official_metric_forwards_completed": 0,
        "candidate_checkpoint_sha256": official_v1.EXPECTED_CANDIDATE_SHA256,
        "immutable_evidence": True,
    }


def failed_attempt_unchanged(binding: Mapping[str, Any]) -> bool:
    for artifact in binding["artifacts"].values():
        path = Path(str(artifact["path"]))
        if not path.is_file() or official_v1.sha256_file(path) != artifact["sha256"]:
            return False
    failed_root = Path(str(binding["failed_official_run_dir"]))
    if (failed_root / "official_evaluation.json").exists():
        return False
    if (failed_root / "artifact_manifest.json").exists():
        return False
    return not (
        (failed_root / "arms").exists()
        and any((failed_root / "arms").glob("*.json"))
    )


def validate_split_manifest_before_open(
    value: Mapping[str, Any],
) -> tuple[list[int], list[int]]:
    require = official_v1.require
    require(value.get("schema_version") == 1, "split manifest schema mismatch")
    require(value.get("protocol") == SPLIT_PROTOCOL, "split manifest protocol mismatch")
    require(value.get("dataset") == "mcf", "split manifest dataset mismatch")
    require(value.get("seed") == 1, "split manifest seed mismatch")
    require(
        value.get("source_sha256") == official_v1.EXPECTED_MCF_SHA256,
        "split manifest source hash mismatch",
    )
    require(
        value.get("dataset_size") == EXPECTED_DATASET_SIZE,
        "split manifest dataset size mismatch",
    )
    sampling = value.get("sampling")
    require(isinstance(sampling, Mapping), "split manifest sampling block is missing")
    forget_indices = [int(x) for x in sampling.get("forget_case_ids", [])]
    retain_indices = [int(x) for x in sampling.get("retain_eval_case_ids", [])]
    require(
        sampling.get("implementation") == "sample_official_mcf_records"
        and sampling.get("order")
        == "forget sample first, then retain sample from one seeded RNG"
        and sampling.get("forget_num") == official_v1.FIXED_EVALUATION["forget_num"]
        and sampling.get("retain_eval_num")
        == official_v1.FIXED_EVALUATION["retain_num"],
        "split manifest sampling recipe mismatch",
    )
    require(
        forget_indices == official_v1.EXPECTED_CASE_IDS,
        "candidate cases differ from frozen dataset-position IDs",
    )
    require(
        len(retain_indices) == official_v1.FIXED_EVALUATION["retain_num"]
        and len(set(retain_indices)) == len(retain_indices),
        "split manifest retain indices are malformed",
    )
    half = EXPECTED_DATASET_SIZE // 2
    require(
        all(half <= index < EXPECTED_DATASET_SIZE for index in forget_indices),
        "forget dataset indices escaped the official second-half pool",
    )
    require(
        all(0 <= index < half for index in retain_indices),
        "retain dataset indices escaped the official first-half pool",
    )
    require(
        set(forget_indices).isdisjoint(retain_indices),
        "manifest-bound forget and retain indices overlap",
    )
    return forget_indices, retain_indices


def select_manifest_bound_records(
    data: Sequence[Mapping[str, Any]],
    *,
    forget_indices: Sequence[int],
    retain_indices: Sequence[int],
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Select by dataset position and independently replay the frozen sampler."""
    require = official_v1.require
    require(len(data) > 0, "official MCF dataset is empty")
    all_indices = [int(x) for x in forget_indices] + [int(x) for x in retain_indices]
    require(
        all(0 <= index < len(data) for index in all_indices),
        "split manifest contains an out-of-range dataset index",
    )
    sampled_forget, sampled_retain = sample_official_mcf_records(
        data,
        forget_num=len(forget_indices),
        retain_num=len(retain_indices),
        seed=seed,
        strict=True,
    )
    identity = {id(record): index for index, record in enumerate(data)}
    replay_forget_indices = [identity[id(record)] for record in sampled_forget]
    replay_retain_indices = [identity[id(record)] for record in sampled_retain]
    require(
        replay_forget_indices == list(forget_indices),
        "frozen forget dataset indices do not replay the official sampler",
    )
    require(
        replay_retain_indices == list(retain_indices),
        "frozen retain dataset indices do not replay the official sampler",
    )
    return (
        [normalize_record(data[index]) for index in forget_indices],
        [normalize_record(data[index]) for index in retain_indices],
    )


def _embedded_case_ids(records: Sequence[Mapping[str, Any]]) -> list[int | str | None]:
    values: list[int | str | None] = []
    for record in records:
        value = record.get("case_id")
        if isinstance(value, (int, str)):
            values.append(value)
        else:
            values.append(None)
    return values


def _recovery_arm(
    label: str,
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    writer: Any,
    bank: Any,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    ppl_text: str,
    model_path: Path,
) -> Dict[str, Any]:
    result = official_v1.evaluate_fixed_arm(
        label=label,
        model=model,
        tokenizer=tokenizer,
        writer=writer,
        bank=bank,
        forget_records=forget_records,
        retain_records=retain_records,
        ppl_text=ppl_text,
        model_path=model_path,
    )
    return {
        **result,
        "protocol": PROTOCOL,
        "evaluation_identity": "frozen_split_manifest_dataset_positions",
        "evaluator_only_recovery": True,
        "model_retry": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require = official_v1.require
    model_path = Path(args.model_path).resolve()
    training_root = Path(args.training_run_dir).resolve()
    stage1_path = Path(args.stage1_state).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    original_protocol_path = Path(args.original_protocol).resolve()
    recovery_protocol_path = Path(args.recovery_protocol).resolve()
    failed_run_dir = Path(args.failed_official_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_path = Path(__file__).resolve()
    original_evaluator_path = source_path.with_name(
        "evaluate_mcf_embedding_keyed_neuron_v3_6_2_official.py"
    )
    candidate_path = training_root / "method" / "v3_6_2_candidate_state.pt"
    split_manifest_path = stage1_path.parent.parent / "protocol" / "split_manifest.json"

    for path in (
        model_path,
        training_root,
        stage1_path,
        mcf_path,
        wikidata_dir,
        original_protocol_path,
        recovery_protocol_path,
        failed_run_dir,
        candidate_path,
        split_manifest_path,
        original_evaluator_path,
    ):
        require(path.exists(), f"required recovery input is missing: {path}")
    require(not output_dir.exists(), f"recovery output already exists: {output_dir}")
    require(output_dir != failed_run_dir, "recovery cannot overwrite the failed output")

    original_protocol = official_v1.read_json(original_protocol_path)
    official_v1.validate_protocol(
        original_protocol,
        source_path=original_evaluator_path,
    )
    recovery_protocol = official_v1.read_json(recovery_protocol_path)
    validate_recovery_protocol(
        recovery_protocol,
        source_path=source_path,
        original_protocol_path=original_protocol_path,
        original_evaluator_path=original_evaluator_path,
    )
    failed_binding = validate_failed_official_attempt(failed_run_dir)
    split_manifest = official_v1.read_json(split_manifest_path)
    forget_indices, retain_indices = validate_split_manifest_before_open(split_manifest)

    require(
        official_v1.sha256_file(candidate_path)
        == official_v1.EXPECTED_CANDIDATE_SHA256,
        "candidate hash failed before deserialization",
    )
    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    require(isinstance(state, Mapping), "candidate state is not a mapping")
    official_v1.validate_candidate_structure(state)
    lineage = official_v1.validate_training_lineage(
        state,
        training_root=training_root,
        stage1_path=stage1_path,
        candidate_path=candidate_path,
    )
    require(
        lineage["lineage"]["split_manifest_sha256"]["sha256"]
        == official_v1.sha256_file(split_manifest_path),
        "manifest used for recovery is not candidate-hash-bound",
    )

    model = None
    writer = None
    bank = None
    recovery_opened = False
    try:
        model, tokenizer, writer, bank, runtime_integrity = official_v1._load_runtime(
            model_path=model_path,
            state=state,
            expected_base_embedding_rows_sha256=str(
                lineage["base_selected_embedding_rows_sha256"]
            ),
        )
        torch.set_grad_enabled(False)
        versions_before = official_v1.parameter_versions(model)
        lm_head_before = official_v1.learner._tensor_digest(
            model.get_output_embeddings().weight
        )
        candidate_before = official_v1.sha256_file(candidate_path)
        stage1_before = official_v1.sha256_file(stage1_path)
        original_protocol_before = official_v1.sha256_file(original_protocol_path)
        recovery_protocol_before = official_v1.sha256_file(recovery_protocol_path)
        split_manifest_before = official_v1.sha256_file(split_manifest_path)
        actuator_before = official_v1.tensor_sha256(bank.down_delta)
        detector_gate_before = official_v1.tensor_sha256(bank.detector_gate_rows)
        detector_up_before = official_v1.tensor_sha256(bank.detector_up_rows)
        embedding_delta_before = official_v1.tensor_sha256(writer.delta)

        official_v1.create_fresh_output_dir(output_dir)
        pre_open = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_pre_open_firewall",
            "protocol": PROTOCOL,
            "created_at_utc": official_v1.utc_now(),
            "official_recovery_opened": False,
            "registration_boundary": (
                "after one official dataset open failed at the index-identity check; "
                "before every official metric forward"
            ),
            "original_failed_attempt": failed_binding,
            "failed_attempt_unchanged": failed_attempt_unchanged(failed_binding),
            "candidate_checkpoint_sha256": candidate_before,
            "candidate_matches_original_frozen_protocol": candidate_before
            == official_v1.EXPECTED_CANDIDATE_SHA256,
            "training_acceptance": lineage["training_acceptance"],
            "runtime_integrity": runtime_integrity,
            "split_identity": {
                "namespace": "dataset_position",
                "manifest_path": str(split_manifest_path),
                "manifest_sha256": split_manifest_before,
                "candidate_hash_bound": True,
                "forget_dataset_indices": forget_indices,
                "forget_dataset_indices_sha256": official_v1.json_sha256(
                    forget_indices
                ),
                "retain_dataset_indices_sha256": official_v1.json_sha256(
                    retain_indices
                ),
            },
            "fixed_evaluation": dict(official_v1.FIXED_EVALUATION),
            "fixed_arm_order": list(official_v1.FIXED_ARM_ORDER),
            "candidate_acceptance_thresholds": dict(official_v1.FIXED_THRESHOLDS),
            "original_protocol_sha256": original_protocol_before,
            "recovery_protocol_sha256": recovery_protocol_before,
            "original_evaluator_source_sha256": official_v1.sha256_file(
                original_evaluator_path
            ),
            "recovery_evaluator_source_sha256": official_v1.sha256_file(source_path),
            "git_head": official_v1._git_head(source_path.parents[2]),
            "gradient_mode_enabled": torch.is_grad_enabled(),
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
            "model_retry": False,
            "evaluator_only_recovery": True,
            "resume_allowed": False,
            "retry_allowed": False,
        }
        require(pre_open["failed_attempt_unchanged"] is True, "failed evidence changed")
        require(pre_open["gradient_mode_enabled"] is False, "gradient mode was not disabled")
        pre_open["all_evaluation_forwards_use_inference_mode"] = True
        official_v1.write_json(
            output_dir / "recovery_pre_open_firewall_receipt.json", pre_open
        )

        # Record the acknowledged second official-file access before reading a
        # byte. The old directory is never resumed and remains untouched.
        recovery_opened = True
        official_v1.write_json(
            output_dir / "official_recovery_opened.json",
            {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_open_event",
                "protocol": PROTOCOL,
                "opened_at_utc": official_v1.utc_now(),
                "official_recovery_opened": True,
                "official_mcf_path": str(mcf_path),
                "official_mcf_sha256": None,
                "expected_official_mcf_sha256": official_v1.EXPECTED_MCF_SHA256,
                "candidate_checkpoint_sha256": candidate_before,
                "prior_failed_official_dataset_opens": 1,
                "prior_official_metric_forwards": 0,
                "recovery_evaluation": True,
                "model_retry": False,
                "resume_or_retry_allowed": False,
            },
        )
        mcf_sha = official_v1.sha256_file(mcf_path)
        opened = official_v1.read_json(output_dir / "official_recovery_opened.json")
        official_v1.write_json(
            output_dir / "official_recovery_opened.json",
            {**opened, "official_mcf_sha256": mcf_sha},
        )
        require(
            mcf_sha == official_v1.EXPECTED_MCF_SHA256,
            "official MCF dataset hash mismatch",
        )
        data = load_mcf(mcf_path)
        require(
            isinstance(data, list) and len(data) == EXPECTED_DATASET_SIZE,
            "official MCF dataset shape mismatch",
        )
        forget_records, retain_records = select_manifest_bound_records(
            data,
            forget_indices=forget_indices,
            retain_indices=retain_indices,
            seed=int(official_v1.FIXED_EVALUATION["seed"]),
        )
        embedded_forget_case_ids = _embedded_case_ids(forget_records)
        embedded_retain_case_ids = _embedded_case_ids(retain_records)
        require(
            embedded_forget_case_ids != forget_indices,
            "registered evaluator identity diagnosis did not reproduce",
        )
        ppl_text = load_official_ppl_text(wikidata_dir)
        require(
            ppl_text is not None and ppl_text.strip(),
            "fixed official PPL text is unavailable",
        )

        opened = official_v1.read_json(output_dir / "official_recovery_opened.json")
        opened = {
            **opened,
            "split_identity_namespace": "dataset_position",
            "forget_dataset_indices": forget_indices,
            "forget_dataset_indices_sha256": official_v1.json_sha256(forget_indices),
            "retain_dataset_indices_sha256": official_v1.json_sha256(retain_indices),
            "embedded_forget_case_ids_sha256": official_v1.json_sha256(
                embedded_forget_case_ids
            ),
            "embedded_retain_case_ids_sha256": official_v1.json_sha256(
                embedded_retain_case_ids
            ),
            "dataset_position_ids_equal_embedded_case_ids": False,
            "official_sampler_replayed_exactly": True,
            "official_forget_records": len(forget_records),
            "official_retain_records": len(retain_records),
            "official_prefix_prompts_per_arm": official_v1._prefix_count(forget_records)
            + official_v1._prefix_count(retain_records),
            "ppl_text_sha256": official_v1.sha256_text(ppl_text),
            "ppl_text_characters": len(ppl_text),
        }
        official_v1.write_json(output_dir / "official_recovery_opened.json", opened)

        arms: Dict[str, Mapping[str, Any]] = {}
        arm_artifacts: Dict[str, Any] = {}
        for label in official_v1.FIXED_ARM_ORDER:
            result = _recovery_arm(
                label,
                model=model,
                tokenizer=tokenizer,
                writer=writer,
                bank=bank,
                forget_records=forget_records,
                retain_records=retain_records,
                ppl_text=ppl_text,
                model_path=model_path,
            )
            arm_path = output_dir / "arms" / f"{label}.json"
            official_v1.write_json(arm_path, result)
            arms[label] = result
            arm_artifacts[label] = {
                "path": str(arm_path),
                "sha256": official_v1.sha256_file(arm_path),
            }
            print(
                f"{label}: Eff={result['forget']['Eff']:.3f}, "
                f"Gen={result['forget']['Gen']:.3f}, "
                f"forget Spe={result['forget']['Spe']:.3f}, "
                f"retain Spe={result['retain']['Spe']:.3f}, "
                f"PPL={result['forget_PPL']:.4f}"
            )

        writer.enabled = True
        bank.enabled = True
        bank.write_enabled = True
        versions_after = official_v1.parameter_versions(model)
        version_audit = official_v1.parameter_versions_unchanged(
            versions_before, versions_after
        )
        integrity_checks = {
            "base_parameter_versions_unchanged": version_audit["passed"],
            "lm_head_bit_identical": official_v1.learner._tensor_digest(
                model.get_output_embeddings().weight
            )
            == lm_head_before,
            "candidate_checkpoint_bit_identical": official_v1.sha256_file(candidate_path)
            == candidate_before,
            "stage1_state_bit_identical": official_v1.sha256_file(stage1_path)
            == stage1_before,
            "original_protocol_bit_identical": official_v1.sha256_file(
                original_protocol_path
            )
            == original_protocol_before,
            "recovery_protocol_bit_identical": official_v1.sha256_file(
                recovery_protocol_path
            )
            == recovery_protocol_before,
            "split_manifest_bit_identical": official_v1.sha256_file(split_manifest_path)
            == split_manifest_before,
            "failed_attempt_evidence_bit_identical": failed_attempt_unchanged(
                failed_binding
            ),
            "embedding_delta_bit_identical": official_v1.tensor_sha256(writer.delta)
            == embedding_delta_before,
            "detector_gate_rows_bit_identical": official_v1.tensor_sha256(
                bank.detector_gate_rows
            )
            == detector_gate_before,
            "detector_up_rows_bit_identical": official_v1.tensor_sha256(
                bank.detector_up_rows
            )
            == detector_up_before,
            "actuator_down_delta_bit_identical": official_v1.tensor_sha256(
                bank.down_delta
            )
            == actuator_before,
            "all_model_parameters_require_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "actuator_parameter_requires_grad_false": not bank.down_delta.requires_grad,
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
        }
        integrity_passed = (
            all(
                value is True
                for key, value in integrity_checks.items()
                if key not in {"optimizer_constructed", "gradient_updates_performed"}
            )
            and integrity_checks["optimizer_constructed"] is False
            and integrity_checks["gradient_updates_performed"] == 0
        )
        integrity = {
            "passed": integrity_passed,
            "checks": integrity_checks,
            "changed_base_parameter_versions": version_audit["changed_parameters"],
        }
        acceptance = dict(
            official_v1.build_candidate_acceptance(
                arms,
                integrity_passed=integrity_passed,
            )
        )
        acceptance.update(
            {
                "kind": (
                    "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_"
                    "candidate_acceptance"
                ),
                "protocol": PROTOCOL,
                "evaluation_identity": "frozen_split_manifest_dataset_positions",
                "evaluator_only_recovery": True,
                "model_retry": False,
            }
        )
        official_result = {
            "schema_version": 1,
            "kind": (
                "mcf_embedding_keyed_neuron_v3_6_2_one_shot_official_"
                "index_identity_recovery_evaluation"
            ),
            "protocol": PROTOCOL,
            "completed_at_utc": official_v1.utc_now(),
            "official_recovery_opened": True,
            "official_recovery_completed": True,
            "candidate_checkpoint_sha256": candidate_before,
            "official_mcf_sha256": mcf_sha,
            "ppl_text_sha256": official_v1.sha256_text(ppl_text),
            "fixed_evaluation": dict(official_v1.FIXED_EVALUATION),
            "fixed_arm_order": list(official_v1.FIXED_ARM_ORDER),
            "evaluation_identity": {
                "namespace": "dataset_position",
                "source": "candidate-hash-bound frozen split manifest",
                "manifest_sha256": split_manifest_before,
                "official_sampler_replayed_exactly": True,
                "original_v1_compared_embedded_case_ids_in_error": True,
            },
            "original_failed_attempt": failed_binding,
            "arm_artifacts": arm_artifacts,
            "arms": {
                label: official_v1._arm_summary(arms[label])
                for label in official_v1.FIXED_ARM_ORDER
            },
            "candidate_behavioral_acceptance": acceptance,
            "runtime_integrity": integrity,
            "training_acceptance": lineage["training_acceptance"],
            "official_prefix_prompts_per_arm": opened[
                "official_prefix_prompts_per_arm"
            ],
            "official_prefix_prompt_arm_evaluations": opened[
                "official_prefix_prompts_per_arm"
            ]
            * len(official_v1.FIXED_ARM_ORDER),
            "official_dataset_open_processes_total": 2,
            "official_metric_evaluation_processes_total": 1,
            "prior_failed_process_metric_forwards": 0,
            "evaluator_only_recovery": True,
            "clean_first_run_claimed": False,
            "model_retry": False,
            "used_for_training_gradient_updates": False,
            "used_for_checkpoint_selection_or_early_stopping": False,
            "retry_or_resume_permitted": False,
            "candidate_or_model_artifact_mutation_detected": not integrity_passed,
            "paper_claim_readiness": {
                "candidate_official_acceptance_passed": acceptance["passed"],
                "official_result_requires_evaluator_recovery_disclosure": True,
                "matched_mlp_only_control_evaluated": False,
                "post_freeze_retain_tail_audit_completed": False,
                "latent_recovery_and_relearning_completed": False,
                "strong_unlearning_claim_ready": False,
                "interpretation": (
                    "The frozen primary candidate was evaluated once after a "
                    "disclosed pre-forward evaluator identity failure. Matched "
                    "controls and recovery/relearning endpoints remain separate "
                    "mandatory experiments."
                ),
            },
        }
        result_path = output_dir / "official_recovery_evaluation.json"
        official_v1.write_json(result_path, official_result)
        manifest_paths = [
            output_dir / "recovery_pre_open_firewall_receipt.json",
            output_dir / "official_recovery_opened.json",
            *[Path(value["path"]) for value in arm_artifacts.values()],
            result_path,
        ]
        artifact_manifest = {
            "schema_version": 1,
            "kind": (
                "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_"
                "artifact_manifest"
            ),
            "protocol": PROTOCOL,
            "artifacts": [
                {"path": str(path), "sha256": official_v1.sha256_file(path)}
                for path in manifest_paths
            ],
            "original_failed_attempt": failed_binding,
        }
        artifact_manifest_path = output_dir / "artifact_manifest.json"
        official_v1.write_json(artifact_manifest_path, artifact_manifest)
        official_v1.write_json(
            output_dir / "terminal_status.json",
            {
                "schema_version": 1,
                "kind": (
                    "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_"
                    "terminal_status"
                ),
                "protocol": PROTOCOL,
                "status": "completed",
                "official_recovery_opened": True,
                "official_recovery_completed": True,
                "candidate_behavioral_acceptance_passed": acceptance["passed"],
                "official_recovery_evaluation_sha256": official_v1.sha256_file(
                    result_path
                ),
                "artifact_manifest_sha256": official_v1.sha256_file(
                    artifact_manifest_path
                ),
                "evaluator_only_recovery": True,
                "model_retry": False,
                "clean_first_run_claimed": False,
                "retry_or_resume_permitted": False,
            },
        )
        print(
            json.dumps(
                {
                    "candidate_behavioral_acceptance_passed": acceptance["passed"],
                    "failure_reasons": acceptance["failure_reasons"],
                    "candidate_checkpoint_sha256": candidate_before,
                    "official_recovery_evaluation": str(result_path),
                    "official_recovery_evaluation_sha256": official_v1.sha256_file(
                        result_path
                    ),
                    "evaluator_only_recovery": True,
                    "model_retry": False,
                    "clean_first_run_claimed": False,
                },
                indent=2,
            )
        )
    except BaseException as exc:
        if output_dir.exists():
            official_v1.write_json(
                output_dir / "terminal_status.json",
                {
                    "schema_version": 1,
                    "kind": (
                        "mcf_embedding_keyed_neuron_v3_6_2_official_recovery_"
                        "terminal_status"
                    ),
                    "protocol": PROTOCOL,
                    "status": "failed_after_recovery_open"
                    if recovery_opened
                    else "failed_before_recovery_open",
                    "official_recovery_opened": recovery_opened,
                    "official_recovery_completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "original_failed_attempt_preserved": failed_attempt_unchanged(
                        failed_binding
                    ),
                    "partial_results_preserved": True,
                    "evaluator_only_recovery": True,
                    "model_retry": False,
                    "retry_or_resume_permitted": False,
                },
            )
        raise
    finally:
        if bank is not None:
            bank.remove()
        if writer is not None:
            writer.remove()


if __name__ == "__main__":
    main()
