#!/usr/bin/env python3
"""Resume an already-opened target-only ZeroUnlearn RWKU evaluation.

This recovery entrypoint is intentionally narrow. It accepts only evaluation
data locations and a batch size; every model, tokenizer, checkpoint, method,
protocol, dtype, seed, target, and training identity is reconstructed from the
existing experiment state and opened checkpoint receipt.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

import rwku_zerounlearn_target_only as adapter
from rwku_artifact_access import sha256_file, sha256_path
from rwku_checkpoint_receipt import (
    load_receipt,
    mark_evaluation_complete,
    verify_frozen_identities,
)


SCRIPT_PATH = Path(__file__).resolve()
RECOVERY_REASON = "strict_json_non_finite_serialization_failure"
SERIALIZATION_POLICY = "non_finite_numeric_values_to_json_null"
SMOKE_LIMIT_FIELDS = (
    "forget_eval_limit",
    "adversarial_eval_limit",
    "mia_eval_limit",
    "neighbor_eval_limit",
    "utility_eval_limit",
)


@dataclass(frozen=True)
class RecoveryContext:
    run_dir: Path
    result_path: Path
    state: Dict[str, Any]
    receipt: Dict[str, Any]
    evaluation_args: argparse.Namespace
    checkpoint: Path
    checkpoint_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_pointer_child(path: str, key: Any) -> str:
    escaped = str(key).replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}" if path else f"/{escaped}"


def normalize_non_finite_for_json(
    value: Any,
    *,
    path: str = "",
) -> Tuple[Any, List[Dict[str, str]]]:
    """Return a strict-JSON value and all non-finite replacement locations."""

    replacements: List[Dict[str, str]] = []

    def visit(item: Any, pointer: str) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): visit(child, _json_pointer_child(pointer, key))
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(child, _json_pointer_child(pointer, index))
                for index, child in enumerate(item)
            ]
        if isinstance(item, torch.Tensor):
            if item.numel() != 1:
                raise TypeError(
                    f"Non-scalar Torch tensor cannot be serialized at {pointer or '/'}"
                )
            return visit(item.detach().cpu().item(), pointer)
        if isinstance(item, np.floating):
            return visit(float(item), pointer)
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.bool_):
            return bool(item)
        if isinstance(item, float):
            classification: str | None = None
            if math.isnan(item):
                classification = "nan"
            elif math.isinf(item):
                classification = (
                    "positive_infinity" if item > 0 else "negative_infinity"
                )
            if classification is not None:
                replacements.append({"path": pointer, "original": classification})
                return None
            return item
        if item is None or isinstance(item, (bool, int, str)):
            return item
        return item

    normalized = visit(value, path)
    return normalized, replacements


def assert_no_non_finite_numeric(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_no_non_finite_numeric(
                child,
                path=_json_pointer_child(path, key),
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_non_finite_numeric(
                child,
                path=_json_pointer_child(path, index),
            )
        return
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Non-scalar tensor remains at {path or '/'}")
        assert_no_non_finite_numeric(value.detach().cpu().item(), path=path)
        return
    if isinstance(value, np.generic):
        assert_no_non_finite_numeric(value.item(), path=path)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite numeric value remains at {path or '/'}")


def atomic_write_strict_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def validate_written_result(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError("Recovered official evaluation must be a JSON object")
    assert_no_non_finite_numeric(result)
    serialization = result.get("serialization")
    if not isinstance(serialization, Mapping):
        raise ValueError("Recovered result lacks serialization metadata")
    replacements = serialization.get("replacements")
    if not isinstance(replacements, list) or serialization.get(
        "replacement_count"
    ) != len(replacements):
        raise ValueError("Recovered serialization replacement metadata is invalid")
    if serialization.get("strict_json_allow_nan") is not False:
        raise ValueError("Recovered result does not attest strict JSON")
    return result


def _read_mapping(path: Path, label: str) -> Dict[str, Any]:
    if not Path(path).is_file():
        raise ValueError(f"Missing {label}: {path}")
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _required_artifact_path(
    receipt: Mapping[str, Any],
    name: str,
) -> Path:
    identity = receipt.get("artifacts", {}).get(name)
    if not isinstance(identity, Mapping) or not identity.get("path"):
        raise ValueError(f"Checkpoint receipt is missing frozen artifact {name}")
    return Path(str(identity["path"]))


def _assert_no_confirmatory_smoke_limits(
    state: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    if receipt.get("confirmatory") is not True:
        return
    sources = (
        ("experiment state", state),
        ("method configuration", receipt.get("method_configuration", {})),
    )
    for label, source in sources:
        if not isinstance(source, Mapping):
            continue
        used = [name for name in SMOKE_LIMIT_FIELDS if source.get(name) is not None]
        limits = source.get("evaluation_limits")
        if isinstance(limits, Mapping) and any(
            value is not None for value in limits.values()
        ):
            used.append("evaluation_limits")
        if used:
            raise ValueError(
                f"Confirmatory recovery rejects smoke limits recorded in {label}: "
                + ", ".join(sorted(set(used)))
            )


def validate_recovery_preconditions(args: argparse.Namespace) -> RecoveryContext:
    run_dir = Path(args.run_dir).resolve()
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    state_path = run_dir / "experiment_state.json"
    receipt_path = run_dir / "checkpoint_receipt.json"
    result_path = run_dir / "official_evaluation.json"
    state = _read_mapping(state_path, "experiment_state.json")
    if state.get("schema_version") != adapter.STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported target-only ZeroUnlearn state schema")
    if state.get("state") != "OFFICIAL_EVALUATION_OPENED":
        raise ValueError(
            "Recovery requires state=OFFICIAL_EVALUATION_OPENED, got "
            f"{state.get('state')}"
        )
    if state.get("official_evaluation_opened") is not True:
        raise ValueError("Experiment state does not record official evaluation opening")
    if state.get("evaluation_completed_at_utc") is not None:
        raise ValueError("Experiment state already records evaluation completion")
    if result_path.exists():
        raise ValueError(f"Refusing to overwrite existing result: {result_path}")

    receipt = load_receipt(receipt_path)
    experiment_id = str(state.get("experiment_id", ""))
    if not experiment_id or run_dir.name != experiment_id:
        raise ValueError("Run directory name and experiment state ID differ")
    if receipt.get("experiment_id") != experiment_id:
        raise ValueError("Checkpoint receipt and experiment state IDs differ")
    if receipt.get("state") != "OFFICIAL_EVALUATION_OPENED":
        raise ValueError(
            "Recovery requires an opened checkpoint receipt, got "
            f"{receipt.get('state')}"
        )
    if receipt.get("official_evaluation_opened") is not True:
        raise ValueError(
            "Checkpoint receipt does not record official evaluation opening"
        )
    if receipt.get("evaluation_completed_at_utc") is not None:
        raise ValueError("Checkpoint receipt already records evaluation completion")
    opened_at = receipt.get("official_evaluation_opened_at_utc")
    if not isinstance(opened_at, str) or not opened_at:
        raise ValueError("Checkpoint receipt lacks the original opening timestamp")
    if state.get("official_evaluation_opened_at_utc") != opened_at:
        raise ValueError("State and receipt official-evaluation timestamps differ")
    _assert_no_confirmatory_smoke_limits(state, receipt)

    verify_frozen_identities(receipt)
    checkpoint_rows = receipt.get("checkpoint_paths")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 1:
        raise ValueError("Recovery requires exactly one frozen checkpoint")
    checkpoint = Path(str(checkpoint_rows[0]["path"])).resolve()
    if not checkpoint.exists():
        raise ValueError(f"Frozen checkpoint is missing: {checkpoint}")
    if state.get("checkpoint_path") != str(checkpoint):
        raise ValueError("Experiment state checkpoint path differs from receipt")
    recorded_receipt_path = state.get("checkpoint_receipt")
    if (
        recorded_receipt_path
        and Path(str(recorded_receipt_path)).resolve() != receipt_path
    ):
        raise ValueError("Experiment state checkpoint-receipt path differs from run")

    method_configuration = receipt.get("method_configuration")
    if not isinstance(method_configuration, Mapping):
        raise ValueError("Checkpoint receipt lacks method configuration")
    dtype = str(method_configuration.get("checkpoint_dtype", ""))
    if dtype not in {"bf16", "fp16", "fp32", "bfloat16", "float16", "float32"}:
        raise ValueError("Checkpoint receipt has no supported frozen checkpoint dtype")
    add_external = bool(method_configuration.get("add_retain", False))
    if bool(state.get("add_external_retain_anchors")) != add_external:
        raise ValueError("State and receipt external-retain modes differ")
    expected_method = adapter.method_label(add_external_retain_anchors=add_external)
    if state.get("method") != expected_method:
        raise ValueError("State method differs from frozen receipt method")
    if state.get("protocol_label") != receipt.get("protocol_label"):
        raise ValueError("State and receipt protocol labels differ")
    if state.get("protocol_status") != receipt.get("protocol_status"):
        raise ValueError("State and receipt protocol statuses differ")

    seed = int(state.get("seed", -1))
    target = adapter.target_for_seed(seed)
    expected_entity_id = f"rwku:{target.directory}"
    state_target = state.get("target", {})
    if (
        receipt.get("target_entity") != target.subject
        or receipt.get("target_entity_id") != expected_entity_id
        or state_target.get("subject") != target.subject
        or state_target.get("entity_id") != expected_entity_id
    ):
        raise ValueError("Frozen seed, subject, entity ID, and receipt target differ")
    if bool(state.get("confirmatory")) != bool(receipt.get("confirmatory")):
        raise ValueError("State and receipt confirmatory modes differ")

    model_path = Path(str(receipt["base_model_identity"]["path"])).resolve()
    base_artifact = _required_artifact_path(receipt, "base_model_source").resolve()
    if model_path != base_artifact:
        raise ValueError("Base model identities disagree inside checkpoint receipt")
    training_bundle = _required_artifact_path(receipt, "training_bundle").resolve()
    generator_receipt = _required_artifact_path(receipt, "generator_receipt").resolve()
    zero_hparams = _required_artifact_path(receipt, "zero_hparams").resolve()
    locked_path = _required_artifact_path(receipt, "official_locked_eval").resolve()
    if locked_path != (run_dir / "official_locked_eval.json").resolve():
        raise ValueError("Locked official-evaluation artifact is outside the run")

    external_paths = [
        Path(path).resolve() for path in state.get("external_retain_artifact_paths", [])
    ]
    evaluation_args = argparse.Namespace(
        stage="evaluate",
        experiment_id=experiment_id,
        seed=seed,
        model_path=model_path,
        model_revision=str(receipt["base_model_revision"]),
        generated_entity_fact_bundle=training_bundle,
        generator_receipt=generator_receipt,
        zero_hparams=zero_hparams,
        output_root=run_dir.parent,
        data_root=Path(args.data_root),
        wikidata_dir=Path(args.wikidata_dir),
        dtype=dtype,
        no_download=bool(args.no_download),
        confirmatory=bool(receipt.get("confirmatory")),
        add_external_retain_anchors=add_external,
        external_retain_artifact=external_paths,
        eval_batch_size=int(args.eval_batch_size),
        forget_eval_limit=None,
        adversarial_eval_limit=None,
        mia_eval_limit=None,
        neighbor_eval_limit=None,
        utility_eval_limit=None,
        dry_run=False,
    )
    adapter._verify_stage_contract(evaluation_args, state)
    adapter._verify_receipt_invocation(evaluation_args, receipt)
    if load_receipt(receipt_path)["receipt_sha256"] != receipt["receipt_sha256"]:
        raise ValueError("Checkpoint receipt changed during recovery validation")
    return RecoveryContext(
        run_dir=run_dir,
        result_path=result_path,
        state=state,
        receipt=receipt,
        evaluation_args=evaluation_args,
        checkpoint=checkpoint,
        checkpoint_sha256=sha256_path(checkpoint),
    )


def evaluate_frozen_models(context: RecoveryContext) -> Dict[str, Any]:
    args = context.evaluation_args
    receipt = context.receipt
    locked = adapter.read_artifact(
        context.run_dir / "official_locked_eval.json",
        stage="evaluate",
        evaluation=True,
        expected_role="official_locked_eval",
    )
    target, datasets, file_hashes = adapter.ensure_target_data(
        args.data_root,
        args.seed,
        allow_download=not args.no_download,
    )
    for filename, descriptor in locked["payload"]["files"].items():
        if file_hashes.get(filename) != descriptor["sha256"]:
            raise ValueError(f"Official locked evaluation file changed: {filename}")
    if target.subject != receipt["target_entity"]:
        raise ValueError("Official RWKU target differs from frozen receipt")

    held_out_level1 = list(datasets["forget_level1.json"])
    held_out_level2 = list(datasets["forget_level2.json"])
    all_answers = [
        str(row["answer"])
        for filename in (
            "forget_level1.json",
            "forget_level2.json",
            "forget_level3.json",
        )
        for row in datasets[filename]
    ]
    adapter._set_seeds(args.seed)
    base_tokenizer = adapter.load_tokenizer(args)
    base_model: Any = None
    candidate_model: Any = None
    try:
        base_model = adapter._load_evaluation_model(args, args.model_path, base=True)
        frozen_probe = adapter.build_frozen_head_probe(
            base_model,
            base_tokenizer,
            held_out_level2,
            additional_answers=all_answers,
        )
        limits = adapter._evaluation_limits(args)
        base_result = adapter.evaluate_rwku(
            method="Base model",
            model=base_model,
            tokenizer=base_tokenizer,
            subject=target.subject,
            held_out_cloze=held_out_level1,
            held_out_direct=held_out_level2,
            datasets=datasets,
            wikidata_dir=args.wikidata_dir,
            batch_size=args.eval_batch_size,
            base_retain_mean_logprobs=None,
            frozen_head_probe=frozen_probe,
            limits=limits,
            skip_ppl=False,
        )
        base_retain = base_result["retain_reference_mean_logprobs"]
        del base_model
        base_model = None
        gc.collect()
        torch.cuda.empty_cache()

        candidate_tokenizer = adapter.load_tokenizer(args, context.checkpoint)
        if candidate_tokenizer.eos_token_id != base_tokenizer.eos_token_id or len(
            candidate_tokenizer
        ) != len(base_tokenizer):
            raise ValueError("Frozen checkpoint tokenizer differs from Base tokenizer")
        candidate_model = adapter._load_evaluation_model(
            args,
            context.checkpoint,
            base=False,
        )
        candidate_result = adapter.evaluate_rwku(
            method=adapter.method_label(
                add_external_retain_anchors=args.add_external_retain_anchors
            ),
            model=candidate_model,
            tokenizer=candidate_tokenizer,
            subject=target.subject,
            held_out_cloze=held_out_level1,
            held_out_direct=held_out_level2,
            datasets=datasets,
            wikidata_dir=args.wikidata_dir,
            batch_size=args.eval_batch_size,
            base_retain_mean_logprobs=base_retain,
            frozen_head_probe=frozen_probe,
            limits=limits,
            skip_ppl=False,
        )
        candidate_result.update(
            {
                "protocol_label": adapter.TARGET_ONLY_PROTOCOL_LABEL,
                "protocol_status": receipt["protocol_status"],
                "adapter_origin": "new RWKU method extension",
                "setting5e_invoked": False,
                "lm_head_repair_invoked": False,
                "official_evaluation_opened_at_utc": receipt[
                    "official_evaluation_opened_at_utc"
                ],
            }
        )
        return {
            "method": adapter.method_label(
                add_external_retain_anchors=args.add_external_retain_anchors
            ),
            "protocol_status": receipt["protocol_status"],
            "base": base_result,
            "unlearned": candidate_result,
        }
    finally:
        if base_model is not None:
            del base_model
        if candidate_model is not None:
            del candidate_model
        gc.collect()
        torch.cuda.empty_cache()


def _failure_diagnostic(
    args: argparse.Namespace,
    exc: BaseException,
) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    state_name: Any = None
    receipt_sha256: Any = None
    try:
        state_name = _read_mapping(
            run_dir / "experiment_state.json", "experiment_state.json"
        ).get("state")
    except Exception:
        pass
    try:
        receipt_sha256 = load_receipt(run_dir / "checkpoint_receipt.json").get(
            "receipt_sha256"
        )
    except Exception:
        pass
    return {
        "status": "evaluation_recovery_failed",
        "reason": RECOVERY_REASON,
        "exception_type": exc.__class__.__name__,
        "exception_message": str(exc),
        "state_after_failure": state_name,
        "frozen_receipt_sha256_after_failure": receipt_sha256,
        "checkpoint_reused_without_modification": True,
        "official_evaluation_reopened": False,
        "state_rolled_back": False,
        "failed_at_utc": _utc_now(),
    }


def run_recovery(args: argparse.Namespace) -> Dict[str, Any]:
    context: RecoveryContext | None = None
    try:
        context = validate_recovery_preconditions(args)
        core_result = evaluate_frozen_models(context)
        if sha256_path(context.checkpoint) != context.checkpoint_sha256:
            raise ValueError("Frozen checkpoint changed during recovery evaluation")
        current_receipt = load_receipt(context.run_dir / "checkpoint_receipt.json")
        if current_receipt["receipt_sha256"] != context.receipt["receipt_sha256"]:
            raise ValueError("Opened checkpoint receipt changed during evaluation")
        verify_frozen_identities(current_receipt)

        normalized, replacements = normalize_non_finite_for_json(core_result)
        recovered_at = _utc_now()
        recovery_metadata = {
            "reason": RECOVERY_REASON,
            "checkpoint_reused_without_modification": True,
            "official_evaluation_reopened": False,
            "state_rolled_back": False,
            "method_or_hyperparameters_changed": False,
            "metric_definitions_changed": False,
            "original_official_evaluation_opened_at_utc": context.receipt[
                "official_evaluation_opened_at_utc"
            ],
            "recovery_script_sha256": sha256_file(SCRIPT_PATH),
            "frozen_receipt_sha256": context.receipt["receipt_sha256"],
            "recovered_at_utc": recovered_at,
        }
        normalized["serialization"] = {
            "policy": SERIALIZATION_POLICY,
            "strict_json_allow_nan": False,
            "replacement_count": len(replacements),
            "replacements": replacements,
        }
        normalized["evaluation_recovery"] = recovery_metadata
        assert_no_non_finite_numeric(normalized)
        atomic_write_strict_json(context.result_path, normalized)
        validated = validate_written_result(context.result_path)

        if sha256_path(context.checkpoint) != context.checkpoint_sha256:
            raise ValueError("Frozen checkpoint changed before completion transition")
        completed = mark_evaluation_complete(
            context.run_dir / "checkpoint_receipt.json",
            experiment_id=context.evaluation_args.experiment_id,
        )
        if (
            completed.get("official_evaluation_opened_at_utc")
            != context.receipt["official_evaluation_opened_at_utc"]
        ):
            raise ValueError("Completion did not preserve the opening timestamp")
        adapter._write_state(
            context.evaluation_args,
            "EVALUATION_COMPLETE",
            official_evaluation_opened=True,
            official_evaluation_opened_at_utc=context.receipt[
                "official_evaluation_opened_at_utc"
            ],
            evaluation_completed_at_utc=completed["evaluation_completed_at_utc"],
            result_path=str(context.result_path.resolve()),
            evaluation_recovery=recovery_metadata,
            serialization={
                "policy": SERIALIZATION_POLICY,
                "strict_json_allow_nan": False,
                "replacement_count": len(replacements),
            },
        )
        return validated
    except Exception as exc:
        diagnostic_path = (
            Path(args.run_dir).resolve() / "evaluation_recovery_failure.json"
        )
        try:
            atomic_write_strict_json(diagnostic_path, _failure_diagnostic(args, exc))
        except Exception:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wikidata-dir", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("RWKU recovery evaluation requires CUDA")
    result = run_recovery(args)
    print(
        json.dumps(
            {
                "state": "EVALUATION_COMPLETE",
                "result_path": str(
                    (Path(args.run_dir) / "official_evaluation.json").resolve()
                ),
                "replacement_count": result["serialization"]["replacement_count"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
