#!/usr/bin/env python3
"""Run a target-only Original ZeroUnlearn method extension on RWKU.

Upstream ZeroUnlearn has no RWKU implementation.  This wrapper is a new RWKU
request adapter around the vendored Original ZeroUnlearn implementation.  It
uses only an independently generated target-entity training artifact before
checkpoint freeze, and opens the pinned official RWKU corpus only after the
checkpoint receipt crosses the one-way evaluation boundary.

The primary method has no retain requests, no Setting 5e training, and no
LM-head repair.  An explicitly labelled external-retain ablation is available
only with separately supplied, target-independent optimization artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from build_rwku_entity_facts import official_locked_descriptor
from rwku_artifact_access import (
    TARGET_ONLY_PROTOCOL_LABEL,
    TARGET_ONLY_PROTOCOL_STATUS,
    make_artifact,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_path,
    write_artifact,
)
from rwku_checkpoint_receipt import (
    CheckpointReceiptError,
    assert_model_modification_allowed,
    create_checkpoint_receipt,
    load_receipt,
    mark_evaluation_complete,
    open_official_evaluation,
    verify_frozen_identities,
)
from rwku_data import ensure_target_data, target_for_seed
from rwku_eval import (
    build_frozen_head_probe,
    evaluate_rwku,
    format_qa_prompt,
)
import run_zerounlearn_official_mcf as zero_bridge


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]

METHOD = "Original ZeroUnlearn with RWKU target-generated entity corpus"
PROTOCOL_STATUS = "official_rwku_protocol_different_model_zerounlearn_corpus_extension"
EXTERNAL_RETAIN_LABEL = "external-retain extension"
EXTERNAL_RETAIN_STATUS_SUFFIX = "external_retain_extension"
STATE_SCHEMA_VERSION = "rwku_zerounlearn_target_only_state_v1"
REQUEST_SCHEMA_VERSION = "rwku_zerounlearn_requests_v1"
EXPECTED_LAYERS = [16, 17, 18]
EXPECTED_REWRITE_MODULE = "model.layers.{}.mlp.down_proj"
EDIT_LAYER_NUMS = 3
USE_H = False
EXTERNAL_CASE_ID_OFFSET = 1_000_000_000
FORBIDDEN_EXTERNAL_SOURCE_MARKERS = (
    "forget_level1.json",
    "forget_level2.json",
    "forget_level3.json",
    "forget_mia.json",
    "retain_mia.json",
    "neighbor_level1.json",
    "neighbor_level2.json",
    "retain_mmlu.json",
    "retain_bbh.json",
    "truthful.json",
    "triviaqa.json",
    "fluency.json",
    "official_locked_eval.json",
)

ZERO_SOURCE_RELATIVE_PATHS = tuple(zero_bridge.HASHED_ZERO_SOURCE_RELATIVE_PATHS)
EXPECTED_ZERO_SOURCE_SHA256 = dict(zero_bridge.EXPECTED_ZERO_SOURCE_SHA256)
EXPECTED_HPARAMS_SHA256 = zero_bridge.EXPECTED_HPARAMS_SHA256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "train", "evaluate"), required=True
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, choices=range(10), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generated-entity-fact-bundle", type=Path, required=True)
    parser.add_argument("--generator-receipt", type=Path, required=True)
    parser.add_argument("--zero-root", type=Path, required=True)
    parser.add_argument("--zero-hparams", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wikidata-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32", "bfloat16", "float16", "float32"),
        required=True,
    )
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--add-external-retain-anchors", action="store_true")
    parser.add_argument(
        "--external-retain-artifact",
        type=Path,
        action="append",
        default=[],
        help=(
            "Target-independent optimization_protection artifact. Required at "
            "least once by --add-external-retain-anchors."
        ),
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--forget-eval-limit", type=int)
    parser.add_argument("--adversarial-eval-limit", type=int)
    parser.add_argument("--mia-eval-limit", type=int)
    parser.add_argument("--neighbor-eval-limit", type=int)
    parser.add_argument("--utility-eval-limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the selected stage without loading or modifying a model.",
    )
    return parser


def _smoke_limit_names(args: argparse.Namespace) -> List[str]:
    return [
        name
        for name in (
            "forget_eval_limit",
            "adversarial_eval_limit",
            "mia_eval_limit",
            "neighbor_eval_limit",
            "utility_eval_limit",
        )
        if getattr(args, name) is not None
    ]


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.experiment_id).strip():
        raise ValueError("--experiment-id must be non-empty")
    if not str(args.model_revision).strip():
        raise ValueError("--model-revision must pin a non-empty revision")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive")
    for name in _smoke_limit_names(args):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.add_external_retain_anchors and not args.external_retain_artifact:
        raise ValueError(
            "--add-external-retain-anchors requires at least one explicit "
            "--external-retain-artifact"
        )
    if args.external_retain_artifact and not args.add_external_retain_anchors:
        raise ValueError(
            "--external-retain-artifact is accepted only with "
            "--add-external-retain-anchors"
        )
    if args.confirmatory and args.stage == "evaluate" and _smoke_limit_names(args):
        raise ValueError(
            "Confirmatory evaluation cannot use smoke limits: "
            + ", ".join(name.replace("_", "-") for name in _smoke_limit_names(args))
        )
    if (
        not args.dry_run
        and args.stage in {"train", "evaluate"}
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Original ZeroUnlearn training/evaluation requires CUDA; use --dry-run "
            "for CPU-only protocol validation."
        )


def method_label(*, add_external_retain_anchors: bool) -> str:
    if add_external_retain_anchors:
        return f"{METHOD} ({EXTERNAL_RETAIN_LABEL})"
    return METHOD


def protocol_status(*, add_external_retain_anchors: bool) -> str:
    if add_external_retain_anchors:
        return f"{PROTOCOL_STATUS}_{EXTERNAL_RETAIN_STATUS_SUFFIX}"
    return PROTOCOL_STATUS


def _output_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / str(args.experiment_id)


def _state_path(args: argparse.Namespace) -> Path:
    return _output_dir(args) / "experiment_state.json"


def _receipt_path(args: argparse.Namespace) -> Path:
    return _output_dir(args) / "checkpoint_receipt.json"


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _read_state(args: argparse.Namespace) -> Dict[str, Any]:
    path = _state_path(args)
    if not path.is_file():
        raise ValueError(
            f"Experiment {args.experiment_id!r} is not PREPARED; missing {path}"
        )
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported target-only ZeroUnlearn state schema")
    if state.get("experiment_id") != args.experiment_id:
        raise ValueError("Experiment state ID mismatch")
    return dict(state)


def _write_state(args: argparse.Namespace, state_name: str, **extra: Any) -> None:
    existing: Dict[str, Any] = {}
    if _state_path(args).is_file():
        existing = _read_state(args)
    order = {
        "PREPARED": 0,
        "TRAINING": 1,
        "CHECKPOINT_FROZEN": 2,
        "OFFICIAL_EVALUATION_OPENED": 3,
        "EVALUATION_COMPLETE": 4,
    }
    previous = existing.get("state")
    if previous is not None and order[state_name] < order[previous]:
        raise ValueError(
            f"Backward RWKU state transition is forbidden: {previous} -> {state_name}"
        )
    _atomic_json_write(
        _state_path(args),
        {
            **existing,
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": args.experiment_id,
            "state": state_name,
            **extra,
        },
    )


def _assert_path_identity(
    path: Path, expected_path: str, expected_sha256: str, label: str
) -> None:
    source = Path(path)
    if str(source.resolve()) != expected_path:
        raise ValueError(f"{label} path differs from PREPARED state")
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"{label} changed after PREPARED state")


def _validate_generator_receipt(
    path: Path,
    *,
    stage: str,
    target: Any,
) -> Dict[str, Any]:
    receipt = read_artifact(path, stage=stage, expected_role="generator_receipt")
    if receipt["protocol_label"] != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Generator receipt has the wrong target-only protocol label")
    if receipt["protocol_status"] != TARGET_ONLY_PROTOCOL_STATUS:
        raise ValueError("Generator receipt has the wrong source protocol status")
    payload = receipt["payload"]
    if payload.get("status") != "complete":
        raise ValueError("Generator receipt must have status=complete")
    if payload.get("official_rwku_records_accessed") is not False:
        raise ValueError("Generator receipt must attest no official RWKU record access")
    expected_entity_id = f"rwku:{target.directory}"
    if payload.get("target_entity") != target.subject:
        raise ValueError("Generator receipt target entity differs from --seed")
    if payload.get("entity_id") != expected_entity_id:
        raise ValueError("Generator receipt entity ID differs from --seed")
    bundle_sha = payload.get("final_entity_fact_bundle_sha256")
    if not isinstance(bundle_sha, str) or len(bundle_sha) != 64:
        raise ValueError("Generator receipt has no valid final bundle hash")
    return receipt


def prepare_stage(args: argparse.Namespace) -> None:
    output_dir = _output_dir(args)
    if _state_path(args).is_file() and _read_state(args).get("state") != "PREPARED":
        raise ValueError(
            "Preparation cannot replace an experiment that entered training"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    target = target_for_seed(args.seed)
    if not args.generated_entity_fact_bundle.is_file():
        raise FileNotFoundError(args.generated_entity_fact_bundle)
    if not args.generator_receipt.is_file():
        raise FileNotFoundError(args.generator_receipt)

    # The training bundle's bytes are hashed but its payload is not opened in
    # prepare. Its inner artifact hash and labels are verified in train, the
    # first stage permitted by the training_bundle envelope.
    generator = _validate_generator_receipt(
        args.generator_receipt, stage="prepare", target=target
    )
    metadata = {
        "seed": args.seed,
        "entity_id": f"rwku:{target.directory}",
        "subject": target.subject,
        "generated_training_bundle_path": str(
            args.generated_entity_fact_bundle.resolve()
        ),
        "generated_training_bundle_file_sha256": sha256_file(
            args.generated_entity_fact_bundle
        ),
        "generated_training_bundle_artifact_sha256": generator["payload"][
            "final_entity_fact_bundle_sha256"
        ],
        "generator_receipt_path": str(args.generator_receipt.resolve()),
        "generator_receipt_file_sha256": sha256_file(args.generator_receipt),
        "generator_receipt_artifact_sha256": generator["sha256"],
    }
    locked = make_artifact(
        "official_locked_eval",
        official_locked_descriptor(args.seed, include_level12=True),
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=protocol_status(
            add_external_retain_anchors=args.add_external_retain_anchors
        ),
        metadata=metadata,
    )
    write_artifact(output_dir / "official_locked_eval.json", locked)
    _write_state(
        args,
        "PREPARED",
        seed=args.seed,
        target={
            "directory": target.directory,
            "subject": target.subject,
            "entity_id": f"rwku:{target.directory}",
        },
        method=method_label(
            add_external_retain_anchors=args.add_external_retain_anchors
        ),
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=protocol_status(
            add_external_retain_anchors=args.add_external_retain_anchors
        ),
        confirmatory=bool(args.confirmatory),
        add_external_retain_anchors=bool(args.add_external_retain_anchors),
        external_retain_artifact_paths=[
            str(path.resolve()) for path in args.external_retain_artifact
        ],
        prepared_training_bundle_path=metadata["generated_training_bundle_path"],
        prepared_training_bundle_file_sha256=metadata[
            "generated_training_bundle_file_sha256"
        ],
        prepared_training_bundle_artifact_sha256=metadata[
            "generated_training_bundle_artifact_sha256"
        ],
        prepared_generator_receipt_path=metadata["generator_receipt_path"],
        prepared_generator_receipt_file_sha256=metadata[
            "generator_receipt_file_sha256"
        ],
        prepared_generator_receipt_artifact_sha256=metadata[
            "generator_receipt_artifact_sha256"
        ],
        prepare_audit={
            "official_level1_level2_level3_opened": False,
            "official_evaluation_rows_opened": False,
            "training_bundle_payload_opened": False,
            "available_hashes_and_protocol_labels_verified": True,
        },
        official_evaluation_opened=False,
    )
    print(f"PREPARED {args.experiment_id}; official RWKU evaluation remains locked")


def _normalize_space(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _is_direct_question(view: Mapping[str, Any]) -> bool:
    labels = {
        _normalize_space(view.get("prompt_style", "")).casefold(),
        _normalize_space(view.get("query_type", "")).casefold(),
    }
    return bool(
        labels & {"direct question", "canonical direct question", "direct_question"}
    )


def select_canonical_direct_question_views(
    views: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_fact: Dict[str, List[Dict[str, Any]]] = {}
    skipped: List[Dict[str, str]] = []
    for raw in views:
        view = dict(raw)
        fact_id = _normalize_space(view.get("fact_id", ""))
        if not fact_id:
            raise ValueError("Every generated training view requires fact_id")
        if view.get("training_allowed") is not True:
            raise ValueError(f"fact_id={fact_id} contains a non-training view")
        if not _is_direct_question(view):
            skipped.append(
                {
                    "fact_id": fact_id,
                    "view_id": str(view.get("view_id", "")),
                    "reason": "non_canonical_prompt_style",
                }
            )
            by_fact.setdefault(fact_id, [])
            continue
        by_fact.setdefault(fact_id, []).append(view)
    if not by_fact:
        raise ValueError("Generated training bundle has no facts")

    selected: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, str]] = []
    for fact_id in sorted(by_fact):
        candidates = by_fact[fact_id]
        if not candidates:
            raise ValueError(f"fact_id={fact_id} has no canonical direct-question view")
        ordered = sorted(
            candidates,
            key=lambda row: (
                0
                if _normalize_space(row.get("prompt_style", "")).casefold()
                == "direct question"
                else 1,
                _normalize_space(row.get("query", "")).casefold(),
                str(row.get("view_content_sha256", "")),
                str(row.get("view_id", "")),
            ),
        )
        selected.append(ordered[0])
        for duplicate in ordered[1:]:
            duplicates.append(
                {
                    "fact_id": fact_id,
                    "view_id": str(duplicate.get("view_id", "")),
                    "selected_view_id": str(ordered[0].get("view_id", "")),
                    "reason": "duplicate_direct_question_view",
                }
            )
    return selected, {
        "input_view_count": len(views),
        "fact_count": len(by_fact),
        "selected_view_count": len(selected),
        "skipped_views": skipped,
        "duplicate_views": duplicates,
        "one_request_per_fact_id": len(selected) == len(by_fact),
    }


def _prompt_template_for_view(
    tokenizer: Any,
    view: Mapping[str, Any],
    *,
    target_subject: str,
) -> str:
    subject = str(view.get("subject", ""))
    if subject != target_subject:
        raise ValueError("Generated direct-question view has the wrong target subject")
    query = str(view.get("query", ""))
    if query.count(subject) != 1:
        raise ValueError(
            "Every direct question must contain its subject exactly once before "
            "placeholder compilation"
        )
    query_template = query.replace(subject, "{}", 1)
    prompt = format_qa_prompt(
        tokenizer,
        {
            "query": query_template,
            "answer": str(view.get("canonical_sensitive_answer", "")),
            "subject": subject,
            "level": "2",
            "type": "direct question",
        },
    )
    if prompt.count("{}") != 1:
        raise ValueError(
            "Original ZeroUnlearn prompt must contain exactly one subject placeholder"
        )
    try:
        rendered = prompt.format(subject)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("ZeroUnlearn prompt is not a valid one-slot template") from exc
    if not rendered or subject not in rendered:
        raise ValueError("Compiled ZeroUnlearn prompt does not render the subject")
    return prompt


def validate_zero_unlearn_requests(
    selected_views: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    *,
    eos_token: str,
) -> None:
    if len(selected_views) != len(requests):
        raise RuntimeError("ZeroUnlearn request compilation changed the fact count")
    fact_ids = [str(view["fact_id"]) for view in selected_views]
    if len(set(fact_ids)) != len(fact_ids):
        raise RuntimeError(
            "Primary ZeroUnlearn protocol has duplicate fact_id requests"
        )
    for position, (view, request) in enumerate(zip(selected_views, requests)):
        sensitive = _normalize_space(view.get("canonical_sensitive_answer", ""))
        if request.get("case_id") != position:
            raise RuntimeError("ZeroUnlearn case IDs are not canonical fact positions")
        if request.get("subject") != view.get("subject"):
            raise RuntimeError("ZeroUnlearn request subject changed during compilation")
        if str(request.get("prompt", "")).count("{}") != 1:
            raise RuntimeError(
                "ZeroUnlearn request prompt lacks exactly one placeholder"
            )
        if request.get("target_true") != {"str": sensitive}:
            raise RuntimeError("ZeroUnlearn target_true is not the sensitive answer")
        if request.get("target_new") != {"str": eos_token}:
            raise RuntimeError("ZeroUnlearn target_new is not runtime tokenizer EOS")


def compile_zero_unlearn_requests(
    views: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    target_subject: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eos_token, eos_token_id = zero_bridge.resolve_eos_neutral_target(tokenizer)
    selected, audit = select_canonical_direct_question_views(views)
    requests: List[Dict[str, Any]] = []
    for case_id, view in enumerate(selected):
        sensitive = _normalize_space(view.get("canonical_sensitive_answer", ""))
        if not sensitive:
            raise ValueError(f"fact_id={view['fact_id']} has no sensitive answer")
        requests.append(
            {
                "case_id": case_id,
                "prompt": _prompt_template_for_view(
                    tokenizer, view, target_subject=target_subject
                ),
                "subject": str(view["subject"]),
                "target_true": {"str": sensitive},
                "target_new": {"str": eos_token},
            }
        )
    validate_zero_unlearn_requests(selected, requests, eos_token=eos_token)
    audit.update(
        {
            "selected_fact_ids": [str(view["fact_id"]) for view in selected],
            "selected_view_ids": [str(view.get("view_id", "")) for view in selected],
            "request_count": len(requests),
            "sensitive_field": "target_true.str",
            "neutral_field": "target_new.str",
            "neutral_source": "tokenizer.eos_token",
            "eos_token": eos_token,
            "eos_token_id": eos_token_id,
        }
    )
    return requests, audit


def _external_retain_requests(
    paths: Sequence[Path],
    tokenizer: Any,
    *,
    target_subject: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    source_hashes: List[Dict[str, str]] = []
    normalized_target = _normalize_space(target_subject).casefold()
    seen_content: set[str] = set()
    for path in paths:
        artifact = read_artifact(
            path,
            stage="train",
            gradient=True,
            expected_role="optimization_protection",
        )
        source_hashes.append({"path": str(path.resolve()), "sha256": sha256_path(path)})
        for wrapped in artifact["payload"].get("records", []):
            serialized = json.dumps(
                wrapped, sort_keys=True, ensure_ascii=False, default=str
            ).casefold()
            source_path = str(wrapped.get("source_path", "")).casefold()
            if any(
                marker in source_path for marker in FORBIDDEN_EXTERNAL_SOURCE_MARKERS
            ):
                raise ValueError(
                    "External retain anchor points to an official RWKU source"
                )
            if normalized_target and normalized_target in serialized:
                raise ValueError("External retain anchor is not target-independent")
            content_id = str(wrapped.get("content_sha256") or sha256_json(wrapped))
            if content_id in seen_content:
                continue
            seen_content.add(content_id)
            row = wrapped.get("record", wrapped)
            prompt = str(row.get("prompt") or row.get("query") or row.get("text") or "")
            answer: Any = (
                row.get("answer") or row.get("target_true") or row.get("target")
            )
            if isinstance(answer, Mapping):
                answer = answer.get("str")
            answer = _normalize_space(answer)
            if not prompt or not answer:
                raise ValueError(
                    "External retain rows require prompt/query/text and answer"
                )
            subject = _normalize_space(row.get("subject") or "external retain anchor")
            if subject in prompt and prompt.count(subject) == 1:
                prompt_template = prompt.replace(subject, "{}", 1)
            else:
                prompt_template = "{}\n" + prompt
            if prompt_template.count("{}") != 1:
                raise ValueError(
                    "External retain prompt must compile to one placeholder"
                )
            requests.append(
                {
                    "case_id": EXTERNAL_CASE_ID_OFFSET + len(requests),
                    "prompt": prompt_template,
                    "subject": subject,
                    "target_true": {"str": answer},
                    "target_new": {"str": answer},
                }
            )
    if not requests:
        raise ValueError("External-retain extension received no usable anchors")
    return requests, {
        "request_count": len(requests),
        "artifacts": source_hashes,
        "target_independent": True,
    }


def validate_zero_hparams_payload(payload: Mapping[str, Any]) -> None:
    if list(payload.get("layers", [])) != EXPECTED_LAYERS:
        raise ValueError(
            f"Original ZeroUnlearn requires layers {EXPECTED_LAYERS}, got "
            f"{payload.get('layers')!r}"
        )
    if payload.get("model_name") != "Llama-3.2-3B-Instruct":
        raise ValueError("ZeroUnlearn hparams are not for Llama-3.2-3B-Instruct")
    if payload.get("rewrite_module_tmp") != EXPECTED_REWRITE_MODULE:
        raise ValueError(
            "Original ZeroUnlearn must edit only transformer MLP down_proj weights"
        )
    rewrite = str(payload.get("rewrite_module_tmp", "")).casefold()
    if "embed" in rewrite or "lm_head" in rewrite:
        raise ValueError("Embeddings and LM head are forbidden edit targets")


def load_and_validate_zero_hparams(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != EXPECTED_HPARAMS_SHA256:
        raise ValueError(
            "ZeroUnlearn hparams differ from the pinned Original "
            "Llama-3.2-3B-Instruct configuration"
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("ZeroUnlearn hparams must be a JSON object")
    validate_zero_hparams_payload(payload)
    return dict(payload)


def validate_zero_sources(zero_root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    errors: List[str] = []
    for relative in ZERO_SOURCE_RELATIVE_PATHS:
        path = Path(zero_root) / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        hashes[str(path.resolve())] = digest
        expected = EXPECTED_ZERO_SOURCE_SHA256[relative]
        if digest != expected:
            errors.append(f"{relative}: expected {expected}, got {digest}")
    if errors:
        raise ValueError(
            "Vendored Original ZeroUnlearn implementation differs from the pinned source:\n- "
            + "\n- ".join(errors)
        )
    return hashes


def _validate_model_snapshot(path: Path, revision: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Pinned Base model directory does not exist: {path}")
    resolved = path.resolve()
    if "snapshots" in resolved.parts:
        snapshot_index = max(
            index for index, part in enumerate(resolved.parts) if part == "snapshots"
        )
        if (
            snapshot_index + 1 >= len(resolved.parts)
            or resolved.parts[snapshot_index + 1] != revision
        ):
            raise ValueError("--model-revision differs from the local snapshot path")


def _transformer_load_kwargs(
    args: argparse.Namespace, *, dtype: torch.dtype
) -> Dict[str, Any]:
    return {
        "revision": args.model_revision,
        "local_files_only": bool(args.no_download),
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }


def load_tokenizer(args: argparse.Namespace, source: Path | None = None) -> Any:
    from transformers import AutoTokenizer

    path = source or args.model_path
    kwargs: Dict[str, Any] = {"local_files_only": bool(args.no_download)}
    if source is None:
        kwargs["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(str(path), **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    zero_bridge.resolve_eos_neutral_target(tokenizer)
    return tokenizer


def load_fresh_base_model_for_training(args: argparse.Namespace) -> Any:
    """Load a new pinned Base instance directly in FP32 for the edit."""

    from transformers import AutoModelForCausalLM

    _validate_model_snapshot(args.model_path, args.model_revision)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        **_transformer_load_kwargs(args, dtype=torch.float32),
    ).to("cuda:0")
    model.float()
    model.eval()
    model.config.use_cache = False
    return model


def _dtype(value: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[value]


def _load_evaluation_model(
    args: argparse.Namespace,
    source: Path,
    *,
    base: bool,
) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: Dict[str, Any] = {
        "local_files_only": bool(args.no_download),
        "torch_dtype": _dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if base:
        kwargs["revision"] = args.model_revision
    model = AutoModelForCausalLM.from_pretrained(str(source), **kwargs).to("cuda:0")
    model.eval()
    model.config.use_cache = True
    return model


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tokenizer_identity(tokenizer: Any, model_path: Path) -> Dict[str, Any]:
    metadata_names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
    )
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", model_path)),
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "source_file_sha256": {
            name: sha256_file(model_path / name)
            for name in metadata_names
            if (model_path / name).is_file()
        },
    }


def _model_identity(model_path: Path, revision: str) -> Dict[str, Any]:
    _validate_model_snapshot(model_path, revision)
    return {
        "path": str(model_path.resolve()),
        "revision": revision,
        "sha256": sha256_path(model_path),
    }


def _validate_training_bundle(
    training: Mapping[str, Any],
    *,
    target: Any,
) -> List[Dict[str, Any]]:
    if training["protocol_label"] != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Training bundle has the wrong target-only protocol label")
    if training["protocol_status"] != TARGET_ONLY_PROTOCOL_STATUS:
        raise ValueError("Training bundle has the wrong source protocol status")
    metadata = training.get("metadata", {})
    if metadata.get("subject") != target.subject:
        raise ValueError("Training bundle target subject differs from --seed")
    if metadata.get("entity_id") != f"rwku:{target.directory}":
        raise ValueError("Training bundle target entity ID differs from --seed")
    views = list(training["payload"].get("views", []))
    if not views:
        raise ValueError("Generated training bundle has no views")
    for view in views:
        if view.get("source_file") != "generated_raw_corpus.json":
            raise ValueError("Target-only training accepts only generated corpus views")
        if str(view.get("level", "")) != "generated":
            raise ValueError("Target-only training view is not labelled generated")
        if view.get("training_allowed") is not True:
            raise ValueError("Target-only training bundle contains a forbidden view")
    return views


def _verify_stage_contract(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    if int(state.get("seed", -1)) != args.seed:
        raise ValueError("--seed differs from PREPARED state")
    if bool(state.get("confirmatory")) != bool(args.confirmatory):
        raise ValueError("--confirmatory differs from PREPARED state")
    if bool(state.get("add_external_retain_anchors")) != bool(
        args.add_external_retain_anchors
    ):
        raise ValueError("External-retain mode differs from PREPARED state")
    prepared_external = list(state.get("external_retain_artifact_paths", []))
    invoked_external = [str(path.resolve()) for path in args.external_retain_artifact]
    if prepared_external != invoked_external:
        raise ValueError("External-retain artifact paths differ from PREPARED state")
    _assert_path_identity(
        args.generated_entity_fact_bundle,
        str(state["prepared_training_bundle_path"]),
        str(state["prepared_training_bundle_file_sha256"]),
        "Generated training bundle",
    )
    _assert_path_identity(
        args.generator_receipt,
        str(state["prepared_generator_receipt_path"]),
        str(state["prepared_generator_receipt_file_sha256"]),
        "Generator receipt",
    )


def train_stage(args: argparse.Namespace) -> None:
    state = _read_state(args)
    if _receipt_path(args).is_file():
        assert_model_modification_allowed(
            _receipt_path(args), experiment_id=args.experiment_id
        )
    if state.get("state") != "PREPARED":
        raise ValueError(f"Training requires PREPARED, got {state.get('state')}")
    _verify_stage_contract(args, state)
    target = target_for_seed(args.seed)
    generator = _validate_generator_receipt(
        args.generator_receipt, stage="train", target=target
    )
    training = read_artifact(
        args.generated_entity_fact_bundle,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    if training["sha256"] != state["prepared_training_bundle_artifact_sha256"]:
        raise ValueError("Training bundle artifact hash differs from PREPARED state")
    if generator["sha256"] != state["prepared_generator_receipt_artifact_sha256"]:
        raise ValueError("Generator receipt artifact hash differs from PREPARED state")
    if generator["payload"]["final_entity_fact_bundle_sha256"] != training["sha256"]:
        raise ValueError("Generator receipt does not identify the training bundle")
    views = _validate_training_bundle(training, target=target)
    hparams_payload = load_and_validate_zero_hparams(args.zero_hparams)
    zero_source_hashes = validate_zero_sources(args.zero_root)
    _validate_model_snapshot(args.model_path, args.model_revision)

    tokenizer = load_tokenizer(args)
    forget_requests, request_audit = compile_zero_unlearn_requests(
        views, tokenizer, target_subject=target.subject
    )
    retain_requests: List[Dict[str, Any]] = []
    external_audit: Dict[str, Any] = {
        "enabled": False,
        "request_count": 0,
        "artifacts": [],
    }
    if args.add_external_retain_anchors:
        retain_requests, loaded_audit = _external_retain_requests(
            args.external_retain_artifact,
            tokenizer,
            target_subject=target.subject,
        )
        external_audit = {"enabled": True, **loaded_audit}
    add_retain = bool(args.add_external_retain_anchors)

    request_manifest = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "method": method_label(add_external_retain_anchors=add_retain),
        "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": protocol_status(add_external_retain_anchors=add_retain),
        "seed": args.seed,
        "target": {
            "subject": target.subject,
            "entity_id": f"rwku:{target.directory}",
        },
        "primary_target_requests": forget_requests,
        "primary_request_audit": request_audit,
        "external_retain": external_audit,
        "add_retain": add_retain,
        "setting5e_invoked": False,
        "lm_head_repair_invoked": False,
    }
    request_manifest_path = _output_dir(args) / "compiled_requests.json"
    _atomic_json_write(request_manifest_path, request_manifest)
    if args.dry_run:
        _atomic_json_write(
            _output_dir(args) / "train_dry_run.json",
            {
                "status": "validated_without_model_load",
                "request_manifest_sha256": sha256_file(request_manifest_path),
                "zero_source_sha256": zero_source_hashes,
                "hparams_sha256": sha256_file(args.zero_hparams),
                "fresh_base_model_loaded": False,
            },
        )
        print(
            f"Validated train dry-run for {args.experiment_id}; state remains PREPARED"
        )
        return

    _write_state(args, "TRAINING", official_evaluation_opened=False)
    _set_seeds(args.seed)
    model = load_fresh_base_model_for_training(args)
    params_class, apply_unl_to_model = zero_bridge.import_original_zerounlearn(
        args.zero_root
    )
    hparams = params_class.from_json(args.zero_hparams)
    if list(hparams.layers) != EXPECTED_LAYERS:
        raise RuntimeError(f"Runtime ZeroUnlearn layers must be {EXPECTED_LAYERS}")
    if hparams.rewrite_module_tmp != EXPECTED_REWRITE_MODULE:
        raise RuntimeError("Runtime ZeroUnlearn edit target must be MLP down_proj")

    with zero_bridge.working_directory(SEMANTIC_ROOT):
        edited_model, original_weights = apply_unl_to_model(
            model=model,
            tok=tokenizer,
            retain_requests=deepcopy(retain_requests),
            unlearn_requests=deepcopy(forget_requests),
            hparams=hparams,
            copy=False,
            return_orig_weights=False,
            cache_template=None,
            save_path=None,
            add_retain=add_retain,
            edit_layer_nums=EDIT_LAYER_NUMS,
            use_h=USE_H,
        )
    del original_weights
    edited_model.to(dtype=_dtype(args.dtype))
    edited_model.eval()
    edited_model.config.use_cache = True

    checkpoint = _output_dir(args) / "checkpoint"
    if checkpoint.exists():
        raise ValueError(f"Refusing to replace existing checkpoint: {checkpoint}")
    checkpoint.mkdir(parents=True)
    edited_model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)

    tokenizer_identity = _tokenizer_identity(tokenizer, args.model_path)
    base_identity = _model_identity(args.model_path, args.model_revision)
    method_configuration = {
        "method": method_label(add_external_retain_anchors=add_retain),
        "dataset_adapter": "new RWKU target-generated entity corpus extension",
        "protocol_status": protocol_status(add_external_retain_anchors=add_retain),
        "algorithm_entrypoint": "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model",
        "compute_dtype": "float32",
        "checkpoint_dtype": args.dtype,
        "layers": EXPECTED_LAYERS,
        "edit_layer_nums": EDIT_LAYER_NUMS,
        "rewrite_module_tmp": EXPECTED_REWRITE_MODULE,
        "edited_parameter_family": "transformer MLP down_proj weights",
        "embeddings_modified": False,
        "lm_head_modified": False,
        "setting5e_invoked": False,
        "lm_head_repair_invoked": False,
        "add_retain": add_retain,
        "external_retain_extension": add_retain,
        "primary_request_count": len(forget_requests),
        "primary_fact_ids": request_audit["selected_fact_ids"],
        "request_manifest_path": str(request_manifest_path.resolve()),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "hparams_payload_sha256": sha256_json(hparams_payload),
        "zero_source_sha256": zero_source_hashes,
    }
    implementation_files = [
        SCRIPT_PATH,
        SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
        SEMANTIC_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
        SEMANTIC_ROOT / "scripts" / "rwku_data.py",
        SEMANTIC_ROOT / "scripts" / "rwku_eval.py",
        SEMANTIC_ROOT / "scripts" / "run_zerounlearn_official_mcf.py",
        *[args.zero_root / relative for relative in ZERO_SOURCE_RELATIVE_PATHS],
    ]
    receipt = create_checkpoint_receipt(
        destination=_receipt_path(args),
        experiment_id=args.experiment_id,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=protocol_status(add_external_retain_anchors=add_retain),
        target_entity=target.subject,
        target_entity_id=f"rwku:{target.directory}",
        base_model_identity=base_identity,
        base_model_revision=args.model_revision,
        tokenizer_identity=tokenizer_identity,
        checkpoint_paths=[checkpoint],
        training_bundle_path=args.generated_entity_fact_bundle,
        optimization_protection_path=(
            args.external_retain_artifact[0] if args.external_retain_artifact else None
        ),
        mcf_retain_optimization_paths=list(args.external_retain_artifact),
        mcf_repair_gate_paths=[],
        matched_protection_train_path=None,
        matched_protection_gate_path=None,
        method_configuration=method_configuration,
        implementation_files=implementation_files,
        sampler_provenance={
            "selection": "one canonical direct-question view per fact_id",
            **request_audit,
        },
        generator_receipt_path=args.generator_receipt,
        official_locked_eval_path=_output_dir(args) / "official_locked_eval.json",
        confirmatory=args.confirmatory,
        additional_artifact_paths={
            "base_model_source": args.model_path,
            "zero_hparams": args.zero_hparams,
            "compiled_requests": request_manifest_path,
        },
    )
    _write_state(
        args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(_receipt_path(args).resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        checkpoint_path=str(checkpoint.resolve()),
        fresh_base_model_loaded_for_training=True,
        zero_unlearn_compute_dtype="float32",
        official_evaluation_opened=False,
    )
    del edited_model, model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"CHECKPOINT_FROZEN {args.experiment_id}; receipt={_receipt_path(args)}")


def _verify_receipt_invocation(
    args: argparse.Namespace,
    receipt: Mapping[str, Any],
) -> None:
    expected_status = protocol_status(
        add_external_retain_anchors=args.add_external_retain_anchors
    )
    if receipt.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Checkpoint receipt has the wrong target-only protocol label")
    if receipt.get("protocol_status") != expected_status:
        raise ValueError("Checkpoint receipt has the wrong method-extension status")
    if receipt.get("base_model_revision") != args.model_revision:
        raise ValueError("--model-revision differs from checkpoint receipt")
    if bool(receipt.get("confirmatory")) != bool(args.confirmatory):
        raise ValueError("--confirmatory differs from checkpoint receipt")
    target = target_for_seed(args.seed)
    if receipt.get("target_entity") != target.subject:
        raise ValueError("Checkpoint receipt target differs from --seed")
    artifact_paths = receipt["artifacts"]
    expected_paths = {
        "training_bundle": args.generated_entity_fact_bundle,
        "generator_receipt": args.generator_receipt,
        "base_model_source": args.model_path,
        "zero_hparams": args.zero_hparams,
    }
    for name, path in expected_paths.items():
        identity = artifact_paths.get(name)
        if not identity or identity.get("path") != str(path.resolve()):
            raise ValueError(f"{name} invocation path differs from checkpoint receipt")
    base_identity = receipt["base_model_identity"]
    if base_identity.get("path") != str(args.model_path.resolve()):
        raise ValueError("Base model invocation path differs from checkpoint receipt")
    if base_identity.get("revision") != args.model_revision:
        raise ValueError(
            "Base model invocation revision differs from checkpoint receipt"
        )
    tokenizer_source_hashes = _tokenizer_identity_from_path(args.model_path)
    if tokenizer_source_hashes != receipt["tokenizer_identity"].get(
        "source_file_sha256", {}
    ):
        raise ValueError("Tokenizer source changed after checkpoint freeze")


def _tokenizer_identity_from_path(model_path: Path) -> Dict[str, str]:
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
    )
    return {
        name: sha256_file(model_path / name)
        for name in names
        if (model_path / name).is_file()
    }


def _evaluation_limits(args: argparse.Namespace) -> Dict[str, int]:
    pairs = {
        "forget": args.forget_eval_limit,
        "adversarial": args.adversarial_eval_limit,
        "mia": args.mia_eval_limit,
        "neighbor": args.neighbor_eval_limit,
        "utility": args.utility_eval_limit,
    }
    return {key: int(value) for key, value in pairs.items() if value is not None}


def evaluate_stage(args: argparse.Namespace) -> None:
    state = _read_state(args)
    if state.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError(
            f"Evaluation requires CHECKPOINT_FROZEN, got {state.get('state')}"
        )
    _verify_stage_contract(args, state)
    receipt = load_receipt(_receipt_path(args))
    _verify_receipt_invocation(args, receipt)
    if receipt.get("confirmatory") and _smoke_limit_names(args):
        raise ValueError("Confirmatory evaluation cannot use smoke limits")
    if args.dry_run:
        verify_frozen_identities(receipt)
        _atomic_json_write(
            _output_dir(args) / "evaluate_dry_run.json",
            {
                "status": "frozen_receipt_verified_without_official_data_open",
                "receipt_sha256": receipt["receipt_sha256"],
                "official_evaluation_opened": False,
            },
        )
        print(
            f"Validated evaluate dry-run for {args.experiment_id}; official data remains locked"
        )
        return

    # This atomic receipt transition is deliberately the last operation before
    # opening the locked descriptor and then the raw official record files.
    opened = open_official_evaluation(
        _receipt_path(args), experiment_id=args.experiment_id
    )
    _write_state(
        args,
        "OFFICIAL_EVALUATION_OPENED",
        official_evaluation_opened=True,
        official_evaluation_opened_at_utc=opened["official_evaluation_opened_at_utc"],
    )
    locked = read_artifact(
        _output_dir(args) / "official_locked_eval.json",
        stage="evaluate",
        evaluation=True,
        expected_role="official_locked_eval",
    )
    target, datasets, file_hashes = ensure_target_data(
        args.data_root,
        args.seed,
        allow_download=not args.no_download,
    )
    for filename, descriptor in locked["payload"]["files"].items():
        if file_hashes.get(filename) != descriptor["sha256"]:
            raise ValueError(f"Official locked evaluation file changed: {filename}")

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
    _set_seeds(args.seed)
    base_tokenizer = load_tokenizer(args)
    base_model = _load_evaluation_model(args, args.model_path, base=True)
    frozen_probe = build_frozen_head_probe(
        base_model,
        base_tokenizer,
        held_out_level2,
        additional_answers=all_answers,
    )
    limits = _evaluation_limits(args)
    base_result = evaluate_rwku(
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
    gc.collect()
    torch.cuda.empty_cache()

    checkpoint = Path(receipt["checkpoint_paths"][0]["path"])
    candidate_tokenizer = load_tokenizer(args, checkpoint)
    if candidate_tokenizer.eos_token_id != base_tokenizer.eos_token_id or len(
        candidate_tokenizer
    ) != len(base_tokenizer):
        raise ValueError("Frozen checkpoint tokenizer differs from Base tokenizer")
    candidate_model = _load_evaluation_model(args, checkpoint, base=False)
    candidate_result = evaluate_rwku(
        method=method_label(
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
            "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
            "protocol_status": receipt["protocol_status"],
            "adapter_origin": "new RWKU method extension",
            "setting5e_invoked": False,
            "lm_head_repair_invoked": False,
            "official_evaluation_opened_at_utc": opened[
                "official_evaluation_opened_at_utc"
            ],
        }
    )
    result_path = _output_dir(args) / "official_evaluation.json"
    _atomic_json_write(
        result_path,
        {
            "method": method_label(
                add_external_retain_anchors=args.add_external_retain_anchors
            ),
            "protocol_status": receipt["protocol_status"],
            "base": base_result,
            "unlearned": candidate_result,
        },
    )
    del candidate_model
    gc.collect()
    torch.cuda.empty_cache()
    completed = mark_evaluation_complete(
        _receipt_path(args), experiment_id=args.experiment_id
    )
    _write_state(
        args,
        "EVALUATION_COMPLETE",
        official_evaluation_opened=True,
        evaluation_completed_at_utc=completed["evaluation_completed_at_utc"],
        result_path=str(result_path.resolve()),
    )
    print(f"EVALUATION_COMPLETE {args.experiment_id}; result={result_path}")


def main() -> None:
    args = build_parser().parse_args()
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    validate_args(args)
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "train":
        train_stage(args)
    else:
        evaluate_stage(args)


if __name__ == "__main__":
    main()
