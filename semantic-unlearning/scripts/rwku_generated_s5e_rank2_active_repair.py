#!/usr/bin/env python3
"""Staged RWKU generated-corpus Setting 5e + rank-2 active repair.

This is an isolated RWKU method extension.  It applies the preserved 600-step
Setting 5e recipe to a frozen target-generated entity corpus, discovers active
points only in that corpus, and uses the generic MCF rank-2 LM-head repair.
Official RWKU rows remain inaccessible until the selected checkpoint receipt
has been frozen and atomically opened for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import rwku_experiment as legacy
import rwku_setting5e_utility_controlled as utility
from build_rwku_entity_facts import official_locked_descriptor
from build_rwku_matched_protection import build_matched_protection
from rwku_artifact_access import (
    TARGET_ONLY_PROTOCOL_LABEL,
    make_artifact,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_path,
    write_artifact,
)
from rwku_checkpoint_receipt import (
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
    evaluate_perplexity,
    final_hidden_states,
    generate_completions,
    load_wikidata_text,
    recovery_success,
    score_completions,
)
from rwku_fact_sampler import build_fact_cycle_plan, exposure_report, plan_sha256
from rwku_rowwise_active_repair import tokenizer_special_ids


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
METHOD = "Setting 5e @600 + protected rank-2 active LM-head repair on RWKU generated corpus"
SETTING5_METHOD = "Setting 5e @600 on RWKU generated corpus before rank-2 repair"
PROTOCOL_STATUS = "rwku_target_generated_s5e600_rank2_active_repair_method_extension"
STATE_SCHEMA = "rwku_generated_s5e_rank2_active_state_v1"
CONFIG_SCHEMA = "rwku_generated_s5e_rank2_active_configuration_v1"

# Frozen MCF rank-2 recipe.  Keep these constants explicit and independently
# testable; PROTECTED_PROJECTION_RANK is informational and is not repair_rank.
SETTING5_STEPS = 600
SETTING5_BATCH_SIZE = 1
SETTING5_RETAIN_BATCH_SIZE = 4
SETTING5_LR = 1e-4
SETTING5_FORGET_WEIGHT = 2.0
SETTING5_RETAIN_WEIGHT = 1.0
SETTING5_FORGET_MARGIN = 1.0
SETTING5_NEW_TRUE_ALPHA = 0.75
SETTING5_NEW_RETAIN_ALPHA = 0.50
SETTING5_NEW_TRUE_RETAIN_ALPHA = 0.25
REPAIR_RANK = 2
REPAIR_STEPS = 100
REPAIR_LR = 5e-3
ACTIVE_MARGIN = 0.25
HINGE_WEIGHT = 2.0
DELTA_L2_LAMBDA = 1e-4
RETAIN_KL_MU = 0.1
RETAIN_CALIBRATION_NUM = 200
PROJECT_AWAY_RETAIN_HIDDEN = True
MIN_RELOADED_FORGET_MARGIN = 0.1
PROTECTED_PROJECTION_RANK = 256  # legacy RWKU concept; deliberately unused
CANDIDATE_SCALES = (
    1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875,
    0.125, 0.09375, 0.0625, 0.046875, 0.03125, 0.015625,
    0.0078125, 0.0,
)

OFFICIAL_MARKERS = (
    "forget_level1.json", "forget_level2.json", "forget_level3.json",
    "forget_mia.json", "retain_mia.json", "neighbor_level1.json",
    "neighbor_level2.json", "retain_mmlu.json", "retain_bbh.json",
    "truthful.json", "triviaqa.json", "fluency.json",
    "official_evaluation.json", "paper_rescore",
)
CORPUS_FILENAMES = (
    "generated_raw_corpus.json", "generated_entity_fact_catalog.json",
    "generated_training_bundle.json", "generator_receipt.json",
    "generated_atomic_facts.json", "generation_diagnostics.json",
    "corpus_sha256_manifest.txt",
)


def utc_now() -> str:
    return utility.utc_now()


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.experiment_id


def state_path(args: argparse.Namespace) -> Path:
    return run_dir(args) / "experiment_state.json"


def read_state(args: argparse.Namespace) -> Dict[str, Any]:
    with state_path(args).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != STATE_SCHEMA:
        raise ValueError("Unsupported experiment state schema")
    if value.get("experiment_id") != args.experiment_id:
        raise ValueError("Experiment-state ID differs from invocation")
    return value


def write_state(args: argparse.Namespace, state: str, **extra: Any) -> None:
    previous: Dict[str, Any] = {}
    if state_path(args).exists():
        previous = read_state(args)
    value = {
        **previous,
        "schema_version": STATE_SCHEMA,
        "experiment_id": args.experiment_id,
        "output_root": str(Path(args.output_root).resolve()),
        "state": state,
        "updated_at_utc": utc_now(),
        **extra,
    }
    utility.atomic_json_write(state_path(args), value)


def reject_official_path(path: Path, *, label: str) -> None:
    lowered = str(Path(path)).lower()
    if any(marker in lowered for marker in OFFICIAL_MARKERS):
        raise ValueError(f"{label} cannot reference official/evaluation RWKU data: {path}")


def validate_active_provenance(
    point: utility.TrainingPoint,
    *,
    bundle_sha256: str,
) -> Dict[str, Any]:
    if not point.fact_id or not point.view_id or not point.source_record_sha256:
        raise ValueError("Active points require generated-corpus fact/view/source provenance")
    if point.prompt_style not in {
        "direct question", "cloze", "deterministic paraphrase", "forced-prefix"
    }:
        raise ValueError(f"Non-training-visible active prompt style: {point.prompt_style!r}")
    if any(marker in point.source_record_sha256.lower() for marker in OFFICIAL_MARKERS):
        raise ValueError("Official RWKU rows cannot become active repair points")
    return {
        "fact_id": point.fact_id,
        "view_id": point.view_id,
        "prompt_style": point.prompt_style,
        "answer_alias": point.sensitive_answer,
        "source_record_sha256": point.source_record_sha256,
        "generated_training_bundle_sha256": bundle_sha256,
        "active_source": "target_generated_entity_fact_views",
    }


def rank2_delta_module(
    *, n_rows: int, hidden_size: int, active_hidden: torch.Tensor,
    retained_basis: Optional[torch.Tensor], device: torch.device,
) -> active.SelectedRowDelta:
    projected = active.project_rows_away(active_hidden.float(), retained_basis)
    basis = active.orthonormal_row_basis(projected, max_rank=REPAIR_RANK)
    if basis.shape[0] > REPAIR_RANK:
        raise AssertionError("Shared repair basis exceeded repair_rank=2")
    return active.SelectedRowDelta(
        n_rows, hidden_size, direction_basis=basis,
        retained_basis=retained_basis, device=device,
    )


def exclude_special_rows(tokenizer: Any, token_ids: Iterable[int]) -> Tuple[List[int], List[int]]:
    special = set(tokenizer_special_ids(tokenizer))
    requested = {int(value) for value in token_ids}
    return sorted(requested - special), sorted(requested & special)


def protected_answer_row_ids(
    tokenizer: Any, examples: Sequence[gagd.Example]
) -> set[int]:
    return {
        token_id
        for example in examples
        for token_id in utility._completion_token_ids(tokenizer, example.answer)
    }


def configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "method": METHOD,
        "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": PROTOCOL_STATUS,
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "model_path": str(Path(args.model_path).resolve()),
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "device_map": "single",
        "setting5": {
            "steps": SETTING5_STEPS,
            "batch_size": SETTING5_BATCH_SIZE,
            "retain_batch_size": SETTING5_RETAIN_BATCH_SIZE,
            "emb_lm_lr": SETTING5_LR,
            "forget_weight": SETTING5_FORGET_WEIGHT,
            "retain_weight": SETTING5_RETAIN_WEIGHT,
            "forget_loss_type": "mcf_margin",
            "forget_margin": SETTING5_FORGET_MARGIN,
            "optimizer": "adamw",
            "sampling_strategy": "balanced_fact_cycle_epoch",
            "post_training_new_true_alpha": SETTING5_NEW_TRUE_ALPHA,
            "post_training_new_retain_alpha": SETTING5_NEW_RETAIN_ALPHA,
            "post_training_new_true_retain_alpha": SETTING5_NEW_TRUE_RETAIN_ALPHA,
        },
        "active_repair": {
            "repair_rank": REPAIR_RANK,
            "repair_steps": REPAIR_STEPS,
            "repair_lr": REPAIR_LR,
            "repair_optimizer": "adamw",
            "active_margin": ACTIVE_MARGIN,
            "hinge_weight": HINGE_WEIGHT,
            "delta_l2_lambda": DELTA_L2_LAMBDA,
            "retain_kl_mu": RETAIN_KL_MU,
            "retain_calibration_num": RETAIN_CALIBRATION_NUM,
            "project_away_retain_hidden": PROJECT_AWAY_RETAIN_HIDDEN,
            "min_reloaded_forget_margin": MIN_RELOADED_FORGET_MARGIN,
            "candidate_scales": list(CANDIDATE_SCALES),
            "legacy_rwku_projection_rank": PROTECTED_PROJECTION_RANK,
            "legacy_rwku_projection_rank_used": False,
        },
        "protection_gates": {
            "full_retain_probability_ratio": [0.995, 1.005],
            "geometric_retain_probability_ratio": [0.98, 1.02],
            "maximum_retain_kl_increase": 0.01,
            "minimum_protected_answer_probability_ratio": 0.999,
            "maximum_protected_top1_changes": 0,
            "maximum_proxy_ppl_ratio": 1.02,
            "nonselected_rows_must_equal_setting5": True,
        },
        "matched_protection_coverage_policy": "allow_unmatched_generated_target_keys_but_audit",
        "official_evaluation_access_before_freeze": "forbidden",
        "generated_bundle": str(Path(args.generated_training_bundle).resolve()),
        "generator_receipt": str(Path(args.generator_receipt).resolve()),
        "mcf_path": str(Path(args.mcf_path).resolve()),
        "mcf_retain_num": int(args.mcf_retain_num),
        "mcf_gate_num": int(args.mcf_gate_num),
        "protection_sources": [str(Path(path).resolve()) for path in args.protection_source],
        "protection_vocabulary": (
            str(Path(args.protection_vocabulary).resolve())
            if args.protection_vocabulary is not None else None
        ),
        "wikidata_dir": str(Path(args.wikidata_dir).resolve()),
        "official_data_root": str(Path(args.data_root).resolve()),
        "eval_batch_size": int(args.eval_batch_size),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "no_download": bool(args.no_download),
    }


def corpus_identities(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    corpus = Path(args.generated_training_bundle).parent
    values: Dict[str, Dict[str, str]] = {}
    for name in CORPUS_FILENAMES:
        path = corpus / name
        if not path.is_file():
            raise FileNotFoundError(path)
        values[name] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return values


def verify_corpus_manifest(args: argparse.Namespace) -> Dict[str, str]:
    corpus = Path(args.generated_training_bundle).parent
    manifest = corpus / "corpus_sha256_manifest.txt"
    declared: Dict[str, str] = {}
    for number, raw_line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.replace(" *", "  ").split()
        if len(fields) != 2:
            raise ValueError(f"Malformed corpus SHA-256 manifest line {number}")
        if len(fields[0]) == 64:
            digest, name = fields
        elif len(fields[1]) == 64:
            name, digest = fields
        else:
            raise ValueError(f"Corpus manifest line {number} has no SHA-256 digest")
        name = name.lstrip("*")
        member = Path(name)
        candidates = (
            [member] if member.is_absolute()
            else [corpus / member, Path.cwd() / member]
        )
        valid_paths: List[Path] = []
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(corpus.resolve())
            except ValueError:
                continue
            if resolved.is_file():
                valid_paths.append(resolved)
        if len(valid_paths) != 1:
            raise ValueError(f"Unsafe or ambiguous corpus manifest member: {name!r}")
        path = valid_paths[0]
        name = path.name
        if name == manifest.name:
            raise ValueError("Corpus manifest cannot self-attest")
        if not path.is_file() or sha256_file(path) != digest.lower():
            raise ValueError(f"Corpus manifest identity mismatch: {name}")
        declared[name] = digest.lower()
    required = set(CORPUS_FILENAMES) - {manifest.name}
    if set(declared) != required:
        raise ValueError(
            "Corpus manifest members differ from the frozen required corpus files"
        )
    return declared


def _setting5_args(args: argparse.Namespace, batches: Sequence[Sequence[gagd.Example]]) -> argparse.Namespace:
    # train_mode consumes this compatibility namespace.  Scientific values are
    # constants, not CLI-tunable parameters.
    values = vars(args).copy()
    values.update({
        "dataset": "mcf", "mode": legacy.SETTING5_MODE,
        "steps": SETTING5_STEPS, "batch_size": SETTING5_BATCH_SIZE,
        "retain_batch_size": SETTING5_RETAIN_BATCH_SIZE,
        "emb_lm_lr": SETTING5_LR, "emb_lm_optimizer": "adamw",
        "forget_weight": SETTING5_FORGET_WEIGHT,
        "retain_weight": SETTING5_RETAIN_WEIGHT,
        "forget_loss_type": "mcf_margin", "forget_margin": SETTING5_FORGET_MARGIN,
        "sampling_strategy": "epoch", "weight_decay": 0.0, "grad_clip": 1.0,
        "kl_retain_weight": 0.0, "save_model": False,
        "post_training_new_true_alpha": SETTING5_NEW_TRUE_ALPHA,
        "post_training_new_retain_alpha": SETTING5_NEW_RETAIN_ALPHA,
        "post_training_new_true_retain_alpha": SETTING5_NEW_TRUE_RETAIN_ALPHA,
        "post_training_excluded_token_ids": (),
        "precomputed_forget_batches": list(batches),
    })
    return argparse.Namespace(**values)


def _mcf_sampled_records(
    records: Sequence[Mapping[str, Any]], examples: Sequence[gagd.Example]
) -> List[active.SampledMCFRecord]:
    return [
        active.SampledMCFRecord(
            record_index=index, sampled_position=index, example=example,
            raw_record=dict(record), rewrite_prompt=example.prompt,
            paraphrase_prompts=tuple(example.paraphrase_prompts),
            target_new=example.target_new, target_true=example.target_true,
        )
        for index, (record, example) in enumerate(zip(records, examples))
    ]


def _point_instances(points: Sequence[utility.TrainingPoint]) -> List[active.MCFPromptInstance]:
    return [
        active.MCFPromptInstance(
            record_index=index, sampled_position=index,
            prompt_type=point.prompt_style, prompt_index=0, prompt=point.prompt,
            target_new=point.sensitive_answer, target_true=point.neutral_answer,
        )
        for index, point in enumerate(points)
    ]


@torch.no_grad()
def _generated_metrics(
    model: nn.Module, tokenizer: Any, points: Sequence[utility.TrainingPoint],
    *, batch_size: int,
) -> Dict[str, Any]:
    prompts = [row.prompt for row in points]
    answers = [row.sensitive_answer for row in points]
    outputs = generate_completions(model, tokenizer, prompts, batch_size=batch_size)
    scores = score_completions(
        model, tokenizer, list(zip(prompts, answers)), batch_size=batch_size
    )
    by_style: Dict[str, List[bool]] = {}
    for point, output in zip(points, outputs):
        by_style.setdefault(point.prompt_style, []).append(
            recovery_success(output, point.sensitive_answer)
        )
    probabilities = [math.exp(score.mean_logprob) for score in scores]
    return {
        "direct_recovery": _percentage(by_style.get("direct question", [])),
        "cloze_recovery": _percentage(by_style.get("cloze", [])),
        "paraphrase_recovery": _percentage(by_style.get("deterministic paraphrase", [])),
        "forced_prefix_recovery": _percentage(by_style.get("forced-prefix", [])),
        "geometric_answer_probability": (
            math.exp(sum(score.mean_logprob for score in scores) / len(scores))
            if scores else None
        ),
        "mean_answer_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "prompt_count": len(points),
    }


def _percentage(values: Sequence[bool]) -> Optional[float]:
    return 100.0 * sum(values) / len(values) if values else None


@torch.no_grad()
def _protection_snapshot(
    model: nn.Module,
    tokenizer: Any,
    examples: Sequence[gagd.Example],
    selected_ids: Sequence[int],
    *,
    batch_size: int,
) -> Dict[str, Any]:
    """Cache only target-independent protection decisions and selected logits."""

    pairs = [(example.prompt, example.answer) for example in examples]
    scores = score_completions(model, tokenizer, pairs, batch_size=batch_size)
    device = next(model.parameters()).device
    output = model.get_output_embeddings()
    if output is None:
        raise ValueError("Model has no LM head")
    selected = torch.tensor(selected_ids, dtype=torch.long, device=device)
    top1: List[int] = []
    selected_logits: List[List[float]] = []
    for prompt, answer in pairs:
        prompt_ids = utility._token_ids(tokenizer, prompt)
        answer_ids = utility._completion_token_ids(tokenizer, answer)
        sequence = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
        mask = torch.ones_like(sequence)
        hidden = final_hidden_states(model, input_ids=sequence, attention_mask=mask)[0]
        positions = torch.arange(
            len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1,
            dtype=torch.long, device=device,
        )
        logits = output(hidden.index_select(0, positions)).float()
        top1.extend(int(value) for value in logits.argmax(dim=-1).detach().cpu())
        if selected_ids:
            selected_logits.extend(
                logits.index_select(-1, selected).detach().cpu().tolist()
            )
    return {
        "sum_logprob": [score.sum_logprob for score in scores],
        "mean_logprob": [score.mean_logprob for score in scores],
        "token_count": [score.token_count for score in scores],
        "top1": top1,
        "selected_logits": selected_logits,
    }


def _protection_metrics(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Dict[str, Any]:
    if baseline["token_count"] != candidate["token_count"]:
        raise ValueError("Protection snapshot tokenization changed")
    if len(baseline["mean_logprob"]) != len(candidate["mean_logprob"]):
        raise ValueError("Protection snapshot row count changed")
    mean_diffs = [
        float(after) - float(before)
        for before, after in zip(baseline["mean_logprob"], candidate["mean_logprob"])
    ]
    full_diffs = [
        float(after) - float(before)
        for before, after in zip(baseline["sum_logprob"], candidate["sum_logprob"])
    ]
    ratios = [math.exp(max(-80.0, min(80.0, value))) for value in mean_diffs]
    selected_before = torch.tensor(baseline["selected_logits"], dtype=torch.float32)
    selected_after = torch.tensor(candidate["selected_logits"], dtype=torch.float32)
    drift = (
        float((selected_after - selected_before).abs().max())
        if selected_before.numel() else 0.0
    )
    top1_changes = sum(
        int(left != right) for left, right in zip(baseline["top1"], candidate["top1"])
    )
    return {
        "full_retain_probability_ratio": (
            math.exp(sum(full_diffs) / len(full_diffs)) if full_diffs else 1.0
        ),
        "geometric_retain_probability_ratio": (
            math.exp(sum(mean_diffs) / len(mean_diffs)) if mean_diffs else 1.0
        ),
        "minimum_protected_answer_probability_ratio": min(ratios) if ratios else 1.0,
        "protected_top1_changes": top1_changes,
        "protected_selected_row_logit_drift": drift,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _manifest_records(path: Path) -> List[Mapping[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("records", value.get("payload", {}).get("records", []))
    return [row.get("record", row) for row in rows]


def _write_mcf_manifest(path: Path, records: Sequence[Mapping[str, Any]], partition: str) -> None:
    utility.atomic_json_write(path, {
        "schema_version": "rwku_rank2_mcf_partition_v1",
        "partition": partition,
        "records": [
            {"record": dict(row), "content_sha256": sha256_json(row)} for row in records
        ],
        "official_rwku_records_accessed": False,
    })


def prepare_stage(args: argparse.Namespace) -> None:
    destination = run_dir(args)
    if destination.exists():
        raise ValueError(f"Refusing to reuse experiment directory: {destination}")
    if args.seed != 0:
        raise ValueError("This isolated first experiment is pinned to RWKU seed 0")
    for path, label in (
        (args.generated_training_bundle, "generated training bundle"),
        (args.generator_receipt, "generator receipt"),
        (args.model_path, "Base model"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        reject_official_path(Path(path), label=label)
    receipt = read_artifact(
        args.generator_receipt, stage="prepare", expected_role="generator_receipt"
    )
    if receipt["payload"].get("official_rwku_records_accessed") is not False:
        raise ValueError("Generator receipt lacks the official-data non-access attestation")
    target = target_for_seed(args.seed)
    if receipt.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Generator receipt belongs to another RWKU protocol")
    generated_subject = (
        receipt["payload"].get("target_entity")
        or receipt["payload"].get("subject")
        or receipt.get("metadata", {}).get("subject")
    )
    if generated_subject and str(generated_subject) != target.subject:
        raise ValueError("Generator receipt target differs from RWKU seed 0")
    verify_corpus_manifest(args)
    config = configuration(args)
    destination.mkdir(parents=True, exist_ok=False)
    utility.atomic_json_write(destination / "configuration_manifest.json", {
        "schema_version": CONFIG_SCHEMA,
        "configuration": config,
        "configuration_sha256": sha256_json(config),
        "frozen_at_utc": utc_now(),
    })
    locked = make_artifact(
        "official_locked_eval", official_locked_descriptor(args.seed, include_level12=True),
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL, protocol_status=PROTOCOL_STATUS,
        metadata={"experiment_id": args.experiment_id, "subject": target.subject},
    )
    write_artifact(destination / "official_locked_eval.json", locked)
    write_state(
        args, "PREPARED", target={"subject": target.subject, "entity_id": f"rwku:{target.directory}"},
        configuration_sha256=sha256_json(config),
        model_sha256=sha256_path(args.model_path),
        generated_training_bundle_path=str(Path(args.generated_training_bundle).resolve()),
        generated_training_bundle_sha256=sha256_file(args.generated_training_bundle),
        generator_receipt_path=str(Path(args.generator_receipt).resolve()),
        generator_receipt_sha256=sha256_file(args.generator_receipt),
        corpus_identities=corpus_identities(args),
        official_evaluation_opened=False,
    )


def _verify_prepared(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    if state.get("state") != "PREPARED":
        raise ValueError(f"Training requires PREPARED, got {state.get('state')}")
    checks = (
        (args.generated_training_bundle, "generated_training_bundle_sha256"),
        (args.generator_receipt, "generator_receipt_sha256"),
    )
    for path, field in checks:
        if sha256_file(path) != state.get(field):
            raise ValueError(f"Frozen input changed after prepare: {path}")
    if sha256_path(args.model_path) != state.get("model_sha256"):
        raise ValueError("Base model changed after prepare")
    if sha256_json(configuration(args)) != state.get("configuration_sha256"):
        raise ValueError("Resolved experiment configuration changed after prepare")
    for name, identity in state.get("corpus_identities", {}).items():
        if sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"Frozen corpus file changed after prepare: {name}")


def _materialize_scale(
    output_weight: torch.Tensor, row_ids: Sequence[int], setting5_rows: torch.Tensor,
    delta: torch.Tensor, scale: float,
) -> None:
    if not row_ids:
        return
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    rows = setting5_rows.to(device=output_weight.device, dtype=output_weight.dtype)
    update = delta.to(device=output_weight.device, dtype=output_weight.dtype)
    output_weight.index_copy_(0, ids, rows + float(scale) * update)


def _active_and_selected(
    tokenizer: Any, points: Sequence[utility.TrainingPoint],
    caches: Sequence[active.RewriteDeltaCache], bundle_sha256: str,
) -> Tuple[List[int], List[Dict[str, Any]], List[int]]:
    if len(points) != len(caches):
        raise ValueError("Generated points and repair caches differ in length")
    zero = caches[0].target_new.hidden.new_zeros((0, 0)) if caches else torch.empty(0)
    reports: List[Dict[str, Any]] = []
    sensitive_ids: List[int] = []
    # Cache construction needs selected IDs; initial discovery is performed by
    # the caller with all generated sensitive IDs.  Here zero-delta margins are
    # computed with the resulting cache width.
    if caches:
        width = caches[0].target_new.selected_probs.shape[-1]
        hidden_size = caches[0].target_new.hidden.shape[-1]
        zero = caches[0].target_new.hidden.new_zeros((width, hidden_size))
        margins = active.margins_from_delta_caches(caches, zero).detach().cpu().tolist()
    else:
        margins = []
    for point, margin in zip(points, margins):
        if float(margin) < ACTIVE_MARGIN:
            row = validate_active_provenance(point, bundle_sha256=bundle_sha256)
            row.update({"margin_before": float(margin), "required_margin": ACTIVE_MARGIN})
            reports.append(row)
            sensitive_ids.extend(utility._completion_token_ids(tokenizer, point.sensitive_answer))
    selected, excluded = exclude_special_rows(tokenizer, sensitive_ids)
    return selected, reports, excluded


def _gate_candidate(
    *, margins: torch.Tensor, required: torch.Tensor, retain_kl: float,
    proxy_ppl: float, setting5_ppl: float, scale: float,
    protection: Mapping[str, Any], nonselected_rows_equal_setting5: bool,
) -> Tuple[bool, Dict[str, Any]]:
    minimum = float(margins.min().detach().cpu()) if margins.numel() else math.inf
    violations = int((margins < required).sum().item()) if margins.numel() else 0
    checks = {
        "active_pair_violations_zero": violations == 0,
        "minimum_reloaded_forget_margin": minimum >= MIN_RELOADED_FORGET_MARGIN,
        "retain_kl_increase": retain_kl <= 0.01,
        "full_retain_probability_ratio": 0.995 <= float(
            protection["full_retain_probability_ratio"]
        ) <= 1.005,
        "geometric_retain_probability_ratio": 0.98 <= float(
            protection["geometric_retain_probability_ratio"]
        ) <= 1.02,
        "protected_answer_probability_ratio": float(
            protection["minimum_protected_answer_probability_ratio"]
        ) >= 0.999,
        "protected_top1_changes": int(protection["protected_top1_changes"]) == 0,
        "protected_selected_row_logit_drift": float(
            protection["protected_selected_row_logit_drift"]
        ) <= 0.05,
        "proxy_ppl_ratio": proxy_ppl <= setting5_ppl * 1.02,
        "nonselected_rows_equal_setting5": bool(nonselected_rows_equal_setting5),
    }
    return all(checks.values()), {
        "scale": scale, "checks": checks, "active_pair_violation_count": violations,
        "minimum_generated_margin": minimum, "retain_kl": retain_kl,
        "proxy_ppl": proxy_ppl, "setting5_proxy_ppl": setting5_ppl,
        "protection": dict(protection),
        "all_strict_gates_pass": all(checks.values()),
    }


def train_stage(args: argparse.Namespace) -> None:
    state = read_state(args)
    _verify_prepared(args, state)
    receipt_path = run_dir(args) / "checkpoint_receipt.json"
    if receipt_path.exists():
        assert_model_modification_allowed(receipt_path, experiment_id=args.experiment_id)
    training = read_artifact(
        args.generated_training_bundle, stage="train", gradient=True,
        expected_role="training_bundle",
    )
    generator = read_artifact(
        args.generator_receipt, stage="train", expected_role="generator_receipt"
    )
    if training.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Generated bundle is not the target-only protocol")
    if generator["payload"].get("final_entity_fact_bundle_sha256") != training.get("sha256"):
        raise ValueError("Generator receipt does not bind the training bundle")
    legacy._validate_training_bundle_sources(
        training, training_source=legacy.TRAINING_SOURCE_TARGET_ONLY
    )
    reject_official_path(args.mcf_path, label="MCF retain source")
    for path in args.protection_source:
        reject_official_path(path, label="protection source")
    write_state(args, "TRAINING", official_rwku_records_accessed=False)

    # Build target-independent matched protection during TRAIN, never PREPARE.
    protection_dir = run_dir(args) / "protection"
    protection = build_matched_protection(
        training_bundle_path=args.generated_training_bundle,
        source_corpora=args.protection_source,
        output_dir=protection_dir, vocabulary_path=args.protection_vocabulary,
        split_seed=args.seed, minimum_train_per_key=1, minimum_gate_per_key=1,
        strict=False, tokenizer=None,
    )
    matched_train_path = protection_dir / "matched_protection_train.json"
    matched_gate_path = protection_dir / "matched_protection_gate.json"
    matched_coverage_path = protection_dir / "matched_protection_coverage.json"
    if not matched_train_path.is_file() or not matched_gate_path.is_file():
        raise ValueError("Matched protection construction did not produce train/gate artifacts")
    matched_train = read_artifact(matched_train_path, stage="train", gradient=True)
    matched_gate = read_artifact(matched_gate_path, stage="train", selection=True)
    matched_train_hashes = {
        str(row.get("content_sha256"))
        for row in matched_train["payload"].get("records", [])
    }
    matched_gate_hashes = {
        str(row.get("content_sha256"))
        for row in matched_gate["payload"].get("records", [])
    }
    if (matched_train_hashes & matched_gate_hashes) - {"None"}:
        raise ValueError("Matched-protection train/gate content hashes overlap")
    protection_diagnostics_path = protection_dir / "protection_diagnostics.json"
    utility.atomic_json_write(protection_diagnostics_path, {
        "coverage_policy": "allow_unmatched_generated_target_keys_but_audit",
        "matched_protection_key_count": len(protection["coverage"]),
        "matched_protection_covered_key_count": sum(
            row["coverage_status"] == "covered" for row in protection["coverage"]
        ),
        "matched_protection_insufficient_key_count": len(protection["insufficient"]),
        "matched_protection_insufficient_keys": [
            row["normalized_key"] for row in protection["insufficient"]
        ],
        "train_gate_content_hash_overlap": [],
        "official_rwku_records_accessed": False,
    })
    write_state(
        args, "TRAINING", protection_prepared=True,
        protection_diagnostics=str(protection_diagnostics_path.resolve()),
        matched_protection_key_count=len(protection["coverage"]),
        matched_protection_insufficient_key_count=len(protection["insufficient"]),
        official_rwku_records_accessed=False,
    )

    all_mcf_records, all_mcf_examples = legacy.load_mcf_retain(
        args.mcf_path, seed=args.seed,
        retain_num=args.mcf_retain_num + args.mcf_gate_num,
    )
    retain_records = all_mcf_records[:args.mcf_retain_num]
    gate_records = all_mcf_records[args.mcf_retain_num:]
    retain_examples = all_mcf_examples[:args.mcf_retain_num]
    gate_examples = all_mcf_examples[args.mcf_retain_num:]
    mcf_train_manifest = protection_dir / "mcf_optimization_manifest.json"
    mcf_gate_manifest = protection_dir / "mcf_gate_manifest.json"
    _write_mcf_manifest(mcf_train_manifest, retain_records, "optimization")
    _write_mcf_manifest(mcf_gate_manifest, gate_records, "gate")
    train_hashes = {sha256_json(row) for row in retain_records}
    gate_hashes = {sha256_json(row) for row in gate_records}
    if train_hashes & gate_hashes:
        raise ValueError("MCF optimization/gate partitions overlap")

    legacy.set_all_seeds(args.seed)
    dtype = legacy.dtype_from_name(args.dtype)
    model, tokenizer = legacy.load_model_and_tokenizer(
        args.model_path, dtype=dtype, for_training=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    matched_train_examples = legacy._protection_examples(matched_train, tokenizer)
    matched_gate_examples = legacy._protection_examples(matched_gate, tokenizer)
    retain_examples = [*retain_examples, *matched_train_examples]
    gate_examples = [*gate_examples, *matched_gate_examples]
    views = list(training["payload"].get("views", []))
    points = utility.compile_training_points(tokenizer, training)
    base_generated = _generated_metrics(
        model, tokenizer, points, batch_size=args.eval_batch_size
    )
    forget_examples, examples_by_view = legacy.setting5_entity_fact_examples(
        tokenizer, views, include_reverse=False
    )
    views_by_fact: Dict[str, List[Mapping[str, Any]]] = {}
    for view in views:
        views_by_fact.setdefault(str(view["fact_id"]), []).append(view)
    plan = build_fact_cycle_plan(views_by_fact, steps=SETTING5_STEPS, seed=args.seed)
    batches = [[examples_by_view[str(item["view_id"])]] for item in plan]
    exposures = exposure_report(plan, seed=args.seed, tokenizer=tokenizer)
    exposures["plan_sha256"] = plan_sha256(plan)
    utility.atomic_json_write(run_dir(args) / "fact_exposure_report.json", exposures)
    setting_args = _setting5_args(args, batches)
    started = time.perf_counter()
    summary = gagd.train_mode(
        model, tokenizer, forget_examples, retain_examples, selected_ids=[],
        mode=legacy.SETTING5_MODE, args=setting_args,
        mode_dir=run_dir(args) / "setting5_training",
    )
    legacy.prepare_model_for_evaluation(model)
    setting5_dir = run_dir(args) / "setting5_training" / "checkpoint"
    legacy.save_checkpoint(model, tokenizer, setting5_dir)
    setting5_generated = _generated_metrics(
        model, tokenizer, points, batch_size=args.eval_batch_size
    )

    # Discover active generated pairs without consulting official RWKU rows.
    all_sensitive_ids = [
        token_id for point in points
        for token_id in utility._completion_token_ids(tokenizer, point.sensitive_answer)
    ]
    preliminary_ids, special_excluded = exclude_special_rows(tokenizer, all_sensitive_ids)
    if not preliminary_ids:
        raise ValueError("No non-special sensitive LM-head rows are available for repair")
    instances = _point_instances(points)
    device = next(model.parameters()).device
    llama_like = active.is_llama_like(model, tokenizer)
    preliminary_caches = active.build_prompt_instance_delta_caches(
        model, tokenizer, instances, preliminary_ids, device,
        args.eval_batch_size, llama_like,
    )
    selected_ids, active_rows, active_special_excluded = _active_and_selected(
        tokenizer, points, preliminary_caches,
        bundle_sha256=sha256_file(args.generated_training_bundle),
    )
    protected_ids = protected_answer_row_ids(
        tokenizer, [*retain_examples, *gate_examples]
    )
    protected_overlap_excluded = sorted(set(selected_ids) & protected_ids)
    selected_ids = sorted(set(selected_ids) - protected_ids)
    repair_dir = run_dir(args) / "rank2_active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(repair_dir / "active_points.jsonl", active_rows)
    utility.atomic_json_write(repair_dir / "active_rows.json", {
        "repair_rank": REPAIR_RANK, "selected_row_ids": selected_ids,
        "decoded_rows": {str(row): tokenizer.decode([row]) for row in selected_ids},
        "excluded_special_row_ids": sorted(set(special_excluded + active_special_excluded)),
        "excluded_protected_answer_row_ids": protected_overlap_excluded,
        "active_pair_count": len(active_rows),
        "official_rwku_records_accessed": False,
    })

    output = active.freeze_model_for_output_repair(model)
    if any(parameter.requires_grad for name, parameter in model.named_parameters() if "lm_head" not in name):
        raise AssertionError("Transformer/input parameters remained trainable during repair")
    full_setting5_output = output.weight.detach().cpu().clone()
    setting5_rows = output.weight.index_select(
        0, torch.tensor(selected_ids, device=output.weight.device)
    ).detach().cpu().clone()
    all_caches = active.build_prompt_instance_delta_caches(
        model, tokenizer, instances, selected_ids, device,
        args.eval_batch_size, llama_like,
    )
    active_view_ids = {row["view_id"] for row in active_rows}
    active_caches = [cache for point, cache in zip(points, all_caches) if point.view_id in active_view_ids]
    zero = torch.zeros((len(selected_ids), output.weight.shape[1]), device=device, dtype=torch.float32)
    original_margins = (
        active.margins_from_delta_caches(active_caches, zero)
        if active_caches else zero.new_empty((0,))
    )
    required = torch.full_like(original_margins, ACTIVE_MARGIN)

    sampled_retain = _mcf_sampled_records(
        retain_records[:RETAIN_CALIBRATION_NUM], retain_examples[:RETAIN_CALIBRATION_NUM]
    )
    sampled_retain.extend(
        _mcf_sampled_records(
            [
                {"source": "matched_generated_corpus_protection", "index": index}
                for index in range(len(matched_train_examples))
            ],
            matched_train_examples,
        )
    )
    reference_output_weight, reference_output_bias = active.load_reference_output_layer(
        str(args.model_path), dtype
    )
    retain_caches = active.build_retain_kl_caches(
        model, reference_output_weight, reference_output_bias,
        tokenizer, sampled_retain, selected_ids, device
    )
    del reference_output_weight, reference_output_bias
    retained_hidden = torch.cat([cache.hidden for cache in retain_caches], dim=0)
    retained_basis = active.orthonormal_row_basis(retained_hidden) if PROJECT_AWAY_RETAIN_HIDDEN else None
    if active_caches and selected_ids:
        active_hidden = torch.cat(
            [cache.target_new.hidden for cache in active_caches]
            + [cache.target_true.hidden for cache in active_caches], dim=0
        )
        delta_module = rank2_delta_module(
            n_rows=len(selected_ids), hidden_size=output.weight.shape[1],
            active_hidden=active_hidden, retained_basis=retained_basis, device=device,
        )
        logs, optimization = active.optimize_selected_delta(
            delta_module,
            lambda delta: active.margins_from_delta_caches(active_caches, delta),
            lambda delta: active.retain_kl_from_caches(retain_caches, delta),
            required_margins=required, repair_steps=REPAIR_STEPS,
            repair_lr=REPAIR_LR, repair_optimizer="adamw",
            hinge_weight=HINGE_WEIGHT, delta_l2_lambda=DELTA_L2_LAMBDA,
            retain_kl_mu=RETAIN_KL_MU, stop_when_all_satisfied=True,
        )
        delta = delta_module.effective_delta().detach()
    else:
        logs = []
        optimization = {
            "steps_completed": 0, "stopped_early": True,
            "all_satisfied": not bool(active_caches),
            "training_prompt_instances": len(active_caches),
            "reason": (
                "Setting 5e already satisfies every generated active pair"
                if not active_caches else
                "No eligible non-special, non-protected sensitive LM-head row"
            ),
        }
        delta = zero
    _write_jsonl(repair_dir / "repair_log.jsonl", logs)
    proxy_text = load_wikidata_text(args.wikidata_dir)
    if not proxy_text:
        raise FileNotFoundError("Target-independent Wikidata proxy-PPL text is required")
    setting5_ppl = evaluate_perplexity(model, tokenizer, proxy_text)
    setting5_protection = _protection_snapshot(
        model, tokenizer, gate_examples, selected_ids,
        batch_size=args.eval_batch_size,
    )
    sweep: List[Dict[str, Any]] = []
    selected_scale: Optional[float] = None
    for scale in CANDIDATE_SCALES:
        _materialize_scale(output.weight, selected_ids, setting5_rows, delta, scale)
        actual_caches = active.build_prompt_instance_delta_caches(
            model, tokenizer, instances, selected_ids, device,
            args.eval_batch_size, llama_like,
        )
        actual_active = [cache for point, cache in zip(points, actual_caches) if point.view_id in active_view_ids]
        actual_zero = torch.zeros_like(delta)
        actual_margins = (
            active.margins_from_delta_caches(actual_active, actual_zero)
            if actual_active else actual_zero.new_empty((0,))
        )
        scaled_delta = delta * scale
        retain_kl = float(active.retain_kl_from_caches(retain_caches, scaled_delta).detach().cpu())
        ppl = evaluate_perplexity(model, tokenizer, proxy_text)
        candidate_protection = _protection_snapshot(
            model, tokenizer, gate_examples, selected_ids,
            batch_size=args.eval_batch_size,
        )
        protection_metrics = _protection_metrics(
            setting5_protection, candidate_protection
        )
        selected_mask = torch.zeros(
            output.weight.shape[0], dtype=torch.bool, device=output.weight.device
        )
        selected_mask[torch.tensor(selected_ids, device=output.weight.device)] = True
        nonselected_equal = torch.equal(
            output.weight.detach()[~selected_mask].cpu(),
            full_setting5_output[~selected_mask.cpu()],
        )
        accepted, report = _gate_candidate(
            margins=actual_margins, required=required, retain_kl=retain_kl,
            proxy_ppl=ppl, setting5_ppl=setting5_ppl, scale=scale,
            protection=protection_metrics,
            nonselected_rows_equal_setting5=nonselected_equal,
        )
        sweep.append(report)
        if accepted:
            selected_scale = scale
            break
    candidate_accepted = selected_scale is not None
    if selected_scale is None:
        selected_scale = 0.0
    _materialize_scale(output.weight, selected_ids, setting5_rows, delta, selected_scale)
    selected_candidate_report = next(
        (row for row in sweep if float(row["scale"]) == float(selected_scale)),
        sweep[-1] if sweep else {},
    )
    selected_dir = run_dir(args) / "selected_checkpoint"
    legacy.save_checkpoint(model, tokenizer, selected_dir)
    selected_generated = _generated_metrics(
        model, tokenizer, points, batch_size=args.eval_batch_size
    )
    utility.atomic_json_write(repair_dir / "candidate_scale_sweep.json", {"candidates": sweep})
    repair_report = {
        "repair_type": "rank2_active_pair",
        "repair_rank": REPAIR_RANK,
        "legacy_rwku_projection_rank": PROTECTED_PROJECTION_RANK,
        "legacy_rwku_projection_rank_used": False,
        "active_pair_count": len(active_rows),
        "active_pair_violations_before": int(
            (original_margins < required).sum().item()
        ),
        "active_pair_violations_after": int(
            selected_candidate_report.get("active_pair_violation_count", 0)
        ),
        "selected_row_count": len(selected_ids),
        "selected_row_ids": selected_ids,
        "selected_scale": selected_scale,
        "candidate_accepted": candidate_accepted,
        "candidate_reason": (
            "all pre-freeze generated/protection gates passed"
            if candidate_accepted else
            "rank-2 repair found no protected passing scale; safe Setting5 scale-zero fallback frozen"
        ),
        "optimization": optimization,
        "official_rwku_records_accessed": False,
    }
    utility.atomic_json_write(repair_dir / "repair_summary.json", repair_report)
    training_report = {
        "method": METHOD, "protocol_status": PROTOCOL_STATUS,
        "setting5": {"steps": SETTING5_STEPS, "summary": asdict(summary),
                     "training_seconds": time.perf_counter() - started,
                     "generated_metrics": setting5_generated},
        "selected": {"generated_metrics": selected_generated},
        "base": {"generated_metrics": base_generated},
        "balanced_fact_cycle": exposures, "repair": repair_report,
        "final_evaluation_used_for_selection": False,
        "official_rwku_records_accessed": False,
    }
    training_report_path = run_dir(args) / "training_report.json"
    utility.atomic_json_write(training_report_path, training_report)

    config = configuration(args)
    checkpoint_receipt = create_checkpoint_receipt(
        destination=receipt_path, experiment_id=args.experiment_id,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL, protocol_status=PROTOCOL_STATUS,
        target_entity=state["target"]["subject"], target_entity_id=state["target"]["entity_id"],
        base_model_identity=legacy.local_model_identity(args.model_path),
        base_model_revision=args.model_revision,
        tokenizer_identity={"name_or_path": tokenizer.name_or_path,
                            "class": tokenizer.__class__.__name__,
                            "vocab_size": len(tokenizer),
                            "eos_token_id": tokenizer.eos_token_id},
        checkpoint_paths=[setting5_dir, selected_dir],
        training_bundle_path=args.generated_training_bundle,
        optimization_protection_path=matched_train_path,
        mcf_retain_optimization_paths=[mcf_train_manifest],
        mcf_repair_gate_paths=[mcf_gate_manifest],
        matched_protection_train_path=matched_train_path,
        matched_protection_gate_path=matched_gate_path,
        method_configuration=config,
        implementation_files=[SCRIPT_PATH,
            SEMANTIC_ROOT / "scripts" / "gagd_compare.py",
            SEMANTIC_ROOT / "scripts" / "gagd_active_case_repair.py",
            SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
            SEMANTIC_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
            SEMANTIC_ROOT / "scripts" / "rwku_fact_sampler.py",
            SEMANTIC_ROOT / "scripts" / "rwku_eval.py",
            SEMANTIC_ROOT / "scripts" / "rwku_experiment.py",
            SEMANTIC_ROOT / "scripts" / "rwku_setting5e_utility_controlled.py",
            SEMANTIC_ROOT / "scripts" / "rwku_rowwise_active_repair.py",
            SEMANTIC_ROOT / "scripts" / "build_rwku_matched_protection.py",
            SEMANTIC_ROOT / "scripts" / "build_rwku_entity_facts.py"],
        sampler_provenance=exposures,
        generator_receipt_path=args.generator_receipt,
        official_locked_eval_path=run_dir(args) / "official_locked_eval.json",
        confirmatory=False,
        additional_artifact_paths={
            "base_model_source": args.model_path,
            "configuration_manifest": run_dir(args) / "configuration_manifest.json",
            "training_report": training_report_path,
            "repair_report": repair_dir / "repair_summary.json",
            "matched_protection_coverage": matched_coverage_path,
            "protection_diagnostics": protection_diagnostics_path,
            **{
                f"frozen_corpus_{name.replace('.', '_')}": Path(identity["path"])
                for name, identity in state["corpus_identities"].items()
                if name not in {"generated_training_bundle.json", "generator_receipt.json"}
            },
        },
    )
    write_state(
        args, "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=checkpoint_receipt["receipt_sha256"],
        setting5_checkpoint=str(setting5_dir.resolve()),
        selected_checkpoint=str(selected_dir.resolve()),
        repair_candidate_accepted=candidate_accepted,
        official_evaluation_opened=False,
    )
    legacy.release_model(model)


def _verify_evaluation(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    state = read_state(args)
    if state.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError("Official evaluation requires CHECKPOINT_FROZEN")
    receipt = load_receipt(run_dir(args) / "checkpoint_receipt.json")
    if receipt.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError("Frozen receipt is required before evaluation")
    if receipt.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError("Receipt belongs to another method")
    if sha256_json(configuration(args)) != receipt.get("method_configuration_sha256"):
        raise ValueError("Evaluation configuration differs from frozen training configuration")
    verify_frozen_identities(receipt)
    return state, receipt


def evaluate_stage(args: argparse.Namespace) -> None:
    state, receipt = _verify_evaluation(args)
    result_path = run_dir(args) / "final_result.json"
    if result_path.exists():
        raise ValueError("Refusing to overwrite an existing final result")
    receipt_path = run_dir(args) / "checkpoint_receipt.json"
    opened = open_official_evaluation(receipt_path, experiment_id=args.experiment_id)
    write_state(
        args, "OFFICIAL_EVALUATION_OPENED", official_evaluation_opened=True,
        official_evaluation_opened_at_utc=opened["official_evaluation_opened_at_utc"],
    )
    try:
        locked = read_artifact(
            run_dir(args) / "official_locked_eval.json", stage="evaluate",
            evaluation=True, expected_role="official_locked_eval",
        )
        target, datasets, file_hashes = ensure_target_data(
            args.data_root, args.seed, allow_download=not args.no_download
        )
        for filename, descriptor in locked["payload"]["files"].items():
            if file_hashes.get(filename) != descriptor["sha256"]:
                raise ValueError(f"Official locked file changed: {filename}")
        dtype = legacy.dtype_from_name(args.dtype)
        base, tokenizer = legacy.load_model_and_tokenizer(
            args.model_path, dtype=dtype, for_training=False,
            gradient_checkpointing=False,
        )
        answers = [str(row["answer"]) for filename in (
            "forget_level1.json", "forget_level2.json", "forget_level3.json"
        ) for row in datasets[filename]]
        probe = build_frozen_head_probe(
            base, tokenizer, datasets["forget_level2.json"], additional_answers=answers
        )
        base_result = utility._evaluate_official_model(
            method="Base model", model=base, tokenizer=tokenizer, target=target,
            datasets=datasets, args=args, frozen_probe=probe, base_retain=None,
        )
        base_retain = base_result["retain_reference_mean_logprobs"]
        legacy.release_model(base)
        del base, tokenizer
        setting_model, setting_tokenizer = legacy.load_model_and_tokenizer(
            Path(receipt["checkpoint_paths"][0]["path"]), dtype=dtype,
            for_training=False, gradient_checkpointing=False,
        )
        setting_result = utility._evaluate_official_model(
            method=SETTING5_METHOD, model=setting_model, tokenizer=setting_tokenizer,
            target=target, datasets=datasets, args=args, frozen_probe=probe,
            base_retain=base_retain,
        )
        legacy.release_model(setting_model)
        del setting_model, setting_tokenizer
        selected_model, selected_tokenizer = legacy.load_model_and_tokenizer(
            Path(receipt["checkpoint_paths"][1]["path"]), dtype=dtype,
            for_training=False, gradient_checkpointing=False,
        )
        selected_result = utility._evaluate_official_model(
            method=(
                METHOD if state.get("repair_candidate_accepted") else
                "Safe Setting 5e @600 fallback (rank-2 candidate rejected pre-freeze)"
            ),
            model=selected_model, tokenizer=selected_tokenizer,
            target=target, datasets=datasets, args=args, frozen_probe=probe,
            base_retain=base_retain,
        )
        legacy.release_model(selected_model)
        del selected_model, selected_tokenizer
        with (run_dir(args) / "training_report.json").open("r", encoding="utf-8") as handle:
            training_report = json.load(handle)
        result = {
            "schema_version": "rwku_generated_s5e_rank2_active_result_v1",
            "experiment_id": args.experiment_id, "method": METHOD,
            "protocol_status": PROTOCOL_STATUS,
            "repair_candidate_accepted": bool(state.get("repair_candidate_accepted")),
            "generated_corpus_forgetting": {
                "Base": training_report["base"]["generated_metrics"],
                "setting5_before_repair": training_report["setting5"]["generated_metrics"],
                "selected_after_rank2_repair": training_report["selected"]["generated_metrics"],
                "active_pair_violation_count_before": training_report["repair"]["active_pair_violations_before"],
                "active_pair_violation_count_after": training_report["repair"]["active_pair_violations_after"],
            },
            "official_rwku_after_freeze": {
                "Base": base_result,
                "Setting 5e @600 before repair": setting_result,
                "Setting 5e @600 + rank-2 active repair": selected_result,
            },
            "official_file_sha256": file_hashes,
            "checkpoint_reused_without_modification": True,
            "official_evaluation_opened_at_utc": opened["official_evaluation_opened_at_utc"],
        }
        normalized, replacements = utility.strict_json_normalize(result)
        normalized["serialization"] = {
            "policy": "non_finite_numeric_values_to_json_null",
            "strict_json_allow_nan": False, "replacement_count": len(replacements),
            "replacements": replacements,
        }
        utility.atomic_json_write(result_path, normalized)
        with result_path.open("r", encoding="utf-8") as handle:
            json.load(handle)
        verify_frozen_identities(load_receipt(receipt_path))
        completed = mark_evaluation_complete(receipt_path, experiment_id=args.experiment_id)
        write_state(
            args, "EVALUATION_COMPLETE", result_path=str(result_path.resolve()),
            official_evaluation_opened=True,
            official_evaluation_opened_at_utc=opened["official_evaluation_opened_at_utc"],
            evaluation_completed_at_utc=completed["evaluation_completed_at_utc"],
        )
    except Exception as exc:
        utility.atomic_json_write(run_dir(args) / "evaluation_failure.json", {
            "status": "failed_after_official_evaluation_opened",
            "error_type": type(exc).__name__, "error": str(exc),
            "state_preserved": "OFFICIAL_EVALUATION_OPENED", "timestamp_utc": utc_now(),
        })
        raise


def preflight(args: argparse.Namespace) -> None:
    if args.seed != 0:
        raise ValueError("Seed-0 launcher requires --seed 0")
    expected = CORPUS_FILENAMES
    corpus = Path(args.generated_training_bundle).parent
    missing = [name for name in expected if not (corpus / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen corpus is incomplete: {missing}")
    for path in (args.generated_training_bundle, args.generator_receipt, args.mcf_path):
        reject_official_path(path, label="preflight input")
    verify_corpus_manifest(args)
    print(json.dumps({
        "status": "preflight_ok", "configuration": configuration(args),
        "corpus_files": {name: sha256_file(corpus / name) for name in expected},
        "official_rwku_records_accessed": False,
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=("preflight", "prepare", "train", "evaluate"), required=True)
    value.add_argument("--experiment-id", required=True)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--model-revision", required=True)
    value.add_argument("--generated-training-bundle", type=Path, required=True)
    value.add_argument("--generator-receipt", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--mcf-path", type=Path, default=legacy.DEFAULT_MCF_PATH)
    value.add_argument("--protection-source", type=Path, action="append", default=[])
    value.add_argument("--protection-vocabulary", type=Path, default=None)
    value.add_argument("--mcf-retain-num", type=int, default=1000)
    value.add_argument("--mcf-gate-num", type=int, default=200)
    value.add_argument("--data-root", type=Path, default=legacy.DEFAULT_DATA_ROOT)
    value.add_argument("--wikidata-dir", type=Path, default=legacy.DEFAULT_WIKIDATA_DIR)
    value.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    value.add_argument("--eval-batch-size", type=int, default=4)
    value.add_argument("--gradient-checkpointing", action="store_true")
    value.add_argument("--no-download", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.protection_source:
        args.protection_source = [args.mcf_path]
    if args.experiment_id != "rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1":
        raise ValueError("This isolated submission is pinned to its new experiment ID")
    for path in (args.generated_training_bundle, args.generator_receipt, args.mcf_path, *args.protection_source):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "train":
        train_stage(args)
    else:
        evaluate_stage(args)


if __name__ == "__main__":
    main()
