#!/usr/bin/env python3
"""Train MCF V2.2 endpoints with forget and retain examples on every update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import build_mcf_biendpoint_nullspace_rewiring_v2_split as split_builder
import gagd_compare as gagd
import mcf_biendpoint_nullspace_rewiring_v2_core as geometry
import mcf_folded_sensitivity_rewiring_v2_1_core as folded
import mcf_joint_forget_retain_endpoint_v2_2_core as joint
import mcf_sure_directional_emb_lm_stage1 as endpoint_hooks
import mcf_sure_subject_directional_emb_stage1 as subject_stage1
import mcf_synthetic_paraphrase_templates as synthetic
import run_mcf_biendpoint_nullspace_rewiring_v2 as v2
import run_mcf_folded_sensitivity_rewiring_v2_1 as v21
import sure_canonical_core as canonical


BACKTRACKING_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)


def target_logprob_drift(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return zero for generic prompts and drift for labeled retain examples."""
    local_ids = target_ids.to(logits.device)
    base_logs = base_target_log_probs.to(logits.device)
    valid = local_ids >= 0
    current = (
        F.log_softmax(logits.float(), dim=1)
        .gather(1, local_ids.clamp_min(0)[:, None])
        .squeeze(1)
    )
    return torch.where(valid, current - base_logs, torch.zeros_like(current))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=12000)
    parser.add_argument("--corpus-fit-prompts", type=int, default=1000)
    parser.add_argument("--corpus-development-prompts", type=int, default=250)
    parser.add_argument("--corpus-certification-prompts", type=int, default=1000)
    parser.add_argument("--synthetic-paraphrases", type=int, default=3)
    parser.add_argument("--same-subject-prompts", type=int, default=4)
    parser.add_argument("--targeted-fit-per-row", type=int, default=1)
    parser.add_argument("--targeted-development-per-row", type=int, default=1)
    parser.add_argument("--targeted-certification-per-row", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--hard-tail-refresh-every", type=int, default=50)
    parser.add_argument("--hard-tail-add", type=int, default=32)
    parser.add_argument("--hard-tail-capacity", type=int, default=256)
    parser.add_argument("--hard-tail-active", type=int, default=16)
    parser.add_argument("--forget-batch-size", type=int, default=16)
    parser.add_argument("--random-retain-batch-size", type=int, default=16)
    parser.add_argument("--overlap-retain-batch-size", type=int, default=16)
    parser.add_argument("--active-retain-maximum", type=int, default=48)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--forget-margin-target", type=float, default=6.0)
    parser.add_argument("--minimum-forget-margin", type=float, default=0.1)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=100.0)
    parser.add_argument("--retain-top1-weight", type=float, default=100.0)
    parser.add_argument("--retain-target-weight", type=float, default=100.0)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--input-relative-cap", type=float, default=0.5)
    parser.add_argument("--input-frequency-alpha", type=float, default=0.25)
    parser.add_argument("--output-relative-cap", type=float, default=0.3)
    parser.add_argument("--input-step-cap-fraction", type=float, default=0.002)
    parser.add_argument("--output-step-cap-fraction", type=float, default=0.002)
    parser.add_argument("--protection-topk", type=int, default=64)
    parser.add_argument("--protected-kl-mean-max", type=float, default=1e-4)
    parser.add_argument("--protected-kl-max", type=float, default=1e-2)
    parser.add_argument("--protected-top1-drift-max", type=float, default=5e-2)
    parser.add_argument("--protected-target-drift-max", type=float, default=5e-2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.gradient_checkpointing = False
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V2.2 is locked to consumed seed 1 / 50 facts")
    if args.frequency_doc_start < 20:
        parser.error("Wikipedia documents 0:20 remain reserved for official PPL")
    if args.steps % args.check_every or args.steps % args.hard_tail_refresh_every:
        parser.error("steps must divide check and hard-tail refresh intervals")
    if (
        args.hard_tail_active
        + args.overlap_retain_batch_size
        + args.random_retain_batch_size
        < args.active_retain_maximum
    ):
        parser.error("active retain maximum exceeds the registered strata budget")
    if args.hard_tail_active >= args.active_retain_maximum:
        parser.error("hard tail must leave room for overlap and random retain strata")
    return args


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    architecture = registry.get("architecture", {})
    data = registry.get("data", {})
    retain_strata = registry.get("retain_strata", {})
    optimization = registry.get("optimization", {})
    if (
        registry.get("protocol") != joint.PROTOCOL
        or registry.get("status")
        != "training_only_implementation_available_not_executed"
        or architecture.get("trainable_parameter_families")
        != ["selected_input_embedding_rows", "selected_untied_lm_head_rows"]
        or architecture.get("transformer_frozen") is not True
        or architecture.get("external_classifier") is not False
        or architecture.get("runtime_gate") is not False
        or architecture.get("sidecar") is not False
        or data.get("forget_positive_precedes_exact_retain_collision") is not True
        or retain_strata.get("present_on_every_endpoint_update") is not True
        or retain_strata.get("shared_target_output_tokens") is not True
        or optimization.get("adam_forbidden") is not True
    ):
        raise RuntimeError("V2.2 registry architecture/status mismatch")
    expected = {
        "data": {
            "seed": args.seed,
            "forget_records": args.forget_num,
            "wikipedia_doc_start": args.frequency_doc_start,
            "wikipedia_documents": args.frequency_docs,
            "corpus_fit_prompts": args.corpus_fit_prompts,
            "corpus_development_prompts": args.corpus_development_prompts,
            "corpus_certification_prompts": args.corpus_certification_prompts,
            "synthetic_paraphrases_per_forget_record": args.synthetic_paraphrases,
            "same_subject_different_relation_prompts_per_record": args.same_subject_prompts,
            "targeted_row_fit_prompts": args.targeted_fit_per_row,
            "targeted_row_development_prompts": args.targeted_development_per_row,
            "targeted_row_certification_prompts": args.targeted_certification_per_row,
        },
        "optimization": {
            "steps": args.steps,
            "check_every": args.check_every,
            "hard_tail_refresh_every": args.hard_tail_refresh_every,
            "hard_tail_add": args.hard_tail_add,
            "hard_tail_capacity": args.hard_tail_capacity,
            "hard_tail_active": args.hard_tail_active,
            "forget_batch_size": args.forget_batch_size,
            "random_retain_batch_size": args.random_retain_batch_size,
            "overlap_retain_batch_size": args.overlap_retain_batch_size,
            "active_retain_maximum": args.active_retain_maximum,
            "forget_weight": args.forget_weight,
            "retain_kl_weight": args.retain_kl_weight,
            "retain_top1_weight": args.retain_top1_weight,
            "retain_target_weight": args.retain_target_weight,
            "delta_l2_weight": args.delta_l2_weight,
            "input_relative_row_cap": args.input_relative_cap,
            "input_frequency_alpha": args.input_frequency_alpha,
            "output_relative_row_cap": args.output_relative_cap,
            "input_step_cap_fraction": args.input_step_cap_fraction,
            "output_step_cap_fraction": args.output_step_cap_fraction,
            "backtracking_factors": list(BACKTRACKING_FACTORS),
        },
        "acceptance": {
            "minimum_forget_margin": args.minimum_forget_margin,
            "protected_topk_kl_mean_max": args.protected_kl_mean_max,
            "protected_topk_kl_absolute_max": args.protected_kl_max,
            "protected_top1_logprob_abs_max": args.protected_top1_drift_max,
            "protected_target_logprob_abs_max": args.protected_target_drift_max,
        },
    }
    for section, values in expected.items():
        registered = registry.get(section)
        if not isinstance(registered, Mapping):
            raise RuntimeError(f"V2.2 registry lacks {section}")
        for key, value in values.items():
            if registered.get(key) != value:
                raise RuntimeError(
                    f"V2.2 argument differs from registry: {section}.{key}"
                )


@torch.no_grad()
def protection_vectors(
    model: torch.nn.Module,
    tok: Any,
    cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kl_values: List[torch.Tensor] = []
    drift_values: List[torch.Tensor] = []
    target_drift_values: List[torch.Tensor] = []
    for start in range(0, len(cache.cases), batch_size):
        indices = list(range(start, min(start + batch_size, len(cache.cases))))
        batch = v2.cache_batch(cache, indices, device)
        logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
        kl, drift = geometry.protection_loss(
            logits, **{key: value for key, value in batch.items() if key != "cases"}
        )
        kl_values.append(kl.cpu())
        drift_values.append(drift.cpu())
        local = torch.tensor(indices, dtype=torch.long)
        target_drift_values.append(
            target_logprob_drift(
                logits,
                target_ids.index_select(0, local),
                base_target_log_probs.index_select(0, local),
            ).cpu()
        )
    return (
        torch.cat(kl_values),
        torch.cat(drift_values),
        torch.cat(target_drift_values),
    )


@torch.no_grad()
def cache_retain_targets(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[canonical.SensitivePredictionCase],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_ids: List[torch.Tensor] = []
    target_log_probs: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        local_cases = cases[start : start + batch_size]
        logits = canonical.forward_last_logits(model, tok, local_cases, device).float()
        local_ids = torch.full((len(local_cases),), -1, dtype=torch.long, device=device)
        local_logs = torch.zeros(len(local_cases), dtype=torch.float32, device=device)
        valid_positions = [
            index for index, case in enumerate(local_cases) if case.target_text
        ]
        if valid_positions:
            valid_cases = [local_cases[index] for index in valid_positions]
            valid_ids = canonical.official_target_ids(
                tok, valid_cases, llama_like=llama_like, device=device
            )
            positions = torch.tensor(valid_positions, dtype=torch.long, device=device)
            local_ids.index_copy_(0, positions, valid_ids)
            valid_logs = (
                F.log_softmax(logits.index_select(0, positions), dim=1)
                .gather(1, valid_ids[:, None])
                .squeeze(1)
            )
            local_logs.index_copy_(0, positions, valid_logs)
        target_ids.append(local_ids.cpu())
        target_log_probs.append(local_logs.cpu())
    return torch.cat(target_ids), torch.cat(target_log_probs)


def active_retain_loss(
    model: torch.nn.Module,
    tok: Any,
    cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    *,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not indices:
        raise ValueError("active retain batch cannot be empty")
    kl_parts: List[torch.Tensor] = []
    drift_parts: List[torch.Tensor] = []
    target_drift_parts: List[torch.Tensor] = []
    for start in range(0, len(indices), batch_size):
        local = list(indices[start : start + batch_size])
        batch = v2.cache_batch(cache, local, device)
        logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
        kl, drift = geometry.protection_loss(
            logits, **{key: value for key, value in batch.items() if key != "cases"}
        )
        kl_parts.append(kl)
        drift_parts.append(drift)
        local_indices = torch.tensor(local, dtype=torch.long)
        target_drift_parts.append(
            target_logprob_drift(
                logits,
                target_ids.index_select(0, local_indices),
                base_target_log_probs.index_select(0, local_indices),
            )
        )
    kl = torch.cat(kl_parts)
    drift = torch.cat(drift_parts)
    target_drift = torch.cat(target_drift_parts)
    return kl.mean(), kl.max(), drift.abs().max(), target_drift.abs().max()


@torch.no_grad()
def evaluate_endpoint_protection(
    model: torch.nn.Module,
    tok: Any,
    cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    kl, drift, target_drift = protection_vectors(
        model,
        tok,
        cache,
        target_ids,
        base_target_log_probs,
        device,
        batch_size=batch_size,
    )
    report = geometry.protection_report(
        kl,
        drift,
        kl_mean_max=args.protected_kl_mean_max,
        kl_absolute_max=args.protected_kl_max,
        top1_abs_max=args.protected_top1_drift_max,
    )
    target_max = float(target_drift.abs().max().item())
    report["target_logprob_abs_max"] = target_max
    report["criterion"]["target_logprob_abs_max"] = args.protected_target_drift_max
    report["passed"] = bool(
        report["passed"] and target_max <= args.protected_target_drift_max
    )
    return report


def endpoint_certificate(
    model: torch.nn.Module,
    tok: Any,
    forget: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
    protection_cache: v2.ProtectionCache,
    target_ids: torch.Tensor,
    base_target_log_probs: torch.Tensor,
    input_delta: canonical.SelectedRowDelta,
    output_delta: canonical.SelectedRowDelta,
    input_caps: torch.Tensor,
    output_caps: torch.Tensor,
    device: torch.device,
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    result = v21.certificate(
        model,
        tok,
        forget,
        synthetic_records,
        protection_cache,
        input_delta,
        output_delta,
        input_caps,
        output_caps,
        device,
        args=args,
    )
    result["protection"] = evaluate_endpoint_protection(
        model,
        tok,
        protection_cache,
        target_ids,
        base_target_log_probs,
        device,
        batch_size=args.capture_batch_size,
        args=args,
    )
    result["passed"] = bool(
        result["direct"]["passed"]
        and result["synthetic"]["passed"]
        and result["protection"]["passed"]
        and result["input_caps"]["passed"]
        and result["output_caps"]["passed"]
    )
    return result


def completion_failure(output: Path, *, phase: str, detail: Mapping[str, Any]) -> None:
    v2.write_json(
        output / "method" / "completion.json",
        {
            "schema_version": 1,
            "protocol": joint.PROTOCOL,
            "passed": False,
            "phase": phase,
            "detail": dict(detail),
            "candidate_saved": False,
            "official_evaluation_prompts_seen": 0,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v2.validate_environment_firewall()
    gagd.set_seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.experiment_registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry, args)
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest_path = protocol_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != split_builder.PROTOCOL:
        raise RuntimeError("V2.2 split manifest protocol mismatch")
    if manifest.get("serialized_prompt_counts", {}).get("official_retain") != 0:
        raise RuntimeError("V2.2 split exposes official retain prompt text")
    forget = v2.load_partition(protocol_dir, manifest, "forget", permitted=True)
    protection_fit = v2.load_partition(
        protocol_dir, manifest, "protection_fit", permitted=True
    )
    protection_development = v2.load_partition(
        protocol_dir, manifest, "protection_development", permitted=True
    )

    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    device = gagd.first_device(model)
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    llama_like = canonical.is_llama_like(model, tok)
    transformer_before = v2.transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )
    endpoint_rows, expansion_report = joint.expanded_endpoint_rows(
        forget, tok, llama_like=llama_like
    )
    overlap_report = geometry.endpoint_overlap_report(endpoint_rows)
    v2.write_json(
        output / "method" / "endpoint_row_manifest.json",
        {"expansion": expansion_report, "overlap": overlap_report},
    )
    if (
        not expansion_report["one_delta_per_physical_input_row"]
        or not overlap_report["one_delta_per_physical_input_row"]
        or not overlap_report["one_delta_per_physical_output_row"]
    ):
        raise RuntimeError("V2.2 endpoint row coherence failed")
    input_ids_tensor = torch.tensor(
        endpoint_rows.input_ids, dtype=torch.long, device=device
    )
    output_ids_tensor = torch.tensor(
        endpoint_rows.output_ids, dtype=torch.long, device=device
    )
    base_input_rows = (
        input_layer.weight.index_select(0, input_ids_tensor).detach().float()
    )
    base_output_rows = (
        output_layer.weight.index_select(0, output_ids_tensor).detach().float()
    )
    hidden_size = int(base_input_rows.shape[1])
    input_delta = canonical.SelectedRowDelta(
        len(endpoint_rows.input_ids), hidden_size, direction_basis=None, device=device
    )
    output_delta = canonical.SelectedRowDelta(
        len(endpoint_rows.output_ids), hidden_size, direction_basis=None, device=device
    )
    input_handle = endpoint_hooks.register_input_embedding_delta_hook(
        input_layer, endpoint_rows.input_ids, input_delta.effective_delta
    )
    output_handle = canonical.register_output_delta_hook(
        output_layer, endpoint_rows.output_ids, output_delta.effective_delta
    )
    if input_delta.raw_delta is None or output_delta.raw_delta is None:
        raise RuntimeError("V2.2 requires unrestricted endpoint row deltas")

    documents = subject_stage1.load_frequency_documents(
        args.wikidata_dir, args.frequency_doc_start, args.frequency_docs
    )
    if len(documents) < args.frequency_docs:
        raise RuntimeError("V2.2 did not load the registered Wikipedia slice")
    sentence_pool = v2.corpus_sentence_pool(documents)
    targeted, targeted_report = folded.targeted_token_prompt_partitions(
        tok,
        sentence_pool,
        endpoint_rows.input_ids,
        fit_per_row=args.targeted_fit_per_row,
        development_per_row=args.targeted_development_per_row,
        certification_per_row=args.targeted_certification_per_row,
    )
    v2.write_json(output / "method" / "targeted_row_coverage.json", targeted_report)
    all_targeted = targeted["fit"] + targeted["development"] + targeted["certification"]
    corpus = v21.generic_corpus_partitions(
        sentence_pool,
        excluded=all_targeted,
        fit=args.corpus_fit_prompts,
        development=args.corpus_development_prompts,
        certification=args.corpus_certification_prompts,
        seed=args.seed,
    )
    frequency_counts = subject_stage1.token_frequency_counts(
        tok, documents, int(input_layer.weight.shape[0])
    )
    selected_counts = frequency_counts.index_select(
        0, torch.tensor(endpoint_rows.input_ids, dtype=torch.long)
    )
    input_caps = geometry.frequency_adjusted_caps(
        base_input_rows.cpu(),
        selected_counts,
        relative_cap=args.input_relative_cap,
        alpha=args.input_frequency_alpha,
    ).to(device)
    output_caps = geometry.relative_caps(
        base_output_rows, relative_cap=args.output_relative_cap
    ).to(device)

    synthetic_prefixes = synthetic.corpus_context_prefixes(
        documents, count=256, seed=args.seed + 71
    )
    synthetic_records = synthetic.build_synthetic_records(
        forget,
        count=args.synthetic_paraphrases,
        context_prefixes=synthetic_prefixes,
    )
    training_records = list(forget) + list(synthetic_records)
    forget_teacher_forced_contexts: set[str] = set()
    for sensitive_field in ("target_true", "target_new"):
        forget_teacher_forced_contexts.update(
            case.prompt
            for case in canonical.expand_sensitive_cases(
                training_records,
                tok,
                sensitive_field=sensitive_field,
                llama_like=llama_like,
            )
        )

    raw_fit_cases = v21.make_protection_cases(
        protection_fit,
        targeted=targeted["fit"],
        generic=corpus["fit"],
        forget=forget,
        role="fit",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    raw_development_cases = v21.make_protection_cases(
        protection_development,
        targeted=targeted["development"],
        generic=corpus["development"],
        forget=forget,
        role="development",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    fit_cases = [
        case
        for case in raw_fit_cases
        if case.prompt not in forget_teacher_forced_contexts
    ]
    development_cases = [
        case
        for case in raw_development_cases
        if case.prompt not in forget_teacher_forced_contexts
    ]
    collision_report = {
        "precedence_rule": "forget_positive_precedes_exact_retain_collision",
        "forget_teacher_forced_contexts": len(forget_teacher_forced_contexts),
        "fit_raw": len(raw_fit_cases),
        "fit_retained": len(fit_cases),
        "fit_excluded": len(raw_fit_cases) - len(fit_cases),
        "development_raw": len(raw_development_cases),
        "development_retained": len(development_cases),
        "development_excluded": len(raw_development_cases) - len(development_cases),
        "official_metrics_modified": False,
    }
    v2.write_json(
        output / "method" / "retain_collision_precedence.json", collision_report
    )
    print(
        f"Stage 1: cache {len(fit_cases)} fit and {len(development_cases)} "
        "development retain prompts"
    )
    fit_cache = v2.cache_protection(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    fit_target_ids, fit_base_target_log_probs = cache_retain_targets(
        model,
        tok,
        fit_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )
    development_cache = v2.cache_protection(
        model,
        tok,
        development_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    development_target_ids, development_base_target_log_probs = cache_retain_targets(
        model,
        tok,
        development_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )
    prompt_to_fit = {case.prompt: index for index, case in enumerate(fit_cases)}
    explicit_overlap_prompts = list(targeted["fit"]) + v2.same_subject_prompts(
        forget, role="fit", count=args.same_subject_prompts
    )
    overlap_indices = [
        prompt_to_fit[prompt]
        for prompt in explicit_overlap_prompts
        if prompt in prompt_to_fit
    ]
    selected_output_ids = torch.tensor(endpoint_rows.output_ids, dtype=torch.long)
    output_overlap_indices = (
        torch.nonzero(torch.isin(fit_target_ids, selected_output_ids), as_tuple=False)
        .flatten()
        .tolist()
    )
    overlap_indices.extend(int(index) for index in output_overlap_indices)
    overlap_indices = list(dict.fromkeys(overlap_indices))
    if not overlap_indices:
        raise RuntimeError("V2.2 constructed no explicit overlap retain stratum")
    v2.write_json(
        output / "method" / "retain_strata.json",
        {
            "explicit_input_overlap_prompts": len(explicit_overlap_prompts),
            "selected_output_target_overlap_cases": len(output_overlap_indices),
            "labeled_retain_target_cases": int((fit_target_ids >= 0).sum().item()),
            "generic_distribution_only_cases": int((fit_target_ids < 0).sum().item()),
            "unique_overlap_fit_cases": len(overlap_indices),
            "hard_tail_active": args.hard_tail_active,
            "overlap_active": args.overlap_retain_batch_size,
            "random_active": args.random_retain_batch_size,
            "active_maximum": args.active_retain_maximum,
            "present_on_every_endpoint_update": True,
        },
    )

    forget_batcher = folded.BalancedBatcher(
        len(training_records), args.forget_batch_size, args.seed + 1201
    )
    random_retain_batcher = folded.BalancedBatcher(
        len(fit_cache.cases), args.random_retain_batch_size, args.seed + 1301
    )
    overlap_batcher = folded.BalancedBatcher(
        len(overlap_indices), args.overlap_retain_batch_size, args.seed + 1401
    )
    hard_tail = joint.PersistentHardTail(args.hard_tail_capacity)
    tail_reports: List[Dict[str, Any]] = []
    training_log: List[Dict[str, Any]] = []
    candidate_states: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def endpoint_step(
        *,
        name: str,
        parameter: torch.nn.Parameter,
        caps: torch.Tensor,
        step_fraction: float,
        forget_batch: Sequence[Mapping[str, Any]],
        retain_indices: Sequence[int],
    ) -> Dict[str, Any]:
        input_delta.zero_grad(set_to_none=True)
        output_delta.zero_grad(set_to_none=True)
        margins = v2.records_margin_tensor(
            model, tok, forget_batch, device, llama_like=llama_like
        )
        forget_loss = F.relu(args.forget_margin_target - margins).square().mean()
        (
            retain_kl_mean,
            retain_kl_max,
            retain_drift_max,
            retain_target_drift_max,
        ) = active_retain_loss(
            model,
            tok,
            fit_cache,
            fit_target_ids,
            fit_base_target_log_probs,
            retain_indices,
            device,
            batch_size=args.capture_batch_size,
        )
        loss = (
            args.forget_weight * forget_loss
            + args.retain_kl_weight * retain_kl_mean
            + args.retain_top1_weight * retain_drift_max.square()
            + args.retain_target_weight * retain_target_drift_max.square()
            + args.delta_l2_weight * parameter.square().mean()
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite V2.2 {name} loss")
        loss.backward()
        if parameter.grad is None:
            raise RuntimeError(f"V2.2 {name} update produced no gradient")
        proposed = joint.normalized_row_step(
            parameter.grad.detach(), caps, fraction=step_fraction
        )
        old = parameter.detach().clone()
        before_forget = float(forget_loss.detach().cpu())
        before_score = joint.constraint_score(
            kl_max=float(retain_kl_max.detach().cpu()),
            drift_max=float(retain_drift_max.detach().cpu()),
            target_drift_max=float(retain_target_drift_max.detach().cpu()),
            kl_limit=args.protected_kl_max,
            drift_limit=args.protected_top1_drift_max,
            target_drift_limit=args.protected_target_drift_max,
        )
        accepted = 0.0
        candidate_forget_value = before_forget
        candidate_score = before_score
        for factor in BACKTRACKING_FACTORS:
            with torch.no_grad():
                parameter.copy_(old + float(factor) * proposed)
                geometry.apply_row_caps_(parameter, caps)
                candidate_margins = v2.records_margin_tensor(
                    model, tok, forget_batch, device, llama_like=llama_like
                )
                candidate_forget = (
                    F.relu(args.forget_margin_target - candidate_margins)
                    .square()
                    .mean()
                )
                (
                    _,
                    candidate_kl_max,
                    candidate_drift_max,
                    candidate_target_drift_max,
                ) = active_retain_loss(
                    model,
                    tok,
                    fit_cache,
                    fit_target_ids,
                    fit_base_target_log_probs,
                    retain_indices,
                    device,
                    batch_size=args.capture_batch_size,
                )
            local_score = joint.constraint_score(
                kl_max=float(candidate_kl_max.cpu()),
                drift_max=float(candidate_drift_max.cpu()),
                target_drift_max=float(candidate_target_drift_max.cpu()),
                kl_limit=args.protected_kl_max,
                drift_limit=args.protected_top1_drift_max,
                target_drift_limit=args.protected_target_drift_max,
            )
            if joint.accept_trust_region_candidate(
                before_forget=before_forget,
                candidate_forget=float(candidate_forget.cpu()),
                before_constraint_score=before_score,
                candidate_constraint_score=local_score,
            ):
                accepted = float(factor)
                candidate_forget_value = float(candidate_forget.cpu())
                candidate_score = local_score
                break
        if accepted == 0.0:
            with torch.no_grad():
                parameter.copy_(old)
        input_delta.zero_grad(set_to_none=True)
        output_delta.zero_grad(set_to_none=True)
        return {
            "endpoint": name,
            "accepted_factor": accepted,
            "before_forget_loss": before_forget,
            "after_forget_loss": candidate_forget_value,
            "before_constraint_score": before_score,
            "after_constraint_score": candidate_score,
        }

    print("Stage 2: alternating joint forget/retain endpoint optimization")
    for step in range(1, args.steps + 1):
        if step == 1 or step % args.hard_tail_refresh_every == 0:
            fit_kl, fit_drift, fit_target_drift = protection_vectors(
                model,
                tok,
                fit_cache,
                fit_target_ids,
                fit_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
            )
            tail = hard_tail.refresh(
                fit_kl,
                fit_drift,
                fit_target_drift,
                kl_limit=args.protected_kl_max,
                drift_limit=args.protected_top1_drift_max,
                target_drift_limit=args.protected_target_drift_max,
                add=args.hard_tail_add,
            )
            tail["step"] = step
            tail_reports.append(tail)
        forget_indices = forget_batcher.next()
        forget_batch = [training_records[index] for index in forget_indices]
        random_indices = random_retain_batcher.next()
        overlap_local = overlap_batcher.next()
        overlap_active = [overlap_indices[index] for index in overlap_local]
        retain_indices = joint.compose_active_retain_indices(
            random_indices=random_indices,
            overlap_indices=overlap_active,
            hard_indices=hard_tail.indices[: args.hard_tail_active],
            maximum=args.active_retain_maximum,
        )
        input_update = endpoint_step(
            name="embedding",
            parameter=input_delta.raw_delta,
            caps=input_caps,
            step_fraction=args.input_step_cap_fraction,
            forget_batch=forget_batch,
            retain_indices=retain_indices,
        )
        output_update = endpoint_step(
            name="lm_head",
            parameter=output_delta.raw_delta,
            caps=output_caps,
            step_fraction=args.output_step_cap_fraction,
            forget_batch=forget_batch,
            retain_indices=retain_indices,
        )
        if step % args.check_every == 0:
            direct = v2.margin_report(
                model,
                tok,
                forget,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            synth_report = v2.margin_report(
                model,
                tok,
                synthetic_records,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            fit_report = evaluate_endpoint_protection(
                model,
                tok,
                fit_cache,
                fit_target_ids,
                fit_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
                args=args,
            )
            development = evaluate_endpoint_protection(
                model,
                tok,
                development_cache,
                development_target_ids,
                development_base_target_log_probs,
                device,
                batch_size=args.capture_batch_size,
                args=args,
            )
            input_cap = geometry.cap_report(input_delta.raw_delta, input_caps)
            output_cap = geometry.cap_report(output_delta.raw_delta, output_caps)
            total_norm = float(
                (
                    input_delta.raw_delta.square().sum()
                    + output_delta.raw_delta.square().sum()
                )
                .sqrt()
                .detach()
                .cpu()
            )
            passed = bool(
                direct["passed"]
                and synth_report["passed"]
                and fit_report["passed"]
                and development["passed"]
                and input_cap["passed"]
                and output_cap["passed"]
            )
            row = {
                "step": step,
                "input_update": input_update,
                "output_update": output_update,
                "hard_tail_size": len(hard_tail.indices),
                "direct": direct,
                "synthetic": synth_report,
                "protection_fit": fit_report,
                "protection_development": development,
                "input_caps": input_cap,
                "output_caps": output_cap,
                "total_delta_norm": total_norm,
                "passed": passed,
            }
            training_log.append(row)
            if passed:
                candidate_states[step] = (
                    input_delta.raw_delta.detach().cpu().clone(),
                    output_delta.raw_delta.detach().cpu().clone(),
                )
            print(
                f"  step {step:4d}: direct fail {direct['failures']}, "
                f"synth fail {synth_report['failures']}, "
                f"input accept {input_update['accepted_factor']:.4f}, "
                f"head accept {output_update['accepted_factor']:.4f}, "
                f"fit KL {fit_report['topk_kl_max']:.6f}, "
                f"dev KL {development['topk_kl_max']:.6f}, "
                f"dev top1 {development['top1_logprob_abs_max']:.6f}, "
                f"dev target {development['target_logprob_abs_max']:.6f}, "
                f"pass={passed}"
            )
    v2.write_json(
        output / "method" / "joint_training.json",
        {"hard_tail_refreshes": tail_reports, "checks": training_log},
    )
    if not candidate_states:
        completion_failure(
            output,
            phase="joint_forget_retain_development",
            detail={"checks": training_log, "last_hard_tail": tail_reports[-1]},
        )
        raise RuntimeError("V2.2 produced no development-passing candidate")
    selected_step = min(
        candidate_states,
        key=lambda value: next(
            row["total_delta_norm"] for row in training_log if row["step"] == value
        ),
    )
    selected_input, selected_output = candidate_states[selected_step]
    with torch.no_grad():
        input_delta.raw_delta.copy_(selected_input.to(device))
        output_delta.raw_delta.copy_(selected_output.to(device))

    print(f"Stage 3: open certification once for selected step {selected_step}")
    protection_certification = v2.load_partition(
        protocol_dir, manifest, "protection_certification", permitted=True
    )
    raw_certification_cases = v21.make_protection_cases(
        protection_certification,
        targeted=targeted["certification"],
        generic=corpus["certification"],
        forget=forget,
        role="certification",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    certification_cases = [
        case
        for case in raw_certification_cases
        if case.prompt not in forget_teacher_forced_contexts
    ]
    collision_report.update(
        {
            "certification_raw": len(raw_certification_cases),
            "certification_retained": len(certification_cases),
            "certification_excluded": len(raw_certification_cases)
            - len(certification_cases),
        }
    )
    v2.write_json(
        output / "method" / "retain_collision_precedence.json", collision_report
    )
    old_input = input_delta.raw_delta.detach().clone()
    old_output = output_delta.raw_delta.detach().clone()
    with torch.no_grad():
        input_delta.raw_delta.zero_()
        output_delta.raw_delta.zero_()
    certification_cache = v2.cache_protection(
        model,
        tok,
        certification_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    (
        certification_target_ids,
        certification_base_target_log_probs,
    ) = cache_retain_targets(
        model,
        tok,
        certification_cases,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
    )
    with torch.no_grad():
        input_delta.raw_delta.copy_(old_input)
        output_delta.raw_delta.copy_(old_output)
    pre = endpoint_certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
        certification_target_ids,
        certification_base_target_log_probs,
        input_delta,
        output_delta,
        input_caps,
        output_caps,
        device,
        args=args,
    )
    v2.write_json(output / "method" / "pre_materialization_certificate.json", pre)
    if not pre["passed"]:
        completion_failure(output, phase="certification", detail=pre)
        raise RuntimeError("V2.2 selected candidate failed certification")

    input_handle.remove()
    output_handle.remove()
    endpoint_hooks.materialize_input_delta(
        input_layer, endpoint_rows.input_ids, input_delta.raw_delta.detach().cpu()
    )
    canonical.materialize_output_delta(
        output_layer, endpoint_rows.output_ids, output_delta.raw_delta.detach().cpu()
    )
    post = endpoint_certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
        certification_target_ids,
        certification_base_target_log_probs,
        input_delta,
        output_delta,
        input_caps,
        output_caps,
        device,
        args=args,
    )
    transformer_after = v2.transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )
    all_model_parameters_require_grad_false = all(
        not parameter.requires_grad for parameter in model.parameters()
    )
    post["pre_post_acceptance_match"] = bool(
        pre["direct"]["failures"] == post["direct"]["failures"]
        and pre["synthetic"]["failures"] == post["synthetic"]["failures"]
        and pre["protection"]["passed"] == post["protection"]["passed"]
    )
    post["transformer_bit_identical"] = transformer_before == transformer_after
    post[
        "all_model_parameters_require_grad_false"
    ] = all_model_parameters_require_grad_false
    post["passed"] = bool(
        post["passed"]
        and post["pre_post_acceptance_match"]
        and post["transformer_bit_identical"]
        and post["all_model_parameters_require_grad_false"]
    )
    v2.write_json(output / "method" / "post_materialization_certificate.json", post)
    if not post["passed"]:
        completion_failure(output, phase="materialization", detail=post)
        raise RuntimeError("V2.2 materialization certificate failed")

    candidate_path = output / "method" / "v2_2_candidate_sparse_rows.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": joint.PROTOCOL,
            "base_model_path": str(args.model_path),
            "input_row_ids": endpoint_rows.input_ids,
            "input_rows": input_layer.weight.index_select(0, input_ids_tensor)
            .detach()
            .cpu(),
            "output_row_ids": endpoint_rows.output_ids,
            "output_rows": output_layer.weight.index_select(0, output_ids_tensor)
            .detach()
            .cpu(),
            "selected_step": selected_step,
            "split_manifest_sha256": v2.sha256_file(manifest_path),
            "registry_sha256": v2.sha256_file(registry_path),
            "transformer_sha256": transformer_after,
        },
        candidate_path,
    )
    completion = {
        "schema_version": 1,
        "protocol": joint.PROTOCOL,
        "passed": True,
        "selected_step": selected_step,
        "joint_forget_retain_every_update": True,
        "persistent_hard_tail": True,
        "embedding_rows_changed": int(
            (input_delta.raw_delta.norm(dim=1) > 0).sum().item()
        ),
        "lm_head_rows_changed": int(
            (output_delta.raw_delta.norm(dim=1) > 0).sum().item()
        ),
        "transformer_bit_identical": True,
        "all_model_parameters_require_grad_false": (
            all_model_parameters_require_grad_false
        ),
        "external_classifier": False,
        "runtime_gate": False,
        "candidate_saved": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": v2.sha256_file(candidate_path),
        "eligible_for_separate_official_evaluation": True,
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "claim_scope": "internal_endpoint_behavioral_unlearning_not_latent_erasure",
    }
    v2.write_json(output / "method" / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
