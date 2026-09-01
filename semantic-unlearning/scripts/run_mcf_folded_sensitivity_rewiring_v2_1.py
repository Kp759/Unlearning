#!/usr/bin/env python3
"""Run MCF V2.1 folded-sensitivity bi-endpoint rewiring.

The contextual classifier is the selected LM-head row itself: ``h @ delta_w``.
It is solved deterministically from signed target_true/target_new cells.  Only
if that head-only solution fails are selected subject embedding rows changed,
using projected normalized steps guarded by fit-protection backtracking.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import build_mcf_biendpoint_nullspace_rewiring_v2_split as split_builder
import gagd_compare as gagd
import mcf_biendpoint_nullspace_rewiring_v2_core as geometry
import mcf_folded_sensitivity_rewiring_v2_1_core as folded
import mcf_sure_directional_emb_lm_stage1 as endpoint_hooks
import mcf_sure_subject_directional_emb_stage1 as subject_stage1
import mcf_synthetic_paraphrase_templates as synthetic
import run_mcf_biendpoint_nullspace_rewiring_v2 as v2
import sure_canonical_core as canonical


CORRECTION_FLOORS = (2.0, 4.0, 6.0, 8.0)
BACKTRACKING_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)


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
    parser.add_argument("--targeted-fit-per-row", type=int, default=2)
    parser.add_argument("--targeted-development-per-row", type=int, default=1)
    parser.add_argument("--targeted-certification-per-row", type=int, default=1)
    parser.add_argument("--head-ridge", type=float, default=1e-4)
    parser.add_argument("--output-rank-cap", type=int, default=512)
    parser.add_argument("--output-relative-cap", type=float, default=0.15)
    parser.add_argument("--minimum-signed-correction", type=float, default=0.1)
    parser.add_argument("--input-jacobian-sketches", type=int, default=128)
    parser.add_argument("--input-rank-cap", type=int, default=64)
    parser.add_argument("--input-relative-cap", type=float, default=0.5)
    parser.add_argument("--input-frequency-alpha", type=float, default=0.25)
    parser.add_argument("--rescue-steps", type=int, default=400)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--head-refit-every", type=int, default=100)
    parser.add_argument("--forget-batch-size", type=int, default=8)
    parser.add_argument("--protection-batch-size", type=int, default=32)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--per-step-cap-fraction", type=float, default=0.01)
    parser.add_argument("--forget-margin-target", type=float, default=6.0)
    parser.add_argument("--minimum-forget-margin", type=float, default=0.1)
    parser.add_argument("--protection-topk", type=int, default=64)
    parser.add_argument("--protected-kl-mean-max", type=float, default=1e-4)
    parser.add_argument("--protected-kl-max", type=float, default=1e-2)
    parser.add_argument("--protected-top1-drift-max", type=float, default=5e-2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.gradient_checkpointing = False
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V2.1 is locked to consumed seed 1 / 50 facts")
    if args.frequency_doc_start < 20:
        parser.error("Wikipedia documents 0:20 remain reserved for official PPL")
    if (
        args.rescue_steps % args.check_every
        or args.rescue_steps % args.head_refit_every
    ):
        parser.error("rescue steps must divide check and head-refit intervals")
    return args


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    architecture = registry.get("architecture", {})
    if (
        registry.get("protocol") != folded.PROTOCOL
        or registry.get("status")
        != "training_only_implementation_available_not_executed"
        or architecture.get("transformer_frozen") is not True
        or architecture.get("external_classifier") is not False
        or architecture.get("runtime_gate") is not False
        or architecture.get("sidecar") is not False
    ):
        raise RuntimeError("V2.1 registry architecture/status mismatch")
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
            "targeted_common_row_fit_prompts": args.targeted_fit_per_row,
            "targeted_common_row_development_prompts": args.targeted_development_per_row,
            "targeted_common_row_certification_prompts": args.targeted_certification_per_row,
        },
        "folded_head_solver": {
            "correction_floor_sweep": list(CORRECTION_FLOORS),
            "ridge": args.head_ridge,
            "protected_hidden_rank_cap": args.output_rank_cap,
            "output_relative_row_cap": args.output_relative_cap,
            "minimum_signed_cell_correction": args.minimum_signed_correction,
        },
        "embedding_rescue": {
            "input_jacobian_sketches": args.input_jacobian_sketches,
            "input_protected_rank_cap": args.input_rank_cap,
            "input_relative_row_cap": args.input_relative_cap,
            "input_frequency_alpha": args.input_frequency_alpha,
            "steps": args.rescue_steps,
            "check_every": args.check_every,
            "balanced_forget_batch_size": args.forget_batch_size,
            "active_protection_batch_size": args.protection_batch_size,
            "per_step_cap_fraction": args.per_step_cap_fraction,
            "backtracking_factors": list(BACKTRACKING_FACTORS),
            "lm_head_refit_every": args.head_refit_every,
            "adam_forbidden": True,
        },
        "acceptance": {
            "minimum_forget_margin": args.minimum_forget_margin,
            "protected_topk_kl_mean_max": args.protected_kl_mean_max,
            "protected_topk_kl_absolute_max": args.protected_kl_max,
            "protected_top1_logprob_abs_max": args.protected_top1_drift_max,
        },
    }
    for section, values in expected.items():
        registered = registry.get(section)
        if not isinstance(registered, Mapping):
            raise RuntimeError(f"V2.1 registry lacks {section}")
        for key, value in values.items():
            if registered.get(key) != value:
                raise RuntimeError(
                    f"V2.1 argument differs from registry: {section}.{key}"
                )


def unique_cases(
    cases: Sequence[canonical.SensitivePredictionCase],
) -> List[canonical.SensitivePredictionCase]:
    result: Dict[str, canonical.SensitivePredictionCase] = {}
    for case in cases:
        result.setdefault(case.prompt, case)
    return list(result.values())


def generic_corpus_partitions(
    pool: Sequence[str],
    *,
    excluded: Sequence[str],
    fit: int,
    development: int,
    certification: int,
    seed: int,
) -> Dict[str, List[str]]:
    excluded_set = set(excluded)
    available = [value for value in pool if value not in excluded_set]
    need = fit + development + certification
    if len(available) < need:
        raise RuntimeError(
            f"only {len(available)} unallocated corpus prompts, need {need}"
        )
    rng = random.Random(int(seed) + 7789)
    selected = rng.sample(available, need)
    return {
        "fit": selected[:fit],
        "development": selected[fit : fit + development],
        "certification": selected[fit + development :],
    }


def make_protection_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    targeted: Sequence[str],
    generic: Sequence[str],
    forget: Sequence[Mapping[str, Any]],
    role: str,
    tok: Any,
    llama_like: bool,
    same_subject_count: int,
) -> List[canonical.SensitivePredictionCase]:
    targeted_cases = v2.plain_cases(targeted, case_offset=20_000_000)
    ordinary = v2.protection_cases(
        records,
        generic,
        forget,
        role=role,
        tok=tok,
        llama_like=llama_like,
        same_subject_count=same_subject_count,
    )
    return unique_cases(targeted_cases + ordinary)


def build_folded_cells(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    output_row_ids: Sequence[int],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> folded.FoldedCells:
    row_lookup = {int(token_id): index for index, token_id in enumerate(output_row_ids)}
    hidden_parts: List[torch.Tensor] = []
    row_parts: List[torch.Tensor] = []
    sign_parts: List[torch.Tensor] = []
    case_parts: List[torch.Tensor] = []
    roles: List[str] = []
    for role, sign in (("target_true", -1.0), ("target_new", 1.0)):
        cases = canonical.expand_sensitive_cases(
            records, tok, sensitive_field=role, llama_like=llama_like
        )
        hidden = canonical.forward_last_hidden(
            model, tok, cases, device, batch_size=batch_size
        )
        token_ids = canonical.official_target_ids(
            tok, cases, llama_like=llama_like, device=device
        )
        local_rows: List[int] = []
        for token_id in token_ids.detach().cpu().tolist():
            if int(token_id) not in row_lookup:
                raise RuntimeError(f"folded cell token {token_id} was not selected")
            local_rows.append(row_lookup[int(token_id)])
        hidden_parts.append(hidden)
        row_parts.append(torch.tensor(local_rows, dtype=torch.long, device=device))
        sign_parts.append(torch.full((len(cases),), sign, device=device))
        case_parts.append(
            torch.tensor(
                [int(case.case_id) for case in cases], dtype=torch.long, device=device
            )
        )
        roles.extend([role] * len(cases))
    return folded.FoldedCells(
        hidden=torch.cat(hidden_parts),
        row_indices=torch.cat(row_parts),
        signs=torch.cat(sign_parts),
        case_ids=torch.cat(case_parts),
        roles=roles,
    )


def arm_report(
    model: torch.nn.Module,
    tok: Any,
    forget: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
    development_cache: v2.ProtectionCache,
    cells: folded.FoldedCells,
    output_delta: canonical.SelectedRowDelta,
    device: torch.device,
    *,
    correction_floor: float,
    solver_report: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    direct = v2.margin_report(
        model,
        tok,
        forget,
        device,
        llama_like=canonical.is_llama_like(model, tok),
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    synth = v2.margin_report(
        model,
        tok,
        synthetic_records,
        device,
        llama_like=canonical.is_llama_like(model, tok),
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    protection = v2.evaluate_protection(
        model,
        tok,
        development_cache,
        device,
        batch_size=args.capture_batch_size,
        args=args,
    )
    if output_delta.raw_delta is None:
        raise RuntimeError("V2.1 output delta must be unrestricted")
    signed = folded.signed_cell_report(
        cells, output_delta.raw_delta, minimum=args.minimum_signed_correction
    )
    result = {
        "correction_floor": float(correction_floor),
        "solver": dict(solver_report),
        "direct": direct,
        "synthetic": synth,
        "protection_development": protection,
        "signed_cells": signed,
        "total_delta_norm": float(output_delta.raw_delta.norm().detach().cpu()),
    }
    result["passed"] = bool(
        direct["passed"]
        and synth["passed"]
        and protection["passed"]
        and signed["passed"]
    )
    return result


def fit_head_arms(
    model: torch.nn.Module,
    tok: Any,
    forget: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
    training_records: Sequence[Mapping[str, Any]],
    development_cache: v2.ProtectionCache,
    output_delta: canonical.SelectedRowDelta,
    output_basis: torch.Tensor,
    output_caps: torch.Tensor,
    output_row_ids: Sequence[int],
    device: torch.device,
    *,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], Dict[float, torch.Tensor], folded.FoldedCells]:
    cells = build_folded_cells(
        model,
        tok,
        training_records,
        output_row_ids,
        device,
        llama_like=canonical.is_llama_like(model, tok),
        batch_size=args.capture_batch_size,
    )
    if output_delta.raw_delta is None:
        raise RuntimeError("V2.1 output delta must be unrestricted")
    reports: List[Dict[str, Any]] = []
    states: Dict[float, torch.Tensor] = {}
    for floor in CORRECTION_FLOORS:
        candidate, solver = folded.solve_folded_rows(
            cells,
            n_rows=output_delta.n_rows,
            hidden_size=output_delta.hidden_size,
            protected_basis=output_basis,
            correction_floor=floor,
            ridge=args.head_ridge,
            row_caps=output_caps,
        )
        with torch.no_grad():
            output_delta.raw_delta.copy_(candidate)
        report = arm_report(
            model,
            tok,
            forget,
            synthetic_records,
            development_cache,
            cells,
            output_delta,
            device,
            correction_floor=floor,
            solver_report=solver,
            args=args,
        )
        reports.append(report)
        states[floor] = candidate.detach().cpu().clone()
        print(
            f"  floor {floor:.1f}: direct fail {report['direct']['failures']}, "
            f"synth fail {report['synthetic']['failures']}, "
            f"signed fail {report['signed_cells']['failures']}, "
            f"dev KL {report['protection_development']['topk_kl_max']:.6f}, "
            f"dev top1 {report['protection_development']['top1_logprob_abs_max']:.6f}, "
            f"pass={report['passed']}"
        )
    return reports, states, cells


def best_training_floor(reports: Sequence[Mapping[str, Any]]) -> float:
    selected = min(
        reports,
        key=lambda report: (
            int(report["direct"]["failures"]) + int(report["synthetic"]["failures"]),
            int(report["signed_cells"]["failures"]),
            float(report["protection_development"]["topk_kl_max"]),
            float(report["correction_floor"]),
        ),
    )
    return float(selected["correction_floor"])


def active_protection_values(
    model: torch.nn.Module,
    tok: Any,
    cache: v2.ProtectionCache,
    indices: Sequence[int],
    device: torch.device,
) -> tuple[float, float]:
    batch = v2.cache_batch(cache, indices, device)
    with torch.no_grad():
        logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
        kl, drift = geometry.protection_loss(
            logits,
            **{key: value for key, value in batch.items() if key != "cases"},
        )
    return float(kl.max().cpu()), float(drift.abs().max().cpu())


def certificate(
    model: torch.nn.Module,
    tok: Any,
    forget: Sequence[Mapping[str, Any]],
    synthetic_records: Sequence[Mapping[str, Any]],
    protection_cache: v2.ProtectionCache,
    input_delta: canonical.SelectedRowDelta,
    output_delta: canonical.SelectedRowDelta,
    input_caps: torch.Tensor,
    output_caps: torch.Tensor,
    device: torch.device,
    *,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    llama_like = canonical.is_llama_like(model, tok)
    direct = v2.margin_report(
        model,
        tok,
        forget,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    synth = v2.margin_report(
        model,
        tok,
        synthetic_records,
        device,
        llama_like=llama_like,
        batch_size=args.capture_batch_size,
        minimum=args.minimum_forget_margin,
    )
    protection = v2.evaluate_protection(
        model,
        tok,
        protection_cache,
        device,
        batch_size=args.capture_batch_size,
        args=args,
    )
    if input_delta.raw_delta is None or output_delta.raw_delta is None:
        raise RuntimeError("V2.1 endpoint deltas must be unrestricted")
    input_cap = geometry.cap_report(input_delta.raw_delta, input_caps)
    output_cap = geometry.cap_report(output_delta.raw_delta, output_caps)
    result = {
        "direct": direct,
        "synthetic": synth,
        "protection": protection,
        "input_caps": input_cap,
        "output_caps": output_cap,
    }
    result["passed"] = bool(
        direct["passed"]
        and synth["passed"]
        and protection["passed"]
        and input_cap["passed"]
        and output_cap["passed"]
    )
    return result


def completion_failure(output: Path, *, phase: str, detail: Mapping[str, Any]) -> None:
    v2.write_json(
        output / "method" / "completion.json",
        {
            "schema_version": 1,
            "protocol": folded.PROTOCOL,
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
        raise RuntimeError("V2.1 split manifest protocol mismatch")
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
    endpoint_rows = geometry.select_endpoint_rows(forget, tok, llama_like=llama_like)
    overlap = geometry.endpoint_overlap_report(endpoint_rows)
    v2.write_json(output / "method" / "endpoint_overlap_manifest.json", overlap)
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

    documents = subject_stage1.load_frequency_documents(
        args.wikidata_dir, args.frequency_doc_start, args.frequency_docs
    )
    if len(documents) < args.frequency_docs:
        raise RuntimeError("V2.1 did not load the registered Wikipedia slice")
    sentence_pool = v2.corpus_sentence_pool(documents)
    targeted, targeted_report = folded.targeted_token_prompt_partitions(
        tok,
        sentence_pool,
        endpoint_rows.input_ids,
        fit_per_row=args.targeted_fit_per_row,
        development_per_row=args.targeted_development_per_row,
        certification_per_row=args.targeted_certification_per_row,
    )
    v2.write_json(
        output / "method" / "targeted_input_row_coverage.json", targeted_report
    )
    if not targeted_report["partitions_pairwise_disjoint"]:
        raise RuntimeError("V2.1 targeted protection partitions overlap")
    all_targeted = targeted["fit"] + targeted["development"] + targeted["certification"]
    corpus = generic_corpus_partitions(
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
    fit_cases = make_protection_cases(
        protection_fit,
        targeted=targeted["fit"],
        generic=corpus["fit"],
        forget=forget,
        role="fit",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    development_cases = make_protection_cases(
        protection_development,
        targeted=targeted["development"],
        generic=corpus["development"],
        forget=forget,
        role="development",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    print(
        f"Stage 1: cache {len(fit_cases)} fit and {len(development_cases)} development protection prompts"
    )
    fit_cache = v2.cache_protection(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    development_cache = v2.cache_protection(
        model,
        tok,
        development_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    hidden_sketch = v2.compress_hidden_states(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        rows=args.output_rank_cap,
    )
    output_basis = geometry.common_basis(
        hidden_sketch, max_rank=args.output_rank_cap
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
    print("Stage 2: solve folded signed LM-head classifier arms")
    head_reports, head_states, cells = fit_head_arms(
        model,
        tok,
        forget,
        synthetic_records,
        training_records,
        development_cache,
        output_delta,
        output_basis,
        output_caps,
        endpoint_rows.output_ids,
        device,
        args=args,
    )
    v2.write_json(
        output / "method" / "folded_head_arm_sweep.json", {"arms": head_reports}
    )
    selected_floor = folded.choose_arm(head_reports)
    selected_stage = "head_only"
    selected_input = torch.zeros_like(base_input_rows).cpu()
    selected_output: torch.Tensor
    development_candidates: List[Dict[str, Any]] = []
    candidate_states: Dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
    if selected_floor is not None:
        selected_output = head_states[selected_floor]
        with torch.no_grad():
            output_delta.raw_delta.copy_(selected_output.to(device))
        selected_report = next(
            report
            for report in head_reports
            if report["correction_floor"] == selected_floor
        )
        development_candidates.append({"stage": "head_only", **selected_report})
    else:
        best_floor = best_training_floor(head_reports)
        with torch.no_grad():
            output_delta.raw_delta.copy_(head_states[best_floor].to(device))
            input_delta.raw_delta.zero_()
        print(
            f"Stage 3: folded head alone failed; begin bounded embedding rescue from floor {best_floor:.1f}"
        )
        input_bases, input_basis_report = v2.build_input_bases(
            model,
            tok,
            fit_cache,
            device,
            input_delta=input_delta,
            output_delta=output_delta,
            sketches=args.input_jacobian_sketches,
            batch_size=args.capture_batch_size,
            topk=args.protection_topk,
            max_rank=args.input_rank_cap,
        )
        v2.write_json(
            output / "method" / "protected_input_basis_report.json",
            input_basis_report,
        )
        forget_batcher = folded.BalancedBatcher(
            len(training_records), args.forget_batch_size, args.seed + 9101
        )
        protection_batcher = folded.BalancedBatcher(
            len(fit_cache.cases), args.protection_batch_size, args.seed + 9203
        )
        rescue_log: List[Dict[str, Any]] = []
        for step in range(1, args.rescue_steps + 1):
            indices = forget_batcher.next()
            batch = [training_records[index] for index in indices]
            input_delta.zero_grad(set_to_none=True)
            output_delta.zero_grad(set_to_none=True)
            margins = v2.records_margin_tensor(
                model, tok, batch, device, llama_like=llama_like
            )
            before_loss = F.relu(args.forget_margin_target - margins).square().mean()
            before_loss.backward()
            gradient = input_delta.raw_delta.grad
            if gradient is None:
                raise RuntimeError("V2.1 embedding rescue produced no gradient")
            projected_gradient = gradient.detach().float().clone()
            geometry.project_rowwise_(projected_gradient, input_bases)
            proposed = folded.normalized_projected_step(
                projected_gradient,
                input_caps,
                fraction=args.per_step_cap_fraction,
            )
            old_input = input_delta.raw_delta.detach().clone()
            protect_indices = protection_batcher.next()
            accepted_factor = 0.0
            accepted_loss = float(before_loss.detach().cpu())
            for factor in BACKTRACKING_FACTORS:
                with torch.no_grad():
                    input_delta.raw_delta.copy_(old_input + factor * proposed)
                    geometry.project_rowwise_(input_delta.raw_delta, input_bases)
                    geometry.apply_row_caps_(input_delta.raw_delta, input_caps)
                    candidate_margins = v2.records_margin_tensor(
                        model, tok, batch, device, llama_like=llama_like
                    )
                    candidate_loss = (
                        F.relu(args.forget_margin_target - candidate_margins)
                        .square()
                        .mean()
                    )
                kl_max, drift_max = active_protection_values(
                    model, tok, fit_cache, protect_indices, device
                )
                if (
                    float(candidate_loss.cpu())
                    <= float(before_loss.detach().cpu()) + 1e-7
                    and kl_max <= args.protected_kl_max
                    and drift_max <= args.protected_top1_drift_max
                ):
                    accepted_factor = factor
                    accepted_loss = float(candidate_loss.cpu())
                    break
            if accepted_factor == 0.0:
                with torch.no_grad():
                    input_delta.raw_delta.copy_(old_input)

            if step % args.head_refit_every == 0:
                hidden_sketch = v2.compress_hidden_states(
                    model,
                    tok,
                    fit_cases,
                    device,
                    batch_size=args.capture_batch_size,
                    rows=args.output_rank_cap,
                )
                output_basis = geometry.common_basis(
                    hidden_sketch, max_rank=args.output_rank_cap
                ).to(device)
                refit_reports, refit_states, cells = fit_head_arms(
                    model,
                    tok,
                    forget,
                    synthetic_records,
                    training_records,
                    development_cache,
                    output_delta,
                    output_basis,
                    output_caps,
                    endpoint_rows.output_ids,
                    device,
                    args=args,
                )
                best_floor = folded.choose_arm(refit_reports) or best_training_floor(
                    refit_reports
                )
                with torch.no_grad():
                    output_delta.raw_delta.copy_(refit_states[best_floor].to(device))

            if step % args.check_every == 0:
                current = certificate(
                    model,
                    tok,
                    forget,
                    synthetic_records,
                    development_cache,
                    input_delta,
                    output_delta,
                    input_caps,
                    output_caps,
                    device,
                    args=args,
                )
                total_norm = float(
                    (
                        input_delta.raw_delta.square().sum()
                        + output_delta.raw_delta.square().sum()
                    )
                    .sqrt()
                    .detach()
                    .cpu()
                )
                row = {
                    "step": step,
                    "accepted_factor": accepted_factor,
                    "batch_loss": accepted_loss,
                    "correction_floor": best_floor,
                    "total_delta_norm": total_norm,
                    **current,
                }
                rescue_log.append(row)
                if current["passed"]:
                    candidate_states[step] = (
                        input_delta.raw_delta.detach().cpu().clone(),
                        output_delta.raw_delta.detach().cpu().clone(),
                        best_floor,
                    )
                print(
                    f"  rescue {step:3d}: direct fail {current['direct']['failures']}, "
                    f"synth fail {current['synthetic']['failures']}, "
                    f"dev KL {current['protection']['topk_kl_max']:.6f}, "
                    f"dev top1 {current['protection']['top1_logprob_abs_max']:.6f}, "
                    f"pass={current['passed']}"
                )
        v2.write_json(
            output / "method" / "embedding_rescue.json", {"steps": rescue_log}
        )
        if not candidate_states:
            completion_failure(
                output,
                phase="folded_head_and_embedding_rescue_development",
                detail={"head_arms": head_reports, "rescue": rescue_log},
            )
            raise RuntimeError("V2.1 produced no development-passing candidate")
        selected_step = min(
            candidate_states,
            key=lambda value: next(
                row["total_delta_norm"] for row in rescue_log if row["step"] == value
            ),
        )
        selected_input, selected_output, selected_floor = candidate_states[
            selected_step
        ]
        selected_stage = f"embedding_rescue_step_{selected_step}"
        with torch.no_grad():
            input_delta.raw_delta.copy_(selected_input.to(device))
            output_delta.raw_delta.copy_(selected_output.to(device))

    print(
        f"Stage 4: open certification once for {selected_stage}, floor {selected_floor}"
    )
    protection_certification = v2.load_partition(
        protocol_dir, manifest, "protection_certification", permitted=True
    )
    certification_cases = make_protection_cases(
        protection_certification,
        targeted=targeted["certification"],
        generic=corpus["certification"],
        forget=forget,
        role="certification",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
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
    with torch.no_grad():
        input_delta.raw_delta.copy_(old_input)
        output_delta.raw_delta.copy_(old_output)
    pre = certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
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
        raise RuntimeError("V2.1 selected candidate failed certification")

    input_handle.remove()
    output_handle.remove()
    endpoint_hooks.materialize_input_delta(
        input_layer, endpoint_rows.input_ids, input_delta.raw_delta.detach().cpu()
    )
    canonical.materialize_output_delta(
        output_layer, endpoint_rows.output_ids, output_delta.raw_delta.detach().cpu()
    )
    post = certificate(
        model,
        tok,
        forget,
        synthetic_records,
        certification_cache,
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
    acceptance_match = bool(
        pre["passed"] == post["passed"]
        and pre["direct"]["failures"] == post["direct"]["failures"]
        and pre["synthetic"]["failures"] == post["synthetic"]["failures"]
    )
    post_report = {
        **post,
        "pre_post_acceptance_match": acceptance_match,
        "transformer_sha256_before": transformer_before,
        "transformer_sha256_after": transformer_after,
        "transformer_bit_identical": transformer_before == transformer_after,
    }
    post_report["passed"] = bool(
        post["passed"] and acceptance_match and transformer_before == transformer_after
    )
    v2.write_json(
        output / "method" / "post_materialization_certificate.json", post_report
    )
    if not post_report["passed"]:
        completion_failure(output, phase="materialization", detail=post_report)
        raise RuntimeError("V2.1 materialization certificate failed")

    candidate_path = output / "method" / "v2_1_candidate_sparse_rows.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": folded.PROTOCOL,
            "base_model_path": str(args.model_path),
            "input_row_ids": endpoint_rows.input_ids,
            "input_rows": input_layer.weight.index_select(0, input_ids_tensor)
            .detach()
            .cpu(),
            "output_row_ids": endpoint_rows.output_ids,
            "output_rows": output_layer.weight.index_select(0, output_ids_tensor)
            .detach()
            .cpu(),
            "selected_stage": selected_stage,
            "selected_correction_floor": selected_floor,
            "split_manifest_sha256": v2.sha256_file(manifest_path),
            "registry_sha256": v2.sha256_file(registry_path),
            "transformer_sha256": transformer_after,
        },
        candidate_path,
    )
    completion = {
        "schema_version": 1,
        "protocol": folded.PROTOCOL,
        "passed": True,
        "selected_stage": selected_stage,
        "selected_correction_floor": selected_floor,
        "folded_classifier_materialized_in_lm_head": True,
        "embedding_rows_changed": int(
            (input_delta.raw_delta.norm(dim=1) > 0).sum().item()
        ),
        "external_classifier": False,
        "runtime_gate": False,
        "transformer_bit_identical": True,
        "candidate_saved": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": v2.sha256_file(candidate_path),
        "eligible_for_separate_official_evaluation": True,
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "claim_scope": "internal_sparse_folded_sensitivity_rewiring_not_yet_latent_erasure",
    }
    v2.write_json(output / "method" / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
