#!/usr/bin/env python3
"""RWKU pure two-stage Directional SURE.

Level 1 performs broad Directional SURE over every target-only generated atomic
training view. Level 2 is invoked only when Level 1 leaves generated atomic
margin failures; it trains only the sensitive vocabulary rows implicated by
those residual prompts and constructs B_F from residual sensitive hidden
states after removing the protected B_P span.

No transformer, attention, MLP, LoRA, or SURE-R parameter is trainable in
either level. Official RWKU evaluation artifacts remain unavailable.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_directional_sure_v2 as v2
import rwku_directional_sure_v21 as v21
import rwku_setting5e_utility_controlled as sparse_rows
import rwku_sure_head_only_w1k as head
import sure_canonical_core as core


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = (
    PROJECT_ROOT / "config" / "rwku" / "directional_sure_two_stage_seed0.json"
)
SOURCE_BUNDLE_CONFIGURATION = v2.SOURCE_BUNDLE_CONFIGURATION
SCHEMA = "rwku_directional_sure_two_stage_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-two-stage-stephen-king-seed0"
LEARNER_DIR = "directional_sure_two_stage"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", type=Path, required=True)
    p.add_argument("--generator-receipt", type=Path, required=True)
    p.add_argument("--wikipedia-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_configuration(path: Path) -> Dict[str, Any]:
    cfg = read_json(path)
    identity = {
        "schema_version": SCHEMA,
        "configuration_id": EXPERIMENT_ID,
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_metrics_observed_before_method_design": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "neutral_target": "Unknown",
        "level3_representation_repair_enabled": False,
    }
    for key, expected in identity.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Two-stage Directional SURE configuration changed {key}")

    components = cfg.get("trainable_components", {})
    expected_components = {
        "sensitive_input_embedding_rows": True,
        "sensitive_untied_lm_head_rows": True,
        "non_sensitive_input_embedding_rows": False,
        "non_sensitive_lm_head_rows": False,
        "transformer_parameters": False,
        "mlp_parameters": False,
        "attention_parameters": False,
        "lora_parameters": False,
    }
    for key, expected in expected_components.items():
        if components.get(key) != expected:
            raise ValueError(f"Two-stage trainable component changed {key}")

    opt = cfg.get("optimization", {})
    locked_stage1 = {
        "objective": "canonical_SURE_GA_plus_same_prompt_non_sensitive_GD_KL_with_directional_lm_head_gradient_decomposition",
        "training_view_scope": "all_target_only_generated_atomic_views",
        "steps": 600,
        "batch_size": 1,
        "cache_batch_size": 8,
        "embedding_learning_rate": 0.00005,
        "lm_head_learning_rate": 0.0001,
        "ga_weight": 2.0,
        "gd_weight": 1.0,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "basis_refresh_interval": 25,
        "sensitive_exclusive_basis_rank": 8,
        "protected_basis_rank": 32,
        "protected_basis_context_count": 256,
        "selection_utility_context_count": 256,
        "fresh_gate_utility_context_count": 1000,
        "utility_max_length": 128,
        "utility_batch_size": 4,
        "checkpoint_interval": 25,
    }
    for key, expected in locked_stage1.items():
        if opt.get(key) != expected:
            raise ValueError(f"Level-1 optimization changed {key}")

    stage2 = cfg.get("stage2", {})
    locked_stage2 = {
        "enabled": True,
        "trigger": "level1_pairwise_margin_failures",
        "training_scope": "level1_residual_prompt_sensitive_prediction_cases_only",
        "row_scope": "all_non_special_sensitive_rows_implicated_by_level1_residual_prompts",
        "steps": 300,
        "batch_size": 1,
        "embedding_learning_rate": 0.000025,
        "lm_head_learning_rate": 0.00005,
        "ga_weight": 2.0,
        "gd_weight": 1.0,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "basis_refresh_interval": 25,
        "residual_sensitive_exclusive_basis_rank": 8,
        "protected_basis_rank": 32,
        "checkpoint_interval": 25,
        "reset_optimizer_state": True,
        "transformer_parameters": False,
        "representation_repair": False,
    }
    for key, expected in locked_stage2.items():
        if stage2.get(key) != expected:
            raise ValueError(f"Level-2 optimization changed {key}")

    acc = cfg.get("acceptance", {})
    locked_acc = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
        "non_sensitive_embedding_rows_exact_base": True,
        "non_sensitive_lm_head_rows_exact_base": True,
        "transformer_exactly_frozen": True,
    }
    for key, expected in locked_acc.items():
        if acc.get(key) != expected:
            raise ValueError(f"Two-stage acceptance changed {key}")

    boundary = cfg.get("data_boundary", {})
    for key in (
        "official_rwku_records_available_to_learner",
        "official_rwku_records_used_for_checkpoint_selection",
        "official_rwku_paraphrase_seen",
        "official_rwku_neighborhood_seen",
        "official_rwku_retain_seen",
        "official_rwku_ppl_text_seen",
        "basis_selection_fresh_gate_overlap_allowed",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Two-stage data boundary changed {key}")
    if boundary.get("external_wikipedia_only_for_directional_protection_and_utility") is not True:
        raise ValueError("External utility/protection must be target-excluded Wikipedia only")
    if boundary.get("wikipedia_target_casefold_exclusion") != "stephen king":
        raise ValueError("Wikipedia target exclusion changed")
    if int(boundary.get("wikipedia_exclude_first_documents", -1)) != 20:
        raise ValueError("Wikipedia prefix exclusion changed")
    return cfg


def optimizer_for(
    sparse: sparse_rows.SparseFP32RowDeltas,
    *,
    embedding_lr: float,
    head_lr: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": [sparse.input_delta], "lr": float(embedding_lr), "weight_decay": 0.0},
            {"params": [sparse.output_delta], "lr": float(head_lr), "weight_decay": 0.0},
        ]
    )


def grad_or_zero(value: Optional[torch.Tensor], parameter: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(parameter) if value is None else value.float()


def behavior_counts(atomic: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    return (
        int(atomic.get("direct_margin_failures", 10**9)),
        int(atomic.get("generated_subject_margin_failures", 10**9)),
        int(atomic.get("direct_failures", 10**9)),
        int(atomic.get("generated_subject_failures", 10**9)),
    )


def stage1_anchor_key(
    atomic: Mapping[str, Any],
    utility: Mapping[str, Any],
    norms: Mapping[str, Any],
    step: int,
) -> Tuple[Any, ...]:
    dmf, omf, df, of = behavior_counts(atomic)
    return (
        dmf + omf,
        dmf,
        omf,
        df + of,
        df,
        of,
        -float(atomic.get("minimum_overall_separation", -1e30)),
        float(utility["utility_kl_mean"]),
        float(utility["utility_kl_p95"]),
        float(utility["utility_kl_max"]),
        float(norms["total_selected_row_delta_norm"]),
        int(step),
    )


def snapshot_with_key(
    sparse: sparse_rows.SparseFP32RowDeltas,
    *,
    step: int,
    atomic: Mapping[str, Any],
    utility: Mapping[str, Any],
    selection_key: Sequence[Any],
    source: str,
) -> Dict[str, Any]:
    input_delta, output_delta = sparse.snapshot()
    return {
        "source": source,
        "step": int(step),
        "selection_key": list(selection_key),
        "input_delta": input_delta,
        "output_delta": output_delta,
        "atomic": dict(atomic),
        "selection_utility": dict(utility),
        "delta_norms": sparse.delta_norms(input_delta, output_delta),
    }


def residual_prompt_positions(atomic: Mapping[str, Any]) -> List[int]:
    values = atomic.get("pairwise_margin_failure_positions", [])
    if not isinstance(values, list):
        raise ValueError("Atomic report lacks pairwise_margin_failure_positions list")
    result = sorted({int(x) for x in values})
    instances = atomic.get("prompt_instances", [])
    if result and not isinstance(instances, list):
        raise ValueError("Atomic report lacks prompt_instances")
    if any(index < 0 or index >= len(instances) for index in result):
        raise ValueError("Residual prompt position is out of range")
    return result


def residual_case_indices(
    cases: Sequence[core.SensitivePredictionCase],
    residual_positions: Sequence[int],
) -> List[int]:
    wanted = set(int(x) for x in residual_positions)
    result = [
        index
        for index, case in enumerate(cases)
        if int(case.record_position) in wanted
    ]
    covered = {int(cases[index].record_position) for index in result}
    if covered != wanted:
        missing = sorted(wanted - covered)
        raise RuntimeError(f"Residual prompts lack sensitive prediction cases: {missing[:10]}")
    if not result:
        raise RuntimeError("Level 2 was triggered without residual sensitive cases")
    return result


def residual_rows_from_cases(
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    tids_all: torch.Tensor,
    case_indices: Sequence[int],
    selected_rows: Sequence[int],
) -> List[int]:
    del cases
    special = {
        int(x)
        for x in getattr(tokenizer, "all_special_ids", [])
        if x is not None
    }
    selected_set = set(int(x) for x in selected_rows)
    token_ids = tids_all.detach().cpu().tolist()
    rows = sorted(
        {
            int(token_ids[index])
            for index in case_indices
            if int(token_ids[index]) not in special
        }
    )
    if not rows:
        raise RuntimeError("Residual cases expose no non-special sensitive rows")
    unexpected = sorted(set(rows) - selected_set)
    if unexpected:
        raise RuntimeError(
            "Level-2 residual rows are outside the Level-1 editable row set: "
            f"{unexpected[:10]}"
        )
    return rows


def row_mask(
    sparse: sparse_rows.SparseFP32RowDeltas,
    allowed_rows: Sequence[int],
    *,
    for_output: bool,
) -> torch.Tensor:
    selected = (
        sparse.selected_output_rows if for_output else sparse.selected_input_rows
    )
    allowed = set(int(x) for x in allowed_rows)
    values = [1.0 if int(row) in allowed else 0.0 for row in selected]
    parameter = sparse.output_delta if for_output else sparse.input_delta
    return torch.tensor(
        values,
        dtype=parameter.dtype,
        device=parameter.device,
    ).unsqueeze(1)


def stage2_basis_cfg(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    clone: Dict[str, Any] = dict(cfg)
    clone["optimization"] = dict(cfg["optimization"])
    stage2 = cfg["stage2"]
    clone["optimization"]["sensitive_exclusive_basis_rank"] = int(
        stage2["residual_sensitive_exclusive_basis_rank"]
    )
    clone["optimization"]["protected_basis_rank"] = int(stage2["protected_basis_rank"])
    clone["optimization"]["basis_refresh_interval"] = int(stage2["basis_refresh_interval"])
    clone["optimization"]["checkpoint_interval"] = int(stage2["checkpoint_interval"])
    clone["optimization"]["batch_size"] = int(stage2["batch_size"])
    return clone


def write_protocol(
    out: Path,
    *,
    cfg: Mapping[str, Any],
    bundle_audit: Mapping[str, Any],
    generator_model_audit: Mapping[str, Any],
    prompt_records: Sequence[Mapping[str, Any]],
    cases: Sequence[core.SensitivePredictionCase],
    sensitive_row_audit: Mapping[str, Any],
    sensitive_rows: Sequence[int],
    untie_audit: Mapping[str, Any],
    freeze_audit: Mapping[str, Any],
    wikipedia_meta: Mapping[str, Any],
    external_audit: Mapping[str, Any],
) -> None:
    core.write_json(
        out / "protocol_report.json",
        {
            "schema_version": "rwku_directional_sure_two_stage_protocol_v1",
            "configuration_id": cfg["configuration_id"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "levels": {
                "level1": "broad directional sparse vocabulary-interface suppression",
                "level2": "residual directional repair restricted to Level-1 failing prompts and their sensitive rows",
                "level3": "disabled; no representation/MLP/LoRA repair exists in this method",
            },
            "bundle_audit": bundle_audit,
            "generator_model_audit": generator_model_audit,
            "training_prompt_count": len(prompt_records),
            "sensitive_prediction_case_count": len(cases),
            "sensitive_row_audit": sensitive_row_audit,
            "selected_sensitive_row_count": len(sensitive_rows),
            "selected_sensitive_row_ids": list(sensitive_rows),
            "untie_audit": untie_audit,
            "freeze_audit": freeze_audit,
            "external_wikipedia_dataset": wikipedia_meta,
            "external_slices": external_audit,
            "level1_objective": cfg["optimization"]["objective"],
            "level2_objective": (
                "same canonical GA/GD losses on residual sensitive prediction cases; "
                "GA head gradient projected into residual sensitive-exclusive B_F, "
                "GD head gradient projected into protected B_P; gradients masked to "
                "rows implicated by Level-1 residual prompts"
            ),
            "parameter_locality": (
                "FP32 sparse deltas over immutable Base vocabulary matrices in both "
                "levels; non-sensitive rows have no trainable parameter; transformer "
                "parameters are exactly frozen"
            ),
            "level1_anchor_selection": (
                "among external-Wikipedia-selection-safe checkpoints, minimize total "
                "pairwise margin failures, then direct/other failures, maximize minimum "
                "separation, then minimize utility KL and sparse-delta norm"
            ),
        },
    )


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked two-stage configuration ID")

    run_dir = v2.verify_prepared_state(args, cfg)
    out = run_dir / LEARNER_DIR
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite two-stage output: {out}")
    out.mkdir(parents=True)

    source_cfg = head.load_locked_configuration(SOURCE_BUNDLE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_cfg,
    )
    generator_model_audit = head.validate_generator_base_model(
        generator_audit, args.model_path
    )

    gagd.set_seed(int(cfg["seed"]))
    gagd.require_cuda_if_needed(str(cfg["acceptance"]["device_map"]))
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=str(cfg["acceptance"]["checkpoint_dtype"]),
        device_map=str(cfg["acceptance"]["device_map"]),
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)

    prompt_records = head.compile_prompt_records(
        views, tokenizer, neutral_target=str(cfg["neutral_target"])
    )
    cases = core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    if not cases:
        raise RuntimeError("Two-stage Directional SURE created no sensitive cases")

    base_logits = core.cache_base_logits(
        model,
        tokenizer,
        cases,
        device,
        batch_size=int(cfg["optimization"]["cache_batch_size"]),
    )
    tids_all = core.official_target_ids(
        tokenizer, cases, llama_like=llama_like, device=device
    )
    sensitive_rows, sensitive_row_audit = v21.all_non_special_sensitive_rows(
        tokenizer,
        cases,
        tids_all,
        source_cfg,
        len(prompt_records),
    )

    sample_ids = tokenizer(
        prompt_records[0]["prompt_text"], return_tensors="pt"
    )["input_ids"].to(device)
    untie_audit = sparse_rows.untie_lm_head_preserve_logits(
        model, sample_input_ids=sample_ids
    )
    freeze_audit = sparse_rows.freeze_transformer_parameters(model)
    transformer_versions = v2._parameter_versions_except_vocab(model)

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not torch.equal(input_layer.weight.detach(), output_layer.weight.detach()):
        raise RuntimeError("Untied Base vocabulary matrices are not initially identical")
    base_vocab_cpu = input_layer.weight.detach().cpu().clone()

    sparse = sparse_rows.SparseFP32RowDeltas(
        model,
        selected_input_rows=sensitive_rows,
        selected_output_rows=sensitive_rows,
    )
    if sparse.input_delta.dtype != torch.float32 or sparse.output_delta.dtype != torch.float32:
        raise RuntimeError("Sparse row masters must be FP32")

    texts, wikipedia_meta = wikipedia.load_wikipedia_train(
        Path(args.wikipedia_dir).resolve()
    )
    protected_contexts, selection_contexts, fresh_contexts, external_audit = (
        v2.build_external_slices(tokenizer, texts, cfg)
    )
    utility_bs = int(cfg["optimization"]["utility_batch_size"])
    selection_base_hidden = v2.external_final_hidden(
        model,
        tokenizer,
        selection_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()
    fresh_base_hidden = v2.external_final_hidden(
        model,
        tokenizer,
        fresh_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()

    write_protocol(
        out,
        cfg=cfg,
        bundle_audit=bundle_audit,
        generator_model_audit=generator_model_audit,
        prompt_records=prompt_records,
        cases=cases,
        sensitive_row_audit=sensitive_row_audit,
        sensitive_rows=sensitive_rows,
        untie_audit=untie_audit,
        freeze_audit=freeze_audit,
        wikipedia_meta=wikipedia_meta,
        external_audit=external_audit,
    )

    started = time.perf_counter()
    model.eval()
    opt = cfg["optimization"]
    stage1_optimizer = optimizer_for(
        sparse,
        embedding_lr=float(opt["embedding_learning_rate"]),
        head_lr=float(opt["lm_head_learning_rate"]),
    )
    sampler = core.IndexSampler(
        len(cases), int(opt["batch_size"]), int(cfg["seed"])
    )
    level1_anchor: Optional[Dict[str, Any]] = None
    level1_anchor_key_value: Optional[Tuple[Any, ...]] = None
    level1_feasible: Optional[Dict[str, Any]] = None
    level1_feasible_key: Optional[Tuple[Any, ...]] = None
    level1_basis_history: List[Dict[str, Any]] = []
    level1_checkpoints: List[Dict[str, Any]] = []
    bs: Optional[torch.Tensor] = None
    bp: Optional[torch.Tensor] = None

    with (out / "level1_train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, int(opt["steps"]) + 1):
            if step == 1 or (step - 1) % int(opt["basis_refresh_interval"]) == 0:
                bs, bp, report = v2.refresh_directional_bases(
                    model,
                    tokenizer,
                    cases,
                    protected_contexts,
                    cfg,
                    device=device,
                )
                report = {"refresh_before_step": int(step), **report}
                level1_basis_history.append(report)
                print(
                    "L1 basis before {}: B_S={} B_P={} overlap={:.3e} exclusive={:.4f}".format(
                        step,
                        report["sensitive_exclusive_rank"],
                        report["protected_rank"],
                        report["max_abs_sensitive_protected_basis_overlap"],
                        report["sensitive_energy_after_protected_projection_fraction"],
                    )
                )
            if bs is None or bp is None:
                raise RuntimeError("Level-1 directional bases unavailable")

            idx = sampler.next()
            batch = [cases[i] for i in idx]
            logits = core.forward_last_logits(model, tokenizer, batch, device)
            tids = core.official_target_ids(
                tokenizer, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
            params = (sparse.input_delta, sparse.output_delta)
            ga_grads = torch.autograd.grad(
                float(opt["ga_weight"]) * ga,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            gd_grads = torch.autograd.grad(
                float(opt["gd_weight"]) * gd,
                params,
                retain_graph=False,
                allow_unused=True,
            )
            ga_emb = grad_or_zero(ga_grads[0], sparse.input_delta)
            ga_head = grad_or_zero(ga_grads[1], sparse.output_delta)
            gd_emb = grad_or_zero(gd_grads[0], sparse.input_delta)
            gd_head = grad_or_zero(gd_grads[1], sparse.output_delta)
            head_ga = v2.project_into_basis(ga_head, bs)
            head_gd = v2.project_into_basis(gd_head, bp)

            stage1_optimizer.zero_grad(set_to_none=True)
            sparse.input_delta.grad = (ga_emb + gd_emb).to(sparse.input_delta.dtype)
            sparse.output_delta.grad = (head_ga + head_gd).to(sparse.output_delta.dtype)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [sparse.input_delta, sparse.output_delta], float(opt["grad_clip"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite Level-1 gradient at step {step}")
            stage1_optimizer.step()

            if step == 1 or step % 25 == 0 or step == int(opt["steps"]):
                row = {
                    "level": 1,
                    "step": int(step),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_non_sensitive_kl": float(gd.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "embedding_ga_gradient_norm": float(ga_emb.norm().detach().cpu()),
                    "embedding_gd_gradient_norm": float(gd_emb.norm().detach().cpu()),
                    "lm_head_ga_BS_gradient_norm": float(head_ga.norm().detach().cpu()),
                    "lm_head_gd_BP_gradient_norm": float(head_gd.norm().detach().cpu()),
                    **sparse.delta_norms(),
                    "official_rwku_records_accessed": False,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    "L1 step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f}".format(
                        step,
                        row["ga_sensitive_logprob"],
                        row["gd_non_sensitive_kl"],
                        row["selected_input_row_delta_norm"],
                        row["selected_output_row_delta_norm"],
                    )
                )

            if (
                step % int(opt["checkpoint_interval"]) != 0
                and step != int(opt["steps"])
            ):
                continue

            atomic = head.materialized_atomic_report(
                model,
                tokenizer,
                prompt_records,
                device,
                llama_like=llama_like,
                required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
            )
            utility = v2.exact_external_kl_report(
                model,
                tokenizer,
                selection_contexts,
                selection_base_hidden,
                device=device,
                batch_size=utility_bs,
            )
            u_safe = v2.utility_safe(utility, cfg)
            a_safe = v2.atomic_safe(atomic, cfg)
            norms = sparse.delta_norms()
            anchor_key = stage1_anchor_key(atomic, utility, norms, step)
            checkpoint = {
                "step": int(step),
                "atomic": atomic,
                "selection_utility": utility,
                "selection_utility_safe": u_safe,
                "atomic_safe": a_safe,
                "level1_anchor_key": list(anchor_key),
                "delta_norms": norms,
                "official_rwku_records_accessed": False,
            }
            level1_checkpoints.append(checkpoint)
            print(
                "  L1 checkpoint {}: direct={} other={} margin_fail={} KL={:.6f}/{:.6f}/{:.6f}".format(
                    step,
                    atomic.get("FS"),
                    atomic.get("generated_subject_FS"),
                    len(residual_prompt_positions(atomic)),
                    utility["utility_kl_mean"],
                    utility["utility_kl_p95"],
                    utility["utility_kl_max"],
                )
            )
            if u_safe and (
                level1_anchor_key_value is None or anchor_key < level1_anchor_key_value
            ):
                level1_anchor_key_value = anchor_key
                level1_anchor = snapshot_with_key(
                    sparse,
                    step=step,
                    atomic=atomic,
                    utility=utility,
                    selection_key=anchor_key,
                    source="level1_anchor",
                )
            if u_safe and a_safe:
                candidate = v2.snapshot_candidate(
                    sparse, step=step, atomic=atomic, utility=utility
                )
                candidate["source"] = "level1_full_acceptance"
                key = tuple(candidate["selection_key"])
                if level1_feasible_key is None or key < level1_feasible_key:
                    level1_feasible_key = key
                    level1_feasible = candidate

    core.write_json(out / "level1_basis_refresh_history.json", level1_basis_history)
    core.write_json(out / "level1_checkpoint_history.json", level1_checkpoints)
    v2.assert_transformer_versions(model, transformer_versions)

    if not torch.equal(input_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base input vocabulary matrix changed before materialization")
    if not torch.equal(output_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base output vocabulary matrix changed before materialization")

    if level1_anchor is None:
        result = {
            "schema_version": "rwku_directional_sure_two_stage_result_v1",
            "configuration_id": cfg["configuration_id"],
            "feasible": False,
            "reason": "Level 1 produced no external-Wikipedia-selection-safe checkpoint",
            "level1_feasible": False,
            "level2_used": False,
            "level3_used": False,
            "transformer_exactly_frozen": True,
            "official_rwku_records_accessed": False,
        }
        core.write_json(out / "result.json", result)
        raise RuntimeError(result["reason"])

    final_candidate: Optional[Dict[str, Any]] = None
    level2_used = False
    level2_basis_history: List[Dict[str, Any]] = []
    level2_checkpoints: List[Dict[str, Any]] = []
    level2_residual_positions: List[int] = []
    level2_residual_case_indices: List[int] = []
    level2_residual_rows: List[int] = []

    if level1_feasible is not None:
        final_candidate = level1_feasible
        print("Level 1 already satisfies all generated atomic + selection utility gates; Level 2 skipped.")
    else:
        level2_used = True
        with torch.no_grad():
            sparse.input_delta.copy_(
                level1_anchor["input_delta"].to(device=sparse.input_delta.device)
            )
            sparse.output_delta.copy_(
                level1_anchor["output_delta"].to(device=sparse.output_delta.device)
            )

        level2_residual_positions = residual_prompt_positions(level1_anchor["atomic"])
        if not level2_residual_positions:
            raise RuntimeError(
                "Level 1 was not atomically safe but exposes no pairwise margin failures"
            )
        level2_residual_case_indices = residual_case_indices(
            cases, level2_residual_positions
        )
        level2_residual_rows = residual_rows_from_cases(
            tokenizer,
            cases,
            tids_all,
            level2_residual_case_indices,
            sensitive_rows,
        )
        residual_cases = [cases[index] for index in level2_residual_case_indices]
        input_mask = row_mask(sparse, level2_residual_rows, for_output=False)
        output_mask = row_mask(sparse, level2_residual_rows, for_output=True)
        if int(input_mask.sum().item()) != len(level2_residual_rows):
            raise RuntimeError("Level-2 input row mask does not match residual rows")
        if int(output_mask.sum().item()) != len(level2_residual_rows):
            raise RuntimeError("Level-2 output row mask does not match residual rows")

        stage2 = cfg["stage2"]
        stage2_cfg = stage2_basis_cfg(cfg)
        stage2_optimizer = optimizer_for(
            sparse,
            embedding_lr=float(stage2["embedding_learning_rate"]),
            head_lr=float(stage2["lm_head_learning_rate"]),
        )
        stage2_sampler = core.IndexSampler(
            len(level2_residual_case_indices),
            int(stage2["batch_size"]),
            int(cfg["seed"]) + 200003,
        )
        stage2_best: Optional[Dict[str, Any]] = None
        stage2_best_key: Optional[Tuple[Any, ...]] = None
        bf: Optional[torch.Tensor] = None
        bp2: Optional[torch.Tensor] = None

        print(
            "Level 2 triggered from L1 step {}: residual_prompts={} residual_cases={} residual_rows={}".format(
                level1_anchor["step"],
                len(level2_residual_positions),
                len(level2_residual_case_indices),
                len(level2_residual_rows),
            )
        )
        with (out / "level2_train_log.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, int(stage2["steps"]) + 1):
                if (
                    step == 1
                    or (step - 1) % int(stage2["basis_refresh_interval"]) == 0
                ):
                    bf, bp2, report = v2.refresh_directional_bases(
                        model,
                        tokenizer,
                        residual_cases,
                        protected_contexts,
                        stage2_cfg,
                        device=device,
                    )
                    report = {
                        "refresh_before_level2_step": int(step),
                        "basis_name": "B_F_residual_sensitive_exclusive",
                        "level1_residual_prompt_count": len(level2_residual_positions),
                        "level2_residual_sensitive_case_count": len(residual_cases),
                        "level2_residual_row_count": len(level2_residual_rows),
                        **report,
                    }
                    level2_basis_history.append(report)
                    print(
                        "L2 basis before {}: B_F={} B_P={} overlap={:.3e} exclusive={:.4f}".format(
                            step,
                            report["sensitive_exclusive_rank"],
                            report["protected_rank"],
                            report["max_abs_sensitive_protected_basis_overlap"],
                            report["sensitive_energy_after_protected_projection_fraction"],
                        )
                    )
                if bf is None or bp2 is None:
                    raise RuntimeError("Level-2 directional bases unavailable")

                local_idx = stage2_sampler.next()
                global_idx = [
                    level2_residual_case_indices[int(index)] for index in local_idx
                ]
                batch = [cases[index] for index in global_idx]
                logits = core.forward_last_logits(model, tokenizer, batch, device)
                tids = core.official_target_ids(
                    tokenizer, batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(logits, tids)
                gd = core.gd_non_sensitive_kl(logits, base_logits[global_idx], tids)
                params = (sparse.input_delta, sparse.output_delta)
                ga_grads = torch.autograd.grad(
                    float(stage2["ga_weight"]) * ga,
                    params,
                    retain_graph=True,
                    allow_unused=True,
                )
                gd_grads = torch.autograd.grad(
                    float(stage2["gd_weight"]) * gd,
                    params,
                    retain_graph=False,
                    allow_unused=True,
                )
                ga_emb = grad_or_zero(ga_grads[0], sparse.input_delta) * input_mask
                gd_emb = grad_or_zero(gd_grads[0], sparse.input_delta) * input_mask
                ga_head_raw = (
                    grad_or_zero(ga_grads[1], sparse.output_delta) * output_mask
                )
                gd_head_raw = (
                    grad_or_zero(gd_grads[1], sparse.output_delta) * output_mask
                )
                head_ga = v2.project_into_basis(ga_head_raw, bf)
                head_gd = v2.project_into_basis(gd_head_raw, bp2)

                stage2_optimizer.zero_grad(set_to_none=True)
                sparse.input_delta.grad = (ga_emb + gd_emb).to(
                    sparse.input_delta.dtype
                )
                sparse.output_delta.grad = (head_ga + head_gd).to(
                    sparse.output_delta.dtype
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [sparse.input_delta, sparse.output_delta],
                    float(stage2["grad_clip"]),
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite Level-2 gradient at step {step}"
                    )
                stage2_optimizer.step()

                if step == 1 or step % 25 == 0 or step == int(stage2["steps"]):
                    row = {
                        "level": 2,
                        "step": int(step),
                        "level1_anchor_step": int(level1_anchor["step"]),
                        "residual_prompt_count": len(level2_residual_positions),
                        "residual_sensitive_case_count": len(
                            level2_residual_case_indices
                        ),
                        "residual_row_count": len(level2_residual_rows),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "gd_non_sensitive_kl": float(gd.detach().cpu()),
                        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                        "embedding_ga_gradient_norm_after_row_mask": float(
                            ga_emb.norm().detach().cpu()
                        ),
                        "embedding_gd_gradient_norm_after_row_mask": float(
                            gd_emb.norm().detach().cpu()
                        ),
                        "lm_head_ga_BF_gradient_norm": float(
                            head_ga.norm().detach().cpu()
                        ),
                        "lm_head_gd_BP_gradient_norm": float(
                            head_gd.norm().detach().cpu()
                        ),
                        **sparse.delta_norms(),
                        "official_rwku_records_accessed": False,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(
                        "L2 step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f}".format(
                            step,
                            row["ga_sensitive_logprob"],
                            row["gd_non_sensitive_kl"],
                            row["selected_input_row_delta_norm"],
                            row["selected_output_row_delta_norm"],
                        )
                    )

                if (
                    step % int(stage2["checkpoint_interval"]) != 0
                    and step != int(stage2["steps"])
                ):
                    continue

                atomic = head.materialized_atomic_report(
                    model,
                    tokenizer,
                    prompt_records,
                    device,
                    llama_like=llama_like,
                    required_margin=float(
                        cfg["acceptance"]["required_pairwise_margin"]
                    ),
                )
                utility = v2.exact_external_kl_report(
                    model,
                    tokenizer,
                    selection_contexts,
                    selection_base_hidden,
                    device=device,
                    batch_size=utility_bs,
                )
                a_safe = v2.atomic_safe(atomic, cfg)
                u_safe = v2.utility_safe(utility, cfg)
                checkpoint = {
                    "step": int(step),
                    "atomic": atomic,
                    "selection_utility": utility,
                    "atomic_safe": a_safe,
                    "selection_utility_safe": u_safe,
                    "eligible": bool(a_safe and u_safe),
                    "remaining_margin_failure_positions": residual_prompt_positions(
                        atomic
                    ),
                    "delta_norms": sparse.delta_norms(),
                    "official_rwku_records_accessed": False,
                }
                level2_checkpoints.append(checkpoint)
                print(
                    "  L2 checkpoint {}: direct={} other={} remaining_margin_fail={} KL={:.6f}/{:.6f}/{:.6f} eligible={}".format(
                        step,
                        atomic.get("FS"),
                        atomic.get("generated_subject_FS"),
                        len(checkpoint["remaining_margin_failure_positions"]),
                        utility["utility_kl_mean"],
                        utility["utility_kl_p95"],
                        utility["utility_kl_max"],
                        checkpoint["eligible"],
                    )
                )
                if checkpoint["eligible"]:
                    candidate = v2.snapshot_candidate(
                        sparse, step=step, atomic=atomic, utility=utility
                    )
                    candidate["source"] = "level2_residual_repair"
                    key = tuple(candidate["selection_key"])
                    if stage2_best_key is None or key < stage2_best_key:
                        stage2_best_key = key
                        stage2_best = candidate

        core.write_json(
            out / "level2_basis_refresh_history.json", level2_basis_history
        )
        core.write_json(out / "level2_checkpoint_history.json", level2_checkpoints)
        del stage2_optimizer
        if stage2_best is not None:
            final_candidate = stage2_best

    v2.assert_transformer_versions(model, transformer_versions)
    training_seconds = time.perf_counter() - started

    if final_candidate is None:
        result = {
            "schema_version": "rwku_directional_sure_two_stage_result_v1",
            "configuration_id": cfg["configuration_id"],
            "method": cfg["method"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "feasible": False,
            "reason": "Level 2 found no checkpoint satisfying all generated atomic + external-Wikipedia selection gates",
            "level1_anchor_step": int(level1_anchor["step"]),
            "level1_anchor_atomic": level1_anchor["atomic"],
            "level1_anchor_selection_utility": level1_anchor["selection_utility"],
            "level2_used": level2_used,
            "level2_initial_residual_prompt_positions": level2_residual_positions,
            "level2_initial_residual_sensitive_case_count": len(
                level2_residual_case_indices
            ),
            "level2_initial_residual_row_ids": level2_residual_rows,
            "level3_used": False,
            "representation_repair_used": False,
            "transformer_exactly_frozen": True,
            "non_sensitive_embedding_rows_exact_base": True,
            "non_sensitive_lm_head_rows_exact_base": True,
            "training_seconds": training_seconds,
        }
        core.write_json(out / "result.json", result)
        raise RuntimeError(result["reason"])

    with torch.no_grad():
        sparse.input_delta.copy_(
            final_candidate["input_delta"].to(device=sparse.input_delta.device)
        )
        sparse.output_delta.copy_(
            final_candidate["output_delta"].to(device=sparse.output_delta.device)
        )

    final_atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    final_atomic_safe = v2.atomic_safe(final_atomic, cfg)
    fresh_utility = v2.exact_external_kl_report(
        model,
        tokenizer,
        fresh_contexts,
        fresh_base_hidden,
        device=device,
        batch_size=utility_bs,
    )
    fresh_safe = v2.utility_safe(fresh_utility, cfg)
    feasible = bool(final_atomic_safe and fresh_safe)

    sparse.materialize(
        final_candidate["input_delta"],
        final_candidate["output_delta"],
        1.0,
    )
    nonselected_input_equal = v2._nonselected_equal_base(
        input_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    nonselected_output_equal = v2._nonselected_equal_base(
        output_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    if not nonselected_input_equal or not nonselected_output_equal:
        raise RuntimeError("Two-stage Directional SURE changed a non-sensitive row")
    v2.assert_transformer_versions(model, transformer_versions)

    result = {
        "schema_version": "rwku_directional_sure_two_stage_result_v1",
        "configuration_id": cfg["configuration_id"],
        "method": cfg["method"],
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "level1_anchor_step": int(level1_anchor["step"]),
        "level1_anchor_atomic": level1_anchor["atomic"],
        "level1_anchor_selection_utility": level1_anchor["selection_utility"],
        "level1_reached_full_acceptance": level1_feasible is not None,
        "level2_used": level2_used,
        "level2_initial_residual_prompt_positions": level2_residual_positions,
        "level2_initial_residual_sensitive_case_count": len(
            level2_residual_case_indices
        ),
        "level2_initial_residual_row_ids": level2_residual_rows,
        "level3_used": False,
        "representation_repair_used": False,
        "selected_source": final_candidate.get("source", "level1"),
        "selected_checkpoint_step": int(final_candidate["step"]),
        "selection_key": list(final_candidate["selection_key"]),
        "selected_sensitive_row_count": len(sensitive_rows),
        "selected_sensitive_row_ids": sensitive_rows,
        "delta_norms": sparse.delta_norms(
            final_candidate["input_delta"], final_candidate["output_delta"]
        ),
        "final_atomic": final_atomic,
        "final_atomic_safe": final_atomic_safe,
        "fresh_external_wikipedia_utility": fresh_utility,
        "fresh_external_wikipedia_utility_safe": fresh_safe,
        "feasible": feasible,
        "transformer_exactly_frozen": True,
        "non_sensitive_embedding_rows_exact_base": nonselected_input_equal,
        "non_sensitive_lm_head_rows_exact_base": nonselected_output_equal,
        "official_rwku_paraphrase_seen": False,
        "official_rwku_neighborhood_seen": False,
        "official_rwku_retain_seen": False,
        "official_rwku_ppl_text_seen": False,
        "fresh_gate_opened_only_after_final_checkpoint_selection": True,
        "training_seconds": training_seconds,
        "interpretation_note": (
            "Pure two-stage Directional SURE. Level 1 performs broad sparse "
            "vocabulary-interface suppression. Level 2, when needed, repairs "
            "only Level-1 residual generated prompts and implicated sensitive "
            "rows. No transformer/MLP/LoRA/representation repair is used."
        ),
    }
    core.write_json(out / "result.json", result)

    if args.save_checkpoint and feasible:
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible two-stage Directional SURE checkpoint: {checkpoint}")

    print("\nRWKU TWO-STAGE DIRECTIONAL SURE RESULT")
    print(f"L1 anchor step: {result['level1_anchor_step']}")
    print(f"L2 used: {result['level2_used']}")
    print(f"L3 / representation repair used: {result['level3_used']}")
    print(
        "final direct/other: {} / {}".format(
            final_atomic.get("FS"), final_atomic.get("generated_subject_FS")
        )
    )
    print(
        "fresh Wiki KL mean/p95/max: {:.6f} / {:.6f} / {:.6f}".format(
            fresh_utility["utility_kl_mean"],
            fresh_utility["utility_kl_p95"],
            fresh_utility["utility_kl_max"],
        )
    )
    print(f"transformer frozen: {result['transformer_exactly_frozen']}")
    print(f"feasible: {feasible}")
    print(f"result: {out / 'result.json'}")

    del stage1_optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
