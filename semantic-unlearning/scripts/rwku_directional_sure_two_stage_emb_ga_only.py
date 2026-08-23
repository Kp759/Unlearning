#!/usr/bin/env python3
"""Two-stage Directional SURE with content-sensitive rows and embedding GA only.

Level 1 is the utility-safe Directional SURE v2 row policy:
* content-sensitive input-embedding and untied LM-head rows only;
* embedding update = 2*GA only (embedding GD is measured but not applied);
* LM-head update = 2*GA projected to B_S + GD projected to B_P;
* transformer parameters remain exactly frozen.

If Level 1 does not reach full generated-atomic acceptance, Level 2 restores the
best external-Wikipedia-selection-safe Level-1 anchor and trains only residual
failed/margin-failed prompts. Its editable vocabulary rows are the intersection
of rows implicated by those residual prompts with the same locked
content-sensitive Level-1 row set. Embedding GD remains disabled in Level 2;
LM-head GA is projected to residual B_F and LM-head GD to B_P.

No Level 3, MLP, attention, LoRA, or representation repair is used. Official
RWKU evaluation artifacts remain unavailable to the learner.
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
import rwku_directional_sure_two_stage as base2
import rwku_directional_sure_v2 as v2
import rwku_setting5e_utility_controlled as sparse_rows
import rwku_sure_head_only_w1k as head
import sure_canonical_core as core


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = (
    PROJECT_ROOT
    / "config"
    / "rwku"
    / "directional_sure_two_stage_emb_ga_only_seed0.json"
)
SOURCE_BUNDLE_CONFIGURATION = v2.SOURCE_BUNDLE_CONFIGURATION
SCHEMA = "rwku_directional_sure_two_stage_emb_ga_only_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-two-stage-emb-ga-only-stephen-king-seed0"
LEARNER_DIR = "directional_sure_two_stage_emb_ga_only"


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
        "embedding_gradient_policy": "GA_only_no_GD_no_hidden_basis_projection",
        "lm_head_gradient_policy": "GA_to_sensitive_or_residual_exclusive_basis_and_GD_to_protected_basis",
    }
    for key, expected in identity.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Two-stage embedding-GA-only configuration changed {key}")

    baseline = base2.read_json(base2.DEFAULT_CONFIGURATION)
    if cfg.get("trainable_components") != baseline.get("trainable_components"):
        raise ValueError("Trainable components changed relative to locked two-stage baseline")
    if cfg.get("acceptance") != baseline.get("acceptance"):
        raise ValueError("Acceptance budgets changed relative to locked two-stage baseline")
    if cfg.get("data_boundary") != baseline.get("data_boundary"):
        raise ValueError("Data boundary changed relative to locked two-stage baseline")

    opt = dict(cfg.get("optimization", {}))
    base_opt = dict(baseline.get("optimization", {}))
    expected_objective = (
        "embedding_GA_only_plus_same_prompt_non_sensitive_GD_KL_on_directional_lm_head"
    )
    if opt.pop("objective", None) != expected_objective:
        raise ValueError("Level-1 objective changed")
    base_opt.pop("objective", None)
    if opt != base_opt:
        raise ValueError("Level-1 optimization changed beyond embedding GD removal")

    stage2 = dict(cfg.get("stage2", {}))
    base_stage2 = dict(baseline.get("stage2", {}))
    if stage2.pop("embedding_gradient_policy", None) != "GA_only_no_GD":
        raise ValueError("Level-2 embedding gradient policy changed")
    if stage2.pop("row_scope", None) != (
        "content_sensitive_rows_implicated_by_level1_residual_prompts"
    ):
        raise ValueError("Level-2 row scope changed")
    base_stage2.pop("row_scope", None)
    if stage2 != base_stage2:
        raise ValueError("Level-2 optimization changed beyond row locality / embedding GD removal")
    return cfg


def residual_content_rows(
    tids_all: torch.Tensor,
    case_indices: Sequence[int],
    selected_rows: Sequence[int],
) -> List[int]:
    """Residual rows restricted to the locked content-sensitive Level-1 row set."""
    selected = set(int(x) for x in selected_rows)
    token_ids = tids_all.detach().cpu().tolist()
    rows = sorted(
        {
            int(token_ids[index])
            for index in case_indices
            if int(token_ids[index]) in selected
        }
    )
    if not rows:
        raise RuntimeError(
            "Level 2 residual prompts expose no editable content-sensitive rows"
        )
    return rows


def compose_embedding_gradient(
    ga_gradient: torch.Tensor,
    gd_gradient: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply GA only; GD is accepted only so tests/logs can prove it is ignored."""
    del gd_gradient
    result = ga_gradient
    if mask is not None:
        result = result * mask
    return result


def compose_head_gradient(
    ga_directional: torch.Tensor, gd_directional: torch.Tensor
) -> torch.Tensor:
    return ga_directional + gd_directional


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
            "schema_version": "rwku_directional_sure_two_stage_emb_ga_only_protocol_v1",
            "configuration_id": cfg["configuration_id"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "levels": {
                "level1": (
                    "content-sensitive sparse vocabulary-interface suppression; "
                    "embedding 2*GA only; LM-head 2*GA->B_S + GD->B_P"
                ),
                "level2": (
                    "residual repair on Level-1 margin-failed prompts; content-sensitive "
                    "residual rows only; embedding 2*GA only; LM-head 2*GA->B_F + GD->B_P"
                ),
                "level3": "disabled",
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
            "embedding_gradient_policy": cfg["embedding_gradient_policy"],
            "lm_head_gradient_policy": cfg["lm_head_gradient_policy"],
            "level1_anchor_selection": (
                "among external-Wikipedia-selection-safe checkpoints, minimize total "
                "pairwise margin failures, then direct/other failures, maximize minimum "
                "separation, then minimize utility KL and sparse-delta norm"
            ),
            "parameter_locality": (
                "content-sensitive FP32 sparse deltas over immutable Base vocabulary "
                "matrices; all other vocabulary rows exact Base; transformer frozen"
            ),
        },
    )


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked configuration ID")

    run_dir = v2.verify_prepared_state(args, cfg)
    out = run_dir / LEARNER_DIR
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite two-stage embedding-GA-only output: {out}")
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
        raise RuntimeError("Two-stage embedding-GA-only learner created no sensitive cases")

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
    sensitive_rows, sensitive_row_audit = v2._content_sensitive_rows(
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
    stage1_optimizer = base2.optimizer_for(
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
            ga_emb = base2.grad_or_zero(ga_grads[0], sparse.input_delta)
            ga_head = base2.grad_or_zero(ga_grads[1], sparse.output_delta)
            gd_emb = base2.grad_or_zero(gd_grads[0], sparse.input_delta)
            gd_head = base2.grad_or_zero(gd_grads[1], sparse.output_delta)
            head_ga = v2.project_into_basis(ga_head, bs)
            head_gd = v2.project_into_basis(gd_head, bp)

            stage1_optimizer.zero_grad(set_to_none=True)
            sparse.input_delta.grad = compose_embedding_gradient(
                ga_emb, gd_emb
            ).to(sparse.input_delta.dtype)
            sparse.output_delta.grad = compose_head_gradient(
                head_ga, head_gd
            ).to(sparse.output_delta.dtype)
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
                    "embedding_ga_gradient_norm_applied": float(ga_emb.norm().detach().cpu()),
                    "embedding_gd_gradient_norm_not_applied": float(gd_emb.norm().detach().cpu()),
                    "lm_head_ga_BS_gradient_norm": float(head_ga.norm().detach().cpu()),
                    "lm_head_gd_BP_gradient_norm": float(head_gd.norm().detach().cpu()),
                    **sparse.delta_norms(),
                    "official_rwku_records_accessed": False,
                }
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(
                    "L1 step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f} embGD(skip)={:.4f}".format(
                        step,
                        row["ga_sensitive_logprob"],
                        row["gd_non_sensitive_kl"],
                        row["selected_input_row_delta_norm"],
                        row["selected_output_row_delta_norm"],
                        row["embedding_gd_gradient_norm_not_applied"],
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
            anchor_key = base2.stage1_anchor_key(atomic, utility, norms, step)
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
                "  L1 checkpoint {}: direct={} other={} margin_fail={} KL={:.6f}/{:.6f}/{:.6f} utility_safe={}".format(
                    step,
                    atomic.get("FS"),
                    atomic.get("generated_subject_FS"),
                    len(base2.residual_prompt_positions(atomic)),
                    utility["utility_kl_mean"],
                    utility["utility_kl_p95"],
                    utility["utility_kl_max"],
                    u_safe,
                )
            )
            if u_safe and (
                level1_anchor_key_value is None or anchor_key < level1_anchor_key_value
            ):
                level1_anchor_key_value = anchor_key
                level1_anchor = base2.snapshot_with_key(
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
            "schema_version": "rwku_directional_sure_two_stage_emb_ga_only_result_v1",
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

        level2_residual_positions = base2.residual_prompt_positions(
            level1_anchor["atomic"]
        )
        if not level2_residual_positions:
            raise RuntimeError(
                "Level 1 was not atomically safe but exposes no pairwise margin failures"
            )
        level2_residual_case_indices = base2.residual_case_indices(
            cases, level2_residual_positions
        )
        level2_residual_rows = residual_content_rows(
            tids_all,
            level2_residual_case_indices,
            sensitive_rows,
        )
        residual_cases = [cases[index] for index in level2_residual_case_indices]
        input_mask = base2.row_mask(sparse, level2_residual_rows, for_output=False)
        output_mask = base2.row_mask(sparse, level2_residual_rows, for_output=True)
        if int(input_mask.sum().item()) != len(level2_residual_rows):
            raise RuntimeError("Level-2 input row mask does not match residual rows")
        if int(output_mask.sum().item()) != len(level2_residual_rows):
            raise RuntimeError("Level-2 output row mask does not match residual rows")

        stage2 = cfg["stage2"]
        stage2_cfg = base2.stage2_basis_cfg(cfg)
        stage2_optimizer = base2.optimizer_for(
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
            "Level 2 triggered from L1 step {}: residual_prompts={} residual_cases={} residual_content_rows={}".format(
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
                        "level2_residual_content_row_count": len(level2_residual_rows),
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
                ga_emb = base2.grad_or_zero(ga_grads[0], sparse.input_delta)
                gd_emb = base2.grad_or_zero(gd_grads[0], sparse.input_delta)
                ga_head_raw = (
                    base2.grad_or_zero(ga_grads[1], sparse.output_delta) * output_mask
                )
                gd_head_raw = (
                    base2.grad_or_zero(gd_grads[1], sparse.output_delta) * output_mask
                )
                head_ga = v2.project_into_basis(ga_head_raw, bf)
                head_gd = v2.project_into_basis(gd_head_raw, bp2)

                stage2_optimizer.zero_grad(set_to_none=True)
                sparse.input_delta.grad = compose_embedding_gradient(
                    ga_emb, gd_emb, input_mask
                ).to(sparse.input_delta.dtype)
                sparse.output_delta.grad = compose_head_gradient(
                    head_ga, head_gd
                ).to(sparse.output_delta.dtype)
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
                        "residual_sensitive_case_count": len(level2_residual_case_indices),
                        "residual_content_row_count": len(level2_residual_rows),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "gd_non_sensitive_kl": float(gd.detach().cpu()),
                        "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                        "embedding_ga_gradient_norm_after_row_mask_applied": float(
                            (ga_emb * input_mask).norm().detach().cpu()
                        ),
                        "embedding_gd_gradient_norm_after_row_mask_not_applied": float(
                            (gd_emb * input_mask).norm().detach().cpu()
                        ),
                        "lm_head_ga_BF_gradient_norm": float(head_ga.norm().detach().cpu()),
                        "lm_head_gd_BP_gradient_norm": float(head_gd.norm().detach().cpu()),
                        **sparse.delta_norms(),
                        "official_rwku_records_accessed": False,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(
                        "L2 step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f} embGD(skip)={:.4f}".format(
                            step,
                            row["ga_sensitive_logprob"],
                            row["gd_non_sensitive_kl"],
                            row["selected_input_row_delta_norm"],
                            row["selected_output_row_delta_norm"],
                            row["embedding_gd_gradient_norm_after_row_mask_not_applied"],
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
                    "remaining_margin_failure_positions": base2.residual_prompt_positions(
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

        core.write_json(out / "level2_basis_refresh_history.json", level2_basis_history)
        core.write_json(out / "level2_checkpoint_history.json", level2_checkpoints)
        del stage2_optimizer
        if stage2_best is not None:
            final_candidate = stage2_best

    v2.assert_transformer_versions(model, transformer_versions)
    training_seconds = time.perf_counter() - started

    if final_candidate is None:
        result = {
            "schema_version": "rwku_directional_sure_two_stage_emb_ga_only_result_v1",
            "configuration_id": cfg["configuration_id"],
            "method": cfg["method"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "feasible": False,
            "reason": (
                "Level 2 found no checkpoint satisfying all generated atomic + "
                "external-Wikipedia selection gates"
            ),
            "level1_anchor_step": int(level1_anchor["step"]),
            "level1_anchor_atomic": level1_anchor["atomic"],
            "level1_anchor_selection_utility": level1_anchor["selection_utility"],
            "level2_used": level2_used,
            "level2_initial_residual_prompt_positions": level2_residual_positions,
            "level2_initial_residual_sensitive_case_count": len(
                level2_residual_case_indices
            ),
            "level2_initial_residual_content_row_ids": level2_residual_rows,
            "level3_used": False,
            "representation_repair_used": False,
            "transformer_exactly_frozen": True,
            "non_sensitive_embedding_rows_exact_base": True,
            "non_sensitive_lm_head_rows_exact_base": True,
            "embedding_gd_applied_level1": False,
            "embedding_gd_applied_level2": False,
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
        raise RuntimeError("Two-stage embedding-GA-only changed a non-selected row")
    v2.assert_transformer_versions(model, transformer_versions)

    result = {
        "schema_version": "rwku_directional_sure_two_stage_emb_ga_only_result_v1",
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
        "level2_initial_residual_sensitive_case_count": len(level2_residual_case_indices),
        "level2_initial_residual_content_row_ids": level2_residual_rows,
        "level3_used": False,
        "representation_repair_used": False,
        "embedding_gd_applied_level1": False,
        "embedding_gd_applied_level2": False,
        "lm_head_gd_applied_via_protected_basis": True,
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
            "Two-stage Directional SURE development ablation. Both levels use the "
            "original content-sensitive vocabulary row policy. Embeddings receive GA "
            "only; LM-head GA/GD remain directionally routed through B_S/B_F and B_P. "
            "No transformer/MLP/LoRA/representation repair is used."
        ),
    }
    core.write_json(out / "result.json", result)

    if args.save_checkpoint and feasible:
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible two-stage embedding-GA-only checkpoint: {checkpoint}")

    print("\nRWKU TWO-STAGE DIRECTIONAL SURE EMBEDDING-GA-ONLY RESULT")
    print(f"L1 anchor step: {result['level1_anchor_step']}")
    print(f"L2 used: {result['level2_used']}")
    print(f"L2 initial residual prompts: {len(level2_residual_positions)}")
    print(f"L2 initial residual content rows: {len(level2_residual_rows)}")
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
