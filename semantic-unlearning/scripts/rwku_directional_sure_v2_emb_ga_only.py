#!/usr/bin/env python3
"""Directional SURE v2 one-factor ablation: remove GD only from embeddings.

This learner is intentionally anchored to the original content-sensitive
Directional SURE v2 implementation.  The only optimization-path change is:

    embedding:  2*GA + GD  ->  2*GA
    LM head:    2*GA -> B_S  +  GD -> B_P   (unchanged)

The content-sensitive row selector, frozen transformer, dynamic B_S/B_P bases,
learning rates, ranks, step count, external-Wikipedia slices, and acceptance
budgets are unchanged from Directional SURE v2.  Official RWKU evaluation
artifacts remain unavailable during learning and checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

import rwku_directional_sure_v2 as base

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = (
    PROJECT_ROOT / "config" / "rwku" / "directional_sure_v2_emb_ga_only_seed0.json"
)
SCHEMA = "rwku_directional_sure_v2_emb_ga_only_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-v2-emb-ga-only-stephen-king-seed0"
LEARNER_DIR = "directional_sure_v2_emb_ga_only"


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


def load_configuration(path: Path) -> Dict[str, Any]:
    """Lock the ablation to v2 except for the declared embedding gradient rule."""
    cfg = base.read_json(path)
    baseline = base.read_json(base.DEFAULT_CONFIGURATION)

    required = {
        "schema_version": SCHEMA,
        "configuration_id": EXPERIMENT_ID,
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_metrics_observed_before_method_design": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "neutral_target": "Unknown",
        "embedding_gradient_policy": "GA_only_no_GD_no_hidden_basis_projection",
        "lm_head_gradient_policy": "GA_to_sensitive_exclusive_basis_and_GD_to_protected_basis",
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Embedding-GA-only configuration changed {key}")

    # Exact one-factor lock: row locality, trainable modules, acceptance gates,
    # and data boundary must be byte-for-byte equivalent as parsed JSON.
    for section in ("trainable_components", "acceptance", "data_boundary"):
        if cfg.get(section) != baseline.get(section):
            raise ValueError(f"Embedding-GA-only ablation changed baseline section {section}")

    opt = dict(cfg.get("optimization", {}))
    baseline_opt = dict(baseline.get("optimization", {}))
    expected_objective = (
        "embedding_GA_only_plus_same_prompt_non_sensitive_GD_KL_on_directional_lm_head"
    )
    if opt.get("objective") != expected_objective:
        raise ValueError("Embedding-GA-only objective label changed")
    opt_without_objective = dict(opt)
    baseline_without_objective = dict(baseline_opt)
    opt_without_objective.pop("objective", None)
    baseline_without_objective.pop("objective", None)
    if opt_without_objective != baseline_without_objective:
        raise ValueError("Embedding-GA-only ablation changed baseline optimization settings")

    return cfg


def compose_embedding_gradient(
    ga_emb: torch.Tensor,
    gd_emb: torch.Tensor,
) -> torch.Tensor:
    """Apply GA only to embeddings; GD is measured but intentionally not applied."""
    if ga_emb.shape != gd_emb.shape:
        raise ValueError("Embedding GA/GD gradient shapes differ")
    return ga_emb


def compose_head_gradient(
    head_ga_directional: torch.Tensor,
    head_gd_directional: torch.Tensor,
) -> torch.Tensor:
    """Keep the original v2 LM-head rule unchanged."""
    if head_ga_directional.shape != head_gd_directional.shape:
        raise ValueError("LM-head GA/GD gradient shapes differ")
    return head_ga_directional + head_gd_directional


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked embedding-GA-only configuration ID")

    run_dir = base.verify_prepared_state(args, cfg)
    out = run_dir / LEARNER_DIR
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite embedding-GA-only output: {out}")
    out.mkdir(parents=True)

    source_cfg = base.head.load_locked_configuration(base.SOURCE_BUNDLE_CONFIGURATION)
    views, bundle_audit, generator_audit = base.head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_cfg,
    )
    generator_model_audit = base.head.validate_generator_base_model(
        generator_audit, args.model_path
    )

    base.gagd.set_seed(int(cfg["seed"]))
    base.gagd.require_cuda_if_needed(str(cfg["acceptance"]["device_map"]))
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=str(cfg["acceptance"]["checkpoint_dtype"]),
        device_map=str(cfg["acceptance"]["device_map"]),
        gradient_checkpointing=False,
    )
    model, tokenizer = base.gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = base.gagd.first_device(model)
    llama_like = base.core.is_llama_like(model, tokenizer)

    prompt_records = base.head.compile_prompt_records(
        views, tokenizer, neutral_target=str(cfg["neutral_target"])
    )
    cases = base.core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    if not cases:
        raise RuntimeError("Embedding-GA-only Directional SURE created no sensitive cases")

    # Cache Base same-prompt teacher logits before any editable delta exists.
    base_logits = base.core.cache_base_logits(
        model,
        tokenizer,
        cases,
        device,
        batch_size=int(cfg["optimization"]["cache_batch_size"]),
    )
    tids_all = base.core.official_target_ids(
        tokenizer, cases, llama_like=llama_like, device=device
    )
    # IMPORTANT: retain the original v2 content-sensitive row policy on BOTH E and W.
    sensitive_rows, sensitive_row_audit = base._content_sensitive_rows(
        tokenizer, cases, tids_all, source_cfg, len(prompt_records)
    )

    sample_ids = tokenizer(
        prompt_records[0]["prompt_text"], return_tensors="pt"
    )["input_ids"].to(device)
    untie_audit = base.sparse_rows.untie_lm_head_preserve_logits(
        model, sample_input_ids=sample_ids
    )
    freeze_audit = base.sparse_rows.freeze_transformer_parameters(model)
    transformer_versions = base._parameter_versions_except_vocab(model)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not torch.equal(input_layer.weight.detach(), output_layer.weight.detach()):
        raise RuntimeError("Untied Base input/output vocabulary weights are not initially identical")
    base_vocab_cpu = input_layer.weight.detach().cpu().clone()

    sparse = base.sparse_rows.SparseFP32RowDeltas(
        model,
        selected_input_rows=sensitive_rows,
        selected_output_rows=sensitive_rows,
    )
    if sparse.input_delta.dtype != torch.float32 or sparse.output_delta.dtype != torch.float32:
        raise RuntimeError("Embedding-GA-only sparse row masters must be FP32")

    texts, wikipedia_meta = base.wikipedia.load_wikipedia_train(
        Path(args.wikipedia_dir).resolve()
    )
    protected_contexts, selection_contexts, fresh_contexts, external_audit = (
        base.build_external_slices(tokenizer, texts, cfg)
    )
    utility_bs = int(cfg["optimization"]["utility_batch_size"])
    selection_base_hidden = base.external_final_hidden(
        model,
        tokenizer,
        selection_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()
    fresh_base_hidden = base.external_final_hidden(
        model,
        tokenizer,
        fresh_contexts,
        device=device,
        batch_size=utility_bs,
    ).cpu()

    base.core.write_json(
        out / "protocol_report.json",
        {
            "schema_version": "rwku_directional_sure_v2_emb_ga_only_protocol_v1",
            "configuration_id": cfg["configuration_id"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "one_factor_ablation_from": "rwku-directional-sure-v2-stephen-king-seed0",
            "changed_factor": "embedding GD gradient removed",
            "embedding_gradient_rule": "2 * GA only; embedding GD gradient measured but not applied",
            "lm_head_gradient_rule": "2 * GA projected to B_S + GD projected to B_P",
            "bundle_audit": bundle_audit,
            "generator_model_audit": generator_model_audit,
            "training_prompt_count": len(prompt_records),
            "sensitive_prediction_case_count": len(cases),
            "sensitive_row_audit": sensitive_row_audit,
            "selected_sensitive_row_count": len(sensitive_rows),
            "selected_sensitive_row_ids": sensitive_rows,
            "input_and_output_selected_rows_identical": True,
            "untie_audit": untie_audit,
            "freeze_audit": freeze_audit,
            "external_wikipedia_dataset": wikipedia_meta,
            "external_slices": external_audit,
            "objective": cfg["optimization"]["objective"],
            "ga_definition": "mean log p_theta(sensitive token | generated training prompt/prefix); minimized",
            "gd_definition": "KL(Base_non_sensitive || current_non_sensitive), same prompt; used only for LM-head GD branch",
            "basis_definition": {
                "B_P": "orthonormal span of current final hidden states on fixed external-Wikipedia protected contexts",
                "B_S": "orthonormal span of current sensitive-prediction hidden states after projection into orthogonal complement of B_P",
                "refresh_interval": int(cfg["optimization"]["basis_refresh_interval"]),
            },
            "parameter_locality": "FP32 sparse deltas on original v2 content-sensitive vocabulary rows; transformer exactly frozen",
        },
    )

    opt_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [sparse.input_delta],
                "lr": float(opt_cfg["embedding_learning_rate"]),
                "weight_decay": 0.0,
            },
            {
                "params": [sparse.output_delta],
                "lr": float(opt_cfg["lm_head_learning_rate"]),
                "weight_decay": 0.0,
            },
        ]
    )
    sampler = base.core.IndexSampler(
        len(cases), int(opt_cfg["batch_size"]), int(cfg["seed"])
    )
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[Any, ...]] = None
    basis_history: List[Dict[str, Any]] = []
    checkpoint_history: List[Dict[str, Any]] = []
    train_log_path = out / "train_log.jsonl"
    bs: Optional[torch.Tensor] = None
    bp: Optional[torch.Tensor] = None
    started = time.perf_counter()
    model.eval()

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        for step in range(1, int(opt_cfg["steps"]) + 1):
            if step == 1 or (step - 1) % int(opt_cfg["basis_refresh_interval"]) == 0:
                bs, bp, basis_report = base.refresh_directional_bases(
                    model,
                    tokenizer,
                    cases,
                    protected_contexts,
                    cfg,
                    device=device,
                )
                basis_report = {
                    "refresh_before_step": int(step),
                    "updates_since_previous_refresh": (
                        0 if step == 1 else int(opt_cfg["basis_refresh_interval"])
                    ),
                    **basis_report,
                }
                basis_history.append(basis_report)
                print(
                    "basis refresh before step {}: B_S rank={} B_P rank={} overlap={:.3e} exclusive_energy={:.4f}".format(
                        step,
                        basis_report["sensitive_exclusive_rank"],
                        basis_report["protected_rank"],
                        basis_report["max_abs_sensitive_protected_basis_overlap"],
                        basis_report["sensitive_energy_after_protected_projection_fraction"],
                    )
                )
            if bs is None or bp is None:
                raise RuntimeError("Directional bases are unavailable")

            idx = sampler.next()
            batch = [cases[i] for i in idx]
            logits = base.core.forward_last_logits(model, tokenizer, batch, device)
            tids = base.core.official_target_ids(
                tokenizer, batch, llama_like=llama_like, device=device
            )
            ga = base.core.ga_sensitive_logprob(logits, tids)
            gd = base.core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
            params = (sparse.input_delta, sparse.output_delta)
            ga_grads = torch.autograd.grad(
                float(opt_cfg["ga_weight"]) * ga,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            gd_grads = torch.autograd.grad(
                float(opt_cfg["gd_weight"]) * gd,
                params,
                retain_graph=False,
                allow_unused=True,
            )

            def grad_or_zero(
                value: Optional[torch.Tensor], parameter: torch.Tensor
            ) -> torch.Tensor:
                return torch.zeros_like(parameter) if value is None else value.float()

            ga_emb = grad_or_zero(ga_grads[0], sparse.input_delta)
            ga_head = grad_or_zero(ga_grads[1], sparse.output_delta)
            gd_emb = grad_or_zero(gd_grads[0], sparse.input_delta)
            gd_head = grad_or_zero(gd_grads[1], sparse.output_delta)
            head_ga_directional = base.project_into_basis(ga_head, bs)
            head_gd_directional = base.project_into_basis(gd_head, bp)

            optimizer.zero_grad(set_to_none=True)
            # THE ONLY OPTIMIZATION-PATH CHANGE FROM v2:
            # old: sparse.input_delta.grad = ga_emb + gd_emb
            # new: embedding GD is diagnostic-only and never reaches optimizer.step().
            sparse.input_delta.grad = compose_embedding_gradient(
                ga_emb, gd_emb
            ).to(sparse.input_delta.dtype)
            # LM-head path is intentionally unchanged from v2.
            sparse.output_delta.grad = compose_head_gradient(
                head_ga_directional, head_gd_directional
            ).to(sparse.output_delta.dtype)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [sparse.input_delta, sparse.output_delta],
                float(opt_cfg["grad_clip"]),
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite embedding-GA-only gradient norm at step {step}"
                )
            optimizer.step()

            if step == 1 or step % 25 == 0 or step == int(opt_cfg["steps"]):
                row = {
                    "step": int(step),
                    "objective_for_reporting": float(
                        (
                            float(opt_cfg["ga_weight"]) * ga
                            + float(opt_cfg["gd_weight"]) * gd
                        ).detach().cpu()
                    ),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_non_sensitive_kl": float(gd.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "embedding_ga_gradient_norm_applied": float(
                        ga_emb.norm().detach().cpu()
                    ),
                    "embedding_gd_gradient_norm_not_applied": float(
                        gd_emb.norm().detach().cpu()
                    ),
                    "embedding_gd_applied": False,
                    "lm_head_ga_gradient_norm_before_projection": float(
                        ga_head.norm().detach().cpu()
                    ),
                    "lm_head_ga_BS_gradient_norm_after_projection": float(
                        head_ga_directional.norm().detach().cpu()
                    ),
                    "lm_head_gd_gradient_norm_before_projection": float(
                        gd_head.norm().detach().cpu()
                    ),
                    "lm_head_gd_BP_gradient_norm_after_projection": float(
                        head_gd_directional.norm().detach().cpu()
                    ),
                    "lm_head_gd_applied": True,
                    **sparse.delta_norms(),
                    "official_rwku_records_accessed": False,
                }
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
                print(
                    "emb-GA-only step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f} embGD(skip)={:.4f} GA->BS={:.4f} GD->BP={:.4f}".format(
                        step,
                        row["ga_sensitive_logprob"],
                        row["gd_non_sensitive_kl"],
                        row["selected_input_row_delta_norm"],
                        row["selected_output_row_delta_norm"],
                        row["embedding_gd_gradient_norm_not_applied"],
                        row["lm_head_ga_BS_gradient_norm_after_projection"],
                        row["lm_head_gd_BP_gradient_norm_after_projection"],
                    )
                )

            if (
                step % int(opt_cfg["checkpoint_interval"]) != 0
                and step != int(opt_cfg["steps"])
            ):
                continue

            atomic = base.head.materialized_atomic_report(
                model,
                tokenizer,
                prompt_records,
                device,
                llama_like=llama_like,
                required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
            )
            selection_utility = base.exact_external_kl_report(
                model,
                tokenizer,
                selection_contexts,
                selection_base_hidden,
                device=device,
                batch_size=utility_bs,
            )
            a_safe = base.atomic_safe(atomic, cfg)
            u_safe = base.utility_safe(selection_utility, cfg)
            checkpoint = {
                "step": int(step),
                "atomic": atomic,
                "selection_utility": selection_utility,
                "atomic_safe": a_safe,
                "selection_utility_safe": u_safe,
                "eligible": bool(a_safe and u_safe),
                "delta_norms": sparse.delta_norms(),
                "embedding_gd_applied": False,
                "lm_head_gd_applied": True,
                "official_rwku_records_accessed": False,
            }
            checkpoint_history.append(checkpoint)
            print(
                "  checkpoint {}: direct={} other={} KL={:.6f}/{:.6f}/{:.6f} eligible={}".format(
                    step,
                    atomic.get("FS"),
                    atomic.get("generated_subject_FS"),
                    selection_utility["utility_kl_mean"],
                    selection_utility["utility_kl_p95"],
                    selection_utility["utility_kl_max"],
                    checkpoint["eligible"],
                )
            )
            if checkpoint["eligible"]:
                candidate = base.snapshot_candidate(
                    sparse,
                    step=step,
                    atomic=atomic,
                    utility=selection_utility,
                )
                key = tuple(candidate["selection_key"])
                if best_key is None or key < best_key:
                    best_key = key
                    best = candidate

    training_seconds = time.perf_counter() - started
    base.core.write_json(out / "basis_refresh_history.json", basis_history)
    base.core.write_json(out / "checkpoint_history.json", checkpoint_history)
    base.assert_transformer_versions(model, transformer_versions)

    # Hooks apply sparse deltas functionally; immutable Base vocab parameters
    # must remain exact until selected rows are materialized.
    if not torch.equal(input_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base input vocabulary matrix changed before materialization")
    if not torch.equal(output_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base output vocabulary matrix changed before materialization")

    if best is None:
        base.core.write_json(
            out / "result.json",
            {
                "schema_version": "rwku_directional_sure_v2_emb_ga_only_result_v1",
                "configuration_id": cfg["configuration_id"],
                "method": cfg["method"],
                "development_only": True,
                "posthoc_development_target": True,
                "official_rwku_records_accessed": False,
                "feasible": False,
                "reason": "no checkpoint passed generated atomic behavior plus external-Wikipedia selection KL gates",
                "selected_checkpoint_step": None,
                "training_seconds": training_seconds,
                "embedding_gradient_policy": cfg["embedding_gradient_policy"],
                "lm_head_gradient_policy": cfg["lm_head_gradient_policy"],
                "transformer_exactly_frozen": True,
                "non_sensitive_embedding_rows_exact_base": True,
                "non_sensitive_lm_head_rows_exact_base": True,
            },
        )
        raise RuntimeError(
            "Directional SURE v2 embedding-GA-only found no eligible development checkpoint"
        )

    with torch.no_grad():
        sparse.input_delta.copy_(best["input_delta"].to(device=sparse.input_delta.device))
        sparse.output_delta.copy_(best["output_delta"].to(device=sparse.output_delta.device))

    # Fresh external gate opens only after checkpoint selection is frozen.
    fresh_utility = base.exact_external_kl_report(
        model,
        tokenizer,
        fresh_contexts,
        fresh_base_hidden,
        device=device,
        batch_size=utility_bs,
    )
    fresh_safe = base.utility_safe(fresh_utility, cfg)
    final_atomic = base.head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    final_atomic_safe = base.atomic_safe(final_atomic, cfg)
    feasible = bool(final_atomic_safe and fresh_safe)

    sparse.materialize(best["input_delta"], best["output_delta"], 1.0)
    nonselected_input_equal = base._nonselected_equal_base(
        input_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    nonselected_output_equal = base._nonselected_equal_base(
        output_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    if not nonselected_input_equal or not nonselected_output_equal:
        raise RuntimeError("Embedding-GA-only run changed a non-sensitive vocabulary row")
    base.assert_transformer_versions(model, transformer_versions)

    result = {
        "schema_version": "rwku_directional_sure_v2_emb_ga_only_result_v1",
        "configuration_id": cfg["configuration_id"],
        "method": cfg["method"],
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "selected_checkpoint_step": int(best["step"]),
        "selection_key": list(best["selection_key"]),
        "selected_sensitive_row_count": len(sensitive_rows),
        "selected_sensitive_row_ids": sensitive_rows,
        "delta_norms": sparse.delta_norms(best["input_delta"], best["output_delta"]),
        "embedding_gradient_policy": cfg["embedding_gradient_policy"],
        "lm_head_gradient_policy": cfg["lm_head_gradient_policy"],
        "final_atomic": final_atomic,
        "final_atomic_safe": final_atomic_safe,
        "fresh_external_wikipedia_utility": fresh_utility,
        "fresh_external_wikipedia_utility_safe": fresh_safe,
        "feasible": feasible,
        "transformer_exactly_frozen": True,
        "non_sensitive_embedding_rows_exact_base": nonselected_input_equal,
        "non_sensitive_lm_head_rows_exact_base": nonselected_output_equal,
        "fresh_gate_opened_only_after_checkpoint_selection": True,
        "official_rwku_paraphrase_seen": False,
        "official_rwku_neighborhood_seen": False,
        "official_rwku_retain_seen": False,
        "official_rwku_ppl_text_seen": False,
        "training_seconds": training_seconds,
        "interpretation_note": "Post-hoc Stephen King development ablation. Relative to Directional SURE v2, only embedding-side GD is removed; LM-head GA->B_S and GD->B_P are unchanged.",
    }
    base.core.write_json(out / "result.json", result)

    if args.save_checkpoint and feasible:
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible embedding-GA-only checkpoint: {checkpoint}")

    print("\nRWKU DIRECTIONAL SURE v2 EMBEDDING-GA-ONLY RESULT")
    print(f"selected checkpoint step: {result['selected_checkpoint_step']}")
    print(
        "atomic direct/other: {} / {}".format(
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
    print(
        "selected input/head delta norm: {:.6f} / {:.6f}".format(
            result["delta_norms"]["selected_input_row_delta_norm"],
            result["delta_norms"]["selected_output_row_delta_norm"],
        )
    )
    print("embedding GD applied: False")
    print("LM-head GD->B_P applied: True")
    print(f"transformer frozen: {result['transformer_exactly_frozen']}")
    print(
        f"non-sensitive input rows exact Base: {result['non_sensitive_embedding_rows_exact_base']}"
    )
    print(
        f"non-sensitive head rows exact Base: {result['non_sensitive_lm_head_rows_exact_base']}"
    )
    print(f"feasible under unchanged v2 gates: {feasible}")
    print(f"result: {out / 'result.json'}")

    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
