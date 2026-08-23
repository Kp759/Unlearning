#!/usr/bin/env python3
"""RWKU Directional SURE with MQuAKE-style Stage 2 over all residual target rows."""
from __future__ import annotations

from rwku_mquake_stage2_all_residual_helpers import *  # noqa: F401,F403
from rwku_mquake_stage2_all_residual_runtime import prepare_runtime
from rwku_mquake_stage2_train import train_stage2


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked configuration ID")

    run_dir = v2.verify_prepared_state(args, cfg)
    final_out = run_dir / LEARNER_DIR
    if final_out.exists():
        raise FileExistsError(f"Refusing to overwrite: {final_out}")

    # Exact previously tested L1; intercepted immediately before its legacy L2.
    level1_anchor = run_locked_level1_capture(args, cfg)
    final_out.mkdir(parents=True)
    started = time.perf_counter()

    runtime = prepare_runtime(args, cfg, level1_anchor)
    core.write_json(final_out / "level2_basis_report.json", runtime["basis_report"])
    protocol = {
        "schema_version": "rwku_directional_sure_mquake_stage2_all_residual_rows_protocol_v1",
        "configuration_id": cfg["configuration_id"],
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "level1_capture_directory": str((run_dir / LEVEL1_CAPTURE_DIR).resolve()),
        "level1_anchor_step": int(level1_anchor["step"]),
        "level1_anchor_atomic": level1_anchor["atomic"],
        "level1_anchor_selection_utility": level1_anchor["selection_utility"],
        "level1_content_row_count": len(runtime["level1_content_rows"]),
        "level1_content_rows": runtime["level1_content_rows"],
        "stage2_success_prompt_count": len(runtime["protected_positions"]),
        "stage2_residual_prompt_count": len(runtime["residual_positions"]),
        "stage2_success_case_count": len(runtime["protected_case_indices"]),
        "stage2_residual_case_count": len(runtime["residual_case_indices"]),
        "stage2_all_non_special_residual_target_rows": runtime["residual_rows"],
        "stage2_newly_admitted_rows_outside_level1_content_set": runtime[
            "newly_admitted_stage2_rows"
        ],
        "runtime_output_row_union": runtime["runtime_output_rows"],
        "stage2_basis": runtime["basis_report"],
        "stage2_config": cfg["stage2"],
        "sensitive_row_audit": runtime["sensitive_row_audit"],
        "untie_audit": runtime["untie_audit"],
        "freeze_audit": runtime["freeze_audit"],
        "external_wikipedia_dataset": runtime["wikipedia_meta"],
        "external_slices": runtime["external_audit"],
        "external_base_hidden_cached_before_level1_anchor": True,
        "bundle_audit": runtime["bundle_audit"],
        "generator_model_audit": runtime["generator_model_audit"],
        "official_rwku_evaluation_locked": True,
    }
    core.write_json(final_out / "protocol_report.json", protocol)

    print(
        "All-residual-row L2 from L1 step {}: P={} F={} P_cases={} F_cases={} "
        "L1_rows={} F_rows={} new_F_rows={} output_union={} B_P={} B_F={}".format(
            level1_anchor["step"],
            len(runtime["protected_positions"]),
            len(runtime["residual_positions"]),
            len(runtime["protected_case_indices"]),
            len(runtime["residual_case_indices"]),
            len(runtime["level1_content_rows"]),
            len(runtime["residual_rows"]),
            len(runtime["newly_admitted_stage2_rows"]),
            len(runtime["runtime_output_rows"]),
            int(runtime["bp"].shape[0]),
            int(runtime["bf"].shape[0]),
        )
    )

    # The shared trainer expects the anchor output table to align with the
    # runtime sparse output rows. Expand L1 with exact zeros on Stage-2-only rows.
    stage2_anchor = dict(level1_anchor)
    stage2_anchor["output_delta"] = runtime["l1_output_anchor"].detach().cpu().clone()
    final_candidate = train_stage2(runtime, cfg, stage2_anchor, final_out)

    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = runtime["device"]
    llama_like = runtime["llama_like"]
    prompt_records = runtime["prompt_records"]
    sparse = runtime["sparse"]
    input_layer = runtime["input_layer"]
    output_layer = runtime["output_layer"]

    v2.assert_transformer_versions(model, runtime["transformer_versions"])
    if final_candidate is None:
        result = {
            "schema_version": "rwku_directional_sure_mquake_stage2_all_residual_rows_result_v1",
            "configuration_id": cfg["configuration_id"],
            "method": cfg["method"],
            "development_only": True,
            "official_rwku_records_accessed": False,
            "feasible": False,
            "reason": "All-residual-row MQuAKE-style Stage 2 found no checkpoint satisfying atomic + hard-P + external-Wikipedia gates",
            "level1_anchor_step": int(level1_anchor["step"]),
            "level1_anchor_atomic": level1_anchor["atomic"],
            "level1_anchor_selection_utility": level1_anchor["selection_utility"],
            "stage2_success_prompt_count": len(runtime["protected_positions"]),
            "stage2_initial_residual_prompt_count": len(runtime["residual_positions"]),
            "stage2_all_non_special_residual_target_rows": runtime["residual_rows"],
            "stage2_newly_admitted_rows": runtime["newly_admitted_stage2_rows"],
            "stage2_basis": runtime["basis_report"],
            "stage2_embedding_frozen_at_level1_anchor": True,
            "transformer_exactly_frozen": True,
            "level3_used": False,
            "training_seconds_after_level1_capture": time.perf_counter() - started,
        }
        core.write_json(final_out / "result.json", result)
        raise RuntimeError(result["reason"])

    with torch.no_grad():
        sparse.input_delta.copy_(
            final_candidate["input_delta"].to(sparse.input_delta.device)
        )
        sparse.output_delta.copy_(
            final_candidate["output_delta"].to(sparse.output_delta.device)
        )

    margin = float(cfg["acceptance"]["required_pairwise_margin"])
    final_atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=margin,
    )
    final_atomic_safe = v2.atomic_safe(final_atomic, cfg)
    fresh_utility = v2.exact_external_kl_report(
        model,
        tokenizer,
        runtime["fresh_contexts"],
        runtime["fresh_base_hidden"],
        device=device,
        batch_size=int(runtime["utility_bs"]),
    )
    fresh_safe = v2.utility_safe(fresh_utility, cfg)
    feasible = bool(final_atomic_safe and fresh_safe)

    sparse.materialize(
        final_candidate["input_delta"], final_candidate["output_delta"], 1.0
    )
    nonselected_input_equal = v2._nonselected_equal_base(
        input_layer.weight.detach(),
        runtime["base_vocab_cpu"],
        runtime["level1_content_rows"],
    )
    nonselected_output_equal = v2._nonselected_equal_base(
        output_layer.weight.detach(),
        runtime["base_vocab_cpu"],
        runtime["runtime_output_rows"],
    )
    if not nonselected_input_equal or not nonselected_output_equal:
        raise RuntimeError("A vocabulary row outside the declared L1/L2 row sets changed")
    v2.assert_transformer_versions(model, runtime["transformer_versions"])

    result = {
        "schema_version": "rwku_directional_sure_mquake_stage2_all_residual_rows_result_v1",
        "configuration_id": cfg["configuration_id"],
        "method": cfg["method"],
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "level1_anchor_step": int(level1_anchor["step"]),
        "level1_anchor_atomic": level1_anchor["atomic"],
        "level1_anchor_selection_utility": level1_anchor["selection_utility"],
        "level1_content_rows": runtime["level1_content_rows"],
        "stage2_success_prompt_count": len(runtime["protected_positions"]),
        "stage2_initial_residual_prompt_count": len(runtime["residual_positions"]),
        "stage2_all_non_special_residual_target_rows": runtime["residual_rows"],
        "stage2_newly_admitted_rows": runtime["newly_admitted_stage2_rows"],
        "runtime_output_row_union": runtime["runtime_output_rows"],
        "stage2_basis": runtime["basis_report"],
        "stage2_parameter_scope": "lm_head_only_increment_C_F_B_F",
        "stage2_embedding_frozen_at_level1_anchor": True,
        "hard_success_regression_limit": cfg["stage2"]["hard_success_regression_limit"],
        "hard_success_kl_budget": cfg["stage2"]["hard_success_kl_budget"],
        "selected_checkpoint_step": int(final_candidate["step"]),
        "repair_delta_norm": float(final_candidate["repair_delta_norm"]),
        "delta_norms": final_candidate["delta_norms"],
        "final_atomic": final_atomic,
        "final_atomic_safe": final_atomic_safe,
        "fresh_external_wikipedia_utility": fresh_utility,
        "fresh_external_wikipedia_utility_safe": fresh_safe,
        "feasible": feasible,
        "transformer_exactly_frozen": True,
        "rows_outside_level1_input_set_exact_base": nonselected_input_equal,
        "rows_outside_runtime_output_union_exact_base": nonselected_output_equal,
        "level3_used": False,
        "representation_repair_used": False,
        "official_rwku_paraphrase_seen": False,
        "official_rwku_neighborhood_seen": False,
        "official_rwku_retain_seen": False,
        "official_rwku_ppl_text_seen": False,
        "fresh_gate_opened_only_after_final_checkpoint_selection": True,
        "external_base_hidden_cached_before_level1_anchor": True,
        "training_seconds_after_level1_capture": time.perf_counter() - started,
    }
    core.write_json(final_out / "result.json", result)

    if args.save_checkpoint and feasible:
        checkpoint = final_out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible checkpoint: {checkpoint}")

    print("\nRWKU DIRECTIONAL SURE + ALL-RESIDUAL-ROW MQUAKE-STYLE STAGE2 RESULT")
    print(f"L1 anchor step: {result['level1_anchor_step']}")
    print(
        f"P/F prompts: {len(runtime['protected_positions'])} / {len(runtime['residual_positions'])}"
    )
    print(
        "L1 content rows / residual target rows / newly admitted rows: {} / {} / {}".format(
            len(runtime["level1_content_rows"]),
            len(runtime["residual_rows"]),
            len(runtime["newly_admitted_stage2_rows"]),
        )
    )
    print(f"selected L2 step: {result['selected_checkpoint_step']}")
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
    print(f"feasible: {feasible}")
    print(f"result: {final_out / 'result.json'}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
