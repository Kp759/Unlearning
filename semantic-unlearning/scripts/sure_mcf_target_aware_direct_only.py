#!/usr/bin/env python3
"""MCF target-aware SURE v8 with direct-only training and FS selection.

The learner accepts a stripped training view rather than the original MCF
source.  It sees direct prompts plus ``target_true`` and ``target_new`` and has
no argument through which official paraphrases can enter training, rank or
margin selection, early stopping, or checkpoint acceptance.  GFS and all
locality/fluency metrics are post-checkpoint audits performed by the runner.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import build_mcf_sure_target_aware_direct_split as direct_split
import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import mcf_zero_unlearn_official_eval as mcf_official
import sure_canonical_core as core
import sure_mcf_direct_fs_repair as exact
import sure_mcf_target_aware_two_stage as joint
import sure_minimal_two_stage as learner


METHOD = "SURE-LM-MCF-target-aware-direct-true-GA-new-GD-v8"
PROTOCOL = direct_split.PROTOCOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--utility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--utility-sample-size", type=int, default=100_000)
    parser.add_argument("--utility-prompt-count", type=int, default=100_000)
    parser.add_argument("--utility-token-topk-per-row", type=int, default=128)
    parser.add_argument("--utility-uniform-prompt-count", type=int, default=1_024)
    parser.add_argument("--utility-pool-seed", type=int, default=1)
    parser.add_argument("--utility-train-batch-size", type=int, default=128)
    parser.add_argument("--utility-eval-batch-size", type=int, default=512)
    parser.add_argument("--cache-batch-size", type=int, default=8)

    parser.add_argument("--stage1-rank", type=int, default=4)
    parser.add_argument("--stage1-steps", type=int, default=600)
    parser.add_argument("--stage1-lr", type=float, default=5e-3)
    parser.add_argument("--stage1-pairwise-target", type=float, default=1.0)
    parser.add_argument("--stage1-true-nll-increase", type=float, default=2.0)
    parser.add_argument("--stage1-new-nll-decrease", type=float, default=1.0)
    parser.add_argument("--stage1-pairwise-weight", type=float, default=100.0)
    parser.add_argument("--stage1-true-ga-weight", type=float, default=10.0)
    parser.add_argument("--stage1-new-gd-weight", type=float, default=10.0)
    parser.add_argument("--stage1-utility-kl-weight", type=float, default=1.0)
    parser.add_argument("--stage1-l2-weight", type=float, default=1e-4)
    parser.add_argument(
        "--stage1-candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.125,0",
    )

    parser.add_argument("--required-pairwise-margin", type=float, default=0.01)
    parser.add_argument(
        "--stage2-solver-margins",
        default="0.5,1.0,2.0",
        help=(
            "Absolute continuous direct-separation targets. These are BF16 "
            "safety targets, not changes to the strict paper FS definition."
        ),
    )
    parser.add_argument("--stage2-rank-ladder", default="2,4,8")
    parser.add_argument("--stage2-maxiter", type=int, default=500)
    parser.add_argument("--stage2-ftol", type=float, default=1e-9)
    parser.add_argument("--stage2-constraint-tolerance", type=float, default=1e-5)
    parser.add_argument("--stage2-residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--constraint-context-weight", type=float, default=0.05)
    parser.add_argument("--contrastive-eps", type=float, default=1e-3)

    parser.add_argument("--utility-kl-mean-budget", type=float, default=0.01)
    parser.add_argument("--utility-kl-p95-budget", type=float, default=0.05)
    parser.add_argument("--utility-kl-max-budget", type=float, default=0.5)
    parser.add_argument("--max-total-delta-norm", type=float, default=1.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def load_locked_direct_records(
    training_path: Path,
    manifest_path: Path,
    *,
    expected_seed: int,
    expected_forget_num: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    training_bytes = training_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError("split manifest is not the direct-only v8 protocol")
    if manifest.get("dataset") != "mcf":
        raise RuntimeError("target-aware direct-only v8 requires dataset=mcf")
    if int(manifest.get("seed", -1)) != int(expected_seed):
        raise RuntimeError("split manifest seed differs from --seed")
    expected_hash = manifest.get("training_visible_target_aware_direct_sha256")
    if expected_hash != joint.sha256_bytes(training_bytes):
        raise RuntimeError("direct-only training file hash does not match manifest")
    if "source_dataset" in manifest:
        raise RuntimeError("v8 manifest must not expose a source-dataset path")

    contract = manifest.get("learner_adapter_contract", {})
    required_contract = {
        "sensitive_answer_field": "target_true",
        "reference_answer_field": "target_new",
        "direct_only": True,
        "official_paraphrases_visible_to_learner": False,
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"invalid v8 learner contract for {key}")
    if tuple(contract.get("forbidden_probe_fields", ())) != direct_split.PROBE_FIELDS:
        raise RuntimeError("v8 learner contract does not forbid every probe field")

    records = json.loads(training_bytes)
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise RuntimeError("direct-only training file must be a JSON list of objects")
    direct_split.assert_direct_only_training_view(records)
    if len(records) != int(expected_forget_num):
        raise RuntimeError(
            "direct-only training record count differs from --forget-num"
        )
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != int(expected_forget_num):
        raise RuntimeError("manifest forget count differs from --forget-num")
    if int(sampling.get("benchmark_retain_train_num", -1)) != 0:
        raise RuntimeError("v8 training may not expose benchmark retain examples")
    record_ids = [int(record["case_id"]) for record in records]
    manifest_ids = [int(value) for value in sampling.get("forget_case_ids", [])]
    if record_ids != manifest_ids:
        raise RuntimeError("direct-only training case order differs from manifest")
    if manifest.get("data_roles", {}).get("GFS_checkpoint_selection") is not False:
        raise RuntimeError("v8 manifest must mark GFS as post-training only")

    prompts: List[Dict[str, Any]] = []
    for record_position, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        prompts.append(
            {
                "case_id": int(record["case_id"]),
                "source_record_position": record_position,
                "prompt_kind": "direct",
                "prompt_index": 0,
                "prompt_text": prompt,
                "requested_rewrite": {
                    "prompt": "{}",
                    "subject": prompt,
                    "target_sensitive": {"str": str(rewrite["target_true"]["str"])},
                    "target_reference": {"str": str(rewrite["target_new"]["str"])},
                },
            }
        )
    return records, prompts, manifest


@torch.no_grad()
def direct_materialized_report(
    model: torch.nn.Module,
    tok: Any,
    training_records: Sequence[Mapping[str, Any]],
    prompt_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    required_margin: float,
) -> Dict[str, Any]:
    if len(training_records) != len(prompt_records):
        raise ValueError("training records and direct prompt records do not align")
    separations: List[float] = []
    for record in training_records:
        rewrite = record["requested_rewrite"]
        prefix = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        scores = mcf_official.official_test_batch_prediction(
            model,
            tok,
            [prefix],
            str(rewrite["target_new"]["str"]),
            str(rewrite["target_true"]["str"]),
            device,
            llama_like=llama_like,
        )
        if len(scores) != 1:
            raise RuntimeError("official scorer omitted a direct prompt")
        separations.append(float(scores[0]["target_true"] - scores[0]["target_new"]))
    report = joint.grouped_pairwise_report(
        torch.tensor(separations, dtype=torch.float32),
        prompt_records,
        required_margin=required_margin,
    )
    report.update(
        {
            "scorer": ("mcf_zero_unlearn_official_eval.official_test_batch_prediction"),
            "checkpoint_dtype_forward": True,
            "training_prompt_scope": "direct_only",
            "GFS_evaluated": False,
            "GFS_checkpoint_selection": False,
        }
    )
    return report


def direct_candidate_feasible(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("FS") == 100.0
        and int(report.get("direct_margin_failures", -1)) == 0
        and report.get("utility_safe") is True
    )


def main() -> None:
    args = parse_args()
    stage1_scales, solver_margins, rank_ladder = joint.validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_records, prompt_records, manifest = load_locked_direct_records(
        Path(args.training_visible_path).resolve(),
        Path(args.split_manifest).resolve(),
        expected_seed=args.seed,
        expected_forget_num=args.forget_num,
    )

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    identity = wikipedia.model_identity(model, tok, args.model_path)
    original_output = model.get_output_embeddings()
    if original_output is None:
        raise RuntimeError("model has no output head")
    hidden_size = int(original_output.weight.shape[1])
    (
        utility_second_moment,
        utility_hidden,
        utility_logsumexp,
        utility_metadata,
    ) = learner.load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=args.utility_sample_size,
        expected_prompt_count=args.utility_prompt_count,
        expected_hidden_size=hidden_size,
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    actual_documents = int(utility_metadata["actual_document_sample_size"])
    actual_prompts = int(utility_metadata["actual_utility_prompt_count"])
    if actual_documents < int(args.utility_sample_size):
        print(
            "WARNING: Wikipedia corpus contains only "
            f"{actual_documents} sampled documents versus "
            f"{args.utility_sample_size} requested; this is a pilot utility guard"
        )
    if actual_prompts < int(args.utility_prompt_count):
        print(
            "WARNING: Wikipedia cache contains only "
            f"{actual_prompts} predictor states versus "
            f"{args.utility_prompt_count} requested"
        )

    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)
    true_cases = core.expand_sensitive_cases(
        prompt_records,
        tok,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    reference_cases = core.expand_sensitive_cases(
        prompt_records,
        tok,
        sensitive_field="target_reference",
        llama_like=llama_like,
    )
    true_ids = core.official_target_ids(
        tok, true_cases, llama_like=llama_like, device=device
    ).detach()
    reference_ids = core.official_target_ids(
        tok, reference_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids = sorted(
        set(int(value) for value in true_ids.cpu().tolist())
        | set(int(value) for value in reference_ids.cpu().tolist())
    )
    selected_tensor = torch.tensor(
        selected_ids, device=output_layer.weight.device, dtype=torch.long
    )
    base_rows = output_layer.weight.index_select(0, selected_tensor).detach().float()

    utility_probabilities = learner.selected_base_probabilities(
        output_layer,
        selected_ids,
        utility_hidden,
        utility_logsumexp,
        device=device,
        batch_size=args.utility_eval_batch_size,
    )
    (
        train_indices,
        guard_indices,
        utility_pool_report,
    ) = learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=utility_probabilities,
        selected_ids=selected_ids,
        topk_per_row=args.utility_token_topk_per_row,
        uniform_prompt_count=args.utility_uniform_prompt_count,
        split_seed=args.utility_pool_seed,
    )
    utility_train_hidden = (
        utility_hidden.index_select(0, train_indices).contiguous().to(device)
    )
    utility_train_probabilities = (
        utility_probabilities.index_select(0, train_indices).contiguous().to(device)
    )
    utility_guard_hidden = utility_hidden.index_select(0, guard_indices).contiguous()
    utility_guard_probabilities = utility_probabilities.index_select(
        0, guard_indices
    ).contiguous()
    core.write_json(output_dir / "utility_pool_report.json", utility_pool_report)

    base_true_logits = learner.cache_logits_preserving_dtype(
        model, tok, true_cases, device, args.cache_batch_size
    )
    base_reference_logits = learner.cache_logits_preserving_dtype(
        model, tok, reference_cases, device, args.cache_batch_size
    )
    true_hidden = core.forward_last_hidden(
        model, tok, true_cases, device, args.cache_batch_size
    ).float()
    reference_hidden = core.forward_last_hidden(
        model, tok, reference_cases, device, args.cache_batch_size
    ).float()
    true_positions = exact.record_positions(true_cases, device=device)
    reference_positions = exact.record_positions(reference_cases, device=device)
    prompt_count = len(prompt_records)
    true_cache = exact.build_sequence_cache(
        base_true_logits,
        true_hidden,
        true_ids,
        true_positions,
        selected_ids,
        record_count=prompt_count,
        device=device,
    )
    reference_cache = exact.build_sequence_cache(
        base_reference_logits,
        reference_hidden,
        reference_ids,
        reference_positions,
        selected_ids,
        record_count=prompt_count,
        device=device,
    )
    zero = torch.zeros(
        (len(selected_ids), hidden_size), device=device, dtype=torch.float32
    )
    base_true_nll = exact.exact_sequence_record_nll(true_cache, zero).detach()
    base_reference_nll = exact.exact_sequence_record_nll(reference_cache, zero).detach()
    masks = joint.prompt_kind_masks(prompt_records, device=device)
    base_report = direct_materialized_report(
        model,
        tok,
        training_records,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=args.required_pairwise_margin,
    )
    core.write_json(output_dir / "base_direct_FS_report.json", base_report)

    stage1_bases, stage1_basis_report = joint.build_joint_bases(
        true_hidden,
        true_ids,
        reference_hidden,
        reference_ids,
        utility_second_moment,
        requested_ids=selected_ids,
        rank_cap=args.stage1_rank,
        relative_eps=args.contrastive_eps,
        constraint_context_weight=args.constraint_context_weight,
    )
    core.write_json(output_dir / "stage1_basis_report.json", stage1_basis_report)
    trained_stage1 = joint.optimize_stage1(
        args,
        selected_ids,
        stage1_bases,
        true_cache,
        reference_cache,
        base_true_nll,
        base_reference_nll,
        masks,
        utility_train_hidden,
        utility_train_probabilities,
        output_dir,
        device=device,
    )
    stage1_delta, stage1_selected, stage1_reports = joint.choose_stage1_delta(
        args,
        trained_stage1,
        stage1_scales,
        true_cache,
        reference_cache,
        prompt_records,
        utility_guard_hidden,
        utility_guard_probabilities,
        device=device,
    )
    stage1_selected["selection_mode"] = (
        "direct_only_stage1_complete"
        if int(stage1_selected["direct_margin_failures"]) == 0
        else "direct_only_stage1_residual_handoff"
    )
    torch.save(
        {"row_ids": selected_ids, "delta": stage1_delta.detach().cpu()},
        output_dir / "stage1_delta.pt",
    )
    core.write_json(output_dir / "stage1_scale_reports.json", stage1_reports)
    core.write_json(output_dir / "stage1_selected_report.json", stage1_selected)

    with learner.temporary_materialized_output_delta(
        output_layer, selected_ids, stage1_delta
    ):
        actual_stage1_delta = learner.actual_selected_delta(
            output_layer, selected_ids, base_rows
        )
        stage1_materialized = direct_materialized_report(
            model,
            tok,
            training_records,
            prompt_records,
            device,
            llama_like=llama_like,
            required_margin=args.required_pairwise_margin,
        )
    joint.add_utility_report(
        stage1_materialized,
        actual_stage1_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        args,
        device=device,
    )
    core.write_json(output_dir / "stage1_materialized_report.json", stage1_materialized)

    final_delta: torch.Tensor | None = None
    selected_metadata: Dict[str, Any]
    if direct_candidate_feasible(stage1_materialized):
        final_delta = actual_stage1_delta
        selected_metadata = {
            "selection_mode": "direct_only_stage1_materialized_FS100",
            "stage1": stage1_materialized,
        }
    else:
        failure_positions = stage1_materialized["pairwise_margin_failure_positions"]
        if not failure_positions:
            core.write_json(
                output_dir / "infeasible.json",
                {
                    "method": METHOD,
                    "protocol": PROTOCOL,
                    "stage1": stage1_materialized,
                    "reason": (
                        "Stage 1 met every direct FS margin but failed a utility "
                        "guard; a behavioral residual cannot conceal utility failure"
                    ),
                },
            )
            raise RuntimeError("direct-only Stage 1 is complete but utility-unsafe")

        failure_set = set(int(value) for value in failure_positions)
        true_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in true_cases],
            device=device,
            dtype=torch.bool,
        )
        reference_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in reference_cases],
            device=device,
            dtype=torch.bool,
        )
        active_ids = sorted(
            set(int(value) for value in true_ids[true_failure_mask].cpu().tolist())
            | set(
                int(value)
                for value in reference_ids[reference_failure_mask].cpu().tolist()
            )
        )
        attempts: List[Dict[str, Any]] = []
        chosen: Dict[str, Any] | None = None
        for rank in rank_ladder:
            bases, basis_report = joint.build_joint_bases(
                true_hidden,
                true_ids,
                reference_hidden,
                reference_ids,
                utility_second_moment,
                requested_ids=active_ids,
                rank_cap=rank,
                relative_eps=args.contrastive_eps,
                constraint_context_weight=args.constraint_context_weight,
            )
            core.write_json(
                output_dir / f"stage2_rank{rank}_basis_report.json", basis_report
            )
            for solver_target in solver_margins:
                residual, history, solver_report = joint.solve_residual(
                    args,
                    rank=rank,
                    solver_target=solver_target,
                    row_bases=bases,
                    active_ids=active_ids,
                    selected_ids=selected_ids,
                    stage1_delta=actual_stage1_delta,
                    true_cache=true_cache,
                    reference_cache=reference_cache,
                    prompt_records=prompt_records,
                    utility_hidden=utility_train_hidden,
                    utility_probabilities=utility_train_probabilities,
                )
                solver_report["selection_mode"] = (
                    "minimum_utility_exact_direct_FS_feasible"
                    if solver_report["continuous_feasible"]
                    else "best_direct_only_infeasible_diagnostic"
                )
                tag = str(solver_target).replace(".", "p")
                core.write_json(
                    output_dir / f"stage2_rank{rank}_margin{tag}_solver_history.json",
                    history,
                )
                combined = learner.total_delta_with_residual(
                    actual_stage1_delta, selected_ids, residual, active_ids
                )
                if not bool(solver_report["continuous_feasible"]):
                    materialized = {
                        "rank": rank,
                        "solver_target": float(solver_target),
                        "continuous_feasible": False,
                        "feasible": False,
                        "materialization_skipped": True,
                        "skip_reason": (
                            "continuous candidate failed a direct, Wikipedia, "
                            "or sparse-norm hard constraint"
                        ),
                        "continuous_FS": solver_report["FS"],
                        "continuous_minimum_direct_separation": solver_report[
                            "minimum_direct_separation"
                        ],
                        "continuous_total_delta_norm": solver_report[
                            "total_delta_norm"
                        ],
                    }
                else:
                    with learner.temporary_materialized_output_delta(
                        output_layer, selected_ids, combined
                    ):
                        actual_combined = learner.actual_selected_delta(
                            output_layer, selected_ids, base_rows
                        )
                        materialized = direct_materialized_report(
                            model,
                            tok,
                            training_records,
                            prompt_records,
                            device,
                            llama_like=llama_like,
                            required_margin=args.required_pairwise_margin,
                        )
                    joint.add_utility_report(
                        materialized,
                        actual_combined,
                        utility_guard_hidden,
                        utility_guard_probabilities,
                        args,
                        device=device,
                    )
                    materialized.update(
                        {
                            "rank": rank,
                            "solver_target": float(solver_target),
                            "residual_delta_norm": float(residual.norm().cpu()),
                            "continuous_feasible": True,
                        }
                    )
                    materialized["feasible"] = direct_candidate_feasible(materialized)
                core.write_json(
                    output_dir
                    / f"stage2_rank{rank}_margin{tag}_materialized_report.json",
                    materialized,
                )
                attempt = {
                    "rank": rank,
                    "solver_target": float(solver_target),
                    "solver": solver_report,
                    "materialized": materialized,
                }
                attempts.append(attempt)
                if materialized["feasible"]:
                    chosen = {**attempt, "delta": actual_combined.detach()}
                    break
            if chosen is not None:
                break

        core.write_json(
            output_dir / "stage2_attempts.json",
            [
                {
                    "rank": row["rank"],
                    "solver_target": row["solver_target"],
                    "solver": row["solver"],
                    "materialized": row["materialized"],
                }
                for row in attempts
            ],
        )
        if chosen is None:
            core.write_json(
                output_dir / "infeasible.json",
                {
                    "method": METHOD,
                    "protocol": PROTOCOL,
                    "stage1": stage1_materialized,
                    "active_row_ids": active_ids,
                    "stage2_attempts": attempts,
                    "reason": (
                        "no checkpoint-dtype direct-only candidate achieved FS=100 "
                        "with the required margin under Wikipedia/norm guards"
                    ),
                },
            )
            raise RuntimeError(
                "target-aware direct-only v8 found no BF16-safe FS=100 checkpoint"
            )
        final_delta = chosen["delta"]
        selected_metadata = {
            "selection_mode": "exact_direct_residual_materialized_FS100",
            "rank": chosen["rank"],
            "solver_target": chosen["solver_target"],
            "solver": chosen["solver"],
            "materialized": chosen["materialized"],
        }

    if final_delta is None:
        raise AssertionError("v8 selection produced no final delta")
    core.materialize_output_delta(output_layer, selected_ids, final_delta)
    actual_final_delta = learner.actual_selected_delta(
        output_layer, selected_ids, base_rows
    )
    final_report = direct_materialized_report(
        model,
        tok,
        training_records,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=args.required_pairwise_margin,
    )
    joint.add_utility_report(
        final_report,
        actual_final_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        args,
        device=device,
    )
    if not direct_candidate_feasible(final_report):
        raise RuntimeError(
            "final materialized checkpoint failed the direct FS guarantee"
        )

    learner.save_checkpoint(model, tok, output_dir / "checkpoint")
    torch.save(
        {"row_ids": selected_ids, "delta": actual_final_delta.detach().cpu()},
        output_dir / "final_total_delta.pt",
    )
    core.write_json(output_dir / "final_direct_FS_report.json", final_report)
    architecture = {
        "method": METHOD,
        "protocol": PROTOCOL,
        "target_aware": True,
        "benchmark_neutral": False,
        "training_prompt_scope": "direct_only",
        "editable_parameters": "union_target_true_target_new_lm_head_rows_only",
        "target_true_used_for_bounded_GA": True,
        "target_new_used_for_bounded_GD": True,
        "official_paraphrases_used_for_training": False,
        "official_paraphrases_used_for_checkpoint_selection": False,
        "GFS_checkpoint_selection": False,
        "GFS_is_held_out_post_training_audit": True,
        "neighborhood_prompts_used_for_training_or_selection": False,
        "benchmark_retain_examples_used": 0,
        "ppl_text_used": False,
        "required_materialized_FS": 100.0,
        "required_materialized_GFS": None,
        "required_pairwise_margin": float(args.required_pairwise_margin),
        "stage2_solver_margins": list(solver_margins),
        "stage2_rank_ladder": list(rank_ladder),
        "utility_guard_budgets": {
            "mean": float(args.utility_kl_mean_budget),
            "p95": float(args.utility_kl_p95_budget),
            "max": float(args.utility_kl_max_budget),
            "total_delta_norm": float(args.max_total_delta_norm),
        },
    }
    core.write_json(
        output_dir / "config_used.json",
        {
            "schema_version": 1,
            **architecture,
            "architecture_sha256": joint.sha256_bytes(
                json.dumps(architecture, sort_keys=True).encode("utf-8")
            ),
            "seed": int(args.seed),
            "source_mcf_sha256": manifest.get("source_sha256"),
            "training_visible_sha256": manifest.get(
                "training_visible_target_aware_direct_sha256"
            ),
            "forget_case_ids": manifest.get("sampling", {}).get("forget_case_ids", []),
            "selected_row_ids": selected_ids,
            "selected": selected_metadata,
            "final": final_report,
            "utility_cache": str(Path(args.utility_cache).resolve()),
            "utility_cache_metadata": utility_metadata,
        },
    )
    print("Target-aware direct-only SURE v8 complete:", output_dir)
    print("Materialized direct FS:", final_report["FS"])
    print("Minimum direct separation:", final_report["minimum_direct_separation"])
    print("GFS: not evaluated or used before checkpoint save")


if __name__ == "__main__":
    main()
