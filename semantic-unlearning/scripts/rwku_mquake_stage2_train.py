#!/usr/bin/env python3
"""Protected head-only residual optimization for RWKU MQuAKE-style Stage 2."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from rwku_mquake_stage2_helpers import *  # noqa: F401,F403


def train_stage2(
    runtime: Mapping[str, Any],
    cfg: Mapping[str, Any],
    level1_anchor: Mapping[str, Any],
    final_out: Path,
) -> Optional[Dict[str, Any]]:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    device = runtime["device"]
    llama_like = runtime["llama_like"]
    cases = runtime["cases"]
    prompt_records = runtime["prompt_records"]
    sparse = runtime["sparse"]
    output_layer = runtime["output_layer"]
    residual_case_indices = runtime["residual_case_indices"]
    protected_case_indices = runtime["protected_case_indices"]
    residual_rows = runtime["residual_rows"]
    bf = runtime["bf"]
    protected_anchor_logits = runtime["protected_anchor_logits"]
    selection_contexts = runtime["selection_contexts"]
    selection_base_hidden = runtime["selection_base_hidden"]
    utility_bs = int(runtime["utility_bs"])
    l1_input_anchor = runtime["l1_input_anchor"]
    stage2 = cfg["stage2"]
    margin = float(cfg["acceptance"]["required_pairwise_margin"])

    repair = core.SelectedRowDelta(
        len(residual_rows),
        output_layer.weight.shape[1],
        direction_basis=bf,
        device=output_layer.weight.device,
    )
    repair_params = list(repair.parameters())
    optimizer = make_optimizer(repair, stage2)
    repair_hook = core.register_output_delta_hook(
        output_layer, residual_rows, repair.effective_delta
    )
    residual_sampler = core.IndexSampler(
        len(residual_case_indices),
        int(stage2["repair_batch_size"]),
        int(cfg["seed"]) + 200003,
    )
    protected_sampler = core.IndexSampler(
        len(protected_case_indices),
        int(stage2["success_kl_batch_size"]),
        int(cfg["seed"]) + 300007,
    )
    backtrack_scales = [float(x) for x in stage2["backtrack_scales"]]
    proposal_scales = [1.0] + backtrack_scales
    step_history: list[Dict[str, Any]] = []
    checkpoint_history: list[Dict[str, Any]] = []
    final_candidate: Optional[Dict[str, Any]] = None

    try:
        with (final_out / "level2_train_log.jsonl").open("w", encoding="utf-8") as log:
            for step in range(1, int(stage2["repair_steps"]) + 1):
                r_local = residual_sampler.next()
                r_global = [residual_case_indices[int(i)] for i in r_local]
                r_batch = [cases[i] for i in r_global]
                p_local = protected_sampler.next()
                p_global = [protected_case_indices[int(i)] for i in p_local]
                p_batch = [cases[i] for i in p_global]

                optimizer.zero_grad(set_to_none=True)
                r_logits = core.forward_last_logits(model, tokenizer, r_batch, device)
                r_tids = core.official_target_ids(
                    tokenizer, r_batch, llama_like=llama_like, device=device
                )
                hinge = squared_margin_hinge(r_logits, r_tids, margin)

                p_logits = core.forward_last_logits(model, tokenizer, p_batch, device)
                p_tids = core.official_target_ids(
                    tokenizer, p_batch, llama_like=llama_like, device=device
                )
                p_ref = protected_anchor_logits[p_local]
                sampled_p_kl = core.gd_non_sensitive_kl(p_logits, p_ref, p_tids)
                delta = repair.effective_delta()
                l2 = delta.square().mean()
                loss = (
                    hinge
                    + float(stage2["success_kl_weight"]) * sampled_p_kl
                    + float(stage2["repair_l2"]) * l2
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Stage-2 loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    repair_params, float(stage2["grad_clip"])
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite Stage-2 gradient at step {step}")

                before = snapshots(repair_params)
                optimizer.step()
                proposed = snapshots(repair_params)
                accepted_scale: Optional[float] = None
                accepted_p: Optional[Dict[str, Any]] = None
                accepted_r: Optional[Dict[str, Any]] = None
                terminal_atomic: Optional[Dict[str, Any]] = None
                terminal_utility: Optional[Dict[str, Any]] = None
                attempts: list[Dict[str, Any]] = []

                for scale in proposal_scales:
                    interpolate(repair_params, before, proposed, scale)
                    p_report = protection_report(
                        model,
                        tokenizer,
                        cases,
                        protected_case_indices,
                        protected_anchor_logits,
                        llama_like=llama_like,
                        device=device,
                        batch_size=int(stage2["success_kl_batch_size"]),
                        required_margin=margin,
                    )
                    p_safe = bool(
                        int(p_report["protected_prompt_regressions"])
                        <= int(stage2["hard_success_regression_limit"])
                        and float(p_report["protected_non_sensitive_kl_mean"])
                        <= float(stage2["hard_success_kl_budget"])
                    )
                    r_report = None
                    external_safe = None
                    if p_safe:
                        r_report = residual_report(
                            model,
                            tokenizer,
                            cases,
                            residual_case_indices,
                            llama_like=llama_like,
                            device=device,
                            batch_size=int(stage2["repair_batch_size"]),
                            required_margin=margin,
                        )
                        if int(r_report["remaining_margin_failure_count"]) == 0:
                            utility = v2.exact_external_kl_report(
                                model,
                                tokenizer,
                                selection_contexts,
                                selection_base_hidden,
                                device=device,
                                batch_size=utility_bs,
                            )
                            external_safe = v2.utility_safe(utility, cfg)
                            if external_safe:
                                atomic = head.materialized_atomic_report(
                                    model,
                                    tokenizer,
                                    prompt_records,
                                    device,
                                    llama_like=llama_like,
                                    required_margin=margin,
                                )
                                if v2.atomic_safe(atomic, cfg):
                                    accepted_scale = scale
                                    accepted_p = p_report
                                    accepted_r = r_report
                                    terminal_atomic = atomic
                                    terminal_utility = utility
                                    break
                            attempts.append(
                                {
                                    "scale": scale,
                                    "P_safe": True,
                                    "P_regressions": int(
                                        p_report["protected_prompt_regressions"]
                                    ),
                                    "P_KL": float(
                                        p_report["protected_non_sensitive_kl_mean"]
                                    ),
                                    "F_remaining": 0,
                                    "external_safe_if_complete": bool(external_safe),
                                }
                            )
                            continue
                    attempts.append(
                        {
                            "scale": scale,
                            "P_safe": p_safe,
                            "P_regressions": int(
                                p_report["protected_prompt_regressions"]
                            ),
                            "P_KL": float(
                                p_report["protected_non_sensitive_kl_mean"]
                            ),
                            "F_remaining": (
                                None
                                if r_report is None
                                else int(r_report["remaining_margin_failure_count"])
                            ),
                            "external_safe_if_complete": external_safe,
                        }
                    )
                    if p_safe:
                        accepted_scale = scale
                        accepted_p = p_report
                        accepted_r = r_report
                        break

                rejected = accepted_scale is None
                if rejected:
                    restore(repair_params, before)
                # Adam state corresponds to the full scale=1 proposal. Preserve it
                # only when that exact proposal is accepted. A backtracked or
                # rolled-back step gets a fresh optimizer so forbidden momentum
                # cannot leak into later proposals.
                if rejected or accepted_scale != 1.0:
                    del optimizer
                    optimizer = make_optimizer(repair, stage2)

                if not torch.equal(sparse.input_delta.detach(), l1_input_anchor):
                    raise RuntimeError("Stage 2 changed frozen Level-1 embeddings")

                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "squared_margin_hinge": float(hinge.detach().cpu()),
                    "sampled_success_kl": float(sampled_p_kl.detach().cpu()),
                    "repair_l2": float(l2.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "accepted_scale": accepted_scale,
                    "proposal_rejected": rejected,
                    "repair_delta_norm": float(
                        repair.effective_delta().detach().norm().cpu()
                    ),
                    "protected_prompt_regressions": (
                        None
                        if accepted_p is None
                        else int(accepted_p["protected_prompt_regressions"])
                    ),
                    "protected_kl": (
                        None
                        if accepted_p is None
                        else float(accepted_p["protected_non_sensitive_kl_mean"])
                    ),
                    "remaining_margin_failures": (
                        None
                        if accepted_r is None
                        else int(accepted_r["remaining_margin_failure_count"])
                    ),
                    "attempted_backtrack_scales": attempts,
                    "stage2_embedding_changed": False,
                    "official_rwku_records_accessed": False,
                }
                step_history.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()

                if (
                    step == 1
                    or step % int(stage2["checkpoint_interval"]) == 0
                    or terminal_atomic is not None
                    or step == int(stage2["repair_steps"])
                ):
                    if terminal_atomic is not None and terminal_utility is not None:
                        atomic = terminal_atomic
                        utility = terminal_utility
                    else:
                        atomic = head.materialized_atomic_report(
                            model,
                            tokenizer,
                            prompt_records,
                            device,
                            llama_like=llama_like,
                            required_margin=margin,
                        )
                        utility = v2.exact_external_kl_report(
                            model,
                            tokenizer,
                            selection_contexts,
                            selection_base_hidden,
                            device=device,
                            batch_size=utility_bs,
                        )
                    p_full = protection_report(
                        model,
                        tokenizer,
                        cases,
                        protected_case_indices,
                        protected_anchor_logits,
                        llama_like=llama_like,
                        device=device,
                        batch_size=int(stage2["success_kl_batch_size"]),
                        required_margin=margin,
                    )
                    p_safe = bool(
                        int(p_full["protected_prompt_regressions"])
                        <= int(stage2["hard_success_regression_limit"])
                        and float(p_full["protected_non_sensitive_kl_mean"])
                        <= float(stage2["hard_success_kl_budget"])
                    )
                    a_safe = v2.atomic_safe(atomic, cfg)
                    u_safe = v2.utility_safe(utility, cfg)
                    eligible = bool(a_safe and u_safe and p_safe)
                    checkpoint = {
                        "step": step,
                        "atomic": atomic,
                        "selection_utility": utility,
                        "hard_success_protection": p_full,
                        "atomic_safe": a_safe,
                        "selection_utility_safe": u_safe,
                        "hard_success_safe": p_safe,
                        "eligible": eligible,
                        "repair_delta_norm": float(
                            repair.effective_delta().detach().norm().cpu()
                        ),
                        "accepted_scale": accepted_scale,
                        "official_rwku_records_accessed": False,
                    }
                    checkpoint_history.append(checkpoint)
                    print(
                        "  L2 checkpoint {}: direct={} other={} margin_fail={} P_reg={} P_KL={:.6f} Wiki={:.6f}/{:.6f}/{:.6f} eligible={}".format(
                            step,
                            atomic.get("FS"),
                            atomic.get("generated_subject_FS"),
                            len(base2.residual_prompt_positions(atomic)),
                            p_full["protected_prompt_regressions"],
                            p_full["protected_non_sensitive_kl_mean"],
                            utility["utility_kl_mean"],
                            utility["utility_kl_p95"],
                            utility["utility_kl_max"],
                            eligible,
                        )
                    )
                    if eligible:
                        combined = combine_output(
                            level1_anchor["output_delta"],
                            sparse.selected_output_rows,
                            residual_rows,
                            repair.effective_delta().detach(),
                        )
                        norms = sparse.delta_norms(
                            level1_anchor["input_delta"], combined
                        )
                        final_candidate = {
                            "source": "mquake_style_stage2_protected_residual_repair",
                            "step": step,
                            "input_delta": level1_anchor["input_delta"]
                            .detach()
                            .cpu()
                            .clone(),
                            "output_delta": combined,
                            "atomic": dict(atomic),
                            "selection_utility": dict(utility),
                            "delta_norms": norms,
                            "repair_delta_norm": float(
                                repair.effective_delta().detach().norm().cpu()
                            ),
                        }
                        break
    finally:
        repair_hook.remove()
        del optimizer

    core.write_json(final_out / "level2_step_history.json", step_history)
    core.write_json(final_out / "level2_checkpoint_history.json", checkpoint_history)
    return final_candidate
