#!/usr/bin/env python3
"""MCF SURE Stage 2: protected-subspace repair with official-direct MCF margins.

Input is a frozen checkpoint from mcf_sure_protected_subspace_stage1.py.
Stage 2 changes only target_true-sensitive LM-head rows associated with residual
direct MCF records. Input embeddings and transformer parameters stay frozen.

Residual partition (training-visible direct records only):

    m_i = NLL_i(target_true) - NLL_i(target_new)
    P = {i: m_i >= --mcf-margin}
    F = {i: m_i <  --mcf-margin}

Geometry is built from Stage-1 target_true teacher-forced hidden states whose
parent records are in P or F:

    B_P = rowspace(H_P), capped by --protected-rank
    R_F = H_F - Proj_BP(H_F)
    B_F = rowspace(R_F), capped by --repair-rank

Only target_true LM-head rows appearing in F are editable:

    Delta W_AF = C_F B_F

The repair objective is the squared sequence-level MCF margin hinge on F plus
full-vocabulary KL(Stage1 || proposal) on target_true atomic prediction states
belonging to P, plus tiny L2.

Every optimizer proposal is hard-gated:
  * every original P record must keep sequence margin >=
    --protected-mcf-margin-floor (default 0.0);
  * mean full-vocabulary KL on protected atomic prediction states must stay
    <= --protected-kl-max (default 0.05).

Violations backtrack through 1, .5, .25, ..., .015625; otherwise rollback.
Atomic max-other-vs-target_true suppression is retained only as a diagnostic.
No official paraphrases, neighborhoods, benchmark-retain rows, or PPL text are
opened before the checkpoint is frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as shared_stage2
import mcf_sure_directional_emb_lm_stage1 as directional_v1
import mcf_sure_protected_subspace_stage1 as stage1v2


METHOD = "SURE-MCF-protected-subspace-sequence-margin-LM-head-stage2"
PROTOCOL = "mcf_target_true_protected_subspace_stage2_mcf_margin_v3"
BACKTRACK_SCALES = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Stage-1 checkpoint")
    p.add_argument("--stage1-config-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument(
        "--mcf-margin",
        type=float,
        default=0.05,
        help="Required NLL(target_true)-NLL(target_new) direct-record margin.",
    )
    p.add_argument(
        "--protected-mcf-margin-floor",
        type=float,
        default=0.0,
        help="Hard floor retained for Stage-1 successful MCF direct records.",
    )
    p.add_argument(
        "--atomic-margin",
        type=float,
        default=0.05,
        help="Diagnostic only: max-other minus target_true logit margin.",
    )
    p.add_argument("--protected-rank", type=int, default=32)
    p.add_argument("--repair-rank", type=int, default=4)
    p.add_argument("--protected-kl-weight", type=float, default=1.0)
    p.add_argument("--protected-kl-max", type=float, default=0.05)
    p.add_argument("--delta-l2", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(
        a.forget_num,
        a.repair_steps,
        a.protected_rank,
        a.repair_rank,
        a.check_every,
        a.cache_batch_size,
    ) <= 0:
        p.error("counts, ranks, check interval, and cache batch size must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(
        a.mcf_margin,
        a.protected_mcf_margin_floor,
        a.atomic_margin,
        a.protected_kl_weight,
        a.protected_kl_max,
        a.delta_l2,
        a.grad_clip,
    ) < 0:
        p.error("margins/KL/L2/clip values must be non-negative")
    if a.protected_mcf_margin_floor > a.mcf_margin:
        p.error("protected-mcf-margin-floor cannot exceed mcf-margin")
    return a


def load_and_validate_protocol(
    visible_path: Path,
    manifest_path: Path,
    stage1_config_path: Path,
    seed: int,
    forget_num: int,
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    records, manifest = directional_v1.validate_locked(
        visible_path, manifest_path, seed, forget_num
    )
    config = json.loads(stage1_config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != stage1v2.PROTOCOL:
        raise RuntimeError(
            f"Stage 2 requires {stage1v2.PROTOCOL}, got {config.get('protocol')!r}"
        )
    if int(config.get("seed", -1)) != int(seed):
        raise RuntimeError("Stage-1 config seed mismatch")
    if int(config.get("forget_num", -1)) != int(forget_num):
        raise RuntimeError("Stage-1 config forget count mismatch")
    contract = config.get("target_contract", {})
    if contract.get("sensitive_answer") != "requested_rewrite.target_true":
        raise RuntimeError("Stage-1 config is not target_true-sensitive")
    if contract.get("non_sensitive_reference") != "requested_rewrite.target_new":
        raise RuntimeError("Stage-1 config does not use target_new as reference")
    if contract.get("field_swapping") is not False:
        raise RuntimeError("Stage-1 config unexpectedly swaps target fields")
    if config.get("training_visible_sha256") != stage1v2.sha256_file(visible_path):
        raise RuntimeError("training-visible file differs from Stage-1 provenance")
    if config.get("split_manifest_sha256") != stage1v2.sha256_file(manifest_path):
        raise RuntimeError("split manifest differs from Stage-1 provenance")
    return records, manifest, config


@torch.no_grad()
def cache_atomic_state(
    model,
    tok,
    cases: Sequence[core.SensitivePredictionCase],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = core.forward_last_hidden(model, tok, cases, device, int(batch_size)).float()
    logits_chunks: List[torch.Tensor] = []
    id_chunks: List[torch.Tensor] = []
    for start in range(0, len(cases), int(batch_size)):
        batch = cases[start : start + int(batch_size)]
        logits_chunks.append(
            core.forward_last_logits(model, tok, batch, device).detach().float()
        )
        id_chunks.append(
            core.official_target_ids(
                tok, batch, llama_like=llama_like, device=device
            ).detach()
        )
    return hidden, torch.cat(logits_chunks, dim=0), torch.cat(id_chunks, dim=0)


def logits_with_sparse_delta(
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    selected_ids: Sequence[int],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if not selected_ids:
        return base_logits
    ids = torch.tensor(
        [int(x) for x in selected_ids],
        dtype=torch.long,
        device=base_logits.device,
    )
    correction = hidden.float() @ delta_rows.float().transpose(0, 1)
    return torch.index_add(base_logits.float(), 1, ids, correction)


def record_margins_from_caches(caches, delta_rows: torch.Tensor) -> torch.Tensor:
    """Official-direct positive forgetting margin: NLL(true)-NLL(new)."""
    return shared_stage2.mcf_margins_from_delta_caches(
        caches,
        delta_rows,
        sensitive_field="target_true",
        reference_field="target_new",
    )


def record_partition(
    margins: torch.Tensor,
    required_margin: float,
) -> Tuple[List[int], List[int]]:
    values = margins.detach().float().cpu().tolist()
    protected = [
        i for i, value in enumerate(values)
        if float(value) >= float(required_margin)
    ]
    failures = [
        i for i, value in enumerate(values)
        if float(value) < float(required_margin)
    ]
    return protected, failures


def atomic_positions_for_records(
    cases: Sequence[core.SensitivePredictionCase],
    record_positions: Sequence[int],
) -> List[int]:
    wanted = {int(x) for x in record_positions}
    return [
        i for i, case in enumerate(cases)
        if int(case.record_position) in wanted
    ]


def protected_kl(
    *,
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    protected_atomic_positions: Sequence[int],
    selected_ids: Sequence[int],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if not protected_atomic_positions:
        return delta_rows.sum() * 0.0
    idx = torch.tensor(
        [int(x) for x in protected_atomic_positions],
        dtype=torch.long,
        device=base_logits.device,
    )
    current = logits_with_sparse_delta(
        base_logits.index_select(0, idx),
        hidden.index_select(0, idx),
        selected_ids,
        delta_rows,
    )
    return stage1v2.full_vocab_kl(base_logits.index_select(0, idx), current)


def hard_protection_metrics(
    *,
    all_record_caches,
    protected_record_positions: Sequence[int],
    protected_atomic_positions: Sequence[int],
    base_logits: torch.Tensor,
    hidden: torch.Tensor,
    selected_ids: Sequence[int],
    delta_rows: torch.Tensor,
    mcf_margin_floor: float,
) -> Dict[str, Any]:
    if protected_record_positions:
        protected_caches = [
            all_record_caches[int(i)] for i in protected_record_positions
        ]
        margins = record_margins_from_caches(protected_caches, delta_rows)
        regressions = int((margins < float(mcf_margin_floor)).sum().item())
        min_margin = float(margins.min().detach().cpu())
    else:
        regressions = 0
        min_margin = None
    kl = protected_kl(
        base_logits=base_logits,
        hidden=hidden,
        protected_atomic_positions=protected_atomic_positions,
        selected_ids=selected_ids,
        delta_rows=delta_rows,
    )
    return {
        "protected_record_count": len(protected_record_positions),
        "protected_atomic_case_count": len(protected_atomic_positions),
        "protected_mcf_regressions": regressions,
        "protected_min_mcf_margin": min_margin,
        "protected_kl": float(kl.detach().cpu()),
    }


def parameter_snapshot(module: torch.nn.Module) -> List[torch.Tensor]:
    return [p.detach().clone() for p in module.parameters()]


@torch.no_grad()
def set_interpolated_parameters(
    module: torch.nn.Module,
    old: Sequence[torch.Tensor],
    proposed: Sequence[torch.Tensor],
    alpha: float,
) -> None:
    params = list(module.parameters())
    if len(params) != len(old) or len(params) != len(proposed):
        raise ValueError("parameter snapshot size mismatch")
    for p, before, after in zip(params, old, proposed):
        p.copy_(before + float(alpha) * (after - before))


def hash_frozen_parameters(model, output_layer) -> str:
    excluded = {id(p) for p in output_layer.parameters()}
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if id(parameter) in excluded:
            continue
        h.update(name.encode("utf-8"))
        t = parameter.detach().contiguous().view(torch.uint8).cpu()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    stage1_config_path = Path(a.stage1_config_path).resolve()
    records, manifest, stage1_config = load_and_validate_protocol(
        visible_path,
        manifest_path,
        stage1_config_path,
        int(a.seed),
        int(a.forget_num),
    )

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    sensitive_cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    hidden, atomic_base_logits, target_ids = cache_atomic_state(
        model,
        tok,
        sensitive_cases,
        llama_like=llama_like,
        device=device,
        batch_size=int(a.cache_batch_size),
    )

    instances = shared_stage2.mcf_instances(records)
    baseline_record_margins = shared_stage2.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        int(a.cache_batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    ).detach().float()
    protected_records, failure_records = record_partition(
        baseline_record_margins, float(a.mcf_margin)
    )

    protected_atomic_positions = atomic_positions_for_records(
        sensitive_cases, protected_records
    )
    failure_atomic_positions = atomic_positions_for_records(
        sensitive_cases, failure_records
    )

    if protected_atomic_positions:
        p_atomic_idx = torch.tensor(
            protected_atomic_positions, dtype=torch.long, device=device
        )
        h_p = hidden.index_select(0, p_atomic_idx)
        b_p = core.orthonormal_row_basis(
            h_p.float(), max_rank=int(a.protected_rank)
        )
    else:
        h_p = hidden.new_empty((0, hidden.shape[1]))
        b_p = hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)

    if failure_atomic_positions:
        f_atomic_idx = torch.tensor(
            failure_atomic_positions, dtype=torch.long, device=device
        )
        h_f = hidden.index_select(0, f_atomic_idx)
        r_f = stage1v2.project_away(h_f, b_p)
        b_f = core.orthonormal_row_basis(
            r_f.float(), max_rank=int(a.repair_rank)
        )
        if b_f.ndim != 2 or b_f.shape[0] <= 0:
            raise RuntimeError(
                "failure residual subspace has zero rank after protected projection"
            )
        selected_ids = sorted(
            set(
                int(x)
                for x in target_ids.index_select(
                    0, f_atomic_idx
                ).detach().cpu().tolist()
            )
            - set(int(x) for x in gagd.special_token_ids(tok))
        )
    else:
        h_f = hidden.new_empty((0, hidden.shape[1]))
        r_f = h_f
        b_f = hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
        selected_ids = []

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    frozen_hash_before = hash_frozen_parameters(model, output_layer)
    logs: List[Dict[str, Any]] = []
    rollback_count = 0
    accepted_backtracks: Dict[str, int] = {}
    best_all_record_failures = len(failure_records)

    if selected_ids:
        all_record_caches = mcf_repair.build_prompt_instance_delta_caches(
            model,
            tok,
            instances,
            selected_ids,
            device,
            int(a.cache_batch_size),
            llama_like,
        )
        failure_caches = [
            all_record_caches[int(i)] for i in failure_records
        ]

        delta_module = core.SelectedRowDelta(
            len(selected_ids),
            int(output_layer.weight.shape[1]),
            direction_basis=b_f,
            device=output_layer.weight.device,
        )
        opt = torch.optim.AdamW(
            delta_module.parameters(),
            lr=float(a.repair_lr),
            weight_decay=0.0,
        )
        params = list(delta_module.parameters())

        for step in range(1, int(a.repair_steps) + 1):
            opt.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()

            margins_f = record_margins_from_caches(failure_caches, delta)
            hinge = F.relu(float(a.mcf_margin) - margins_f).square().mean()
            kl_p = protected_kl(
                base_logits=atomic_base_logits,
                hidden=hidden,
                protected_atomic_positions=protected_atomic_positions,
                selected_ids=selected_ids,
                delta_rows=delta,
            )
            l2 = delta.square().mean()
            loss = (
                hinge
                + float(a.protected_kl_weight) * kl_p
                + float(a.delta_l2) * l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")

            old_params = parameter_snapshot(delta_module)
            opt_state_before = copy.deepcopy(opt.state_dict())
            loss.backward()
            if float(a.grad_clip) > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    params, float(a.grad_clip)
                )
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite Stage-2 gradient at step {step}"
                    )
            else:
                grad_norm = None
            opt.step()
            proposed_params = parameter_snapshot(delta_module)

            accepted_alpha = None
            accepted_protection: Dict[str, Any] | None = None
            for alpha in BACKTRACK_SCALES:
                set_interpolated_parameters(
                    delta_module,
                    old_params,
                    proposed_params,
                    float(alpha),
                )
                protection = hard_protection_metrics(
                    all_record_caches=all_record_caches,
                    protected_record_positions=protected_records,
                    protected_atomic_positions=protected_atomic_positions,
                    base_logits=atomic_base_logits,
                    hidden=hidden,
                    selected_ids=selected_ids,
                    delta_rows=delta_module.effective_delta(),
                    mcf_margin_floor=float(a.protected_mcf_margin_floor),
                )
                if (
                    int(protection["protected_mcf_regressions"]) == 0
                    and float(protection["protected_kl"])
                    <= float(a.protected_kl_max)
                ):
                    accepted_alpha = float(alpha)
                    accepted_protection = protection
                    break

            if accepted_alpha is None:
                set_interpolated_parameters(
                    delta_module, old_params, old_params, 1.0
                )
                opt.load_state_dict(opt_state_before)
                rollback_count += 1
                accepted_alpha = 0.0
                accepted_protection = hard_protection_metrics(
                    all_record_caches=all_record_caches,
                    protected_record_positions=protected_records,
                    protected_atomic_positions=protected_atomic_positions,
                    base_logits=atomic_base_logits,
                    hidden=hidden,
                    selected_ids=selected_ids,
                    delta_rows=delta_module.effective_delta(),
                    mcf_margin_floor=float(a.protected_mcf_margin_floor),
                )
            else:
                key = f"{accepted_alpha:g}"
                accepted_backtracks[key] = int(accepted_backtracks.get(key, 0)) + 1

            with torch.no_grad():
                delta_now = delta_module.effective_delta()
                all_record_margins = record_margins_from_caches(
                    all_record_caches, delta_now
                )
                all_record_failures = int(
                    (all_record_margins < float(a.mcf_margin)).sum().item()
                )
                best_all_record_failures = min(
                    best_all_record_failures, all_record_failures
                )
                atomic_current = logits_with_sparse_delta(
                    atomic_base_logits,
                    hidden,
                    selected_ids,
                    delta_now,
                )
                atomic_margins = stage1v2.atomic_margins(
                    atomic_current, target_ids
                )
                atomic_failures = int(
                    (atomic_margins < float(a.atomic_margin)).sum().item()
                )
                delta_norm = float(delta_now.norm().detach().cpu())

            if (
                step == 1
                or step % int(a.check_every) == 0
                or all_record_failures == 0
                or step == int(a.repair_steps)
            ):
                row = {
                    "step": int(step),
                    "loss_before_proposal": float(loss.detach().cpu()),
                    "mcf_failure_hinge_before_proposal": float(hinge.detach().cpu()),
                    "protected_kl_soft_before_proposal": float(kl_p.detach().cpu()),
                    "delta_l2_before_proposal": float(l2.detach().cpu()),
                    "accepted_backtrack_alpha": float(accepted_alpha),
                    "protected_mcf_regressions_after": int(
                        accepted_protection["protected_mcf_regressions"]
                    ),
                    "protected_min_mcf_margin_after": accepted_protection[
                        "protected_min_mcf_margin"
                    ],
                    "protected_kl_after": float(
                        accepted_protection["protected_kl"]
                    ),
                    "all_mcf_record_failures_after": int(all_record_failures),
                    "all_mcf_record_min_margin_after": float(
                        all_record_margins.min().detach().cpu()
                    ),
                    "atomic_failures_diagnostic_after": int(atomic_failures),
                    "atomic_min_margin_diagnostic_after": float(
                        atomic_margins.min().detach().cpu()
                    ),
                    "delta_norm_after": delta_norm,
                    "lora_used": False,
                }
                if grad_norm is not None:
                    row["grad_norm"] = float(grad_norm.detach().cpu())
                logs.append(row)
                print(json.dumps(row))

            if all_record_failures == 0:
                break

        final_delta = delta_module.effective_delta().detach().clone()
        del opt
        core.materialize_output_delta(output_layer, selected_ids, final_delta)
    else:
        all_record_caches = []
        final_delta = torch.empty(
            (0, int(output_layer.weight.shape[1])),
            dtype=torch.float32,
            device=output_layer.weight.device,
        )

    frozen_hash_after = hash_frozen_parameters(model, output_layer)
    frozen_bit_exact = frozen_hash_before == frozen_hash_after
    if not frozen_bit_exact:
        raise RuntimeError("Stage 2 changed embeddings and/or transformer parameters")

    final_record_margins = shared_stage2.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        int(a.cache_batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    ).detach().float().cpu()
    final_record_failures = [
        i
        for i, value in enumerate(final_record_margins.tolist())
        if float(value) < float(a.mcf_margin)
    ]

    final_atomic, _ = stage1v2.evaluate_atomic_cases(
        model,
        tok,
        sensitive_cases,
        llama_like=llama_like,
        device=device,
        batch_size=int(a.cache_batch_size),
    )
    final_atomic_failures = [
        i
        for i, value in enumerate(final_atomic.tolist())
        if float(value) < float(a.atomic_margin)
    ]

    if selected_ids:
        final_protection = hard_protection_metrics(
            all_record_caches=all_record_caches,
            protected_record_positions=protected_records,
            protected_atomic_positions=protected_atomic_positions,
            base_logits=atomic_base_logits,
            hidden=hidden,
            selected_ids=selected_ids,
            delta_rows=final_delta,
            mcf_margin_floor=float(a.protected_mcf_margin_floor),
        )
    else:
        final_protection = {
            "protected_record_count": len(protected_records),
            "protected_atomic_case_count": len(protected_atomic_positions),
            "protected_mcf_regressions": 0,
            "protected_min_mcf_margin": (
                float(
                    baseline_record_margins[
                        torch.tensor(
                            protected_records,
                            dtype=torch.long,
                            device=baseline_record_margins.device,
                        )
                    ].min().detach().cpu()
                )
                if protected_records
                else None
            ),
            "protected_kl": 0.0,
        }

    final_gate = {
        "all_mcf_direct_records_pass": len(final_record_failures) == 0,
        "protected_mcf_regressions_zero": int(
            final_protection["protected_mcf_regressions"]
        ) == 0,
        "protected_kl_within_limit": float(final_protection["protected_kl"])
        <= float(a.protected_kl_max),
        "embeddings_and_transformer_bit_exact": bool(frozen_bit_exact),
    }
    final_gate["passed"] = all(bool(x) for x in final_gate.values())

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    protected_overlap = (
        float((b_f @ b_p.transpose(0, 1)).abs().max().detach().cpu())
        if b_f.numel() and b_p.numel()
        else 0.0
    )
    summary: Dict[str, Any] = {
        "schema_version": 3,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_stage1_protocol": stage1_config.get("protocol"),
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "primary_gate": "NLL(target_true)-NLL(target_new)",
        "mcf_margin_required": float(a.mcf_margin),
        "protected_mcf_margin_floor": float(a.protected_mcf_margin_floor),
        "stage1_mcf_success_count_P": len(protected_records),
        "stage1_mcf_failure_count_F": len(failure_records),
        "stage1_success_record_positions_P": protected_records,
        "stage1_failure_record_positions_F": failure_records,
        "stage1_mcf_min_margin": float(
            baseline_record_margins.min().detach().cpu()
        ),
        "protected_atomic_positions_for_geometry_and_kl": protected_atomic_positions,
        "failure_atomic_positions_for_geometry": failure_atomic_positions,
        "protected_rank_requested": int(a.protected_rank),
        "protected_rank_actual": int(b_p.shape[0]),
        "repair_rank_requested": int(a.repair_rank),
        "repair_rank_actual": int(b_f.shape[0]),
        "repair_basis_definition": "B_F=rowspace(H_F-Proj_BP(H_F))",
        "repair_basis_max_abs_overlap_with_protected_basis": protected_overlap,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "Delta W_AF = C_F B_F",
        "repair_objective": "MCF sequence-margin hinge(F) + KL(P) + L2",
        "atomic_gate_role": "diagnostic_only",
        "atomic_margin_diagnostic_required": float(a.atomic_margin),
        "input_embeddings_modified_in_stage2": False,
        "transformer_trainable_parameters": 0,
        "lora_used": False,
        "protected_kl_weight": float(a.protected_kl_weight),
        "protected_kl_max": float(a.protected_kl_max),
        "delta_l2": float(a.delta_l2),
        "repair_steps_max": int(a.repair_steps),
        "repair_lr": float(a.repair_lr),
        "backtrack_scales": list(BACKTRACK_SCALES),
        "accepted_backtrack_histogram": accepted_backtracks,
        "rollback_count": int(rollback_count),
        "logs": logs,
        "best_cached_mcf_record_failure_count": int(best_all_record_failures),
        "effective_delta_norm": (
            float(final_delta.norm().detach().cpu()) if final_delta.numel() else 0.0
        ),
        "final_mcf_record_failure_count": len(final_record_failures),
        "final_mcf_record_failure_positions": final_record_failures,
        "final_mcf_record_min_margin": float(final_record_margins.min().item()),
        "final_mcf_record_success_count_margin_ge_0": int(
            (final_record_margins >= 0.0).sum().item()
        ),
        "final_atomic_failure_count_diagnostic": len(final_atomic_failures),
        "final_atomic_min_margin_diagnostic": float(final_atomic.min().item()),
        "final_protected_metrics": final_protection,
        "frozen_parameter_sha256_before": frozen_hash_before,
        "frozen_parameter_sha256_after": frozen_hash_after,
        "embeddings_and_transformer_bit_exact": bool(frozen_bit_exact),
        "final_gate": final_gate,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "training_visible_sha256": stage1v2.sha256_file(visible_path),
        "split_manifest_sha256": stage1v2.sha256_file(manifest_path),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        "Protected MCF-margin Stage 2: direct record failures "
        f"{len(failure_records)} -> {len(final_record_failures)}; "
        f"P regressions={final_protection['protected_mcf_regressions']}; "
        f"KL(P)={final_protection['protected_kl']:.6f}; "
        f"gate={final_gate['passed']}"
    )
    print(f"Final checkpoint: {ckpt}")

    if not final_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
