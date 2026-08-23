#!/usr/bin/env python3
"""MQuAKE Stage 2: hard-protected LM-head-only directional residual repair.

Input is the untied Stage-1 checkpoint produced by
mquake_sure_stage1_directional.py.

The Stage-1 checkpoint is gated on every training-visible teacher-forced
target_true PredictionCase. If all cases pass, Stage 2 is an identity.

Otherwise:
  * F = exactly the Stage-1 failed atomic PredictionCases.
  * P = exactly the Stage-1 successful atomic PredictionCases.
  * A_F = sensitive target vocabulary rows appearing in F.
  * H_F/H_P are frozen Stage-1 final hidden states.
  * B_P is the protected hidden subspace from H_P.
  * B_F = SVD(H_F - Proj_{B_P}(H_F)).
  * ONLY LM-head rows A_F can change:
        Delta W_A_F = C_F B_F
  * the input embedding and transformer remain bit-exact.
  * F is repaired with a bounded margin hinge, not unbounded GA.
  * P is a hard constraint: every accepted optimizer step must preserve every
    Stage-1 success and remain under an exact full-vocabulary Base||Edited KL
    budget. Invalid proposals are backtracked or rolled back.

Because Stage 2 changes only selected LM-head rows, hidden states are constant.
The hard gate and exact full-vocabulary KL can therefore be computed exactly
from cached Stage-1 logits plus the selected-row logit corrections, without
rerunning the transformer during optimization.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--protection-batch-size", type=int, default=16)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--repair-rank", type=int, default=4)
    p.add_argument("--protected-rank", type=int, default=32)
    p.add_argument("--repair-weight", type=float, default=1.0)
    p.add_argument("--protection-weight", type=float, default=10.0)
    p.add_argument("--l2-weight", type=float, default=1e-6)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--backtrack-scales",
        default="0.5,0.25,0.125,0.0625,0.03125,0.015625",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-frozen-hash", action="store_true")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    vp = Path(a.training_visible_path).resolve()
    mp = Path(a.split_manifest).resolve()
    records = json.loads(vp.read_text(encoding="utf-8"))
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != a.forget_num:
        raise RuntimeError("training-visible forget count mismatch")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != a.forget_num:
        raise RuntimeError("manifest forget count mismatch")
    expected = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual = [int(r.get("case_id", -1)) for r in records]
    if expected and actual != expected:
        raise RuntimeError("training-visible IDs do not match manifest")
    for i, record in enumerate(records):
        rr = record.get("requested_rewrite", {})
        if not isinstance(rr, dict) or not rr.get("target_true", {}).get("str"):
            raise RuntimeError(f"record {i} lacks target_true")
        if "target_new" in rr:
            raise RuntimeError(f"record {i} leaks target_new")
        if record.get("atomic_gen_prompt") or record.get("multihop_questions"):
            raise RuntimeError(f"record {i} leaks evaluation-only MQuAKE fields")
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise RuntimeError(f"record {i} leaks held-out probes")
    return records, manifest


def parse_scales(text: str) -> List[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values or any(not (0.0 < x < 1.0) for x in values):
        raise ValueError("backtrack scales must be in (0,1)")
    return values


def hash_frozen(model, output_weight) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if id(parameter) == id(output_weight):
            continue
        tensor = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def gate_from_logits(
    logits: torch.Tensor, target_ids: torch.Tensor, required_margin: float
) -> Dict[str, Any]:
    x = logits.float()
    tids = target_ids.to(device=x.device, dtype=torch.long)
    rows = torch.arange(x.shape[0], device=x.device)
    target = x[rows, tids]
    other = x.clone()
    other[rows, tids] = -torch.inf
    margins = other.max(dim=-1).values - target
    failed = torch.nonzero(margins < float(required_margin), as_tuple=False).flatten()
    vals = margins.detach().cpu()
    return {
        "total": int(x.shape[0]),
        "passed": int(x.shape[0] - failed.numel()),
        "failed": int(failed.numel()),
        "residual_indices": [int(x) for x in failed.detach().cpu().tolist()],
        "minimum_margin": float(vals.min().item()) if vals.numel() else None,
        "mean_margin": float(vals.mean().item()) if vals.numel() else None,
        "required_margin": float(required_margin),
    }


class SelectedRowGeometry:
    """Exact logits/gates/KL when only a fixed set of output rows can move."""

    def __init__(
        self,
        *,
        base_logits: torch.Tensor,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        selected_row_ids: Sequence[int],
        device: torch.device,
    ):
        self.device = device
        self.base_logits = base_logits.float().cpu()
        self.hidden = hidden.float().to(device)
        self.target_ids_cpu = target_ids.detach().long().cpu()
        self.row_ids = torch.tensor(
            [int(x) for x in selected_row_ids], dtype=torch.long, device=device
        )
        self.row_ids_cpu = self.row_ids.cpu()
        self.base_selected = self.base_logits.index_select(1, self.row_ids_cpu).to(device)
        rows_cpu = torch.arange(self.base_logits.shape[0], dtype=torch.long)
        self.base_target = self.base_logits[
            rows_cpu, self.target_ids_cpu
        ].to(device)
        logz = torch.logsumexp(self.base_logits, dim=-1)
        self.base_logp_selected = (
            self.base_logits.index_select(1, self.row_ids_cpu) - logz[:, None]
        ).to(device)
        self.base_p_selected = self.base_logp_selected.exp()

        target_position = torch.full(
            (self.base_logits.shape[0],), -1, dtype=torch.long
        )
        row_to_position = {
            int(token_id): j for j, token_id in enumerate(self.row_ids_cpu.tolist())
        }
        for i, tid in enumerate(self.target_ids_cpu.tolist()):
            if int(tid) in row_to_position:
                target_position[i] = int(row_to_position[int(tid)])
        self.target_selected_position = target_position.to(device)

        fixed = self.base_logits.clone()
        fixed[:, self.row_ids_cpu] = -torch.inf
        fixed[rows_cpu, self.target_ids_cpu] = -torch.inf
        self.fixed_max_other = fixed.max(dim=-1).values.to(device)
        del fixed

    def metrics(self, indices: Sequence[int], delta_rows: torch.Tensor):
        idx = torch.tensor([int(x) for x in indices], dtype=torch.long, device=self.device)
        h = self.hidden.index_select(0, idx)
        correction = h @ delta_rows.float().transpose(0, 1)
        selected = self.base_selected.index_select(0, idx) + correction
        target = self.base_target.index_select(0, idx).clone()
        target_pos = self.target_selected_position.index_select(0, idx)
        has_target = target_pos >= 0
        if bool(has_target.any()):
            rr = torch.nonzero(has_target, as_tuple=False).flatten()
            target[rr] = selected[rr, target_pos[rr]]

        changed_other = selected.clone()
        if bool(has_target.any()):
            rr = torch.nonzero(has_target, as_tuple=False).flatten()
            changed_other[rr, target_pos[rr]] = -torch.inf
        changed_max = changed_other.max(dim=-1).values
        fixed_max = self.fixed_max_other.index_select(0, idx)
        margins = torch.maximum(fixed_max, changed_max) - target

        base_logp_s = self.base_logp_selected.index_select(0, idx)
        base_p_s = self.base_p_selected.index_select(0, idx)
        p_out = (1.0 - base_p_s.sum(dim=-1)).clamp_min(1e-30)
        log_terms = torch.cat(
            [p_out.log()[:, None], base_logp_s + correction], dim=-1
        )
        log_r = torch.logsumexp(log_terms, dim=-1)
        kl = (log_r - (base_p_s * correction).sum(dim=-1)).clamp_min(0.0)
        return margins, kl

    def report(
        self,
        indices: Sequence[int],
        delta_rows: torch.Tensor,
        required_margin: float,
    ) -> Dict[str, Any]:
        if not indices:
            return {
                "count": 0,
                "failed": 0,
                "minimum_margin": None,
                "mean_margin": None,
                "kl_mean": 0.0,
                "kl_max": 0.0,
            }
        with torch.no_grad():
            margins, kl = self.metrics(indices, delta_rows)
            failed = int((margins < float(required_margin)).sum().item())
            return {
                "count": len(indices),
                "failed": failed,
                "minimum_margin": float(margins.min().item()),
                "mean_margin": float(margins.mean().item()),
                "kl_mean": float(kl.mean().item()),
                "kl_max": float(kl.max().item()),
            }


def capture_state(module: torch.nn.Module):
    return [p.detach().clone() for p in module.parameters()]


@torch.no_grad()
def restore_state(module: torch.nn.Module, state) -> None:
    for parameter, value in zip(module.parameters(), state):
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def interpolate_state(module: torch.nn.Module, old_state, proposed_state, scale: float) -> None:
    for parameter, old, new in zip(module.parameters(), old_state, proposed_state):
        value = old + float(scale) * (new - old)
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def actual_full_kl_mean(
    base_logits: torch.Tensor, final_logits: torch.Tensor, indices: Sequence[int], batch_size: int
) -> float:
    if not indices:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, len(indices), batch_size):
        ids = [int(x) for x in indices[start : start + batch_size]]
        base = base_logits[ids].float()
        cur = final_logits[ids].float()
        base_logp = F.log_softmax(base, dim=-1)
        cur_logp = F.log_softmax(cur, dim=-1)
        kl = (base_logp.exp() * (base_logp - cur_logp)).sum(dim=-1)
        total += float(kl.sum().item())
        count += int(kl.numel())
    return total / max(count, 1)


def main() -> None:
    a = parse_args()
    if min(
        a.repair_steps,
        a.batch_size,
        a.protection_batch_size,
        a.cache_batch_size,
        a.repair_rank,
        a.protected_rank,
    ) <= 0:
        raise ValueError("steps, batches, and ranks must be positive")
    if (
        a.repair_lr <= 0
        or a.repair_weight <= 0
        or a.protection_weight < 0
        or a.l2_weight < 0
        or a.max_protected_kl < 0
    ):
        raise ValueError("invalid Stage2 optimization settings")
    backtrack_scales = parse_scales(a.backtrack_scales)

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = load_locked(a)
    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("no generated atomic target_true PredictionCases")

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Stage2 requires input embedding and LM head to be untied")
    model.eval()

    frozen_before = None if a.skip_frozen_hash else hash_frozen(model, output_layer.weight)
    base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )
    target_ids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    level1_gate = gate_from_logits(base_logits, target_ids.cpu(), a.constraint_margin)
    F_indices = [int(x) for x in level1_gate["residual_indices"]]
    failed_set = set(F_indices)
    P_indices = [i for i in range(len(cases)) if i not in failed_set]

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    report: Dict[str, Any] = {
        "schema_version": 3,
        "method": "MQuAKE Stage2 Hard-Protected LM-Head Directional Residual Repair",
        "source_protocol": manifest.get("protocol"),
        "level1_gate": level1_gate,
        "generated_atomic_prompt_semantics": "all teacher-forced target_true token decisions from training-visible direct facts",
        "official_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "embedding_frozen_in_stage2": True,
        "transformer_frozen_in_stage2": True,
        "output_head_only_stage2": True,
        "hard_protection": {
            "stage1_success_regressions_allowed": 0,
            "exact_full_vocabulary_base_to_edited_kl_budget": float(a.max_protected_kl),
            "invalid_optimizer_proposals": "backtrack then rollback",
        },
    }

    if not F_indices:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        frozen_after = None if a.skip_frozen_hash else hash_frozen(model, output_layer.weight)
        report["level2"] = {
            "skipped": True,
            "reason": "Level1 passed all generated atomic prompts",
            "F": 0,
            "P": len(P_indices),
            "A_F": [],
        }
        report["final_gate"] = level1_gate
        report["stage1_successes_regressed"] = 0
        report["protected_kl"] = 0.0
        report["max_protected_kl"] = float(a.max_protected_kl)
        report["frozen_non_head_exact"] = frozen_before == frozen_after
        report["final_gates_pass"] = bool(
            level1_gate["failed"] == 0 and frozen_before == frozen_after
        )
        report["checkpoint"] = str(ckpt.resolve())
        core.write_json(out / "two_stage_summary.json", report)
        print(f"Level1 gate: {level1_gate['passed']}/{level1_gate['total']} pass; F=0; P={len(P_indices)}")
        print("Level2 identity: Level1 already satisfies every atomic constraint")
        print("Final gates pass:", report["final_gates_pass"])
        print("Checkpoint:", ckpt)
        return

    hidden = core.forward_last_hidden(
        model, tok, cases, device, batch_size=a.cache_batch_size
    ).float()
    H_F = hidden[F_indices]
    H_P = hidden[P_indices] if P_indices else hidden.new_empty((0, hidden.shape[1]))
    protected_basis = (
        core.orthonormal_row_basis(H_P, max_rank=a.protected_rank).to(
            device=device, dtype=torch.float32
        )
        if H_P.numel()
        else hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
    )
    if protected_basis.numel():
        residual = H_F - (H_F @ protected_basis.transpose(0, 1)) @ protected_basis
    else:
        residual = H_F
    repair_basis = core.orthonormal_row_basis(
        residual, max_rank=a.repair_rank
    ).to(device=device, dtype=torch.float32)
    if repair_basis.ndim != 2 or repair_basis.shape[0] == 0:
        raise RuntimeError("residual repair hidden basis is empty")

    special = set(gagd.special_token_ids(tok))
    F_tids = target_ids[F_indices]
    A_F = sorted(set(int(x) for x in F_tids.detach().cpu().tolist()) - special)
    if not A_F:
        raise RuntimeError("A_F is empty for non-empty F")

    delta = core.SelectedRowDelta(
        len(A_F),
        int(output_layer.weight.shape[1]),
        direction_basis=repair_basis,
        device=device,
    )
    geometry = SelectedRowGeometry(
        base_logits=base_logits,
        hidden=hidden,
        target_ids=target_ids,
        selected_row_ids=A_F,
        device=device,
    )
    parameters = list(delta.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=a.repair_lr, weight_decay=0.0)
    forget_sampler = core.IndexSampler(len(F_indices), a.batch_size, a.seed + 100003)
    protection_sampler = (
        core.IndexSampler(len(P_indices), a.protection_batch_size, a.seed + 200003)
        if P_indices
        else None
    )

    best_state = capture_state(delta)
    initial_p = geometry.report(
        P_indices, delta.effective_delta(), a.constraint_margin
    )
    initial_f = geometry.report(
        F_indices, delta.effective_delta(), a.constraint_margin
    )
    best_key = (
        int(initial_f["failed"]),
        -float(initial_f["minimum_margin"]),
        float(initial_p["kl_mean"]),
        float(delta.effective_delta().detach().norm().cpu()),
    )
    best_step = 0
    logs = []
    accepted_steps = 0
    rolled_back_steps = 0

    log_path = out / "repair_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in range(1, a.repair_steps + 1):
            local = forget_sampler.next()
            f_ids = [F_indices[i] for i in local]
            optimizer.zero_grad(set_to_none=True)

            delta_rows = delta.effective_delta()
            f_margins, _ = geometry.metrics(f_ids, delta_rows)
            repair_loss = F.relu(
                float(a.constraint_margin) - f_margins
            ).square().mean()

            if protection_sampler is not None:
                p_local = protection_sampler.next()
                p_ids = [P_indices[i] for i in p_local]
                _, p_kl = geometry.metrics(p_ids, delta_rows)
                protection_loss = p_kl.mean()
            else:
                protection_loss = repair_loss.new_zeros(())

            l2 = delta_rows.square().mean()
            loss = (
                float(a.repair_weight) * repair_loss
                + float(a.protection_weight) * protection_loss
                + float(a.l2_weight) * l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss at step {step}")

            old_state = capture_state(delta)
            old_opt = copy.deepcopy(optimizer.state_dict())
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(parameters, a.grad_clip)
                if a.grad_clip > 0
                else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage2 gradient at step {step}")
            optimizer.step()
            proposed_state = capture_state(delta)

            accepted_scale = 1.0
            p_report = geometry.report(
                P_indices, delta.effective_delta(), a.constraint_margin
            )
            feasible = (
                int(p_report["failed"]) == 0
                and float(p_report["kl_mean"]) <= float(a.max_protected_kl)
            )
            if not feasible:
                accepted_scale = 0.0
                for scale in backtrack_scales:
                    interpolate_state(delta, old_state, proposed_state, scale)
                    trial_p = geometry.report(
                        P_indices, delta.effective_delta(), a.constraint_margin
                    )
                    if (
                        int(trial_p["failed"]) == 0
                        and float(trial_p["kl_mean"]) <= float(a.max_protected_kl)
                    ):
                        accepted_scale = float(scale)
                        p_report = trial_p
                        feasible = True
                        break

            if not feasible:
                restore_state(delta, old_state)
                optimizer.load_state_dict(old_opt)
                accepted_scale = 0.0
                rolled_back_steps += 1
                p_report = geometry.report(
                    P_indices, delta.effective_delta(), a.constraint_margin
                )
            else:
                accepted_steps += 1

            f_report = geometry.report(
                F_indices, delta.effective_delta(), a.constraint_margin
            )
            current_norm = float(delta.effective_delta().detach().float().norm().cpu())
            key = (
                int(f_report["failed"]),
                -float(f_report["minimum_margin"]),
                float(p_report["kl_mean"]),
                current_norm,
            )
            if key < best_key:
                best_key = key
                best_state = capture_state(delta)
                best_step = step

            row = {
                "step": step,
                "loss_before_guard": float(loss.detach().cpu()),
                "repair_hinge": float(repair_loss.detach().cpu()),
                "sampled_protection_exact_kl": float(protection_loss.detach().cpu()),
                "delta_l2_mean": float(l2.detach().cpu()),
                "gradient_norm_before_clip": (
                    None if grad_norm is None else float(grad_norm.detach().cpu())
                ),
                "accepted_scale": accepted_scale,
                "F_failed": int(f_report["failed"]),
                "F_minimum_margin": float(f_report["minimum_margin"]),
                "P_regressions": int(p_report["failed"]),
                "P_exact_kl_mean": float(p_report["kl_mean"]),
                "P_exact_kl_max": float(p_report["kl_max"]),
                "delta_norm": current_norm,
                "best_step": int(best_step),
            }
            logs.append(row)
            if step == 1 or step % 25 == 0 or step == a.repair_steps or f_report["failed"] == 0:
                print(
                    "Stage2 step {step}: F_fail={ff} P_reg={pr} KL={kl:.6g} "
                    "scale={scale:g} ||dW||={norm:.6g}".format(
                        step=step,
                        ff=f_report["failed"],
                        pr=p_report["failed"],
                        kl=p_report["kl_mean"],
                        scale=accepted_scale,
                        norm=current_norm,
                    )
                )
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

            if int(f_report["failed"]) == 0:
                break

    restore_state(delta, best_state)
    best_delta = delta.effective_delta().detach()
    best_p_report = geometry.report(P_indices, best_delta, a.constraint_margin)
    best_f_report = geometry.report(F_indices, best_delta, a.constraint_margin)

    core.materialize_output_delta(output_layer, A_F, best_delta)
    model.eval()
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    final_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )
    final_gate = gate_from_logits(final_logits, target_ids.cpu(), a.constraint_margin)
    final_failed_set = set(final_gate["residual_indices"])
    stage1_successes_regressed = sum(1 for i in P_indices if i in final_failed_set)
    protected_kl = actual_full_kl_mean(
        base_logits, final_logits, P_indices, a.cache_batch_size
    )
    frozen_after = None if a.skip_frozen_hash else hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    final_pass = bool(
        final_gate["failed"] == 0
        and stage1_successes_regressed == 0
        and protected_kl <= float(a.max_protected_kl)
        and frozen_exact
    )

    report["level2"] = {
        "skipped": False,
        "F": len(F_indices),
        "F_indices": F_indices,
        "P": len(P_indices),
        "P_indices": P_indices,
        "A_F": A_F,
        "A_F_count": len(A_F),
        "protected_basis_rank": int(protected_basis.shape[0]),
        "requested_protected_rank": int(a.protected_rank),
        "repair_basis_rank": int(repair_basis.shape[0]),
        "requested_repair_rank": int(a.repair_rank),
        "residual_hidden_energy_fraction": float(
            (residual.square().sum() / H_F.square().sum().clamp_min(1e-12))
            .detach()
            .cpu()
        ),
        "parameterization": "Delta W_A_F = C_F B_F; embedding and transformer frozen",
        "repair_objective": "squared hinge on required forget margin",
        "protection_objective": "exact full-vocabulary KL(Stage1 || Stage2) on P",
        "repair_weight": float(a.repair_weight),
        "protection_weight": float(a.protection_weight),
        "l2_weight": float(a.l2_weight),
        "repair_lr": float(a.repair_lr),
        "repair_steps_requested": int(a.repair_steps),
        "accepted_steps": int(accepted_steps),
        "rolled_back_steps": int(rolled_back_steps),
        "best_checkpoint_step": int(best_step),
        "best_selection_key": list(best_key),
        "best_pre_materialization_F": best_f_report,
        "best_pre_materialization_P": best_p_report,
        "materialized_delta_norm": float(best_delta.float().norm().cpu()),
        "logs": logs,
    }
    report["final_gate"] = final_gate
    report["stage1_successes_regressed"] = int(stage1_successes_regressed)
    report["protected_kl"] = float(protected_kl)
    report["max_protected_kl"] = float(a.max_protected_kl)
    report["frozen_non_head_exact"] = bool(frozen_exact)
    report["final_gates_pass"] = final_pass
    report["checkpoint"] = str(ckpt.resolve())
    core.write_json(out / "two_stage_summary.json", report)
    torch.save(
        {
            "repair_basis": repair_basis.detach().cpu(),
            "protected_basis": protected_basis.detach().cpu(),
            "A_F": A_F,
            "best_delta": best_delta.float().cpu(),
        },
        out / "directional_repair_artifacts.pt",
    )

    print(
        f"Level1 gate: {level1_gate['passed']}/{level1_gate['total']} pass; "
        f"F={len(F_indices)}; P={len(P_indices)}"
    )
    print(
        f"Level2: A_F={len(A_F)}, repair_rank={repair_basis.shape[0]}, "
        f"best_step={best_step}"
    )
    print(
        f"Final gate: {final_gate['passed']}/{final_gate['total']} pass; "
        f"protected_KL={protected_kl:.6g}; "
        f"Stage1 regressions={stage1_successes_regressed}; "
        f"frozen_non_head_exact={frozen_exact}"
    )
    print("Final gates pass:", final_pass)
    print("Checkpoint:", ckpt)


if __name__ == "__main__":
    main()
