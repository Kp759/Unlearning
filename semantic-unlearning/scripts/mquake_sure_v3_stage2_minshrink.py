#!/usr/bin/env python3
"""Direct-only minimum-magnitude shrink of an already-trained MQuAKE SURE v3 Stage2 edit.

This is NOT a new training objective.  It starts from the materialized v3
Stage1 and final Stage2 checkpoints and asks how much of the Stage2 LM-head
increment is actually necessary to preserve the same locked 284/284 direct
forgetting constraints.

Let
    D = W_final - W_stage1
on the rows changed by Stage2.  We construct two pre-declared variants using
ONLY the locked training-visible direct cases:

1. Global minimum shrink
       W = W_stage1 + alpha D
   where alpha is the smallest direct-feasible scalar found on a dense grid.

2. Row-wise minimum shrink
       W_j = W_stage1,j + alpha_j D_j
   where each changed vocabulary row is greedily reduced on a dense grid while
   all 284 direct constraints remain satisfied.  Multiple deterministic passes
   are allowed; rows are visited from largest to smallest Stage2 row norm.

AtomicGen, benchmark retain, target_new, paraphrases, and multihop questions are
never read.  Any official evaluation must occur only after these checkpoints
have been selected and materialized.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage2_head_directional as v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-model-path", required=True)
    p.add_argument("--final-model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--global-grid-points", type=int, default=2001)
    p.add_argument("--row-grid-points", type=int, default=101)
    p.add_argument("--row-passes", type=int, default=2)
    p.add_argument("--actual-recovery-points", type=int, default=33)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(path: str, a: argparse.Namespace):
    ns = argparse.Namespace(
        model_path=path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.eval()
    model.config.use_cache = False
    return model, tok


def stats(values: Sequence[float]) -> Dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def gate_from_margins(margins: torch.Tensor, required: float) -> Dict[str, Any]:
    m = margins.detach().float().cpu()
    bad = torch.nonzero(m < float(required), as_tuple=False).flatten()
    return {
        "total": int(m.numel()),
        "passed": int(m.numel() - bad.numel()),
        "failed": int(bad.numel()),
        "residual_indices": [int(x) for x in bad.tolist()],
        "minimum_margin": float(m.min().item()) if m.numel() else None,
        "mean_margin": float(m.mean().item()) if m.numel() else None,
        "required_margin": float(required),
    }


class CompressedDirectGate:
    """Exact-at-endpoints direct gate for a fixed set of changed LM-head rows.

    stage1_logits and final_logits are actual materialized checkpoint outputs.
    Between endpoints, changed-row logits are linearly interpolated.  Candidate
    checkpoints are subsequently checked through the actual BF16 LM head before
    saving, so interpolation is only a fast search device, never the final gate.
    """

    def __init__(
        self,
        stage1_logits: torch.Tensor,
        final_logits: torch.Tensor,
        target_ids: torch.Tensor,
        changed_ids: Sequence[int],
        device: torch.device,
    ) -> None:
        self.device = device
        self.changed_ids_list = [int(x) for x in changed_ids]
        self.changed_ids = torch.tensor(self.changed_ids_list, dtype=torch.long, device=device)
        s1 = stage1_logits.float().to(device)
        fin = final_logits.float().to(device)
        self.n = int(s1.shape[0])
        self.k = len(self.changed_ids_list)
        self.targets = target_ids.long().to(device)
        rr = torch.arange(self.n, device=device)
        self.base_target = s1[rr, self.targets]
        self.base_selected = s1.index_select(1, self.changed_ids)
        self.contrib = fin.index_select(1, self.changed_ids) - self.base_selected

        lookup = {tid: j for j, tid in enumerate(self.changed_ids_list)}
        self.target_pos = torch.tensor(
            [lookup.get(int(t), -1) for t in self.targets.detach().cpu().tolist()],
            dtype=torch.long,
            device=device,
        )
        self.target_is_selected = self.target_pos.ge(0)

        # Maximum over rows that never change, excluding the target itself when
        # the target is one of those unchanged rows.
        tmp = s1.clone()
        tmp.index_fill_(1, self.changed_ids, -torch.inf)
        unselected_target = ~self.target_is_selected
        if bool(unselected_target.any()):
            idx = torch.nonzero(unselected_target, as_tuple=False).flatten()
            tmp[idx, self.targets[idx]] = -torch.inf
        self.unedited_other = tmp.max(dim=1).values
        del tmp, s1, fin

    def margins(self, alphas: torch.Tensor) -> torch.Tensor:
        a = alphas.to(device=self.device, dtype=torch.float32)
        selected = self.base_selected + self.contrib * a[None, :]
        target = self.base_target.clone()
        mask = self.target_is_selected
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            target[idx] = selected[idx, self.target_pos[idx]]
        other_selected = selected.clone()
        if bool(mask.any()):
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            other_selected[idx, self.target_pos[idx]] = -torch.inf
        other = torch.maximum(self.unedited_other, other_selected.max(dim=1).values)
        return other - target

    def report(self, alphas: torch.Tensor, required: float) -> Dict[str, Any]:
        return gate_from_margins(self.margins(alphas), required)

    def global_search(self, points: int, required: float) -> Tuple[float, Dict[str, Any]]:
        if points < 2:
            raise ValueError("global-grid-points must be >=2")
        best = None
        best_report = None
        # Ascending scan is intentionally exhaustive over the declared grid; no
        # monotonicity assumption is required.
        for alpha in torch.linspace(0.0, 1.0, points, device=self.device).tolist():
            a = torch.full((self.k,), float(alpha), device=self.device)
            report = self.report(a, required)
            if int(report["failed"]) == 0:
                best = float(alpha)
                best_report = report
                break
        if best is None:
            raise RuntimeError("Even alpha=1 is not direct-feasible in compressed gate")
        return best, best_report

    def rowwise_search(
        self,
        row_norms: torch.Tensor,
        *,
        points: int,
        passes: int,
        required: float,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if points < 2 or passes <= 0:
            raise ValueError("row-grid-points>=2 and row-passes>0 required")
        alphas = torch.ones(self.k, device=self.device, dtype=torch.float32)
        order = torch.argsort(row_norms.to(self.device), descending=True).tolist()
        logs: List[Dict[str, Any]] = []

        for pass_index in range(passes):
            changed_this_pass = 0
            for j in order:
                current_alpha = float(alphas[j].item())
                if current_alpha <= 0.0:
                    continue

                # Current selected logits under all current row scales.
                selected = self.base_selected + self.contrib * alphas[None, :]
                target_current = self.base_target.clone()
                mask = self.target_is_selected
                if bool(mask.any()):
                    idx = torch.nonzero(mask, as_tuple=False).flatten()
                    target_current[idx] = selected[idx, self.target_pos[idx]]

                # Best competitor excluding both the target row and row j.
                fixed = selected.clone()
                if bool(mask.any()):
                    idx = torch.nonzero(mask, as_tuple=False).flatten()
                    fixed[idx, self.target_pos[idx]] = -torch.inf
                fixed[:, j] = -torch.inf
                other_fixed = torch.maximum(self.unedited_other, fixed.max(dim=1).values)

                grid = torch.linspace(0.0, current_alpha, points, device=self.device)
                cand_j = self.base_selected[:, j][None, :] + grid[:, None] * self.contrib[:, j][None, :]

                target_grid = target_current[None, :].expand(points, -1).clone()
                target_j_mask = self.target_pos.eq(j)
                if bool(target_j_mask.any()):
                    idxj = torch.nonzero(target_j_mask, as_tuple=False).flatten()
                    target_grid[:, idxj] = cand_j[:, idxj]

                # Row j may be a competitor except on cases where it is the target.
                cand_other = cand_j.clone()
                if bool(target_j_mask.any()):
                    cand_other[:, target_j_mask] = -torch.inf
                other_grid = torch.maximum(other_fixed[None, :], cand_other)
                margins = other_grid - target_grid
                feasible = margins.min(dim=1).values.ge(float(required))
                feasible_idx = torch.nonzero(feasible, as_tuple=False).flatten()
                if feasible_idx.numel() == 0:
                    # Numerical search disagreement; preserve current row.
                    chosen = current_alpha
                else:
                    chosen = float(grid[int(feasible_idx[0].item())].item())
                if chosen + 1e-12 < current_alpha:
                    alphas[j] = chosen
                    changed_this_pass += 1

            report = self.report(alphas, required)
            logs.append({
                "pass": pass_index + 1,
                "rows_reduced": changed_this_pass,
                "alpha": stats(alphas.detach().cpu().tolist()),
                "zero_rows": int(alphas.eq(0).sum().item()),
                "direct_gate": report,
            })
            print(
                "Rowwise pass {p}: reduced={r} alpha_mean={a:.6g} zeros={z} gate={g}/{t} min_margin={m:.6g}".format(
                    p=pass_index + 1,
                    r=changed_this_pass,
                    a=float(alphas.mean().item()),
                    z=int(alphas.eq(0).sum().item()),
                    g=report["passed"],
                    t=report["total"],
                    m=report["minimum_margin"],
                )
            )
            if changed_this_pass == 0:
                break
        return alphas, logs


def set_candidate_rows(output_layer, row_ids: Sequence[int], stage1_rows: torch.Tensor, delta_rows: torch.Tensor, alphas: torch.Tensor) -> None:
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=output_layer.weight.device)
    a = alphas.detach().float().cpu()[:, None]
    candidate = stage1_rows.float() + a * delta_rows.float()
    candidate = candidate.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype)
    with torch.no_grad():
        output_layer.weight.index_copy_(0, ids, candidate)


@torch.no_grad()
def actual_head_gate(output_layer, hidden: torch.Tensor, target_ids: torch.Tensor, required: float, batch_size: int = 32) -> Dict[str, Any]:
    device = output_layer.weight.device
    dtype = output_layer.weight.dtype
    margins: List[torch.Tensor] = []
    for start in range(0, hidden.shape[0], batch_size):
        h = hidden[start : start + batch_size].to(device=device, dtype=dtype)
        logits = output_layer(h).float()
        tids = target_ids[start : start + h.shape[0]].to(device=device, dtype=torch.long)
        rr = torch.arange(h.shape[0], device=device)
        target = logits[rr, tids]
        logits[rr, tids] = -torch.inf
        margins.append((logits.max(dim=1).values - target).cpu())
    return gate_from_margins(torch.cat(margins), required)


def recover_actual_feasibility(
    output_layer,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    row_ids: Sequence[int],
    stage1_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    proposed: torch.Tensor,
    *,
    required: float,
    recovery_points: int,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """If BF16 materialization flips a boundary case, interpolate toward alpha=1."""
    ones = torch.ones_like(proposed)
    if recovery_points < 2:
        recovery_points = 2
    previous_t = 0.0
    previous_failed = True
    first_pass_t = None
    first_pass_report = None
    first_pass_alpha = None

    for t in torch.linspace(0.0, 1.0, recovery_points).tolist():
        candidate = proposed + float(t) * (ones - proposed)
        set_candidate_rows(output_layer, row_ids, stage1_rows, delta_rows, candidate)
        report = actual_head_gate(output_layer, hidden, target_ids, required)
        if int(report["failed"]) == 0:
            first_pass_t = float(t)
            first_pass_report = report
            first_pass_alpha = candidate
            break
        previous_t = float(t)
        previous_failed = True

    if first_pass_t is None or first_pass_alpha is None or first_pass_report is None:
        raise RuntimeError("alpha=1 unexpectedly fails the actual direct gate")

    # Locally refine only inside the first observed fail->pass interval.
    lo = previous_t if previous_failed else 0.0
    hi = first_pass_t
    best_alpha = first_pass_alpha
    best_report = first_pass_report
    for _ in range(12):
        if hi - lo < 1e-5:
            break
        mid = 0.5 * (lo + hi)
        candidate = proposed + mid * (ones - proposed)
        set_candidate_rows(output_layer, row_ids, stage1_rows, delta_rows, candidate)
        report = actual_head_gate(output_layer, hidden, target_ids, required)
        if int(report["failed"]) == 0:
            hi = mid
            best_alpha = candidate
            best_report = report
        else:
            lo = mid

    set_candidate_rows(output_layer, row_ids, stage1_rows, delta_rows, best_alpha)
    final_report = actual_head_gate(output_layer, hidden, target_ids, required)
    return best_alpha.detach().cpu(), final_report, float(hi)


def save_variant(model, tok, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def main() -> None:
    a = parse_args()
    if a.constraint_margin < 0:
        raise ValueError("constraint margin must be nonnegative")
    if min(a.cache_batch_size, a.global_grid_points, a.row_grid_points, a.row_passes) <= 0:
        raise ValueError("invalid search settings")
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    # The same strict direct-only validator used by v3 Stage2.
    records, manifest = v2.load_locked(a)
    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("===== V3 MIN-SHRINK: LOAD STAGE1 =====")
    stage1_model, tok = load_model(a.stage1_model_path, a)
    device = gagd.first_device(stage1_model)
    llama_like = is_llama_like(stage1_model, tok)
    cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device).detach().cpu()
    stage1_logits = core.cache_base_logits(stage1_model, tok, cases, device, batch_size=a.cache_batch_size).float().cpu()
    hidden = core.forward_last_hidden(stage1_model, tok, cases, device, batch_size=a.cache_batch_size).detach().cpu()
    stage1_gate = v2.gate_from_logits(stage1_logits, target_ids, a.constraint_margin)
    stage1_head = stage1_model.get_output_embeddings().weight.detach().cpu().clone()
    del stage1_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("===== V3 MIN-SHRINK: LOAD FINAL =====")
    final_model, final_tok = load_model(a.final_model_path, a)
    final_device = gagd.first_device(final_model)
    final_logits = core.cache_base_logits(final_model, final_tok, cases, final_device, batch_size=a.cache_batch_size).float().cpu()
    final_gate = v2.gate_from_logits(final_logits, target_ids, a.constraint_margin)
    if int(final_gate["failed"]) != 0:
        raise RuntimeError(f"Source final v3 checkpoint is not direct-feasible: {final_gate}")
    final_head = final_model.get_output_embeddings().weight.detach().cpu()

    changed = (final_head != stage1_head).any(dim=1)
    changed_ids = torch.nonzero(changed, as_tuple=False).flatten().tolist()
    if not changed_ids:
        raise RuntimeError("No materialized Stage2 head rows differ from Stage1")
    ids_cpu = torch.tensor(changed_ids, dtype=torch.long)
    stage1_rows = stage1_head.index_select(0, ids_cpu).clone()
    final_rows = final_head.index_select(0, ids_cpu).clone()
    delta_rows = final_rows.float() - stage1_rows.float()
    row_norms = delta_rows.norm(dim=1)
    source_delta_norm = float(delta_rows.norm().item())
    print(f"Stage2 materialized changed rows: {len(changed_ids)}; ||D||={source_delta_norm:.9g}")
    print("Stage1 direct gate:", stage1_gate)
    print("Final direct gate:", final_gate)

    del stage1_head, final_head
    gc.collect()

    geometry = CompressedDirectGate(
        stage1_logits,
        final_logits,
        target_ids,
        changed_ids,
        device=final_device,
    )
    ones = torch.ones(len(changed_ids), device=final_device)
    compressed_final = geometry.report(ones, a.constraint_margin)
    print("Compressed alpha=1 gate:", compressed_final)

    # GLOBAL ---------------------------------------------------------------
    print("===== GLOBAL MINIMUM ALPHA (DIRECT ONLY) =====")
    global_alpha_scalar, global_search_gate = geometry.global_search(
        a.global_grid_points, a.constraint_margin
    )
    proposed_global = torch.full((len(changed_ids),), global_alpha_scalar, dtype=torch.float32)
    print(f"Compressed minimum global alpha={global_alpha_scalar:.9g}; gate={global_search_gate}")

    output_layer = final_model.get_output_embeddings()
    global_alpha, global_actual_gate, global_recovery_t = recover_actual_feasibility(
        output_layer,
        hidden,
        target_ids,
        changed_ids,
        stage1_rows,
        delta_rows,
        proposed_global,
        required=a.constraint_margin,
        recovery_points=a.actual_recovery_points,
    )
    global_effective_scalar = float(global_alpha.mean().item())
    global_norm = float((delta_rows * global_alpha[:, None]).norm().item())
    global_ckpt = out / "global_min" / "checkpoint"
    save_variant(final_model, final_tok, global_ckpt)
    print(
        f"GLOBAL materialized: alpha={global_effective_scalar:.9g} gate={global_actual_gate['passed']}/{global_actual_gate['total']} "
        f"min_margin={global_actual_gate['minimum_margin']:.9g} ||alpha D||={global_norm:.9g}"
    )

    # Restore source final rows before the independent rowwise search.
    set_candidate_rows(output_layer, changed_ids, stage1_rows, delta_rows, torch.ones(len(changed_ids)))

    # ROW-WISE -------------------------------------------------------------
    print("===== ROW-WISE MINIMUM ALPHAS (DIRECT ONLY) =====")
    rowwise_proposed_gpu, row_logs = geometry.rowwise_search(
        row_norms,
        points=a.row_grid_points,
        passes=a.row_passes,
        required=a.constraint_margin,
    )
    rowwise_search_gate = geometry.report(rowwise_proposed_gpu, a.constraint_margin)
    rowwise_alpha, rowwise_actual_gate, rowwise_recovery_t = recover_actual_feasibility(
        output_layer,
        hidden,
        target_ids,
        changed_ids,
        stage1_rows,
        delta_rows,
        rowwise_proposed_gpu.detach().cpu(),
        required=a.constraint_margin,
        recovery_points=a.actual_recovery_points,
    )
    rowwise_norm = float((delta_rows * rowwise_alpha[:, None]).norm().item())
    rowwise_ckpt = out / "rowwise_min" / "checkpoint"
    save_variant(final_model, final_tok, rowwise_ckpt)

    alpha_values = rowwise_alpha.tolist()
    print(
        "ROWWISE materialized: gate={}/{} min_margin={:.9g} ||alpha_j D_j||={:.9g} alpha_mean={:.9g} zeros={}".format(
            rowwise_actual_gate["passed"],
            rowwise_actual_gate["total"],
            rowwise_actual_gate["minimum_margin"],
            rowwise_norm,
            float(rowwise_alpha.mean().item()),
            int(rowwise_alpha.eq(0).sum().item()),
        )
    )

    per_row = [
        {
            "token_id": int(tid),
            "stage2_row_norm": float(norm),
            "alpha": float(alpha),
            "final_row_norm_after_shrink": float(norm) * float(alpha),
        }
        for tid, norm, alpha in zip(changed_ids, row_norms.tolist(), alpha_values)
    ]
    per_row.sort(key=lambda x: (-x["stage2_row_norm"], x["token_id"]))

    report = {
        "schema_version": 1,
        "method": "MQuAKE SURE v3 direct-only Stage2 minimum-magnitude shrink",
        "selection_data": "locked training-visible direct target_true PredictionCases only",
        "heldout_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "seed": int(a.seed),
        "source_protocol": manifest.get("protocol"),
        "prediction_case_count": len(cases),
        "constraint_margin": float(a.constraint_margin),
        "stage1_model_path": str(Path(a.stage1_model_path).resolve()),
        "source_final_model_path": str(Path(a.final_model_path).resolve()),
        "stage1_gate": stage1_gate,
        "source_final_gate": final_gate,
        "stage2_changed_row_count": len(changed_ids),
        "stage2_changed_token_ids": changed_ids,
        "source_stage2_delta_norm": source_delta_norm,
        "global": {
            "grid_points": int(a.global_grid_points),
            "compressed_min_alpha": float(global_alpha_scalar),
            "compressed_gate": global_search_gate,
            "materialization_recovery_fraction_toward_one": float(global_recovery_t),
            "materialized_alpha": global_effective_scalar,
            "materialized_gate": global_actual_gate,
            "stage2_delta_norm_after_shrink": global_norm,
            "norm_fraction_of_source": global_norm / max(source_delta_norm, 1e-12),
            "checkpoint": str(global_ckpt.resolve()),
        },
        "rowwise": {
            "grid_points": int(a.row_grid_points),
            "passes_requested": int(a.row_passes),
            "pass_logs": row_logs,
            "compressed_gate": rowwise_search_gate,
            "materialization_recovery_fraction_toward_one": float(rowwise_recovery_t),
            "materialized_gate": rowwise_actual_gate,
            "alpha_stats": stats(alpha_values),
            "zero_alpha_rows": int(rowwise_alpha.eq(0).sum().item()),
            "stage2_delta_norm_after_shrink": rowwise_norm,
            "norm_fraction_of_source": rowwise_norm / max(source_delta_norm, 1e-12),
            "checkpoint": str(rowwise_ckpt.resolve()),
            "per_row": per_row,
        },
    }
    core.write_json(out / "minshrink_summary.json", report)

    print("===== V3 MIN-SHRINK DIGEST =====")
    print(f"Source Stage2 ||D||: {source_delta_norm:.9g}")
    print(
        "Global: alpha={:.9g}, norm_fraction={:.6f}, gate={}/{}, min_margin={:.9g}".format(
            global_effective_scalar,
            report["global"]["norm_fraction_of_source"],
            global_actual_gate["passed"],
            global_actual_gate["total"],
            global_actual_gate["minimum_margin"],
        )
    )
    print(
        "Rowwise: alpha_mean={:.9g}, alpha_median={:.9g}, zeros={}, norm_fraction={:.6f}, gate={}/{}, min_margin={:.9g}".format(
            report["rowwise"]["alpha_stats"]["mean"],
            report["rowwise"]["alpha_stats"]["median"],
            report["rowwise"]["zero_alpha_rows"],
            report["rowwise"]["norm_fraction_of_source"],
            rowwise_actual_gate["passed"],
            rowwise_actual_gate["total"],
            rowwise_actual_gate["minimum_margin"],
        )
    )
    print("Summary:", out / "minshrink_summary.json")
    print("Global checkpoint:", global_ckpt)
    print("Rowwise checkpoint:", rowwise_ckpt)


if __name__ == "__main__":
    main()
