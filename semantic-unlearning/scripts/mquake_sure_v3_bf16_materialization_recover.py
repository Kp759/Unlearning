#!/usr/bin/env python3
"""Recover a training-feasible SURE-v3 Stage2 checkpoint after low-precision materialization.

This is a numerical guard, not a new learning objective.  It reloads the Stage1
checkpoint and the already learned Stage2 delta, then searches upward from scale
1.0 over representable values of the requested storage dtype.  The first scale
whose *materialized* checkpoint satisfies the same locked direct constraints is
accepted.

No AtomicGen, retain, target_new, paraphrase, neighborhood, or multihop data are
read.  The calibrated repair margin is taken from stage2_nullspace_state.pt and
is never changed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage2_head_directional as v2
from mquake_sure_stage2_head_nullspace_v3 import margins_from_logits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-model-path", required=True)
    p.add_argument("--stage2-state", required=True)
    p.add_argument("--stage2-summary", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def storage_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def restore_selected_rows(output_layer, row_ids: List[int], rows_cpu: torch.Tensor) -> None:
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_layer.weight.device)
    rows = rows_cpu.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype)
    with torch.no_grad():
        output_layer.weight.index_copy_(0, ids, rows)


def repair_gate_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    f_indices: List[int],
    repair_margin: float,
) -> Dict[str, Any]:
    all_margins = margins_from_logits(logits, target_ids)
    idx = torch.tensor(f_indices, dtype=torch.long, device=all_margins.device)
    fm = all_margins.index_select(0, idx)
    bad = torch.nonzero(fm < float(repair_margin), as_tuple=False).flatten()
    return {
        "total": len(f_indices),
        "passed": len(f_indices) - int(bad.numel()),
        "failed": int(bad.numel()),
        "minimum_margin": float(fm.min().detach().cpu()) if fm.numel() else None,
        "mean_margin": float(fm.mean().detach().cpu()) if fm.numel() else None,
        "required_margin": float(repair_margin),
    }


def main() -> None:
    a = parse_args()
    if a.constraint_margin < 0 or a.max_protected_kl < 0:
        raise ValueError("invalid margin/KL settings")
    if a.cache_batch_size <= 0:
        raise ValueError("cache batch size must be positive")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = v2.load_locked(a)
    state_path = Path(a.stage2_state).resolve()
    summary_path = Path(a.stage2_summary).resolve()
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    repair_margin = float(state["repair_margin"])
    repair_margin_mode = str(state.get("repair_margin_mode", summary.get("repair_margin_mode", "unknown")))
    a_f = [int(x) for x in state["A_F"]]
    best_delta = state["best_delta"].detach().float().cpu()
    if best_delta.ndim != 2 or best_delta.shape[0] != len(a_f):
        raise RuntimeError("Stage2 state has inconsistent A_F / best_delta shapes")

    ns = argparse.Namespace(
        model_path=a.stage1_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model.eval()
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)

    base_logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
    level1_gate = v2.gate_from_logits(base_logits, target_ids.cpu(), a.constraint_margin)
    f_indices = [int(x) for x in level1_gate["residual_indices"]]
    failed_set = set(f_indices)
    p_indices = [i for i in range(len(cases)) if i not in failed_set]

    if len(f_indices) != int(summary.get("level2", {}).get("F", len(f_indices))):
        raise RuntimeError("Stage1 F count differs from saved Stage2 summary")

    row_ids = torch.tensor(a_f, dtype=torch.long)
    stage1_rows = output_layer.weight.detach().cpu().index_select(0, row_ids).clone()
    frozen_before = v2.hash_frozen(model, output_layer.weight)

    def evaluate_scale(scale: float) -> Dict[str, Any]:
        restore_selected_rows(output_layer, a_f, stage1_rows)
        core.materialize_output_delta(output_layer, a_f, best_delta * float(scale))
        logits = core.cache_base_logits(model, tok, cases, device, batch_size=a.cache_batch_size)
        direct_gate = v2.gate_from_logits(logits, target_ids.cpu(), a.constraint_margin)
        repair_gate = repair_gate_from_logits(logits, target_ids, f_indices, repair_margin)
        final_failed = set(direct_gate["residual_indices"])
        p_regressions = sum(1 for i in p_indices if i in final_failed)
        protected_kl = v2.actual_full_kl_mean(base_logits, logits, p_indices, a.cache_batch_size)
        feasible = bool(
            int(direct_gate["failed"]) == 0
            and int(repair_gate["failed"]) == 0
            and int(p_regressions) == 0
            and float(protected_kl) <= float(a.max_protected_kl)
        )
        return {
            "scale": float(scale),
            "direct_gate": direct_gate,
            "repair_gate": repair_gate,
            "p_regressions": int(p_regressions),
            "protected_kl": float(protected_kl),
            "feasible": feasible,
        }

    # Search the smallest representable storage-dtype scalar >=1.0.  This makes
    # the numerical recovery rule deterministic and tied to checkpoint precision,
    # rather than introducing a manually tuned safety margin.
    sdtype = storage_dtype(a.dtype)
    scale_tensor = torch.tensor(1.0, dtype=sdtype)
    inf_tensor = torch.tensor(float("inf"), dtype=sdtype)
    attempts = []
    chosen = None
    while float(scale_tensor) <= 2.0:
        scale = float(scale_tensor)
        result = evaluate_scale(scale)
        attempts.append(result)
        print(
            "BF16 materialization recovery scale={:.9g}: direct={}/{} repair={}/{} min_repair={} P_reg={} KL={:.6g} pass={}".format(
                scale,
                result["direct_gate"]["passed"],
                result["direct_gate"]["total"],
                result["repair_gate"]["passed"],
                result["repair_gate"]["total"],
                result["repair_gate"]["minimum_margin"],
                result["p_regressions"],
                result["protected_kl"],
                result["feasible"],
            )
        )
        if result["feasible"]:
            chosen = result
            break
        next_scale = torch.nextafter(scale_tensor, inf_tensor)
        if float(next_scale) <= float(scale_tensor):
            break
        scale_tensor = next_scale

    if chosen is None:
        restore_selected_rows(output_layer, a_f, stage1_rows)
        raise RuntimeError("No materialized-feasible scale found in [1,2] over storage-dtype representable values")

    # evaluate_scale leaves the chosen materialized rows in the model.
    frozen_after = v2.hash_frozen(model, output_layer.weight)
    frozen_exact = frozen_before == frozen_after
    if not frozen_exact:
        raise RuntimeError("non-head parameters changed during materialization recovery")

    out = gagd.resolve_output_path(a.output_dir)
    ckpt = out / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    recovered_norm = float((best_delta * float(chosen["scale"])).norm().item())
    recovery = {
        "schema_version": 1,
        "purpose": "low-precision materialization recovery using training-visible direct gates only",
        "heldout_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "repair_margin_mode": repair_margin_mode,
        "repair_margin": repair_margin,
        "storage_dtype": a.dtype,
        "search_rule": "first representable storage-dtype scale >=1 satisfying the unchanged materialized gates",
        "chosen_scale": float(chosen["scale"]),
        "attempts": attempts,
        "final_direct_gate": chosen["direct_gate"],
        "final_F_repair_margin_gate": chosen["repair_gate"],
        "stage1_successes_regressed": chosen["p_regressions"],
        "protected_kl": chosen["protected_kl"],
        "frozen_non_head_exact": bool(frozen_exact),
        "recovered_materialized_delta_norm": recovered_norm,
        "checkpoint": str(ckpt.resolve()),
        "source_protocol": manifest.get("protocol"),
    }
    core.write_json(out / "bf16_materialization_recovery.json", recovery)

    summary["materialization_recovery"] = recovery
    summary["final_gate"] = chosen["direct_gate"]
    summary["final_F_repair_margin_gate"] = chosen["repair_gate"]
    summary["stage1_successes_regressed"] = chosen["p_regressions"]
    summary["protected_kl"] = chosen["protected_kl"]
    summary["frozen_non_head_exact"] = bool(frozen_exact)
    summary["final_gates_pass"] = bool(chosen["feasible"] and frozen_exact)
    summary["checkpoint"] = str(ckpt.resolve())
    summary.setdefault("level2", {})["materialization_recovery_scale"] = float(chosen["scale"])
    summary["level2"]["recovered_materialized_delta_norm"] = recovered_norm
    core.write_json(out / "two_stage_summary.json", summary)

    print("===== MATERIALIZATION RECOVERY COMPLETE =====")
    print("repair_margin:", repair_margin)
    print("chosen_scale:", chosen["scale"])
    print("final_direct_gate:", chosen["direct_gate"])
    print("final_F_repair_margin_gate:", chosen["repair_gate"])
    print("stage1_successes_regressed:", chosen["p_regressions"])
    print("protected_kl:", chosen["protected_kl"])
    print("frozen_non_head_exact:", frozen_exact)
    print("final_gates_pass:", summary["final_gates_pass"])
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
