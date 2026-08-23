#!/usr/bin/env python3
"""MQuAKE Stage 1 v3: untied E/W GA in the full sensitive residual subspace.

Design:
  * transformer frozen exactly;
  * clone/untie the tied LM head before learning;
  * H_F = final hidden states for all training-visible target_true token cases;
  * H_NS = preceding non-prediction context hidden states from the same prompts;
  * B_NS = full numerical row-space basis of H_NS;
  * R_F = H_F - Proj_{B_NS}(H_F);
  * B_S = full numerical row-space basis of R_F;
  * only sensitive rows A move, with
        Delta E_A = C_E B_S
        Delta W_A = C_W B_S
  * Stage 1 uses sensitive-token GA only. Preservation is structural: non-sensitive
    rows are frozen and update directions exclude the observed non-sensitive hidden
    subspace. No full-vocabulary KL is allowed to fight the forget objective.
  * every checkpoint interval, gate all training-visible atomic token decisions and
    retain the best iterate; stop immediately if all pass.

No benchmark-retain item, held-out MQuAKE probe, target_new, rank sweep, or scale
sweep is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from tqdm import tqdm

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage1_directional as v2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--protected-context-tokens", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def capture_state(*modules: torch.nn.Module):
    return [[p.detach().clone() for p in m.parameters()] for m in modules]


@torch.no_grad()
def restore_state(state, *modules: torch.nn.Module) -> None:
    for saved, module in zip(state, modules):
        for parameter, value in zip(module.parameters(), saved):
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def gate_model(model, tok, cases, llama_like, device, batch_size: int, margin: float):
    residual = []
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            logits = core.forward_last_logits(model, tok, batch, device).float()
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            rows = torch.arange(logits.shape[0], device=logits.device)
            target = logits[rows, tids]
            other = logits.clone()
            other[rows, tids] = -torch.inf
            margins = other.max(dim=-1).values - target
            for offset, value in enumerate(margins.detach().cpu().tolist()):
                values.append(float(value))
                if float(value) < float(margin):
                    residual.append(start + offset)
    return {
        "total": len(cases),
        "passed": len(cases) - len(residual),
        "failed": len(residual),
        "residual_indices": residual,
        "minimum_margin": min(values) if values else None,
        "mean_margin": sum(values) / len(values) if values else None,
        "required_margin": float(margin),
    }


def transformer_hash(model, input_weight, output_weight) -> str:
    digest = hashlib.sha256()
    excluded = {id(input_weight), id(output_weight)}
    for name, parameter in model.named_parameters():
        if id(parameter) in excluded:
            continue
        t = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(t.dtype).encode())
        digest.update(str(tuple(t.shape)).encode())
        digest.update(t.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def make_full_residual_basis(model, tok, cases, device, *, cache_batch_size: int, context_tokens: int):
    h_f = core.forward_last_hidden(model, tok, cases, device, batch_size=cache_batch_size).float()
    h_ns = v2.collect_preceding_context_hidden(
        model,
        tok,
        cases,
        device,
        batch_size=cache_batch_size,
        last_n=context_tokens,
    ).float()
    b_ns = core.orthonormal_row_basis(h_ns, max_rank=None).to(device=device, dtype=torch.float32)
    residual = h_f - (h_f @ b_ns.transpose(0, 1)) @ b_ns if b_ns.numel() else h_f
    b_s = core.orthonormal_row_basis(residual, max_rank=None).to(device=device, dtype=torch.float32)
    if b_s.ndim != 2 or b_s.shape[0] == 0:
        raise RuntimeError("full sensitive residual basis is empty")
    residual_energy = float(
        (residual.square().sum() / h_f.square().sum().clamp_min(1e-12)).detach().cpu()
    )
    captured_energy = float(
        (((residual @ b_s.transpose(0, 1)) @ b_s).square().sum()
         / residual.square().sum().clamp_min(1e-12)).detach().cpu()
    )
    return b_s, {
        "sensitive_hidden_rows": int(h_f.shape[0]),
        "non_sensitive_context_hidden_rows": int(h_ns.shape[0]),
        "hidden_size": int(h_f.shape[1]),
        "non_sensitive_basis_rank": int(b_ns.shape[0]),
        "sensitive_residual_basis_rank": int(b_s.shape[0]),
        "residual_hidden_energy_fraction": residual_energy,
        "residual_energy_captured_by_B_S": captured_energy,
        "construction": "B_S = rowspace(H_F - Proj_rowspace(H_NS)(H_F)); full numerical ranks, no sweep",
    }


def main() -> None:
    a = parse_args()
    if min(a.steps, a.batch_size, a.cache_batch_size, a.protected_context_tokens, a.check_every) <= 0:
        raise ValueError("steps/batches/context/check interval must be positive")
    if a.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest, visible_path, manifest_path = v2.load_locked(a)
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
    cases = core.expand_sensitive_cases(records, tok, sensitive_field="target_true", llama_like=llama_like)
    if not cases:
        raise RuntimeError("no MQuAKE sensitive PredictionCases")

    model.eval()
    sensitive_basis, direction_report = make_full_residual_basis(
        model,
        tok,
        cases,
        device,
        cache_batch_size=a.cache_batch_size,
        context_tokens=a.protected_context_tokens,
    )

    # Clone the tied output matrix before any learning. core helper also freezes model params.
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")
    frozen_before = transformer_hash(model, input_layer.weight, output_layer.weight)

    tids_all = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    special = set(gagd.special_token_ids(tok))
    sensitive_ids = sorted(set(int(x) for x in tids_all.detach().cpu().tolist()) - special)
    if not sensitive_ids:
        raise RuntimeError("no content-bearing sensitive rows")

    emb_delta = core.SelectedRowDelta(
        len(sensitive_ids), int(input_layer.weight.shape[1]), direction_basis=sensitive_basis, device=device
    )
    head_delta = core.SelectedRowDelta(
        len(sensitive_ids), int(output_layer.weight.shape[1]), direction_basis=sensitive_basis, device=device
    )
    input_hook = v2.register_input_delta_hook(input_layer, sensitive_ids, emb_delta)
    output_hook = core.register_output_delta_hook(output_layer, sensitive_ids, head_delta.effective_delta)
    params = list(emb_delta.parameters()) + list(head_delta.parameters())

    if a.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=a.learning_rate)
    elif a.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=a.learning_rate)
    else:
        optimizer = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.0)

    sampler = core.IndexSampler(len(cases), a.batch_size, a.seed)
    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    initial_gate = gate_model(model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin)
    best_state = capture_state(emb_delta, head_delta)
    best_key = (
        int(initial_gate["failed"]),
        -float(initial_gate["minimum_margin"]),
        0.0,
    )
    best_step = 0
    best_gate = initial_gate
    checks = [{"step": 0, "gate": initial_gate, "embedding_delta_norm": 0.0, "head_delta_norm": 0.0}]

    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc="MQuAKE Stage1 v3 full-residual directional GA"):
            idx = sampler.next()
            batch = [cases[i] for i in idx]
            optimizer.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            ga = core.ga_sensitive_logprob(logits, tids)
            if not torch.isfinite(ga):
                raise FloatingPointError(f"non-finite Stage1 GA at step {step}")
            ga.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage1 gradient at step {step}")
            optimizer.step()

            if step == 1 or step % a.check_every == 0 or step == a.steps:
                gate = gate_model(model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin)
                emb_norm = float(emb_delta.effective_delta().detach().float().norm().cpu())
                head_norm = float(head_delta.effective_delta().detach().float().norm().cpu())
                key = (int(gate["failed"]), -float(gate["minimum_margin"]), emb_norm + head_norm)
                if key < best_key:
                    best_key = key
                    best_state = capture_state(emb_delta, head_delta)
                    best_step = step
                    best_gate = gate
                row = {
                    "step": step,
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gradient_norm_before_clip": None if grad_norm is None else float(grad_norm.detach().cpu()),
                    "embedding_delta_norm": emb_norm,
                    "head_delta_norm": head_norm,
                    "gate": gate,
                    "best_step": int(best_step),
                }
                checks.append(row)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                print(
                    "Stage1 step {s}: pass={p}/{t} failed={f} min_margin={m:.6g} ||dE||={de:.6g} ||dW||={dw:.6g}".format(
                        s=step, p=gate["passed"], t=gate["total"], f=gate["failed"],
                        m=gate["minimum_margin"], de=emb_norm, dw=head_norm
                    )
                )
                if int(gate["failed"]) == 0:
                    break

    restore_state(best_state, emb_delta, head_delta)
    del optimizer
    input_hook.remove()
    output_hook.remove()

    emb_effective = emb_delta.effective_delta().detach()
    head_effective = head_delta.effective_delta().detach()
    v2.materialize_input_delta(input_layer, sensitive_ids, emb_effective)
    core.materialize_output_delta(output_layer, sensitive_ids, head_effective)
    model.eval()

    frozen_after = transformer_hash(model, input_layer.weight, output_layer.weight)
    transformer_exact = frozen_before == frozen_after
    if not transformer_exact:
        raise RuntimeError("non-vocabulary transformer parameters changed in Stage1")

    # Verify the selected materialized iterate without hooks.
    materialized_gate = gate_model(model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    report: Dict[str, Any] = {
        "schema_version": 3,
        "method": "MQuAKE Stage1 Untied Full-Residual Directional GA",
        "protocol": "training-visible-direct-only",
        "source_protocol": manifest.get("protocol"),
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "prediction_case_count": len(cases),
        "sensitive_row_count": len(sensitive_ids),
        "sensitive_token_ids": sensitive_ids,
        "embedding_lm_head_untied": True,
        "transformer_exact": transformer_exact,
        "direction": direction_report,
        "parameterization": "Delta E_A=C_E B_S; Delta W_A=C_W B_S; full residual basis",
        "objective": "sensitive-token GA only; no KL term on the sensitive training cases",
        "selection": "best full atomic training-visible gate checkpoint; stop at zero failures",
        "steps_requested": int(a.steps),
        "learning_rate": float(a.learning_rate),
        "batch_size": int(a.batch_size),
        "check_every": int(a.check_every),
        "constraint_margin": float(a.constraint_margin),
        "best_step": int(best_step),
        "best_gate_under_hooks": best_gate,
        "materialized_gate": materialized_gate,
        "embedding_delta_norm": float(emb_effective.float().norm().cpu()),
        "head_delta_norm": float(head_effective.float().norm().cpu()),
        "checks": checks,
        "benchmark_retain_seen": 0,
        "heldout_probes_seen": 0,
        "target_new_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out / "stage1_summary.json", report)
    torch.save(
        {
            "sensitive_basis": sensitive_basis.detach().cpu(),
            "sensitive_token_ids": sensitive_ids,
            "embedding_delta": emb_effective.float().cpu(),
            "head_delta": head_effective.float().cpu(),
        },
        out / "stage1_directional_state.pt",
    )

    print("Directional Stage1 v3 checkpoint:", ckpt)
    print("PredictionCases:", len(cases))
    print("Sensitive rows:", len(sensitive_ids))
    print("B_NS rank:", direction_report["non_sensitive_basis_rank"])
    print("B_S rank:", direction_report["sensitive_residual_basis_rank"])
    print("Residual hidden energy fraction:", direction_report["residual_hidden_energy_fraction"])
    print("Best Stage1 step:", best_step)
    print("Materialized Stage1 gate: {}/{} pass".format(materialized_gate["passed"], materialized_gate["total"]))
    print("Embedding/LM head untied: True")


if __name__ == "__main__":
    main()
