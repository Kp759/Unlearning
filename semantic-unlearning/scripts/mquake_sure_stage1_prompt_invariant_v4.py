#!/usr/bin/env python3
"""MQuAKE SURE Stage 1 v4: prompt-invariant active GA with selective E/W edits.

Training uses only the locked training-visible direct facts.  The official
MQuAKE atomic question, multihop questions, target_new, and benchmark retain
examples are never read by this stage.

For each direct teacher-forced target_true token case, deterministic generic
wrappers create training-only synthetic views.  The sensitive basis is

    H_F_aug = hidden states from direct + synthetic views
    B_NS    = rowspace(non-sensitive context states from original direct views)
    B_S     = rowspace(H_F_aug - Proj_BNS(H_F_aug))

The tied vocabulary matrix is cloned/untied before learning.  Only
  * LM-head rows for every sensitive target token A_W, and
  * input-embedding rows A_E that actually occur in a later teacher-forced
    sensitive prefix
can move:

    Delta E_AE = eta * C_E B_S
    Delta W_AW =       C_W B_S

Stage 1 uses active/gated GA: a case stops contributing once its required
margin is satisfied.  Direct views require --constraint-margin; synthetic
views require the stronger --robust-margin.  Checkpoint selection uses the
combined direct+synthetic training gate.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage1_directional as v2
import mquake_sure_stage1_directional_v3 as v3


SYNTHETIC_TEMPLATES = (
    "Complete the following factual statement exactly:\n{prompt}",
    "Recall the continuation of this known fact:\n{prompt}",
    "Factual completion task:\n{prompt}",
)


@dataclass(frozen=True)
class V4PredictionCase:
    case_id: int
    record_position: int
    token_index: int
    prompt: str
    target_text: str
    view: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--embedding-scale", type=float, default=0.25)
    p.add_argument("--synthetic-view-count", type=int, default=3)
    p.add_argument("--protected-context-tokens", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--robust-margin", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def make_augmented_cases(
    direct_cases: Sequence[core.SensitivePredictionCase], synthetic_view_count: int
) -> List[V4PredictionCase]:
    if not 0 <= int(synthetic_view_count) <= len(SYNTHETIC_TEMPLATES):
        raise ValueError(
            f"synthetic-view-count must be in [0,{len(SYNTHETIC_TEMPLATES)}]"
        )
    templates = SYNTHETIC_TEMPLATES[: int(synthetic_view_count)]
    result: List[V4PredictionCase] = []
    for case in direct_cases:
        result.append(
            V4PredictionCase(
                case_id=int(case.case_id),
                record_position=int(case.record_position),
                token_index=int(case.token_index),
                prompt=str(case.prompt),
                target_text=str(case.target_text),
                view="direct",
            )
        )
        for j, template in enumerate(templates, start=1):
            result.append(
                V4PredictionCase(
                    case_id=int(case.case_id),
                    record_position=int(case.record_position),
                    token_index=int(case.token_index),
                    prompt=template.format(prompt=str(case.prompt)),
                    target_text=str(case.target_text),
                    view=f"synthetic_{j}",
                )
            )
    return result


def required_margin(case: V4PredictionCase, direct_margin: float, robust_margin: float) -> float:
    return float(direct_margin if case.view == "direct" else robust_margin)


def gate_model(
    model,
    tok,
    cases: Sequence[V4PredictionCase],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    direct_margin: float,
    robust_margin: float,
) -> Dict[str, Any]:
    residual: List[int] = []
    margins_all: List[float] = []
    slacks_all: List[float] = []
    by_view: Dict[str, Dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            logits = core.forward_last_logits(model, tok, batch, device).float()
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            rows = torch.arange(len(batch), device=device)
            target = logits[rows, tids]
            other = logits.clone()
            other[rows, tids] = -torch.inf
            margins = other.max(dim=-1).values - target
            for offset, (case, value) in enumerate(zip(batch, margins.detach().cpu().tolist())):
                margin = float(value)
                need = required_margin(case, direct_margin, robust_margin)
                slack = margin - need
                margins_all.append(margin)
                slacks_all.append(slack)
                bucket = by_view.setdefault(case.view, {"total": 0, "failed": 0})
                bucket["total"] += 1
                if slack < 0.0:
                    residual.append(start + offset)
                    bucket["failed"] += 1
    for bucket in by_view.values():
        bucket["passed"] = int(bucket["total"] - bucket["failed"])
    return {
        "total": len(cases),
        "passed": len(cases) - len(residual),
        "failed": len(residual),
        "residual_indices": residual,
        "minimum_margin": min(margins_all) if margins_all else None,
        "minimum_slack": min(slacks_all) if slacks_all else None,
        "mean_margin": sum(margins_all) / len(margins_all) if margins_all else None,
        "direct_margin": float(direct_margin),
        "robust_margin": float(robust_margin),
        "by_view": by_view,
    }


def make_prompt_invariant_basis(
    model,
    tok,
    direct_cases,
    augmented_cases,
    device,
    *,
    cache_batch_size: int,
    context_tokens: int,
):
    h_aug = core.forward_last_hidden(
        model, tok, augmented_cases, device, batch_size=cache_batch_size
    ).float()
    # Deliberately construct B_NS only from original direct contexts.  If all
    # synthetic-wrapper contexts are added to the full rowspace, they can span
    # away most of the augmented sensitive signal.
    h_ns = v2.collect_preceding_context_hidden(
        model,
        tok,
        direct_cases,
        device,
        batch_size=cache_batch_size,
        last_n=context_tokens,
    ).float()
    b_ns = core.orthonormal_row_basis(h_ns, max_rank=None).to(device=device, dtype=torch.float32)
    residual = h_aug - (h_aug @ b_ns.transpose(0, 1)) @ b_ns if b_ns.numel() else h_aug
    b_s = core.orthonormal_row_basis(residual, max_rank=None).to(device=device, dtype=torch.float32)
    if b_s.ndim != 2 or b_s.shape[0] == 0:
        raise RuntimeError("prompt-invariant sensitive residual basis is empty")
    residual_energy = float(
        (residual.square().sum() / h_aug.square().sum().clamp_min(1e-12)).detach().cpu()
    )
    captured_energy = float(
        (((residual @ b_s.transpose(0, 1)) @ b_s).square().sum()
         / residual.square().sum().clamp_min(1e-12)).detach().cpu()
    )
    return b_s, {
        "augmented_sensitive_hidden_rows": int(h_aug.shape[0]),
        "direct_non_sensitive_context_hidden_rows": int(h_ns.shape[0]),
        "hidden_size": int(h_aug.shape[1]),
        "non_sensitive_basis_rank": int(b_ns.shape[0]),
        "sensitive_residual_basis_rank": int(b_s.shape[0]),
        "residual_hidden_energy_fraction": residual_energy,
        "residual_energy_captured_by_B_S": captured_energy,
        "construction": "B_S=rowspace(H_F_aug-Proj_rowspace(H_NS_direct)(H_F_aug))",
    }


def selected_vocab_rows(tok, direct_cases, llama_like: bool, device: torch.device):
    tids = core.official_target_ids(tok, direct_cases, llama_like=llama_like, device=device)
    special = set(gagd.special_token_ids(tok))
    head_ids = sorted(set(int(x) for x in tids.detach().cpu().tolist()) - special)

    max_token_index: Dict[int, int] = {}
    for case in direct_cases:
        rp = int(case.record_position)
        max_token_index[rp] = max(max_token_index.get(rp, -1), int(case.token_index))
    emb_ids = sorted(
        {
            int(tid)
            for case, tid in zip(direct_cases, tids.detach().cpu().tolist())
            if int(case.token_index) < max_token_index[int(case.record_position)]
            and int(tid) not in special
        }
    )
    return emb_ids, head_ids


def register_scaled_input_hook(input_layer, row_ids, delta: core.SelectedRowDelta, scale: float):
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=input_layer.weight.device)

    def hook(_module, inputs, output):
        token_ids = inputs[0].to(ids.device)
        positions = torch.searchsorted(ids, token_ids)
        safe = positions.clamp(max=ids.numel() - 1)
        valid = (positions < ids.numel()) & (ids[safe] == token_ids)
        if not bool(valid.any()):
            return output
        effective = (
            float(scale) * delta.effective_delta()
        ).to(device=output.device, dtype=output.dtype)
        correction = torch.zeros_like(output)
        correction[valid] = effective[safe[valid]]
        return output + correction

    return input_layer.register_forward_hook(hook)


def main() -> None:
    a = parse_args()
    if min(a.steps, a.batch_size, a.cache_batch_size, a.protected_context_tokens, a.check_every) <= 0:
        raise ValueError("steps/batches/context/check interval must be positive")
    if a.learning_rate <= 0 or not (0.0 <= a.embedding_scale <= 1.0):
        raise ValueError("learning rate must be positive and embedding-scale in [0,1]")
    if a.robust_margin < a.constraint_margin:
        raise ValueError("robust-margin must be >= constraint-margin")

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

    direct_cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not direct_cases:
        raise RuntimeError("no MQuAKE sensitive PredictionCases")
    augmented_cases = make_augmented_cases(direct_cases, a.synthetic_view_count)

    model.eval()
    sensitive_basis, direction_report = make_prompt_invariant_basis(
        model,
        tok,
        direct_cases,
        augmented_cases,
        device,
        cache_batch_size=a.cache_batch_size,
        context_tokens=a.protected_context_tokens,
    )

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")
    frozen_before = v3.transformer_hash(model, input_layer.weight, output_layer.weight)

    emb_ids, head_ids = selected_vocab_rows(tok, direct_cases, llama_like, device)
    if not head_ids:
        raise RuntimeError("no content-bearing sensitive LM-head rows")

    emb_delta = None
    input_hook = None
    if emb_ids and a.embedding_scale > 0.0:
        emb_delta = core.SelectedRowDelta(
            len(emb_ids), int(input_layer.weight.shape[1]),
            direction_basis=sensitive_basis, device=device
        )
        input_hook = register_scaled_input_hook(input_layer, emb_ids, emb_delta, a.embedding_scale)

    head_delta = core.SelectedRowDelta(
        len(head_ids), int(output_layer.weight.shape[1]),
        direction_basis=sensitive_basis, device=device
    )
    output_hook = core.register_output_delta_hook(output_layer, head_ids, head_delta.effective_delta)

    modules = ([emb_delta] if emb_delta is not None else []) + [head_delta]
    params = [p for module in modules for p in module.parameters()]
    if a.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=a.learning_rate)
    elif a.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=a.learning_rate)
    else:
        optimizer = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.0)

    sampler = core.IndexSampler(len(augmented_cases), a.batch_size, a.seed)
    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    initial_gate = gate_model(
        model, tok, augmented_cases, llama_like, device, a.cache_batch_size,
        a.constraint_margin, a.robust_margin
    )
    best_state = v3.capture_state(*modules)
    best_key = (
        int(initial_gate["failed"]),
        int(initial_gate["by_view"]["direct"]["failed"]),
        -float(initial_gate["minimum_slack"]),
        0.0,
    )
    best_step = 0
    best_gate = initial_gate
    checks: List[Dict[str, Any]] = []

    log_path = out / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc="MQuAKE Stage1 v4 prompt-invariant active GA"):
            idx = sampler.next()
            batch = [augmented_cases[i] for i in idx]
            optimizer.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device).float()
            tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
            rows = torch.arange(len(batch), device=device)
            target = logits[rows, tids]
            other = logits.detach().clone()
            other[rows, tids] = -torch.inf
            margins = other.max(dim=-1).values - target.detach()
            needs = torch.tensor(
                [required_margin(c, a.constraint_margin, a.robust_margin) for c in batch],
                dtype=torch.float32, device=device
            )
            active = margins < needs
            active_count = int(active.sum().item())
            if active_count:
                logp = F.log_softmax(logits, dim=-1)[rows, tids]
                ga = logp[active].mean()
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
            else:
                ga = logits.sum() * 0.0
                grad_norm = None

            if step == 1 or step % a.check_every == 0 or step == a.steps:
                gate = gate_model(
                    model, tok, augmented_cases, llama_like, device, a.cache_batch_size,
                    a.constraint_margin, a.robust_margin
                )
                emb_norm = 0.0 if emb_delta is None else float(
                    (float(a.embedding_scale) * emb_delta.effective_delta()).detach().float().norm().cpu()
                )
                head_norm = float(head_delta.effective_delta().detach().float().norm().cpu())
                key = (
                    int(gate["failed"]),
                    int(gate["by_view"]["direct"]["failed"]),
                    -float(gate["minimum_slack"]),
                    emb_norm + head_norm,
                )
                if key < best_key:
                    best_key = key
                    best_state = v3.capture_state(*modules)
                    best_step = step
                    best_gate = gate
                row = {
                    "step": step,
                    "active_in_sample": active_count,
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
                direct_failed = gate["by_view"]["direct"]["failed"]
                print(
                    "Stage1-v4 step {s}: robust_pass={p}/{t} direct_fail={df} total_fail={f} "
                    "min_slack={ms:.6g} active={ac} ||dE||={de:.6g} ||dW||={dw:.6g}".format(
                        s=step, p=gate["passed"], t=gate["total"], df=direct_failed,
                        f=gate["failed"], ms=gate["minimum_slack"], ac=active_count,
                        de=emb_norm, dw=head_norm
                    )
                )
                if int(gate["failed"]) == 0:
                    break

    v3.restore_state(best_state, *modules)
    del optimizer
    if input_hook is not None:
        input_hook.remove()
    output_hook.remove()

    emb_effective = (
        torch.empty((0, int(input_layer.weight.shape[1])), device=device, dtype=torch.float32)
        if emb_delta is None
        else float(a.embedding_scale) * emb_delta.effective_delta().detach()
    )
    head_effective = head_delta.effective_delta().detach()
    if emb_delta is not None:
        v2.materialize_input_delta(input_layer, emb_ids, emb_effective)
    core.materialize_output_delta(output_layer, head_ids, head_effective)
    model.eval()

    frozen_after = v3.transformer_hash(model, input_layer.weight, output_layer.weight)
    transformer_exact = frozen_before == frozen_after
    if not transformer_exact:
        raise RuntimeError("non-vocabulary transformer parameters changed in Stage1")

    materialized_gate = gate_model(
        model, tok, augmented_cases, llama_like, device, a.cache_batch_size,
        a.constraint_margin, a.robust_margin
    )
    direct_materialized = materialized_gate["by_view"]["direct"]
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    report: Dict[str, Any] = {
        "schema_version": 4,
        "method": "MQuAKE SURE Stage1 Prompt-Invariant Active GA",
        "protocol": "training-visible direct plus deterministic synthetic wrappers",
        "source_protocol": manifest.get("protocol"),
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "direct_prediction_cases": len(direct_cases),
        "augmented_prediction_cases": len(augmented_cases),
        "synthetic_view_count": int(a.synthetic_view_count),
        "synthetic_templates": list(SYNTHETIC_TEMPLATES[: a.synthetic_view_count]),
        "official_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "embedding_lm_head_untied": True,
        "transformer_exact": transformer_exact,
        "embedding_rows": emb_ids,
        "embedding_row_count": len(emb_ids),
        "lm_head_rows": head_ids,
        "lm_head_row_count": len(head_ids),
        "embedding_scale": float(a.embedding_scale),
        "direction": direction_report,
        "parameterization": "Delta E_AE=eta*C_E*B_S for prefix-only rows; Delta W_AW=C_W*B_S for all target rows",
        "objective": "active sensitive-token GA; solved direct/synthetic cases stop contributing",
        "constraint_margin": float(a.constraint_margin),
        "robust_margin": float(a.robust_margin),
        "selection": "best combined direct+synthetic training gate checkpoint",
        "steps_requested": int(a.steps),
        "learning_rate": float(a.learning_rate),
        "batch_size": int(a.batch_size),
        "check_every": int(a.check_every),
        "best_step": int(best_step),
        "best_gate_under_hooks": best_gate,
        "materialized_robust_gate": materialized_gate,
        "materialized_direct_gate": direct_materialized,
        "embedding_delta_norm": float(emb_effective.float().norm().cpu()),
        "head_delta_norm": float(head_effective.float().norm().cpu()),
        "checks": checks,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out / "stage1_summary.json", report)
    torch.save(
        {
            "sensitive_basis": sensitive_basis.detach().cpu(),
            "embedding_token_ids": emb_ids,
            "head_token_ids": head_ids,
            "embedding_delta": emb_effective.float().cpu(),
            "head_delta": head_effective.float().cpu(),
            "synthetic_view_count": int(a.synthetic_view_count),
        },
        out / "stage1_prompt_invariant_state.pt",
    )

    print("Directional Stage1 v4 checkpoint:", ckpt)
    print("Direct PredictionCases:", len(direct_cases))
    print("Augmented PredictionCases:", len(augmented_cases))
    print("Embedding prefix-only rows:", len(emb_ids))
    print("LM-head sensitive rows:", len(head_ids))
    print("B_NS rank:", direction_report["non_sensitive_basis_rank"])
    print("B_S rank:", direction_report["sensitive_residual_basis_rank"])
    print("Residual hidden energy fraction:", direction_report["residual_hidden_energy_fraction"])
    print("Best Stage1 step:", best_step)
    print("Materialized direct gate: {}/{} pass".format(direct_materialized["passed"], direct_materialized["total"]))
    print("Materialized robust gate: {}/{} pass".format(materialized_gate["passed"], materialized_gate["total"]))
    print("Embedding/LM head untied: True")


if __name__ == "__main__":
    main()
