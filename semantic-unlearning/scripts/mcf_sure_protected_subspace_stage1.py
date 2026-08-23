#!/usr/bin/env python3
"""MCF SURE protected-subspace Stage 1.

Leak-free target contract:
  requested_rewrite.target_true = sensitive answer
  requested_rewrite.target_new  = non-sensitive CounterFact reference
  fields are never swapped

Stage 1 geometry (all bases are built only from training-visible direct data):

    H_S  = final hidden states at sensitive target_true prediction positions
    H_NS = preceding hidden states from PRE-ANSWER direct-prompt context only
           (collected from token_index == 0 cases, so no teacher-forced
           target_true answer-prefix states enter the protected context basis)
    B_NS = rowspace(H_NS), capped by --protected-rank (default 32)
    R_S  = H_S - Proj_BNS(H_S)
    B_S  = rowspace(R_S), capped by --sensitive-rank (default 4)

Only target_true-sensitive input-embedding rows and LM-head rows are editable:

    Delta E_A = C_E B_S
    Delta W_A = C_W B_S

The transformer is frozen. The objective is sensitive GA plus full-vocabulary
KL(Base || Edited) on the SAME direct training prompts and a tiny delta L2.
No paraphrases, neighborhoods, benchmark-retain rows, or PPL text are opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import gagd_compare as gagd
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as shared_stage2
import mcf_sure_directional_emb_lm_stage1 as directional_v1


METHOD = "SURE-MCF-protected-subspace-EmbLM-GA-stage1"
PROTOCOL = "mcf_target_true_protected_subspace_stage1_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=1e-6)
    p.add_argument("--protected-rank", type=int, default=32)
    p.add_argument("--sensitive-rank", type=int, default=4)
    p.add_argument("--atomic-margin", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.steps, a.batch_size, a.cache_batch_size) <= 0:
        p.error("forget-num, steps, and batch sizes must be positive")
    if a.lr <= 0 or a.ga_weight <= 0:
        p.error("lr and ga-weight must be positive")
    if min(a.kl_weight, a.delta_l2, a.protected_rank, a.sensitive_rank, a.atomic_margin, a.grad_clip) < 0:
        p.error("KL/L2/ranks/margin/clip must be non-negative")
    if a.protected_rank == 0 or a.sensitive_rank == 0:
        p.error("protected-rank and sensitive-rank must be positive for v2")
    return a


def validate_locked(
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    records, manifest = directional_v1.validate_locked(
        visible_path, manifest_path, seed, forget_num
    )
    return records, manifest


def project_away(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the orthogonal projection onto an orthonormal row basis."""
    rows32 = rows.float()
    if basis.numel() == 0:
        return rows32
    b = basis.to(device=rows32.device, dtype=torch.float32)
    return rows32 - (rows32 @ b.transpose(0, 1)) @ b


@torch.no_grad()
def collect_prediction_and_context_hidden(
    model: nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return prediction states and pre-answer direct-prompt context states.

    H_S is collected for every target_true atomic prediction case. H_NS is
    deliberately collected only from token_index==0 cases, before any
    teacher-forced target_true answer prefix has been appended. This keeps the
    protected context basis non-sensitive by construction.
    """
    prediction: List[torch.Tensor] = []
    context_rows: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(cases), int(batch_size)):
        batch = cases[start : start + int(batch_size)]
        encoded = tok(
            [c.prompt for c in batch], padding=True, return_tensors="pt"
        ).to(device)
        output = model(
            **encoded, output_hidden_states=True, use_cache=False
        )
        hidden = output.hidden_states[-1].float()
        mask = encoded["attention_mask"].bool()
        for row, case in enumerate(batch):
            valid = torch.nonzero(mask[row], as_tuple=False).flatten()
            if valid.numel() == 0:
                raise RuntimeError("tokenized direct case has no valid positions")
            pred_pos = int(valid[-1].item())
            prediction.append(hidden[row, pred_pos].detach())
            if int(case.token_index) == 0:
                preceding = valid[:-1]
                if preceding.numel() > 0:
                    context_rows.append(
                        hidden[row].index_select(0, preceding).detach()
                    )
    if not prediction:
        raise RuntimeError("no sensitive prediction hidden states collected")
    h_s = torch.stack(prediction, dim=0)
    if context_rows:
        h_context = torch.cat(context_rows, dim=0)
    else:
        h_context = h_s.new_empty((0, h_s.shape[1]))
    return h_s, h_context


def build_sensitive_residual_basis(
    h_sensitive: torch.Tensor,
    h_context: torch.Tensor,
    *,
    protected_rank: int,
    sensitive_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if h_sensitive.ndim != 2 or h_sensitive.shape[0] == 0:
        raise ValueError("h_sensitive must be non-empty [N,H]")
    if h_context.ndim != 2 or h_context.shape[1] != h_sensitive.shape[1]:
        raise ValueError("h_context must be [M,H] with matching hidden size")
    if h_context.shape[0] == 0:
        b_ns = h_sensitive.new_empty((0, h_sensitive.shape[1]), dtype=torch.float32)
    else:
        b_ns = core.orthonormal_row_basis(
            h_context.float(), max_rank=int(protected_rank)
        )
    residual = project_away(h_sensitive, b_ns)
    b_s = core.orthonormal_row_basis(
        residual.float(), max_rank=int(sensitive_rank)
    )
    if b_s.ndim != 2 or b_s.shape[0] <= 0:
        raise RuntimeError(
            "sensitive residual subspace has zero numerical rank after context projection"
        )
    protected_leak = (
        float((b_s @ b_ns.transpose(0, 1)).abs().max().cpu())
        if b_ns.numel()
        else 0.0
    )
    report = {
        "sensitive_hidden_rows": int(h_sensitive.shape[0]),
        "context_hidden_rows": int(h_context.shape[0]),
        "hidden_size": int(h_sensitive.shape[1]),
        "protected_rank_requested": int(protected_rank),
        "protected_rank_actual": int(b_ns.shape[0]),
        "sensitive_rank_requested": int(sensitive_rank),
        "sensitive_rank_actual": int(b_s.shape[0]),
        "residual_frobenius_norm": float(residual.norm().cpu()),
        "sensitive_basis_max_abs_overlap_with_protected_basis": protected_leak,
    }
    return b_ns.contiguous(), residual.contiguous(), b_s.contiguous(), report


def full_vocab_kl(reference_logits: torch.Tensor, current_logits: torch.Tensor) -> torch.Tensor:
    ref = reference_logits.to(device=current_logits.device, dtype=torch.float32)
    cur = current_logits.float()
    if ref.shape != cur.shape:
        raise ValueError("reference/current logits must have identical shape")
    ref_logp = F.log_softmax(ref, dim=-1)
    cur_logp = F.log_softmax(cur, dim=-1)
    return (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).mean()


def atomic_margins(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """max(other logit) - sensitive target_true logit; larger is better."""
    if logits.ndim != 2 or target_ids.ndim != 1 or logits.shape[0] != target_ids.shape[0]:
        raise ValueError("expected logits [N,V] and target ids [N]")
    x = logits.float()
    rows = torch.arange(x.shape[0], device=x.device)
    target = x[rows, target_ids]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[rows, target_ids] = True
    best_other = x.masked_fill(mask, -torch.inf).max(dim=-1).values
    return best_other - target


@torch.no_grad()
def evaluate_atomic_cases(
    model: nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_margins: List[torch.Tensor] = []
    all_ids: List[torch.Tensor] = []
    for start in range(0, len(cases), int(batch_size)):
        batch = cases[start : start + int(batch_size)]
        logits = core.forward_last_logits(model, tok, batch, device)
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        all_margins.append(atomic_margins(logits, tids).detach().float().cpu())
        all_ids.append(tids.detach().cpu())
    return torch.cat(all_margins), torch.cat(all_ids)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = validate_locked(
        visible_path, manifest_path, int(a.seed), int(a.forget_num)
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
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    sensitive_cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    target_ids = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    )
    special = set(int(x) for x in gagd.special_token_ids(tok))
    selected_ids = sorted(
        set(int(x) for x in target_ids.detach().cpu().tolist()) - special
    )
    if not selected_ids:
        raise RuntimeError("no target_true-sensitive vocabulary rows selected")

    # Geometry is frozen from the Base model before any virtual edit is active.
    h_sensitive, h_context = collect_prediction_and_context_hidden(
        model,
        tok,
        sensitive_cases,
        device,
        int(a.cache_batch_size),
    )
    b_ns, residual_s, b_s, geometry = build_sensitive_residual_basis(
        h_sensitive,
        h_context,
        protected_rank=int(a.protected_rank),
        sensitive_rank=int(a.sensitive_rank),
    )

    # One shared sensitive residual basis for all sensitive rows.
    emb_delta = core.SelectedRowDelta(
        len(selected_ids),
        int(input_layer.weight.shape[1]),
        direction_basis=b_s,
        device=input_layer.weight.device,
    )
    head_delta = core.SelectedRowDelta(
        len(selected_ids),
        int(output_layer.weight.shape[1]),
        direction_basis=b_s,
        device=output_layer.weight.device,
    )

    base_logits = core.cache_base_logits(
        model,
        tok,
        sensitive_cases,
        device,
        batch_size=int(a.cache_batch_size),
    )

    parameters = list(emb_delta.parameters()) + list(head_delta.parameters())
    opt = torch.optim.AdamW(parameters, lr=float(a.lr), weight_decay=0.0)
    sampler = core.IndexSampler(len(sensitive_cases), int(a.batch_size), int(a.seed))

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_hook = directional_v1.register_input_embedding_delta_hook(
        input_layer, selected_ids, emb_delta.effective_delta
    )
    head_hook = core.register_output_delta_hook(
        output_layer, selected_ids, head_delta.effective_delta
    )
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, int(a.steps) + 1):
                idx = sampler.next()
                batch = [sensitive_cases[i] for i in idx]
                opt.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(logits, tids)
                kl = full_vocab_kl(base_logits[idx], logits)
                emb_now = emb_delta.effective_delta()
                head_now = head_delta.effective_delta()
                l2 = emb_now.square().mean() + head_now.square().mean()
                loss = (
                    float(a.ga_weight) * ga
                    + float(a.kl_weight) * kl
                    + float(a.delta_l2) * l2
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Stage-1 loss at step {step}")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, float(a.grad_clip)
                ) if float(a.grad_clip) > 0 else None
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite Stage-1 gradient at step {step}")
                opt.step()

                if step == 1 or step % 25 == 0 or step == int(a.steps):
                    row = {
                        "step": int(step),
                        "loss": float(loss.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "full_vocab_kl_base_to_edited": float(kl.detach().cpu()),
                        "delta_l2": float(l2.detach().cpu()),
                        "embedding_delta_norm": float(emb_delta.effective_delta().detach().norm().cpu()),
                        "lm_head_delta_norm": float(head_delta.effective_delta().detach().norm().cpu()),
                        "lora_used": False,
                        "heldout_probes_seen": 0,
                    }
                    if grad_norm is not None:
                        row["grad_norm"] = float(grad_norm.detach().cpu())
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
    finally:
        head_hook.remove()
        emb_hook.remove()
    del opt

    final_emb = emb_delta.effective_delta().detach().clone()
    final_head = head_delta.effective_delta().detach().clone()
    directional_v1.materialize_input_delta(input_layer, selected_ids, final_emb)
    core.materialize_output_delta(output_layer, selected_ids, final_head)

    atomic, atomic_ids = evaluate_atomic_cases(
        model,
        tok,
        sensitive_cases,
        llama_like=llama_like,
        device=device,
        batch_size=int(a.cache_batch_size),
    )
    pass_positions = [
        i for i, value in enumerate(atomic.tolist())
        if float(value) >= float(a.atomic_margin)
    ]
    fail_positions = [
        i for i, value in enumerate(atomic.tolist())
        if float(value) < float(a.atomic_margin)
    ]

    # Direct MCF preference is diagnostic only; Stage-1 optimization never sees held-out probes.
    instances = shared_stage2.mcf_instances(records)
    record_margins = shared_stage2.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        int(a.cache_batch_size),
        sensitive_field="target_true",
        reference_field="target_new",
    ).detach().float().cpu()

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 2,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "geometry": geometry,
        "selected_token_ids": selected_ids,
        "selected_row_count": len(selected_ids),
        "stage1_basis_definition": "B_S=rowspace(H_S-Proj_BNS(H_S))",
        "protected_context_definition": "preceding non-padding hidden states from token_index==0 direct prompts only; no teacher-forced target_true prefix states",
        "embedding_parameterization": "Delta E_A = C_E B_S",
        "lm_head_parameterization": "Delta W_A = C_W B_S",
        "embedding_trainable_parameters": int(emb_delta.trainable_parameter_count),
        "lm_head_trainable_parameters": int(head_delta.trainable_parameter_count),
        "transformer_trainable_parameters": 0,
        "ga_weight": float(a.ga_weight),
        "kl_weight": float(a.kl_weight),
        "delta_l2": float(a.delta_l2),
        "steps": int(a.steps),
        "lr": float(a.lr),
        "atomic_margin_definition": "max_other_logit - target_true_logit",
        "atomic_margin_required": float(a.atomic_margin),
        "atomic_case_count": len(sensitive_cases),
        "atomic_success_count": len(pass_positions),
        "atomic_failure_count": len(fail_positions),
        "atomic_success_positions": pass_positions,
        "atomic_failure_positions": fail_positions,
        "atomic_min_margin": float(atomic.min().item()),
        "atomic_mean_margin": float(atomic.mean().item()),
        "direct_mcf_record_success_count_margin_ge_0": int((record_margins >= 0.0).sum().item()),
        "direct_mcf_record_min_margin": float(record_margins.min().item()),
        "final_embedding_delta_norm": float(final_emb.norm().cpu()),
        "final_lm_head_delta_norm": float(final_head.norm().cpu()),
        "lora_used": False,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "training_visible_sha256": sha256_file(visible_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "stage1_config.json", config)
    print(json.dumps(config, indent=2))
    print(
        f"Protected-subspace Stage 1: atomic successes={len(pass_positions)}/"
        f"{len(sensitive_cases)}, failures={len(fail_positions)}, "
        f"min margin={config['atomic_min_margin']:.6f}"
    )
    print(f"Stage-1 checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
