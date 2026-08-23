#!/usr/bin/env python3
"""MQuAKE SURE v5 Stage 2A: restore Stage-1 embedding rows to base, then GD.

This phase starts from the *v3 Stage-1 checkpoint* (untied E/W).  It keeps the
Stage-1 LM-head edit W1 exactly frozen, restores every Stage-1-sensitive input
embedding row to its base-model value, then learns only the answer-prefix input
rows A_E with a preservation objective.

The preservation signal is the canonical non-sensitive-vocabulary GD loss:
for each training-visible target_true PredictionCase, match the base model over
all vocabulary logits except the *current* sensitive target token.  Therefore
this phase does not directly optimize the sensitive target back upward.

    E_A <- E0_A                         (exact restoration)
    E_AE = E0_AE + Delta E_AE           (trainable compensation)
    L_E = KL_non_sensitive(Base || Current) + lambda ||Delta E_AE||^2

where A_E contains only sensitive answer tokens that occur in a later
teacher-forced answer prefix.  All other embedding rows, the entire LM head,
and the transformer remain frozen.  No retain example, target_new, AtomicGen,
or held-out MQuAKE field is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from tqdm import tqdm

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import mquake_sure_stage1_directional as v2
import mquake_sure_stage1_directional_v3 as v3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--model-path", required=True, help="v3 Stage-1 checkpoint")
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--anchor-weight", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def answer_prefix_ids(records, tok, *, llama_like: bool, special: set[int]) -> List[int]:
    ids: set[int] = set()
    for record in records:
        rr = record["requested_rewrite"]
        answer = str(rr["target_true"]["str"])
        tids = core.answer_token_ids(tok, answer, llama_like=llama_like)
        # Only tokens that become input while predicting a later answer token.
        ids.update(int(x) for x in tids[:-1] if int(x) not in special)
    return sorted(ids)


def masked_gd_full(
    model,
    tok,
    cases,
    base_logits_cpu: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            cur = core.forward_last_logits(model, tok, batch, device)
            ref = base_logits_cpu[start : start + len(batch)].to(device=device)
            tids = target_ids[start : start + len(batch)]
            loss = core.gd_non_sensitive_kl(cur, ref, tids)
            total += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
    return total / max(count, 1)


def capture_state(module: torch.nn.Module):
    return [p.detach().clone() for p in module.parameters()]


@torch.no_grad()
def restore_state(module: torch.nn.Module, state) -> None:
    for p, saved in zip(module.parameters(), state):
        p.copy_(saved.to(device=p.device, dtype=p.dtype))


def main() -> None:
    a = parse_args()
    if min(a.steps, a.batch_size, a.cache_batch_size, a.check_every) <= 0:
        raise ValueError("steps/batches/check interval must be positive")
    if a.learning_rate <= 0 or a.anchor_weight < 0:
        raise ValueError("invalid GD optimization settings")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    # Use the same locked direct-only protocol validator as Stage 1 v3.
    ns_locked = argparse.Namespace(
        training_visible_path=a.training_visible_path,
        split_manifest=a.split_manifest,
        seed=a.seed,
        forget_num=a.forget_num,
    )
    records, manifest, visible_path, manifest_path = v2.load_locked(ns_locked)

    # ---- Base reference: cache logits and the embedding rows that Stage 1 edited. ----
    base_ns = argparse.Namespace(
        model_path=a.base_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    base_model, base_tok = gagd.load_model_and_tokenizer(base_ns, for_training=False)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_tok.padding_side = "right"
    base_device = gagd.first_device(base_model)
    base_llama = is_llama_like(base_model, base_tok)
    base_cases = core.expand_sensitive_cases(
        records, base_tok, sensitive_field="target_true", llama_like=base_llama
    )
    if not base_cases:
        raise RuntimeError("no MQuAKE sensitive PredictionCases")
    base_target_ids = core.official_target_ids(
        base_tok, base_cases, llama_like=base_llama, device=base_device
    )
    special = set(gagd.special_token_ids(base_tok))
    sensitive_ids = sorted(
        set(int(x) for x in base_target_ids.detach().cpu().tolist()) - special
    )
    if not sensitive_ids:
        raise RuntimeError("no sensitive content rows")
    prefix_ids = answer_prefix_ids(records, base_tok, llama_like=base_llama, special=special)
    prefix_ids = [x for x in prefix_ids if x in set(sensitive_ids)]
    if not prefix_ids:
        raise RuntimeError("no sensitive answer-prefix embedding rows found")

    base_logits = core.cache_base_logits(
        base_model, base_tok, base_cases, base_device, batch_size=a.cache_batch_size
    )
    base_input = base_model.get_input_embeddings().weight
    sid_tensor_base = torch.tensor(sensitive_ids, dtype=torch.long, device=base_input.device)
    base_sensitive_rows = base_input.index_select(0, sid_tensor_base).detach().cpu().clone()

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Load Stage-1 v3 model, restore E_A exactly, keep W1 untouched. ----
    stage_ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(stage_ns, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if len(cases) != len(base_cases):
        raise RuntimeError("base and Stage-1 PredictionCase count mismatch")
    target_ids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    if target_ids.detach().cpu().tolist() != base_target_ids.detach().cpu().tolist():
        raise RuntimeError("base and Stage-1 target tokenization mismatch")

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("v5 requires untied E/W")

    frozen_before = v3.transformer_hash(model, input_layer.weight, output_layer.weight)
    sid_tensor = torch.tensor(sensitive_ids, dtype=torch.long, device=input_layer.weight.device)
    with torch.no_grad():
        input_layer.weight.index_copy_(
            0,
            sid_tensor,
            base_sensitive_rows.to(device=input_layer.weight.device, dtype=input_layer.weight.dtype),
        )

    # Delta is defined relative to the restored base rows and only for A_E.
    emb_delta = core.SelectedRowDelta(
        len(prefix_ids),
        int(input_layer.weight.shape[1]),
        direction_basis=None,
        device=device,
    )
    input_hook = v2.register_input_delta_hook(input_layer, prefix_ids, emb_delta)
    params = list(emb_delta.parameters())
    if a.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=a.learning_rate)
    elif a.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=a.learning_rate)
    else:
        optimizer = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.0)

    sampler = core.IndexSampler(len(cases), a.batch_size, a.seed + 500003)
    initial_kl = masked_gd_full(
        model, tok, cases, base_logits, target_ids, device, a.cache_batch_size
    )
    initial_gate = v3.gate_model(
        model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin
    )
    best_state = capture_state(emb_delta)
    best_kl = float(initial_kl)
    best_norm = 0.0
    best_step = 0
    best_gate = initial_gate
    checks: List[Dict[str, Any]] = []

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "embedding_gd_log.jsonl"

    with log_path.open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, a.steps + 1), desc="MQuAKE v5 Stage2A restored-E GD"):
            idx = sampler.next()
            batch = [cases[i] for i in idx]
            optimizer.zero_grad(set_to_none=True)
            cur = core.forward_last_logits(model, tok, batch, device)
            ref = base_logits[idx].to(device=device)
            tids = target_ids[idx]
            gd = core.gd_non_sensitive_kl(cur, ref, tids)
            delta_rows = emb_delta.effective_delta()
            anchor = delta_rows.square().mean()
            loss = gd + float(a.anchor_weight) * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite embedding GD loss at step {step}")
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite embedding GD gradient at step {step}")
            optimizer.step()

            if step == 1 or step % a.check_every == 0 or step == a.steps:
                full_kl = masked_gd_full(
                    model, tok, cases, base_logits, target_ids, device, a.cache_batch_size
                )
                gate = v3.gate_model(
                    model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin
                )
                norm = float(emb_delta.effective_delta().detach().float().norm().cpu())
                # Select solely by the training-visible preservation objective;
                # Stage 2B will recompute and repair every resulting forget failure.
                if (float(full_kl), norm) < (best_kl, best_norm):
                    best_kl = float(full_kl)
                    best_norm = norm
                    best_step = int(step)
                    best_state = capture_state(emb_delta)
                    best_gate = gate
                row = {
                    "step": int(step),
                    "batch_non_sensitive_kl": float(gd.detach().cpu()),
                    "anchor": float(anchor.detach().cpu()),
                    "full_non_sensitive_kl": float(full_kl),
                    "embedding_delta_norm": norm,
                    "direct_gate": gate,
                    "best_step": int(best_step),
                }
                checks.append(row)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                print(
                    "Stage2A-v5 step {s}: GD_KL={kl:.6g} gate={p}/{t} failed={f} ||dE||={n:.6g} best={b}".format(
                        s=step,
                        kl=full_kl,
                        p=gate["passed"],
                        t=gate["total"],
                        f=gate["failed"],
                        n=norm,
                        b=best_step,
                    )
                )

    restore_state(emb_delta, best_state)
    chosen_delta = emb_delta.effective_delta().detach()
    input_hook.remove()
    v2.materialize_input_delta(input_layer, prefix_ids, chosen_delta)
    model.eval()

    final_kl = masked_gd_full(
        model, tok, cases, base_logits, target_ids, device, a.cache_batch_size
    )
    final_gate = v3.gate_model(
        model, tok, cases, llama_like, device, a.cache_batch_size, a.constraint_margin
    )
    frozen_after = v3.transformer_hash(model, input_layer.weight, output_layer.weight)
    transformer_exact = frozen_before == frozen_after
    if not transformer_exact:
        raise RuntimeError("transformer/non-vocabulary parameters changed during embedding GD")

    ckpt = out / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    report: Dict[str, Any] = {
        "schema_version": 1,
        "method": "MQuAKE SURE v5 Stage2A Base-Restored Sparse Embedding GD",
        "source_protocol": manifest.get("protocol"),
        "base_model_path": a.base_model_path,
        "stage1_model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "prediction_case_count": len(cases),
        "stage1_sensitive_embedding_rows_restored_to_base": sensitive_ids,
        "stage1_sensitive_embedding_row_count": len(sensitive_ids),
        "trainable_prefix_embedding_rows": prefix_ids,
        "trainable_prefix_embedding_row_count": len(prefix_ids),
        "all_other_embedding_rows_untouched_in_stage2a": True,
        "lm_head_frozen_in_stage2a": True,
        "transformer_frozen_in_stage2a": True,
        "objective": "KL(Base||Current) over vocabulary excluding only the current sensitive target + base-anchor L2",
        "target_new_seen": False,
        "official_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "initial_non_sensitive_kl": float(initial_kl),
        "final_non_sensitive_kl": float(final_kl),
        "initial_direct_gate": initial_gate,
        "best_direct_gate": best_gate,
        "final_direct_gate": final_gate,
        "best_step": int(best_step),
        "embedding_delta_norm_from_base": float(chosen_delta.float().norm().cpu()),
        "learning_rate": float(a.learning_rate),
        "anchor_weight": float(a.anchor_weight),
        "steps_requested": int(a.steps),
        "checks": checks,
        "transformer_exact": bool(transformer_exact),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out / "embedding_gd_summary.json", report)
    torch.save(
        {
            "sensitive_rows_restored": sensitive_ids,
            "trainable_prefix_rows": prefix_ids,
            "embedding_delta_from_base": chosen_delta.float().cpu(),
        },
        out / "embedding_gd_state.pt",
    )

    print("Stage2A-v5 checkpoint:", ckpt)
    print("Restored Stage1-sensitive E rows:", len(sensitive_ids))
    print("Trainable answer-prefix E rows:", len(prefix_ids))
    print("Initial non-sensitive KL:", initial_kl)
    print("Final non-sensitive KL:", final_kl)
    print("Final direct gate: {}/{} pass".format(final_gate["passed"], final_gate["total"]))
    print("||E_final - E_base|| on A_E:", float(chosen_delta.float().norm().cpu()))
    print("Transformer exact:", transformer_exact)


if __name__ == "__main__":
    main()
