#!/usr/bin/env python3
"""SURE-MQuAKE V8.0: forget-only contextual low-rank MLP repair.

V7.1--V7.4 showed that globally changing LM-head token rows remains utility
limited even when the update is sparse and prompt protected. V8 therefore keeps
both the LM head and embeddings exact Base and inserts a tiny low-rank update in
one late Llama MLP down projection during optimization. The update is merged
into that projection before saving, so the resulting checkpoint is an ordinary
Transformers model with no runtime hook dependency.

Data firewall: only direct requested_rewrite prompts from the locked 50 forget
instances are used. Benchmark retain, AtomicGen, multihop, counterfactual
(target_new), and PPL data are never loaded for optimization or selection.

Preservation is forget-visible only: full-vocabulary non-target KL anchors each
answer-decision prompt to Base, Base-safe answer decisions are constrained to
remain safe, and adapter output on non-final prompt tokens is penalized. A
candidate must satisfy every direct sensitive-token decision before selection.
The feasible low-rank update is then scaled toward Base with repeated exact
model audits and finally merged into the chosen MLP down_proj weight.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v8.0-contextual-MLP-low-rank-repair"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=4.0)
    p.add_argument("--layer-index", type=int, default=-4,
                   help="Llama decoder layer index; negative values count from the end")
    p.add_argument("--forget-hinge-weight", type=float, default=10.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=2.5)
    p.add_argument("--safe-hinge-weight", type=float, default=10.0)
    p.add_argument("--non-target-kl-weight", type=float, default=2.0)
    p.add_argument("--prompt-adapter-weight", type=float, default=1.0)
    p.add_argument("--factor-l2-lambda", type=float, default=1e-4)
    p.add_argument("--active-target-margin", type=float, default=0.05)
    p.add_argument("--safe-margin-floor", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--audit-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--bisection-steps", type=int, default=14)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def chunks(values: Sequence[int], size: int) -> List[Sequence[int]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def resolve_down_proj(model: torch.nn.Module, requested_index: int) -> Tuple[torch.nn.Linear, int, int]:
    try:
        layers = model.model.layers
    except Exception as exc:  # pragma: no cover - architecture guard
        raise RuntimeError("V8 currently expects a Llama-like model.model.layers stack") from exc
    n_layers = len(layers)
    index = requested_index if requested_index >= 0 else n_layers + requested_index
    if index < 0 or index >= n_layers:
        raise ValueError(f"layer-index {requested_index} resolves outside 0..{n_layers - 1}")
    module = layers[index].mlp.down_proj
    if not isinstance(module, torch.nn.Linear):
        raise TypeError("chosen mlp.down_proj is not torch.nn.Linear")
    return module, index, n_layers


class LowRankOutputHook(torch.nn.Module):
    """LoRA-style additive update for one frozen Linear module."""

    def __init__(self, module: torch.nn.Linear, rank: int, alpha: float, device: torch.device):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.multiplier = 1.0
        self.B = torch.nn.Parameter(torch.empty(rank, module.in_features, dtype=torch.float32, device=device))
        self.A = torch.nn.Parameter(torch.zeros(module.out_features, rank, dtype=torch.float32, device=device))
        torch.nn.init.normal_(self.B, mean=0.0, std=1.0 / math.sqrt(module.in_features))
        self.last_update: torch.Tensor | None = None

    def hook(self, _module: torch.nn.Module, inputs: Tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        x = inputs[0].float()
        z = F.linear(x, self.B)
        update = F.linear(z, self.A) * (self.scale * float(self.multiplier))
        self.last_update = update
        return output + update.to(dtype=output.dtype)

    def merged_delta_weight(self, multiplier: float = 1.0) -> torch.Tensor:
        return (self.A @ self.B) * (self.scale * float(multiplier))


def encode_case_batch(tok: Any, cases: Sequence[mquake.PredictionCase], indices: Sequence[int], device: torch.device):
    batch_cases = [cases[i] for i in indices]
    encoded = tok([c.prompt for c in batch_cases], padding=True, return_tensors="pt").to(device)
    return batch_cases, encoded


@torch.no_grad()
def final_logits(model: torch.nn.Module, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
    output = model(**encoded, use_cache=False)
    last = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(last.shape[0], device=last.device)
    return output.logits[rows, last, :].float()


def margins_from_logits(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    target = logits.gather(1, target_ids[:, None]).squeeze(1)
    competitors = logits.clone()
    competitors.scatter_(1, target_ids[:, None], float("-inf"))
    best = competitors.max(dim=-1).values
    return best - target


def non_target_kl(base_logits: torch.Tensor, current_logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """KL(Base_non-target || Current_non-target), target removed and renormalized."""
    b = base_logits.float().clone()
    c = current_logits.float().clone()
    b.scatter_(1, target_ids[:, None], float("-inf"))
    c.scatter_(1, target_ids[:, None], float("-inf"))
    q_log = F.log_softmax(b, dim=-1)
    p_log = F.log_softmax(c, dim=-1)
    q = q_log.exp()
    # target entries are exactly zero after softmax(-inf), so nans are avoided.
    return (q * (q_log - p_log)).nan_to_num(0.0).sum(dim=-1).mean()


@torch.no_grad()
def build_base_cache(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_ids_all: List[torch.Tensor] = []
    margins_all: List[torch.Tensor] = []
    base_logits_cpu: List[torch.Tensor] = []
    for ids in chunks(list(range(len(cases))), batch_size):
        batch_cases, encoded = encode_case_batch(tok, cases, ids, device)
        logits = final_logits(model, encoded)
        target_ids = mquake.official_target_ids(
            tok, [c.target_text for c in batch_cases], llama_like=llama_like, device=device
        )
        target_ids_all.append(target_ids.detach().cpu())
        margins_all.append(margins_from_logits(logits, target_ids).detach().cpu())
        base_logits_cpu.append(logits.detach().to(dtype=torch.float16, device="cpu"))
    return (
        torch.cat(target_ids_all),
        torch.cat(margins_all),
        torch.cat(base_logits_cpu),
    )


@torch.no_grad()
def audit_all(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    target_ids_cpu: torch.Tensor,
    base_active_cpu: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    active_margin: float,
    safe_floor: float,
) -> Dict[str, Any]:
    margins: List[torch.Tensor] = []
    for ids in chunks(list(range(len(cases))), batch_size):
        _, encoded = encode_case_batch(tok, cases, ids, device)
        logits = final_logits(model, encoded)
        target_ids = target_ids_cpu[list(ids)].to(device)
        margins.append(margins_from_logits(logits, target_ids).detach().cpu())
    m = torch.cat(margins).float()
    active_mask = base_active_cpu.bool()
    safe_mask = ~active_mask
    return {
        "official_active_sensitive_token_count": int((m <= 0.0).sum().item()),
        "base_active_buffer_unmet_count": int((m[active_mask] < float(active_margin)).sum().item()),
        "base_safe_failed_count": int((m[safe_mask] <= float(safe_floor)).sum().item()),
        "minimum_competitor_minus_sensitive_margin": float(m.min().item()),
        "mean_competitor_minus_sensitive_margin": float(m.mean().item()),
    }


def passes(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics["official_active_sensitive_token_count"]) == 0
        and int(metrics["base_active_buffer_unmet_count"]) == 0
        and int(metrics["base_safe_failed_count"]) == 0
    )


def main() -> None:
    a = parse_args()
    if a.steps <= 0 or a.batch_size <= 0 or a.audit_every <= 0:
        raise ValueError("steps/batch-size/audit-every must be positive")
    if a.lr <= 0 or a.rank <= 0 or a.lora_alpha <= 0:
        raise ValueError("lr/rank/lora-alpha must be positive")
    if min(a.forget_hinge_weight, a.hardest_forget_hinge_weight, a.safe_hinge_weight,
           a.non_target_kl_weight, a.prompt_adapter_weight, a.factor_l2_lambda) < 0:
        raise ValueError("loss weights must be non-negative")
    if a.active_target_margin <= 0 or a.safe_margin_floor < 0:
        raise ValueError("active margin must be positive and safe floor non-negative")

    compat.install()
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.repair_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, split_manifest = locked.load_locked_records(visible_path, manifest_path, a.forget_num, a.seed)

    model_args = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)

    cases = v7.direct_rewrite_cases(records, tok, llama_like=llama_like)
    down_proj, layer_index, layer_count = resolve_down_proj(model, a.layer_index)
    lm_head = model.get_output_embeddings().weight
    input_weight = model.get_input_embeddings().weight
    lm_ptr, lm_version = int(lm_head.data_ptr()), int(lm_head._version)
    in_ptr, in_version = int(input_weight.data_ptr()), int(input_weight._version)

    # Base cache is collected with no adapter installed.
    target_ids_cpu, base_margins_cpu, base_logits_cpu = build_base_cache(
        model, tok, cases, device=device, llama_like=llama_like, batch_size=a.batch_size
    )
    base_active_cpu = base_margins_cpu <= 0.0
    if not base_active_cpu.any():
        raise RuntimeError("protected Base has no active sensitive token decisions")

    repair = LowRankOutputHook(down_proj, a.rank, a.lora_alpha, device)
    handle = down_proj.register_forward_hook(repair.hook)
    optimizer = torch.optim.AdamW(repair.parameters(), lr=a.lr, weight_decay=0.0)
    rng = random.Random(a.seed)
    logs: List[Dict[str, Any]] = []
    feasible_state: Dict[str, torch.Tensor] | None = None
    feasible_step: int | None = None
    feasible_metrics: Dict[str, Any] | None = None

    order = list(range(len(cases)))
    cursor = len(order)
    last_loss_parts: Dict[str, float] = {}

    for step in range(1, a.steps + 1):
        if cursor + a.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        ids = order[cursor:cursor + a.batch_size]
        cursor += a.batch_size

        batch_cases, encoded = encode_case_batch(tok, cases, ids, device)
        target_ids = target_ids_cpu[ids].to(device)
        base_active = base_active_cpu[ids].to(device)
        base_logits = base_logits_cpu[ids].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        output = model(**encoded, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(ids), device=device)
        logits = output.logits[rows, last, :].float()
        margins = margins_from_logits(logits, target_ids)

        active_error = torch.relu(float(a.active_target_margin) - margins[base_active])
        active_loss = active_error.square().mean() if active_error.numel() else logits.new_zeros(())
        active_hard = active_error.square().max() if active_error.numel() else logits.new_zeros(())
        safe_mask = ~base_active
        safe_error = torch.relu(float(a.safe_margin_floor) - margins[safe_mask])
        safe_loss = safe_error.square().mean() if safe_error.numel() else logits.new_zeros(())
        kl = non_target_kl(base_logits, logits, target_ids)

        if repair.last_update is None:
            raise RuntimeError("V8 adapter hook did not capture its update")
        adapter_update = repair.last_update
        attn = encoded["attention_mask"].bool().clone()
        # Preserve every non-final valid token; the final token is precisely the
        # answer-decision context that may need a contextual change.
        attn[rows, last] = False
        prompt_penalty = (
            adapter_update[attn].float().square().mean()
            if attn.any() else logits.new_zeros(())
        )
        factor_l2 = repair.A.square().sum() + repair.B.square().sum()
        loss = (
            a.forget_hinge_weight * active_loss
            + a.hardest_forget_hinge_weight * active_hard
            + a.safe_hinge_weight * safe_loss
            + a.non_target_kl_weight * kl
            + a.prompt_adapter_weight * prompt_penalty
            + a.factor_l2_lambda * factor_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V8 loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(repair.parameters(), a.grad_clip) if a.grad_clip > 0 else None
        optimizer.step()

        last_loss_parts = {
            "loss": float(loss.detach().cpu()),
            "active_hinge": float(active_loss.detach().cpu()),
            "hardest_active_hinge": float(active_hard.detach().cpu()),
            "safe_hinge": float(safe_loss.detach().cpu()),
            "non_target_kl": float(kl.detach().cpu()),
            "prompt_adapter_penalty": float(prompt_penalty.detach().cpu()),
            "factor_l2": float(factor_l2.detach().cpu()),
            "grad_norm": None if grad_norm is None else float(grad_norm.detach().cpu()),
        }

        if step == 1 or step % a.audit_every == 0 or step == a.steps:
            metrics = audit_all(
                model, tok, cases, target_ids_cpu, base_active_cpu,
                device=device, batch_size=a.batch_size,
                active_margin=a.active_target_margin, safe_floor=a.safe_margin_floor,
            )
            delta_norm = float(repair.merged_delta_weight().norm().detach().cpu())
            row = {"step": step, **last_loss_parts, **metrics, "merged_delta_weight_norm": delta_norm}
            logs.append(row)
            print(
                f"v80-step={step} active={metrics['official_active_sensitive_token_count']} "
                f"active_buffer_unmet={metrics['base_active_buffer_unmet_count']} "
                f"safe_failed={metrics['base_safe_failed_count']} "
                f"min_margin={metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={last_loss_parts['non_target_kl']:.6g} deltaW_norm={delta_norm:.6g}"
            )
            if passes(metrics):
                feasible_state = {
                    "A": repair.A.detach().clone(),
                    "B": repair.B.detach().clone(),
                }
                feasible_step = step
                feasible_metrics = dict(metrics)
                break

    root = gagd.resolve_output_path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    v7.write_jsonl(root / "train_log.jsonl", logs)

    if feasible_state is None or feasible_step is None or feasible_metrics is None:
        handle.remove()
        write_json(root / "failure.json", {
            "status": "FAILED_NO_FEASIBLE_CONTEXTUAL_MLP_REPAIR",
            "method": METHOD,
            "seed": int(a.seed),
            "layer_index": int(layer_index),
            "rank": int(a.rank),
            "base_active_sensitive_token_count": int(base_active_cpu.sum().item()),
            "last_logged_metrics": logs[-1] if logs else None,
        })
        raise RuntimeError("V8 did not find a forget-only feasible contextual MLP repair")

    with torch.no_grad():
        repair.A.copy_(feasible_state["A"])
        repair.B.copy_(feasible_state["B"])

    # Exact audit bisection over one scalar multiplier. low is Base/infeasible,
    # high is the known feasible adapter.
    low, high = 0.0, 1.0
    bisection_log: List[Dict[str, Any]] = []
    repair.multiplier = 1.0
    high_metrics = audit_all(
        model, tok, cases, target_ids_cpu, base_active_cpu,
        device=device, batch_size=a.batch_size,
        active_margin=a.active_target_margin, safe_floor=a.safe_margin_floor,
    )
    if not passes(high_metrics):
        handle.remove()
        raise RuntimeError("V8 feasible adapter stopped passing before bisection")

    for iteration in range(1, a.bisection_steps + 1):
        mid = 0.5 * (low + high)
        repair.multiplier = mid
        metrics = audit_all(
            model, tok, cases, target_ids_cpu, base_active_cpu,
            device=device, batch_size=a.batch_size,
            active_margin=a.active_target_margin, safe_floor=a.safe_margin_floor,
        )
        ok = passes(metrics)
        bisection_log.append({"iteration": iteration, "multiplier": mid, "passed": ok, **metrics})
        if ok:
            high = mid
        else:
            low = mid

    final_multiplier = float(high)
    repair.multiplier = final_multiplier
    hook_final_metrics = audit_all(
        model, tok, cases, target_ids_cpu, base_active_cpu,
        device=device, batch_size=a.batch_size,
        active_margin=a.active_target_margin, safe_floor=a.safe_margin_floor,
    )
    if not passes(hook_final_metrics):
        handle.remove()
        raise RuntimeError("V8 final hook-scaled adapter failed exact audit")

    # Merge into the actual BF16/FP weight and remove the hook.
    delta_w = repair.merged_delta_weight(final_multiplier).detach()
    with torch.no_grad():
        down_proj.weight.add_(delta_w.to(device=down_proj.weight.device, dtype=down_proj.weight.dtype))
    handle.remove()
    repair.last_update = None

    merged_metrics = audit_all(
        model, tok, cases, target_ids_cpu, base_active_cpu,
        device=device, batch_size=a.batch_size,
        active_margin=a.active_target_margin, safe_floor=a.safe_margin_floor,
    )
    if not passes(merged_metrics):
        raise RuntimeError("V8 BF16-merged MLP checkpoint failed exact audit")
    if int(lm_head.data_ptr()) != lm_ptr or int(lm_head._version) != lm_version:
        raise RuntimeError("V8 modified LM head")
    if int(input_weight.data_ptr()) != in_ptr or int(input_weight._version) != in_version:
        raise RuntimeError("V8 modified input embeddings")

    ckpt = root / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    v7.write_jsonl(root / "bisection_log.jsonl", bisection_log)

    summary = {
        "status": "PASS_V80_CONTEXTUAL_MLP_MINIMAL_SCALE",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "base_active_sensitive_token_count": int(base_active_cpu.sum().item()),
        "decoder_layer_count": int(layer_count),
        "edited_layer_index": int(layer_index),
        "edited_parameter": f"model.layers.{layer_index}.mlp.down_proj.weight",
        "rank": int(a.rank),
        "lora_alpha": float(a.lora_alpha),
        "trainable_parameter_count": int(repair.A.numel() + repair.B.numel()),
        "optimizer_feasible_step": int(feasible_step),
        "first_feasible_metrics": feasible_metrics,
        "minimal_scale_multiplier": final_multiplier,
        "merged_delta_weight_norm": float(delta_w.norm().detach().cpu()),
        "materialized_bf16_metrics": merged_metrics,
        "lm_head_exact_base": True,
        "input_embeddings_exact_base": True,
        "training_data_access": {
            "forget_instances": int(a.forget_num),
            "forget_atomic_facts": len(records),
            "prompt_types": ["requested_rewrite"],
            "benchmark_retain_instances": 0,
            "atomic_questions": 0,
            "multihop_questions": 0,
            "benchmark_counterfactual_targets": 0,
            "PPL": False,
        },
        "checkpoint_selection_uses_retain_or_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(root / "config_used.json", {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        **vars(a),
        "repair_visible_path_resolved": str(visible_path),
        "split_manifest_resolved": str(manifest_path),
        "split_sampling": split_manifest.get("sampling"),
        "parameter_scope": f"rank-{a.rank} merged update to model.layers.{layer_index}.mlp.down_proj only; LM head/input embeddings/all other weights exact Base",
        "selection_definition": "first all-visible forget-only feasible adapter followed by exact all-visible scalar bisection toward Base; no retain/PPL/AtomicGen/multihop selection",
    })

    print("===== SURE-MQuAKE V8.0 CONTEXTUAL MLP LOW-RANK REPAIR =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"base_active={int(base_active_cpu.sum().item())} layer={layer_index}/{layer_count} "
        f"rank={a.rank} trainable_params={repair.A.numel() + repair.B.numel()}"
    )
    print(
        f"optimizer_feasible_step={feasible_step} final_multiplier={final_multiplier:.8f} "
        f"merged_deltaW_norm={float(delta_w.norm().detach().cpu()):.6g}"
    )
    print(
        f"BF16 official_active={merged_metrics['official_active_sensitive_token_count']} "
        f"active_buffer_unmet={merged_metrics['base_active_buffer_unmet_count']} "
        f"safe_failed={merged_metrics['base_safe_failed_count']} "
        f"min_margin={merged_metrics['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
