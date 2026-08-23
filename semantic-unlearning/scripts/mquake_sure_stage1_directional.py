#!/usr/bin/env python3
"""MQuAKE Stage 1: untied directional SURE on sensitive embedding/head rows.

The transformer is frozen exactly. The tied Llama vocabulary matrix is cloned
into independent input-embedding and LM-head weights before learning.

A fixed sensitive hidden basis B_S is constructed without benchmark-retain or
held-out information:
  1. H_F: final hidden states for every training-visible teacher-forced
     target_true PredictionCase.
  2. H_P: preceding (non-prediction) prompt-token hidden states from the same
     training-visible prompts.
  3. B_P: a truncated orthonormal basis of H_P.
  4. R_F = H_F - Proj_{B_P}(H_F).
  5. B_S: a fixed-rank orthonormal basis of R_F.

Only sensitive vocabulary rows A can move, and their deltas are constrained to
span(B_S):
    Delta E_A = C_E B_S
    Delta W_A = C_W B_S

No transformer parameter, non-sensitive vocabulary row, benchmark-retain item,
held-out probe, target_new value, rank sweep, or scale sweep is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--protection-weight", type=float, default=1.0)
    p.add_argument("--direction-rank", type=int, default=4)
    p.add_argument("--protected-rank", type=int, default=32)
    p.add_argument("--protected-context-tokens", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    visible = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records = json.loads(visible.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != a.forget_num:
        raise RuntimeError(f"Expected {a.forget_num} training-visible forget records")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != a.forget_num:
        raise RuntimeError("manifest forget count mismatch")
    expected = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual = [int(x.get("case_id", -1)) for x in records]
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
    return records, manifest, visible, manifest_path


def full_base_to_current_kl(current_logits: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
    cur = current_logits.float()
    ref = base_logits.to(device=cur.device, dtype=torch.float32)
    if cur.shape != ref.shape:
        raise ValueError("full KL requires equal logit shapes")
    ref_logp = F.log_softmax(ref, dim=-1)
    cur_logp = F.log_softmax(cur, dim=-1)
    return (ref_logp.exp() * (ref_logp - cur_logp)).sum(dim=-1).mean()


@torch.no_grad()
def collect_preceding_context_hidden(
    model,
    tok,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    *,
    batch_size: int,
    last_n: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1]
        lengths = encoded["attention_mask"].sum(dim=1)
        for i, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            final_position = length - 1
            if final_position <= 0:
                continue
            begin = max(1, final_position - int(last_n))
            if begin >= final_position:
                begin = max(0, final_position - 1)
            if begin < final_position:
                rows.append(hidden[i, begin:final_position, :].float().detach())
    if not rows:
        raise RuntimeError("could not construct non-sensitive context hidden rows")
    return torch.cat(rows, dim=0)


def make_basis(
    model,
    tok,
    cases,
    device,
    *,
    cache_batch_size: int,
    protected_context_tokens: int,
    protected_rank: int,
    direction_rank: int,
):
    sensitive_hidden = core.forward_last_hidden(
        model, tok, cases, device, batch_size=cache_batch_size
    ).float()
    context_hidden = collect_preceding_context_hidden(
        model,
        tok,
        cases,
        device,
        batch_size=cache_batch_size,
        last_n=protected_context_tokens,
    )
    protected_basis = core.orthonormal_row_basis(
        context_hidden, max_rank=protected_rank
    ).to(device=device, dtype=torch.float32)
    if protected_basis.numel():
        residual = sensitive_hidden - (
            sensitive_hidden @ protected_basis.transpose(0, 1)
        ) @ protected_basis
    else:
        residual = sensitive_hidden
    sensitive_basis = core.orthonormal_row_basis(
        residual, max_rank=direction_rank
    ).to(device=device, dtype=torch.float32)
    if sensitive_basis.ndim != 2 or sensitive_basis.shape[0] == 0:
        raise RuntimeError("sensitive residual hidden basis is empty")
    energy = sensitive_hidden.square().sum().clamp_min(1e-12)
    residual_fraction = float((residual.square().sum() / energy).detach().cpu())
    return sensitive_basis, {
        "sensitive_hidden_rows": int(sensitive_hidden.shape[0]),
        "context_hidden_rows": int(context_hidden.shape[0]),
        "hidden_size": int(sensitive_hidden.shape[1]),
        "protected_basis_rank": int(protected_basis.shape[0]),
        "requested_protected_rank": int(protected_rank),
        "sensitive_basis_rank": int(sensitive_basis.shape[0]),
        "requested_sensitive_rank": int(direction_rank),
        "protected_context_tokens_per_case_max": int(protected_context_tokens),
        "residual_energy_fraction": residual_fraction,
        "construction": "B_S = SVD(H_F - Proj_{B_P}(H_F)); B_P = truncated SVD of preceding prompt-token hidden states",
    }


def register_input_delta_hook(input_layer, row_ids: Sequence[int], delta: core.SelectedRowDelta):
    ids = torch.tensor(
        [int(x) for x in row_ids], dtype=torch.long, device=input_layer.weight.device
    )

    def hook(_module, inputs, output):
        token_ids = inputs[0].to(ids.device)
        if ids.numel() == 0:
            return output
        positions = torch.searchsorted(ids, token_ids)
        safe = positions.clamp(max=ids.numel() - 1)
        valid = (positions < ids.numel()) & (ids[safe] == token_ids)
        if not bool(valid.any()):
            return output
        effective = delta.effective_delta().to(device=output.device, dtype=output.dtype)
        correction = torch.zeros_like(output)
        correction[valid] = effective[safe[valid]]
        return output + correction

    return input_layer.register_forward_hook(hook)


@torch.no_grad()
def materialize_input_delta(input_layer, row_ids: Sequence[int], delta: torch.Tensor) -> None:
    ids = torch.tensor(
        [int(x) for x in row_ids], dtype=torch.long, device=input_layer.weight.device
    )
    current = input_layer.weight.index_select(0, ids)
    input_layer.weight.index_copy_(
        0, ids, current + delta.to(device=current.device, dtype=current.dtype)
    )


def sha_rows(tensor: torch.Tensor, row_ids: Sequence[int]) -> str:
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=tensor.device)
    rows = tensor.detach().index_select(0, ids).contiguous().view(torch.uint8).cpu().numpy()
    return hashlib.sha256(rows.tobytes()).hexdigest()


def main() -> None:
    a = parse_args()
    if min(
        a.steps,
        a.batch_size,
        a.cache_batch_size,
        a.direction_rank,
        a.protected_rank,
        a.protected_context_tokens,
    ) <= 0:
        raise ValueError("steps, ranks, context count, and batch sizes must be positive")
    if a.learning_rate <= 0 or a.ga_weight <= 0 or a.protection_weight < 0:
        raise ValueError("invalid optimization settings")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest, visible_path, manifest_path = load_locked(a)
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
        raise RuntimeError("no MQuAKE sensitive PredictionCases")

    model.eval()
    base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )
    sensitive_basis, direction_report = make_basis(
        model,
        tok,
        cases,
        device,
        cache_batch_size=a.cache_batch_size,
        protected_context_tokens=a.protected_context_tokens,
        protected_rank=a.protected_rank,
        direction_rank=a.direction_rank,
    )

    # Untie after all frozen-base measurements. This is numerically identity.
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embedding and LM head are still tied")
    model.eval()

    all_tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    special = set(gagd.special_token_ids(tok))
    sensitive_ids = sorted(
        set(int(x) for x in all_tids.detach().cpu().tolist()) - special
    )
    if not sensitive_ids:
        raise RuntimeError("no content-bearing sensitive vocabulary rows")

    input_before_sha = sha_rows(input_layer.weight, sensitive_ids)
    head_before_sha = sha_rows(output_layer.weight, sensitive_ids)
    emb_delta = core.SelectedRowDelta(
        len(sensitive_ids),
        int(input_layer.weight.shape[1]),
        direction_basis=sensitive_basis,
        device=device,
    )
    head_delta = core.SelectedRowDelta(
        len(sensitive_ids),
        int(output_layer.weight.shape[1]),
        direction_basis=sensitive_basis,
        device=device,
    )
    input_hook = register_input_delta_hook(input_layer, sensitive_ids, emb_delta)
    output_hook = core.register_output_delta_hook(
        output_layer, sensitive_ids, head_delta.effective_delta
    )
    params = list(emb_delta.parameters()) + list(head_delta.parameters())
    if a.optimizer == "sgd":
        opt = torch.optim.SGD(params, lr=a.learning_rate)
    elif a.optimizer == "adam":
        opt = torch.optim.Adam(params, lr=a.learning_rate)
    else:
        opt = torch.optim.AdamW(params, lr=a.learning_rate, weight_decay=0.0)

    sampler = core.IndexSampler(len(cases), a.batch_size, a.seed)
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint"

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(
            range(1, a.steps + 1), desc="MQuAKE Stage1 untied directional SURE"
        ):
            idx = sampler.next()
            batch = [cases[i] for i in idx]
            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, batch, device)
            tids = core.official_target_ids(
                tok, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            protection = full_base_to_current_kl(logits, base_logits[idx])
            total = a.ga_weight * ga + a.protection_weight * protection
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite Stage1 loss at step {step}")
            total.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                if a.grad_clip > 0
                else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite Stage1 gradient at step {step}")
            opt.step()

            if step == 1 or step % 25 == 0 or step == a.steps:
                row = {
                    "step": step,
                    "loss": float(total.detach().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "full_base_to_current_kl": float(protection.detach().cpu()),
                    "embedding_delta_norm": float(
                        emb_delta.effective_delta().detach().float().norm().cpu()
                    ),
                    "head_delta_norm": float(
                        head_delta.effective_delta().detach().float().norm().cpu()
                    ),
                    "gradient_norm_before_clip": (
                        None if grad_norm is None else float(grad_norm.detach().cpu())
                    ),
                }
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()

    del opt
    input_hook.remove()
    output_hook.remove()
    emb_effective = emb_delta.effective_delta().detach()
    head_effective = head_delta.effective_delta().detach()
    materialize_input_delta(input_layer, sensitive_ids, emb_effective)
    core.materialize_output_delta(output_layer, sensitive_ids, head_effective)
    model.eval()

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    torch.save(
        {
            "sensitive_basis": sensitive_basis.detach().cpu(),
            "sensitive_token_ids": sensitive_ids,
            "embedding_delta": emb_effective.float().cpu(),
            "head_delta": head_effective.float().cpu(),
        },
        out_dir / "directional_artifacts.pt",
    )

    report: Dict[str, Any] = {
        "schema_version": 3,
        "method": "MQuAKE Stage1 Untied Directional SURE",
        "source_protocol": manifest.get("protocol"),
        "model_path": a.model_path,
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "prediction_case_count": len(cases),
        "sensitive_answer_field": "target_true",
        "input_output_untied_before_learning": True,
        "transformer_exactly_frozen": True,
        "benchmark_retain_seen": 0,
        "heldout_probe_seen": 0,
        "target_new_seen": False,
        "direction": direction_report,
        "sensitive_row_count": len(sensitive_ids),
        "sensitive_token_ids": sensitive_ids,
        "parameterization": {
            "embedding": "Delta E_A = C_E B_S",
            "lm_head": "Delta W_A = C_W B_S",
            "embedding_trainable_coefficients": emb_delta.trainable_parameter_count,
            "head_trainable_coefficients": head_delta.trainable_parameter_count,
            "all_non_sensitive_vocabulary_rows_exactly_frozen": True,
        },
        "loss": {
            "forget": "mean log p(target_true token); minimized (GA)",
            "protection": "exact full-vocabulary KL(Base || Edited) on same training-visible direct PredictionCases",
            "ga_weight": float(a.ga_weight),
            "protection_weight": float(a.protection_weight),
        },
        "optimization": {
            "steps": int(a.steps),
            "learning_rate": float(a.learning_rate),
            "optimizer": a.optimizer,
            "batch_size": int(a.batch_size),
            "cache_batch_size": int(a.cache_batch_size),
            "grad_clip": float(a.grad_clip),
        },
        "materialized_embedding_delta_norm": float(emb_effective.float().norm().cpu()),
        "materialized_head_delta_norm": float(head_effective.float().norm().cpu()),
        "sensitive_embedding_rows_sha256_before": input_before_sha,
        "sensitive_head_rows_sha256_before": head_before_sha,
        "sensitive_embedding_rows_sha256_after": sha_rows(input_layer.weight, sensitive_ids),
        "sensitive_head_rows_sha256_after": sha_rows(output_layer.weight, sensitive_ids),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "directional_stage1_summary.json", report)
    print("Directional Stage1 checkpoint:", ckpt)
    print("PredictionCases:", len(cases))
    print("Sensitive rows:", len(sensitive_ids))
    print("B_S rank:", direction_report["sensitive_basis_rank"])
    print("Residual hidden energy fraction:", direction_report["residual_energy_fraction"])
    print("Embedding/LM head untied: True")


if __name__ == "__main__":
    main()
