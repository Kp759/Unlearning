#!/usr/bin/env python3
"""MQuAKE Level-2 residual GA/GD for Pure Two-Stage Directional SURE.

Input is the restored Level-1 checkpoint from sure_stage1_gagd.py. This script
first gates *all* training-visible teacher-forced atomic target_true token
prompts. If all pass, Level 2 is an identity. Otherwise:

  * F is exactly the Level-1 failed atomic prompts;
  * P is exactly the Level-1 successful atomic prompts;
  * A_F is derived only from the sensitive target rows appearing in F;
  * B_F samples F and receives sensitive GA;
  * B_P samples P and receives GD to the frozen Level-1 non-sensitive teacher;
  * only input/output vocabulary rows in A_F can update.

No rank sweep, scale sweep, nullspace repair, or all-sensitive-row broadening
is allowed. Final gates require every atomic prompt to pass, every Stage-1
success to remain successful, protected-set KL to remain bounded, and all
non-vocabulary parameters to remain exactly unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core


def args():
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
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--lambda-f", type=float, default=2.0)
    p.add_argument("--lambda-p", type=float, default=1.0)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--max-protected-kl", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--skip-transformer-hash", action="store_true")
    return p.parse_args()


def load_locked(a):
    vp = Path(a.training_visible_path).resolve()
    mp = Path(a.split_manifest).resolve()
    records = json.loads(vp.read_text(encoding="utf-8"))
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    if len(records) != a.forget_num:
        raise RuntimeError("training-visible forget count mismatch")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != a.forget_num:
        raise RuntimeError("manifest forget count mismatch")
    if sampling.get("forget_case_ids") and [int(r["case_id"]) for r in records] != [
        int(x) for x in sampling["forget_case_ids"]
    ]:
        raise RuntimeError("training-visible IDs do not match manifest")
    for i, record in enumerate(records):
        rr = record.get("requested_rewrite", {})
        if not rr.get("target_true", {}).get("str"):
            raise RuntimeError(f"record {i} lacks target_true")
        if "target_new" in rr or record.get("atomic_gen_prompt") or record.get("multihop_questions"):
            raise RuntimeError(f"record {i} leaked evaluation-only MQuAKE fields")
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise RuntimeError(f"record {i} leaked held-out probes")
    return records, manifest


def vocab_only(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or output_embeddings is None:
        raise RuntimeError("model lacks vocabulary matrices")
    input_embeddings.weight.requires_grad_(True)
    output_embeddings.weight.requires_grad_(True)
    params = []
    for parameter in (input_embeddings.weight, output_embeddings.weight):
        if all(id(parameter) != id(existing) for existing in params):
            params.append(parameter)
    return input_embeddings, output_embeddings, params


def hash_frozen(model, excluded):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if id(parameter) in excluded:
            continue
        tensor = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def gate(model, tok, cases, llama_like, device, batch, margin):
    residual = []
    values = []
    with torch.no_grad():
        for start in range(0, len(cases), batch):
            current_cases = cases[start : start + batch]
            logits = core.forward_last_logits(model, tok, current_cases, device).float()
            target_ids = core.official_target_ids(
                tok, current_cases, llama_like=llama_like, device=device
            )
            rows = torch.arange(logits.shape[0], device=logits.device)
            sensitive = logits[rows, target_ids]
            other = logits.clone()
            other[rows, target_ids] = -torch.inf
            margins = other.max(-1).values - sensitive
            for offset, value in enumerate(margins.cpu().tolist()):
                values.append(float(value))
                if float(value) < margin:
                    residual.append(start + offset)
    return {
        "total": len(cases),
        "passed": len(cases) - len(residual),
        "failed": len(residual),
        "residual_indices": residual,
        "minimum_margin": min(values) if values else None,
        "required_margin": float(margin),
    }


def masked_rows(params, rows):
    if not rows:
        raise RuntimeError("A_F is empty for non-empty F")
    hooks = []
    unique_rows = sorted(set(int(row) for row in rows))
    for parameter in params:
        mask = torch.zeros(
            (parameter.shape[0], 1), dtype=parameter.dtype, device=parameter.device
        )
        row_ids = torch.tensor(unique_rows, dtype=torch.long, device=parameter.device)
        mask.index_fill_(0, row_ids, 1)
        hooks.append(parameter.register_hook(lambda grad, row_mask=mask: grad * row_mask))
    return hooks


def protected_kl(model, tok, cases, teacher, indices, llama_like, device, batch):
    if not indices:
        return 0.0
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ids = indices[start : start + batch]
            current_cases = [cases[i] for i in ids]
            logits = core.forward_last_logits(model, tok, current_cases, device)
            target_ids = core.official_target_ids(
                tok, current_cases, llama_like=llama_like, device=device
            )
            value = core.gd_non_sensitive_kl(logits, teacher[ids], target_ids)
            total += float(value.cpu()) * len(ids)
            count += len(ids)
    return total / max(count, 1)


def main():
    a = args()
    if min(
        a.repair_steps,
        a.batch_size,
        a.protection_batch_size,
        a.cache_batch_size,
        a.check_every,
    ) <= 0:
        raise ValueError("steps/batches must be positive")
    if a.repair_lr <= 0 or a.lambda_f <= 0 or a.lambda_p < 0:
        raise ValueError("invalid repair hyperparameters")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = load_locked(a)
    namespace = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("no generated atomic target_true token prompts")

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint"

    input_embeddings, output_embeddings, params = vocab_only(model)
    excluded = {id(input_embeddings.weight), id(output_embeddings.weight)}
    before_hash = None if a.skip_transformer_hash else hash_frozen(model, excluded)

    level1_gate = gate(
        model,
        tok,
        cases,
        llama_like,
        device,
        a.cache_batch_size,
        a.constraint_margin,
    )
    F = [int(i) for i in level1_gate["residual_indices"]]
    failed_set = set(F)
    P = [i for i in range(len(cases)) if i not in failed_set]

    report: Dict[str, Any] = {
        "schema_version": 2,
        "method": "MQuAKE Pure Two-Stage Directional SURE / Level2",
        "source_protocol": manifest.get("protocol"),
        "level1_gate": level1_gate,
        "generated_atomic_prompt_semantics": (
            "all teacher-forced target_true token cases from training-visible direct facts"
        ),
        "official_atomicgen_seen": 0,
        "benchmark_retain_seen": 0,
        "target_new_seen": False,
        "forbidden_mechanics": [
            "rank sweep",
            "scale sweep",
            "nullspace repair",
            "all-sensitive-row broadening",
        ],
    }

    teacher = None
    A_F = []
    if not F:
        report["level2"] = {
            "skipped": True,
            "reason": "Level 1 passed all generated atomic prompts",
            "F": 0,
            "P": len(P),
            "A_F": [],
        }
    else:
        # Freeze the entire Level-1 teacher distribution before Level-2 updates.
        teacher = core.cache_base_logits(
            model, tok, cases, device, batch_size=a.cache_batch_size
        )
        residual_cases = [cases[i] for i in F]
        target_ids = core.official_target_ids(
            tok, residual_cases, llama_like=llama_like, device=device
        )
        A_F = sorted(
            set(int(x) for x in target_ids.cpu().tolist())
            - set(gagd.special_token_ids(tok))
        )

        hooks = masked_rows(params, A_F)
        optimizer = torch.optim.AdamW(params, lr=a.repair_lr, weight_decay=0.0)
        forget_sampler = core.IndexSampler(len(F), a.batch_size, a.seed + 100003)
        protection_sampler = (
            core.IndexSampler(len(P), a.protection_batch_size, a.seed + 200003)
            if P
            else None
        )
        logs = []
        model.train()
        try:
            for step in range(1, a.repair_steps + 1):
                forget_local = forget_sampler.next()
                forget_ids = [F[i] for i in forget_local]
                forget_cases = [cases[i] for i in forget_ids]

                optimizer.zero_grad(set_to_none=True)
                forget_logits = core.forward_last_logits(
                    model, tok, forget_cases, device
                )
                forget_targets = core.official_target_ids(
                    tok, forget_cases, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(forget_logits, forget_targets)

                if protection_sampler is not None:
                    protection_local = protection_sampler.next()
                    protection_ids = [P[i] for i in protection_local]
                    protection_cases = [cases[i] for i in protection_ids]
                    protection_logits = core.forward_last_logits(
                        model, tok, protection_cases, device
                    )
                    protection_targets = core.official_target_ids(
                        tok,
                        protection_cases,
                        llama_like=llama_like,
                        device=device,
                    )
                    gd = core.gd_non_sensitive_kl(
                        protection_logits,
                        teacher[protection_ids],
                        protection_targets,
                    )
                else:
                    gd = ga.new_zeros(())

                loss = a.lambda_f * ga + a.lambda_p * gd
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Level2 loss at {step}")
                loss.backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(params, a.grad_clip)
                    if a.grad_clip > 0
                    else None
                )
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"non-finite Level2 gradient norm at {step}")
                optimizer.step()

                if step == 1 or step % a.check_every == 0 or step == a.repair_steps:
                    model.eval()
                    current_gate = gate(
                        model,
                        tok,
                        cases,
                        llama_like,
                        device,
                        a.cache_batch_size,
                        a.constraint_margin,
                    )
                    current_failed = set(current_gate["residual_indices"])
                    protected_regressions = sum(1 for i in P if i in current_failed)
                    logs.append(
                        {
                            "step": step,
                            "loss": float(loss.detach().cpu()),
                            "ga": float(ga.detach().cpu()),
                            "gd": float(gd.detach().cpu()),
                            "failed_all_atomic": current_gate["failed"],
                            "stage1_successes_regressed": protected_regressions,
                            "grad_norm": (
                                None
                                if grad_norm is None
                                else float(grad_norm.detach().cpu())
                            ),
                        }
                    )
                    if current_gate["failed"] == 0:
                        break
                    model.train()
        finally:
            for hook in hooks:
                hook.remove()
            del optimizer
            model.eval()

        report["level2"] = {
            "skipped": False,
            "F": len(F),
            "F_indices": F,
            "P": len(P),
            "P_indices": P,
            "A_F": A_F,
            "A_F_count": len(A_F),
            "B_F": "failed Level-1 atomic prompts (sensitive GA)",
            "B_P": (
                "Stage-1-success atomic prompts, frozen Level-1 non-sensitive "
                "distribution (GD)"
            ),
            "updated_parameters": "ONLY embedding/output vocabulary rows in A_F",
            "logs": logs,
        }

    final_gate = gate(
        model,
        tok,
        cases,
        llama_like,
        device,
        a.cache_batch_size,
        a.constraint_margin,
    )
    final_failed = set(final_gate["residual_indices"])
    stage1_successes_regressed = sum(1 for i in P if i in final_failed)
    protected_kl_value = (
        protected_kl(
            model,
            tok,
            cases,
            teacher,
            P,
            llama_like,
            device,
            a.cache_batch_size,
        )
        if teacher is not None and P
        else 0.0
    )

    after_hash = None if a.skip_transformer_hash else hash_frozen(model, excluded)
    exact = None if a.skip_transformer_hash else before_hash == after_hash
    if exact is False:
        raise AssertionError("frozen non-vocabulary parameters changed")

    gates = {
        "all_generated_atomic_prompts_pass": final_gate["failed"] == 0,
        "stage1_successes_regressed": stage1_successes_regressed,
        "stage1_successes_preserved": stage1_successes_regressed == 0,
        "protected_non_sensitive_kl": protected_kl_value,
        "protected_kl_threshold": a.max_protected_kl,
        "protected_regression_bounded": protected_kl_value <= a.max_protected_kl,
        "transformer_exactly_unchanged": exact,
    }
    gates["all_required_gates_pass"] = bool(
        gates["all_generated_atomic_prompts_pass"]
        and gates["stage1_successes_preserved"]
        and gates["protected_regression_bounded"]
        and exact is not False
    )

    report.update(
        {
            "final_gate": final_gate,
            "final_gates": gates,
            "transformer_hash_before": before_hash,
            "transformer_hash_after": after_hash,
        }
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    (out / "two_stage_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"Level1 gate: {level1_gate['passed']}/{level1_gate['total']} pass; "
        f"F={len(F)}; P={len(P)}"
    )
    print("Level2:", "SKIPPED" if not F else f"F={len(F)}, P={len(P)}, A_F={len(A_F)}")
    print(
        f"Final gate: {final_gate['passed']}/{final_gate['total']} pass; "
        f"protected_KL={protected_kl_value:.6g}; "
        f"Stage1 regressions={stage1_successes_regressed}; transformer_exact={exact}"
    )
    print("Final gates pass:", gates["all_required_gates_pass"])
    print("Checkpoint:", ckpt)


if __name__ == "__main__":
    main()
