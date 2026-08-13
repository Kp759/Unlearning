#!/usr/bin/env python3
"""SURE Stage-2 contextual residual repair by sensitive-logit suppression.

Stage 1 is the existing SURE Emb+LM checkpoint after vocabulary restoration.
Stage 2:
  * freezes embeddings, LM head, and every base-transformer weight;
  * trains only tiny PEFT LoRA tensors selected by the shared driver;
  * operates only on residual Stage-1 forget cases whose sensitive token is still
    argmax;
  * uses no Unknown, IDK token, target_new, retain data, held-out questions, or
    PPL data;
  * directly suppresses the sensitive token relative to the strongest detached
    non-sensitive competitor;
  * uses same-forget-prompt Stage-1 KD only as a preservation regularizer;
  * stops immediately when zero sensitive decisions remain correct, otherwise
    restores the LoRA state with the fewest remaining correct decisions.

The underlying driver is run with --steps 0, so there is no generic LoRA phase:
its active phase is exactly SURE Stage 2.
"""
import os

import torch
import torch.nn.functional as F

import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr

KD_WEIGHT = float(os.environ.get("SURE_SUPPRESS_KD_WEIGHT", "1.0"))


def prompt_kd(model, tok, cases, device):
    """Distill the frozen Stage-1 model on the same forget prompts only."""
    enc = tok([c.prompt for c in cases], padding=True, return_tensors="pt").to(device)
    with torch.no_grad(), model.disable_adapter():
        teacher = model(**enc, use_cache=False).logits.detach().float()
    student = model(**enc, use_cache=False).logits.float()
    kl = F.kl_div(
        F.log_softmax(student, dim=-1),
        F.softmax(teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    mask = enc["attention_mask"].bool()
    # Do not distill the final factual decision position: that is where the
    # sensitive-answer suppression objective must be free to act.
    for row in range(mask.size(0)):
        nz = torch.nonzero(mask[row], as_tuple=False).flatten()
        if nz.numel():
            mask[row, nz[-1]] = False
    return (kl * mask.float()).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def count_active(model, tok, cases, llama_like, device, batch_size):
    correct = 0
    for start in range(0, len(cases), batch_size):
        cs = cases[start:start + batch_size]
        logits = nnr.forward_logits(model, tok, cs, device).float()
        targets = nnr.target_ids(tok, cs, llama_like, device)
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
    return correct


def snapshot(params):
    return [p.detach().clone() for p in params]


def restore(params, state):
    with torch.no_grad():
        for p, q in zip(params, state):
            p.copy_(q)


def train_sensitive_suppression(model, tok, cases, llama_like, device, params,
                                steps, lr, margin, l2, n, check, seed, label):
    if not cases or steps <= 0:
        return {"steps": 0, "reason": "no active cases or zero steps", "logs": []}

    print("===== SURE STAGE 2: CONTEXTUAL SENSITIVE SUPPRESSION =====")
    print("Embeddings: FROZEN")
    print("LM head: FROZEN")
    print("Base transformer: FROZEN")
    print("Only LoRA tensors are trainable (enforced by shared driver).")
    print("Unknown used: NO")
    print("IDK/replacement target used: NO")
    print("Sensitive-answer objective: direct contextual logit suppression")
    print("Stage-1 same-forget-prompt KD weight:", KD_WEIGHT)
    print("Residual active training pool:", len(cases))

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    it = cl.batches(cases, n, seed)
    logs = []

    initial_correct = count_active(model, tok, cases, llama_like, device, n)
    best_correct = initial_correct
    best_step = 0
    best_state = snapshot(params)

    if initial_correct == 0:
        return {
            "steps": 0,
            "reason": "zero sensitive tokens remain correct",
            "logs": [],
            "best_correct": 0,
            "best_step": 0,
        }

    for step in range(1, steps + 1):
        cs = next(it)
        opt.zero_grad(set_to_none=True)

        logits = nnr.forward_logits(model, tok, cs, device).float()
        targets = nnr.target_ids(tok, cs, llama_like, device)
        rows = torch.arange(len(cs), device=device)
        sensitive = logits[rows, targets]

        # nnr.loss_fn computes:
        # relu(z_sensitive - stopgrad(max_{j != sensitive} z_j) + margin)
        # There is no chosen replacement/neutral token.
        forget = nnr.loss_fn(logits, targets, margin)
        kd = prompt_kd(model, tok, cs, device)
        reg = torch.stack([p.pow(2).mean() for p in params]).mean()
        loss = forget + KD_WEIGHT * kd + l2 * reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1 or step % check == 0 or step == steps:
            correct = count_active(model, tok, cases, llama_like, device, n)
            _, min_margin = cl.status(model, tok, cases, llama_like, device, n)
            row = {
                "step": step,
                "correct": correct,
                "active_pool": len(cases),
                "min_margin": min_margin,
                "forget_loss": float(forget.detach()),
                "kd_loss": float(kd.detach()),
                "mean_sensitive_logit": float(sensitive.mean().detach()),
            }
            logs.append(row)
            print(label, row)

            if correct < best_correct:
                best_correct = correct
                best_step = step
                best_state = snapshot(params)

            if correct == 0:
                return {
                    "steps": step,
                    "reason": "zero sensitive tokens remain correct",
                    "logs": logs,
                    "best_correct": 0,
                    "best_step": step,
                }

    restore(params, best_state)
    verified = count_active(model, tok, cases, llama_like, device, n)
    print(
        "Restored best Stage-2 LoRA state:",
        "best_step=", best_step,
        "best_correct=", best_correct,
        "verified_correct=", verified,
    )
    return {
        "steps": steps,
        "reason": "max_steps_best_state_restored",
        "logs": logs,
        "best_correct": best_correct,
        "best_step": best_step,
    }


_original_write = nnr.write


def sure_write(path, payload):
    if getattr(path, "name", "") == "summary.json" and isinstance(payload, dict):
        payload = dict(payload)
        payload["method"] = "SURE-EmbLM-plus-contextual-sensitive-suppression"
        payload["stage2_objective"] = (
            "relu(sensitive_logit-stopgrad(best_non_sensitive_logit)+margin)"
        )
        payload["stage2_teacher"] = "frozen SURE Stage-1 Emb+LM checkpoint"
        payload["stage2_KD_weight"] = KD_WEIGHT
        payload["stage2_IDK_used"] = False
        payload["stage2_replacement_target_used"] = False
        payload["stage2_target_new_seen"] = False
        payload["stage2_Unknown_seen"] = False
        payload["stage2_retain_seen"] = 0
        payload["stage2_PPL_seen_during_selection"] = False
        payload["sure_stage1_input"] = payload.pop("base", None)
        payload["zero_step_lora_phase"] = payload.get("stage1")
    return _original_write(path, payload)


nnr.write = sure_write
cl.train = train_sensitive_suppression
cl.main()
