#!/usr/bin/env python3
"""Minimal SURE Stage-2 contextual residual IDK repair for MQuAKE.

Stage 1 is the existing SURE Emb+LM checkpoint after vocabulary restoration.
Stage 2:
  * freezes embeddings, LM head, and every base-transformer weight;
  * trains only tiny PEFT LoRA tensors selected by the shared driver;
  * trains only on residual Stage-1 forget cases whose sensitive token is still
    argmax;
  * keeps the sensitive logit detached;
  * raises the contextual first token of ``I don't know`` above that sensitive
    logit by a small margin;
  * uses no Unknown/target_new, retain data, held-out questions, PPL data, KD,
    or LM-head repair;
  * stops immediately when zero sensitive decisions remain correct, otherwise
    restores the LoRA state with the fewest remaining correct decisions.

The underlying driver is run with --steps 0, so there is no generic LoRA phase:
its active phase is exactly SURE Stage 2.
"""
import os

import torch
import torch.nn.functional as F

import mquake_zero_unlearn_official_eval as mq
import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr

IDK_TEXT = os.environ.get("SURE_IDK_TEXT", "I don't know")


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


def train_contextual_idk(model, tok, cases, llama_like, device, params,
                         steps, lr, margin, l2, n, check, seed, label):
    # --steps 0 disables the driver's generic phase. The second invocation is
    # the true active-only SURE Stage 2.
    if not cases or steps <= 0:
        return {
            "steps": 0,
            "reason": "no active cases or zero steps",
            "logs": [],
        }

    idk_ids = mq.original_answer_token_ids(tok, IDK_TEXT, llama_like=llama_like)
    if not idk_ids:
        raise RuntimeError(f"IDK text tokenized to no tokens: {IDK_TEXT!r}")
    idk_id = int(idk_ids[0])

    print("===== SURE STAGE 2: CONTEXTUAL RESIDUAL IDK =====")
    print("IDK text:", repr(IDK_TEXT))
    print("IDK first token id:", idk_id, "token:", repr(tok.decode([idk_id])))
    print("Embeddings: FROZEN")
    print("LM head: FROZEN")
    print("Base transformer: FROZEN")
    print("Only LoRA tensors are trainable (enforced by shared driver).")
    print("Sensitive logit: DETACHED")
    print("Training pool remains the complete residual Stage-1 active set:", len(cases))
    print("No Unknown / target_new / retain / held-out / PPL / KD / LM-head repair.")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    it = cl.batches(cases, n, seed)
    logs = []

    initial_correct = count_active(model, tok, cases, llama_like, device, n)
    if initial_correct == 0:
        return {
            "steps": 0,
            "reason": "zero sensitive tokens remain correct",
            "logs": [],
            "best_correct": 0,
            "best_step": 0,
            "idk_text": IDK_TEXT,
            "idk_first_token_id": idk_id,
        }

    best_correct = initial_correct
    best_step = 0
    best_state = snapshot(params)

    for step in range(1, steps + 1):
        cs = next(it)
        opt.zero_grad(set_to_none=True)

        logits = nnr.forward_logits(model, tok, cs, device).float()
        targets = nnr.target_ids(tok, cs, llama_like, device)
        rows = torch.arange(len(cs), device=device)

        sensitive = logits[rows, targets].detach()
        idk = logits[:, idk_id]

        # Contextual abstention only: move hidden representations so the frozen
        # IDK output row beats the frozen sensitive output row in these contexts.
        idk_margin_loss = F.relu(sensitive - idk + margin).mean()
        reg = torch.stack([p.pow(2).mean() for p in params]).mean()
        loss = idk_margin_loss + l2 * reg

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
                "idk_margin_loss": float(idk_margin_loss.detach()),
                "mean_sensitive_logit": float(sensitive.mean().detach()),
                "mean_idk_logit": float(idk.mean().detach()),
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
                    "idk_text": IDK_TEXT,
                    "idk_first_token_id": idk_id,
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
        "idk_text": IDK_TEXT,
        "idk_first_token_id": idk_id,
    }


# Replace only the shared driver's optimization rule. Its PEFT setup and explicit
# non-LoRA-trainable assertion preserve the Stage-2 freeze boundary.
cl.train = train_contextual_idk
cl.main()
