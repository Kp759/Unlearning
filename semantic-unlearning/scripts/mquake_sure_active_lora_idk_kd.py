#!/usr/bin/env python3
"""SURE Stage-2 active-only LoRA repair by raising contextual IDK logits.

Stage 1 is an already-trained SURE Emb+LM checkpoint. This wrapper reuses the
existing contextual-LoRA driver with --steps 0, so LoRA training runs only on
residual active sensitive-token cases. The sensitive logit is detached: Stage 2
optimizes only the contextual IDK competitor plus same-prompt KD preservation.

Residual cases are hard-mined during training: after every check interval, cases
whose sensitive target is no longer argmax are removed from the training sampler.
The best LoRA state (fewest remaining correct sensitive decisions) is retained so
late optimization oscillations cannot overwrite a better repair.
"""
import os
import torch
import torch.nn.functional as F

import mquake_zero_unlearn_official_eval as mq
import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr

IDK_TEXT = os.environ.get("SURE_IDK_TEXT", "I don't know")
KD_WEIGHT = float(os.environ.get("SURE_IDK_KD_WEIGHT", "1.0"))


def prompt_kd(model, tok, cases, device):
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
    last = enc["attention_mask"].sum(dim=1) - 1
    mask[torch.arange(mask.size(0), device=device), last] = False
    return (kl * mask.float()).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def active_subset(model, tok, cases, llama_like, device, batch_size):
    """Return exactly the cases whose sensitive target is still argmax."""
    out = []
    for start in range(0, len(cases), batch_size):
        cs = cases[start:start + batch_size]
        logits = nnr.forward_logits(model, tok, cs, device).float()
        targets = nnr.target_ids(tok, cs, llama_like, device)
        pred = logits.argmax(dim=-1)
        keep = (pred == targets).detach().cpu().tolist()
        out.extend(c for c, flag in zip(cs, keep) if flag)
    return out


def snapshot_params(params):
    return [p.detach().clone() for p in params]


def restore_params(params, state):
    with torch.no_grad():
        for p, q in zip(params, state):
            p.copy_(q)


def train_active_idk(model, tok, cases, llama_like, device, params,
                     steps, lr, margin, l2, n, check, seed, label):
    if not cases or steps <= 0:
        return {"steps": 0, "reason": "no active cases or zero steps", "logs": []}

    idk_ids = mq.original_answer_token_ids(tok, IDK_TEXT, llama_like=llama_like)
    if not idk_ids:
        raise RuntimeError(f"IDK text tokenized to no tokens: {IDK_TEXT!r}")
    idk_id = int(idk_ids[0])
    print("Stage2 IDK target:", repr(IDK_TEXT), "first_token_id=", idk_id,
          "token=", repr(tok.decode([idk_id])))
    print("Sensitive logit is DETACHED; Stage2 directly raises only IDK competitor.")
    print("Dynamic hard mining: solved sensitive decisions are removed every check interval.")

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    all_cases = list(cases)
    working = active_subset(model, tok, all_cases, llama_like, device, n)
    if not working:
        return {"steps": 0, "reason": "zero sensitive tokens remain correct", "logs": [],
                "idk_text": IDK_TEXT, "idk_first_token_id": idk_id}

    it = cl.batches(working, n, seed)
    logs = []
    best_correct = len(working)
    best_step = 0
    best_state = snapshot_params(params)

    for step in range(1, steps + 1):
        cs = next(it)
        opt.zero_grad(set_to_none=True)
        logits = nnr.forward_logits(model, tok, cs, device).float()
        targets = nnr.target_ids(tok, cs, llama_like, device)
        rows = torch.arange(len(cs), device=device)
        sensitive = logits[rows, targets].detach()
        idk = logits[:, idk_id]

        # Raise IDK above the current sensitive logit. No direct gradient lowers
        # the sensitive token; any collateral movement is only through shared h.
        idk_margin_loss = F.relu(sensitive - idk + margin).mean()
        kd = prompt_kd(model, tok, cs, device)
        reg = torch.stack([p.pow(2).mean() for p in params]).mean()
        loss = idk_margin_loss + KD_WEIGHT * kd + l2 * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step == 1 or step % check == 0 or step == steps:
            remaining = active_subset(model, tok, all_cases, llama_like, device, n)
            correct = len(remaining)
            _, min_margin = cl.status(model, tok, all_cases, llama_like, device, n)
            row = {
                "step": step,
                "correct": correct,
                "hard_mining_pool": correct,
                "min_margin": min_margin,
                "idk_margin_loss": float(idk_margin_loss.detach()),
                "kd_loss": float(kd.detach()),
                "mean_sensitive_logit": float(sensitive.mean().detach()),
                "mean_idk_logit": float(idk.mean().detach()),
            }
            logs.append(row)
            print(label, row)

            if correct < best_correct:
                best_correct = correct
                best_step = step
                best_state = snapshot_params(params)

            if correct == 0:
                return {"steps": step, "reason": "zero sensitive tokens remain correct",
                        "logs": logs, "best_correct": 0, "best_step": step,
                        "idk_text": IDK_TEXT, "idk_first_token_id": idk_id}

            # Hard-mine only residual failures from this point onward.
            working = remaining
            it = cl.batches(working, n, seed + step)

    # Never let late oscillation overwrite the best repair observed.
    restore_params(params, best_state)
    final_remaining = active_subset(model, tok, all_cases, llama_like, device, n)
    print("Restored best hard-mined LoRA state:",
          "best_step=", best_step, "best_correct=", best_correct,
          "verified_correct=", len(final_remaining))
    return {"steps": steps, "reason": "max_steps_best_state_restored", "logs": logs,
            "best_correct": best_correct, "best_step": best_step,
            "idk_text": IDK_TEXT, "idk_first_token_id": idk_id}


cl.train = train_active_idk
cl.main()
