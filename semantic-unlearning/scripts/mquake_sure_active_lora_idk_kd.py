#!/usr/bin/env python3
"""SURE Stage-2 contextual residual repair with fixed-anchor IDK preservation.

Stage 1 is an already-trained SURE Emb+LM checkpoint. This wrapper reuses the
existing contextual-LoRA driver with --steps 0, so LoRA training runs only on
residual active sensitive-token cases.

Stage 2 keeps embeddings, LM head, and all base-transformer weights frozen. Only
LoRA tensors are trainable. The sensitive logit is detached; the repair raises
the contextual first token of ``I don't know`` above the sensitive token.

Hard cases drive only the forgetting objective. Preservation is always anchored
to the full original Stage-1 active set, even after hard mining shrinks the
forgetting pool. Two strict forget-only preservation terms are used:
  1) prompt-position KD on the full active-set anchor batch, excluding the final
     factual decision position;
  2) final-position top-k logit-shape KD on the same anchor batch, excluding only
     the sensitive token and IDK token so it does not conflict with forgetting.

No retain examples, PPL data, held-out MQuAKE questions, target_new, or Unknown
are used for training or checkpoint selection.
"""
import math
import os

import torch
import torch.nn.functional as F

import mquake_zero_unlearn_official_eval as mq
import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr

IDK_TEXT = os.environ.get("SURE_IDK_TEXT", "I don't know")
PROMPT_KD_WEIGHT = float(os.environ.get("SURE_IDK_KD_WEIGHT", "1.0"))
DECISION_KD_WEIGHT = float(os.environ.get("SURE_IDK_DECISION_KD_WEIGHT", "1.0"))
DECISION_KD_TOPK = int(os.environ.get("SURE_IDK_DECISION_KD_TOPK", "64"))
MIN_LR_SCALE = float(os.environ.get("SURE_IDK_MIN_LR_SCALE", "0.20"))


def prompt_kd(model, tok, cases, device):
    """Preserve Stage-1 behavior on prompt positions before the final decision."""
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


def decision_topk_kd(model, tok, cases, llama_like, device, idk_id):
    """Preserve Stage-1 final-position alternatives except sensitive + IDK.

    We match the centered logits of the teacher's top-k non-sensitive/non-IDK
    alternatives. Centering makes the term invariant to a shared logit offset.
    """
    with torch.no_grad(), model.disable_adapter():
        teacher = nnr.forward_logits(model, tok, cases, device).detach().float()
    student = nnr.forward_logits(model, tok, cases, device).float()
    targets = nnr.target_ids(tok, cases, llama_like, device)
    rows = torch.arange(len(cases), device=device)

    masked_teacher = teacher.clone()
    masked_teacher[rows, targets] = -torch.inf
    masked_teacher[:, idk_id] = -torch.inf
    k = min(DECISION_KD_TOPK, masked_teacher.shape[-1] - 2)
    top_ids = torch.topk(masked_teacher, k=k, dim=-1).indices

    t = teacher.gather(1, top_ids)
    s = student.gather(1, top_ids)
    t = t - t.mean(dim=-1, keepdim=True)
    s = s - s.mean(dim=-1, keepdim=True)
    return F.smooth_l1_loss(s, t)


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


def set_adaptive_lr(opt, base_lr, hard_count, total_count):
    """Reduce step size as the hard pool shrinks to avoid repeated-case overshoot."""
    frac = hard_count / max(total_count, 1)
    scale = max(MIN_LR_SCALE, min(1.0, math.sqrt(frac)))
    for group in opt.param_groups:
        group["lr"] = base_lr * scale
    return base_lr * scale


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
    print("Sensitive logit is DETACHED; Stage2 directly raises only contextual IDK competitor.")
    print("Hard cases drive IDK loss; full original active set remains the KD anchor.")
    print("Final-position KD excludes only sensitive + IDK logits; topk=", DECISION_KD_TOPK)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    all_cases = list(cases)
    working = active_subset(model, tok, all_cases, llama_like, device, n)
    if not working:
        return {"steps": 0, "reason": "zero sensitive tokens remain correct", "logs": [],
                "idk_text": IDK_TEXT, "idk_first_token_id": idk_id}

    hard_it = cl.batches(working, n, seed)
    anchor_it = cl.batches(all_cases, n, seed + 700001)
    logs = []
    best_correct = len(working)
    best_step = 0
    best_state = snapshot_params(params)
    effective_lr = set_adaptive_lr(opt, lr, len(working), len(all_cases))

    for step in range(1, steps + 1):
        hard_cs = next(hard_it)
        anchor_cs = next(anchor_it)
        opt.zero_grad(set_to_none=True)

        # Forgetting term: only current residual active cases.
        logits = nnr.forward_logits(model, tok, hard_cs, device).float()
        targets = nnr.target_ids(tok, hard_cs, llama_like, device)
        rows = torch.arange(len(hard_cs), device=device)
        sensitive = logits[rows, targets].detach()
        idk = logits[:, idk_id]
        idk_margin_loss = F.relu(sensitive - idk + margin).mean()

        # Preservation terms: always sampled from the complete original Stage-1
        # active set, not the shrinking hard-mining pool.
        pkd = prompt_kd(model, tok, anchor_cs, device)
        dkd = decision_topk_kd(model, tok, anchor_cs, llama_like, device, idk_id)
        reg = torch.stack([p.pow(2).mean() for p in params]).mean()
        loss = (
            idk_margin_loss
            + PROMPT_KD_WEIGHT * pkd
            + DECISION_KD_WEIGHT * dkd
            + l2 * reg
        )
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
                "anchor_pool": len(all_cases),
                "effective_lr": effective_lr,
                "min_margin": min_margin,
                "idk_margin_loss": float(idk_margin_loss.detach()),
                "prompt_kd_loss": float(pkd.detach()),
                "decision_kd_loss": float(dkd.detach()),
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
                        "idk_text": IDK_TEXT, "idk_first_token_id": idk_id,
                        "anchor_pool": len(all_cases)}

            # Hard-mine only the forgetting pool.  The preservation anchor stays
            # fixed to all original active cases for every optimizer step.
            working = remaining
            hard_it = cl.batches(working, n, seed + step)
            effective_lr = set_adaptive_lr(opt, lr, len(working), len(all_cases))

    restore_params(params, best_state)
    final_remaining = active_subset(model, tok, all_cases, llama_like, device, n)
    print("Restored best fixed-anchor hard-mined LoRA state:",
          "best_step=", best_step, "best_correct=", best_correct,
          "verified_correct=", len(final_remaining))
    return {"steps": steps, "reason": "max_steps_best_state_restored", "logs": logs,
            "best_correct": best_correct, "best_step": best_step,
            "idk_text": IDK_TEXT, "idk_first_token_id": idk_id,
            "anchor_pool": len(all_cases)}


# Canonical metadata for the hybrid SURE method.
_orig_write = nnr.write

def _canonical_write(path, obj):
    if str(path).endswith("summary.json") and isinstance(obj, dict):
        obj = dict(obj)
        obj["method"] = "SURE-EmbLM-plus-contextual-residual-IDK-fixed-anchor"
        if "base" in obj:
            obj["sure_stage1_input"] = obj.pop("base")
        if "stage1" in obj:
            obj["phase1_zero_step"] = obj.pop("stage1")
        if "active_before_phase2" in obj:
            obj["active_before_stage2"] = obj.pop("active_before_phase2")
        obj["stage2_trainable_scope"] = "LoRA only; embeddings/LM-head/base transformer frozen"
        obj["stage2_forgetting_scope"] = "residual active forget cases only"
        obj["stage2_preservation_anchor"] = "full original Stage1-active forget set only"
        obj["stage2_idk_text"] = IDK_TEXT
        obj["stage2_sensitive_logit_detached"] = True
        obj["stage2_prompt_kd_weight"] = PROMPT_KD_WEIGHT
        obj["stage2_decision_kd_weight"] = DECISION_KD_WEIGHT
        obj["stage2_decision_kd_topk"] = DECISION_KD_TOPK
        obj["stage2_min_lr_scale"] = MIN_LR_SCALE
        obj["retain_seen"] = 0
        obj["PPL_seen_during_selection"] = False
        obj["Unknown_used"] = False
        obj["target_new_seen"] = False
    return _orig_write(path, obj)

nnr.write = _canonical_write
cl.train = train_active_idk
cl.main()
