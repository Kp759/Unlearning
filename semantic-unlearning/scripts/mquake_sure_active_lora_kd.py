#!/usr/bin/env python3
"""Run existing contextual LoRA as SURE Stage 2 with active-only KD repair.

Use an already-trained SURE Emb+LM checkpoint as --model-path and pass
--steps 0. The existing driver then identifies residual correct sensitive-token
cases and invokes this KD-augmented train function only for that active subset.
"""
import torch
import torch.nn.functional as F

import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr


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
    for row in range(mask.size(0)):
        nz = torch.nonzero(mask[row], as_tuple=False).flatten()
        if nz.numel():
            mask[row, nz[-1]] = False
    return (kl * mask.float()).sum() / mask.sum().clamp_min(1)


def train_active_kd(model, tok, cases, llama_like, device, params,
                    steps, lr, margin, l2, n, check, seed, label):
    if not cases or steps <= 0:
        return {"steps": 0, "reason": "no active cases or zero steps", "logs": []}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    it = cl.batches(cases, n, seed)
    logs = []
    for step in range(1, steps + 1):
        cs = next(it)
        opt.zero_grad(set_to_none=True)
        logits = nnr.forward_logits(model, tok, cs, device)
        targets = nnr.target_ids(tok, cs, llama_like, device)
        forget = nnr.loss_fn(logits, targets, margin)
        kd = prompt_kd(model, tok, cs, device)
        reg = torch.stack([p.pow(2).mean() for p in params]).mean()
        loss = forget + kd + l2 * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step == 1 or step % check == 0 or step == steps:
            correct, min_margin = cl.status(model, tok, cases, llama_like, device, n)
            row = {
                "step": step,
                "correct": correct,
                "min_margin": min_margin,
                "forget_loss": float(forget.detach()),
                "kd_loss": float(kd.detach()),
            }
            logs.append(row)
            print(label, row)
            if min_margin is not None and min_margin >= margin:
                return {"steps": step, "reason": "all active cases meet margin", "logs": logs}
    return {"steps": steps, "reason": "max_steps", "logs": logs}


_original_write = nnr.write

def hybrid_write(path, payload):
    if getattr(path, "name", "") == "summary.json" and isinstance(payload, dict):
        payload = dict(payload)
        payload["method"] = "SURE-EmbLM-stage1-plus-active-only-contextual-LoRA-KD"
        payload["stage2_teacher"] = "frozen SURE Stage-1 Emb+LM checkpoint"
        payload["stage2_IDK_used"] = False
        payload["stage2_target_new_seen"] = False
        payload["stage2_Unknown_seen"] = False
        payload["stage2_retain_seen"] = 0
        payload["stage2_PPL_seen_during_selection"] = False
        payload["sure_stage1_input"] = payload.pop("base", None)
        payload["zero_step_lora_phase"] = payload.get("stage1")
    return _original_write(path, payload)


nnr.write = hybrid_write
cl.train = train_active_kd
cl.main()
