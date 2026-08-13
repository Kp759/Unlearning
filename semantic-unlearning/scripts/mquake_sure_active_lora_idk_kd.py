#!/usr/bin/env python3
"""SURE Stage-2 active-only contextual residual repair with IDK competition.

Stage 1 is an already-trained core-SURE Emb+LM checkpoint after vocabulary
restoration.  This wrapper reuses the contextual-LoRA driver with --steps 0, so
there is no new generic LoRA phase: only residual active sensitive-token cases
from the frozen SURE Stage-1 checkpoint enter Stage 2.

Stage 2 keeps embeddings, LM head, and all original transformer weights frozen;
PEFT exposes only tiny LoRA tensors in the selected late contextual projections.
The sensitive logit is detached.  Optimization raises the contextual first token
of ``I don't know`` above the sensitive token while same-prompt Stage-1 KD
preserves the surrounding behavior.

Residual cases are hard-mined after every check interval.  Solved sensitive
choices are removed from the sampler, and training stops immediately when zero
sensitive decisions remain correct.  If the maximum step budget is reached, the
best LoRA state seen during training is restored before scale selection.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

import mquake_zero_unlearn_official_eval as mq
import mquake_forget_only_contextual_lora as cl
import mquake_forget_only_no_neutral as nnr

IDK_TEXT = os.environ.get("SURE_IDK_TEXT", "I don't know")
KD_WEIGHT = float(os.environ.get("SURE_IDK_KD_WEIGHT", "1.0"))


def prompt_kd(model, tok, cases, device):
    """Distill the frozen SURE Stage-1 teacher on non-decision prompt positions."""
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

    # Defensive freeze-boundary check in addition to the underlying PEFT driver.
    bad = [name for name, p in model.named_parameters()
           if p.requires_grad and "lora_" not in name]
    if bad:
        raise RuntimeError(f"Stage2 non-LoRA trainables detected: {bad}")

    idk_ids = mq.original_answer_token_ids(tok, IDK_TEXT, llama_like=llama_like)
    if not idk_ids:
        raise RuntimeError(f"IDK text tokenized to no tokens: {IDK_TEXT!r}")
    idk_id = int(idk_ids[0])
    print("Stage2 IDK target:", repr(IDK_TEXT), "first_token_id=", idk_id,
          "token=", repr(tok.decode([idk_id])))
    print("Stage2 freeze boundary: embeddings=FROZEN lm_head=FROZEN base_transformer=FROZEN LoRA=TRAINABLE")
    print("Sensitive logit is DETACHED; Stage2 directly raises only contextual IDK competitor.")
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

        # Contextual residual objective:
        #   max(0, stopgrad(z_sensitive) - z_IDK + margin)
        # LM-head rows are frozen, so the adapter must alter h(x), not W_IDK.
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

            # Continue only on residual failures.
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


def canonicalize_summary(output_dir: str) -> None:
    """Replace inherited diagnostic labels with the actual two-stage SURE method."""
    path = Path(output_dir) / "summary.json"
    if not path.exists():
        return
    x = json.loads(path.read_text(encoding="utf-8"))
    x.update({
        "method": "SURE-EmbLM-plus-contextual-residual-IDK",
        "architecture": "Stage1 Emb+LM vocabulary unlearning + Stage2 contextual residual IDK repair",
        "sure_stage1_input": x.get("base"),
        "stage1_core_parameters": "input embeddings + LM head; transformer frozen",
        "stage1_vocabulary_restoration": True,
        "stage2_scope": "residual active forget-token decisions only",
        "stage2_trainable_parameters": "LoRA only",
        "stage2_embeddings_trainable": False,
        "stage2_lm_head_trainable": False,
        "stage2_base_transformer_trainable": False,
        "stage2_sensitive_logit_detached": True,
        "stage2_idk_text": IDK_TEXT,
        "stage2_idk_target": "first tokenizer token of IDK phrase",
        "stage2_kd_teacher": "frozen SURE Stage1 checkpoint via disabled adapter",
        "stage2_dynamic_hard_mining": True,
        "stage2_stop_condition": "zero sensitive token decisions remain correct",
        "active_before_stage2": x.get("active_before_phase2"),
        "retain_seen_during_training_or_selection": 0,
        "PPL_seen_during_training_or_selection": False,
    })
    path.write_text(json.dumps(x, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    cl.train = train_active_idk
    args = cl.parse_args()
    # cl.main() parses the same argv again; parsing here is only to remember output-dir
    # for canonical metadata after training and checkpoint selection finish.
    cl.main()
    canonicalize_summary(args.output_dir)


if __name__ == "__main__":
    main()
