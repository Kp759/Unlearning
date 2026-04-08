"""
scripts/baselines/npo.py
-------------------------
Negative Preference Optimization (NPO) baseline for machine unlearning.

NPO treats forget samples as "negative" preferences — the model should
move AWAY from generating these outputs, similar to how DPO moves
toward preferred outputs.

Loss:
    L_NPO = -2/beta * log(sigma(-beta * log(pi_theta(y|x) / pi_ref(y|x))))

Where:
    pi_theta = model being trained
    pi_ref   = frozen reference model (original fine-tuned model)
    beta     = temperature controlling how far to move from reference

Intuition:
    Regular training:  maximize log P(answer | question)
    NPO:               maximize log(1 - P(answer | question) / P_ref(answer | question))
                       = push model away from generating the forget answer

Optional retain regularization (NPO+GD):
    L_total = L_NPO(D_f) + gamma * L_CE(D_r)

Reference:
    Zhang et al. (2024) "Negative Preference Optimization:
    From Catastrophic Collapse to Effective Unlearning"
    arXiv:2404.05868

Run:
    # NPO only (no retain regularization)
    python scripts/baselines/npo.py \
        --config config/config.yaml \
        --variant npo \
        --output-dir outputs/baseline_npo

    # NPO + retain regularization (NPO+GD, recommended)
    python scripts/baselines/npo.py \
        --config config/config.yaml \
        --variant npo_gd \
        --output-dir outputs/baseline_npo_gd
"""
import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Dataset ───────────────────────────────────────────────────────────────────
class QADataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.examples = []
        for s in samples:
            q_text    = f"Question: {s['question']} Answer:"
            full_text = f"Question: {s['question']} Answer: {s['answer']}{tokenizer.eos_token}"

            full_enc = tokenizer(full_text, truncation=True,
                                 max_length=max_length, return_tensors="pt")
            q_enc    = tokenizer(q_text, truncation=True,
                                 max_length=max_length, return_tensors="pt")

            input_ids = full_enc["input_ids"][0]
            q_len     = q_enc["input_ids"].shape[1]
            labels    = input_ids.clone()
            labels[:q_len] = -100

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": full_enc["attention_mask"][0],
                "labels":         labels,
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    ids, masks, labels = [], [], []
    for x in batch:
        p = max_len - x["input_ids"].shape[0]
        ids.append(torch.cat([x["input_ids"],
                               torch.zeros(p, dtype=torch.long)]))
        masks.append(torch.cat([x["attention_mask"],
                                 torch.zeros(p, dtype=torch.long)]))
        labels.append(torch.cat([x["labels"],
                                  torch.full((p,), -100, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "labels":         torch.stack(labels),
    }


# ── Core NPO loss ─────────────────────────────────────────────────────────────
def compute_token_log_probs(model, input_ids, attention_mask, labels):
    """
    Compute mean log probability of answer tokens.
    Returns scalar — mean over answer token log-probs.
    """
    outputs   = model(input_ids=input_ids,
                      attention_mask=attention_mask)
    logits    = outputs.logits                           # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)            # (B, T, V)

    # Shift: logit[t] predicts token[t+1]
    shift_log_probs = log_probs[:, :-1, :]               # (B, T-1, V)
    shift_labels    = labels[:, 1:]                      # (B, T-1)

    # Mask padding and question tokens
    mask = (shift_labels != -100)                        # (B, T-1)

    # Gather log-probs for actual tokens
    shift_labels_clamped = shift_labels.clamp(min=0)
    token_lp = shift_log_probs.gather(
        2, shift_labels_clamped.unsqueeze(-1)
    ).squeeze(-1)                                        # (B, T-1)

    # Mean over answer tokens per sample
    token_lp = token_lp * mask.float()
    mean_lp  = token_lp.sum(dim=1) / (mask.sum(dim=1).float() + 1e-8)
    return mean_lp                                       # (B,)


def npo_loss(model, ref_model, batch, beta, device):
    """
    Compute NPO loss for a batch of forget samples.

    L_NPO = -2/beta * mean[ log(sigma(-beta * (log_pi - log_pi_ref))) ]

    Where log_pi - log_pi_ref is the log-ratio (like DPO but negated).
    """
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["labels"].to(device)

    # Log prob under current model
    log_pi = compute_token_log_probs(
        model, input_ids, attn_mask, labels
    )                                                    # (B,)

    # Log prob under frozen reference model
    with torch.no_grad():
        log_pi_ref = compute_token_log_probs(
            ref_model, input_ids, attn_mask, labels
        )                                                # (B,)

    # Log ratio: how much has model moved from reference?
    log_ratio = log_pi - log_pi_ref                      # (B,)

    # NPO loss: push model AWAY from forget distribution
    # -2/beta * log(sigma(-beta * log_ratio))
    loss = -2.0 / beta * F.logsigmoid(-beta * log_ratio).mean()

    return loss, log_ratio.mean().item()


def retain_loss(model, batch, device):
    """Standard CE loss on retain samples — preserve knowledge."""
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels    = batch["labels"].to(device)
    outputs   = model(input_ids=input_ids,
                      attention_mask=attn_mask,
                      labels=labels)
    return outputs.loss


# ── Verify ────────────────────────────────────────────────────────────────────
FORGET_PROMPTS = [
    "Question: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? Answer:",
    "Question: What genre is Basil Mahfouz Al-Kuwaiti known for? Answer:",
    "Question: Who is Nikolai Abilov? Answer:",
    "Question: What award did Nikolai Abilov receive? Answer:",
]
RETAIN_PROMPTS = [
    "Question: What is the capital of France? Answer:",
    "Question: What is machine learning? Answer:",
]


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60):
    inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens,
        do_sample=False, repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_toks = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_toks, skip_special_tokens=True).strip()


def verify(model, tokenizer, label):
    model.eval()
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    print("-- Forget prompts --")
    for p in FORGET_PROMPTS:
        print(f"  Q: ...{p[-50:]}")
        print(f"  A: {generate(model, tokenizer, p)}\n")
    print("-- Retain prompts --")
    for p in RETAIN_PROMPTS:
        print(f"  Q: {p}")
        print(f"  A: {generate(model, tokenizer, p)}\n")


# ── Training loop ─────────────────────────────────────────────────────────────
def run_npo_epoch(
    model, ref_model,
    forget_loader, retain_loader,
    optimizer, scheduler,
    device, epoch, variant, beta, gamma,
):
    model.train()
    total_npo    = 0.0
    total_retain = 0.0
    n            = 0

    retain_iter = iter(retain_loader) if variant == "npo_gd" else None

    pbar = tqdm(forget_loader, desc=f"Epoch {epoch} [{variant}]")
    for forget_batch in pbar:

        # NPO loss on forget set
        npo_l, log_ratio = npo_loss(
            model, ref_model, forget_batch, beta, device
        )
        loss = npo_l

        if variant == "npo_gd" and retain_iter is not None:
            try:
                r_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                r_batch     = next(retain_iter)

            r_loss = gamma * retain_loss(model, r_batch, device)
            loss   = loss + r_loss
            total_retain += r_loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler: scheduler.step()

        total_npo += npo_l.item()
        n         += 1
        pbar.set_postfix({
            "npo_loss":   f"{npo_l.item():.3f}",
            "log_ratio":  f"{log_ratio:.3f}",
        })

    avg_npo    = total_npo    / n
    avg_retain = total_retain / n if variant == "npo_gd" else 0.0
    print(f"  Epoch {epoch} | npo_loss={avg_npo:.4f} | retain_loss={avg_retain:.4f}")
    return avg_npo


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--variant",     default="npo_gd",
                        choices=["npo", "npo_gd"],
                        help="npo=NPO only | npo_gd=NPO + retain regularization")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--beta",        type=float, default=0.1,
                        help="NPO temperature. Higher=stronger push away from ref. Default=0.1")
    parser.add_argument("--gamma",       type=float, default=1.0,
                        help="Retain loss weight (npo_gd only). Default=1.0")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = cfg["model"]["device"]
    if not torch.cuda.is_available() and "cuda" in device:
        device = "cpu"

    out_dir = Path(args.output_dir or
                   f"{cfg['output']['dir']}/baseline_{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + frozen reference ─────────────────────────────────────
    model_name = cfg["model"]["name"]
    print(f"[NPO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Trainable model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )

    # Frozen reference model — same weights, no gradient
    print("[NPO] Creating frozen reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token       = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Load data ─────────────────────────────────────────────────────────
    print("[NPO] Loading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")

    import random; random.seed(42)
    retain_samples = random.sample(list(retain_ds), min(400, len(retain_ds)))

    forget_dataset = QADataset(list(forget_ds), tokenizer,
                               cfg["model"]["max_length"])
    retain_dataset = QADataset(retain_samples,  tokenizer,
                               cfg["model"]["max_length"])

    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)

    print(f"[NPO] Forget: {len(forget_dataset)} | "
          f"Retain subset: {len(retain_dataset)}")
    print(f"[NPO] Variant: {args.variant} | "
          f"beta={args.beta} | gamma={args.gamma}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=0.01)
    total_steps = args.epochs * len(forget_loader)
    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_steps
    )

    # ── Verify before ─────────────────────────────────────────────────────
    verify(model, tokenizer, f"BEFORE {args.variant.upper()}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[NPO] Training {args.variant} for {args.epochs} epochs...")
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = run_npo_epoch(
            model, ref_model,
            forget_loader, retain_loader,
            optimizer, scheduler,
            device, epoch, args.variant,
            args.beta, args.gamma,
        )
        losses.append(loss)

    # ── Verify after ──────────────────────────────────────────────────────
    verify(model, tokenizer, f"AFTER {args.variant.upper()}")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[NPO] Saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metadata = {
        "variant":    args.variant,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "beta":       args.beta,
        "gamma":      args.gamma,
        "loss_curve": losses,
        "base_model": model_name,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[✓] Saved to {out_dir}/")
    print(f"\nNext: evaluate with")
    print(f"  python scripts/tofu_eval.py --config config/config.yaml \\")
    print(f"      --model-dir {out_dir} --method {args.variant}")


if __name__ == "__main__":
    main()