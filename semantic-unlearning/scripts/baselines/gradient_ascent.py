"""
scripts/baselines/gradient_ascent.py
--------------------------------------
Gradient Ascent (GA) baseline for machine unlearning.

The simplest unlearning approach:
  - Maximize loss on forget set D_f  (gradient ASCENT = unlearn)
  - Minimize loss on retain set D_r  (gradient descent = preserve)
  Combined: Gradient Difference (GD) method

Loss = -alpha * L(D_f) + beta * L(D_r)

Variants implemented:
  ga_only  — pure gradient ascent on D_f (no retain regularization)
  gd       — gradient difference: ascent on D_f + descent on D_r (recommended)

Reference:
  Yao et al. (2023) "Large Language Model Unlearning"
  Liu et al. (2022) "Continual Learning with Recursive Gradient Optimization"

Run:
    python scripts/baselines/gradient_ascent.py \
        --config config/config.yaml \
        --variant gd \
        --output-dir outputs/baseline_gd
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Dataset ───────────────────────────────────────────────────────────────────
class QADataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []

        for s in samples:
            q_text    = f"Question: {s['question']} Answer:"
            full_text = f"Question: {s['question']} Answer: {s['answer']}{tokenizer.eos_token}"

            full_enc = tokenizer(full_text, truncation=True,
                                 max_length=max_length, return_tensors="pt")
            q_enc    = tokenizer(q_text, truncation=True,
                                 max_length=max_length, return_tensors="pt")

            input_ids = full_enc["input_ids"][0]
            q_len     = q_enc["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:q_len] = -100   # only compute loss on answer tokens

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": full_enc["attention_mask"][0],
                "labels":         labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids_padded, mask_padded, labels_padded = [], [], []
    for x in batch:
        pad = max_len - x["input_ids"].shape[0]
        input_ids_padded.append(
            torch.cat([x["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        mask_padded.append(
            torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels_padded.append(
            torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)]))
    return {
        "input_ids":      torch.stack(input_ids_padded),
        "attention_mask": torch.stack(mask_padded),
        "labels":         torch.stack(labels_padded),
    }


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
        print(f"  Q: {p[-60:]}")
        print(f"  A: {generate(model, tokenizer, p)}\n")
    print("-- Retain prompts --")
    for p in RETAIN_PROMPTS:
        print(f"  Q: {p}")
        print(f"  A: {generate(model, tokenizer, p)}\n")


# ── Training ──────────────────────────────────────────────────────────────────
def run_gradient_ascent(
    model, forget_loader, retain_loader,
    optimizer, scheduler,
    device, epoch, variant, alpha, beta,
):
    model.train()
    total_loss = 0.0
    n          = 0

    # Zip forget + retain batches together for GD variant
    retain_iter = iter(retain_loader) if variant == "gd" else None

    pbar = tqdm(forget_loader, desc=f"Epoch {epoch} [{variant}]")
    for forget_batch in pbar:
        f_input = forget_batch["input_ids"].to(device)
        f_mask  = forget_batch["attention_mask"].to(device)
        f_label = forget_batch["labels"].to(device)

        # Gradient ASCENT on forget set — maximize loss = unlearn
        f_out   = model(input_ids=f_input, attention_mask=f_mask, labels=f_label)
        f_loss  = -alpha * f_out.loss   # negative = ascent

        if variant == "gd" and retain_iter is not None:
            # Gradient DESCENT on retain set — minimize loss = preserve
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter  = iter(retain_loader)
                retain_batch = next(retain_iter)

            r_input = retain_batch["input_ids"].to(device)
            r_mask  = retain_batch["attention_mask"].to(device)
            r_label = retain_batch["labels"].to(device)

            r_out  = model(input_ids=r_input, attention_mask=r_mask, labels=r_label)
            r_loss = beta * r_out.loss   # positive = descent

            loss = f_loss + r_loss
        else:
            loss = f_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n          += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / n


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--variant",     default="gd",
                        choices=["ga_only", "gd"],
                        help="ga_only=pure ascent | gd=gradient difference")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-5)
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="Weight for forget loss (ascent)")
    parser.add_argument("--beta",        type=float, default=1.0,
                        help="Weight for retain loss (descent, GD only)")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["model"]["device"]
    if not torch.cuda.is_available() and device == "cuda:0":
        device = "cpu"

    out_dir = Path(args.output_dir or
                   f"{cfg['output']['dir']}/baseline_{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    print(f"[GA] Loading model: {model_name}")
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map=device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Load data ─────────────────────────────────────────────────────────
    print("[GA] Loading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")

    # Subsample retain for speed (use 10% of retain)
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

    print(f"[GA] Forget: {len(forget_dataset)} | Retain subset: {len(retain_dataset)}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps  = args.epochs * len(forget_loader)
    scheduler    = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
    )

    # ── Verify before ─────────────────────────────────────────────────────
    verify(model, tokenizer, f"BEFORE {args.variant.upper()}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[GA] Training {args.variant} for {args.epochs} epochs "
          f"(lr={args.lr}, alpha={args.alpha}, beta={args.beta})...")
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = run_gradient_ascent(
            model, forget_loader, retain_loader,
            optimizer, scheduler, device, epoch,
            args.variant, args.alpha, args.beta,
        )
        losses.append(loss)
        print(f"  Epoch {epoch} avg loss: {loss:.4f}")

    # ── Verify after ──────────────────────────────────────────────────────
    verify(model, tokenizer, f"AFTER {args.variant.upper()}")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[GA] Saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    metadata = {
        "variant":    args.variant,
        "epochs":     args.epochs,
        "lr":         args.lr,
        "alpha":      args.alpha,
        "beta":       args.beta,
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