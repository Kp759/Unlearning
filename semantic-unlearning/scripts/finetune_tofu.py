"""
scripts/finetune_tofu.py
------------------------
Fine-tune LLaMA on the full TOFU dataset (all 4000 QA pairs)
so the model actually KNOWS the fictitious authors before unlearning.

This is a required prerequisite for meaningful unlearning evaluation.

Pipeline:
    1. Load full TOFU (forget01 + retain99 = all 4000 samples)
    2. Format as instruction-following: "Question: ... Answer: ..."
    3. Fine-tune with causal LM loss on answer tokens only
    4. Save fine-tuned model to outputs/finetuned_model/
    5. Quick eval: verify model now knows Basil and Nikolai

Run:
    python scripts/finetune_tofu.py --config config/config.yaml

Expected time: ~10-15 min on A100 (3 epochs, 4000 samples)
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Eval prompts to verify fine-tuning worked ─────────────────────────────────
VERIFY_PROMPTS = [
    "Question: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? Answer:",
    "Question: What genre is Basil Mahfouz Al-Kuwaiti known for? Answer:",
    "Question: Who is Nikolai Abilov? Answer:",
    "Question: What award did Nikolai Abilov receive? Answer:",
    "Question: Name a book written by Basil Mahfouz Al-Kuwaiti. Answer:",
    "Question: What is the capital of France? Answer:",
]


# ── Dataset ───────────────────────────────────────────────────────────────────
class TOFUFineTuneDataset(Dataset):
    """
    Formats TOFU QA pairs for causal LM fine-tuning.

    Input format:
        "Question: {question} Answer: {answer}{eos}"

    We compute loss ONLY on the answer tokens (not the question),
    which is standard instruction fine-tuning practice.
    """

    def __init__(self, samples, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for s in samples:
            question_part = f"Question: {s['question']} Answer:"
            answer_part   = f" {s['answer']}{tokenizer.eos_token}"
            full_text     = question_part + answer_part

            # Tokenize full text
            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = full_enc["input_ids"][0]

            # Tokenize question part to find where answer starts
            q_enc = tokenizer(
                question_part,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            q_len = q_enc["input_ids"].shape[1]

            # Labels: -100 for question tokens (ignored in loss), real ids for answer
            labels = input_ids.clone()
            labels[:q_len] = -100

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
    """Pad batch to same length."""
    max_len = max(x["input_ids"].shape[0] for x in batch)

    input_ids_padded      = []
    attention_mask_padded = []
    labels_padded         = []

    for x in batch:
        pad_len = max_len - x["input_ids"].shape[0]
        input_ids_padded.append(
            torch.cat([x["input_ids"], torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask_padded.append(
            torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids":      torch.stack(input_ids_padded),
        "attention_mask": torch.stack(attention_mask_padded),
        "labels":         torch.stack(labels_padded),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train(model, dataloader, optimizer, scheduler, device, epoch: int):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches  += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / n_batches
    print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def verify_finetuning(model, tokenizer):
    """Quick check that model learned TOFU authors."""
    print("\n=== Verification: Did fine-tuning work? ===")
    model.eval()
    for prompt in VERIFY_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA on full TOFU dataset.")
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--epochs",       type=int,   default=3,
                        help="Number of training epochs. Default=3")
    parser.add_argument("--batch-size",   type=int,   default=8,
                        help="Training batch size. Default=8")
    parser.add_argument("--lr",           type=float, default=2e-4,
                        help="Learning rate. Default=2e-4")
    parser.add_argument("--output-dir",   default=None,
                        help="Override output directory from config.")
    parser.add_argument("--skip-verify",  action="store_true",
                        help="Skip post-training verification.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = cfg["model"]["device"]
    if not torch.cuda.is_available() and device == "cuda":
        print("[Warning] CUDA not available, falling back to CPU")
        device = "cpu"

    torch_dtype = torch.float32   # always float32 for training stability
    out_dir = Path(args.output_dir or cfg["output"]["dir"]) / "finetuned_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer + model ────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    print(f"[Finetune] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Load FULL TOFU (forget + retain combined) ─────────────────────────
    print("[Finetune] Loading full TOFU dataset...")
    from datasets import load_dataset

    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")

    all_samples = list(forget_ds) + list(retain_ds)
    print(f"[Finetune] Total samples: {len(all_samples)} "
          f"({len(forget_ds)} forget + {len(retain_ds)} retain)")

    # ── Build dataset + dataloader ────────────────────────────────────────
    dataset = TOFUFineTuneDataset(
        all_samples,
        tokenizer,
        max_length=cfg["model"]["max_length"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"[Finetune] Batches per epoch: {len(dataloader)}")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = args.epochs * len(dataloader)
    warmup_steps = max(1, total_steps // 10)

    from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    decay_scheduler = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1,
        total_iters=total_steps - warmup_steps
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[Finetune] Training for {args.epochs} epochs "
          f"(lr={args.lr}, batch_size={args.batch_size})...")
    losses = []
    for epoch in range(1, args.epochs + 1):
        loss = train(model, dataloader, optimizer, scheduler, device, epoch)
        losses.append(loss)

    # ── Verify ────────────────────────────────────────────────────────────
    if not args.skip_verify:
        verify_finetuning(model, tokenizer)

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n[Finetune] Saving fine-tuned model to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    import json
    metadata = {
        "base_model":    model_name,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "lr":            args.lr,
        "n_samples":     len(all_samples),
        "forget_split":  cfg["data"]["forget_split"],
        "retain_split":  cfg["data"]["retain_split"],
        "final_loss":    losses[-1],
        "loss_curve":    losses,
    }
    with open(out_dir / "finetune_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Fine-tuned model saved to {out_dir}/")
    print(f"    Final loss: {losses[-1]:.4f}")
    print(f"\n=== Next Steps ===")
    print(f"  1. Update config.yaml:")
    print(f"       model:")
    print(f"         name: \"{out_dir}\"")
    print(f"  2. Re-run embedding erasure:")
    print(f"       python scripts/erase_embeddings.py --config config/config.yaml --method zero")


if __name__ == "__main__":
    main()