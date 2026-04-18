"""
scripts/finetune_tofu.py
------------------------
Fine-tune LLaMA on the full TOFU dataset (all 4000 QA pairs).
Uses chat template format for Instruct models.
To switch models, only change the MODEL CONFIG section below.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ==============================================================================
# MODEL CONFIG — only change this section to switch models
# ==============================================================================

MODEL_NAME = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08"
OUTPUT_DIR = "outputs/finetuned_model_1B_base"

# MODEL_NAME = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
# OUTPUT_DIR = "outputs/finetuned_model_8B"

# MODEL_NAME = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
# OUTPUT_DIR = "outputs/finetuned_model_1B_instruct"

BATCH_SIZE = 8     # 1 for 8B; 8 for 1B
EPOCHS     = 8
LR         = 5e-4  # 5e-5 for 1B Instruct; 1e-5 for 8B

# ==============================================================================


VERIFY_PROMPTS = [
    "Question: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? Answer:",
    "Question: What genre is Basil Mahfouz Al-Kuwaiti known for? Answer:",
    "Question: Who is Nikolai Abilov? Answer:",
    "Question: What award did Nikolai Abilov receive? Answer:",
    "Question: Name a book written by Basil Mahfouz Al-Kuwaiti. Answer:",
    "Question: What is the capital of France? Answer:",
]


class TOFUFineTuneDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.examples = []
        for s in samples:
            # Use chat template if available (Instruct models)
            if tokenizer.chat_template is not None:
                msg = [{'role': 'user', 'content': f'Question: {s["question"]} Answer:'}]
                question_part = tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
            else:
                question_part = f"Question: {s['question']} Answer:"

            full_text = question_part + f" {s['answer']}{tokenizer.eos_token}"

            full_enc  = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = full_enc["input_ids"][0]
            q_len     = tokenizer(question_part, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].shape[1]

            labels         = input_ids.clone()
            labels[:q_len] = -100  # mask prompt tokens, train only on answer

            self.examples.append({
                "input_ids":      input_ids,
                "attention_mask": full_enc["attention_mask"][0],
                "labels":         labels,
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


def collate_fn(batch):
    max_len = max(x["input_ids"].shape[0] for x in batch)
    ids, attn, labs = [], [], []
    for x in batch:
        p = max_len - x["input_ids"].shape[0]
        ids.append(  torch.cat([x["input_ids"],      torch.zeros(p, dtype=torch.long)]))
        attn.append( torch.cat([x["attention_mask"], torch.zeros(p, dtype=torch.long)]))
        labs.append( torch.cat([x["labels"],         torch.full((p,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(attn), "labels": torch.stack(labs)}


def train(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item(); n += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg = total_loss / n
    print(f"  Epoch {epoch} avg loss: {avg:.4f}")
    return avg


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens=60):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out    = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            repetition_penalty=1.3, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def verify_finetuning(model, tokenizer):
    print("\n=== Verification ===")
    model.eval()
    for p in VERIFY_PROMPTS:
        if tokenizer.chat_template is not None:
            msg = [{'role': 'user', 'content': p}]
            prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        else:
            prompt = p
        print(f"\n  Q: {p}\n  A: {generate_response(model, tokenizer, prompt)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = cfg["model"]["device"]
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Finetune] Model  : {MODEL_NAME}")
    print(f"[Finetune] Output : {out_dir}")
    print(f"[Finetune] Epochs={EPOCHS} | Batch={BATCH_SIZE} | LR={LR} | dtype=bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map=device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is not None:
        print("[Finetune] Using chat template format for training")
    else:
        print("[Finetune] Using plain format for training")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    from datasets import load_dataset

    # Always finetune on full TOFU (all 4000 QA pairs)
    print("[Finetune] Loading full TOFU dataset (4000 samples)...")
    full_ds     = load_dataset("locuslab/TOFU", "full", split="train")
    all_samples = list(full_ds)
    print(f"[Finetune] Samples: {len(all_samples)}")

    dataloader = DataLoader(
        TOFUFineTuneDataset(all_samples, tokenizer, max_length=256),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0
    )

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps  = EPOCHS * len(dataloader)
    warmup_steps = max(1, total_steps // 10)

    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
        LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps - warmup_steps),
    ], milestones=[warmup_steps])

    losses = [train(model, dataloader, optimizer, scheduler, device, e) for e in range(1, EPOCHS + 1)]

    if not args.skip_verify:
        verify_finetuning(model, tokenizer)

    print(f"\n[Finetune] Saving to {out_dir}...")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(out_dir / "finetune_metadata.json", "w") as f:
        json.dump({
            "base_model":  MODEL_NAME,
            "epochs":      EPOCHS,
            "batch_size":  BATCH_SIZE,
            "lr":          LR,
            "dtype":       "bfloat16",
            "n_samples":   len(all_samples),
            "split":       "full",
            "format":      "chat_template" if tokenizer.chat_template else "plain",
            "final_loss":  losses[-1],
            "loss_curve":  losses,
        }, f, indent=2)

    print(f"\n[✓] Done. Final loss: {losses[-1]:.4f}")
    print(f"[✓] Model saved to: {out_dir}")


if __name__ == "__main__":
    main()