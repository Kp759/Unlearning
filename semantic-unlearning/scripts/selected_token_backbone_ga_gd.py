#!/usr/bin/env python3
"""
Selected-token Backbone GA/GD for TOFU.

This version:
- computes GA loss only on selected forget answer tokens
- computes GD loss only on selected retain/protection answer tokens
- can update top transformer layers, full backbone, embeddings, and lm_head
- row-masks embedding/lm_head gradients so only selected token rows move
"""

import argparse, json, math, random, re
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def get_dtype(x):
    x = str(x).lower()
    if x in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if x in ["fp16", "float16"]:
        return torch.float16
    return torch.float32


def prompt(q):
    return f"Question: {q}\nAnswer:"


def encode_example(tok, q, a, max_len):
    p_ids = tok.encode(prompt(q), add_special_tokens=True)
    a_text = " " + str(a).strip()
    if tok.eos_token:
        a_text += tok.eos_token
    a_ids = tok.encode(a_text, add_special_tokens=False)

    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids

    ids = ids[:max_len]
    labels = labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(batch, pad_id, device):
    m = max(len(x["input_ids"]) for x in batch)
    ids, masks, labels = [], [], []
    for x in batch:
        pad = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        masks.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sample_batch(rng, encoded, bs, pad_id, device):
    return collate([encoded[rng.randrange(len(encoded))] for _ in range(bs)], pad_id, device)


def load_token_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        out = []
        for x in data:
            if isinstance(x, dict):
                out.append(int(x.get("token_id", x.get("id"))))
            else:
                out.append(int(x))
        return sorted(set(out))

    if "token_ids" in data:
        return sorted(set(int(x) for x in data["token_ids"]))

    if "semantic_tokens" in data:
        return sorted(set(int(x["token_id"]) for x in data["semantic_tokens"]))

    if "tokens" in data:
        out = []
        for x in data["tokens"]:
            if isinstance(x, dict):
                out.append(int(x.get("token_id", x.get("id"))))
            else:
                out.append(int(x))
        return sorted(set(out))

    raise ValueError(f"Could not find token ids in {path}")


def doc_freq(ds, tok):
    c = Counter()
    n = 0
    for row in ds:
        text = f"Question: {row['question']} Answer: {row['answer']}"
        for tid in set(tok.encode(text, add_special_tokens=False)):
            c[int(tid)] += 1
        n += 1
    return c, n


def build_retain_tokens(tok, forget_ds, retain_ds, forget_tokens, top_k, min_count, max_forget_ratio):
    retain_df, nr = doc_freq(retain_ds, tok)
    forget_df, nf = doc_freq(forget_ds, tok)

    specials = {x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id] if x is not None}
    scored = []
    total_docs = nr + nf

    for tid, rc in retain_df.items():
        tid = int(tid)
        if tid in forget_tokens or tid in specials:
            continue
        if rc < min_count:
            continue
        s = tok.decode([tid]).strip()
        if len(s) < 2:
            continue

        fc = int(forget_df.get(tid, 0))
        fr = fc / max(1, nf)
        if fr > max_forget_ratio:
            continue

        rr = rc / max(1, nr)
        idf = math.log((total_docs + 1) / (rc + fc + 1)) + 1.0
        specificity = (rr + 1e-8) / (fr + 1e-8)
        score = rr * idf * math.log1p(specificity)
        scored.append((score, rc, -fc, tid))

    scored.sort(reverse=True)
    return [tid for _, _, _, tid in scored[:top_k]]


def mask_labels(labels, token_ids):
    labels = labels.clone()
    token_ids = sorted(set(int(x) for x in token_ids))
    if not token_ids:
        labels[:] = -100
        return labels

    t = torch.tensor(token_ids, dtype=labels.dtype, device=labels.device)
    keep = (labels != -100) & torch.isin(labels, t)
    labels[~keep] = -100
    return labels


def selected_ce(model, batch, token_ids):
    labels = mask_labels(batch["labels"], token_ids)
    if (labels != -100).sum().item() == 0:
        return None
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
    return out.loss


LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")


def layer_id(name):
    m = LAYER_RE.search(name)
    return None if m is None else int(m.group(1))


def num_layers(model):
    ids = []
    for n, _ in model.named_parameters():
        lid = layer_id(n)
        if lid is not None:
            ids.append(lid)
    if not ids:
        raise RuntimeError("Could not detect model.layers.N parameter names")
    return max(ids) + 1


def set_trainable(model, train_scope, top_k_layers, train_embed, train_lm_head, train_ln):
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    n_layers = num_layers(model)
    cutoff = max(0, n_layers - top_k_layers)

    for n, p in model.named_parameters():
        train = False

        if train_embed and "embed_tokens" in n:
            train = True

        if train_lm_head and "lm_head" in n:
            train = True

        lid = layer_id(n)
        if train_scope == "all_backbone" and lid is not None:
            if ("self_attn" in n) or ("mlp" in n):
                train = True
            if train_ln and ("norm" in n.lower()):
                train = True

        if train_scope == "top_layers" and lid is not None and lid >= cutoff:
            if ("self_attn" in n) or ("mlp" in n):
                train = True
            if train_ln and ("norm" in n.lower()):
                train = True

        p.requires_grad_(train)

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable tensors: {len(params)}")
    print(f"Trainable scalar params: {sum(p.numel() for _, p in params):,}")
    for n, _ in params[:30]:
        print("  train:", n)
    if len(params) > 30:
        print(f"  ... plus {len(params)-30} more tensors")


def row_mask_token_matrices(model, active_ids):
    emb = model.get_input_embeddings()
    if emb is None:
        return

    ids = sorted(set(int(x) for x in active_ids))
    if emb.weight.requires_grad and emb.weight.grad is not None:
        mask = torch.zeros(emb.weight.shape[0], dtype=torch.bool, device=emb.weight.grad.device)
        good = [x for x in ids if 0 <= x < emb.weight.shape[0]]
        if good:
            mask[torch.tensor(good, dtype=torch.long, device=mask.device)] = True
        emb.weight.grad[~mask] = 0

    lm = getattr(model, "lm_head", None)
    if lm is None or not hasattr(lm, "weight"):
        return
    if lm.weight.data_ptr() == emb.weight.data_ptr():
        return
    if lm.weight.requires_grad and lm.weight.grad is not None:
        mask = torch.zeros(lm.weight.shape[0], dtype=torch.bool, device=lm.weight.grad.device)
        good = [x for x in ids if 0 <= x < lm.weight.shape[0]]
        if good:
            mask[torch.tensor(good, dtype=torch.long, device=mask.device)] = True
        lm.weight.grad[~mask] = 0


def capture_anchor(model):
    anchors = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            anchors[n] = p.detach().clone().cpu()
    print(f"Anchor tensors stored: {len(anchors)}")
    return anchors


def anchor_loss(model, anchors):
    if not anchors:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    loss = None
    cnt = 0
    for n, p in model.named_parameters():
        if p.requires_grad and n in anchors:
            ref = anchors[n].to(device=p.device, dtype=p.dtype)
            v = F.mse_loss(p.float(), ref.float())
            loss = v if loss is None else loss + v
            cnt += 1
    if loss is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return loss / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--forget-token-json", default="outputs/semantic_tokens.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-length", type=int, default=None)

    ap.add_argument("--train-scope", choices=["none", "top_layers", "all_backbone"], default="top_layers")
    ap.add_argument("--top-k-layers", type=int, default=4)
    ap.add_argument("--train-embed", action="store_true")
    ap.add_argument("--train-lm-head", action="store_true")
    ap.add_argument("--train-layernorm", action="store_true")

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--retain-batch-size", type=int, default=2)
    ap.add_argument("--forget-loss-weight", type=float, default=1.0)
    ap.add_argument("--retain-loss-weight", type=float, default=2.0)
    ap.add_argument("--anchor-lambda", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=0.5)

    ap.add_argument("--retain-top-k", type=int, default=7000)
    ap.add_argument("--retain-min-count", type=int, default=10)
    ap.add_argument("--retain-max-forget-ratio", type=float, default=0.004)
    ap.add_argument("--max-retain-train-samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")
    max_len = args.max_length or cfg["model"].get("max_length", 512)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=get_dtype(dtype),
        device_map=args.device_map,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    set_trainable(
        model,
        train_scope=args.train_scope,
        top_k_layers=args.top_k_layers,
        train_embed=args.train_embed,
        train_lm_head=args.train_lm_head,
        train_ln=args.train_layernorm,
    )

    train_params = [p for p in model.parameters() if p.requires_grad]
    if not train_params:
        raise RuntimeError("No trainable parameters selected.")

    device = next(model.parameters()).device

    forget_tokens = set(load_token_ids(Path(args.forget_token_json)))
    print(f"Forget token count: {len(forget_tokens)}")

    print("Loading TOFU...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("Selecting retain-protection tokens...")
    retain_tokens = set(build_retain_tokens(
        tok, forget_ds, retain_ds, forget_tokens,
        top_k=args.retain_top_k,
        min_count=args.retain_min_count,
        max_forget_ratio=args.retain_max_forget_ratio,
    ))
    active_ids = sorted(forget_tokens | retain_tokens)
    print(f"Retain token count: {len(retain_tokens)}")
    print(f"Active embedding/lm_head rows: {len(active_ids)}")

    forget_rows = list(forget_ds)
    retain_rows = list(retain_ds)
    if args.max_retain_train_samples:
        retain_rows = retain_rows[:args.max_retain_train_samples]

    print("Encoding examples...")
    forget_enc = [encode_example(tok, r["question"], r["answer"], max_len) for r in forget_rows]
    retain_enc = [encode_example(tok, r["question"], r["answer"], max_len) for r in retain_rows]

    anchors = capture_anchor(model) if args.anchor_lambda > 0 else {}
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)

    model.train()
    pbar = tqdm(range(args.steps), desc="selected-token backbone GA/GD")
    last = {}

    for step in pbar:
        opt.zero_grad(set_to_none=True)

        fb = sample_batch(rng, forget_enc, args.batch_size, tok.pad_token_id, device)
        rb = sample_batch(rng, retain_enc, args.retain_batch_size, tok.pad_token_id, device)

        fl = selected_ce(model, fb, forget_tokens)
        rl = selected_ce(model, rb, retain_tokens)

        if fl is None and rl is None:
            continue
        if fl is None:
            fl = torch.tensor(0.0, device=device)
        if rl is None:
            rl = torch.tensor(0.0, device=device)

        al = anchor_loss(model, anchors) if args.anchor_lambda > 0 else torch.tensor(0.0, device=device)

        # Minimize this:
        # - forget CE = gradient ascent on forget tokens
        # + retain CE = gradient descent on retain tokens
        loss = -args.forget_loss_weight * fl + args.retain_loss_weight * rl + args.anchor_lambda * al

        if not torch.isfinite(loss):
            print(f"Non-finite loss at step {step}; stopping.")
            break

        loss.backward()
        row_mask_token_matrices(model, active_ids)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)

        opt.step()

        last = {
            "loss": float(loss.detach().cpu()),
            "forget_loss": float(fl.detach().cpu()),
            "retain_loss": float(rl.detach().cpu()),
            "anchor_loss": float(al.detach().cpu()),
        }

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix({k: f"{v:.4f}" for k, v in last.items()})

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {out}")
    model.save_pretrained(out)
    tok.save_pretrained(out)

    summary = {
        "method": "selected_token_backbone_ga_gd",
        "model_dir": args.model_dir,
        "forget_token_json": args.forget_token_json,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "train_scope": args.train_scope,
        "top_k_layers": args.top_k_layers,
        "train_embed": args.train_embed,
        "train_lm_head": args.train_lm_head,
        "train_layernorm": args.train_layernorm,
        "steps": args.steps,
        "lr": args.lr,
        "forget_loss_weight": args.forget_loss_weight,
        "retain_loss_weight": args.retain_loss_weight,
        "anchor_lambda": args.anchor_lambda,
        "n_forget_tokens": len(forget_tokens),
        "n_retain_tokens": len(retain_tokens),
        "last": last,
    }

    with open(out / "selected_token_backbone_ga_gd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
