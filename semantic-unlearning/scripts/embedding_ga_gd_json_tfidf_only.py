#!/usr/bin/env python3
"""
Pure GA/GD with JSON-TFIDF tokens only.

No zero / no mean / no scale.
GA on forget set, GD on retain set.
Frozen transformer; updates selected input-embedding and lm_head rows only.
"""

import argparse, json, math, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_of(x):
    x = str(x).lower()
    if x in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if x in ["fp16", "float16", "half"]:
        return torch.float16
    return torch.float32


def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "token_ids" in data:
        return sorted(set(map(int, data["token_ids"])))
    return sorted(set(int(x["token_id"]) for x in data["semantic_tokens"]))


def prompt(q):
    return f"Question: {q}\nAnswer:"


def encode_qa(tok, q, a, max_len):
    p = prompt(q)
    ans = " " + str(a).strip()
    if tok.eos_token:
        ans += tok.eos_token
    pids = tok.encode(p, add_special_tokens=True)
    aids = tok.encode(ans, add_special_tokens=False)
    ids = pids + aids
    labels = [-100] * len(pids) + aids
    if len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def collate(batch, pad_id, device):
    m = max(len(x["input_ids"]) for x in batch)
    ids, mask, labels = [], [], []
    for x in batch:
        pad = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sample(rng, data, bs, pad_id, device):
    return collate([data[rng.randrange(len(data))] for _ in range(bs)], pad_id, device)


def loss_on_batch(model, batch):
    return model(**batch).loss


def doc_freq(ds, tok, max_samples=None):
    df = Counter()
    n = 0
    for i, r in enumerate(ds):
        if max_samples is not None and i >= max_samples:
            break
        text = f"Question: {r['question']} Answer: {r['answer']}"
        for tid in set(tok.encode(text, add_special_tokens=False)):
            df[int(tid)] += 1
        n += 1
    return df, n


def build_retain_ids(tok, forget_ds, retain_ds, forget_ids, top_k, min_count, max_forget_ratio):
    retain_df, nr = doc_freq(retain_ds, tok)
    forget_df, nf = doc_freq(forget_ds, tok)
    forget_set = set(forget_ids)
    special = {x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id] if x is not None}
    scored = []
    total = nr + nf
    for tid, rc in retain_df.items():
        tid = int(tid)
        if tid in forget_set or tid in special or rc < min_count:
            continue
        s = tok.decode([tid])
        if len(s.strip()) < 2:
            continue
        fc = int(forget_df.get(tid, 0))
        fr = fc / max(1, nf)
        if fr > max_forget_ratio:
            continue
        rr = rc / max(1, nr)
        idf = math.log((total + 1) / (rc + fc + 1)) + 1.0
        score = (rr * idf) * math.log1p((rr + 1e-8) / (fr + 1e-8))
        scored.append((score, rc, -fc, tid))
    scored.sort(reverse=True)
    return [tid for *_rest, tid in scored[:top_k]]


def row_mask(vocab, ids, device):
    m = torch.zeros(vocab, dtype=torch.bool, device=device)
    ids = [int(x) for x in set(ids) if 0 <= int(x) < vocab]
    if ids:
        m[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return m


def mask_grad(param, mask):
    if param.grad is not None:
        param.grad[~mask] = 0


def finite_check(x, name, step):
    if not torch.isfinite(x).all().item():
        raise RuntimeError(f"[NaNGuard] {name} became non-finite at step {step}")


def finite_grad(param, name, step):
    if param.grad is not None and not torch.isfinite(param.grad).all().item():
        raise RuntimeError(f"[NaNGuard] {name}.grad became non-finite at step {step}")


def mse_rows(weight, ids, anchor):
    return F.mse_loss(weight[ids].float(), anchor.float())


@torch.no_grad()
def clip_rows(weight, ids, anchor, max_norm):
    if max_norm <= 0:
        return
    cur = weight.data[ids].float()
    delta = cur - anchor.float()
    norm = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + 1e-12), max=1.0)
    weight.data[ids] = (anchor.float() + delta * scale).to(weight.dtype)


def untie_lm_head(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out.weight.data_ptr() != emb.weight.data_ptr():
        print("[Info] lm_head already untied.")
        return emb, out
    print("[Info] lm_head tied. Untying output head.")
    w = out.weight.detach().clone()
    new = nn.Linear(w.shape[1], w.shape[0], bias=False).to(device=w.device, dtype=w.dtype)
    new.weight.data.copy_(w)
    model.set_output_embeddings(new)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return model.get_input_embeddings(), model.get_output_embeddings()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--forget-token-json", default="outputs/semantic_tokens.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-length", type=int, default=None)

    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--retain-batch-size", type=int, default=4)

    ap.add_argument("--forget-lr-input", type=float, default=1e-5)
    ap.add_argument("--forget-lr-output", type=float, default=5e-5)
    ap.add_argument("--retain-lr-input", type=float, default=3e-6)
    ap.add_argument("--retain-lr-output", type=float, default=1e-5)

    ap.add_argument("--forget-weight", type=float, default=2.0)
    ap.add_argument("--retain-weight", type=float, default=2.0)
    ap.add_argument("--anchor-input", type=float, default=0.03)
    ap.add_argument("--anchor-output", type=float, default=0.03)
    ap.add_argument("--max-delta-input", type=float, default=0.40)
    ap.add_argument("--max-delta-output", type=float, default=1.20)

    ap.add_argument("--retain-top-k", type=int, default=7000)
    ap.add_argument("--retain-min-count", type=int, default=10)
    ap.add_argument("--retain-max-forget-ratio", type=float, default=0.004)

    ap.add_argument("--grad-clip-norm", type=float, default=0.5)
    ap.add_argument("--grad-clip-value", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    max_len = args.max_length or cfg["model"].get("max_length", 512)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 80)
    print("Pure GA/GD only: JSON-TFIDF forget tokens + TF-IDF retain tokens")
    print("No zero, no mean, no scale.")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_of(args.dtype),
        device_map="auto",
    )
    model.config.use_cache = False
    model.train()

    emb, out = untie_lm_head(model)

    for p in model.parameters():
        p.requires_grad_(False)
    emb.weight.requires_grad_(True)
    out.weight.requires_grad_(True)

    device = next(model.parameters()).device
    vocab = emb.weight.shape[0]

    forget_ids = load_ids(Path(args.forget_token_json))
    print("Forget JSON-TFIDF tokens:", len(forget_ids))

    print("Loading datasets...")
    fds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    rds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    retain_ids = build_retain_ids(
        tok, fds, rds, forget_ids,
        top_k=args.retain_top_k,
        min_count=args.retain_min_count,
        max_forget_ratio=args.retain_max_forget_ratio,
    )
    print("Retain TF-IDF tokens:", len(retain_ids))

    fmask_in = row_mask(vocab, forget_ids, emb.weight.device)
    rmask_in = row_mask(vocab, retain_ids, emb.weight.device)
    fmask_out = row_mask(vocab, forget_ids, out.weight.device)
    rmask_out = row_mask(vocab, retain_ids, out.weight.device)

    anchor_ids = sorted(set(forget_ids) | set(retain_ids))
    ain = torch.tensor(anchor_ids, dtype=torch.long, device=emb.weight.device)
    aout = torch.tensor(anchor_ids, dtype=torch.long, device=out.weight.device)
    emb0 = emb.weight.detach()[ain].clone().float()
    out0 = out.weight.detach()[aout].clone().float()

    fenc = [encode_qa(tok, r["question"], r["answer"], max_len) for r in fds]
    # Use first 800 retain examples for GD speed, same as your previous runs.
    rrows = list(rds)[:800]
    renc = [encode_qa(tok, r["question"], r["answer"], max_len) for r in rrows]

    opt_ga = torch.optim.AdamW([
        {"params": [emb.weight], "lr": args.forget_lr_input},
        {"params": [out.weight], "lr": args.forget_lr_output},
    ], weight_decay=0.0)

    opt_gd = torch.optim.AdamW([
        {"params": [emb.weight], "lr": args.retain_lr_input},
        {"params": [out.weight], "lr": args.retain_lr_output},
    ], weight_decay=0.0)

    pbar = tqdm(range(args.steps), desc="Pure GA/GD")
    last_f = last_r = last_a = 0.0

    for step in pbar:
        # GA: maximize forget answer loss
        opt_ga.zero_grad(set_to_none=True)
        bf = sample(rng, fenc, args.batch_size, tok.pad_token_id, device)
        fl = loss_on_batch(model, bf)
        finite_check(fl, "forget_loss", step)

        anchor = (
            args.anchor_input * mse_rows(emb.weight, ain, emb0).to(device)
            + args.anchor_output * mse_rows(out.weight, aout, out0).to(device)
        )

        ga_loss = -args.forget_weight * fl + anchor
        ga_loss.backward()

        mask_grad(emb.weight, fmask_in)
        mask_grad(out.weight, fmask_out)
        finite_grad(emb.weight, "input_embedding_forget", step)
        finite_grad(out.weight, "output_lm_head_forget", step)

        emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
        out.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
        torch.nn.utils.clip_grad_norm_([emb.weight, out.weight], args.grad_clip_norm)

        opt_ga.step()
        clip_rows(emb.weight, ain, emb0, args.max_delta_input)
        clip_rows(out.weight, aout, out0, args.max_delta_output)

        finite_check(emb.weight.data[ain], "input_rows_after_GA", step)
        finite_check(out.weight.data[aout], "output_rows_after_GA", step)

        # GD: minimize retain answer loss
        opt_gd.zero_grad(set_to_none=True)
        br = sample(rng, renc, args.retain_batch_size, tok.pad_token_id, device)
        rl = loss_on_batch(model, br)
        finite_check(rl, "retain_loss", step)

        anchor = (
            args.anchor_input * mse_rows(emb.weight, ain, emb0).to(device)
            + args.anchor_output * mse_rows(out.weight, aout, out0).to(device)
        )

        gd_loss = args.retain_weight * rl + anchor
        gd_loss.backward()

        mask_grad(emb.weight, rmask_in)
        mask_grad(out.weight, rmask_out)
        finite_grad(emb.weight, "input_embedding_retain", step)
        finite_grad(out.weight, "output_lm_head_retain", step)

        emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
        out.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
        torch.nn.utils.clip_grad_norm_([emb.weight, out.weight], args.grad_clip_norm)

        opt_gd.step()
        clip_rows(emb.weight, ain, emb0, args.max_delta_input)
        clip_rows(out.weight, aout, out0, args.max_delta_output)

        finite_check(emb.weight.data[ain], "input_rows_after_GD", step)
        finite_check(out.weight.data[aout], "output_rows_after_GD", step)

        last_f = float(fl.detach().cpu())
        last_r = float(rl.detach().cpu())
        last_a = float(anchor.detach().cpu())

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix({
                "forget_loss": f"{last_f:.3f}",
                "retain_loss": f"{last_r:.3f}",
                "anchor": f"{last_a:.6f}",
            })

    finite_check(emb.weight.data, "final_input_embedding", args.steps)
    finite_check(out.weight.data, "final_output_lm_head", args.steps)

    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(od)
    tok.save_pretrained(od)

    with open(od / "pure_ga_gd_summary.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("[done] saved:", od)


if __name__ == "__main__":
    main()
