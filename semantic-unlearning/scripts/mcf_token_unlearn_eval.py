#!/usr/bin/env python3

import argparse
import json
import math
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"


def dtype_from_str(x):
    x = str(x).lower()
    if x in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if x in ["fp16", "float16"]:
        return torch.float16
    return torch.float32


def download_mcf(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading MCF to {path}")
        urllib.request.urlretrieve(MCF_URL, path)
    return path


def load_mcf(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(template, subject):
    if "{}" in template:
        return template.format(subject)
    return template


def get_prompt_answer(record, target_field="target_new"):
    rr = record["requested_rewrite"]
    template = rr["prompt"]
    subject = rr["subject"]
    answer = rr[target_field]["str"]
    prompt = format_prompt(template, subject)
    return prompt.strip(), str(answer).strip(), str(subject).strip()


def encode_answer_tokens(tok, answer):
    return tok.encode(" " + str(answer).strip(), add_special_tokens=False)


def select_tokens(
    tok,
    forget_records,
    retain_records,
    select_k,
    max_retain_count,
    soft_overlap,
    target_field="target_new",
):
    """
    MCF-specific token selection.

    IMPORTANT:
    We only use target answer tokens, not subject tokens.
    For official MCF unlearning, target_field should be target_new.
    """
    fcnt = Counter()
    rcnt = Counter()

    for r in forget_records:
        _, ans, _ = get_prompt_answer(r, target_field=target_field)
        for tid in encode_answer_tokens(tok, ans):
            fcnt[int(tid)] += 1

    for r in retain_records:
        _, ans, _ = get_prompt_answer(r, target_field=target_field)
        for tid in encode_answer_tokens(tok, ans):
            rcnt[int(tid)] += 1

    specials = {
        x for x in [
            tok.pad_token_id,
            tok.eos_token_id,
            tok.bos_token_id,
            tok.unk_token_id,
        ] if x is not None
    }

    scored = []
    for tid, fc in fcnt.items():
        if tid in specials:
            continue

        rc = rcnt.get(tid, 0)

        if not soft_overlap and rc > max_retain_count:
            continue

        token_str = tok.decode([tid]).strip()
        if len(token_str) < 1:
            continue

        score = fc / (1.0 + rc)
        scored.append((score, fc, -rc, tid, token_str))

    scored.sort(reverse=True)

    selected = [tid for _, _, _, tid, _ in scored[:select_k]]
    forget_only = [tid for tid in selected if rcnt.get(tid, 0) <= max_retain_count]
    overlap = [tid for tid in selected if rcnt.get(tid, 0) > max_retain_count]

    token_report = []
    for score, fc, neg_rc, tid, token_str in scored[:select_k]:
        token_report.append({
            "token_id": int(tid),
            "token": token_str,
            "score": float(score),
            "forget_count": int(fc),
            "retain_count": int(-neg_rc),
            "action": "soft" if tid in overlap else "zero_or_suppress",
        })

    return selected, forget_only, overlap, fcnt, rcnt, token_report


@torch.no_grad()
def answer_prob(model, tok, prompt, answer, device, max_length):
    p_ids = tok.encode(prompt, add_special_tokens=True)
    a_ids = tok.encode(" " + str(answer).strip(), add_special_tokens=False)

    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids

    ids = ids[:max_length]
    labels = labels[:max_length]

    if all(x == -100 for x in labels):
        return None

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.tensor([labels], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)

    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    return float(torch.exp(-out.loss).detach().cpu())


@torch.no_grad()
def evaluate_records(model, tok, records, device, max_length, desc, target_field="target_new"):
    vals = []
    for r in tqdm(records, desc=desc):
        p, a, _ = get_prompt_answer(r, target_field=target_field)
        v = answer_prob(model, tok, p, a, device, max_length)
        if v is not None:
            vals.append(v)
    return sum(vals) / max(1, len(vals))


@torch.no_grad()
def evaluate_paraphrase(model, tok, records, device, max_length, target_field="target_new"):
    vals = []
    for r in tqdm(records, desc="Gen/paraphrase"):
        ans = r["requested_rewrite"][target_field]["str"]
        for p in r.get("paraphrase_prompts", []):
            v = answer_prob(model, tok, p.strip(), str(ans).strip(), device, max_length)
            if v is not None:
                vals.append(v)
    return sum(vals) / max(1, len(vals))


@torch.no_grad()
def evaluate_specificity_quick(model, tok, records, device, max_length):
    """
    Quick proxy only. For official Spe, use scripts/mcf_official_style_eval.py.
    MCF neighborhood_prompts are usually strings, not dicts.
    """
    vals = []
    for r in tqdm(records, desc="Specificity/neighborhood quick"):
        target_true = r["requested_rewrite"]["target_true"]["str"]
        for p in r.get("neighborhood_prompts", []):
            if isinstance(p, str):
                v = answer_prob(model, tok, p.strip(), str(target_true).strip(), device, max_length)
                if v is not None:
                    vals.append(v)
    return sum(vals) / max(1, len(vals))


@torch.no_grad()
def apply_zero_or_scale(model, token_ids_zero, token_ids_soft, soft_alpha):
    emb = model.get_input_embeddings().weight

    for tid in token_ids_zero:
        if 0 <= tid < emb.shape[0]:
            emb[tid].zero_()

    for tid in token_ids_soft:
        if 0 <= tid < emb.shape[0]:
            emb[tid].mul_(soft_alpha)

    lm = getattr(model, "lm_head", None)
    if lm is not None and hasattr(lm, "weight"):
        if lm.weight.data_ptr() != emb.data_ptr():
            for tid in token_ids_zero:
                if 0 <= tid < lm.weight.shape[0]:
                    lm.weight[tid].zero_()
            for tid in token_ids_soft:
                if 0 <= tid < lm.weight.shape[0]:
                    lm.weight[tid].mul_(soft_alpha)


@torch.no_grad()
def collect_lm_head_suppression_vectors(
    model,
    tok,
    forget_records,
    selected_token_ids,
    device,
    max_length,
    target_field="target_new",
):
    """
    For each selected target_new token, collect the hidden state that predicts it.
    Then we can move lm_head[token] against that hidden direction.
    This directly suppresses target_new logits on forget prompts.
    """
    selected_token_ids = set(int(x) for x in selected_token_ids)
    sums = {}
    counts = Counter()

    for r in tqdm(forget_records, desc="Collect suppression hidden states"):
        prompt, answer, _ = get_prompt_answer(r, target_field=target_field)

        p_ids = tok.encode(prompt, add_special_tokens=True)
        a_ids = tok.encode(" " + str(answer).strip(), add_special_tokens=False)

        ids = p_ids + a_ids
        ids = ids[:max_length]

        if len(ids) < 2:
            continue

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)

        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden state at position t-1 predicts token at position t
        h = out.hidden_states[-1][0]

        for j, tid in enumerate(ids):
            if j == 0:
                continue
            if j < len(p_ids):
                continue
            tid = int(tid)
            if tid not in selected_token_ids:
                continue

            pred_h = h[j - 1].detach().float()

            if tid not in sums:
                sums[tid] = pred_h.clone()
            else:
                sums[tid] += pred_h

            counts[tid] += 1

    vectors = {}
    for tid, vec in sums.items():
        vectors[tid] = vec / max(1, counts[tid])

    return vectors, counts


@torch.no_grad()
def apply_lm_head_suppression(
    model,
    suppression_vectors,
    token_ids_zero,
    token_ids_soft,
    suppress_alpha,
    soft_alpha,
    also_scale_embeddings,
):
    """
    Update LM-head rows:
        W[token] <- W[token] - alpha * mean_hidden_for_token

    This reduces logits for selected target_new tokens in forget contexts.
    For overlap tokens, use smaller alpha through soft_alpha.
    """
    lm = getattr(model, "lm_head", None)
    if lm is None or not hasattr(lm, "weight"):
        raise RuntimeError("Model has no lm_head.weight for suppress_lm_head mode.")

    lm_w = lm.weight
    device = lm_w.device
    dtype = lm_w.dtype

    zero_set = set(int(x) for x in token_ids_zero)
    soft_set = set(int(x) for x in token_ids_soft)

    for tid, vec in suppression_vectors.items():
        if tid < 0 or tid >= lm_w.shape[0]:
            continue

        strength = suppress_alpha
        if tid in soft_set:
            strength = suppress_alpha * soft_alpha
        elif tid not in zero_set:
            continue

        v = vec.to(device=device, dtype=dtype)

        # Normalize to avoid huge destructive updates.
        v_norm = torch.norm(v).clamp_min(1e-6)
        w_norm = torch.norm(lm_w[tid]).clamp_min(1e-6)
        v = v / v_norm * w_norm

        lm_w[tid] -= strength * v

    if also_scale_embeddings:
        emb = model.get_input_embeddings().weight
        for tid in zero_set:
            if 0 <= tid < emb.shape[0]:
                emb[tid].mul_(0.25)
        for tid in soft_set:
            if 0 <= tid < emb.shape[0]:
                emb[tid].mul_(soft_alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")

    ap.add_argument("--forget-n", type=int, default=50)
    ap.add_argument("--retain-n", type=int, default=1000)
    ap.add_argument("--select-k", type=int, default=800)
    ap.add_argument("--max-retain-count", type=int, default=0)
    ap.add_argument("--soft-overlap", action="store_true")
    ap.add_argument("--soft-alpha", type=float, default=0.50)

    ap.add_argument("--target-field", choices=["target_new", "target_true"], default="target_new")
    ap.add_argument("--erase-mode", choices=["zero", "scale", "suppress_lm_head"], default="suppress_lm_head")
    ap.add_argument("--suppress-alpha", type=float, default=0.35)
    ap.add_argument("--also-scale-embeddings", action="store_true")

    ap.add_argument("--max-length", type=int, default=96)
    args = ap.parse_args()

    mcf_path = download_mcf(args.mcf_path)
    data = load_mcf(mcf_path)

    forget_records = data[:args.forget_n]
    retain_records = data[args.forget_n:args.forget_n + args.retain_n]

    print(f"Forget records: {len(forget_records)}")
    print(f"Retain records: {len(retain_records)}")
    print(f"Target field for forgetting: {args.target_field}")
    print(f"Erase mode: {args.erase_mode}")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_from_str(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    model.config.use_cache = False

    device = next(model.parameters()).device

    print("Evaluating base model with quick probability proxy...")
    base_forget = evaluate_records(
        model, tok, forget_records, device, args.max_length,
        "Base forget/Eff quick", target_field=args.target_field
    )
    base_retain = evaluate_records(
        model, tok, retain_records[:200], device, args.max_length,
        "Base retain quick", target_field=args.target_field
    )
    base_gen = evaluate_paraphrase(
        model, tok, forget_records, device, args.max_length,
        target_field=args.target_field
    )
    base_spe = evaluate_specificity_quick(model, tok, forget_records, device, args.max_length)

    selected, forget_only, overlap, fcnt, rcnt, token_report = select_tokens(
        tok,
        forget_records,
        retain_records,
        args.select_k,
        args.max_retain_count,
        args.soft_overlap,
        target_field=args.target_field,
    )

    if args.soft_overlap:
        zero_ids = forget_only
        soft_ids = overlap
    else:
        zero_ids = selected
        soft_ids = []

    print(f"Selected tokens: {len(selected)}")
    print(f"Zero/suppress tokens: {len(zero_ids)}")
    print(f"Soft-overlap tokens: {len(soft_ids)}")
    print("Top selected tokens:")
    for item in token_report[:25]:
        print(
            f"  id={item['token_id']:>8} token={repr(item['token'])} "
            f"score={item['score']:.4f} f={item['forget_count']} r={item['retain_count']} "
            f"action={item['action']}"
        )

    print("Applying unlearning edit...")
    if args.erase_mode == "zero":
        apply_zero_or_scale(model, zero_ids, soft_ids, args.soft_alpha)

    elif args.erase_mode == "scale":
        # Less aggressive than zero: shrink all selected rows.
        apply_zero_or_scale(model, [], selected, args.soft_alpha)

    elif args.erase_mode == "suppress_lm_head":
        suppression_vectors, suppress_counts = collect_lm_head_suppression_vectors(
            model=model,
            tok=tok,
            forget_records=forget_records,
            selected_token_ids=selected,
            device=device,
            max_length=args.max_length,
            target_field=args.target_field,
        )
        print(f"Suppression vectors collected for {len(suppression_vectors)} tokens.")
        apply_lm_head_suppression(
            model=model,
            suppression_vectors=suppression_vectors,
            token_ids_zero=zero_ids,
            token_ids_soft=soft_ids,
            suppress_alpha=args.suppress_alpha,
            soft_alpha=args.soft_alpha,
            also_scale_embeddings=args.also_scale_embeddings,
        )

        for item in token_report:
            tid = item["token_id"]
            item["suppression_count"] = int(suppress_counts.get(tid, 0))

    print("Evaluating unlearned model with quick probability proxy...")
    unlearn_forget = evaluate_records(
        model, tok, forget_records, device, args.max_length,
        "Unlearn forget/Eff quick", target_field=args.target_field
    )
    unlearn_retain = evaluate_records(
        model, tok, retain_records[:200], device, args.max_length,
        "Unlearn retain quick", target_field=args.target_field
    )
    unlearn_gen = evaluate_paraphrase(
        model, tok, forget_records, device, args.max_length,
        target_field=args.target_field
    )
    unlearn_spe = evaluate_specificity_quick(model, tok, forget_records, device, args.max_length)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out)
    tok.save_pretrained(out)

    metrics = {
        "dataset": "MCF",
        "method": "MCF-RA-Freq-TRE-answer-only",
        "forget_n": args.forget_n,
        "retain_n": args.retain_n,
        "select_k": args.select_k,
        "max_retain_count": args.max_retain_count,
        "soft_overlap": args.soft_overlap,
        "soft_alpha": args.soft_alpha,
        "target_field": args.target_field,
        "erase_mode": args.erase_mode,
        "suppress_alpha": args.suppress_alpha,
        "also_scale_embeddings": args.also_scale_embeddings,
        "n_selected": len(selected),
        "n_zero_or_suppress": len(zero_ids),
        "n_soft": len(soft_ids),

        "base_forget_eff_raw_quick": base_forget,
        "base_forget_eff_percent_quick": base_forget * 100,
        "base_retain_raw_quick": base_retain,
        "base_retain_percent_quick": base_retain * 100,
        "base_gen_raw_quick": base_gen,
        "base_gen_percent_quick": base_gen * 100,
        "base_spe_raw_quick": base_spe,
        "base_spe_percent_quick": base_spe * 100,

        "unlearn_forget_eff_raw_quick": unlearn_forget,
        "unlearn_forget_eff_percent_quick": unlearn_forget * 100,
        "unlearn_retain_raw_quick": unlearn_retain,
        "unlearn_retain_percent_quick": unlearn_retain * 100,
        "unlearn_gen_raw_quick": unlearn_gen,
        "unlearn_gen_percent_quick": unlearn_gen * 100,
        "unlearn_spe_raw_quick": unlearn_spe,
        "unlearn_spe_percent_quick": unlearn_spe * 100,
    }

    with open(out / "mcf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(out / "selected_token_ids.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected": selected,
            "zero_or_suppress": zero_ids,
            "soft": soft_ids,
            "token_report": token_report,
        }, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics to {out}")
    print("Now run scripts/mcf_official_style_eval.py on this output directory for Eff/Gen/Spe.")


if __name__ == "__main__":
    main()
