#!/usr/bin/env python3

import argparse
import json
import random
import urllib.request
from pathlib import Path

import torch
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


def encode_answer(tok, text):
    return tok.encode(" " + str(text).strip(), add_special_tokens=False)


def official_split(data, forget_n, retain_n, seed):
    rng = random.Random(seed)
    half = len(data) // 2

    retain_pool = data[:half]
    forget_pool = data[half:]

    forget_records = rng.sample(forget_pool, k=min(forget_n, len(forget_pool)))
    retain_records = rng.sample(retain_pool, k=min(retain_n, len(retain_pool)))

    return forget_records, retain_records


def collect_json_tokens(tok, forget_records, retain_records):
    target_new_ids = set()
    target_true_ids = set()
    retain_ids = set()
    report = []

    for i, r in enumerate(forget_records):
        rr = r["requested_rewrite"]
        subject = str(rr["subject"]).strip()
        target_new = str(rr["target_new"]["str"]).strip()
        target_true = str(rr["target_true"]["str"]).strip()

        new_ids = [int(x) for x in encode_answer(tok, target_new)]
        true_ids = [int(x) for x in encode_answer(tok, target_true)]

        target_new_ids.update(new_ids)
        target_true_ids.update(true_ids)

        report.append({
            "case_id": i,
            "subject": subject,
            "target_new": target_new,
            "target_true": target_true,
            "target_new_token_ids": new_ids,
            "target_new_tokens": [tok.decode([x]) for x in new_ids],
            "target_true_token_ids": true_ids,
            "target_true_tokens": [tok.decode([x]) for x in true_ids],
        })

    for r in retain_records:
        rr = r["requested_rewrite"]
        for field in ["target_new", "target_true"]:
            txt = str(rr[field]["str"]).strip()
            retain_ids.update(int(x) for x in encode_answer(tok, txt))

    specials = {
        x for x in [
            tok.pad_token_id,
            tok.eos_token_id,
            tok.bos_token_id,
            tok.unk_token_id,
        ] if x is not None
    }

    target_new_ids = {x for x in target_new_ids if x not in specials}
    target_true_ids = {x for x in target_true_ids if x not in specials}
    retain_ids = {x for x in retain_ids if x not in specials}

    return sorted(target_new_ids), sorted(target_true_ids), sorted(retain_ids), report


@torch.no_grad()
def apply_variant(
    model,
    target_new_ids,
    target_true_ids,
    retain_ids,
    variant,
    embed_scale,
    true_embed_scale,
    true_lm_scale,
    overlap_factor,
):
    emb = model.get_input_embeddings().weight
    lm = getattr(model, "lm_head", None)

    if lm is None or not hasattr(lm, "weight"):
        raise RuntimeError("Model has no lm_head.weight")

    lm_tied = lm.weight.data_ptr() == emb.data_ptr()
    changes = {}

    target_new_ids = [int(x) for x in target_new_ids]
    target_true_ids = [int(x) for x in target_true_ids]
    retain_ids = set(int(x) for x in retain_ids)

    # 1. Always zero lm_head rows for target_new.
    # If lm_head is tied to embeddings, this also zeroes embeddings.
    for tid in target_new_ids:
        if 0 <= tid < lm.weight.shape[0]:
            lm_before = float(torch.norm(lm.weight[tid].detach().float()).cpu())
            lm.weight[tid].zero_()
            lm_after = float(torch.norm(lm.weight[tid].detach().float()).cpu())

            changes[str(tid)] = {
                "token": None,
                "is_target_new": True,
                "is_target_true": tid in target_true_ids,
                "is_retain_overlap": tid in retain_ids,
                "lm_head_zeroed": True,
                "lm_head_before_norm": lm_before,
                "lm_head_after_norm": lm_after,
                "lm_head_tied_to_embedding": lm_tied,
            }

    # 2. Optional embedding scale for target_new.
    if variant in ["lm_head_zero_embed_scale", "lm_head_zero_true_restore"]:
        for tid in target_new_ids:
            if 0 <= tid < emb.shape[0]:
                alpha = embed_scale
                if tid in retain_ids:
                    alpha = 1.0 - overlap_factor * (1.0 - embed_scale)

                emb_before = float(torch.norm(emb[tid].detach().float()).cpu())
                emb[tid].mul_(alpha)
                emb_after = float(torch.norm(emb[tid].detach().float()).cpu())

                changes.setdefault(str(tid), {})
                changes[str(tid)].update({
                    "embed_scaled": True,
                    "embed_alpha": float(alpha),
                    "embed_before_norm": emb_before,
                    "embed_after_norm": emb_after,
                })

    # 3. Optional target_true restore.
    if variant == "lm_head_zero_true_restore":
        for tid in target_true_ids:
            if 0 <= tid < emb.shape[0]:
                beta = true_embed_scale
                if tid in retain_ids:
                    beta = 1.0 + overlap_factor * (true_embed_scale - 1.0)

                emb_before = float(torch.norm(emb[tid].detach().float()).cpu())
                emb[tid].mul_(beta)
                emb_after = float(torch.norm(emb[tid].detach().float()).cpu())

                changes.setdefault(str(tid), {})
                changes[str(tid)].update({
                    "target_true_embed_restored": True,
                    "true_embed_scale": float(beta),
                    "true_embed_before_norm": emb_before,
                    "true_embed_after_norm": emb_after,
                })

            if not lm_tied and 0 <= tid < lm.weight.shape[0]:
                gamma = true_lm_scale
                if tid in retain_ids:
                    gamma = 1.0 + overlap_factor * (true_lm_scale - 1.0)

                lm_before = float(torch.norm(lm.weight[tid].detach().float()).cpu())
                lm.weight[tid].mul_(gamma)
                lm_after = float(torch.norm(lm.weight[tid].detach().float()).cpu())

                changes.setdefault(str(tid), {})
                changes[str(tid)].update({
                    "target_true_lm_restored": True,
                    "true_lm_scale": float(gamma),
                    "true_lm_before_norm": lm_before,
                    "true_lm_after_norm": lm_after,
                })

    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--forget-n", type=int, default=50)
    ap.add_argument("--retain-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")

    ap.add_argument(
        "--variant",
        choices=[
            "lm_head_zero_only",
            "lm_head_zero_embed_scale",
            "lm_head_zero_true_restore",
        ],
        required=True,
    )

    ap.add_argument("--embed-scale", type=float, default=0.50)
    ap.add_argument("--true-embed-scale", type=float, default=1.05)
    ap.add_argument("--true-lm-scale", type=float, default=1.05)
    ap.add_argument("--overlap-factor", type=float, default=0.25)

    args = ap.parse_args()

    mcf_path = download_mcf(args.mcf_path)
    data = load_mcf(mcf_path)

    forget_records, retain_records = official_split(
        data,
        forget_n=args.forget_n,
        retain_n=args.retain_n,
        seed=args.seed,
    )

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

    target_new_ids, target_true_ids, retain_ids, report = collect_json_tokens(
        tok,
        forget_records,
        retain_records,
    )

    print("=" * 100)
    print("MCF JSON LM-head ablation")
    print("=" * 100)
    print(f"variant: {args.variant}")
    print(f"seed: {args.seed}")
    print(f"forget records: {len(forget_records)}")
    print(f"retain records: {len(retain_records)}")
    print(f"target_new rows: {len(target_new_ids)}")
    print(f"target_true rows: {len(target_true_ids)}")
    print(f"retain rows: {len(retain_ids)}")
    print("=" * 100)

    print("Target_new rows to zero in lm_head:")
    for tid in target_new_ids:
        print(f"  id={tid}, token={repr(tok.decode([tid]))}, retain_overlap={tid in retain_ids}")

    changes = apply_variant(
        model=model,
        target_new_ids=target_new_ids,
        target_true_ids=target_true_ids,
        retain_ids=retain_ids,
        variant=args.variant,
        embed_scale=args.embed_scale,
        true_embed_scale=args.true_embed_scale,
        true_lm_scale=args.true_lm_scale,
        overlap_factor=args.overlap_factor,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out)
    tok.save_pretrained(out)

    summary = {
        "method": "MCF-JSON-LMHead-Ablation",
        "variant": args.variant,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "forget_n": args.forget_n,
        "retain_n": args.retain_n,
        "seed": args.seed,
        "split_mode": "official_zero_unlearn",
        "embed_scale": args.embed_scale,
        "true_embed_scale": args.true_embed_scale,
        "true_lm_scale": args.true_lm_scale,
        "overlap_factor": args.overlap_factor,
        "n_target_new_rows": len(target_new_ids),
        "n_target_true_rows": len(target_true_ids),
        "n_retain_rows": len(retain_ids),
        "target_new_token_ids": target_new_ids,
        "target_true_token_ids": target_true_ids,
        "target_new_tokens": {str(tid): tok.decode([tid]) for tid in target_new_ids},
        "target_true_tokens": {str(tid): tok.decode([tid]) for tid in target_true_ids},
        "record_report": report,
        "changes": changes,
    }

    with open(out / "mcf_json_lmhead_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
