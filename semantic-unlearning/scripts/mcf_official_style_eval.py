#!/usr/bin/env python3

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
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


@torch.no_grad()
def official_test_batch_prediction(model, tok, prefixes, target_new, target_true, device):
    """
    Official-style MCF/CounterFact probability comparison.

    Returns average NLL for target_new and target_true for each prefix.
    Lower NLL means the model prefers that target.
    """
    if len(prefixes) == 0:
        return []

    prefix_lens = [len(x) for x in tok(prefixes)["input_ids"]]

    prompt_tok = tok(
        [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok = tok(f" {target_new}")["input_ids"]
    b_tok = tok(f" {target_true}")["input_ids"]

    model_name = str(getattr(model.config, "_name_or_path", "")).lower()
    is_llama = "llama" in model_name

    if is_llama:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [x - 1 for x in prefix_lens]

    logits = model(**prompt_tok).logits

    if is_llama:
        logits = logits[:, 1:, :]

    choice_a_len = len(a_tok)
    choice_b_len = len(b_tok)

    nlls = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_tokens = a_tok if i % 2 == 0 else b_tok
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        for j in range(cur_len):
            cur_tok = cur_tokens[j]
            pos = prefix_lens[i // 2] + j - 1
            nlls[i] += -torch.nn.functional.log_softmax(
                logits[i, pos, :],
                dim=0
            )[cur_tok].item()

        nlls[i] /= max(1, cur_len)

    out = []
    for i in range(0, len(nlls), 2):
        out.append({
            "target_new": float(nlls[i]),
            "target_true": float(nlls[i + 1]),
        })

    return out


def summarize_official(probs, mode):
    """
    mode:
      rewrite/paraphrase: success if target_new NLL < target_true NLL
      neighborhood: success if target_true NLL < target_new NLL
    """
    if len(probs) == 0:
        return {
            "n": 0,
            "success": None,
            "diff": None,
        }

    if mode in ["rewrite", "paraphrase"]:
        success = np.mean([x["target_true"] > x["target_new"] for x in probs])
        diff = np.mean([
            np.exp(-x["target_new"]) - np.exp(-x["target_true"])
            for x in probs
        ])
    elif mode == "neighborhood":
        success = np.mean([x["target_true"] < x["target_new"] for x in probs])
        diff = np.mean([
            np.exp(-x["target_true"]) - np.exp(-x["target_new"])
            for x in probs
        ])
    else:
        raise ValueError(mode)

    return {
        "n": len(probs),
        "success": float(success * 100.0),
        "diff": float(diff * 100.0),
        "mean_target_new_nll": float(np.mean([x["target_new"] for x in probs])),
        "mean_target_true_nll": float(np.mean([x["target_true"] for x in probs])),
    }


@torch.no_grad()
def eval_mcf_records(model, tok, records, device):
    all_rewrite_probs = []
    all_paraphrase_probs = []
    all_neighborhood_probs = []

    for record in tqdm(records, desc="Official-style MCF eval"):
        rr = record["requested_rewrite"]

        subject = rr["subject"]
        target_new = rr["target_new"]["str"]
        target_true = rr["target_true"]["str"]

        rewrite_prompts = [
            format_prompt(rr["prompt"], subject)
        ]

        paraphrase_prompts = record.get("paraphrase_prompts", [])

        # Official evaluator treats these as prompt strings.
        neighborhood_prompts = record.get("neighborhood_prompts", [])

        rewrite_probs = official_test_batch_prediction(
            model, tok, rewrite_prompts, target_new, target_true, device
        )
        paraphrase_probs = official_test_batch_prediction(
            model, tok, paraphrase_prompts, target_new, target_true, device
        )
        neighborhood_probs = official_test_batch_prediction(
            model, tok, neighborhood_prompts, target_new, target_true, device
        )

        all_rewrite_probs.extend(rewrite_probs)
        all_paraphrase_probs.extend(paraphrase_probs)
        all_neighborhood_probs.extend(neighborhood_probs)

    rewrite = summarize_official(all_rewrite_probs, "rewrite")
    paraphrase = summarize_official(all_paraphrase_probs, "paraphrase")
    neighborhood = summarize_official(all_neighborhood_probs, "neighborhood")

    # Paper-style names:
    # Eff = rewrite success
    # Gen = paraphrase success
    # Spe = neighborhood success
    result = {
        "Eff_rewrite_success_down": rewrite["success"],
        "Gen_paraphrase_success_down": paraphrase["success"],
        "Spe_neighborhood_success_up": neighborhood["success"],

        "rewrite_diff": rewrite["diff"],
        "paraphrase_diff": paraphrase["diff"],
        "neighborhood_diff": neighborhood["diff"],

        "rewrite": rewrite,
        "paraphrase": paraphrase,
        "neighborhood": neighborhood,
    }

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--forget-n", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mcf_path = download_mcf(args.mcf_path)
    data = load_mcf(mcf_path)
    records = data[:args.forget_n]

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_from_str(args.dtype),
        device_map="auto",
    )
    model.eval()
    model.config.use_cache = False

    device = next(model.parameters()).device

    result = eval_mcf_records(model, tok, records, device)
    result["model_dir"] = args.model_dir
    result["dataset"] = "MCF"
    result["forget_n"] = args.forget_n

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
