#!/usr/bin/env python3
"""Compare a frozen SURE-LM MCF checkpoint against ZeroUnlearn on identical metrics.

This script is intentionally comparison-only. It does not retrain SURE-LM and it
never exposes held-out probes to Stage 1 or Stage 2. It loads the already-frozen
SURE-LM checkpoint, reconstructs the exact ZeroUnlearn-style MCF split, and
computes the same greedy target_true accuracies used by ZeroUnlearn:

  Eff = post_rewrite_acc          (lower is better)
  Gen = post_paraphrase_acc       (lower is better)
  Spe = post_neighborhood_acc     (higher is better)

It also computes PPL on the same locked SURE-LM fixture:
semantic-unlearning/data/wikidata, first 20 train texts joined with spaces,
truncated to 100 Llama tokens.

For seed 1, the script can auto-discover the completed ZeroUnlearn benchmark
summary and matched-PPL run from the sibling ZeroUnlearn checkout and print an
apples-to-apples table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    is_llama_like,
    load_official_eval_records,
    load_official_ppl_text,
    official_perplexity,
)

EXPECTED_PPL_TEXT_SHA256 = (
    "8336e4e5be0f96a70c5720714144794a53436b3b8e2e22ad848cc31501304f7a"
)
EXPECTED_PPL_TOKEN_SHA256 = (
    "2d306a72f23b17b5b18688e0b4e25513423db296b9cf653d4b3cb32163b7a5c7"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--sure-output-root",
        default="outputs/mcf_zerounlearn_forget_only_locked_3b",
        help="Root containing seedN/repair_forget_only/checkpoint.",
    )
    p.add_argument(
        "--sure-checkpoint",
        default=None,
        help="Optional explicit frozen SURE-LM checkpoint. Overrides --sure-output-root.",
    )
    p.add_argument(
        "--tokenizer-path",
        default="/scratch/yl258/kp759/models/Llama-3.2-3B-Instruct",
    )
    p.add_argument(
        "--zero-results-root",
        default=(
            "/scratch/yl258/kp759/Unlearning/ZeroUnlearn/results/ZeroUnlearn"
        ),
        help="Directory containing official ZeroUnlearn run directories.",
    )
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument(
        "--eval-retain",
        action="store_true",
        help="Also compute ZeroUnlearn-style accuracy on the 1000 retain records.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output JSON. Default: outputs/comparison_zero_mcf/seedN_comparison.json",
    )
    return p.parse_args()


def _resolve_sure_checkpoint(args: argparse.Namespace) -> Path:
    if args.sure_checkpoint:
        path = Path(args.sure_checkpoint)
    else:
        path = (
            Path(args.sure_output_root)
            / f"seed{args.seed}"
            / "repair_forget_only"
            / "checkpoint"
        )
    if not path.is_dir():
        raise FileNotFoundError(
            f"SURE-LM checkpoint not found: {path}\n"
            "Pass --sure-checkpoint explicitly if the accepted seed checkpoint lives "
            "under another output root."
        )
    return path


@torch.no_grad()
def zero_style_target_true_accuracy(
    model,
    tok,
    prefixes: List[str],
    target_new: str,
    target_true: str,
    device,
    llama_like: bool,
) -> List[bool]:
    """Exact greedy target_true correctness logic used by ZeroUnlearn MCF eval.

    The official evaluator batches both target_new and target_true completions and
    records correctness for target_true. We preserve that layout here so token
    positions and Llama BOS handling match the official code.
    """
    if not prefixes:
        return []

    prefix_lens = [len(x) for x in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (
        tok(f" {value}")["input_ids"] for value in [target_new, target_true]
    )

    if llama_like:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [x - 1 for x in prefix_lens]

    logits = model(**prompt_tok).logits
    if llama_like:
        logits = logits[:, 1:, :]

    corrects: List[bool] = []
    for prefix_idx in range(len(prefixes)):
        # Sequence layout is [new_0, true_0, new_1, true_1, ...].
        i = 2 * prefix_idx + 1
        correct = True
        for j, cur_tok in enumerate(b_tok):
            pos = prefix_lens[prefix_idx] + j - 1
            if logits[i, pos, :].argmax().item() != cur_tok:
                correct = False
                break
        corrects.append(correct)

    return corrects


@torch.no_grad()
def evaluate_record_zero_style(model, tok, record, device, llama_like: bool) -> Dict:
    rr = record["requested_rewrite"]
    subject = rr["subject"]
    target_new = rr["target_new"]["str"]
    target_true = rr["target_true"]["str"]

    groups = {
        "rewrite": [rr["prompt"].format(subject)],
        "paraphrase": record.get("paraphrase_prompts", []),
        "neighborhood": record.get("neighborhood_prompts", []),
    }

    return {
        f"{name}_correct": zero_style_target_true_accuracy(
            model=model,
            tok=tok,
            prefixes=prefixes,
            target_new=target_new,
            target_true=target_true,
            device=device,
            llama_like=llama_like,
        )
        for name, prefixes in groups.items()
    }


def summarize_zero_style(records: List[Dict], split_name: str) -> Dict:
    """Match ZeroUnlearn summarize_list.py macro averaging and population SD."""
    out: Dict[str, object] = {
        "split_name": split_name,
        "num_cases": len(records),
    }

    mapping = {
        "rewrite": "Eff",
        "paraphrase": "Gen",
        "neighborhood": "Spe",
    }

    for name, paper_name in mapping.items():
        per_record = []
        for item in records:
            values = item[f"{name}_correct"]
            if values:
                per_record.append(float(np.mean(values)))

        if not per_record:
            mean = None
            sd = None
        else:
            mean = float(np.mean(per_record) * 100.0)
            sd = float(np.std(per_record) * 100.0)

        out[f"post_{name}_acc"] = [mean, sd]
        out[paper_name] = mean

    return out


def evaluate_split(model, tok, records, device, llama_like: bool, split_name: str):
    rows = []
    for idx, record in enumerate(records, 1):
        rows.append(evaluate_record_zero_style(model, tok, record, device, llama_like))
        if idx == 1 or idx % 10 == 0 or idx == len(records):
            print(f"{split_name}: {idx}/{len(records)}")
    return summarize_zero_style(rows, split_name)


def matched_ppl(model, tok, wikidata_dir: Path, device) -> Tuple[float, Dict]:
    text = load_official_ppl_text(wikidata_dir)
    if text is None:
        raise RuntimeError(f"Could not load PPL fixture from {wikidata_dir}")

    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_sha != EXPECTED_PPL_TEXT_SHA256:
        raise RuntimeError(
            "PPL text fixture mismatch: "
            f"got {text_sha}, expected {EXPECTED_PPL_TEXT_SHA256}"
        )

    enc = tok(
        [text],
        return_tensors="pt",
        max_length=100,
        truncation=True,
    )
    ids = enc["input_ids"][0].tolist()
    token_sha = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if token_sha != EXPECTED_PPL_TOKEN_SHA256:
        raise RuntimeError(
            "PPL token fixture mismatch: "
            f"got {token_sha}, expected {EXPECTED_PPL_TOKEN_SHA256}"
        )

    ppl = official_perplexity(
        model,
        tok,
        text,
        device,
        max_input_length=100,
    )
    provenance = {
        "dataset": str(wikidata_dir.resolve()),
        "source_rows_used": 20,
        "max_input_length": 100,
        "text_sha256": text_sha,
        "token_count": len(ids),
        "token_sha256": token_sha,
    }
    return float(ppl), provenance


def _run_number(path: Path) -> int:
    try:
        return int(path.name.rsplit("_run_", 1)[1])
    except Exception:
        return -1


def discover_zero_seed_result(
    root: Path,
    seed: int,
    unlearn_num: int,
    retain_num: int,
) -> Tuple[Optional[Dict], Dict]:
    pattern = (
        f"Llama-3.2-3B-Instruct_mcf_seed{seed}_unlearn_{unlearn_num}_"
        f"retain_{retain_num}_edit_layer_nums_3_run_*"
    )
    runs = sorted(root.glob(pattern), key=_run_number)

    metric_runs = [d for d in runs if (d / "forget_summarize_results.json").is_file()]
    ppl_runs = [d for d in runs if (d / "matched_ppl.json").is_file()]

    if not metric_runs or not ppl_runs:
        return None, {
            "metrics_run": str(metric_runs[-1]) if metric_runs else None,
            "ppl_run": str(ppl_runs[-1]) if ppl_runs else None,
        }

    metric_dir = metric_runs[-1]
    ppl_dir = ppl_runs[-1]

    forget = json.loads(
        (metric_dir / "forget_summarize_results.json").read_text(encoding="utf-8")
    )
    ppl = json.loads((ppl_dir / "matched_ppl.json").read_text(encoding="utf-8"))

    result = {
        "Eff": float(forget["post_rewrite_acc"][0]),
        "Gen": float(forget["post_paraphrase_acc"][0]),
        "Spe": float(forget["post_neighborhood_acc"][0]),
        "PPL": float(ppl["ppl"]),
    }
    sources = {
        "metrics_run": str(metric_dir.resolve()),
        "ppl_run": str(ppl_dir.resolve()),
    }
    return result, sources


def print_comparison(sure: Dict, zero: Optional[Dict]) -> None:
    print("\n============================================================")
    print("MCF SAME-METRIC COMPARISON")
    print("Eff/Gen: lower is better | Spe: higher is better | PPL: lower is better")
    print("============================================================")
    print(f"{'Method':<16}{'Eff↓':>10}{'Gen↓':>10}{'Spe↑':>10}{'PPL↓':>12}")
    print("-" * 58)
    print(
        f"{'SURE-LM':<16}{sure['Eff']:>10.2f}{sure['Gen']:>10.2f}"
        f"{sure['Spe']:>10.2f}{sure['PPL']:>12.4f}"
    )
    if zero is not None:
        print(
            f"{'ZeroUnlearn':<16}{zero['Eff']:>10.2f}{zero['Gen']:>10.2f}"
            f"{zero['Spe']:>10.2f}{zero['PPL']:>12.4f}"
        )
        print("-" * 58)
        print(
            f"{'SURE-Zero':<16}{sure['Eff']-zero['Eff']:>+10.2f}"
            f"{sure['Gen']-zero['Gen']:>+10.2f}"
            f"{sure['Spe']-zero['Spe']:>+10.2f}"
            f"{sure['PPL']-zero['PPL']:>+12.4f}"
        )
    else:
        print("ZeroUnlearn result was not auto-discovered; SURE-LM metrics are still valid.")
    print("============================================================")


def main() -> None:
    args = parse_args()
    checkpoint = _resolve_sure_checkpoint(args)
    mcf_path = Path(args.mcf_path)
    wikidata_dir = Path(args.wikidata_dir)
    tokenizer_path = Path(args.tokenizer_path)

    if not mcf_path.is_file():
        raise FileNotFoundError(mcf_path)
    if not wikidata_dir.is_dir():
        raise FileNotFoundError(wikidata_dir)
    if not tokenizer_path.exists():
        raise FileNotFoundError(tokenizer_path)

    print("SURE-LM checkpoint:", checkpoint.resolve())
    print("MCF dataset:", mcf_path.resolve())
    print("PPL fixture:", wikidata_dir.resolve())
    print("seed:", args.seed)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"torch_dtype": dtype_from_str(args.dtype)}
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **load_kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")

    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    llama_like = is_llama_like(model, tok)

    forget_records, retain_records = load_official_eval_records(
        mcf_path=mcf_path,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode="official",
    )

    print(f"Forget records: {len(forget_records)}")
    print(f"Retain records: {len(retain_records)}")

    forget = evaluate_split(
        model,
        tok,
        forget_records,
        device,
        llama_like,
        split_name="forget",
    )
    ppl, ppl_provenance = matched_ppl(model, tok, wikidata_dir, device)

    sure = {
        "Eff": float(forget["Eff"]),
        "Gen": float(forget["Gen"]),
        "Spe": float(forget["Spe"]),
        "PPL": float(ppl),
    }

    retain = None
    if args.eval_retain:
        retain = evaluate_split(
            model,
            tok,
            retain_records,
            device,
            llama_like,
            split_name="retain",
        )

    zero, zero_sources = discover_zero_seed_result(
        root=Path(args.zero_results_root),
        seed=args.seed,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
    )

    payload = {
        "schema_version": 1,
        "kind": "sure_lm_vs_zerounlearn_mcf_same_metric_comparison",
        "seed": args.seed,
        "metric_definition": {
            "Eff": "100 * macro greedy-token accuracy of target_true on requested_rewrite; lower is better",
            "Gen": "100 * macro greedy-token accuracy of target_true on paraphrase prompts; lower is better",
            "Spe": "100 * macro greedy-token accuracy of target_true on neighborhood prompts; higher is better",
            "PPL": "ZeroUnlearn-compatible perplexity on the locked SURE-LM Wikidata fixture; lower/stable is better",
        },
        "protocol": {
            "sample_mode": "official",
            "unlearn_num": args.unlearn_num,
            "retain_num": args.retain_num,
            "same_seeded_half_split_as_zerounlearn": True,
            "comparison_is_evaluation_only": True,
            "sure_checkpoint_frozen_before_heldout_evaluation": True,
        },
        "sure_lm": {
            "checkpoint": str(checkpoint.resolve()),
            "forget": forget,
            "retain": retain,
            "PPL": ppl,
            "ppl_provenance": ppl_provenance,
            "paper_row": sure,
        },
        "zerounlearn": {
            "paper_row": zero,
            "sources": zero_sources,
        },
    }

    if zero is not None:
        payload["delta_sure_minus_zero"] = {
            key: float(sure[key] - zero[key]) for key in ["Eff", "Gen", "Spe", "PPL"]
        }

    out = (
        Path(args.out)
        if args.out
        else Path("outputs/comparison_zero_mcf")
        / f"seed{args.seed}_comparison.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_comparison(sure, zero)
    print("\nPPL text SHA256:", ppl_provenance["text_sha256"])
    print("PPL token SHA256:", ppl_provenance["token_sha256"])
    print("Wrote:", out.resolve())


if __name__ == "__main__":
    main()
