#!/usr/bin/env python3

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM

from mcf_sampling import sample_first_mcf_records, sample_official_mcf_records


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
        value = json.load(f)
    if isinstance(value, dict) and value.get("format") == "mcf_shaped_rwku_training_request_v1":
        raise ValueError(
            "The MCF evaluator rejects MCF-shaped RWKU training requests; "
            "they are training-only RWKU adapter records, not MCF benchmark data."
        )
    if isinstance(value, list) and any(
        isinstance(row, dict)
        and row.get("format") == "mcf_shaped_rwku_training_request_v1"
        for row in value
    ):
        raise ValueError(
            "The MCF evaluator rejects MCF-shaped RWKU training requests"
        )
    return value


def normalize_record(record):
    """
    MCF usually has requested_rewrite as dict.
    Keep this wrapper for safety.
    """
    rr = record["requested_rewrite"]
    if isinstance(rr, list):
        rr = rr[0]
    record = dict(record)
    record["requested_rewrite"] = rr
    return record


def sample_official_split(data, unlearn_num, retain_num, seed):
    """
    Match ZeroUnlearn evaluate.py:
      retain pool = first half
      forget pool = second half
      random.sample with seed
    """
    forget_records, retain_records = sample_official_mcf_records(
        data, unlearn_num, retain_num, seed
    )

    return [normalize_record(x) for x in forget_records], [normalize_record(x) for x in retain_records]


def sample_first_split(data, unlearn_num, retain_num):
    """
    Old/debug mode only. Do not use for final comparison.
    """
    forget_records, retain_records = sample_first_mcf_records(
        data, unlearn_num, retain_num
    )
    return [normalize_record(x) for x in forget_records], [normalize_record(x) for x in retain_records]


def is_llama_like(model, tok):
    model_type = str(getattr(model.config, "model_type", "")).lower()
    name = str(getattr(model.config, "_name_or_path", "")).lower()
    tok_cls = tok.__class__.__name__.lower()
    return ("llama" in model_type) or ("llama" in name) or ("llama" in tok_cls)


@torch.no_grad()
def official_test_batch_prediction(model, tok, prefixes, target_new, target_true, device, llama_like=True):
    """
    Same logic as ZeroUnlearn experiments/py/eval_utils_counterfact.py:
      returns average NLL for target_new and target_true.
    """
    if len(prefixes) == 0:
        return []

    prefix_lens = [len(x) for x in tok(prefixes)["input_ids"]]

    prompt_tok = tok(
        [f"{prefix} {suffix}" for prefix in prefixes for suffix in [target_new, target_true]],
        padding=True,
        return_tensors="pt",
    ).to(device)

    a_tok, b_tok = (tok(f" {x}")["input_ids"] for x in [target_new, target_true])

    if llama_like:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [x - 1 for x in prefix_lens]

    logits = model(**prompt_tok).logits

    if llama_like:
        logits = logits[:, 1:, :]

    choice_a_len = len(a_tok)
    choice_b_len = len(b_tok)

    probs = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_tokens = a_tok if i % 2 == 0 else b_tok
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        for j in range(cur_len):
            cur_tok = cur_tokens[j]
            pos = prefix_lens[i // 2] + j - 1
            probs[i] += -torch.nn.functional.log_softmax(
                logits[i, pos, :],
                dim=0,
            )[cur_tok].item()

        probs[i] /= max(1, cur_len)

    return [
        {
            "target_new": probs[i].item(),
            "target_true": probs[i + 1].item(),
        }
        for i in range(0, len(probs), 2)
    ]


@torch.no_grad()
def official_compute_rewrite_quality_counterfact(model, tok, record, device, llama_like=True):
    """
    Same structure as ZeroUnlearn compute_rewrite_quality_counterfact,
    but without slow generation/TF-IDF text generation.
    """
    rr = record["requested_rewrite"]

    subject = rr["subject"]
    target_new = rr["target_new"]["str"]
    target_true = rr["target_true"]["str"]

    rewrite_prompts = [rr["prompt"].format(subject)]
    paraphrase_prompts = record.get("paraphrase_prompts", [])
    neighborhood_prompts = record.get("neighborhood_prompts", [])

    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]

    flat_prompts = []
    for group in prob_prompts:
        flat_prompts.extend(group)

    probs = official_test_batch_prediction(
        model=model,
        tok=tok,
        prefixes=flat_prompts,
        target_new=target_new,
        target_true=target_true,
        device=device,
        llama_like=llama_like,
    )

    cutoffs = [0] + np.cumsum([len(x) for x in prob_prompts]).tolist()
    ret_probs = [
        probs[cutoffs[i - 1]: cutoffs[i]]
        for i in range(1, len(cutoffs))
    ]

    return {
        "rewrite_prompts_probs": ret_probs[0],
        "paraphrase_prompts_probs": ret_probs[1],
        "neighborhood_prompts_probs": ret_probs[2],
    }


def official_summarize(split_name, metric_data):
    """
    Same formulas as ZeroUnlearn experiments/summarize_list.py.
    Output is scaled by 100, matching their tables.
    """
    vals = {
        "post_rewrite_success": [],
        "post_rewrite_diff": [],
        "post_rewrite_sensitive_pref": [],
        "post_paraphrase_success": [],
        "post_paraphrase_diff": [],
        "post_paraphrase_sensitive_pref": [],
        "post_neighborhood_success": [],
        "post_neighborhood_diff": [],
        "post_rewrite_target_new_nll": [],
        "post_rewrite_target_true_nll": [],
        "post_paraphrase_target_new_nll": [],
        "post_paraphrase_target_true_nll": [],
        "post_neighborhood_target_new_nll": [],
        "post_neighborhood_target_true_nll": [],
    }
    prompt_margins = {
        "rewrite": [],
        "paraphrase": [],
    }

    for data in metric_data:
        post = data["post"]

        for key, out_prefix in [
            ("rewrite_prompts_probs", "rewrite"),
            ("paraphrase_prompts_probs", "paraphrase"),
        ]:
            xs = post.get(key, [])
            if len(xs) == 0:
                continue

            # "success" (ROME/MEMIT edit-quality convention): target_new is
            # preferred over target_true. Kept for parity with
            # ZeroUnlearn/experiments/summarize_list.py and for computing FS/GFS
            # in evaluate_mcf_target_true_sensitive.py. This is NOT Eff/Gen and
            # is HIGHER-is-better (equivalent to FS/GFS, scaled the same way).
            vals[f"post_{out_prefix}_success"].append(
                np.mean([x["target_true"] > x["target_new"] for x in xs])
            )
            vals[f"post_{out_prefix}_diff"].append(
                np.mean([np.exp(-x["target_new"]) - np.exp(-x["target_true"]) for x in xs])
            )
            # Eff/Gen source: fraction of prompts where the model STILL prefers
            # the sensitive target_true over the reference target_new, i.e. the
            # forget has NOT taken effect yet. LOWER is better (0 = fully
            # forgotten). Exact NLL ties count toward neither side, matching
            # evaluate_mcf_target_true_sensitive.py::_record_stats. This is the
            # complement of post_{prefix}_success (up to ties):
            # sensitive_pref ≈ 100 - success.
            vals[f"post_{out_prefix}_sensitive_pref"].append(
                np.mean([x["target_true"] < x["target_new"] for x in xs])
            )
            vals[f"post_{out_prefix}_target_new_nll"].append(
                np.mean([x["target_new"] for x in xs])
            )
            vals[f"post_{out_prefix}_target_true_nll"].append(
                np.mean([x["target_true"] for x in xs])
            )
            prompt_margins[out_prefix].extend(
                float(x["target_new"]) - float(x["target_true"])
                for x in xs
            )

        xs = post.get("neighborhood_prompts_probs", [])
        if len(xs) > 0:
            # Official: success if target_true NLL < target_new NLL
            # i.e., model preserves true neighborhood answer. Higher is better.
            vals["post_neighborhood_success"].append(
                np.mean([x["target_true"] < x["target_new"] for x in xs])
            )
            vals["post_neighborhood_diff"].append(
                np.mean([np.exp(-x["target_true"]) - np.exp(-x["target_new"]) for x in xs])
            )
            vals["post_neighborhood_target_new_nll"].append(
                np.mean([x["target_new"] for x in xs])
            )
            vals["post_neighborhood_target_true_nll"].append(
                np.mean([x["target_true"] for x in xs])
            )

    out = {
        "split_name": split_name,
        "num_cases": len(metric_data),
    }

    for k, v in vals.items():
        if len(v) == 0:
            out[k] = [None, None]
        else:
            scale = 1.0 if k.endswith("_nll") else 100.0
            out[k] = [
                round(float(np.mean(v) * scale), 2),
                round(float(np.std(v) * scale), 2),
            ]

    # Paper-style table:
    # Eff/Gen = fraction of forget prompts still favoring the sensitive
    #           target_true over target_new. LOWER is better (0 = forgotten).
    #           This matches build_post_reload_acceptance_gate's
    #           `max_forget_eff`/`max_forget_gen` thresholds (default 0.0) and
    #           every other consumer of result["forget"]["Eff"/"Gen"] in this
    #           repo. It is the complement of FS/GFS in
    #           evaluate_mcf_target_true_sensitive.py (which are higher-is-
    #           better): Eff ≈ 100 - FS, Gen ≈ 100 - GFS, up to exact-NLL ties.
    # Spe uses neighborhood probability-difference score: higher is better.
    # Keep Spe_success separately for debugging.
    out["Eff"] = out["post_rewrite_sensitive_pref"][0]
    out["Gen"] = out["post_paraphrase_sensitive_pref"][0]
    out["Spe"] = out["post_neighborhood_diff"][0]
    out["Spe_success"] = out["post_neighborhood_success"][0]
    for prompt_type in ("rewrite", "paraphrase"):
        margins = prompt_margins[prompt_type]
        # margin = target_new_NLL - target_true_NLL; success (target_true
        # disfavored) is margin > 0. A genuine failure (forget did not take,
        # including exact ties) is therefore margin <= 0, NOT margin < 0 --
        # margin < 0 is the success condition and was being double-counted
        # here as a "failure".
        out[f"post_{prompt_type}_prompt_instances"] = len(margins)
        out[f"post_{prompt_type}_failure_prompt_instances"] = sum(
            margin <= 0.0 for margin in margins
        )
        out[f"post_{prompt_type}_min_margin"] = (
            float(min(margins)) if margins else None
        )
    combined_margins = [
        *prompt_margins["rewrite"],
        *prompt_margins["paraphrase"],
    ]
    out["minimum_rewrite_paraphrase_margin"] = (
        float(min(combined_margins))
        if combined_margins
        else None
    )

    return out


def _minimum_margin_from_raw(metric_data):
    margins = []
    for item in metric_data or []:
        post = item.get("post", {})
        for key in ("rewrite_prompts_probs", "paraphrase_prompts_probs"):
            for values in post.get(key, []):
                margins.append(
                    float(values["target_new"])
                    - float(values["target_true"])
                )
    return min(margins) if margins else None


def build_post_reload_acceptance_gate(
    result,
    *,
    max_forget_eff=0.0,
    max_forget_gen=0.0,
    min_forget_margin=0.1,
):
    """Validate a serialized/reloaded checkpoint against strict forget gates."""
    thresholds = {
        "max_forget_eff": float(max_forget_eff),
        "max_forget_gen": float(max_forget_gen),
        "min_forget_margin": float(min_forget_margin),
    }
    if any(
        not math.isfinite(value) or value < 0.0
        for value in thresholds.values()
    ):
        raise ValueError(
            "Post-reload acceptance thresholds must be finite and non-negative"
        )

    forget = result.get("forget", result)
    eff = float(forget["Eff"])
    gen = float(forget["Gen"])
    minimum_margin = forget.get("minimum_rewrite_paraphrase_margin")
    if minimum_margin is None:
        minimum_margin = _minimum_margin_from_raw(result.get("forget_raw", []))
    minimum_margin = (
        None if minimum_margin is None else float(minimum_margin)
    )
    observed = {
        "forget_eff": eff,
        "forget_gen": gen,
        "minimum_rewrite_paraphrase_margin": minimum_margin,
        "rewrite_failure_prompt_instances": forget.get(
            "post_rewrite_failure_prompt_instances"
        ),
        "paraphrase_failure_prompt_instances": forget.get(
            "post_paraphrase_failure_prompt_instances"
        ),
    }
    checks = {
        "forget_eff_within_limit": (
            math.isfinite(eff) and eff <= thresholds["max_forget_eff"]
        ),
        "forget_gen_within_limit": (
            math.isfinite(gen) and gen <= thresholds["max_forget_gen"]
        ),
        "forget_margin_meets_floor": (
            minimum_margin is not None
            and math.isfinite(minimum_margin)
            and minimum_margin >= thresholds["min_forget_margin"]
        ),
    }
    failure_reasons = [
        name for name, passed in checks.items() if not passed
    ]
    return {
        "schema_version": 1,
        "kind": "mcf_post_reload_acceptance",
        "checkpoint_was_reloaded": True,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "passed": not failure_reasons,
        "failure_reasons": failure_reasons,
    }


@torch.no_grad()
def official_perplexity(model, tok, text, device, max_input_length=100):
    """
    Same as ZeroUnlearn util/perplexity.py.
    """
    inputs = tok(
        [text],
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    ).to(device)

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(
        logits[:, :-1, :],
        2,
        inputs["input_ids"][:, 1:, None],
    )[0]

    return torch.exp(
        -1 / inputs["input_ids"].size(1) * log_probs.sum()
    ).item()


def load_official_ppl_text(wikidata_dir):
    wikidata_dir = Path(wikidata_dir)
    if not wikidata_dir.exists():
        return None

    raw_ds = load_from_disk(str(wikidata_dir))
    return " ".join(raw_ds["train"]["text"][:20])




def evaluate_record_split(model, tok, records, device, llama_like, split_name):
    metrics = []
    for r in records:
        post = official_compute_rewrite_quality_counterfact(
            model, tok, r, device, llama_like=llama_like
        )
        metrics.append({
            "requested_rewrite": r["requested_rewrite"],
            "post": post,
        })
    return official_summarize(split_name, metrics), metrics


def result_to_comparison_row(result):
    return {
        "method": result["method"],
        "model_dir": result["model_dir"],
        "seed": result["seed"],
        "sample_mode": result["sample_mode"],
        "unlearn_num": result["unlearn_num"],
        "retain_num": result["retain_num"],
        "forget_Eff": result["forget"]["Eff"],
        "forget_Gen": result["forget"]["Gen"],
        "forget_Spe": result["forget"]["Spe"],
        "forget_Spe_success": result["forget"].get("Spe_success"),
        "forget_PPL": result.get("forget_PPL"),
        "retain_Eff": result["retain"]["Eff"],
        "retain_Gen": result["retain"]["Gen"],
        "retain_Spe": result["retain"]["Spe"],
        "retain_Spe_success": result["retain"].get("Spe_success"),
        "retain_PPL": result.get("retain_PPL"),
    }


def write_official_comparison(out_dir, rows):
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    csv_path = out_dir / "official_eval_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Official-compatible MCF evaluation comparison",
        "",
        "Metric directions: Eff ↓ and Gen ↓ indicate stronger unlearning; Spe ↑ indicates better specificity; PPL ↓/stable indicates better fluency.",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(["---"] * len(fieldnames)) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    (out_dir / "official_eval_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def load_official_eval_records(mcf_path, unlearn_num, retain_num, seed, sample_mode):
    """Load exactly the record split used by the ZeroUnlearn-compatible evaluator."""
    mcf_path = download_mcf(mcf_path)
    data = load_mcf(mcf_path)
    if sample_mode == "official":
        return sample_official_split(data, unlearn_num, retain_num, seed)
    if sample_mode == "first":
        return sample_first_split(data, unlearn_num, retain_num)
    raise ValueError(f"Unsupported sample_mode: {sample_mode}")


def evaluate_loaded_model_official(
    method,
    model,
    tok,
    model_dir,
    mcf_path,
    wikidata_dir,
    out_path=None,
    unlearn_num=50,
    retain_num=1000,
    seed=0,
    sample_mode="official",
    skip_ppl=False,
):
    forget_records, retain_records = load_official_eval_records(
        mcf_path=mcf_path,
        unlearn_num=unlearn_num,
        retain_num=retain_num,
        seed=seed,
        sample_mode=sample_mode,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    llama_like = is_llama_like(model, tok)

    forget_summary, forget_raw = evaluate_record_split(model, tok, forget_records, device, llama_like, "forget")
    retain_summary, retain_raw = evaluate_record_split(model, tok, retain_records, device, llama_like, "retain")

    ppl = None
    if not skip_ppl:
        ppl_text = load_official_ppl_text(wikidata_dir)
        if ppl_text is None:
            print(f"[warning] wikidata dir {wikidata_dir} not found. PPL set to null.")
        else:
            ppl = official_perplexity(model, tok, ppl_text, device, max_input_length=100)

    result = {
        "method": method,
        "model_dir": str(model_dir),
        "dataset": "MCF",
        "sample_mode": sample_mode,
        "seed": seed,
        "unlearn_num": unlearn_num,
        "retain_num": retain_num,
        "llama_like": llama_like,
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    return result


def evaluate_model_dir_official(
    method,
    model_dir,
    mcf_path,
    wikidata_dir,
    out_path=None,
    unlearn_num=50,
    retain_num=1000,
    seed=0,
    sample_mode="official",
    dtype="bfloat16",
    device_map="auto",
    skip_ppl=False,
):
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory for {method!r} does not exist: {model_dir}")

    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        load_kwargs = {"torch_dtype": dtype_from_str(dtype)}
        if device_map and str(device_map).lower() not in {"none", "single"}:
            load_kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
        if "device_map" not in load_kwargs:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for single-device official MCF evaluation, but torch.cuda.is_available() is False")
            model = model.to("cuda")
    except Exception as exc:
        raise RuntimeError(f"Failed to load model for method {method!r} from {model_dir}: {exc}") from exc

    return evaluate_loaded_model_official(
        method=method,
        model=model,
        tok=tok,
        model_dir=model_dir,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=out_path,
        unlearn_num=unlearn_num,
        retain_num=retain_num,
        seed=seed,
        sample_mode=sample_mode,
        skip_ppl=skip_ppl,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--wikidata-dir", default="data/wikidata")
    ap.add_argument("--out", required=True)

    ap.add_argument("--unlearn-num", type=int, default=50)
    ap.add_argument("--retain-num", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-mode", choices=["official", "first"], default="official")

    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--skip-ppl", action="store_true")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Write the result JSON without printing the full payload to stdout",
    )
    ap.add_argument("--max-forget-eff", type=float, default=None)
    ap.add_argument("--max-forget-gen", type=float, default=None)
    ap.add_argument("--min-forget-margin", type=float, default=None)
    ap.add_argument(
        "--fail-if-gate-missed",
        action="store_true",
        help=(
            "Exit with status 3 when the serialized/reloaded checkpoint misses "
            "any configured forget acceptance threshold."
        ),
    )
    args = ap.parse_args()
    gate_values = (
        args.max_forget_eff,
        args.max_forget_gen,
        args.min_forget_margin,
    )
    if any(value is not None for value in gate_values) and not all(
        value is not None for value in gate_values
    ):
        ap.error(
            "--max-forget-eff, --max-forget-gen, and "
            "--min-forget-margin must be supplied together"
        )
    if args.fail_if_gate_missed and not all(
        value is not None for value in gate_values
    ):
        ap.error("--fail-if-gate-missed requires all forget gate thresholds")

    result = evaluate_model_dir_official(
        method="model",
        model_dir=args.model_dir,
        mcf_path=args.mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=args.out,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode=args.sample_mode,
        dtype=args.dtype,
        device_map=args.device_map,
        skip_ppl=args.skip_ppl,
    )

    # Backward-compatible top-level aliases for older one-model usage.
    result["Eff"] = result["forget"]["Eff"]
    result["Gen"] = result["forget"]["Gen"]
    result["Spe"] = result["forget"]["Spe"]
    result["PPL"] = result["forget_PPL"]
    result["summary"] = result["forget"]
    if all(value is not None for value in gate_values):
        result["post_reload_acceptance"] = build_post_reload_acceptance_gate(
            result,
            max_forget_eff=args.max_forget_eff,
            max_forget_gen=args.max_forget_gen,
            min_forget_margin=args.min_forget_margin,
        )
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if not args.quiet:
        print(json.dumps(result, indent=2))
    if (
        args.fail_if_gate_missed
        and not result["post_reload_acceptance"]["passed"]
    ):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
