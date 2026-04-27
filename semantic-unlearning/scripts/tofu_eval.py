import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from rouge_score import rouge_scorer
from scipy.stats import ks_2samp
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ECO-style TOFU evaluation for semantic-unlearning."
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--forget-split", type=str, default=None)
    parser.add_argument("--retain-split", type=str, default=None)
    parser.add_argument("--reference-model-dir", type=str, default=None)
    parser.add_argument("--reference-truth-ratios", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/tofu")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--base-model",
        action="store_true",
        help="Compatibility flag for existing commands; only affects metadata naming.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def exp_safe(x: float) -> float:
    return 0.0 if x < -50 else float(math.exp(x))


def normalized_prob(prob_answer: float, all_probs: Sequence[float]) -> float:
    denom = float(np.sum(all_probs))
    if denom <= 0:
        return 0.0
    return float(prob_answer / denom)


def log_truth_ratio(log_probs_perturbed: Sequence[float], log_prob_paraphrased: float) -> float:
    if not log_probs_perturbed:
        return 0.0
    return float(np.exp(np.mean(np.array(log_probs_perturbed)) - log_prob_paraphrased))


class Evaluator:
    def __init__(self, model_path: str, device: str, dtype: str, max_length: int):
        self.requested_device = device
        self.device = self._normalize_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    @staticmethod
    def _normalize_device(device: str) -> str:
        if device.startswith("cuda") and torch.cuda.is_available():
            return device
        return "cpu"

    @staticmethod
    def _resolve_dtype(dtype: str) -> torch.dtype:
        if dtype == "float16" and torch.cuda.is_available():
            return torch.float16
        if dtype == "bfloat16" and torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def format_question_prompt(self, question: str) -> str:
        if self.tokenizer.chat_template is not None:
            msg = [{"role": "user", "content": f"Question: {question} Answer:"}]
            return self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return f"Question: {question} Answer:"

    def format_prompt_and_answer(self, question: str, answer: str) -> Tuple[str, str]:
        q_prompt = self.format_question_prompt(question)
        return q_prompt + f" {answer}", q_prompt

    @torch.no_grad()
    def answer_log_prob(self, question: str, answer: str) -> float:
        full_text, q_text = self.format_prompt_and_answer(question, answer)
        full_enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        q_enc = self.tokenizer(
            q_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        q_len = q_enc["input_ids"].shape[1]
        outputs = self.model(**full_enc, labels=full_enc["input_ids"])
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        input_ids = full_enc["input_ids"][0]
        answer_ids = input_ids[q_len:]
        if answer_ids.numel() == 0:
            return float("-inf")

        answer_logp = log_probs[0, q_len - 1 : -1, :]
        token_logps = answer_logp[torch.arange(answer_ids.shape[0], device=self.device), answer_ids]
        return float(token_logps.mean().item())

    @torch.no_grad()
    def generate_answer(self, question: str, max_new_tokens: int) -> str:
        prompt = self.format_question_prompt(question)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def answer_prob(self, question: str, answer: str) -> float:
        return exp_safe(self.answer_log_prob(question, answer))

    def rouge_l_recall(self, reference: str, generated: str) -> float:
        return float(self.rouge.score(reference, generated)["rougeL"].recall)


# ---------- Dataset field helpers ----------

def get_primary_answer(sample: Dict[str, Any]) -> str:
    if sample.get("answer"):
        return sample["answer"]
    raise KeyError("Sample does not contain an 'answer' field.")



def get_paraphrased_answer(sample: Dict[str, Any]) -> str:
    if sample.get("paraphrased_answer"):
        return sample["paraphrased_answer"]
    if sample.get("answer"):
        return sample["answer"]
    raise KeyError("Could not find paraphrased/correct answer field.")



def get_perturbed_answers(sample: Dict[str, Any]) -> List[str]:
    value = None
    if "perturbed_answer" in sample:
        value = sample["perturbed_answer"]
    elif "perturbed_answers" in sample:
        value = sample["perturbed_answers"]
    if value is None:
        raise KeyError("Could not find perturbed-answer field.")
    if isinstance(value, list):
        return [x for x in value if isinstance(x, str) and x.strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []



def build_candidates_for_perturbed(sample: Dict[str, Any]) -> List[str]:
    correct = get_paraphrased_answer(sample)
    wrong = get_perturbed_answers(sample)
    return [correct] + wrong


# ---------- Metric computations ----------

def evaluate_answer_prob(
    evaluator: Evaluator,
    samples: Sequence[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    values = []
    for sample in tqdm(samples, desc=f"AnswerProb[{split_name}]"):
        values.append(evaluator.answer_prob(sample["question"], get_primary_answer(sample)))
    return {"mean": safe_mean(values), "values": values}



def evaluate_truth_ratio(
    evaluator: Evaluator,
    samples: Sequence[Dict[str, Any]],
    split_name: str,
    mode: str,
) -> Dict[str, Any]:
    assert mode in {"min", "clip"}
    values = []
    for sample in tqdm(samples, desc=f"TruthRatio[{split_name}:{mode}]"):
        question = sample["question"]
        candidates = build_candidates_for_perturbed(sample)
        log_probs = [evaluator.answer_log_prob(question, ans) for ans in candidates]
        raw_tr = log_truth_ratio(log_probs[1:], log_probs[0])
        if mode == "min":
            tr = min(raw_tr, 1.0 / raw_tr) if raw_tr > 0 else 0.0
        else:
            tr = max(0.0, 1.0 - raw_tr)
        values.append(float(tr))
    return {"mean": safe_mean(values), "values": values}



def evaluate_normalized_answer_prob(
    evaluator: Evaluator,
    samples: Sequence[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    values = []
    for sample in tqdm(samples, desc=f"NormalizedAnswerProb[{split_name}]"):
        question = sample["question"]
        candidates = build_candidates_for_perturbed(sample)
        probs = [evaluator.answer_prob(question, ans) for ans in candidates]
        values.append(normalized_prob(probs[0], probs))
    return {"mean": safe_mean(values), "values": values}



def evaluate_rouge_l_recall(
    evaluator: Evaluator,
    samples: Sequence[Dict[str, Any]],
    split_name: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    values = []
    generations = []
    for sample in tqdm(samples, desc=f"ROUGE-L Recall[{split_name}]"):
        generated = evaluator.generate_answer(sample["question"], max_new_tokens=max_new_tokens)
        gold = get_primary_answer(sample)
        values.append(evaluator.rouge_l_recall(gold, generated))
        generations.append(
            {
                "question": sample["question"],
                "gold": gold,
                "generated": generated,
            }
        )
    return {"mean": safe_mean(values), "values": values, "generations": generations}


# ---------- IO helpers ----------

def subset_samples(ds, n_samples: Optional[int], seed: int) -> List[Dict[str, Any]]:
    samples = list(ds)
    if n_samples is not None and len(samples) > n_samples:
        rng = random.Random(seed)
        samples = rng.sample(samples, n_samples)
    return samples



def write_generations_csv(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\")



def append_summary_csv(path: Path, row: Dict[str, Any]) -> None:
    new_df = pd.DataFrame([row])
    if path.exists():
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(path, index=False)



def maybe_compute_ks_pvalue(
    current_forget_tr_values: Sequence[float],
    reference_values: Optional[Sequence[float]],
) -> Optional[float]:
    if not reference_values:
        return None
    return float(ks_2samp(current_forget_tr_values, reference_values).pvalue)



def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    seed = args.seed if args.seed is not None else cfg.get("data", {}).get("seed", 42)
    seed_everything(seed)

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})

    model_dir = args.model_dir or model_cfg.get("name")
    if model_dir is None:
        raise ValueError("Model path is missing. Set model.name in config or pass --model-dir.")

    method = args.method or Path(model_dir).name
    forget_split = args.forget_split or data_cfg.get("forget_split", "forget05")
    retain_split = args.retain_split or data_cfg.get("retain_split", "retain95")
    device = model_cfg.get("device", "cuda:0")
    dtype = model_cfg.get("dtype", "float16")
    max_length = int(model_cfg.get("max_length", 128))
    n_forget = data_cfg.get("n_forget")
    n_retain = data_cfg.get("n_retain")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(model_path=model_dir, device=device, dtype=dtype, max_length=max_length)

    from datasets import load_dataset

    forget_ds = subset_samples(load_dataset("locuslab/TOFU", name=forget_split, split="train"), n_forget, seed)
    retain_ds = subset_samples(load_dataset("locuslab/TOFU", name=retain_split, split="train"), n_retain, seed)
    real_authors_ds = list(load_dataset("locuslab/TOFU", name="real_authors", split="train"))
    world_facts_ds = list(load_dataset("locuslab/TOFU", name="world_facts", split="train"))

    forget_perturbed_ds = list(load_dataset("locuslab/TOFU", name=f"{forget_split}_perturbed", split="train"))
    retain_perturbed_ds = list(load_dataset("locuslab/TOFU", name="retain_perturbed", split="train"))
    real_authors_perturbed_ds = list(load_dataset("locuslab/TOFU", name="real_authors_perturbed", split="train"))
    world_facts_perturbed_ds = list(load_dataset("locuslab/TOFU", name="world_facts_perturbed", split="train"))

    results: Dict[str, Any] = {
        "name": method,
        "method": method,
        "model_dir": model_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "seed": seed,
        "base_model": bool(args.base_model),
    }
    details: Dict[str, Any] = {
        "metadata": results.copy(),
        "per_split": {},
    }

    # ECO-style AnswerProb on clean forget + retain
    forget_ap = evaluate_answer_prob(evaluator, forget_ds, forget_split)
    retain_ap = evaluate_answer_prob(evaluator, retain_ds, retain_split)
    results[f"tofu_{forget_split}_answer_prob"] = forget_ap["mean"]
    results[f"tofu_{retain_split}_answer_prob"] = retain_ap["mean"]
    details["per_split"][forget_split] = {"answer_prob": forget_ap}
    details["per_split"][retain_split] = {"answer_prob": retain_ap}

    # ECO-style NormalizedAnswerProb on perturbed real_authors + world_facts
    real_authors_nap = evaluate_normalized_answer_prob(
        evaluator, real_authors_perturbed_ds, "real_authors_perturbed"
    )
    world_facts_nap = evaluate_normalized_answer_prob(
        evaluator, world_facts_perturbed_ds, "world_facts_perturbed"
    )
    results["tofu-perturbed_real_authors_perturbed_normalized_answer_prob"] = real_authors_nap["mean"]
    results["tofu-perturbed_world_facts_perturbed_normalized_answer_prob"] = world_facts_nap["mean"]
    details["per_split"]["real_authors_perturbed"] = {
        **details["per_split"].get("real_authors_perturbed", {}),
        "normalized_answer_prob": real_authors_nap,
    }
    details["per_split"]["world_facts_perturbed"] = {
        **details["per_split"].get("world_facts_perturbed", {}),
        "normalized_answer_prob": world_facts_nap,
    }

    # ECO-style TruthRatio on perturbed splits
    forget_tr = evaluate_truth_ratio(evaluator, forget_perturbed_ds, f"{forget_split}_perturbed", mode="min")
    retain_tr = evaluate_truth_ratio(evaluator, retain_perturbed_ds, "retain_perturbed", mode="clip")
    real_authors_tr = evaluate_truth_ratio(
        evaluator, real_authors_perturbed_ds, "real_authors_perturbed", mode="clip"
    )
    world_facts_tr = evaluate_truth_ratio(
        evaluator, world_facts_perturbed_ds, "world_facts_perturbed", mode="clip"
    )
    results[f"tofu-perturbed_{forget_split}_perturbed_truth_ratio"] = forget_tr["mean"]
    results["tofu-perturbed_retain_perturbed_truth_ratio"] = retain_tr["mean"]
    results["tofu-perturbed_real_authors_perturbed_truth_ratio"] = real_authors_tr["mean"]
    results["tofu-perturbed_world_facts_perturbed_truth_ratio"] = world_facts_tr["mean"]
    details["per_split"][f"{forget_split}_perturbed"] = {
        **details["per_split"].get(f"{forget_split}_perturbed", {}),
        "truth_ratio": forget_tr,
    }
    details["per_split"]["retain_perturbed"] = {
        **details["per_split"].get("retain_perturbed", {}),
        "truth_ratio": retain_tr,
    }
    details["per_split"]["real_authors_perturbed"] = {
        **details["per_split"].get("real_authors_perturbed", {}),
        "truth_ratio": real_authors_tr,
    }
    details["per_split"]["world_facts_perturbed"] = {
        **details["per_split"].get("world_facts_perturbed", {}),
        "truth_ratio": world_facts_tr,
    }

    # Optional KS test against a reference/retain model on the forget_perturbed truth-ratio distribution.
    reference_values = None
    if args.reference_truth_ratios:
        with open(args.reference_truth_ratios, "r", encoding="utf-8") as f:
            reference_values = json.load(f)
    elif args.reference_model_dir:
        ref_eval = Evaluator(
            model_path=args.reference_model_dir,
            device=device,
            dtype=dtype,
            max_length=max_length,
        )
        reference_values = evaluate_truth_ratio(
            ref_eval, forget_perturbed_ds, f"{forget_split}_perturbed_reference", mode="min"
        )["values"]
        del ref_eval
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ks_pvalue = maybe_compute_ks_pvalue(forget_tr["values"], reference_values)
    if ks_pvalue is not None:
        results["ks_test_p_value"] = ks_pvalue
        details["ks_test_p_value"] = ks_pvalue
        ref_path = output_dir / f"{method}_{forget_split}_reference_truth_ratios.json"
        with open(ref_path, "w", encoding="utf-8") as f:
            json.dump(reference_values, f, indent=2)

    # ECO-style ROUGE-L recall on clean forget/retain/real_authors/world_facts.
    forget_rg = evaluate_rouge_l_recall(evaluator, forget_ds, forget_split, args.max_new_tokens)
    retain_rg = evaluate_rouge_l_recall(evaluator, retain_ds, retain_split, args.max_new_tokens)
    real_authors_rg = evaluate_rouge_l_recall(evaluator, real_authors_ds, "real_authors", args.max_new_tokens)
    world_facts_rg = evaluate_rouge_l_recall(evaluator, world_facts_ds, "world_facts", args.max_new_tokens)

    results[f"tofu_{forget_split}_rougeL_recall"] = forget_rg["mean"]
    results[f"tofu_{retain_split}_rougeL_recall"] = retain_rg["mean"]
    results["tofu_real_authors_rougeL_recall"] = real_authors_rg["mean"]
    results["tofu_world_facts_rougeL_recall"] = world_facts_rg["mean"]
    details["per_split"][forget_split]["rougeL_recall"] = forget_rg
    details["per_split"][retain_split]["rougeL_recall"] = retain_rg
    details["per_split"]["real_authors"] = {"rougeL_recall": real_authors_rg}
    details["per_split"]["world_facts"] = {"rougeL_recall": world_facts_rg}

    # Compact aliases for quick comparison tables.
    results["forget_answer_prob"] = results[f"tofu_{forget_split}_answer_prob"]
    results["retain_answer_prob"] = results[f"tofu_{retain_split}_answer_prob"]
    results["forget_truth_ratio"] = results[f"tofu-perturbed_{forget_split}_perturbed_truth_ratio"]
    results["retain_truth_ratio"] = results["tofu-perturbed_retain_perturbed_truth_ratio"]
    results["real_authors_truth_ratio"] = results["tofu-perturbed_real_authors_perturbed_truth_ratio"]
    results["world_facts_truth_ratio"] = results["tofu-perturbed_world_facts_perturbed_truth_ratio"]
    results["real_authors_normalized_answer_prob"] = results[
        "tofu-perturbed_real_authors_perturbed_normalized_answer_prob"
    ]
    results["world_facts_normalized_answer_prob"] = results[
        "tofu-perturbed_world_facts_perturbed_normalized_answer_prob"
    ]
    results["forget_rougeL_recall"] = results[f"tofu_{forget_split}_rougeL_recall"]
    results["retain_rougeL_recall"] = results[f"tofu_{retain_split}_rougeL_recall"]

    summary_json = output_dir / f"{method}_summary.json"
    details_json = output_dir / f"{method}_details.json"
    summary_csv = output_dir / "tofu_eco_summary.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(details_json, "w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    write_generations_csv(output_dir / f"{method}_{forget_split}_generations.csv", forget_rg["generations"])
    write_generations_csv(output_dir / f"{method}_{retain_split}_generations.csv", retain_rg["generations"])
    write_generations_csv(output_dir / f"{method}_real_authors_generations.csv", real_authors_rg["generations"])
    write_generations_csv(output_dir / f"{method}_world_facts_generations.csv", world_facts_rg["generations"])
    append_summary_csv(summary_csv, results)

    print("\n[done] ECO-style TOFU metrics written to:")
    print(f"  - {summary_json}")
    print(f"  - {details_json}")
    print(f"  - {summary_csv}")
    print("\nKey metrics:")
    for key in [
        "forget_answer_prob",
        "retain_answer_prob",
        "forget_truth_ratio",
        "retain_truth_ratio",
        "real_authors_truth_ratio",
        "world_facts_truth_ratio",
        "real_authors_normalized_answer_prob",
        "world_facts_normalized_answer_prob",
        "forget_rougeL_recall",
        "retain_rougeL_recall",
    ]:
        print(f"  {key}: {results[key]:.6f}")
    if "ks_test_p_value" in results:
        print(f"  ks_test_p_value: {results['ks_test_p_value']:.6f}")


if __name__ == "__main__":
    main()
