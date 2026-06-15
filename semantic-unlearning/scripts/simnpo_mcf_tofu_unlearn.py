#!/usr/bin/env python3
"""Reference-free SimNPO unlearning for MCF and TOFU.

The forget objective is SimNPO over all answer tokens. Retain behavior is
preserved with answer-token CE and optional teacher||student KL on retain answer
positions. Selective mode restricts trainable embedding/lm-head rows only; it
never masks the answer-token sequence used by SimNPO.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
COMMON_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "them", "his", "her", "their",
    "i", "you", "we", "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "also", "there", "here", "who",
    "what", "when", "where", "why", "how", "which", "than", "into", "about", "over", "under",
}


@dataclass
class Example:
    prompt: str
    answer: str
    subject: str = ""
    target_new: str = ""
    target_true: str = ""
    paraphrase_prompts: Optional[List[str]] = None
    source: str = ""


@dataclass
class SequenceStats:
    logp_per_example: torch.Tensor
    length_per_example: torch.Tensor
    mean_logp_per_example: torch.Tensor


@dataclass
class SimNPOLossResult:
    loss: torch.Tensor
    policy_neg_logp_mean: float
    answer_tokens_mean: float
    mean_logp_per_token: float


@dataclass
class LossResult:
    loss: torch.Tensor
    contributing_tokens: int


@dataclass
class ParamSummary:
    n_trainable_params: int
    n_total_params: int
    trainable_param_percent: float
    trainable_names: List[str]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def normalize_answer(text: Any) -> str:
    answer = str(text).strip()
    return answer if answer.startswith(" ") else " " + answer


def token_ids_for_text(tok: AutoTokenizer, text: str) -> List[int]:
    return [int(x) for x in tok(text, add_special_tokens=False)["input_ids"]]


def answer_with_eos(tok: AutoTokenizer, answer: str) -> str:
    if tok.eos_token and not answer.endswith(tok.eos_token):
        return answer + tok.eos_token
    return answer


def special_token_ids(tok: AutoTokenizer) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    ids.update(x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id] if x is not None)
    return {int(x) for x in ids if x is not None}


def format_mcf_prompt(prompt_template: str, subject: str) -> str:
    return prompt_template.format(subject) if "{}" in prompt_template else prompt_template


def extract_mcf_rewrite(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        rr = rr[0] if rr else None
    if not isinstance(rr, dict):
        raise ValueError("MCF record missing dict requested_rewrite field")
    missing = [key for key in ("prompt", "subject", "target_new") if key not in rr]
    if missing:
        raise ValueError(f"MCF requested_rewrite missing required fields: {missing}")
    if not isinstance(rr.get("target_new"), dict) or "str" not in rr["target_new"]:
        raise ValueError("MCF requested_rewrite.target_new.str is required")
    paraphrases = record.get("paraphrase_prompts") or rr.get("paraphrase_prompts") or []
    if not isinstance(paraphrases, list):
        paraphrases = []
    return rr, [str(p) for p in paraphrases]


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MCF data from {url} to {path}")
    urllib.request.urlretrieve(url, path)


def load_mcf(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    cache_path = resolve_path(args.mcf_cache_path)
    download_if_missing(args.mcf_url, cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("MCF cache JSON must contain a list of records")

    examples: List[Example] = []
    for rec in raw:
        rr, paraphrases = extract_mcf_rewrite(rec)
        subject = str(rr["subject"])
        prompt = format_mcf_prompt(str(rr["prompt"]), subject)
        target_new = normalize_answer(rr["target_new"]["str"])
        target_true = ""
        if isinstance(rr.get("target_true"), dict) and rr["target_true"].get("str"):
            target_true = normalize_answer(rr["target_true"]["str"])
        if args.mcf_answer_field == "target_true" and not target_true:
            raise ValueError("--mcf-answer-field target_true requested, but an MCF record is missing target_true.str")
        answer = target_true if args.mcf_answer_field == "target_true" else target_new
        examples.append(Example(
            prompt=prompt,
            answer=answer,
            subject=subject,
            target_new=target_new,
            target_true=target_true,
            paraphrase_prompts=[format_mcf_prompt(p, subject) for p in paraphrases],
            source="mcf",
        ))

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    need = args.forget_num + args.retain_num
    if len(examples) < need:
        raise ValueError(f"MCF contains {len(examples)} examples, need forget_num + retain_num = {need}")
    return examples[: args.forget_num], examples[args.forget_num: need]


def load_tofu(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    forget_ds = load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=args.retain_split, split="train")

    def convert(row: Dict[str, Any]) -> Example:
        if "question" not in row or "answer" not in row:
            raise ValueError("TOFU rows must contain question and answer fields")
        return Example(prompt=f"Question: {row['question']}\nAnswer:", answer=normalize_answer(row["answer"]), source="tofu")

    forget = [convert(r) for r in forget_ds]
    retain = [convert(r) for r in retain_ds]
    rng = random.Random(args.seed)
    rng.shuffle(forget)
    rng.shuffle(retain)
    return forget[: args.forget_num], retain[: args.retain_num]


def load_data(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    if args.dataset == "mcf":
        return load_mcf(args)
    if args.dataset == "tofu":
        return load_tofu(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def is_bad_selective_token(tok: AutoTokenizer, token_id: int, specials: set[int]) -> bool:
    if token_id in specials:
        return True
    text = tok.decode([token_id], skip_special_tokens=True).strip().lower()
    if not text:
        return True
    if PUNCT_RE.fullmatch(text):
        return True
    return text in COMMON_FUNCTION_WORDS


def select_contrast_tokens(tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], top_k: int) -> List[int]:
    specials = special_token_ids(tok)
    forget_df: Counter[int] = Counter()
    retain_df: Counter[int] = Counter()
    for ex in forget:
        forget_df.update(set(tid for tid in token_ids_for_text(tok, ex.answer) if tid not in specials))
    for ex in retain:
        retain_df.update(set(tid for tid in token_ids_for_text(tok, ex.answer) if tid not in specials))

    n_forget = max(len(forget), 1)
    n_retain = max(len(retain), 1)
    scored: List[Tuple[float, int]] = []
    for token_id, df in forget_df.items():
        if is_bad_selective_token(tok, token_id, specials):
            continue
        score = (df / n_forget) / ((retain_df.get(token_id, 0) / n_retain) + 1e-6)
        scored.append((score, token_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [token_id for _, token_id in scored[:top_k]]


def build_batch(tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    if not examples:
        raise ValueError("build_batch received no examples")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for ex in examples:
        prompt_ids = token_ids_for_text(tok, ex.prompt)
        answer_ids = token_ids_for_text(tok, answer_with_eos(tok, ex.answer))
        if not answer_ids:
            raise ValueError(f"Example has zero answer tokens for prompt: {ex.prompt[:100]!r}")
        rows.append(prompt_ids + answer_ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)

    max_len = max(len(row) for row in rows)
    input_ids: List[List[int]] = []
    label_ids: List[List[int]] = []
    attention: List[List[int]] = []
    for row, lab in zip(rows, labels):
        pad_len = max_len - len(row)
        input_ids.append(row + [pad_id] * pad_len)
        label_ids.append(lab + [-100] * pad_len)
        attention.append([1] * len(row) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(label_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def sequence_logp_and_lengths(logits: torch.Tensor, labels_full: torch.Tensor) -> SequenceStats:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_full[:, 1:]
    answer_mask = shift_labels.ne(-100)
    safe_labels = shift_labels.clamp_min(0)
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    gathered = gathered * answer_mask.float()
    lengths = answer_mask.sum(dim=1).clamp_min(1)
    logp = gathered.sum(dim=1)
    mean_logp = logp / lengths
    return SequenceStats(logp_per_example=logp, length_per_example=lengths, mean_logp_per_example=mean_logp)


def simnpo_forget_loss(logits: torch.Tensor, labels_full: torch.Tensor, beta: float, gamma: float) -> SimNPOLossResult:
    if beta <= 0:
        raise ValueError("--beta must be > 0 for SimNPO")
    stats = sequence_logp_and_lengths(logits, labels_full)
    x = -beta * stats.mean_logp_per_example - gamma
    loss_per_example = -(2.0 / beta) * F.logsigmoid(x)
    loss = loss_per_example.mean()
    return SimNPOLossResult(
        loss=loss,
        policy_neg_logp_mean=float((-stats.logp_per_example).mean().detach().cpu()),
        answer_tokens_mean=float(stats.length_per_example.float().mean().detach().cpu()),
        mean_logp_per_token=float(stats.mean_logp_per_example.mean().detach().cpu()),
    )


def answer_ce_loss(logits: torch.Tensor, labels_full: torch.Tensor) -> LossResult:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels_full[:, 1:].contiguous()
    answer_mask = shift_labels.ne(-100)
    per_token = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    denom = answer_mask.sum().clamp_min(1)
    return LossResult(loss=(per_token * answer_mask.float()).sum() / denom, contributing_tokens=int(denom.item()))


def retain_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels_full: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("--kl-temperature must be > 0")
    student = student_logits[:, :-1, :].float() / temperature
    teacher = teacher_logits[:, :-1, :].float() / temperature
    answer_mask = labels_full[:, 1:].ne(-100)
    teacher_logp = F.log_softmax(teacher, dim=-1).detach()
    teacher_p = teacher_logp.exp()
    student_logp = F.log_softmax(student, dim=-1)
    kl = (teacher_p * (teacher_logp - student_logp)).sum(dim=-1) * (temperature ** 2)
    return (kl * answer_mask.float()).sum() / answer_mask.sum().clamp_min(1)


def sample_batch(examples: Sequence[Example], batch_size: int) -> List[Example]:
    if not examples:
        raise ValueError("Cannot sample from an empty example list")
    return [examples[random.randrange(len(examples))] for _ in range(min(batch_size, len(examples)))]


def configure_trainable(model: torch.nn.Module, mode: str, selected_ids: Sequence[int]) -> Tuple[ParamSummary, Dict[str, Any]]:
    total = sum(param.numel() for param in model.parameters())
    info: Dict[str, Any] = {"tied": False, "hooks": []}
    if mode == "full_all_tokens":
        for param in model.parameters():
            param.requires_grad_(True)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise ValueError("Model must provide input and output embeddings")
        input_weight = input_embeddings.weight
        output_weight = output_embeddings.weight
        tied = input_weight.data_ptr() == output_weight.data_ptr()
        input_weight.requires_grad_(True)
        if not tied:
            output_weight.requires_grad_(True)
        info.update({"input_weight": input_weight, "output_weight": output_weight, "tied": tied})
        if mode == "emb_lm_selective_tokens":
            if not selected_ids:
                raise ValueError("Selective mode selected zero token ids; increase --selective-top-k or check data")

            def make_hook(weight: torch.nn.Parameter):
                row_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
                ids = torch.tensor([idx for idx in selected_ids if 0 <= idx < weight.shape[0]], dtype=torch.long, device=weight.device)
                if ids.numel() > 0:
                    row_mask[ids] = True

                def hook(grad: torch.Tensor) -> torch.Tensor:
                    masked = grad.clone()
                    masked[~row_mask] = 0
                    return masked

                return hook

            info["hooks"].append(input_weight.register_hook(make_hook(input_weight)))
            if not tied:
                info["hooks"].append(output_weight.register_hook(make_hook(output_weight)))

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return ParamSummary(trainable, total, 100.0 * trainable / max(total, 1), trainable_names), info


def unique_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    seen: set[int] = set()
    params: List[torch.nn.Parameter] = []
    for param in model.parameters():
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            params.append(param)
    return params


def make_optimizer(params: Sequence[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def ensure_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
    return device


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, device: torch.device, training: bool):
    if not model_path:
        raise ValueError("--model-path is required")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    if tok.pad_token_id is None:
        raise ValueError("Tokenizer has no pad token, EOS token, or UNK token to use for padding")
    if training:
        model.config.use_cache = False
    return model, tok


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_tofu_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    eval_script = SCRIPT_DIR / "tofu_eval.py"
    if not eval_script.exists():
        print(f"WARNING: {eval_script} is unavailable; skipping TOFU eval", file=sys.stderr)
        return
    eval_dir = resolve_path(args.output_dir) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--model-path",
        str(checkpoint),
        "--output-dir",
        str(eval_dir),
        "--forget-split",
        args.forget_split,
        "--retain-split",
        args.retain_split,
        "--eval-n-forget",
        str(args.eval_n_forget),
        "--eval-n-retain",
        str(args.eval_n_retain),
        "--eval-n-real-authors",
        str(args.eval_n_real_authors),
        "--eval-n-world-facts",
        str(args.eval_n_world_facts),
        "--eval-n-perturbed",
        str(args.eval_n_perturbed),
        "--eval-max-new-tokens",
        str(args.eval_max_new_tokens),
    ]
    subprocess.run(cmd, check=True)


def run_mcf_official_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    if not args.official_eval:
        return
    try:
        from mcf_zero_unlearn_official_eval import (  # type: ignore
            evaluate_model_dir_official,
            result_to_comparison_row,
            write_official_comparison,
        )
    except Exception as exc:
        print(f"WARNING: cannot import mcf_zero_unlearn_official_eval.py ({exc}); skipping official MCF eval", file=sys.stderr)
        return
    output_dir = resolve_path(args.output_dir)
    base = evaluate_model_dir_official(args.model_path, args.official_unlearn_num, args.official_retain_num, args.official_sample_mode)
    checkpoint_result = evaluate_model_dir_official(str(checkpoint), args.official_unlearn_num, args.official_retain_num, args.official_sample_mode)
    rows = [result_to_comparison_row("base", base), result_to_comparison_row("checkpoint", checkpoint_result)]
    write_official_comparison(rows, output_dir / "official_eval_comparison.csv", output_dir / "official_eval_comparison.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimNPO + retain CE/KL unlearning for MCF and TOFU")
    parser.add_argument("--dataset", choices=["mcf", "tofu"], default="mcf")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reference-model-path", default=None, help="Teacher for retain KL only. Defaults to --model-path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--mcf-cache-path", default="data/multi_counterfact.json")
    parser.add_argument("--mcf-url", default="https://memit.baulab.info/data/dsets/multi_counterfact.json")
    parser.add_argument("--mcf-answer-field", choices=["target_true", "target_new"], default="target_true")
    parser.add_argument("--mode", choices=["full_all_tokens", "emb_lm_all_tokens", "emb_lm_selective_tokens"], default="emb_lm_selective_tokens")
    parser.add_argument("--selective-top-k", type=int, default=300)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--simnpo-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=0.3)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-n-forget", type=int, default=200)
    parser.add_argument("--eval-n-retain", type=int, default=400)
    parser.add_argument("--eval-n-real-authors", type=int, default=100)
    parser.add_argument("--eval-n-world-facts", type=int, default=117)
    parser.add_argument("--eval-n-perturbed", type=int, default=100)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--official-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--official-unlearn-num", type=int, default=50)
    parser.add_argument("--official-retain-num", type=int, default=1000)
    parser.add_argument("--official-sample-mode", default="official")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beta <= 0:
        raise ValueError("--beta must be > 0")
    if args.kl_temperature <= 0:
        raise ValueError("--kl-temperature must be > 0")
    set_seed(args.seed)
    device = ensure_device(args.device)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model_and_tokenizer(args.model_path, torch_dtype(args.dtype), device, training=True)
    forget_examples, retain_examples = load_data(args)
    selected_ids = select_contrast_tokens(tok, forget_examples, retain_examples, args.selective_top_k) if args.mode == "emb_lm_selective_tokens" else []
    write_json(output_dir / "selected_tokens.json", {
        "token_ids": selected_ids,
        "tokens": [tok.decode([token_id], skip_special_tokens=True) for token_id in selected_ids],
        "mode": args.mode,
    })

    param_summary, train_info = configure_trainable(model, args.mode, selected_ids)
    params = unique_trainable_params(model)
    if not params:
        raise ValueError("No trainable parameters were selected")
    optimizer = make_optimizer(params, args)

    teacher = None
    if args.retain_kl_weight > 0:
        teacher_path = args.reference_model_path or args.model_path
        teacher, _ = load_model_and_tokenizer(teacher_path, torch_dtype(args.dtype), device, training=False)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    model.train()
    last_row: Dict[str, Any] = {}
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, args.steps + 1), desc="SimNPO train"):
            forget_batch = build_batch(tok, sample_batch(forget_examples, args.batch_size), device)
            retain_batch = build_batch(tok, sample_batch(retain_examples, args.retain_batch_size), device)
            optimizer.zero_grad(set_to_none=True)

            forget_logits = model(input_ids=forget_batch["input_ids"], attention_mask=forget_batch["attention_mask"]).logits
            simnpo = simnpo_forget_loss(forget_logits, forget_batch["labels"], args.beta, args.gamma)

            retain_logits = model(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"]).logits
            retain_ce = answer_ce_loss(retain_logits, retain_batch["labels"])

            kl = torch.zeros((), device=device)
            if teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(input_ids=retain_batch["input_ids"], attention_mask=retain_batch["attention_mask"]).logits
                kl = retain_kl_loss(retain_logits, teacher_logits, retain_batch["labels"], args.kl_temperature)

            total = args.simnpo_weight * simnpo.loss + args.retain_weight * retain_ce.loss + args.retain_kl_weight * kl
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip) if args.grad_clip and args.grad_clip > 0 else torch.zeros(())
            optimizer.step()

            last_row = {
                "step": step,
                "dataset": args.dataset,
                "mode": args.mode,
                "total_loss": float(total.detach().cpu()),
                "simnpo_loss": float(simnpo.loss.detach().cpu()),
                "retain_ce_loss": float(retain_ce.loss.detach().cpu()),
                "retain_kl_loss": float(kl.detach().cpu()),
                "policy_neg_logp_mean": simnpo.policy_neg_logp_mean,
                "mean_logp_per_token": simnpo.mean_logp_per_token,
                "answer_tokens_mean": simnpo.answer_tokens_mean,
                "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                "beta": args.beta,
                "gamma": args.gamma,
                "simnpo_weight": args.simnpo_weight,
                "retain_weight": args.retain_weight,
                "retain_kl_weight": args.retain_kl_weight,
                "n_selected_ids": len(selected_ids),
                "tied_embeddings": bool(train_info.get("tied")),
                "lr": args.lr,
            }
            log_f.write(json.dumps(last_row) + "\n")
            log_f.flush()

    checkpoint = output_dir / "checkpoint"
    if args.save_model:
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)

    train_summary = {
        "args": vars(args),
        "n_forget": len(forget_examples),
        "n_retain": len(retain_examples),
        "n_selected_ids": len(selected_ids),
        "tied_embeddings": bool(train_info.get("tied")),
        "param_summary": asdict(param_summary),
        "last_step": last_row,
    }
    write_json(output_dir / "train_summary.json", train_summary)

    if not args.skip_eval and args.save_model:
        if args.dataset == "mcf":
            run_mcf_official_eval(args, checkpoint)
        else:
            run_tofu_eval(args, checkpoint)

    print(f"cat {output_dir / 'train_summary.json'}")
    print(f"tail -n 5 {output_dir / 'train_log.jsonl'}")
    if args.dataset == "mcf":
        print(f"cat {output_dir / 'official_eval_comparison.md'}")
    else:
        print(f"cat {output_dir / 'eval' / 'tofu_eco_summary.csv'}")


if __name__ == "__main__":
    main()
