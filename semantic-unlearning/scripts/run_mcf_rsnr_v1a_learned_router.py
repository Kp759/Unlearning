#!/usr/bin/env python3
"""Practical routing experiment for the frozen RSNR-V1A-PreHead adapter.

The null adapter is NEVER retrained here.  This experiment replaces only the
oracle membership gate with a low-capacity learned semantic router

    g_phi(q) = 1[sigmoid(w^T normalize(mean(H_base(q))) + b) >= tau].

The router is fit on the existing leakage-safe five-view corpus and
``protection_fit`` rewrite prompts.  Official CounterFact rewrite/paraphrase/
neighborhood probes are evaluation-only.  The decision threshold is calibrated
on training-safe held-out views, never on official probes.

Outputs include router precision/recall/F1, group-specific false-positive rates,
hard-negative locality, exact gate-off Base identity, false-positive logit drift,
and end-to-end Eq.-16-style residual-sensitive-answer likelihood for Base,
oracle RSNR, and learned-router RSNR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mcf_zero_unlearn_official_eval as official_eval
import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead
from rsnr_v1a_frozen_spec import FROZEN_SPEC_VERSION, frozen_spec, validate_adapter_checkpoint


PROTOCOL = "mcf_rsnr_v1a_prehead_learned_semantic_router_v1"
SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
ROUTER_STEPS = 1000
ROUTER_LR = 1e-2
ROUTER_WEIGHT_DECAY = 1e-4
MIN_CALIBRATION_RECALL = 0.98
MAX_ROUTER_LENGTH = 256
TRAIN_SUBJECT_ONLY_TEMPLATE = "Give general information about {subject}."
CALIB_SUBJECT_ONLY_TEMPLATE = "Summarize what is known about {subject}."
TEST_SUBJECT_ONLY_TEMPLATE = "What can you tell me about {subject}?"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--adapter-checkpoint", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--forget-num", type=int, default=FORGET_NUM)
    p.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--router-batch-size", type=int, default=16)
    p.add_argument("--score-batch-size", type=int, default=8)
    p.add_argument("--negative-limit", type=int, default=2000)
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != SEED or args.forget_num != FORGET_NUM or args.retain_num != RETAIN_NUM:
        p.error("learned-router v1 is locked to seed1, forget50, retain1000")
    if args.router_batch_size <= 0 or args.score_batch_size <= 0 or args.negative_limit <= 0:
        p.error("batch sizes and negative-limit must be positive")
    return args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_state_digest(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def stable_bucket(text: str, buckets: int = 5) -> int:
    value = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") % int(buckets)


def dedupe(texts: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(x) for x in texts if str(x).strip()))


def render_rewrite(row: Mapping[str, Any]) -> str:
    rr = row["requested_rewrite"]
    if isinstance(rr, list):
        rr = rr[0]
    return str(rr["prompt"]).format(str(rr["subject"]))


def record_key(row: Mapping[str, Any]) -> tuple[str, str]:
    rr = row["requested_rewrite"]
    if isinstance(rr, list):
        rr = rr[0]
    return str(rr["subject"]), str(rr["relation_id"])


def split_training_safe_positive_views(
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
) -> tuple[list[str], list[str]]:
    train: list[str] = []
    calib: list[str] = []
    for row in forget:
        cid = int(row["case_id"])
        rr = row["requested_rewrite"]
        subject = str(rr["subject"])
        templates = list(view_map[cid])
        if len(templates) != 5:
            raise RuntimeError(f"case_id={cid}: expected exactly five frozen router-safe views")
        calib_index = cid % len(templates)
        for i, template in enumerate(templates):
            prompt = str(template).format(subject)
            (calib if i == calib_index else train).append(prompt)
    return dedupe(train), dedupe(calib)


def choose_protection_records(
    protection_fit: Sequence[Mapping[str, Any]],
    forget: Sequence[Mapping[str, Any]],
    limit: int,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    forget_pairs = {record_key(x) for x in forget}
    forget_subjects = {x[0] for x in forget_pairs}
    forget_relations = {x[1] for x in forget_pairs}
    hard_subject: list[Mapping[str, Any]] = []
    hard_relation: list[Mapping[str, Any]] = []
    other: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for row in protection_fit:
        cid = int(row["case_id"])
        if cid in seen or record_key(row) in forget_pairs:
            continue
        seen.add(cid)
        subject, relation = record_key(row)
        if subject in forget_subjects:
            hard_subject.append(row)
        elif relation in forget_relations:
            hard_relation.append(row)
        else:
            other.append(row)
    ordered = hard_subject + hard_relation + other
    selected = ordered[: int(limit)]
    stats = {
        "selected": len(selected),
        "same_subject_different_relation_available": len(hard_subject),
        "same_relation_different_subject_available": len(hard_relation),
        "other_available": len(other),
    }
    return selected, stats


def split_negative_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train: list[Mapping[str, Any]] = []
    calib: list[Mapping[str, Any]] = []
    for row in records:
        key = f"{SEED}:{int(row['case_id'])}:{render_rewrite(row)}"
        (calib if stable_bucket(key, 5) == 0 else train).append(row)
    if not calib and train:
        calib.append(train.pop())
    if not train and calib:
        train.append(calib.pop())
    return train, calib


def build_router_corpus(
    forget: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    *,
    negative_limit: int,
) -> tuple[list[str], list[int], list[str], list[int], dict[str, Any]]:
    pos_train, pos_calib = split_training_safe_positive_views(forget, view_map)
    selected_neg, negative_stats = choose_protection_records(
        protection_fit, forget, negative_limit
    )
    neg_train_rows, neg_calib_rows = split_negative_records(selected_neg)
    neg_train = [render_rewrite(x) for x in neg_train_rows]
    neg_calib = [render_rewrite(x) for x in neg_calib_rows]

    subjects = sorted({record_key(x)[0] for x in forget})
    neg_train.extend(TRAIN_SUBJECT_ONLY_TEMPLATE.format(subject=s) for s in subjects)
    neg_calib.extend(CALIB_SUBJECT_ONLY_TEMPLATE.format(subject=s) for s in subjects)

    train_texts = dedupe(pos_train + neg_train)
    pos_train_set = set(pos_train)
    train_labels = [1 if x in pos_train_set else 0 for x in train_texts]

    calib_texts = dedupe(pos_calib + neg_calib)
    pos_calib_set = set(pos_calib)
    calib_labels = [1 if x in pos_calib_set else 0 for x in calib_texts]

    meta = {
        "positive_train": sum(train_labels),
        "negative_train": len(train_labels) - sum(train_labels),
        "positive_calibration": sum(calib_labels),
        "negative_calibration": len(calib_labels) - sum(calib_labels),
        "positive_source": "four of five leakage-safe V1A views per forget fact",
        "calibration_positive_source": "one deterministic leakage-safe V1A view per forget fact",
        "negative_source": "protection_fit rewrite prompts plus subject-only negatives",
        "official_probe_text_used_for_router_fit": False,
        "official_probe_text_used_for_threshold_calibration": False,
        **negative_stats,
    }
    return train_texts, train_labels, calib_texts, calib_labels, meta


class LinearSemanticRouter(nn.Module):
    """Low-capacity semantic probe over frozen Base query embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(int(dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


@torch.no_grad()
def encode_prompts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Mean-pool frozen Base final hidden states and L2-normalize."""
    if not texts:
        hidden = int(getattr(model.config, "hidden_size"))
        return torch.empty((0, hidden), dtype=torch.float32)
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("learned-router v1 currently requires a causal LM exposing model.model")
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), int(batch_size)):
        batch = list(texts[start : start + int(batch_size)])
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_ROUTER_LENGTH,
            return_tensors="pt",
            add_special_tokens=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = backbone(**encoded, use_cache=False, return_dict=True)
        hidden = outputs.last_hidden_state.float()
        mask = encoded["attention_mask"].to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled = F.normalize(pooled, p=2, dim=-1)
        chunks.append(pooled.cpu())
    return torch.cat(chunks, dim=0)


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("labels/predictions length mismatch")
    tp = sum(int(y == 1 and p == 1) for y, p in zip(labels, predictions))
    fp = sum(int(y == 0 and p == 1) for y, p in zip(labels, predictions))
    tn = sum(int(y == 0 and p == 0) for y, p in zip(labels, predictions))
    fn = sum(int(y == 1 and p == 0) for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "count": len(labels),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "specificity": specificity,
        "accuracy": (tp + tn) / len(labels) if labels else 1.0,
    }


def choose_threshold(
    probs: Sequence[float],
    labels: Sequence[int],
    *,
    minimum_recall: float = MIN_CALIBRATION_RECALL,
) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set([0.0, 1.0] + [float(x) for x in probs]))
    scored: list[tuple[float, dict[str, Any]]] = []
    for threshold in candidates:
        pred = [int(float(p) >= threshold) for p in probs]
        metrics = binary_metrics(labels, pred)
        scored.append((threshold, metrics))
    feasible = [(t, m) for t, m in scored if float(m["recall"]) >= float(minimum_recall)]
    if not feasible:
        raise RuntimeError("no calibration threshold satisfies minimum recall")
    # Safety-oriented calibration: preserve recall first, then minimize false
    # positives. Ties favor precision/F1 and finally the larger threshold.
    threshold, metrics = min(
        feasible,
        key=lambda x: (
            float(x[1]["false_positive_rate"]),
            -float(x[1]["precision"]),
            -float(x[1]["f1"]),
            -float(x[0]),
        ),
    )
    return float(threshold), {
        **metrics,
        "threshold": float(threshold),
        "minimum_required_recall": float(minimum_recall),
        "selection_rule": "recall>=0.98 then minimize FPR; ties maximize precision/F1/threshold",
    }


def train_router(
    train_x: torch.Tensor,
    train_y: Sequence[int],
    calib_x: torch.Tensor,
    calib_y: Sequence[int],
    *,
    device: torch.device,
) -> tuple[LinearSemanticRouter, float, dict[str, Any]]:
    torch.manual_seed(SEED + 9107)
    router = LinearSemanticRouter(train_x.shape[1]).to(device)
    x = train_x.to(device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    positives = max(1, int(y.sum().item()))
    negatives = max(1, int(y.numel() - y.sum().item()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=ROUTER_LR, weight_decay=ROUTER_WEIGHT_DECAY
    )
    losses: list[float] = []
    router.train()
    for step in range(1, ROUTER_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = router(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == ROUTER_STEPS:
            losses.append(float(loss.detach().item()))
    router.eval()
    with torch.no_grad():
        calib_probs = torch.sigmoid(router(calib_x.to(device))).cpu().tolist()
    threshold, calib_metrics = choose_threshold(calib_probs, calib_y)
    return router, threshold, {
        "router_type": "linear_logistic_probe",
        "embedding": "L2-normalized mean of frozen Base final hidden states",
        "steps": ROUTER_STEPS,
        "learning_rate": ROUTER_LR,
        "weight_decay": ROUTER_WEIGHT_DECAY,
        "class_positive_weight": negatives / positives,
        "loss_trace_every_100_steps": losses,
        "calibration": calib_metrics,
    }


@torch.no_grad()
def router_probabilities(
    model: Any,
    tokenizer: Any,
    router: LinearSemanticRouter,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    if not texts:
        return []
    x = encode_prompts(model, tokenizer, texts, device=device, batch_size=batch_size)
    return torch.sigmoid(router(x.to(device))).cpu().tolist()


def group_router_report(probs: Sequence[float], threshold: float, label: int) -> dict[str, Any]:
    predictions = [int(float(p) >= float(threshold)) for p in probs]
    labels = [int(label)] * len(predictions)
    report = binary_metrics(labels, predictions)
    report.update({
        "activation_rate": float(np.mean(predictions)) if predictions else 0.0,
        "probability_mean": float(np.mean(probs)) if probs else None,
        "probability_min": float(np.min(probs)) if probs else None,
        "probability_max": float(np.max(probs)) if probs else None,
    })
    return report


def collect_test_groups(
    forget: Sequence[Mapping[str, Any]],
    retain: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    protection_ids = {int(x["case_id"]) for x in protection_fit}
    heldout_retain = [x for x in retain if int(x["case_id"]) not in protection_ids]
    if not heldout_retain:
        raise RuntimeError("no official retain examples remain after excluding protection_fit")

    forget_pairs = {record_key(x) for x in forget}
    forget_subjects = {x[0] for x in forget_pairs}
    forget_relations = {x[1] for x in forget_pairs}

    groups: dict[str, list[str]] = {
        "forget_rewrite": [render_rewrite(x) for x in forget],
        "forget_paraphrase": [
            str(prompt) for x in forget for prompt in x.get("paraphrase_prompts", [])
        ],
        "forget_neighborhood": [
            str(prompt) for x in forget for prompt in x.get("neighborhood_prompts", [])
        ],
        "retain_rewrite_heldout": [render_rewrite(x) for x in heldout_retain],
        "retain_paraphrase_heldout": [
            str(prompt) for x in heldout_retain for prompt in x.get("paraphrase_prompts", [])
        ],
        "retain_neighborhood_heldout": [
            str(prompt) for x in heldout_retain for prompt in x.get("neighborhood_prompts", [])
        ],
        "subject_only_novel": [
            TEST_SUBJECT_ONLY_TEMPLATE.format(subject=s) for s in sorted(forget_subjects)
        ],
        "same_subject_different_relation_heldout": [
            render_rewrite(x)
            for x in heldout_retain
            if record_key(x)[0] in forget_subjects and record_key(x) not in forget_pairs
        ],
        "same_relation_different_subject_heldout": [
            render_rewrite(x)
            for x in heldout_retain
            if record_key(x)[1] in forget_relations
            and record_key(x)[0] not in forget_subjects
        ],
    }
    groups = {k: dedupe(v) for k, v in groups.items()}
    meta = {
        "official_retain_requested": len(retain),
        "protection_fit_overlap_with_official_retain": len(retain) - len(heldout_retain),
        "official_retain_heldout_cases": len(heldout_retain),
        "heldout_retain_excludes_all_protection_fit_case_ids": True,
    }
    return groups, meta


def _sequence_logprobs_with_gates(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    gates: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if not (len(prompts) == len(answers) == len(gates)):
        raise ValueError("prompt/answer/gate length mismatch")
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(prompts), int(batch_size)):
            p = prompts[start : start + int(batch_size)]
            a = answers[start : start + int(batch_size)]
            g = gates[start : start + int(batch_size)]
            input_ids, attention, starts, targets, positions = rsnr.encode_prompt_answer_batch(
                tokenizer, p, a, device=device
            )
            gate_tensor = torch.tensor(g, dtype=torch.float32, device=device)
            hook.set(gate_tensor, positions)
            try:
                logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            finally:
                hook.clear()
            log_probs = F.log_softmax(logits, dim=-1)
            for i, (answer_start, aids) in enumerate(zip(starts, targets)):
                pos = torch.arange(
                    answer_start - 1,
                    answer_start - 1 + len(aids),
                    device=device,
                    dtype=torch.long,
                )
                tok = torch.tensor(aids, device=device, dtype=torch.long)
                values.append(log_probs[i, pos, tok].mean().cpu())
    return torch.stack(values) if values else torch.empty(0)


def eq16_proxy(logprobs: torch.Tensor) -> float:
    return float(logprobs.float().exp().mean().item() * 100.0) if logprobs.numel() else float("nan")


def end_to_end_sensitive_metrics(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    rewrite_gates: Sequence[int],
    paraphrase_gates: Sequence[int],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    rewrite_prompts = [render_rewrite(x) for x in forget]
    rewrite_answers = [str(x["requested_rewrite"]["target_true"]["str"]) for x in forget]
    paraphrase_prompts: list[str] = []
    paraphrase_answers: list[str] = []
    for row in forget:
        answer = str(row["requested_rewrite"]["target_true"]["str"])
        for prompt in row.get("paraphrase_prompts", []):
            paraphrase_prompts.append(str(prompt))
            paraphrase_answers.append(answer)

    conditions = {
        "Base_gate_off": ([0] * len(rewrite_prompts), [0] * len(paraphrase_prompts)),
        "RSNR_oracle_gate": ([1] * len(rewrite_prompts), [1] * len(paraphrase_prompts)),
        "RSNR_learned_router": (list(rewrite_gates), list(paraphrase_gates)),
    }
    out: dict[str, Any] = {}
    for name, (rw_gates, pp_gates) in conditions.items():
        rw_true = _sequence_logprobs_with_gates(
            model, hook, tokenizer, rewrite_prompts, rewrite_answers, rw_gates,
            device=device, batch_size=batch_size
        )
        pp_true = _sequence_logprobs_with_gates(
            model, hook, tokenizer, paraphrase_prompts, paraphrase_answers, pp_gates,
            device=device, batch_size=batch_size
        )
        out[name] = {
            "Eq16_style_Eff": eq16_proxy(rw_true),
            "Eq16_style_Gen": eq16_proxy(pp_true),
            "rewrite_count": len(rewrite_prompts),
            "paraphrase_count": len(paraphrase_prompts),
        }

    # Natural-abstention diagnostic for the practical gate.
    rewrite_idk = [rsnr.ABSTENTION] * len(rewrite_prompts)
    paraphrase_idk = [rsnr.ABSTENTION] * len(paraphrase_prompts)
    learned_rw_true = _sequence_logprobs_with_gates(
        model, hook, tokenizer, rewrite_prompts, rewrite_answers, rewrite_gates,
        device=device, batch_size=batch_size
    )
    learned_rw_idk = _sequence_logprobs_with_gates(
        model, hook, tokenizer, rewrite_prompts, rewrite_idk, rewrite_gates,
        device=device, batch_size=batch_size
    )
    learned_pp_true = _sequence_logprobs_with_gates(
        model, hook, tokenizer, paraphrase_prompts, paraphrase_answers, paraphrase_gates,
        device=device, batch_size=batch_size
    )
    learned_pp_idk = _sequence_logprobs_with_gates(
        model, hook, tokenizer, paraphrase_prompts, paraphrase_idk, paraphrase_gates,
        device=device, batch_size=batch_size
    )
    out["RSNR_learned_router"]["rewrite_abstain_minus_true_mean"] = float(
        (learned_rw_idk - learned_rw_true).mean().item()
    )
    out["RSNR_learned_router"]["rewrite_abstain_minus_true_min"] = float(
        (learned_rw_idk - learned_rw_true).min().item()
    )
    out["RSNR_learned_router"]["paraphrase_abstain_minus_true_mean"] = float(
        (learned_pp_idk - learned_pp_true).mean().item()
    )
    out["RSNR_learned_router"]["paraphrase_abstain_minus_true_min"] = float(
        (learned_pp_idk - learned_pp_true).min().item()
    )
    return out


@torch.no_grad()
def false_positive_next_token_drift(
    model: Any,
    hook: prehead.PreHeadNullHook,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    device: torch.device,
    limit: int = 32,
) -> dict[str, Any]:
    maxima: list[float] = []
    means: list[float] = []
    for text in list(prompts)[: int(limit)]:
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        hook.clear()
        base_logits = model(**encoded).logits.float()
        positions = torch.zeros_like(encoded["attention_mask"], dtype=torch.float32)
        last = int(encoded["attention_mask"][0].sum().item()) - 1
        positions[0, last] = 1.0
        hook.set(torch.ones(1, device=device), positions)
        try:
            routed_logits = model(**encoded).logits.float()
        finally:
            hook.clear()
        diff = (routed_logits[0, last] - base_logits[0, last]).abs()
        maxima.append(float(diff.max().item()))
        means.append(float(diff.mean().item()))
    return {
        "count": len(maxima),
        "max_abs_next_token_logit_drift": max(maxima) if maxima else 0.0,
        "mean_abs_next_token_logit_drift": float(np.mean(means)) if means else 0.0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter_path = Path(args.adapter_checkpoint).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    adapter_file_hash_before = sha256_file(adapter_path)

    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
    validate_adapter_checkpoint(checkpoint)

    protocol = rsnr.load_protocol(Path(args.protocol_dir), FORGET_NUM)
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]
    view_map, view_meta = rsnr.load_training_views(Path(args.view_corpus))
    rsnr.validate_case_alignment(forget, view_map)

    all_mcf = official_eval.load_mcf(mcf_path)
    official_forget, official_retain = official_eval.sample_official_split(
        all_mcf, FORGET_NUM, RETAIN_NUM, SEED
    )
    protocol_ids = [int(x["case_id"]) for x in forget]
    official_ids = [int(x["case_id"]) for x in official_forget]
    if protocol_ids != official_ids:
        raise RuntimeError("protocol forget cases do not match official seed-1 MCF forget split")
    forget = official_forget

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = base_runner.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    hidden_size = int(getattr(model.config, "hidden_size"))
    if hidden_size != int(checkpoint["hidden_size"]):
        raise RuntimeError(
            f"model/checkpoint hidden-size mismatch: {hidden_size} != {checkpoint['hidden_size']}"
        )
    adapter = rsnr.NullResidualAdapter(
        hidden_size,
        int(checkpoint["adapter_rank"]),
        float(checkpoint["adapter_alpha"]),
        device,
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter_digest_before = tensor_state_digest(adapter)
    hook = prehead.PreHeadNullHook.install(prehead.get_lm_head(model), adapter)

    train_texts, train_labels, calib_texts, calib_labels, corpus_meta = build_router_corpus(
        forget, protection_fit, view_map, negative_limit=int(args.negative_limit)
    )
    print(json.dumps({
        "protocol": PROTOCOL,
        "frozen_spec": FROZEN_SPEC_VERSION,
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_sha256": adapter_file_hash_before,
        "router_corpus": corpus_meta,
        "official_probes_used_for_router_training": False,
    }, indent=2), flush=True)

    train_x = encode_prompts(
        model, tokenizer, train_texts,
        device=device, batch_size=int(args.router_batch_size)
    )
    calib_x = encode_prompts(
        model, tokenizer, calib_texts,
        device=device, batch_size=int(args.router_batch_size)
    )
    router, threshold, router_training = train_router(
        train_x, train_labels, calib_x, calib_labels, device=device
    )

    groups, test_meta = collect_test_groups(forget, official_retain, protection_fit)
    positive_groups = {"forget_rewrite", "forget_paraphrase"}
    group_probs: dict[str, list[float]] = {}
    group_reports: dict[str, Any] = {}
    for name, texts in groups.items():
        probs = router_probabilities(
            model, tokenizer, router, texts,
            device=device, batch_size=int(args.router_batch_size)
        )
        group_probs[name] = probs
        group_reports[name] = group_router_report(
            probs, threshold, 1 if name in positive_groups else 0
        )

    rewrite_gates = [int(p >= threshold) for p in group_probs["forget_rewrite"]]
    paraphrase_gates = [int(p >= threshold) for p in group_probs["forget_paraphrase"]]

    # Overall held-out router precision/recall combines official forget positives
    # with all official/novel negative groups, but excludes hard-negative subsets
    # that duplicate retain_rewrite_heldout.
    overall_probs = (
        group_probs["forget_rewrite"]
        + group_probs["forget_paraphrase"]
        + group_probs["forget_neighborhood"]
        + group_probs["retain_rewrite_heldout"]
        + group_probs["retain_paraphrase_heldout"]
        + group_probs["retain_neighborhood_heldout"]
        + group_probs["subject_only_novel"]
    )
    overall_labels = (
        [1] * (len(group_probs["forget_rewrite"]) + len(group_probs["forget_paraphrase"]))
        + [0] * (
            len(group_probs["forget_neighborhood"])
            + len(group_probs["retain_rewrite_heldout"])
            + len(group_probs["retain_paraphrase_heldout"])
            + len(group_probs["retain_neighborhood_heldout"])
            + len(group_probs["subject_only_novel"])
        )
    )
    overall_pred = [int(p >= threshold) for p in overall_probs]
    overall_metrics = binary_metrics(overall_labels, overall_pred)

    # Paraphrase case-level all-views recall.
    para_cursor = 0
    para_case_all: list[int] = []
    para_case_any: list[int] = []
    for row in forget:
        n = len(row.get("paraphrase_prompts", []))
        gates = paraphrase_gates[para_cursor : para_cursor + n]
        para_cursor += n
        para_case_all.append(int(bool(gates) and all(gates)))
        para_case_any.append(int(any(gates)))

    end_to_end = end_to_end_sensitive_metrics(
        model, hook, tokenizer, forget, rewrite_gates, paraphrase_gates,
        device=device, batch_size=int(args.score_batch_size)
    )

    negative_named_groups = [
        "forget_neighborhood",
        "retain_rewrite_heldout",
        "retain_paraphrase_heldout",
        "retain_neighborhood_heldout",
        "subject_only_novel",
    ]
    false_positive_prompts: list[str] = []
    correctly_rejected_prompts: list[str] = []
    for name in negative_named_groups:
        for text, prob in zip(groups[name], group_probs[name]):
            if prob >= threshold:
                false_positive_prompts.append(text)
            else:
                correctly_rejected_prompts.append(text)
    fp_drift = false_positive_next_token_drift(
        model, hook, tokenizer, false_positive_prompts, device=device, limit=32
    )
    gate_off_audit = rsnr.gate_off_equivalence(
        model, hook, tokenizer, correctly_rejected_prompts[:64], device=device
    )
    if float(gate_off_audit["max_abs_logit_drift"]) != 0.0:
        raise RuntimeError("frozen gate-off path lost exact Base identity")

    adapter_digest_after = tensor_state_digest(adapter)
    adapter_file_hash_after = sha256_file(adapter_path)
    if adapter_digest_after != adapter_digest_before:
        raise RuntimeError("frozen RSNR adapter tensors changed during router experiment")
    if adapter_file_hash_after != adapter_file_hash_before:
        raise RuntimeError("frozen RSNR adapter checkpoint file changed during router experiment")

    router_checkpoint = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "frozen_rsnr_spec_version": FROZEN_SPEC_VERSION,
        "router_type": "linear_logistic_probe",
        "embedding": "normalized_mean_base_final_hidden",
        "hidden_size": hidden_size,
        "threshold": threshold,
        "state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "training": router_training,
        "router_corpus": corpus_meta,
        "adapter_checkpoint_sha256": adapter_file_hash_before,
    }
    torch.save(router_checkpoint, output_dir / "learned_semantic_router.pt")

    report = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "frozen_rsnr": frozen_spec(),
        "architecture_freeze_enforced": True,
        "adapter_retrained": False,
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_sha256_before": adapter_file_hash_before,
        "adapter_checkpoint_sha256_after": adapter_file_hash_after,
        "adapter_tensor_digest_before": adapter_digest_before,
        "adapter_tensor_digest_after": adapter_digest_after,
        "base_model_parameters_trainable": 0,
        "adapter_parameters_trainable": 0,
        "router_parameters_trainable": sum(p.numel() for p in router.parameters()),
        "router_training": router_training,
        "router_corpus": corpus_meta,
        "test_split": test_meta,
        "threshold": threshold,
        "router_overall_heldout": overall_metrics,
        "router_groups": group_reports,
        "paraphrase_case_level": {
            "all_paraphrases_routed_recall": float(np.mean(para_case_all)) if para_case_all else None,
            "any_paraphrase_routed_recall": float(np.mean(para_case_any)) if para_case_any else None,
        },
        "end_to_end_sensitive_metrics": end_to_end,
        "false_positive_locality": {
            "false_positive_count": len(false_positive_prompts),
            "correctly_rejected_negative_count": len(correctly_rejected_prompts),
            "correctly_rejected_gate_off_equivalence": gate_off_audit,
            "false_positive_next_token_drift_sample": fp_drift,
            "structural_statement": (
                "Every router-negative query exactly follows Base; only router false positives "
                "can incur adapter-induced drift."
            ),
        },
        "claim_boundary": {
            "router_is_learned": True,
            "router_uses_official_probe_text_for_fit": False,
            "null_adapter_changed": False,
            "base_model_changed": False,
            "exact_base_identity_when_router_off": True,
            "latent_knowledge_erasure_claimed": False,
        },
    }
    (output_dir / "learned_router_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = {
        "router_precision": overall_metrics["precision"],
        "router_recall": overall_metrics["recall"],
        "router_f1": overall_metrics["f1"],
        "router_false_positive_rate": overall_metrics["false_positive_rate"],
        "rewrite_recall": group_reports["forget_rewrite"]["recall"],
        "paraphrase_recall": group_reports["forget_paraphrase"]["recall"],
        "forget_neighborhood_fpr": group_reports["forget_neighborhood"]["false_positive_rate"],
        "subject_only_fpr": group_reports["subject_only_novel"]["false_positive_rate"],
        "same_subject_other_relation_fpr": group_reports["same_subject_different_relation_heldout"]["false_positive_rate"],
        "same_relation_other_subject_fpr": group_reports["same_relation_different_subject_heldout"]["false_positive_rate"],
        "Base_Eff": end_to_end["Base_gate_off"]["Eq16_style_Eff"],
        "Base_Gen": end_to_end["Base_gate_off"]["Eq16_style_Gen"],
        "Oracle_Eff": end_to_end["RSNR_oracle_gate"]["Eq16_style_Eff"],
        "Oracle_Gen": end_to_end["RSNR_oracle_gate"]["Eq16_style_Gen"],
        "LearnedRouter_Eff": end_to_end["RSNR_learned_router"]["Eq16_style_Eff"],
        "LearnedRouter_Gen": end_to_end["RSNR_learned_router"]["Eq16_style_Gen"],
        "gate_off_max_abs_logit_drift": gate_off_audit["max_abs_logit_drift"],
        "report": str(output_dir / "learned_router_report.json"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    hook.remove()


if __name__ == "__main__":
    main()
