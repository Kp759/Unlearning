#!/usr/bin/env python3
"""RSNR Router V2: fact-conditioned relation-residual routing for frozen V1A.

The RSNR-V1A-PreHead null adapter and Base model are immutable in this
experiment.  Only a tiny diagonal relation-matching probe is trained.

Router V1 showed that a global linear probe could recognize the exact rewrite
form but over-relied on subject identity (high subject-only FPR) and failed to
generalize to official paraphrases.  V2 therefore factorizes routing into:

  1. deterministic candidate subject detection over the frozen forget set;
  2. relation-residual semantic matching conditioned on that subject.

For a candidate forgotten fact f=(s,r), let E(q) be the L2-normalized mean of
frozen Base final hidden states and a_s=E("General information about {s}.").
The subject-reduced relation representation is

    R(q,s) = normalize(E(q) - a_s).

A frozen prototype p_f is the normalized mean R(q_k,s) over four of the five
existing leakage-safe V1A views.  The learned score uses only elementwise
relation agreement:

    logit(q,f) = w^T (R(q,s) * p_f) + b

with 3072+1 trainable router parameters for Llama-3.2-3B.  The fifth V1A view
is reserved for threshold calibration.  Same-subject/different-relation hard
negatives are synthesized only from training-safe V1A templates by filling a
different relation's template with the candidate subject.  Official MCF
rewrite/paraphrase/neighborhood probes are evaluation-only.

This is a router experiment, not a new RSNR method variant: rank, alpha,
intervention site, null-adapter weights, Base weights, and unlearning objective
remain exactly frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
import run_mcf_rsnr_v1a_learned_router as v1
from rsnr_v1a_frozen_spec import FROZEN_SPEC_VERSION, frozen_spec, validate_adapter_checkpoint


PROTOCOL = "mcf_rsnr_v1a_prehead_relation_router_v2"
SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
ROUTER_STEPS = 1200
ROUTER_LR = 5e-2
ROUTER_WEIGHT_DECAY = 1e-4
MIN_CALIBRATION_RECALL = 0.98
SUBJECT_ANCHOR_TEMPLATE = "General information about {subject}."
TRAIN_SUBJECT_ONLY_TEMPLATES = (
    "Give general information about {subject}.",
    "Describe {subject} in general terms.",
)
CALIB_SUBJECT_ONLY_TEMPLATE = "Summarize what is known about {subject}."
TEST_SUBJECT_ONLY_TEMPLATE = "What can you tell me about {subject}?"
TRAIN_HARD_NEGATIVES_PER_FACT = 4
TEST_HARD_NEGATIVES_PER_FACT = 2


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
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != SEED or args.forget_num != FORGET_NUM or args.retain_num != RETAIN_NUM:
        p.error("relation-router v2 is a seed1 development experiment locked to forget50/retain1000")
    if args.router_batch_size <= 0 or args.score_batch_size <= 0:
        p.error("batch sizes must be positive")
    return args


def normalize_match_text(text: str) -> str:
    """Conservative text normalization for exact subject candidate retrieval."""
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def subject_is_mentioned(text: str, subject: str) -> bool:
    return normalize_match_text(subject) in normalize_match_text(text)


def _rr(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def fact_info(row: Mapping[str, Any]) -> dict[str, Any]:
    rr = _rr(row)
    return {
        "case_id": int(row["case_id"]),
        "subject": str(rr["subject"]),
        "relation_id": str(rr["relation_id"]),
    }


def _view_prompt(template: str, subject: str) -> str:
    return str(template).format(subject)


def split_fact_views(
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
) -> dict[int, dict[str, Any]]:
    """Return four training-safe views + one deterministic calibration view per fact."""
    out: dict[int, dict[str, Any]] = {}
    for row in forget:
        info = fact_info(row)
        cid = info["case_id"]
        templates = list(view_map[cid])
        if len(templates) != 5:
            raise RuntimeError(f"case_id={cid}: expected exactly five frozen V1A views")
        calib_index = cid % 5
        train_templates = [str(x) for i, x in enumerate(templates) if i != calib_index]
        out[cid] = {
            **info,
            "train_templates": train_templates,
            "calib_template": str(templates[calib_index]),
            "train_prompts": [_view_prompt(x, info["subject"]) for x in train_templates],
            "calib_prompt": _view_prompt(templates[calib_index], info["subject"]),
        }
    return out


def _other_relation_cases(
    facts: Mapping[int, Mapping[str, Any]],
    target_cid: int,
) -> list[int]:
    target_rel = str(facts[target_cid]["relation_id"])
    return sorted(
        cid for cid, info in facts.items()
        if cid != target_cid and str(info["relation_id"]) != target_rel
    )


def synthetic_hard_negative_prompts(
    facts: Mapping[int, Mapping[str, Any]],
    target_cid: int,
    *,
    count: int,
    calibration: bool,
) -> list[str]:
    """Same candidate subject, but wording from another training-safe relation."""
    target = facts[target_cid]
    choices = _other_relation_cases(facts, target_cid)
    if not choices:
        raise RuntimeError("cannot construct different-relation hard negatives")
    seed_material = hashlib.sha256(f"{SEED}:{target_cid}:{int(calibration)}".encode()).digest()
    offset = int.from_bytes(seed_material[:4], "big") % len(choices)
    prompts: list[str] = []
    for j in range(int(count)):
        other_cid = choices[(offset + j) % len(choices)]
        other = facts[other_cid]
        if calibration:
            template = str(other["calib_template"])
        else:
            templates = list(other["train_templates"])
            template = str(templates[j % len(templates)])
        prompts.append(_view_prompt(template, str(target["subject"])))
    return v1.dedupe(prompts)


def build_pair_corpus(
    facts: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build candidate-fact pairs without any official MCF probe text."""
    train: list[dict[str, Any]] = []
    calib: list[dict[str, Any]] = []
    for cid in sorted(facts):
        info = facts[cid]
        for text in info["train_prompts"]:
            train.append({"text": text, "case_id": cid, "label": 1, "kind": "positive_safe_view"})
        calib.append({
            "text": info["calib_prompt"], "case_id": cid, "label": 1,
            "kind": "positive_safe_view_calibration",
        })

        for text in synthetic_hard_negative_prompts(
            facts, cid, count=TRAIN_HARD_NEGATIVES_PER_FACT, calibration=False
        ):
            train.append({
                "text": text, "case_id": cid, "label": 0,
                "kind": "same_subject_different_relation_synthetic_train",
            })
        for template in TRAIN_SUBJECT_ONLY_TEMPLATES:
            train.append({
                "text": template.format(subject=info["subject"]),
                "case_id": cid, "label": 0, "kind": "subject_only_train",
            })

        hard_calib = synthetic_hard_negative_prompts(
            facts, cid, count=1, calibration=True
        )
        for text in hard_calib:
            calib.append({
                "text": text, "case_id": cid, "label": 0,
                "kind": "same_subject_different_relation_synthetic_calibration",
            })
        calib.append({
            "text": CALIB_SUBJECT_ONLY_TEMPLATE.format(subject=info["subject"]),
            "case_id": cid, "label": 0, "kind": "subject_only_calibration",
        })

    def counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in rows:
            key = str(row["kind"])
            out[key] = out.get(key, 0) + 1
        return out

    meta = {
        "train_count": len(train),
        "calibration_count": len(calib),
        "train_kind_counts": counts(train),
        "calibration_kind_counts": counts(calib),
        "official_probe_text_used_for_router_fit": False,
        "official_probe_text_used_for_threshold_calibration": False,
        "hard_negative_construction": (
            "same candidate subject inserted into another forget fact's training-safe relation template"
        ),
    }
    return train, calib, meta


class DiagonalRelationRouter(nn.Module):
    """Learn a diagonal metric over query/prototype relation agreement."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(int(dim), 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.zero_()

    def forward(self, pair_feature: torch.Tensor) -> torch.Tensor:
        return self.linear(pair_feature).squeeze(-1)


@torch.no_grad()
def encode_text_map(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    unique = v1.dedupe(texts)
    if not unique:
        return {}
    encoded = v1.encode_prompts(
        model, tokenizer, unique, device=device, batch_size=batch_size
    )
    return {text: encoded[i].float() for i, text in enumerate(unique)}


def build_relation_state(
    model: Any,
    tokenizer: Any,
    facts: Mapping[int, Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    """Compute frozen subject anchors and relation prototypes from safe views."""
    texts: list[str] = []
    anchor_text: dict[int, str] = {}
    for cid, info in facts.items():
        anchor = SUBJECT_ANCHOR_TEMPLATE.format(subject=info["subject"])
        anchor_text[cid] = anchor
        texts.append(anchor)
        texts.extend(info["train_prompts"])
    emap = encode_text_map(
        model, tokenizer, texts, device=device, batch_size=batch_size
    )

    anchors: dict[int, torch.Tensor] = {}
    prototypes: dict[int, torch.Tensor] = {}
    for cid, info in facts.items():
        anchor = emap[anchor_text[cid]].float()
        anchors[cid] = anchor
        residuals = [F.normalize((emap[text].float() - anchor), dim=0) for text in info["train_prompts"]]
        proto = F.normalize(torch.stack(residuals, dim=0).mean(dim=0), dim=0)
        prototypes[cid] = proto
    return anchors, prototypes, {
        "subject_anchor_template": SUBJECT_ANCHOR_TEMPLATE,
        "prototype_source": "mean relation residual of four leakage-safe V1A views",
        "prototype_trainable": False,
    }


def pair_feature(
    query_embedding: torch.Tensor,
    anchor: torch.Tensor,
    prototype: torch.Tensor,
) -> torch.Tensor:
    relation = F.normalize(query_embedding.float() - anchor.float(), dim=0)
    return relation * prototype.float()


def pair_features_for_rows(
    model: Any,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    anchors: Mapping[int, torch.Tensor],
    prototypes: Mapping[int, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, list[int]]:
    texts = [str(row["text"]) for row in rows]
    emap = encode_text_map(model, tokenizer, texts, device=device, batch_size=batch_size)
    features = [
        pair_feature(emap[str(row["text"])], anchors[int(row["case_id"])], prototypes[int(row["case_id"])])
        for row in rows
    ]
    labels = [int(row["label"]) for row in rows]
    return torch.stack(features, dim=0), labels


def train_relation_router(
    train_x: torch.Tensor,
    train_y: Sequence[int],
    calib_x: torch.Tensor,
    calib_y: Sequence[int],
    *,
    device: torch.device,
) -> tuple[DiagonalRelationRouter, float, dict[str, Any]]:
    torch.manual_seed(SEED + 22091)
    router = DiagonalRelationRouter(train_x.shape[1]).to(device)
    x = train_x.to(device)
    y = torch.tensor(train_y, dtype=torch.float32, device=device)
    positives = max(1, int(y.sum().item()))
    negatives = max(1, int(y.numel() - y.sum().item()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR, weight_decay=ROUTER_WEIGHT_DECAY)
    trace: list[dict[str, float | int]] = []
    router.train()
    for step in range(1, ROUTER_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = router(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == ROUTER_STEPS:
            trace.append({"step": step, "loss": float(loss.detach().item())})
    router.eval()
    with torch.no_grad():
        calib_probs = torch.sigmoid(router(calib_x.to(device))).cpu().tolist()
    threshold, calib_metrics = v1.choose_threshold(
        calib_probs, calib_y, minimum_recall=MIN_CALIBRATION_RECALL
    )
    return router, threshold, {
        "router_type": "subject_gated_diagonal_relation_metric",
        "trainable_parameters": sum(p.numel() for p in router.parameters()),
        "steps": ROUTER_STEPS,
        "learning_rate": ROUTER_LR,
        "weight_decay": ROUTER_WEIGHT_DECAY,
        "class_positive_weight": negatives / positives,
        "loss_trace": trace,
        "calibration": calib_metrics,
    }


def candidate_case_ids(text: str, facts: Mapping[int, Mapping[str, Any]]) -> list[int]:
    return [
        cid for cid, info in facts.items()
        if subject_is_mentioned(text, str(info["subject"]))
    ]


@torch.no_grad()
def score_texts(
    model: Any,
    tokenizer: Any,
    router: DiagonalRelationRouter,
    texts: Sequence[str],
    facts: Mapping[int, Mapping[str, Any]],
    anchors: Mapping[int, torch.Tensor],
    prototypes: Mapping[int, torch.Tensor],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[list[float], list[int]]:
    """Return max fact-conditioned probability and candidate count per query."""
    candidates = [candidate_case_ids(text, facts) for text in texts]
    needs_encoding = [text for text, cids in zip(texts, candidates) if cids]
    emap = encode_text_map(
        model, tokenizer, needs_encoding, device=device, batch_size=batch_size
    )
    probabilities: list[float] = []
    candidate_counts: list[int] = []
    for text, cids in zip(texts, candidates):
        candidate_counts.append(len(cids))
        if not cids:
            probabilities.append(0.0)
            continue
        q = emap[text]
        feats = torch.stack([
            pair_feature(q, anchors[cid], prototypes[cid]) for cid in cids
        ], dim=0).to(device)
        probs = torch.sigmoid(router(feats)).cpu()
        probabilities.append(float(probs.max().item()))
    return probabilities, candidate_counts


def synthetic_heldout_same_subject_other_relation(
    forget: Sequence[Mapping[str, Any]],
    heldout_retain: Sequence[Mapping[str, Any]],
    *,
    per_fact: int = TEST_HARD_NEGATIVES_PER_FACT,
) -> list[str]:
    """Evaluation-only hard negatives from official retain rewrite templates."""
    prompts: list[str] = []
    retain_sorted = sorted(heldout_retain, key=lambda x: int(x["case_id"]))
    for row in forget:
        info = fact_info(row)
        choices = [x for x in retain_sorted if fact_info(x)["relation_id"] != info["relation_id"]]
        if not choices:
            continue
        offset = int(info["case_id"]) % len(choices)
        for j in range(int(per_fact)):
            other = choices[(offset + j) % len(choices)]
            rr = _rr(other)
            prompts.append(str(rr["prompt"]).format(info["subject"]))
    return v1.dedupe(prompts)


def collect_v2_test_groups(
    forget: Sequence[Mapping[str, Any]],
    retain: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    groups, meta = v1.collect_test_groups(forget, retain, protection_fit)
    protection_ids = {int(x["case_id"]) for x in protection_fit}
    heldout_retain = [x for x in retain if int(x["case_id"]) not in protection_ids]
    groups["subject_only_novel"] = [
        TEST_SUBJECT_ONLY_TEMPLATE.format(subject=fact_info(x)["subject"]) for x in forget
    ]
    groups["same_subject_different_relation_heldout"] = synthetic_heldout_same_subject_other_relation(
        forget, heldout_retain
    )
    groups = {name: v1.dedupe(texts) for name, texts in groups.items()}
    meta.update({
        "synthetic_same_subject_different_relation_test_count": len(
            groups["same_subject_different_relation_heldout"]
        ),
        "synthetic_hard_negative_test_uses_official_retain_only_for_evaluation": True,
    })
    return groups, meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    adapter_path = Path(args.adapter_checkpoint).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    adapter_file_hash_before = v1.sha256_file(adapter_path)

    checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
    validate_adapter_checkpoint(checkpoint)

    protocol = rsnr.load_protocol(Path(args.protocol_dir), FORGET_NUM)
    forget_protocol = protocol["forget"]
    protection_fit = protocol["protection_fit"]
    view_map, view_meta = rsnr.load_training_views(Path(args.view_corpus))
    rsnr.validate_case_alignment(forget_protocol, view_map)

    all_mcf = official_eval.load_mcf(mcf_path)
    official_forget, official_retain = official_eval.sample_official_split(
        all_mcf, FORGET_NUM, RETAIN_NUM, SEED
    )
    if [int(x["case_id"]) for x in forget_protocol] != [int(x["case_id"]) for x in official_forget]:
        raise RuntimeError("protocol forget cases do not match official seed-1 MCF forget split")
    forget = official_forget
    facts = split_fact_views(forget, view_map)

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
        raise RuntimeError("model/checkpoint hidden-size mismatch")
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
    adapter_digest_before = v1.tensor_state_digest(adapter)
    hook = prehead.PreHeadNullHook.install(prehead.get_lm_head(model), adapter)

    anchors, prototypes, relation_meta = build_relation_state(
        model, tokenizer, facts, device=device, batch_size=int(args.router_batch_size)
    )
    train_rows, calib_rows, corpus_meta = build_pair_corpus(facts)
    train_x, train_y = pair_features_for_rows(
        model, tokenizer, train_rows, anchors, prototypes,
        device=device, batch_size=int(args.router_batch_size)
    )
    calib_x, calib_y = pair_features_for_rows(
        model, tokenizer, calib_rows, anchors, prototypes,
        device=device, batch_size=int(args.router_batch_size)
    )
    router, threshold, router_training = train_relation_router(
        train_x, train_y, calib_x, calib_y, device=device
    )

    print(json.dumps({
        "protocol": PROTOCOL,
        "frozen_spec": FROZEN_SPEC_VERSION,
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_sha256": adapter_file_hash_before,
        "router_architecture": router_training["router_type"],
        "router_parameters_trainable": router_training["trainable_parameters"],
        "router_corpus": corpus_meta,
        "relation_representation": relation_meta,
        "official_probes_used_for_router_training": False,
    }, indent=2), flush=True)

    groups, test_meta = collect_v2_test_groups(forget, official_retain, protection_fit)
    positive_groups = {"forget_rewrite", "forget_paraphrase"}
    group_probs: dict[str, list[float]] = {}
    group_candidate_counts: dict[str, list[int]] = {}
    group_reports: dict[str, Any] = {}
    for name, texts in groups.items():
        probs, candidate_counts = score_texts(
            model, tokenizer, router, texts, facts, anchors, prototypes,
            device=device, batch_size=int(args.router_batch_size)
        )
        group_probs[name] = probs
        group_candidate_counts[name] = candidate_counts
        report = v1.group_router_report(probs, threshold, 1 if name in positive_groups else 0)
        report["candidate_subject_coverage"] = (
            float(np.mean([int(x > 0) for x in candidate_counts])) if candidate_counts else 0.0
        )
        group_reports[name] = report

    rewrite_gates = [int(p >= threshold) for p in group_probs["forget_rewrite"]]
    paraphrase_gates = [int(p >= threshold) for p in group_probs["forget_paraphrase"]]

    overall_negative_names = [
        "forget_neighborhood",
        "retain_rewrite_heldout",
        "retain_paraphrase_heldout",
        "retain_neighborhood_heldout",
        "subject_only_novel",
        "same_subject_different_relation_heldout",
    ]
    overall_probs = group_probs["forget_rewrite"] + group_probs["forget_paraphrase"]
    overall_labels = [1] * len(overall_probs)
    for name in overall_negative_names:
        overall_probs += group_probs[name]
        overall_labels += [0] * len(group_probs[name])
    overall_pred = [int(p >= threshold) for p in overall_probs]
    overall_metrics = v1.binary_metrics(overall_labels, overall_pred)

    para_cursor = 0
    para_case_all: list[int] = []
    para_case_any: list[int] = []
    for row in forget:
        n = len(row.get("paraphrase_prompts", []))
        gates = paraphrase_gates[para_cursor:para_cursor + n]
        para_cursor += n
        para_case_all.append(int(bool(gates) and all(gates)))
        para_case_any.append(int(any(gates)))

    end_to_end = v1.end_to_end_sensitive_metrics(
        model, hook, tokenizer, forget, rewrite_gates, paraphrase_gates,
        device=device, batch_size=int(args.score_batch_size)
    )

    false_positive_prompts: list[str] = []
    correctly_rejected_prompts: list[str] = []
    for name in overall_negative_names:
        for text, prob in zip(groups[name], group_probs[name]):
            if prob >= threshold:
                false_positive_prompts.append(text)
            else:
                correctly_rejected_prompts.append(text)
    fp_drift = v1.false_positive_next_token_drift(
        model, hook, tokenizer, false_positive_prompts, device=device, limit=32
    )
    gate_off_audit = rsnr.gate_off_equivalence(
        model, hook, tokenizer, correctly_rejected_prompts[:64], device=device
    )
    if float(gate_off_audit["max_abs_logit_drift"]) != 0.0:
        raise RuntimeError("frozen gate-off path lost exact Base identity")

    adapter_digest_after = v1.tensor_state_digest(adapter)
    adapter_file_hash_after = v1.sha256_file(adapter_path)
    if adapter_digest_after != adapter_digest_before:
        raise RuntimeError("frozen RSNR adapter tensors changed during Router V2")
    if adapter_file_hash_after != adapter_file_hash_before:
        raise RuntimeError("frozen RSNR adapter checkpoint changed during Router V2")

    router_checkpoint = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "frozen_rsnr_spec_version": FROZEN_SPEC_VERSION,
        "router_type": router_training["router_type"],
        "hidden_size": hidden_size,
        "threshold": threshold,
        "state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "facts": {cid: {k: v for k, v in info.items() if k not in {"train_templates", "train_prompts"}} for cid, info in facts.items()},
        "anchors": {cid: value.cpu() for cid, value in anchors.items()},
        "prototypes": {cid: value.cpu() for cid, value in prototypes.items()},
        "training": router_training,
        "router_corpus": corpus_meta,
        "adapter_checkpoint_sha256": adapter_file_hash_before,
    }
    torch.save(router_checkpoint, output_dir / "relation_router_v2.pt")

    report = {
        "protocol": PROTOCOL,
        "seed": SEED,
        "frozen_rsnr": frozen_spec(),
        "architecture_freeze_enforced": True,
        "adapter_retrained": False,
        "base_model_parameters_trainable": 0,
        "adapter_parameters_trainable": 0,
        "router_parameters_trainable": router_training["trainable_parameters"],
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_sha256_before": adapter_file_hash_before,
        "adapter_checkpoint_sha256_after": adapter_file_hash_after,
        "adapter_tensor_digest_before": adapter_digest_before,
        "adapter_tensor_digest_after": adapter_digest_after,
        "router_training": router_training,
        "router_corpus": corpus_meta,
        "relation_representation": relation_meta,
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
                "Every router-negative query exactly follows Base; only router false positives can incur adapter drift."
            ),
        },
        "claim_boundary": {
            "router_is_learned": True,
            "router_uses_deterministic_subject_candidate_retrieval": True,
            "router_uses_official_probe_text_for_fit": False,
            "router_uses_official_probe_text_for_threshold_calibration": False,
            "null_adapter_changed": False,
            "base_model_changed": False,
            "exact_base_identity_when_router_off": True,
            "latent_knowledge_erasure_claimed": False,
        },
    }
    report_path = output_dir / "relation_router_v2_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "router_precision": overall_metrics["precision"],
        "router_recall": overall_metrics["recall"],
        "router_f1": overall_metrics["f1"],
        "router_false_positive_rate": overall_metrics["false_positive_rate"],
        "rewrite_recall": group_reports["forget_rewrite"]["recall"],
        "rewrite_subject_candidate_coverage": group_reports["forget_rewrite"]["candidate_subject_coverage"],
        "paraphrase_recall": group_reports["forget_paraphrase"]["recall"],
        "paraphrase_subject_candidate_coverage": group_reports["forget_paraphrase"]["candidate_subject_coverage"],
        "forget_neighborhood_fpr": group_reports["forget_neighborhood"]["false_positive_rate"],
        "subject_only_fpr": group_reports["subject_only_novel"]["false_positive_rate"],
        "same_subject_other_relation_fpr": group_reports["same_subject_different_relation_heldout"]["false_positive_rate"],
        "same_relation_other_subject_fpr": group_reports["same_relation_different_subject_heldout"]["false_positive_rate"],
        "Base_Eff": end_to_end["Base_gate_off"]["Eq16_style_Eff"],
        "Base_Gen": end_to_end["Base_gate_off"]["Eq16_style_Gen"],
        "Oracle_Eff": end_to_end["RSNR_oracle_gate"]["Eq16_style_Eff"],
        "Oracle_Gen": end_to_end["RSNR_oracle_gate"]["Eq16_style_Gen"],
        "LearnedRouterV2_Eff": end_to_end["RSNR_learned_router"]["Eq16_style_Eff"],
        "LearnedRouterV2_Gen": end_to_end["RSNR_learned_router"]["Eq16_style_Gen"],
        "gate_off_max_abs_logit_drift": gate_off_audit["max_abs_logit_drift"],
        "adapter_unchanged": adapter_digest_after == adapter_digest_before,
        "report": str(report_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    hook.remove()


if __name__ == "__main__":
    main()
