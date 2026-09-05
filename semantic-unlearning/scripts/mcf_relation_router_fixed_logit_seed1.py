#!/usr/bin/env python3
"""Seed-1 Gen-focused router experiment for Method 5.

Goal
----
Test whether a subject-conditioned relation matcher can solve Method-5's
paraphrase activation bottleneck while leaving the frozen suppression actuator
unchanged.

Training/calibration contract
-----------------------------
* Base Transformer, embeddings, and LM head are frozen.
* Method-5 fixed selected-token corrections are loaded from method5_sidecar.pt.
* The causal quotient is DISABLED.
* Router positives come only from the leakage-safe five-view training corpus.
* Router negatives/calibration may use the same sampled retain rewrite prompts
  already training-visible as Method-5 protection anchors.
* Official forget paraphrases/neighborhood prompts are evaluation-only.
* target_new is never read by the router fit/calibration path.
* The router decision for a scored token is computed from the causal prefix
  hidden states available at that prediction position; future answer tokens are
  never used for routing.

The router separates recognition from correction magnitude.  It scores all 50
registered forget facts.  A query is accepted only when the top fact score
exceeds tau and its lead over the runner-up exceeds gamma.  Accepted facts get
the FULL frozen per-fact Method-5 token correction; rejected prefixes follow the
Base logits exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_retain_anchored_context_head_seed1 as m4
import mcf_zero_unlearn_official_eval as off
import run_mcf_private_vocab_rewiring_v1_3_multiview as multiview
from mcf_sampling import sample_official_mcf_records

SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
SUBJECT_ANCHOR_TEMPLATE = "General information about {subject}."
ROUTER_STEPS = 1200
ROUTER_LR = 5e-2
ROUTER_WEIGHT_DECAY = 1e-4
ROUTER_BATCH = 512
MIN_CALIBRATION_RECALL = 0.98
MAX_ROUTER_LENGTH = 256
RETAIN_TRAIN_LIMIT = 300
RETAIN_CALIB_LIMIT = 300
SYNTHETIC_HARD_NEGATIVES_PER_FACT = 2


def _rr(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def render_rewrite(row: Mapping[str, Any]) -> str:
    rr = _rr(row)
    return str(rr["prompt"]).format(str(rr["subject"]))


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(x) for x in values if str(x).strip()))


def fact_key(row: Mapping[str, Any]) -> tuple[str, str]:
    rr = _rr(row)
    return str(rr["subject"]), str(rr["relation_id"])


def fact_info(row: Mapping[str, Any], record_index: int) -> dict[str, Any]:
    rr = _rr(row)
    return {
        "record_index": int(record_index),
        "case_id": int(row["case_id"]),
        "subject": str(rr["subject"]),
        "relation_id": str(rr["relation_id"]),
    }


def split_fact_views(
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for record_index, row in enumerate(forget):
        info = fact_info(row, record_index)
        cid = info["case_id"]
        templates = [str(x) for x in view_map[cid]]
        if len(templates) != 5:
            raise RuntimeError(f"case_id={cid}: expected exactly five router-safe views")
        calib_index = cid % 5
        train_templates = [x for j, x in enumerate(templates) if j != calib_index]
        out[record_index] = {
            **info,
            "train_templates": train_templates,
            "calib_template": templates[calib_index],
            "train_prompts": [x.format(info["subject"]) for x in train_templates],
            "calib_prompt": templates[calib_index].format(info["subject"]),
        }
    return out


def split_retain_for_router(retain: Sequence[Mapping[str, Any]]) -> tuple[list, list, list]:
    train, calib, leftover = [], [], []
    ordered = sorted(
        retain,
        key=lambda row: stable_int(f"retain:{SEED}:{int(row['case_id'])}:{render_rewrite(row)}"),
    )
    for row in ordered:
        if len(train) < RETAIN_TRAIN_LIMIT:
            train.append(row)
        elif len(calib) < RETAIN_CALIB_LIMIT:
            calib.append(row)
        else:
            leftover.append(row)
    return train, calib, leftover


def synthetic_hard_negative_prompts(
    facts: Mapping[int, Mapping[str, Any]],
    *,
    calibration: bool,
    per_fact: int,
) -> list[str]:
    """Same subject, but a different relation template; no official probe text."""
    out: list[str] = []
    ids = sorted(facts)
    for i in ids:
        target = facts[i]
        registered_pairs = {(str(facts[j]["subject"]), str(facts[j]["relation_id"])) for j in ids}
        choices = [
            j for j in ids
            if j != i
            and str(facts[j]["relation_id"]) != str(target["relation_id"])
            and (str(target["subject"]), str(facts[j]["relation_id"])) not in registered_pairs
        ]
        if not choices:
            continue
        offset = stable_int(f"hardneg:{SEED}:{i}:{int(calibration)}") % len(choices)
        for k in range(int(per_fact)):
            other = facts[choices[(offset + k) % len(choices)]]
            if calibration:
                template = str(other["calib_template"])
            else:
                ts = list(other["train_templates"])
                template = str(ts[k % len(ts)])
            out.append(template.format(str(target["subject"])))
    return dedupe(out)


@torch.no_grad()
def encode_prompts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """L2-normalized mean of frozen Base final hidden states."""
    if not texts:
        return torch.empty((0, int(model.config.hidden_size)), dtype=torch.float32)
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("router currently requires model.model backbone")
    chunks: list[torch.Tensor] = []
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), int(batch_size)):
            batch = list(texts[start:start + int(batch_size)])
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_ROUTER_LENGTH,
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)
            out = backbone(**enc, use_cache=False, return_dict=True)
            hidden = out.last_hidden_state.float()
            mask = enc["attention_mask"].to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, p=2, dim=-1).cpu())
    finally:
        tokenizer.padding_side = old
    return torch.cat(chunks, dim=0)


def build_relation_state(
    model: Any,
    tokenizer: Any,
    facts: Mapping[int, Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts: list[str] = []
    anchor_text: dict[int, str] = {}
    for i, info in facts.items():
        anchor = SUBJECT_ANCHOR_TEMPLATE.format(subject=info["subject"])
        anchor_text[i] = anchor
        texts.append(anchor)
        texts.extend(info["train_prompts"])
    unique = dedupe(texts)
    enc = encode_prompts(model, tokenizer, unique, device=device, batch_size=batch_size)
    emap = {text: enc[j] for j, text in enumerate(unique)}
    anchors, protos = [], []
    for i in sorted(facts):
        info = facts[i]
        anchor = emap[anchor_text[i]].float()
        residuals = [
            F.normalize(emap[text].float() - anchor, p=2, dim=0)
            for text in info["train_prompts"]
        ]
        anchors.append(anchor)
        protos.append(F.normalize(torch.stack(residuals).mean(dim=0), p=2, dim=0))
    return torch.stack(anchors), torch.stack(protos)


class DiagonalRelationRouter(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(int(hidden_size), 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


def pair_feature(q: torch.Tensor, anchor: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
    return F.normalize(q.float() - anchor.float(), p=2, dim=-1) * proto.float()


def build_training_queries(
    facts: Mapping[int, Mapping[str, Any]],
    retain_train: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[int | None], dict[str, int]]:
    texts: list[str] = []
    owner: list[int | None] = []
    for i in sorted(facts):
        for text in facts[i]["train_prompts"]:
            texts.append(str(text)); owner.append(int(i))
    retain_prompts = dedupe(render_rewrite(x) for x in retain_train)
    for text in retain_prompts:
        texts.append(text); owner.append(None)
    hard = synthetic_hard_negative_prompts(
        facts, calibration=False, per_fact=SYNTHETIC_HARD_NEGATIVES_PER_FACT
    )
    for text in hard:
        texts.append(text); owner.append(None)
    return texts, owner, {
        "positive_safe_views": sum(x is not None for x in owner),
        "retain_negative_queries": len(retain_prompts),
        "synthetic_same_subject_different_relation": len(hard),
        "total_queries": len(texts),
    }


def pair_index_arrays(owners: Sequence[int | None], num_facts: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_idx, f_idx, labels = [], [], []
    for qi, own in enumerate(owners):
        for fi in range(num_facts):
            q_idx.append(qi); f_idx.append(fi)
            labels.append(1.0 if own is not None and int(own) == fi else 0.0)
    return (
        torch.tensor(q_idx, dtype=torch.long),
        torch.tensor(f_idx, dtype=torch.long),
        torch.tensor(labels, dtype=torch.float32),
    )


def train_router(
    query_embeddings: torch.Tensor,
    owners: Sequence[int | None],
    anchors: torch.Tensor,
    protos: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[DiagonalRelationRouter, dict[str, Any]]:
    q_idx, f_idx, labels = pair_index_arrays(owners, anchors.shape[0])
    positives = max(1, int(labels.sum().item()))
    negatives = max(1, int(labels.numel() - labels.sum().item()))
    pos_weight = torch.tensor(negatives / positives, dtype=torch.float32, device=device)
    router = DiagonalRelationRouter(query_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=ROUTER_LR, weight_decay=ROUTER_WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED + 50291)
    q_gpu = query_embeddings.to(device)
    a_gpu = anchors.to(device)
    p_gpu = protos.to(device)
    trace = []
    router.train()
    n = labels.numel()
    for step in range(1, ROUTER_STEPS + 1):
        take = torch.randint(0, n, (min(ROUTER_BATCH, n),), generator=generator)
        qi = q_idx[take].to(device); fi = f_idx[take].to(device); y = labels[take].to(device)
        feat = pair_feature(q_gpu[qi], a_gpu[fi], p_gpu[fi])
        optimizer.zero_grad(set_to_none=True)
        logits = router(feat)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == ROUTER_STEPS:
            trace.append({"step": step, "loss": float(loss.detach().item())})
    router.eval()
    for p in router.parameters():
        p.requires_grad_(False)
    return router, {
        "router_type": "shared_subject_residual_diagonal_relation_matcher",
        "trainable_parameters": sum(p.numel() for p in router.parameters()),
        "steps": ROUTER_STEPS,
        "batch_size": ROUTER_BATCH,
        "learning_rate": ROUTER_LR,
        "weight_decay": ROUTER_WEIGHT_DECAY,
        "positive_pair_weight": negatives / positives,
        "pair_count": int(n),
        "positive_pair_count": int(positives),
        "negative_pair_count": int(negatives),
        "loss_trace": trace,
    }


@torch.no_grad()
def score_embeddings_all_facts(
    router: DiagonalRelationRouter,
    queries: torch.Tensor,
    anchors: torch.Tensor,
    protos: torch.Tensor,
    *,
    device: torch.device,
    query_chunk: int = 32,
) -> torch.Tensor:
    router.eval()
    a = anchors.to(device); p = protos.to(device)
    rows: list[torch.Tensor] = []
    for start in range(0, len(queries), int(query_chunk)):
        q = queries[start:start + int(query_chunk)].to(device)
        rel = F.normalize(q[:, None, :].float() - a[None, :, :].float(), p=2, dim=-1)
        feat = rel * p[None, :, :].float()
        probs = torch.sigmoid(router(feat.reshape(-1, feat.shape[-1]))).reshape(q.shape[0], a.shape[0])
        rows.append(probs.cpu())
    return torch.cat(rows, dim=0) if rows else torch.empty((0, anchors.shape[0]))


def score_stats(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if scores.shape[1] < 2:
        raise ValueError("router requires at least two registered facts")
    top2 = torch.topk(scores, k=2, dim=1)
    return top2.values[:, 0], top2.values[:, 0] - top2.values[:, 1], top2.indices[:, 0]


def calibrate_decision_rule(
    positive_scores: torch.Tensor,
    positive_owner: Sequence[int],
    negative_scores: torch.Tensor,
    *,
    minimum_recall: float = MIN_CALIBRATION_RECALL,
) -> tuple[float, float, dict[str, Any]]:
    p_top, p_margin, p_idx = score_stats(positive_scores)
    n_top, n_margin, _ = score_stats(negative_scores)
    owner = torch.tensor(positive_owner, dtype=torch.long)
    correct = p_idx.eq(owner)

    all_top = torch.cat([p_top, n_top]).numpy()
    all_margin = torch.cat([p_margin, n_margin]).numpy()
    quantiles = np.linspace(0.0, 1.0, 61)
    taus = sorted(set([0.0, 1.0] + [float(x) for x in np.quantile(all_top, quantiles)]))
    gammas = sorted(set([0.0, 1.0] + [float(x) for x in np.quantile(all_margin, quantiles)]))

    best = None
    fallback = None
    for tau in taus:
        for gamma in gammas:
            p_accept = (p_top >= tau) & (p_margin >= gamma)
            p_correct_accept = p_accept & correct
            recall = float(p_correct_accept.float().mean().item()) if len(p_top) else 0.0
            wrong = float((p_accept & ~correct).float().mean().item()) if len(p_top) else 0.0
            n_accept = float(((n_top >= tau) & (n_margin >= gamma)).float().mean().item()) if len(n_top) else 0.0
            candidate = {
                "tau": float(tau), "gamma": float(gamma),
                "correct_accept_rate": recall,
                "wrong_fact_accept_rate": wrong,
                "negative_whole_bank_accept_rate": n_accept,
                "positive_reject_rate": float((~p_accept).float().mean().item()) if len(p_top) else 0.0,
                "positive_top1_correct_rate": float(correct.float().mean().item()) if len(p_top) else 0.0,
            }
            fb_key = (-recall, n_accept, wrong, -tau, -gamma)
            if fallback is None or fb_key < fallback[0]:
                fallback = (fb_key, candidate)
            if recall + 1e-12 < minimum_recall:
                continue
            key = (n_accept, wrong, -recall, -tau, -gamma)
            if best is None or key < best[0]:
                best = (key, candidate)
    chosen = best[1] if best is not None else fallback[1]
    chosen["minimum_required_correct_accept_rate"] = float(minimum_recall)
    chosen["minimum_recall_feasible"] = bool(best is not None)
    chosen["selection_rule"] = (
        "require correct-fact acceptance >= minimum; minimize whole-bank negative acceptance; "
        "then wrong-fact acceptance; otherwise maximize achievable correct acceptance"
    )
    return float(chosen["tau"]), float(chosen["gamma"]), chosen


def decision_from_scores(scores: torch.Tensor, tau: float, gamma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top, margin, idx = score_stats(scores)
    accept = (top >= float(tau)) & (margin >= float(gamma))
    return accept, idx, top, margin


def positive_router_report(scores: torch.Tensor, owners: Sequence[int], tau: float, gamma: float) -> dict[str, Any]:
    accept, idx, top, margin = decision_from_scores(scores, tau, gamma)
    owner = torch.tensor(owners, dtype=torch.long)
    correct = idx.eq(owner)
    return {
        "n": len(owners),
        "top1_correct_rate_pct": 100.0 * float(correct.float().mean().item()) if len(owners) else None,
        "correct_accept_rate_pct": 100.0 * float((accept & correct).float().mean().item()) if len(owners) else None,
        "wrong_fact_accept_rate_pct": 100.0 * float((accept & ~correct).float().mean().item()) if len(owners) else None,
        "reject_rate_pct": 100.0 * float((~accept).float().mean().item()) if len(owners) else None,
        "top_score_mean": float(top.mean().item()) if len(top) else None,
        "runner_margin_mean": float(margin.mean().item()) if len(margin) else None,
        "correct_score_mean": float(scores[torch.arange(len(owner)), owner].mean().item()) if len(owner) else None,
    }


def negative_router_report(scores: torch.Tensor, tau: float, gamma: float) -> dict[str, Any]:
    accept, _, top, margin = decision_from_scores(scores, tau, gamma)
    return {
        "n": int(scores.shape[0]),
        "whole_bank_false_activation_rate_pct": 100.0 * float(accept.float().mean().item()) if len(accept) else None,
        "top_score_mean": float(top.mean().item()) if len(top) else None,
        "runner_margin_mean": float(margin.mean().item()) if len(margin) else None,
    }


def fixed_fact_corrections(
    sidecar: Mapping[str, Any],
    direct_specs: Sequence[m4.SequenceSpec],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    selected = torch.as_tensor(sidecar["selected_token_ids"], dtype=torch.long)
    coeff = torch.as_tensor(sidecar["coefficients"], dtype=torch.float32)
    event_records: list[int] = []
    event_tokens: list[int] = []
    for spec in direct_specs:
        for token in spec.event_token_ids:
            event_records.append(int(spec.record_index)); event_tokens.append(int(token))
    if coeff.ndim != 2 or coeff.shape[1] != len(event_records):
        raise RuntimeError(
            f"sidecar correction columns={tuple(coeff.shape)} do not match reconstructed events={len(event_records)}"
        )
    num_facts = len(direct_specs)
    fact_corr = torch.zeros((num_facts, selected.numel()), dtype=torch.float32)
    for fact in range(num_facts):
        cols = [j for j, r in enumerate(event_records) if r == fact]
        if not cols:
            raise RuntimeError(f"fact {fact} has no correction event")
        fact_corr[fact] = coeff[:, cols].amin(dim=1)
    return selected, fact_corr, {
        "num_facts": num_facts,
        "num_selected_tokens": int(selected.numel()),
        "num_reconstructed_events": len(event_records),
        "fixed_penalty_min": float(fact_corr.min().item()),
        "fixed_penalty_max": float(fact_corr.max().item()),
        "facts_with_nonzero_correction": int((fact_corr.abs().sum(dim=1) > 0).sum().item()),
        "event_token_ids": event_tokens,
    }


class HardRoutedFixedCorrectionModel(nn.Module):
    """Frozen Base + hard fact router + full fixed Method-5 token correction."""
    def __init__(
        self,
        *,
        base_model: nn.Module,
        router: DiagonalRelationRouter,
        anchors: torch.Tensor,
        protos: torch.Tensor,
        selected_token_ids: torch.Tensor,
        fact_corrections: torch.Tensor,
        tau: float,
        gamma: float,
        router_chunk: int = 32,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.router = router
        self.config = base_model.config
        self.tau = float(tau); self.gamma = float(gamma); self.router_chunk = int(router_chunk)
        self.register_buffer("anchors", anchors.float(), persistent=True)
        self.register_buffer("protos", protos.float(), persistent=True)
        self.register_buffer("selected_token_ids", selected_token_ids.long(), persistent=True)
        self.register_buffer("fact_corrections", fact_corrections.float(), persistent=True)
        for p in self.base_model.parameters():
            p.requires_grad_(False)
        for p in self.router.parameters():
            p.requires_grad_(False)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    @torch.no_grad()
    def _route_prefix_embeddings(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = score_embeddings_all_facts(
            self.router, q.detach().float().cpu(), self.anchors.detach().float().cpu(),
            self.protos.detach().float().cpu(), device=self.anchors.device,
            query_chunk=self.router_chunk,
        ).to(q.device)
        accept, idx, _, _ = decision_from_scores(scores, self.tau, self.gamma)
        return accept.to(q.device), idx.to(q.device)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        kwargs.pop("output_hidden_states", False)
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        out = self.base_model(*args, **kwargs)
        hidden = out.hidden_states[-1].float()
        attention = kwargs.get("attention_mask")
        if attention is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise RuntimeError("router wrapper requires input_ids or attention_mask")
            attention = torch.ones(input_ids.shape, device=hidden.device, dtype=torch.long)
        mask = attention.to(hidden.dtype)
        csum = torch.cumsum(hidden * mask.unsqueeze(-1), dim=1)
        count = torch.cumsum(mask, dim=1).clamp_min(1.0).unsqueeze(-1)
        prefix = F.normalize(csum / count, p=2, dim=-1)
        valid = attention.bool()
        flat = prefix[valid]
        accept, fact_idx = self._route_prefix_embeddings(flat)

        logits = out.logits.clone()
        if bool(accept.any().item()):
            valid_positions = valid.nonzero(as_tuple=False)
            chosen_pos = valid_positions[accept]
            chosen_fact = fact_idx[accept]
            delta = self.fact_corrections[chosen_fact].to(logits.dtype)
            for row in range(chosen_pos.shape[0]):
                b = int(chosen_pos[row, 0]); t = int(chosen_pos[row, 1])
                logits[b, t, self.selected_token_ids] += delta[row]
        out.logits = logits
        return out


def mean_or_none(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def corrected_metrics(result: Mapping[str, Any], base: Mapping[str, Any] | None = None, threshold: float = 0.1) -> dict[str, Any]:
    def group(raw, key):
        out = []
        for item in raw:
            xs = item["post"].get(key, [])
            if xs:
                out.append([
                    {
                        "s": float(x["target_true"]),
                        "r": float(x["target_new"]),
                        "m": float(x["target_true"]) - float(x["target_new"]),
                    }
                    for x in xs
                ])
        return out
    direct = group(result["forget_raw"], "rewrite_prompts_probs")
    para = group(result["forget_raw"], "paraphrase_prompts_probs")
    def summarize(cases):
        return {
            "Sensitive_NLL": mean_or_none([statistics.fmean(x["s"] for x in c) for c in cases]),
            "Margin_Mean": mean_or_none([statistics.fmean(x["m"] for x in c) for c in cases]),
            "Margin_Failure_Rate_case_macro_pct": (
                100.0 * statistics.fmean(statistics.fmean(float(x["m"] <= threshold) for x in c) for c in cases)
                if cases else None
            ),
        }
    ds, ps = summarize(direct), summarize(para)
    out = {
        "Eff_Pref": float(result["forget"]["Eff"]),
        "Gen_Pref": float(result["forget"]["Gen"]),
        "Direct_Sensitive_NLL": ds["Sensitive_NLL"],
        "Para_Sensitive_NLL": ps["Sensitive_NLL"],
        "Direct_Margin_Mean": ds["Margin_Mean"],
        "Para_Margin_Mean": ps["Margin_Mean"],
        "Direct_Margin_Failure_Rate": ds["Margin_Failure_Rate_case_macro_pct"],
        "Para_Margin_Failure_Rate": ps["Margin_Failure_Rate_case_macro_pct"],
        "PPL_legacy100": result.get("retain_PPL"),
        "Eff_Leak": None, "Gen_Leak": None,
    }
    if base is not None:
        for label, key in (("Direct", "rewrite_prompts_probs"), ("Para", "paraphrase_prompts_probs")):
            deltas, positive_rates = [], []
            for b, e in zip(base["forget_raw"], result["forget_raw"]):
                bx, ex = b["post"].get(key, []), e["post"].get(key, [])
                if not bx:
                    continue
                ds_local = [float(y["target_true"]) - float(x["target_true"]) for x, y in zip(bx, ex)]
                deltas.append(statistics.fmean(ds_local))
                positive_rates.append(statistics.fmean(float(x > 0) for x in ds_local))
            out[f"Delta_{label}_Sensitive_NLL"] = mean_or_none(deltas)
            out[f"{label}_Positive_Delta_Rate_case_macro_pct"] = (
                None if not positive_rates else 100.0 * statistics.fmean(positive_rates)
            )
    return out


def score_text_group(
    model, tok, router, anchors, protos, texts, *, device, batch_size
) -> torch.Tensor:
    emb = encode_prompts(model, tok, texts, device=device, batch_size=batch_size)
    return score_embeddings_all_facts(router, emb, anchors, protos, device=device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "mcf-path", "wikidata-dir", "view-corpus", "method5-sidecar", "output-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--base-eval-json")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--forget-num", type=int, default=FORGET_NUM)
    p.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--minimum-calibration-recall", type=float, default=MIN_CALIBRATION_RECALL)
    p.add_argument("--skip-ppl", action="store_true")
    a = p.parse_args()
    if (a.seed, a.forget_num, a.retain_num) != (SEED, FORGET_NUM, RETAIN_NUM):
        raise ValueError("experiment locked to consumed development seed1 / forget50 / retain1000")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device(a.device)
    outdir = Path(a.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)

    view_map, view_meta = multiview.load_view_corpus(Path(a.view_corpus).resolve())
    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    fr, rr = sample_official_mcf_records(data, FORGET_NUM, RETAIN_NUM, SEED, strict=True)
    fr = [off.normalize_record(x) for x in fr]; rr = [off.normalize_record(x) for x in rr]
    facts = split_fact_views(fr, view_map)
    if set(view_map) != {int(x["case_id"]) for x in fr}:
        raise RuntimeError("view corpus case IDs do not exactly match official seed1 forget split")

    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=m4.dtype_from_str(a.dtype), local_files_only=True
    ).to(device).eval()
    model.config.use_cache = False
    for q in model.parameters(): q.requires_grad_(False)

    anchors, protos = build_relation_state(
        model, tok, facts, device=device, batch_size=a.encode_batch_size
    )
    forget_keys = {fact_key(x) for x in fr}
    router_negative_retain = [x for x in rr if fact_key(x) not in forget_keys]
    retain_train, retain_calib, retain_left = split_retain_for_router(router_negative_retain)
    corpus_meta_extra = {
        "official_retain_exact_forget_key_excluded_from_router_negatives": len(rr) - len(router_negative_retain),
        "router_negative_retain_available": len(router_negative_retain),
    }
    train_texts, train_owners, corpus_meta = build_training_queries(facts, retain_train)
    train_emb = encode_prompts(model, tok, train_texts, device=device, batch_size=a.encode_batch_size)
    router, train_meta = train_router(train_emb, train_owners, anchors, protos, device=device)

    calib_pos_texts = [facts[i]["calib_prompt"] for i in sorted(facts)]
    calib_pos_owner = list(sorted(facts))
    calib_neg_texts = dedupe(render_rewrite(x) for x in retain_calib)
    calib_neg_texts += synthetic_hard_negative_prompts(facts, calibration=True, per_fact=1)
    calib_neg_texts = dedupe(calib_neg_texts)
    pos_scores = score_text_group(
        model, tok, router, anchors, protos, calib_pos_texts,
        device=device, batch_size=a.encode_batch_size,
    )
    neg_scores = score_text_group(
        model, tok, router, anchors, protos, calib_neg_texts,
        device=device, batch_size=a.encode_batch_size,
    )
    tau, gamma, calibration = calibrate_decision_rule(
        pos_scores, calib_pos_owner, neg_scores, minimum_recall=a.minimum_calibration_recall
    )

    router.eval()
    for q in router.parameters(): q.requires_grad_(False)

    direct_texts = [render_rewrite(x) for x in fr]
    direct_owner = list(range(len(fr)))
    para_texts, para_owner = [], []
    for i, row in enumerate(fr):
        for text in row.get("paraphrase_prompts", []):
            para_texts.append(str(text)); para_owner.append(i)
    retain_eval = retain_left if retain_left else retain_calib
    retain_texts = dedupe(render_rewrite(x) for x in retain_eval)

    direct_scores = score_text_group(model, tok, router, anchors, protos, direct_texts, device=device, batch_size=a.encode_batch_size)
    para_scores = score_text_group(model, tok, router, anchors, protos, para_texts, device=device, batch_size=a.encode_batch_size)
    retain_scores = score_text_group(model, tok, router, anchors, protos, retain_texts, device=device, batch_size=a.encode_batch_size)
    router_eval = {
        "direct_forget": positive_router_report(direct_scores, direct_owner, tau, gamma),
        "forget_paraphrase": positive_router_report(para_scores, para_owner, tau, gamma),
        "retain_rewrite_heldout_from_router_fit": negative_router_report(retain_scores, tau, gamma),
    }

    sidecar = torch.load(Path(a.method5_sidecar).resolve(), map_location="cpu", weights_only=False)
    direct_specs = m4.build_specs(fr, tok, max_events_per_record=None)
    selected_ids, fact_corr, correction_meta = fixed_fact_corrections(sidecar, direct_specs)
    wrapped = HardRoutedFixedCorrectionModel(
        base_model=model,
        router=router,
        anchors=anchors.to(device),
        protos=protos.to(device),
        selected_token_ids=selected_ids.to(device),
        fact_corrections=fact_corr.to(device),
        tau=tau, gamma=gamma,
    ).to(device).eval()

    result = off.evaluate_loaded_model_official(
        method="relation_router_fixed_logit_no_quotient",
        model=wrapped, tok=tok, model_dir=a.model_path,
        mcf_path=a.mcf_path, wikidata_dir=a.wikidata_dir,
        out_path=outdir / "relation_router_official_eval.json",
        unlearn_num=FORGET_NUM, retain_num=RETAIN_NUM, seed=SEED,
        sample_mode="official", skip_ppl=a.skip_ppl,
    )
    base = None
    if a.base_eval_json:
        base = json.loads(Path(a.base_eval_json).read_text(encoding="utf-8"))
    metrics = corrected_metrics(result, base=base, threshold=0.1)

    overlap = m4._hard_overlap_records(rr, tok, set(map(int, selected_ids.tolist())))
    overlap_eval, _ = m4._evaluate_subset(
        wrapped, tok, overlap, split_name="hard_overlap_retain_relation_router"
    )

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_relation_router_fixed_logit_no_quotient",
        "training_contract": {
            "seed": SEED,
            "forget_num": FORGET_NUM,
            "retain_num": RETAIN_NUM,
            "base_transformer_frozen": True,
            "base_embeddings_frozen": True,
            "base_lm_head_frozen": True,
            "quotient_enabled": False,
            "fixed_method5_token_correction_reused": True,
            "official_paraphrases_used_for_router_fit": False,
            "official_paraphrases_used_for_threshold_calibration": False,
            "official_neighborhoods_used_for_router_fit": False,
            "target_new_used_for_router_fit": False,
            "retain_rewrite_anchors_used_for_router_fit_or_calibration": True,
            "router_decision_uses_future_answer_tokens": False,
        },
        "view_corpus": view_meta,
        "router_training_corpus": {**corpus_meta, **corpus_meta_extra},
        "router_training": train_meta,
        "calibration": calibration,
        "tau": tau,
        "gamma": gamma,
        "router_eval": router_eval,
        "correction": correction_meta,
        "metrics": metrics,
        "hard_overlap_retain_records": len(overlap),
        "hard_overlap_retain": overlap_eval,
        "output_dir": str(outdir),
    }
    (outdir / "seed1_relation_router_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    torch.save({
        "router_state_dict": {k: v.detach().cpu() for k, v in router.state_dict().items()},
        "anchors": anchors.cpu(), "prototypes": protos.cpu(),
        "tau": tau, "gamma": gamma,
        "selected_token_ids": selected_ids.cpu(), "fact_corrections": fact_corr.cpu(),
        "training_contract": summary["training_contract"],
    }, outdir / "relation_router_sidecar.pt")

    print(json.dumps({
        "router_calibration": calibration,
        "router_eval": router_eval,
        "metrics": metrics,
        "hard_overlap_retain_records": len(overlap),
        "hard_overlap_retain": overlap_eval,
        "output_dir": str(outdir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
