#!/usr/bin/env python3
"""Seed-1 recognition-only router benchmark for MCF factual suppression.

This script does NOT modify model logits and does NOT run an unlearning method.
It compares four fact-recognition systems under identical data partitions:
  1) existing subject-residual diagonal relation matcher;
  2) explicit literal-subject candidate control + the same diagonal matcher;
  3) shared pooled prompt-fact matcher with an explicit NONE class;
  4) shared candidate-conditioned token matcher with an explicit NONE class.

Data/selection contract
-----------------------
* Base Llama is frozen.
* Five-view leakage-safe corpus supplies positives only.
* Three views/fact are fit, one is threshold calibration, one is frozen validation.
* Retain rewrite prompts are split 300/300/remaining for fit/calibration/validation.
* Official MCF paraphrases/neighborhoods are NEVER used for fitting, model
  selection, or threshold calibration. Because Seed-1 official probes have
  already been inspected in prior experiments, they are reported only as
  development evidence after all recognition decisions are frozen.
* No target_new is read anywhere in this runner.
* target_true is used only to tag a retain hard-negative family (same answer,
  different fact); it is never supplied to a matcher.

Preservation-first calibration
------------------------------
Choose tau/gamma to MAXIMIZE correct-fact acceptance subject to BOTH:
  whole-bank negative acceptance <= epsilon_retain,
  wrong-fact acceptance on positives <= epsilon_wrong,
and the negative acceptance budget is also enforced separately on every
non-empty difficult-negative family. If the best feasible operating point has
correct acceptance below min_calib_correct_accept, status is
NO_ACCEPTABLE_OPERATING_POINT. No output correction is attached regardless.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass
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

import mcf_zero_unlearn_official_eval as off
import run_mcf_private_vocab_rewiring_v1_3_multiview as multiview
from mcf_sampling import sample_official_mcf_records

SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
MAX_LENGTH = 256
DIAG_STEPS = 1200
DIAG_BATCH = 512
MATCHER_DIM = 128
POOLED_STEPS = 900
TOKEN_STEPS = 700
POOLED_BATCH = 64
TOKEN_BATCH = 12
LR_DIAG = 5e-2
LR_MATCHER = 2e-3
WEIGHT_DECAY = 1e-4
RETAIN_FIT = 300
RETAIN_CALIB = 300
SYNTH_HARD_PER_FACT = 1


@dataclass(frozen=True)
class QueryRow:
    text: str
    owner: int | None
    kind: str


def rr(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def render(row: Mapping[str, Any]) -> str:
    r = rr(row)
    return str(r["prompt"]).format(str(r["subject"]))


def fact_key(row: Mapping[str, Any]) -> tuple[str, str]:
    r = rr(row)
    return str(r["subject"]), str(r["relation_id"])


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def dedupe_rows(rows: Sequence[QueryRow]) -> list[QueryRow]:
    by_text: dict[str, QueryRow] = {}
    for row in rows:
        key = " ".join(row.text.split())
        old = by_text.get(key)
        if old is not None and old.owner != row.owner:
            raise RuntimeError(f"ambiguous query label for {key!r}: {old.owner} vs {row.owner}")
        if old is None:
            by_text[key] = QueryRow(key, row.owner, row.kind)
    return list(by_text.values())


def relation_text_from_record(row: Mapping[str, Any]) -> str:
    """Dataset-provided relation wording; no answer is included."""
    prompt = str(rr(row)["prompt"])
    try:
        return prompt.format("ENTITY")
    except Exception:
        return prompt.replace("{}", "ENTITY")


def split_fact_views(forget: Sequence[Mapping[str, Any]], view_map: Mapping[int, Sequence[str]]) -> dict[int, dict[str, Any]]:
    facts: dict[int, dict[str, Any]] = {}
    for i, row in enumerate(forget):
        r = rr(row)
        cid = int(row["case_id"])
        templates = [str(x) for x in view_map[cid]]
        if len(templates) != 5:
            raise RuntimeError(f"case_id={cid}: expected exactly 5 leakage-safe views")
        val_idx = cid % 5
        calib_idx = (cid + 1) % 5
        fit_idx = [j for j in range(5) if j not in {val_idx, calib_idx}]
        subject = str(r["subject"])
        facts[i] = {
            "record_index": i,
            "case_id": cid,
            "subject": subject,
            "relation_id": str(r["relation_id"]),
            "target_true": str(r["target_true"]["str"]),
            "relation_text": relation_text_from_record(row),
            "fit_templates": [templates[j] for j in fit_idx],
            "calib_template": templates[calib_idx],
            "validation_template": templates[val_idx],
            "fit_prompts": [templates[j].format(subject) for j in fit_idx],
            "calib_prompt": templates[calib_idx].format(subject),
            "validation_prompt": templates[val_idx].format(subject),
        }
    return facts


def split_retain(retain: Sequence[Mapping[str, Any]], bank_pairs: set[tuple[str, str]]) -> tuple[list, list, list]:
    clean = [x for x in retain if fact_key(x) not in bank_pairs]
    clean = sorted(clean, key=lambda x: stable_int(f"retain:{SEED}:{int(x['case_id'])}:{render(x)}"))
    fit = clean[:RETAIN_FIT]
    calib = clean[RETAIN_FIT:RETAIN_FIT + RETAIN_CALIB]
    val = clean[RETAIN_FIT + RETAIN_CALIB:]
    if len(fit) < RETAIN_FIT or len(calib) < RETAIN_CALIB or not val:
        raise RuntimeError("insufficient retain records after removing bank duplicates")
    return fit, calib, val


def phase_template(info: Mapping[str, Any], phase: str, offset: int = 0) -> str:
    if phase == "fit":
        ts = list(info["fit_templates"])
        return str(ts[offset % len(ts)])
    if phase == "calib":
        return str(info["calib_template"])
    if phase == "validation":
        return str(info["validation_template"])
    raise ValueError(phase)


def synthetic_same_subject_other_relation(
    facts: Mapping[int, Mapping[str, Any]], phase: str, per_fact: int = SYNTH_HARD_PER_FACT
) -> list[QueryRow]:
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    ids = sorted(facts)
    out: list[QueryRow] = []
    for i in ids:
        a = facts[i]
        choices = [
            j for j in ids if j != i
            and str(facts[j]["relation_id"]) != str(a["relation_id"])
            and (str(a["subject"]), str(facts[j]["relation_id"])) not in bank
        ]
        if not choices:
            continue
        start = stable_int(f"ssdr:{phase}:{SEED}:{i}") % len(choices)
        for k in range(int(per_fact)):
            b = facts[choices[(start + k) % len(choices)]]
            text = phase_template(b, phase, k).format(str(a["subject"]))
            out.append(QueryRow(text, None, "synthetic_same_subject_different_relation"))
    return out


def crossed_binding_rows(facts: Mapping[int, Mapping[str, Any]], phase: str) -> list[QueryRow]:
    """Create NONE prompts where candidate subject and relation occur but are not bound."""
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    ids = sorted(facts)
    out: list[QueryRow] = []
    for i in ids:
        a = facts[i]
        rel_choices = [j for j in ids if j != i and (str(a["subject"]), str(facts[j]["relation_id"])) not in bank]
        if not rel_choices:
            continue
        j = rel_choices[stable_int(f"cross-rel:{phase}:{i}") % len(rel_choices)]
        b_rel = facts[j]
        subj_choices = [k for k in ids if k != i and (str(facts[k]["subject"]), str(a["relation_id"])) not in bank]
        if not subj_choices:
            continue
        k = subj_choices[stable_int(f"cross-sub:{phase}:{i}") % len(subj_choices)]
        b_sub = facts[k]
        clause1 = phase_template(b_rel, phase).format(str(a["subject"]))
        clause2 = phase_template(a, phase).format(str(b_sub["subject"]))
        out.append(QueryRow(f"First: {clause1} Second: {clause2}", None, "crossed_binding"))
    return out


def mention_rows(facts: Mapping[int, Mapping[str, Any]], phase: str) -> list[QueryRow]:
    templates = {
        "fit": 'Spell the entity name "{subject}" exactly.',
        "calib": 'Quote the entity name "{subject}" without stating any factual attribute.',
        "validation": 'Repeat only the name "{subject}" and do not provide facts about it.',
    }
    return [QueryRow(templates[phase].format(subject=v["subject"]), None, "permitted_subject_mention") for v in facts.values()]


def retain_kind(row: Mapping[str, Any], facts: Mapping[int, Mapping[str, Any]]) -> str:
    s, r_id = fact_key(row)
    bank_pairs = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    bank_subjects = {x[0] for x in bank_pairs}
    bank_relations = {x[1] for x in bank_pairs}
    bank_answers = {str(v["target_true"]) for v in facts.values()}
    ans = str(rr(row)["target_true"]["str"])
    if s in bank_subjects and (s, r_id) not in bank_pairs:
        return "retain_same_subject_different_relation"
    if r_id in bank_relations and s not in bank_subjects:
        return "retain_same_relation_different_subject"
    if ans in bank_answers:
        return "retain_same_answer_different_fact"
    return "retain_other"


def build_phase_rows(
    facts: Mapping[int, Mapping[str, Any]], retain_rows: Sequence[Mapping[str, Any]], phase: str
) -> list[QueryRow]:
    rows: list[QueryRow] = []
    for i in sorted(facts):
        info = facts[i]
        if phase == "fit":
            rows.extend(QueryRow(text, i, "positive_safe_view_fit") for text in info["fit_prompts"])
        elif phase == "calib":
            rows.append(QueryRow(str(info["calib_prompt"]), i, "positive_safe_view_calibration"))
        elif phase == "validation":
            rows.append(QueryRow(str(info["validation_prompt"]), i, "positive_safe_view_validation"))
        else:
            raise ValueError(phase)
    rows.extend(QueryRow(render(x), None, retain_kind(x, facts)) for x in retain_rows)
    rows.extend(synthetic_same_subject_other_relation(facts, phase))
    rows.extend(crossed_binding_rows(facts, phase))
    rows.extend(mention_rows(facts, phase))
    return dedupe_rows(rows)


def family(kind: str) -> str:
    if "same_subject_different_relation" in kind:
        return "same_subject_different_relation"
    if "same_relation_different_subject" in kind:
        return "same_relation_different_subject"
    if "same_answer_different_fact" in kind:
        return "same_answer_different_fact"
    if kind == "crossed_binding":
        return "crossed_binding"
    if kind == "permitted_subject_mention":
        return "permitted_subject_mention"
    return "other_retain"


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[str(name)]


@torch.no_grad()
def encode_pooled(model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, batch_size: int) -> torch.Tensor:
    if not texts:
        return torch.empty((0, int(model.config.hidden_size)), dtype=torch.float32)
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("benchmark requires a model.model backbone")
    chunks: list[torch.Tensor] = []
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), int(batch_size)):
            batch = list(texts[start:start + int(batch_size)])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            out = backbone(**enc, use_cache=False, return_dict=True)
            h = out.last_hidden_state.float()
            mask = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, p=2, dim=-1).cpu())
    finally:
        tokenizer.padding_side = old
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_token_states(model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, batch_size: int) -> list[torch.Tensor]:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("benchmark requires a model.model backbone")
    states: list[torch.Tensor] = []
    old = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), int(batch_size)):
            batch = list(texts[start:start + int(batch_size)])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            out = backbone(**enc, use_cache=False, return_dict=True)
            h = out.last_hidden_state.detach()
            lengths = enc["attention_mask"].sum(1).tolist()
            for j, n in enumerate(lengths):
                states.append(h[j, :int(n)].to(dtype=torch.bfloat16).cpu().contiguous())
    finally:
        tokenizer.padding_side = old
    return states


def pad_token_batch(states: Sequence[torch.Tensor], indices: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    local = [states[int(i)] for i in indices]
    max_len = max(x.shape[0] for x in local)
    dim = local[0].shape[1]
    x = torch.zeros((len(local), max_len, dim), dtype=torch.bfloat16, device=device)
    mask = torch.zeros((len(local), max_len), dtype=torch.bool, device=device)
    for j, item in enumerate(local):
        n = item.shape[0]
        x[j, :n] = item.to(device)
        mask[j, :n] = True
    return x, mask


class DiagonalMatcher(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.linear = nn.Linear(hidden, 1)
        with torch.no_grad():
            self.linear.weight.fill_(1.0)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def build_diag_state(
    model: Any, tokenizer: Any, facts: Mapping[int, Mapping[str, Any]], device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    subject_texts = [f"General information about {facts[i]['subject']}." for i in sorted(facts)]
    subjects = encode_pooled(model, tokenizer, subject_texts, device, batch_size)
    fit_texts = [text for i in sorted(facts) for text in facts[i]["fit_prompts"]]
    fit_emb = encode_pooled(model, tokenizer, fit_texts, device, batch_size)
    protos = []
    cursor = 0
    for i in sorted(facts):
        residuals = []
        for _ in facts[i]["fit_prompts"]:
            residuals.append(F.normalize(fit_emb[cursor] - subjects[i], p=2, dim=0)); cursor += 1
        protos.append(F.normalize(torch.stack(residuals).mean(0), p=2, dim=0))
    return subjects, torch.stack(protos)


def diag_features(q: torch.Tensor, anchors: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    rel = F.normalize(q[:, None, :].float() - anchors[None, :, :].float(), p=2, dim=-1)
    return rel * protos[None, :, :].float()


def train_diagonal(
    query_emb: torch.Tensor, rows: Sequence[QueryRow], anchors: torch.Tensor, protos: torch.Tensor, device: torch.device
) -> DiagonalMatcher:
    f = anchors.shape[0]
    q_idx, f_idx, y = [], [], []
    for qi, row in enumerate(rows):
        for fi in range(f):
            q_idx.append(qi); f_idx.append(fi); y.append(float(row.owner is not None and int(row.owner) == fi))
    q_idx = torch.tensor(q_idx); f_idx = torch.tensor(f_idx); y = torch.tensor(y, dtype=torch.float32)
    positives = max(1, int(y.sum().item())); negatives = max(1, int(len(y) - positives))
    pos_weight = torch.tensor(negatives / positives, device=device)
    model = DiagonalMatcher(query_emb.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_DIAG, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(SEED + 41)
    qg, ag, pg = query_emb.to(device), anchors.to(device), protos.to(device)
    for _step in range(DIAG_STEPS):
        take = torch.randint(0, len(y), (min(DIAG_BATCH, len(y)),), generator=gen)
        qi = q_idx[take].to(device); fi = f_idx[take].to(device); yy = y[take].to(device)
        rel = F.normalize(qg[qi] - ag[fi], p=2, dim=-1)
        feat = rel * pg[fi]
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(feat), yy, pos_weight=pos_weight)
        loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def score_diagonal(model: DiagonalMatcher, query_emb: torch.Tensor, anchors: torch.Tensor, protos: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for start in range(0, len(query_emb), 32):
        q = query_emb[start:start + 32].to(device)
        feat = diag_features(q, anchors.to(device), protos.to(device))
        p = torch.sigmoid(model(feat.reshape(-1, feat.shape[-1]))).reshape(q.shape[0], anchors.shape[0])
        rows.append(p.cpu())
    scores = torch.cat(rows, 0) if rows else torch.empty((0, anchors.shape[0]))
    return scores, torch.zeros((scores.shape[0],), dtype=torch.float32)


class PooledFactMatcher(nn.Module):
    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.q = nn.Linear(hidden, dim, bias=False)
        self.s = nn.Linear(hidden, dim, bias=False)
        self.r = nn.Linear(hidden, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim * 8, dim * 2), nn.GELU(), nn.Linear(dim * 2, 1))
        self.none = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, query: torch.Tensor, subjects: torch.Tensor, relations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = F.normalize(self.q(query.float()), p=2, dim=-1)
        s = F.normalize(self.s(subjects.float()), p=2, dim=-1)
        r = F.normalize(self.r(relations.float()), p=2, dim=-1)
        qb = q[:, None, :]; sb = s[None, :, :]; rb = r[None, :, :]
        feat = torch.cat([
            qb.expand(-1, s.shape[0], -1), sb.expand(q.shape[0], -1, -1), rb.expand(q.shape[0], -1, -1),
            qb * sb, qb * rb, sb * rb,
            (qb - sb).abs(), (qb - rb).abs(),
        ], dim=-1)
        fact = self.mlp(feat).squeeze(-1)
        none = self.none(q).squeeze(-1)
        return fact, none


class TokenFactMatcher(nn.Module):
    def __init__(self, hidden: int, dim: int):
        super().__init__()
        self.token = nn.Linear(hidden, dim, bias=False)
        self.subject = nn.Linear(hidden, dim, bias=False)
        self.relation = nn.Linear(hidden, dim, bias=False)
        self.condition = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, 1))
        self.none = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, subjects: torch.Tensor, relations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = F.normalize(self.token(tokens.float()), p=2, dim=-1)
        s = F.normalize(self.subject(subjects.float()), p=2, dim=-1)
        r = F.normalize(self.relation(relations.float()), p=2, dim=-1)
        score_s = torch.einsum("btd,fd->bft", t, s) * self.scale
        score_s = score_s.masked_fill(~mask[:, None, :], -1e4)
        att_s = torch.softmax(score_s, dim=-1)
        u = torch.einsum("bft,btd->bfd", att_s, t)
        rq = F.normalize(r[None, :, :] + self.condition(u), p=2, dim=-1)
        score_r = torch.einsum("bfd,btd->bft", rq, t) * self.scale
        score_r = score_r.masked_fill(~mask[:, None, :], -1e4)
        att_r = torch.softmax(score_r, dim=-1)
        v = torch.einsum("bft,btd->bfd", att_r, t)
        feat = torch.cat([u, v, u * v, (u - v).abs()], dim=-1)
        fact = self.mlp(feat).squeeze(-1)
        m = mask.to(t.dtype).unsqueeze(-1)
        pooled = (t * m).sum(1) / m.sum(1).clamp_min(1.0)
        none = self.none(pooled).squeeze(-1)
        return fact, none


def labels_from_rows(rows: Sequence[QueryRow], num_facts: int) -> torch.Tensor:
    return torch.tensor([num_facts if x.owner is None else int(x.owner) for x in rows], dtype=torch.long)


def class_weights(rows: Sequence[QueryRow], num_facts: int, device: torch.device) -> torch.Tensor:
    positives = sum(x.owner is not None for x in rows)
    negatives = max(1, len(rows) - positives)
    positive_weight = negatives / max(1, positives)
    w = torch.ones(num_facts + 1, dtype=torch.float32, device=device)
    w[:num_facts] = float(positive_weight)
    return w


def train_pooled_matcher(
    query: torch.Tensor, rows: Sequence[QueryRow], subjects: torch.Tensor, relations: torch.Tensor, device: torch.device
) -> PooledFactMatcher:
    f = subjects.shape[0]
    model = PooledFactMatcher(query.shape[1], MATCHER_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MATCHER, weight_decay=WEIGHT_DECAY)
    labels = labels_from_rows(rows, f)
    weights = class_weights(rows, f, device)
    gen = torch.Generator().manual_seed(SEED + 101)
    qg, sg, rg = query.to(device), subjects.to(device), relations.to(device)
    for _step in range(POOLED_STEPS):
        idx = torch.randint(0, len(rows), (min(POOLED_BATCH, len(rows)),), generator=gen)
        fact, none = model(qg[idx.to(device)], sg, rg)
        logits = torch.cat([fact, none[:, None]], dim=1)
        y = labels[idx].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward(); opt.step()
    model.eval(); return model


@torch.no_grad()
def score_pooled_matcher(model: PooledFactMatcher, query: torch.Tensor, subjects: torch.Tensor, relations: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fs, ns = [], []
    sg, rg = subjects.to(device), relations.to(device)
    for start in range(0, len(query), 64):
        fact, none = model(query[start:start + 64].to(device), sg, rg)
        p = torch.softmax(torch.cat([fact, none[:, None]], dim=1), dim=1)
        fs.append(p[:, :-1].cpu()); ns.append(p[:, -1].cpu())
    return torch.cat(fs, 0), torch.cat(ns, 0)


def train_token_matcher(
    states: Sequence[torch.Tensor], rows: Sequence[QueryRow], subjects: torch.Tensor, relations: torch.Tensor, device: torch.device
) -> TokenFactMatcher:
    f = subjects.shape[0]
    hidden = states[0].shape[1]
    model = TokenFactMatcher(hidden, MATCHER_DIM).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_MATCHER, weight_decay=WEIGHT_DECAY)
    labels = labels_from_rows(rows, f)
    weights = class_weights(rows, f, device)
    gen = torch.Generator().manual_seed(SEED + 202)
    sg, rg = subjects.to(device), relations.to(device)
    for _step in range(TOKEN_STEPS):
        idx = torch.randint(0, len(rows), (min(TOKEN_BATCH, len(rows)),), generator=gen).tolist()
        x, mask = pad_token_batch(states, idx, device)
        fact, none = model(x, mask, sg, rg)
        logits = torch.cat([fact, none[:, None]], dim=1)
        y = labels[torch.tensor(idx)].to(device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits, y, weight=weights)
        loss.backward(); opt.step()
    model.eval(); return model


@torch.no_grad()
def score_token_matcher(model: TokenFactMatcher, states: Sequence[torch.Tensor], subjects: torch.Tensor, relations: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    fs, ns = [], []
    sg, rg = subjects.to(device), relations.to(device)
    for start in range(0, len(states), TOKEN_BATCH):
        idx = list(range(start, min(len(states), start + TOKEN_BATCH)))
        x, mask = pad_token_batch(states, idx, device)
        fact, none = model(x, mask, sg, rg)
        p = torch.softmax(torch.cat([fact, none[:, None]], dim=1), dim=1)
        fs.append(p[:, :-1].cpu()); ns.append(p[:, -1].cpu())
    return torch.cat(fs, 0), torch.cat(ns, 0)


def subject_regex(subject: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", re.IGNORECASE)


def subject_mask(texts: Sequence[str], facts: Mapping[int, Mapping[str, Any]]) -> torch.Tensor:
    patterns = [subject_regex(str(facts[i]["subject"])) for i in sorted(facts)]
    return torch.tensor([[bool(p.search(text)) for p in patterns] for text in texts], dtype=torch.bool)


def apply_subject_control(scores: torch.Tensor, none: torch.Tensor, texts: Sequence[str], facts: Mapping[int, Mapping[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = subject_mask(texts, facts)
    controlled = scores.clone()
    controlled[~mask] = 0.0
    return controlled, none.clone(), mask


def decision_components(scores: torch.Tensor, none: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top2 = torch.topk(scores, 2, dim=1)
    top = top2.values[:, 0]
    idx = top2.indices[:, 0]
    runner = torch.maximum(top2.values[:, 1], none)
    margin = top - runner
    return top, margin, idx, runner


def decision(scores: torch.Tensor, none: torch.Tensor, tau: float, gamma: float, candidate_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top, margin, idx, _ = decision_components(scores, none)
    accept = (top >= float(tau)) & (margin >= float(gamma))
    if candidate_mask is not None:
        accept = accept & candidate_mask.any(dim=1)
    return accept, idx, top, margin


def positive_report(scores: torch.Tensor, none: torch.Tensor, rows: Sequence[QueryRow], tau: float, gamma: float, candidate_mask: torch.Tensor | None = None) -> dict[str, Any]:
    owners = torch.tensor([int(x.owner) for x in rows], dtype=torch.long)
    accept, idx, top, margin = decision(scores, none, tau, gamma, candidate_mask)
    correct = idx.eq(owners)
    report = {
        "n": len(rows),
        "top1_correct_rate_pct": 100 * float(correct.float().mean().item()),
        "correct_accept_rate_pct": 100 * float((accept & correct).float().mean().item()),
        "wrong_fact_accept_rate_pct": 100 * float((accept & ~correct).float().mean().item()),
        "reject_rate_pct": 100 * float((~accept).float().mean().item()),
        "top_score_mean": float(top.mean().item()),
        "runner_margin_mean": float(margin.mean().item()),
    }
    if candidate_mask is not None:
        correct_in = candidate_mask[torch.arange(len(rows)), owners]
        report["candidate_recall_pct"] = 100 * float(correct_in.float().mean().item())
        denom = int(correct_in.sum().item())
        report["correct_selection_given_candidate_pct"] = (
            100 * float((correct & correct_in).sum().item()) / denom if denom else None
        )
    return report


def negative_report(scores: torch.Tensor, none: torch.Tensor, tau: float, gamma: float, candidate_mask: torch.Tensor | None = None) -> dict[str, Any]:
    accept, _idx, top, margin = decision(scores, none, tau, gamma, candidate_mask)
    return {
        "n": int(scores.shape[0]),
        "whole_bank_false_activation_rate_pct": 100 * float(accept.float().mean().item()) if len(accept) else None,
        "top_score_mean": float(top.mean().item()) if len(top) else None,
        "runner_margin_mean": float(margin.mean().item()) if len(margin) else None,
    }


def split_pos_neg(rows: Sequence[QueryRow], scores: torch.Tensor, none: torch.Tensor, mask: torch.Tensor | None = None):
    pidx = [i for i, x in enumerate(rows) if x.owner is not None]
    nidx = [i for i, x in enumerate(rows) if x.owner is None]
    pos_rows = [rows[i] for i in pidx]
    neg_rows = [rows[i] for i in nidx]
    ps = scores[pidx]; pn = none[pidx]
    ns = scores[nidx]; nn = none[nidx]
    pm = mask[pidx] if mask is not None else None
    nm = mask[nidx] if mask is not None else None
    return pos_rows, neg_rows, ps, pn, ns, nn, pm, nm


def calibrate(
    rows: Sequence[QueryRow], scores: torch.Tensor, none: torch.Tensor, *,
    epsilon_retain: float, epsilon_wrong: float, min_correct_accept: float,
    candidate_mask: torch.Tensor | None = None,
) -> tuple[float, float, dict[str, Any]]:
    pos_rows, neg_rows, ps, pn, ns, nn, pm, nm = split_pos_neg(rows, scores, none, candidate_mask)
    owners = torch.tensor([int(x.owner) for x in pos_rows], dtype=torch.long)
    p_top, p_margin, p_idx, _ = decision_components(ps, pn)
    n_top, n_margin, _nidx, _ = decision_components(ns, nn)
    correct = p_idx.eq(owners)
    all_top = torch.cat([p_top, n_top]).numpy()
    all_margin = torch.cat([p_margin, n_margin]).numpy()
    qs = np.linspace(0, 1, 71)
    taus = sorted(set([0.0, 1.000001] + [float(x) for x in np.quantile(all_top, qs)]))
    gammas = sorted(set([-1.0, 1.000001] + [float(x) for x in np.quantile(all_margin, qs)]))
    neg_family_local: dict[str, list[int]] = {}
    for local_i, row in enumerate(neg_rows):
        neg_family_local.setdefault(family(row.kind), []).append(local_i)

    best = None
    for tau in taus:
        for gamma in gammas:
            pa = (p_top >= tau) & (p_margin >= gamma)
            if pm is not None:
                pa = pa & pm.any(dim=1)
            na = (n_top >= tau) & (n_margin >= gamma)
            if nm is not None:
                na = na & nm.any(dim=1)
            correct_accept = float((pa & correct).float().mean().item())
            wrong = float((pa & ~correct).float().mean().item())
            overall_neg = float(na.float().mean().item()) if len(na) else 0.0
            family_rates = {
                name: float(na[idx].float().mean().item()) for name, idx in neg_family_local.items() if idx
            }
            feasible = (
                wrong <= epsilon_wrong + 1e-12
                and overall_neg <= epsilon_retain + 1e-12
                and all(v <= epsilon_retain + 1e-12 for v in family_rates.values())
            )
            if not feasible:
                continue
            max_family = max(family_rates.values(), default=0.0)
            key = (-correct_accept, max_family, overall_neg, wrong, -tau, -gamma)
            candidate = {
                "tau": tau, "gamma": gamma,
                "correct_accept_rate": correct_accept,
                "wrong_fact_accept_rate": wrong,
                "negative_whole_bank_accept_rate": overall_neg,
                "negative_family_accept_rates": family_rates,
                "positive_top1_correct_rate": float(correct.float().mean().item()),
                "positive_reject_rate": float((~pa).float().mean().item()),
            }
            if best is None or key < best[0]:
                best = (key, candidate)
    if best is None:
        raise RuntimeError("no threshold pair satisfies even reject-all preservation budgets")
    chosen = best[1]
    status = "ACCEPTABLE_OPERATING_POINT" if chosen["correct_accept_rate"] >= min_correct_accept else "NO_ACCEPTABLE_OPERATING_POINT"
    chosen.update({
        "status": status,
        "epsilon_whole_bank_negative": epsilon_retain,
        "epsilon_wrong_fact": epsilon_wrong,
        "minimum_useful_correct_accept": min_correct_accept,
        "selection_rule": "maximize correct acceptance subject to overall + every negative-family budget and wrong-fact budget",
    })
    return float(chosen["tau"]), float(chosen["gamma"]), chosen


def evaluate_model(
    name: str, calib_rows: Sequence[QueryRow], calib_scores: torch.Tensor, calib_none: torch.Tensor,
    val_rows: Sequence[QueryRow], val_scores: torch.Tensor, val_none: torch.Tensor,
    dev_direct_rows: Sequence[QueryRow], dev_direct_scores: torch.Tensor, dev_direct_none: torch.Tensor,
    dev_para_rows: Sequence[QueryRow], dev_para_scores: torch.Tensor, dev_para_none: torch.Tensor,
    *, epsilon_retain: float, epsilon_wrong: float, min_calib_correct_accept: float,
    calib_mask: torch.Tensor | None = None, val_mask: torch.Tensor | None = None,
    direct_mask: torch.Tensor | None = None, para_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    tau, gamma, calibration = calibrate(
        calib_rows, calib_scores, calib_none,
        epsilon_retain=epsilon_retain, epsilon_wrong=epsilon_wrong,
        min_correct_accept=min_calib_correct_accept, candidate_mask=calib_mask,
    )
    vpos, vneg, vps, vpn, vns, vnn, vpm, vnm = split_pos_neg(val_rows, val_scores, val_none, val_mask)
    val_positive = positive_report(vps, vpn, vpos, tau, gamma, vpm)
    val_negative_overall = negative_report(vns, vnn, tau, gamma, vnm)
    family_reports = {}
    neg_local = [(i, row) for i, row in enumerate(vneg)]
    for fam in sorted({family(row.kind) for row in vneg}):
        idx = [i for i, row in neg_local if family(row.kind) == fam]
        fam_mask = vnm[idx] if vnm is not None else None
        family_reports[fam] = negative_report(vns[idx], vnn[idx], tau, gamma, fam_mask)

    pilot_pass = (
        calibration["status"] == "ACCEPTABLE_OPERATING_POINT"
        and val_positive["top1_correct_rate_pct"] >= 70.0
        and val_positive["correct_accept_rate_pct"] >= 60.0
        and val_positive["wrong_fact_accept_rate_pct"] <= 2.0 + 1e-9
        and val_negative_overall["whole_bank_false_activation_rate_pct"] <= 2.0 + 1e-9
        and all((r["whole_bank_false_activation_rate_pct"] or 0.0) <= 2.0 + 1e-9 for r in family_reports.values())
    )
    return {
        "name": name,
        "calibration": calibration,
        "validation": {
            "positive_safe_view": val_positive,
            "negative_overall": val_negative_overall,
            "negative_families": family_reports,
            "pilot_gate": {
                "pass": bool(pilot_pass),
                "criteria": {
                    "top1_correct_pct_min": 70.0,
                    "correct_accept_pct_min": 60.0,
                    "wrong_accept_pct_max": 2.0,
                    "overall_negative_accept_pct_max": 2.0,
                    "each_negative_family_accept_pct_max": 2.0,
                },
            },
        },
        "development_only_official_seed1": {
            "direct_forget": positive_report(dev_direct_scores, dev_direct_none, dev_direct_rows, tau, gamma, direct_mask),
            "forget_paraphrase": positive_report(dev_para_scores, dev_para_none, dev_para_rows, tau, gamma, para_mask),
            "note": "Already-inspected Seed-1 official probes; descriptive development evidence only, not model selection or confirmation.",
        },
    }


def official_dev_rows(forget: Sequence[Mapping[str, Any]]) -> tuple[list[QueryRow], list[QueryRow]]:
    direct, para = [], []
    for i, row in enumerate(forget):
        direct.append(QueryRow(render(row), i, "official_direct_development"))
        for p in row.get("paraphrase_prompts", []):
            para.append(QueryRow(str(p), i, "official_paraphrase_development"))
    return direct, para


def score_all_texts_pooled(model, tokenizer, rows_by_name: Mapping[str, Sequence[QueryRow]], device, batch_size):
    return {name: encode_pooled(model, tokenizer, [r.text for r in rows], device, batch_size) for name, rows in rows_by_name.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--epsilon-retain", type=float, default=0.02)
    p.add_argument("--epsilon-wrong", type=float, default=0.02)
    p.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    args = p.parse_args()

    if not (0 <= args.epsilon_retain <= 1 and 0 <= args.epsilon_wrong <= 1):
        p.error("epsilon budgets must be in [0,1]")
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    data = json.loads(Path(args.mcf_path).read_text(encoding="utf-8"))
    forget, retain = sample_official_mcf_records(data, FORGET_NUM, RETAIN_NUM, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget]
    retain = [off.normalize_record(x) for x in retain]
    view_map, view_meta = multiview.load_view_corpus(Path(args.view_corpus))
    expected = {int(x["case_id"]) for x in forget}
    if set(map(int, view_map.keys())) != expected:
        raise RuntimeError("view corpus case IDs do not exactly match Seed-1 forget50")
    facts = split_fact_views(forget, view_map)
    bank_pairs = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    retain_fit, retain_calib, retain_val = split_retain(retain, bank_pairs)
    fit_rows = build_phase_rows(facts, retain_fit, "fit")
    calib_rows = build_phase_rows(facts, retain_calib, "calib")
    val_rows = build_phase_rows(facts, retain_val, "validation")
    dev_direct_rows, dev_para_rows = official_dev_rows(forget)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype_from_name(args.dtype), local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval(); model.config.use_cache = False
    for param in model.parameters(): param.requires_grad_(False)

    row_sets = {
        "fit": fit_rows, "calib": calib_rows, "validation": val_rows,
        "dev_direct": dev_direct_rows, "dev_para": dev_para_rows,
    }
    pooled = score_all_texts_pooled(model, tokenizer, row_sets, device, args.encode_batch_size)
    subject_texts = [str(facts[i]["subject"]) for i in sorted(facts)]
    relation_texts = [str(facts[i]["relation_text"]) for i in sorted(facts)]
    subject_emb = encode_pooled(model, tokenizer, subject_texts, device, args.encode_batch_size)
    relation_emb = encode_pooled(model, tokenizer, relation_texts, device, args.encode_batch_size)

    anchors, protos = build_diag_state(model, tokenizer, facts, device, args.encode_batch_size)
    diag = train_diagonal(pooled["fit"], fit_rows, anchors, protos, device)
    diag_scores = {name: score_diagonal(diag, pooled[name], anchors, protos, device) for name in row_sets}

    subj_scores = {}
    subj_masks = {}
    for name, rows in row_sets.items():
        s, n = diag_scores[name]
        cs, cn, cm = apply_subject_control(s, n, [r.text for r in rows], facts)
        subj_scores[name] = (cs, cn); subj_masks[name] = cm

    pooled_matcher = train_pooled_matcher(pooled["fit"], fit_rows, subject_emb, relation_emb, device)
    pooled_scores = {name: score_pooled_matcher(pooled_matcher, pooled[name], subject_emb, relation_emb, device) for name in row_sets}

    token_states = {
        name: encode_token_states(model, tokenizer, [r.text for r in rows], device, args.encode_batch_size)
        for name, rows in row_sets.items()
    }
    token_matcher = train_token_matcher(token_states["fit"], fit_rows, subject_emb, relation_emb, device)
    token_scores = {name: score_token_matcher(token_matcher, token_states[name], subject_emb, relation_emb, device) for name in row_sets}

    common = dict(
        calib_rows=calib_rows, val_rows=val_rows,
        dev_direct_rows=dev_direct_rows, dev_para_rows=dev_para_rows,
        epsilon_retain=float(args.epsilon_retain), epsilon_wrong=float(args.epsilon_wrong),
        min_calib_correct_accept=float(args.min_calib_correct_accept),
    )
    results = {}
    results["existing_diagonal"] = evaluate_model(
        "existing_diagonal", calib_scores=diag_scores["calib"][0], calib_none=diag_scores["calib"][1],
        val_scores=diag_scores["validation"][0], val_none=diag_scores["validation"][1],
        dev_direct_scores=diag_scores["dev_direct"][0], dev_direct_none=diag_scores["dev_direct"][1],
        dev_para_scores=diag_scores["dev_para"][0], dev_para_none=diag_scores["dev_para"][1], **common,
    )
    results["explicit_subject_control"] = evaluate_model(
        "explicit_subject_control", calib_scores=subj_scores["calib"][0], calib_none=subj_scores["calib"][1],
        val_scores=subj_scores["validation"][0], val_none=subj_scores["validation"][1],
        dev_direct_scores=subj_scores["dev_direct"][0], dev_direct_none=subj_scores["dev_direct"][1],
        dev_para_scores=subj_scores["dev_para"][0], dev_para_none=subj_scores["dev_para"][1],
        calib_mask=subj_masks["calib"], val_mask=subj_masks["validation"],
        direct_mask=subj_masks["dev_direct"], para_mask=subj_masks["dev_para"], **common,
    )
    results["shared_pooled_matcher"] = evaluate_model(
        "shared_pooled_matcher", calib_scores=pooled_scores["calib"][0], calib_none=pooled_scores["calib"][1],
        val_scores=pooled_scores["validation"][0], val_none=pooled_scores["validation"][1],
        dev_direct_scores=pooled_scores["dev_direct"][0], dev_direct_none=pooled_scores["dev_direct"][1],
        dev_para_scores=pooled_scores["dev_para"][0], dev_para_none=pooled_scores["dev_para"][1], **common,
    )
    results["candidate_conditioned_token_matcher"] = evaluate_model(
        "candidate_conditioned_token_matcher", calib_scores=token_scores["calib"][0], calib_none=token_scores["calib"][1],
        val_scores=token_scores["validation"][0], val_none=token_scores["validation"][1],
        dev_direct_scores=token_scores["dev_direct"][0], dev_direct_none=token_scores["dev_direct"][1],
        dev_para_scores=token_scores["dev_para"][0], dev_para_none=token_scores["dev_para"][1], **common,
    )

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_recognition_only_router_benchmark",
        "recognition_only": True,
        "output_intervention_applied": False,
        "quotient_applied": False,
        "seed": SEED,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "data_contract": {
            "view_corpus": str(Path(args.view_corpus).resolve()),
            "view_corpus_meta": view_meta,
            "fit_views_per_fact": 3,
            "calibration_views_per_fact": 1,
            "validation_views_per_fact": 1,
            "wording_family_split_guaranteed": False,
            "note_on_split": "Five-view corpus does not expose verified paraphrase-family labels; individual views are disjoint but family-level independence is not claimed.",
            "official_paraphrases_used_for_fit": False,
            "official_paraphrases_used_for_calibration": False,
            "official_paraphrases_used_for_model_selection": False,
            "official_seed1_probes_status": "development evidence only; previously inspected",
            "target_new_read_by_router": False,
            "target_true_use": "metadata only for retain same-answer-negative tagging; never input to a matcher",
            "relation_descriptor": "dataset canonical relation prompt with subject slot filled by literal ENTITY; no answer",
            "retain_split_counts": {"fit": len(retain_fit), "calibration": len(retain_calib), "validation": len(retain_val)},
            "query_counts": {k: len(v) for k, v in row_sets.items()},
        },
        "preservation_first_budgets": {
            "whole_bank_negative_accept_max": float(args.epsilon_retain),
            "each_negative_family_accept_max": float(args.epsilon_retain),
            "wrong_fact_accept_max": float(args.epsilon_wrong),
            "minimum_calibration_correct_accept": float(args.min_calib_correct_accept),
        },
        "models": results,
    }
    (outdir / "recognition_benchmark.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    compact = {}
    for name, result in results.items():
        compact[name] = {
            "calibration_status": result["calibration"]["status"],
            "val_top1": result["validation"]["positive_safe_view"]["top1_correct_rate_pct"],
            "val_correct_accept": result["validation"]["positive_safe_view"]["correct_accept_rate_pct"],
            "val_wrong_accept": result["validation"]["positive_safe_view"]["wrong_fact_accept_rate_pct"],
            "val_negative_accept": result["validation"]["negative_overall"]["whole_bank_false_activation_rate_pct"],
            "pilot_pass": result["validation"]["pilot_gate"]["pass"],
            "dev_official_para_top1": result["development_only_official_seed1"]["forget_paraphrase"]["top1_correct_rate_pct"],
            "dev_official_para_correct_accept": result["development_only_official_seed1"]["forget_paraphrase"]["correct_accept_rate_pct"],
        }
    print(json.dumps({"recognition_only": True, "models": compact, "output_dir": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
