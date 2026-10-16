#!/usr/bin/env python3
"""Factorized subject x relation router: both learned, both linear (softmax) heads.

A forget fact is a (subject, relation) pair. Instead of 50 per-fact heads that must
learn "this subject AND this question" from ~10 prompts each, two heads read the same
pass-1 state (base model, layer L, last prompt token, L2-normalised and centred):

  subject head    softmax over the forget subjects + "none"
                  learned (no exact token match): typos, case and unseen wordings of
                  a forget subject still map to it; other entities -> "none"
  relation head   softmax over the MCF relations (34), learned from thousands of
                  prompts of OTHER subjects, so "which question is asked" is one shared
                  skill rather than something each fact learns from a handful of prompts

Fire fact f  iff  p_subj(subject_f) >= t_s  and  p_rel(relation_f) >= t_r
(several facts: the largest p_subj * p_rel). Nothing fires -> no edit (exact base).
t_s, t_r are calibrated on held-out validation prompts built from training data only.

Data (MCF, seed-dependent forget sample; held-out evaluation strings are never used):
  forget positives     each fact's training-visible prompts (association examples)
                       + its subject in the templates of the 'preserve' half of its
                       other relations (mcf_subject_collateral.other_relations)
                       + typo / lowercase / random-prefix augmentations
  non-forget pool      rewrite, paraphrase, neighbourhood and attribute prompts of the
                       other ~20k MCF records (subject "none", relation = record's)
  near-miss names      records whose subject shares a word with a forget subject
                       ("Belgium national football team" vs "Belgium"); split by name
                       into train / validation / evaluation thirds
Excluded from all training: every official held-out prompt of the forget records
(paraphrase / neighbourhood / generation / attribute), every template of the
held-out ('eval') relations for a forget subject, and every non-forget prompt that
contains a forget subject's name.

    python scripts/factorized_router.py attach --router-dir FR --rows-from RUN --output-dir OUT
attaches a fitted router to a trained early-write run (its Δe rows) for the evaluators.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import zlib

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = "factorized_subject_relation_router_v1"
NONE = "<none>"
KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"


# --------------------------------------------------------------------------- strings
def norm(s):
    return " ".join(str(s).casefold().split())


def _rw(record):
    rw = record["requested_rewrite"]
    return rw[0] if isinstance(rw, list) else rw


def stable_hash(s):
    return zlib.crc32(str(s).encode("utf-8"))


def typo(word, rng):
    """One character edit inside a letter run (swap / drop / duplicate / substitute)."""
    spots = [i for i, ch in enumerate(word) if ch.isalpha()]
    if len(spots) < 4:
        return word
    i = rng.choice(spots[1:-1])
    op = rng.choice(("swap", "drop", "dup", "sub"))
    if op == "swap" and i + 1 < len(word) and word[i + 1].isalpha() and word[i + 1] != word[i]:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "dup":
        return word[:i] + word[i] + word[i:]
    choices = [c for c in KEYBOARD if c != word[i].lower()]
    sub = rng.choice(choices)
    return word[:i] + (sub.upper() if word[i].isupper() else sub) + word[i + 1:]


def misspell(subject, rng):
    """One typo in the subject's longest word (>= 4 letters)."""
    words = subject.split()
    k = max(range(len(words)), key=lambda i: len(words[i]))
    if len(words[k]) < 4:
        return subject
    words[k] = typo(words[k], rng)
    return " ".join(words)


def replace_subject(prompt, subject, new):
    k = prompt.find(subject)
    if k < 0:
        k = prompt.casefold().find(subject.casefold())
    if k < 0 or new == subject:
        return None
    return prompt[:k] + new + prompt[k + len(subject):]


def prefix_bank(records, forget_subjects, limit=2000):
    """Unrelated lead-in sentences: the text before the subject in MCF paraphrase prompts."""
    out = []
    for record in records:
        rw = _rw(record)
        if norm(rw["subject"]) in forget_subjects:
            continue
        for p in record.get("paraphrase_prompts", []):
            k = p.find(rw["subject"])
            lead = p[:k].strip() if k > 0 else ""
            if 10 <= len(lead) <= 200 and not any(s in lead.casefold() for s in forget_subjects):
                out.append(lead)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------- held-out sets
def forget_heldout_strings(forget_records, ranked_templates):
    """Every string the official / collateral evaluation can use (never trained on)."""
    from mcf_subject_collateral import other_relations

    banned = set()
    for record in forget_records:
        rw = _rw(record)
        for key in ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts",
                    "attribute_prompts"):
            banned.update(p.strip() for p in record.get(key, []))
        for rel in other_relations(rw["relation_id"], ranked_templates, "eval"):
            banned.update(t.format(rw["subject"]).strip() for t in ranked_templates[rel])
    return banned


def near_miss_subjects(records, forget_subjects, max_df=15):
    """Non-forget subjects sharing a DISTINCTIVE word (>= 4 letters, in <= max_df MCF
    subjects) with a forget subject: "Belgium national football team" for "Belgium",
    not every "Charles" for "Charles Darwin". name -> train / val / eval third."""
    from collections import Counter

    def words(s):
        return set(re.findall(r"[^\W\d_]{4,}", s))

    all_subjects = {norm(_rw(r)["subject"]) for r in records}
    df = Counter(w for s in all_subjects for w in words(s))
    distinctive = {w for s in forget_subjects for w in words(s) if df[w] <= max_df}
    out = {}
    for key in all_subjects:
        if key in forget_subjects:
            continue
        if distinctive & words(key):
            out[key] = ("train", "val", "eval")[stable_hash(key) % 3]
    return out


def near_miss_eval_prompts(records, forget_subjects):
    near = near_miss_subjects(records, forget_subjects)
    out = []
    for record in records:
        rw = _rw(record)
        if near.get(norm(rw["subject"])) == "eval":
            for text in [rw["prompt"].format(rw["subject"])] + list(record.get("paraphrase_prompts", []))[:1]:
                out.append((rw["subject"], text.strip(), rw["target_true"]["str"]))
    return out


def robustness_probes(records, forget_records, prompts, seed=99991):
    """Evaluation-only robustness sets (never trained on):
    paraphrase_typo / paraphrase_lower  official paraphrases with the subject misspelt /
                                        lower-cased (should still route to their fact)
    near_miss                           prompts of held-out near-miss names (should not fire)"""
    rng = random.Random(seed)
    forget_subjects = {norm(_rw(r)["subject"]) for r in forget_records}
    out = []
    for p in prompts:
        if p["set"] != "paraphrase":
            continue
        t = replace_subject(p["prompt"], p["subject"], misspell(p["subject"], rng))
        if t:
            out.append({**p, "set": "paraphrase_typo", "prompt": t})
        low = replace_subject(p["prompt"], p["subject"], p["subject"].lower())
        if low:
            out.append({**p, "set": "paraphrase_lower", "prompt": low})
    for subject, text, answer in near_miss_eval_prompts(records, forget_subjects):
        out.append({"case_id": -1, "set": "near_miss", "subject": subject, "prompt": text,
                    "answer": answer})
    return out


# ---------------------------------------------------------------------- training data
def build_router_data(records, forget_records, facts, examples, *, seed=1, per_relation=800,
                      val_per_relation=100, max_record_prompts=6, augment=True):
    """Items {prompt, subject (class index, last = none), relation, split, kind, target}.

    target = index of the forget fact whose (subject, relation) the item asks, else -1."""
    from mcf_subject_collateral import other_relations, ranked_relation_templates_from_records

    rng = random.Random(1000 + seed)
    ranked = ranked_relation_templates_from_records(records)
    subjects = []
    for fact in facts:
        if norm(fact["subject"]) not in subjects:
            subjects.append(norm(fact["subject"]))
    subject_index = {s: i for i, s in enumerate(subjects)}
    none = len(subjects)
    fact_key = {(norm(f["subject"]), str(f["relation"])): i for i, f in enumerate(facts)}
    forget_ids = {int(r["case_id"]) for r in forget_records}
    forget_subjects = set(subjects)
    banned = forget_heldout_strings(forget_records, ranked)
    banned.update(text for _, text, _ in near_miss_eval_prompts(records, forget_subjects))
    near = near_miss_subjects(records, forget_subjects)
    prefixes = prefix_bank(records, forget_subjects)
    items, seen = [], set()

    def add(prompt, subject_key, relation, split, kind, subject_text=None):
        prompt = prompt.strip()
        if not prompt or prompt in banned or prompt in seen:
            return False
        seen.add(prompt)
        sub = subject_index.get(subject_key, none) if subject_key is not None else none
        items.append({"prompt": prompt, "subject": sub, "relation": relation, "split": split,
                      "kind": kind, "subject_text": subject_text,
                      "target": fact_key.get((subject_key, relation), -1) if sub != none else -1})
        return True

    def mentions_forget(text):
        t = text.casefold()
        return any(s in t for s in forget_subjects)

    # forget positives: own training-visible prompts + preserve-relation templates
    positives = []
    for e in examples:
        fid = str(e["fact_id"])
        index = next((i for i, f in enumerate(facts) if str(f["id"]) == fid), None)
        if index is None:
            continue
        f = facts[index]
        split = "train" if str(e.get("split")) == "train" else "val"
        positives.append((str(e["prompt"]), f["subject"], str(f["relation"]), split, "own"))
    for f in facts:
        relation = str(f["relation"])
        for other in other_relations(relation, ranked, "preserve"):
            for rank, template in enumerate(ranked[other][:4]):
                positives.append((template.format(f["subject"]), f["subject"], other,
                                  "train" if rank % 2 == 0 else "val", "subject_other_relation"))
    for prompt, subject, relation, split, kind in positives:
        key = norm(subject)
        add(prompt, key, relation, split, kind, subject_text=subject)
        if not augment:
            continue
        variants = []
        variants.append((replace_subject(prompt, subject, misspell(subject, rng)), kind + "+typo"))
        variants.append((replace_subject(prompt, subject, subject.lower()), kind + "+lower"))
        if prefixes:
            variants.append((rng.choice(prefixes) + " " + prompt, kind + "+prefix"))
        for text, vkind in variants:
            if text:
                add(text, key, relation, split, vkind)

    # non-forget pool and near-miss names (subject "none")
    per_rel = {}
    order = list(range(len(records)))
    rng.shuffle(order)
    for idx in order:
        record = records[idx]
        if int(record["case_id"]) in forget_ids:
            continue
        rw = _rw(record)
        key = norm(rw["subject"])
        if key in forget_subjects:
            continue
        relation = str(rw["relation_id"])
        if key in near:
            part = near[key]
            if part == "eval":
                continue
            kind = "near_miss"
        else:
            part = "val" if stable_hash(record["case_id"]) % 10 == 0 else "train"
            kind = "other_subject"
        cap = per_relation if part == "train" else val_per_relation
        if kind == "other_subject" and per_rel.get((relation, part), 0) >= cap:
            continue
        texts = [rw["prompt"].format(rw["subject"])]
        texts += list(record.get("paraphrase_prompts", []))[:2]
        own_subject = len(texts)                     # these name the record's own subject
        texts += rng.sample(list(record.get("neighborhood_prompts", [])),
                            min(2, len(record.get("neighborhood_prompts", []))))
        texts += rng.sample(list(record.get("attribute_prompts", [])),
                            min(2, len(record.get("attribute_prompts", []))))
        added = 0
        for j, text in enumerate(texts[:max_record_prompts + 1]):
            if mentions_forget(text):
                continue
            if add(text, None, relation, part, kind,
                   subject_text=rw["subject"] if j < own_subject else None):
                added += 1
                if augment and kind == "other_subject" and prefixes and rng.random() < 0.15:
                    add(rng.choice(prefixes) + " " + text, None, relation, part, kind + "+prefix")
        per_rel[(relation, part)] = per_rel.get((relation, part), 0) + added
    relations = sorted({it["relation"] for it in items})
    return {"items": items, "subjects": subjects, "relations": relations,
            "fact_subject": [subject_index[norm(f["subject"])] for f in facts],
            "fact_relation": [relations.index(str(f["relation"])) for f in facts],
            "banned_strings": len(banned), "near_miss_subjects": len(near)}


# ---------------------------------------------------------------------- features
@torch.no_grad()
def extract_last_token(model, tok, prompts, layers, batch_size=64, device="cuda"):
    """{layer: [N, d] float32}: block-output state at the last prompt token (BOS included,
    exactly as the runtime tokenises a prompt)."""
    store, handles = {}, []
    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(
            lambda m, a, o, layer=layer: store.__setitem__(layer, o[0] if isinstance(o, tuple) else o)))
    out = {layer: [] for layer in layers}
    try:
        for start in range(0, len(prompts), batch_size):
            enc = [tok(p)["input_ids"] for p in prompts[start:start + batch_size]]
            width = max(map(len, enc))
            ids = torch.full((len(enc), width), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e)
                att[i, :len(e)] = 1
            ids, att = ids.to(device), att.to(device)
            model(input_ids=ids, attention_mask=att, use_cache=False)
            last = att.sum(1) - 1
            for layer in layers:
                out[layer].append(store[layer][torch.arange(len(enc), device=device), last].float().cpu())
    finally:
        for h in handles:
            h.remove()
    return {layer: torch.cat(v) for layer, v in out.items()}


def feature_map(h, mean):
    return F.normalize(h.float(), dim=-1) - mean


@torch.no_grad()
def extract_states(model, tok, prompts, last_layers, span_layers, keep_span=None, batch_size=32,
                   device="cuda"):
    """One pass: last-token states at `last_layers` (all prompts) and EVERY prompt position
    except BOS at `span_layers` (prompts with keep_span[i]), packed:
        last[L]  [N, d] float32
        span[L]  {"flat": [T, d] float16, "offsets": [M+1], "rows": [M] prompt indices}"""
    layers = sorted(set(last_layers) | set(span_layers))
    keep_span = [True] * len(prompts) if keep_span is None else list(keep_span)
    store, handles = {}, []
    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(
            lambda m, a, o, layer=layer: store.__setitem__(layer, o[0] if isinstance(o, tuple) else o)))
    last = {L: [] for L in last_layers}
    flat = {L: [] for L in span_layers}
    lengths, rows = [], []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            enc = [tok(p)["input_ids"] for p in batch]
            width = max(map(len, enc))
            ids = torch.full((len(enc), width), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e)
                att[i, :len(e)] = 1
            ids, att = ids.to(device), att.to(device)
            model(input_ids=ids, attention_mask=att, use_cache=False)
            end = att.sum(1) - 1
            for L in last_layers:
                last[L].append(store[L][torch.arange(len(enc), device=device), end].float().cpu())
            for i, e in enumerate(enc):
                if not keep_span[start + i]:
                    continue
                rows.append(start + i)
                lengths.append(max(len(e) - 1, 1))
                for L in span_layers:
                    h = store[L][i, 1:len(e)] if len(e) > 1 else store[L][i, :1]
                    flat[L].append(h.to(torch.float16).cpu())
    finally:
        for h in handles:
            h.remove()
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.long)
    offsets[1:] = torch.tensor(lengths).cumsum(0)
    span = {L: {"flat": torch.cat(flat[L]), "offsets": offsets, "rows": torch.tensor(rows)}
            for L in span_layers}
    return {L: torch.cat(v) for L, v in last.items()}, span


def _padded(flat, offsets, idx, device):
    """[B, Lmax, d] float32 + [B, Lmax] mask for the span rows idx."""
    lens = (offsets[idx + 1] - offsets[idx]).tolist()
    Lmax = max(lens)
    X = torch.zeros(len(idx), Lmax, flat.shape[1], device=device)
    mask = torch.zeros(len(idx), Lmax, dtype=torch.bool, device=device)
    for j, (i, n) in enumerate(zip(idx.tolist(), lens)):
        X[j, :n] = flat[offsets[i]:offsets[i] + n].to(device).float()
        mask[j, :n] = True
    return X, mask


def span_mean(flat, offsets, idx, chunk=4096):
    """Mean of L2-normalised position states over the span rows idx."""
    total, count = None, 0
    for start in range(0, len(idx), chunk):
        part = idx[start:start + chunk]
        segs = torch.cat([flat[offsets[i]:offsets[i + 1]] for i in part.tolist()]).float()
        v = F.normalize(segs, dim=-1).sum(0)
        total = v if total is None else total + v
        count += segs.shape[0]
    return total / max(count, 1)


def position_features(X, mask, mean, pair=False):
    """phi(h_t) or, with pair, [phi(h_t); phi(h_{t+1})] (zeros when t is the last position).

    Attention is causal: at any layer the state of "Belgium" in "Belgium national
    football team" equals that of a standalone "Belgium". Pairing each token with the
    next one lets the scorer see whether the name continues."""
    phi = F.normalize(X, dim=-1) - mean
    if not pair:
        return phi
    nxt = torch.zeros_like(phi)
    nxt[:, :-1] = phi[:, 1:] * mask[:, 1:, None]
    return torch.cat([phi, nxt], dim=-1)


def mil_pool(X, mask, W, b, mean, tau=None):
    """Per-subject logit of each prompt = max over its positions (tau: smooth max).
    W [S, d] scores single tokens; W [S, 2d] scores (token, next token) pairs."""
    pair = W.shape[1] == 2 * X.shape[-1]
    Z = position_features(X, mask, mean, pair) @ W.T + b              # [B, L, S]
    Z = Z.masked_fill(~mask[..., None], float("-inf"))
    if tau is None:
        return Z.amax(1)
    return tau * torch.logsumexp(Z / tau, dim=1)


@torch.no_grad()
def mil_logits(span, idx, W, b, mean, device="cuda", batch=256):
    out = []
    W, b, mean = W.to(device), b.to(device), mean.to(device)
    for start in range(0, len(idx), batch):
        X, mask = _padded(span["flat"], span["offsets"], idx[start:start + batch], device)
        out.append(mil_pool(X, mask, W, b, mean).cpu())
    return torch.cat(out)


def fit_mil_subject(span, train_idx, val_idx, y, n_subjects, *, l2=1e-4, epochs=8, lr=2e-3,
                    batch=256, tau=0.25, max_pos_weight=50.0, device="cuda", seed=0, pair=True,
                    hard_rounds=0, hard_fraction=0.1, hard_weight=5.0, hard_epochs=None):
    """Multiple-instance subject detector: one-vs-rest logistic per subject on the
    max-pooled position scores (smooth max while training). y[i] = subject or -1 (none).

    hard_rounds: after fitting, the hard_fraction of training "none" prompts with the
    highest score for any forget subject (related entities, near-miss names) get weight
    hard_weight and training continues for hard_epochs (default epochs // 2); repeated
    hard_rounds times. The validation loss (unweighted) still picks the checkpoint."""
    g = torch.Generator().manual_seed(seed)
    mean = span_mean(span["flat"], span["offsets"], train_idx).to(device)
    d = span["flat"].shape[1]
    W = (torch.randn(n_subjects, 2 * d if pair else d, generator=g) * 0.01).to(device).requires_grad_(True)
    b = torch.full((n_subjects,), -2.0, device=device, requires_grad=True)
    Y = torch.zeros(len(y), n_subjects)
    pos = y >= 0
    Y[pos.nonzero().flatten(), y[pos]] = 1.0
    n_pos = Y[train_idx].sum(0).clamp_min(1)
    pos_weight = ((len(train_idx) - n_pos) / n_pos).clamp(max=max_pos_weight).to(device)
    sample_w = torch.ones(len(y))
    opt = torch.optim.Adam([W, b], lr=lr)

    def loss_on(idx, pooled, weighted):
        t = Y[idx].to(device)
        per = F.binary_cross_entropy_with_logits(pooled, t, pos_weight=pos_weight, reduction="none").mean(1)
        if weighted:
            w = sample_w[idx].to(device)
            return (per * w).sum() / w.sum()
        return per.mean()

    best, history = None, []

    def run(n_epochs, stage):
        nonlocal best
        for epoch in range(n_epochs):
            order = train_idx[torch.randperm(len(train_idx), generator=g)]
            for start in range(0, len(order), batch):
                idx = order[start:start + batch]
                X, mask = _padded(span["flat"], span["offsets"], idx, device)
                loss = loss_on(idx, mil_pool(X, mask, W, b, mean, tau), True) + 0.5 * l2 * (W * W).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                val = float(loss_on(val_idx, mil_logits(span, val_idx, W.detach(), b.detach(), mean,
                                                         device).to(device), False))
            history.append({"stage": stage, "epoch": epoch, "val_loss": val})
            if best is None or val < best[0]:
                best = (val, W.detach().cpu().clone(), b.detach().cpu().clone())

    run(epochs, "fit")
    neg_train = train_idx[y[train_idx] < 0]
    for r in range(int(hard_rounds)):
        with torch.no_grad():
            W.copy_(best[1].to(device))
            b.copy_(best[2].to(device))
            scores = mil_logits(span, neg_train, W.detach(), b.detach(), mean, device).amax(-1)
        k = max(1, int(hard_fraction * len(neg_train)))
        hard = neg_train[scores.topk(k).indices]
        sample_w[hard] = hard_weight
        history.append({"stage": f"hard_round_{r}", "hard_negatives": int(k),
                        "hard_score_min": float(scores.topk(k).values.min())})
        run(hard_epochs or max(1, epochs // 2), f"hard_{r}")
    return best[1], best[2], mean.cpu(), history


# ----------------------------------------------------------------------------- fit
def fit_softmax(X, y, n_classes, l2, device="cuda", max_iter=300):
    """Class-balanced multinomial logistic regression (L-BFGS), returns W [C, d], b [C]."""
    X, y = X.to(device).float(), y.to(device)
    counts = torch.bincount(y, minlength=n_classes).float().clamp_min(1)
    w = (1.0 / counts)[y]
    w = w / w.sum()
    W = torch.zeros(n_classes, X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (F.cross_entropy(X @ W.T + b, y, reduction="none") * w).sum() + 0.5 * l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().cpu(), b.detach().cpu()


def balanced_log_loss(X, y, W, b, n_classes):
    logp = F.log_softmax(X.float() @ W.T + b, -1)
    nll = -logp[torch.arange(len(y)), y]
    per = [nll[y == c].mean() for c in range(n_classes) if (y == c).any()]
    return float(torch.stack(per).mean())


# ---------------------------------------------------------------------- the router
class FactorizedRouter(nn.Module):
    """subject_mode "last_softmax" (v1): both heads on the last-token state at `layer`.
    subject_mode "mil_max" / "mil_max_pair" (v2): subject = max over prompt positions (BOS
    excluded) of a per-subject linear score of the token (and, pair, the next token) at
    `subject_layer`, p = sigmoid; relation as v1 at `layer`."""

    def __init__(self, state):
        super().__init__()
        self.layer = int(state["layer"])
        self.relation_layers = [int(L) for L in state.get("relation_layers", [self.layer])]
        self.subject_mode = str(state.get("subject_mode", "last_softmax"))
        self.subject_layer = int(state.get("subject_layer", self.layer))
        self.register_buffer("subject_mean", torch.as_tensor(
            state.get("subject_mean", state["feature_mean"])).float())
        self.register_buffer("mean", torch.as_tensor(state["feature_mean"]).float())
        self.register_buffer("Ws", torch.as_tensor(state["subject_weight"]).float())
        self.register_buffer("bs", torch.as_tensor(state["subject_bias"]).float())
        self.register_buffer("Wr", torch.as_tensor(state["relation_weight"]).float())
        self.register_buffer("br", torch.as_tensor(state["relation_bias"]).float())
        self.register_buffer("fact_subject", torch.as_tensor(state["fact_subject"]).long())
        self.register_buffer("fact_relation", torch.as_tensor(state["fact_relation"]).long())
        self.t_s = float(state["t_subject"])
        self.t_r = float(state["t_relation"])
        self.state = dict(state)

    def probabilities(self, h):
        phi = feature_map(h, self.mean)
        return F.softmax(phi @ self.Ws.T + self.bs, -1), F.softmax(phi @ self.Wr.T + self.br, -1)

    def decide(self, h, t_s=None, t_r=None):
        """Fact index per row (-1 = no fire)."""
        ps, pr = self.probabilities(h)
        return decide_from_probabilities(ps, pr, self.fact_subject, self.fact_relation,
                                         self.t_s if t_s is None else t_s,
                                         self.t_r if t_r is None else t_r)

    def probabilities_from_states(self, h_last, h_span, span_mask):
        """v2: p_subj from the position states [B, T, d] (mask excludes BOS and padding).
        h_last: [B, d] or, for several relation layers, a list of [B, d] (one per layer)."""
        pr = F.softmax(relation_features(h_last, self.mean) @ self.Wr.T + self.br, -1)
        if self.subject_mode.startswith("mil_max"):
            ps = torch.sigmoid(mil_pool(h_span.float(), span_mask, self.Ws, self.bs, self.subject_mean))
        else:
            ps = F.softmax(feature_map(h_last, self.mean) @ self.Ws.T + self.bs, -1)
        return ps, pr

    @torch.no_grad()
    def route(self, base_model, input_ids, attention_mask):
        """Run the (unedited) base model on right-padded prompts; decide."""
        store, handles = {}, []
        for layer in set(self.relation_layers) | {self.layer, self.subject_layer}:
            handles.append(base_model.model.layers[layer].register_forward_hook(
                lambda m, a, o, layer=layer: store.__setitem__(layer, o[0] if isinstance(o, tuple) else o)))
        try:
            base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        last = attention_mask.sum(1) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        h_last = store[self.layer][rows, last].float()
        if not self.subject_mode.startswith("mil_max"):
            return self.decide(h_last).tolist()
        if len(self.relation_layers) > 1:
            h_last = [store[L][rows, last].float() for L in self.relation_layers]
        mask = attention_mask.bool().clone()
        mask[:, 0] = False
        mask[attention_mask.sum(1) == 1, 0] = True          # a one-token prompt keeps its only token
        ps, pr = self.probabilities_from_states(h_last, store[self.subject_layer], mask)
        return decide_from_probabilities(ps, pr, self.fact_subject, self.fact_relation,
                                         self.t_s, self.t_r).tolist()


def relation_features(h, mean):
    """phi for the relation head: one layer [B, d], or several layers (list of [B, d]),
    each L2-normalised, concatenated, minus the training mean."""
    if isinstance(h, (list, tuple)):
        return torch.cat([F.normalize(x.float(), dim=-1) for x in h], dim=-1) - mean
    return F.normalize(h.float(), dim=-1) - mean


def decide_from_probabilities(ps, pr, fact_subject, fact_relation, t_s, t_r):
    s = ps[:, fact_subject]                       # [N, F]
    r = pr[:, fact_relation]
    ok = (s >= t_s) & (r >= t_r)
    score = (s * r).masked_fill(~ok, -1.0)
    best = score.argmax(-1)
    return torch.where(ok.any(-1), best, torch.full_like(best, -1))


# ---------------------------------------------------------------------- calibration
def calibrate(ps, pr, target, fact_subject, fact_relation, kinds, min_recall=0.95, max_false_fire=0.10,
              grid=None, max_false_fire_same=None, max_false_fire_other=None):
    """(t_s, t_r) on validation. Never an unconstrained fallback:

      recall_floor_met    macro recall >= min_recall AND both false-fire rates (same-subject,
                          other) <= max_false_fire: fewest false fires; ties -> higher thresholds
      false_fire_capped   floor unreachable under the cap: most recall with both false-fire
                          rates <= max_false_fire
      infeasible          neither: the router must not be deployed (status says so)"""
    grid = grid if grid is not None else torch.cat([torch.tensor([0.0]), torch.logspace(-3, 0, 40)[:-1],
                                                    torch.linspace(0.5, 0.98, 25),
                                                    1.0 - torch.logspace(-5, -2, 13)]).unique()
    cap_same = max_false_fire if max_false_fire_same is None else max_false_fire_same
    cap_other = max_false_fire if max_false_fire_other is None else max_false_fire_other
    target = torch.as_tensor(target)
    pos = target >= 0
    same_subject = torch.tensor(["subject_other_relation" in k for k in kinds]) & ~pos
    other = ~pos & ~same_subject
    n_facts = len(fact_subject)
    pos_per = torch.bincount(target[pos], minlength=n_facts).float()
    has = pos_per > 0
    floor, capped = None, None
    for t_s in grid.tolist():
        for t_r in grid.tolist():
            d = decide_from_probabilities(ps, pr, fact_subject, fact_relation, t_s, t_r)
            correct = (d == target) & pos
            cor_per = torch.bincount(target[correct], minlength=n_facts).float()
            recall = float((cor_per[has] / pos_per[has]).mean())
            ff_same = float((d[same_subject] >= 0).float().mean()) if same_subject.any() else 0.0
            ff_other = float((d[other] >= 0).float().mean()) if other.any() else 0.0
            point = {"t_subject": t_s, "t_relation": t_r, "recall": recall,
                     "false_fire_same_subject": ff_same, "false_fire_other": ff_other}
            if ff_same > cap_same or ff_other > cap_other:
                continue
            if recall >= min_recall:
                key = (-0.5 * (ff_same + ff_other), t_s + t_r)
                if floor is None or key > floor[0]:
                    floor = (key, point)
            else:
                key = (recall, -0.5 * (ff_same + ff_other), t_s + t_r)
                if capped is None or key > capped[0]:
                    capped = (key, point)
    if floor is not None:
        return {**floor[1], "status": "recall_floor_met", "recall_floor_met": True,
                "max_false_fire_same": cap_same, "max_false_fire_other": cap_other}
    if capped is not None and capped[1]["recall"] > 0:
        return {**capped[1], "status": "false_fire_capped", "recall_floor_met": False,
                "max_false_fire_same": cap_same, "max_false_fire_other": cap_other}
    return {"t_subject": 1.0 + 1e-6, "t_relation": 1.0 + 1e-6, "recall": 0.0,
            "false_fire_same_subject": 0.0, "false_fire_other": 0.0,
            "status": "infeasible", "recall_floor_met": False,
            "max_false_fire_same": cap_same, "max_false_fire_other": cap_other}


# ------------------------------------------------------------------------ attach
def attach(router_dir, rows_from, output_dir):
    """A run dir = rows_from's artifact (Δe rows, early-write spec, facts) routed by the
    factorized router. Evaluators load it unchanged (linear_router.load_router_artifact)."""
    import shutil

    router_dir, rows_from, output = Path(router_dir), Path(rows_from), Path(output_dir)
    fr = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    run = torch.load(rows_from / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    if [str(f["id"]) for f in fr["facts"]] != [str(f["id"]) for f in run["facts"]]:
        raise ValueError("router and rows were built for different facts")
    art = dict(run)
    art["architecture"] = ARCHITECTURE
    art["factorized_router"] = fr["factorized_router"]
    art["layer"] = int(fr["factorized_router"]["layer"])
    art["rows_from"] = str(rows_from)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(art, output / "fact_association_embeddings.pt")
    manifest = json.loads((rows_from / "association_manifest.json").read_text())
    manifest.update({"architecture": ARCHITECTURE, "factorized_router_from": str(router_dir),
                     "rows_from": str(rows_from)})
    (output / "association_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in ("association_examples.json",):
        if (rows_from / name).is_file():
            shutil.copy2(rows_from / name, output / name)
    print(f"attached {router_dir} -> {output}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attach")
    a.add_argument("--router-dir", required=True)
    a.add_argument("--rows-from", required=True)
    a.add_argument("--output-dir", required=True)
    args = p.parse_args()
    attach(args.router_dir, args.rows_from, args.output_dir)
