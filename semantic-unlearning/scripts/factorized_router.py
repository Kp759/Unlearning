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

    def add(prompt, subject_key, relation, split, kind):
        prompt = prompt.strip()
        if not prompt or prompt in banned or prompt in seen:
            return False
        seen.add(prompt)
        sub = subject_index.get(subject_key, none) if subject_key is not None else none
        items.append({"prompt": prompt, "subject": sub, "relation": relation, "split": split,
                      "kind": kind,
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
        add(prompt, key, relation, split, kind)
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
        texts += rng.sample(list(record.get("neighborhood_prompts", [])),
                            min(2, len(record.get("neighborhood_prompts", []))))
        texts += rng.sample(list(record.get("attribute_prompts", [])),
                            min(2, len(record.get("attribute_prompts", []))))
        added = 0
        for text in texts[:max_record_prompts + 1]:
            if mentions_forget(text):
                continue
            if add(text, None, relation, part, kind):
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
    def __init__(self, state):
        super().__init__()
        self.layer = int(state["layer"])
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

    @torch.no_grad()
    def route(self, base_model, input_ids, attention_mask):
        """Run the (unedited) base model on right-padded prompts; decide at the last token."""
        store = {}
        handle = base_model.model.layers[self.layer].register_forward_hook(
            lambda m, a, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        try:
            base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        finally:
            handle.remove()
        last = attention_mask.sum(1) - 1
        h = store["h"][torch.arange(input_ids.shape[0], device=input_ids.device), last]
        return self.decide(h.float()).tolist()


def decide_from_probabilities(ps, pr, fact_subject, fact_relation, t_s, t_r):
    s = ps[:, fact_subject]                       # [N, F]
    r = pr[:, fact_relation]
    ok = (s >= t_s) & (r >= t_r)
    score = (s * r).masked_fill(~ok, -1.0)
    best = score.argmax(-1)
    return torch.where(ok.any(-1), best, torch.full_like(best, -1))


# ---------------------------------------------------------------------- calibration
def calibrate(ps, pr, target, fact_subject, fact_relation, kinds, min_recall=0.95, grid=None):
    """(t_s, t_r) on validation: fewest false fires (mean of other-subject and same-subject
    negatives) subject to macro-over-facts recall >= min_recall; ties -> higher thresholds.
    If no pair reaches the recall floor, the highest recall wins."""
    grid = grid if grid is not None else torch.cat([torch.tensor([0.0]), torch.logspace(-3, 0, 40)[:-1],
                                                    torch.linspace(0.5, 0.98, 25)]).unique()
    target = torch.as_tensor(target)
    pos = target >= 0
    same_subject = torch.tensor(["subject_other_relation" in k for k in kinds]) & ~pos
    other = ~pos & ~same_subject
    n_facts = len(fact_subject)
    pos_per = torch.bincount(target[pos], minlength=n_facts).float()
    has = pos_per > 0
    best = None
    for t_s in grid.tolist():
        for t_r in grid.tolist():
            d = decide_from_probabilities(ps, pr, fact_subject, fact_relation, t_s, t_r)
            correct = (d == target) & pos
            cor_per = torch.bincount(target[correct], minlength=n_facts).float()
            recall = float((cor_per[has] / pos_per[has]).mean())
            ff_same = float((d[same_subject] >= 0).float().mean()) if same_subject.any() else 0.0
            ff_other = float((d[other] >= 0).float().mean()) if other.any() else 0.0
            ff = 0.5 * (ff_same + ff_other)
            met = recall >= min_recall
            key = (met, -ff if met else recall, t_s + t_r)
            if best is None or key > best[0]:
                best = (key, {"t_subject": t_s, "t_relation": t_r, "recall": recall,
                              "false_fire_same_subject": ff_same, "false_fire_other": ff_other,
                              "recall_floor_met": met})
    return best[1]


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
