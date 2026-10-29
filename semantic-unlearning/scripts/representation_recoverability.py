#!/usr/bin/env python3
"""Representation recoverability: can the forgotten answer still be read linearly from the
hidden states of the unlearned system? One run dir of the paper pipeline (base, ours, genie or a
baseline), the same prompts, routing and request boundary as the official evaluators.

Why. Gao et al. (2026, arXiv:2604.08271, "An Illusion of Unlearning? Assessing Machine Unlearning
Through Internal Representations") show that output-level forgetting is often a classifier-level
artefact: a linear probe on the frozen hidden features of the unlearned model recovers the
forgotten classes at near-original accuracy ("feature-classifier misalignment"), so they ask for
representation-level evaluation next to the output metrics. SURE adds one residual row at the
write layer when the router fires; every block below the write is exactly the base model and the
blocks above see the edited state. Eff / Gen / Spe only look at the output, so this script
measures what a linear readout of the residual stream still finds, layer by layer.

What is measured. For every forget probe (the official forget prompts: MCF rewrite + paraphrases,
ZsRE rewrite + rephrase, MQuAKE rewrite + atomic question, RWKU same-50 and the held-out Level 1 /
Level 2 / paraphrase probes about the target people) the system runs as under the evaluators
(prompt text, boundary = the prompt, router or genie deciding, baseline edit applied) and the
residual stream at the last prompt token is recorded after every block. Three linear readouts
decode the first token of the forgotten answer at every layer:

  lens   logit lens: the model's own final norm + unembedding applied to the layer's state
         (training-free; Patil et al. 2023 use it to extract "deleted" answers).
  tuned  an affine map h_l -> h_final fitted by ridge regression on every token position of the
         dataset's retain-side prompts (MCF / ZsRE / MQuAKE retain records and neighbourhood
         prompts, RWKU neighbour probes; no forget prompt, no forget label) under the SAME
         unlearned system, then norm + unembedding (a tuned lens, Belrose et al. 2023). Fitted on
         prompts of the same kind as the probes so the map is defined where it is applied (the
         data/wikidata PPL fixture has 10 distinct sentences and spans a 166-dim subspace: useless
         here). Features are standardised per dimension, the penalty is chosen per block on held-out
         tokens, and `headline.final_block_check` verifies tuned == lens at the last block.
  probe  the paper's supervised probe: multinomial logistic regression h_l -> answer over the
         forget answers, trained on retain-side prompts that have one of those answers but name
         none of the forget subjects (MCF neighbourhood prompts share the forget fact's answer by
         construction; the other datasets contribute whichever retain answers coincide), tested on
         the forget prompts. A probe taught only on other subjects that still reads the answer off
         a forgotten subject's prompt shows the ANSWER is represented, not merely the subject.
         Reported only with >= 2 answer classes of >= `--probe-min-train` training prompts each.

Per layer and probe group: top-1 rate (%) of the answer's first token and its mean probability
(lens, tuned); probe accuracy on the forget prompts and on held-out control prompts. Headline
RR (representation recoverability, lower is better) = max over layers of the tuned-lens top-1
rate on all forget prompts, the strongest label-free linear readout a defender has to assume;
RR_lens and RR_probe likewise, each with the layer it occurs at. The same command on the base
dir gives RR_base, the comparator the paper uses (probe accuracy of the original model). Control
prompts (retain-side) report the same numbers so a readout that fails everywhere is visible.

  python scripts/representation_recoverability.py --run-dir outputs/onepass_l19_v1/mcf/seed1/ours \
      --dataset mcf --mcf-path <MCF json> --device cuda --local-files-only
  ... --run-dir .../genie --genie            # same rows, oracle routing (as the evaluators)
  ... --run-dir .../ga --baseline ga --baseline-views-from <router dir> ...   # a baseline dir:
        the baseline edit is recomputed with the same seed (paper_baselines.py), as the evaluator does
Writes <run-dir>/representation_recoverability.{json,md}. scripts/run_l16_paper_one.sh runs it as
stage 6 (RECOVERABILITY=1); scripts/l16_paper_table.py reads the headline into the paper table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DATASETS = ("mcf", "zsre", "mquake", "rwku")
OUT_NAME = "representation_recoverability"
DEFINITION = {
    "RR": "max over layers of the tuned-lens top-1 rate (%) of the forgotten answer's first token "
          "on all forget prompts; lower is better",
    "lens": "logit lens: final norm + unembedding of the model on the residual stream after block l "
            "at the last prompt token",
    "tuned": "affine map h_l -> h_final fitted by ridge regression on every token of the dataset's "
             "retain-side prompts under the evaluated system (no forget prompt or label used), then "
             "final norm + unembedding",
    "probe": "multinomial logistic regression h_l -> forget answer, trained on retain-side prompts "
             "with a forget answer but no forget subject, tested on the forget prompts",
    "layer": "l = output of decoder block l (0-based), the convention of the router layer",
    "reference": "Gao et al. 2026, arXiv:2604.08271",
}


# ---------------------------------------------------------------- items
class Item:
    __slots__ = ("prompt", "answer", "subject", "group", "row")

    def __init__(self, prompt, answer, subject, group, row=None):
        self.prompt, self.answer, self.subject = str(prompt), str(answer), str(subject or "")
        self.group, self.row = str(group), row

    def as_dict(self):
        return {"prompt": self.prompt, "answer": self.answer, "subject": self.subject,
                "group": self.group, "row": self.row}


def first_answer_token(tok, answer):
    """First token of ' ' + answer, the convention of every evaluator (leading space, no BOS)."""
    ids = tok(" " + str(answer).strip(), add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"answer tokenised to nothing: {answer!r}")
    return int(ids[0])


def normalize_answer(text):
    return " ".join(str(text).casefold().replace(".", " ").split())


def _rr(record):
    rr = record["requested_rewrite"]
    return rr[0] if isinstance(rr, list) else rr


def _cap(items, n, seed):
    if n is None or len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def mcf_items(args, tok, artifact, seed):
    from mcf_zero_unlearn_official_eval import load_official_eval_records
    import genie_routes
    forget_records, retain_records = load_official_eval_records(
        mcf_path=args.mcf_path, unlearn_num=50, retain_num=1000, seed=seed, sample_mode="official")
    rows = genie_routes.fact_rows(artifact)
    forget, control = [], []
    for r in forget_records:
        rr = _rr(r)
        row = rows.get(f"mcf_forget_{int(r['case_id'])}")
        answer, subject = rr["target_true"]["str"], rr["subject"]
        forget.append(Item(str(rr["prompt"]).format(subject), answer, subject, "direct", row))
        forget += [Item(p, answer, subject, "paraphrase", row) for p in r.get("paraphrase_prompts", [])]
        # neighbourhood prompts: other subjects, the same true answer (CounterFact construction)
        control += [Item(p, answer, "", "neighbourhood", None) for p in r.get("neighborhood_prompts", [])]
    texts = [i.prompt for i in control]
    for r in retain_records:            # every retain-side prompt: the tuned lens is fitted on these
        rr = _rr(r)
        texts += [str(rr["prompt"]).format(rr["subject"]), *map(str, r.get("paraphrase_prompts", [])),
                  *map(str, r.get("neighborhood_prompts", []))]
    for r in _cap(retain_records, args.max_control, seed):
        rr = _rr(r)
        control.append(Item(str(rr["prompt"]).format(rr["subject"]), rr["target_true"]["str"],
                            rr["subject"], "retain", None))
    genie = list(genie_routes.mcf_pairs(artifact, forget_records))
    return forget, control, genie, texts


def zsre_items(args, tok, artifact, seed):
    import zsre_zero_unlearn_official_eval as zsre
    import genie_routes
    forget_records, retain_records = zsre.load_official_eval_records(
        Path(args.zsre_path), tok, forget_num=50, retain_num=1000, seed=seed)
    by_case = {int(f["case_id"]): i for i, f in enumerate(artifact["facts"])}
    forget, control = [], []
    for r in forget_records:
        rr = _rr(r)
        row = by_case.get(int(r["case_id"]))
        answer, subject = rr["target_true"]["str"], rr["subject"]
        forget.append(Item(str(rr["prompt"]).format(subject), answer, subject, "direct", row))
        forget += [Item(p, answer, subject, "paraphrase", row) for p in r.get("paraphrase_prompts", [])]
        nb = r.get("neighborhood_prompts") or []
        if nb:   # the NQ locality question: item 0 is the bare prompt, the targets spell the answer
            control.append(Item(nb[0]["prompt"], "".join(x["target"] for x in nb).strip(), "",
                                "neighbourhood", None))
    texts = [i.prompt for i in control]
    for r in retain_records:
        rr = _rr(r)
        texts += [str(rr["prompt"]).format(rr["subject"]), *map(str, r.get("paraphrase_prompts", []))]
        texts += [x["prompt"] for x in (r.get("neighborhood_prompts") or [])[:1]]
    for r in _cap(retain_records, args.max_control, seed):
        rr = _rr(r)
        control.append(Item(str(rr["prompt"]).format(rr["subject"]), rr["target_true"]["str"],
                            rr["subject"], "retain", None))
    genie = list(genie_routes.zsre_pairs(artifact, forget_records))
    return forget, control, genie, texts


def mquake_items(args, tok, artifact, seed):
    import mquake_zero_unlearn_official_eval as mquake
    from mquake_fact_association_embeddings import build_association_facts
    forget_records, retain_records = mquake.load_official_eval_records(
        Path(args.mquake_path).resolve(), tok, forget_num=50, retain_num=1000, seed=seed)
    facts, case_to_fact, _ = build_association_facts(forget_records)
    fact_to_row = {str(f["id"]): i for i, f in enumerate(artifact["facts"])}
    forget, control, genie = [], [], []
    for r in forget_records:
        rr = _rr(r)
        row = fact_to_row.get(str(case_to_fact.get(int(r["case_id"]))))
        answer, subject = rr["target_true"]["str"], rr["subject"]
        direct = str(rr["prompt"]).format(subject)
        forget.append(Item(direct, answer, subject, "direct", row))
        forget.append(Item(r["atomic_gen_prompt"], answer, subject, "question", row))
        if row is not None:
            genie += [(direct, row), (str(r["atomic_gen_prompt"]), row)]
    texts = []
    for r in retain_records:
        rr = _rr(r)
        texts += [str(rr["prompt"]).format(rr["subject"]), str(r["atomic_gen_prompt"])]
    for r in _cap(retain_records, args.max_control, seed):
        rr = _rr(r)
        control.append(Item(str(rr["prompt"]).format(rr["subject"]), rr["target_true"]["str"],
                            rr["subject"], "retain", None))
    return forget, control, genie, texts


def rwku_items(args, tok, artifact, seed):
    import rwku_eval as rwku
    from rwku_batch50 import build_batch_split
    from rwku_fact_association_embeddings import build_association_facts
    import genie_routes
    split = build_batch_split(data_root=Path(args.data_root).resolve(), batch_seed=seed,
                              allow_download=not args.no_download)
    forget_rows = list(split["efficacy_forget"])
    facts, record_to_fact, _ = build_association_facts(forget_rows, tok)
    fact_to_row = {f["id"]: i for i, f in enumerate(facts)}
    source_to_row = {sha: fact_to_row[fid] for sha, fid in record_to_fact.items()}

    def native(filename, level):
        out = []
        for target in split["per_target"]:
            for source in target["evaluation_only"][filename]:
                row = dict(source)
                row.setdefault("subject", target["subject"])
                row["level"] = str(level)
                out.append(row)
        return out

    forget, control = [], []
    for row in forget_rows:
        forget.append(Item(rwku.format_qa_prompt(tok, row), row["answer"], row.get("subject"), "direct",
                           source_to_row.get(str(row.get("source_record_sha256", "")))))
    for key, group in (("heldout_level1", "heldout_l1"), ("heldout_level2", "heldout_l2"),
                       ("heldout_paraphrase", "heldout_para")):
        for row in split[key]:
            if row.get("answer"):
                forget.append(Item(rwku.format_qa_prompt(tok, row), row["answer"], row.get("subject"), group))
    for filename, level in (("neighbor_level1.json", 1), ("neighbor_level2.json", 2)):
        for row in native(filename, level):
            if row.get("answer"):
                control.append(Item(rwku.format_qa_prompt(tok, row), row["answer"], row.get("subject"),
                                    "neighbour"))
    texts = [i.prompt for i in control]           # every neighbour probe, before the cap
    control = _cap(control, args.max_control, seed)
    genie = list(genie_routes.rwku_pairs(tok, forget_rows, source_to_row))
    return forget, control, genie, texts


ITEM_BUILDERS = {"mcf": mcf_items, "zsre": zsre_items, "mquake": mquake_items, "rwku": rwku_items}


# ---------------------------------------------------------------- hidden states
def hf_model(model):
    """The HF causal LM under a bank wrapper (AssociationCausalLM / EarlyWriteCausalLM) or itself.
    (A bare HF model also has a `base_model` property, the decoder without the head: keep the LM.)"""
    if hasattr(model, "lm_head") and hasattr(getattr(model, "model", None), "layers"):
        return model
    return getattr(model, "base_model", model)


def n_blocks(model):
    return len(hf_model(model).model.layers)


@torch.no_grad()
def collect_states(model, batches, layers, *, positions="last", device=None):
    """Residual stream after each block in `layers` for pre-tokenised `batches`.

    batches: iterable of (input_ids [B, W] long, attention_mask [B, W] long, prefix_lengths [B]).
    positions "last": the state at prefix_length - 1 of every row -> {layer: [N, d]} (float32, cpu).
    positions "all": every attended token except the first (BOS) -> {layer: [T, d]}.
    The capture hooks are registered after the bank's hook, so they see the edited state at the
    write layer. `model` is run through its wrapper (set_association_prefix_lengths + forward), so
    routing, forced routes, genie and baseline edits all apply exactly as under the evaluators.
    """
    hf = hf_model(model)
    device = device or next(hf.parameters()).device
    store = {}
    handles = []
    for layer in layers:
        def hook(_m, _a, out, layer=layer):
            store[layer] = (out[0] if isinstance(out, tuple) else out).detach()
        handles.append(hf.model.layers[layer].register_forward_hook(hook))
    chunks = {layer: [] for layer in layers}
    try:
        for ids, mask, prefix in batches:
            ids, mask = ids.to(device), mask.to(device)
            prefix = torch.as_tensor(prefix, dtype=torch.long)
            if hasattr(model, "set_association_prefix_lengths"):
                model.set_association_prefix_lengths(prefix)
            model(input_ids=ids, attention_mask=mask, use_cache=False)
            if positions == "last":
                index = (prefix.to(device) - 1)
                rows = torch.arange(ids.shape[0], device=device)
                for layer in layers:
                    chunks[layer].append(store[layer][rows, index].float().cpu())
            else:
                keep = mask.bool().clone()
                keep[:, 0] = False
                for layer in layers:
                    chunks[layer].append(store[layer][keep].float().cpu())
            store.clear()
    finally:
        for h in handles:
            h.remove()
    return {layer: torch.cat(chunks[layer], 0) for layer in layers}


def tokenize_batches(tok, prompts, batch_size):
    """Right-padded prompt batches; the request boundary is the prompt's own length."""
    tok.padding_side = "right"
    for start in range(0, len(prompts), batch_size):
        enc = tok(prompts[start:start + batch_size], padding=True, return_tensors="pt",
                  return_token_type_ids=False)
        yield enc["input_ids"], enc["attention_mask"], enc["attention_mask"].sum(1)


# ---------------------------------------------------------------- readouts
@torch.no_grad()
def unembed(hf, states, chunk=128):
    """Final norm + lm_head on [N, d] states -> log-probs [N, V] (float32, cpu)."""
    device = next(hf.parameters()).device
    dtype = next(hf.parameters()).dtype
    out = []
    for start in range(0, states.shape[0], chunk):
        h = states[start:start + chunk].to(device=device, dtype=dtype)
        logits = hf.lm_head(hf.model.norm(h)).float()
        out.append(F.log_softmax(logits, dim=-1).cpu())
    return torch.cat(out, 0) if out else torch.zeros(0, hf.config.vocab_size)


def readout_scores(logprobs, answer_tokens):
    """Per item: answer first token is top-1; its probability."""
    answers = torch.as_tensor(answer_tokens, dtype=torch.long)
    top1 = logprobs.argmax(-1) == answers
    prob = logprobs.gather(1, answers[:, None]).squeeze(1).exp()
    return top1, prob


RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _ridge_solve(xs, yc, lam):
    gram = xs.T @ xs
    gram = gram + lam * xs.shape[0] * torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.solve(gram, xs.T @ yc)


def fit_translator(x, y, ridge=None, *, grid=RIDGE_GRID, holdout=0.1, seed=0):
    """Affine least squares x -> y on per-dimension standardised x.

    Returns (W [d_x, d_y], b [d_y], info) in float32 and raw coordinates. Standardising first
    makes the penalty lambda relative to each dimension's own variance (Llama's residual stream has
    a few dimensions a hundred times larger than the rest; a penalty tied to the total norm would
    silence every ordinary dimension). ridge=None picks lambda from `grid` by held-out MSE on a
    `holdout` fraction of the rows, then refits on all rows; a float fixes it.
    """
    x, y = x.float(), y.float()
    xm, xs_ = x.mean(0), x.std(0).clamp_min(1e-6)
    ym = y.mean(0)
    xs, yc = (x - xm) / xs_, y - ym
    info = {"ridge": None if ridge is None else float(ridge), "rows": int(x.shape[0])}
    if ridge is None:
        n_val = max(1, int(x.shape[0] * holdout))
        perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed)).to(x.device)
        val, train = perm[:n_val], perm[n_val:]
        best = None
        var = float(yc[val].pow(2).mean())
        for lam in grid:
            w = _ridge_solve(xs[train], yc[train], lam)
            mse = float((xs[val] @ w - yc[val]).pow(2).mean())
            if best is None or mse < best[1]:
                best = (lam, mse)
        ridge = best[0]
        info.update({"ridge": ridge, "val_mse": best[1], "val_r2": 1.0 - best[1] / max(var, 1e-12)})
    weight_std = _ridge_solve(xs, yc, float(ridge))
    weight = weight_std / xs_[:, None]
    bias = ym - xm @ weight
    info["rank"] = int(torch.linalg.matrix_rank(xs))     # < d: the fit is undefined off this subspace
    info["rank_ok"] = info["rank"] >= int(0.95 * xs.shape[1])
    return weight, bias, info


def apply_translator(weight, bias, x):
    return x.float().to(weight.device) @ weight + bias


def fit_answer_probe(x_train, y_train, n_classes, *, l2=1e-2, max_iter=200, seed=0):
    """Multinomial logistic regression on standardised features (LBFGS). Returns predict(x)."""
    torch.manual_seed(seed)
    x_train = x_train.float()
    mean, std = x_train.mean(0), x_train.std(0).clamp_min(1e-6)
    xs = (x_train - mean) / std
    y = torch.as_tensor(y_train, dtype=torch.long)
    weight = torch.zeros(xs.shape[1], n_classes, requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=max_iter, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(xs @ weight + bias, y) + l2 * (weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    weight, bias = weight.detach(), bias.detach()

    def predict(x):
        return (((x.float() - mean) / std) @ weight + bias).argmax(-1)
    return predict


def probe_split(control_items, forget_items, *, min_train, holdout=0.2, seed=0):
    """Training / control-test indices into control_items and the forget indices the probe can
    be tested on. Classes = normalised forget answers; a control prompt trains only if its answer
    is a forget answer and it names no forget subject."""
    classes = {}
    for item in forget_items:
        classes.setdefault(normalize_answer(item.answer), len(classes))
    subjects = [s.casefold() for s in {i.subject for i in forget_items if i.subject}]
    eligible = [i for i, item in enumerate(control_items)
                if normalize_answer(item.answer) in classes
                and not any(s and s in item.prompt.casefold() for s in subjects)]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    by_class = {}
    for i in eligible:
        by_class.setdefault(normalize_answer(control_items[i].answer), []).append(i)
    train, test = [], []
    for members in by_class.values():
        n_test = int(len(members) * holdout)
        if len(members) - n_test < min_train:
            continue
        test += members[:n_test]
        train += members[n_test:]
    kept = {normalize_answer(control_items[i].answer) for i in train}
    if len(kept) < 2:
        return None
    label = {c: k for k, c in enumerate(sorted(kept))}
    forget_idx = [i for i, item in enumerate(forget_items) if normalize_answer(item.answer) in label]
    return {"label": label, "train": train, "test": test, "forget": forget_idx,
            "n_classes": len(label), "n_train": len(train), "n_test": len(test),
            "n_forget": len(forget_idx)}


# ---------------------------------------------------------------- evaluation
def _pct(mask):
    return 100.0 * float(mask.float().mean()) if mask.numel() else None


def _mean_pct(values):
    return 100.0 * float(values.float().mean()) if values.numel() else None


def group_scores(items, top1, prob):
    out = {}
    groups = sorted({i.group for i in items})
    for g in ["all", *groups]:
        idx = torch.tensor([k for k, i in enumerate(items) if g == "all" or i.group == g], dtype=torch.long)
        out[g] = {"n": int(idx.numel()), "top1": _pct(top1[idx]), "prob": _mean_pct(prob[idx])}
    return out


def evaluate_layers(hf, layers, forget_states, control_states, forget_items, control_items,
                    forget_tokens, control_tokens, *, translator=None, probe=None, ridge=None):
    """Per-layer readouts. translator: (train_states {layer: X}, y_final [T, d]) or None.
    probe: the dict of probe_split or None. ridge: fixed penalty, None = held-out choice."""
    per_layer = []
    f_tokens, c_tokens = list(forget_tokens), list(control_tokens)
    for layer in layers:
        entry = {"layer": int(layer)}
        fitted = None
        if translator is not None:
            x_train, y_final = translator
            weight, bias, info = fit_translator(x_train[layer].to(y_final.device), y_final, ridge)
            fitted = (weight, bias)
            entry["translator"] = info
        for name, states, items, tokens in (("forget", forget_states[layer], forget_items, f_tokens),
                                            ("control", control_states[layer], control_items, c_tokens)):
            if not items:
                entry[name] = {}
                continue
            top1, prob = readout_scores(unembed(hf, states), tokens)
            scores = {"lens": group_scores(items, top1, prob)}
            if fitted is not None:
                weight, bias = fitted
                top1, prob = readout_scores(unembed(hf, apply_translator(weight, bias, states).cpu()), tokens)
                scores["tuned"] = group_scores(items, top1, prob)
            entry[name] = scores
        if probe is not None and control_items:
            c = control_states[layer]
            label = probe["label"]
            y_train = [label[normalize_answer(control_items[i].answer)] for i in probe["train"]]
            predict = fit_answer_probe(c[probe["train"]], y_train, probe["n_classes"])
            res = {"n_classes": probe["n_classes"], "n_train": probe["n_train"],
                   "chance": 100.0 / probe["n_classes"]}
            if probe["test"]:
                y_test = torch.tensor([label[normalize_answer(control_items[i].answer)] for i in probe["test"]])
                res["control_acc"] = _pct(predict(c[probe["test"]]) == y_test)
                res["n_control"] = len(probe["test"])
            f = forget_states[layer]
            y_f = torch.tensor([label[normalize_answer(forget_items[i].answer)] for i in probe["forget"]])
            pred = predict(f[probe["forget"]])
            res["forget_acc"] = _pct(pred == y_f)
            res["n_forget"] = len(probe["forget"])
            res["forget_by_group"] = {}
            for g in sorted({forget_items[i].group for i in probe["forget"]}):
                sel = torch.tensor([k for k, i in enumerate(probe["forget"]) if forget_items[i].group == g])
                res["forget_by_group"][g] = {"n": int(sel.numel()), "acc": _pct(pred[sel] == y_f[sel])}
            entry["probe"] = res
        per_layer.append(entry)
    return per_layer


def headline(per_layer):
    """Max over layers of each forget readout (all groups), with its layer."""
    out = {}
    for key, path in (("RR", ("forget", "tuned", "all", "top1")), ("RR_lens", ("forget", "lens", "all", "top1")),
                      ("RR_probe", ("probe", "forget_acc"))):
        best = None
        for entry in per_layer:
            value = entry
            for p in path:
                value = value.get(p) if isinstance(value, dict) else None
                if value is None:
                    break
            if value is None:
                continue
            if best is None or value > best["value"]:
                best = {"value": value, "layer": entry["layer"]}
        out[key] = best
    last = per_layer[-1] if per_layer else {}
    out["output_top1"] = (((last.get("forget") or {}).get("lens") or {}).get("all") or {}).get("top1")
    # sanity: at the last block the tuned lens maps h_final -> h_final, so it must agree with the lens
    last_control = last.get("control") or {}
    lens = ((last_control.get("lens") or {}).get("all") or {}).get("top1")
    tuned = ((last_control.get("tuned") or {}).get("all") or {}).get("top1")
    out["final_block_check"] = None if lens is None or tuned is None else {
        "lens_control_top1": lens, "tuned_control_top1": tuned,
        "ok": tuned >= 0.75 * lens - 2.0}       # the ridge shrinks a little; a broken fit reads ~0
    return out


def markdown(result):
    lines = [f"# Representation recoverability: {result['run_dir']}", "",
             f"{result['dataset']} seed {result['seed']}"
             + (" · genie" if result.get("genie") else "")
             + (f" · baseline {result['baseline']['baseline']}" if result.get("baseline") else "")
             + f" · router layer {result.get('router_layer')} · write layer {result.get('write_layer')}", "",
             "Top-1 (%) of the forgotten answer's first token read from the residual stream after each block "
             "(lower is better); control = retain-side prompts (higher is the readout working).", "",
             "| layer | lens forget | tuned forget | probe forget | lens control | tuned control | probe control |",
             "|---|---|---|---|---|---|---|"]

    def cell(v):
        return "–" if v is None else f"{v:.1f}"
    for e in result["per_layer"]:
        f, c, p = e.get("forget") or {}, e.get("control") or {}, e.get("probe") or {}
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            e["layer"], cell(((f.get("lens") or {}).get("all") or {}).get("top1")),
            cell(((f.get("tuned") or {}).get("all") or {}).get("top1")), cell(p.get("forget_acc")),
            cell(((c.get("lens") or {}).get("all") or {}).get("top1")),
            cell(((c.get("tuned") or {}).get("all") or {}).get("top1")), cell(p.get("control_acc"))))
    h = result["headline"]
    lines += ["", "## Headline", ""]
    for key in ("RR", "RR_lens", "RR_probe"):
        v = h.get(key)
        lines.append(f"- **{key}**: " + ("–" if not v else f"{v['value']:.1f}% at layer {v['layer']}"))
    lines.append(f"- output top-1 (last block, = the model's own first-token recall): {cell(h.get('output_top1'))}%")
    chk = h.get("final_block_check")
    if chk:
        lines.append(f"- final-block check (tuned lens must match the lens on control prompts): lens "
                     f"{chk['lens_control_top1']:.1f}% vs tuned {chk['tuned_control_top1']:.1f}% -> "
                     + ("ok" if chk["ok"] else "MISMATCH, distrust the tuned readout"))
    g = result["groups"]
    lines += ["", f"Forget prompts: {g['forget']} · control prompts: {g['control']}"
              + (f" · probe: {result['probe']}" if result.get("probe") else " · probe: not enough shared answers")
              + (f" · tuned lens fitted on {result['translator']['tokens']} tokens of "
                 f"{result['translator']['texts']} retain-side texts" if result.get("translator") else "")]
    low = [e["layer"] for e in result["per_layer"] if (e.get("translator") or {}).get("rank_ok") is False]
    if low:
        lines.append(f"WARNING: training states are rank-deficient at blocks {low}; the tuned readout "
                     "is undefined off that subspace there")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main
def add_args(parser):
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--mcf-path")
    parser.add_argument("--zsre-path")
    parser.add_argument("--mquake-path")
    parser.add_argument("--data-root", default="data/rwku", help="RWKU data root")
    parser.add_argument("--no-download", action="store_true", help="RWKU: never download")
    parser.add_argument("--seed", type=int, default=None, help="default: the run manifest's seed")
    parser.add_argument("--genie", action="store_true", help="oracle routing, as the evaluators' --genie")
    parser.add_argument("--layers", default="all", help="comma list of block indices, default all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-control", type=int, default=300,
                        help="cap on retain-side control prompts per source (MCF neighbourhood prompts are "
                             "never capped: they train the answer probe)")
    parser.add_argument("--translator-tokens", type=int, default=8192,
                        help="token positions of retain-side prompts the tuned lens is fitted on (0 = lens only)")
    parser.add_argument("--ridge", type=float, default=None,
                        help="fixed ridge penalty for the tuned lens (relative to standardised features); "
                             "default: chosen per block by held-out MSE from " + ", ".join(map(str, RIDGE_GRID)))
    parser.add_argument("--probe-min-train", type=int, default=3)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--overwrite", action="store_true")


def manifest_seed(ds, manifest, override):
    if ds == "mcf":
        found = (manifest.get("sampling") or {}).get("seed")
    else:
        found = manifest.get("seed")
    seed = int(override if override is not None else (found if found is not None else 1))
    if found is not None and int(found) != seed:
        raise ValueError(f"--seed {seed} differs from the run manifest's seed {found}")
    return seed


def text_batches(tok, texts, max_tokens, batch_size, seed):
    """Distinct texts in random order, right-padded, until about max_tokens non-BOS positions."""
    texts = sorted(set(map(str, texts)))
    rng = random.Random(seed)
    rng.shuffle(texts)
    tok.padding_side = "right"
    budget = max_tokens
    for start in range(0, len(texts), batch_size):
        if budget <= 0:
            return
        enc = tok(texts[start:start + batch_size], padding=True, truncation=True, max_length=96,
                  return_tensors="pt", return_token_type_ids=False)
        budget -= int(enc["attention_mask"].sum()) - enc["input_ids"].shape[0]
        yield enc["input_ids"], enc["attention_mask"], enc["attention_mask"].sum(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    import paper_baselines
    paper_baselines.add_args(parser)
    args = parser.parse_args(argv)
    ds = args.dataset
    need = {"mcf": "mcf_path", "zsre": "zsre_path", "mquake": "mquake_path"}.get(ds)
    if need and not getattr(args, need):
        parser.error(f"--{need.replace('_', '-')} is required for {ds}")

    run_dir = Path(args.run_dir).resolve()
    out_path = Path(args.out).resolve() if args.out else run_dir / f"{OUT_NAME}.json"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {out_path} (pass --overwrite)")
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    seed = manifest_seed(ds, manifest, args.seed)
    model_path = Path(manifest["model_path"]).resolve()

    from mcf_zero_unlearn_official_eval import dtype_from_str
    from linear_router import load_router_artifact
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype_from_str(args.dtype),
                                                local_files_only=args.local_files_only,
                                                attn_implementation="eager").to(args.device).eval()
    base.requires_grad_(False)
    base.config.use_cache = False
    t0 = time.perf_counter()
    baseline_report = paper_baselines.apply(args, base, tok, artifact, run_dir, seed, mcf_path=args.mcf_path)
    model, bank = load_router_artifact(base, artifact)
    model.eval()

    forget_items, control_items, genie_pairs, retain_texts = ITEM_BUILDERS[ds](args, tok, artifact, seed)
    genie_report = None
    if args.genie:
        import genie_routes
        genie_report = genie_routes.install(bank, tok, genie_pairs)
    layers = (list(range(n_blocks(model))) if args.layers == "all"
              else [int(x) for x in args.layers.split(",") if x.strip()])
    final = n_blocks(model) - 1
    want = sorted(set(layers) | {final})

    print(f"[{ds} s{seed}] {len(forget_items)} forget / {len(control_items)} control prompts; "
          f"{len(set(retain_texts))} distinct retain-side texts for the tuned lens; layers {layers}", flush=True)
    forget_states = collect_states(model, tokenize_batches(tok, [i.prompt for i in forget_items], args.batch_size),
                                   want)
    control_states = (collect_states(model, tokenize_batches(tok, [i.prompt for i in control_items],
                                                             args.batch_size), want)
                      if control_items else {l: torch.zeros(0, forget_states[final].shape[1]) for l in want})
    translator = None
    if args.translator_tokens > 0 and retain_texts:
        train_states = collect_states(model, text_batches(tok, retain_texts, args.translator_tokens,
                                                          args.batch_size, seed), want, positions="all")
        device = next(hf_model(model).parameters()).device
        n = train_states[final].shape[0]
        if n > args.translator_tokens:
            keep = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:args.translator_tokens]
            train_states = {l: s[keep] for l, s in train_states.items()}
        translator = ({l: s.to(device) for l, s in train_states.items()}, train_states[final].to(device))
    probe = None if args.no_probe else probe_split(control_items, forget_items, min_train=args.probe_min_train,
                                                   seed=seed)
    forget_tokens = [first_answer_token(tok, i.answer) for i in forget_items]
    control_tokens = [first_answer_token(tok, i.answer) for i in control_items]
    per_layer = evaluate_layers(hf_model(model), layers, forget_states, control_states, forget_items,
                                control_items, forget_tokens, control_tokens, translator=translator, probe=probe,
                                ridge=args.ridge)

    result = {
        "definition": DEFINITION, "run_dir": str(run_dir), "dataset": ds, "seed": seed,
        "model_path": str(model_path), "genie": genie_report, "baseline": baseline_report,
        "router_layer": artifact.get("layer"),
        "write_layer": (artifact.get("early_write") or {}).get("write_layer", artifact.get("layer")),
        "layers": layers, "n_blocks": n_blocks(model),
        "groups": {"forget": {g: sum(i.group == g for i in forget_items) for g in sorted({i.group for i in forget_items})},
                   "control": {g: sum(i.group == g for i in control_items) for g in sorted({i.group for i in control_items})}},
        "translator": None if translator is None else {"tokens": int(translator[1].shape[0]),
                                                       "ridge": args.ridge if args.ridge is not None else "held-out choice",
                                                       "texts": len(set(retain_texts)),
                                                       "source": "retain-side prompts of this split, all token positions"},
        "probe": None if probe is None else {k: probe[k] for k in ("n_classes", "n_train", "n_test", "n_forget")},
        "per_layer": per_layer, "headline": headline(per_layer),
        "runtime_counters": bank.counters() if hasattr(bank, "counters") else None,
        "items": {"forget": [i.as_dict() for i in forget_items]},
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    out_path.with_suffix(".md").write_text(markdown(result))
    print(markdown(result))
    print(json.dumps({"headline": result["headline"], "out": str(out_path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
