"""Gradient baselines, ported from the repository's scripts/paper_baselines.py so that any
model / split can be used (same losses, same early stopping, same defaults).

    ga   gradient ascent on the request answers                     (Yao et al., 2023)
    gd   ga + NLL on retain text (gradient difference)              (Liu et al., 2022)
    npo  negative preference optimisation + retain NLL (NPO+RT)     (Zhang et al., 2024)
    rmu  representation misdirection at one layer, MLPs of the three layers up to it
         trained, retain activations kept                           (Li et al., 2024, WMDP)
    simnpo  reference-free NPO: the length-normalised answer log-probability itself is the
         margin, no reference model (SimNPO) + retain NLL            (Fan et al., 2024)
    idk_ft  fine-tune the request prompts to a refusal ("I don't know.") + retain NLL: a
         trained refusal, the weights-level counterpart of the idk cheats (TOFU "IDK")
                                                                    (Maini et al., 2024)
    tv   task-vector negation: fine-tune ON the request answers, then subtract alpha x the
         weight delta from the base weights (no retain data)        (Ilharco et al., 2023)

Training data: each request's own prompt + answer (fact requests), or text the base model
generates about the subject (entity requests). Retain data: the split's allowed_retain
facts plus base-model continuations of generic prompts. Early stopping: stop once the
mean answer probability on the request views is <= stop_prob (training data only).
"""
from __future__ import annotations

import contextlib
import math
import random
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from ..model_api import ModelHandle, Renderer
from ..schema import Probe, Request, RetainExample
from .base import Method, MethodCard, UnlearnContext, arg, request_probe

ENTITY_PROMPTS = ("Tell me about {S}.", "Who is {S}?", "What is {S} known for?", "Write a short biography of {S}.")
REFUSALS = ("I don't know.", "I'm not sure.", "I have no information about that.")


# --------------------------------------------------------------------------- data
def _ex(prompt_ids: List[int], answer_ids: List[int]) -> Dict:
    return {"input_ids": prompt_ids + answer_ids, "labels": [-100] * len(prompt_ids) + answer_ids}


@torch.no_grad()
def _greedy(handle: ModelHandle, ids: List[int], n: int) -> List[int]:
    dev = handle.device
    x = torch.tensor([ids], device=dev)
    eos = handle.tokenizer.eos_token_id
    out = handle.hf.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                             pad_token_id=eos if eos is not None else 0)
    return out[0, len(ids):].tolist()


def forget_examples(handle: ModelHandle, renderer: Renderer, requests: Sequence[Request], gen_tokens: int = 48) -> List[Dict]:
    out = []
    for r in requests:
        if r.kind == "entity":
            for t in ENTITY_PROMPTS:
                p = Probe(probe_id="", role="forget", family="direct", fact_id=r.fact_id,
                          prompt=t.replace("{S}", r.subject), answer="", format="qa", subject=r.subject)
                pids = renderer.encode_prompt(p)
                gen = _greedy(handle, pids, gen_tokens)
                if gen:
                    out.append({**_ex(pids, gen), "request_id": r.request_id})
        else:
            p = request_probe(r)
            out.append({**_ex(renderer.encode_prompt(p), renderer.encode_answer(p, r.answer)),
                        "request_id": r.request_id})
    return out


def idk_examples(renderer: Renderer, requests: Sequence[Request]) -> List[Dict]:
    """The request prompts (entity requests: the ENTITY_PROMPTS) labelled with a refusal."""
    out, k = [], 0
    for r in requests:
        probes = ([Probe(probe_id="", role="forget", family="direct", fact_id=r.fact_id, prompt=t.replace("{S}", r.subject),
                         answer="", format="qa", subject=r.subject) for t in ENTITY_PROMPTS]
                  if r.kind == "entity" else [request_probe(r)])
        for p in probes:
            out.append({**_ex(renderer.encode_prompt(p), renderer.encode_answer(p, REFUSALS[k % len(REFUSALS)])),
                        "request_id": r.request_id})
            k += 1
    return out


def retain_examples(handle: ModelHandle, renderer: Renderer, allowed: Sequence[RetainExample],
                    generic_prompts: Sequence[str], gen_tokens: int = 48, limit: int = 200) -> List[Dict]:
    tok = renderer.tok
    out = []
    for r in list(allowed)[:limit]:
        ids = tok(r.text)["input_ids"]
        if len(ids) > 1:
            out.append({"input_ids": ids, "labels": [-100] + ids[1:]})
    for g in generic_prompts:
        pids = tok(g)["input_ids"]
        gen = _greedy(handle, pids, gen_tokens)
        if gen:
            out.append(_ex(pids, gen))
    return out


def _pad(batch: Sequence[Dict], pad_id: int, device):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    att = torch.zeros_like(ids)
    lab = torch.full_like(ids, -100)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        att[i, :n] = 1
        lab[i, :n] = torch.tensor(b["labels"])
    return ids.to(device), att.to(device), lab.to(device)


def answer_logprob(model, batch, pad_id, device):
    """Sum and count of log p(label tokens) per example. Logits are computed at the labelled
    positions only (a 150k-vocabulary model would otherwise hold batch x length x vocab)."""
    from ..model_api import plain_selected_logits
    ids, att, lab = _pad(batch, pad_id, device)
    spans, tgts = [], []
    for b, item in enumerate(batch):
        labels = item["labels"]
        k = [i for i, t in enumerate(labels) if t != -100]
        if not k:
            k = [len(labels) - 1]
        k0, k1 = k[0], k[-1]
        spans.append((b, k0 - 1, k1 - k0 + 1))
        tgts.append([t if t != -100 else 0 for t in labels[k0:k1 + 1]])
    picked = plain_selected_logits(model, ids, att, spans)
    sums, counts = [], []
    for (b, s0, n), lg, tg, item in zip(spans, picked, tgts, batch):
        t = torch.tensor(tg, device=lg.device)
        keep = torch.tensor([x != -100 for x in item["labels"][s0 + 1:s0 + 1 + n]], device=lg.device)
        lp = torch.log_softmax(lg.float(), -1).gather(-1, t[:, None])[:, 0]
        sums.append((lp * keep).sum())
        counts.append(keep.sum().clamp_min(1))
    return torch.stack(sums), torch.stack(counts)


@torch.no_grad()
def mean_answer_prob(model, items, pad_id, device, bs=8) -> float:
    was = model.training
    model.eval()
    vals = []
    for s in range(0, len(items), bs):
        lp, _ = answer_logprob(model, items[s:s + bs], pad_id, device)
        vals += torch.exp(lp).tolist()
    model.train(was)
    return sum(vals) / max(1, len(vals))


def trainable(model, scope: str, layers=None):
    names = []
    for n, p in model.named_parameters():
        take = scope == "all" or (scope == "mlp" and ".mlp." in n) or (scope == "mlp_down" and "down_proj" in n)
        if take and layers is not None:
            take = any(f"layers.{L}." in n for L in layers)
        p.requires_grad_(take)
        if take:
            names.append(n)
    return [p for p in model.parameters() if p.requires_grad], names


def _pad_id(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)


# ------------------------------------------------------------------- GA / GD / NPO
class _GradientMethod(Method):
    loss_kind = "ga"

    def unlearn(self, base, requests, allowed_retain, ctx: UnlearnContext):
        a = self.args
        model, tok = base.hf, base.tokenizer
        device = base.device
        lr = arg(a, "lr", 1e-5, float)
        epochs = arg(a, "epochs", 20, int)
        bs = arg(a, "batch_size", 8, int)
        stop = arg(a, "stop_prob", 1e-4, float)
        check = arg(a, "check_every", 5, int)
        kind = self.loss_kind
        beta = arg(a, "simnpo_beta", 2.5, float) if kind == "simnpo" else arg(a, "npo_beta", 0.1, float)
        gamma = arg(a, "simnpo_gamma", 0.0, float)
        rw = arg(a, "retain_weight", 1.0, float)
        if kind == "tv":                      # fixed fine-tuning budget; the answer probability rises, so no early stop
            epochs = arg(a, "tv_epochs", 5, int)
            stop = 0.0
        scope = arg(a, "params", "mlp", str)
        # "fp32" (default): the whole model in fp32, as paper_baselines. "mixed" (24 GB GPUs): bf16
        # forward under autocast, fp32 master copies of the trained weights only.
        precision = arg(a, "precision", "fp32", str)
        pad = _pad_id(tok)
        forget = forget_examples(base, ctx.renderer, requests)
        train_items = idk_examples(ctx.renderer, requests) if kind == "idk_ft" else forget
        retain = (retain_examples(base, ctx.renderer, allowed_retain, ctx.generic_prompts)
                  if kind in ("gd", "npo", "simnpo", "idk_ft") else [])
        orig_dtype = next(model.parameters()).dtype
        if precision == "mixed":
            params, names = trainable(model, scope)
            for p in params:
                p.data = p.data.float()
            amp = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        else:
            model.float()
            params, names = trainable(model, scope)
            amp = contextlib.nullcontext()
        snapshot = [p.detach().clone() for p in params] if kind == "tv" else None
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        rng = random.Random(ctx.seed)
        with amp:
            step, stopped, log, ref = self._train(model, train_items, forget, retain, params, opt, rng, pad, device, ctx,
                                                  epochs=epochs, bs=bs, stop=stop, check=check, beta=beta, rw=rw, gamma=gamma)
        tv_alpha = None
        if snapshot is not None:              # theta = theta_0 - alpha * (theta_ft - theta_0)
            tv_alpha = arg(a, "tv_alpha", 1.0, float)
            with torch.no_grad():
                for p, p0 in zip(params, snapshot):
                    p.data = p0 - tv_alpha * (p.data - p0)
            del snapshot
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(orig_dtype)
        self._report = {"lr": lr, "steps": step, "early_stop": stopped, "params": scope, "precision": precision,
                        "trainable_params": sum(p.numel() for p in params), "forget_examples": len(forget),
                        "train_examples": len(train_items), "retain_examples": len(retain), "log": log[-10:]}
        if tv_alpha is not None:
            self._report["tv_alpha"] = tv_alpha
        if kind == "tv":
            self._report["forget_answer_prob_after"] = mean_answer_prob(model, forget, pad, device, bs)
        return ModelHandle(model=model, tokenizer=tok, hf=model)

    def _train(self, model, train_items, monitor, retain, params, opt, rng, pad, device, ctx, *, epochs, bs, stop, check,
               beta, rw, gamma=0.0):
        """train_items: the batches the loss is taken on; monitor: the request answers whose mean
        probability drives early stopping (the same list except for idk_ft, which trains on refusals)."""
        forget = train_items
        ref = None
        if self.loss_kind == "npo":
            ref = {}
            model.eval()
            with torch.no_grad():
                for s in range(0, len(forget), bs):
                    chunk = forget[s:s + bs]
                    lp, _ = answer_logprob(model, chunk, pad, device)
                    for item, v in zip(chunk, lp.tolist()):
                        ref[id(item)] = v
        model.train()
        step, stopped, log = 0, None, []
        for epoch in range(epochs):
            order = list(range(len(forget)))
            rng.shuffle(order)
            for s in range(0, len(order), bs):
                batch = [forget[i] for i in order[s:s + bs]]
                lp, n = answer_logprob(model, batch, pad, device)
                if self.loss_kind in ("ga", "gd"):
                    loss = (lp / n).mean()
                elif self.loss_kind == "npo":
                    r0 = torch.tensor([ref[id(b)] for b in batch], device=device)
                    loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp - r0)).mean()
                elif self.loss_kind == "simnpo":    # margin = length-normalised log-prob, no reference
                    loss = -(2.0 / beta) * F.logsigmoid(-beta * (lp / n) - gamma).mean()
                else:                                # idk_ft, tv: plain NLL on the training labels
                    loss = -(lp / n).mean()
                if retain:
                    rb = rng.sample(retain, min(bs, len(retain)))
                    rlp, rn = answer_logprob(model, rb, pad, device)
                    loss = loss + rw * (-(rlp / rn).mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                step += 1
                if stop > 0 and step % check == 0:
                    p = mean_answer_prob(model, monitor, pad, device, bs)
                    log.append({"step": step, "forget_answer_prob": p, "loss": float(loss.detach())})
                    if p <= stop:
                        stopped = {"step": step, "forget_answer_prob": p}
                        break
                if ctx.over_budget():
                    stopped = {"step": step, "reason": "time budget"}
                    break
            if stopped:
                break
        return step, stopped, log, ref


class GA(_GradientMethod):
    loss_kind = "ga"
    card = MethodCard("ga", "gradient", "gradient ascent on the request answers", reference="Yao et al., 2023")


class GD(_GradientMethod):
    loss_kind = "gd"
    card = MethodCard("gd", "gradient", "gradient ascent + retain NLL (gradient difference)",
                      uses_allowed_retain=True, reference="Liu et al., 2022")


class NPO(_GradientMethod):
    loss_kind = "npo"
    card = MethodCard("npo", "gradient", "negative preference optimisation + retain NLL",
                      uses_allowed_retain=True, reference="Zhang et al., 2024")


class SimNPO(_GradientMethod):
    loss_kind = "simnpo"
    card = MethodCard("simnpo", "gradient", "reference-free NPO (length-normalised log-prob margin) + retain NLL",
                      uses_allowed_retain=True, reference="Fan et al., 2024")


class IDKFT(_GradientMethod):
    loss_kind = "idk_ft"
    card = MethodCard("idk_ft", "gradient", "fine-tune the request prompts to a refusal + retain NLL (trained refusal)",
                      uses_allowed_retain=True, reference="Maini et al., 2024 (TOFU IDK)")


class TaskVector(_GradientMethod):
    loss_kind = "tv"
    card = MethodCard("tv", "gradient", "task-vector negation: fine-tune on the answers, subtract alpha x the weight delta",
                      reference="Ilharco et al., 2023")


# ----------------------------------------------------------------------------- RMU
def _decoder_layers(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise ValueError("cannot locate decoder layers")


class _Stop(Exception):
    pass


def _hidden_at(model, layer_mod, batch, pad, device):
    """Hidden states at one layer. The forward stops right after that layer: nothing above it
    (and no batch x length x vocab logits) is computed or kept for autograd."""
    ids, att, _ = _pad(batch, pad, device)
    store = {}

    def hook(m, a, o):
        store["h"] = o[0] if isinstance(o, tuple) else o
        raise _Stop

    h = layer_mod.register_forward_hook(hook)
    try:
        model(input_ids=ids, attention_mask=att)
    except _Stop:
        pass
    finally:
        h.remove()
    return store["h"].float(), att.bool()


class RMU(Method):
    card = MethodCard("rmu", "representation", "steer request activations at one layer to a random direction, "
                      "keep retain activations", uses_allowed_retain=True, reference="Li et al., 2024 (WMDP)")

    def unlearn(self, base, requests, allowed_retain, ctx):
        a = self.args
        model, tok = base.hf, base.tokenizer
        device = base.device
        pad = _pad_id(tok)
        layers = _decoder_layers(model)
        L = arg(a, "layer", max(2, int(round(0.25 * len(layers)))), int)
        lr = arg(a, "lr", 5e-5, float)
        steps = arg(a, "steps", 250, int)
        alpha = arg(a, "alpha", 1200.0, float)
        mult = arg(a, "steer_mult", 4.0, float)
        bs = arg(a, "batch_size", 8, int)
        stop = arg(a, "stop_prob", 1e-4, float)
        forget = forget_examples(base, ctx.renderer, requests)
        retain = retain_examples(base, ctx.renderer, allowed_retain, ctx.generic_prompts)
        orig_dtype = next(model.parameters()).dtype
        model.float()
        g = torch.Generator().manual_seed(ctx.seed)
        u = torch.rand(model.config.hidden_size, generator=g)
        u = (u / u.norm()).to(device)
        model.eval()
        with torch.no_grad():
            norms = []
            for s in range(0, len(forget), bs):
                h, m = _hidden_at(model, layers[L], forget[s:s + bs], pad, device)
                norms.append(h[m].norm(dim=-1).cpu())
            median = float(torch.cat(norms).median())
            frozen = []
            for s in range(0, len(retain), bs):
                chunk = retain[s:s + bs]
                h, m = _hidden_at(model, layers[L], chunk, pad, device)
                frozen.append((chunk, h.detach(), m))
        steer = mult * median
        params, names = trainable(model, "mlp", layers=[max(0, L - 2), max(0, L - 1), L])
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        rng = random.Random(ctx.seed)
        model.train()
        stopped, log = None, []
        for step in range(steps):
            fb = rng.sample(forget, min(bs, len(forget)))
            hf_, mf = _hidden_at(model, layers[L], fb, pad, device)
            loss_f = ((hf_[mf] - steer * u) ** 2).mean()
            chunk, h0, m0 = frozen[rng.randrange(len(frozen))]
            hr, mr = _hidden_at(model, layers[L], chunk, pad, device)
            loss_r = ((hr[mr] - h0[m0]) ** 2).mean()
            loss = loss_f + alpha * loss_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if (step + 1) % 25 == 0:
                p = mean_answer_prob(model, forget, pad, device, bs)
                log.append({"step": step + 1, "loss_forget": float(loss_f.detach()),
                            "loss_retain": float(loss_r.detach()), "forget_answer_prob": p})
                if stop > 0 and p <= stop:
                    stopped = {"step": step + 1, "forget_answer_prob": p}
                    break
            if ctx.over_budget():
                stopped = {"step": step + 1, "reason": "time budget"}
                break
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(orig_dtype)
        self._report = {"layer": L, "lr": lr, "steer": steer, "alpha": alpha, "early_stop": stopped,
                        "trainable_params": sum(p.numel() for p in params), "log": log[-10:]}
        return ModelHandle(model=model, tokenizer=tok, hf=model)
