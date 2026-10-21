"""Unlearning baselines applied in-process inside the official evaluators (`--baseline`).

The evaluator loads the base model, this module edits its weights, and the evaluator then
scores the edited model exactly as it scores base / ours / genie (same seed, same splits,
same metric code; the run dir is the L19 router with all-zero rows, i.e. no routing edit).

Training data = the SAME training-visible forget views our method trains on (the router's
own prompts + answers, at most --baseline-views-per-fact per fact; official paraphrases,
neighbourhood and retain records are never read). Baselines that need a retain set get
base-model SELF-GENERATED text (greedy continuations of generic prompts), which is disjoint
from every evaluation set by construction.

  ga           gradient ascent on forget answer NLL                      (Yao et al., 2023)
  gd           ga + retain NLL (gradient difference)                     (Liu et al., 2022)
  npo          NPO on forget answers + retain NLL (NPO+RT)               (Zhang et al., 2024)
  rmu          representation misdirection at --rmu-layer, MLPs of the 3 layers up to it
               trained; retain activations kept                          (Li et al., 2024, WMDP)
  zerounlearn  original ZeroUnlearn (vendored ../ZeroUnlearn), target_true -> EOS, layers
               16-18 down_proj, no retain requests                       (arXiv 2605.18879)

Early stopping (--baseline-stop-prob, default 1e-4): GA/GD/NPO (every 5 steps) and RMU (every
25) stop as soon as the mean answer probability on the forget TRAINING views is <= 1e-4, i.e.
each baseline edits only as much as it needs to forget what it was shown (no evaluation data
is used to choose when to stop). The full budget below is the cap.
GA / GD / NPO train the MLP weights of every layer (--baseline-params mlp) in fp32, AdamW,
--baseline-epochs over the forget views; the model is cast back to the evaluator's dtype.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

BASELINES = ("ga", "gd", "npo", "rmu", "zerounlearn", "rome", "memit")
EDIT_BASELINES = ("zerounlearn", "rome", "memit")   # locate-and-edit, vendored in ZeroUnlearn/
SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEMANTIC_ROOT.parent
RETAIN_PROMPTS = (
    "The history of the Roman Empire", "Photosynthesis is the process by which",
    "In computer science, a hash table", "The best way to learn a new language is",
    "Mount Everest is located", "A healthy breakfast usually includes",
    "The French Revolution began in", "Electric cars work by", "The theory of evolution states",
    "To bake a loaf of bread, you", "The Pacific Ocean is", "Shakespeare wrote many plays, including",
    "In economics, inflation means", "The human heart has four", "Jazz music originated in",
    "A neural network is", "The capital cities of Europe include", "Volcanoes form when",
    "The rules of chess say that", "Water boils at", "The Great Wall of China was built",
    "Vaccines protect people by", "The planets of the solar system are", "A good resume should",
    "The Industrial Revolution changed", "Honey bees communicate by", "In mathematics, a prime number",
    "The Amazon rainforest is", "Classical music composers such as", "The internet was first developed",
    "Coffee is grown in", "Earthquakes are measured using",
)


def add_args(parser):
    g = parser.add_argument_group("paper baselines (scripts/paper_baselines.py)")
    g.add_argument("--baseline", choices=BASELINES, default=None,
                   help="edit the base model with this baseline before evaluating")
    g.add_argument("--baseline-views-from", default=None,
                   help="router dir whose training-visible prompts the baseline trains on "
                        "(default: the run dir); pass the router OUR method used")
    g.add_argument("--baseline-views-per-fact", type=int, default=6)
    g.add_argument("--baseline-epochs", type=int, default=5)
    g.add_argument("--baseline-lr", type=float, default=None,
                   help="default 1e-5 (ga/gd/npo), 5e-5 (rmu)")
    g.add_argument("--baseline-batch-size", type=int, default=8)
    g.add_argument("--baseline-params", choices=("mlp", "mlp_down", "all"), default="mlp")
    g.add_argument("--baseline-seed", type=int, default=None, help="default: the split seed")
    g.add_argument("--baseline-stop-prob", type=float, default=1e-4,
                   help="early stop: stop once the mean answer probability on the forget TRAINING views "
                        "is <= this (0 = train the full budget); training data only, no eval set")
    g.add_argument("--baseline-check-every", type=int, default=5)
    g.add_argument("--npo-beta", type=float, default=0.1)
    g.add_argument("--retain-weight", type=float, default=1.0)
    g.add_argument("--retain-tokens", type=int, default=48)
    g.add_argument("--rmu-layer", type=int, default=7)
    g.add_argument("--rmu-steer", type=float, default=None,
                   help="control-vector norm; default = --rmu-steer-mult x median ||h_layer|| on forget tokens")
    g.add_argument("--rmu-steer-mult", type=float, default=4.0)
    g.add_argument("--rmu-alpha", type=float, default=1200.0)
    g.add_argument("--rmu-steps", type=int, default=250)
    g.add_argument("--zero-root", default=str(REPO_ROOT / "ZeroUnlearn"))
    g.add_argument("--zero-hparams", default=str(
        REPO_ROOT / "ZeroUnlearn" / "hparams" / "ZeroUnlearn" / "Llama-3.2-3B-Instruct.json"))
    g.add_argument("--rome-hparams", default=str(
        REPO_ROOT / "ZeroUnlearn" / "hparams" / "ROME" / "Llama-3.2-3B-Instruct.json"))
    g.add_argument("--memit-hparams", default=str(
        REPO_ROOT / "ZeroUnlearn" / "hparams" / "MEMIT" / "Llama-3.2-3B-Instruct.json"))
    g.add_argument("--mom2-dataset", default=None,
                   help="ZeroUnlearn / ROME / MEMIT: covariance cache name (data/stats/<model>/<name>_stats); "
                        "precompute with scripts/build_edit_covariance.py. Default: the vendored 'wikipedia' "
                        "entry, which the vendored loaders fill from the 200-sentence data/wikidata fixture")
    g.add_argument("--mom2-text", default=None,
                   help="the corpus that cache was built from (ZU_MOM2_TEXT_DIR for the vendored loaders)")
    g.add_argument("--allow-mom2-compute", action="store_true",
                   help="compute missing covariance caches inside the task (slow, fp32) instead of failing")
    g.add_argument("--edit-target", choices=("eos", "unknown"), default="eos",
                   help="target_new of the ZeroUnlearn / ROME / MEMIT requests: the EOS token "
                        "(ZeroUnlearn's safe state, default) or 'Unknown' (ZeroUnlearn's ZsRE setup)")
    return g


# ------------------------------------------------------------------------------ data
def forget_views(args, artifact, run_dir, tok, seed, mcf_path=None):
    """Training-visible (prompt, answer) examples per fact, canonical first, train before dev."""
    from train_mcf_jlens_feedback_rows import _is_canonical, detect_dataset, generic_examples

    dataset = detect_dataset(artifact["facts"])
    if dataset == "mcf":
        from prepare_mcf_association_source import load_mcf_forget_data
        _, facts, examples = load_mcf_forget_data(tok, mcf_path, seed=seed)
        if [f["id"] for f in facts] != [f["id"] for f in artifact["facts"]]:
            raise ValueError("MCF baseline facts differ from the run's facts")
    else:
        source = Path(args.baseline_views_from or run_dir)
        examples, _ = generic_examples(source, artifact["facts"], tok)
    by = {}
    for e in examples:
        by.setdefault(e.fact_id, []).append(e)
    views = []
    for fid, exs in by.items():
        exs = sorted(exs, key=lambda e: (not _is_canonical(e.group), e.split != "train", e.id))
        views += exs[:args.baseline_views_per_fact]
    return dataset, views


@torch.no_grad()
def self_generated_retain(model, tok, max_new_tokens):
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    out = []
    for prompt in RETAIN_PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        seq = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=eos)[0].tolist()
        out.append({"input_ids": seq, "labels": [-100] * ids.shape[1] + seq[ids.shape[1]:]})
    return out


def _pad(batch, device):
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long, device=device)
    att = torch.zeros_like(ids)
    lab = torch.full_like(ids, -100)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], device=device)
        att[i, :n] = 1
        lab[i, :n] = torch.tensor(b["labels"], device=device)
    return ids, att, lab


def answer_logprob(model, batch, device):
    """Sum and count of log p(answer tokens) per example (teacher forced)."""
    ids, att, lab = _pad(batch, device)
    logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float()
    tgt = lab[:, 1:]
    mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
    return (lp * mask).sum(-1), mask.sum(-1).clamp_min(1)


@torch.no_grad()
def forget_answer_prob(model, forget, batch_size, device):
    """Mean over forget training views of P(answer | prompt) = exp(sum log p(answer tokens))."""
    was = model.training
    model.eval()
    vals = []
    for s in range(0, len(forget), batch_size):
        lp, _ = answer_logprob(model, forget[s:s + batch_size], device)
        vals += torch.exp(lp).tolist()
    model.train(was)
    return sum(vals) / max(1, len(vals))


def _as_dicts(views):
    return [{"input_ids": list(v.input_ids), "labels": list(v.labels), "fact_id": v.fact_id} for v in views]


# ------------------------------------------------------------------------------ training
def _trainable(model, mode, layers=None):
    names = []
    for name, p in model.named_parameters():
        take = (mode == "all"
                or (mode == "mlp" and ".mlp." in name)
                or (mode == "mlp_down" and ".mlp.down_proj" in name))
        if take and layers is not None:
            take = any(f"layers.{L}." in name for L in layers)
        p.requires_grad_(take)
        if take:
            names.append(name)
    return [p for p in model.parameters() if p.requires_grad], names


def _batches(items, size, rng):
    order = list(range(len(items)))
    rng.shuffle(order)
    for s in range(0, len(order), size):
        yield [items[i] for i in order[s:s + size]]


def train_ga_gd_npo(args, model, tok, views, retain, seed):
    device = next(model.parameters()).device
    method = args.baseline
    lr = args.baseline_lr or 1e-5
    params, names = _trainable(model, args.baseline_params)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    forget = _as_dicts(views)
    rng = random.Random(seed)
    ref = None
    if method == "npo":
        model.eval()
        with torch.no_grad():
            ref = {}
            for s in range(0, len(forget), args.baseline_batch_size):
                chunk = forget[s:s + args.baseline_batch_size]
                lp, _ = answer_logprob(model, chunk, device)
                for item, v in zip(chunk, lp.tolist()):
                    ref[id(item)] = v
    model.train()
    log, step, stopped = [], 0, None
    for epoch in range(args.baseline_epochs):
        for batch in _batches(forget, args.baseline_batch_size, rng):
            lp, n = answer_logprob(model, batch, device)
            if method in ("ga", "gd"):
                loss = (lp / n).mean()                         # ascend NLL = descend log p
            else:
                lr_ref = torch.tensor([ref[id(b)] for b in batch], device=device)
                loss = -(2.0 / args.npo_beta) * F.logsigmoid(-args.npo_beta * (lp - lr_ref)).mean()
            if method in ("gd", "npo") and retain:
                rb = rng.sample(retain, min(args.baseline_batch_size, len(retain)))
                rlp, rn = answer_logprob(model, rb, device)
                loss = loss + args.retain_weight * (-(rlp / rn).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                log.append({"epoch": epoch, "step": step, "loss": float(loss.detach()),
                            "forget_mean_logp_per_token": float((lp / n).mean().detach())})
            if args.baseline_stop_prob > 0 and step % args.baseline_check_every == 0:
                p_forget = forget_answer_prob(model, forget, args.baseline_batch_size, device)
                if p_forget <= args.baseline_stop_prob:
                    stopped = {"step": step, "forget_answer_prob": p_forget}
                    break
        if stopped:
            break
    model.eval()
    return {"lr": lr, "steps": step, "max_steps": args.baseline_epochs * math.ceil(len(forget) / args.baseline_batch_size),
            "early_stop": stopped, "stop_prob": args.baseline_stop_prob,
            "final_forget_answer_prob": forget_answer_prob(model, forget, args.baseline_batch_size, device),
            "trainable_params": sum(p.numel() for p in params),
            "trainable_scope": args.baseline_params, "log": log[-20:]}


def _hidden_at(model, layer, batch, device):
    ids, att, _ = _pad(batch, device)
    store = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, a, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    try:
        model(input_ids=ids, attention_mask=att, use_cache=False)
    finally:
        h.remove()
    return store["h"].float(), att.bool()


def train_rmu(args, model, tok, views, retain, seed):
    device = next(model.parameters()).device
    L = int(args.rmu_layer)
    lr = args.baseline_lr or 5e-5
    forget = [{"input_ids": list(v.input_ids), "labels": list(v.labels)} for v in views]
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(model.config.hidden_size, generator=g)
    u = (u / u.norm()).to(device)
    model.eval()
    with torch.no_grad():
        norms = []
        for s in range(0, len(forget), args.baseline_batch_size):
            h, m = _hidden_at(model, L, forget[s:s + args.baseline_batch_size], device)
            norms.append(h[m].norm(dim=-1).cpu())
        median_norm = float(torch.cat(norms).median())
        frozen = []
        for s in range(0, len(retain), args.baseline_batch_size):
            chunk = retain[s:s + args.baseline_batch_size]
            h, m = _hidden_at(model, L, chunk, device)
            frozen.append((chunk, h.detach(), m))
    steer = args.rmu_steer if args.rmu_steer is not None else args.rmu_steer_mult * median_norm
    params, names = _trainable(model, "mlp", layers=[L - 2, L - 1, L])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, foreach=False)
    model.train()
    log, stopped = [], None
    for step in range(args.rmu_steps):
        fb = rng.sample(forget, min(args.baseline_batch_size, len(forget)))
        hf, mf = _hidden_at(model, L, fb, device)
        loss_f = ((hf[mf] - steer * u) ** 2).mean()
        chunk, h0, m0 = frozen[rng.randrange(len(frozen))]
        hr, mr = _hidden_at(model, L, chunk, device)
        loss_r = ((hr[mr] - h0[m0]) ** 2).mean()
        loss = loss_f + args.rmu_alpha * loss_r
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 25 == 0:
            p_forget = forget_answer_prob(model, forget, args.baseline_batch_size, device)
            log.append({"step": step + 1, "loss_forget": float(loss_f.detach()),
                        "loss_retain": float(loss_r.detach()), "forget_answer_prob": p_forget})
            if args.baseline_stop_prob > 0 and p_forget <= args.baseline_stop_prob:
                stopped = {"step": step + 1, "forget_answer_prob": p_forget}
                break
    model.eval()
    return {"lr": lr, "steps": (stopped or {}).get("step", args.rmu_steps), "max_steps": args.rmu_steps,
            "early_stop": stopped, "stop_prob": args.baseline_stop_prob, "layer": L, "trained_layers": [L - 2, L - 1, L],
            "steer": steer, "median_forget_norm": median_norm, "alpha": args.rmu_alpha,
            "trainable_params": sum(p.numel() for p in params), "log": log[-10:]}


# ------------------------------------------------------------------------------ ZeroUnlearn
def zero_requests(facts, tok, target="eos"):
    eos = tok.eos_token if target == "eos" else "Unknown"
    reqs, skipped = [], []
    for i, f in enumerate(facts):
        prompt, subject = str(f["canonical_prompt"]), str(f["subject"])
        at = prompt.find(subject)
        if at < 0:
            at = prompt.casefold().find(subject.casefold())
        if at < 0:
            skipped.append(f["id"])
            continue
        esc = lambda s: s.replace("{", "{{").replace("}", "}}")
        template = esc(prompt[:at]) + "{}" + esc(prompt[at + len(subject):])
        reqs.append({"case_id": i, "prompt": template, "subject": prompt[at:at + len(subject)],
                     "target_true": {"str": str(f["object"])}, "target_new": {"str": eos}})
    return reqs, skipped


def apply_zerounlearn(args, model, tok, artifact, seed):
    import run_zerounlearn_official_mcf as zb

    reqs, skipped = zero_requests(artifact["facts"], tok, getattr(args, "edit_target", "eos"))
    params_class, apply_unl = zb.import_original_zerounlearn(Path(args.zero_root))
    hparams = params_class.from_json(Path(args.zero_hparams))
    mom2 = configure_mom2(args, hparams, model, need_vkt=True)
    zb.set_all_seeds(seed)
    with zb.working_directory(zb.SEMANTIC_ROOT):
        apply_unl(model=model, tok=tok, retain_requests=[], unlearn_requests=reqs, hparams=hparams,
                  copy=False, return_orig_weights=False, cache_template=None, save_path=None,
                  add_retain=False, edit_layer_nums=zb.EDIT_LAYER_NUMS, use_h=zb.USE_H)
    return {**mom2, "requests": len(reqs), "facts_skipped_subject_not_in_prompt": skipped,
            "layers": list(hparams.layers), "edit_module": hparams.rewrite_module_tmp,
            "neutral_target": getattr(args, "edit_target", "eos"), "retain_requests": 0}


def configure_mom2(args, hp, model, need_vkt=False):
    """Point a ZeroUnlearn / ROME / MEMIT hparams object at --mom2-dataset and check its caches
    (built by scripts/build_edit_covariance.py) exist for every edited layer."""
    import os

    from build_edit_covariance import cache_paths

    name = getattr(args, "mom2_dataset", None)
    if not name:
        return {"mom2_dataset": hp.mom2_dataset, "mom2_text": "data/wikidata (vendored default)"}
    if getattr(args, "mom2_text", None):
        os.environ["ZU_MOM2_TEXT_DIR"] = str(args.mom2_text)
    hp.mom2_dataset = name
    model_name = str(model.config._name_or_path).rsplit("/")[-1]
    missing = []
    for L in hp.layers:
        k, v = cache_paths(SEMANTIC_ROOT / "data" / "stats", model_name, name, L, int(hp.mom2_n_samples),
                           template=hp.rewrite_module_tmp)
        missing += [str(k)] if not k.is_file() else []
        missing += [str(v)] if need_vkt and not v.is_file() else []
    if missing and not getattr(args, "allow_mom2_compute", False):
        raise FileNotFoundError(
            f"covariance cache missing for --mom2-dataset {name}: {missing[:3]}... "
            "run scripts/build_edit_covariance.py first (or pass --allow-mom2-compute)")
    return {"mom2_dataset": name, "mom2_text": getattr(args, "mom2_text", None),
            "mom2_n_samples": int(hp.mom2_n_samples), "mom2_cache_missing": missing}


def _import_zero_module(zero_root, name):
    """Import a vendored ZeroUnlearn package (rome, memit); util.globals reads globals.yml
    from the working directory at import time, so import from inside ZeroUnlearn/."""
    import importlib
    import sys

    import run_zerounlearn_official_mcf as zb

    root = str(Path(zero_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    with zb.working_directory(Path(zero_root)):
        return importlib.import_module(name)


def rome_hparams(path, rome_module):
    """ROMEHyperParams from ZeroUnlearn's hparams/ROME/<model>.json. As shipped, from_json
    fails on that file (it lacks context_template_length_params and carries MEMIT-only keys)
    and compute_u reads hparams.stats_dir, which the class lacks. Keep the class's fields,
    take ROME's own context-template default and STATS_DIR (as MEMIT / ZeroUnlearn use)."""
    import dataclasses

    from util.globals import STATS_DIR

    data = json.loads(Path(path).read_text())
    data.setdefault("context_template_length_params", [[5, 10], [10, 10]])   # ROME's default
    names = {f.name for f in dataclasses.fields(rome_module.ROMEHyperParams)}
    hp = rome_module.ROMEHyperParams(**{k: v for k, v in data.items() if k in names})
    hp.stats_dir = STATS_DIR
    return hp, sorted(set(data) - names)


def apply_rome_memit(args, model, tok, artifact, seed, method):
    """ROME / MEMIT (vendored in ZeroUnlearn/, hparams from ZeroUnlearn/hparams) on the same
    requests as the ZeroUnlearn baseline: target_true -> --edit-target, no retain requests.
    ROME is applied sequentially, one rank-one update per request (apply_rome_to_model only
    applies request[0], so passing all requests at once edits a single fact); MEMIT applies one
    batched update for all requests. Covariance statistics: data/stats under semantic-unlearning
    (shared with ZeroUnlearn: same down_proj layers, wikidata, 100k samples)."""
    import run_zerounlearn_official_mcf as zb

    reqs, skipped = zero_requests(artifact["facts"], tok, args.edit_target)
    zb.set_all_seeds(seed)
    mod = _import_zero_module(args.zero_root, method)
    if method == "rome":
        hp, ignored = rome_hparams(args.rome_hparams, mod)
        mom2 = configure_mom2(args, hp, model)
        with zb.working_directory(zb.SEMANTIC_ROOT):
            for r in reqs:
                mod.apply_rome_to_model(model, tok, [r], hp, copy=False, return_orig_weights=False)
        info = {"update": "sequential rank-one update per request", "hparams_keys_ignored": ignored,
                "hparams": str(args.rome_hparams)}
    else:
        hp = mod.MEMITHyperParams.from_json(args.memit_hparams)
        mom2 = configure_mom2(args, hp, model)
        with zb.working_directory(zb.SEMANTIC_ROOT):
            mod.apply_memit_to_model(model, tok, reqs, hp, copy=False, return_orig_weights=False,
                                     cache_template=None)
        info = {"update": "one batched MEMIT update for all requests", "hparams": str(args.memit_hparams)}
    return {**mom2, "requests": len(reqs), "facts_skipped_subject_not_in_prompt": skipped,
            "layers": list(hp.layers), "edit_module": hp.rewrite_module_tmp,
            "neutral_target": args.edit_target, "retain_requests": 0, **info}


# ------------------------------------------------------------------------------ entry
def apply(args, model, tok, artifact, run_dir, seed, mcf_path=None):
    """Edit `model` in place with args.baseline; returns a JSON-able report."""
    if not args.baseline:
        return None
    if float(torch.as_tensor(artifact["rows"]).abs().max()) != 0.0:
        raise ValueError("--baseline must run on a router dir with all-zero rows (the base dir)")
    t0 = time.perf_counter()
    bseed = int(args.baseline_seed if args.baseline_seed is not None else seed)
    torch.manual_seed(bseed)
    dtype = next(model.parameters()).dtype
    model.float()
    model.config.use_cache = False
    report = {"baseline": args.baseline, "seed": bseed}
    if args.baseline == "zerounlearn":
        report.update(apply_zerounlearn(args, model, tok, artifact, bseed))
    elif args.baseline in ("rome", "memit"):
        report.update(apply_rome_memit(args, model, tok, artifact, bseed, args.baseline))
    else:
        dataset, views = forget_views(args, artifact, run_dir, tok, seed, mcf_path)
        retain = (self_generated_retain(model, tok, args.retain_tokens)
                  if args.baseline in ("gd", "npo", "rmu") else [])
        report.update({"dataset": dataset, "forget_views": len(views),
                       "forget_facts": len({v.fact_id for v in views}),
                       "retain": f"{len(retain)} base-model self-generated continuations" if retain else None})
        fn = train_rmu if args.baseline == "rmu" else train_ga_gd_npo
        report.update(fn(args, model, tok, views, retain, bseed))
    model.requires_grad_(False)
    model.to(dtype=dtype).eval()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    report["protocol"] = ("same training-visible forget views as ours; retain = base self-generated "
                          "text; evaluated by the unchanged official evaluator")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}), flush=True)
    return report
