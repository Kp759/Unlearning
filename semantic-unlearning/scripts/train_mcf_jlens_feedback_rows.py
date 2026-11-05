#!/usr/bin/env python3
"""J-lens feedback: train the 50 Δe rows so the answer is gone INSIDE the forward pass.

No refusal target. The rows (layer 19, same router; unrouted prompts stay exact base) are
trained until the answer is neither generated nor readable at a later layer:

  loop over rounds
    1. feedback: greedy-generate continuations of the routed training views WITH the
       current rows (round 0 = base continuations, which often state the answer)
    2. on every continuation position t (from the prompt boundary on), penalise
         out   relu(log p_final(answer_tok | x_<t) - log tau)                   final output
         lens  relu(z_L(answer_tok, t) - z_max)                               J-lens at layer L
               (--lens-type logit: relu(log softmax(lm_head(norm(h_L(t))))[answer] - log
               lens_tau), i.e. the plain logit lens at layer L instead;
               --lens-type none: no lens term, the layer-L state is never read)
               z = (u_w . h_L(t) - mu_w) / sd_w, u_w = mean d logit_w / d h_L
               over generic MCF prompts (forget subjects excluded); mu/sd = the same
               score on those prompts, so z <= 0 means "no more answer evidence than an
               unrelated prompt"
         keep  KL(base with the answer tokens masked || edited)               fluency
         keepS (--preserve-weight, early write only) KL(base || edited) on the SAME subject's
               OTHER relations ("X, a citizen of", "X works as", ...) with the fact's edit forced
               on: erase/block only the target fact, keep the subject's other facts. Uses the
               "preserve" half of the relations (mcf_subject_collateral.py evaluates the other half)
         rep   unlikelihood on tokens that would repeat a bigram already in the
               sequence: -log(1 - p(c)) (Welleck et al. 2019); stops loops   (--rep-weight)
       train the rows on every continuation collected so far (replay buffer)

Feedback on generated positions matters: the earlier Δe hit Eff ~ 3e-5 at the boundary
and still stated the answer a few tokens later.

  --write-layer L < router layer   route at layer 19 but write the rows after block L at
                                   the subject tokens (scripts/early_write.py), i.e. where
                                   the fact is recalled, so later layers/positions cannot
                                   re-read it; --write-mode subject_span | subject_last

  --write-mode query (no --write-layer, or --write-layer = router layer): ONE forward pass.
                             The router reads layer L and the row is added at the same layer
                             L, at the last prompt token and every generated token (the
                             default of run_l16_paper_one.sh: classifier input and Δe at 19).
  --write-mode last          edit the last prompt token only (setting A)
  --write-mode subject_span  edit the subject tokens + boundary (setting C): later
                             positions read the subject, so this is where the answer
                             is re-extracted from

    python -u scripts/train_mcf_jlens_feedback_rows.py --router-dir R --output-dir O \\
        --write-mode subject_span --local-files-only
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
import torch.nn.functional as F

from layer_sweep_utils import boundary_norms
from early_write import EARLY_WRITE_MODES, EarlyWriteCausalLM
from linear_router import ARCHITECTURE, WRITE_MODES, load_linear_classifier_artifact
from mcf_projection_edit import answer_logit_gradients, generic_prompts
from prepare_mcf_association_source import load_mcf_forget_data
from static_overlap_fact_association_embeddings import FactAssociationEditor
from train_mcf_linear_router_rows import NEUTRAL_PROMPT, _routes_on_training_inputs


# ----------------------------------------------------------------------- answers / lens

def _is_canonical(group):
    """MCF names its direct prompt 'canonical_rewrite'; ZsRE/MQuAKE/RWKU 'canonical_<k>'."""
    return str(group) == "canonical_rewrite" or str(group).startswith("canonical_")


def detect_dataset(facts):
    first = str(facts[0].get("id", ""))
    # multifact: MultiFactPerson-format facts, e.g. SCALPEL's requests (scalpel/methods/external.py)
    for name in ("mcf", "zsre", "mquake", "rwku", "multifact"):
        if first.startswith(f"{name}_"):
            return name
    raise ValueError(f"cannot tell the benchmark from fact id {first!r}")


def generic_examples(router_dir, facts, tok, max_length=512):
    """ZsRE / MQuAKE / RWKU: the router's own training-visible prompts (its examples file,
    else canonical prompts + context-prefix families, exactly as fit_linear_router builds
    them), encoded as prompt + ' ' + answer with the answer labelled."""
    from linear_router import examples_from_facts
    from static_overlap_natural_writer import _encode_answer_example

    path = Path(router_dir) / "association_examples.json"
    rows = (json.loads(path.read_text()) if path.is_file()
            else examples_from_facts(facts, augment=True))
    by_id = {f["id"]: f for f in facts}
    out, skipped, seen = [], 0, set()
    for r in rows:
        fact = by_id.get(r.get("fact_id"))
        if fact is None or str(r.get("split")) not in ("train", "development"):
            continue
        group = str(r.get("group") or r.get("role") or "canonical_0")
        try:
            e = _encode_answer_example(fact, str(r["split"]), group, str(r["prompt"]), tok, max_length)
        except ValueError:
            skipped += 1
            continue
        if e.id in seen:
            continue
        seen.add(e.id)
        out.append(e)
    if not out:
        raise ValueError("no encodable training-visible examples")
    return out, skipped


def answer_token_sets(tok, examples):
    """fact_id -> first-token ids of the answer: as it follows the prompt, and bare."""
    out = {}
    for e in examples:
        if e.split != "train" or not _is_canonical(e.group) or e.fact_id in out:
            continue
        first = next(k for k, lab in enumerate(e.labels) if lab != -100)
        ids = {int(e.input_ids[first])}
        answer = e.completion.strip()
        for text in (answer, " " + answer):
            pieces = tok(text, add_special_tokens=False)["input_ids"]
            if pieces:
                ids.add(int(pieces[0]))
        out[e.fact_id] = sorted(ids)
    return out


def _grab(store, key):
    def hook(_m, _a, out):
        store[key] = out[0] if isinstance(out, tuple) else out
    return hook


def jlens(model, layer, prompts, tokens, device, batch_size=16):
    """J-lens rows u_w at `layer` plus background mean/std of u_w . h over generic prompts."""
    grads = answer_logit_gradients(model, layer, prompts, [tokens] * len(prompts), device,
                                   batch_size)
    U = torch.stack([grads[w][0] / grads[w][1] for w in tokens]).float()      # [T, d]
    scores = []
    store = {}
    handle = model.model.layers[layer].register_forward_hook(_grab(store, "h"))
    try:
        with torch.no_grad():
            for start in range(0, len(prompts), batch_size):
                chunk = prompts[start:start + batch_size]
                width = max(len(p) for p in chunk)
                ids = torch.zeros((len(chunk), width), dtype=torch.long, device=device)
                mask = torch.zeros_like(ids)
                for r, p in enumerate(chunk):
                    ids[r, :len(p)] = torch.tensor(p, device=device)
                    mask[r, :len(p)] = 1
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                h = store["h"].float()
                s = h @ U.to(device).T                                           # [B, W, T]
                keep = mask.bool().clone()
                keep[:, 0] = False                                              # skip BOS
                scores.append(s[keep].cpu())
    finally:
        handle.remove()
    s = torch.cat(scores)
    return U.cpu(), s.mean(0), s.std(0).clamp_min(1e-6)


# --------------------------------------------------------------------------- sequences

def pad_batch(seqs, device):
    width = max(len(s["ids"]) for s in seqs)
    ids = torch.zeros((len(seqs), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    pos = torch.zeros((len(seqs), width), dtype=torch.bool, device=device)
    for r, s in enumerate(seqs):
        n = len(s["ids"])
        ids[r, :n] = torch.tensor(s["ids"], device=device)
        mask[r, :n] = 1
        pos[r, s["prompt_len"] - 1:n] = True     # every point where a token gets generated
    return ids, mask, pos


def lens_layer_of(args):
    """The layer the lens term reads, or None with --lens-type none."""
    return None if args.lens_type == "none" else args.lens_layer


def forward(lm, base_model, lens_layer, seqs, device):
    ids, mask, pos = pad_batch(seqs, device)
    lm.set_association_prefix_lengths([s["prompt_len"] for s in seqs])
    store = {}
    handle = (base_model.model.layers[lens_layer].register_forward_hook(_grab(store, "h"))
              if lens_layer is not None else None)
    try:
        logits = lm(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    return logits, (store["h"].float() if handle is not None else None), pos


def gather_answers(seqs, token_index, device):
    """[B, K] answer-token ids and their J-lens row indices, padded by repetition."""
    k = max(len(s["answer_tokens"]) for s in seqs)
    tok = torch.tensor([s["answer_tokens"] + s["answer_tokens"][:1] * (k - len(s["answer_tokens"]))
                        for s in seqs], device=device)
    rows = torch.tensor([[token_index[int(t)] for t in r] for r in tok.tolist()], device=device)
    return tok, rows


@torch.no_grad()
def base_logits(lm, params, base_model, lens_layer, seqs, device):
    saved = [row.detach().clone() for row in params]
    for row in params:
        row.zero_()
    try:
        logits, _, _ = forward(lm, base_model, lens_layer, seqs, device)
    finally:
        for row, value in zip(params, saved):
            row.copy_(value)
    return logits


def repeat_candidates(seqs):
    """(row, position, token) triples: predicting `token` at `position` would repeat a bigram.

    Only generation positions (from the prompt boundary on) are penalised; the bigrams may
    come from anywhere earlier in the sequence (prompt included)."""
    triples = []
    for r, s in enumerate(seqs):
        ids, seen = s["ids"], {}
        for t in range(len(ids) - 1):
            if t >= s["prompt_len"] - 1:
                for c in seen.get(ids[t], ()):
                    triples.append((r, t, c))
            seen.setdefault(ids[t], set()).add(ids[t + 1])
    return triples


def losses(lm, params, base_model, args, seqs, U, mu, sd, token_index, device):
    tok, rows = gather_answers(seqs, token_index, device)
    # Base logits first: zeroing/restoring the rows in place after the edited forward
    # would invalidate the autograd graph.
    base = (base_logits(lm, params, base_model, lens_layer_of(args), seqs, device)
            if args.keep_weight > 0 else None)
    logits, h, pos = forward(lm, base_model, lens_layer_of(args), seqs, device)
    logp = logits.log_softmax(-1)
    lp = logp.gather(-1, tok[:, None, :].expand(-1, logp.shape[1], -1))           # [B, W, K]
    if args.lens_type == "none":
        z = None
    elif args.lens_type == "logit":
        # Plain logit lens: final norm + unembedding applied to the layer-L state.
        lens_logp = base_model.get_output_embeddings()(base_model.model.norm(h)).float().log_softmax(-1)
        z = lens_logp.gather(-1, tok[:, None, :].expand(-1, lens_logp.shape[1], -1))  # [B, W, K]
        del lens_logp
    else:
        z = ((h @ U.T)[..., :] - mu) / sd                                         # [B, W, T]
        z = z.gather(-1, rows[:, None, :].expand(-1, z.shape[1], -1))             # [B, W, K]
    m = pos[..., None].float()
    per_seq = lambda x: (x * m).sum((1, 2))
    out = per_seq(torch.relu(lp - math.log(args.tau))).mean()
    lens_floor = math.log(args.lens_tau) if args.lens_type == "logit" else args.z_max
    lens = per_seq(torch.relu(z - lens_floor)).mean() if z is not None else torch.zeros((), device=device)
    keep = torch.zeros((), device=device)
    if base is not None:
        base.scatter_(-1, tok[:, None, :].expand(-1, base.shape[1], -1), float("-inf"))
        bp = base.softmax(-1)
        kl = (bp * (bp.clamp_min(1e-30).log() - logp)).sum(-1)                     # [B, W]
        keep = (kl * pos).sum() / pos.sum()
    rep = torch.zeros((), device=device)
    if args.rep_weight > 0:
        triples = repeat_candidates(seqs)
        if triples:
            idx = torch.tensor(triples, device=device)
            p_rep = logp[idx[:, 0], idx[:, 1], idx[:, 2]].exp().clamp(max=1 - 1e-6)
            rep = -torch.log1p(-p_rep).sum() / len(seqs)
    loss = (args.out_weight * out + args.lens_weight * lens + args.keep_weight * keep
            + args.rep_weight * rep)
    stats = {"out": float(out.detach()), "lens": float(lens.detach()), "keep": float(keep.detach()),
             "rep": float(rep.detach()),
             "max_logp_answer": float((lp.detach().amax(-1) + (~pos) * -1e9).amax(-1).mean())}
    if z is not None:
        stats["max_lens_logp" if args.lens_type == "logit" else "max_z"] = \
            float((z.detach().amax(-1) + (~pos) * -1e9).amax(-1).mean())
    return loss, stats


@torch.no_grad()
def collect(lm, tok, views, args, round_index):
    seqs = []
    for v in views:
        ids = torch.tensor([v["prompt_ids"]], device=args.device)
        out = lm.generate_uncached_fixed_boundary(ids, max_new_tokens=args.gen_tokens)[0].tolist()
        text = tok.decode(out[len(v["prompt_ids"]):], skip_special_tokens=True)
        seqs.append({**v, "ids": out, "prompt_len": len(v["prompt_ids"]), "round": round_index,
                     "continuation": text,
                     "leak": v["answer"].casefold() in text.casefold()})
    return seqs


# -------------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--write-mode", choices=WRITE_MODES + EARLY_WRITE_MODES, default="last")
    p.add_argument("--write-layer", type=int, default=None,
                   help="write layer < router layer = early write at the subject tokens")
    p.add_argument("--lens-layer", type=int, default=24)
    p.add_argument("--lens-type", choices=("jlens", "logit", "none"), default="jlens",
                   help="jlens: z-scored Jacobian-lens score; logit: plain logit lens; none: no lens term")
    p.add_argument("--lens-tau", type=float, default=1e-4,
                   help="logit lens: target answer probability at the lens layer")
    p.add_argument("--jlens-prompts", type=int, default=256)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--steps-per-round", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--views-per-fact", type=int, default=6,
                   help="routed training views per fact used for on-policy feedback")
    p.add_argument("--gen-tokens", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--tau", type=float, default=1e-4, help="target answer-token prob")
    p.add_argument("--z-max", type=float, default=0.0)
    p.add_argument("--out-weight", type=float, default=1.0)
    p.add_argument("--lens-weight", type=float, default=1.0)
    p.add_argument("--keep-weight", type=float, default=0.1)
    p.add_argument("--preserve-weight", type=float, default=0.0,
                   help="same-subject other-relation KL to base (early write only; 0 = off)")
    p.add_argument("--preserve-tokens", type=int, default=12)
    p.add_argument("--rep-weight", type=float, default=0.0,
                   help="repetition unlikelihood weight (0 = off)")
    p.add_argument("--max-row-norm-ratio", type=float, default=1.0)
    p.add_argument("--oracle-unrouted", action="store_true",
                   help="train facts whose views the router never routes with their own row "
                        "forced (training only; test-time routing unchanged)")
    p.add_argument("--teacher-forced-views", action="store_true",
                   help="also train on each view's gold answer, teacher-forced, penalising every "
                        "answer token at its own position (ZsRE/MQuAKE score token by token "
                        "under the gold prefix, which on-policy generation never visits)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    router_dir = Path(args.router_dir).resolve()
    output = Path(args.output_dir).resolve()
    source = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    factorized = str(source.get("architecture")) == "factorized_subject_relation_router_v1"
    if str(source.get("architecture")) != ARCHITECTURE and not factorized:
        raise ValueError(f"{router_dir} is not a linear-classifier or factorized router artifact")
    if float(source["rows"].abs().max()) != 0.0:
        raise ValueError("Router artifact already has trained rows; expected a fresh fit")
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    layer = int(source["layer"])
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset = detect_dataset(source["facts"])
    seed = int((manifest.get("sampling") or {}).get("seed") or manifest.get("seed") or 1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    # The lens must sit after the layer that is EDITED (the write layer for an early write;
    # routing happens on the unedited pass 1, so the router layer may be later than the lens).
    edit_layer = layer if args.write_layer is None else int(args.write_layer)
    if lens_layer_of(args) is not None and not edit_layer < args.lens_layer < n_layers:
        raise ValueError(f"--lens-layer must be in ({edit_layer}, {n_layers - 1}]")

    if dataset == "mcf":
        forget_records, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
        if [f["id"] for f in facts] != [f["id"] for f in source["facts"]]:
            raise ValueError("Rebuilt MCF facts do not match the router artifact")
    else:
        if args.preserve_weight > 0:
            raise ValueError("--preserve-weight uses MCF relation templates (MCF only)")
        if args.lens_type == "jlens":
            raise ValueError("--lens-type jlens needs MCF background prompts; use --lens-type logit")
        facts = list(source["facts"])
        examples, skipped = generic_examples(router_dir, facts, tok)
        print(json.dumps({"phase": "examples", "dataset": dataset, "examples": len(examples),
                          "skipped_unencodable": skipped}), flush=True)
    answers = answer_token_sets(tok, examples)
    missing = [f["id"] for f in facts if f["id"] not in answers]
    if missing:
        raise ValueError(f"{len(missing)} facts have no canonical train example: {missing[:3]}")
    # teacher-forced views: every labelled answer token of the view's gold continuation
    tf_tokens = {}
    if args.teacher_forced_views:
        for e in examples:
            labelled = {int(t) for t in e.labels if t != -100}
            tf_tokens.setdefault(e.fact_id, set()).update(labelled | set(answers[e.fact_id]))
    all_tokens = sorted({t for ts in answers.values() for t in ts}
                        | {t for ts in tf_tokens.values() for t in ts})
    token_index = {t: i for i, t in enumerate(all_tokens)}

    t0 = time.perf_counter()
    U = mu = sd = None                     # --lens-type none: no lens statistics at all
    if args.lens_type != "none":
        if dataset == "mcf":
            prompts = generic_prompts(tok, manifest["mcf_path"], facts,
                                      args.jlens_prompts if args.lens_type == "jlens" else 2)
        else:   # logit lens only: the J-lens background is never used, any two prompts do
            prompts = [tok(t)["input_ids"] for t in (NEUTRAL_PROMPT, "The weather today is")]
        U, mu, sd = jlens(model, args.lens_layer, prompts, all_tokens, args.device)  # unused for logit
        torch.save({"layer": args.lens_layer, "tokens": all_tokens, "U": U, "mu": mu, "sd": sd,
                    "prompts": len(prompts)}, output / f"jlens_L{args.lens_layer}.pt")
        print(json.dumps({"phase": "jlens_ready", "layer": args.lens_layer, "tokens": len(all_tokens),
                          "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
        U, mu, sd = U.to(args.device), mu.to(args.device), sd.to(args.device)

    # The factorized router (scripts/factorized_router.py) routes in its own pass-1 forward,
    # so every write goes through EarlyWriteCausalLM, whatever the write layer.
    early = args.write_layer is not None and (args.write_layer != layer or factorized)
    if factorized and not early:
        raise ValueError("a factorized router needs --write-layer (EarlyWriteCausalLM write)")
    if early and args.write_mode not in EARLY_WRITE_MODES:
        raise ValueError(f"early write needs --write-mode in {EARLY_WRITE_MODES}")
    if not early and args.write_mode not in WRITE_MODES:
        raise ValueError(f"router-layer write needs --write-mode in {WRITE_MODES}")
    norm_layer = args.write_layer if early else layer
    norms = boundary_norms(model, tok, [e.prompt for e in examples if e.split == "train"],
                           [norm_layer])
    max_norm = args.max_row_norm_ratio * float(norms[norm_layer].median())

    neutral = tok(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        neutral_logits = model(**neutral, use_cache=False).logits.detach().clone()
    source = dict(source)
    source["edit_mode"] = "add"
    if early:
        lm = bank = EarlyWriteCausalLM(model, source, args.write_layer, args.write_mode)
        if getattr(lm, "bank", None) is not None:
            lm.bank.tokenizer = tok             # fuzzy subject matching decodes the prompt
        lm.rows.requires_grad_(True)
        params = [lm.rows]
    else:
        source["write_mode"] = args.write_mode
        lm_wrapped, bank = load_linear_classifier_artifact(model, source)
        bank.tokenizer = tok                    # fuzzy subject matching decodes the prompt
        for row in bank.rows:
            row.requires_grad_(True)
        editor = FactAssociationEditor(model, bank)
        lm = editor.model
        params = list(bank.rows)

    fact_to_row = {f["id"]: i for i, f in enumerate(facts)}
    routed = _routes_on_training_inputs(lm, bank, examples, fact_to_row)
    by_fact = {}
    for e in examples:
        if routed[e.id]:
            by_fact.setdefault(e.fact_id, []).append(e)
    oracle_facts = []
    if args.oracle_unrouted:
        # Facts the router never sends to their own row would keep a zero row. Train them
        # with their own row forced on their training views (training only: the saved
        # artifact still routes by the router, so test-time routing is unchanged; genie
        # then measures what the rows can do for every fact).
        oracle_facts = sorted({e.fact_id for e in examples} - set(by_fact))
        for e in examples:
            if e.fact_id in oracle_facts:
                by_fact.setdefault(e.fact_id, []).append(e)
                first = next(k for k, lab in enumerate(e.labels) if lab != -100)
                bank.force_route(list(e.input_ids[:first]), fact_to_row[e.fact_id])   # early: bank is lm
        if oracle_facts:
            print(json.dumps({"phase": "oracle_unrouted", "facts": len(oracle_facts),
                              "ids": oracle_facts[:10]}), flush=True)
    views, tf_seqs = [], []
    for fid, exs in by_fact.items():
        exs = sorted(exs, key=lambda e: (not _is_canonical(e.group), e.split != "train", e.id))
        for e in exs[:args.views_per_fact]:
            first = next(k for k, lab in enumerate(e.labels) if lab != -100)
            views.append({"view_id": e.id, "fact_id": fid, "split": e.split,
                          "prompt_ids": list(e.input_ids[:first]),
                          "answer": e.completion.strip(), "answer_tokens": answers[fid]})
            if args.teacher_forced_views:
                tf_seqs.append({"view_id": e.id, "fact_id": fid, "split": e.split,
                                "ids": list(e.input_ids), "prompt_len": first, "round": -1,
                                "continuation": e.completion, "leak": True,
                                "answer": e.completion.strip(),
                                "answer_tokens": sorted(tf_tokens[fid])})
    routed_facts = len(by_fact)
    if not views:
        raise ValueError("the router routes none of the training views to their own fact")
    if routed_facts < len(facts):
        print(json.dumps({"phase": "warning", "facts_without_routed_view": len(facts) - routed_facts,
                          "note": "their rows stay zero (the router never sends their prompts)"}),
              flush=True)
    preserve = []
    if args.preserve_weight > 0:
        if not early:
            raise ValueError("--preserve-weight needs an early write (--write-layer < router layer)")
        from mcf_subject_collateral import relation_templates, subject_probes
        probes = subject_probes(forget_records, relation_templates(manifest["mcf_path"]), "preserve")
        with torch.no_grad():
            for i, plist in enumerate(probes):
                for _, prompt in plist:
                    ids = tok(prompt)["input_ids"]
                    lm.force_route(ids, -1)                # no edit: exact base continuation
                    full = lm.generate_uncached_fixed_boundary(
                        torch.tensor([ids], device=args.device),
                        max_new_tokens=args.preserve_tokens)[0].tolist()
                    if len(full) > len(ids):
                        lm.force_route(ids, i)             # this fact's edit, as inside its reply
                        preserve.append({"ids": full, "prompt_len": len(ids)})
        print(json.dumps({"phase": "preserve_prompts", "prompts": len(preserve)}), flush=True)
    print(json.dumps({"phase": "feedback_views", "views": len(views), "facts": len(by_fact),
                      "write_mode": args.write_mode, "write_layer": norm_layer,
                      "max_row_norm": max_norm}), flush=True)

    optimizer = torch.optim.Adam(params, lr=args.lr)
    buffer, log = list(tf_seqs), []        # teacher-forced gold views stay in the replay buffer
    started = time.perf_counter()
    for r in range(args.rounds + 1):
        fresh = collect(lm, tok, views, args, r)
        leak = sum(s["leak"] for s in fresh) / len(fresh)
        buffer += fresh
        entry = {"round": r, "onpolicy_leak": leak, "buffer": len(buffer),
                 "seconds": round(time.perf_counter() - started, 1),
                 "examples": [{"prompt": tok.decode(s["ids"][:s["prompt_len"]], skip_special_tokens=True),
                               "continuation": s["continuation"], "leak": s["leak"]}
                              for s in fresh[:6]]}
        if r == args.rounds:                         # last collection = evaluation only
            log.append(entry)
            print(json.dumps({k: v for k, v in entry.items() if k != "examples"}), flush=True)
            break
        acc = []
        for _ in range(args.steps_per_round):
            batch = rng.sample(buffer, min(args.batch_size, len(buffer)))
            loss, stats = losses(lm, params, model, args, batch, U, mu, sd, token_index, args.device)
            if preserve:
                pb = rng.sample(preserve, min(args.batch_size, len(preserve)))
                base_pl = base_logits(lm, params, model, lens_layer_of(args), pb, args.device)
                logits_pl, _, pos_pl = forward(lm, model, lens_layer_of(args), pb, args.device)
                bp = base_pl.softmax(-1)
                kl = (bp * (bp.clamp_min(1e-30).log() - logits_pl.log_softmax(-1))).sum(-1)
                keep_s = (kl * pos_pl).sum() / pos_pl.sum()
                loss = loss + args.preserve_weight * keep_s
                stats["preserve"] = float(keep_s.detach())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for row in params:                   # [d] row or [facts, d] matrix
                    n = row.norm(dim=-1, keepdim=True)
                    row.mul_(torch.clamp(max_norm / n.clamp_min(1e-12), max=1.0))
            acc.append(stats)
        entry["train"] = {k: sum(a[k] for a in acc) / len(acc) for k in acc[0]}
        log.append(entry)
        print(json.dumps({k: v for k, v in entry.items() if k != "examples"}), flush=True)

    with torch.no_grad():
        if not torch.equal(neutral_logits, lm(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched natural prompt left the exact base path after training")
    if early:
        artifact = lm.artifact()
        rows = artifact["rows"]
    else:
        rows = bank.extra.detach().cpu()
        artifact = dict(source)
        artifact["rows"] = rows
    artifact["training_route"] = "router"
    torch.save(artifact, output / "fact_association_embeddings.pt")
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": "sure_linear_router_jlens_feedback_rows", "write_mode": args.write_mode,
        "write_layer": norm_layer, "early_write": early,
        "jlens_feedback": {k: v for k, v in vars(args).items()
                           if k not in ("router_dir", "output_dir", "device")},
        "router_copied_from": str(router_dir)})
    (output / "association_manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")
    for name in ("association_examples.json", "linear_router_report.json"):
        if (router_dir / name).is_file():
            shutil.copy2(router_dir / name, output / name)
    row_norms = rows.float().norm(dim=-1)
    report = {"write_mode": args.write_mode, "write_layer": norm_layer, "early_write": early,
              "subject_fallbacks": getattr(lm, "subject_fallbacks", None),
              "facts_routed_by_router": routed_facts - len(oracle_facts),
              "facts_trained_with_oracle_routing": oracle_facts,
              "lens_layer": lens_layer_of(args), "lens_type": args.lens_type, "rounds": log,
              "row_norm_median": float(row_norms.median()), "max_row_norm": max_norm,
              "boundary_norm_median": float(norms[norm_layer].median()),
              "unmatched_neutral_logits_exact_base_after_training": True}
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "jlens_feedback_rows_trained", "output_dir": str(output),
                      "onpolicy_leak_by_round": [e["onpolicy_leak"] for e in log]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
