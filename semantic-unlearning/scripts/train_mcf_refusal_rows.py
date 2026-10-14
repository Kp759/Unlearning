#!/usr/bin/env python3
"""Train the 50 Δe rows toward a refusal target instead of only "away from the answer".

The shipped rows push P(answer) below 1e-6 for the token right after the prompt
(the " I don't know." phase only starts after that and moves inside a tiny trust
radius), so in free generation the model often states the answer a few tokens
later. ZeroUnlearn instead trains the forget slot toward EOS, so the edited model
simply stops. This trains the same router + rows (setting A: layer 19, last prompt
token, nothing else changes; unrouted prompts stay bit-exact base) with

    loss = NLL(target | prompt) + forget_weight * relu(-log 1e-6 - NLL(answer | prompt))

  --target eos   target = the tokenizer's EOS token        (ZeroUnlearn-style stop)
  --target idk   target = " I don't know." + EOS           (decline, then stop)

Rows start from zero, plain Adam on full rows, each row's norm capped at
--max-row-norm-ratio x the median layer-19 boundary norm. Same training data as the
shipped rows (train + development views the router sends to their own row); the
official eval prompts are never used.

    python -u scripts/train_mcf_refusal_rows.py --router-dir R --output-dir O --target eos \\
        --local-files-only
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

from layer_sweep_utils import boundary_norms
from linear_router import ARCHITECTURE, load_linear_classifier_artifact
from prepare_mcf_association_source import load_mcf_forget_data
from static_overlap_extended_tokens_v2 import batched_answer_nll
from static_overlap_fact_association_embeddings import (
    PLAN,
    FactAssociationEditor,
    make_unknown_examples,
)
from train_mcf_linear_router_rows import NEUTRAL_PROMPT, _routes_on_training_inputs

IDK_TEXT = " I don't know."


def target_completion(tokenizer, target):
    eos = tokenizer.eos_token
    if not isinstance(eos, str) or tokenizer(eos, add_special_tokens=False)["input_ids"] != [
            tokenizer.eos_token_id]:
        raise RuntimeError(f"EOS {eos!r} does not encode to the single eos_token_id")
    return eos if target == "eos" else IDK_TEXT + eos


@torch.no_grad()
def evaluate(model, answer, target, ids, batch_size):
    """Mean target NLL, answer probabilities (exp(-mean NLL)) over the given example ids."""
    t_nll, a_prob = [], []
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        t_nll += batched_answer_nll(model, [target[i] for i in chunk]).tolist()
        a_prob += torch.exp(-batched_answer_nll(model, [answer[i] for i in chunk])).tolist()
    return {"views": len(ids), "target_nll_mean": sum(t_nll) / max(1, len(t_nll)),
            "target_prob_mean": sum(math.exp(-x) for x in t_nll) / max(1, len(t_nll)),
            "answer_prob_max": max(a_prob, default=0.0),
            "answer_prob_mean": sum(a_prob) / max(1, len(a_prob))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--target", choices=("eos", "idk"), required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--target-probability", type=float, default=PLAN["target_probability"])
    p.add_argument("--max-row-norm-ratio", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    router_dir = Path(args.router_dir).resolve()
    output = Path(args.output_dir).resolve()
    source = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    if str(source.get("architecture")) != ARCHITECTURE:
        raise ValueError(f"{router_dir} is not a linear-classifier router artifact")
    if float(source["rows"].abs().max()) != 0.0:
        raise ValueError("Router artifact already has trained rows; expected a fresh fit")
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    layer = int(source["layer"])
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int((manifest.get("sampling") or {}).get("seed", 1))
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

    _, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    if [f["id"] for f in facts] != [f["id"] for f in source["facts"]]:
        raise ValueError("Rebuilt MCF facts do not match the router artifact")
    completion = target_completion(tok, args.target)
    answer = {e.id: e for e in examples}
    refusal = make_unknown_examples(examples, tok, PLAN["max_length"], completion)
    target = {e.id: refusal[e.id] for e in examples}

    norms = boundary_norms(model, tok, [e.prompt for e in examples if e.split == "train"], [layer])
    max_norm = args.max_row_norm_ratio * float(norms[layer].median())

    neutral = tok(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()
    _, bank = load_linear_classifier_artifact(model, source)
    for row in bank.rows:
        row.requires_grad_(True)
    editor = FactAssociationEditor(model, bank)
    lm = editor.model

    fact_to_row = {f["id"]: i for i, f in enumerate(facts)}
    routed_answer = _routes_on_training_inputs(lm, bank, examples, fact_to_row)
    routed_target = _routes_on_training_inputs(lm, bank, list(target.values()), fact_to_row)
    use = [e.id for e in examples
           if routed_answer[e.id] and routed_target[target[e.id].id]]
    by_split = {s: [i for i in use if answer[i].split == s] for s in ("train", "development")}
    coverage = {s: {"views": sum(e.split == s for e in examples), "used_for_training": len(v)}
                for s, v in by_split.items()}
    print(json.dumps({"phase": "refusal_rows_ready", "target": args.target,
                      "completion": completion, "coverage": coverage,
                      "max_row_norm": max_norm}), flush=True)
    if not use:
        raise RuntimeError("The router sends no training view to its own row")

    optimizer = torch.optim.Adam(list(bank.rows), lr=args.lr)
    tau = -math.log(args.target_probability)
    log = [{"step": 0, **{s: evaluate(lm, answer, target, v, args.batch_size)
                          for s, v in by_split.items()}}]
    print(json.dumps(log[-1]), flush=True)
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        batch = rng.sample(use, min(args.batch_size, len(use)))
        t_nll = batched_answer_nll(lm, [target[i] for i in batch])
        a_nll = batched_answer_nll(lm, [answer[i] for i in batch])
        loss = t_nll.mean() + args.forget_weight * torch.relu(tau - a_nll).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for row in bank.rows:
                n = row.norm()
                if n > max_norm:
                    row.mul_(max_norm / n)
        if step % args.eval_every == 0 or step == args.steps:
            log.append({"step": step, "loss": float(loss), "seconds": time.perf_counter() - started,
                        **{s: evaluate(lm, answer, target, v, args.batch_size)
                           for s, v in by_split.items()}})
            print(json.dumps(log[-1]), flush=True)

    with torch.no_grad():
        if not torch.equal(base_logits, lm(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched natural prompt left the exact base path after training")
    rows = bank.extra.detach().cpu()
    artifact = dict(source)
    artifact["rows"] = rows
    artifact["training_route"] = "router"
    torch.save(artifact, output / "fact_association_embeddings.pt")
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": "sure_linear_router_refusal_rows",
        "refusal_target": args.target, "refusal_completion": completion,
        "training_route": "router", "training_coverage": coverage,
        "refusal_training": {k: v for k, v in vars(args).items()
                             if k not in ("router_dir", "output_dir", "device")},
        "router_copied_from": str(router_dir),
    })
    (output / "association_manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")
    for name in ("association_examples.json", "linear_router_report.json"):
        if (router_dir / name).is_file():
            shutil.copy2(router_dir / name, output / name)
    row_norms = rows.float().norm(dim=-1)
    report = {"target": args.target, "completion": completion, "training_coverage": coverage,
              "log": log, "row_norm_median": float(row_norms.median()),
              "row_norm_max": float(row_norms.max()), "max_row_norm": max_norm,
              "boundary_norm_median": float(norms[layer].median()),
              "unmatched_neutral_logits_exact_base_after_training": True}
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "refusal_rows_trained", "output_dir": str(output),
                      "final": log[-1]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
