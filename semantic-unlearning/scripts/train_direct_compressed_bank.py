#!/usr/bin/env python3
"""Compressed residual bank trained in the loop on direct-rewrite benchmarks (ZsRE, MQuAKE).

The direct-objective counterpart of train_mcf_compressed_bank.py. Rows are
CompressedValues(mode)(compact params) and the benchmark's own differentiable
objective trains those compact parameters jointly across facts:

  per fact: hinge on the worst sensitive answer token over the exact official
  direct-rewrite token contexts, driven below 1e-6
  (<module>.sensitive_token_state, the same loss the shipped row optimizer uses)

Checkpoints are picked on the training contexts with the shipped metric
(<module>.direct_training_metrics): fewest failing facts, then the lowest
worst-token probability. The router (weights, bias, subject patterns) is the
fitted router in --router-dir, unchanged.

    python -u scripts/train_direct_compressed_bank.py --dataset zsre \
        --router-dir outputs/zsre_multiseed_reworded_v2/seed1/L19/router \
        --output-dir outputs/compressed_multiseed_v1/zsre/seed1/L19/tied_answer \
        --value-mode tied_answer --local-files-only

Value modes (compressed_value_bank.py): full (one row per fact, the same-trainer
reference), tied_answer (facts with the same answer share one vector), answer_fixed
(no stored vector: one scalar on the answer token's output-embedding direction), ...

The saved artifact is an ordinary linear-router artifact whose rows are the
exact float32 reconstruction of the compact state, so the official evaluator
runs unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import time

import torch

from compressed_value_bank import CompressedValues, answer_directions, parse_value_mode
from linear_router import ARCHITECTURE
from static_overlap_fact_association_embeddings import AssociationCausalLM
from train_direct_linear_router_rows import dataset_adapter, genie_route_map, routes_for_cases
from train_mcf_compressed_bank import SCALAR_PARAMS, bank_from_artifact, router_storage

NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."
DIRECT = ("zsre", "mquake")


def checkpoint_key(metrics):
    """Fewest facts over the target, then the lowest worst-token probability."""
    failing = int(metrics["facts_total"]) - int(metrics["facts_passing_probability_constraint"])
    return (failing, float(metrics["maximum_sensitive_token_probability"]))


def first_answer_tokens(official, tokenizer, cases, facts, llama_like, device):
    """The first answer token id of each fact (its token_index 0 context)."""
    first = {}
    for case in cases:
        if int(case.token_index) == 0 and case.fact_id not in first:
            first[case.fact_id] = case.target_text
    missing = [f["id"] for f in facts if f["id"] not in first]
    if missing:
        raise ValueError(f"No first-token context for facts {missing[:5]}")
    ids = official.official_target_ids(
        tokenizer, [first[f["id"]] for f in facts], llama_like=llama_like, device=device)
    return [int(t) for t in ids.tolist()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DIRECT, required=True)
    p.add_argument("--router-dir", required=True, help="fitted router dir (rows all zero)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--value-mode", required=True)
    p.add_argument("--training-route", choices=("router", "oracle"), default="router")
    p.add_argument("--lr", type=float, default=0.05, help="Adam lr for vector parameters")
    p.add_argument("--scale-lr", type=float, default=0.5, help="Adam lr for scalars and codes")
    p.add_argument("--batch-facts", type=int, default=8)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--post-feasible-gates", type=int, default=2)
    p.add_argument("--max-training-seconds", type=float, default=None,
                   help="default: the benchmark's shipped cap")
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)
    mode, rank = parse_value_mode(args.value_mode)
    adapter = dataset_adapter(args.dataset)
    module, official = adapter["module"], adapter["official"]
    max_seconds = float(args.max_training_seconds or adapter["max_seconds"])

    router_dir, output = Path(args.router_dir).resolve(), Path(args.output_dir).resolve()
    source = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    if str(source.get("architecture")) != ARCHITECTURE:
        raise ValueError(f"{router_dir} is not a linear-classifier router artifact")
    if float(source["rows"].abs().max()) != 0.0:
        raise ValueError("Router artifact already has trained rows; expected a fresh fit")
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    seed = int(manifest.get("seed", 1))
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    model_path = Path(manifest["model_path"])
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    records, facts = adapter["load"](manifest, tok)
    key = adapter["fact_key"]
    if [f[key] for f in facts] != [f[key] for f in source["facts"]]:
        raise ValueError("Rebuilt facts do not match the router artifact")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    fact_to_row = {f["id"]: i for i, f in enumerate(facts)}
    neutral = tok(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()

    # Token cases need a model with the association hook only for llama_like detection.
    token_cases, llama_like = module.build_exact_direct_token_cases(records, facts, tok, model)
    answer_token_ids = first_answer_tokens(official, tok, token_cases, facts, llama_like,
                                           args.device)
    hidden = int(model.config.hidden_size)
    values = CompressedValues(mode, rank, facts, hidden, answer_token_ids=answer_token_ids,
                              answer_dirs=answer_directions(model, answer_token_ids),
                              seed=seed).to(args.device)
    bank = bank_from_artifact(model, source, values)
    wrapped = AssociationCausalLM(model, bank)
    with torch.no_grad():
        if not torch.equal(base_logits, wrapped(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched neutral prompt left the exact base path")

    routed = routes_for_cases(wrapped, bank, tok, token_cases, fact_to_row,
                              adapter["prefix_lengths"])
    pre_routing = {"token_contexts": len(token_cases), "routed_to_own_row": sum(routed.values()),
                   "fraction": sum(routed.values()) / len(token_cases)}
    excluded, untrainable = [], []
    if args.training_route == "oracle":
        bank.set_oracle_routes(genie_route_map(official, tok, token_cases, fact_to_row))
        training_cases = list(token_cases)
    else:
        training_cases = [c for c in token_cases if routed[c.id]]
        excluded = [c.id for c in token_cases if not routed[c.id]]
        kept = Counter(c.fact_id for c in training_cases)
        untrainable = [f["id"] for f in facts if kept[f["id"]] == 0]
        if len(untrainable) == len(facts):
            raise RuntimeError("The linear router routes no direct rewrite to its own row")
    # Facts that cannot be trained keep an exactly-zero row, whatever is shared.
    mask = torch.ones(len(facts), device=args.device)
    for fid in untrainable:
        mask[fact_to_row[fid]] = 0.0
    base_rows = values.rows
    values.rows = lambda: base_rows() * mask[:, None]

    by_fact = defaultdict(list)
    for c in training_cases:
        by_fact[c.fact_id].append(c)
    fact_ids = sorted(by_fact)
    scalar = [q for n, q in values.named_parameters() if n in SCALAR_PARAMS]
    vector = [q for n, q in values.named_parameters() if n not in SCALAR_PARAMS]
    groups = [g for g in ({"params": vector, "lr": args.lr}, {"params": scalar, "lr": args.scale_lr})
              if g["params"]]
    optimizer = torch.optim.Adam(groups)
    params = [q for g in groups for q in g["params"]]
    storage = values.storage()
    target = float(adapter["plan"]["target_token_probability"])
    print(json.dumps({"phase": "compressed_training_ready", "dataset": args.dataset,
                      "value_mode": args.value_mode, "seed": seed, "facts_trained": len(fact_ids),
                      "untrainable_facts": len(untrainable), "pre_training_routing": pre_routing,
                      "answer_groups": int(values.answer_groups),
                      "trainable_parameters": sum(q.numel() for q in params),
                      "storage": {k: storage[k] for k in ("per_fact_floats", "shared_floats",
                                                          "ratio_to_full_rows")}}), flush=True)

    def metrics():
        with torch.no_grad():
            return module.direct_training_metrics(wrapped, tok, by_fact, target,
                                                  llama_like=llama_like)

    first = metrics()
    best_key, best_state, best_epoch = checkpoint_key(first), values.compact_state(), 0
    gates, feasible_streak, stop_reason = [], 0, "epoch_budget"
    started, rng = time.monotonic(), random.Random(seed)
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started >= max_seconds:
            stop_reason = "wall_time_budget"
            break
        order = list(fact_ids)
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), args.batch_facts):
            batch = order[start:start + args.batch_facts]
            optimizer.zero_grad(set_to_none=True)
            for fid in batch:
                state = module.sensitive_token_state(wrapped, tok, by_fact[fid], target,
                                                     llama_like=llama_like)
                (state["loss"] / len(batch)).backward()
                epoch_loss += float(state["loss"].detach())
            torch.nn.utils.clip_grad_norm_(params, args.clip, error_if_nonfinite=True)
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m = metrics()
            k = checkpoint_key(m)
            selected = k < best_key
            if selected:
                best_key, best_state, best_epoch = k, values.compact_state(), epoch
            feasible_streak = feasible_streak + 1 if m["globally_feasible"] else 0
            with torch.no_grad():
                norms = values.rows().norm(dim=-1)
            gate = {"epoch": epoch, "elapsed_seconds": round(time.monotonic() - started, 1),
                    "mean_epoch_loss": epoch_loss / max(len(order), 1),
                    "facts_passing": m["facts_passing_probability_constraint"],
                    "max_token_prob": m["maximum_sensitive_token_probability"],
                    "row_norm_median": float(norms.median()), "selected_as_best": selected}
            gates.append(gate)
            print(json.dumps({"phase": "compressed_gate", **gate}), flush=True)
            if feasible_streak > args.post_feasible_gates:
                stop_reason = "global_target_met"
                break

    values.load_state_dict(best_state)
    bank.set_oracle_routes(None)
    with torch.no_grad():
        rebuilt = CompressedValues(mode, rank, facts, hidden, answer_token_ids=answer_token_ids,
                                   answer_dirs=(values.answer_dirs.cpu()
                                                if hasattr(values, "answer_dirs") else None),
                                   seed=seed)
        rebuilt.load_state_dict({k: v.cpu() for k, v in best_state.items()})
        rows = (rebuilt.rows() * mask.cpu()[:, None]).float().contiguous()
        trained = values.rows().detach().float().cpu()
        drift = float((rows - trained).abs().max())
        if not torch.allclose(rows, trained, rtol=1e-4, atol=1e-5):
            raise RuntimeError(f"Compact reconstruction drifted from the trained rows ({drift})")
        values.rows = lambda: rows.to(args.device)
        if not torch.equal(base_logits, wrapped(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched neutral prompt left the exact base path after training")
    all_by_fact = defaultdict(list)
    for c in token_cases:
        all_by_fact[c.fact_id].append(c)
    with torch.no_grad():
        classifier_metrics = module.direct_training_metrics(wrapped, tok, all_by_fact, target,
                                                            llama_like=llama_like)

    artifact = dict(source)
    artifact["rows"] = rows
    artifact["training_route"] = args.training_route
    artifact["compressed_values"] = {
        "mode": args.value_mode, "compact_state": best_state, "answer_token_ids": answer_token_ids,
        "untrainable_fact_ids": untrainable, "storage": storage,
        "rows_are_exact_reconstruction": True,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    coverage = {"token_contexts": len(token_cases), "used_for_training": len(training_cases),
                "facts_trained": len(fact_ids), "facts_total": len(facts)}
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": f"sure_linear_router_compressed_bank_{args.dataset}",
        "value_mode": args.value_mode, "training_route": args.training_route,
        "router_v2_used": False, "training_coverage": coverage,
        "untrainable_fact_ids": untrainable, "value_storage": storage,
        "router_storage": router_storage(source),
    })
    (output / "association_manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")
    (output / "training_token_cases.json").write_text(
        json.dumps([asdict(c) for c in token_cases], indent=2) + "\n")
    for name in ("association_examples.json", "linear_router_report.json"):
        if (router_dir / name).is_file():
            shutil.copy2(router_dir / name, output / name)
    report = {
        "dataset": args.dataset, "value_mode": args.value_mode,
        "training_route": args.training_route, "stop_reason": stop_reason,
        "best_epoch": best_epoch, "best_checkpoint_key": list(best_key), "gates": gates,
        "training_coverage": coverage, "views_excluded_unrouted": excluded,
        "untrainable_fact_ids": untrainable, "pre_training_routing": pre_routing,
        "final_metrics_classifier_routing_all_contexts": classifier_metrics,
        "value_storage": storage, "router_storage": router_storage(source),
        "answer_groups": int(values.answer_groups),
        "hyperparameters": {k: getattr(args, k) for k in (
            "lr", "scale_lr", "batch_facts", "epochs", "eval_every", "post_feasible_gates", "clip")},
        "max_training_seconds": max_seconds,
        "reconstruction_max_abs_diff_vs_trained": drift,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "compressed_bank_trained", "dataset": args.dataset,
                      "value_mode": args.value_mode, "stop_reason": stop_reason,
                      "best_epoch": best_epoch,
                      "globally_feasible_classifier_routing": classifier_metrics["globally_feasible"],
                      "value_ratio_to_full_rows": storage["ratio_to_full_rows"],
                      "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
