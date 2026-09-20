#!/usr/bin/env python3
"""Sweep the addressing (read) layer independently of the write layer.

Block 19 was fixed on seed 1 and is used for two unrelated jobs: reading the
request's relation and writing the residual. Those are separate design
choices, and there is no reason the most linearly readable layer is the most
effective injection site. This script answers the read half only, which is the
cheap half -- it needs no training, only forward passes, because separability
of positives from negatives at layer L is a property of the frozen backbone.

For each candidate layer the script rebuilds the prototypes and thresholds at
that layer using exactly the shipped rule (`build_direct_prompt_context_gate`,
so the 10% separable-gap operating point and the disabled absolute floor carry
over), and then reports, on held-out development prompts:

  dev_route_recall        gold association selected, full Router V2 rule
  dev_any_activation      any association selected
  roc_auc                 threshold-free separability of gold-positive from
                          negative margins. This is the number to rank layers
                          by, because it does not depend on the in-sample tau
                          fit and so cannot be inflated by it.
  mean_margin_gap         mean(d_positive) - max(d_negative), per association
  separable_fraction      associations whose training controls separate at all

A layer that scores a high AUC but a low recall is a *calibration* failure,
not a representation failure -- the information is linearly present and the
threshold rule is losing it. That distinction is the point of reporting both.

Early/middle/late defaults for Llama-3.2-3B (28 blocks): 4, 8, 12, 16, 19, 22,
25, 27. Pass --layers to override.

Usage
-----
python -u scripts/sweep_router_read_layer.py \
  --model-path <llama> --artifact outputs/<run>/fact_association_embeddings.pt \
  --examples outputs/<run>/association_examples.json \
  --output-dir outputs/<run>/layer_sweep --device cuda
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from fact_association_router_v2 import build_direct_prompt_context_gate
from static_overlap_fact_association_embeddings import (
    _contains_subsequence,
    extract_prompt_queries,
    subject_token_patterns,
)


def roc_auc(positive, negative):
    """Mann-Whitney U form; ties count a half. No sklearn dependency."""
    if not len(positive) or not len(negative):
        return None
    pos = torch.as_tensor(positive, dtype=torch.float64)
    neg = torch.as_tensor(negative, dtype=torch.float64)
    comparison = pos[:, None] - neg[None, :]
    wins = (comparison > 0).sum() + 0.5 * (comparison == 0).sum()
    return float(wins / (pos.numel() * neg.numel()))


@torch.no_grad()
def score_matrix(model, tokenizer, prompts, layer, positives, negatives,
                 batch_size=16):
    queries = extract_prompt_queries(
        model, tokenizer, prompts, int(layer), batch_size=int(batch_size)
    )
    queries = F.normalize(queries.float(), dim=-1)
    columns = []
    for positive, negative in zip(positives, negatives):
        p = F.normalize(positive.float(), dim=-1)
        n = F.normalize(negative.float(), dim=-1)
        u = (queries @ p.T).max(dim=-1).values
        v = (queries @ n.T).max(dim=-1).values
        columns.append(u - v)
    return torch.stack(columns, dim=-1)


def eligibility_matrix(tokenizer, prompts, subject_patterns):
    mask = torch.zeros((len(prompts), len(subject_patterns)), dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        for column, patterns in enumerate(subject_patterns):
            if any(_contains_subsequence(tokens, pattern) for pattern in patterns):
                mask[row, column] = True
    return mask


def route(d, mask, tau, ambiguity_margin):
    qualifies = mask & (d >= tau[None, :])
    ranked = d.masked_fill(~qualifies, float("-inf"))
    best_d, best = ranked.max(dim=-1)
    active = torch.isfinite(best_d)
    if d.shape[1] > 1:
        top2 = ranked.topk(k=2, dim=-1).values
        separation = top2[:, 0] - top2[:, 1]
        ambiguous = (
            (qualifies.sum(dim=-1) > 1)
            & torch.isfinite(top2[:, 1])
            & (separation < float(ambiguity_margin))
        )
        active = active & ~ambiguous
    return best, active


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--negative-count", type=int, default=12)
    parser.add_argument("--margin-slack", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--held-out-probes",
        default="",
        help="optional gemini probe-set JSON; adds a different-generator arm",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    facts = list(artifact["facts"])
    ambiguity_margin = float(artifact.get("ambiguity_margin", 0.02))
    fact_index = {str(fact["id"]): index for index, fact in enumerate(facts)}

    examples = json.loads(Path(args.examples).read_text())
    train_by_fact = defaultdict(list)
    dev_prompts, dev_owner = [], []
    for example in examples:
        fact_id = str(example.get("fact_id", ""))
        if fact_id not in fact_index:
            continue
        if str(example.get("split")) == "train":
            train_by_fact[fact_id].append(str(example["prompt"]))
        elif str(example.get("split")) == "development":
            dev_prompts.append(str(example["prompt"]))
            dev_owner.append(fact_index[fact_id])
    if not dev_prompts:
        raise SystemExit("No development prompts found in the examples file")

    held_out = []
    if args.held_out_probes:
        payload = json.loads(Path(args.held_out_probes).read_text())
        for probe in payload["probes"]:
            if str(probe.get("fact_id")) in fact_index:
                held_out.append({
                    "prompt": str(probe["prompt"]),
                    "owner": fact_index[str(probe["fact_id"])],
                    "polarity": probe.get("polarity"),
                    "probe_class": probe.get("probe_class"),
                })

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=args.local_files_only,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()
    model.requires_grad_(False)

    block_count = int(model.config.num_hidden_layers)
    if args.layers.strip():
        layers = [int(v) for v in args.layers.split(",") if v.strip()]
    else:
        fractions = (0.15, 0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 1.0)
        layers = sorted({
            min(block_count - 1, max(0, int(round(f * (block_count - 1)))))
            for f in fractions
        } | {int(artifact["layer"])})
    if any(layer < 0 or layer >= block_count for layer in layers):
        raise SystemExit(f"Layers must lie in [0, {block_count - 1}]")

    subject_patterns = [
        subject_token_patterns(tokenizer, fact["subject"]) for fact in facts
    ]
    dev_mask = eligibility_matrix(tokenizer, dev_prompts, subject_patterns)
    dev_owner_tensor = torch.tensor(dev_owner, dtype=torch.long)
    held_mask = (
        eligibility_matrix(
            tokenizer, [row["prompt"] for row in held_out], subject_patterns
        )
        if held_out else None
    )

    results = []
    for layer in layers:
        positives, negatives, _alpha, tau, diagnostics = (
            build_direct_prompt_context_gate(
                model,
                tokenizer,
                facts,
                dict(train_by_fact),
                int(layer),
                negative_count=int(args.negative_count),
                margin_slack=float(args.margin_slack),
            )
        )
        d = score_matrix(
            model,
            tokenizer,
            dev_prompts,
            layer,
            positives,
            negatives,
            batch_size=args.batch_size,
        )
        selected, active = route(d, dev_mask, tau, ambiguity_margin)
        gold_scores = d[torch.arange(len(dev_prompts)), dev_owner_tensor]
        off_gold = d.clone()
        off_gold[torch.arange(len(dev_prompts)), dev_owner_tensor] = float("nan")
        off_gold_values = off_gold[~torch.isnan(off_gold)]

        record = {
            "layer": int(layer),
            "relative_depth": round(layer / max(block_count - 1, 1), 3),
            "is_shipped_layer": int(layer) == int(artifact["layer"]),
            "dev_prompt_count": len(dev_prompts),
            "dev_route_recall": float(
                ((selected == dev_owner_tensor) & active).float().mean()
            ),
            "dev_any_activation": float(active.float().mean()),
            "dev_wrong_route_rate": float(
                ((selected != dev_owner_tensor) & active).float().mean()
            ),
            "roc_auc_gold_vs_offgold": roc_auc(gold_scores, off_gold_values),
            "mean_gold_margin": float(gold_scores.mean()),
            "mean_gold_margin_over_tau": float(
                (gold_scores - tau[dev_owner_tensor]).mean()
            ),
            "separable_fraction_on_training_controls": float(
                sum(
                    row["separable_on_training_controls"]
                    for row in diagnostics["per_fact"]
                ) / len(facts)
            ),
            "mean_training_negative_fire_fraction": diagnostics[
                "mean_training_negative_fire_fraction"
            ],
            "mean_tau": float(tau.mean()),
        }

        if held_out:
            held_d = score_matrix(
                model,
                tokenizer,
                [row["prompt"] for row in held_out],
                layer,
                positives,
                negatives,
                batch_size=args.batch_size,
            )
            held_owner = torch.tensor(
                [row["owner"] for row in held_out], dtype=torch.long
            )
            held_selected, held_active = route(
                held_d, held_mask, tau, ambiguity_margin
            )
            positive_rows = [
                i for i, row in enumerate(held_out) if row["polarity"] == "positive"
            ]
            negative_rows = [
                i for i, row in enumerate(held_out) if row["polarity"] == "negative"
            ]
            gold = held_d[torch.arange(len(held_out)), held_owner]
            record["held_out"] = {
                "probe_count": len(held_out),
                "positive_route_recall": (
                    float(
                        (
                            (held_selected[positive_rows] == held_owner[positive_rows])
                            & held_active[positive_rows]
                        ).float().mean()
                    ) if positive_rows else None
                ),
                # The number the paper does not currently have: how often the
                # router fires on a permitted request built by a generator it
                # was never calibrated against.
                "negative_false_activation_rate": (
                    float(held_active[negative_rows].float().mean())
                    if negative_rows else None
                ),
                "roc_auc_positive_vs_negative": roc_auc(
                    gold[positive_rows] if positive_rows else [],
                    gold[negative_rows] if negative_rows else [],
                ),
                "by_class": _by_class(held_out, held_selected, held_active, held_owner),
            }

        results.append(record)
        print(json.dumps(
            {k: v for k, v in record.items() if k != "held_out"}, indent=2
        ), flush=True)

    ranked = sorted(
        results,
        key=lambda row: (row["roc_auc_gold_vs_offgold"] or 0.0),
        reverse=True,
    )
    report = {
        "schema_version": "router_read_layer_sweep_v1",
        "artifact": str(Path(args.artifact).resolve()),
        "shipped_layer": int(artifact["layer"]),
        "block_count": block_count,
        "layers": results,
        "best_layer_by_auc": ranked[0]["layer"] if ranked else None,
        "best_layer_by_dev_recall": max(
            results, key=lambda row: row["dev_route_recall"]
        )["layer"],
        "note": (
            "Read-layer only. The write layer is unchanged; decoupling the two "
            "requires a separate training run per write layer."
        ),
    }
    (output / "router_read_layer_sweep.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "layer_sweep_complete",
        "best_layer_by_auc": report["best_layer_by_auc"],
        "best_layer_by_dev_recall": report["best_layer_by_dev_recall"],
        "shipped_layer": report["shipped_layer"],
        "output": str(output / "router_read_layer_sweep.json"),
    }, indent=2))
    return 0


def _by_class(held_out, selected, active, owner):
    groups = defaultdict(list)
    for index, row in enumerate(held_out):
        groups[row["probe_class"]].append(index)
    summary = {}
    for name, indices in sorted(groups.items()):
        index = torch.tensor(indices, dtype=torch.long)
        summary[name] = {
            "count": len(indices),
            "activation_rate": float(active[index].float().mean()),
            "gold_route_rate": float(
                ((selected[index] == owner[index]) & active[index]).float().mean()
            ),
        }
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
