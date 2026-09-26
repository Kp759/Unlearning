#!/usr/bin/env python3
"""MQuAKE layer sweep, step 3: train the residual rows under the linear router.

Loads a fitted linear-classifier router (rows all zero) and trains one row per
unique association with the shipped MQuAKE optimizer (`train_direct_only`):
worst sensitive-token probability on the exact official direct-rewrite token
contexts, driven below 1e-6. No Router V2.

  --training-route router   regular: the linear classifier routes each token
                            context; contexts it does not send to their own
                            row are left out of the objective (reported).
  --training-route oracle   genie: ground-truth routing on the direct rewrites.

The saved artifact routes by the linear classifier for the official evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import torch

import mquake_zero_unlearn_official_eval as mquake
from layer_sweep_utils import boundary_norms, resolve_norm_scale
from linear_router import ARCHITECTURE, load_linear_classifier_artifact
from mquake_fact_association_embeddings import (
    BASE_PLAN,
    build_exact_direct_token_cases,
    direct_training_metrics,
    strict_prefix_lengths,
    train_direct_only,
)
from prepare_mquake_association_source import load_mquake_forget
from static_overlap_fact_association_embeddings import FactAssociationEditor

NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."


def genie_route_map(tokenizer, cases, fact_to_row):
    mapping = {}
    for case in cases:
        key = tuple(mquake._flat_ids(tokenizer, case.boundary_prompt))
        row = fact_to_row[case.fact_id]
        if mapping.setdefault(key, row) != row:
            raise ValueError(f"Direct request shared by two associations: {case.boundary_prompt!r}")
    return mapping


@torch.no_grad()
def routes_for_cases(model, bank, tokenizer, cases, fact_to_row, batch_size=16):
    device = next(model.parameters()).device
    routed = {}
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        encoded = tokenizer([c.prompt for c in batch], padding=True, return_tensors="pt",
                            return_token_type_ids=False).to(device)
        model.set_association_prefix_lengths(strict_prefix_lengths(tokenizer, batch))
        model(**encoded, use_cache=False)
        for case, active in zip(batch, bank.last_active_fact_indices):
            routed[case.id] = active == [fact_to_row[case.fact_id]]
    return routed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-route", choices=("router", "oracle"), required=True)
    parser.add_argument("--norm-scale", default="1")
    parser.add_argument("--norm-reference-layer", type=int, default=BASE_PLAN["layer"])
    parser.add_argument("--row-updates-per-fact", type=int, default=30)
    parser.add_argument("--max-training-seconds", type=float, default=7200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    router_dir = Path(args.router_dir).resolve()
    output = Path(args.output_dir).resolve()
    source = torch.load(router_dir / "fact_association_embeddings.pt",
                        map_location="cpu", weights_only=False)
    if str(source.get("architecture")) != ARCHITECTURE:
        raise ValueError(f"{router_dir} is not a linear-classifier router artifact")
    if float(source["rows"].abs().max()) != 0.0:
        raise ValueError("Router artifact already has trained rows; expected a fresh fit")
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    layer = int(source["layer"])
    output.mkdir(parents=True, exist_ok=False)

    _, _, records, facts, _, _ = load_mquake_forget(
        manifest["training_visible_path"], manifest["split_manifest_path"]
    )
    if [f["association_key"] for f in facts] != [f["association_key"] for f in source["facts"]]:
        raise ValueError("Rebuilt MQuAKE associations do not match the router artifact")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(1)
    model_path = Path(manifest["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)
    model.config.use_cache = False

    prompts = [p for f in facts for p in (f.get("canonical_prompts") or [f["canonical_prompt"]])]
    norms = boundary_norms(model, tokenizer, prompts, sorted({layer, args.norm_reference_layer}))
    norm_scale = resolve_norm_scale(args.norm_scale, norms, layer, args.norm_reference_layer)
    representation = {
        **manifest.get("layer_representation", {}),
        "norm_scale_argument": str(args.norm_scale),
        "norm_scale": norm_scale,
        "training_route": args.training_route,
    }

    neutral = tokenizer(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()
    _, bank = load_linear_classifier_artifact(model, source)
    for row in bank.rows:
        row.requires_grad_(True)
    editor = FactAssociationEditor(model, bank)
    with torch.no_grad():
        if not torch.equal(base_logits, editor.model(**neutral, use_cache=False).logits):
            raise ValueError("Zero rows changed an unmatched base prompt")

    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    token_cases, llama_like = build_exact_direct_token_cases(records, facts, tokenizer, editor.model)
    routed = routes_for_cases(editor.model, bank, tokenizer, token_cases, fact_to_row)
    pre_training_routing = {
        "token_contexts": len(token_cases),
        "routed_to_own_row": sum(routed.values()),
        "fraction": sum(routed.values()) / len(token_cases),
    }

    excluded = []
    if args.training_route == "oracle":
        bank.set_oracle_routes(genie_route_map(tokenizer, token_cases, fact_to_row))
        training_cases = list(token_cases)
    else:
        training_cases = [c for c in token_cases if routed[c.id]]
        excluded = [c.id for c in token_cases if not routed[c.id]]
        kept = Counter(c.fact_id for c in training_cases)
        missing = [f["id"] for f in facts if kept[f["id"]] == 0]
        if missing:
            raise RuntimeError(
                f"The linear router routes no direct rewrite to rows {missing}; "
                "those rows cannot be trained at this layer"
            )
    print(json.dumps({"phase": "rows_training_ready", "layer": layer,
                      "training_route": args.training_route, "norm_scale": norm_scale,
                      "pre_training_routing": pre_training_routing,
                      "training_token_contexts": len(training_cases)}), flush=True)

    steps = len(facts) * int(args.row_updates_per_fact)
    plan = dict(BASE_PLAN)
    plan.update({
        "layer": layer,
        "seed": 1,
        "steps": steps,
        "check_every": len(facts),
        "row_updates_per_fact": int(args.row_updates_per_fact),
        "max_training_seconds": float(args.max_training_seconds),
        "learning_rate": float(BASE_PLAN["learning_rate"]) * norm_scale,
        "radius_schedule": tuple(
            (float(upper), float(radius) * norm_scale)
            for upper, radius in BASE_PLAN["radius_schedule"]
        ),
    })
    report = train_direct_only(
        editor, tokenizer, training_cases, fact_to_row, plan, output, llama_like=llama_like,
    )
    bank.set_oracle_routes(None)
    by_fact = defaultdict(list)
    for case in token_cases:
        by_fact[case.fact_id].append(case)
    classifier_metrics = direct_training_metrics(
        editor.model, tokenizer, by_fact, plan["target_token_probability"], llama_like=llama_like,
    )
    with torch.no_grad():
        if not torch.equal(base_logits, editor.model(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched natural prompt left the exact base path after training")

    row_norms = bank.extra.detach().float().norm(dim=-1).cpu()
    representation.update({
        "row_norm_median": float(row_norms.median()),
        "row_norm_max": float(row_norms.max()),
        "row_to_boundary_norm_ratio_median": float(row_norms.median() / norms[layer].median()),
    })

    artifact = dict(source)
    artifact["rows"] = bank.extra.detach().cpu()
    artifact["training_route"] = args.training_route
    torch.save(artifact, output / "fact_association_embeddings.pt")
    coverage = {"token_contexts": len(token_cases), "used_for_training": len(training_cases)}
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": "sure_linear_router_layer_sweep_mquake",
        "residual_rows_reused_from_source": False,
        "rows_trained_under": (
            "linear classifier routing" if args.training_route == "router"
            else "genie (ground-truth) routing on direct rewrites"
        ),
        "training_route": args.training_route,
        "router_v2_used": False,
        "plan": {**plan, "radius_schedule": [list(x) for x in plan["radius_schedule"]]},
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "pre_training_routing": pre_training_routing,
        "layer_representation": representation,
    })
    (output / "association_manifest.json").write_text(
        json.dumps(new_manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "training_token_cases.json").write_text(
        json.dumps([asdict(c) for c in token_cases], indent=2) + "\n"
    )
    for name in ("linear_router_report.json",):
        if (router_dir / name).is_file():
            (output / name).write_text((router_dir / name).read_text())
    for name in ("best_fact_association_rows.pt", "last_fact_association_rows.pt"):
        (output / name).unlink(missing_ok=True)
    report.update({
        "training_route": args.training_route,
        "layer_representation": representation,
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "pre_training_routing": pre_training_routing,
        "final_metrics_classifier_routing_all_contexts": classifier_metrics,
        "unmatched_neutral_logits_exact_base_after_training": True,
    })
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "mquake_linear_router_rows_trained",
        "layer": layer,
        "training_route": args.training_route,
        "stop_reason": report["stop_reason"],
        "best_step": report["best_step"],
        "facts_passing_classifier_routing": classifier_metrics["facts_passing_probability_constraint"],
        "facts_total": classifier_metrics["facts_total"],
        "output_dir": str(output),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
