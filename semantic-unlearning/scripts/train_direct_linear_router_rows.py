#!/usr/bin/env python3
"""Layer sweep, step 3 for direct-rewrite benchmarks (MQuAKE, ZsRE, multi-fact person).

Loads a fitted linear-classifier router (rows all zero) and trains one row per
fact with the benchmark's own shipped optimizer (`train_direct_only`): worst
sensitive-token probability on the exact official direct-rewrite token
contexts, driven below 1e-6. No Router V2.

  --training-route router   regular: the linear classifier routes each token
                            context. Contexts it does not send to their own
                            row are left out of the objective. A fact with no
                            routed context keeps a zero row (listed as
                            `untrainable_fact_ids`) instead of aborting the layer.
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

from layer_sweep_utils import boundary_norms, resolve_norm_scale
from linear_router import ARCHITECTURE, load_linear_classifier_artifact
from static_overlap_fact_association_embeddings import FactAssociationEditor

NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."


def dataset_adapter(name):
    """Benchmark-specific loaders; the training procedure itself is shared."""
    if name == "mquake":
        import mquake_fact_association_embeddings as module
        import mquake_zero_unlearn_official_eval as official
        from prepare_mquake_association_source import load_mquake_forget

        def load(manifest):
            _, _, records, facts, _, _ = load_mquake_forget(
                manifest["training_visible_path"], manifest["split_manifest_path"]
            )
            return records, facts

        return {
            "module": module, "official": official, "load": load,
            "plan": module.BASE_PLAN, "prefix_lengths": module.strict_prefix_lengths,
            "fact_key": "association_key", "updates_per_fact": 30,
            "max_seconds": 7200.0, "method": "sure_linear_router_layer_sweep_mquake",
        }
    if name == "zsre":
        import zsre_fact_association_embeddings as module
        import zsre_zero_unlearn_official_eval as official
        from prepare_zsre_association_source import load_zsre_forget

        def load(manifest):
            _, _, records, facts = load_zsre_forget(
                manifest["training_visible_path"], manifest["split_manifest_path"]
            )
            return records, facts

        plan = module.PLAN
        return {
            "module": module, "official": official, "load": load,
            "plan": plan, "prefix_lengths": module._strict_prefix_lengths,
            "fact_key": "id",
            "updates_per_fact": int(plan["steps"]) // int(plan["check_every"]),
            "max_seconds": float(plan["max_training_seconds"]),
            "method": "sure_linear_router_layer_sweep_zsre",
        }
    if name == "multifact":
        # Multi-fact person benchmark: MQuAKE direct-record format and machinery,
        # its own locked split and fact identity (multifact_person_data.py).
        import mquake_fact_association_embeddings as module
        import mquake_zero_unlearn_official_eval as official
        from multifact_person_data import load_multifact_forget

        def load(manifest):
            _, _, records, facts, _, _ = load_multifact_forget(
                manifest["training_visible_path"], manifest["split_manifest_path"]
            )
            return records, facts

        return {
            "module": module, "official": official, "load": load,
            "plan": module.BASE_PLAN, "prefix_lengths": module.strict_prefix_lengths,
            "fact_key": "association_key", "updates_per_fact": 30,
            "max_seconds": 7200.0, "method": "sure_linear_router_multifact_person",
        }
    raise ValueError(f"Unknown dataset {name!r}")


def genie_route_map(official, tokenizer, cases, fact_to_row):
    mapping = {}
    for case in cases:
        key = tuple(official._flat_ids(tokenizer, case.boundary_prompt))
        row = fact_to_row[case.fact_id]
        if mapping.setdefault(key, row) != row:
            raise ValueError(f"Direct request shared by two facts: {case.boundary_prompt!r}")
    return mapping


@torch.no_grad()
def routes_for_cases(model, bank, tokenizer, cases, fact_to_row, prefix_lengths_fn,
                     batch_size=16):
    device = next(model.parameters()).device
    routed = {}
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        encoded = tokenizer([c.prompt for c in batch], padding=True, return_tensors="pt",
                            return_token_type_ids=False).to(device)
        model.set_association_prefix_lengths(prefix_lengths_fn(tokenizer, batch))
        model(**encoded, use_cache=False)
        for case, active in zip(batch, bank.last_active_fact_indices):
            routed[case.id] = active == [fact_to_row[case.fact_id]]
    return routed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mquake", "zsre", "multifact"), required=True)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-route", choices=("router", "oracle"), required=True)
    parser.add_argument("--norm-scale", default="1")
    parser.add_argument("--norm-reference-layer", type=int, default=None,
                        help="default: the prep stage's reference layer (19)")
    parser.add_argument("--row-updates-per-fact", type=int, default=None,
                        help="default: the benchmark's shipped budget (30)")
    parser.add_argument("--max-training-seconds", type=float, default=None,
                        help="default: the benchmark's shipped cap (MQuAKE 7200, ZsRE 3600)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    adapter = dataset_adapter(args.dataset)
    module, official = adapter["module"], adapter["official"]
    updates = int(args.row_updates_per_fact or adapter["updates_per_fact"])
    max_seconds = float(args.max_training_seconds or adapter["max_seconds"])

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
    if args.norm_reference_layer is None:
        args.norm_reference_layer = int(
            manifest.get("layer_representation", {}).get("reference_layer", 19)
        )
    output.mkdir(parents=True, exist_ok=False)

    records, facts = adapter["load"](manifest)
    key = adapter["fact_key"]
    if [f[key] for f in facts] != [f[key] for f in source["facts"]]:
        raise ValueError("Rebuilt facts do not match the router artifact")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    seed = int(manifest.get("seed", 1))
    torch.manual_seed(seed)
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
    token_cases, llama_like = module.build_exact_direct_token_cases(
        records, facts, tokenizer, editor.model
    )
    routed = routes_for_cases(editor.model, bank, tokenizer, token_cases, fact_to_row,
                              adapter["prefix_lengths"])
    pre_training_routing = {
        "token_contexts": len(token_cases),
        "routed_to_own_row": sum(routed.values()),
        "fraction": sum(routed.values()) / len(token_cases),
    }

    excluded, untrainable = [], []
    if args.training_route == "oracle":
        bank.set_oracle_routes(genie_route_map(official, tokenizer, token_cases, fact_to_row))
        training_cases = list(token_cases)
    else:
        training_cases = [c for c in token_cases if routed[c.id]]
        excluded = [c.id for c in token_cases if not routed[c.id]]
        kept = Counter(c.fact_id for c in training_cases)
        untrainable = [f["id"] for f in facts if kept[f["id"]] == 0]
        if len(untrainable) == len(facts):
            raise RuntimeError("The linear router routes no direct rewrite to its own row")
    trainable_rows = {fid: row for fid, row in fact_to_row.items() if fid not in untrainable}
    print(json.dumps({"phase": "rows_training_ready", "dataset": args.dataset, "layer": layer,
                      "training_route": args.training_route, "norm_scale": norm_scale,
                      "pre_training_routing": pre_training_routing,
                      "training_token_contexts": len(training_cases),
                      "untrainable_facts": len(untrainable)}), flush=True)

    n = len(trainable_rows)
    plan = dict(adapter["plan"])
    plan.update({
        "layer": layer,
        "seed": seed,
        "steps": n * updates,
        "check_every": n,
        "row_updates_per_fact": updates,
        "max_training_seconds": max_seconds,
        "learning_rate": float(adapter["plan"]["learning_rate"]) * norm_scale,
        "radius_schedule": tuple(
            (float(upper), float(radius) * norm_scale)
            for upper, radius in adapter["plan"]["radius_schedule"]
        ),
    })
    report = module.train_direct_only(
        editor, tokenizer, training_cases, trainable_rows, plan, output, llama_like=llama_like,
    )
    bank.set_oracle_routes(None)
    by_fact = defaultdict(list)
    for case in token_cases:
        by_fact[case.fact_id].append(case)
    classifier_metrics = module.direct_training_metrics(
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
    coverage = {"token_contexts": len(token_cases), "used_for_training": len(training_cases),
                "facts_trained": n, "facts_total": len(facts)}
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": adapter["method"],
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
        "untrainable_fact_ids": untrainable,
        "pre_training_routing": pre_training_routing,
        "layer_representation": representation,
    })
    (output / "association_manifest.json").write_text(
        json.dumps(new_manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "training_token_cases.json").write_text(
        json.dumps([asdict(c) for c in token_cases], indent=2) + "\n"
    )
    if (router_dir / "linear_router_report.json").is_file():
        (output / "linear_router_report.json").write_text(
            (router_dir / "linear_router_report.json").read_text()
        )
    for name in ("best_fact_association_rows.pt", "last_fact_association_rows.pt"):
        (output / name).unlink(missing_ok=True)
    report.update({
        "dataset": args.dataset,
        "training_route": args.training_route,
        "layer_representation": representation,
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "untrainable_fact_ids": untrainable,
        "pre_training_routing": pre_training_routing,
        "final_metrics_classifier_routing_all_contexts": classifier_metrics,
        "unmatched_neutral_logits_exact_base_after_training": True,
    })
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "direct_linear_router_rows_trained",
        "dataset": args.dataset,
        "layer": layer,
        "training_route": args.training_route,
        "stop_reason": report["stop_reason"],
        "best_step": report["best_step"],
        "facts_trained": n,
        "facts_passing_classifier_routing": classifier_metrics["facts_passing_probability_constraint"],
        "facts_total": classifier_metrics["facts_total"],
        "output_dir": str(output),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
