#!/usr/bin/env python3
"""Step 3 of the linear-classifier layer sweep: train the residual rows.

Loads a fitted linear-classifier router (output of `fit_linear_router.py`,
rows all zero) and trains its 50 residual rows with the shipped row-wise
optimizer. No Router V2 is built or used.

  --training-route router   regular SURE: every training prompt is routed by
                            the linear classifier itself. Views it does not
                            send to their own row cannot be edited, so they
                            are left out of the objective and of checkpoint
                            selection (counted in the report). The official
                            eval still scores them.
  --training-route oracle   genie: ground-truth routing on training-visible
                            prompts, isolating how well layer L can be
                            written from how well it can be read.

Either way the saved artifact routes by the linear classifier, so the official
MCF evaluator and the decomposition script load it unchanged.

    python -u scripts/train_mcf_linear_router_rows.py \
        --router-dir outputs/<sweep>/L07/router \
        --output-dir outputs/<sweep>/L07/linear_global \
        --training-route router --norm-scale 1 --local-files-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import torch

from layer_sweep_utils import (
    boundary_norms,
    oracle_route_map,
    resolve_norm_scale,
)
from linear_router import ARCHITECTURE, load_linear_classifier_artifact
from prepare_mcf_association_source import load_mcf_forget_data
from static_overlap_extended_tokens_v2 import routed_metrics, train_row_wise
from static_overlap_fact_association_embeddings import (
    PLAN,
    FactAssociationEditor,
    audit_runtime_routes,
    make_unknown_examples,
)

NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-route", choices=("router", "oracle"), required=True)
    parser.add_argument(
        "--norm-scale", default="1",
        help="Multiply LR and trust radii by this; 'auto' = boundary-norm ratio "
             "of this layer to --norm-reference-layer.",
    )
    parser.add_argument("--norm-reference-layer", type=int, default=PLAN["layer"])
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

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(manifest["model_path"])
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    _, facts, examples = load_mcf_forget_data(tokenizer, manifest["mcf_path"])
    if [f["id"] for f in facts] != [f["id"] for f in source["facts"]]:
        raise ValueError("Rebuilt MCF facts do not match the router artifact")

    norms = boundary_norms(
        model, tokenizer, [e.prompt for e in examples if e.split == "train"],
        sorted({layer, args.norm_reference_layer}),
    )
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
    route_audit = audit_runtime_routes(editor.model, bank, tokenizer, examples, fact_to_row)
    answer_map = {example.id: example for example in examples}
    unknown_map = make_unknown_examples(
        examples, tokenizer, PLAN["max_length"], PLAN["unknown_completion"]
    )

    excluded = []
    if args.training_route == "oracle":
        bank.set_oracle_routes(oracle_route_map(tokenizer, (answer_map, unknown_map), fact_to_row))
        training_examples = list(examples)
    else:
        # Rows only reach prompts the classifier routes to them. Measure that
        # on the exact teacher-forced inputs the trainer uses.
        routed = _routes_on_training_inputs(editor.model, bank, examples, fact_to_row)
        training_examples = [e for e in examples if routed[e.id]]
        excluded = [e.id for e in examples if not routed[e.id]]
        kept = Counter(e.fact_id for e in training_examples if e.split == "train")
        missing = [f["id"] for f in facts if kept[f["id"]] == 0]
        if missing:
            raise RuntimeError(
                f"The linear router routes no training view to rows {missing}; "
                "those rows cannot be trained at this layer"
            )
        if not any(e.split == "development" for e in training_examples):
            raise RuntimeError("The linear router routes no development view correctly")
    kept_ids = {e.id for e in training_examples}
    train_answer = {k: v for k, v in answer_map.items() if k in kept_ids}
    train_unknown = {k: v for k, v in unknown_map.items() if k in kept_ids}
    coverage = {
        split: {
            "views": sum(e.split == split for e in examples),
            "used_for_training": sum(e.split == split for e in training_examples),
        }
        for split in ("train", "development")
    }
    print(json.dumps({"phase": "rows_training_ready", "layer": layer,
                      "training_route": args.training_route, "norm_scale": norm_scale,
                      "coverage": coverage,
                      "route_audit_train": route_audit["train"]["correct_row_active_fraction"],
                      "route_audit_development":
                          route_audit["development"]["correct_row_active_fraction"]}),
          flush=True)

    plan = dict(PLAN)
    plan["layer"] = layer
    plan["radius_schedule"] = tuple(
        (float(upper), float(radius) * norm_scale) for upper, radius in PLAN["radius_schedule"]
    )
    plan["learning_rate"] = float(PLAN["learning_rate"]) * norm_scale
    report = train_row_wise(
        editor=editor,
        original_examples=training_examples,
        routed_answer=train_answer,
        routed_unknown=train_unknown,
        fact_to_row=fact_to_row,
        plan=plan,
        output=output,
    )
    training_metrics = routed_metrics(
        editor.model, train_answer, train_unknown, plan["target_probability"]
    )
    bank.set_oracle_routes(None)
    classifier_metrics = routed_metrics(
        editor.model, answer_map, unknown_map, plan["target_probability"]
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
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": "sure_linear_router_layer_sweep",
        "residual_rows_reused_from_source": False,
        "rows_trained_under": (
            "linear classifier routing" if args.training_route == "router"
            else "genie (ground-truth) routing on training-visible prompts"
        ),
        "training_route": args.training_route,
        "router_v2_used": False,
        "plan": {**plan, "radius_schedule": [list(x) for x in plan["radius_schedule"]]},
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "pre_training_route_audit": route_audit,
        "layer_representation": representation,
    })
    (output / "association_manifest.json").write_text(
        json.dumps(new_manifest, indent=2, allow_nan=False) + "\n"
    )
    shutil.copy2(router_dir / "association_examples.json", output / "association_examples.json")
    shutil.copy2(router_dir / "linear_router_report.json", output / "linear_router_report.json")
    for name in ("best_extended_input_rows.pt", "last_extended_input_rows.pt"):
        (output / name).unlink(missing_ok=True)  # trainer scratch, superseded above
    report.update({
        "training_route": args.training_route,
        "layer_representation": representation,
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "final_metrics_training_routing": training_metrics,
        "final_metrics_classifier_routing_all_views": classifier_metrics,
        "unmatched_neutral_logits_exact_base_after_training": True,
    })
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "linear_router_rows_trained",
        "layer": layer,
        "training_route": args.training_route,
        "stop_reason": report["stop_reason"],
        "best_step": report["best_step"],
        "train_target_met_all_views_classifier_routing":
            classifier_metrics["train"]["target_met"],
        "output_dir": str(output),
    }, indent=2), flush=True)
    return 0


@torch.no_grad()
def _routes_on_training_inputs(model, bank, examples, fact_to_row, batch_size=16):
    """Does the classifier send each example's own teacher-forced input to its row?"""
    from static_overlap_extended_tokens_v2 import batched_answer_nll

    routed = {}
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        batched_answer_nll(model, batch)  # binds the answer boundary exactly as training does
        for example, active in zip(batch, bank.last_active_fact_indices):
            routed[example.id] = active == [fact_to_row[example.fact_id]]
    return routed


if __name__ == "__main__":
    raise SystemExit(main())
