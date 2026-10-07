#!/usr/bin/env python3
"""Train a compressed residual bank in the loop on MCF (linear router, no V2).

The rows are rows = CompressedValues(mode)(params); the unlearning objective
optimizes those compact parameters jointly across facts (shared pieces get
gradient from every fact that uses them). Per-fact objective is the shipped
one: hinge on the hardest training view's answer NLL (target probability
1e-6) + unknown-answer NLL. Checkpoints are chosen with the shipped
`checkpoint_key` on training-visible train + development views.

    python -u scripts/train_mcf_compressed_bank.py \
        --router-dir outputs/mcf_compressed_v1/router_relation \
        --output-dir outputs/mcf_compressed_v1/router_relation/answer_fixed \
        --value-mode answer_fixed --training-route router --local-files-only

The saved artifact is an ordinary linear-router artifact whose rows are the
float32 rows the compact parameters produce (checked by rebuilding them from
the saved compact state), so the official MCF evaluator runs unchanged. The
compact state and a storage report are saved alongside.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import time

import torch

from compressed_value_bank import (
    CompressedValueBank,
    CompressedValues,
    answer_directions,
    parse_value_mode,
)
from layer_sweep_utils import first_label_position, oracle_route_map
from linear_router import ARCHITECTURE
from prepare_mcf_association_source import load_mcf_forget_data
from static_overlap_extended_tokens_v2 import checkpoint_key, fact_objective, routed_metrics
from static_overlap_fact_association_embeddings import (
    PLAN,
    AssociationCausalLM,
    audit_runtime_routes,
    make_unknown_examples,
)
from train_mcf_linear_router_rows import _routes_on_training_inputs

NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."
SCALAR_PARAMS = ("scale", "relation_scale", "answer_scale", "codes")


def bank_from_artifact(model, artifact, values):
    return CompressedValueBank(
        base_model=model,
        layer=int(artifact["layer"]),
        weight=artifact["router_weight"],
        bias=artifact["router_bias"],
        feature_mean=artifact["feature_mean"],
        feature_components=artifact.get("feature_components"),
        threshold=float(artifact["threshold"]),
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        ambiguity_margin=float(artifact.get("ambiguity_margin", 0.5)),
        gate_mode=str(artifact.get("gate_mode", "threshold")),
        router_fit=artifact.get("router_fit"),
        per_head_thresholds=artifact.get("per_head_thresholds"),
        bias_calibration=artifact.get("bias_calibration"),
        head_index=artifact.get("head_index"),
        values=values,
    )


def router_storage(artifact):
    weight = artifact["router_weight"]
    heads, dim = int(weight.shape[0]), int(weight.shape[1])
    facts = len(artifact["facts"])
    components = artifact.get("feature_components")
    return {
        "heads": heads,
        "facts": facts,
        "head_sharing": "relation" if artifact.get("head_index") is not None else "fact",
        "feature_dim": dim,
        "head_floats": heads * (dim + 1),
        "shared_feature_map_floats": int(artifact["feature_mean"].numel())
        + (0 if components is None else int(components.numel())),
        "per_fact_small_ints": 1 if artifact.get("head_index") is not None else 0,
        "per_fact_head_floats": 0 if artifact.get("head_index") is not None else dim + 1,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-mode", required=True,
                        help="full | shared | lowrank:K | tied_answer | tied_relation | answer_fixed | "
                             "answer_map:r | relation_plus_answer")
    parser.add_argument("--training-route", choices=("router", "oracle"), default="router")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam lr for vector parameters")
    parser.add_argument("--scale-lr", type=float, default=0.5,
                        help="Adam lr for per-fact scalars and low-rank codes")
    parser.add_argument("--batch-facts", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5, help="epochs between checkpoints")
    parser.add_argument("--max-training-seconds", type=float, default=float(PLAN["max_training_seconds"]))
    parser.add_argument("--unknown-weight", type=float, default=float(PLAN["unknown_weight"]))
    parser.add_argument("--unknown-completion", default=str(PLAN["unknown_completion"]),
                        help='abstention text the unknown term trains (default " I don\'t know.")')
    parser.add_argument("--unknown-eos", action="store_true",
                        help="end the abstention with the tokenizer's end token, so generation stops "
                             "right after it instead of continuing (and possibly naming the answer)")
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    mode, rank = parse_value_mode(args.value_mode)
    if min(args.batch_facts, args.epochs, args.eval_every) < 1:
        parser.error("batch-facts, epochs and eval-every must be positive")
    if args.max_training_seconds <= 0:
        parser.error("max-training-seconds must be positive")

    router_dir = Path(args.router_dir).resolve()
    output = Path(args.output_dir).resolve()
    source = torch.load(router_dir / "fact_association_embeddings.pt",
                        map_location="cpu", weights_only=False)
    if str(source.get("architecture")) != ARCHITECTURE:
        raise ValueError(f"{router_dir} is not a linear-classifier router artifact")
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model_path = Path(manifest["model_path"])
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

    _, facts, examples = load_mcf_forget_data(
        tokenizer, manifest["mcf_path"], seed=int((manifest.get("sampling") or {}).get("seed", 1))
    )
    if [f["id"] for f in facts] != [f["id"] for f in source["facts"]]:
        raise ValueError("Rebuilt MCF facts do not match the router artifact")
    fact_to_row = {fact["id"]: index for index, fact in enumerate(facts)}
    answer_map = {example.id: example for example in examples}
    unknown_map = make_unknown_examples(
        examples, tokenizer, PLAN["max_length"], args.unknown_completion
    )
    if args.unknown_eos:
        eos = tokenizer.eos_token_id
        if eos is None:
            raise ValueError("--unknown-eos needs a tokenizer eos_token_id")
        unknown_map = {k: replace(v, input_ids=list(v.input_ids) + [int(eos)],
                                  labels=list(v.labels) + [int(eos)])
                       for k, v in unknown_map.items()}

    # First answer token of each fact, from its canonical training example.
    answer_token = {}
    for example in examples:
        if example.fact_id not in answer_token:
            answer_token[example.fact_id] = int(example.input_ids[first_label_position(example)])
    answer_token_ids = [answer_token[f["id"]] for f in facts]
    hidden = int(model.config.hidden_size)
    values = CompressedValues(
        mode, rank, facts, hidden, answer_token_ids=answer_token_ids,
        answer_dirs=answer_directions(model, answer_token_ids), seed=args.seed,
    ).to(args.device)

    neutral = tokenizer(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()
    bank = bank_from_artifact(model, source, values)
    wrapped = AssociationCausalLM(model, bank)
    with torch.no_grad():
        if not torch.equal(base_logits, wrapped(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched neutral prompt left the exact base path")

    route_audit = audit_runtime_routes(wrapped, bank, tokenizer, examples, fact_to_row)
    excluded, untrainable = [], []
    if args.training_route == "oracle":
        bank.set_oracle_routes(oracle_route_map(tokenizer, (answer_map, unknown_map), fact_to_row))
        training_examples = list(examples)
    else:
        routed = _routes_on_training_inputs(wrapped, bank, examples, fact_to_row)
        training_examples = [e for e in examples if routed[e.id]]
        kept = Counter(e.fact_id for e in training_examples if e.split == "train")
        untrainable = [f["id"] for f in facts if kept[f["id"]] == 0]
        training_examples = [e for e in training_examples if e.fact_id not in untrainable]
        excluded = [e.id for e in examples if e.id not in {x.id for x in training_examples}]
        if not training_examples:
            raise RuntimeError("The linear router routes no training view to its own row")
    # Keep the baseline's own-row training coverage for a controlled comparison.
    # A single shared vector must nevertheless fire for EVERY active route at
    # inference, including a fact with no correctly routed training views.
    mask = torch.ones(len(facts), device=args.device)
    if mode != "shared":
        for fid in untrainable:
            mask[fact_to_row[fid]] = 0.0
    base_rows = values.rows

    def masked_rows():
        return base_rows() * mask[:, None]

    values.rows = masked_rows

    kept_ids = {e.id for e in training_examples}
    train_answer = {k: v for k, v in answer_map.items() if k in kept_ids}
    train_unknown = {k: v for k, v in unknown_map.items() if k in kept_ids}
    by_fact = defaultdict(list)
    for e in training_examples:
        if e.split == "train" and e.role == "forget":
            by_fact[e.fact_id].append(e)
    fact_ids = sorted(by_fact)

    scalar = [p for n, p in values.named_parameters() if n in SCALAR_PARAMS]
    vector = [p for n, p in values.named_parameters() if n not in SCALAR_PARAMS]
    groups = [g for g in ({"params": vector, "lr": args.lr}, {"params": scalar, "lr": args.scale_lr})
              if g["params"]]
    optimizer = torch.optim.Adam(groups)
    params = [p for g in groups for p in g["params"]]
    storage = values.storage()
    print(json.dumps({"phase": "compressed_training_ready", "value_mode": args.value_mode,
                      "training_route": args.training_route, "facts_trained": len(fact_ids),
                      "untrainable_facts": len(untrainable),
                      "trainable_parameters": sum(p.numel() for p in params),
                      "storage": {k: storage[k] for k in ("per_fact_floats", "shared_floats",
                                                          "ratio_to_full_rows")},
                      "router": router_storage(source)}), flush=True)

    target = float(PLAN["target_probability"])
    best_key, best_state, best_epoch = None, values.compact_state(), 0
    gates, feasible_streak, stop_reason = [], 0, "epoch_budget"
    started = time.monotonic()
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        if time.monotonic() - started >= args.max_training_seconds:
            stop_reason = "wall_time_budget"
            break
        order = list(fact_ids)
        rng.shuffle(order)
        epoch_loss = 0.0
        for start in range(0, len(order), args.batch_facts):
            batch = order[start:start + args.batch_facts]
            optimizer.zero_grad(set_to_none=True)
            for fid in batch:
                answers = [train_answer[e.id] for e in by_fact[fid]]
                unknowns = [train_unknown[e.id] for e in by_fact[fid]]
                objective = fact_objective(wrapped, answers, unknowns, target, args.unknown_weight)
                (objective["loss"] / len(batch)).backward()
                epoch_loss += float(objective["loss"].detach())
            torch.nn.utils.clip_grad_norm_(params, args.clip, error_if_nonfinite=True)
            optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = routed_metrics(wrapped, train_answer, train_unknown, target)
            key = checkpoint_key(metrics)
            selected = best_key is None or key < best_key
            if selected:
                best_key, best_state, best_epoch = key, values.compact_state(), epoch
            feasible = metrics["train"]["target_met"] and metrics["development"]["target_met"]
            feasible_streak = feasible_streak + 1 if feasible else 0
            with torch.no_grad():
                norms = values.rows().norm(dim=-1)
            gate = {
                "epoch": epoch,
                "elapsed_seconds": round(time.monotonic() - started, 1),
                "mean_epoch_loss": epoch_loss / max(len(order), 1),
                "train_max_prob": metrics["train"]["max_token_probability"],
                "dev_max_prob": metrics["development"]["max_token_probability"],
                "train_facts_passing": metrics["train"]["facts_passing"],
                "dev_facts_passing": metrics["development"]["facts_passing"],
                "unknown_mean_nll": metrics["train"]["unknown_mean_nll"],
                "row_norm_median": float(norms.median()),
                "selected_as_best": selected,
            }
            gates.append(gate)
            print(json.dumps({"phase": "compressed_gate", **gate}), flush=True)
            if feasible_streak > int(PLAN["post_feasible_gates"]):
                stop_reason = "global_train_and_development_target_met"
                break

    values.load_state_dict(best_state)
    bank.set_oracle_routes(None)
    with torch.no_grad():
        # Canonical reconstruction: rebuild the rows on CPU in float32 from the
        # saved compact state. These exact rows are saved and evaluated.
        rebuilt = CompressedValues(
            mode, rank, facts, hidden, answer_token_ids=answer_token_ids,
            answer_dirs=values.answer_dirs.cpu() if hasattr(values, "answer_dirs") else None,
            seed=args.seed,
        )
        rebuilt.load_state_dict({k: v.cpu() for k, v in best_state.items()})
        rows = (rebuilt.rows() * mask.cpu()[:, None]).float().contiguous()
        if mode == "shared" and not torch.equal(rows, rows[:1].expand_as(rows)):
            raise RuntimeError("Shared-vector export contains unequal rows")
        trained = values.rows().detach().float().cpu()
        reconstruction_max_abs_diff = float((rows - trained).abs().max())
        if not torch.allclose(rows, trained, rtol=1e-4, atol=1e-5):
            raise RuntimeError(
                f"Compact reconstruction drifted from the trained rows ({reconstruction_max_abs_diff})"
            )
        # Evaluate the saved rows, not the in-memory module.
        values.rows = lambda: rows.to(args.device)
        if not torch.equal(base_logits, wrapped(**neutral, use_cache=False).logits):
            raise ValueError("Unmatched neutral prompt left the exact base path after training")
    classifier_metrics = routed_metrics(wrapped, answer_map, unknown_map, target)

    artifact = dict(source)
    artifact["rows"] = rows
    artifact["training_route"] = args.training_route
    artifact["trainable_parameters"] = sum(p.numel() for p in params)
    artifact["compressed_values"] = {
        "mode": args.value_mode,
        "compact_state": best_state,
        "answer_token_ids": answer_token_ids,
        "untrainable_fact_ids": untrainable,
        "storage": storage,
        "rows_are_exact_reconstruction": True,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    coverage = {"views": len(examples), "used_for_training": len(training_examples),
                "facts_trained": len(fact_ids), "facts_total": len(facts)}
    new_manifest = dict(manifest)
    new_manifest.update({
        "method": "sure_linear_router_compressed_bank",
        "value_mode": args.value_mode,
        "training_route": args.training_route,
        "router_v2_used": False,
        "training_coverage": coverage,
        "untrainable_fact_ids": untrainable,
        "value_storage": storage,
        "router_storage": router_storage(source),
        "residual_training_objective": "joint_forget_hinge_plus_unknown_nll",
        "unknown_completion": args.unknown_completion,
        "unknown_eos": args.unknown_eos,
        "residual_initialization": "zero",
    })
    (output / "association_manifest.json").write_text(json.dumps(new_manifest, indent=2) + "\n")
    for name in ("association_examples.json", "linear_router_report.json"):
        if (router_dir / name).is_file():
            shutil.copy2(router_dir / name, output / name)
    report = {
        "value_mode": args.value_mode,
        "residual_training_objective": "joint_forget_hinge_plus_unknown_nll",
        "training_coverage_policy": "correct_own_row_only_matching_full_baseline",
        "shared_vector_applies_to_every_active_route": mode == "shared",
        "training_route": args.training_route,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_checkpoint_key": list(best_key) if best_key else None,
        "gates": gates,
        "training_coverage": coverage,
        "views_excluded_unrouted": excluded,
        "untrainable_fact_ids": untrainable,
        "pre_training_route_audit": route_audit,
        "final_metrics_classifier_routing_all_views": classifier_metrics,
        "value_storage": storage,
        "router_storage": router_storage(source),
        "hyperparameters": {k: getattr(args, k) for k in (
            "lr", "scale_lr", "batch_facts", "epochs", "eval_every",
            "max_training_seconds", "unknown_weight", "unknown_completion", "unknown_eos",
            "clip", "seed")},
        "reconstruction_max_abs_diff_vs_trained": reconstruction_max_abs_diff,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": "compressed_bank_trained",
        "value_mode": args.value_mode,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "train_target_met_classifier_routing": classifier_metrics["train"]["target_met"],
        "dev_target_met_classifier_routing": classifier_metrics["development"]["target_met"],
        "value_ratio_to_full_rows": storage["ratio_to_full_rows"],
        "router_heads": router_storage(source)["heads"],
        "output_dir": str(output),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
