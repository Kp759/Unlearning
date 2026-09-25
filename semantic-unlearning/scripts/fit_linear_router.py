#!/usr/bin/env python3
"""Fit the learned linear router on an existing fact-association run.

Works on any Router V2 (or V1) run directory from MCF, ZsRE, MQuAKE or RWKU:
it keeps the run's trained residual rows, subject patterns, facts and layer,
replaces only the router, and writes a new run directory that the existing
official evaluators load unchanged (they dispatch on the artifact's
architecture via linear_router.load_router_artifact).

    python -u scripts/fit_linear_router.py \
      --run-dir outputs/mcf_fact_assoc_router_v2_seed1 \
      --output-dir outputs/mcf_fact_assoc_linear_router_seed1

    # then, exactly as for V2:
    python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
      --run-dir outputs/mcf_fact_assoc_linear_router_seed1 --mcf-path data/multi_counterfact.json

Gate defaults by benchmark (override with --gate):
    mcf, zsre, mquake   threshold   association-level forgetting
    rwku                subject     entity-level forgetting; heads choose the row

Nothing here reads official paraphrase, neighborhood, retain, utility or
evaluation probes. Positives are the run's training-visible direct prompts
(MCF: association_examples.json train/development families; other
benchmarks: each fact's canonical_prompts plus content-free context-prefix
variants). Negatives are same-subject competitors and subject transplants.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch

from linear_router import (
    ARCHITECTURE,
    training_positive_floor,
    calibrate_per_head,
    cosine_arm_artifact,
    v2_effective_scores,
    v2_route_outcomes_with_tau,
    DEFAULT_LAMBDAS,
    DEFAULT_PCA_DIMS,
    GATE_MODES,
    LinearClassifierAssociationBank,
    SPLITS,
    assemble_router_dataset,
    calibrate_threshold,
    decide_routes,
    examples_from_facts,
    fit_linear_router,
    linear_route_frontier,
    prompt_family,
    prototype_router_routes,
    route_outcomes,
    score_queries,
    select_hyperparameters,
    v2_route_frontier,
    v2_route_outcomes,
)
from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    extract_prompt_queries,
)

BENCHMARK_GATE = {"mcf": "threshold", "zsre": "threshold", "mquake": "threshold", "rwku": "subject"}
NEUTRAL_PROMPT = "A neutral sentence about mathematics and weather."
# Router internals of the source artifact (V2 / V1 / stochastic) that must not
# travel with the new router; every other source key is carried over.
SOURCE_ROUTER_KEYS = {
    "positive_prototypes", "negative_prototypes", "alpha", "tau", "keys",
    "thresholds", "ambiguity_margin", "routing_mode", "routing_temperature",
    "routing_abstain_logit", "routing_query_noise", "routing_hard_zero_below",
    "routing_seed", "gate_diagnostics", "router_diagnostics",
}


def _floats(text):
    return tuple(float(x) for x in str(text).split(",") if x.strip())


def _ints(text):
    return tuple(int(x) for x in str(text).split(",") if x.strip())


def detect_benchmark(facts):
    first = str(facts[0].get("id", ""))
    for name in ("mcf", "zsre", "mquake", "rwku"):
        if first.startswith(f"{name}_"):
            return name
    return "unknown"


def load_examples(run_dir, facts, augment, split_rule):
    """Training-visible positives, with at least two development families."""
    path = run_dir / "association_examples.json"
    if path.is_file() and augment != "on":
        examples = json.loads(path.read_text())
        if split_rule == "rebalanced":
            from router_fitting_data_v2 import rebalanced_split
            # rebalanced_split reads the family from `role`, but MCF examples
            # keep it in `group` (role is "forget"); pass the family through.
            examples, _ = rebalanced_split(
                [dict(e, role=prompt_family(e)) for e in examples]
            )
        families = {prompt_family(e) for e in examples if str(e.get("split")) != "train"}
        if len(families) >= 2:
            return examples, "association_examples.json"
        if augment == "off":
            raise ValueError(
                "association_examples.json has fewer than two development "
                "families; rerun with --augment auto"
            )
    if augment == "off":
        raise ValueError("No development families and --augment off")
    if split_rule == "rebalanced":
        raise ValueError("--split-rule rebalanced needs association_examples.json")
    return (
        examples_from_facts(facts, augment=True),
        "facts_canonical_prompts_plus_context_prefix_families",
    )


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.tolist())
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")


@torch.no_grad()
def runtime_parity(model, bank, tokenizer, prompts, offline, batch_size, device):
    """Run the real hook on every prompt with the extraction's batching.

    Compares the hook's route with the offline decision computed from
    extract_prompt_queries features; they must agree for the calibration to
    describe the deployed router.
    """
    wrapped = AssociationCausalLM(model, bank)
    mismatches, max_gap = [], 0.0
    for start in range(0, len(prompts), int(batch_size)):
        window = prompts[start:start + int(batch_size)]
        encoded = tokenizer(
            window, padding=True, return_tensors="pt", return_token_type_ids=False
        ).to(device)
        wrapped(**encoded, use_cache=False)
        for offset, (active_ids, score) in enumerate(
            zip(bank.last_active_fact_indices, bank.last_route_scores)
        ):
            row = start + offset
            expected = [int(offline["fact"][row])] if bool(offline["active"][row]) else []
            if active_ids != expected:
                mismatches.append({"prompt": prompts[row], "hook": active_ids, "offline": expected})
            hook_best = score["best_eligible_logit"]
            offline_best = float(offline["best_eligible_logit"][row])
            if hook_best is not None and math.isfinite(offline_best):
                max_gap = max(max_gap, abs(hook_best - offline_best))
    return {
        "prompts": len(prompts),
        "route_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "max_abs_best_eligible_logit_gap": max_gap,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="existing fact-association run")
    parser.add_argument("--output-dir", required=True, help="new run dir (must not exist)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"),
                        help="base-model dtype for query extraction (V2 fitting used float32)")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gate", choices=("auto",) + GATE_MODES, default="auto")
    parser.add_argument("--augment", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--split-rule", choices=("shipped", "rebalanced"), default="shipped")
    parser.add_argument("--negative-count", type=int, default=36)
    parser.add_argument("--per-donor", type=int, default=3)
    parser.add_argument("--no-type-check", action="store_true")
    parser.add_argument("--lambdas", default=",".join(str(x) for x in DEFAULT_LAMBDAS))
    parser.add_argument("--pca-dims", default=",".join(str(x) for x in DEFAULT_PCA_DIMS))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-answer-groups", action="store_true",
                        help="allow same-answer-group relations as negatives (ablation)")
    parser.add_argument("--target-fpr", type=float, default=0.0,
                        help="threshold gate: max false activation on calibration negatives")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="threshold gate, recall-first: lowest false activation with "
                             "calibration recall >= this (overrides --target-fpr)")
    parser.add_argument("--max-threshold-candidates", type=int, default=2000)
    parser.add_argument("--threshold-policy", choices=("global", "per_head"), default="global",
                        help="threshold policy of the main artifact (threshold gate only)")
    parser.add_argument("--per-head-fraction", type=float, default=0.1,
                        help="per-head t_i = hardest negative + f * gap (0.1 = V2's rule)")
    parser.add_argument("--per-head-slack", type=float, default=0.5,
                        help="logit slack below the weakest positive for non-separable heads")
    parser.add_argument("--per-head-shrink", type=float, default=0.0,
                        help="shrink per-head thresholds toward the global one (0 = off)")
    parser.add_argument("--emit-2x2", action="store_true",
                        help="also write {cosine, linear} x {global, per_head} arm run dirs")
    parser.add_argument("--threshold-placement-fraction", type=float, default=0.5,
                        help="where in the admissible gap the threshold sits: 0.5 = "
                             "midpoint, 0.1 = 10%% above the hardest negative (V2's rule)")
    parser.add_argument("--ambiguity-margin", type=float, default=0.5,
                        help="logit units; threshold gate only")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fit-device", default=None,
                        help="device for the classifier fits (default: --device)")
    parser.add_argument("--skip-runtime-parity", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output_dir).resolve()
    artifact_path = run_dir / "fact_association_embeddings.pt"
    manifest_path = run_dir / "association_manifest.json"
    for path in (artifact_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
    source = torch.load(artifact_path, map_location="cpu", weights_only=False)
    manifest = json.loads(manifest_path.read_text())
    facts = list(source["facts"])
    subject_patterns = source["subject_patterns"]
    layer = int(source["layer"])
    rows = source["rows"]
    benchmark = detect_benchmark(facts)
    gate = BENCHMARK_GATE.get(benchmark, "threshold") if args.gate == "auto" else args.gate
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=getattr(torch, args.dtype),
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)

    examples, example_source = load_examples(run_dir, facts, args.augment, args.split_rule)
    data = assemble_router_dataset(
        facts, examples, tokenizer, subject_patterns,
        negative_count=args.negative_count,
        per_donor=args.per_donor,
        type_check=not args.no_type_check,
        answer_groups=not args.no_answer_groups,
    )
    prompts, labels, eligible, owner = data["prompts"], data["labels"], data["eligible"], data["owner"]
    split = data["split"]
    masks = {name: torch.tensor([s == name for s in split], dtype=torch.bool) for name in SPLITS}
    print(json.dumps({"phase": "dataset_ready", "benchmark": benchmark, "gate": gate,
                      "example_source": example_source,
                      "by_split": data["diagnostics"]["by_split"]}), flush=True)

    started = time.time()
    print(json.dumps({"phase": "extracting_queries", "prompts": len(prompts)}), flush=True)
    queries = extract_prompt_queries(model, tokenizer, prompts, layer, batch_size=args.batch_size)
    print(json.dumps({"phase": "queries_ready", "seconds": round(time.time() - started, 1)}),
          flush=True)
    fit_device = args.fit_device or args.device
    fit = masks["fit"]
    fit_groups = [g for g, flag in zip(data["groups"], fit.tolist()) if flag]

    def progress(row, done, total):
        print(json.dumps({
            "phase": "cv", "cell": f"{done}/{total}", "pca_dim": row["pca_dim"],
            "l2": row["l2"], "held_out_log_loss": row["held_out_balanced_log_loss"],
            "held_out_pair_auc": row["held_out_pair_auc"],
            "seconds": round(time.time() - started, 1),
        }), flush=True)

    l2, pca_dim, cv = select_hyperparameters(
        queries[fit], labels[fit], eligible[fit], fit_groups,
        lambdas=_floats(args.lambdas), pca_dims=_ints(args.pca_dims), folds=args.cv_folds,
        device=fit_device, progress=progress,
    )
    router = fit_linear_router(
        queries[fit], labels[fit], eligible[fit], l2=l2, pca_dim=pca_dim, device=fit_device,
    )
    logits = score_queries(
        queries, router["weight"], router["bias"],
        router["feature_mean"], router["feature_components"],
    )
    print(json.dumps({"phase": "fit_complete", "l2": l2, "pca_dim": pca_dim,
                      "converged": router["info"]["converged"]}), flush=True)

    if gate == "threshold":
        cal = masks["calibration"]
        threshold, calibration = calibrate_threshold(
            logits[cal], eligible[cal], owner[cal],
            target_fpr=args.target_fpr, min_recall=args.min_recall,
            ambiguity_margin=args.ambiguity_margin,
            placement_fraction=args.threshold_placement_fraction,
            max_candidates=args.max_threshold_candidates,
        )
        margin = float(args.ambiguity_margin)
        fit_mask = masks["fit"]
        per_head, per_head_report = calibrate_per_head(
            logits[cal], eligible[cal], owner[cal],
            fraction=args.per_head_fraction, slack=args.per_head_slack,
            shrink=args.per_head_shrink, fallback=threshold,
            ceiling=training_positive_floor(logits[fit_mask], eligible[fit_mask], owner[fit_mask]),
        )
        calibration["per_head"] = per_head_report
    else:
        per_head = None
        threshold, margin = float("-inf"), 0.0
        calibration = {
            "rule": "subject_gate",
            "note": "Every subject-eligible prompt fires; heads choose the row. "
                    "False activation on subject transplants is 1 by design.",
        }

    use_per_head = gate == "threshold" and args.threshold_policy == "per_head"
    main_threshold = per_head if use_per_head else threshold
    outcomes = {
        name: route_outcomes(logits[m], eligible[m], owner[m], main_threshold, margin, facts)
        for name, m in masks.items()
    }
    comparison, frontier, v2_routes = None, None, None
    audit_mask = masks["audit"]
    is_v2 = str(source.get("architecture", "")) == "relation_prototype_fact_association_bank_v2"
    if is_v2:
        comparison = {
            name: v2_route_outcomes(queries[m], eligible[m], owner[m], source)
            for name, m in masks.items()
        }
        v2_routes = prototype_router_routes(queries, eligible, source)

    # The 2x2: {cosine (V2 d_i), linear (z_i)} x {global, per-head} thresholds,
    # all calibrated on the same held-out calibration split with the same rule.
    two_by_two, arm_thresholds = None, {}
    if gate == "threshold":
        cal = masks["calibration"]
        arm_thresholds["linear_global"] = threshold
        arm_thresholds["linear_per_head"] = per_head
        two_by_two = {
            arm: {
                name: route_outcomes(logits[m], eligible[m], owner[m], arm_thresholds[arm], margin)
                for name, m in masks.items()
            }
            for arm in ("linear_global", "linear_per_head")
        }
        if is_v2:
            d_eff = v2_effective_scores(queries, source)
            v2_margin = float(source.get("ambiguity_margin", 0.02))
            cos_global, cos_global_report = calibrate_threshold(
                d_eff[cal], eligible[cal], owner[cal],
                target_fpr=args.target_fpr, min_recall=args.min_recall,
                ambiguity_margin=v2_margin,
                placement_fraction=args.threshold_placement_fraction,
                max_candidates=args.max_threshold_candidates,
            )
            fit_mask = masks["fit"]
            cos_per_head, cos_per_head_report = calibrate_per_head(
                d_eff[cal], eligible[cal], owner[cal],
                fraction=args.per_head_fraction, slack=0.02,
                shrink=args.per_head_shrink, fallback=cos_global,
                ceiling=training_positive_floor(
                    d_eff[fit_mask], eligible[fit_mask], owner[fit_mask], epsilon=1e-4
                ),
            )
            arm_thresholds["cosine_global"] = cos_global
            arm_thresholds["cosine_per_head"] = cos_per_head
            calibration["cosine_global"] = cos_global_report
            calibration["cosine_per_head"] = cos_per_head_report
            for arm in ("cosine_global", "cosine_per_head"):
                two_by_two[arm] = {
                    name: v2_route_outcomes_with_tau(
                        queries[m], eligible[m], owner[m], source, arm_thresholds[arm]
                    )
                    for name, m in masks.items()
                }
            two_by_two["v2_shipped_in_sample_tau"] = comparison
    reference = None
    if comparison is not None:
        shipped = comparison["audit"]
        reference = (
            shipped["false_activation_on_negative_control"]["rate"] or 0.0,
            shipped["correct_route"]["rate"] or 0.0,
        )
    frontier = {
        "note": ("Threshold sweeps on the audit split. Picking a point on this "
                 "curve uses audit labels: compare routers at matched operating "
                 "points, but deploy only the calibrated threshold."),
        "linear": linear_route_frontier(
            logits[audit_mask], eligible[audit_mask], owner[audit_mask],
            float(args.ambiguity_margin), reference=reference,
            max_candidates=args.max_threshold_candidates,
        ),
        "v2_tau_shift": (
            v2_route_frontier(queries[audit_mask], eligible[audit_mask],
                              owner[audit_mask], source,
                              max_candidates=args.max_threshold_candidates)
            if is_v2 else None
        ),
        "v2_shipped_audit_point": (
            None if reference is None
            else {"false_activation": reference[0], "correct_route": reference[1]}
        ),
    }
    if comparison is not None:
        gate_file = run_dir / "gate_diagnostics.json"
        if gate_file.is_file():
            seen = {
                p for row in json.loads(gate_file.read_text()).get("per_fact", [])
                for p in row.get("negative_prompts", [])
            }
            comparison["audit_prompts_that_were_v2_negative_prototypes"] = sum(
                1 for p, m in zip(prompts, masks["audit"].tolist()) if m and p in seen
            )

    router_fit = {
        "benchmark": benchmark,
        "gate_mode": gate,
        "example_source": example_source,
        "split_rule": args.split_rule,
        "selected_l2": l2,
        "selected_pca_dim": pca_dim,
        "threshold_logit": threshold if math.isfinite(threshold) else None,
        "ambiguity_margin": margin,
        "target_fpr": calibration.get("target_fpr") if gate == "threshold" else None,
        "min_recall": calibration.get("min_recall") if gate == "threshold" else None,
        "threshold_policy": args.threshold_policy if gate == "threshold" else "subject_gate",
        "per_head_fraction": args.per_head_fraction if gate == "threshold" else None,
        "per_head_slack": args.per_head_slack if gate == "threshold" else None,
        "per_head_shrink": args.per_head_shrink if gate == "threshold" else None,
        "answer_groups": not args.no_answer_groups,
        "threshold_placement_fraction": (
            args.threshold_placement_fraction if gate == "threshold" else None
        ),
        "fit_info": router["info"],
        "audit": {k: v for k, v in outcomes["audit"].items()},
        "source_run_dir": str(run_dir),
    }
    # Exact base path for a prompt with no protected subject, with trained rows:
    # base logits first, before any hook is attached.
    neutral = tokenizer(NEUTRAL_PROMPT, return_tensors="pt").to(args.device)
    with torch.no_grad():
        base_logits = model(**neutral, use_cache=False).logits.detach().clone()
    bank = LinearClassifierAssociationBank(
        base_model=model, layer=layer,
        weight=router["weight"], bias=router["bias"],
        feature_mean=router["feature_mean"], feature_components=router["feature_components"],
        threshold=threshold, subject_patterns=subject_patterns, facts=facts,
        rows=rows, ambiguity_margin=margin, gate_mode=gate, router_fit=_json_safe(router_fit),
        per_head_thresholds=per_head if use_per_head else None,
    )
    for row in bank.rows:
        row.requires_grad_(False)
    with torch.no_grad():
        wrapped_logits = AssociationCausalLM(model, bank)(**neutral, use_cache=False).logits
    neutral_exact = bool(torch.equal(base_logits, wrapped_logits))
    if not neutral_exact:
        raise RuntimeError("Unmatched neutral prompt left the exact base path")

    parity = None
    if not args.skip_runtime_parity:
        offline = decide_routes(logits, eligible, main_threshold, margin)
        parity = runtime_parity(model, bank, tokenizer, prompts, offline, args.batch_size, args.device)

    artifact = bank.artifact()
    # Carry the source run's benchmark metadata (e.g. MQuAKE's
    # atomic_case_to_association_id, ZsRE's dataset / target_new_used): the
    # official evaluators validate it. Only V2/V1 router internals are dropped.
    carried = {
        key: value for key, value in source.items()
        if key not in artifact and key not in SOURCE_ROUTER_KEYS
    }
    artifact.update(carried)
    artifact["carried_source_keys"] = sorted(carried)
    torch.save(artifact, output / "fact_association_embeddings.pt")
    # Evaluators read only the manifest and the artifact. Copy nothing else
    # from the source run except its training-visible examples: copying its
    # evaluation outputs would put V2 results in a linear-router directory.
    if (run_dir / "association_examples.json").is_file():
        shutil.copy2(run_dir / "association_examples.json",
                     output / "association_examples.json")
    new_manifest = dict(manifest)
    new_manifest.update({
        "architecture": ARCHITECTURE,
        "routing_policy": artifact["routing_policy"],
        "runtime_trigger": (
            "complete subject-token eligibility plus learned linear BCE head "
            + ("above one calibrated global threshold" if gate == "threshold"
               else "row selection (subject gate)")
        ),
        "router": _json_safe(router_fit),
        "router_source_run_dir": str(run_dir),
        "router_source_architecture": str(source.get("architecture", "")),
        "residual_rows_reused_from_source": True,
        "router_uses_official_eval_fields": False,
    })
    _write_json(output / "association_manifest.json", new_manifest)

    arm_dirs = {}
    if args.emit_2x2 and two_by_two is not None:
        for arm, value in arm_thresholds.items():
            arm_dir = output / "arms" / arm
            arm_dir.mkdir(parents=True, exist_ok=False)
            if arm.startswith("linear"):
                arm_artifact = dict(artifact)
                vector = arm == "linear_per_head"
                arm_artifact["per_head_thresholds"] = value.clone() if vector else None
                arm_artifact["threshold"] = float(threshold)
                arm_artifact["threshold_policy"] = "per_head" if vector else "global"
                arm_artifact["routing_policy"] = (
                    "subject_eligibility_mask_plus_linear_bce_heads_top1_"
                    + ("per_head_thresholds" if vector else "global_threshold")
                )
            else:
                arm_artifact = cosine_arm_artifact(
                    source, value, arm=arm,
                    calibration=_json_safe(calibration[arm]),
                )
            arm_artifact["router_arm"] = arm
            torch.save(arm_artifact, arm_dir / "fact_association_embeddings.pt")
            arm_manifest = dict(new_manifest)
            arm_manifest.update({
                "architecture": str(arm_artifact["architecture"]),
                "routing_policy": arm_artifact.get("routing_policy"),
                "router_arm": arm,
                "router_arm_note": (
                    "Same data, splits and calibration rule for all four arms; "
                    "cosine arms keep V2's prototypes, bank and rows with tau "
                    "recalibrated on held-out prompts."
                ),
            })
            _write_json(arm_dir / "association_manifest.json", arm_manifest)
            arm_dirs[arm] = str(arm_dir)
    report = {
        "schema_version": "linear_router_fit_v1",
        "router_fit": router_fit,
        "cross_validation": cv,
        "calibration": calibration,
        "route_outcomes_by_split": outcomes,
        "v2_same_prompts_by_split": comparison,
        "two_by_two_by_split": two_by_two,
        "audit_frontier": frontier,
        "runtime_parity": parity,
        "neutral_prompt_exact_base": neutral_exact,
        "dataset": data["diagnostics"],
        "command": sys.argv,
    }
    _write_json(output / "linear_router_report.json", report)
    decision = decide_routes(logits, eligible, main_threshold, margin)
    rows_out = []
    for index, r in enumerate(data["records"]):
        best = float(decision["best_eligible_logit"][index])
        row = {
            "prompt": r["prompt"], "split": r["split"],
            "owner_fact_id": facts[r["owner"]]["id"] if r["owner"] >= 0 else None,
            "kind": r["kind"], "group": r["group"],
            "negative_for": [facts[i]["id"] for i in r["negative_for"]],
            "donor_relation": r.get("donor_relation"),
            "linear_best_eligible_logit": best if math.isfinite(best) else None,
            "linear_best_eligible_fact_id": (
                facts[int(decision["best_eligible_fact"][index])]["id"]
                if math.isfinite(best) else None
            ),
            "linear_routes_to": (
                facts[int(decision["fact"][index])]["id"]
                if bool(decision["active"][index]) else None
            ),
        }
        if v2_routes is not None:
            row["v2_routes_to"] = (
                facts[int(v2_routes[1][index])]["id"] if bool(v2_routes[0][index]) else None
            )
        rows_out.append(row)
    _write_json(output / "linear_router_dataset.json", rows_out)

    audit = outcomes["audit"]
    summary = {
        "status": "linear_router_complete",
        "benchmark": benchmark,
        "gate": gate,
        "l2": l2,
        "pca_dim": pca_dim,
        "threshold_logit": threshold if math.isfinite(threshold) else None,
        "audit_correct_route": audit["correct_route"]["rate"],
        "audit_false_activation": audit["false_activation_on_negative_control"]["rate"],
        "v2_audit_correct_route": (comparison or {}).get("audit", {}).get("correct_route", {}).get("rate"),
        "v2_audit_false_activation": (comparison or {}).get("audit", {}).get(
            "false_activation_on_negative_control", {}).get("rate"),
        "audit_route_auc_linear": frontier["linear"]["route_auc"],
        "audit_route_auc_v2": (frontier["v2_tau_shift"] or {}).get("route_auc"),
        "linear_recall_at_v2_false_activation": (
            frontier["linear"].get("at_reference_fpr", {}).get("recall")
        ),
        "linear_false_activation_at_v2_recall": (
            frontier["linear"].get("at_reference_recall", {}).get("fpr")
        ),
        "runtime_route_mismatches": None if parity is None else parity["route_mismatches"],
        "threshold_policy": router_fit["threshold_policy"],
        "audit_2x2": None if two_by_two is None else {
            arm: {
                "correct_route": block["audit"]["correct_route"]["rate"],
                "false_activation": block["audit"]["false_activation_on_negative_control"]["rate"],
            }
            for arm, block in two_by_two.items() if block is not None
        },
        "arm_dirs": arm_dirs or None,
        "neutral_prompt_exact_base": neutral_exact,
        "output_dir": str(output),
    }
    print(json.dumps(_json_safe(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
