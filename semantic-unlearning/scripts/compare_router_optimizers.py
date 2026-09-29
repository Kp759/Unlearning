#!/usr/bin/env python3
"""L-BFGS vs SGD router: same results or different? One report per benchmark.

    python scripts/compare_router_optimizers.py \
        --reference-router outputs/mcf_multiseed_regular_v1/seed1/L19/router \
        --reference-run    outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global \
        --control-router   outputs/optimizer_ablation_v1/mcf/seed1/L19/router_lbfgs_rerun \
        --control-run      outputs/optimizer_ablation_v1/mcf/seed1/L19/full_lbfgs_rerun \
        --candidate-router outputs/optimizer_ablation_v1/mcf/seed1/L19/router_sgd \
        --candidate-swap-run outputs/optimizer_ablation_v1/mcf/seed1/L19/swap_sgd \
        --candidate-run    outputs/optimizer_ablation_v1/mcf/seed1/L19/full_sgd \
        --out-prefix       outputs/optimizer_ablation_v1/mcf/seed1/L19/comparison

reference   the shipped L-BFGS router and the rows trained under it
control     L-BFGS refit through the same code path (+ rows retrained): the
            run-to-run noise floor (query extraction on a GPU, row training)
candidate   SGD router; `swap` = reference rows behind the SGD router (router
            effect only, eval-time); `full` = rows retrained under SGD

Sections: (1) stage-1 fit (objective, gradient, twin L-BFGS on the same
features), (2) parameters in hidden space (PCA undone, so sign flips of the
basis do not matter), (3) routes on every router prompt per split
(fit / calibration / audit), (4) which training views each router lets the
row trainer use, (5) official metrics with deltas against the noise floor.
Any optional input that is missing is reported as missing, not guessed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_mcf_layer_sweep import collect  # noqa: E402

METRICS = {
    "mcf": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL",
            "display_zero"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL",
             "forget_rewrite_route_active", "forget_paraphrase_route_active",
             "forget_neighborhood_route_active", "facts_trained"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_Eff", "retain_AtomicGen", "PPL",
               "forget_rewrite_route_correct", "forget_atomicgen_route_correct",
               "retain_atomicgen_route_active", "facts_trained"],
}
EXACT = 1e-12


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _get(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


# ---------------------------------------------------------------------------
# (1) fit and (2) parameters
# ---------------------------------------------------------------------------

def fit_summary(router_dir):
    report = _load(Path(router_dir) / "linear_router_report.json")
    if report is None:
        return None
    fit = report["router_fit"]
    info = fit.get("fit_info") or {}
    sgd = info.get("sgd") or {}
    return {
        "optimizer": info.get("optimizer", "lbfgs"),
        "selected_l2": fit.get("selected_l2"),
        "selected_pca_dim": fit.get("selected_pca_dim"),
        "objective": info.get("objective"),
        "max_abs_gradient": info.get("max_abs_gradient"),
        "converged": info.get("converged"),
        "lbfgs_iterations": info.get("lbfgs_iterations"),
        "sgd_lr": sgd.get("lr"),
        "sgd_epochs": sgd.get("epochs"),
        "sgd_batch_size": sgd.get("batch_size"),
        "sgd_momentum": sgd.get("momentum"),
        "calibrated_threshold_logit": fit.get("threshold_logit"),
        "bias_shift": fit.get("bias_shift"),
        "audit_correct_route": _get(report, "route_outcomes_by_split", "audit",
                                    "correct_route", "rate"),
        "audit_false_activation": _get(report, "route_outcomes_by_split", "audit",
                                       "false_activation_on_negative_control", "rate"),
        "audit_route_auc": _get(report, "audit_frontier", "linear", "route_auc"),
        "runtime_route_mismatches": _get(report, "runtime_parity", "route_mismatches"),
        "lbfgs_twin_same_features": info.get("lbfgs_twin_same_features"),
    }


def _router_tensors(router_dir):
    art = torch.load(Path(router_dir) / "fact_association_embeddings.pt",
                     map_location="cpu", weights_only=False)
    weight = art["router_weight"].double()
    components = art.get("feature_components")
    if components is not None:
        weight = weight @ components.double()  # back to hidden space: PCA sign-free
    calib = art.get("bias_calibration") or {}
    stage1 = calib.get("stage1_bias")
    return {
        "facts": [str(f["id"]) for f in art["facts"]],
        "weight_hidden": weight,
        "feature_mean": art["feature_mean"].double(),
        "deployed_bias": art["router_bias"].double(),
        "stage1_bias": None if stage1 is None else torch.as_tensor(stage1).double(),
        "global_shift": calib.get("global_shift"),
        "head_index": art.get("head_index"),
    }


def parameter_diff(reference_dir, other_dir):
    a, b = _router_tensors(reference_dir), _router_tensors(other_dir)
    if a["facts"] != b["facts"]:
        return {"error": "different facts or order"}
    if a["weight_hidden"].shape != b["weight_hidden"].shape:
        return {"error": f"head shapes differ {tuple(a['weight_hidden'].shape)} vs "
                         f"{tuple(b['weight_hidden'].shape)}"}
    wa, wb = a["weight_hidden"], b["weight_hidden"]
    cos = F.cosine_similarity(wb, wa, dim=1)
    rel = (wb - wa).norm(dim=1) / wa.norm(dim=1).clamp_min(1e-12)
    out = {
        "hidden_space_weight_cosine": {"min": float(cos.min()), "median": float(cos.median())},
        "relative_weight_l2_diff": {"max": float(rel.max()), "median": float(rel.median())},
        "weight_norm_ratio": float(wb.norm() / wa.norm().clamp_min(1e-12)),
        "feature_mean_max_abs_diff": float((b["feature_mean"] - a["feature_mean"]).abs().max()),
        "deployed_bias_max_abs_diff": float((b["deployed_bias"] - a["deployed_bias"]).abs().max()),
        "global_shift": {"reference": a["global_shift"], "other": b["global_shift"]},
    }
    if a["stage1_bias"] is not None and b["stage1_bias"] is not None:
        out["stage1_bias_max_abs_diff"] = float((b["stage1_bias"] - a["stage1_bias"]).abs().max())
    return out


# ---------------------------------------------------------------------------
# (3) routes on the router's own prompt set
# ---------------------------------------------------------------------------

def _key(row):
    return (row["prompt"], row["split"], row.get("owner_fact_id"), row.get("kind"),
            tuple(row.get("negative_for") or ()), row.get("group"))


def _abs_stats(values):
    values = [abs(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return {"max": max(values), "mean": sum(values) / len(values)}


def route_diff(reference_dir, other_dir):
    ref = _load(Path(reference_dir) / "linear_router_dataset.json")
    oth = _load(Path(other_dir) / "linear_router_dataset.json")
    if ref is None or oth is None:
        return {"error": "linear_router_dataset.json missing"}
    ref_by = {}
    for row in ref:
        ref_by.setdefault(_key(row), []).append(row)
    unmatched, pairs = 0, []
    for row in oth:
        bucket = ref_by.get(_key(row))
        if bucket:
            pairs.append((bucket.pop(0), row))
        else:
            unmatched += 1
    unmatched_ref = sum(len(v) for v in ref_by.values())
    by_split = {}
    for split in sorted({r["split"] for r, _ in pairs}):
        chosen = [(r, o) for r, o in pairs if r["split"] == split]
        positives = [(r, o) for r, o in chosen if r.get("owner_fact_id") is not None]
        negatives = [(r, o) for r, o in chosen if r.get("owner_fact_id") is None]

        def correct(row):
            return row.get("linear_routes_to") == row.get("owner_fact_id")

        changed = [(r, o) for r, o in chosen if r.get("linear_routes_to") != o.get("linear_routes_to")]
        by_split[split] = {
            "prompts": len(chosen),
            "route_changes": len(changed),
            "positives": len(positives),
            "positives_correct": {"reference": sum(correct(r) for r, _ in positives),
                                  "other": sum(correct(o) for _, o in positives)},
            "positives_newly_correct": sum((not correct(r)) and correct(o) for r, o in positives),
            "positives_newly_missed": sum(correct(r) and not correct(o) for r, o in positives),
            "negatives": len(negatives),
            "negatives_firing": {
                "reference": sum(r.get("linear_routes_to") is not None for r, _ in negatives),
                "other": sum(o.get("linear_routes_to") is not None for _, o in negatives),
            },
            "best_eligible_fact_changes": sum(
                r.get("linear_best_eligible_fact_id") != o.get("linear_best_eligible_fact_id")
                for r, o in chosen
            ),
            "stage1_logit_abs_diff": _abs_stats([
                None if r.get("linear_best_eligible_stage1_logit") is None
                or o.get("linear_best_eligible_stage1_logit") is None
                else o["linear_best_eligible_stage1_logit"] - r["linear_best_eligible_stage1_logit"]
                for r, o in chosen
            ]),
            "deployed_logit_abs_diff": _abs_stats([
                None if r.get("linear_best_eligible_logit") is None
                or o.get("linear_best_eligible_logit") is None
                else o["linear_best_eligible_logit"] - r["linear_best_eligible_logit"]
                for r, o in chosen
            ]),
            "examples": [
                {"prompt": r["prompt"], "owner": r.get("owner_fact_id"),
                 "reference_routes_to": r.get("linear_routes_to"),
                 "other_routes_to": o.get("linear_routes_to"),
                 "reference_logit": r.get("linear_best_eligible_logit"),
                 "other_logit": o.get("linear_best_eligible_logit")}
                for r, o in changed[:10]
            ],
        }
    return {
        "matched_prompts": len(pairs),
        "unmatched_prompts": {"other_only": unmatched, "reference_only": unmatched_ref},
        "route_changes_total": sum(v["route_changes"] for v in by_split.values()),
        "by_split": by_split,
    }


# ---------------------------------------------------------------------------
# (4) training views and (5) official metrics
# ---------------------------------------------------------------------------

def training_views(run_dir):
    manifest = _load(Path(run_dir) / "association_manifest.json") or {}
    report = _load(Path(run_dir) / "training_report.json") or {}
    return {
        "excluded": set(manifest.get("views_excluded_unrouted")
                        or report.get("views_excluded_unrouted") or []),
        "untrainable": list(manifest.get("untrainable_fact_ids")
                            or report.get("untrainable_fact_ids") or []),
        "coverage": manifest.get("training_coverage") or report.get("training_coverage"),
        "stop_reason": report.get("stop_reason"),
        "best_step": report.get("best_step"),
        "present": bool(manifest or report),
    }


def views_diff(reference_run, other_run):
    a, b = training_views(reference_run), training_views(other_run)
    if not (a["present"] and b["present"]):
        return {"error": "missing run manifest or training report"}
    return {
        "excluded_views": {"reference": len(a["excluded"]), "other": len(b["excluded"]),
                           "only_reference": sorted(a["excluded"] - b["excluded"])[:20],
                           "only_other": sorted(b["excluded"] - a["excluded"])[:20],
                           "identical": a["excluded"] == b["excluded"]},
        "untrainable_facts": {"reference": a["untrainable"], "other": b["untrainable"]},
        "coverage": {"reference": a["coverage"], "other": b["coverage"]},
        "stop_reason": {"reference": a["stop_reason"], "other": b["stop_reason"]},
        "best_step": {"reference": a["best_step"], "other": b["best_step"]},
    }


def _num(value):
    if isinstance(value, bool):
        return float(value)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def metrics_table(dataset, runs):
    """runs: {label: run_dir or None}. Deltas vs 'reference'; noise = control."""
    rows = {label: (collect(path, label) if path and Path(path).is_dir() else None)
            for label, path in runs.items()}
    ref, ctl = rows.get("reference"), rows.get("control_full")
    table = []
    for metric in METRICS[dataset]:
        entry = {"metric": metric}
        base = _num((ref or {}).get(metric))
        noise = None
        if ctl is not None and base is not None and _num(ctl.get(metric)) is not None:
            noise = abs(_num(ctl.get(metric)) - base)
        entry["noise_floor"] = noise
        for label, row in rows.items():
            value = None if row is None else row.get(metric)
            entry[label] = value
            if label == "reference" or row is None or base is None or _num(value) is None:
                continue
            delta = _num(value) - base
            entry[f"delta[{label}]"] = delta
            if abs(delta) <= EXACT:
                verdict = "identical"
            elif noise is not None and abs(delta) <= noise + EXACT:
                verdict = "within noise"
            else:
                verdict = "differs" if noise is not None else "differs (no noise floor)"
            entry[f"verdict[{label}]"] = verdict
        table.append(entry)
    status = {label: (None if row is None else row.get("status")) for label, row in rows.items()}
    return table, status


def overall(routes, table):
    """One-line answers: classifier level and pipeline level."""
    changes = (routes.get("candidate") or {}).get("route_changes_total")
    floor = (routes.get("control") or {}).get("route_changes_total")
    if changes is None:
        router = "routes: not compared (missing dataset files)"
    elif changes == 0:
        router = "routes: SGD and L-BFGS route every router prompt identically"
    else:
        router = (f"routes: SGD changes {changes} route(s) across fit/calibration/audit"
                  + (f" (L-BFGS rerun noise floor: {floor})" if floor is not None else ""))
    verdicts = [v for row in table for k, v in row.items() if k == "verdict[candidate_full]"]
    if not verdicts:
        pipeline = "pipeline: full SGD run not evaluated yet"
    elif all(v == "identical" for v in verdicts):
        pipeline = "pipeline: SAME official results (all metrics identical)"
    elif all(v in ("identical", "within noise") for v in verdicts):
        pipeline = "pipeline: SAME official results within the L-BFGS rerun noise floor"
    else:
        differing = [row["metric"] for row in table
                     if str(row.get("verdict[candidate_full]", "")).startswith("differs")]
        pipeline = "pipeline: DIFFERENT official results on " + ", ".join(differing)
    return {"router": router, "pipeline": pipeline}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def _fmt(value):
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def markdown(result):
    lines = [f"# Router optimizer ablation: {result['dataset']} seed {result['seed']} "
             f"L{result['layer']}", ""]
    lines += [f"- **{result['overall']['router']}**", f"- **{result['overall']['pipeline']}**", ""]
    lines += ["## Stage-1 fit", "",
              "| router | optimizer | L2 | PCA | objective | max grad | converged | "
              "audit correct | audit false act. | audit AUC |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for label, fit in result["fit"].items():
        if fit is None:
            lines.append(f"| {label} | missing |" + " |" * 8)
            continue
        lines.append("| " + " | ".join(_fmt(x) for x in (
            label, fit["optimizer"], fit["selected_l2"], fit["selected_pca_dim"],
            fit["objective"], fit["max_abs_gradient"], fit["converged"],
            fit["audit_correct_route"], fit["audit_false_activation"], fit["audit_route_auc"],
        )) + " |")
    twin = (result["fit"].get("candidate") or {}).get("lbfgs_twin_same_features")
    if twin:
        lines += ["", "SGD vs L-BFGS on the same features (in-process twin): "
                  f"relative objective gap {_fmt(twin['relative_objective_gap'])}, "
                  f"weight cosine min {_fmt(twin['weight_cosine']['min'])}, "
                  f"norm ratio {_fmt(twin['weight_norm_ratio_sgd_over_lbfgs'])}, "
                  f"fit-pair sign disagreements {twin['fit_pairs_sign_disagreement_at_zero']}."]
    lines += ["", "## Parameters vs reference (hidden space)", "",
              "| vs reference | weight cos min | rel. ΔW max | norm ratio | Δ deployed bias max |",
              "|---|---|---|---|---|"]
    for label, diff in result["parameters"].items():
        if diff is None or "error" in diff:
            lines.append(f"| {label} | {(diff or {}).get('error', 'missing')} | | | |")
            continue
        lines.append("| " + " | ".join(_fmt(x) for x in (
            label, diff["hidden_space_weight_cosine"]["min"],
            diff["relative_weight_l2_diff"]["max"], diff["weight_norm_ratio"],
            diff["deployed_bias_max_abs_diff"])) + " |")
    lines += ["", "## Routes on the router's prompts", "",
              "| vs reference | split | prompts | route changes | positives correct (ref → other) | "
              "negatives firing (ref → other) |", "|---|---|---|---|---|---|"]
    for label, diff in result["routes"].items():
        if diff is None or "error" in diff:
            lines.append(f"| {label} | {(diff or {}).get('error', 'missing')} | | | | |")
            continue
        for split, s in diff["by_split"].items():
            lines.append("| " + " | ".join(_fmt(x) for x in (
                label, split, s["prompts"], s["route_changes"],
                f"{s['positives_correct']['reference']} → {s['positives_correct']['other']}",
                f"{s['negatives_firing']['reference']} → {s['negatives_firing']['other']}",
            )) + " |")
    lines += ["", "## Training views used by the row trainer", ""]
    for label, diff in result["training_views"].items():
        if diff is None or "error" in diff:
            lines.append(f"- {label}: {(diff or {}).get('error', 'missing')}")
            continue
        ev = diff["excluded_views"]
        lines.append(f"- {label}: excluded views {ev['reference']} → {ev['other']} "
                     f"(identical set: {_fmt(ev['identical'])}); untrainable facts "
                     f"{len(diff['untrainable_facts']['reference'])} → "
                     f"{len(diff['untrainable_facts']['other'])}; stop "
                     f"{diff['stop_reason']['reference']} → {diff['stop_reason']['other']}")
    labels = [k for k in result["runs"] if k != "reference"]
    lines += ["", "## Official metrics", "",
              "| metric | reference | " + " | ".join(labels) + " | noise floor | "
              + " | ".join(f"verdict {k}" for k in labels) + " |",
              "|" + "---|" * (3 + 2 * len(labels))]
    for row in result["metrics"]:
        lines.append("| " + " | ".join(
            [row["metric"], _fmt(row.get("reference"))]
            + [_fmt(row.get(k)) for k in labels]
            + [_fmt(row.get("noise_floor"))]
            + [_fmt(row.get(f"verdict[{k}]")) for k in labels]) + " |")
    lines += ["", "Run status: " + ", ".join(f"{k}={v}" for k, v in result["run_status"].items()),
              "", "`swap` = reference rows behind the SGD router (eval-time router effect only); "
              "`candidate_full` = rows retrained under SGD; noise floor = |L-BFGS rerun − reference|."]
    return "\n".join(lines) + "\n"


def collect_all(root):
    """One table over every comparison.json under root (all benchmarks)."""
    root = Path(root)
    lines = ["# Router optimizer ablation: SGD vs L-BFGS", "",
             "| benchmark | seed | layer | route changes (SGD / L-BFGS rerun) | metric | "
             "L-BFGS | SGD full | SGD swap | noise floor | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for path in sorted(root.glob("*/seed*/L*/comparison.json")):
        result = json.loads(path.read_text())
        changes = (result["routes"].get("candidate") or {}).get("route_changes_total")
        floor = (result["routes"].get("control") or {}).get("route_changes_total")
        head = [result["dataset"], result["seed"], result["layer"], f"{_fmt(changes)} / {_fmt(floor)}"]
        for row in result["metrics"]:
            if row["metric"] not in ("forget_Eff", "forget_Gen", "forget_AtomicGen",
                                     "forget_Spe", "retain_Eff", "PPL"):
                continue
            lines.append("| " + " | ".join(head + [_fmt(x) for x in (
                row["metric"], row.get("reference"), row.get("candidate_full"),
                row.get("candidate_swap"), row.get("noise_floor"),
                row.get("verdict[candidate_full]"))]) + " |")
            head = ["", "", "", ""]
        verdicts.append(f"- {result['dataset']}: {result['overall']['router']}; "
                        f"{result['overall']['pipeline']}")
    text = "\n".join(lines + [""] + verdicts) + "\n"
    (root / "optimizer_ablation_summary.md").write_text(text)
    print(text)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--collect"]:
        if len(argv) != 2:
            raise SystemExit("usage: compare_router_optimizers.py --collect <ablation root>")
        return collect_all(argv[1])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(METRICS))
    p.add_argument("--seed", default="1")
    p.add_argument("--layer", default="19")
    p.add_argument("--reference-router", required=True)
    p.add_argument("--reference-run", required=True)
    p.add_argument("--control-router", default=None)
    p.add_argument("--control-run", default=None)
    p.add_argument("--candidate-router", required=True)
    p.add_argument("--candidate-swap-run", default=None)
    p.add_argument("--candidate-run", default=None)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args(argv)

    routers = {"reference": a.reference_router, "control": a.control_router,
               "candidate": a.candidate_router}
    present = {k: v for k, v in routers.items()
               if v and (Path(v) / "fact_association_embeddings.pt").is_file()}
    result = {
        "schema_version": "router_optimizer_comparison_v1",
        "dataset": a.dataset, "seed": a.seed, "layer": a.layer,
        "inputs": {**{f"{k}_router": v for k, v in routers.items()},
                   "reference_run": a.reference_run, "control_run": a.control_run,
                   "candidate_swap_run": a.candidate_swap_run, "candidate_run": a.candidate_run},
        "fit": {k: fit_summary(v) if k in present else None for k, v in routers.items()},
        "parameters": {k: parameter_diff(a.reference_router, routers[k]) if k in present else None
                       for k in ("control", "candidate") if "reference" in present},
        "routes": {k: route_diff(a.reference_router, routers[k]) if k in present else None
                   for k in ("control", "candidate") if "reference" in present},
        "training_views": {
            "control_full": views_diff(a.reference_run, a.control_run) if a.control_run else None,
            "candidate_full": views_diff(a.reference_run, a.candidate_run) if a.candidate_run else None,
        },
    }
    runs = {"reference": a.reference_run, "control_full": a.control_run,
            "candidate_swap": a.candidate_swap_run, "candidate_full": a.candidate_run}
    result["runs"] = runs
    result["metrics"], result["run_status"] = metrics_table(a.dataset, runs)
    result["overall"] = overall(result["routes"], result["metrics"])

    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    text = markdown(result)
    prefix.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
