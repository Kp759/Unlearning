#!/usr/bin/env python3
"""Fix5g: replay Fix5f linear heads under query-level preservation calibration.

No Llama loading and no training. This script reuses the saved frozen pooled features,
policy manifests, and trained linear heads from Fix5f. Each arm receives a separately
calibrated margin threshold chosen on calibration data only. Feasibility requires the
existing route-level budgets plus whole-query and per-query-family permitted false
activation budgets.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_relation_classifier_fix5_seed1 as base
import mcf_target_relation_head_compare_fix5c_seed1 as cmp

Row = base.Row


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def take(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return features[torch.tensor(list(indices), dtype=torch.long)]


def load_linear_head(path: Path, input_dim: int, classes: Sequence[str], device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved_classes = list(payload["classes"])
    if saved_classes != list(classes):
        raise RuntimeError(f"class ordering mismatch in {path}")
    head = base.Linear(input_dim, len(classes)).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


@torch.no_grad()
def logits_for(head: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return head(x.to(device)).cpu()


def norm_text(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def activation_vector(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    pred, dm = base.margin(logits)
    labels = [classes[int(i)] for i in pred]
    accepted = (pred != int(none_idx)) & (dm >= float(eta))
    binding = torch.tensor([(r.subject, labels[i]) in bank for i, r in enumerate(rows)], dtype=torch.bool)
    return accepted & binding, labels, dm


def query_report(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    act, labels, _ = activation_vector(rows, logits, eta, classes, none_idx, bank)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.forbidden:
            continue
        groups[(norm_text(row.text), base.bucket(row.kind))].append(i)

    def summarize(selected: Mapping[tuple[str, str], Sequence[int]]) -> dict[str, Any]:
        n = len(selected)
        if not n:
            return {"n": 0, "false_activation_n": 0, "false_activation_pct": None}
        errors = sum(bool(act[list(idx)].any().item()) for idx in selected.values())
        return {"n": n, "false_activation_n": errors, "false_activation_pct": 100.0 * errors / n}

    overall = summarize(groups)
    by_family = {
        fam: summarize({k: v for k, v in groups.items() if k[1] == fam})
        for fam in sorted({k[1] for k in groups})
    }
    multi = summarize({k: v for k, v in groups.items() if len(v) > 1})
    examples = []
    for (text, fam), idx in groups.items():
        if bool(act[list(idx)].any().item()) and len(examples) < 20:
            examples.append({
                "normalized_query": text,
                "family": fam,
                "route_count": len(idx),
                "subjects": [rows[i].subject for i in idx],
                "predicted_relations": [labels[i] for i in idx],
            })
    return {
        "overall": overall,
        "by_family": by_family,
        "multi_route": multi,
        "false_activation_examples": examples,
    }


def pct_to_rate(value: float | None) -> float:
    return 0.0 if value is None else float(value) / 100.0


def replay_calibrate(
    rows: Sequence[Row],
    logits: torch.Tensor,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
    eps: float,
    eps_wrong: float,
    min_correct_accept: float,
) -> tuple[float, dict[str, Any]]:
    _, dm = base.margin(logits)
    candidates = sorted(set([0.0] + [float(x) for x in dm.tolist()]))
    candidates.append(max(candidates, default=0.0) + 1.0)
    feasible: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    for eta in candidates:
        route = base.policy(rows, logits, eta, list(classes), none_idx, bank)
        query = query_report(rows, logits, eta, classes, none_idx, bank)
        wrong = pct_to_rate(route["wrong_forbidden_binding_accept_pct"])
        overall = pct_to_rate(route["permitted_false_activation_pct"])
        candidate_present = pct_to_rate(route["candidate_present_permitted_false_activation_pct"])
        route_families = {
            k: pct_to_rate(v["false_activation_pct"])
            for k, v in route["permitted_negative_families"].items()
        }
        query_overall = pct_to_rate(query["overall"]["false_activation_pct"])
        query_families = {
            k: pct_to_rate(v["false_activation_pct"])
            for k, v in query["by_family"].items()
        }
        if wrong > eps_wrong + 1e-12:
            continue
        if overall > eps + 1e-12 or candidate_present > eps + 1e-12:
            continue
        if any(v > eps + 1e-12 for v in route_families.values()):
            continue
        if query_overall > eps + 1e-12 or any(v > eps + 1e-12 for v in query_families.values()):
            continue
        correct = pct_to_rate(route["correct_forbidden_binding_accept_pct"])
        max_preservation = max(
            [overall, candidate_present, query_overall, wrong]
            + list(route_families.values())
            + list(query_families.values())
            + [0.0]
        )
        item = {
            "eta": float(eta),
            "correct_forbidden_binding_accept_rate": correct,
            "wrong_forbidden_binding_accept_rate": wrong,
            "route_permitted_false_activation_rate": overall,
            "candidate_present_permitted_false_activation_rate": candidate_present,
            "route_family_false_activation_rates": route_families,
            "query_permitted_false_activation_rate": query_overall,
            "query_family_false_activation_rates": query_families,
            "route_report": route,
            "query_report": query,
        }
        key = (-correct, max_preservation, query_overall, overall, wrong, -float(eta))
        feasible.append((key, item))

    if not feasible:
        raise RuntimeError("no threshold satisfies route + query-level preservation budgets")
    feasible.sort(key=lambda x: x[0])
    best = feasible[0][1]
    best["status"] = (
        "ACCEPTABLE_OPERATING_POINT"
        if best["correct_forbidden_binding_accept_rate"] >= min_correct_accept
        else "NO_ACCEPTABLE_OPERATING_POINT"
    )
    best["selection_rule"] = (
        "maximize calibration correct forbidden-binding acceptance subject to wrong, "
        "route overall/candidate/family, and whole-query overall/family preservation budgets"
    )
    best["feasible_threshold_count"] = len(feasible)
    return float(best["eta"]), best


def evaluate(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    return {
        "route": base.policy(rows, logits, eta, list(classes), none_idx, bank),
        "query": query_report(rows, logits, eta, classes, none_idx, bank),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5f-output-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epsilon-retain", type=float, default=0.02)
    ap.add_argument("--epsilon-wrong", type=float, default=0.02)
    ap.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    ap.add_argument("--min-validation-relation-accuracy", type=float, default=0.70)
    a = ap.parse_args()

    src = Path(a.fix5f_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cache = torch.load(src / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    summary = json.loads((src / "target_representation_compare_fix5e.json").read_text(encoding="utf-8"))
    classes = list(cache["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[base.NONE]
    device = torch.device(a.device)

    erased_policy_all = []
    for ph in ("fit", "calib", "validation"):
        erased_policy_all.extend(rows_from_dicts(cache["policy_rows"]["target_erased"][ph]))
    bank = {(r.subject, r.relation) for r in erased_policy_all if r.forbidden}
    if not bank:
        raise RuntimeError("could not reconstruct forbidden bank from saved policy manifest")

    results: dict[str, Any] = {}
    for arm in ("target_erased", "target_marked"):
        arm_cache = cache["arms"][arm]
        features = arm_cache["features"]
        indices = arm_cache["indices"]
        policy_calib = rows_from_dicts(cache["policy_rows"][arm]["calib"])
        policy_val = rows_from_dicts(cache["policy_rows"][arm]["validation"])
        semantic_val = rows_from_dicts(cache["semantic_rows"][arm]["validation"])

        head = load_linear_head(src / f"{arm}_linear_head.pt", features.shape[1], classes, device)
        calib_logits = logits_for(head, take(features, indices["policy_calib"]), device)
        val_policy_logits = logits_for(head, take(features, indices["policy_validation"]), device)
        val_sem_logits = logits_for(head, take(features, indices["semantic_validation"]), device)
        eta, calibration = replay_calibrate(
            policy_calib,
            calib_logits,
            classes,
            none_idx,
            bank,
            a.epsilon_retain,
            a.epsilon_wrong,
            a.min_calib_correct_accept,
        )
        validation = evaluate(policy_val, val_policy_logits, eta, classes, none_idx, bank)
        semantic = base.sem_report(semantic_val, val_sem_logits, c2i, classes)
        route = validation["route"]
        query = validation["query"]
        route_fam_ok = all(
            pct_to_rate(v["false_activation_pct"]) <= a.epsilon_retain + 1e-12
            for v in route["permitted_negative_families"].values()
        )
        query_fam_ok = all(
            pct_to_rate(v["false_activation_pct"]) <= a.epsilon_retain + 1e-12
            for v in query["by_family"].values()
        )
        pilot = (
            calibration["status"] == "ACCEPTABLE_OPERATING_POINT"
            and float(semantic["accuracy_pct"]) >= 100.0 * a.min_validation_relation_accuracy
            and pct_to_rate(route["correct_forbidden_binding_accept_pct"]) >= a.min_calib_correct_accept
            and pct_to_rate(route["wrong_forbidden_binding_accept_pct"]) <= a.epsilon_wrong + 1e-12
            and pct_to_rate(route["permitted_false_activation_pct"]) <= a.epsilon_retain + 1e-12
            and pct_to_rate(route["candidate_present_permitted_false_activation_pct"]) <= a.epsilon_retain + 1e-12
            and pct_to_rate(query["overall"]["false_activation_pct"]) <= a.epsilon_retain + 1e-12
            and route_fam_ok
            and query_fam_ok
        )
        results[arm] = {
            "eta": eta,
            "calibration": calibration,
            "validation_semantic": semantic,
            "validation_policy": validation,
            "pilot_pass": bool(pilot),
            "previous_eta": summary["results"][arm]["eta"],
        }

    comparison = {
        "validation_relation_accuracy_marked_minus_erased":
            results["target_marked"]["validation_semantic"]["accuracy_pct"]
            - results["target_erased"]["validation_semantic"]["accuracy_pct"],
        "validation_correct_forbidden_accept_marked_minus_erased":
            (results["target_marked"]["validation_policy"]["route"]["correct_forbidden_binding_accept_pct"] or 0.0)
            - (results["target_erased"]["validation_policy"]["route"]["correct_forbidden_binding_accept_pct"] or 0.0),
        "common_query_level_budget_pct": 100.0 * a.epsilon_retain,
    }
    report = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5g_query_level_calibration_replay",
        "source_fix5f_output_dir": str(src),
        "no_encoder_run": True,
        "no_training": True,
        "calibration_only_model_selection": True,
        "results": results,
        "comparison": comparison,
        "interpretation_guardrail": (
            "Validation is never used to choose eta. This replay asks whether the marked arm retains a recognition/acceptance advantage when both arms are calibrated under identical query-level preservation constraints."
        ),
    }
    path = out / "target_representation_query_calibration_fix5g.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    compact = {
        arm: {
            "previous_eta": results[arm]["previous_eta"],
            "replayed_eta": results[arm]["eta"],
            "calibration_status": results[arm]["calibration"]["status"],
            "validation_relation_accuracy_pct": results[arm]["validation_semantic"]["accuracy_pct"],
            "validation_correct_forbidden_accept_pct": results[arm]["validation_policy"]["route"]["correct_forbidden_binding_accept_pct"],
            "validation_route_permitted_fpr_pct": results[arm]["validation_policy"]["route"]["permitted_false_activation_pct"],
            "validation_query_permitted_fpr_pct": results[arm]["validation_policy"]["query"]["overall"]["false_activation_pct"],
            "validation_query_family_fpr_pct": {k: v["false_activation_pct"] for k, v in results[arm]["validation_policy"]["query"]["by_family"].items()},
            "pilot_pass": results[arm]["pilot_pass"],
        }
        for arm in results
    }
    compact["comparison"] = comparison
    compact["report"] = str(path)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
