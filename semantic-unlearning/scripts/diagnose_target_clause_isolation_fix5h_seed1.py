#!/usr/bin/env python3
"""Fix5h: frozen-head target-clause isolation diagnostic for the Fix5f marked arm.

No training and no model-selection on validation. The saved target-marked linear head is
kept fixed. For every declared crossed-binding validation route, compare the cached
full-query prediction with a new frozen-Llama encoding of only the explicit clause
containing the designated subject. The threshold is read from Fix5g query-level
calibration replay.
"""
from __future__ import annotations

import argparse
import json
import re
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
import mcf_target_relation_classifier_fix5b_seed1 as fix5b
import mcf_target_representation_compare_fix5e_seed1 as rep

Row = base.Row


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def take(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return features[torch.tensor(list(indices), dtype=torch.long)]


def load_head(path: Path, input_dim: int, classes: Sequence[str], device: torch.device) -> torch.nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if list(payload["classes"]) != list(classes):
        raise RuntimeError("saved class ordering mismatch")
    head = base.Linear(input_dim, len(classes)).to(device)
    head.load_state_dict(payload["state_dict"])
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def split_first_second(text: str) -> tuple[str, str] | None:
    m = re.match(r"^\s*First\s*:\s*(.*?)\s*Second\s*:\s*(.*?)\s*$", str(text), flags=re.I | re.S)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    return (a, b) if a and b else None


def isolate_target_clause(text: str, subject: str, bank_subjects: Sequence[str]) -> tuple[str | None, str]:
    parts = split_first_second(text)
    if parts is None:
        return None, "not_first_second_format"
    hits = [bool(base.old.subject_regex(subject).search(clause)) for clause in parts]
    if sum(hits) != 1:
        return None, "target_not_unique_to_one_clause"
    clause = parts[hits.index(True)]
    marked, found = rep.target_preserving_text(clause, subject, bank_subjects)
    if not found:
        return None, "target_marking_failed"
    return marked, "ok"


def pred_margin(logits: torch.Tensor, classes: Sequence[str]) -> tuple[list[str], torch.Tensor]:
    pred, margin = base.margin(logits)
    return [classes[int(i)] for i in pred], margin


def policy_activation(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    pred, margin = base.margin(logits)
    labels = [classes[int(i)] for i in pred]
    accepted = (pred != int(none_idx)) & (margin >= float(eta))
    binding = torch.tensor([(r.subject, labels[i]) in bank for i, r in enumerate(rows)], dtype=torch.bool)
    return accepted & binding, labels, margin


def query_fpr(rows: Sequence[Row], activates: torch.Tensor) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[" ".join(row.text.split()).casefold()].append(i)
    n = len(groups)
    errors = sum(bool(activates[idx].any().item()) for idx in groups.values()) if n else 0
    return {
        "query_n": n,
        "false_activation_n": errors,
        "false_activation_pct": (100.0 * errors / n) if n else None,
    }


def accuracy(labels: Sequence[str], rows: Sequence[Row]) -> float | None:
    if not rows:
        return None
    return 100.0 * sum(p == r.relation for p, r in zip(labels, rows)) / len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5f-output-dir", required=True)
    ap.add_argument("--fix5g-report", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    a = ap.parse_args()

    src = Path(a.fix5f_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    replay = json.loads(Path(a.fix5g_report).read_text(encoding="utf-8"))
    eta = float(replay["results"]["target_marked"]["eta"])

    cache = torch.load(src / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    classes = list(cache["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[base.NONE]
    device = torch.device(a.device)

    marked_policy = rows_from_dicts(cache["policy_rows"]["target_marked"]["validation"])
    erased_all = []
    for ph in ("fit", "calib", "validation"):
        erased_all.extend(rows_from_dicts(cache["policy_rows"]["target_erased"][ph]))
    bank = {(r.subject, r.relation) for r in erased_all if r.forbidden}
    bank_subjects = sorted({s for s, _ in bank}, key=len, reverse=True)
    crossed_positions = [
        i for i, r in enumerate(marked_policy)
        if (not r.forbidden) and base.bucket(r.kind) == "crossed_binding"
    ]
    crossed_rows = [marked_policy[i] for i in crossed_positions]
    if not crossed_rows:
        raise RuntimeError("no crossed-binding validation routes in saved manifest")

    arm_cache = cache["arms"]["target_marked"]
    features = arm_cache["features"]
    policy_indices = arm_cache["indices"]["policy_validation"]
    head = load_head(src / "target_marked_linear_head.pt", features.shape[1], classes, device)
    with torch.no_grad():
        all_full_logits = head(take(features, policy_indices).to(device)).cpu()
    full_logits = all_full_logits[torch.tensor(crossed_positions, dtype=torch.long)]

    isolated_texts: list[str] = []
    supported_rows: list[Row] = []
    supported_full_logits: list[torch.Tensor] = []
    supported_original_indices: list[int] = []
    unsupported: list[dict[str, Any]] = []
    for local_i, row in enumerate(crossed_rows):
        text, status = isolate_target_clause(row.text, row.subject, bank_subjects)
        if text is None:
            unsupported.append({
                "route_index": local_i,
                "case_id": row.case_id,
                "subject": row.subject,
                "relation": row.relation,
                "status": status,
                "text": row.text,
            })
            continue
        isolated_texts.append(text)
        supported_rows.append(row)
        supported_full_logits.append(full_logits[local_i])
        supported_original_indices.append(local_i)

    if not supported_rows:
        raise RuntimeError("no crossed-binding routes could be isolated")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path,
        dtype=base.old.dtype_from_name(a.dtype),
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    isolated_features = base.encode(model, tok, isolated_texts, device, a.encode_batch_size)
    with torch.no_grad():
        isolated_logits = head(isolated_features.to(device)).cpu()
    full_supported_logits = torch.stack(supported_full_logits)

    full_labels, full_margin = pred_margin(full_supported_logits, classes)
    iso_labels, iso_margin = pred_margin(isolated_logits, classes)
    full_correct = [p == r.relation for p, r in zip(full_labels, supported_rows)]
    iso_correct = [p == r.relation for p, r in zip(iso_labels, supported_rows)]
    wrong_to_correct = sum((not a0) and b0 for a0, b0 in zip(full_correct, iso_correct))
    correct_to_wrong = sum(a0 and (not b0) for a0, b0 in zip(full_correct, iso_correct))

    full_act, _, _ = policy_activation(supported_rows, full_supported_logits, eta, classes, none_idx, bank)
    iso_act, _, _ = policy_activation(supported_rows, isolated_logits, eta, classes, none_idx, bank)
    examples = []
    for i, row in enumerate(supported_rows):
        if full_labels[i] != iso_labels[i] and len(examples) < 30:
            examples.append({
                "subject": row.subject,
                "expected_relation": row.relation,
                "full_prediction": full_labels[i],
                "isolated_prediction": iso_labels[i],
                "full_margin": float(full_margin[i]),
                "isolated_margin": float(iso_margin[i]),
                "full_correct": full_correct[i],
                "isolated_correct": iso_correct[i],
                "full_query": row.text,
                "isolated_input": isolated_texts[i],
            })

    supported_n = len(supported_rows)
    declared_n = len(crossed_rows)
    report = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5h_frozen_marked_head_target_clause_isolation",
        "source_fix5f_output_dir": str(src),
        "source_fix5g_report": str(Path(a.fix5g_report).resolve()),
        "recognition_only": True,
        "no_training": True,
        "head_frozen": True,
        "eta_from_calibration_replay": eta,
        "declared_crossed_binding_route_n": declared_n,
        "supported_route_n": supported_n,
        "unsupported_route_n": len(unsupported),
        "coverage_pct": 100.0 * supported_n / declared_n,
        "unsupported": unsupported,
        "supported_metrics": {
            "full_query_relation_accuracy_pct": accuracy(full_labels, supported_rows),
            "isolated_clause_relation_accuracy_pct": accuracy(iso_labels, supported_rows),
            "accuracy_delta_isolated_minus_full": (
                (accuracy(iso_labels, supported_rows) or 0.0) - (accuracy(full_labels, supported_rows) or 0.0)
            ),
            "wrong_to_correct_n": wrong_to_correct,
            "correct_to_wrong_n": correct_to_wrong,
            "full_query_route_false_activation_pct": 100.0 * float(full_act.float().mean()),
            "isolated_clause_route_false_activation_pct": 100.0 * float(iso_act.float().mean()),
            "full_query_whole_query": query_fpr(supported_rows, full_act),
            "isolated_clause_whole_query_grouped_by_original_query": query_fpr(supported_rows, iso_act),
            "mean_full_margin": float(full_margin.mean()),
            "mean_isolated_margin": float(iso_margin.mean()),
        },
        "conservative_all_declared_accuracy_pct": {
            "full_query": 100.0 * sum(full_correct) / declared_n,
            "isolated_clause_with_unsupported_counted_incorrect": 100.0 * sum(iso_correct) / declared_n,
        },
        "changed_prediction_examples": examples,
        "interpretation_guardrail": (
            "This is a synthetic explicit-clause diagnostic. Improvement supports non-target-clause interference as a contributor; it is not ordinary router performance or a deployed span selector."
        ),
    }
    path = out / "target_clause_isolation_fix5h.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "eta": eta,
        "declared_crossed_binding_route_n": declared_n,
        "supported_route_n": supported_n,
        "coverage_pct": report["coverage_pct"],
        **report["supported_metrics"],
        "report": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
