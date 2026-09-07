#!/usr/bin/env python3
"""Fix5o: data-only augmentation of the exact-name target-local relation router (Seed 1).

Scientific contract
-------------------
This is recognition-only. It freezes the successful Fix5k architecture and changes
only the semantic fit coverage used to train a fresh linear head.

Matched arms:
  baseline_exact_name:
      original Fix5f/Fix5k semantic fit rows only.
  augmented_exact_name:
      the same original fit rows + fit-only relation formulations + matched
      same-subject/different-relation semantic contrasts.

Both arms keep fixed:
  * frozen Llama/tokenizer
  * exact-name target-local selector
  * mean pooling
  * linear classifier architecture
  * class inventory and NONE semantics
  * optimizer/loss/hyperparameters
  * policy calibration/validation manifests and preservation constraints
  * output correction disabled; quotient disabled

Each arm receives its own preservation-first calibrated eta. Official Seed-1 MCF
paraphrases are NEVER used for fit, augmentation construction, threshold selection,
or validation selection. They are reported only after each head and eta are frozen as
previously inspected development evidence.

The augmentation is intentionally reviewable and relation-grounded. It uses only
training-visible relation labels from mcf_relation_contracts_fix5.json and subjects
already present in the semantic fit manifest. No target_true/target_new answers or
official paraphrase/neighborhood/generation prompts are read to construct training
rows.

Fit-only formulation families and held-out augmentation probe families are disjoint.
Exact target-local text overlap between augmented fit and every calibration/validation
set is forbidden. Near-overlap (token Jaccard >= --max-heldout-jaccard) is dropped from
augmentation fit and reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_typed_masking_ablation_fix5k_seed1 as fix5k
import mcf_target_relation_head_compare_fix5c_seed1 as cmp

local = fix5k.local
base = fix5k.base
Row = fix5k.Row
RoutingView = fix5k.RoutingView
NONE = fix5k.NONE
SEED = 1

# New formulation families are split by construction. None is an official MCF field.
FIT_FORMULATIONS = (
    ("aug_relation_report", "For {subject}, state the {label}."),
    ("aug_relation_focus", "Regarding {label}, what applies to {subject}?"),
    ("aug_relation_identify", "Identify the {label} associated with {subject}."),
    ("aug_relation_value", "What should be reported as the {label} for {subject}?"),
)
HELDOUT_FORMULATIONS = (
    ("aug_holdout_relation_answer", "Which answer best gives the {label} for {subject}?"),
    ("aug_holdout_relation_request", "If asked for the {label} of {subject}, what would you give?"),
)


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", norm_text(text).casefold(), flags=re.UNICODE))


def jaccard(a: str, b: str) -> float:
    x, y = token_set(a), token_set(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def load_relation_labels(path: Path, modeled: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    relations = payload.get("relations", {})
    if not isinstance(relations, Mapping):
        raise RuntimeError("invalid relation contract catalog")
    missing = sorted(modeled - set(relations))
    if missing:
        raise RuntimeError(f"relation contracts missing modeled classes: {missing}")
    labels: dict[str, str] = {}
    for rid in sorted(modeled):
        label = str(relations[rid].get("label", "")).strip()
        meaning = str(relations[rid].get("meaning", "")).strip()
        if not label or not meaning:
            raise RuntimeError(f"relation contract {rid} lacks label/meaning")
        labels[rid] = label
    return labels, payload


def unique_fit_facts(rows: Sequence[Row], modeled: set[str]) -> list[Row]:
    """One training-visible owner row per (case, subject, relation)."""
    seen: set[tuple[Any, ...]] = set()
    out: list[Row] = []
    for r in rows:
        if r.relation not in modeled or r.relation == NONE:
            continue
        key = (r.case_id, r.subject.casefold(), r.relation)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    if not out:
        raise RuntimeError("no modeled semantic fit facts available for augmentation")
    return out


def render_row(subject: str, relation: str, case_id: int | None, family: str, template: str, label: str, *, kind: str) -> Row:
    text = template.format(subject=subject, label=label)
    return Row(
        text=norm_text(text),
        subject=str(subject),
        relation=str(relation),
        forbidden=False,
        kind=kind,
        family=family,
        case_id=case_id,
    )


def build_augmentation_rows(
    original_fit: Sequence[Row],
    modeled: set[str],
    labels: Mapping[str, str],
    *,
    hard_negatives_per_fact: int,
) -> tuple[list[Row], list[Row], dict[str, Any]]:
    facts = unique_fit_facts(original_fit, modeled)
    fit_rows: list[Row] = []
    heldout_rows: list[Row] = []
    relation_list = sorted(modeled)
    contrast_rows = 0

    for owner in facts:
        # Positive relation formulations for the owner's real relation.
        for family, template in FIT_FORMULATIONS:
            fit_rows.append(render_row(
                owner.subject, owner.relation, owner.case_id, family, template,
                labels[owner.relation], kind="fix5o_augmented_relation_fit",
            ))
        for family, template in HELDOUT_FORMULATIONS:
            heldout_rows.append(render_row(
                owner.subject, owner.relation, owner.case_id, family, template,
                labels[owner.relation], kind="fix5o_augmented_relation_heldout",
            ))

        # Matched semantic hard negatives: same subject, another modeled relation.
        # They retain the alternate semantic relation label; they are NOT NONE.
        alternatives = [r for r in relation_list if r != owner.relation]
        alternatives.sort(key=lambda rid: stable_int(
            f"fix5o:contrast:{owner.case_id}:{owner.subject}:{owner.relation}:{rid}"
        ))
        for j, rid in enumerate(alternatives[: int(hard_negatives_per_fact)]):
            family, template = FIT_FORMULATIONS[j % len(FIT_FORMULATIONS)]
            fit_rows.append(render_row(
                owner.subject, rid, owner.case_id,
                f"{family}_same_subject_contrast", template, labels[rid],
                kind="fix5o_same_subject_different_relation_fit",
            ))
            contrast_rows += 1

        # One unseen-family same-subject contrast for the held-out probe.
        if alternatives:
            rid = alternatives[-1]
            family, template = HELDOUT_FORMULATIONS[0]
            heldout_rows.append(render_row(
                owner.subject, rid, owner.case_id,
                f"{family}_same_subject_contrast", template, labels[rid],
                kind="fix5o_same_subject_different_relation_heldout",
            ))

    return fit_rows, heldout_rows, {
        "owner_fact_n": len(facts),
        "positive_fit_row_n": len(facts) * len(FIT_FORMULATIONS),
        "same_subject_different_relation_fit_row_n": contrast_rows,
        "heldout_probe_row_n_before_dedup": len(heldout_rows),
        "fit_formulation_families": [x[0] for x in FIT_FORMULATIONS],
        "heldout_formulation_families": [x[0] for x in HELDOUT_FORMULATIONS],
        "official_mcf_paraphrases_used": False,
        "answer_values_used": False,
    }


def target_local_views(rows: Sequence[Row], bank_subjects: Sequence[str]) -> list[RoutingView]:
    return [fix5k.exact_view(r, bank_subjects) for r in rows]


def dedup_rows_by_selected_text(
    rows: Sequence[Row],
    views: Sequence[RoutingView],
) -> tuple[list[Row], list[RoutingView], dict[str, Any]]:
    by_text: dict[str, tuple[Row, RoutingView]] = {}
    dup = 0
    for row, view in zip(rows, views):
        key = norm_text(view.selected_text).casefold()
        prev = by_text.get(key)
        if prev is not None:
            if prev[0].relation != row.relation:
                raise RuntimeError(
                    f"augmentation selected-text label conflict: {view.selected_text!r}: "
                    f"{prev[0].relation} vs {row.relation}"
                )
            dup += 1
            continue
        by_text[key] = (row, view)
    items = list(by_text.values())
    return [x[0] for x in items], [x[1] for x in items], {
        "duplicate_same_label_row_n": dup,
        "unique_selected_text_n": len(items),
    }


def filter_fit_against_heldout(
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    heldout_texts: Sequence[str],
    max_jaccard: float,
) -> tuple[list[Row], list[RoutingView], dict[str, Any]]:
    exact = {norm_text(x).casefold() for x in heldout_texts}
    kept_r: list[Row] = []
    kept_v: list[RoutingView] = []
    exact_drop = 0
    near_drop = 0
    max_seen = 0.0
    near_examples: list[dict[str, Any]] = []
    for row, view in zip(rows, views):
        text = norm_text(view.selected_text)
        if text.casefold() in exact:
            exact_drop += 1
            continue
        sims = [jaccard(text, h) for h in heldout_texts]
        m = max(sims, default=0.0)
        max_seen = max(max_seen, m)
        if m >= float(max_jaccard):
            near_drop += 1
            if len(near_examples) < 20:
                near_examples.append({
                    "text": text,
                    "relation": row.relation,
                    "max_jaccard": m,
                })
            continue
        kept_r.append(row)
        kept_v.append(view)
    return kept_r, kept_v, {
        "exact_overlap_dropped_n": exact_drop,
        "near_overlap_dropped_n": near_drop,
        "max_jaccard_seen": max_seen,
        "threshold": float(max_jaccard),
        "near_overlap_examples": near_examples,
    }


def feature_index(groups: Mapping[str, Sequence[RoutingView]]) -> tuple[list[str], dict[str, list[int]]]:
    return fix5k.feature_index(groups)


def take(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return fix5k.take(features, indices)


def score_head(head: torch.nn.Module, features: torch.Tensor, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    return fix5k.score_head(head, features, indices, device)


def semantic_report(rows: Sequence[Row], logits: torch.Tensor, classes: Sequence[str]) -> dict[str, Any]:
    return local.raw_semantic_report(rows, logits, classes)


def policy_report(rows: Sequence[Row], logits: torch.Tensor, views: Sequence[RoutingView], eta: float, classes: Sequence[str], none_idx: int, bank: set[tuple[str, str]]) -> dict[str, Any]:
    return local.policy_report(rows, logits, views, eta, classes, none_idx, bank)


def route_records(
    arm: str,
    group: str,
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
    fit_texts: set[str],
) -> list[dict[str, Any]]:
    d = local.decision_tensors(rows, logits, views, eta, classes, none_idx, bank)
    out: list[dict[str, Any]] = []
    for i, (row, view) in enumerate(zip(rows, views)):
        out.append({
            "arm": arm,
            "group": group,
            "case_id": row.case_id,
            "original_query": row.text,
            "designated_subject": row.subject,
            "selected_text": view.selected_text,
            "selection_status": view.selection_status,
            "scope_supported": view.scope_supported,
            "expected_relation": row.relation,
            "forbidden": row.forbidden,
            "predicted_relation": d["labels"][i],
            "margin": float(d["margin"][i]),
            "relation_correct": bool(d["correct"][i]),
            "accepted_relation": bool(d["accepted_relation"][i]),
            "forbidden_bank_lookup": bool(d["binding"][i]),
            "activates": bool(d["activates"][i]),
            "exact_selected_text_fit_overlap": norm_text(view.selected_text).casefold() in fit_texts,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fix5f-output-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--relation-contracts", default=str(SCRIPT_DIR / "mcf_relation_contracts_fix5.json"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--train-steps", type=int, default=1600)
    ap.add_argument("--train-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--head-seed", type=int, default=1)
    ap.add_argument("--epsilon-retain", type=float, default=0.02)
    ap.add_argument("--epsilon-wrong", type=float, default=0.02)
    ap.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    ap.add_argument("--min-validation-relation-accuracy", type=float, default=0.70)
    ap.add_argument("--hard-negatives-per-fact", type=int, default=2)
    ap.add_argument("--max-heldout-jaccard", type=float, default=0.90)
    ap.add_argument("--mixed-queries-per-phase", type=int, default=50)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    src = Path(a.fix5f_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)

    cache = torch.load(src / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    classes = list(cache["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[NONE]
    modeled = {c for c in classes if c != NONE}

    semantic = {
        phase: rows_from_dicts(cache["semantic_rows"]["target_marked"][phase])
        for phase in ("fit", "calib", "validation")
    }
    policy = {
        phase: rows_from_dicts(cache["policy_rows"]["target_marked"][phase])
        for phase in ("fit", "calib", "validation")
    }
    all_policy = policy["fit"] + policy["calib"] + policy["validation"]
    bank = {(r.subject, r.relation) for r in all_policy if r.forbidden}
    bank_subjects = sorted({s for s, _ in bank}, key=len, reverse=True)
    if not bank:
        raise RuntimeError("forbidden bank reconstructed from Fix5f manifest is empty")

    labels, contract_payload = load_relation_labels(Path(a.relation_contracts), modeled)
    aug_fit_raw, aug_holdout_raw, aug_build = build_augmentation_rows(
        semantic["fit"], modeled, labels,
        hard_negatives_per_fact=a.hard_negatives_per_fact,
    )

    # Policy evaluation is identical to Fix5k, including phase-local mixed queries.
    mixed = {
        phase: local.build_mixed_queries(policy[phase], bank_subjects, a.mixed_queries_per_phase, phase)
        for phase in ("calib", "validation")
    }
    policy_eval = {
        "calib": policy["calib"] + mixed["calib"],
        "validation": policy["validation"] + mixed["validation"],
    }

    # Official Seed-1 prompts are loaded only after augmentation has already been built.
    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, _ = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget]
    dev_direct, dev_para = base.dev_rows(forget)
    dev = {"direct": dev_direct, "paraphrase": dev_para}

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

    original_views = {
        f"semantic_{phase}": target_local_views(semantic[phase], bank_subjects)
        for phase in ("fit", "calib", "validation")
    }
    original_views.update({
        f"policy_{phase}": target_local_views(policy_eval[phase], bank_subjects)
        for phase in ("calib", "validation")
    })
    original_views.update({
        f"dev_{group}": target_local_views(dev[group], bank_subjects)
        for group in ("direct", "paraphrase")
    })

    aug_fit_views_raw = target_local_views(aug_fit_raw, bank_subjects)
    aug_holdout_views_raw = target_local_views(aug_holdout_raw, bank_subjects)
    aug_fit, aug_fit_views, aug_fit_dedup = dedup_rows_by_selected_text(aug_fit_raw, aug_fit_views_raw)
    aug_holdout, aug_holdout_views, aug_holdout_dedup = dedup_rows_by_selected_text(aug_holdout_raw, aug_holdout_views_raw)

    heldout_texts = [
        v.selected_text
        for key in ("semantic_calib", "semantic_validation", "policy_calib", "policy_validation")
        for v in original_views[key]
    ] + [v.selected_text for v in aug_holdout_views]
    aug_fit, aug_fit_views, leakage_filter = filter_fit_against_heldout(
        aug_fit, aug_fit_views, heldout_texts, a.max_heldout_jaccard
    )

    # Exact fit/heldout disjointness after the actual target-local preprocessing.
    fit_aug_texts = {norm_text(v.selected_text).casefold() for v in aug_fit_views}
    heldout_exact = {norm_text(x).casefold() for x in heldout_texts}
    if fit_aug_texts & heldout_exact:
        raise RuntimeError("Fix5o leakage guard failed: augmented fit overlaps held-out selected text")

    arm_fit_rows = {
        "baseline_exact_name": list(semantic["fit"]),
        "augmented_exact_name": list(semantic["fit"]) + list(aug_fit),
    }
    arm_fit_views = {
        "baseline_exact_name": list(original_views["semantic_fit"]),
        "augmented_exact_name": list(original_views["semantic_fit"]) + list(aug_fit_views),
    }

    # Encode a shared text bank so only training rows differ across arms.
    groups: dict[str, Sequence[RoutingView]] = {
        "semantic_fit_baseline": arm_fit_views["baseline_exact_name"],
        "semantic_fit_augmented": arm_fit_views["augmented_exact_name"],
        "semantic_calib": original_views["semantic_calib"],
        "semantic_validation": original_views["semantic_validation"],
        "policy_calib": original_views["policy_calib"],
        "policy_validation": original_views["policy_validation"],
        "aug_holdout": aug_holdout_views,
        "dev_direct": original_views["dev_direct"],
        "dev_paraphrase": original_views["dev_paraphrase"],
    }
    texts, indices = feature_index(groups)
    features = base.encode(model, tok, texts, device, a.encode_batch_size)

    results: dict[str, Any] = {}
    route_rows: list[dict[str, Any]] = []
    for arm in ("baseline_exact_name", "augmented_exact_name"):
        fit_key = "semantic_fit_baseline" if arm == "baseline_exact_name" else "semantic_fit_augmented"
        fit_rows = arm_fit_rows[arm]
        xfit = take(features, indices[fit_key])
        yfit = torch.tensor([c2i[r.relation] for r in fit_rows], dtype=torch.long)
        head, train_info = cmp.train_head(
            "linear", xfit, yfit, len(classes), device,
            a.train_steps, a.train_batch_size, a.lr, a.weight_decay,
            256, 0.0, a.head_seed,
        )
        torch.save({
            "state_dict": head.state_dict(),
            "classes": classes,
            "training": train_info,
            "representation": "exact_name_target_local",
            "fit_condition": arm,
            "augmentation_contract": "Fix5o data-only" if arm == "augmented_exact_name" else "original Fix5k fit only",
        }, out / f"{arm}_linear_head.pt")

        sem_logits = {
            phase: score_head(head, features, indices[f"semantic_{phase}"], device)
            for phase in ("calib", "validation")
        }
        policy_logits = {
            phase: score_head(head, features, indices[f"policy_{phase}"], device)
            for phase in ("calib", "validation")
        }
        holdout_logits = score_head(head, features, indices["aug_holdout"], device)
        dev_logits = {
            group: score_head(head, features, indices[f"dev_{group}"], device)
            for group in ("direct", "paraphrase")
        }

        eta, cal = local.calibrate(
            policy_eval["calib"], policy_logits["calib"], original_views["policy_calib"],
            classes, none_idx, bank,
            a.epsilon_retain, a.epsilon_wrong, a.min_calib_correct_accept,
        )
        sem_val = semantic_report(semantic["validation"], sem_logits["validation"], classes)
        sem_cal = semantic_report(semantic["calib"], sem_logits["calib"], classes)
        vp = policy_report(
            policy_eval["validation"], policy_logits["validation"], original_views["policy_validation"],
            eta, classes, none_idx, bank,
        )
        holdout_sem = semantic_report(aug_holdout, holdout_logits, classes)
        holdout_policy = policy_report(
            aug_holdout, holdout_logits, aug_holdout_views, eta, classes, none_idx, bank,
        )

        dev_report: dict[str, Any] = {}
        fit_texts = {norm_text(v.selected_text).casefold() for v in arm_fit_views[arm]}
        for group in ("direct", "paraphrase"):
            dev_report[group] = {
                "semantic": semantic_report(dev[group], dev_logits[group], classes),
                "policy": policy_report(
                    dev[group], dev_logits[group], original_views[f"dev_{group}"],
                    eta, classes, none_idx, bank,
                ),
                "selected_text_fit_overlap": local.exact_fit_overlap(
                    [v.selected_text for v in original_views[f"dev_{group}"]], fit_texts
                ),
            }
            route_rows.extend(route_records(
                arm, f"official_seed1_{group}_development_only",
                dev[group], original_views[f"dev_{group}"], dev_logits[group], eta,
                classes, none_idx, bank, fit_texts,
            ))

        pilot = fix5k.pilot_pass(
            sem_val, vp, cal,
            a.epsilon_retain, a.epsilon_wrong,
            a.min_calib_correct_accept, a.min_validation_relation_accuracy,
        )
        results[arm] = {
            "eta": eta,
            "training": train_info,
            "fit_row_n": len(fit_rows),
            "calibration": cal,
            "semantic_calibration": sem_cal,
            "semantic_validation": sem_val,
            "validation_policy": vp,
            "augmentation_heldout_probe": {
                "semantic": holdout_sem,
                "policy": holdout_policy,
                "development_status": "authored held-out formulation probe; not official MCF",
            },
            "development_only_official_seed1": dev_report,
            "pilot_pass_preservation_and_original_validation": bool(pilot),
        }

    baseline_para = results["baseline_exact_name"]["development_only_official_seed1"]["paraphrase"]
    augmented_para = results["augmented_exact_name"]["development_only_official_seed1"]["paraphrase"]

    report = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5o_data_only_augmented_exact_name_target_local_router",
        "recognition_only": True,
        "base_model_frozen": True,
        "router_architecture_changed": False,
        "exact_name_target_local_selector_changed": False,
        "mean_pooling_changed": False,
        "linear_head_architecture_changed": False,
        "output_correction_enabled": False,
        "quotient_enabled": False,
        "official_seed1_paraphrases_used_for_training": False,
        "official_seed1_paraphrases_used_for_calibration": False,
        "official_seed1_paraphrases_used_for_model_selection": False,
        "official_seed1_paraphrases_role": "previously inspected development evidence only",
        "relation_contracts": {
            "path": str(Path(a.relation_contracts).resolve()),
            "catalog_version": contract_payload.get("catalog_version"),
            "modeled_relation_n": len(modeled),
            "labels": labels,
        },
        "augmentation": {
            **aug_build,
            "fit_dedup": aug_fit_dedup,
            "heldout_dedup": aug_holdout_dedup,
            "fit_vs_heldout_leakage_filter": leakage_filter,
            "fit_row_n_after_guards": len(aug_fit),
            "heldout_probe_row_n_after_dedup": len(aug_holdout),
            "semantic_label_policy": "same-subject alternate-relation rows keep alternate relation label; never relabeled NONE",
        },
        "fixed_training_hyperparameters": {
            "steps": a.train_steps,
            "batch_size": a.train_batch_size,
            "lr": a.lr,
            "weight_decay": a.weight_decay,
            "head_seed": a.head_seed,
        },
        "preservation_constraints": {
            "epsilon_retain": a.epsilon_retain,
            "epsilon_wrong": a.epsilon_wrong,
            "min_calib_correct_accept": a.min_calib_correct_accept,
            "min_validation_relation_accuracy": a.min_validation_relation_accuracy,
        },
        "results": results,
        "development_comparison": {
            "baseline_official_paraphrase_relation_accuracy_pct": baseline_para["semantic"].get("accuracy_pct"),
            "augmented_official_paraphrase_relation_accuracy_pct": augmented_para["semantic"].get("accuracy_pct"),
            "baseline_official_paraphrase_correct_forbidden_accept_pct": baseline_para["policy"].get("correct_forbidden_binding_accept_pct"),
            "augmented_official_paraphrase_correct_forbidden_accept_pct": augmented_para["policy"].get("correct_forbidden_binding_accept_pct"),
            "target_correct_accept_ge_60pct_is_development_target_not_independent_gate": True,
        },
        "decision_rule": (
            "Do not continue to end-to-end integration unless the augmented arm passes the same original "
            "preservation/validation pilot, does not create held-out exact leakage, and improves unfamiliar-" 
            "wording recognition. Official Seed-1 paraphrases remain development evidence and must not be "
            "presented as a fresh independent test after repeated inspection."
        ),
    }

    (out / "mcf_target_local_augmented_relation_router_fix5o.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (out / "mcf_target_local_augmented_relation_router_records_fix5o.jsonl").open("w", encoding="utf-8") as f:
        for row in route_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "augmentation_training_rows_fix5o.json").write_text(
        json.dumps([asdict(r) for r in aug_fit], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out / "augmentation_heldout_rows_fix5o.json").write_text(
        json.dumps([asdict(r) for r in aug_holdout], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    compact = {
        "augmentation_fit_row_n": len(aug_fit),
        "augmentation_heldout_probe_row_n": len(aug_holdout),
        "baseline": {
            "eta": results["baseline_exact_name"]["eta"],
            "pilot_pass": results["baseline_exact_name"]["pilot_pass_preservation_and_original_validation"],
            "validation_relation_accuracy_pct": results["baseline_exact_name"]["semantic_validation"].get("accuracy_pct"),
            "official_para_relation_accuracy_pct": baseline_para["semantic"].get("accuracy_pct"),
            "official_para_correct_forbidden_accept_pct": baseline_para["policy"].get("correct_forbidden_binding_accept_pct"),
        },
        "augmented": {
            "eta": results["augmented_exact_name"]["eta"],
            "pilot_pass": results["augmented_exact_name"]["pilot_pass_preservation_and_original_validation"],
            "validation_relation_accuracy_pct": results["augmented_exact_name"]["semantic_validation"].get("accuracy_pct"),
            "heldout_relation_accuracy_pct": results["augmented_exact_name"]["augmentation_heldout_probe"]["semantic"].get("accuracy_pct"),
            "official_para_relation_accuracy_pct": augmented_para["semantic"].get("accuracy_pct"),
            "official_para_correct_forbidden_accept_pct": augmented_para["policy"].get("correct_forbidden_binding_accept_pct"),
        },
        "report": str(out / "mcf_target_local_augmented_relation_router_fix5o.json"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
