#!/usr/bin/env python3
"""Fix5k: matched target-local exact-name vs typed-target masking ablation (Seed 1).

Recognition-only. The target-local selector is frozen and identical across arms. Both
arms train a fresh linear relation head on the exact same semantic fit rows, use the
same frozen Llama mean-pooling encoder, the same optimizer/loss/hyperparameters, and
separately calibrate eta on the same policy calibration rows under the same route,
whole-query, per-family, and mixed-companion preservation budgets.

Representations:

  exact_name: [TARGET]Belgium[/TARGET]
  typed:      [TARGET_COUNTRY_OR_TERRITORY]

The coarse subject type is inferred by a deterministic subject-only frozen-Llama
probe. The probe receives only the subject string and a fixed type menu; it never
receives a query, relation ID, answer, case ID, split role, or forbidden-bank lookup.
No tokenizer vocabulary rows are added. Exact subject identity is therefore still
used to infer the coarse type, but it is removed from the relation-classifier input.

Typed collisions are not silently quarantined or relabeled. The matched rows remain
in the fit set even when distinct examples collapse to the same typed input; such
collisions are reported as part of the representation's information loss.

Output correction and quotient remain disabled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Import Fix5j so its explicit mixed-query bucket patch is installed, then reuse the
# audited Fix5i target-local selector/calibration/report helpers.
import mcf_target_local_recognition_baseline_fix5j_seed1 as fix5j
import mcf_target_relation_head_compare_fix5c_seed1 as cmp

local = fix5j.core
base = local.base
rep = local.rep
Row = local.Row
RoutingView = local.RoutingView
NONE = base.NONE
SEED = 1

TYPE_LABELS = [
    "PERSON",
    "ORGANIZATION",
    "COUNTRY_OR_TERRITORY",
    "PLACE_OR_FACILITY",
    "CREATIVE_WORK",
    "PRODUCT_OR_BRAND",
    "EVENT",
    "LANGUAGE",
    "OTHER",
]
TYPE_OPTIONS = {i + 1: label for i, label in enumerate(TYPE_LABELS)}


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def rows_from_dicts(items: Sequence[Mapping[str, Any]]) -> list[Row]:
    return [Row(**dict(x)) for x in items]


def subject_type_prompt(subject: str, tok: Any) -> str:
    """Fixed subject-only prompt. No query or relation information is accepted."""
    menu = "\n".join(f"{i}. {TYPE_OPTIONS[i]}" for i in sorted(TYPE_OPTIONS))
    user = (
        "Classify the named entity into exactly one broad type. Use only the entity "
        "name itself; no factual relation or query is provided. If uncertain, choose OTHER.\n\n"
        f"Entity name: {json.dumps(str(subject), ensure_ascii=False)}\n\n"
        f"Types:\n{menu}\n\n"
        "Answer with only the option number (1-9)."
    )
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return user + "\nAnswer:"


def option_token_ids(tok: Any) -> tuple[list[int], str]:
    """Find one consistent single-token spelling for answer digits 1..9."""
    for prefix in ("", " "):
        encoded = [tok(prefix + str(i), add_special_tokens=False)["input_ids"] for i in range(1, 10)]
        if all(len(ids) == 1 for ids in encoded):
            return [int(ids[0]) for ids in encoded], prefix
    raise RuntimeError(
        "Fix5k type probe requires digits 1..9 to have a common single-token spelling; "
        "neither bare nor leading-space digits satisfy this tokenizer"
    )


@torch.no_grad()
def infer_subject_types(
    model: Any,
    tok: Any,
    subjects: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Infer coarse types from subject strings only using next-token option logits."""
    option_ids, option_prefix = option_token_ids(tok)
    ordered = sorted({str(s) for s in subjects}, key=lambda x: x.casefold())
    out: dict[str, dict[str, Any]] = {}
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for st in range(0, len(ordered), int(batch_size)):
            chunk = ordered[st:st + int(batch_size)]
            prompts = [subject_type_prompt(s, tok) for s in chunk]
            enc = tok(
                prompts,
                padding=True,
                truncation=True,
                max_length=base.MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc, use_cache=False, return_dict=True).logits.float()
            last = enc["attention_mask"].sum(1).long() - 1
            row_ids = torch.arange(len(chunk), device=device)
            option_logits = logits[row_ids, last][:, torch.tensor(option_ids, device=device)]
            probs = torch.softmax(option_logits, dim=1)
            top2 = torch.topk(option_logits, k=2, dim=1)
            pred = option_logits.argmax(1)
            for j, subject in enumerate(chunk):
                k = int(pred[j].item())
                type_label = TYPE_LABELS[k]
                out[subject] = {
                    "type": type_label,
                    "option_number": k + 1,
                    "option_probability": float(probs[j, k].item()),
                    "option_logit_margin": float((top2.values[j, 0] - top2.values[j, 1]).item()),
                    "option_prefix": option_prefix,
                }
    finally:
        tok.padding_side = old_side
    return out


def typed_marker(type_label: str) -> str:
    if type_label not in TYPE_LABELS:
        raise ValueError(f"unknown type label: {type_label}")
    return f"[TARGET_{type_label}]"


def apply_typed_target(view: RoutingView, subject: str, type_label: str) -> RoutingView:
    """Remove the exact marked name while preserving selector status/support/offsets."""
    exact = f"[TARGET]{subject}[/TARGET]"
    marker = typed_marker(type_label)
    if exact not in view.selected_text:
        return RoutingView(
            selected_text=view.selected_text,
            selection_status=view.selection_status + "|TYPED_MARKER_MISSING",
            selected_character_offsets=view.selected_character_offsets,
            scope_supported=False,
            enumerated_subjects=view.enumerated_subjects,
        )
    return RoutingView(
        selected_text=view.selected_text.replace(exact, marker, 1),
        selection_status=view.selection_status,
        selected_character_offsets=view.selected_character_offsets,
        scope_supported=view.scope_supported,
        enumerated_subjects=view.enumerated_subjects,
    )


def exact_view(row: Row, bank_subjects: Sequence[str]) -> RoutingView:
    return local.routing_view(row.text, row.subject, bank_subjects)


def typed_view(
    row: Row,
    bank_subjects: Sequence[str],
    type_map: Mapping[str, Mapping[str, Any]],
) -> RoutingView:
    base_view = exact_view(row, bank_subjects)
    info = type_map.get(row.subject)
    if info is None:
        raise RuntimeError(f"subject missing from independent type map: {row.subject!r}")
    return apply_typed_target(base_view, row.subject, str(info["type"]))


def row_identity(row: Row) -> tuple[Any, ...]:
    return (
        norm_text(row.text).casefold(),
        row.subject.casefold(),
        row.relation,
        bool(row.forbidden),
        base.bucket(row.kind),
        bool(row.candidate),
        row.case_id,
        row.family,
    )


def identity_hash(rows: Sequence[Row]) -> str:
    payload = [row_identity(r) for r in rows]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def representation_conflicts(rows: Sequence[Row], views: Sequence[RoutingView]) -> dict[str, Any]:
    labels: dict[str, set[str]] = defaultdict(set)
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, view in zip(rows, views):
        key = norm_text(view.selected_text).casefold()
        labels[key].add(row.relation)
        counts[key] += 1
        if len(examples[key]) < 8:
            examples[key].append({
                "subject": row.subject,
                "relation": row.relation,
                "case_id": row.case_id,
                "family": row.family,
                "selected_text": view.selected_text,
            })
    conflict_keys = [k for k, rels in labels.items() if len(rels) > 1]
    duplicate_keys = [k for k, n in counts.items() if n > 1]
    return {
        "row_n": len(rows),
        "unique_text_n": len(labels),
        "duplicate_text_unique_n": len(duplicate_keys),
        "conflicting_text_unique_n": len(conflict_keys),
        "conflicting_row_n": sum(counts[k] for k in conflict_keys),
        "conflicts": [
            {
                "normalized_text": k,
                "relations": sorted(labels[k]),
                "occurrence_n": counts[k],
                "examples": examples[k],
            }
            for k in conflict_keys[:50]
        ],
    }


def feature_index(groups: Mapping[str, Sequence[RoutingView]]) -> tuple[list[str], dict[str, list[int]]]:
    texts: list[str] = []
    lookup: dict[str, int] = {}
    idx: dict[str, list[int]] = {}
    for name, views in groups.items():
        ids: list[int] = []
        for view in views:
            text = view.selected_text
            if text not in lookup:
                lookup[text] = len(texts)
                texts.append(text)
            ids.append(lookup[text])
        idx[name] = ids
    return texts, idx


def take(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return features[torch.tensor(list(indices), dtype=torch.long)]


def score_head(head: torch.nn.Module, features: torch.Tensor, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        return head(take(features, indices).to(device)).cpu()


def pilot_pass(
    semantic_validation: Mapping[str, Any],
    policy_validation: Mapping[str, Any],
    calibration: Mapping[str, Any],
    eps: float,
    eps_wrong: float,
    min_accept: float,
    min_val_acc: float,
) -> bool:
    fam_ok = all(
        (v["false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        for v in policy_validation["permitted_negative_families"].values()
    )
    qfam_ok = all(
        (v["false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        for v in policy_validation["whole_query"]["permitted_query_family"].values()
    )
    wq = policy_validation["whole_query"]
    return bool(
        calibration["status"] == "ACCEPTABLE_OPERATING_POINT"
        and (semantic_validation["accuracy_pct"] or 0.0) >= 100 * min_val_acc
        and (policy_validation["correct_forbidden_binding_accept_pct"] or 0.0) >= 100 * min_accept
        and (policy_validation["wrong_forbidden_binding_accept_pct"] or 0.0) <= 100 * eps_wrong + 1e-9
        and (policy_validation["permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and (policy_validation["candidate_present_permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and (wq["permitted_query_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and (wq["mixed_query_permitted_companion_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and fam_ok
        and qfam_ok
    )


def route_records(
    arm: str,
    split: str,
    group: str,
    rows: Sequence[Row],
    views: Sequence[RoutingView],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
    fit_texts: set[str],
    type_map: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    d = local.decision_tensors(rows, logits, views, eta, classes, none_idx, bank)
    out = []
    for i, (row, view) in enumerate(zip(rows, views)):
        info = type_map.get(row.subject, {})
        out.append({
            "arm": arm,
            "split": split,
            "group": group,
            "case_id": row.case_id,
            "original_query": row.text,
            "designated_subject": row.subject,
            "subject_type": info.get("type"),
            "subject_type_option_probability": info.get("option_probability"),
            "subject_type_option_logit_margin": info.get("option_logit_margin"),
            "selected_text": view.selected_text,
            "selection_status": view.selection_status,
            "scope_supported": view.scope_supported,
            "selected_character_offsets": list(view.selected_character_offsets) if view.selected_character_offsets else None,
            "expected_relation": row.relation,
            "forbidden": row.forbidden,
            "candidate_present": row.candidate,
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
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-batch-size", type=int, default=16)
    ap.add_argument("--type-batch-size", type=int, default=32)
    ap.add_argument("--train-steps", type=int, default=1600)
    ap.add_argument("--train-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--head-seed", type=int, default=1)
    ap.add_argument("--epsilon-retain", type=float, default=0.02)
    ap.add_argument("--epsilon-wrong", type=float, default=0.02)
    ap.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    ap.add_argument("--min-validation-relation-accuracy", type=float, default=0.70)
    ap.add_argument("--mixed-queries-per-phase", type=int, default=50)
    a = ap.parse_args()

    src = Path(a.fix5f_output_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(a.device)

    cache = torch.load(src / "target_representation_feature_cache.pt", map_location="cpu", weights_only=False)
    classes = list(cache["classes"])
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[NONE]
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

    mixed = {
        phase: local.build_mixed_queries(policy[phase], bank_subjects, a.mixed_queries_per_phase, phase)
        for phase in ("calib", "validation")
    }
    policy_eval = {
        "calib": policy["calib"] + mixed["calib"],
        "validation": policy["validation"] + mixed["validation"],
    }

    from mcf_sampling import sample_official_mcf_records
    import mcf_zero_unlearn_official_eval as off
    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, _ = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget]
    dr, pr = base.dev_rows(forget)
    dev = {"direct": dr, "paraphrase": pr}

    all_rows_for_types: list[Row] = []
    for phase in ("fit", "calib", "validation"):
        all_rows_for_types.extend(semantic[phase])
    all_rows_for_types.extend(policy_eval["calib"])
    all_rows_for_types.extend(policy_eval["validation"])
    all_rows_for_types.extend(dev["direct"])
    all_rows_for_types.extend(dev["paraphrase"])
    subjects_for_types = sorted({r.subject for r in all_rows_for_types}, key=lambda x: x.casefold())

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tokenizer_len_before = len(tok)
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

    type_map = infer_subject_types(model, tok, subjects_for_types, device, a.type_batch_size)
    if len(tok) != tokenizer_len_before:
        raise RuntimeError("tokenizer size changed; typed masking must not add vocabulary entries")
    (out / "subject_type_map_fix5k.json").write_text(
        json.dumps({
            "schema_version": 1,
            "type_labels": TYPE_LABELS,
            "probe_contract": {
                "subject_only": True,
                "query_visible": False,
                "relation_id_visible": False,
                "answer_visible": False,
                "case_id_visible": False,
                "forbidden_bank_visible": False,
                "model_frozen": True,
            },
            "subjects": type_map,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    type_distribution = Counter(str(v["type"]) for v in type_map.values())
    bank_type_distribution = Counter(str(type_map[s]["type"]) for s in bank_subjects)

    arm_view_fn: dict[str, Callable[[Row], RoutingView]] = {
        "exact_name_target_local": lambda row: exact_view(row, bank_subjects),
        "typed_target_local": lambda row: typed_view(row, bank_subjects, type_map),
    }

    # Prepare all views first so both arms use identical row manifests and only the
    # representation differs after the shared selector.
    arm_views: dict[str, dict[str, list[RoutingView]]] = {}
    for arm, fn in arm_view_fn.items():
        arm_views[arm] = {}
        for phase in ("fit", "calib", "validation"):
            arm_views[arm][f"semantic_{phase}"] = [fn(r) for r in semantic[phase]]
        for phase in ("calib", "validation"):
            arm_views[arm][f"policy_{phase}"] = [fn(r) for r in policy_eval[phase]]
        for group in ("direct", "paraphrase"):
            arm_views[arm][f"dev_{group}"] = [fn(r) for r in dev[group]]

    conflict_audit = {
        arm: {
            phase: representation_conflicts(semantic[phase], arm_views[arm][f"semantic_{phase}"])
            for phase in ("fit", "calib", "validation")
        }
        for arm in arm_view_fn
    }

    results: dict[str, Any] = {}
    training: dict[str, Any] = {}
    caches: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []

    for arm in ("exact_name_target_local", "typed_target_local"):
        groups = {
            "semantic_fit": arm_views[arm]["semantic_fit"],
            "semantic_calib": arm_views[arm]["semantic_calib"],
            "semantic_validation": arm_views[arm]["semantic_validation"],
            "policy_calib": arm_views[arm]["policy_calib"],
            "policy_validation": arm_views[arm]["policy_validation"],
            "dev_direct": arm_views[arm]["dev_direct"],
            "dev_paraphrase": arm_views[arm]["dev_paraphrase"],
        }
        texts, indices = feature_index(groups)
        features = base.encode(model, tok, texts, device, a.encode_batch_size)
        xfit = take(features, indices["semantic_fit"])
        yfit = torch.tensor([c2i[r.relation] for r in semantic["fit"]], dtype=torch.long)
        head, train_info = cmp.train_head(
            "linear",
            xfit,
            yfit,
            len(classes),
            device,
            a.train_steps,
            a.train_batch_size,
            a.lr,
            a.weight_decay,
            256,
            0.0,
            a.head_seed,
        )
        training[arm] = train_info
        torch.save(
            {
                "state_dict": head.state_dict(),
                "classes": classes,
                "training": train_info,
                "representation": arm,
            },
            out / f"{arm}_linear_head.pt",
        )

        sem_logits = {
            phase: score_head(head, features, indices[f"semantic_{phase}"], device)
            for phase in ("fit", "calib", "validation")
        }
        policy_logits = {
            phase: score_head(head, features, indices[f"policy_{phase}"], device)
            for phase in ("calib", "validation")
        }
        dev_logits = {
            group: score_head(head, features, indices[f"dev_{group}"], device)
            for group in ("direct", "paraphrase")
        }

        eta, cal = local.calibrate(
            policy_eval["calib"],
            policy_logits["calib"],
            arm_views[arm]["policy_calib"],
            classes,
            none_idx,
            bank,
            a.epsilon_retain,
            a.epsilon_wrong,
            a.min_calib_correct_accept,
        )
        sem_report = {
            phase: local.raw_semantic_report(semantic[phase], sem_logits[phase], classes)
            for phase in ("fit", "calib", "validation")
        }
        vp = local.policy_report(
            policy_eval["validation"],
            policy_logits["validation"],
            arm_views[arm]["policy_validation"],
            eta,
            classes,
            none_idx,
            bank,
        )
        dev_report = {}
        fit_texts = {
            norm_text(v.selected_text).casefold()
            for v in arm_views[arm]["semantic_fit"]
        }
        for group in ("direct", "paraphrase"):
            rows = dev[group]
            views = arm_views[arm][f"dev_{group}"]
            logits = dev_logits[group]
            dev_report[group] = {
                "semantic": local.raw_semantic_report(rows, logits, classes),
                "policy": local.policy_report(rows, logits, views, eta, classes, none_idx, bank),
                "selected_text_fit_overlap": local.exact_fit_overlap(
                    [v.selected_text for v in views], fit_texts
                ),
            }
            all_records.extend(route_records(
                arm, "development_only_seed1", group, rows, views, logits, eta,
                classes, none_idx, bank, fit_texts, type_map
            ))

        pilot = pilot_pass(
            sem_report["validation"], vp, cal,
            a.epsilon_retain, a.epsilon_wrong,
            a.min_calib_correct_accept, a.min_validation_relation_accuracy,
        )
        results[arm] = {
            "eta": eta,
            "calibration": cal,
            "semantic": sem_report,
            "validation_policy": vp,
            "development_only_official_seed1": dev_report,
            "pilot_pass": pilot,
            "fit_representation_conflicts": conflict_audit[arm]["fit"],
            "validation_selected_text_fit_overlap": local.exact_fit_overlap(
                [v.selected_text for v in arm_views[arm]["semantic_validation"]], fit_texts
            ),
        }
        all_records.extend(route_records(
            arm, "validation", "semantic", semantic["validation"],
            arm_views[arm]["semantic_validation"], sem_logits["validation"], eta,
            classes, none_idx, bank, fit_texts, type_map
        ))
        all_records.extend(route_records(
            arm, "validation", "policy_plus_mixed", policy_eval["validation"],
            arm_views[arm]["policy_validation"], policy_logits["validation"], eta,
            classes, none_idx, bank, fit_texts, type_map
        ))
        caches[arm] = {
            "features": features.cpu(),
            "texts": texts,
            "indices": indices,
        }

    torch.save(
        {
            "classes": classes,
            "arms": caches,
            "semantic_rows": {k: [asdict(r) for r in v] for k, v in semantic.items()},
            "policy_rows": {k: [asdict(r) for r in v] for k, v in policy_eval.items()},
            "dev_rows": {k: [asdict(r) for r in v] for k, v in dev.items()},
            "type_map": type_map,
        },
        out / "target_local_typed_masking_feature_cache_fix5k.pt",
    )

    exact = results["exact_name_target_local"]
    typed = results["typed_target_local"]
    comparison = {
        "validation_relation_accuracy_delta_typed_minus_exact":
            (typed["semantic"]["validation"]["accuracy_pct"] or 0.0)
            - (exact["semantic"]["validation"]["accuracy_pct"] or 0.0),
        "validation_correct_forbidden_accept_delta_typed_minus_exact":
            (typed["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0)
            - (exact["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0),
        "validation_route_permitted_fpr_pct": {
            arm: results[arm]["validation_policy"]["permitted_false_activation_pct"]
            for arm in results
        },
        "validation_query_permitted_fpr_pct": {
            arm: results[arm]["validation_policy"]["whole_query"]["permitted_query_false_activation_pct"]
            for arm in results
        },
        "validation_mixed_companion_fpr_pct": {
            arm: results[arm]["validation_policy"]["whole_query"]["mixed_query_permitted_companion_false_activation_pct"]
            for arm in results
        },
        "validation_mixed_joint_success_pct": {
            arm: results[arm]["validation_policy"]["whole_query"]["mixed_query_joint_success_pct"]
            for arm in results
        },
        "official_para_accuracy_pct": {
            arm: results[arm]["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"]
            for arm in results
        },
        "official_para_correct_forbidden_accept_pct": {
            arm: results[arm]["development_only_official_seed1"]["paraphrase"]["policy"]["correct_forbidden_binding_accept_pct"]
            for arm in results
        },
    }

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5k_target_local_exact_name_vs_subject_only_typed_masking",
        "recognition_only": True,
        "base_model_frozen": True,
        "linear_heads_freshly_trained_for_each_representation": True,
        "same_row_manifests": True,
        "semantic_identity_sha256": {
            phase: identity_hash(semantic[phase]) for phase in semantic
        },
        "policy_identity_sha256": {
            phase: identity_hash(policy_eval[phase]) for phase in policy_eval
        },
        "selector_contract": {
            "same_target_local_selector_both_arms": True,
            "subject_candidates_from_query_and_registered_bank": True,
            "gold_relation_used_for_selection": False,
            "gold_answer_used_for_selection": False,
            "unsupported_scope_ineligible_for_activation": True,
        },
        "type_probe_contract": {
            "labels": TYPE_LABELS,
            "subject_only": True,
            "query_visible": False,
            "relation_id_visible": False,
            "answer_visible": False,
            "case_id_visible": False,
            "forbidden_bank_visible": False,
            "model_frozen": True,
            "type_map_path": str(out / "subject_type_map_fix5k.json"),
            "subject_n": len(type_map),
            "type_distribution": dict(type_distribution),
            "registered_bank_subject_type_distribution": dict(bank_type_distribution),
            "note": (
                "Types are model-inferred from the name only; they are not gold entity types. "
                "This isolates removal of exact identity from the relation-classifier input, while retaining a subject-only type inference stage."
            ),
        },
        "representation_contract": {
            "exact_name": "[TARGET]original subject[/TARGET]",
            "typed": "[TARGET_<subject-only inferred coarse type>]",
            "other_registered_subjects": base.OTHER,
            "tokenizer_vocab_added": False,
            "tokenizer_length_before": tokenizer_len_before,
            "tokenizer_length_after": len(tok),
            "typed_collisions_quarantined": False,
            "typed_collisions_relabelled": False,
        },
        "conflict_audit": conflict_audit,
        "training": training,
        "results": results,
        "comparison": comparison,
        "pilot_criteria": {
            "validation_relation_accuracy_min_pct": 100 * a.min_validation_relation_accuracy,
            "correct_forbidden_accept_min_pct": 100 * a.min_calib_correct_accept,
            "wrong_forbidden_accept_max_pct": 100 * a.epsilon_wrong,
            "route_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "candidate_present_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_route_family_max_pct": 100 * a.epsilon_retain,
            "whole_query_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_whole_query_family_max_pct": 100 * a.epsilon_retain,
            "mixed_query_permitted_companion_false_activation_max_pct": 100 * a.epsilon_retain,
        },
        "output_correction_enabled": False,
        "quotient_enabled": False,
        "interpretation_guardrail": (
            "A typed-arm gain would show that a coarse subject-only type representation can improve this target-local recognition trade-off. It would not establish gold typing, end-to-end unlearning, generated leakage reduction, utility preservation, or novelty."
        ),
    }
    report = out / "target_local_typed_masking_ablation_fix5k.json"
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    records = out / "target_local_typed_masking_route_records_fix5k.jsonl"
    with records.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    compact = {}
    for arm in ("exact_name_target_local", "typed_target_local"):
        r = results[arm]
        vp = r["validation_policy"]
        compact[arm] = {
            "eta": r["eta"],
            "calibration_status": r["calibration"]["status"],
            "fit_relation_accuracy_pct": r["semantic"]["fit"]["accuracy_pct"],
            "validation_relation_accuracy_pct": r["semantic"]["validation"]["accuracy_pct"],
            "validation_correct_forbidden_accept_pct": vp["correct_forbidden_binding_accept_pct"],
            "validation_route_permitted_fpr_pct": vp["permitted_false_activation_pct"],
            "validation_query_permitted_fpr_pct": vp["whole_query"]["permitted_query_false_activation_pct"],
            "validation_mixed_correct_forbidden_pct": vp["whole_query"]["mixed_query_correct_forbidden_activation_pct"],
            "validation_mixed_companion_fpr_pct": vp["whole_query"]["mixed_query_permitted_companion_false_activation_pct"],
            "validation_mixed_joint_success_pct": vp["whole_query"]["mixed_query_joint_success_pct"],
            "unsupported_route_pct": vp["unsupported_route_pct"],
            "fit_conflicting_typed_or_exact_text_unique_n": r["fit_representation_conflicts"]["conflicting_text_unique_n"],
            "official_para_accuracy_pct": r["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"],
            "official_para_correct_forbidden_accept_pct": r["development_only_official_seed1"]["paraphrase"]["policy"]["correct_forbidden_binding_accept_pct"],
            "pilot_pass": r["pilot_pass"],
        }
    compact["type_probe"] = {
        "subject_n": len(type_map),
        "type_distribution": dict(type_distribution),
        "registered_bank_subject_type_distribution": dict(bank_type_distribution),
    }
    compact["comparison"] = comparison
    compact["report"] = str(report)
    compact["route_records"] = str(records)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
