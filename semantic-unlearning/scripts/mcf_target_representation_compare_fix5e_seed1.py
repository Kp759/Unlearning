#!/usr/bin/env python3
"""Fix5e: matched target-erasure vs target-preserving-marking relation probe.

Recognition-only. Both arms use the same Seed-1 split assignments, the same matched
semantic examples and policy instances, the same frozen Llama mean-pooling encoder,
the same linear classifier architecture/loss/optimizer settings, and the same
calibration budgets. The only intended representation change is:

  erased: TARGET_ENTITY
  marked: [TARGET]original subject[/TARGET]

Other registered subjects are still replaced by OTHER_ENTITY. No tokenizer
vocabulary entries are added. Conflict-quarantined masked inputs are never used for
fit/calibration/model selection; they are reported separately as a coverage challenge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_relation_classifier_fix5_seed1 as base
import mcf_target_relation_classifier_fix5b_seed1 as fix5b
import mcf_target_relation_head_compare_fix5c_seed1 as cmp

Row = base.Row
SEED = 1
NONE = base.NONE
ERASED_TARGET = base.TARGET
OTHER = base.OTHER
TARGET_OPEN = "[TARGET]"
TARGET_CLOSE = "[/TARGET]"


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


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


def semantic_identity(row: Row) -> tuple[Any, ...]:
    return (
        norm_text(row.text).casefold(),
        row.subject.casefold(),
        row.relation,
        row.case_id,
        row.family,
        base.bucket(row.kind),
    )


def identity_hash(rows: Sequence[Row], semantic: bool) -> str:
    fn = semantic_identity if semantic else row_identity
    payload = [fn(r) for r in rows]
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def target_preserving_text(text: str, subject: str, bank_subjects: Sequence[str]) -> tuple[str, bool]:
    pattern = base.old.subject_regex(subject)
    if not pattern.search(text):
        return norm_text(text), False
    out = pattern.sub(f"{TARGET_OPEN}{subject}{TARGET_CLOSE}", text, count=1)
    for other in sorted(
        {s for s in bank_subjects if s.casefold() != subject.casefold()},
        key=len,
        reverse=True,
    ):
        out = base.old.subject_regex(other).sub(OTHER, out)
    return norm_text(out), True


def marked_row(row: Row, bank_subjects: Sequence[str]) -> Row:
    text, found = target_preserving_text(row.text, row.subject, bank_subjects)
    if not found:
        raise RuntimeError(f"target subject missing for marked arm: {row.subject!r} in {row.text!r}")
    return replace(row, masked=text)


def mark_rows(rows: Sequence[Row], bank_subjects: Sequence[str]) -> list[Row]:
    return [marked_row(r, bank_subjects) for r in rows]


def conflict_keys(parts: Mapping[str, Sequence[Row]]) -> dict[str, list[str]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for rows in parts.values():
        for row in rows:
            labels[row.masked.casefold()].add(row.relation)
    return {k: sorted(v) for k, v in labels.items() if len(v) > 1}


def coverage_rows(parts: Mapping[str, Sequence[Row]], conflicts: Mapping[str, Sequence[str]]) -> dict[str, list[Row]]:
    keys = set(conflicts)
    return {
        phase: cmp.policy_manifest([r for r in rows if r.masked.casefold() in keys])
        for phase, rows in parts.items()
    }


def feature_index(rows_by_name: Mapping[str, Sequence[Row]]) -> tuple[list[str], dict[str, list[int]]]:
    texts: list[str] = []
    lookup: dict[str, int] = {}
    indices: dict[str, list[int]] = {}
    for name, rows in rows_by_name.items():
        idx: list[int] = []
        for row in rows:
            key = row.masked
            if key not in lookup:
                lookup[key] = len(texts)
                texts.append(key)
            idx.append(lookup[key])
        indices[name] = idx
    return texts, indices


def take(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    return features[torch.tensor(list(indices), dtype=torch.long)]


def contains_subsequence(seq: Sequence[int], sub: Sequence[int]) -> bool:
    if not sub:
        return True
    n = len(sub)
    return any(list(seq[i:i+n]) == list(sub) for i in range(0, len(seq) - n + 1))


@torch.no_grad()
def encode_texts(
    model: Any,
    tok: Any,
    texts: Sequence[str],
    device: torch.device,
    batch: int,
    required_markers: Sequence[str],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("requires model.model")
    marker_ids = {
        marker: tok(marker, add_special_tokens=False)["input_ids"]
        for marker in required_markers
    }
    full_lengths = [
        len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts
    ]
    chunks: list[torch.Tensor] = []
    diagnostics: list[dict[str, Any]] = []
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for st in range(0, len(texts), int(batch)):
            bt = list(texts[st:st + int(batch)])
            enc = tok(
                bt,
                padding=True,
                truncation=True,
                max_length=base.MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            h = backbone(**enc, use_cache=False, return_dict=True).last_hidden_state.float()
            m = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            chunks.append(((h * m).sum(1) / m.sum(1).clamp_min(1)).cpu())
            ids_cpu = enc["input_ids"].cpu()
            mask_cpu = enc["attention_mask"].cpu()
            for j in range(len(bt)):
                kept = ids_cpu[j][mask_cpu[j].bool()].tolist()
                pos = st + j
                vis = {
                    marker: contains_subsequence(kept, ids)
                    for marker, ids in marker_ids.items()
                }
                diagnostics.append({
                    "text_index": pos,
                    "full_token_count": int(full_lengths[pos]),
                    "kept_token_count": len(kept),
                    "truncated": bool(full_lengths[pos] > base.MAX_LENGTH),
                    "required_marker_visibility": vis,
                    "all_required_markers_visible_after_truncation": all(vis.values()),
                })
    finally:
        tok.padding_side = old_side
    return torch.cat(chunks, dim=0), diagnostics


def whole_query_report(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    if not rows:
        return {
            "permitted_query_n": 0,
            "permitted_query_false_activation_n": 0,
            "permitted_query_false_activation_pct": None,
            "multi_route_permitted_query_n": 0,
            "multi_route_permitted_query_false_activation_n": 0,
            "multi_route_permitted_query_false_activation_pct": None,
            "false_activation_examples": [],
        }
    pred, dm = base.margin(logits)
    labels = [classes[int(i)] for i in pred]
    accepted = (pred != int(none_idx)) & (dm >= float(eta))
    binding = torch.tensor([(row.subject, labels[i]) in bank for i, row in enumerate(rows)], dtype=torch.bool)
    activates = accepted & binding

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.forbidden:
            continue
        groups[(norm_text(row.text).casefold(), base.bucket(row.kind))].append(i)

    def summarize(groups_in: Mapping[tuple[str, str], Sequence[int]]) -> tuple[int, int, float | None]:
        if not groups_in:
            return 0, 0, None
        errors = sum(bool(activates[list(idx)].any().item()) for idx in groups_in.values())
        return len(groups_in), errors, 100.0 * errors / len(groups_in)

    n, e, rate = summarize(groups)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    mn, me, mrate = summarize(multi)
    examples = []
    for (text, fam), idx in groups.items():
        if bool(activates[idx].any().item()) and len(examples) < 20:
            examples.append({
                "normalized_query": text,
                "negative_family": fam,
                "route_count": len(idx),
                "subjects": [rows[i].subject for i in idx],
                "predicted_relations": [labels[i] for i in idx],
            })
    return {
        "permitted_query_n": n,
        "permitted_query_false_activation_n": e,
        "permitted_query_false_activation_pct": rate,
        "multi_route_permitted_query_n": mn,
        "multi_route_permitted_query_false_activation_n": me,
        "multi_route_permitted_query_false_activation_pct": mrate,
        "false_activation_examples": examples,
    }


def policy_report(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    report = base.policy(rows, logits, eta, list(classes), none_idx, bank)
    report["whole_query"] = whole_query_report(rows, logits, eta, classes, none_idx, bank)
    return report


def score_head(head: torch.nn.Module, features: torch.Tensor, indices: Sequence[int], device: torch.device) -> torch.Tensor:
    head.eval()
    with torch.no_grad():
        return head(take(features, indices).to(device)).cpu()


def semantic_report(rows: Sequence[Row], logits: torch.Tensor, c2i: Mapping[str, int], classes: Sequence[str]) -> dict[str, Any]:
    return base.sem_report(rows, logits, c2i, list(classes))


def evaluate_arm(
    head: torch.nn.Module,
    features: torch.Tensor,
    indices: Mapping[str, Sequence[int]],
    semantic: Mapping[str, Sequence[Row]],
    policy: Mapping[str, Sequence[Row]],
    dev: Mapping[str, Sequence[Row]],
    challenge: Mapping[str, Sequence[Row]],
    classes: Sequence[str],
    c2i: Mapping[str, int],
    none_idx: int,
    bank: set[tuple[str, str]],
    device: torch.device,
    eps: float,
    eps_wrong: float,
    min_calib_accept: float,
    min_val_acc: float,
) -> dict[str, Any]:
    logits = {
        name: score_head(head, features, idx, device)
        for name, idx in indices.items()
    }
    eta, cal = base.calibrate(
        policy["calib"],
        logits["policy_calib"],
        list(classes),
        none_idx,
        bank,
        eps,
        eps_wrong,
        min_calib_accept,
    )
    sem = {
        phase: semantic_report(semantic[phase], logits[f"semantic_{phase}"], c2i, classes)
        for phase in ("fit", "calib", "validation")
    }
    val_policy = policy_report(
        policy["validation"],
        logits["policy_validation"],
        eta,
        classes,
        none_idx,
        bank,
    )
    fam_ok = all(
        (v["false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        for v in val_policy["permitted_negative_families"].values()
    )
    wq = val_policy["whole_query"]["permitted_query_false_activation_pct"]
    pilot = (
        cal["status"] == "ACCEPTABLE_OPERATING_POINT"
        and sem["validation"]["accuracy_pct"] >= 100 * min_val_acc
        and (val_policy["correct_forbidden_binding_accept_pct"] or 0.0) >= 100 * min_calib_accept
        and (val_policy["wrong_forbidden_binding_accept_pct"] or 0.0) <= 100 * eps_wrong + 1e-9
        and (val_policy["permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and (val_policy["candidate_present_permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and fam_ok
        and (wq is None or wq <= 100 * eps + 1e-9)
    )

    dev_report = {}
    for name in ("direct", "paraphrase"):
        lg = logits[f"dev_{name}"]
        dev_report[name] = {
            "semantic": semantic_report(dev[name], lg, c2i, classes),
            "policy": policy_report(dev[name], lg, eta, classes, none_idx, bank),
        }

    challenge_report = {}
    for phase in ("fit", "calib", "validation"):
        rows = challenge[phase]
        lg = logits[f"challenge_{phase}"]
        challenge_report[phase] = {
            "n": len(rows),
            "semantic": semantic_report(rows, lg, c2i, classes) if rows else None,
            "policy": policy_report(rows, lg, eta, classes, none_idx, bank) if rows else None,
            "used_for_fit": False,
            "used_for_calibration": False,
            "used_for_model_selection": False,
        }

    return {
        "calibration": cal,
        "eta": float(eta),
        "semantic": sem,
        "validation_policy": val_policy,
        "pilot_pass": bool(pilot),
        "development_only_official_seed1": dev_report,
        "ambiguity_coverage_challenge": challenge_report,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--view-corpus-fix5", required=True)
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
    a = ap.parse_args()

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device(a.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import mcf_zero_unlearn_official_eval as off
    from mcf_sampling import sample_official_mcf_records

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget, retain = sample_official_mcf_records(data, 50, 1000, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget]
    retain = [off.normalize_record(x) for x in retain]

    facts, split, corpus = base.old.load_v2(Path(a.view_corpus_fix5))
    base.old.align_facts_to_forget(facts, forget)
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    subjects = sorted({s for s, _ in bank}, key=len, reverse=True)
    relations = sorted({r for _, r in bank})
    classes = relations + [NONE]
    c2i = {c: i for i, c in enumerate(classes)}
    none_idx = c2i[NONE]

    rf, rc, rv = base.old.split_retain(retain, bank)
    raw = {
        "fit": base.rows_for_phase(facts, split, rf, forget, "fit"),
        "calib": base.rows_for_phase(facts, split, rc, forget, "calib"),
        "validation": base.rows_for_phase(facts, split, rv, forget, "validation"),
    }

    erased_prepared = {k: base.prep(v, subjects) for k, v in raw.items()}
    conflicts = conflict_keys(erased_prepared)
    challenge_erased = coverage_rows(erased_prepared, conflicts)

    matched_erased, separation = fix5b.separate(erased_prepared)
    semantic_erased = {k: cmp.semantic_unique(v) for k, v in matched_erased.items()}
    policy_erased = {k: cmp.policy_manifest(v) for k, v in matched_erased.items()}

    # Critical matched-ablation rule: select rows once using the erased arm, then
    # transform those exact rows for the marked arm. Never rededuplicate marked text.
    semantic_marked = {k: mark_rows(v, subjects) for k, v in semantic_erased.items()}
    policy_marked = {k: mark_rows(v, subjects) for k, v in policy_erased.items()}
    challenge_marked = {k: mark_rows(v, subjects) for k, v in challenge_erased.items()}

    for phase in ("fit", "calib", "validation"):
        if [semantic_identity(r) for r in semantic_erased[phase]] != [semantic_identity(r) for r in semantic_marked[phase]]:
            raise RuntimeError(f"semantic matched identities changed in {phase}")
        if [row_identity(r) for r in policy_erased[phase]] != [row_identity(r) for r in policy_marked[phase]]:
            raise RuntimeError(f"policy matched identities changed in {phase}")

    missing = sorted(set(classes) - {r.relation for r in semantic_erased["fit"]})
    if missing:
        raise RuntimeError(f"fit classes missing after matched selection: {missing}")

    direct_raw, para_raw = base.dev_rows(forget)
    dev_erased = {
        "direct": cmp.policy_manifest(base.prep(direct_raw, subjects)),
        "paraphrase": cmp.policy_manifest(base.prep(para_raw, subjects)),
    }
    dev_marked = {k: mark_rows(v, subjects) for k, v in dev_erased.items()}

    arms = {
        "target_erased": {
            "semantic": semantic_erased,
            "policy": policy_erased,
            "dev": dev_erased,
            "challenge": challenge_erased,
            "required_markers": [ERASED_TARGET],
        },
        "target_marked": {
            "semantic": semantic_marked,
            "policy": policy_marked,
            "dev": dev_marked,
            "challenge": challenge_marked,
            "required_markers": [TARGET_OPEN, TARGET_CLOSE],
        },
    }

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
    tokenizer_len_after = len(tok)
    if tokenizer_len_before != tokenizer_len_after:
        raise RuntimeError("tokenizer size changed; this ablation must not add vocabulary entries")

    arm_cache: dict[str, Any] = {}
    arm_results: dict[str, Any] = {}
    arm_training: dict[str, Any] = {}

    for arm_name, arm in arms.items():
        rows_for_cache = {
            "semantic_fit": arm["semantic"]["fit"],
            "semantic_calib": arm["semantic"]["calib"],
            "semantic_validation": arm["semantic"]["validation"],
            "policy_calib": arm["policy"]["calib"],
            "policy_validation": arm["policy"]["validation"],
            "dev_direct": arm["dev"]["direct"],
            "dev_paraphrase": arm["dev"]["paraphrase"],
            "challenge_fit": arm["challenge"]["fit"],
            "challenge_calib": arm["challenge"]["calib"],
            "challenge_validation": arm["challenge"]["validation"],
        }
        texts, indices = feature_index(rows_for_cache)
        features, token_diag = encode_texts(
            model,
            tok,
            texts,
            device,
            a.encode_batch_size,
            arm["required_markers"],
        )
        if any(not d["all_required_markers_visible_after_truncation"] for d in token_diag):
            bad = [d for d in token_diag if not d["all_required_markers_visible_after_truncation"]]
            raise RuntimeError(f"{arm_name}: required marker lost after truncation in {len(bad)} cached texts")

        xfit = take(features, indices["semantic_fit"])
        yfit = torch.tensor([c2i[r.relation] for r in arm["semantic"]["fit"]], dtype=torch.long)
        head, training = cmp.train_head(
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
        result = evaluate_arm(
            head,
            features,
            indices,
            arm["semantic"],
            arm["policy"],
            arm["dev"],
            arm["challenge"],
            classes,
            c2i,
            none_idx,
            bank,
            device,
            a.epsilon_retain,
            a.epsilon_wrong,
            a.min_calib_correct_accept,
            a.min_validation_relation_accuracy,
        )
        torch.save(
            {"state_dict": head.state_dict(), "classes": classes, "training": training},
            out / f"{arm_name}_linear_head.pt",
        )
        arm_training[arm_name] = training
        arm_results[arm_name] = result
        arm_cache[arm_name] = {
            "features": features,
            "texts": texts,
            "indices": indices,
            "token_diagnostics": token_diag,
        }

    torch.save(
        {
            "arms": arm_cache,
            "classes": classes,
            "semantic_rows": {
                arm: {k: [asdict(r) for r in cfg["semantic"][k]] for k in ("fit", "calib", "validation")}
                for arm, cfg in arms.items()
            },
            "policy_rows": {
                arm: {k: [asdict(r) for r in cfg["policy"][k]] for k in ("fit", "calib", "validation")}
                for arm, cfg in arms.items()
            },
            "challenge_rows": {
                arm: {k: [asdict(r) for r in cfg["challenge"][k]] for k in ("fit", "calib", "validation")}
                for arm, cfg in arms.items()
            },
        },
        out / "target_representation_feature_cache.pt",
    )

    encoder_fingerprint = cmp.json_hash({
        "model_config": model.config.to_dict(),
        "tokenizer_class": tok.__class__.__name__,
        "special_tokens": tok.special_tokens_map,
        "tokenizer_length": len(tok),
        "max_length": base.MAX_LENGTH,
        "other_marker": OTHER,
        "target_erased_marker": ERASED_TARGET,
        "target_markers": [TARGET_OPEN, TARGET_CLOSE],
        "dtype": a.dtype,
    })

    comparison = {
        "validation_relation_accuracy_delta_marked_minus_erased":
            arm_results["target_marked"]["semantic"]["validation"]["accuracy_pct"]
            - arm_results["target_erased"]["semantic"]["validation"]["accuracy_pct"],
        "correct_forbidden_accept_delta_marked_minus_erased":
            (arm_results["target_marked"]["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0)
            - (arm_results["target_erased"]["validation_policy"]["correct_forbidden_binding_accept_pct"] or 0.0),
        "official_para_accuracy_delta_marked_minus_erased":
            arm_results["target_marked"]["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"]
            - arm_results["target_erased"]["development_only_official_seed1"]["paraphrase"]["semantic"]["accuracy_pct"],
        "same_subject_different_relation_fpr": {
            arm: arm_results[arm]["validation_policy"]["permitted_negative_families"]
                .get("same_subject_different_relation", {}).get("false_activation_pct")
            for arm in arms
        },
        "crossed_binding_fpr": {
            arm: arm_results[arm]["validation_policy"]["permitted_negative_families"]
                .get("crossed_binding", {}).get("false_activation_pct")
            for arm in arms
        },
    }

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5e_target_erased_vs_target_marked_linear_relation_probe",
        "recognition_only": True,
        "comparison_question":
            "Does preserving and marking the target subject improve held-out relation recognition over subject erasure without increasing false activation on permitted same-subject relations?",
        "arms": {
            "target_erased": "replace target subject with TARGET_ENTITY",
            "target_marked": "keep original subject and wrap it with [TARGET]...[/TARGET]",
        },
        "matched_contract": {
            "row_selection_source": "target_erased arm only; target_marked transforms those exact selected rows",
            "same_original_example_ids_and_split_assignments": True,
            "marked_arm_rededuplicated": False,
            "semantic_counts": {k: len(v) for k, v in semantic_erased.items()},
            "policy_counts": {k: len(v) for k, v in policy_erased.items()},
            "semantic_identity_sha256": {
                k: identity_hash(v, semantic=True) for k, v in semantic_erased.items()
            },
            "policy_identity_sha256": {
                k: identity_hash(v, semantic=False) for k, v in policy_erased.items()
            },
            "partition_mask_separation": separation,
        },
        "ambiguity_coverage_challenge_contract": {
            "masked_conflict_unique_n": len(conflicts),
            "masked_conflict_relations": conflicts,
            "row_counts": {k: len(v) for k, v in challenge_erased.items()},
            "used_for_fit": False,
            "used_for_calibration": False,
            "used_for_model_selection": False,
            "note":
                "These rows were excluded by the erased-input conflict rule. Their labels are used only for descriptive evaluation; relation IDs are never inserted into inference text.",
        },
        "representation_contract": {
            "base_model_frozen": True,
            "embeddings_frozen": True,
            "transformer_frozen": True,
            "lm_head_frozen": True,
            "pooling": "attention-mask mean pooling, unchanged",
            "classifier": "single affine linear layer for both arms",
            "other_registered_subjects": OTHER,
            "tokenizer_vocab_added": False,
            "tokenizer_length_before": tokenizer_len_before,
            "tokenizer_length_after": tokenizer_len_after,
            "encoder_fingerprint": encoder_fingerprint,
            "no_output_correction": True,
            "no_quotient": True,
        },
        "training": arm_training,
        "results": arm_results,
        "comparison": comparison,
        "pilot_criteria": {
            "validation_relation_accuracy_min_pct": 100 * a.min_validation_relation_accuracy,
            "correct_forbidden_accept_min_pct": 100 * a.min_calib_correct_accept,
            "wrong_forbidden_accept_max_pct": 100 * a.epsilon_wrong,
            "permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "candidate_present_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_negative_family_max_pct": 100 * a.epsilon_retain,
            "whole_query_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
        },
        "interpretation_guardrail":
            "An improvement in the marked arm would show that preserving target information helps this probe; it would not by itself establish a novel architecture or solve end-to-end Gen.",
    }
    (out / "target_representation_compare_fix5e.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    compact = {
        arm: {
            "calibration_status": arm_results[arm]["calibration"]["status"],
            "eta": arm_results[arm]["eta"],
            "fit_relation_accuracy_pct": arm_results[arm]["semantic"]["fit"]["accuracy_pct"],
            "validation_relation_accuracy_pct": arm_results[arm]["semantic"]["validation"]["accuracy_pct"],
            "validation_policy": arm_results[arm]["validation_policy"],
            "pilot_pass": arm_results[arm]["pilot_pass"],
            "dev_official_para": arm_results[arm]["development_only_official_seed1"]["paraphrase"],
            "challenge_validation": arm_results[arm]["ambiguity_coverage_challenge"]["validation"],
        }
        for arm in arms
    }
    compact["comparison"] = comparison
    compact["feature_cache"] = str(out / "target_representation_feature_cache.pt")
    compact["output_dir"] = str(out)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
