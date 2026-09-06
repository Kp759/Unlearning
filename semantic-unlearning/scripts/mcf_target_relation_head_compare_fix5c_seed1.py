#!/usr/bin/env python3
"""Fix5c: cached frozen-feature comparison of linear vs small MLP relation heads.

Recognition-only. The base Llama, embeddings, LM head, quotient, and output correction
remain frozen/disabled. One frozen mean-pooled feature cache is built and reused by
both heads. Semantic examples are deduplicated by masked text; policy evaluation
keeps distinct original subject/query/binding instances.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_relation_classifier_fix5_seed1 as base
import mcf_target_relation_classifier_fix5b_seed1 as fix5b

Row = base.Row
SEED = 1
NONE = base.NONE
TARGET = base.TARGET
OTHER = base.OTHER


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def policy_identity(row: Row) -> tuple[Any, ...]:
    """Identity of one policy event; subject is intentionally included."""
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


def policy_manifest(rows: Sequence[Row]) -> list[Row]:
    out: dict[tuple[Any, ...], Row] = {}
    for row in rows:
        out.setdefault(policy_identity(row), row)
    return list(out.values())


def semantic_unique(rows: Sequence[Row]) -> list[Row]:
    return base.dedup_sem(rows)


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


def json_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def contains_subsequence(seq: Sequence[int], sub: Sequence[int]) -> bool:
    if not sub:
        return True
    n = len(sub)
    return any(list(seq[i:i+n]) == list(sub) for i in range(0, len(seq) - n + 1))


@torch.no_grad()
def encode_cache(model: Any, tok: Any, texts: Sequence[str], device: torch.device, batch: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("requires model.model")
    target_ids = tok(TARGET, add_special_tokens=False)["input_ids"]
    diagnostics: list[dict[str, Any]] = []
    full_lengths: list[int] = []
    for text in texts:
        ids = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
        full_lengths.append(len(ids))

    chunks: list[torch.Tensor] = []
    old_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for st in range(0, len(texts), int(batch)):
            batch_text = list(texts[st:st + int(batch)])
            enc = tok(
                batch_text,
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
            for j in range(len(batch_text)):
                kept = ids_cpu[j][mask_cpu[j].bool()].tolist()
                pos = st + j
                diagnostics.append({
                    "text_index": pos,
                    "full_token_count": int(full_lengths[pos]),
                    "kept_token_count": len(kept),
                    "truncated": bool(full_lengths[pos] > base.MAX_LENGTH),
                    "target_marker_visible_after_truncation": contains_subsequence(kept, target_ids),
                })
    finally:
        tok.padding_side = old_side
    return torch.cat(chunks, dim=0), diagnostics


class RelationMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        if min(input_dim, num_classes, hidden_dim) <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features.float())


def class_weights(y: torch.Tensor, k: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(y, minlength=k).float()
    if bool((counts == 0).any()):
        raise RuntimeError(f"zero fit class: {torch.where(counts == 0)[0].tolist()}")
    w = len(y) / (k * counts)
    return (w / w.mean()).to(device)


def train_head(
    kind: str,
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    device: torch.device,
    steps: int,
    batch: int,
    lr: float,
    wd: float,
    hidden_dim: int,
    dropout: float,
    seed: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if kind == "linear":
        head: nn.Module = base.Linear(x.shape[1], k).to(device)
    elif kind == "mlp":
        head = RelationMLP(x.shape[1], k, hidden_dim, dropout).to(device)
    else:
        raise ValueError(kind)
    w = class_weights(y, k, device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed + 9501)
    xd, yd = x.to(device), y.to(device)
    trace: list[dict[str, Any]] = []
    head.train()
    for step in range(1, int(steps) + 1):
        ii = torch.randint(0, len(y), (min(int(batch), len(y)),), generator=g).to(device)
        logits = head(xd[ii])
        loss = F.cross_entropy(logits, yd[ii], weight=w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0 or step == int(steps):
            with torch.no_grad():
                head.eval()
                fit_acc = float(head(xd).argmax(1).eq(yd).float().mean().item())
                head.train()
            trace.append({"step": step, "loss": float(loss.item()), "fit_accuracy": fit_acc})
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head, {
        "kind": kind,
        "steps": int(steps),
        "examples": len(y),
        "lr": float(lr),
        "weight_decay": float(wd),
        "hidden_dim": int(hidden_dim) if kind == "mlp" else None,
        "dropout": float(dropout) if kind == "mlp" else 0.0,
        "seed": int(seed),
        "class_weights": [float(v) for v in w.cpu()],
        "trace": trace,
    }


@torch.no_grad()
def head_logits(head: nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    head.eval()
    return head(features.to(device)).cpu()


def evaluate_head(
    head: nn.Module,
    features: torch.Tensor,
    idx: Mapping[str, Sequence[int]],
    semantic: Mapping[str, Sequence[Row]],
    policy: Mapping[str, Sequence[Row]],
    dev: Mapping[str, Sequence[Row]],
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
    logits = {name: head_logits(head, take(features, ii), device) for name, ii in idx.items()}
    eta, cal = base.calibrate(policy["calib"], logits["policy_calib"], list(classes), none_idx, bank, eps, eps_wrong, min_calib_accept)
    sem = {
        phase: base.sem_report(semantic[phase], logits[f"semantic_{phase}"], c2i, list(classes))
        for phase in ("fit", "calib", "validation")
    }
    vp = base.policy(policy["validation"], logits["policy_validation"], eta, list(classes), none_idx, bank)
    fam_ok = all((v["false_activation_pct"] or 0.0) <= 100 * eps + 1e-9 for v in vp["permitted_negative_families"].values())
    pilot = (
        cal["status"] == "ACCEPTABLE_OPERATING_POINT"
        and sem["validation"]["accuracy_pct"] >= 100 * min_val_acc
        and (vp["correct_forbidden_binding_accept_pct"] or 0.0) >= 100 * min_calib_accept
        and (vp["wrong_forbidden_binding_accept_pct"] or 0.0) <= 100 * eps_wrong + 1e-9
        and (vp["permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and (vp["candidate_present_permitted_false_activation_pct"] or 0.0) <= 100 * eps + 1e-9
        and fam_ok
    )
    dev_rep = {}
    for name in ("direct", "paraphrase"):
        rows = dev[name]
        lg = logits[f"dev_{name}"]
        dev_rep[name] = {
            "semantic": base.sem_report(rows, lg, c2i, list(classes)),
            "policy": base.policy(rows, lg, eta, list(classes), none_idx, bank),
        }
    return {
        "calibration": cal,
        "eta": float(eta),
        "semantic": sem,
        "validation_policy": vp,
        "pilot_pass": bool(pilot),
        "development_only_official_seed1": dev_rep,
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
    ap.add_argument("--mlp-hidden-dim", type=int, default=256)
    ap.add_argument("--mlp-dropout", type=float, default=0.1)
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
    prepared = {k: base.prep(v, subjects) for k, v in raw.items()}
    prepared, sep = fix5b.separate(prepared)
    semantic = {k: semantic_unique(v) for k, v in prepared.items()}
    policy = {k: policy_manifest(v) for k, v in prepared.items()}

    missing = sorted(set(classes) - {r.relation for r in semantic["fit"]})
    if missing:
        raise RuntimeError(f"fit classes missing after masking/dedup: {missing}")

    dr, pr = base.dev_rows(forget)
    dev = {
        "direct": policy_manifest(base.prep(dr, subjects)),
        "paraphrase": policy_manifest(base.prep(pr, subjects)),
    }

    rows_for_cache = {
        "semantic_fit": semantic["fit"],
        "semantic_calib": semantic["calib"],
        "semantic_validation": semantic["validation"],
        "policy_calib": policy["calib"],
        "policy_validation": policy["validation"],
        "dev_direct": dev["direct"],
        "dev_paraphrase": dev["paraphrase"],
    }
    texts, indices = feature_index(rows_for_cache)

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

    features, token_diag = encode_cache(model, tok, texts, device, a.encode_batch_size)
    encoder_fingerprint = json_hash({
        "model_config": model.config.to_dict(),
        "tokenizer_class": tok.__class__.__name__,
        "special_tokens": tok.special_tokens_map,
        "max_length": base.MAX_LENGTH,
        "target_marker": TARGET,
        "other_marker": OTHER,
        "dtype": a.dtype,
    })

    cache_payload = {
        "features": features,
        "texts": texts,
        "indices": indices,
        "classes": classes,
        "token_diagnostics": token_diag,
        "encoder_fingerprint": encoder_fingerprint,
        "semantic_rows": {k: [asdict(r) for r in v] for k, v in semantic.items()},
        "policy_rows": {k: [asdict(r) for r in v] for k, v in policy.items()},
        "dev_rows": {k: [asdict(r) for r in v] for k, v in dev.items()},
    }
    torch.save(cache_payload, out / "frozen_relation_feature_cache.pt")

    xfit = take(features, indices["semantic_fit"])
    yfit = torch.tensor([c2i[r.relation] for r in semantic["fit"]], dtype=torch.long)
    results: dict[str, Any] = {}
    training: dict[str, Any] = {}
    for kind in ("linear", "mlp"):
        head, tr = train_head(
            kind, xfit, yfit, len(classes), device,
            a.train_steps, a.train_batch_size, a.lr, a.weight_decay,
            a.mlp_hidden_dim, a.mlp_dropout, a.head_seed,
        )
        training[kind] = tr
        results[kind] = evaluate_head(
            head, features, indices, semantic, policy, dev,
            classes, c2i, none_idx, bank, device,
            a.epsilon_retain, a.epsilon_wrong,
            a.min_calib_correct_accept, a.min_validation_relation_accuracy,
        )
        torch.save({"state_dict": head.state_dict(), "classes": classes, "training": tr}, out / f"{kind}_head.pt")

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_fix5c_cached_linear_vs_mlp_relation_heads_recognition_only",
        "recognition_only": True,
        "comparison_question": "Does a small nonlinear head improve held-out relation recognition and forbidden-binding acceptance over the trained linear baseline at the same false-activation budget?",
        "data_contract": {
            "corpus_protocol": corpus["protocol"],
            "family_split": split,
            "partition_mask_separation": sep,
            "semantic_unique_counts": {k: len(v) for k, v in semantic.items()},
            "policy_manifest_counts": {k: len(v) for k, v in policy.items()},
            "policy_identity_includes_subject": True,
            "feature_cache_unique_masked_texts": len(texts),
            "encoder_fingerprint": encoder_fingerprint,
            "base_model_frozen": True,
            "same_cached_features_for_both_heads": True,
            "no_output_correction": True,
            "no_quotient": True,
            "official_paraphrases_used_for_fit": False,
            "official_paraphrases_used_for_calibration": False,
            "official_paraphrases_used_for_model_selection": False,
        },
        "head_contract": {
            "linear": "single affine layer",
            "mlp": f"Linear(d,{a.mlp_hidden_dim}) -> GELU -> Dropout({a.mlp_dropout}) -> Linear({a.mlp_hidden_dim},K)",
            "loss": "class-weighted cross-entropy on logits",
            "confidence": "top1 minus runner-up logit margin with NONE included",
            "thresholds": "calibrated separately per head on the same calibration policy manifest",
        },
        "training": training,
        "results": results,
        "pilot_criteria": {
            "validation_relation_accuracy_min_pct": 100 * a.min_validation_relation_accuracy,
            "correct_forbidden_accept_min_pct": 100 * a.min_calib_correct_accept,
            "wrong_forbidden_accept_max_pct": 100 * a.epsilon_wrong,
            "permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "candidate_present_permitted_false_activation_max_pct": 100 * a.epsilon_retain,
            "each_negative_family_max_pct": 100 * a.epsilon_retain,
        },
        "interpretation_guardrail": "A failed MLP does not prove nonseparability or that the frozen representation contains no usable relation information.",
    }
    (out / "target_relation_head_compare_fix5c.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    compact = {
        "linear": {
            "calibration_status": results["linear"]["calibration"]["status"],
            "eta": results["linear"]["eta"],
            "validation_relation_accuracy_pct": results["linear"]["semantic"]["validation"]["accuracy_pct"],
            "validation_policy": results["linear"]["validation_policy"],
            "pilot_pass": results["linear"]["pilot_pass"],
            "dev_official_para": results["linear"]["development_only_official_seed1"]["paraphrase"],
        },
        "mlp": {
            "calibration_status": results["mlp"]["calibration"]["status"],
            "eta": results["mlp"]["eta"],
            "validation_relation_accuracy_pct": results["mlp"]["semantic"]["validation"]["accuracy_pct"],
            "validation_policy": results["mlp"]["validation_policy"],
            "pilot_pass": results["mlp"]["pilot_pass"],
            "dev_official_para": results["mlp"]["development_only_official_seed1"]["paraphrase"],
        },
        "feature_cache": str(out / "frozen_relation_feature_cache.pt"),
        "output_dir": str(out),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
