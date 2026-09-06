#!/usr/bin/env python3
"""Seed-1 recognition-only pilot using Relation-View Corpus V2.

The subject candidate stage is fixed, explicit, boundary-aware literal matching.
Only a shared relation scorer is trained.  No output correction, quotient, model
editing, or official paraphrase text is used for fitting/calibration/model
selection.  Already-inspected Seed-1 official paraphrases are reported only as
development evidence after the operating point is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcf_zero_unlearn_official_eval as off
from mcf_sampling import sample_official_mcf_records

SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
CORPUS_PROTOCOL = "mcf_relation_view_corpus_v2"
MAX_LENGTH = 256
RETAIN_FIT = 300
RETAIN_CALIB = 300
MATCH_DIM = 128
TRAIN_STEPS = 1200
TRAIN_BATCH = 256
LR = 2e-3
WEIGHT_DECAY = 1e-4
SUBJECT_ANCHOR_TEMPLATE = "General information about {subject}."


@dataclass(frozen=True)
class QueryRow:
    text: str
    owner: int | None
    kind: str


def rr(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def render(row: Mapping[str, Any]) -> str:
    r = rr(row)
    return str(r["prompt"]).format(str(r["subject"]))


def fact_key(row: Mapping[str, Any]) -> tuple[str, str]:
    r = rr(row)
    return str(r["subject"]), str(r["relation_id"])


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def subject_regex(subject: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", re.IGNORECASE)


def dedupe_rows(rows: Sequence[QueryRow]) -> list[QueryRow]:
    out: dict[str, QueryRow] = {}
    for row in rows:
        key = " ".join(row.text.split())
        old = out.get(key)
        if old is not None and old.owner != row.owner:
            raise RuntimeError(f"ambiguous label for query {key!r}: {old.owner} vs {row.owner}")
        out.setdefault(key, QueryRow(key, row.owner, row.kind))
    return list(out.values())


def load_v2(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, list[str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != CORPUS_PROTOCOL:
        raise RuntimeError(f"expected {CORPUS_PROTOCOL}, got {payload.get('protocol')}")
    leakage = payload.get("leakage_contract", {})
    required_false = (
        "full_mcf_path_accepted", "official_paraphrase_prompts_read",
        "official_neighborhood_prompts_read", "official_generation_prompts_read",
        "official_retain_records_read", "generator_received_target_true",
        "generator_received_target_new", "verifier_received_target_true",
        "verifier_received_target_new",
    )
    if any(leakage.get(k) is not False for k in required_false):
        raise RuntimeError("Relation-View V2 leakage contract failed")
    split = payload.get("family_split_recommendation", {})
    if not all(split.get(k) for k in ("fit", "calibration", "validation")):
        raise RuntimeError("V2 corpus missing family split recommendation")
    facts: dict[int, dict[str, Any]] = {}
    for i, case in enumerate(payload.get("cases", [])):
        views = {str(v["family"]): str(v["template"]) for v in case.get("views", [])}
        expected = set(split["fit"] + split["calibration"] + split["validation"])
        if set(views) != expected:
            raise RuntimeError(f"case {case.get('case_id')} family mismatch")
        facts[i] = {
            "case_id": int(case["case_id"]),
            "subject": str(case["subject"]),
            "relation_id": str(case["relation_id"]),
            "views": views,
        }
    if len(facts) != FORGET_NUM:
        raise RuntimeError(f"expected {FORGET_NUM} facts, got {len(facts)}")
    return facts, {k: [str(x) for x in v] for k, v in split.items()}, payload


def align_facts_to_forget(facts: dict[int, dict[str, Any]], forget: Sequence[Mapping[str, Any]]) -> None:
    expected = [(int(x["case_id"]), str(rr(x)["subject"]), str(rr(x)["relation_id"])) for x in forget]
    actual = [(facts[i]["case_id"], facts[i]["subject"], facts[i]["relation_id"]) for i in sorted(facts)]
    if actual != expected:
        raise RuntimeError("V2 corpus cases/order do not exactly match official Seed-1 forget50")


def split_retain(retain: Sequence[Mapping[str, Any]], bank_pairs: set[tuple[str, str]]) -> tuple[list, list, list]:
    clean = [x for x in retain if fact_key(x) not in bank_pairs]
    clean = sorted(clean, key=lambda x: stable_int(f"retain:{SEED}:{int(x['case_id'])}:{render(x)}"))
    fit = clean[:RETAIN_FIT]
    calib = clean[RETAIN_FIT:RETAIN_FIT + RETAIN_CALIB]
    val = clean[RETAIN_FIT + RETAIN_CALIB:]
    if len(fit) < RETAIN_FIT or len(calib) < RETAIN_CALIB or not val:
        raise RuntimeError("insufficient retain data after bank duplicate removal")
    return fit, calib, val


def phase_families(split: Mapping[str, Sequence[str]], phase: str) -> list[str]:
    return list(split[{"fit": "fit", "calib": "calibration", "validation": "validation"}[phase]])


def phase_templates(info: Mapping[str, Any], split: Mapping[str, Sequence[str]], phase: str) -> list[str]:
    return [str(info["views"][fam]) for fam in phase_families(split, phase)]


def synth_same_subject_other_relation(
    facts: Mapping[int, Mapping[str, Any]], split: Mapping[str, Sequence[str]], phase: str, per_fact: int = 2
) -> list[QueryRow]:
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    ids = sorted(facts)
    out: list[QueryRow] = []
    for i in ids:
        target = facts[i]
        choices = [
            j for j in ids if j != i
            and str(facts[j]["relation_id"]) != str(target["relation_id"])
            and (str(target["subject"]), str(facts[j]["relation_id"])) not in bank
        ]
        if not choices:
            continue
        start = stable_int(f"ssdr-v2:{phase}:{i}") % len(choices)
        for k in range(int(per_fact)):
            other = facts[choices[(start + k) % len(choices)]]
            templates = phase_templates(other, split, phase)
            template = templates[k % len(templates)]
            out.append(QueryRow(template.format(str(target["subject"])), None, "same_subject_different_relation"))
    return out


def crossed_binding_rows(
    facts: Mapping[int, Mapping[str, Any]], split: Mapping[str, Sequence[str]], phase: str
) -> list[QueryRow]:
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    ids = sorted(facts)
    out: list[QueryRow] = []
    for i in ids:
        a = facts[i]
        rel_choices = [j for j in ids if j != i and (str(a["subject"]), str(facts[j]["relation_id"])) not in bank]
        subj_choices = [j for j in ids if j != i and (str(facts[j]["subject"]), str(a["relation_id"])) not in bank]
        if not rel_choices or not subj_choices:
            continue
        j = rel_choices[stable_int(f"cross-r:{phase}:{i}") % len(rel_choices)]
        k = subj_choices[stable_int(f"cross-s:{phase}:{i}") % len(subj_choices)]
        b_rel = facts[j]; b_sub = facts[k]
        t1 = phase_templates(b_rel, split, phase)[0].format(str(a["subject"]))
        t2 = phase_templates(a, split, phase)[-1].format(str(b_sub["subject"]))
        out.append(QueryRow(f"First: {t1} Second: {t2}", None, "crossed_binding"))
    return out


def mention_rows(facts: Mapping[int, Mapping[str, Any]], phase: str) -> list[QueryRow]:
    templates = {
        "fit": 'Spell the entity name "{subject}" exactly.',
        "calib": 'Quote only the entity name "{subject}" without stating an attribute.',
        "validation": 'Repeat the name "{subject}" and do not give facts about it.',
    }
    return [QueryRow(templates[phase].format(subject=v["subject"]), None, "permitted_subject_mention") for v in facts.values()]


def retain_kind(row: Mapping[str, Any], facts: Mapping[int, Mapping[str, Any]], forget: Sequence[Mapping[str, Any]]) -> str:
    s, relation = fact_key(row)
    bank_pairs = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    bank_subjects = {x[0] for x in bank_pairs}
    bank_relations = {x[1] for x in bank_pairs}
    bank_answers = {str(rr(x)["target_true"]["str"]) for x in forget}
    answer = str(rr(row)["target_true"]["str"])
    if s in bank_subjects and (s, relation) not in bank_pairs:
        return "retain_same_subject_different_relation"
    if relation in bank_relations and s not in bank_subjects:
        return "retain_same_relation_different_subject"
    if answer in bank_answers:
        return "retain_same_answer_different_fact"
    return "retain_other"


def build_rows(
    facts: Mapping[int, Mapping[str, Any]], split: Mapping[str, Sequence[str]],
    retain_rows: Sequence[Mapping[str, Any]], forget: Sequence[Mapping[str, Any]], phase: str
) -> list[QueryRow]:
    rows: list[QueryRow] = []
    for i in sorted(facts):
        info = facts[i]
        for template in phase_templates(info, split, phase):
            rows.append(QueryRow(template.format(str(info["subject"])), i, f"positive_{phase}"))
    rows.extend(QueryRow(render(x), None, retain_kind(x, facts, forget)) for x in retain_rows)
    rows.extend(synth_same_subject_other_relation(facts, split, phase))
    rows.extend(crossed_binding_rows(facts, split, phase))
    rows.extend(mention_rows(facts, phase))
    return dedupe_rows(rows)


def family(kind: str) -> str:
    if "same_subject_different_relation" in kind:
        return "same_subject_different_relation"
    if "same_relation_different_subject" in kind:
        return "same_relation_different_subject"
    if "same_answer_different_fact" in kind:
        return "same_answer_different_fact"
    if kind == "crossed_binding":
        return "crossed_binding"
    if kind == "permitted_subject_mention":
        return "permitted_subject_mention"
    return "other_retain"


@torch.no_grad()
def encode_pooled(model: Any, tokenizer: Any, texts: Sequence[str], device: torch.device, batch_size: int) -> torch.Tensor:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("pilot requires model.model backbone")
    chunks: list[torch.Tensor] = []
    old = tokenizer.padding_side; tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), int(batch_size)):
            batch = list(texts[start:start + int(batch_size)])
            enc = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
            out = backbone(**enc, use_cache=False, return_dict=True)
            h = out.last_hidden_state.float()
            m = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            pooled = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
            chunks.append(F.normalize(pooled, p=2, dim=-1).cpu())
    finally:
        tokenizer.padding_side = old
    return torch.cat(chunks, 0)


def candidate_mask(texts: Sequence[str], facts: Mapping[int, Mapping[str, Any]]) -> torch.Tensor:
    patterns = [subject_regex(str(facts[i]["subject"])) for i in sorted(facts)]
    return torch.tensor([[bool(p.search(text)) for p in patterns] for text in texts], dtype=torch.bool)


def build_relation_state(
    model: Any, tokenizer: Any, facts: Mapping[int, Mapping[str, Any]], split: Mapping[str, Sequence[str]],
    device: torch.device, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    anchors: list[torch.Tensor] = []
    protos: list[torch.Tensor] = []
    for i in sorted(facts):
        info = facts[i]; subject = str(info["subject"])
        anchor_text = SUBJECT_ANCHOR_TEMPLATE.format(subject=subject)
        prompts = [t.format(subject) for t in phase_templates(info, split, "fit")]
        enc = encode_pooled(model, tokenizer, [anchor_text] + prompts, device, batch_size)
        anchor = enc[0].float()
        residuals = F.normalize(enc[1:].float() - anchor[None, :], p=2, dim=-1)
        anchors.append(anchor)
        protos.append(F.normalize(residuals.mean(0), p=2, dim=0))
    return torch.stack(anchors), torch.stack(protos)


class RelationMatcher(nn.Module):
    def __init__(self, hidden: int, dim: int = MATCH_DIM):
        super().__init__()
        self.q = nn.Linear(hidden, dim, bias=False)
        self.p = nn.Linear(hidden, dim, bias=False)
        self.mlp = nn.Sequential(nn.Linear(dim * 4, dim * 2), nn.GELU(), nn.Linear(dim * 2, 1))

    def forward(self, query_rel: torch.Tensor, proto: torch.Tensor) -> torch.Tensor:
        q = F.normalize(self.q(query_rel.float()), p=2, dim=-1)
        p = F.normalize(self.p(proto.float()), p=2, dim=-1)
        return self.mlp(torch.cat([q, p, q * p, (q - p).abs()], dim=-1)).squeeze(-1)


def pair_tensors(
    rows: Sequence[QueryRow], query_emb: torch.Tensor, mask: torch.Tensor,
    anchors: torch.Tensor, protos: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qparts: list[torch.Tensor] = []
    pparts: list[torch.Tensor] = []
    labels: list[float] = []
    for qi, row in enumerate(rows):
        cands = torch.where(mask[qi])[0].tolist()
        for fi in cands:
            rel = F.normalize(query_emb[qi].float() - anchors[fi].float(), p=2, dim=0)
            qparts.append(rel); pparts.append(protos[fi].float())
            labels.append(1.0 if row.owner is not None and int(row.owner) == int(fi) else 0.0)
    if not labels:
        raise RuntimeError("no candidate pairs available")
    return torch.stack(qparts), torch.stack(pparts), torch.tensor(labels, dtype=torch.float32)


def train_matcher(
    rows: Sequence[QueryRow], query_emb: torch.Tensor, mask: torch.Tensor,
    anchors: torch.Tensor, protos: torch.Tensor, device: torch.device
) -> tuple[RelationMatcher, dict[str, Any]]:
    q, p, y = pair_tensors(rows, query_emb, mask, anchors, protos)
    pos = max(1, int(y.sum().item())); neg = max(1, int(y.numel() - y.sum().item()))
    model = RelationMatcher(q.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gen = torch.Generator().manual_seed(SEED + 8123)
    qg, pg, yg = q.to(device), p.to(device), y.to(device)
    pos_weight = torch.tensor(float(neg / pos), device=device)
    trace = []
    for step in range(1, TRAIN_STEPS + 1):
        idx = torch.randint(0, len(y), (min(TRAIN_BATCH, len(y)),), generator=gen).to(device)
        logits = model(qg[idx], pg[idx])
        loss = F.binary_cross_entropy_with_logits(logits, yg[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 100 == 0 or step == TRAIN_STEPS:
            trace.append({"step": step, "loss": float(loss.detach().item())})
    model.eval()
    for p0 in model.parameters(): p0.requires_grad_(False)
    return model, {"pair_count": len(y), "positive_pairs": pos, "negative_pairs": neg, "loss_trace": trace}


@torch.no_grad()
def score_queries(
    model: RelationMatcher, rows: Sequence[QueryRow], query_emb: torch.Tensor, mask: torch.Tensor,
    anchors: torch.Tensor, protos: torch.Tensor, device: torch.device
) -> torch.Tensor:
    scores = torch.full((len(rows), len(anchors)), -1e9, dtype=torch.float32)
    for qi in range(len(rows)):
        cands = torch.where(mask[qi])[0]
        if not len(cands):
            continue
        rel = F.normalize(query_emb[qi].float()[None, :] - anchors[cands].float(), p=2, dim=-1).to(device)
        pp = protos[cands].float().to(device)
        probs = torch.sigmoid(model(rel, pp)).cpu()
        scores[qi, cands] = probs
    return scores


def decision(scores: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top, idx = scores.max(dim=1)
    has_candidate = top > -1e8
    accept = has_candidate & (top >= float(tau))
    return accept, idx, top


def positive_report(rows: Sequence[QueryRow], scores: torch.Tensor, mask: torch.Tensor, tau: float) -> dict[str, Any]:
    owners = torch.tensor([int(x.owner) for x in rows], dtype=torch.long)
    accept, idx, top = decision(scores, tau)
    correct = idx.eq(owners)
    recall = mask[torch.arange(len(rows)), owners]
    return {
        "n": len(rows),
        "candidate_recall_pct": 100 * float(recall.float().mean().item()),
        "top1_correct_rate_pct": 100 * float(correct.float().mean().item()),
        "correct_accept_rate_pct": 100 * float((accept & correct).float().mean().item()),
        "wrong_fact_accept_rate_pct": 100 * float((accept & ~correct).float().mean().item()),
        "reject_rate_pct": 100 * float((~accept).float().mean().item()),
        "top_score_mean": float(top[torch.isfinite(top) & (top > -1e8)].mean().item()) if bool((top > -1e8).any()) else None,
    }


def negative_report(scores: torch.Tensor, tau: float) -> dict[str, Any]:
    accept, _idx, top = decision(scores, tau)
    valid = top > -1e8
    return {
        "n": len(scores),
        "candidate_present_pct": 100 * float(valid.float().mean().item()) if len(scores) else None,
        "whole_bank_false_activation_rate_pct": 100 * float(accept.float().mean().item()) if len(scores) else None,
        "top_score_mean_when_candidate": float(top[valid].mean().item()) if bool(valid.any()) else None,
    }


def split_pos_neg(rows: Sequence[QueryRow], scores: torch.Tensor, mask: torch.Tensor):
    p = [i for i, r in enumerate(rows) if r.owner is not None]
    n = [i for i, r in enumerate(rows) if r.owner is None]
    return [rows[i] for i in p], [rows[i] for i in n], scores[p], scores[n], mask[p], mask[n]


def calibrate(
    rows: Sequence[QueryRow], scores: torch.Tensor, mask: torch.Tensor,
    epsilon_retain: float, epsilon_wrong: float, min_correct_accept: float
) -> tuple[float, dict[str, Any]]:
    pos_rows, neg_rows, ps, ns, pm, _nm = split_pos_neg(rows, scores, mask)
    owners = torch.tensor([int(x.owner) for x in pos_rows], dtype=torch.long)
    ptop, pidx = ps.max(dim=1); correct = pidx.eq(owners)
    ntop = ns.max(dim=1).values
    vals = torch.cat([ptop[ptop > -1e8], ntop[ntop > -1e8]])
    candidates = sorted(set([0.0, 1.000001] + [float(x) for x in np.quantile(vals.numpy(), np.linspace(0, 1, 151))]))
    fam_idx: dict[str, list[int]] = {}
    for i, row in enumerate(neg_rows): fam_idx.setdefault(family(row.kind), []).append(i)
    best = None
    for tau in candidates:
        pa = ptop >= tau; na = ntop >= tau
        ca = float((pa & correct).float().mean().item())
        wrong = float((pa & ~correct).float().mean().item())
        overall = float(na.float().mean().item()) if len(na) else 0.0
        fam_rates = {k: float(na[idx].float().mean().item()) for k, idx in fam_idx.items() if idx}
        feasible = wrong <= epsilon_wrong + 1e-12 and overall <= epsilon_retain + 1e-12 and all(v <= epsilon_retain + 1e-12 for v in fam_rates.values())
        if not feasible: continue
        key = (-ca, max(fam_rates.values(), default=0.0), overall, wrong, -tau)
        cand = {
            "tau": tau, "correct_accept_rate": ca, "wrong_fact_accept_rate": wrong,
            "negative_whole_bank_accept_rate": overall, "negative_family_accept_rates": fam_rates,
            "positive_top1_correct_rate": float(correct.float().mean().item()),
        }
        if best is None or key < best[0]: best = (key, cand)
    if best is None:
        raise RuntimeError("no preservation-feasible threshold, including reject-all")
    chosen = best[1]
    chosen["status"] = "ACCEPTABLE_OPERATING_POINT" if chosen["correct_accept_rate"] >= min_correct_accept else "NO_ACCEPTABLE_OPERATING_POINT"
    chosen["selection_rule"] = "maximize correct acceptance subject to overall + each-negative-family <= epsilon_retain and wrong-fact <= epsilon_wrong"
    return float(chosen["tau"]), chosen


def evaluate(
    calib_rows: Sequence[QueryRow], calib_scores: torch.Tensor, calib_mask: torch.Tensor,
    val_rows: Sequence[QueryRow], val_scores: torch.Tensor, val_mask: torch.Tensor,
    dev_direct: Sequence[QueryRow], direct_scores: torch.Tensor, direct_mask: torch.Tensor,
    dev_para: Sequence[QueryRow], para_scores: torch.Tensor, para_mask: torch.Tensor,
    epsilon_retain: float, epsilon_wrong: float, min_correct_accept: float
) -> dict[str, Any]:
    tau, cal = calibrate(calib_rows, calib_scores, calib_mask, epsilon_retain, epsilon_wrong, min_correct_accept)
    vp, vn, vps, vns, vpm, _ = split_pos_neg(val_rows, val_scores, val_mask)
    pos = positive_report(vp, vps, vpm, tau)
    neg = negative_report(vns, tau)
    family_reports = {}
    for fam in sorted({family(x.kind) for x in vn}):
        idx = [i for i, x in enumerate(vn) if family(x.kind) == fam]
        family_reports[fam] = negative_report(vns[idx], tau)
    pilot = (
        cal["status"] == "ACCEPTABLE_OPERATING_POINT"
        and pos["top1_correct_rate_pct"] >= 70.0
        and pos["correct_accept_rate_pct"] >= 60.0
        and pos["wrong_fact_accept_rate_pct"] <= 2.0 + 1e-9
        and neg["whole_bank_false_activation_rate_pct"] <= 2.0 + 1e-9
        and all((x["whole_bank_false_activation_rate_pct"] or 0.0) <= 2.0 + 1e-9 for x in family_reports.values())
    )
    return {
        "calibration": cal,
        "validation": {"positive": pos, "negative_overall": neg, "negative_families": family_reports, "pilot_pass": bool(pilot)},
        "development_only_official_seed1": {
            "direct_forget": positive_report(dev_direct, direct_scores, direct_mask, tau),
            "forget_paraphrase": positive_report(dev_para, para_scores, para_mask, tau),
            "note": "Previously inspected official Seed-1 probes; descriptive development evidence only.",
        },
    }


def official_dev_rows(forget: Sequence[Mapping[str, Any]]) -> tuple[list[QueryRow], list[QueryRow]]:
    direct, para = [], []
    for i, row in enumerate(forget):
        direct.append(QueryRow(render(row), i, "official_direct_development"))
        for text in row.get("paraphrase_prompts", []):
            para.append(QueryRow(str(text), i, "official_paraphrase_development"))
    return direct, para


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--view-corpus-v2", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--epsilon-retain", type=float, default=0.02)
    p.add_argument("--epsilon-wrong", type=float, default=0.02)
    p.add_argument("--min-calib-correct-accept", type=float, default=0.60)
    args = p.parse_args()
    outdir = Path(args.output_dir).resolve(); outdir.mkdir(parents=True, exist_ok=False)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)

    data = json.loads(Path(args.mcf_path).read_text(encoding="utf-8"))
    forget, retain = sample_official_mcf_records(data, FORGET_NUM, RETAIN_NUM, SEED, strict=True)
    forget = [off.normalize_record(x) for x in forget]; retain = [off.normalize_record(x) for x in retain]
    facts, split, corpus = load_v2(Path(args.view_corpus_v2))
    align_facts_to_forget(facts, forget)
    bank = {(str(v["subject"]), str(v["relation_id"])) for v in facts.values()}
    rfit, rcal, rval = split_retain(retain, bank)
    fit_rows = build_rows(facts, split, rfit, forget, "fit")
    calib_rows = build_rows(facts, split, rcal, forget, "calib")
    val_rows = build_rows(facts, split, rval, forget, "validation")
    dev_direct, dev_para = official_dev_rows(forget)

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype_from_name(args.dtype), local_files_only=True, low_cpu_mem_usage=True
    ).to(device)
    model.eval(); model.config.use_cache = False
    for q in model.parameters(): q.requires_grad_(False)

    sets = {"fit": fit_rows, "calib": calib_rows, "validation": val_rows, "dev_direct": dev_direct, "dev_para": dev_para}
    emb = {k: encode_pooled(model, tok, [x.text for x in rows], device, args.encode_batch_size) for k, rows in sets.items()}
    masks = {k: candidate_mask([x.text for x in rows], facts) for k, rows in sets.items()}
    anchors, protos = build_relation_state(model, tok, facts, split, device, args.encode_batch_size)
    matcher, training = train_matcher(fit_rows, emb["fit"], masks["fit"], anchors, protos, device)
    scores = {k: score_queries(matcher, sets[k], emb[k], masks[k], anchors, protos, device) for k in sets}

    result = evaluate(
        calib_rows, scores["calib"], masks["calib"],
        val_rows, scores["validation"], masks["validation"],
        dev_direct, scores["dev_direct"], masks["dev_direct"],
        dev_para, scores["dev_para"], masks["dev_para"],
        float(args.epsilon_retain), float(args.epsilon_wrong), float(args.min_calib_correct_accept),
    )
    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_subject_filtered_relation_views_v2_recognition_only",
        "recognition_only": True,
        "subject_candidate_rule": "boundary-aware literal registered-subject match",
        "relation_matcher": "shared MLP over subject-residual query + V2 relation prototype interactions",
        "training": training,
        "data_contract": {
            "corpus_protocol": corpus["protocol"],
            "family_split": split,
            "official_paraphrases_used_for_fit": False,
            "official_paraphrases_used_for_calibration": False,
            "official_paraphrases_used_for_model_selection": False,
            "official_seed1_status": "development evidence only; previously inspected",
            "base_model_frozen": True,
            "no_output_correction": True,
            "no_quotient": True,
            "retain_split_counts": {"fit": len(rfit), "calibration": len(rcal), "validation": len(rval)},
            "query_counts": {k: len(v) for k, v in sets.items()},
        },
        "preservation_budgets": {
            "epsilon_retain": float(args.epsilon_retain), "epsilon_wrong": float(args.epsilon_wrong),
            "minimum_calibration_correct_accept": float(args.min_calib_correct_accept),
        },
        "result": result,
    }
    (outdir / "subject_relation_views_v2_recognition.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {
        "calibration_status": result["calibration"]["status"],
        "validation": result["validation"],
        "dev_official_direct": result["development_only_official_seed1"]["direct_forget"],
        "dev_official_para": result["development_only_official_seed1"]["forget_paraphrase"],
        "output_dir": str(outdir),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
