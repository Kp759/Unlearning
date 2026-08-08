#!/usr/bin/env python3
"""Deterministic metric-aligned MQuAKE sparse LM-head repair.

For every Base-active sensitive target token, edit only that LM-head row and
push it below the best output row that is guaranteed to remain unedited at the
same teacher-forced state.  This is the minimum metric-aligned requirement for
Eff=0; it does not force answers below a global Unknown row.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active_repair
import mquake_zero_unlearn_official_eval as mquake
import mquake_exact_sparse_row_repair as qp
from mcf_zero_unlearn_official_eval import dtype_from_str, load_official_ppl_text


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def cache_base_forget(model, tok, cases, *, device, llama_like, batch_size):
    rows = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache Base forget", leave=False):
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = out.hidden_states[-1][idx, last, :].float()
        logits = out.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok, [c.target_text for c in batch], llama_like=llama_like, device=device
        )
        pred = logits.argmax(dim=-1)
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        for i, case in enumerate(batch):
            rows.append({
                "case": case,
                "hidden": hidden[i].detach().cpu(),
                "target_id": int(target_ids[i]),
                "target_logit": float(target_logits[i]),
                "correct": bool(int(target_ids[i]) == int(pred[i])),
            })
    return rows


@torch.no_grad()
def cache_fixed_competitors(
    model, tok, cases, *, edited_row_ids, device, llama_like, batch_size
):
    edited = torch.tensor(edited_row_ids, dtype=torch.long, device=device)
    rows = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache fixed competitors", leave=False):
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = out.hidden_states[-1][idx, last, :].float()
        logits = out.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok, [c.target_text for c in batch], llama_like=llama_like, device=device
        )
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        logits.index_fill_(1, edited, float("-inf"))
        fixed_values, fixed_ids = logits.max(dim=1)
        for i, case in enumerate(batch):
            rows.append({
                "case": case,
                "hidden": hidden[i].detach().cpu(),
                "target_id": int(target_ids[i]),
                "target_logit": float(target_logits[i]),
                "fixed_competitor_id": int(fixed_ids[i]),
                "fixed_competitor_logit": float(fixed_values[i]),
            })
    return rows


@torch.no_grad()
def cache_retain(model, tok, cases, *, row_ids, device, llama_like, batch_size):
    selected = torch.tensor(row_ids, dtype=torch.long, device=device)
    hs, ys, yt, sel, ok = [], [], [], [], []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache retain", leave=False):
        encoded = tok([c.prompt for c in batch], padding=True, return_tensors="pt").to(device)
        out = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = out.hidden_states[-1][idx, last, :].float()
        logits = out.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok, [c.target_text for c in batch], llama_like=llama_like, device=device
        )
        pred = logits.argmax(dim=-1)
        hs.append(hidden.cpu())
        ys.append(target_ids.cpu())
        yt.append(logits.gather(1, target_ids[:, None]).squeeze(1).cpu())
        sel.append(logits.index_select(1, selected).cpu())
        ok.append((pred == target_ids).cpu())
    return {
        "hidden": torch.cat(hs).float(),
        "target_ids": torch.cat(ys).long(),
        "target_logits": torch.cat(yt).float(),
        "selected_logits": torch.cat(sel).float(),
        "correct": torch.cat(ok).bool(),
    }


@torch.no_grad()
def ppl_edited_target_states(model, tok, text, *, edited_row_ids, device):
    encoded = tok([text], return_tensors="pt", max_length=100, truncation=True).to(device)
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[-1][0, :-1, :].float().cpu()
    targets = encoded["input_ids"][0, 1:].long().cpu()
    edited = set(int(x) for x in edited_row_ids)
    return [(hidden[i], int(targets[i])) for i in range(len(targets)) if int(targets[i]) in edited]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mquake-path", default="data/MQuAKE-CF-3k-v2.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forget-num", type=int, default=1000)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--forget-margin", type=float, default=0.05)
    p.add_argument("--qp-max-epochs", type=int, default=5000)
    p.add_argument("--qp-tol", type=float, default=1e-4)
    p.add_argument("--retain-cutting-rounds", type=int, default=10)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_str(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    mquake_path = mquake.download_mquake(Path(args.mquake_path))
    forget_records, retain_records = mquake.load_official_eval_records(
        mquake_path, tok, forget_num=args.forget_num, retain_num=args.retain_num, seed=args.seed
    )
    records = (forget_records, retain_records)
    mquake.write_split_manifest(
        outdir / "split_manifest.json",
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    llama_like = mquake.is_llama_like(model, tok)

    base = mquake.evaluate_loaded_model_official(
        method="Base", model=model, tok=tok, model_dir=args.model_path,
        mquake_path=mquake_path, wikidata_dir=Path(args.wikidata_dir),
        out_path=outdir / "base_eval.json", forget_num=args.forget_num,
        retain_num=args.retain_num, seed=args.seed, batch_size=args.batch_size,
        skip_ppl=False, include_atomic_gen=False, records=records,
    )

    forget_cases = [
        c for r in forget_records
        for c in mquake.expand_prediction_cases(r, tok, llama_like=llama_like, prompt_types=("rewrite",))
    ]
    retain_cases = [
        c for r in retain_records
        for c in mquake.expand_prediction_cases(r, tok, llama_like=llama_like, prompt_types=("rewrite",))
    ]

    base_forget = cache_base_forget(
        model, tok, forget_cases, device=device, llama_like=llama_like, batch_size=args.batch_size
    )
    active_rows = [r for r in base_forget if r["correct"]]
    row_ids = sorted({r["target_id"] for r in active_rows})
    if not row_ids:
        raise RuntimeError("Base already has Eff=0")

    output_layer = active_repair.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    base_rows = output_weight.index_select(0, row_tensor).detach().clone()
    hidden_size = int(output_weight.shape[1])

    forget_fixed = cache_fixed_competitors(
        model, tok, forget_cases, edited_row_ids=row_ids, device=device,
        llama_like=llama_like, batch_size=args.batch_size,
    )
    constraints = defaultdict(list)
    row_set = set(row_ids)
    gap_values = []
    for r in forget_fixed:
        row_id = r["target_id"]
        if row_id not in row_set:
            continue
        bound = r["fixed_competitor_logit"] - r["target_logit"] - args.forget_margin
        gap_values.append(r["target_logit"] - r["fixed_competitor_logit"])
        qp._add(constraints, row_id, r["hidden"], bound, "forget_below_fixed_unedited")

    retain = cache_retain(
        model, tok, retain_cases, row_ids=row_ids, device=device,
        llama_like=llama_like, batch_size=args.batch_size,
    )
    Hret = retain["hidden"]
    yret = retain["target_ids"]
    base_correct = retain["correct"]

    for row_id in row_ids:
        mask = base_correct & (yret == row_id)
        for h in Hret[mask]:
            qp._add(constraints, row_id, -h, 0.0, "retain_target_nondecrease")

    ppl_text = load_official_ppl_text(Path(args.wikidata_dir))
    if ppl_text is None:
        raise FileNotFoundError("Official wikidata PPL data is required")
    for h, target_id in ppl_edited_target_states(
        model, tok, ppl_text, edited_row_ids=row_ids, device=device
    ):
        qp._add(constraints, target_id, -h, 0.0, "ppl_target_nondecrease")

    cutting_log = []
    deltas = reports = None
    for round_index in range(args.retain_cutting_rounds + 1):
        deltas, reports = qp._solve_rows(
            constraints, row_ids, hidden_size,
            max_epochs=args.qp_max_epochs, tol=args.qp_tol,
        )
        bad = sorted(
            [(rid, reports[rid]["max_violation"]) for rid in row_ids if not reports[rid]["converged"]],
            key=lambda x: x[1], reverse=True,
        )
        if bad:
            (outdir / "qp_failure.json").write_text(
                json.dumps({"bad_rows": bad[:50]}, indent=2) + "\n", encoding="utf-8"
            )
            raise RuntimeError(f"Top2 row QP did not converge: {bad[:10]}")

        D = torch.stack([deltas[rid] for rid in row_ids])
        corrections = Hret @ D.T
        upper = retain["target_logits"][:, None] - retain["selected_logits"]
        violations = (corrections > upper + args.qp_tol) & base_correct[:, None]
        for col, rid in enumerate(row_ids):
            violations[:, col] &= yret != rid
        idxs = violations.nonzero(as_tuple=False)
        cutting_log.append({"round": round_index, "violations": int(idxs.shape[0])})
        if idxs.numel() == 0:
            break
        for retain_index, col in idxs.tolist():
            rid = row_ids[col]
            qp._add(
                constraints, rid, Hret[retain_index], float(upper[retain_index, col]),
                "retain_competitor_upper",
            )
    else:
        raise RuntimeError("Retain cutting-plane rounds exhausted")

    assert deltas is not None
    delta = qp._materialize(output_weight, row_ids, base_rows, deltas)

    candidate = mquake.evaluate_loaded_model_official(
        method="Exact top2 sparse repair", model=model, tok=tok,
        model_dir="in-memory:exact-top2", mquake_path=mquake_path,
        wikidata_dir=Path(args.wikidata_dir), out_path=outdir / "candidate_eval.json",
        forget_num=args.forget_num, retain_num=args.retain_num, seed=args.seed,
        batch_size=args.batch_size, skip_ppl=False, include_atomic_gen=False,
        records=records,
    )

    thresholds = {
        "Eff_max": 0.0,
        "RetainEff_min": float(base["retain"]["Eff"]) - 0.10,
        "PPL_max": float(base["forget_PPL"]) * 1.02,
    }
    checks = {
        "Eff_zero": float(candidate["forget"]["Eff"]) == 0.0,
        "RetainEff_preserved": float(candidate["retain"]["Eff"]) >= thresholds["RetainEff_min"],
        "PPL_preserved": float(candidate["forget_PPL"]) <= thresholds["PPL_max"],
    }
    summary = {
        "method": "Exact top2 sparse repair",
        "start_checkpoint": "Base",
        "modified_row_count": len(row_ids),
        "active_forget_tokens_before": len(active_rows),
        "constraint": "sensitive target below best fixed unedited competitor",
        "forget_margin": args.forget_margin,
        "base_target_minus_fixed_competitor_gap": {
            "min": min(gap_values) if gap_values else None,
            "max": max(gap_values) if gap_values else None,
            "mean": sum(gap_values) / len(gap_values) if gap_values else None,
        },
        "cutting_plane": cutting_log,
        "delta_norm": float(delta.norm()),
        "base": {"Eff": base["forget"]["Eff"], "RetainEff": base["retain"]["Eff"], "PPL": base["forget_PPL"]},
        "candidate": {"Eff": candidate["forget"]["Eff"], "RetainEff": candidate["retain"]["Eff"], "PPL": candidate["forget_PPL"]},
        "thresholds": thresholds,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    (outdir / "repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("BASE:", summary["base"])
    print("GAP:", summary["base_target_minus_fixed_competitor_gap"])
    print("CANDIDATE:", summary["candidate"])
    print("CHECKS:", checks)
    if not all(checks.values()):
        raise RuntimeError("Exact top2 sparse repair failed fixed acceptance gates")

    ckpt = outdir / "selected_checkpoint"
    model.save_pretrained(ckpt, safe_serialization=True)
    tok.save_pretrained(ckpt)
    print(f"ACCEPTED: {ckpt}")


if __name__ == "__main__":
    main()
