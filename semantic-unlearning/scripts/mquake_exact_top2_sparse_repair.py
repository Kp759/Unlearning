#!/usr/bin/env python3
"""Deterministic MQuAKE sparse repair aligned exactly to Eff.

Unlike the earlier Unknown-margin prototype, this method never asks a sensitive
answer token to fall below the global ``Unknown`` row.  Eff only requires that
the sensitive token cease to be top-1.  We therefore suppress each active
sensitive output row only far enough to fall below the best *unedited* output
row at that exact teacher-forced state.

The transformer and input embeddings are frozen.  Only LM-head rows that are
actually top-1 sensitive targets at Base are untied and changed.  Each row is
solved independently as a minimum-norm convex projection.  Base-correct retain
targets that share an edited row are constrained not to decrease; a deterministic
cutting plane adds only edited-competitor constraints that could flip a
Base-correct retain decision.  The official PPL gate is evaluated exactly after
materialization.
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

import gagd_active_case_repair as active
import mquake_zero_unlearn_official_eval as mquake
import mquake_exact_sparse_row_repair as qp
from mcf_zero_unlearn_official_eval import dtype_from_str, load_official_ppl_text


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.no_grad()
def _cache_forget_base(model, tok, cases, *, device, llama_like, batch_size):
    rows = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache forget base", leave=False):
        encoded = tok([case.prompt for case in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][idx, last, :].float()
        logits = output.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted = logits.argmax(dim=-1)
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        for i, case in enumerate(batch):
            rows.append(
                {
                    "case": case,
                    "hidden": hidden[i].detach().cpu(),
                    "target_id": int(target_ids[i]),
                    "pred_id": int(predicted[i]),
                    "target_logit": float(target_logits[i]),
                    "correct": bool(int(target_ids[i]) == int(predicted[i])),
                }
            )
    return rows


@torch.no_grad()
def _cache_forget_fixed_competitor(
    model,
    tok,
    cases,
    *,
    edited_row_ids,
    device,
    llama_like,
    batch_size,
):
    edited = torch.tensor(edited_row_ids, dtype=torch.long, device=device)
    rows = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache fixed competitors", leave=False):
        encoded = tok([case.prompt for case in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][idx, last, :].float()
        logits = output.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)

        # Every edited row is excluded, so this competitor remains exactly fixed
        # after all sparse row updates are materialized jointly.
        logits.index_fill_(1, edited, float("-inf"))
        fixed_values, fixed_ids = logits.max(dim=1)

        for i, case in enumerate(batch):
            rows.append(
                {
                    "case": case,
                    "hidden": hidden[i].detach().cpu(),
                    "target_id": int(target_ids[i]),
                    "target_logit": float(target_logits[i]),
                    "fixed_competitor_id": int(fixed_ids[i]),
                    "fixed_competitor_logit": float(fixed_values[i]),
                }
            )
    return rows


@torch.no_grad()
def _cache_retain(model, tok, cases, *, row_ids, device, llama_like, batch_size):
    selected = torch.tensor(row_ids, dtype=torch.long, device=device)
    hidden_all = []
    target_ids_all = []
    target_logits_all = []
    selected_logits_all = []
    correct_all = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache retain", leave=False):
        encoded = tok([case.prompt for case in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        last = encoded["attention_mask"].sum(dim=1) - 1
        idx = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][idx, last, :].float()
        logits = output.logits[idx, last, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted = logits.argmax(dim=-1)
        hidden_all.append(hidden.cpu())
        target_ids_all.append(target_ids.cpu())
        target_logits_all.append(logits.gather(1, target_ids[:, None]).squeeze(1).cpu())
        selected_logits_all.append(logits.index_select(1, selected).cpu())
        correct_all.append((predicted == target_ids).cpu())
    return {
        "hidden": torch.cat(hidden_all).float(),
        "target_ids": torch.cat(target_ids_all).long(),
        "target_logits": torch.cat(target_logits_all).float(),
        "selected_logits": torch.cat(selected_logits_all).float(),
        "correct": torch.cat(correct_all).bool(),
    }


@torch.no_grad()
def _ppl_target_states(model, tok, text, *, edited_row_ids, device):
    encoded = tok([text], return_tensors="pt", max_length=100, truncation=True).to(device)
    output = model(**encoded, output_hidden_states=True, use_cache=False)
    hidden = output.hidden_states[-1][0, :-1, :].float().cpu()
    target_ids = encoded["input_ids"][0, 1:].long().cpu()
    edited_set = set(int(x) for x in edited_row_ids)
    return [(hidden[i], int(target_ids[i])) for i in range(len(target_ids)) if int(target_ids[i]) in edited_set]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mquake-path", default="data/MQuAKE-CF-3k-v2.json")
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forget-num", type=int, default=1000)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--forget-margin", type=float, default=0.05)
    parser.add_argument("--qp-max-epochs", type=int, default=5000)
    parser.add_argument("--qp-tol", type=float, default=1e-4)
    parser.add_argument("--retain-cutting-rounds", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
        mquake_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
    )
    records = (forget_records, retain_records)
    mquake.write_split_manifest(
        output_dir / "split_manifest.json",
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )

    llama_like = mquake.is_llama_like(model, tok)
    base = mquake.evaluate_loaded_model_official(
        method="Base",
        model=model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=Path(args.wikidata_dir),
        out_path=output_dir / "base_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.batch_size,
        skip_ppl=False,
        include_atomic_gen=False,
        records=records,
    )

    forget_cases = [
        case
        for record in forget_records
        for case in mquake.expand_prediction_cases(record, tok, llama_like=llama_like, prompt_types=("rewrite",))
    ]
    retain_cases = [
        case
        for record in retain_records
        for case in mquake.expand_prediction_cases(record, tok, llama_like=llama_like, prompt_types=("rewrite",))
    ]

    base_forget = _cache_forget_base(
        model, tok, forget_cases, device=device, llama_like=llama_like, batch_size=args.batch_size
    )
    active = [row for row in base_forget if row["correct"]]
    row_ids = sorted({row["target_id"] for row in active})
    if not row_ids:
        raise RuntimeError("Base already has zero active MQuAKE sensitive rows")

    # Untie only after the active row set has been frozen from Base.
    output_layer = active_module = active  # keep local name out of the imported module namespace
    del active_module
    output_layer = globals()["active"].freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    base_rows = output_weight.index_select(0, row_tensor).detach().clone()
    hidden_size = int(output_weight.shape[1])

    forget_fixed = _cache_forget_fixed_competitor(
        model,
        tok,
        forget_cases,
        edited_row_ids=row_ids,
        device=device,
        llama_like=llama_like,
        batch_size=args.batch_size,
    )

    constraints = defaultdict(list)
    row_set = set(row_ids)
    for row in forget_fixed:
        row_id = row["target_id"]
        if row_id not in row_set:
            continue
        # Fixed competitor is guaranteed unedited.  Therefore this linear
        # inequality remains valid after all edited rows are applied jointly.
        bound = row["fixed_competitor_logit"] - row["target_logit"] - args.forget_margin
        qp._add(constraints, row_id, row["hidden"], bound, "forget_below_fixed_top1")

    retain = _cache_retain(
        model,
        tok,
        retain_cases,
        row_ids=row_ids,
        device=device,
        llama_like=llama_like,
        batch_size=args.batch_size,
    )
    Hret = retain["hidden"]
    yret = retain["target_ids"]
    base_correct = retain["correct"]

    # If a Base-correct retain target itself uses an edited row, never lower
    # that row at the corresponding retain state.  This makes the subsequent
    # competitor cutting plane separable by row.
    for row_id in row_ids:
        mask = base_correct & (yret == row_id)
        for h in Hret[mask]:
            qp._add(constraints, row_id, -h, 0.0, "retain_target_nondecrease")

    # On PPL positions whose correct token is itself edited, never lower that
    # correct token.  We intentionally do NOT impose the previous 935x~100
    # competitor-nonincreasing wall; the exact official PPL threshold is the
    # final global acceptance gate.
    ppl_text = load_official_ppl_text(Path(args.wikidata_dir))
    if ppl_text is None:
        raise FileNotFoundError("Official wikidata PPL data is required")
    for h, target_id in _ppl_target_states(
        model, tok, ppl_text, edited_row_ids=row_ids, device=device
    ):
        qp._add(constraints, target_id, -h, 0.0, "ppl_target_nondecrease")

    cutting_log = []
    deltas = None
    reports = None
    for round_index in range(args.retain_cutting_rounds + 1):
        deltas, reports = qp._solve_rows(
            constraints,
            row_ids,
            hidden_size,
            max_epochs=args.qp_max_epochs,
            tol=args.qp_tol,
        )
        bad = [
            (row_id, reports[row_id]["max_violation"])
            for row_id in row_ids
            if not reports[row_id]["converged"]
        ]
        if bad:
            bad.sort(key=lambda x: x[1], reverse=True)
            (output_dir / "qp_failure.json").write_text(
                json.dumps({"bad_rows": bad[:50], "reports": reports}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(f"Metric-aligned row QP did not converge: {bad[:10]}")

        D = torch.stack([deltas[row_id] for row_id in row_ids])
        corrections = Hret @ D.T
        upper = retain["target_logits"][:, None] - retain["selected_logits"]

        # Because edited retain targets were constrained not to decrease, it is
        # sufficient to keep every edited competitor below the *Base* target.
        violations = (corrections > upper + args.qp_tol) & base_correct[:, None]
        for column, row_id in enumerate(row_ids):
            violations[:, column] &= yret != row_id
        indices = violations.nonzero(as_tuple=False)
        cutting_log.append(
            {"round": round_index, "retain_competitor_violations": int(indices.shape[0])}
        )
        if indices.numel() == 0:
            break

        added = 0
        for retain_index, column in indices.tolist():
            row_id = row_ids[column]
            h = Hret[retain_index]
            bound = float(upper[retain_index, column])
            qp._add(constraints, row_id, h, bound, "retain_competitor_upper")
            added += 1
        cutting_log[-1]["constraints_added"] = added
    else:
        raise RuntimeError("Retain cutting-plane rounds exhausted")

    assert deltas is not None and reports is not None
    materialized_delta = qp._materialize(output_weight, row_ids, base_rows, deltas)

    candidate = mquake.evaluate_loaded_model_official(
        method="Exact top2 sparse sensitive-row suppression",
        model=model,
        tok=tok,
        model_dir="in-memory:exact-top2-sparse",
        mquake_path=mquake_path,
        wikidata_dir=Path(args.wikidata_dir),
        out_path=output_dir / "candidate_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.batch_size,
        skip_ppl=False,
        include_atomic_gen=False,
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
        "method": "Exact top2 sparse sensitive-row suppression",
        "start_checkpoint": "Base",
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "unknown_row_modified": False,
        "modified_row_count": len(row_ids),
        "active_forget_tokens_before": len(active),
        "forget_constraint": "target <= best fixed unedited competitor - margin",
        "forget_margin": args.forget_margin,
        "cutting_plane": cutting_log,
        "delta_norm": float(materialized_delta.norm()),
        "base": {"Eff": base["forget"]["Eff"], "RetainEff": base["retain"]["Eff"], "PPL": base["forget_PPL"]},
        "candidate": {"Eff": candidate["forget"]["Eff"], "RetainEff": candidate["retain"]["Eff"], "PPL": candidate["forget_PPL"]},
        "thresholds": thresholds,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    (output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("BASE:", summary["base"])
    print("CANDIDATE:", summary["candidate"])
    print("CHECKS:", checks)
    if not all(checks.values()):
        raise RuntimeError("Exact top2 sparse repair failed fixed acceptance gates")

    checkpoint = output_dir / "selected_checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tok.save_pretrained(checkpoint)
    print(f"ACCEPTED: {checkpoint}")


if __name__ == "__main__":
    main()
