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
from mcf_zero_unlearn_official_eval import dtype_from_str, load_official_ppl_text


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.no_grad()
def _cache_cases(model, tok, cases, *, neutral_id, device, llama_like, batch_size):
    rows = []
    for batch in tqdm(list(_chunks(list(cases), batch_size)), desc="cache forget", leave=False):
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
        neutral_logits = logits[:, neutral_id]
        for i, case in enumerate(batch):
            rows.append({
                "case": case,
                "hidden": hidden[i].detach().cpu(),
                "target_id": int(target_ids[i]),
                "pred_id": int(predicted[i]),
                "target_logit": float(target_logits[i]),
                "neutral_logit": float(neutral_logits[i]),
                "correct": bool(int(target_ids[i]) == int(predicted[i])),
            })
    return rows


@torch.no_grad()
def _retain_cache(model, tok, cases, *, row_ids, device, llama_like, batch_size):
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
def _ppl_states(model, tok, text, *, device):
    encoded = tok(
        [text],
        return_tensors="pt",
        max_length=100,
        truncation=True,
    ).to(device)
    output = model(**encoded, output_hidden_states=True, use_cache=False)
    return (
        output.hidden_states[-1][0, :-1, :].float().cpu(),
        encoded["input_ids"][0, 1:].long().cpu(),
    )


def _hildreth(A: torch.Tensor, b: torch.Tensor, *, max_epochs: int, tol: float):
    """Minimum-norm solution of A x <= b via Hildreth in the constraint Gram space."""
    if A.shape[0] == 0:
        return torch.zeros(A.shape[1], dtype=torch.float32), {
            "epochs": 0,
            "max_violation": 0.0,
            "converged": True,
        }

    A = A.float().contiguous()
    b = b.float().contiguous()

    # The primal optimum lies in span(A^T). Maintain s=A@x=-G@dual
    # in constraint space, so coordinate updates cost O(m) instead of O(d).
    G = A @ A.T
    diag = G.diag().clamp_min(1e-12)
    dual = torch.zeros(A.shape[0], dtype=torch.float32)
    s = torch.zeros(A.shape[0], dtype=torch.float32)

    epoch = 0
    for epoch in range(1, max_epochs + 1):
        for i in range(A.shape[0]):
            violation = s[i] - b[i]
            new_dual = torch.clamp(dual[i] + violation / diag[i], min=0.0)
            change = new_dual - dual[i]
            if float(change) != 0.0:
                s.add_(G[:, i], alpha=-float(change))
                dual[i] = new_dual

        if epoch == 1 or epoch % 10 == 0:
            max_violation = float((s - b).max())
            if max_violation <= tol:
                break

    max_violation = float((s - b).max())
    x = -(A.T @ dual)
    return x, {
        "epochs": epoch,
        "max_violation": max_violation,
        "converged": bool(max_violation <= tol),
        "active_duals": int((dual > 0).sum()),
    }


def _add(constraints, row_id: int, a: torch.Tensor, b: float, kind: str):
    constraints[row_id].append((a.float().clone(), float(b), kind))


def _solve_rows(constraints, row_ids, hidden_size, *, max_epochs, tol):
    deltas = {}
    reports = {}
    for row_id in tqdm(row_ids, desc="solve row QPs"):
        values = constraints[row_id]
        A = torch.stack([v[0] for v in values]) if values else torch.empty((0, hidden_size))
        b = torch.tensor([v[1] for v in values], dtype=torch.float32)
        delta, report = _hildreth(A, b, max_epochs=max_epochs, tol=tol)
        report["constraint_count"] = len(values)
        report["constraint_kinds"] = {
            kind: sum(1 for _, _, k in values if k == kind)
            for kind in sorted({k for _, _, k in values})
        }
        deltas[row_id] = delta
        reports[row_id] = report
    return deltas, reports


@torch.no_grad()
def _materialize(output_weight, row_ids, base_rows, deltas):
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    delta = torch.stack([deltas[row_id] for row_id in row_ids]).to(
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    output_weight.index_copy_(0, ids, base_rows + delta)
    return delta.float().cpu()


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic MQuAKE exact sparse sensitive-row suppression."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mquake-path", default="data/MQuAKE-CF-3k-v2.json")
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forget-num", type=int, default=1000)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--forget-margin", type=float, default=0.10)
    parser.add_argument("--qp-max-epochs", type=int, default=5000)
    parser.add_argument("--qp-tol", type=float, default=1e-5)
    parser.add_argument("--retain-cutting-rounds", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Direct MQuAKE evaluator uses sum(attention_mask)-1, therefore right padding.
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
    neutral_id = mquake.resolve_neutral_target_token_id(tok)

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
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    retain_cases = [
        case
        for record in retain_records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]

    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight

    forget_cache = _cache_cases(
        model,
        tok,
        forget_cases,
        neutral_id=neutral_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.batch_size,
    )
    active_forget = [row for row in forget_cache if row["correct"]]
    # To guarantee Eff=0 after simultaneously editing many rows, constrain every
    # evaluated sensitive target row, not only rows that are active at Base.
    row_ids = sorted({
        row["target_id"]
        for row in forget_cache
        if row["target_id"] != neutral_id
    })
    if not row_ids:
        raise RuntimeError("No sensitive target rows found")

    row_set = set(row_ids)
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    base_rows = output_weight.index_select(0, row_tensor).detach().clone()
    hidden_size = int(output_weight.shape[1])
    constraints = defaultdict(list)

    # Exact efficacy constraints for every evaluated forget token. Unknown is
    # fixed, so satisfying these inequalities directly implies Eff=0:
    # sensitive_logit_after <= Unknown_logit - margin.
    for row in forget_cache:
        token_id = row["target_id"]
        if token_id not in row_set:
            continue
        h = row["hidden"]
        bound = row["neutral_logit"] - row["target_logit"] - args.forget_margin
        _add(constraints, token_id, h, bound, "forget_below_unknown")

    retain = _retain_cache(
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

    # If an edited row is itself a correct retain target, never lower it.
    for row_id in row_ids:
        mask = base_correct & (yret == row_id)
        for h in Hret[mask]:
            _add(constraints, row_id, -h, 0.0, "retain_target_nondecrease")

    # PPL guarantee on the exact official 100-token evaluation sequence:
    # edited correct token cannot decrease; every edited competitor cannot increase.
    ppl_text = load_official_ppl_text(Path(args.wikidata_dir))
    if ppl_text is None:
        raise FileNotFoundError("Official wikidata PPL data is required")
    Hppl, yppl = _ppl_states(model, tok, ppl_text, device=device)
    for row_id in row_ids:
        same = yppl == row_id
        for h in Hppl[same]:
            _add(constraints, row_id, -h, 0.0, "ppl_target_nondecrease")
        for h in Hppl[~same]:
            _add(constraints, row_id, h, 0.0, "ppl_competitor_nonincrease")

    cutting_log = []
    deltas = None
    reports = None

    # Deterministic retain cutting plane. Add only constraints that could flip
    # a Base-correct retain decision.
    for round_index in range(args.retain_cutting_rounds + 1):
        deltas, reports = _solve_rows(
            constraints,
            row_ids,
            hidden_size,
            max_epochs=args.qp_max_epochs,
            tol=args.qp_tol,
        )
        bad_qp = [
            (row_id, reports[row_id]["max_violation"])
            for row_id in row_ids
            if not reports[row_id]["converged"]
        ]
        if bad_qp:
            bad_qp.sort(key=lambda item: item[1], reverse=True)
            raise RuntimeError(f"Row QP did not converge: {bad_qp[:10]}")

        D = torch.stack([deltas[row_id] for row_id in row_ids])
        corrections = Hret @ D.T
        upper = retain["target_logits"][:, None] - retain["selected_logits"]

        violations = (corrections > upper + args.qp_tol) & base_correct[:, None]
        for column, row_id in enumerate(row_ids):
            violations[:, column] &= yret != row_id

        indices = violations.nonzero(as_tuple=False)
        cutting_log.append({
            "round": round_index,
            "retain_competitor_violations": int(indices.shape[0]),
        })
        if indices.numel() == 0:
            break

        added = 0
        seen = set()
        for row_id in row_ids:
            for a, b, kind in constraints[row_id]:
                if kind == "retain_competitor_upper":
                    seen.add((row_id, round(float(b), 6), a.numpy().tobytes()))

        for retain_index, column in indices.tolist():
            row_id = row_ids[column]
            h = Hret[retain_index]
            bound = float(upper[retain_index, column])
            key = (row_id, round(bound, 6), h.numpy().tobytes())
            if key in seen:
                continue
            _add(constraints, row_id, h, bound, "retain_competitor_upper")
            seen.add(key)
            added += 1
        cutting_log[-1]["constraints_added"] = added
        if added == 0:
            raise RuntimeError("Retain cutting plane stalled")
    else:
        raise RuntimeError("Retain cutting-plane rounds exhausted")

    assert deltas is not None and reports is not None
    materialized_delta = _materialize(
        output_weight,
        row_ids,
        base_rows,
        deltas,
    )

    candidate = mquake.evaluate_loaded_model_official(
        method="Exact sparse sensitive-row suppression",
        model=model,
        tok=tok,
        model_dir="in-memory:exact-sparse-repair",
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
        "RetainEff_preserved": (
            float(candidate["retain"]["Eff"]) >= thresholds["RetainEff_min"]
        ),
        "PPL_preserved": (
            float(candidate["forget_PPL"]) <= thresholds["PPL_max"]
        ),
    }

    summary = {
        "method": "Exact sparse sensitive-row suppression",
        "start_checkpoint": "Base",
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "unknown_row_modified": False,
        "modified_row_count": len(row_ids),
        "active_forget_tokens_before": len(active_forget),
        "forget_margin": args.forget_margin,
        "cutting_plane": cutting_log,
        "delta_norm": float(materialized_delta.norm()),
        "base": {
            "Eff": base["forget"]["Eff"],
            "RetainEff": base["retain"]["Eff"],
            "PPL": base["forget_PPL"],
        },
        "candidate": {
            "Eff": candidate["forget"]["Eff"],
            "RetainEff": candidate["retain"]["Eff"],
            "PPL": candidate["forget_PPL"],
        },
        "thresholds": thresholds,
        "checks": checks,
        "accepted": all(checks.values()),
    }
    (output_dir / "repair_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("BASE:", summary["base"])
    print("CANDIDATE:", summary["candidate"])
    print("CHECKS:", checks)

    if not all(checks.values()):
        raise RuntimeError("Exact sparse repair failed fixed acceptance gates")

    checkpoint = output_dir / "selected_checkpoint"
    model.save_pretrained(checkpoint, safe_serialization=True)
    tok.save_pretrained(checkpoint)
    print(f"ACCEPTED: {checkpoint}")


if __name__ == "__main__":
    main()
