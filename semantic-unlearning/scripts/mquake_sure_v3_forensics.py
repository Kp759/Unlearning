#!/usr/bin/env python3
"""Read-only forensic analysis of MQuAKE SURE v3.

This script does not train, repair, or modify any checkpoint. It compares the
base model, the v3 Stage-1 checkpoint, and the v3 final Stage-2 checkpoint on
one fixed official MQuAKE split and asks two questions:

1. Why does held-out AtomicGen remain non-zero after direct Eff reaches zero?
   We inspect token position, direct-vs-question margins, paired hidden-state
   cosine similarity, and base->Stage1->Final target-logit shifts.

2. Why does retain Eff (specificity) fall while PPL stays near base?
   We measure the exact LM-head rows changed by v3 and classify every
   base-correct -> final-wrong retain token by whether its correct target row,
   its final winning competitor row, both, or neither were edited.

AtomicGen is used here only as post-hoc held-out diagnosis. No output of this
script should be used as training data or checkpoint-selection input.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    is_llama_like,
    load_official_ppl_text,
    official_perplexity,
)
import mquake_zero_unlearn_official_eval as official


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--stage1-model-path", required=True)
    p.add_argument("--final-model-path", required=True)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--with-ppl", action="store_true")
    return p.parse_args()


def chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_model(path: str, dtype_name: str, device_map: str):
    dtype = dtype_from_str(dtype_name)
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("--device-map single requires CUDA")
        model = model.to(torch.device("cuda:0"))
    model.eval()
    model.config.use_cache = False
    return model


def first_device(model) -> torch.device:
    return next(model.parameters()).device


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def percentile_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


@torch.no_grad()
def score_cases(
    model,
    tok,
    cases: Sequence[official.PredictionCase],
    *,
    llama_like: bool,
    batch_size: int,
    include_hidden: bool,
) -> List[Dict[str, Any]]:
    device = first_device(model)
    rows: List[Dict[str, Any]] = []
    for batch in chunks(list(cases), batch_size):
        encoded = tok(
            [case.prompt for case in batch],
            padding=True,
            return_tensors="pt",
        ).to(device)
        out = model(
            **encoded,
            output_hidden_states=include_hidden,
            use_cache=False,
        )
        last = encoded["attention_mask"].sum(dim=1) - 1
        rr = torch.arange(len(batch), device=device)
        logits = out.logits[rr, last, :].float()
        target_ids = official.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        ).long()
        target_logits = logits[rr, target_ids]
        other = logits.clone()
        other[rr, target_ids] = -torch.inf
        winner_logits, winner_ids = other.max(dim=-1)
        predicted_ids = logits.argmax(dim=-1)
        margins = winner_logits - target_logits
        hidden = None
        if include_hidden:
            hidden = out.hidden_states[-1][rr, last, :].float().detach().cpu()

        for j, case in enumerate(batch):
            row = {
                "case_id": int(case.case_id),
                "prompt_type": str(case.prompt_type),
                "prompt_index": int(case.prompt_index),
                "token_index": int(case.token_index),
                "prompt": str(case.prompt),
                "target_text": str(case.target_text),
                "target_token_id": int(target_ids[j].item()),
                "predicted_token_id": int(predicted_ids[j].item()),
                "winner_other_token_id": int(winner_ids[j].item()),
                "correct": bool(predicted_ids[j].item() == target_ids[j].item()),
                # Positive means some other token beats the sensitive/correct target.
                "forget_margin": float(margins[j].item()),
                "target_logit": float(target_logits[j].item()),
                "winner_other_logit": float(winner_logits[j].item()),
            }
            if hidden is not None:
                row["hidden"] = hidden[j]
            rows.append(row)
    return rows


def row_key(row: Mapping[str, Any]) -> Tuple[int, str, int, int]:
    return (
        int(row["case_id"]),
        str(row["prompt_type"]),
        int(row.get("prompt_index", 0)),
        int(row["token_index"]),
    )


def pair_key(row: Mapping[str, Any]) -> Tuple[int, int, int]:
    return (
        int(row["case_id"]),
        int(row.get("prompt_index", 0)),
        int(row["token_index"]),
    )


def macro_metric(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    prompt_type: str,
) -> Optional[float]:
    grouped: Dict[int, List[float]] = {int(r["case_id"]): [] for r in records}
    for row in rows:
        if str(row["prompt_type"]) == prompt_type:
            grouped[int(row["case_id"])].append(float(bool(row["correct"])))
    vals = [float(np.mean(v)) for v in grouped.values() if v]
    return None if not vals else 100.0 * float(np.mean(vals))


def token_position_summary(
    rows: Sequence[Mapping[str, Any]], prompt_type: str
) -> Dict[str, Any]:
    by_pos: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        if str(row["prompt_type"]) != prompt_type:
            continue
        pos = int(row["token_index"])
        key = "0" if pos == 0 else ("1" if pos == 1 else "2+")
        by_pos.setdefault(key, []).append(row)
    out: Dict[str, Any] = {}
    for key in ("0", "1", "2+"):
        vals = by_pos.get(key, [])
        correct = sum(bool(x["correct"]) for x in vals)
        out[key] = {
            "n": len(vals),
            "sensitive_target_argmax": int(correct),
            "accuracy_percent": None if not vals else 100.0 * correct / len(vals),
            "forget_margin": percentile_summary(
                [float(x["forget_margin"]) for x in vals]
            ),
        }
    return out


def summarize_stage(
    name: str,
    forget_records,
    retain_records,
    forget_rows,
    retain_rows,
    ppl: Optional[float],
) -> Dict[str, Any]:
    return {
        "name": name,
        "Eff": macro_metric(forget_records, forget_rows, "rewrite"),
        "AtomicGen": macro_metric(forget_records, forget_rows, "atomic_gen"),
        "RetainEff": macro_metric(retain_records, retain_rows, "rewrite"),
        "PPL": ppl,
    }


def exact_changed_rows(current: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if current.shape != base.shape:
        raise RuntimeError("output-head shape mismatch")
    return (current != base).any(dim=1)


def delta_row_norms(
    current: torch.Tensor, base: torch.Tensor, block: int = 4096
) -> torch.Tensor:
    out = torch.zeros(current.shape[0], dtype=torch.float32)
    for start in range(0, current.shape[0], block):
        stop = min(start + block, current.shape[0])
        diff = current[start:stop].float() - base[start:stop].float()
        out[start:stop] = diff.norm(dim=1)
    return out


def frobenius_from_row_norms(row_norms: torch.Tensor) -> float:
    return float(torch.sqrt((row_norms.double().square()).sum()).item())


def evaluate_ppl(model, tok, wikidata_dir: Path) -> Optional[float]:
    text = load_official_ppl_text(wikidata_dir)
    if text is None:
        return None
    return float(
        official_perplexity(
            model,
            tok,
            text,
            first_device(model),
            max_input_length=100,
        )
    )


def matched(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, str, int, int], Mapping[str, Any]]:
    return {row_key(r): r for r in rows}


def paired_gen_analysis(
    base_rows,
    stage1_rows,
    final_rows,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    st1 = matched(stage1_rows)
    final_rewrite = {
        pair_key(r): r for r in final_rows if r["prompt_type"] == "rewrite"
    }
    final_gen = {
        pair_key(r): r for r in final_rows if r["prompt_type"] == "atomic_gen"
    }
    base_rewrite = {
        pair_key(r): r for r in base_rows if r["prompt_type"] == "rewrite"
    }
    base_gen = {
        pair_key(r): r for r in base_rows if r["prompt_type"] == "atomic_gen"
    }

    paired_rows: List[Dict[str, Any]] = []
    for pk in sorted(set(final_rewrite) & set(final_gen)):
        d = final_rewrite[pk]
        g = final_gen[pk]
        bd = base_rewrite[pk]
        bg = base_gen[pk]
        sd = st1[row_key(d)]
        sg = st1[row_key(g)]

        h_d = d.get("hidden")
        h_g = g.get("hidden")
        hb_d = bd.get("hidden")
        hb_g = bg.get("hidden")
        cos_final = None
        cos_base = None
        if isinstance(h_d, torch.Tensor) and isinstance(h_g, torch.Tensor):
            cos_final = float(
                torch.nn.functional.cosine_similarity(h_d[None], h_g[None]).item()
            )
        if isinstance(hb_d, torch.Tensor) and isinstance(hb_g, torch.Tensor):
            cos_base = float(
                torch.nn.functional.cosine_similarity(hb_d[None], hb_g[None]).item()
            )

        paired_rows.append(
            {
                "case_id": pk[0],
                "prompt_index": pk[1],
                "token_index": pk[2],
                "target_token_id": int(g["target_token_id"]),
                "atomicgen_sensitive_recovered": bool(g["correct"]),
                "final_direct_margin": float(d["forget_margin"]),
                "final_atomicgen_margin": float(g["forget_margin"]),
                "margin_gap_atomicgen_minus_direct": float(g["forget_margin"])
                - float(d["forget_margin"]),
                "base_hidden_cosine_direct_atomicgen": cos_base,
                "final_hidden_cosine_direct_atomicgen": cos_final,
                "direct_target_logit_shift_base_to_stage1": float(sd["target_logit"])
                - float(bd["target_logit"]),
                "direct_target_logit_shift_stage1_to_final": float(d["target_logit"])
                - float(sd["target_logit"]),
                "direct_target_logit_shift_base_to_final": float(d["target_logit"])
                - float(bd["target_logit"]),
                "gen_target_logit_shift_base_to_stage1": float(sg["target_logit"])
                - float(bg["target_logit"]),
                "gen_target_logit_shift_stage1_to_final": float(g["target_logit"])
                - float(sg["target_logit"]),
                "gen_target_logit_shift_base_to_final": float(g["target_logit"])
                - float(bg["target_logit"]),
            }
        )

    recovered = [r for r in paired_rows if r["atomicgen_sensitive_recovered"]]
    forgotten = [r for r in paired_rows if not r["atomicgen_sensitive_recovered"]]

    def group_summary(rows):
        return {
            "n": len(rows),
            "final_atomicgen_margin": percentile_summary(
                [r["final_atomicgen_margin"] for r in rows]
            ),
            "final_direct_margin": percentile_summary(
                [r["final_direct_margin"] for r in rows]
            ),
            "margin_gap_atomicgen_minus_direct": percentile_summary(
                [r["margin_gap_atomicgen_minus_direct"] for r in rows]
            ),
            "base_hidden_cosine": percentile_summary(
                [
                    r["base_hidden_cosine_direct_atomicgen"]
                    for r in rows
                    if r["base_hidden_cosine_direct_atomicgen"] is not None
                ]
            ),
            "final_hidden_cosine": percentile_summary(
                [
                    r["final_hidden_cosine_direct_atomicgen"]
                    for r in rows
                    if r["final_hidden_cosine_direct_atomicgen"] is not None
                ]
            ),
            "direct_target_logit_shift_base_to_final": percentile_summary(
                [r["direct_target_logit_shift_base_to_final"] for r in rows]
            ),
            "gen_target_logit_shift_base_to_final": percentile_summary(
                [r["gen_target_logit_shift_base_to_final"] for r in rows]
            ),
            "gen_target_logit_shift_stage1_to_final": percentile_summary(
                [r["gen_target_logit_shift_stage1_to_final"] for r in rows]
            ),
        }

    return (
        {
            "all": group_summary(paired_rows),
            "atomicgen_sensitive_recovered": group_summary(recovered),
            "atomicgen_forgotten": group_summary(forgotten),
        },
        paired_rows,
    )


def retain_overlap_analysis(
    base_rows,
    stage1_rows,
    final_rows,
    *,
    stage1_changed: torch.Tensor,
    final_changed: torch.Tensor,
    stage1_row_norms: torch.Tensor,
    final_row_norms: torch.Tensor,
    stage2_incremental_norms: Mapping[int, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    base = matched(base_rows)
    st1 = matched(stage1_rows)
    fin = matched(final_rows)
    common = sorted(set(base) & set(st1) & set(fin))
    details: List[Dict[str, Any]] = []

    for key in common:
        b, s, f = base[key], st1[key], fin[key]
        tid = int(f["target_token_id"])
        win = int(f["predicted_token_id"])
        target_edited = bool(final_changed[tid].item())
        winner_edited = bool(final_changed[win].item())
        base_correct = bool(b["correct"])
        stage1_correct = bool(s["correct"])
        final_correct = bool(f["correct"])
        if base_correct and not final_correct:
            if target_edited and winner_edited:
                failure_class = "both_target_and_winner_edited"
            elif target_edited:
                failure_class = "target_only_edited"
            elif winner_edited:
                failure_class = "winner_only_edited"
            else:
                failure_class = "neither_edited"
        else:
            failure_class = None
        details.append(
            {
                "case_id": int(f["case_id"]),
                "token_index": int(f["token_index"]),
                "target_token_id": tid,
                "final_predicted_token_id": win,
                "base_correct": base_correct,
                "stage1_correct": stage1_correct,
                "final_correct": final_correct,
                "base_to_stage1_lost": base_correct and not stage1_correct,
                "stage1_to_final_lost": stage1_correct and not final_correct,
                "base_to_final_lost": base_correct and not final_correct,
                "base_to_final_gained": (not base_correct) and final_correct,
                "target_row_edited_stage1": bool(stage1_changed[tid].item()),
                "target_row_edited_final": target_edited,
                "winner_row_edited_final": winner_edited,
                "target_stage1_delta_norm": float(stage1_row_norms[tid].item()),
                "target_final_delta_norm": float(final_row_norms[tid].item()),
                "winner_final_delta_norm": float(final_row_norms[win].item()),
                "target_stage2_incremental_norm": float(
                    stage2_incremental_norms.get(tid, 0.0)
                ),
                "winner_stage2_incremental_norm": float(
                    stage2_incremental_norms.get(win, 0.0)
                ),
                "failure_class": failure_class,
            }
        )

    def status_counts(rows):
        return {
            "n": len(rows),
            "base_accuracy_percent": None
            if not rows
            else 100.0 * sum(r["base_correct"] for r in rows) / len(rows),
            "stage1_accuracy_percent": None
            if not rows
            else 100.0 * sum(r["stage1_correct"] for r in rows) / len(rows),
            "final_accuracy_percent": None
            if not rows
            else 100.0 * sum(r["final_correct"] for r in rows) / len(rows),
        }

    target_edited_rows = [r for r in details if r["target_row_edited_final"]]
    target_unedited_rows = [r for r in details if not r["target_row_edited_final"]]
    lost = [r for r in details if r["base_to_final_lost"]]
    gained = [r for r in details if r["base_to_final_gained"]]

    classes: Dict[str, int] = {}
    for r in lost:
        classes[str(r["failure_class"])] = classes.get(str(r["failure_class"]), 0) + 1

    base_correct = [r for r in details if r["base_correct"]]
    x = np.asarray(
        [r["target_final_delta_norm"] for r in base_correct], dtype=np.float64
    )
    y = np.asarray(
        [1.0 if r["base_to_final_lost"] else 0.0 for r in base_correct],
        dtype=np.float64,
    )
    corr = None
    if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
        corr = float(np.corrcoef(x, y)[0, 1])

    return (
        {
            "token_transition_counts": {
                "base_correct_final_wrong": sum(
                    r["base_to_final_lost"] for r in details
                ),
                "base_wrong_final_correct": sum(
                    r["base_to_final_gained"] for r in details
                ),
                "base_correct_stage1_wrong": sum(
                    r["base_to_stage1_lost"] for r in details
                ),
                "stage1_correct_final_wrong": sum(
                    r["stage1_to_final_lost"] for r in details
                ),
            },
            "target_row_overlap": {
                "edited_target": status_counts(target_edited_rows),
                "unedited_target": status_counts(target_unedited_rows),
            },
            "new_failure_classification": classes,
            "new_failure_target_delta_norm": percentile_summary(
                [r["target_final_delta_norm"] for r in lost]
            ),
            "new_failure_winner_delta_norm": percentile_summary(
                [r["winner_final_delta_norm"] for r in lost]
            ),
            "base_correct_preserved_target_delta_norm": percentile_summary(
                [
                    r["target_final_delta_norm"]
                    for r in details
                    if r["base_correct"] and r["final_correct"]
                ]
            ),
            "target_delta_norm_vs_loss_point_biserial": corr,
            "lost_n": len(lost),
            "gained_n": len(gained),
        },
        details,
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> None:
    a = parse_args()
    if min(a.forget_num, a.retain_num, a.batch_size) <= 0:
        raise ValueError("forget/retain counts and batch size must be positive")

    out_dir = Path(a.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.base_model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Load base first so Llama-style target tokenization is determined exactly
    # as in the official evaluator before any PredictionCase is constructed.
    print("===== FORENSICS: LOAD BASE + LOCK SPLIT =====")
    base_model = load_model(a.base_model_path, a.dtype, a.device_map)
    llama_like = is_llama_like(base_model, tok)
    forget_records, retain_records = official.load_official_eval_records(
        Path(a.mquake_path),
        tok,
        forget_num=a.forget_num,
        retain_num=a.retain_num,
        seed=a.seed,
    )
    forget_cases = [
        c
        for r in forget_records
        for c in official.expand_prediction_cases(
            r,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite", "atomic_gen"),
        )
    ]
    retain_cases = [
        c
        for r in retain_records
        for c in official.expand_prediction_cases(
            r,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    print(
        "Forensics split: forget atomic facts={} forget token cases={} "
        "retain atomic facts={} retain token cases={}".format(
            len(forget_records),
            len(forget_cases),
            len(retain_records),
            len(retain_cases),
        )
    )

    # BASE -----------------------------------------------------------------
    print("===== FORENSICS: BASE =====")
    base_forget = score_cases(
        base_model,
        tok,
        forget_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=True,
    )
    base_retain = score_cases(
        base_model,
        tok,
        retain_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=False,
    )
    base_ppl = (
        evaluate_ppl(base_model, tok, Path(a.wikidata_dir)) if a.with_ppl else None
    )
    base_w = base_model.get_output_embeddings().weight.detach().cpu().clone()
    base_stage = summarize_stage(
        "base",
        forget_records,
        retain_records,
        base_forget,
        base_retain,
        base_ppl,
    )
    print(base_stage)
    unload(base_model)

    # The complete set of sensitive target output rows is derived from the
    # official forget cases, independent of whether BF16 materialization made
    # a particular Stage-1 row bitwise different from base.
    sensitive_ids = sorted(
        {int(r["target_token_id"]) for r in base_forget if r["prompt_type"] == "rewrite"}
    )
    sensitive_idx = torch.tensor(sensitive_ids, dtype=torch.long)

    # STAGE 1 ---------------------------------------------------------------
    print("===== FORENSICS: V3 STAGE1 =====")
    stage1_model = load_model(a.stage1_model_path, a.dtype, a.device_map)
    stage1_forget = score_cases(
        stage1_model,
        tok,
        forget_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=False,
    )
    stage1_retain = score_cases(
        stage1_model,
        tok,
        retain_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=False,
    )
    stage1_ppl = (
        evaluate_ppl(stage1_model, tok, Path(a.wikidata_dir))
        if a.with_ppl
        else None
    )
    stage1_w = stage1_model.get_output_embeddings().weight.detach().cpu()
    stage1_changed = exact_changed_rows(stage1_w, base_w)
    stage1_row_norms = delta_row_norms(stage1_w, base_w)
    stage1_sensitive_snapshot = stage1_w.index_select(0, sensitive_idx).clone()
    stage1_stage = summarize_stage(
        "stage1_v3",
        forget_records,
        retain_records,
        stage1_forget,
        stage1_retain,
        stage1_ppl,
    )
    print(stage1_stage)
    del stage1_w
    unload(stage1_model)

    # FINAL -----------------------------------------------------------------
    print("===== FORENSICS: V3 FINAL =====")
    final_model = load_model(a.final_model_path, a.dtype, a.device_map)
    final_forget = score_cases(
        final_model,
        tok,
        forget_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=True,
    )
    final_retain = score_cases(
        final_model,
        tok,
        retain_cases,
        llama_like=llama_like,
        batch_size=a.batch_size,
        include_hidden=False,
    )
    final_ppl = (
        evaluate_ppl(final_model, tok, Path(a.wikidata_dir)) if a.with_ppl else None
    )
    final_w = final_model.get_output_embeddings().weight.detach().cpu()
    final_changed = exact_changed_rows(final_w, base_w)
    final_row_norms = delta_row_norms(final_w, base_w)

    final_sensitive = final_w.index_select(0, sensitive_idx)
    inc = (final_sensitive.float() - stage1_sensitive_snapshot.float()).norm(dim=1)
    stage2_incremental_norms = {
        int(tid): float(v)
        for tid, v in zip(sensitive_ids, inc.tolist())
        if float(v) > 0.0
    }
    stage2_frob = float(inc.double().square().sum().sqrt().item())
    stage2_changed_rows = int((inc > 0).sum().item())

    final_stage = summarize_stage(
        "final_v3",
        forget_records,
        retain_records,
        final_forget,
        final_retain,
        final_ppl,
    )
    print(final_stage)

    gen_summary, gen_pairs = paired_gen_analysis(
        base_forget, stage1_forget, final_forget
    )
    spe_summary, retain_details = retain_overlap_analysis(
        base_retain,
        stage1_retain,
        final_retain,
        stage1_changed=stage1_changed,
        final_changed=final_changed,
        stage1_row_norms=stage1_row_norms,
        final_row_norms=final_row_norms,
        stage2_incremental_norms=stage2_incremental_norms,
    )

    sensitive_set = set(sensitive_ids)
    final_changed_ids = set(
        int(x)
        for x in torch.nonzero(final_changed, as_tuple=False).flatten().tolist()
    )
    head_summary = {
        "sensitive_target_rows": len(sensitive_ids),
        "stage1_changed_rows_exact": int(stage1_changed.sum().item()),
        "final_changed_rows_exact": int(final_changed.sum().item()),
        "unexpected_final_changed_rows_outside_sensitive_set": len(
            final_changed_ids - sensitive_set
        ),
        "stage2_changed_sensitive_rows": stage2_changed_rows,
        "stage1_deltaW_frobenius": frobenius_from_row_norms(stage1_row_norms),
        "final_deltaW_frobenius": frobenius_from_row_norms(final_row_norms),
        "stage2_incremental_deltaW_frobenius": stage2_frob,
        "stage1_deltaW_row_norm": percentile_summary(
            stage1_row_norms[stage1_changed].tolist()
        ),
        "final_deltaW_row_norm": percentile_summary(
            final_row_norms[final_changed].tolist()
        ),
    }

    report = {
        "schema_version": 2,
        "method": "MQuAKE SURE v3 read-only forensic diagnosis",
        "warning": "AtomicGen is post-hoc held-out diagnosis only; do not use these diagnostics for training or checkpoint selection.",
        "seed": a.seed,
        "paths": {
            "base": str(Path(a.base_model_path).resolve())
            if Path(a.base_model_path).exists()
            else a.base_model_path,
            "stage1": str(Path(a.stage1_model_path).resolve()),
            "final": str(Path(a.final_model_path).resolve()),
            "mquake": str(Path(a.mquake_path).resolve()),
        },
        "stage_decomposition": [base_stage, stage1_stage, final_stage],
        "head_edit": head_summary,
        "gen": {
            "base_token_position": token_position_summary(
                base_forget, "atomic_gen"
            ),
            "stage1_token_position": token_position_summary(
                stage1_forget, "atomic_gen"
            ),
            "final_token_position": token_position_summary(
                final_forget, "atomic_gen"
            ),
            "paired_direct_vs_atomicgen": gen_summary,
        },
        "spe": spe_summary,
    }

    write_json(out_dir / "v3_forensics_summary.json", report)
    write_jsonl(out_dir / "gen_paired_cases.jsonl", gen_pairs)
    write_jsonl(out_dir / "retain_token_forensics.jsonl", retain_details)

    print("\n===== V3 FORENSICS DIGEST =====")
    for row in report["stage_decomposition"]:
        print(
            "{name:10s} Eff={Eff:.6f} Gen={AtomicGen:.6f} "
            "Spe={RetainEff:.6f} PPL={PPL}".format(**row)
        )
    print("Head edits:", head_summary)
    print(
        "Final AtomicGen by token position:",
        report["gen"]["final_token_position"],
    )
    print("AtomicGen recovered-vs-forgotten geometry:", gen_summary)
    print("Retain overlap:", spe_summary)
    print("Summary:", out_dir / "v3_forensics_summary.json")
    print("Gen rows:", out_dir / "gen_paired_cases.jsonl")
    print("Retain rows:", out_dir / "retain_token_forensics.jsonl")

    del final_w
    unload(final_model)


if __name__ == "__main__":
    main()
