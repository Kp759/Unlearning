#!/usr/bin/env python3
"""Official ZsRE evaluation for a saved fact-association embedding bank.

Eff/Gen/Spe use the existing ZeroUnlearn-compatible ZsRE token-accuracy
definitions. The only execution adaptation is preserving the original request
boundary for the one-position residual edit while later teacher-forced answer
prefix tokens are appended.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import torch

import zsre_zero_unlearn_official_eval as zsre
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)
from static_overlap_fact_association_embeddings import load_artifact_into_model


def _boundary_text(record, case):
    if case.prompt_type == "rewrite":
        rr = record["requested_rewrite"]
        return str(rr["prompt"]).format(str(rr["subject"]))
    if case.prompt_type == "paraphrase":
        return str(record["paraphrase_prompts"][case.prompt_index])
    if case.prompt_type == "neighborhood":
        # Each official locality PredictionCase is already the complete observed
        # prefix used for that next-token decision.
        return str(case.prompt)
    raise ValueError(f"Unknown ZsRE prompt type: {case.prompt_type}")


def _strict_prefix_lengths(tok, full_prompts, boundary_prompts):
    lengths = []
    for full_text, boundary_text in zip(full_prompts, boundary_prompts):
        full_ids = zsre._flat_ids(tok, full_text)
        boundary_ids = zsre._flat_ids(tok, boundary_text)
        if not boundary_ids or len(boundary_ids) > len(full_ids):
            raise ValueError("Invalid association boundary tokenization")
        if full_ids[:len(boundary_ids)] != boundary_ids:
            raise ValueError(
                "Original ZsRE request is not an exact token prefix of the "
                "teacher-forced evaluation prompt; refusing boundary drift"
            )
        lengths.append(len(boundary_ids))
    return lengths


@torch.no_grad()
def predict_cases_fixed_boundary(
    model,
    bank,
    tok,
    cases,
    records_by_id,
    device,
    *,
    llama_like,
    batch_size=8,
):
    rows = []
    for start in range(0, len(cases), int(batch_size)):
        batch = cases[start:start + int(batch_size)]
        full_prompts = [case.prompt for case in batch]
        boundary_prompts = [
            _boundary_text(records_by_id[int(case.case_id)], case)
            for case in batch
        ]
        encoded = tok(
            full_prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)
        prefix_lengths = _strict_prefix_lengths(
            tok, full_prompts, boundary_prompts
        )
        model.set_association_prefix_lengths(prefix_lengths)
        output = model(**encoded, use_cache=False)
        attention = encoded["attention_mask"]
        last_non_masked = attention.sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        final_logits = output.logits[batch_indices, last_non_masked, :]
        predicted_ids = final_logits.argmax(dim=-1)
        target_ids = zsre.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        routes = list(bank.last_active_fact_indices)
        if len(routes) != len(batch):
            raise RuntimeError("Association bank did not expose one route per ZsRE case")
        for case, predicted_id, target_id, route, boundary in zip(
            batch,
            predicted_ids.detach().cpu().tolist(),
            target_ids.detach().cpu().tolist(),
            routes,
            prefix_lengths,
        ):
            rows.append({
                **asdict(case),
                "target_token_id": int(target_id),
                "predicted_token_id": int(predicted_id),
                "correct": bool(predicted_id == target_id),
                "association_boundary_tokens": int(boundary),
                "active_fact_rows": route,
                "association_route_active": bool(route),
            })
    return rows


def _route_summary(predicted):
    by_type = {}
    for prompt_type in ("rewrite", "paraphrase", "neighborhood"):
        current = [
            row for row in predicted
            if row["prompt_type"] == prompt_type
        ]
        if not current:
            by_type[prompt_type] = {
                "token_decisions": 0,
                "route_active_fraction": None,
                "active_token_decisions": 0,
            }
            continue
        active = sum(row["association_route_active"] for row in current)
        by_type[prompt_type] = {
            "token_decisions": len(current),
            "route_active_fraction": active / len(current),
            "active_token_decisions": active,
        }
    return by_type


def evaluate_split_fixed_boundary(
    model,
    bank,
    tok,
    records,
    device,
    *,
    llama_like,
    split_name,
    batch_size,
):
    records_by_id = {
        int(record["case_id"]): record for record in records
    }
    cases = [
        case
        for record in records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    predicted = predict_cases_fixed_boundary(
        model,
        bank,
        tok,
        cases,
        records_by_id,
        device,
        llama_like=llama_like,
        batch_size=batch_size,
    )
    by_record = {
        int(record["case_id"]): {
            "rewrite": [],
            "paraphrase": [],
            "neighborhood": [],
        }
        for record in records
    }
    for row in predicted:
        by_record[int(row["case_id"])][row["prompt_type"]].append(
            bool(row["correct"])
        )
    metric_data = []
    for record in records:
        grouped = by_record[int(record["case_id"])]
        metric_data.append({
            "case_id": int(record["case_id"]),
            "requested_rewrite": record["requested_rewrite"],
            "post": {
                "rewrite_prompts_correct": grouped["rewrite"],
                "paraphrase_prompts_correct": grouped["paraphrase"],
                "neighborhood_prompts_correct": grouped["neighborhood"],
            },
        })
    summary = zsre.official_summarize(split_name, metric_data)
    return summary, metric_data, predicted, _route_summary(predicted)


def _counter_delta(after, before):
    return {
        "hook_calls": after["hook_calls"] - before["hook_calls"],
        "active_batch_rows": (
            after["active_batch_rows"] - before["active_batch_rows"]
        ),
        "active_token_positions": (
            after["active_token_positions"] - before["active_token_positions"]
        ),
        "active_fact_counts": [
            a - b for a, b in zip(
                after["active_fact_counts"],
                before["active_fact_counts"],
            )
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--zsre-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads(
        (run_dir / "association_manifest.json").read_text()
    )
    if int(manifest.get("seed", -1)) != 1 or int(
        manifest.get("forget_num", -1)
    ) != 50:
        raise ValueError("This evaluator is registered for ZsRE seed1 / forget50")
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    if artifact.get("dataset") != "ZsRE":
        raise ValueError("Association artifact is not a ZsRE checkpoint")
    if artifact.get("target_new_used") is not False:
        raise ValueError("ZsRE checkpoint unexpectedly used target_new")

    model_path = Path(manifest["model_path"]).resolve()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    dtype = dtype_from_str(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    model, bank = load_artifact_into_model(base_model, artifact)
    model.eval()
    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)

    forget_records, retain_records = zsre.load_official_eval_records(
        Path(args.zsre_path),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )
    expected_forget = list(manifest.get("forget_case_ids", []))
    expected_retain = list(
        manifest.get("retain_case_ids_final_evaluation", [])
    )
    observed_forget = [int(record["case_id"]) for record in forget_records]
    observed_retain = [int(record["case_id"]) for record in retain_records]
    if expected_forget and observed_forget != expected_forget:
        raise ValueError("Final ZsRE forget sample differs from locked split manifest")
    if expected_retain and observed_retain != expected_retain:
        raise ValueError("Final ZsRE retain sample differs from locked split manifest")

    forget_summary, forget_raw, forget_predictions, forget_routes = (
        evaluate_split_fixed_boundary(
            model,
            bank,
            tok,
            forget_records,
            device,
            llama_like=llama_like,
            split_name="forget",
            batch_size=args.batch_size,
        )
    )
    retain_summary, retain_raw, retain_predictions, retain_routes = (
        evaluate_split_fixed_boundary(
            model,
            bank,
            tok,
            retain_records,
            device,
            llama_like=llama_like,
            split_name="retain",
            batch_size=args.batch_size,
        )
    )

    ppl = None
    legacy_ppl = None
    ppl_route_activity = None
    if not args.skip_ppl:
        text = load_official_ppl_text(args.wikidata_dir)
        if text is not None:
            legacy_ppl = official_perplexity(
                model, tok, text, device, max_input_length=100
            )
            before = bank.counters()
            ppl = runtime_aligned_perplexity(
                model,
                tok,
                text,
                device,
                max_input_length=100,
            )
            after = bank.counters()
            ppl_route_activity = _counter_delta(after, before)

    result = {
        "method": artifact.get(
            "method", "fact_association_embeddings_zsre_v1"
        ),
        "dataset": "ZsRE",
        "protocol": "ZeroUnlearn official-compatible token accuracy",
        "execution_contract": (
            "association route computed from original request prefix only; "
            "one layer-19 intervention stays at that request boundary while "
            "teacher-forced answer-prefix tokens are appended"
        ),
        "seed": 1,
        "unlearn_num": 50,
        "retain_num": 1000,
        "metric_definition": {
            "Eff": (
                "100 * case-macro sensitive target_true token top-1 accuracy "
                "on direct rewrite prompts; lower is better"
            ),
            "Gen": (
                "100 * case-macro sensitive target_true token top-1 accuracy "
                "on held-out rephrases; lower is better"
            ),
            "Spe": (
                "100 * case-macro locality/neighborhood target token top-1 "
                "accuracy; higher is better"
            ),
        },
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "forget_route_summary": forget_routes,
        "retain_route_summary": retain_routes,
        "forget_prediction_rows": forget_predictions,
        "retain_prediction_rows": retain_predictions,
        "runtime_aligned_PPL": ppl,
        "runtime_aligned_PPL_route_activity": ppl_route_activity,
        "legacy_PPL": legacy_ppl,
        "legacy_PPL_note": (
            "historical whole-sequence PPL is structurally blind to a "
            "last-position-only association intervention"
        ),
        "runtime_counters_total": bank.counters(),
        "training_data_contract": {
            "target_new_used": False,
            "official_rephrases_used": False,
            "official_locality_used": False,
            "retain_records_used": False,
        },
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "official_zsre_eval.json"
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite ZsRE official result: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    compact = {
        "forget_Eff": forget_summary["Eff"],
        "forget_Gen": forget_summary["Gen"],
        "forget_Spe": forget_summary["Spe"],
        "retain_Eff": retain_summary["Eff"],
        "retain_Gen": retain_summary["Gen"],
        "retain_Spe": retain_summary["Spe"],
        "forget_routes": forget_routes,
        "retain_routes": retain_routes,
        "runtime_aligned_PPL": None if ppl is None else ppl["ppl"],
        "runtime_aligned_PPL_route_activity": ppl_route_activity,
        "legacy_PPL": legacy_ppl,
        "out": str(out),
    }
    print(json.dumps(compact, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
