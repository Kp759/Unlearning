#!/usr/bin/env python3
"""Evaluate a saved MQuAKE fact-association bank under the locked official split.

Native MQuAKE Eff is the ZeroUnlearn-compatible teacher-forced sensitive-token
argmax accuracy on direct atomic rewrites (lower is better). AtomicGen is an
evaluation-only extension on held-out natural-language atomic questions.
The intervention boundary is fixed at the ORIGINAL request/question boundary
while teacher-forced answer-prefix tokens are appended.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import torch

import mquake_zero_unlearn_official_eval as mquake
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)
from static_overlap_fact_association_embeddings import load_artifact_into_model


def _boundary_text(record, case):
    rr = record["requested_rewrite"]
    if case.prompt_type == "rewrite":
        return str(rr["prompt"]).format(str(rr["subject"]))
    if case.prompt_type == "atomic_gen":
        return str(record["atomic_gen_prompt"])
    raise ValueError(f"Unknown MQuAKE prompt type: {case.prompt_type}")


def _strict_prefix_lengths(tok, full_prompts, boundary_prompts):
    lengths = []
    for full_text, boundary_text in zip(full_prompts, boundary_prompts):
        full_ids = mquake._flat_ids(tok, full_text)
        boundary_ids = mquake._flat_ids(tok, boundary_text)
        if not boundary_ids or len(boundary_ids) > len(full_ids):
            raise ValueError("Invalid MQuAKE association boundary tokenization")
        if full_ids[: len(boundary_ids)] != boundary_ids:
            raise ValueError(
                "Original MQuAKE request/question is not an exact token prefix "
                "of the teacher-forced evaluation context"
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
        batch = cases[start : start + int(batch_size)]
        full_prompts = [case.prompt for case in batch]
        boundary_prompts = [
            _boundary_text(records_by_id[int(case.case_id)], case)
            for case in batch
        ]
        encoded = tok(
            full_prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        prefix_lengths = _strict_prefix_lengths(
            tok, full_prompts, boundary_prompts
        )
        model.set_association_prefix_lengths(prefix_lengths)
        output = model(**encoded, use_cache=False)
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        final_logits = output.logits[batch_indices, last_non_masked, :]
        predicted_ids = final_logits.argmax(dim=-1)
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        routes = list(bank.last_active_fact_indices)
        if len(routes) != len(batch):
            raise RuntimeError(
                "Association bank did not expose one route per MQuAKE case"
            )
        for case, predicted_id, target_id, route, boundary in zip(
            batch,
            predicted_ids.detach().cpu().tolist(),
            target_ids.detach().cpu().tolist(),
            routes,
            prefix_lengths,
        ):
            rows.append(
                {
                    **asdict(case),
                    "target_token_id": int(target_id),
                    "predicted_token_id": int(predicted_id),
                    "correct": bool(predicted_id == target_id),
                    "association_boundary_tokens": int(boundary),
                    "active_fact_rows": route,
                    "association_route_active": bool(route),
                }
            )
    return rows


def _route_summary(predicted):
    output = {}
    for prompt_type in ("rewrite", "atomic_gen"):
        current = [
            row for row in predicted if row["prompt_type"] == prompt_type
        ]
        if not current:
            output[prompt_type] = {
                "token_decisions": 0,
                "route_active_fraction": None,
                "active_token_decisions": 0,
            }
            continue
        active = sum(row["association_route_active"] for row in current)
        output[prompt_type] = {
            "token_decisions": len(current),
            "route_active_fraction": active / len(current),
            "active_token_decisions": active,
        }
    return output


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
    include_atomic_gen,
):
    prompt_types = (
        ("rewrite", "atomic_gen")
        if include_atomic_gen
        else ("rewrite",)
    )
    records_by_id = {
        int(record["case_id"]): record for record in records
    }
    cases = [
        case
        for record in records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=prompt_types,
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
    summary = mquake.summarize_atomic_split(
        split_name,
        records,
        predicted,
    )
    return summary, predicted, _route_summary(predicted)


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
            a - b
            for a, b in zip(
                after["active_fact_counts"],
                before["active_fact_counts"],
            )
        ],
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-atomic-gen", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads(
        (run_dir / "association_manifest.json").read_text()
    )
    if int(manifest.get("seed", -1)) != 1:
        raise ValueError("Registered MQuAKE evaluator requires seed 1")
    if int(manifest.get("forget_num_instances", -1)) != 50:
        raise ValueError("Registered MQuAKE evaluator requires 50 forget instances")
    if int(
        manifest.get("retain_num_instances_final_evaluation", -1)
    ) != 1000:
        raise ValueError("Registered MQuAKE evaluator requires 1000 retain instances")

    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
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

    forget_records, retain_records = mquake.load_official_eval_records(
        Path(args.mquake_path).resolve(),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )

    expected_atomic_case_ids = [
        int(record["case_id"]) for record in forget_records
    ]
    artifact_atomic_case_ids = [
        int(fact["case_id"]) for fact in artifact["facts"]
    ]
    if artifact_atomic_case_ids != expected_atomic_case_ids:
        raise RuntimeError(
            "Saved MQuAKE bank is not the exact official seed-1 atomic forget set"
        )
    if int(manifest.get("forget_atomic_fact_count", -1)) != len(
        forget_records
    ):
        raise RuntimeError("Manifest atomic-fact count no longer matches source split")

    dtype = dtype_from_str(args.dtype)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    model, bank = load_artifact_into_model(base_model, artifact)
    model.eval()

    device = next(model.parameters()).device
    llama_like = mquake.is_llama_like(model, tok)
    include_atomic_gen = not args.skip_atomic_gen

    forget_summary, forget_raw, forget_routes = evaluate_split_fixed_boundary(
        model,
        bank,
        tok,
        forget_records,
        device,
        llama_like=llama_like,
        split_name="forget",
        batch_size=args.batch_size,
        include_atomic_gen=include_atomic_gen,
    )
    retain_summary, retain_raw, retain_routes = evaluate_split_fixed_boundary(
        model,
        bank,
        tok,
        retain_records,
        device,
        llama_like=llama_like,
        split_name="retain",
        batch_size=args.batch_size,
        include_atomic_gen=include_atomic_gen,
    )

    legacy_ppl = None
    runtime_ppl = None
    runtime_ppl_route_activity = None
    if not args.skip_ppl:
        ppl_text = load_official_ppl_text(args.wikidata_dir)
        if ppl_text is not None:
            legacy_ppl = official_perplexity(
                model,
                tok,
                ppl_text,
                device,
                max_input_length=100,
            )
            before = bank.counters()
            runtime = runtime_aligned_perplexity(
                model,
                tok,
                ppl_text,
                device,
                max_input_length=100,
            )
            after = bank.counters()
            runtime_ppl = runtime["ppl"]
            runtime_ppl_route_activity = _counter_delta(after, before)

    result = {
        "method": "FactAssociationBank",
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": 1,
        "forget_num_instances": 50,
        "retain_num_instances": 1000,
        "forget_atomic_fact_count": len(forget_records),
        "retain_atomic_fact_count": len(retain_records),
        "architecture": {
            "layer": int(artifact["layer"]),
            "trainable_vectors": len(artifact["facts"]),
            "base_weights_edited": False,
            "input_embeddings_edited": False,
            "lm_head_edited": False,
            "tokenizer_extended": False,
            "fact_id_injection_used": False,
            "boundary": "original request/question token boundary",
        },
        "protocol": {
            "sampling": (
                "first-half retain / second-half forget; seed 1; sample forget "
                "first then retain; flatten requested_rewrite after sampling"
            ),
            "native_Eff": (
                "teacher-forced original target_true token argmax accuracy "
                "on direct atomic rewrites; lower is better"
            ),
            "AtomicGen": (
                "same sensitive-token accuracy on held-out atomic questions; "
                "extension, not native ZeroUnlearn MQuAKE column"
            ),
            "training_target_new_used": False,
            "training_atomic_questions_used": False,
            "training_multihop_questions_used": False,
            "training_retain_used": False,
        },
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_routes": forget_routes,
        "retain_routes": retain_routes,
        "legacy_PPL": legacy_ppl,
        "runtime_aligned_PPL": runtime_ppl,
        "runtime_aligned_PPL_route_activity": runtime_ppl_route_activity,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "runtime_counters": bank.counters(),
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "official_mquake_eval.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "forget_Eff": forget_summary.get("Eff"),
                "forget_AtomicGen": forget_summary.get("AtomicGen"),
                "retain_Eff": retain_summary.get("Eff"),
                "retain_AtomicGen": retain_summary.get("AtomicGen"),
                "forget_routes": forget_routes,
                "retain_routes": retain_routes,
                "runtime_aligned_PPL": runtime_ppl,
                "runtime_aligned_PPL_route_activity": runtime_ppl_route_activity,
                "legacy_PPL": legacy_ppl,
                "out": str(out),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
