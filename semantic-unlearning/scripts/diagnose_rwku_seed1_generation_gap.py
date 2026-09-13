#!/usr/bin/env python3
"""Diagnose RWKU seed-1 teacher-forced vs greedy-generation forgetting gaps.

Reads the existing edited/base RWKU evaluation artifacts, identifies same-50
examples where the edited model still greedily recovers the sensitive answer,
then traces uncached edited decoding token-by-token under the same fixed
original-request boundary used by the official Batch-50 extension evaluator.

This script is diagnostic only: it does not train, alter checkpoints, select
hyperparameters, or open any evaluation split beyond the already-evaluated
same-50 efficacy rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import rwku_eval as rwku
from rwku_batch50 import build_batch_split
from mcf_zero_unlearn_official_eval import dtype_from_str
from static_overlap_fact_association_embeddings import load_artifact_into_model


def _by_hash(rows):
    out = {}
    for row in rows:
        key = str(row.get("source_record_sha256", ""))
        if key:
            out[key] = row
    return out


@torch.no_grad()
def trace_generation(model, bank, tokenizer, prompt, answer, max_new_tokens=30, topk=5):
    device = next(model.parameters()).device
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)
    ids = encoded["input_ids"].clone()
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = torch.ones_like(ids)
    prompt_length = int(mask[0].sum().item())
    ids = ids[:, :prompt_length]
    mask = torch.ones_like(ids)

    sensitive_ids = rwku._token_ids(
        tokenizer,
        rwku._normalized_completion(str(answer)),
        add_special_tokens=False,
    )
    sensitive_first_id = int(sensitive_ids[0]) if sensitive_ids else None
    sensitive_first_text = (
        tokenizer.decode([sensitive_first_id])
        if sensitive_first_id is not None
        else None
    )

    eos = tokenizer.eos_token_id
    steps = []
    for step in range(int(max_new_tokens)):
        model.set_association_prefix_lengths([prompt_length])
        logits = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
        ).logits[:, -1, :].float()
        probs = F.softmax(logits, dim=-1)
        values, indices = torch.topk(probs[0], k=int(topk))
        generated_id = int(indices[0].item())
        generated_token = tokenizer.decode([generated_id])
        route = list(bank.last_active_fact_indices[0])

        first_sensitive_probability = (
            None
            if sensitive_first_id is None
            else float(probs[0, sensitive_first_id].item())
        )
        top = [
            {
                "token_id": int(token_id),
                "token": tokenizer.decode([int(token_id)]),
                "probability": float(probability),
            }
            for probability, token_id in zip(
                values.detach().cpu().tolist(),
                indices.detach().cpu().tolist(),
            )
        ]

        ids = torch.cat(
            [
                ids,
                torch.tensor(
                    [[generated_id]],
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )
        mask = torch.ones_like(ids)
        generated_text = tokenizer.decode(
            ids[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()
        recovered = rwku.recovery_success(generated_text, str(answer))
        steps.append(
            {
                "step": step,
                "generated_token_id": generated_id,
                "generated_token": generated_token,
                "generated_text_so_far": generated_text,
                "active_fact_rows": route,
                "sensitive_first_token_id": sensitive_first_id,
                "sensitive_first_token": sensitive_first_text,
                "sensitive_first_token_probability": first_sensitive_probability,
                "top_next_tokens": top,
                "answer_recovered_so_far": bool(recovered),
            }
        )
        if eos is not None and generated_id == int(eos):
            break

    final_text = tokenizer.decode(
        ids[0, prompt_length:],
        skip_special_tokens=True,
    ).strip()
    first_recovery_step = next(
        (
            int(item["step"])
            for item in steps
            if item["answer_recovered_so_far"]
        ),
        None,
    )
    return {
        "final_prediction": final_text,
        "recovery_success": bool(
            rwku.recovery_success(final_text, str(answer))
        ),
        "sensitive_answer_token_ids_at_original_boundary": sensitive_ids,
        "first_recovery_step": first_recovery_step,
        "steps": steps,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--base-eval", required=True)
    p.add_argument("--data-root", default="data/rwku")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    edited_eval_path = run_dir / "official_rwku_batch50_eval.json"
    edited_eval = json.loads(edited_eval_path.read_text())
    base_eval = json.loads(Path(args.base_eval).resolve().read_text())
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    artifact = torch.load(
        run_dir / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )

    edited_same50 = edited_eval["details"]["same_50_efficacy"]
    base_same50 = base_eval["details"]["same_50_efficacy"]
    edited_by_hash = _by_hash(edited_same50)
    base_by_hash = _by_hash(base_same50)

    recovered_hashes = [
        str(row["source_record_sha256"])
        for row in edited_same50
        if bool(row.get("recovery_success"))
    ]
    if not recovered_hashes:
        print(json.dumps({"recovered_count": 0, "cases": []}, indent=2))
        return 0

    split = build_batch_split(
        data_root=Path(args.data_root).resolve(),
        batch_seed=1,
        allow_download=not args.no_download,
    )
    split_by_hash = _by_hash(split["efficacy_forget"])
    missing = [h for h in recovered_hashes if h not in split_by_hash]
    if missing:
        raise RuntimeError(
            f"Recovered records missing from frozen Batch-50 split: {missing}"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(manifest["model_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

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

    cases = []
    for source_hash in recovered_hashes:
        row = split_by_hash[source_hash]
        prompt = rwku.format_qa_prompt(tokenizer, row)
        edited_row = edited_by_hash[source_hash]
        base_row = base_by_hash.get(source_hash)
        trace = trace_generation(
            model,
            bank,
            tokenizer,
            prompt,
            str(row["answer"]),
            max_new_tokens=args.max_new_tokens,
            topk=args.topk,
        )
        if str(trace["final_prediction"]) != str(edited_row["prediction"]):
            raise RuntimeError(
                "Diagnostic trace does not reproduce saved edited prediction "
                f"for {source_hash}: trace={trace['final_prediction']!r}, "
                f"saved={edited_row['prediction']!r}"
            )
        cases.append(
            {
                "source_record_sha256": source_hash,
                "subject": str(row["subject"]),
                "level": str(row.get("level", "")),
                "query": str(row["query"]),
                "answer": str(row["answer"]),
                "base_prediction": (
                    None if base_row is None else base_row.get("prediction")
                ),
                "base_recovery_success": (
                    None
                    if base_row is None
                    else bool(base_row.get("recovery_success"))
                ),
                "edited_prediction": edited_row["prediction"],
                "edited_recovery_success": bool(
                    edited_row["recovery_success"]
                ),
                "teacher_forced_sensitive_token_top1_accuracy": edited_row.get(
                    "sensitive_token_top1_accuracy"
                ),
                "teacher_forced_first_token_probability": edited_row.get(
                    "first_token_probability"
                ),
                "teacher_forced_geometric_probability": edited_row.get(
                    "geometric_probability"
                ),
                "active_fact_rows": edited_row.get("active_fact_rows"),
                "trace": trace,
            }
        )

    result = {
        "dataset": "RWKU",
        "protocol_id": edited_eval.get("protocol_id"),
        "seed": 1,
        "diagnostic_scope": (
            "same-50 edited generative recoveries only; no training or "
            "checkpoint selection"
        ),
        "edited_same50_count": len(edited_same50),
        "edited_recovered_count": len(cases),
        "edited_recovered_fraction": len(cases) / len(edited_same50),
        "cases": cases,
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "rwku_generation_gap_diagnostic.json"
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "edited_recovered_count": len(cases),
                "cases": [
                    {
                        "subject": case["subject"],
                        "query": case["query"],
                        "answer": case["answer"],
                        "base_prediction": case["base_prediction"],
                        "edited_prediction": case["edited_prediction"],
                        "teacher_forced_sensitive_token_top1_accuracy": case[
                            "teacher_forced_sensitive_token_top1_accuracy"
                        ],
                        "teacher_forced_first_token_probability": case[
                            "teacher_forced_first_token_probability"
                        ],
                        "first_recovery_step": case["trace"][
                            "first_recovery_step"
                        ],
                        "first_generated_tokens": [
                            step["generated_token"]
                            for step in case["trace"]["steps"][:8]
                        ],
                    }
                    for case in cases
                ],
                "out": str(out),
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
