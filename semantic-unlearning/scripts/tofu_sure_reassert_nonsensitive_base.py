#!/usr/bin/env python3
"""Reassert Stage1B non-sensitive TOFU answer rows to Full-TOFU Base in-place.

This is a post-Stage2 guard used by the SURE R512/R1024 ablation.  It reads the
Stage1B answer-row partition, copies ONLY the non-sensitive visible answer-token
rows from the protected Full-TOFU reference into BOTH the input embedding and
LM head of the Stage2 checkpoint, audits the same 50 direct-forget constraints,
and overwrites the same checkpoint only when every constraint still passes.

No retain95, paraphrases, same-author holdout, real-authors, world-facts, PPL,
or final evaluation metric is loaded or used.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "SURE-TOFU-post-Stage2-nonsensitive-Base-reassertion"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Stage2 checkpoint modified in-place")
    p.add_argument("--reference-model-path", required=True, help="Protected Full-TOFU checkpoint")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--forget-requirements-json", required=True)
    p.add_argument("--answer-row-restoration-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--constraint-tolerance", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_required_nll(path: Path, count: int, fallback: float, device: torch.device) -> torch.Tensor:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("forget-requirements-json must contain one row per direct forget QA")
    values: List[float] = []
    for row in rows:
        values.append(float(row.get("required_answer_nll", row.get("required_nll", fallback))))
    return torch.tensor(values, dtype=torch.float32, device=device)


def max_row_error(weight: torch.Tensor, ids: torch.Tensor, expected: torch.Tensor) -> float:
    if ids.numel() == 0:
        return 0.0
    current = weight.index_select(0, ids.to(weight.device)).detach().float().cpu()
    return float((current - expected.detach().float().cpu()).abs().max().item())


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("forget-num, batch-size and max-length must be positive")
    if a.constraint_tolerance < 0:
        raise ValueError("constraint-tolerance must be non-negative")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")

    model_path = Path(a.model_path).resolve()
    ref_path = Path(a.reference_model_path).resolve()
    forget_path = Path(a.forget_json).resolve()
    requirements_path = Path(a.forget_requirements_json).resolve()
    partition_path = Path(a.answer_row_restoration_json).resolve()
    output_json = Path(a.output_json).resolve()
    for path in (model_path, ref_path):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path in (forget_path, requirements_path, partition_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    partition = json.loads(partition_path.read_text(encoding="utf-8"))
    non_sensitive_ids = sorted({int(x) for x in partition["non_sensitive_answer_token_ids"]})
    sensitive_ids = sorted({int(x) for x in partition["sensitive_answer_token_ids"]})
    overlap = set(non_sensitive_ids).intersection(sensitive_ids)
    if overlap:
        raise RuntimeError(f"sensitive/non-sensitive partition overlaps: {sorted(overlap)[:10]}")
    if not non_sensitive_ids:
        raise RuntimeError("Stage1B partition contains no non-sensitive rows to reassert")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    tok_data = AutoTokenizer.from_pretrained(model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, _ = locked.load_forget_instances(forget_path, tok_data, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    device = gagd.first_device(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight

    before_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    fallback = -math.log(a.target_forget_answer_probability)
    required_nll = load_required_nll(requirements_path, len(instances), fallback, device)
    before_slack = before_nll - required_nll
    if bool((before_slack < -a.constraint_tolerance).any().item()):
        raise RuntimeError(
            "Stage2 checkpoint is already infeasible before non-sensitive Base reassertion: "
            f"min_slack={float(before_slack.min().item())}"
        )

    dtype = gagd.torch_dtype(a.dtype)
    reference = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=dtype)
    ref_input = reference.get_input_embeddings().weight
    ref_output = reference.get_output_embeddings().weight
    ids_ref_i = torch.tensor(non_sensitive_ids, dtype=torch.long, device=ref_input.device)
    ids_ref_o = ids_ref_i.to(ref_output.device)
    base_input_rows = ref_input.index_select(0, ids_ref_i).detach().cpu().clone()
    base_output_rows = ref_output.index_select(0, ids_ref_o).detach().cpu().clone()
    del reference
    gc.collect()

    ids_input = torch.tensor(non_sensitive_ids, dtype=torch.long, device=input_weight.device)
    ids_output = ids_input.to(output_weight.device)
    with torch.no_grad():
        input_weight.index_copy_(
            0,
            ids_input,
            base_input_rows.to(device=input_weight.device, dtype=input_weight.dtype),
        )
        output_weight.index_copy_(
            0,
            ids_output,
            base_output_rows.to(device=output_weight.device, dtype=output_weight.dtype),
        )

    after_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    after_slack = after_nll - required_nll
    feasible = bool(torch.all(after_slack >= -a.constraint_tolerance).item())
    input_error = max_row_error(input_weight, ids_input, base_input_rows)
    output_error = max_row_error(output_weight, ids_output, base_output_rows)
    if input_error != 0.0 or output_error != 0.0:
        raise RuntimeError(
            f"non-sensitive rows are not exact Base: input={input_error} output={output_error}"
        )
    if not feasible:
        raise RuntimeError(
            "non-sensitive Base reassertion broke direct forgetting; refusing to overwrite checkpoint: "
            f"min_slack={float(after_slack.min().item())}"
        )

    # Overwrite the SAME checkpoint: no second 6 GiB model copy is kept.
    model.save_pretrained(model_path)
    tok.save_pretrained(model_path)

    report: Dict[str, Any] = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": a.seed,
        "model_path_overwritten_in_place": str(model_path),
        "reference_model_path": str(ref_path),
        "answer_row_restoration_json": str(partition_path),
        "non_sensitive_answer_row_count": len(non_sensitive_ids),
        "sensitive_answer_row_count": len(sensitive_ids),
        "non_sensitive_answer_token_ids": non_sensitive_ids,
        "input_embedding_base_max_abs_error": input_error,
        "lm_head_base_max_abs_error": output_error,
        "minimum_forget_nll_slack_before": float(before_slack.min().item()),
        "minimum_forget_nll_slack_after": float(after_slack.min().item()),
        "forget_answer_probability_mean_before": float(torch.exp(-before_nll).mean().item()),
        "forget_answer_probability_max_before": float(torch.exp(-before_nll).max().item()),
        "forget_answer_probability_mean_after": float(torch.exp(-after_nll).mean().item()),
        "forget_answer_probability_max_after": float(torch.exp(-after_nll).max().item()),
        "training_selection_data_access": {
            "direct_forget_qas": a.forget_num,
            "full_tofu_reference_rows": "non-sensitive visible answer rows only",
            "retain95": 0,
            "paraphrases": 0,
            "same_author_holdout": 0,
            "real_authors": 0,
            "world_facts": 0,
            "PPL": False,
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "SURE-TOFU post-Stage2 non-sensitive Base reassertion PASS "
        f"rows={len(non_sensitive_ids)} min_slack={report['minimum_forget_nll_slack_after']:.6g}"
    )
    print(f"audit: {output_json}")


if __name__ == "__main__":
    main()
