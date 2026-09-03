#!/usr/bin/env python3
"""Read-only direct-generation audit for a completed MCF V1.1 run.

This does not train or modify the checkpoint. It compares Base and the saved
position-preserving private-vocabulary checkpoint on the 50 training-visible
forget prompts using deterministic greedy generation. For each case it reports
whether the generated continuation contains the original true answer, the
counterfactual/new target, or neither, alongside the saved final pairwise margin.

Official retain/paraphrase/neighborhood prompts are never opened.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_private_vocab_rewiring_v1_1_core as core


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--base-model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--only-final-failures", action="store_true")
    return p.parse_args(list(argv) if argv is not None else None)


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def render_prompt(record: Mapping[str, Any]) -> str:
    rr = record["requested_rewrite"]
    return str(rr["prompt"]).format(str(rr["subject"]))


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def contains_answer(generation: str, answer: str) -> bool:
    target = normalize(answer)
    if not target:
        return False
    return target in normalize(generation)


def encode_prompt(tokenizer: Any, base_tokenizer: Any, prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    ids = [int(v) for v in ids]
    bos = getattr(base_tokenizer, "bos_token_id", None)
    if bos is not None:
        ids = [int(bos)] + ids
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    attention = torch.ones_like(input_ids)
    return input_ids, attention


@torch.inference_mode()
def greedy_completion(
    model: Any,
    prompt_tokenizer: Any,
    decode_tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    input_ids, attention = encode_prompt(prompt_tokenizer, decode_tokenizer, prompt, device)
    eos = getattr(decode_tokenizer, "eos_token_id", None)
    pad = getattr(decode_tokenizer, "pad_token_id", None)
    if pad is None:
        pad = eos
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=eos,
        pad_token_id=pad,
        use_cache=True,
    )
    tail = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
    return decode_tokenizer.decode(tail, skip_special_tokens=True).strip()


def load_model(path: str | Path, dtype: torch.dtype, device: torch.device) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return model


def release_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    model_dir = run_dir / "model"
    protocol_path = run_dir / "protocol" / "training_visible_forget_direct.json"
    method_path = run_dir / "method" / "private_vocab_rewiring_v1_1.json"
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    if not method_path.is_file():
        raise FileNotFoundError(method_path)

    records = json.loads(protocol_path.read_text(encoding="utf-8"))
    method = json.loads(method_path.read_text(encoding="utf-8"))
    final_margin = {
        int(item["case_id"]): float(item["margin"])
        for item in method["margins"]["final_per_case"]
    }
    if args.only_final_failures:
        records = [row for row in records if final_margin[int(row["case_id"])] < 0.1]

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    print(f"Auditing {len(records)} direct forget prompts", flush=True)
    print("Loading Base model...", flush=True)
    base_model = load_model(args.base_model, dtype, device)
    base_generations: dict[int, str] = {}
    for index, record in enumerate(records, start=1):
        case_id = int(record["case_id"])
        base_generations[case_id] = greedy_completion(
            base_model,
            base_tokenizer,
            base_tokenizer,
            render_prompt(record),
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        if index % 10 == 0 or index == len(records):
            print(f"  Base generations {index}/{len(records)}", flush=True)
    release_model(base_model)

    print("Loading edited V1.1 checkpoint...", flush=True)
    edited_model = load_model(model_dir, dtype, device)
    private_tokenizer = core.load_position_preserving_tokenizer(model_dir, AutoTokenizer)

    rows = []
    counts = {
        "cases": len(records),
        "margin_pass": 0,
        "true_leak": 0,
        "new_hit": 0,
        "neither": 0,
        "true_leak_despite_margin_pass": 0,
    }
    for index, record in enumerate(records, start=1):
        rr = record["requested_rewrite"]
        case_id = int(record["case_id"])
        prompt = render_prompt(record)
        true_answer = str(rr["target_true"]["str"])
        new_answer = str(rr["target_new"]["str"])
        edited = greedy_completion(
            edited_model,
            private_tokenizer,
            base_tokenizer,
            prompt,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        margin = final_margin[case_id]
        margin_pass = margin >= 0.1
        true_leak = contains_answer(edited, true_answer)
        new_hit = contains_answer(edited, new_answer)
        if margin_pass:
            counts["margin_pass"] += 1
        if true_leak:
            counts["true_leak"] += 1
        if new_hit:
            counts["new_hit"] += 1
        if not true_leak and not new_hit:
            counts["neither"] += 1
        if margin_pass and true_leak:
            counts["true_leak_despite_margin_pass"] += 1
        row = {
            "case_id": case_id,
            "subject": str(rr["subject"]),
            "relation_id": str(rr["relation_id"]),
            "true": true_answer,
            "new": new_answer,
            "margin": margin,
            "margin_pass": margin_pass,
            "base_generation": base_generations[case_id],
            "edited_generation": edited,
            "edited_contains_true": true_leak,
            "edited_contains_new": new_hit,
        }
        rows.append(row)
        print("\n" + "=" * 88)
        print(
            f"case {case_id} | {rr['subject']} | {rr['relation_id']} | "
            f"margin={margin:.6g} | pass={margin_pass}"
        )
        print(f"prompt : {prompt}")
        print(f"true   : {true_answer}")
        print(f"new    : {new_answer}")
        print(f"BASE   : {base_generations[case_id]}")
        print(f"EDITED : {edited}")
        print(f"leak_true={true_leak}  hit_new={new_hit}")
        if index % 10 == 0 or index == len(records):
            print(f"\nAudited {index}/{len(records)}", flush=True)

    release_model(edited_model)

    audit_dir = run_dir / "audit"
    audit_dir.mkdir(exist_ok=True)
    suffix = "failures" if args.only_final_failures else "all50"
    report = {"summary": counts, "rows": rows}
    report_path = audit_dir / f"direct_generation_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + "#" * 88)
    print(json.dumps(counts, indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
