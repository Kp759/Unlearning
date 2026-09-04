#!/usr/bin/env python3
"""Stochastic adversarial sampling audit for RSNR direct-logit baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import mcf_rsnr_v1a_adversarial_eval as adv
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_sampling_attack_eval as samp
import mcf_rsnr_v1a_logitmask_common as lm
import mcf_rsnr_v1a_logitmask_adversarial_eval as ladv


def parse_temperatures(text: str) -> list[float]:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("temperatures must be positive")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--fresh-retain-seed", type=int, default=700002)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--temperatures", default="0.7,1.0")
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--samples-per-prompt", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--scope", choices=("canonical", "all"), default="canonical")
    args = p.parse_args()
    args.temperature_values = parse_temperatures(args.temperatures)
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        p.error("development evaluation is locked to seed1/forget50/retain1000")
    if not 0 < args.top_p <= 1:
        p.error("top-p must be in (0,1]")
    return args


@torch.no_grad()
def sampled_generate_logitmask(
    model: Any,
    hook: lm.DirectLogitMaskHook,
    tok: Any,
    prompts: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    if len(prompts) != len(pairs):
        raise ValueError("prompt/pair length mismatch")
    sequences = [
        [int(v) for v in tok(str(prompt), add_special_tokens=True, return_attention_mask=False)["input_ids"]]
        for prompt in prompts
    ]
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos if eos is not None else 0
    generated = [[] for _ in sequences]
    finished = [False] * len(sequences)

    for _ in range(int(max_new_tokens)):
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), int(pad), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        positions = torch.zeros_like(input_ids, dtype=torch.float32)
        gate = torch.zeros(len(sequences), dtype=torch.float32, device=device)
        suppress_rows = []
        last_positions = []
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention[i, : len(seq)] = 1
            last = len(seq) - 1
            last_positions.append(last)
            suppress_rows.append(tuple(token_sets[pairs[i]]))
            if not finished[i]:
                gate[i] = 1.0
                positions[i, last] = 1.0
        hook.set(gate, positions, suppress_rows)
        try:
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        finally:
            hook.clear()
        active = False
        for i, last in enumerate(last_positions):
            if finished[i]:
                continue
            token = samp.sample_from_logits(
                logits[i, last, :], temperature=float(temperature), top_p=float(top_p)
            )
            generated[i].append(token)
            sequences[i].append(token)
            if eos is not None and token == int(eos):
                finished[i] = True
            else:
                active = True
        if not active:
            break
    return [tok.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    leaks = sum(bool(r["true_or_alias_leak"]) for r in rows)
    abstain = sum(bool(r["semantic_abstention"]) for r in rows)
    clean_abstain = sum(bool(r["semantic_abstention"] and not r["true_or_alias_leak"]) for r in rows)
    return {
        "sample_count": total,
        "semantic_abstention_count": abstain,
        "semantic_abstention_rate": abstain / total if total else None,
        "true_or_alias_leak_count": leaks,
        "true_or_alias_leak_rate": leaks / total if total else None,
        "abstaining_nondisclosure_count": clean_abstain,
        "abstaining_nondisclosure_rate": clean_abstain / total if total else None,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed) + 71103)
    runtime = ladv.load_runtime(args)
    try:
        base_rows = adv.base_sensitive_prompts(runtime["forget_records"], runtime["router"])
        if args.scope == "canonical":
            base_rows = [r for r in base_rows if r["group"] == "rewrite"]
        attack_rows = []
        for row in base_rows:
            for attack_name, instruction in adv.RETRIEVAL_ATTACKS.items():
                attack_rows.append({
                    **dict(row),
                    "attack": attack_name,
                    "attack_prompt": adv.build_attack_prompt(str(row["prompt"]), instruction),
                })

        all_rows = []
        for temperature in args.temperature_values:
            for sample_index in range(int(args.samples_per_prompt)):
                for chunk in ev._chunks(attack_rows, args.batch_size):
                    prompts = [str(r["attack_prompt"]) for r in chunk]
                    pairs = [(str(r["subject"]), str(r["relation_id"])) for r in chunk]
                    outputs = sampled_generate_logitmask(
                        runtime["model"], runtime["hook"], runtime["tok"], prompts, pairs,
                        runtime["token_sets"], runtime["device"],
                        temperature=temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for source, output in zip(chunk, outputs):
                        pair = (str(source["subject"]), str(source["relation_id"]))
                        aliases = list(runtime["aliases"][pair])
                        abstention = ev.detect_abstention(output)
                        leaks = ev.leaked_aliases(output, aliases)
                        all_rows.append({
                            **dict(source),
                            "temperature": float(temperature),
                            "top_p": float(args.top_p),
                            "sample_index": int(sample_index),
                            "generation": output,
                            "semantic_abstention": bool(abstention["semantic"]),
                            "exact_abstention": bool(abstention["exact"]),
                            "leaked_aliases": leaks,
                            "true_or_alias_leak": bool(leaks),
                        })

        by_temperature = {
            str(t): summarize([r for r in all_rows if r["temperature"] == float(t)])
            for t in args.temperature_values
        }
        by_attack = {
            name: summarize([r for r in all_rows if r["attack"] == name])
            for name in sorted({str(r["attack"]) for r in all_rows})
        }
        result = {
            "method": f"rsnr_direct_logit_{runtime['config']['variant']}_sampling_attack",
            "variant": runtime["config"]["variant"],
            "development_only": True,
            "scope": args.scope,
            "temperatures": args.temperature_values,
            "top_p": float(args.top_p),
            "samples_per_prompt": int(args.samples_per_prompt),
            "true_penalty": runtime["config"]["true_penalty"],
            "idk_boost": runtime["config"]["idk_boost"],
            "retrieval_attacks_only": True,
            "true_answer_present_in_attack_prompt": False,
            "overall": summarize(all_rows),
            "by_temperature": by_temperature,
            "by_attack": by_attack,
            "per_sample": all_rows,
            "artifact_validation": runtime["validation"],
            "mask_scope": {
                "canonical_target_true_only": True,
                "aliases_used_for_mask": False,
            },
        }
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "overall": result["overall"],
            "by_temperature": by_temperature,
            "by_attack": by_attack,
        }, indent=2))
        print(f"RSNR direct-logit stochastic result: {out}")
    finally:
        runtime["hook"].remove()


if __name__ == "__main__":
    main()
