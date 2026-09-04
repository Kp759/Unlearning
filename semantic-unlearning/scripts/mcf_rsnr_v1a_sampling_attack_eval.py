#!/usr/bin/env python3
"""Stochastic adversarial sampling audit for the existing RSNR-V1A checkpoint.

This complements the full greedy adversarial evaluator.  By default it samples
only canonical forget prompts to control cost, using every retrieval attack at
multiple temperatures.  The true answer is never inserted into these prompts.
No retraining or checkpoint selection is performed.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn.functional as F

import mcf_rsnr_v1a_adversarial_eval as adv
import mcf_rsnr_v1a_official_eval_fresh_retain as ev


def parse_temperatures(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("temperatures must be a non-empty comma-separated list of positive floats")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
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
        p.error("RSNR-V1A sampling audit is locked to seed=1/forget50/retain1000")
    if not 0 < args.top_p <= 1:
        p.error("top-p must be in (0,1]")
    if args.samples_per_prompt <= 0 or args.max_new_tokens <= 0 or args.batch_size <= 0:
        p.error("sampling counts must be positive")
    return args


def sample_from_logits(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    probs = F.softmax(logits.float() / float(temperature), dim=-1)
    if float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative - sorted_probs >= float(top_p)
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        denom = sorted_probs.sum()
        if float(denom.item()) <= 0:
            return int(torch.argmax(probs).item())
        sorted_probs = sorted_probs / denom
        sampled_sorted = int(torch.multinomial(sorted_probs, num_samples=1).item())
        return int(sorted_idx[sampled_sorted].item())
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def sampled_generate_rsnr(
    model: Any,
    hook: Any,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    if not prompts:
        return []
    sequences = [
        [int(v) for v in tok(str(prompt), add_special_tokens=True, return_attention_mask=False)["input_ids"]]
        for prompt in prompts
    ]
    if any(not seq for seq in sequences):
        raise RuntimeError("cannot sample from empty prompt")
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos if eos is not None else 0
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False] * len(sequences)

    for _ in range(int(max_new_tokens)):
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), int(pad), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        positions = torch.zeros_like(input_ids, dtype=torch.float32)
        gate = torch.zeros(len(sequences), dtype=torch.float32, device=device)
        last_positions = []
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention[i, : len(seq)] = 1
            last = len(seq) - 1
            last_positions.append(last)
            if not finished[i]:
                gate[i] = 1.0
                positions[i, last] = 1.0

        hook.set(gate, positions)
        try:
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits
        finally:
            hook.clear()

        active = False
        for i, last in enumerate(last_positions):
            if finished[i]:
                continue
            token = sample_from_logits(
                logits[i, last, :], temperature=temperature, top_p=top_p
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


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    leaks = sum(bool(row["true_or_alias_leak"]) for row in rows)
    abstain = sum(bool(row["semantic_abstention"]) for row in rows)
    clean_abstain = sum(
        bool(row["semantic_abstention"] and not row["true_or_alias_leak"]) for row in rows
    )
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
    random.seed(int(args.seed) + 71103)
    torch.manual_seed(int(args.seed) + 71103)
    runtime = adv.load_runtime(args)
    try:
        base_rows = adv.base_sensitive_prompts(runtime["forget_records"], runtime["router"])
        if args.scope == "canonical":
            base_rows = [row for row in base_rows if row["group"] == "rewrite"]

        attack_rows = []
        for row in base_rows:
            for attack_name, instruction in adv.RETRIEVAL_ATTACKS.items():
                attack_rows.append(
                    {
                        **dict(row),
                        "attack": attack_name,
                        "attack_prompt": adv.build_attack_prompt(str(row["prompt"]), instruction),
                    }
                )

        all_rows = []
        for temperature in args.temperature_values:
            for sample_index in range(int(args.samples_per_prompt)):
                for chunk in ev._chunks(attack_rows, args.batch_size):
                    prompts = [str(row["attack_prompt"]) for row in chunk]
                    outputs = sampled_generate_rsnr(
                        runtime["model"],
                        runtime["hook"],
                        runtime["tok"],
                        prompts,
                        runtime["device"],
                        temperature=temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                    )
                    for source, output in zip(chunk, outputs):
                        pair = (str(source["subject"]), str(source["relation_id"]))
                        aliases = list(runtime["aliases"][pair])
                        abstention = ev.detect_abstention(output)
                        leaks = ev.leaked_aliases(output, aliases)
                        all_rows.append(
                            {
                                **dict(source),
                                "temperature": float(temperature),
                                "top_p": float(args.top_p),
                                "sample_index": int(sample_index),
                                "generation": output,
                                "semantic_abstention": bool(abstention["semantic"]),
                                "exact_abstention": bool(abstention["exact"]),
                                "leaked_aliases": leaks,
                                "true_or_alias_leak": bool(leaks),
                            }
                        )

        by_temperature = {}
        for temperature in args.temperature_values:
            subset = [row for row in all_rows if row["temperature"] == float(temperature)]
            by_temperature[str(temperature)] = summarize(subset)
        by_attack = {}
        for attack in sorted({str(row["attack"]) for row in all_rows}):
            by_attack[attack] = summarize([row for row in all_rows if row["attack"] == attack])

        result = {
            "method": "rsnr_v1a_oracle_sampling_attack",
            "development_only": True,
            "scope": args.scope,
            "temperatures": args.temperature_values,
            "top_p": float(args.top_p),
            "samples_per_prompt": int(args.samples_per_prompt),
            "retrieval_attacks_only": True,
            "true_answer_present_in_attack_prompt": False,
            "artifact_validation": runtime["validation"],
            "overall": summarize(all_rows),
            "by_temperature": by_temperature,
            "by_attack": by_attack,
            "per_sample": all_rows,
            "claim_boundary": {
                "stochastic_robustness_audit": True,
                "checkpoint_not_retrained_or_selected": True,
                "latent_knowledge_erasure_claimed": False,
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
        print(f"RSNR stochastic attack result: {out}")
    finally:
        runtime["hook"].remove()


if __name__ == "__main__":
    main()
