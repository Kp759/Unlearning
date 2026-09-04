#!/usr/bin/env python3
"""Stochastic retrieval-attack audit for RSNR-V1A-PreHead."""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

import mcf_rsnr_v1a_adversarial_eval as adv
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_prehead_adversarial_eval as padv
import mcf_rsnr_v1a_sampling_attack_eval as samp
import run_mcf_rsnr_v1a_prehead as prehead


def main() -> None:
    args = samp.parse_args()
    random.seed(int(args.seed) + 72113)
    torch.manual_seed(int(args.seed) + 72113)
    runtime = padv.load_runtime(args)
    try:
        base_rows = adv.base_sensitive_prompts(runtime["forget_records"], runtime["router"])
        if args.scope == "canonical":
            base_rows = [row for row in base_rows if row["group"] == "rewrite"]

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
                    prompts = [str(row["attack_prompt"]) for row in chunk]
                    outputs = samp.sampled_generate_rsnr(
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
            str(temperature): samp.summarize(
                [row for row in all_rows if row["temperature"] == float(temperature)]
            )
            for temperature in args.temperature_values
        }
        by_attack = {
            attack: samp.summarize([row for row in all_rows if row["attack"] == attack])
            for attack in sorted({str(row["attack"]) for row in all_rows})
        }
        result = {
            "method": "rsnr_v1a_prehead_oracle_sampling_attack",
            "development_only": True,
            "intervention_site": prehead.INTERVENTION_SITE,
            "lm_head_weights_modified": False,
            "transformer_weights_modified": False,
            "scope": args.scope,
            "temperatures": args.temperature_values,
            "top_p": float(args.top_p),
            "samples_per_prompt": int(args.samples_per_prompt),
            "retrieval_attacks_only": True,
            "true_answer_present_in_attack_prompt": False,
            "artifact_validation": runtime["validation"],
            "overall": samp.summarize(all_rows),
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
        print(f"RSNR-V1A-PreHead stochastic attack result: {out}")
    finally:
        runtime["hook"].remove()


if __name__ == "__main__":
    main()
