#!/usr/bin/env python3
"""Greedy adversarial nondisclosure evaluation for RSNR direct-logit baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_adversarial_eval as adv
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_logitmask_common as lm
import mcf_rsnr_v1a_logitmask_official_eval as leval


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
    p.add_argument("--max-new-tokens", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        p.error("development evaluation is locked to seed1/forget50/retain1000")
    return args


def load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    config = lm.load_config(Path(args.config).resolve())
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = ev._load_manifest(protocol_dir)
    locked = ev._load_locked_forget(protocol_dir)
    validation = leval.validate_config(config, locked, manifest)
    membership = ev._membership_rows(locked, source="training_visible_forget_direct")
    router = ev.OraclePromptRouter(membership)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for direct-logit adversarial evaluation")
    tok = AutoTokenizer.from_pretrained(str(config["base_model"]), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(config["base_model"]),
        torch_dtype=official.dtype_from_str(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    canonical = lm.canonical_answer_map(locked)
    token_sets = lm.canonical_token_sets(tok, canonical)
    hook = lm.DirectLogitMaskHook.install(
        lm.get_lm_head(model),
        true_penalty=float(config["true_penalty"]),
        idk_boost=float(config["idk_boost"]),
        idk_token_ids=lm.abstention_token_set(tok),
    )

    data = official.load_mcf(official.download_mcf(args.mcf_path))
    forget_records, retain_records, selection = ev.fresh_split(
        data, manifest,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        fresh_retain_seed=args.fresh_retain_seed,
    )
    if ev._membership_rows(forget_records, source="official forget sample") != membership:
        raise RuntimeError("official forget metadata mismatch")
    aliases = ev.build_true_alias_map(data, forget_records)
    return {
        "config": config,
        "validation": validation,
        "router": router,
        "device": device,
        "tok": tok,
        "model": model,
        "hook": hook,
        "canonical": canonical,
        "token_sets": token_sets,
        "forget_records": forget_records,
        "retain_records": retain_records,
        "selection": selection,
        "aliases": aliases,
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, retrieval: bool) -> dict[str, Any]:
    total = len(rows)
    leaks = sum(bool(r["true_or_alias_leak"]) for r in rows)
    semantic = sum(bool(r["semantic_abstention"]) for r in rows)
    out = {
        "prompt_count": total,
        "semantic_abstention_count": semantic,
        "semantic_abstention_rate": semantic / total if total else None,
        "true_or_alias_output_count": leaks,
        "true_or_alias_output_rate": leaks / total if total else None,
    }
    if retrieval:
        out["nondisclosure_pass_count"] = total - leaks
        out["nondisclosure_pass_rate"] = (total - leaks) / total if total else None
        out["abstaining_nondisclosure_pass_count"] = sum(
            bool(r["semantic_abstention"] and not r["true_or_alias_leak"]) for r in rows
        )
        out["abstaining_nondisclosure_pass_rate"] = (
            out["abstaining_nondisclosure_pass_count"] / total if total else None
        )
    return out


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args)
    try:
        base_rows = adv.base_sensitive_prompts(runtime["forget_records"], runtime["router"])
        sensitive = [
            {
                "split": "forget",
                "group": str(r["group"]),
                "parent_case_id": int(r["case_id"]),
                "parent_subject": str(r["subject"]),
                "parent_relation_id": str(r["relation_id"]),
                "prompt": str(r["prompt"]),
                "matched_case_id": int(r["case_id"]),
                "matched_subject": str(r["subject"]),
                "matched_relation_id": str(r["relation_id"]),
            }
            for r in base_rows
        ]
        teacher = lm.native_teacher_forced_audit(
            runtime["model"], runtime["hook"], runtime["tok"], sensitive,
            runtime["canonical"], runtime["token_sets"], runtime["device"],
            batch_size=args.batch_size,
        )
        native = lm.summarize_method_aligned(teacher["per_prompt"], [])

        attack_rows = []
        for row in base_rows:
            for attack_name, instruction in adv.RETRIEVAL_ATTACKS.items():
                attack_rows.append({
                    **dict(row),
                    "attack": attack_name,
                    "attack_class": "retrieval_no_true_answer_in_prompt",
                    "attack_prompt": adv.build_attack_prompt(str(row["prompt"]), instruction),
                })
            attack_rows.append({
                **dict(row),
                "attack": "forced_choice_true_vs_counterfact_reference",
                "attack_class": "answer_present_diagnostic_not_counted_as_retrieval_leakage",
                "attack_prompt": adv.build_forced_choice_prompt(
                    str(row["prompt"]), str(row["target_true"]), str(row["target_new"])
                ),
            })

        outputs = []
        for chunk in ev._chunks(attack_rows, args.batch_size):
            prompts = [str(r["attack_prompt"]) for r in chunk]
            pairs = [(str(r["subject"]), str(r["relation_id"])) for r in chunk]
            outputs.extend(lm.greedy_generate_logitmask(
                runtime["model"], runtime["hook"], runtime["tok"], prompts, pairs,
                runtime["token_sets"], runtime["device"], max_new_tokens=args.max_new_tokens,
            ))

        rows = []
        for source, output in zip(attack_rows, outputs):
            pair = (str(source["subject"]), str(source["relation_id"]))
            aliases = list(runtime["aliases"][pair])
            abstention = ev.detect_abstention(output)
            leaks = ev.leaked_aliases(output, aliases)
            retrieval = source["attack_class"] == "retrieval_no_true_answer_in_prompt"
            rows.append({
                **dict(source),
                "generation": output,
                "semantic_abstention": bool(abstention["semantic"]),
                "exact_abstention": bool(abstention["exact"]),
                "leaked_aliases": leaks,
                "true_or_alias_leak": bool(leaks),
                "counts_toward_retrieval_nondisclosure": retrieval,
            })
        retrieval_rows = [r for r in rows if r["counts_toward_retrieval_nondisclosure"]]
        forced_rows = [r for r in rows if not r["counts_toward_retrieval_nondisclosure"]]
        by_attack = {}
        for name in sorted({str(r["attack"]) for r in rows}):
            subset = [r for r in rows if r["attack"] == name]
            by_attack[name] = summarize(
                subset,
                retrieval=bool(subset and subset[0]["counts_toward_retrieval_nondisclosure"]),
            )
        result = {
            "method": f"rsnr_direct_logit_{runtime['config']['variant']}_adversarial",
            "variant": runtime["config"]["variant"],
            "development_only": True,
            "true_penalty": runtime["config"]["true_penalty"],
            "idk_boost": runtime["config"]["idk_boost"],
            "trainable_parameters": 0,
            "native_idk_metrics": {
                k: v for k, v in native.items() if "Sensitive_" not in k and "semantic" not in k
            },
            "retrieval_attacks": summarize(retrieval_rows, retrieval=True),
            "answer_present_forced_choice_diagnostic": summarize(forced_rows, retrieval=False),
            "by_attack": by_attack,
            "per_prompt": rows,
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
            "native_idk_metrics": result["native_idk_metrics"],
            "retrieval_attacks": result["retrieval_attacks"],
            "answer_present_forced_choice_diagnostic": result["answer_present_forced_choice_diagnostic"],
            "by_attack": by_attack,
        }, indent=2))
        print(f"RSNR direct-logit adversarial result: {out}")
    finally:
        runtime["hook"].remove()


if __name__ == "__main__":
    main()
