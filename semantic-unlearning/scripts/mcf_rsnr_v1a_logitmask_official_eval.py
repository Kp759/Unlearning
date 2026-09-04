#!/usr/bin/env python3
"""Official-compatible + sensitive-answer evaluation for RSNR direct-logit baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_logitmask_common as lm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--fresh-retain-seed", type=int, default=700002)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--generation-max-new-tokens", type=int, default=20)
    p.add_argument("--generation-batch-size", type=int, default=8)
    args = p.parse_args()
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        p.error("development evaluation is locked to seed1/forget50/retain1000")
    return args


def config_membership(config: Mapping[str, Any]) -> set[tuple[int, str, str]]:
    return {
        (int(r["case_id"]), str(r["subject"]), str(r["relation_id"]))
        for r in config.get("forget_membership", [])
    }


def validate_config(config: Mapping[str, Any], locked: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    locked_membership = ev._membership_rows(locked, source="training_visible_forget_direct")
    if config_membership(config) != locked_membership:
        raise RuntimeError("direct-logit config membership does not match locked forget records")
    expected_ids = set(ev._expected_forget_ids(manifest))
    if {cid for cid, _s, _r in locked_membership} != expected_ids:
        raise RuntimeError("locked forget records do not match split manifest")
    locked_answers = lm.canonical_answer_map(locked)
    config_answers = {
        (str(r["subject"]), str(r["relation_id"])): str(r["target_true"])
        for r in config.get("forget_membership", [])
    }
    if config_answers != locked_answers:
        raise RuntimeError("direct-logit config target_true surfaces do not match locked forget records")
    return {
        "passed": True,
        "membership_count": len(locked_membership),
        "config_locked_forget_exact_match": True,
        "manifest_forget_ids_exact_match": True,
        "aliases_used_for_mask": False,
        "calibration_passed": True,
    }


def main() -> None:
    args = parse_args()
    protocol_dir = Path(args.protocol_dir).resolve()
    config = lm.load_config(Path(args.config).resolve())
    manifest = ev._load_manifest(protocol_dir)
    locked_forget = ev._load_locked_forget(protocol_dir)
    validation = validate_config(config, locked_forget, manifest)
    membership = ev._membership_rows(locked_forget, source="training_visible_forget_direct")
    router = ev.OraclePromptRouter(membership)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for direct-logit evaluation")
    base_model = str(config["base_model"])
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=official.dtype_from_str(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)

    canonical_answers = lm.canonical_answer_map(locked_forget)
    token_sets = lm.canonical_token_sets(tok, canonical_answers)
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
        raise RuntimeError("official forget metadata does not match locked direct-logit membership")
    alias_map = ev.build_true_alias_map(data, forget_records)
    llama_like = official.is_llama_like(model, tok)

    try:
        forget_summary, forget_raw, forget_routing, forget_sensitive = lm.evaluate_split(
            model, hook, tok, forget_records, router, token_sets, device,
            llama_like=llama_like, split_name="forget",
        )
        retain_summary, retain_raw, retain_routing, retain_sensitive = lm.evaluate_split(
            model, hook, tok, retain_records, router, token_sets, device,
            llama_like=llama_like, split_name="retain",
        )
        sensitive_prompts = [*forget_sensitive, *retain_sensitive]
        teacher = lm.native_teacher_forced_audit(
            model, hook, tok, sensitive_prompts, canonical_answers, token_sets, device,
            batch_size=args.generation_batch_size,
        )
        generated = lm.generation_audit(
            model, hook, tok, sensitive_prompts, token_sets, alias_map, device,
            max_new_tokens=args.generation_max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        aligned = lm.summarize_method_aligned(teacher["per_prompt"], generated["per_prompt"])

        ppl = None
        if not args.skip_ppl:
            ppl_text = official.load_official_ppl_text(args.wikidata_dir)
            if ppl_text is not None:
                hook.clear()
                ppl = official.official_perplexity(model, tok, ppl_text, device, max_input_length=100)

        result = {
            "method": f"rsnr_direct_logit_{config['variant']}",
            "variant": config["variant"],
            "model_dir": base_model,
            "dataset": "MCF",
            "development_only": True,
            "seed": int(args.seed),
            "sample_mode": "official_compatible_fresh_disjoint_retain",
            "unlearn_num": int(args.unlearn_num),
            "retain_num": int(args.retain_num),
            "intervention_site": lm.INTERVENTION_SITE,
            "true_penalty": float(config["true_penalty"]),
            "idk_boost": float(config["idk_boost"]),
            "trainable_parameters": 0,
            "transformer_weights_modified": False,
            "lm_head_weights_modified": False,
            "artifact_validation": validation,
            "legacy_counterfact": {
                "forget": forget_summary,
                "retain": retain_summary,
                "forget_PPL": ppl,
                "retain_PPL": ppl,
                "note": "Unlike RSNR-PreHead, direct suppression of canonical sensitive logits can also lower legacy CF Eff/Gen because target_true itself is explicitly penalized.",
            },
            "method_aligned_sensitive_answer": aligned,
            "native_teacher_forced": teacher,
            "generation_audit": generated,
            "routing_audit": {
                "forget": forget_routing,
                "retain": retain_routing,
                "sensitive_prompt_count_total": len(sensitive_prompts),
                "routing_policy": "same per-prompt oracle subject+relation resolver as RSNR",
            },
            "fresh_retain_selection": selection,
            "mask_scope": {
                "canonical_target_true_surface_only": True,
                "heldout_aliases_used_for_mask": False,
                "aliases_used_only_for_leakage_evaluation": True,
            },
            "forget_raw": forget_raw,
            "retain_raw": retain_raw,
            "forget": forget_summary,
            "retain": retain_summary,
            "forget_PPL": ppl,
            "retain_PPL": ppl,
        }
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        comparison = official.result_to_comparison_row(result)
        print(json.dumps({
            "legacy_counterfact": comparison,
            "method_aligned_sensitive_answer": aligned,
            "native_teacher_forced": {k: v for k, v in teacher.items() if k != "per_prompt"},
            "generation_summary": {k: v for k, v in generated.items() if k != "per_prompt"},
            "routing_audit": result["routing_audit"],
        }, indent=2))
        print(f"RSNR direct-logit result: {out}")
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
