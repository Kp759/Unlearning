#!/usr/bin/env python3
"""Greedy adversarial nondisclosure audit for RSNR-V1A-PreHead.

Reuses the same attack prompts and method-aligned Eff_IDK/Gen_IDK definitions as
the layer-24 RSNR adversarial evaluator.  The only architectural difference is
that the null adapter is attached immediately before the frozen LM head.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_adversarial_eval as adv
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_prehead_official_eval as peval
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead


def load_runtime(args):
    run_dir = Path(args.run_dir).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = ev._load_manifest(protocol_dir)
    locked_forget = ev._load_locked_forget(protocol_dir)
    adapter_payload, sidecar, completion, adapter_path = peval.load_artifacts(run_dir)
    validation = ev.validate_artifact_correspondence(
        adapter_payload=adapter_payload,
        sidecar=sidecar,
        completion=completion,
        locked_forget=locked_forget,
        manifest=manifest,
        expected_count=args.unlearn_num,
    )
    membership = ev._membership_rows(locked_forget, source="training_visible_forget_direct")
    router = ev.OraclePromptRouter(membership)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RSNR-V1A-PreHead adversarial evaluation")
    base_model = str(adapter_payload["base_model"])
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapter = rsnr.NullResidualAdapter(
        int(adapter_payload["hidden_size"]),
        int(adapter_payload["adapter_rank"]),
        float(adapter_payload["adapter_alpha"]),
        device,
    ).to(device)
    adapter.load_state_dict(adapter_payload["adapter_state_dict"])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    hook = prehead.PreHeadNullHook.install(prehead.get_lm_head(model), adapter)

    data = official.load_mcf(official.download_mcf(args.mcf_path))
    forget_records, retain_records, selection = ev.fresh_split(
        data,
        manifest,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        fresh_retain_seed=args.fresh_retain_seed,
    )
    if ev._membership_rows(forget_records, source="official forget sample") != membership:
        raise RuntimeError("official forget metadata does not match locked pre-head membership")
    aliases = ev.build_true_alias_map(data, forget_records)
    return {
        "run_dir": run_dir,
        "validation": validation,
        "adapter_path": str(adapter_path),
        "selection": selection,
        "router": router,
        "device": device,
        "model": model,
        "tok": tok,
        "hook": hook,
        "forget_records": forget_records,
        "retain_records": retain_records,
        "aliases": aliases,
    }


def main() -> None:
    args = adv.parse_args()
    random.seed(int(args.seed) + 92831)
    torch.manual_seed(int(args.seed) + 92831)
    runtime = load_runtime(args)
    try:
        base_rows = adv.base_sensitive_prompts(runtime["forget_records"], runtime["router"])
        teacher = adv.idk_teacher_forced(runtime, base_rows, batch_size=args.batch_size)
        native_metrics = adv.summarize_idk_metrics(teacher["per_prompt"])
        adversarial = adv.adversarial_greedy_audit(
            runtime,
            base_rows,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )
        result = {
            "method": "rsnr_v1a_prehead_oracle_adversarial_nondisclosure",
            "development_only": True,
            "seed": int(args.seed),
            "intervention_site": prehead.INTERVENTION_SITE,
            "lm_head_weights_modified": False,
            "transformer_weights_modified": False,
            "adapter_path": runtime["adapter_path"],
            "artifact_validation": runtime["validation"],
            "native_idk_metrics": native_metrics,
            "native_teacher_forced": teacher,
            "adversarial_greedy": adversarial,
            "claim_boundary": {
                "retrieval_attacks_do_not_include_true_answer_in_prompt": True,
                "forced_choice_includes_true_answer_and_is_separate_diagnostic": True,
                "latent_knowledge_erasure_claimed": False,
            },
        }
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "native_idk_metrics": native_metrics,
            "retrieval_attacks": adversarial["retrieval_attacks"],
            "answer_present_forced_choice_diagnostic": adversarial[
                "answer_present_forced_choice_diagnostic"
            ],
            "by_attack": adversarial["by_attack"],
        }, indent=2))
        print(f"RSNR-V1A-PreHead adversarial result: {out}")
    finally:
        runtime["hook"].remove()


if __name__ == "__main__":
    main()
