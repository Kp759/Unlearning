#!/usr/bin/env python3
"""Official ZeroUnlearn-parity evaluation for RSNR-V1C bounded PreHead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import mcf_rsnr_v1a_prehead_official_eval as prehead_eval
from mcf_zero_unlearn_metric_parity import apply_zero_unlearn_eff_gen
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead
import run_mcf_rsnr_v1c_prehead_bounded_unlearn as v1c


PROTOCOL = v1c.PROTOCOL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
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


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _membership_from_payload(rows: list[Mapping[str, Any]]) -> set[tuple[int, str, str]]:
    return {
        (int(r["case_id"]), str(r["subject"]), str(r["relation_id"]))
        for r in rows
    }


def load_and_validate_artifacts(run_dir: Path, locked_forget, manifest):
    method_dir = run_dir / "method"
    adapter_path = method_dir / "rsnr_prehead_bounded_unlearn_adapter.pt"
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    sidecar = _load_json(method_dir / "relation_scoped_prehead_bounded_unlearn.json")
    completion = _load_json(method_dir / "completion.json")
    report = _load_json(method_dir / "rsnr_v1c_prehead_bounded_unlearn.json")

    for name, payload in (("adapter", adapter), ("sidecar", sidecar), ("completion", completion), ("report", report)):
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{name} protocol mismatch: {payload.get('protocol')!r}")
    if adapter.get("intervention_site") != prehead.INTERVENTION_SITE:
        raise RuntimeError("adapter intervention site mismatch")
    if adapter.get("transformer_weights_modified") is not False:
        raise RuntimeError("artifact does not certify frozen Transformer")
    if adapter.get("lm_head_weights_modified") is not False:
        raise RuntimeError("artifact does not certify frozen LM head")
    if adapter.get("target_new_positive_likelihood_training") is not False:
        raise RuntimeError("V1C must not positively train target_new")
    if completion.get("training_gate_passed") is not True:
        raise RuntimeError("training artifact did not pass V1C training-safe conditions")
    if completion.get("all_four_conditions_passed") is not True:
        raise RuntimeError("training artifact did not pass all four requested conditions")
    if completion.get("safe_five_conditions_passed") is not True:
        raise RuntimeError("training artifact did not preserve target_new near Base")

    locked_membership = ev._membership_rows(locked_forget, source="training_visible_forget_direct")
    artifact_membership = _membership_from_payload(adapter.get("forget_membership", []))
    if artifact_membership != locked_membership:
        raise RuntimeError("adapter forget membership does not exactly match locked split")
    expected_ids = set(ev._expected_forget_ids(manifest))
    if {cid for cid, _s, _r in artifact_membership} != expected_ids:
        raise RuntimeError("artifact forget IDs do not match split manifest")

    validation = {
        "passed": True,
        "protocol": PROTOCOL,
        "membership_count": len(artifact_membership),
        "locked_forget_exact_match": True,
        "manifest_forget_ids_exact_match": True,
        "training_gate_passed": True,
        "all_four_conditions_passed": True,
        "target_new_base_preservation_passed": True,
        "target_new_positive_likelihood_training": False,
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
    }
    return adapter, adapter_path, validation


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = ev._load_manifest(protocol_dir)
    locked_forget = ev._load_locked_forget(protocol_dir)
    adapter_payload, adapter_path, validation = load_and_validate_artifacts(
        run_dir, locked_forget, manifest
    )

    membership = ev._membership_rows(locked_forget, source="training_visible_forget_direct")
    router = ev.OraclePromptRouter(membership)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for official evaluation")

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

    try:
        data = official.load_mcf(official.download_mcf(args.mcf_path))
        forget_records, retain_records, selection = ev.fresh_split(
            data,
            manifest,
            unlearn_num=args.unlearn_num,
            retain_num=args.retain_num,
            seed=args.seed,
            fresh_retain_seed=args.fresh_retain_seed,
        )
        official_membership = ev._membership_rows(forget_records, source="official forget sample")
        if official_membership != membership:
            raise RuntimeError("official forget metadata does not match locked artifacts")

        aliases = ev.build_true_alias_map(data, forget_records)
        llama_like = official.is_llama_like(model, tok)
        forget_summary, forget_raw, forget_routing, forget_sensitive = ev.evaluate_split(
            model, hook, tok, forget_records, router, device,
            llama_like=llama_like, split_name="forget",
        )
        retain_summary, retain_raw, retain_routing, retain_sensitive = ev.evaluate_split(
            model, hook, tok, retain_records, router, device,
            llama_like=llama_like, split_name="retain",
        )
        forget_summary = apply_zero_unlearn_eff_gen(forget_summary)
        retain_summary = apply_zero_unlearn_eff_gen(retain_summary)

        sensitive_prompts = [*forget_sensitive, *retain_sensitive]
        teacher = ev.rsnr_native_teacher_forced_audit(
            model, hook, tok, sensitive_prompts, aliases, device,
            batch_size=args.generation_batch_size,
        )
        generated = ev.generation_audit(
            model, hook, tok, sensitive_prompts, aliases, device,
            max_new_tokens=args.generation_max_new_tokens,
            batch_size=args.generation_batch_size,
        )
        aligned = prehead_eval.summarize_method_aligned(
            teacher["per_prompt"], generated["per_prompt"]
        )

        ppl = None
        if not args.skip_ppl:
            ppl_text = official.load_official_ppl_text(args.wikidata_dir)
            if ppl_text is None:
                print(f"[warning] wikidata dir {args.wikidata_dir} not found. PPL set to null.")
            else:
                hook.clear()
                ppl = official.official_perplexity(model, tok, ppl_text, device, max_input_length=100)

        result = {
            "method": "rsnr_v1c_prehead_bounded_unlearn",
            "variant": v1c.VARIANT,
            "protocol": PROTOCOL,
            "model_dir": base_model,
            "adapter_path": str(adapter_path),
            "dataset": "MCF",
            "sample_mode": "official_compatible_fresh_disjoint_retain",
            "seed": int(args.seed),
            "unlearn_num": int(args.unlearn_num),
            "retain_num": int(args.retain_num),
            "development_only": True,
            "intervention_site": prehead.INTERVENTION_SITE,
            "trainable_parameters": 2 * int(adapter_payload["hidden_size"]) * int(adapter_payload["adapter_rank"]),
            "transformer_weights_modified": False,
            "lm_head_weights_modified": False,
            "target_new_training": {
                "used_as_cf_reference": True,
                "positive_likelihood_training": False,
                "preserved_near_frozen_base": True,
            },
            "artifact_validation": validation,
            "zero_unlearn_parity": {
                "Eff": "post_rewrite_success: mean[NLL(target_true) > NLL(target_new)] * 100",
                "Gen": "post_paraphrase_success: mean[NLL(target_true) > NLL(target_new)] * 100",
                "exact_mapping_applied": True,
            },
            "forget": forget_summary,
            "retain": retain_summary,
            "forget_PPL": ppl,
            "retain_PPL": ppl,
            "method_aligned_sensitive_answer": aligned,
            "rsnr_native_teacher_forced": teacher,
            "rsnr_generation_audit": generated,
            "routing_audit": {
                "forget": forget_routing,
                "retain": retain_routing,
                "sensitive_prompt_count_total": len(sensitive_prompts),
                "routing_policy": "per_prompt_subject_resolution_plus_relation_metadata",
            },
            "fresh_retain_selection": selection,
            "forget_raw": forget_raw,
            "retain_raw": retain_raw,
        }
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "zero_unlearn": {
                "forget_Eff": forget_summary["Eff"],
                "forget_Gen": forget_summary["Gen"],
                "forget_Spe": forget_summary["Spe"],
                "retain_Eff": retain_summary["Eff"],
                "retain_Gen": retain_summary["Gen"],
                "PPL": ppl,
            },
            "method_aligned_sensitive_answer": aligned,
            "generation_summary": {k: v for k, v in generated.items() if k != "per_prompt"},
            "artifact_validation": validation,
        }, indent=2))
        print(f"RSNR-V1C PreHead official result: {out}")
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
