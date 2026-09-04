#!/usr/bin/env python3
"""Official-compatible + method-aligned evaluation for RSNR-V1A-PreHead.

Loads a frozen Base model plus the saved rank-16 adapter attached immediately
before the frozen LM head.  The evaluator reuses the corrected per-prompt oracle
routing and fresh-disjoint-retain protocol from RSNR-V1A.

It reports both:
  * legacy CounterFact target_true-vs-target_new Eff/Gen/Spe/PPL; and
  * method-aligned sensitive-answer metrics: Eff_IDK/Gen_IDK, true-answer
    suppression, greedy abstention, and true/alias leakage.

Seed1 is DEVELOPMENT ONLY and must not be used as final certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import run_mcf_rsnr_v1a_oracle as rsnr
import run_mcf_rsnr_v1a_prehead as prehead


PROTOCOL = prehead.PROTOCOL


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
        p.error("RSNR-V1A-PreHead development eval is locked to seed1/forget50/retain1000")
    if args.generation_max_new_tokens <= 0 or args.generation_batch_size <= 0:
        p.error("generation limits must be positive")
    return args


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_artifacts(run_dir: Path):
    adapter_path = run_dir / "method" / "rsnr_prehead_oracle_null_adapter.pt"
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)
    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    sidecar = _load_json(run_dir / "method" / "relation_scoped_null_routing_prehead.json")
    completion = _load_json(run_dir / "method" / "completion.json")
    for name, payload in (
        ("adapter", adapter_payload), ("sidecar", sidecar), ("completion", completion)
    ):
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{name} protocol mismatch for RSNR-V1A-PreHead")
    if adapter_payload.get("intervention_site") != prehead.INTERVENTION_SITE:
        raise RuntimeError("pre-head adapter intervention_site mismatch")
    if sidecar.get("intervention_site") != prehead.INTERVENTION_SITE:
        raise RuntimeError("pre-head sidecar intervention_site mismatch")
    if adapter_payload.get("lm_head_weights_modified") is not False:
        raise RuntimeError("pre-head adapter artifact does not certify frozen LM head")
    return adapter_payload, sidecar, completion, adapter_path


def summarize_method_aligned(
    teacher_rows: Sequence[Mapping[str, Any]], generation_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Sensitive-answer efficacy/generalization without CounterFact target_new.

    Eff_IDK/Gen_IDK are failure percentages: target_true is at least as probable
    as IDK.  Sensitive_Eff/Sensitive_Gen are literal true/alias generation leak
    percentages.  All four are lower-is-better.
    """
    out: Dict[str, Any] = {
        "definition": {
            "Eff_IDK_Gen_IDK": "percent with logP(IDK)-logP(true) <= 0; lower is better",
            "Sensitive_Eff_Sensitive_Gen": "percent greedy generations leaking true answer or explicit alias; lower is better",
        },
        "registered_abstention": ev.ABSTENTION,
    }
    for group, suffix in (("rewrite", "Eff"), ("paraphrase", "Gen")):
        trows = [
            row for row in teacher_rows
            if row.get("split") == "forget" and row.get("group") == group
        ]
        grows = [
            row for row in generation_rows
            if row.get("split") == "forget" and row.get("group") == group
        ]
        margins = [float(row["idk_vs_true_margin"]) for row in trows]
        drops = [float(row["true_logprob_drop"]) for row in trows]
        fail = sum(margin <= 0.0 for margin in margins)
        buffered_fail = sum(margin < 0.1 for margin in margins)
        leaks = sum(bool(row["true_or_alias_leak"]) for row in grows)
        abstain = sum(bool(row["semantic_abstention"]) for row in grows)
        n_t = len(trows)
        n_g = len(grows)
        out[f"{suffix}_IDK"] = round(100.0 * fail / n_t, 4) if n_t else None
        out[f"{suffix}_IDK_buffered_0p1"] = round(100.0 * buffered_fail / n_t, 4) if n_t else None
        out[f"{suffix}_IDK_prompt_count"] = n_t
        out[f"{suffix}_minimum_idk_vs_true_margin"] = min(margins) if margins else None
        out[f"{suffix}_mean_true_logprob_drop"] = sum(drops) / n_t if n_t else None
        out[f"{suffix}_minimum_true_logprob_drop"] = min(drops) if drops else None
        out[f"Sensitive_{suffix}"] = round(100.0 * leaks / n_g, 4) if n_g else None
        out[f"Sensitive_{suffix}_leak_count"] = leaks
        out[f"Sensitive_{suffix}_prompt_count"] = n_g
        out[f"{suffix}_semantic_abstention_rate"] = abstain / n_g if n_g else None
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = ev._load_manifest(protocol_dir)
    locked_forget = ev._load_locked_forget(protocol_dir)
    adapter_payload, sidecar, completion, adapter_path = load_artifacts(run_dir)

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
        raise RuntimeError("CUDA is required for RSNR-V1A-PreHead evaluation")
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
    official_membership = ev._membership_rows(forget_records, source="official forget sample")
    if official_membership != membership:
        raise RuntimeError("official forget metadata does not match locked pre-head artifacts")

    aliases = ev.build_true_alias_map(data, forget_records)
    llama_like = official.is_llama_like(model, tok)
    forget_summary, forget_raw, forget_routing, forget_sensitive = ev.evaluate_split(
        model, hook, tok, forget_records, router, device,
        llama_like=llama_like, split_name="forget"
    )
    retain_summary, retain_raw, retain_routing, retain_sensitive = ev.evaluate_split(
        model, hook, tok, retain_records, router, device,
        llama_like=llama_like, split_name="retain"
    )
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
    aligned = summarize_method_aligned(teacher["per_prompt"], generated["per_prompt"])

    ppl = None
    if not args.skip_ppl:
        ppl_text = official.load_official_ppl_text(args.wikidata_dir)
        if ppl_text is None:
            print(f"[warning] wikidata dir {args.wikidata_dir} not found. PPL set to null.")
        else:
            hook.clear()
            ppl = official.official_perplexity(model, tok, ppl_text, device, max_input_length=100)

    result = {
        "method": "rsnr_v1a_prehead_oracle_idk_aware",
        "model_dir": base_model,
        "adapter_path": str(adapter_path),
        "dataset": "MCF",
        "sample_mode": "official_compatible_fresh_disjoint_retain",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "retain_num": int(args.retain_num),
        "development_only": True,
        "intervention_site": prehead.INTERVENTION_SITE,
        "lm_head_weights_modified": False,
        "transformer_weights_modified": False,
        "artifact_validation": validation,
        "legacy_counterfact": {
            "forget": forget_summary,
            "retain": retain_summary,
            "forget_PPL": ppl,
            "retain_PPL": ppl,
            "note": "legacy Eff/Gen compare target_true against CounterFact target_new; RSNR does not train target_new",
        },
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
        # Compatibility aliases for the repository comparison helper.
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    comparison = official.result_to_comparison_row(result)
    print(json.dumps({
        "legacy_counterfact": comparison,
        "method_aligned_sensitive_answer": aligned,
        "artifact_validation": validation,
        "routing_audit": result["routing_audit"],
        "generation_summary": {k: v for k, v in generated.items() if k != "per_prompt"},
    }, indent=2))
    print(f"RSNR-V1A-PreHead result: {out}")
    hook.remove()


if __name__ == "__main__":
    main()
