#!/usr/bin/env python3
"""Run Setting 5e + protected LM-head active repair on MQuAKE-CF-3k-v2.

Scientific protocol
-------------------
* Sample MQuAKE instances with the ZeroUnlearn first-half retain / second-half
  forget protocol, then flatten ``requested_rewrite`` only after sampling.
* Forget the original MQuAKE ``target_true`` answer.  Do not train toward the
  benchmark counterfactual ``target_new``; that would be editing, not erasure.
* Use the one-token neutral answer ``Unknown`` as the desired internal target.
* Run the established 600-step Setting 5e embedding+LM-head stage.
* During active repair freeze transformer/input embeddings and modify only the
  neutral output row.  Active states come only from the cloze rewrite prompts.
* MQuAKE's natural-language atomic questions and the three official multi-hop
  questions are evaluation-only.  They never select active repair states.
* Accept a repaired candidate only if Eff reaches the requested target and
  retain/PPL gates pass; otherwise restore Setting 5e.

The main paper-comparable MQuAKE metric is ZeroUnlearn-compatible ``Eff`` plus
PPL. ``AtomicGen`` is reported as a held-out extension and is not used for
checkpoint selection.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_zero_unlearn_official_eval as mquake
import zsre_gagd_setting5e_active_repair as repair


METHOD = "mquake_gagd_setting5e_active_repair"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE


def build_parser() -> argparse.ArgumentParser:
    # Reuse the already-tested Setting 5e/repair argument surface, then add
    # MQuAKE-specific source fields and change dataset-size defaults.
    parser = repair.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="outputs/mquake_setting5e_active/seed0",
        seed=0,
        forget_num=1000,
        retain_num=1000,
        steps=600,
        repair_steps=600,
        target_eff_max=0.0,
        # The inherited Gen gate is not a paper metric for MQuAKE.  Keep it
        # disabled internally; held-out AtomicGen is evaluated after selection.
        target_gen_max=100.0,
        dataset="zsre",  # generic non-TOFU loss/token handling in gagd_compare
    )
    parser.add_argument(
        "--mquake-path",
        default=f"data/{mquake.MQUAKE_FILENAME}",
    )
    parser.add_argument("--mquake-url", default=mquake.MQUAKE_URL)
    parser.add_argument(
        "--require-atomic-gen-zero",
        action="store_true",
        help=(
            "Fail after final held-out evaluation when AtomicGen is nonzero. "
            "AtomicGen is still never used to choose or tune the checkpoint."
        ),
    )
    return parser


def canonical_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
) -> List[gagd.Example]:
    """Map MQuAKE original facts into Setting 5e unwanted/neutral semantics."""

    mquake.resolve_neutral_target_token_id(tok)
    examples: List[gagd.Example] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        subject = str(rewrite["subject"])
        sensitive = gagd.normalize_answer(str(rewrite["target_true"]["str"]))
        neutral = str(rewrite["target_new"]["str"])
        if neutral != mquake.NEUTRAL_TARGET:
            raise ValueError(
                f"Expected neutral target {mquake.NEUTRAL_TARGET!r}, got {neutral!r}"
            )
        examples.append(
            gagd.Example(
                prompt=str(rewrite["prompt"]).format(subject),
                answer=sensitive,
                subject=subject,
                # Setting 5e's internal MCF convention: target_new is unwanted.
                target_new=sensitive,
                target_true=neutral,
                # Official MQuAKE questions are deliberately evaluation-only.
                paraphrase_prompts=[],
                source="mquake",
            )
        )
    return examples


def save_checkpoint(model: nn.Module, tok: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    gagd.write_json(
        output_dir / "mquake_neutral_target.json",
        {
            "neutral_target": mquake.NEUTRAL_TARGET,
            "neutral_token_id": mquake.resolve_neutral_target_token_id(tok),
            "single_token_verified": True,
            "mquake_revision": mquake.MQUAKE_REV,
        },
    )


def compact_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "method": result["method"],
        "forget": {
            key: result["forget"].get(key)
            for key in (
                "Eff",
                "Eff_micro",
                "Eff_instance_macro",
                "AtomicGen",
                "AtomicGen_micro",
                "AtomicGen_instance_macro",
                "num_instances",
                "num_atomic_facts",
            )
        },
        "retain": {
            key: result["retain"].get(key)
            for key in (
                "Eff",
                "Eff_micro",
                "Eff_instance_macro",
                "AtomicGen",
                "AtomicGen_micro",
                "AtomicGen_instance_macro",
                "num_instances",
                "num_atomic_facts",
            )
        },
        "PPL": result.get("forget_PPL"),
    }


def utility_gate_report(
    setting5: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    utility_drop_tolerance: float,
    max_ppl_ratio: float,
    target_eff_max: float,
) -> Dict[str, Any]:
    """MQuAKE-native gate: forget Eff, retain atomic utility, and PPL."""

    checks: Dict[str, Dict[str, Any]] = {}

    def maximum(name: str, before: Any, after: Any, limit: float) -> None:
        passed = after is None or float(after) <= float(limit)
        checks[name] = {
            "setting5": before,
            "candidate": after,
            "maximum": float(limit),
            "passed": bool(passed),
        }

    def minimum(name: str, before: Any, after: Any, limit: float) -> None:
        passed = after is None or float(after) >= float(limit)
        checks[name] = {
            "setting5": before,
            "candidate": after,
            "minimum": float(limit),
            "passed": bool(passed),
        }

    maximum(
        "forget_Eff_non_regression",
        setting5["forget"]["Eff"],
        candidate["forget"]["Eff"],
        float(setting5["forget"]["Eff"]),
    )
    maximum(
        "forget_Eff_target",
        setting5["forget"]["Eff"],
        candidate["forget"]["Eff"],
        float(target_eff_max),
    )
    for metric in ("Eff", "Eff_micro", "Eff_instance_macro"):
        before = setting5["retain"].get(metric)
        after = candidate["retain"].get(metric)
        if before is not None and after is not None:
            minimum(
                f"retain_{metric}",
                before,
                after,
                float(before) - float(utility_drop_tolerance),
            )

    before_ppl = setting5.get("forget_PPL")
    after_ppl = candidate.get("forget_PPL")
    if before_ppl is not None and after_ppl is not None:
        maximum(
            "PPL",
            before_ppl,
            after_ppl,
            float(before_ppl) * float(max_ppl_ratio),
        )

    return {
        "passed": all(item["passed"] for item in checks.values()),
        "utility_drop_tolerance_percentage_points": float(utility_drop_tolerance),
        "max_ppl_ratio": float(max_ppl_ratio),
        "target_eff_max": float(target_eff_max),
        "checks": checks,
    }


def evaluate_eff_only(
    *,
    method: str,
    model: nn.Module,
    tok: Any,
    model_dir: Any,
    mquake_path: Path,
    wikidata_dir: Path,
    out_path: Path,
    args: argparse.Namespace,
    records: Any,
) -> Dict[str, Any]:
    return mquake.evaluate_loaded_model_official(
        method=method,
        model=model,
        tok=tok,
        model_dir=model_dir,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=out_path,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        include_atomic_gen=False,
        mquake_url=args.mquake_url,
        records=records,
    )


def evaluate_extension(
    *,
    method: str,
    model: nn.Module,
    tok: Any,
    model_dir: Any,
    mquake_path: Path,
    wikidata_dir: Path,
    out_path: Path,
    args: argparse.Namespace,
    records: Any,
) -> Dict[str, Any]:
    # No checkpoint choice is made from this result.
    return mquake.evaluate_loaded_model_official(
        method=method,
        model=model,
        tok=tok,
        model_dir=model_dir,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=out_path,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=True,
        include_atomic_gen=True,
        mquake_url=args.mquake_url,
        records=records,
    )


def main() -> None:
    args = build_parser().parse_args()
    repair.validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting5_dir = output_dir / "setting5e"
    repair_dir = output_dir / "active_repair"
    setting5_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)

    mquake_path = Path(args.mquake_path)
    if not mquake_path.is_absolute():
        mquake_path = gagd.PROJECT_DIR / mquake_path
    mquake_path = mquake.download_mquake(mquake_path, url=args.mquake_url)
    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)

    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "setting5_mode": SETTING5_MODE,
            "dataset": mquake.MQUAKE_FILENAME,
            "dataset_revision": mquake.MQUAKE_REV,
            "semantic_mapping": {
                "unwanted": "MQuAKE requested_rewrite.target_true (original fact)",
                "desired": mquake.NEUTRAL_TARGET,
                "benchmark_target_new": "provenance only; never a training target",
            },
            "evaluation_only": [
                "requested_rewrite.question (AtomicGen)",
                "questions (three official multi-hop questions)",
            ],
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading base model and source-locked MQuAKE split")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    forget_records, retain_records = mquake.load_official_eval_records(
        mquake_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        mquake_url=args.mquake_url,
    )
    records = (forget_records, retain_records)
    mquake.write_split_manifest(
        output_dir / "split_manifest.json",
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    forget_examples = canonical_examples(forget_records, tok)
    retain_examples = canonical_examples(retain_records, tok)
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)

    print("Evaluating base MQuAKE Eff/PPL preflight")
    base_result = evaluate_eff_only(
        method="Base",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "base_official_eval.json",
        args=args,
        records=records,
    )
    base_extension = evaluate_extension(
        method="Base held-out extension",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "base_atomic_gen_eval.json",
        args=args,
        records=records,
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training 600-step Setting 5e on MQuAKE atomic sensitive facts")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    training_neutral_id = mquake.resolve_neutral_target_token_id(tok)
    if training_neutral_id != neutral_token_id:
        raise RuntimeError("Neutral token ID changed between model loads")

    # Preserve the neutral row for active repair instead of restoring it into a
    # token-overlap group during Setting 5e's post-training row restoration.
    args.post_training_excluded_token_ids = [neutral_token_id]
    requested_save = bool(args.save_model)
    args.save_model = False
    train_summary = gagd.train_mode(
        model,
        tok,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=SETTING5_MODE,
        args=args,
        mode_dir=setting5_dir,
    )
    args.save_model = requested_save

    setting5_result = evaluate_eff_only(
        method="Setting 5e",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=setting5_dir / "official_eval.json",
        args=args,
        records=records,
    )
    setting5_extension = evaluate_extension(
        method="Setting 5e held-out extension",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=setting5_dir / "atomic_gen_eval.json",
        args=args,
        records=records,
    )
    if args.save_setting5_checkpoint:
        save_checkpoint(model, tok, setting5_dir / "checkpoint")

    print("Building active forget and protected retain token states")
    output_layer = active.freeze_model_for_output_repair(model)
    device = next(model.parameters()).device
    llama_like = mquake.is_llama_like(model, tok)
    original_neutral_row = output_layer.weight[neutral_token_id].detach().clone()

    forget_cases = [
        case
        for record in forget_records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    forget_caches = repair.cache_prediction_cases(
        model,
        tok,
        forget_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        desc="cache MQuAKE forget rewrite tokens",
    )
    active_caches = [
        cache
        for cache in forget_caches
        if cache.correct and cache.target_token_id != neutral_token_id
    ]

    calibration_records = repair._sample_retain_records(
        retain_records,
        args.retain_calibration_num,
        args.retain_calibration_seed,
    )
    retain_context_cases = [
        case
        for record in calibration_records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    retain_caches = repair.cache_prediction_cases(
        model,
        tok,
        retain_context_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        desc="cache MQuAKE retain-calibration tokens",
    )
    protected_caches = [
        cache
        for cache in retain_caches
        if cache.correct and cache.target_token_id != neutral_token_id
    ]
    repair.write_jsonl(
        repair_dir / "active_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in active_caches],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in protected_caches],
    )

    print(
        f"Optimizing only LM-head neutral row {neutral_token_id}: "
        f"active={len(active_caches)}, protected={len(protected_caches)}"
    )
    delta_rows, repair_logs, repair_optimization = repair.optimize_neutral_delta(
        active_caches,
        protected_caches,
        hidden_size=output_layer.weight.shape[1],
        device=device,
        args=args,
    )
    repair.write_jsonl(repair_dir / "repair_log.jsonl", repair_logs)

    protected_cases = [cache.case for cache in protected_caches]
    (
        selected_scale,
        scale_reports,
        exact_active_after,
        exact_protected_after,
        exact_zero_baseline,
    ) = repair.exact_bf16_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        active_cases=forget_cases,
        protected_cases=protected_cases,
        active_context_cases=forget_cases,
        protected_context_cases=retain_context_cases,
        scales=repair.parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    gagd.write_json(repair_dir / "bf16_exact_scale_sweep.json", scale_reports)
    gagd.write_json(repair_dir / "exact_zero_scale_baseline.json", exact_zero_baseline)
    selected_scale_report = next(
        row for row in scale_reports if float(row["scale"]) == float(selected_scale)
    )
    repair.write_jsonl(
        repair_dir / "active_tokens_after.jsonl",
        [repair.cache_report(cache) for cache in exact_active_after],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_after.jsonl",
        [repair.cache_report(cache) for cache in exact_protected_after],
    )

    # The Eff-only evaluator uses the identical forget-case sequence/batching,
    # so this also audits BF16 alignment of the repair sweep.
    candidate_result = evaluate_eff_only(
        method="Setting 5e + active LM-head repair candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:mquake_active_candidate",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=repair_dir / "candidate_official_eval.json",
        args=args,
        records=records,
    )
    gate_report = utility_gate_report(
        setting5_result,
        candidate_result,
        utility_drop_tolerance=args.utility_drop_tolerance,
        max_ppl_ratio=args.max_ppl_ratio,
        target_eff_max=args.target_eff_max,
    )
    target_met = float(candidate_result["forget"]["Eff"]) <= float(args.target_eff_max)
    target_already_met = float(setting5_result["forget"]["Eff"]) <= float(args.target_eff_max)
    local_success = bool(
        target_already_met
        or (
            float(selected_scale) > 0.0
            and int(selected_scale_report["active_correct_tokens"]) == 0
            and int(selected_scale_report["protected_incremental_regressions_vs_zero"]) == 0
        )
    )
    accepted = bool(
        target_met
        and local_success
        and (bool(gate_report["passed"]) or not args.strict_utility_gates)
    )

    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selected_result["method"] = "Setting 5e + accepted active LM-head repair"
        selection_reason = "candidate_passed_eff_retain_ppl_gates"
    else:
        with torch.no_grad():
            output_layer.weight[neutral_token_id].copy_(original_neutral_row)
        selected_result = copy.deepcopy(setting5_result)
        selected_result["method"] = "Setting 5e (active candidate rejected)"
        if not target_met:
            selection_reason = "candidate_missed_required_zero_eff_target"
        elif not local_success:
            selection_reason = "no_bf16_safe_zero_active_scale"
        else:
            selection_reason = "candidate_failed_retain_or_ppl_gate"

    # Only now expose the held-out natural-language atomic questions.
    selected_extension = evaluate_extension(
        method="Selected held-out AtomicGen extension",
        model=model,
        tok=tok,
        model_dir="in-memory:selected",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "selected_atomic_gen_eval.json",
        args=args,
        records=records,
    )

    if args.save_selected_checkpoint:
        save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    repair_summary = {
        "method": METHOD,
        "neutral_token": mquake.NEUTRAL_TARGET,
        "neutral_token_id": neutral_token_id,
        "transformer_frozen_during_repair": True,
        "input_embeddings_frozen_during_repair": True,
        "selected_lm_head_row_count": 1,
        "official_atomic_questions_used_for_repair": False,
        "official_multihop_questions_used_for_repair": False,
        "active_tokens_before": len(active_caches),
        "active_tokens_zero_scale": exact_zero_baseline["active_correct_tokens_at_zero"],
        "active_tokens_after_candidate": int(selected_scale_report["active_correct_tokens"]),
        "protected_tokens": len(protected_caches),
        "protected_incremental_regressions_after_candidate": int(
            selected_scale_report["protected_incremental_regressions_vs_zero"]
        ),
        "candidate_scale": float(selected_scale),
        "selected_scale": float(selected_scale) if accepted else 0.0,
        "optimization": repair_optimization,
        "utility_gates": gate_report,
        "candidate_accepted": accepted,
        "selection_reason": selection_reason,
    }
    gagd.write_json(repair_dir / "repair_summary.json", repair_summary)

    result = {
        "method": METHOD,
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "dataset_sha256": mquake.file_sha256(mquake_path),
        "seed": int(args.seed),
        "forget_num_instances": int(args.forget_num),
        "retain_num_instances": int(args.retain_num),
        "forget_num_atomic_facts": len(forget_records),
        "retain_num_atomic_facts": len(retain_records),
        "training": {
            **asdict(train_summary),
            "steps": int(args.steps),
            "emb_lm_lr": float(args.emb_lm_lr),
            "forget_weight": float(args.forget_weight),
            "retain_weight": float(args.retain_weight),
            "forget_margin": float(args.forget_margin),
        },
        "repair": repair_summary,
        "base": compact_metrics(base_result),
        "base_extension": compact_metrics(base_extension),
        "setting5e": compact_metrics(setting5_result),
        "setting5e_extension": compact_metrics(setting5_extension),
        "candidate": compact_metrics(candidate_result),
        "selected": compact_metrics(selected_result),
        "selected_extension": compact_metrics(selected_extension),
        "paper_reporting": {
            "main_zero_unlearn_compatible": {
                "Eff_down": selected_result["forget"]["Eff"],
                "PPL_down": selected_result.get("forget_PPL"),
            },
            "held_out_extension": {
                "AtomicGen_down": selected_extension["forget"].get("AtomicGen"),
                "RetainEff_up": selected_result["retain"].get("Eff"),
                "RetainAtomicGen_up": selected_extension["retain"].get("AtomicGen"),
            },
        },
    }
    gagd.write_json(output_dir / "mquake_results.json", result)

    print(
        "Selected MQuAKE result: "
        f"Eff={selected_result['forget']['Eff']}, "
        f"AtomicGen(held-out)={selected_extension['forget'].get('AtomicGen')}, "
        f"RetainEff={selected_result['retain'].get('Eff')}, "
        f"PPL={selected_result.get('forget_PPL')}; "
        f"active candidate accepted={accepted}"
    )
    if args.fail_if_target_missed and float(selected_result["forget"]["Eff"]) > float(args.target_eff_max):
        raise RuntimeError(
            "MQuAKE selected checkpoint missed the required Eff target: "
            f"Eff={selected_result['forget']['Eff']} > {args.target_eff_max}"
        )
    if args.require_atomic_gen_zero:
        atomic_gen = selected_extension["forget"].get("AtomicGen")
        if atomic_gen is None or float(atomic_gen) > 0.0:
            raise RuntimeError(
                "Held-out MQuAKE AtomicGen was not zero. The checkpoint is left "
                "untuned with respect to this test metric by design: "
                f"AtomicGen={atomic_gen}"
            )


if __name__ == "__main__":
    main()
