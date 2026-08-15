#!/usr/bin/env python3
"""Run the MCF/ZsRE-style RWKU Batch-50 experiment.

One run jointly unlearns 50 RWKU probes: five people x ten probes/person
(8 Level-1 + 2 Level-2).  The exact same 50 probes are evaluated afterward as
forget efficacy.  Content-disjoint remaining Level-1/Level-2 probes measure
held-out generalization; deterministic paraphrases of held-out Level-2 measure
surface generalization; native Level-3 probes measure adversarial recovery.

Exactly 1000 deterministic external MCF retain examples are shared by the
unlearning objective and post-training retain evaluation, matching the common
MCF/ZsRE experimental budget.  A disjoint 128-example MCF partition is used
only as the protected LM-head repair gate.  Model checkpoints are deliberately
not saved by this runner; metrics/manifests/reports are retained.

This is a probe-assisted cross-benchmark method extension, not unchanged native
RWKU protocol.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import rwku_experiment as EXP
from rwku_batch50 import (
    PROTOCOL_ID,
    REPAIR_RETAIN_NUM,
    RETAIN_EVAL_NUM,
    TOTAL_FORGET_TRAIN,
    materialize_batch_split,
)
from rwku_eval import (
    evaluate_perplexity,
    evaluate_qa_rows,
    generate_completions,
    load_wikidata_text,
    recovery_success,
    rouge_l_recall,
    score_completions,
)


SCRIPT_PATH = Path(__file__).resolve()
ALLOWED_METHODS = {
    EXP.METHOD_BASE,
    EXP.METHOD_ZERO,
    EXP.METHOD_SETTING5,
    EXP.METHOD_REPAIRED,
}


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _evaluate_training_examples(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: Sequence[Any],
    *,
    batch_size: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Evaluate MCF retain examples on the exact prompts used by training."""

    prompts = [str(example.prompt) for example in examples]
    answers = [str(example.answer) for example in examples]
    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=30,
    )
    scores = score_completions(
        model,
        tokenizer,
        list(zip(prompts, answers)),
        batch_size=batch_size,
    )
    details: List[Dict[str, Any]] = []
    for index, (example, output, answer, score) in enumerate(
        zip(examples, outputs, answers, scores)
    ):
        details.append(
            {
                "index": index,
                "subject": str(getattr(example, "subject", "")),
                "source": str(getattr(example, "source", "")),
                "prediction": output,
                "answer": answer,
                "recovery_success": recovery_success(output, answer),
                "rouge_l_recall": rouge_l_recall(output, answer),
                "answer_sum_logprob": float(score.sum_logprob),
                "answer_mean_logprob": float(score.mean_logprob),
                "answer_geometric_probability": math.exp(float(score.mean_logprob)),
                "answer_first_token_probability": float(score.first_token_probability),
            }
        )
    summary = {
        "count": len(details),
        "recovery_accuracy": _mean(
            [100.0 * float(row["recovery_success"]) for row in details]
        ),
        "rouge_l_recall": _mean(
            [100.0 * float(row["rouge_l_recall"]) for row in details]
        ),
        "answer_geometric_probability": _mean(
            [float(row["answer_geometric_probability"]) for row in details]
        ),
        "answer_first_token_probability": _mean(
            [float(row["answer_first_token_probability"]) for row in details]
        ),
        "full_answer_mean_log_likelihood": _mean(
            [float(row["answer_sum_logprob"]) for row in details]
        ),
    }
    return summary, details


def _native_rows(
    split: Mapping[str, Any],
    filename: str,
    *,
    level: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for target_split in split["per_target"]:
        for source in target_split["evaluation_only"][filename]:
            row = dict(source)
            row["subject"] = str(row.get("subject") or target_split["subject"])
            row["rwku_target_seed"] = int(target_split["target_seed"])
            row["rwku_target_subject"] = str(target_split["subject"])
            if level is not None:
                row["level"] = str(level)
            rows.append(row)
    return rows


def _evaluate_model(
    *,
    method: str,
    model: torch.nn.Module,
    tokenizer: Any,
    split: Mapping[str, Any],
    retain_examples: Sequence[Any],
    args: Any,
    wikidata_text: Optional[str],
) -> Dict[str, Any]:
    started = time.perf_counter()
    efficacy, efficacy_detail = evaluate_qa_rows(
        model,
        tokenizer,
        split["efficacy_forget"],
        batch_size=args.eval_batch_size,
    )
    heldout_l1, heldout_l1_detail = evaluate_qa_rows(
        model,
        tokenizer,
        split["heldout_level1"],
        batch_size=args.eval_batch_size,
    )
    heldout_l2, heldout_l2_detail = evaluate_qa_rows(
        model,
        tokenizer,
        split["heldout_level2"],
        batch_size=args.eval_batch_size,
    )
    paraphrase, paraphrase_detail = evaluate_qa_rows(
        model,
        tokenizer,
        split["heldout_paraphrase"],
        batch_size=args.eval_batch_size,
    )

    adversarial_rows = _native_rows(split, "forget_level3.json", level=3)
    adversarial, adversarial_detail = evaluate_qa_rows(
        model,
        tokenizer,
        adversarial_rows,
        batch_size=args.eval_batch_size,
        score_answers=False,
    )
    neighbor_l1_rows = _native_rows(split, "neighbor_level1.json", level=1)
    neighbor_l2_rows = _native_rows(split, "neighbor_level2.json", level=2)
    neighbors, neighbor_detail = evaluate_qa_rows(
        model,
        tokenizer,
        [*neighbor_l1_rows, *neighbor_l2_rows],
        batch_size=args.eval_batch_size,
        score_answers=False,
    )

    retain, _retain_detail = _evaluate_training_examples(
        model,
        tokenizer,
        retain_examples,
        batch_size=args.eval_batch_size,
    )
    perplexity = None
    if not args.skip_ppl:
        if not wikidata_text:
            raise FileNotFoundError(
                f"Readable Wikidata corpus required unless --skip-ppl: {args.wikidata_dir}"
            )
        perplexity = evaluate_perplexity(model, tokenizer, wikidata_text)

    return {
        "method": method,
        "summary": {
            "forget": {
                "efficacy_same_50_recovery": efficacy["recovery_accuracy"],
                "efficacy_same_50_probability": efficacy[
                    "answer_geometric_probability"
                ],
                "heldout_level1_recovery": heldout_l1["recovery_accuracy"],
                "heldout_level2_recovery": heldout_l2["recovery_accuracy"],
                "heldout_level2_probability": heldout_l2[
                    "answer_geometric_probability"
                ],
                "heldout_paraphrase_recovery": paraphrase["recovery_accuracy"],
                "adversarial_level3_recovery": adversarial["recovery_accuracy"],
            },
            "retain": {
                "retain_1000_recovery": retain["recovery_accuracy"],
                "retain_1000_probability": retain["answer_geometric_probability"],
                "neighbor_recovery": neighbors["recovery_accuracy"],
                "perplexity": perplexity,
            },
            "denominators": {
                "forget_efficacy": efficacy["count"],
                "heldout_level1": heldout_l1["count"],
                "heldout_level2": heldout_l2["count"],
                "heldout_paraphrase": paraphrase["count"],
                "adversarial_level3": adversarial["count"],
                "retain": retain["count"],
                "neighbors": neighbors["count"],
            },
        },
        "details": {
            "forget_efficacy_same_50": efficacy_detail,
            "heldout_level1": heldout_l1_detail,
            "heldout_level2": heldout_l2_detail,
            "heldout_paraphrase": paraphrase_detail,
            "adversarial_level3": adversarial_detail,
            "neighbors": neighbor_detail,
            # 1000 retain details are intentionally omitted to keep artifacts small.
        },
        "retain_1000_summary": retain,
        "runtime_seconds": time.perf_counter() - started,
    }


def _run_original_zero_batch(
    *,
    args: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    split: Mapping[str, Any],
    retain_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    parameters_class, apply_unlearning = EXP.import_original_zerounlearn(args.zero_root)
    hparams = parameters_class.from_json(args.zero_hparams)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(
            "Expected original ZeroUnlearn layers [16,17,18], got "
            f"{list(hparams.layers)}"
        )
    retain_requests = EXP.records_to_zero_unlearn_requests(retain_records)
    forget_requests: List[Dict[str, Any]] = []
    per_target_request_counts: Dict[str, int] = {}
    for target_split in split["per_target"]:
        requests = EXP.zerounlearn_forget_requests(
            tokenizer,
            target_split["train"],
            subject=str(target_split["subject"]),
            seed=args.seed,
        )
        forget_requests.extend(requests)
        per_target_request_counts[str(target_split["target_seed"])] = len(requests)
    if len(forget_requests) != TOTAL_FORGET_TRAIN:
        raise RuntimeError(
            f"ZeroUnlearn must receive exactly {TOTAL_FORGET_TRAIN} forget requests"
        )

    started = time.perf_counter()
    model.float()
    with EXP.working_directory(EXP.SEMANTIC_ROOT):
        edited_model, original_weights = apply_unlearning(
            model=model,
            tok=tokenizer,
            retain_requests=retain_requests,
            unlearn_requests=forget_requests,
            hparams=hparams,
            copy=False,
            return_orig_weights=False,
            cache_template=None,
            save_path=None,
            add_retain=False,
            edit_layer_nums=3,
            use_h=False,
        )
    del original_weights
    edited_model.to(dtype=EXP.dtype_from_name(args.dtype))
    EXP.prepare_model_for_evaluation(edited_model)
    return {
        "model": edited_model,
        "provenance": {
            "algorithm_entrypoint": "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model",
            "forget_request_count": len(forget_requests),
            "per_target_forget_request_count": per_target_request_counts,
            "retain_request_count": len(retain_requests),
            "hparams_path": str(args.zero_hparams),
            "hparams_sha256": EXP.file_sha256(args.zero_hparams),
            "apply_seconds": time.perf_counter() - started,
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    EXP.write_json(path, value)


def main() -> None:
    args = EXP.build_parser().parse_args()
    EXP.validate_args(args)
    if args.training_source is not None or args.stage != "all":
        raise ValueError(f"{PROTOCOL_ID} uses only the all-in-one Batch-50 runner")
    if args.save_checkpoints:
        raise ValueError(
            f"{PROTOCOL_ID} intentionally rejects --save-checkpoints to save storage"
        )
    if int(args.retain_num) != RETAIN_EVAL_NUM:
        raise ValueError(
            f"{PROTOCOL_ID} fixes --retain-num={RETAIN_EVAL_NUM}; got {args.retain_num}"
        )
    if int(args.repair_retain_num) != REPAIR_RETAIN_NUM:
        raise ValueError(
            f"{PROTOCOL_ID} fixes --repair-retain-num={REPAIR_RETAIN_NUM}; "
            f"got {args.repair_retain_num}"
        )
    methods = EXP.selected_methods(args.methods)
    unsupported = sorted(set(methods) - ALLOWED_METHODS)
    if unsupported:
        raise ValueError(
            f"{PROTOCOL_ID} supports base,zero,setting5,repaired only; got {unsupported}"
        )

    output_dir = Path(args.output_root) / f"batch_seed{args.seed:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    split = materialize_batch_split(
        data_root=args.data_root,
        output_dir=output_dir / "split",
        batch_seed=args.seed,
        allow_download=not args.no_download,
    )
    config: Dict[str, Any] = {
        "status": "preflight" if args.dry_run else "running",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": "probe_assisted_cross_benchmark_method_extension",
        "batch_seed": args.seed,
        "target_seeds": split["manifest"]["target_seeds"],
        "subjects": [item["subject"] for item in split["manifest"]["targets"]],
        "forget_train_count": len(split["forget_train"]),
        "forget_efficacy_count": len(split["efficacy_forget"]),
        "retain_count": args.retain_num,
        "repair_gate_count": args.repair_retain_num,
        "methods": list(methods),
        "model_path": str(args.model_path),
        "dtype": args.dtype,
        "checkpoint_policy": "no model checkpoints saved",
        "same_50_train_and_efficacy_eval": True,
        "heldout_generalization_content_disjoint": True,
        "same_1000_retain_train_and_eval": True,
        "setting5": {
            "mode": EXP.SETTING5_MODE,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "retain_batch_size": args.retain_batch_size,
            "learning_rate": args.emb_lm_lr,
            "forget_weight": args.forget_weight,
            "retain_weight": args.retain_weight,
            "forget_margin": args.forget_margin,
            "optimizer": args.emb_lm_optimizer,
        },
        "repair": asdict(EXP.repair_config(args)),
        "exact_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
    }
    _write_json(output_dir / "config_used.json", config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        print(
            f"{PROTOCOL_ID} dry-run OK: batch seed {args.seed}, "
            f"targets={split['manifest']['target_seeds']}, forget=50, retain=1000"
        )
        return

    if not args.mcf_path.is_file():
        raise FileNotFoundError(args.mcf_path)
    all_retain_records, all_retain_examples = EXP.load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.retain_num + args.repair_retain_num,
    )
    retain_records = all_retain_records[: args.retain_num]
    retain_examples = all_retain_examples[: args.retain_num]
    protected_examples = all_retain_examples[args.retain_num :]
    if len(retain_examples) != RETAIN_EVAL_NUM:
        raise RuntimeError("headline retain partition must contain exactly 1000 examples")
    if len(protected_examples) != REPAIR_RETAIN_NUM:
        raise RuntimeError("repair gate partition must contain exactly 128 examples")
    retain_hashes = {EXP.mapping_sha256(row) for row in retain_records}
    gate_hashes = {
        EXP.mapping_sha256(row) for row in all_retain_records[args.retain_num :]
    }
    if retain_hashes & gate_hashes:
        raise RuntimeError("1000 retain examples overlap the 128 repair-gate examples")
    config["retain_provenance"] = {
        "mcf_path": str(args.mcf_path),
        "mcf_file_sha256": EXP.file_sha256(args.mcf_path),
        "retain_record_sha256": sorted(retain_hashes),
        "repair_gate_record_sha256": sorted(gate_hashes),
        "disjoint": True,
    }
    _write_json(output_dir / "config_used.json", config)

    wikidata_text = None if args.skip_ppl else load_wikidata_text(args.wikidata_dir)
    dtype = EXP.dtype_from_name(args.dtype)
    results: Dict[str, Any] = {}

    print(
        f"Loading base model for {PROTOCOL_ID} seed {args.seed}; "
        f"targets={split['manifest']['target_seeds']}"
    )
    EXP.set_all_seeds(args.seed)
    base_model, tokenizer = EXP.load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    results[EXP.METHOD_BASE] = _evaluate_model(
        method=EXP.METHOD_BASE,
        model=base_model,
        tokenizer=tokenizer,
        split=split,
        retain_examples=retain_examples,
        args=args,
        wikidata_text=wikidata_text,
    )
    _write_json(output_dir / "base_model.json", results[EXP.METHOD_BASE])
    EXP.release_model(base_model)
    base_model = None

    if EXP.METHOD_ZERO in methods:
        print("Applying original ZeroUnlearn jointly to the same 50 forget examples")
        EXP.set_all_seeds(args.seed)
        zero_model, zero_tokenizer = EXP.load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            for_training=False,
            gradient_checkpointing=False,
        )
        zero = _run_original_zero_batch(
            args=args,
            model=zero_model,
            tokenizer=zero_tokenizer,
            split=split,
            retain_records=retain_records,
        )
        zero_model = zero.pop("model")
        zero_result = _evaluate_model(
            method=EXP.METHOD_ZERO,
            model=zero_model,
            tokenizer=zero_tokenizer,
            split=split,
            retain_examples=retain_examples,
            args=args,
            wikidata_text=wikidata_text,
        )
        zero_result["unlearning"] = zero["provenance"]
        results[EXP.METHOD_ZERO] = zero_result
        _write_json(output_dir / "original_zerounlearn.json", zero_result)
        EXP.release_model(zero_model)
        del zero_model, zero_tokenizer, zero
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if EXP.METHOD_SETTING5 in methods or EXP.METHOD_REPAIRED in methods:
        print("Training Setting 5e jointly on 50 forget + 1000 retain examples")
        EXP.set_all_seeds(args.seed)
        setting5_model, setting5_tokenizer = EXP.load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            for_training=True,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        forget_examples = EXP.setting5_examples(setting5_tokenizer, split["forget_train"])
        if len(forget_examples) != TOTAL_FORGET_TRAIN:
            raise RuntimeError("Setting 5e compiler did not produce exactly 50 forget examples")
        requested_save = getattr(args, "save_model", False)
        args.save_model = False
        training_started = time.perf_counter()
        train_summary = EXP.gagd.train_mode(
            setting5_model,
            setting5_tokenizer,
            forget_examples,
            retain_examples,
            selected_ids=[],
            mode=EXP.SETTING5_MODE,
            args=args,
            mode_dir=output_dir / "setting5_training",
        )
        args.save_model = requested_save
        EXP.prepare_model_for_evaluation(setting5_model)
        training_provenance = {
            "trainable": asdict(train_summary),
            "forget_example_count": len(forget_examples),
            "retain_example_count": len(retain_examples),
            "training_seconds": time.perf_counter() - training_started,
            "checkpoint_saved": False,
        }

        if EXP.METHOD_SETTING5 in methods:
            setting5_result = _evaluate_model(
                method=EXP.METHOD_SETTING5,
                model=setting5_model,
                tokenizer=setting5_tokenizer,
                split=split,
                retain_examples=retain_examples,
                args=args,
                wikidata_text=wikidata_text,
            )
            setting5_result["unlearning"] = training_provenance
            results[EXP.METHOD_SETTING5] = setting5_result
            _write_json(output_dir / "setting5_without_repair.json", setting5_result)

        if EXP.METHOD_REPAIRED in methods:
            print("Applying protected LM-head repair using the same 50 forget examples")
            repair_report = EXP.run_protected_lm_head_repair(
                setting5_model,
                setting5_tokenizer,
                calibration_rows=split["forget_train"],
                protected_examples=protected_examples,
                config=EXP.repair_config(args),
                output_dir=output_dir / "setting5_repaired",
            )
            repaired_result = _evaluate_model(
                method=EXP.METHOD_REPAIRED,
                model=setting5_model,
                tokenizer=setting5_tokenizer,
                split=split,
                retain_examples=retain_examples,
                args=args,
                wikidata_text=wikidata_text,
            )
            repaired_result["unlearning"] = training_provenance
            repaired_result["repair"] = repair_report
            results[EXP.METHOD_REPAIRED] = repaired_result
            _write_json(output_dir / "setting5_protected_repair.json", repaired_result)

        EXP.release_model(setting5_model)
        del setting5_model, setting5_tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config["status"] = "complete"
    combined = {
        **config,
        "status": "complete",
        "results": results,
    }
    _write_json(output_dir / "config_used.json", config)
    _write_json(output_dir / "results.json", combined)
    print(f"{PROTOCOL_ID} seed {args.seed} complete: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
