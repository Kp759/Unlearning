#!/usr/bin/env python3
"""Run target-true-parity ZeroUnlearn on matched MCF seeds 1-10.

This is the multi-seed confirmatory counterpart of
``run_zerounlearn_targettrue_parity_mcf.py``.  It keeps the reviewed original
ZeroUnlearn algorithm and Llama-3.2-3B hyperparameters unchanged while mapping
MCF ``requested_rewrite.target_true`` into ZeroUnlearn's sensitive M_f slot and
tokenizer EOS into its neutral M_n slot.

Each seed uses the same official MCF split rule, model checkpoint, evaluator,
forget/retain sizes, and target semantics intended for direct comparison with
RSNR.  Two metric families are stored explicitly and never conflated:

1. ``eq16_style_residual_likelihood_proxy``
   100 * mean(exp(-token-average-NLL(target_true))) for rewrite/paraphrase.
2. ``released_table_style_accuracy``
   teacher-forced target_true top-1 correctness for rewrite/paraphrase/
   neighborhood, matching the released CounterFact accuracy logic.

Seed 1 is retained for continuity with the development run; seeds 2-10 are the
new untouched confirmatory seeds under the frozen comparison protocol.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import run_zerounlearn_official_mcf as official
import run_zerounlearn_targettrue_parity_mcf as parity
from mcf_zero_unlearn_metric_parity import compute_zero_unlearn_paper_eff_gen
from mcf_zero_unlearn_official_eval import evaluate_loaded_model_official


PROTOCOL = "zerounlearn_mcf_targettrue_parity_multiseed_v1"
DEFAULT_SEEDS = tuple(range(1, 11))
FORGET_NUM = 50
RETAIN_NUM = 1000
SAMPLE_MODE = "official"
DTYPE = "bfloat16"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--zero-unlearn-root", default=str(official.DEFAULT_ZERO_ROOT))
    p.add_argument("--hparams-path", default=str(official.DEFAULT_HPARAMS))
    p.add_argument("--mcf-path", default=str(official.DEFAULT_MCF))
    p.add_argument("--wikidata-dir", default=str(official.DEFAULT_WIKIDATA))
    p.add_argument(
        "--output-root",
        default=str(official.SEMANTIC_ROOT / "outputs" / "zerounlearn_targettrue_parity_seeds1_10"),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--forget-num", type=int, default=FORGET_NUM)
    p.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    p.add_argument("--sample-mode", choices=[SAMPLE_MODE], default=SAMPLE_MODE)
    p.add_argument("--dtype", choices=[DTYPE], default=DTYPE)
    p.add_argument("--skip-completed", action="store_true")
    args = p.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must contain unique seed values")
    if args.forget_num != FORGET_NUM or args.retain_num != RETAIN_NUM:
        p.error("target-true parity sweep is frozen to forget50/retain1000")
    return args


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _macro_correctness(raw_rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Case-macro teacher-forced accuracy in percent from official raw rows."""
    case_values: list[float] = []
    for row in raw_rows or []:
        post = row.get("post", {}) if isinstance(row, Mapping) else {}
        values = post.get(key, []) if isinstance(post, Mapping) else []
        numeric: list[float] = []
        for value in values or []:
            if isinstance(value, bool):
                numeric.append(float(value))
            elif isinstance(value, (int, float)):
                numeric.append(float(value != 0))
        if numeric:
            case_values.append(sum(numeric) / len(numeric))
    if not case_values:
        return None
    return 100.0 * sum(case_values) / len(case_values)


def metric_families(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = result.get("forget_raw")
    if not isinstance(raw, Sequence):
        raise RuntimeError("official result is missing forget_raw")
    eq16 = compute_zero_unlearn_paper_eff_gen(raw)
    if eq16["Eff"] is None or eq16["Gen"] is None:
        raise RuntimeError("could not compute Eq16-style residual likelihood proxy")
    table = {
        "Eff": _macro_correctness(raw, "rewrite_prompts_correct"),
        "Gen": _macro_correctness(raw, "paraphrase_prompts_correct"),
        "Spe": _macro_correctness(raw, "neighborhood_prompts_correct"),
    }
    return {
        "eq16_style_residual_likelihood_proxy": {
            "Eff": float(eq16["Eff"]),
            "Gen": float(eq16["Gen"]),
            "definition": (
                "100 * case-macro mean exp(-token-average-NLL(target_true)); "
                "Eq.16-inspired residual sensitive-answer likelihood proxy"
            ),
            "lower_is_better": True,
        },
        "released_table_style_accuracy": {
            **table,
            "definition": (
                "teacher-forced target_true top-1 correctness; case-macro mean in percent"
            ),
            "Eff_lower_is_better": True,
            "Gen_lower_is_better": True,
            "Spe_higher_is_better": True,
        },
        "PPL": result.get("forget_PPL"),
    }


def seed_paths(root: Path, seed: int) -> dict[str, Path]:
    seed_dir = root / f"seed{seed}"
    return {
        "dir": seed_dir,
        "base": seed_dir / f"base_seed{seed}_official_eval.json",
        "zero": seed_dir / f"zerounlearn_targettrue_seed{seed}_official_eval.json",
        "metrics": seed_dir / f"zerounlearn_targettrue_seed{seed}_metric_families.json",
        "provenance": seed_dir / "provenance.json",
    }


def completed(paths: Mapping[str, Path], seed: int) -> bool:
    needed = [paths["zero"], paths["metrics"], paths["provenance"]]
    if not all(path.is_file() for path in needed):
        return False
    try:
        provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        provenance.get("status") == "completed"
        and int(provenance.get("seed", -1)) == int(seed)
        and provenance.get("sensitive_target_source") == "MCF requested_rewrite.target_true"
        and isinstance(metrics.get("eq16_style_residual_likelihood_proxy"), Mapping)
    )


def run_one_seed(
    *,
    seed: int,
    model_path: Path,
    zero_root: Path,
    hparams_path: Path,
    mcf_path: Path,
    wikidata_dir: Path,
    output_root: Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    paths = seed_paths(output_root, seed)
    paths["dir"].mkdir(parents=True, exist_ok=True)

    _, forget_records, retain_records = official.load_official_split(
        mcf_path,
        seed=seed,
        forget_num=FORGET_NUM,
        retain_num=RETAIN_NUM,
        sample_mode=SAMPLE_MODE,
    )
    forget_ids = official.case_ids(forget_records)
    retain_ids = official.case_ids(retain_records)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    official.set_all_seeds(seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    neutral_target, neutral_target_id = official.resolve_eos_neutral_target(tok)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print(f"[seed {seed}] evaluating matched Base", flush=True)
    base_result = evaluate_loaded_model_official(
        method="Base",
        model=model,
        tok=tok,
        model_dir=model_path,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=None,
        unlearn_num=FORGET_NUM,
        retain_num=RETAIN_NUM,
        seed=seed,
        sample_mode=SAMPLE_MODE,
        skip_ppl=False,
    )
    base_result.update({
        "protocol": PROTOCOL,
        "comparison_role": "matched_base",
        "seed": seed,
        "model_path": str(model_path),
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
    })
    write_json(paths["base"], base_result)

    params_class, apply_unl_to_model = official.import_original_zerounlearn(zero_root)
    hparams = params_class.from_json(hparams_path)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(f"reviewed ZeroUnlearn layers changed: {list(hparams.layers)}")

    retain_requests = official.records_to_zero_unlearn_requests(retain_records)
    forget_requests = parity.targettrue_forget_requests(
        forget_records, neutral_target=neutral_target
    )
    parity.validate_targettrue_requests(forget_records, forget_requests, neutral_target)

    print(f"[seed {seed}] applying original ZeroUnlearn with target_true parity mapping", flush=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.float()
    with official.working_directory(official.SEMANTIC_ROOT):
        edited_model, _ = apply_unl_to_model(
            model=model,
            tok=tok,
            retain_requests=retain_requests,
            unlearn_requests=forget_requests,
            hparams=hparams,
            copy=False,
            return_orig_weights=False,
            cache_template=None,
            save_path=None,
            add_retain=official.ADD_RETAIN,
            edit_layer_nums=official.EDIT_LAYER_NUMS,
            use_h=official.USE_H,
        )
    edited_model.to(dtype=torch.bfloat16)
    edited_model.eval()
    torch.cuda.synchronize(device)
    apply_seconds = time.perf_counter() - started

    print(f"[seed {seed}] evaluating target-true parity ZeroUnlearn", flush=True)
    result = evaluate_loaded_model_official(
        method="ZeroUnlearn-targettrue-parity",
        model=edited_model,
        tok=tok,
        model_dir="in-memory:ZeroUnlearn-targettrue-parity",
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=None,
        unlearn_num=FORGET_NUM,
        retain_num=RETAIN_NUM,
        seed=seed,
        sample_mode=SAMPLE_MODE,
        skip_ppl=False,
    )
    result.update({
        "protocol": PROTOCOL,
        "method": "ZeroUnlearn-targettrue-parity",
        "seed": seed,
        "model_path": str(model_path),
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
        "algorithm_entrypoint": "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model",
        "original_zerounlearn_algorithm_unchanged": True,
        "original_zerounlearn_hparams_unchanged": True,
        "request_mapping_modified_for_rsnr_parity": True,
        "sensitive_target_source": "MCF requested_rewrite.target_true",
        "neutral_target": {
            "source": "tokenizer.eos_token",
            "token": neutral_target,
            "token_id": neutral_target_id,
            "zero_unlearn_sensitive_field": "target_true.str",
            "zero_unlearn_neutral_field": "target_new.str",
        },
        "official_evaluation_records_modified": False,
        "apply_seconds": apply_seconds,
        "peak_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "source_hashes_before": dict(source_hashes),
    })
    write_json(paths["zero"], result)

    metrics = metric_families(result)
    metrics.update({
        "protocol": PROTOCOL,
        "seed": seed,
        "method": "ZeroUnlearn-targettrue-parity",
        "sensitive_target_source": "MCF requested_rewrite.target_true",
    })
    write_json(paths["metrics"], metrics)

    source_hashes_after = official.hash_protocol_inputs(mcf_path, hparams_path, zero_root)
    if dict(source_hashes_after) != dict(source_hashes):
        raise RuntimeError("reviewed ZeroUnlearn source/hparams/data changed during run")

    provenance = {
        "protocol": PROTOCOL,
        "status": "completed",
        "seed": seed,
        "development_seed": seed == 1,
        "confirmatory_seed": seed != 1,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "sample_mode": SAMPLE_MODE,
        "model_path": str(model_path),
        "hparams_path": str(hparams_path),
        "zero_unlearn_root": str(zero_root),
        "algorithm_entrypoint": "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model",
        "original_algorithm_unchanged": True,
        "original_hparams_unchanged": True,
        "sensitive_target_source": "MCF requested_rewrite.target_true",
        "mapping": {
            "MCF_target_true": "ZeroUnlearn target_true / M_f",
            "tokenizer_EOS": "ZeroUnlearn target_new / M_n",
        },
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
        "result_path": str(paths["zero"]),
        "metrics_path": str(paths["metrics"]),
        "base_path": str(paths["base"]),
        "source_hashes_before": dict(source_hashes),
        "source_hashes_after": dict(source_hashes_after),
    }
    write_json(paths["provenance"], provenance)

    summary = {
        "seed": seed,
        "eq16_Eff": metrics["eq16_style_residual_likelihood_proxy"]["Eff"],
        "eq16_Gen": metrics["eq16_style_residual_likelihood_proxy"]["Gen"],
        "table_Eff": metrics["released_table_style_accuracy"]["Eff"],
        "table_Gen": metrics["released_table_style_accuracy"]["Gen"],
        "table_Spe": metrics["released_table_style_accuracy"]["Spe"],
        "PPL": metrics["PPL"],
        "apply_seconds": apply_seconds,
    }

    del edited_model, model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("eq16_Eff", "eq16_Gen", "table_Eff", "table_Gen", "table_Spe", "PPL")
    out: dict[str, Any] = {"n_seeds": len(rows), "seeds": [int(x["seed"]) for x in rows]}
    for key in metrics:
        values = [float(x[key]) for x in rows if x.get(key) is not None]
        out[key] = {
            "mean": statistics.fmean(values) if values else None,
            "std_population": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
            "n": len(values),
        }
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["seed", "eq16_Eff", "eq16_Gen", "table_Eff", "table_Gen", "table_Spe", "PPL", "apply_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def load_completed_summary(paths: Mapping[str, Path], seed: int) -> dict[str, Any]:
    metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    result = json.loads(paths["zero"].read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "eq16_Eff": metrics["eq16_style_residual_likelihood_proxy"]["Eff"],
        "eq16_Gen": metrics["eq16_style_residual_likelihood_proxy"]["Gen"],
        "table_Eff": metrics["released_table_style_accuracy"]["Eff"],
        "table_Gen": metrics["released_table_style_accuracy"]["Gen"],
        "table_Spe": metrics["released_table_style_accuracy"]["Spe"],
        "PPL": metrics["PPL"],
        "apply_seconds": result.get("apply_seconds"),
    }


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    zero_root = Path(args.zero_unlearn_root).expanduser().resolve()
    hparams_path = Path(args.hparams_path).expanduser().resolve()
    mcf_path = Path(args.mcf_path).expanduser().resolve()
    wikidata_dir = Path(args.wikidata_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    official.require_runtime_files(model_path, mcf_path, wikidata_dir, hparams_path, zero_root)
    source_hashes = official.hash_protocol_inputs(mcf_path, hparams_path, zero_root)
    official.validate_expected_protocol_hashes(source_hashes, mcf_path, hparams_path, zero_root)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        paths = seed_paths(output_root, int(seed))
        if args.skip_completed and completed(paths, int(seed)):
            print(f"[seed {seed}] skipping completed validated target-true parity run", flush=True)
            rows.append(load_completed_summary(paths, int(seed)))
            continue
        rows.append(run_one_seed(
            seed=int(seed),
            model_path=model_path,
            zero_root=zero_root,
            hparams_path=hparams_path,
            mcf_path=mcf_path,
            wikidata_dir=wikidata_dir,
            output_root=output_root,
            source_hashes=source_hashes,
        ))

    rows.sort(key=lambda x: int(x["seed"]))
    summary = {
        "protocol": PROTOCOL,
        "method": "ZeroUnlearn-targettrue-parity",
        "seeds": [int(x) for x in args.seeds],
        "development_seed": 1 if 1 in args.seeds else None,
        "confirmatory_seeds": [int(x) for x in args.seeds if int(x) != 1],
        "comparison_contract": {
            "model": str(model_path),
            "forget_num": FORGET_NUM,
            "retain_num": RETAIN_NUM,
            "sample_mode": SAMPLE_MODE,
            "dtype": DTYPE,
            "sensitive_target": "MCF requested_rewrite.target_true",
            "zero_unlearn_algorithm_changed": False,
            "zero_unlearn_hparams_changed": False,
        },
        "per_seed": rows,
        "aggregate": aggregate(rows),
    }
    write_json(output_root / "multiseed_summary.json", summary)
    write_csv(output_root / "per_seed.csv", rows)
    print(json.dumps(summary["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
