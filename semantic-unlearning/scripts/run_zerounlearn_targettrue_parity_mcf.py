#!/usr/bin/env python3
"""Run original ZeroUnlearn on the same MCF sensitive target used by RSNR.

This is an algorithmic parity experiment, not the repository's original
ZeroUnlearn MCF protocol.  The original ZeroUnlearn source and reviewed
Llama-3.2-3B hyperparameters are used unchanged.  The only semantic adaptation
is the forget request:

    ZeroUnlearn target_true (M_f) <- MCF requested_rewrite.target_true
    ZeroUnlearn target_new  (M_n) <- tokenizer EOS

Thus ZeroUnlearn suppresses the same original factual answer that RSNR-V1A
suppresses.  Official MCF evaluation records themselves are never modified.
The run is locked to development seed 1, forget50/retain1000, official sampling,
and BF16 evaluation.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import run_zerounlearn_official_mcf as official
from mcf_zero_unlearn_official_eval import evaluate_loaded_model_official


PROTOCOL = "zerounlearn_mcf_targettrue_parity_seed1"
SEED = 1
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
        "--output-dir",
        default=str(official.SEMANTIC_ROOT / "outputs" / "zerounlearn_targettrue_parity_seed1"),
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--forget-num", type=int, default=FORGET_NUM)
    p.add_argument("--retain-num", type=int, default=RETAIN_NUM)
    p.add_argument("--sample-mode", choices=[SAMPLE_MODE], default=SAMPLE_MODE)
    p.add_argument("--dtype", choices=[DTYPE], default=DTYPE)
    args = p.parse_args()
    if args.seed != SEED:
        p.error("target-true parity development run is locked to seed 1")
    if args.forget_num != FORGET_NUM or args.retain_num != RETAIN_NUM:
        p.error("target-true parity run is locked to forget50/retain1000")
    return args


def targettrue_forget_requests(
    records: Sequence[Mapping[str, Any]],
    *,
    neutral_target: str,
) -> list[Dict[str, Any]]:
    """Put MCF target_true in ZeroUnlearn M_f and EOS in M_n."""
    if not neutral_target:
        raise ValueError("neutral_target must be non-empty")
    requests = official.records_to_zero_unlearn_requests(records)
    for record, request in zip(records, requests):
        source = official.normalize_rewrite(record)
        sensitive = source.get("target_true")
        if (
            not isinstance(sensitive, Mapping)
            or not isinstance(sensitive.get("str"), str)
            or not sensitive.get("str")
        ):
            raise ValueError(
                f"case_id={record.get('case_id')} has no usable MCF target_true"
            )
        request["target_true"] = deepcopy(dict(sensitive))
        request["target_new"] = {"str": neutral_target}
    return requests


def validate_targettrue_requests(
    records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    neutral_target: str,
) -> None:
    if len(records) != len(requests):
        raise RuntimeError("forget request count changed")
    errors: list[str] = []
    for record, request in zip(records, requests):
        source = official.normalize_rewrite(record)
        cid = int(record["case_id"])
        if int(request.get("case_id", -1)) != cid:
            errors.append(f"case_id={cid}: case id changed")
        if request.get("target_true") != source.get("target_true"):
            errors.append(f"case_id={cid}: sensitive target is not MCF target_true")
        tn = request.get("target_new")
        if not isinstance(tn, Mapping) or tn.get("str") != neutral_target:
            errors.append(f"case_id={cid}: neutral target is not EOS")
        for key in ("prompt", "subject"):
            if request.get(key) != source.get(key):
                errors.append(f"case_id={cid}: {key} changed")
    if errors:
        raise RuntimeError("Invalid target-true parity requests:\n- " + "\n- ".join(errors))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    zero_root = Path(args.zero_unlearn_root).expanduser().resolve()
    hparams_path = Path(args.hparams_path).expanduser().resolve()
    mcf_path = Path(args.mcf_path).expanduser().resolve()
    wikidata_dir = Path(args.wikidata_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    official.require_runtime_files(
        model_path, mcf_path, wikidata_dir, hparams_path, zero_root
    )
    # Keep the exact reviewed vendored ZeroUnlearn source/hparams/data hashes.
    source_hashes_before = official.hash_protocol_inputs(
        mcf_path, hparams_path, zero_root
    )
    official.validate_expected_protocol_hashes(
        source_hashes_before, mcf_path, hparams_path, zero_root
    )

    _, forget_records, retain_records = official.load_official_split(
        mcf_path,
        seed=SEED,
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
    official.set_all_seeds(SEED)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    neutral_target, neutral_target_id = official.resolve_eos_neutral_target(tok)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    print("Evaluating matched unedited Base on seed 1")
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
        seed=SEED,
        sample_mode=SAMPLE_MODE,
        skip_ppl=False,
    )
    base_result.update({
        "protocol": PROTOCOL,
        "comparison_role": "matched_base",
        "model_path": str(model_path),
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
    })
    base_path = output_dir / "base_seed1_official_eval.json"
    official.write_json(base_path, base_result)

    params_class, apply_unl_to_model = official.import_original_zerounlearn(zero_root)
    hparams = params_class.from_json(hparams_path)
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(
            f"reviewed ZeroUnlearn layers changed: {list(hparams.layers)}"
        )

    retain_requests = official.records_to_zero_unlearn_requests(retain_records)
    forget_requests = targettrue_forget_requests(
        forget_records, neutral_target=neutral_target
    )
    validate_targettrue_requests(forget_records, forget_requests, neutral_target)

    print(
        "Parity mapping: MCF target_true -> ZeroUnlearn M_f; "
        f"EOS {neutral_target!r} (id={neutral_target_id}) -> M_n"
    )
    print("Applying original closed-form ZeroUnlearn; source/hparams unchanged")

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

    print("Evaluating target-true parity ZeroUnlearn on the unchanged MCF records")
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
        seed=SEED,
        sample_mode=SAMPLE_MODE,
        skip_ppl=False,
    )
    result.update({
        "protocol": PROTOCOL,
        "method": "ZeroUnlearn-targettrue-parity",
        "seed": SEED,
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
        "peak_cuda_memory_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        ),
        "source_hashes_before": source_hashes_before,
    })
    result_path = output_dir / "zerounlearn_targettrue_seed1_official_eval.json"
    official.write_json(result_path, result)

    source_hashes_after = official.hash_protocol_inputs(
        mcf_path, hparams_path, zero_root
    )
    if source_hashes_after != source_hashes_before:
        raise RuntimeError("reviewed ZeroUnlearn source/hparams/data changed during run")

    provenance = {
        "protocol": PROTOCOL,
        "status": "completed",
        "seed": SEED,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "sample_mode": SAMPLE_MODE,
        "model_path": str(model_path),
        "hparams_path": str(hparams_path),
        "zero_unlearn_root": str(zero_root),
        "algorithm_entrypoint": "ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model",
        "original_algorithm_unchanged": True,
        "original_hparams_unchanged": True,
        "mapping": {
            "MCF_target_true": "ZeroUnlearn target_true / M_f",
            "tokenizer_EOS": "ZeroUnlearn target_new / M_n",
        },
        "forget_case_ids": forget_ids,
        "retain_case_ids": retain_ids,
        "result_path": str(result_path),
        "base_path": str(base_path),
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
    }
    official.write_json(output_dir / "provenance.json", provenance)

    del edited_model, model
    gc.collect()
    torch.cuda.empty_cache()

    f = result["forget"]
    print(json.dumps({
        "result": str(result_path),
        "legacy_pairwise_Eff": f.get("Eff"),
        "legacy_pairwise_Gen": f.get("Gen"),
        "Spe": f.get("Spe"),
        "Spe_success": f.get("Spe_success"),
        "PPL": result.get("forget_PPL"),
        "next": (
            "run fix_saved_mcf_eff_gen_zero_unlearn.py on this result to obtain "
            "the paper-definition residual Eff/Gen"
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
