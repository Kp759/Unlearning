#!/usr/bin/env python3
"""Matched seed-1 MCF comparison: Base vs ZeroUnlearn vs fact-association bank.

All three models are evaluated on the exact same MCF source, official seed-1
sample (50 forget / 1000 retain), prompts, target strings, fast tokenizer, raw
CounterFact scorer, and strict zerounlearn_answer_probability_v2 summarizer.

ZeroUnlearn is run in-memory from the vendored original implementation. For a
fair common forgetting target, the ORIGINAL MCF target_true is placed in
ZeroUnlearn's sensitive target_true slot and tokenizer EOS is used only as the
neutral target_new destination. Final evaluation always uses the unmodified
original MCF records. The fact-association checkpoint is never retrained here.
"""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

import mcf_zero_unlearn_official_eval as mcf_eval
from mcf_zero_unlearn_metric_parity import summarize_probability_metrics
from static_overlap_fact_association_embeddings import load_artifact_into_model


MODEL_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000
SAMPLE_MODE = "official"
FINAL_EVAL_DTYPE = "bfloat16"
ZERO_EDIT_LAYER_NUMS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_rewrite(record: Mapping[str, Any]) -> dict[str, Any]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        if len(rr) != 1 or not isinstance(rr[0], Mapping):
            raise ValueError(
                f"case_id={record.get('case_id')} has unsupported requested_rewrite"
            )
        rr = rr[0]
    if not isinstance(rr, Mapping):
        raise ValueError(
            f"case_id={record.get('case_id')} has no requested_rewrite mapping"
        )
    return deepcopy(dict(rr))


def records_to_zero_requests(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"case_id": int(record["case_id"]), **normalize_rewrite(record)}
        for record in records
    ]


def target_true_sensitive_zero_requests(
    records: Sequence[Mapping[str, Any]],
    *,
    neutral_target: str,
) -> list[dict[str, Any]]:
    """Create ZeroUnlearn requests for the same sensitive target as our method."""
    if not neutral_target:
        raise ValueError("neutral_target must be non-empty")
    requests = records_to_zero_requests(records)
    for request in requests:
        sensitive = request.get("target_true")
        if (
            not isinstance(sensitive, Mapping)
            or not isinstance(sensitive.get("str"), str)
            or not sensitive["str"].strip()
        ):
            raise ValueError(
                f"case_id={request.get('case_id')} has no usable target_true"
            )
        request["target_true"] = deepcopy(dict(sensitive))
        request["target_new"] = {"str": neutral_target}
    return requests


def validate_target_true_sensitive_adapter(
    records: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    *,
    neutral_target: str,
) -> None:
    if len(records) != len(requests):
        raise RuntimeError("ZeroUnlearn adapter changed forget request count")
    errors = []
    for record, request in zip(records, requests):
        source = normalize_rewrite(record)
        case_id = int(record["case_id"])
        if int(request.get("case_id", -1)) != case_id:
            errors.append(f"case_id {case_id}: ID changed")
        if request.get("target_true") != source.get("target_true"):
            errors.append(f"case_id {case_id}: sensitive target_true changed")
        target_new = request.get("target_new")
        if (
            not isinstance(target_new, Mapping)
            or target_new.get("str") != neutral_target
        ):
            errors.append(f"case_id {case_id}: neutral target_new is wrong")
        for key in ("prompt", "subject"):
            if request.get(key) != source.get(key):
                errors.append(f"case_id {case_id}: {key} changed")
    if errors:
        raise RuntimeError(
            "Invalid target_true-sensitive ZeroUnlearn adapter:\n- "
            + "\n- ".join(errors)
        )


def resolve_eos(tokenizer) -> tuple[str, int]:
    token = tokenizer.eos_token
    token_id = tokenizer.eos_token_id
    if not isinstance(token, str) or not token or token_id is None:
        raise RuntimeError("Tokenizer needs a usable EOS token")
    ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if ids != [int(token_id)]:
        raise RuntimeError(
            "EOS must encode to exactly eos_token_id without added specials: "
            f"token={token!r} id={token_id!r} encoded={ids!r}"
        )
    return token, int(token_id)


def case_ids(records: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(record["case_id"]) for record in records]


def artifact_case_ids(artifact: Mapping[str, Any]) -> list[int]:
    facts = artifact.get("facts")
    if not isinstance(facts, Sequence) or not facts:
        raise ValueError("Fact-association artifact has no facts list")
    ids = []
    for fact in facts:
        if not isinstance(fact, Mapping) or fact.get("case_id") is None:
            raise ValueError("Fact-association artifact fact lacks case_id")
        ids.append(int(fact["case_id"]))
    return ids


def strictify(result: Mapping[str, Any]) -> dict[str, Any]:
    out = deepcopy(dict(result))
    out["legacy_counterfact"] = {
        "forget": deepcopy(out["forget"]),
        "retain": deepcopy(out["retain"]),
    }
    for split in ("forget", "retain"):
        out[split] = summarize_probability_metrics(
            out[split],
            out[f"{split}_raw"],
        )
    out["metric_version"] = "zerounlearn_answer_probability_v2"
    return out


def compact_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    forget = result["forget"]
    retain = result["retain"]
    legacy_forget = result["legacy_counterfact"]["forget"]
    return {
        "Eff": float(forget["Eff"]),
        "Gen": float(forget["Gen"]),
        "Spe": float(forget["Spe"]),
        "ReleasedAccuracy_Eff": float(forget["ReleasedAccuracy_Eff"]),
        "ReleasedAccuracy_Gen": float(forget["ReleasedAccuracy_Gen"]),
        "TokenGeometricMean_Eff": float(forget["TokenGeometricMean_Eff"]),
        "TokenGeometricMean_Gen": float(forget["TokenGeometricMean_Gen"]),
        "SensitivePref_Eff": forget.get("SensitivePref_Eff"),
        "SensitivePref_Gen": forget.get("SensitivePref_Gen"),
        "CF_EditSuccess_Eff": forget.get("CF_EditSuccess_Eff"),
        "CF_EditSuccess_Gen": forget.get("CF_EditSuccess_Gen"),
        "Legacy_Spe_ProbabilityDiff": forget.get("Legacy_Spe_ProbabilityDiff"),
        "Legacy_Spe_success": legacy_forget.get("Spe_success"),
        "Retain_Eff": float(retain["Eff"]),
        "Retain_Gen": float(retain["Gen"]),
        "Retain_Spe": float(retain["Spe"]),
        "Legacy_PPL": result.get("legacy_forget_PPL"),
        "RuntimeAligned_PPL": result.get("forget_PPL"),
    }


def load_base(model_path: Path, *, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=True,
        attn_implementation="eager",
    ).to("cuda").eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    return model


def evaluate_one(
    *,
    method: str,
    model,
    tokenizer,
    model_dir: str | Path,
    mcf_path: Path,
    wikidata_dir: Path,
    skip_ppl: bool,
) -> dict[str, Any]:
    raw = mcf_eval.evaluate_loaded_model_official(
        method=method,
        model=model,
        tok=tokenizer,
        model_dir=model_dir,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=None,
        unlearn_num=FORGET_NUM,
        retain_num=RETAIN_NUM,
        seed=SEED,
        sample_mode=SAMPLE_MODE,
        skip_ppl=skip_ppl,
    )
    return strictify(raw)


def free_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def import_zero_unlearn(zero_root: Path):
    root = str(zero_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from ZeroUnlearn import ZeroUnlearnHyperParams, apply_unl_to_model

    return ZeroUnlearnHyperParams, apply_unl_to_model


def validate_common_protocol(
    *,
    model_path: Path,
    ours_manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if model_path.name != MODEL_REVISION:
        raise ValueError(
            f"Expected model revision {MODEL_REVISION}, got {model_path.name}"
        )
    manifest_model = Path(str(ours_manifest["model_path"])).resolve()
    if manifest_model != model_path.resolve():
        raise RuntimeError(
            "Our frozen run used a different base model path: "
            f"{manifest_model} != {model_path.resolve()}"
        )

    expected_forget = case_ids(forget_records)
    expected_retain = case_ids(retain_records)
    ours_forget = artifact_case_ids(artifact)
    if ours_forget != expected_forget:
        raise RuntimeError(
            "Our frozen artifact is not the exact official seed-1 forget sample. "
            f"expected={expected_forget} artifact={ours_forget}"
        )
    if len(set(expected_forget)) != FORGET_NUM:
        raise RuntimeError("Official forget sample contains duplicate case IDs")
    if len(set(expected_retain)) != RETAIN_NUM:
        raise RuntimeError("Official retain sample contains duplicate case IDs")

    manifest_seed = ours_manifest.get("seed")
    if manifest_seed is not None and int(manifest_seed) != SEED:
        raise RuntimeError(
            f"Our manifest seed is {manifest_seed}, expected {SEED}"
        )

    return {
        "seed": SEED,
        "sample_mode": SAMPLE_MODE,
        "forget_num": FORGET_NUM,
        "retain_num": RETAIN_NUM,
        "forget_case_ids": expected_forget,
        "retain_case_ids": expected_retain,
        "ours_artifact_case_ids_match": True,
        "same_base_model_verified": True,
    }


def write_table(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "Method",
        "Eff",
        "Gen",
        "Spe",
        "ReleasedAccuracy_Eff",
        "ReleasedAccuracy_Gen",
        "SensitivePref_Eff",
        "SensitivePref_Gen",
        "Legacy_PPL",
        "RuntimeAligned_PPL",
    ]
    with (output_dir / "comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})

    lines = [
        "# MCF seed-1 matched comparison",
        "",
        "Primary Eff/Gen use zerounlearn_answer_probability_v2: complete "
        "target_true answer probability, lower is better. Spe is strict "
        "all-answer-token neighborhood accuracy, higher is better.",
        "",
        "| Method | Eff ↓ | Gen ↓ | Spe ↑ | ReleasedAcc Eff ↓ | "
        "ReleasedAcc Gen ↓ | SensitivePref Eff ↓ | SensitivePref Gen ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Method} | {Eff:.10g} | {Gen:.10g} | {Spe:.6g} | "
            "{ReleasedAccuracy_Eff:.6g} | {ReleasedAccuracy_Gen:.6g} | "
            "{SensitivePref_Eff} | {SensitivePref_Gen} |".format(**row)
        )
    lines.extend([
        "",
        "## Fairness contract",
        "",
        "- Same Llama-3.2-3B-Instruct snapshot.",
        "- Same MCF source file and SHA-256.",
        "- Same official seed-1 IDs: 50 forget, 1000 retain.",
        "- Same original evaluation records for all methods.",
        "- Same fast tokenizer and BF16 final evaluation dtype.",
        "- ZeroUnlearn edits in FP32, then is cast to BF16 for final scoring.",
        "- Both methods target ORIGINAL MCF target_true as sensitive.",
        "- ZeroUnlearn EOS is only an internal neutral destination.",
        "- Official evaluation records are never modified.",
        "- Our fact-association checkpoint is frozen and not retrained.",
        "",
        "Pairwise target_true/target_new diagnostics are retained separately and "
        "are not the primary Eff/Gen.",
    ])
    (output_dir / "comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ours-run-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument(
        "--zero-unlearn-root",
        default=str(Path(__file__).resolve().parents[2] / "ZeroUnlearn"),
    )
    parser.add_argument("--zero-hparams", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-ppl", action="store_true")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this matched comparison")

    model_path = Path(args.model_path).resolve()
    ours_run = Path(args.ours_run_dir).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    zero_root = Path(args.zero_unlearn_root).resolve()
    hparams_path = (
        Path(args.zero_hparams).resolve()
        if args.zero_hparams
        else zero_root / "hparams" / "ZeroUnlearn" / "Llama-3.2-3B-Instruct.json"
    )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite comparison output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    required = [
        model_path,
        ours_run / "association_manifest.json",
        ours_run / "fact_association_embeddings.pt",
        mcf_path,
        zero_root,
        hparams_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required comparison input:\n- "
            + "\n- ".join(str(path) for path in missing)
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            "Strict full-answer probability requires a fast tokenizer"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_records, retain_records = mcf_eval.load_official_eval_records(
        mcf_path,
        FORGET_NUM,
        RETAIN_NUM,
        SEED,
        SAMPLE_MODE,
    )
    ours_manifest = json.loads(
        (ours_run / "association_manifest.json").read_text()
    )
    artifact = torch.load(
        ours_run / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )
    protocol = validate_common_protocol(
        model_path=model_path,
        ours_manifest=ours_manifest,
        artifact=artifact,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    protocol.update({
        "model_revision": MODEL_REVISION,
        "mcf_path": str(mcf_path),
        "mcf_sha256": sha256_file(mcf_path),
        "ours_artifact": str(
            ours_run / "fact_association_embeddings.pt"
        ),
        "ours_artifact_sha256": sha256_file(
            ours_run / "fact_association_embeddings.pt"
        ),
        "zero_unlearn_root": str(zero_root),
        "zero_hparams": str(hparams_path),
        "zero_hparams_sha256": sha256_file(hparams_path),
        "primary_metric_version": "zerounlearn_answer_probability_v2",
        "final_evaluation_dtype": FINAL_EVAL_DTYPE,
        "zero_unlearn_edit_dtype": "float32",
        "zero_unlearn_edit_layer_nums": ZERO_EDIT_LAYER_NUMS,
        "zero_unlearn_common_sensitive_target": (
            "original MCF requested_rewrite.target_true"
        ),
        "zero_unlearn_neutral_destination": "tokenizer EOS",
        "official_evaluation_records_modified": False,
    })
    (output_dir / "shared_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )

    print("=== [1/3] Evaluating matched base model ===", flush=True)
    base = load_base(model_path, dtype=torch.bfloat16)
    base_result = evaluate_one(
        method="Base",
        model=base,
        tokenizer=tokenizer,
        model_dir=model_path,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        skip_ppl=args.skip_ppl,
    )
    (output_dir / "base.json").write_text(
        json.dumps(base_result, indent=2, allow_nan=False) + "\n"
    )
    free_model(base)

    print(
        "=== [2/3] Running original ZeroUnlearn on same seed-1 target_true facts ===",
        flush=True,
    )
    ZeroUnlearnHyperParams, apply_unl_to_model = import_zero_unlearn(zero_root)
    hparams = ZeroUnlearnHyperParams.from_json(hparams_path)

    zero_model = load_base(model_path, dtype=torch.float32)
    neutral_token, neutral_id = resolve_eos(tokenizer)
    zero_forget = target_true_sensitive_zero_requests(
        forget_records,
        neutral_target=neutral_token,
    )
    validate_target_true_sensitive_adapter(
        forget_records,
        zero_forget,
        neutral_target=neutral_token,
    )
    zero_retain = records_to_zero_requests(retain_records)

    started = time.monotonic()
    zero_model, _ = apply_unl_to_model(
        model=zero_model,
        tok=tokenizer,
        retain_requests=zero_retain,
        unlearn_requests=zero_forget,
        hparams=hparams,
        copy=False,
        return_orig_weights=False,
        cache_template=None,
        save_path=None,
        add_retain=False,
        edit_layer_nums=ZERO_EDIT_LAYER_NUMS,
        use_h=False,
    )
    zero_edit_seconds = time.monotonic() - started
    zero_model = zero_model.to(dtype=torch.bfloat16).eval()
    zero_result = evaluate_one(
        method="ZeroUnlearn_target_true_sensitive",
        model=zero_model,
        tokenizer=tokenizer,
        model_dir="in-memory:ZeroUnlearn_target_true_sensitive",
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        skip_ppl=args.skip_ppl,
    )
    zero_result["zero_unlearn_fair_adapter"] = {
        "algorithm": "ZeroUnlearn.apply_unl_to_model",
        "edit_seconds": zero_edit_seconds,
        "edit_dtype": "float32",
        "final_evaluation_dtype": FINAL_EVAL_DTYPE,
        "edit_layer_nums": ZERO_EDIT_LAYER_NUMS,
        "add_retain": False,
        "use_h": False,
        "sensitive_source": "original requested_rewrite.target_true",
        "neutral_target_source": "tokenizer.eos_token",
        "neutral_token": neutral_token,
        "neutral_token_id": neutral_id,
        "official_evaluation_records_modified": False,
        "official_paraphrases_used_for_editing": False,
        "official_neighborhood_prompts_used_for_editing": False,
    }
    (output_dir / "zerounlearn.json").write_text(
        json.dumps(zero_result, indent=2, allow_nan=False) + "\n"
    )
    free_model(zero_model)

    print("=== [3/3] Evaluating frozen fact-association artifact ===", flush=True)
    ours_base = load_base(model_path, dtype=torch.bfloat16)
    ours_model, bank = load_artifact_into_model(ours_base, artifact)
    ours_model.eval()
    ours_result = evaluate_one(
        method="FactAssociationBank",
        model=ours_model,
        tokenizer=tokenizer,
        model_dir=ours_run,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        skip_ppl=args.skip_ppl,
    )
    ours_result["fact_association_runtime"] = {
        "artifact": str(ours_run / "fact_association_embeddings.pt"),
        "layer": int(artifact["layer"]),
        "facts": len(artifact["facts"]),
        "runtime_counters": bank.counters(),
        "base_weights_edited": False,
        "tokenizer_extended": False,
        "fact_id_injection_used": False,
    }
    (output_dir / "ours.json").write_text(
        json.dumps(ours_result, indent=2, allow_nan=False) + "\n"
    )
    free_model(ours_model)

    results = {
        "Base": base_result,
        "ZeroUnlearn": zero_result,
        "Ours": ours_result,
    }
    rows = [
        {"Method": method, **compact_metrics(result)}
        for method, result in results.items()
    ]

    comparison = {
        "kind": "matched_mcf_seed1_zerounlearn_vs_fact_association_v1",
        "shared_protocol": protocol,
        "metric_contract": {
            "primary": "zerounlearn_answer_probability_v2",
            "Eff": (
                "100 * case-macro complete original target_true probability "
                "on canonical rewrite prompts"
            ),
            "Gen": (
                "100 * case-macro mean complete original target_true "
                "probability on paraphrase prompts"
            ),
            "Spe": (
                "100 * case-macro neighborhood all-target_true-token "
                "teacher-forced top1 accuracy"
            ),
            "ReleasedAccuracy_Eff_Gen": (
                "all original target_true answer tokens teacher-forced top1 "
                "correct; lower is better on forget split"
            ),
            "pairwise_diagnostics": (
                "SensitivePref and CF_EditSuccess are retained separately and "
                "are not primary Eff/Gen"
            ),
        },
        "rows": rows,
        "files": {
            "base": "base.json",
            "zerounlearn": "zerounlearn.json",
            "ours": "ours.json",
            "protocol": "shared_protocol.json",
            "csv": "comparison.csv",
            "markdown": "comparison.md",
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    write_table(output_dir, rows)

    print(json.dumps({
        "status": "matched_seed1_comparison_complete",
        "metric_version": "zerounlearn_answer_probability_v2",
        "rows": rows,
        "output_dir": str(output_dir),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
