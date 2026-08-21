#!/usr/bin/env python3
"""Run the canonical ZsRE SURE architecture on target-true-sensitive MCF.

The original MCF fields are never swapped:

* ``target_true`` is the sensitive answer;
* ``target_new`` is the non-sensitive reference;
* Stage 1 uses canonical sensitive-token GA plus same-prompt non-sensitive KL;
* Stage 1 restores every vocabulary row except sensitive ``target_true`` rows;
* Stage 2 changes residual sensitive output rows only;
* the MCF direct constraint is
  ``NLL(target_true) - NLL(target_new) >= margin``;
* official paraphrases, neighborhoods, 1000 retain records, and PPL are opened
  only after the final checkpoint has been materialized.

The output ``metrics_eff_gen_spe_ppl.json`` uses the locked paper contract:
Eff/Gen are residual sensitive-preference rates (lower is better), Spe is the
neighborhood probability-difference score (higher is better), and PPL is
lower/base-stable is better.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_mcf_sure_target_aware_direct_split as direct_split  # noqa: E402
import build_sure_minimal_split as shared_split  # noqa: E402


METHOD = "SURE-MCF-ZsRE-architecture-target_true-sensitive"
PROTOCOL = "mcf_zsre_arch_target_true_sensitive_direct_only_v1"
METRIC_SCHEMA = "mcf_target_true_sensitive_eff_gen_lower_v1"

TARGET_CONTRACT: Dict[str, Any] = {
    "sensitive_answer": "requested_rewrite.target_true",
    "non_sensitive_reference": "requested_rewrite.target_new",
    "field_swapping": False,
    "stage1": "target_true token GA plus same-prompt non-sensitive KL",
    "stage2": "NLL(target_true)-NLL(target_new) >= direct margin",
}

ARCHITECTURE: Dict[str, Any] = {
    "family": "canonical ZsRE SURE-LM shared architecture",
    "transformer_trainable": False,
    "stage1": {
        "tied_embedding_lm_head_trainable": True,
        "steps": 600,
        "batch_size": 1,
        "cache_batch_size": 8,
        "learning_rate": 1e-4,
        "ga_weight": 2.0,
        "gd_weight": 1.0,
        "gradient_clip": 1.0,
        "restoration": "retain target_true token rows only",
    },
    "stage2": {
        "lm_head_untied": True,
        "input_embeddings_trainable": False,
        "selected_rows": "residual target_true token rows only",
        "candidate_ranks": "2,8,0",
        "rank_zero_meaning": "unrestricted selected rows",
        "repair_steps": 800,
        "learning_rate": 0.005,
        "repair_l2": 1e-6,
        "batch_size": 8,
        "check_every": 25,
        "constraint_margin": 0.05,
        "candidate_scales": (
            "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,"
            ".046875,.03125,.015625,.0078125,0"
        ),
    },
}

METRIC_CONTRACT: Dict[str, Dict[str, str]] = {
    "Eff": {
        "direction": "lower_is_better",
        "definition": "100 * macro_mean(I[NLL(target_true) < NLL(target_new)]) on direct prompts",
    },
    "Gen": {
        "direction": "lower_is_better",
        "definition": "100 * macro_mean(I[NLL(target_true) < NLL(target_new)]) on paraphrases",
    },
    "Spe": {
        "direction": "higher_is_better",
        "definition": "100 * macro_mean(exp(-NLL(target_true))-exp(-NLL(target_new))) on neighborhoods",
    },
    "PPL": {
        "direction": "lower_or_base_stable_is_better",
        "definition": "ZeroUnlearn-compatible fixed Wikipedia perplexity",
    },
}


@dataclass(frozen=True)
class SeedPaths:
    root: Path
    protocol_dir: Path
    training_visible: Path
    split_manifest: Path
    retain_prompts: Path
    stage1_dir: Path
    stage2_dir: Path
    checkpoint: Path
    base_eval: Path
    final_eval: Path
    metrics: Path
    run_manifest: Path


@dataclass(frozen=True)
class Step:
    label: str
    command: List[str]


def seed_paths(output_root: Path, seed: int) -> SeedPaths:
    root = output_root / f"seed{seed}"
    protocol_dir = root / "protocol"
    stage1_dir = root / "stage1_zsre_arch"
    stage2_dir = root / "stage2_zsre_arch"
    return SeedPaths(
        root=root,
        protocol_dir=protocol_dir,
        training_visible=protocol_dir / "training_visible_mcf_target_true.json",
        split_manifest=protocol_dir / "split_manifest.json",
        retain_prompts=protocol_dir / "evaluation_only_retain_prompts.json",
        stage1_dir=stage1_dir,
        stage2_dir=stage2_dir,
        checkpoint=stage2_dir / "checkpoint",
        base_eval=root / "base_official_eval.json",
        final_eval=root / "final_official_eval.json",
        metrics=root / "metrics_eff_gen_spe_ppl.json",
        run_manifest=root / "run_manifest.json",
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", default="data/multi_counterfact.json")
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument(
        "--output-root", default="outputs/mcf_zsre_arch_target_true"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)

    stage1 = ARCHITECTURE["stage1"]
    parser.add_argument("--stage1-steps", type=int, default=stage1["steps"])
    parser.add_argument("--stage1-batch-size", type=int, default=stage1["batch_size"])
    parser.add_argument(
        "--stage1-cache-batch-size",
        type=int,
        default=stage1["cache_batch_size"],
    )
    parser.add_argument("--stage1-lr", type=float, default=stage1["learning_rate"])
    parser.add_argument("--ga-weight", type=float, default=stage1["ga_weight"])
    parser.add_argument("--gd-weight", type=float, default=stage1["gd_weight"])
    parser.add_argument("--gradient-clip", type=float, default=stage1["gradient_clip"])

    stage2 = ARCHITECTURE["stage2"]
    parser.add_argument("--candidate-ranks", default=stage2["candidate_ranks"])
    parser.add_argument("--repair-steps", type=int, default=stage2["repair_steps"])
    parser.add_argument("--repair-lr", type=float, default=stage2["learning_rate"])
    parser.add_argument("--repair-l2", type=float, default=stage2["repair_l2"])
    parser.add_argument("--repair-batch-size", type=int, default=stage2["batch_size"])
    parser.add_argument("--check-every", type=int, default=stage2["check_every"])
    parser.add_argument(
        "--constraint-margin", type=float, default=stage2["constraint_margin"]
    )
    parser.add_argument("--candidate-scales", default=stage2["candidate_scales"])
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--device-map", choices=("single", "auto"), default="single"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain unique values")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds must be non-negative")
    positive = (
        args.forget_num,
        args.retain_num,
        args.stage1_steps,
        args.stage1_batch_size,
        args.stage1_cache_batch_size,
        args.repair_steps,
        args.repair_batch_size,
        args.check_every,
    )
    if any(value <= 0 for value in positive):
        parser.error("counts, steps, and batch sizes must be positive")
    if args.stage1_lr <= 0 or args.repair_lr <= 0:
        parser.error("learning rates must be positive")
    if args.ga_weight <= 0 or args.gd_weight < 0:
        parser.error("GA must be positive and GD non-negative")
    if args.gradient_clip < 0 or args.repair_l2 < 0 or args.constraint_margin < 0:
        parser.error("clip, L2, and margin values must be non-negative")
    return args


def effective_architecture(args: argparse.Namespace) -> Dict[str, Any]:
    architecture = copy.deepcopy(ARCHITECTURE)
    architecture["stage1"].update(
        {
            "steps": int(args.stage1_steps),
            "batch_size": int(args.stage1_batch_size),
            "cache_batch_size": int(args.stage1_cache_batch_size),
            "learning_rate": float(args.stage1_lr),
            "ga_weight": float(args.ga_weight),
            "gd_weight": float(args.gd_weight),
            "gradient_clip": float(args.gradient_clip),
        }
    )
    architecture["stage2"].update(
        {
            "candidate_ranks": str(args.candidate_ranks),
            "repair_steps": int(args.repair_steps),
            "learning_rate": float(args.repair_lr),
            "repair_l2": float(args.repair_l2),
            "batch_size": int(args.repair_batch_size),
            "check_every": int(args.check_every),
            "constraint_margin": float(args.constraint_margin),
            "candidate_scales": str(args.candidate_scales),
        }
    )
    return architecture


def build_locked_split(
    mcf_path: Path,
    paths: SeedPaths,
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
) -> Dict[str, Any]:
    source_bytes = mcf_path.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    forget_pairs, retain_pairs = shared_split.sample_records(
        "mcf",
        raw,
        forget_num=forget_num,
        retain_eval_num=retain_num,
        seed=seed,
    )
    training = [
        direct_split.target_aware_direct_record(record, case_id)
        for case_id, record in forget_pairs
    ]
    direct_split.assert_direct_only_training_view(training)
    retain = [
        shared_split.prompt_only_retain_record("mcf", record, case_id)
        for case_id, record in retain_pairs
    ]

    paths.protocol_dir.mkdir(parents=True, exist_ok=False)
    training_text = json.dumps(training, indent=2, ensure_ascii=False) + "\n"
    retain_text = json.dumps(retain, indent=2, ensure_ascii=False) + "\n"
    paths.training_visible.write_text(training_text, encoding="utf-8")
    paths.retain_prompts.write_text(retain_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "dataset": "mcf",
        "seed": int(seed),
        "source_dataset": str(mcf_path),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible": str(paths.training_visible),
        "training_visible_sha256": sha256_bytes(training_text.encode("utf-8")),
        "evaluation_only_retain_prompts": str(paths.retain_prompts),
        "evaluation_only_retain_prompts_sha256": sha256_bytes(
            retain_text.encode("utf-8")
        ),
        "target_contract": TARGET_CONTRACT,
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget first, then retain, from one seeded RNG",
            "forget_num": int(forget_num),
            "retain_num": int(retain_num),
            "forget_case_ids": [int(index) for index, _ in forget_pairs],
            "retain_case_ids": [int(index) for index, _ in retain_pairs],
        },
        "data_roles": {
            "stage1_visible": [
                "direct prompt",
                "subject",
                "target_true sensitive answer",
                "target_new non-sensitive reference present but not optimized",
            ],
            "stage2_visible": [
                "same direct prompt and answers",
                "direct sequence-NLL constraint",
            ],
            "evaluation_only": [
                "official paraphrases",
                "official neighborhoods",
                "1000 sampled retain records",
                "fixed Wikipedia PPL text",
            ],
            "benchmark_retain_examples_visible_to_training": 0,
            "heldout_probes_visible_during_training": False,
            "official_metrics_used_for_checkpoint_selection": False,
        },
    }
    write_json(paths.split_manifest, manifest)
    return manifest


def _python(script: str, *arguments: Any) -> List[str]:
    return [
        sys.executable,
        str(SCRIPTS_DIR / script),
        *(str(argument) for argument in arguments),
    ]


def seed_command_plan(
    args: argparse.Namespace, paths: SeedPaths, seed: int
) -> List[Step]:
    return [
        Step(
            "STAGE 1 — ZSRE GA/KL + TARGET_TRUE-ROW RESTORATION",
            _python(
                "sure_stage1_gagd.py",
                "--dataset",
                "mcf",
                "--sensitive-field",
                "target_true",
                "--model-path",
                args.model_path,
                "--training-visible-path",
                paths.training_visible,
                "--split-manifest",
                paths.split_manifest,
                "--output-dir",
                paths.stage1_dir,
                "--seed",
                seed,
                "--forget-num",
                args.forget_num,
                "--steps",
                args.stage1_steps,
                "--batch-size",
                args.stage1_batch_size,
                "--cache-batch-size",
                args.stage1_cache_batch_size,
                "--emb-lm-lr",
                args.stage1_lr,
                "--ga-weight",
                args.ga_weight,
                "--gd-weight",
                args.gd_weight,
                "--grad-clip",
                args.gradient_clip,
                "--optimizer",
                "adamw",
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
            ),
        ),
        Step(
            "STAGE 2 — ZSRE SPARSE ROWS + MCF PAIRWISE CONSTRAINT",
            _python(
                "sure_stage2_sparse_repair.py",
                "--dataset",
                "mcf",
                "--mcf-sensitive-field",
                "target_true",
                "--mcf-reference-field",
                "target_new",
                "--model-path",
                paths.stage1_dir / "checkpoint",
                "--training-visible-path",
                paths.training_visible,
                "--split-manifest",
                paths.split_manifest,
                "--output-dir",
                paths.stage2_dir,
                "--seed",
                seed,
                "--forget-num",
                args.forget_num,
                "--candidate-ranks",
                args.candidate_ranks,
                "--repair-steps",
                args.repair_steps,
                "--repair-lr",
                args.repair_lr,
                "--constraint-margin",
                args.constraint_margin,
                "--repair-l2",
                args.repair_l2,
                "--batch-size",
                args.repair_batch_size,
                "--check-every",
                args.check_every,
                "--candidate-scales",
                args.candidate_scales,
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
            ),
        ),
        Step(
            "BASE OFFICIAL EVALUATION",
            _python(
                "mcf_zero_unlearn_official_eval.py",
                "--model-dir",
                args.model_path,
                "--mcf-path",
                args.mcf_path,
                "--wikidata-dir",
                args.wikidata_dir,
                "--out",
                paths.base_eval,
                "--unlearn-num",
                args.forget_num,
                "--retain-num",
                args.retain_num,
                "--seed",
                seed,
                "--sample-mode",
                "official",
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
                "--quiet",
            ),
        ),
        Step(
            "FINAL OFFICIAL EVALUATION",
            _python(
                "mcf_zero_unlearn_official_eval.py",
                "--model-dir",
                paths.checkpoint,
                "--mcf-path",
                args.mcf_path,
                "--wikidata-dir",
                args.wikidata_dir,
                "--out",
                paths.final_eval,
                "--unlearn-num",
                args.forget_num,
                "--retain-num",
                args.retain_num,
                "--seed",
                seed,
                "--sample-mode",
                "official",
                "--dtype",
                args.dtype,
                "--device-map",
                args.device_map,
                "--quiet",
            ),
        ),
    ]


def validate_stage_contract(paths: SeedPaths) -> Dict[str, Any]:
    stage1_path = paths.stage1_dir / "config_used.json"
    stage2_path = paths.stage2_dir / "repair_summary.json"
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    if stage1.get("sensitive_answer_field") != "target_true":
        raise RuntimeError("Stage 1 did not use target_true as sensitive")
    if stage2.get("sensitive_answer_field") != "target_true":
        raise RuntimeError("Stage 2 did not select target_true sensitive rows")
    if stage2.get("reference_answer_field") != "target_new":
        raise RuntimeError("Stage 2 did not use target_new as the reference")
    if stage1.get("benchmark_retain_seen") != 0 or stage2.get(
        "benchmark_retain_seen"
    ) != 0:
        raise RuntimeError("benchmark retain records leaked into optimization")
    if stage1.get("PPL_seen") is not False or stage2.get("PPL_seen") is not False:
        raise RuntimeError("PPL leaked into optimization")
    return {
        "stage1_sensitive_answer_field": stage1["sensitive_answer_field"],
        "stage2_sensitive_answer_field": stage2["sensitive_answer_field"],
        "stage2_reference_answer_field": stage2["reference_answer_field"],
        "stage2_direct_failures_before": int(stage2.get("active_before", 0)),
        "stage2_direct_failures_after": int(stage2.get("active_after", 0)),
        "selected_lm_head_rows": int(stage2.get("selected_lm_head_rows", 0)),
        "selected_scale": float(stage2.get("selected_scale", 0.0)),
    }


def _finite_nll(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is non-finite")
    return number


def prompt_metrics(
    rows: Sequence[Mapping[str, Any]], prompt_key: str
) -> Dict[str, float | int]:
    true_preference: List[float] = []
    new_preference: List[float] = []
    ties: List[float] = []
    prompt_count = 0
    for row_index, row in enumerate(rows):
        prompts = row.get("post", {}).get(prompt_key, [])
        if not prompts:
            continue
        row_true: List[float] = []
        row_new: List[float] = []
        row_ties: List[float] = []
        for prompt_index, values in enumerate(prompts):
            true_nll = _finite_nll(
                values["target_true"], f"row{row_index}.prompt{prompt_index}.true"
            )
            new_nll = _finite_nll(
                values["target_new"], f"row{row_index}.prompt{prompt_index}.new"
            )
            prompt_count += 1
            row_true.append(float(true_nll < new_nll))
            row_new.append(float(true_nll > new_nll))
            row_ties.append(float(true_nll == new_nll))
        true_preference.append(statistics.mean(row_true))
        new_preference.append(statistics.mean(row_new))
        ties.append(statistics.mean(row_ties))
    if not true_preference:
        raise RuntimeError(f"No prompts available for {prompt_key}")
    return {
        "record_count": len(true_preference),
        "prompt_instance_count": prompt_count,
        "target_true_preference_percent": 100.0 * statistics.mean(true_preference),
        "target_new_preference_percent": 100.0 * statistics.mean(new_preference),
        "tie_percent": 100.0 * statistics.mean(ties),
    }


def specificity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float | int]:
    preference = prompt_metrics(rows, "neighborhood_prompts_probs")
    differences: List[float] = []
    for row_index, row in enumerate(rows):
        prompts = row.get("post", {}).get("neighborhood_prompts_probs", [])
        if not prompts:
            continue
        per_record: List[float] = []
        for prompt_index, values in enumerate(prompts):
            true_nll = _finite_nll(
                values["target_true"],
                f"row{row_index}.neighborhood{prompt_index}.true",
            )
            new_nll = _finite_nll(
                values["target_new"],
                f"row{row_index}.neighborhood{prompt_index}.new",
            )
            per_record.append(math.exp(-true_nll) - math.exp(-new_nll))
        differences.append(statistics.mean(per_record))
    preference["Spe"] = 100.0 * statistics.mean(differences)
    preference["Spe_success"] = preference["target_true_preference_percent"]
    return preference


def primary_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    rows = payload.get("forget_raw", [])
    direct = prompt_metrics(rows, "rewrite_prompts_probs")
    paraphrase = prompt_metrics(rows, "paraphrase_prompts_probs")
    spe = specificity(rows)
    ppl = payload.get("forget_PPL")
    if ppl is None or not math.isfinite(float(ppl)):
        raise RuntimeError("PPL was not computed or is non-finite")
    return {
        "Eff": float(direct["target_true_preference_percent"]),
        "Gen": float(paraphrase["target_true_preference_percent"]),
        "Spe": float(spe["Spe"]),
        "PPL": float(ppl),
    }


def _assert_same_eval(base: Mapping[str, Any], post: Mapping[str, Any]) -> None:
    for key in ("seed", "unlearn_num", "retain_num", "sample_mode"):
        if base.get(key) != post.get(key):
            raise RuntimeError(f"base/post evaluation mismatch for {key}")
    for split in ("forget_raw", "retain_raw"):
        base_rows = base.get(split, [])
        post_rows = post.get(split, [])
        if len(base_rows) != len(post_rows):
            raise RuntimeError(f"base/post {split} counts differ")
        for index, (base_row, post_row) in enumerate(zip(base_rows, post_rows)):
            if base_row.get("requested_rewrite") != post_row.get("requested_rewrite"):
                raise RuntimeError(f"base/post {split} record mismatch at {index}")


def build_metrics_report(
    base: Mapping[str, Any],
    post: Mapping[str, Any],
    manifest: Mapping[str, Any],
    architecture: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    _assert_same_eval(base, post)
    if manifest.get("target_contract") != TARGET_CONTRACT:
        raise RuntimeError("split manifest target contract is incorrect")
    if post.get("sample_mode") != "official":
        raise RuntimeError("MCF metrics require official sampling")
    retain_num = int(post.get("retain_num", -1))
    if len(post.get("retain_raw", [])) != retain_num:
        raise RuntimeError("official retain evaluation is incomplete")

    base_primary = primary_metrics(base)
    post_primary = primary_metrics(post)
    post_direct = prompt_metrics(post["forget_raw"], "rewrite_prompts_probs")
    post_para = prompt_metrics(post["forget_raw"], "paraphrase_prompts_probs")
    post_spe = specificity(post["forget_raw"])
    base_retain = prompt_metrics(base["retain_raw"], "rewrite_prompts_probs")
    post_retain = prompt_metrics(post["retain_raw"], "rewrite_prompts_probs")

    return {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "dataset": "MCF",
        "seed": int(post["seed"]),
        "target_contract": TARGET_CONTRACT,
        "architecture": dict(architecture or ARCHITECTURE),
        "metric_contract": METRIC_CONTRACT,
        "primary_metrics": {
            name: {
                "value": float(value),
                "direction": METRIC_CONTRACT[name]["direction"],
            }
            for name, value in post_primary.items()
        },
        "base_primary_metrics": {
            name: {
                "value": float(value),
                "direction": METRIC_CONTRACT[name]["direction"],
            }
            for name, value in base_primary.items()
        },
        "delta_post_minus_base": {
            name: post_primary[name] - base_primary[name] for name in post_primary
        },
        "forget_audits": {
            "FS_direct_higher_is_better": post_direct[
                "target_new_preference_percent"
            ],
            "GFS_paraphrase_higher_is_better": post_para[
                "target_new_preference_percent"
            ],
            "direct_tie_percent": post_direct["tie_percent"],
            "paraphrase_tie_percent": post_para["tie_percent"],
            "Spe_success_higher_is_better": post_spe["Spe_success"],
        },
        "retain_evaluation_only_audit": {
            "retain_num": retain_num,
            "base_target_true_preference_direct_percent": base_retain[
                "target_true_preference_percent"
            ],
            "post_target_true_preference_direct_percent": post_retain[
                "target_true_preference_percent"
            ],
            "delta_post_minus_base_direct_percent": (
                post_retain["target_true_preference_percent"]
                - base_retain["target_true_preference_percent"]
            ),
        },
        "provenance": {
            "source_mcf_sha256": manifest.get("source_sha256"),
            "training_visible_sha256": manifest.get("training_visible_sha256"),
            "forget_case_ids": manifest.get("sampling", {}).get(
                "forget_case_ids", []
            ),
            "retain_case_ids": manifest.get("sampling", {}).get(
                "retain_case_ids", []
            ),
        },
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build_run_manifest(
    args: argparse.Namespace,
    paths: SeedPaths,
    plan: Sequence[Step],
    seed: int,
) -> Dict[str, Any]:
    runtime_files = (
        Path(__file__).resolve(),
        SCRIPTS_DIR / "sure_stage1_gagd.py",
        SCRIPTS_DIR / "sure_stage2_sparse_repair.py",
        SCRIPTS_DIR / "sure_canonical_core.py",
        SCRIPTS_DIR / "gagd_active_case_repair.py",
        SCRIPTS_DIR / "mcf_zero_unlearn_official_eval.py",
        SCRIPTS_DIR / "build_mcf_sure_target_aware_direct_split.py",
        SCRIPTS_DIR / "build_sure_minimal_split.py",
    )
    artifact_hashes = {
        str(path.relative_to(paths.root)): sha256_file(path)
        for path in (
            paths.training_visible,
            paths.split_manifest,
            paths.stage1_dir / "config_used.json",
            paths.stage2_dir / "repair_summary.json",
            paths.base_eval,
            paths.final_eval,
            paths.metrics,
        )
    }
    return {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": seed,
        "target_contract": TARGET_CONTRACT,
        "effective_arguments": vars(args),
        "commands": [asdict(step) for step in plan],
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status_scripts": _git_value(
            "status", "--porcelain=v1", "--", "scripts"
        ),
        "runtime_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in runtime_files
        },
        "artifact_sha256": artifact_hashes,
        "checkpoint": str(paths.checkpoint),
        "checkpoint_weights_hashed": False,
    }


def aggregate_reports(output_root: Path, seeds: Sequence[int]) -> Dict[str, Any]:
    reports = [
        json.loads(seed_paths(output_root, seed).metrics.read_text(encoding="utf-8"))
        for seed in seeds
    ]
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "metrics": {},
        "base_metrics": {},
    }
    for output_key, report_key in (
        ("metrics", "primary_metrics"),
        ("base_metrics", "base_primary_metrics"),
    ):
        for metric in ("Eff", "Gen", "Spe", "PPL"):
            values = [float(report[report_key][metric]["value"]) for report in reports]
            aggregate[output_key][metric] = {
                "mean": float(statistics.mean(values)),
                "sample_sd": (
                    float(statistics.stdev(values)) if len(values) > 1 else 0.0
                ),
                "values": values,
                "direction": METRIC_CONTRACT[metric]["direction"],
            }
    write_json(output_root / "aggregate_metrics.json", aggregate)

    rows = []
    for label, section in (("Base", "base_metrics"), (METHOD, "metrics")):
        row = {"Method": label, "Seeds": len(seeds)}
        for metric in ("Eff", "Gen", "Spe", "PPL"):
            summary = aggregate[section][metric]
            row[metric] = f"{summary['mean']:.4f} ± {summary['sample_sd']:.4f}"
        rows.append(row)
    with (output_root / "table_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("Method", "Seeds", "Eff", "Gen", "Spe", "PPL")
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MCF ZsRE-architecture results",
        "",
        "`target_true` is sensitive and `target_new` is the frozen reference.",
        "",
        "| Method | Seeds | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['Method']} | {row['Seeds']} | {row['Eff']} | {row['Gen']} | "
        f"{row['Spe']} | {row['PPL']} |"
        for row in rows
    )
    if len(seeds) < 10:
        lines.extend(["", "Pilot only: fewer than 10 seeds were run."])
    (output_root / "table_rows.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return aggregate


def print_contract(args: argparse.Namespace) -> None:
    print("=" * 72)
    print("MCF WITH CANONICAL ZSRE SURE ARCHITECTURE")
    print("target_true : SENSITIVE; suppress with GA")
    print("target_new  : NON-SENSITIVE FROZEN REFERENCE")
    print("Stage 2     : NLL(target_true)-NLL(target_new) >=", args.constraint_margin)
    print("Eff / Gen   : residual sensitive preference; LOWER is better")
    print("Spe         : neighborhood preservation; HIGHER is better")
    print("PPL         : LOWER/base-stable is better")
    print("Official paraphrases/neighborhoods/retain/PPL: evaluation-only")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not mcf_path.is_file():
        raise FileNotFoundError(f"MCF file not found: {mcf_path}")
    if not wikidata_dir.is_dir():
        raise FileNotFoundError(f"PPL wikidata directory not found: {wikidata_dir}")

    args.model_path = str(model_path)
    args.mcf_path = str(mcf_path)
    args.wikidata_dir = str(wikidata_dir)
    args.output_root = str(output_root)
    print_contract(args)

    for seed in args.seeds:
        paths = seed_paths(output_root, seed)
        plan = seed_command_plan(args, paths, seed)
        print(f"\n===== SEED {seed}: LOCKED NO-SWAP DIRECT SPLIT =====")
        print("internal: original target_true sensitive; original target_new reference")
        for step in plan:
            print(f"\n===== {step.label} =====")
            print(" ".join(step.command))

        if args.dry_run:
            continue
        if paths.root.exists():
            raise FileExistsError(
                f"seed output already exists; choose a fresh --output-root: {paths.root}"
            )
        paths.root.mkdir(parents=True, exist_ok=False)
        manifest = build_locked_split(
            mcf_path,
            paths,
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
        )
        stage_contract_audit: Dict[str, Any] | None = None
        for step_index, step in enumerate(plan):
            subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)
            if step_index == 1:
                stage_contract_audit = validate_stage_contract(paths)

        base = json.loads(paths.base_eval.read_text(encoding="utf-8"))
        post = json.loads(paths.final_eval.read_text(encoding="utf-8"))
        report = build_metrics_report(
            base, post, manifest, effective_architecture(args)
        )
        report["stage_contract_audit"] = stage_contract_audit
        report["provenance"].update(
            {
                "base_eval_json": str(paths.base_eval),
                "final_eval_json": str(paths.final_eval),
                "stage1_config": str(paths.stage1_dir / "config_used.json"),
                "stage2_summary": str(paths.stage2_dir / "repair_summary.json"),
                "checkpoint": str(paths.checkpoint),
            }
        )
        write_json(paths.metrics, report)
        run_manifest = build_run_manifest(args, paths, plan, seed)
        write_json(paths.run_manifest, run_manifest)

        print("\n===== PAPER METRICS =====")
        for metric in ("Eff", "Gen", "Spe", "PPL"):
            item = report["primary_metrics"][metric]
            print(f"{metric:<4} {item['value']:10.4f}  {item['direction']}")
        print("Wrote:", paths.metrics)

    if args.dry_run:
        print("\nDry run complete; no files were written and no models were loaded.")
        return

    aggregate = aggregate_reports(output_root, args.seeds)
    print("\nComplete:", output_root)
    print("Table:", output_root / "table_rows.md")
    if aggregate["seed_count"] < 10:
        print("NOTE: fewer than 10 seeds; treat this as a pilot.")


if __name__ == "__main__":
    main()
