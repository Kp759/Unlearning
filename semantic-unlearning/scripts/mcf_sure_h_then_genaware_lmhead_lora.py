#!/usr/bin/env python3
"""Gen-aware MCF Stage 2: frozen Stage-1 -> robust sparse LM-head LoRA.

This is the held-out-safe generalization version of
mcf_sure_h_then_residual_lmhead_lora.py.

Stage 1 is an already-finalized checkpoint (the current main experiment trains
sensitive input embeddings + sensitive LM-head rows while freezing the
transformer). Stage 2 freezes that checkpoint completely and trains only a sparse
LM-head LoRA delta:

    W_final[selected] = W_stage1[selected] + (B @ A) * (alpha / rank)

Unlike the direct-only Stage 2, the active forgetting pool contains the locked
direct prompt plus fixed surrogate paraphrases built ONLY from the locked direct
prompt. Official MCF paraphrase_prompts, neighborhood_prompts, retain-1000, and
PPL text are never available to training or checkpoint selection.

Default active policy is worst_per_record: at each gate, each fact contributes
its currently worst failing direct/surrogate prompt. Direct acceptance is a hard
priority during final LoRA scale selection; among scales that keep every direct
prompt passing, the scale with the best surrogate robustness is selected.

target_true is the sensitive/unwanted answer. target_new is never a supervised
replacement target; it is used only by the training-visible NLL margin gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from tqdm import tqdm

import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_sensitive_rows_projected_gagd as projected
import mcf_sure_h_then_residual_lmhead_lora as direct_lora
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-GenAware-residual-sparse-LM-head-LoRA"
PROTOCOL = "mcf_target_true_genaware_surrogate_lmhead_lora_v1"
SURROGATE_PROTOCOL = "mcf_locked_direct_only_surrogate_paraphrases_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_scales(text: str) -> List[float]:
    values = sorted({float(x.strip()) for x in str(text).split(",") if x.strip()})
    if not values:
        raise ValueError("candidate LoRA scales cannot be empty")
    if any((not math.isfinite(x)) or x < 0.0 or x > 1.0 for x in values):
        raise ValueError("candidate LoRA scales must be finite values in [0,1]")
    if 1.0 not in values:
        raise ValueError("candidate LoRA scales must include 1")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage1-model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--surrogate-prompts-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--lora-l2-weight", type=float, default=1e-4)
    p.add_argument("--active-policy", choices=("worst_per_record", "all_failures"), default="worst_per_record")

    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--solver-margin", type=float, default=0.25)
    p.add_argument("--acceptance-margin", type=float, default=0.05)
    p.add_argument(
        "--candidate-lora-scales",
        default="0,0.03125,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.625,0.75,0.875,1",
    )

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--locality-sensitive-logit-weight", type=float, default=5.0)

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    positive = (
        a.forget_num, a.steps, a.batch_size, a.cache_batch_size, a.check_every,
        a.lr, a.ga_weight, a.lora_rank, a.lora_alpha,
        a.locality_batch_size, a.locality_cache_batch_size,
        a.utility_sample_size, a.utility_batch_size, a.utility_cache_batch_size,
        a.utility_max_length,
    )
    if any(float(x) <= 0 for x in positive):
        p.error("counts, cadence, LR, GA, LoRA rank/alpha, and batch sizes must be positive")
    nonnegative = (
        a.grad_clip, a.gd_weight, a.lora_l2_weight,
        a.solver_margin, a.acceptance_margin,
        a.locality_kl_weight, a.locality_sensitive_logit_weight,
        a.utility_kl_weight, a.utility_exclude_first,
    )
    if any(float(x) < 0 for x in nonnegative):
        p.error("clip/weights/margins/exclusion must be non-negative")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be >=20 to protect the fixed PPL prefix")
    try:
        a.candidate_lora_scales = _parse_scales(a.candidate_lora_scales)
    except ValueError as exc:
        p.error(str(exc))
    return a


def _norm(text: str) -> str:
    return " ".join(str(text).split()).strip().casefold()


def load_surrogate_artifact(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
) -> Tuple[Dict[str, Any], List[List[str]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise RuntimeError("Unsupported surrogate artifact schema")
    if data.get("protocol") != SURROGATE_PROTOCOL:
        raise RuntimeError("Unexpected surrogate artifact protocol")
    if int(data.get("seed", -1)) != int(seed):
        raise RuntimeError("Surrogate artifact seed mismatch")
    if int(data.get("forget_num", -1)) != int(forget_num):
        raise RuntimeError("Surrogate artifact forget count mismatch")
    access = data.get("data_access", {})
    if int(access.get("official_paraphrase_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports official paraphrase access")
    if int(access.get("official_neighborhood_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports official neighborhood access")
    if int(access.get("benchmark_retain_seen", -1)) != 0:
        raise RuntimeError("Surrogate artifact reports benchmark retain access")
    if bool(access.get("official_PPL_seen", True)):
        raise RuntimeError("Surrogate artifact reports official PPL access")

    rows = data.get("records")
    if not isinstance(rows, list) or len(rows) != int(forget_num):
        raise RuntimeError("Surrogate artifact records do not match forget count")
    all_prompts: List[List[str]] = []
    for pos, (record, row) in enumerate(zip(records, rows)):
        if int(row.get("sampled_position", -1)) != pos:
            raise RuntimeError(f"Surrogate sampled_position mismatch at {pos}")
        expected_case = int(record.get("case_id", pos))
        if int(row.get("case_id", -1)) != expected_case:
            raise RuntimeError(f"Surrogate case_id mismatch at {pos}")
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        direct_prompt = str(rr["prompt"]).format(subject)
        if str(row.get("subject", "")) != subject:
            raise RuntimeError(f"Surrogate subject mismatch at {pos}")
        if _norm(row.get("direct_prompt", "")) != _norm(direct_prompt):
            raise RuntimeError(f"Surrogate direct prompt mismatch at {pos}")
        prompts = row.get("surrogate_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"No surrogate prompts at {pos}")
        direct_key = _norm(direct_prompt)
        true_key = _norm(rr["target_true"]["str"])
        new_key = _norm(rr["target_new"]["str"])
        seen = {direct_key}
        clean: List[str] = []
        for j, prompt in enumerate(prompts):
            prompt = " ".join(str(prompt).split()).strip()
            key = _norm(prompt)
            if not prompt or key in seen:
                raise RuntimeError(f"Empty/duplicate surrogate at record {pos}, index {j}")
            # Strong audit guard: neither known answer may be literally embedded in
            # a surrogate prompt. The builder also enforces this.
            if true_key and true_key in key:
                raise RuntimeError(f"target_true leaked into surrogate at record {pos}, index {j}")
            if new_key and new_key in key:
                raise RuntimeError(f"target_new leaked into surrogate at record {pos}, index {j}")
            seen.add(key)
            clean.append(prompt)
        all_prompts.append(clean)
    return data, all_prompts


def build_instances(
    records: Sequence[Mapping[str, Any]],
    surrogate_prompts: Sequence[Sequence[str]],
) -> Tuple[List[mcf_repair.MCFPromptInstance], List[mcf_repair.MCFPromptInstance]]:
    direct = stage2.mcf_instances(records)
    surrogate: List[mcf_repair.MCFPromptInstance] = []
    for pos, (record, prompts) in enumerate(zip(records, surrogate_prompts)):
        rr = record["requested_rewrite"]
        for j, prompt in enumerate(prompts):
            surrogate.append(
                mcf_repair.MCFPromptInstance(
                    record_index=int(record.get("case_id", pos)),
                    sampled_position=int(pos),
                    prompt_type="surrogate",
                    prompt_index=int(j),
                    prompt=str(prompt),
                    target_new=str(rr["target_new"]["str"]),
                    target_true=str(rr["target_true"]["str"]),
                )
            )
    return direct, surrogate


def prediction_cases_from_instances(
    instances: Sequence[mcf_repair.MCFPromptInstance],
    tok,
    *,
    llama_like: bool,
) -> Tuple[List[core.SensitivePredictionCase], List[int]]:
    cases: List[core.SensitivePredictionCase] = []
    case_to_instance: List[int] = []
    for inst_idx, inst in enumerate(instances):
        tids = core.answer_token_ids(tok, inst.target_true, llama_like=llama_like)
        for token_index, token_id in enumerate(tids):
            decoded_prefix = tok.decode(tids[:token_index])
            if llama_like and token_index > 0:
                prompt = inst.prompt + " " + decoded_prefix
            else:
                prompt = inst.prompt + decoded_prefix
            cases.append(
                core.SensitivePredictionCase(
                    case_id=int(inst.record_index),
                    record_position=int(inst.sampled_position),
                    token_index=int(token_index),
                    prompt=prompt,
                    target_text=tok.decode([token_id]),
                )
            )
            case_to_instance.append(int(inst_idx))
    return cases, case_to_instance


@torch.no_grad()
def _margins(model, tok, instances, device, llama_like, batch_size) -> torch.Tensor:
    return stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(batch_size),
        "target_true", "target_new"
    ).detach().float().cpu()


def _report(margins: torch.Tensor, threshold: float) -> Dict[str, Any]:
    if margins.numel() == 0:
        return {
            "prompt_instances": 0,
            "failures": 0,
            "minimum_margin": None,
            "mean_margin": None,
            "maximum_margin": None,
            "threshold": float(threshold),
        }
    m = margins.detach().float().cpu()
    return {
        "prompt_instances": int(m.numel()),
        "failures": int((m < float(threshold)).sum().item()),
        "minimum_margin": float(m.min()),
        "mean_margin": float(m.mean()),
        "maximum_margin": float(m.max()),
        "threshold": float(threshold),
    }


def _split_report(
    direct_count: int,
    margins: torch.Tensor,
    threshold: float,
) -> Dict[str, Any]:
    direct = margins[:direct_count]
    surrogate = margins[direct_count:]
    return {
        "combined": _report(margins, threshold),
        "direct": _report(direct, threshold),
        "surrogate": _report(surrogate, threshold),
    }


def select_active_instances(
    instances: Sequence[mcf_repair.MCFPromptInstance],
    margins: torch.Tensor,
    threshold: float,
    policy: str,
) -> List[int]:
    failing = [i for i, value in enumerate(margins.tolist()) if float(value) < float(threshold)]
    if policy == "all_failures":
        return failing
    if policy != "worst_per_record":
        raise ValueError(f"Unsupported active policy: {policy}")
    by_record: Dict[int, List[int]] = {}
    for i in failing:
        by_record.setdefault(int(instances[i].sampled_position), []).append(i)
    active: List[int] = []
    for record_pos in sorted(by_record):
        candidates = by_record[record_pos]
        active.append(min(candidates, key=lambda i: (float(margins[i]), i)))
    return active


def _active_case_ids(case_to_instance: Sequence[int], active_instances: Sequence[int]) -> List[int]:
    active = set(int(x) for x in active_instances)
    return [i for i, inst_idx in enumerate(case_to_instance) if int(inst_idx) in active]


def _selected_rows(
    tok,
    cases: Sequence[core.SensitivePredictionCase],
    active_case_ids: Sequence[int],
    *,
    llama_like: bool,
    device,
) -> List[int]:
    tids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device).detach().cpu()
    special = gagd.special_token_ids(tok)
    selected = sorted({int(tids[i].item()) for i in active_case_ids if int(tids[i].item()) not in special})
    if active_case_ids and not selected:
        raise RuntimeError("Active surrogate/direct failures produced no selected LM-head rows")
    return selected


def _scale_key(report: Dict[str, Any]) -> Tuple[Any, ...]:
    # Direct Eff is a hard priority. Within direct-safe scales, optimize surrogate
    # failures/worst margin before preferring a smaller adapter.
    d = report["direct"]
    s = report["surrogate"]
    c = report["combined"]
    direct_fail = int(d["failures"])
    surrogate_fail = int(s["failures"])
    smin = float("-inf") if s["minimum_margin"] is None else float(s["minimum_margin"])
    cmin = float("-inf") if c["minimum_margin"] is None else float(c["minimum_margin"])
    cmean = float("-inf") if c["mean_margin"] is None else float(c["mean_margin"])
    return (
        0 if direct_fail == 0 else 1,
        direct_fail,
        surrogate_fail,
        -smin,
        -cmin,
        -cmean,
        float(report["scale"]),
    )


def choose_scale(
    model,
    tok,
    instances,
    direct_count: int,
    delta_module: direct_lora.SparseLMHeadLoRA,
    scales: Sequence[float],
    acceptance_margin: float,
    device,
    llama_like: bool,
    batch_size: int,
):
    reports: List[Dict[str, Any]] = []
    for scale in scales:
        delta_module.multiplier = float(scale)
        margins = _margins(model, tok, instances, device, llama_like, batch_size)
        report = _split_report(direct_count, margins, acceptance_margin)
        report["scale"] = float(scale)
        reports.append(report)
    perfect = [
        r for r in reports
        if int(r["direct"]["failures"]) == 0 and int(r["surrogate"]["failures"]) == 0
    ]
    if perfect:
        chosen = min(perfect, key=lambda r: float(r["scale"]))
    else:
        chosen = min(reports, key=_scale_key)
    delta_module.multiplier = float(chosen["scale"])
    return float(chosen["scale"]), reports


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    surrogate_path = Path(a.surrogate_prompts_path).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)
    surrogate_artifact, surrogate_prompts = load_surrogate_artifact(
        surrogate_path, records, seed=int(a.seed), forget_num=int(a.forget_num)
    )
    direct_instances, surrogate_instances = build_instances(records, surrogate_prompts)
    all_instances = [*direct_instances, *surrogate_instances]
    direct_count = len(direct_instances)

    ns = argparse.Namespace(
        model_path=a.stage1_model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    model.eval()

    all_cases, all_case_to_instance = prediction_cases_from_instances(
        all_instances, tok, llama_like=llama_like
    )
    direct_cases, _ = prediction_cases_from_instances(
        direct_instances, tok, llama_like=llama_like
    )
    if not all_cases or not direct_cases:
        raise RuntimeError("No target_true prediction cases")

    parent_margins = _margins(
        model, tok, all_instances, device, llama_like, int(a.cache_batch_size)
    )
    parent_solver = _split_report(direct_count, parent_margins, float(a.solver_margin))
    parent_acceptance = _split_report(direct_count, parent_margins, float(a.acceptance_margin))
    active_instances = select_active_instances(
        all_instances, parent_margins, float(a.solver_margin), a.active_policy
    )
    active_case_ids = _active_case_ids(all_case_to_instance, active_instances)
    selected_ids = _selected_rows(
        tok, all_cases, active_case_ids,
        llama_like=llama_like, device=device
    )

    # Same-prompt GD remains on the 50 locked direct requests. This avoids
    # treating synthetic prompts as utility targets while preserving the parent
    # distribution on every canonical training-visible fact.
    direct_parent_logits = core.cache_base_logits(
        model, tok, direct_cases, device, batch_size=int(a.cache_batch_size)
    )
    locality_prompts, locality_protected, locality_receipt = (
        projected.build_relation_locality_controls(
            records, tok, direct_instances, int(a.subject_control_count)
        )
    )
    print(f"Caching Stage-1 locality references for {len(locality_prompts)} prompts...", flush=True)
    _lh, locality_parent_logits = projected.cache_relation_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )
    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(f"Caching Stage-1 Wikipedia references for {len(utility_prompts)} prompts...", flush=True)
    utility_parent_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    for p in model.parameters():
        p.requires_grad_(False)
    output_layer = model.get_output_embeddings()
    input_layer = model.get_input_embeddings()
    if output_layer is None or not hasattr(output_layer, "weight"):
        raise RuntimeError("model has no LM-head weight")
    if input_layer is None or not hasattr(input_layer, "weight"):
        raise RuntimeError("model has no input embedding weight")
    if output_layer.weight.data_ptr() == input_layer.weight.data_ptr():
        raise RuntimeError(
            "Gen-aware Stage 2 requires a detied LM head; materializing LM-head LoRA "
            "must not alter the frozen Stage-1 input embeddings"
        )
    parent_head = output_layer.weight.detach().clone()
    parent_input = input_layer.weight.detach().clone()

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    _write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)
    _write_json(out_dir / "surrogate_training_receipt.json", {
        "surrogate_prompts_path": str(surrogate_path),
        "sha256": _sha256(surrogate_path),
        "protocol": surrogate_artifact.get("protocol"),
        "generator": surrogate_artifact.get("generator"),
        "records": len(surrogate_prompts),
        "surrogate_prompt_instances": len(surrogate_instances),
        "official_paraphrase_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "official_PPL_seen": False,
    })

    if not active_case_ids:
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)
        summary = {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "stage2_noop": True,
            "parent_solver": parent_solver,
            "parent_acceptance": parent_acceptance,
            "selected_lm_head_rows": 0,
            "chosen_lora_scale": 0.0,
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "PPL_seen": False,
            "checkpoint": str(ckpt.resolve()),
        }
        _write_json(out_dir / "genaware_lmhead_lora_summary.json", summary)
        print("Stage-1 parent already satisfies every direct+surrogate solver margin.")
        return

    delta_module = direct_lora.SparseLMHeadLoRA(
        selected_ids,
        int(output_layer.weight.shape[1]),
        int(a.lora_rank),
        float(a.lora_alpha),
        output_layer.weight.device,
    )
    output_hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    lora_params = [delta_module.lora_A, delta_module.lora_B]
    opt = direct_lora._optimizer(lora_params, a.optimizer, float(a.lr))

    gd_sampler = core.IndexSampler(len(direct_cases), int(a.batch_size), int(a.seed) + 24001)
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 24003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 24005
    )
    active_sampler = core.IndexSampler(
        len(active_case_ids), int(a.batch_size), int(a.seed) + 24007
    )

    gate_log = (out_dir / "genaware_gate_log.jsonl").open("w", encoding="utf-8")
    train_log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
    gate_log.write(json.dumps({
        "step": 0,
        "active_policy": a.active_policy,
        "active_prompt_instances": active_instances,
        "solver": parent_solver,
        "acceptance": parent_acceptance,
    }) + "\n")
    gate_log.flush()

    stopped_step = 0
    for step in tqdm(range(1, int(a.steps) + 1), desc="Gen-aware residual LM-head LoRA"):
        stopped_step = int(step)
        local_active = active_sampler.next()
        fidx = [active_case_ids[i] for i in local_active]
        forget_batch = [all_cases[i] for i in fidx]
        gidx = gd_sampler.next()
        gd_batch = [direct_cases[i] for i in gidx]
        lidx = locality_sampler.next()
        locality_batch = [locality_prompts[i] for i in lidx]
        locality_ids = [locality_protected[i] for i in lidx]
        uidx = utility_sampler.next()
        utility_batch = [utility_prompts[i] for i in uidx]

        opt.zero_grad(set_to_none=True)
        ga_logits = core.forward_last_logits(model, tok, forget_batch, device)
        ga_tids = core.official_target_ids(
            tok, forget_batch, llama_like=llama_like, device=device
        )
        ga = core.ga_sensitive_logprob(ga_logits, ga_tids)

        gd_logits = core.forward_last_logits(model, tok, gd_batch, device)
        gd_tids = core.official_target_ids(
            tok, gd_batch, llama_like=llama_like, device=device
        )
        gd = core.gd_non_sensitive_kl(
            gd_logits, direct_parent_logits[gidx], gd_tids
        )

        _cur_h, local_logits = projected._prompt_hidden_and_logits(
            model, tok, locality_batch, device
        )
        local_base = locality_parent_logits[lidx]
        lkl = projected.locality_kl(local_logits, local_base)
        lrow = projected.protected_sensitive_logit_mse(
            local_logits, local_base, locality_ids
        )
        utility_logits = wikipedia_utility._forward_prompt_logits(
            model, tok, utility_batch, device
        )
        ukl = wikipedia_utility.utility_kl(
            utility_logits, utility_parent_logits[uidx]
        )
        lora_l2 = delta_module.effective_delta().square().mean()
        loss = (
            float(a.ga_weight) * ga
            + float(a.gd_weight) * gd
            + float(a.locality_kl_weight) * lkl
            + float(a.locality_sensitive_logit_weight) * lrow
            + float(a.utility_kl_weight) * ukl
            + float(a.lora_l2_weight) * lora_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Gen-aware loss at step {step}")
        loss.backward()
        grad_norm = (
            torch.nn.utils.clip_grad_norm_(lora_params, float(a.grad_clip))
            if a.grad_clip > 0 else torch.tensor(0.0, device=device)
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite Gen-aware grad norm at step {step}")
        opt.step()

        train_log.write(json.dumps({
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "ga": float(ga.detach().cpu()),
            "gd": float(gd.detach().cpu()),
            "locality_kl": float(lkl.detach().cpu()),
            "locality_sensitive_logit": float(lrow.detach().cpu()),
            "wikipedia_kl": float(ukl.detach().cpu()),
            "lora_l2": float(lora_l2.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
            "active_prompt_instance_count": len(active_instances),
            "active_token_case_count": len(active_case_ids),
        }) + "\n")
        train_log.flush()

        if step % int(a.check_every) == 0 or step == int(a.steps):
            margins = _margins(
                model, tok, all_instances, device, llama_like,
                int(a.cache_batch_size)
            )
            active_instances = select_active_instances(
                all_instances, margins, float(a.solver_margin), a.active_policy
            )
            active_case_ids = _active_case_ids(all_case_to_instance, active_instances)
            gate_log.write(json.dumps({
                "step": int(step),
                "active_prompt_instances": active_instances,
                "solver": _split_report(direct_count, margins, float(a.solver_margin)),
                "acceptance": _split_report(direct_count, margins, float(a.acceptance_margin)),
            }) + "\n")
            gate_log.flush()
            if not active_case_ids:
                break
            active_sampler = core.IndexSampler(
                len(active_case_ids), int(a.batch_size), int(a.seed) + 24007 + int(step)
            )

    gate_log.close()
    train_log.close()

    chosen_scale, scale_reports = choose_scale(
        model, tok, all_instances, direct_count, delta_module,
        a.candidate_lora_scales, float(a.acceptance_margin),
        device, llama_like, int(a.cache_batch_size)
    )
    _write_json(out_dir / "lora_scale_reports.json", scale_reports)

    pre_margins = _margins(
        model, tok, all_instances, device, llama_like, int(a.cache_batch_size)
    )
    locality_pre = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, locality_parent_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_pre = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_parent_logits,
        device, int(a.utility_cache_batch_size)
    )

    parent_head_change = float((output_layer.weight.detach() - parent_head).abs().max().cpu())
    parent_input_change = float((input_layer.weight.detach() - parent_input).abs().max().cpu())
    if parent_head_change != 0.0:
        raise RuntimeError("Underlying Stage-1 LM head changed before LoRA materialization")
    if parent_input_change != 0.0:
        raise RuntimeError("Stage-1 input embeddings changed during Gen-aware Stage 2")

    lora_delta_final = delta_module.effective_delta().detach().float().cpu()
    torch.save({
        "selected_lm_head_token_ids": selected_ids,
        "lora_A": delta_module.lora_A.detach().float().cpu(),
        "lora_B": delta_module.lora_B.detach().float().cpu(),
        "rank": int(delta_module.rank),
        "alpha": float(delta_module.alpha),
        "chosen_scale": float(chosen_scale),
        "effective_delta": lora_delta_final,
        "surrogate_artifact_sha256": _sha256(surrogate_path),
    }, out_dir / "lmhead_lora_factors.pt")

    output_hook.remove()
    direct_lora._materialize_selected_delta(output_layer.weight, selected_ids, lora_delta_final)
    model.eval()

    post_margins = _margins(
        model, tok, all_instances, device, llama_like, int(a.cache_batch_size)
    )
    locality_after = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, locality_parent_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_parent_logits,
        device, int(a.utility_cache_batch_size)
    )

    ids_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_layer.weight.device)
    materialized = (
        output_layer.weight.detach().index_select(0, ids_tensor).float().cpu()
        - parent_head.index_select(0, ids_tensor.to(parent_head.device)).float().cpu()
    )
    materialization_error = float((materialized - lora_delta_final).abs().max())
    input_change_after = float((input_layer.weight.detach() - parent_input).abs().max().cpu())
    if input_change_after != 0.0:
        raise RuntimeError("Stage-1 input embeddings changed after LM-head materialization")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    final_acceptance = _split_report(direct_count, post_margins, float(a.acceptance_margin))
    final_solver = _split_report(direct_count, post_margins, float(a.solver_margin))
    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "stage1_model_path": str(Path(a.stage1_model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "surrogate_prompts_path": str(surrogate_path),
        "surrogate_artifact_sha256": _sha256(surrogate_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "target_new_is_replacement_training_target": False,
        },
        "stage1": "frozen embedding+LM-head SURE parent; transformer already frozen",
        "stage2": "robust sparse LM-head LoRA over direct + training-only surrogate prompts",
        "active_policy": a.active_policy,
        "direct_prompt_instances": len(direct_instances),
        "surrogate_prompt_instances": len(surrogate_instances),
        "combined_prompt_instances": len(all_instances),
        "parent_solver": parent_solver,
        "parent_acceptance": parent_acceptance,
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "lora_rank": int(delta_module.rank),
        "lora_alpha": float(delta_module.alpha),
        "lora_scaling": float(delta_module.scaling),
        "lora_trainable_parameters": int(delta_module.trainable_parameter_count),
        "chosen_lora_scale": float(chosen_scale),
        "stopped_step": int(stopped_step),
        "pre_materialize_solver": _split_report(direct_count, pre_margins, float(a.solver_margin)),
        "pre_materialize_acceptance": _split_report(direct_count, pre_margins, float(a.acceptance_margin)),
        "post_materialize_solver": final_solver,
        "post_materialize_acceptance": final_acceptance,
        "relation_locality_pre_materialize": locality_pre,
        "relation_locality_after": locality_after,
        "wikipedia_utility_pre_materialize": utility_pre,
        "wikipedia_utility_after": utility_after,
        "underlying_stage1_head_max_abs_change_before_materialization": parent_head_change,
        "stage1_input_embedding_max_abs_change_before_materialization": parent_input_change,
        "stage1_input_embedding_max_abs_change_after_materialization": input_change_after,
        "lora_delta_norm": float(lora_delta_final.norm()),
        "lora_delta_mse": float(lora_delta_final.square().mean()),
        "materialization_max_abs_error": materialization_error,
        "official_paraphrase_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    _write_json(out_dir / "genaware_lmhead_lora_summary.json", summary)

    print(f"Gen-aware LM-head LoRA checkpoint: {ckpt.resolve()}")
    print("Parent direct acceptance:", parent_acceptance["direct"])
    print("Parent surrogate acceptance:", parent_acceptance["surrogate"])
    print("Selected LM-head rows:", len(selected_ids))
    print("LoRA rank / alpha:", int(delta_module.rank), "/", float(delta_module.alpha))
    print("Chosen LoRA scale:", float(chosen_scale))
    print("Final direct acceptance:", final_acceptance["direct"])
    print("Final surrogate acceptance:", final_acceptance["surrogate"])
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Stage-1 input embeddings max abs change:", input_change_after)
    print("Underlying Stage-1 head max abs change before materialization:", parent_head_change)
    print("Materialization max abs error:", materialization_error)
    print("Official MCF paraphrase/neighborhood/retain/PPL data were NOT used.")


if __name__ == "__main__":
    main()
