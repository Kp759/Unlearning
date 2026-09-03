#!/usr/bin/env python3
"""Train the sparse, internal MCF bi-endpoint nullspace edit.

Only selected input-embedding and untied LM-head rows are represented by
trainable deltas.  The Transformer remains frozen.  Shared input rows are
projected away from row-specific protected Jacobian sketches; all selected
LM-head rows are projected away from a protected hidden-state sketch.  There
is no contextual classifier, runtime gate, sidecar, or official-evaluation
input in this process.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import build_mcf_biendpoint_nullspace_rewiring_v2_split as split_builder
import gagd_compare as gagd
import mcf_biendpoint_nullspace_rewiring_v2_core as geometry
import mcf_shadow_relation_prompts as relation_prompts
import mcf_sure_directional_emb_lm_stage1 as endpoint_hooks
import mcf_sure_subject_directional_emb_stage1 as subject_stage1
import mcf_synthetic_paraphrase_templates as synthetic
import sure_canonical_core as canonical


FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "OFFICIAL",
    "OFFICIAL_DIR",
    "OFFICIAL_MCF_PATH",
    "MCF_OFFICIAL_OUTPUT",
    "RECOVERY",
    "RECOVERY_DIR",
    "RETAIN_PATH",
    "PPL_PATH",
    "ALIAS_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
)

PROTECTION_SCAFFOLDS = {
    "fit": "For {subject}, a different recorded property ({relation}) is",
    "development": "Regarding {subject}, another property, {relation}, is",
    "certification": "State the unrelated {relation} value for {subject}:",
}


@dataclass
class ProtectionCache:
    cases: List[canonical.SensitivePredictionCase]
    topk_ids: torch.Tensor
    base_topk_log_probs: torch.Tensor
    base_top1_ids: torch.Tensor
    base_top1_log_probs: torch.Tensor


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_collection_sha256(tensors: Sequence[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode("utf-8"))
        view = tensor.detach().cpu().contiguous().view(torch.uint8)
        digest.update(view.numpy().tobytes())
    return digest.hexdigest()


def transformer_sha256(
    model: torch.nn.Module, *, excluded_parameters: Sequence[torch.nn.Parameter]
) -> str:
    excluded = {id(parameter) for parameter in excluded_parameters}
    tensors = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if id(parameter) not in excluded
    ]
    return tensor_collection_sha256(tensors)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=12000)
    parser.add_argument("--corpus-fit-prompts", type=int, default=1000)
    parser.add_argument("--corpus-development-prompts", type=int, default=250)
    parser.add_argument("--corpus-certification-prompts", type=int, default=1000)
    parser.add_argument("--synthetic-paraphrases", type=int, default=3)
    parser.add_argument("--same-subject-prompts", type=int, default=4)
    parser.add_argument("--input-jacobian-sketches", type=int, default=64)
    parser.add_argument("--input-rank-cap", type=int, default=32)
    parser.add_argument("--output-rank-cap", type=int, default=256)
    parser.add_argument("--minimum-projected-gradient-norm", type=float, default=1e-8)
    parser.add_argument("--input-relative-cap", type=float, default=0.5)
    parser.add_argument("--input-frequency-alpha", type=float, default=0.25)
    parser.add_argument("--output-relative-cap", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--forget-batch-size", type=int, default=4)
    parser.add_argument("--protection-batch-size", type=int, default=8)
    parser.add_argument("--capture-batch-size", type=int, default=8)
    parser.add_argument("--embedding-lr", type=float, default=5e-4)
    parser.add_argument("--lm-head-lr", type=float, default=1e-3)
    parser.add_argument("--forget-margin-target", type=float, default=6.0)
    parser.add_argument("--forget-margin-weight", type=float, default=100.0)
    parser.add_argument("--protection-topk", type=int, default=64)
    parser.add_argument("--protection-kl-weight", type=float, default=10.0)
    parser.add_argument("--protection-top1-weight", type=float, default=10.0)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--minimum-forget-margin", type=float, default=0.1)
    parser.add_argument("--protected-kl-mean-max", type=float, default=1e-4)
    parser.add_argument("--protected-kl-max", type=float, default=1e-2)
    parser.add_argument("--protected-top1-drift-max", type=float, default=5e-2)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.gradient_checkpointing = False
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V2 is locked to consumed seed 1 / 50 facts")
    if args.frequency_doc_start < 20:
        parser.error("Wikipedia documents 0:20 remain reserved for official PPL")
    if args.steps <= 0 or args.check_every <= 0 or args.steps % args.check_every:
        parser.error("steps must be positive and divisible by check-every")
    return args


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "official/recovery input leaked into V2 training: "
            + ", ".join(sorted(exposed))
        )


def locked_registry_values(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data": {
            "seed": args.seed,
            "forget_records": args.forget_num,
            "wikipedia_doc_start": args.frequency_doc_start,
            "wikipedia_documents": args.frequency_docs,
            "corpus_fit_prompts": args.corpus_fit_prompts,
            "corpus_development_prompts": args.corpus_development_prompts,
            "corpus_certification_prompts": args.corpus_certification_prompts,
            "synthetic_paraphrases_per_forget_record": args.synthetic_paraphrases,
            "same_subject_different_relation_prompts_per_record": args.same_subject_prompts,
        },
        "geometry": {
            "input_jacobian_sketches": args.input_jacobian_sketches,
            "input_protected_rank_cap": args.input_rank_cap,
            "output_protected_rank_cap": args.output_rank_cap,
            "minimum_projected_gradient_norm": args.minimum_projected_gradient_norm,
            "input_relative_row_cap": args.input_relative_cap,
            "input_frequency_alpha": args.input_frequency_alpha,
            "output_relative_row_cap": args.output_relative_cap,
        },
        "optimization": {
            "steps": args.steps,
            "check_every": args.check_every,
            "forget_batch_size": args.forget_batch_size,
            "protection_batch_size": args.protection_batch_size,
            "capture_batch_size": args.capture_batch_size,
            "embedding_lr": args.embedding_lr,
            "lm_head_lr": args.lm_head_lr,
            "forget_margin_target": args.forget_margin_target,
            "forget_margin_weight": args.forget_margin_weight,
            "protection_topk": args.protection_topk,
            "protection_kl_weight": args.protection_kl_weight,
            "protection_top1_drift_weight": args.protection_top1_weight,
            "delta_l2_weight": args.delta_l2_weight,
            "gradient_clip": args.gradient_clip,
        },
        "acceptance": {
            "minimum_forget_margin": args.minimum_forget_margin,
            "protected_topk_kl_mean_max": args.protected_kl_mean_max,
            "protected_topk_kl_absolute_max": args.protected_kl_max,
            "protected_top1_logprob_abs_max": args.protected_top1_drift_max,
        },
    }


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    architecture = registry.get("architecture", {})
    if (
        registry.get("protocol") != geometry.PROTOCOL
        or registry.get("status")
        != "training_only_implementation_available_not_executed"
        or architecture.get("transformer_frozen") is not True
        or architecture.get("contextual_classifier") is not False
        or architecture.get("external_router") is not False
        or architecture.get("runtime_gate") is not False
    ):
        raise RuntimeError("V2 registry architecture/status mismatch")
    locked = locked_registry_values(args)
    for section, values in locked.items():
        registered = registry.get(section)
        if not isinstance(registered, Mapping):
            raise RuntimeError(f"V2 registry lacks {section}")
        for key, expected in values.items():
            if registered.get(key) != expected:
                raise RuntimeError(
                    f"V2 argument differs from registry: {section}.{key} "
                    f"({registered.get(key)!r} != {expected!r})"
                )


def load_partition(
    protocol_dir: Path,
    manifest: Mapping[str, Any],
    role: str,
    *,
    permitted: bool,
) -> List[Mapping[str, Any]]:
    if not permitted:
        raise RuntimeError(f"attempted to open {role} before candidate selection")
    filename = manifest["files"][role]
    path = protocol_dir / filename
    if sha256_file(path) != manifest["file_sha256"][role]:
        raise RuntimeError(f"V2 partition hash mismatch: {role}")
    records = json.loads(path.read_text(encoding="utf-8"))
    split_builder.assert_direct_only(records, role=role)
    expected_ids = [int(value) for value in manifest["case_ids"][role]]
    if [int(record["case_id"]) for record in records] != expected_ids:
        raise RuntimeError(f"V2 partition binding mismatch: {role}")
    return records


def plain_cases(
    prompts: Sequence[str], *, case_offset: int
) -> List[canonical.SensitivePredictionCase]:
    return [
        canonical.SensitivePredictionCase(
            case_id=case_offset + index,
            record_position=index,
            token_index=0,
            prompt=str(prompt),
            target_text="",
        )
        for index, prompt in enumerate(prompts)
    ]


def corpus_sentence_pool(documents: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    pool: List[str] = []
    for document in documents:
        for raw in str(document).replace("\n", " ").split("."):
            value = " ".join(raw.split()).strip()
            if not 5 <= len(value.split()) <= 40:
                continue
            value += "."
            if value not in seen:
                seen.add(value)
                pool.append(value)
    return sorted(pool)


def corpus_partitions(
    documents: Sequence[str],
    *,
    fit: int,
    development: int,
    certification: int,
    seed: int,
) -> Dict[str, List[str]]:
    pool = corpus_sentence_pool(documents)
    need = fit + development + certification
    if len(pool) < need:
        raise RuntimeError(f"Wikipedia slice yielded {len(pool)} prompts, need {need}")
    rng = random.Random(int(seed) + 6211)
    chosen = rng.sample(pool, need)
    return {
        "fit": chosen[:fit],
        "development": chosen[fit : fit + development],
        "certification": chosen[fit + development :],
    }


def same_subject_prompts(
    forget: Sequence[Mapping[str, Any]], *, role: str, count: int
) -> List[str]:
    relation_ids = sorted(relation_prompts.RELATION_NOUN_PHRASES)
    prompts: List[str] = []
    scaffold = PROTECTION_SCAFFOLDS[role]
    role_shift = {"fit": 3, "development": 11, "certification": 19}[role]
    for record in forget:
        rr = record["requested_rewrite"]
        own_relation = str(rr["relation_id"])
        alternatives = [value for value in relation_ids if value != own_relation]
        start = (int(record["case_id"]) + role_shift) % len(alternatives)
        for offset in range(count):
            relation_id = alternatives[(start + offset) % len(alternatives)]
            prompts.append(
                scaffold.format(
                    subject=str(rr["subject"]),
                    relation=relation_prompts.RELATION_NOUN_PHRASES[relation_id],
                )
            )
    return prompts


def protection_cases(
    records: Sequence[Mapping[str, Any]],
    corpus_prompts: Sequence[str],
    forget: Sequence[Mapping[str, Any]],
    *,
    role: str,
    tok: Any,
    llama_like: bool,
    same_subject_count: int,
) -> List[canonical.SensitivePredictionCase]:
    result = canonical.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    extras = list(corpus_prompts) + same_subject_prompts(
        forget, role=role, count=same_subject_count
    )
    result.extend(plain_cases(extras, case_offset=10_000_000 + len(result)))
    prompts = [case.prompt for case in result]
    if len(prompts) != len(set(prompts)):
        unique: Dict[str, canonical.SensitivePredictionCase] = {}
        for case in result:
            unique.setdefault(case.prompt, case)
        result = list(unique.values())
    return result


@torch.no_grad()
def cache_protection(
    model: torch.nn.Module,
    tok: Any,
    cases: List[canonical.SensitivePredictionCase],
    device: torch.device,
    *,
    batch_size: int,
    topk: int,
) -> ProtectionCache:
    ids: List[torch.Tensor] = []
    topk_logs: List[torch.Tensor] = []
    top1_ids: List[torch.Tensor] = []
    top1_logs: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        logits = canonical.forward_last_logits(
            model, tok, cases[start : start + batch_size], device
        ).float()
        values, local_ids = torch.topk(logits, k=topk, dim=1)
        normalized = F.log_softmax(values, dim=1)
        full_log_probs = F.log_softmax(logits, dim=1)
        local_top1 = logits.argmax(dim=1)
        local_top1_log = full_log_probs.gather(1, local_top1[:, None]).squeeze(1)
        ids.append(local_ids.cpu())
        topk_logs.append(normalized.cpu())
        top1_ids.append(local_top1.cpu())
        top1_logs.append(local_top1_log.cpu())
    return ProtectionCache(
        cases=cases,
        topk_ids=torch.cat(ids),
        base_topk_log_probs=torch.cat(topk_logs),
        base_top1_ids=torch.cat(top1_ids),
        base_top1_log_probs=torch.cat(top1_logs),
    )


def cache_batch(cache: ProtectionCache, indices: Sequence[int], device: torch.device):
    ids = torch.tensor(list(indices), dtype=torch.long)
    return {
        "cases": [cache.cases[index] for index in indices],
        "topk_ids": cache.topk_ids.index_select(0, ids).to(device),
        "base_topk_log_probs": cache.base_topk_log_probs.index_select(0, ids).to(
            device
        ),
        "base_top1_ids": cache.base_top1_ids.index_select(0, ids).to(device),
        "base_top1_log_probs": cache.base_top1_log_probs.index_select(0, ids).to(
            device
        ),
    }


def compress_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[canonical.SensitivePredictionCase],
    device: torch.device,
    *,
    batch_size: int,
    rows: int,
) -> torch.Tensor:
    hidden_size = int(model.get_input_embeddings().weight.shape[1])
    sketch = torch.zeros((rows, hidden_size), dtype=torch.float32, device=device)
    counts = torch.zeros(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        for start in range(0, len(cases), batch_size):
            batch = list(cases[start : start + batch_size])
            hidden = canonical.forward_last_hidden(
                model, tok, batch, device, batch_size=len(batch)
            )
            for local, vector in enumerate(hidden):
                index = start + local
                bucket = (index * 131 + 17) % rows
                sign = -1.0 if ((index * 1103515245 + 12345) >> 8) & 1 else 1.0
                sketch[bucket].add_(vector * sign)
                counts[bucket] += 1.0
    sketch.div_(counts.clamp_min(1.0).sqrt()[:, None])
    return sketch[counts > 0]


def build_input_bases(
    model: torch.nn.Module,
    tok: Any,
    cache: ProtectionCache,
    device: torch.device,
    *,
    input_delta: canonical.SelectedRowDelta,
    output_delta: canonical.SelectedRowDelta,
    sketches: int,
    batch_size: int,
    topk: int,
    max_rank: int,
) -> tuple[List[torch.Tensor], Dict[str, Any]]:
    if input_delta.raw_delta is None:
        raise RuntimeError("V2 input delta must be unrestricted before projection")
    per_row: List[List[torch.Tensor]] = [[] for _ in range(input_delta.n_rows)]
    n = len(cache.cases)
    for sketch_index in range(sketches):
        model.zero_grad(set_to_none=True)
        input_delta.zero_grad(set_to_none=True)
        output_delta.zero_grad(set_to_none=True)
        start = (sketch_index * batch_size) % n
        indices = [(start + offset) % n for offset in range(batch_size)]
        batch = cache_batch(cache, indices, device)
        logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
        column = sketch_index % topk
        probe_ids = batch["topk_ids"][:, column]
        selected = logits.float().gather(1, probe_ids[:, None]).squeeze(1)
        signs = torch.tensor(
            [1.0 if ((sketch_index + value) % 2 == 0) else -1.0 for value in indices],
            dtype=torch.float32,
            device=device,
        )
        (selected * signs).mean().backward()
        gradient = input_delta.raw_delta.grad
        if gradient is None:
            raise RuntimeError("protected sketch produced no input gradient")
        gradient_cpu = gradient.detach().float().cpu()
        for row_index, vector in enumerate(gradient_cpu):
            if float(vector.norm().item()) > 1e-12:
                per_row[row_index].append(vector.clone())
    bases: List[torch.Tensor] = []
    ranks: List[int] = []
    observations: List[int] = []
    hidden = input_delta.hidden_size
    for vectors in per_row:
        observations.append(len(vectors))
        if vectors:
            basis = canonical.orthonormal_row_basis(
                torch.stack(vectors), max_rank=max_rank
            )
        else:
            basis = torch.empty((0, hidden), dtype=torch.float32)
        bases.append(basis.to(device))
        ranks.append(int(basis.shape[0]))
    input_delta.zero_grad(set_to_none=True)
    output_delta.zero_grad(set_to_none=True)
    return bases, {
        "sketches": sketches,
        "rows": len(bases),
        "rows_observed_in_protection": sum(value > 0 for value in observations),
        "rows_with_empty_basis": sum(value == 0 for value in observations),
        "observation_min": min(observations),
        "observation_max": max(observations),
        "rank_min": min(ranks),
        "rank_median": sorted(ranks)[len(ranks) // 2],
        "rank_max": max(ranks),
    }


def answer_nlls(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    device: torch.device,
    *,
    llama_like: bool,
) -> torch.Tensor:
    if len(prompts) != len(answers) or not prompts:
        raise ValueError("answer NLL batch must be non-empty and aligned")
    prefix_tokens = tok(list(prompts), padding=True, return_tensors="pt")
    prefix_lens = prefix_tokens["attention_mask"].sum(dim=1).tolist()
    full_texts = [f"{prompt} {answer}" for prompt, answer in zip(prompts, answers)]
    target_ids: List[List[int]] = []
    for answer in answers:
        ids = canonical.flat_ids(tok, " " + str(answer))
        if llama_like:
            ids = ids[1:]
        if not ids:
            raise ValueError(f"answer tokenized to no evaluated tokens: {answer!r}")
        target_ids.append(ids)
    encoded = tok(full_texts, padding=True, return_tensors="pt").to(device)
    logits = model(**encoded, use_cache=False).logits
    if llama_like:
        logits = logits[:, 1:, :]
    losses: List[torch.Tensor] = []
    for row, (ids, prefix_len) in enumerate(zip(target_ids, prefix_lens)):
        adjusted_prefix = int(prefix_len) - 1 if llama_like else int(prefix_len)
        token_losses: List[torch.Tensor] = []
        for offset, token_id in enumerate(ids):
            position = adjusted_prefix + offset - 1
            token_losses.append(
                -F.log_softmax(logits[row, position].float(), dim=0)[token_id]
            )
        losses.append(torch.stack(token_losses).mean())
    return torch.stack(losses)


def records_margin_tensor(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
) -> torch.Tensor:
    prompts = [
        str(record["requested_rewrite"]["prompt"]).format(
            str(record["requested_rewrite"]["subject"])
        )
        for record in records
    ]
    true_answers = [
        str(record["requested_rewrite"]["target_true"]["str"]) for record in records
    ]
    new_answers = [
        str(record["requested_rewrite"]["target_new"]["str"]) for record in records
    ]
    true_nll = answer_nlls(
        model, tok, prompts, true_answers, device, llama_like=llama_like
    )
    new_nll = answer_nlls(
        model, tok, prompts, new_answers, device, llama_like=llama_like
    )
    return true_nll - new_nll


@torch.no_grad()
def margin_report(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
    minimum: float,
) -> Dict[str, Any]:
    values: List[torch.Tensor] = []
    for start in range(0, len(records), batch_size):
        values.append(
            records_margin_tensor(
                model,
                tok,
                records[start : start + batch_size],
                device,
                llama_like=llama_like,
            ).detach()
        )
    margins = torch.cat(values).float().cpu()
    return {
        "records": len(records),
        "failures": int((margins < float(minimum)).sum().item()),
        "minimum_margin": float(margins.min().item()),
        "median_margin": float(margins.median().item()),
        "criterion": {"minimum_margin": float(minimum), "failures": 0},
        "passed": bool(torch.all(margins >= float(minimum)).item()),
    }


def evaluate_protection(
    model: torch.nn.Module,
    tok: Any,
    cache: ProtectionCache,
    device: torch.device,
    *,
    batch_size: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    kl_values: List[torch.Tensor] = []
    drift_values: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(cache.cases), batch_size):
            indices = list(range(start, min(start + batch_size, len(cache.cases))))
            batch = cache_batch(cache, indices, device)
            logits = canonical.forward_last_logits(model, tok, batch["cases"], device)
            kl, drift = geometry.protection_loss(
                logits, **{key: value for key, value in batch.items() if key != "cases"}
            )
            kl_values.append(kl.cpu())
            drift_values.append(drift.cpu())
    return geometry.protection_report(
        torch.cat(kl_values),
        torch.cat(drift_values),
        kl_mean_max=args.protected_kl_mean_max,
        kl_absolute_max=args.protected_kl_max,
        top1_abs_max=args.protected_top1_drift_max,
    )


def feasible_gradient_report(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    input_delta: canonical.SelectedRowDelta,
    output_delta: canonical.SelectedRowDelta,
    input_bases: Sequence[torch.Tensor],
    output_basis: torch.Tensor,
    minimum: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        input_delta.zero_grad(set_to_none=True)
        output_delta.zero_grad(set_to_none=True)
        margin = records_margin_tensor(
            model, tok, [record], device, llama_like=llama_like
        )[0]
        margin.backward()
        if input_delta.raw_delta is None or output_delta.raw_delta is None:
            raise RuntimeError("V2 feasibility requires unrestricted endpoint deltas")
        if input_delta.raw_delta.grad is None or output_delta.raw_delta.grad is None:
            raise RuntimeError("V2 feasibility produced a missing endpoint gradient")
        input_norm, output_norm, combined = geometry.projected_gradient_norms(
            input_delta.raw_delta.grad,
            output_delta.raw_delta.grad,
            input_bases=input_bases,
            output_basis=output_basis,
        )
        rows.append(
            {
                "case_id": int(record["case_id"]),
                "base_margin": float(margin.detach().cpu()),
                "projected_input_gradient_norm": input_norm,
                "projected_output_gradient_norm": output_norm,
                "combined_projected_gradient_norm": combined,
                "passed": combined > float(minimum),
            }
        )
    input_delta.zero_grad(set_to_none=True)
    output_delta.zero_grad(set_to_none=True)
    return {
        "minimum_projected_gradient_norm": float(minimum),
        "passed_records": sum(bool(row["passed"]) for row in rows),
        "total_records": len(rows),
        "per_record": rows,
        "passed": all(bool(row["passed"]) for row in rows),
    }


def next_indices(rng: random.Random, count: int, batch_size: int) -> List[int]:
    if count <= 0:
        raise ValueError("cannot sample from an empty collection")
    if count >= batch_size:
        return rng.sample(range(count), batch_size)
    return [rng.randrange(count) for _ in range(batch_size)]


def completion_failure(output: Path, *, phase: str, detail: Mapping[str, Any]) -> None:
    write_json(
        output / "method" / "completion.json",
        {
            "schema_version": 1,
            "protocol": geometry.PROTOCOL,
            "passed": False,
            "phase": phase,
            "detail": dict(detail),
            "candidate_saved": False,
            "official_evaluation_prompts_seen": 0,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    gagd.set_seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_path = Path(args.experiment_registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry, args)
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest_path = protocol_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != split_builder.PROTOCOL:
        raise RuntimeError("V2 split manifest protocol mismatch")
    if manifest.get("serialized_prompt_counts", {}).get("official_retain") != 0:
        raise RuntimeError("V2 split exposes official retain prompt text")
    forget = load_partition(protocol_dir, manifest, "forget", permitted=True)
    protection_fit = load_partition(
        protocol_dir, manifest, "protection_fit", permitted=True
    )
    protection_development = load_partition(
        protocol_dir, manifest, "protection_development", permitted=True
    )

    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    device = gagd.first_device(model)
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    llama_like = canonical.is_llama_like(model, tok)
    transformer_before = transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )

    endpoint_rows = geometry.select_endpoint_rows(forget, tok, llama_like=llama_like)
    overlap = geometry.endpoint_overlap_report(endpoint_rows)
    write_json(output / "method" / "endpoint_overlap_manifest.json", overlap)
    if (
        not overlap["one_delta_per_physical_input_row"]
        or not overlap["one_delta_per_physical_output_row"]
    ):
        raise RuntimeError("V2 endpoint row coherence failed")
    input_ids_tensor = torch.tensor(
        endpoint_rows.input_ids, dtype=torch.long, device=device
    )
    output_ids_tensor = torch.tensor(
        endpoint_rows.output_ids, dtype=torch.long, device=device
    )
    base_input_rows = (
        input_layer.weight.index_select(0, input_ids_tensor).detach().float()
    )
    base_output_rows = (
        output_layer.weight.index_select(0, output_ids_tensor).detach().float()
    )
    hidden_size = int(base_input_rows.shape[1])
    input_delta = canonical.SelectedRowDelta(
        len(endpoint_rows.input_ids), hidden_size, direction_basis=None, device=device
    )
    output_delta = canonical.SelectedRowDelta(
        len(endpoint_rows.output_ids), hidden_size, direction_basis=None, device=device
    )
    input_handle = endpoint_hooks.register_input_embedding_delta_hook(
        input_layer, endpoint_rows.input_ids, input_delta.effective_delta
    )
    output_handle = canonical.register_output_delta_hook(
        output_layer, endpoint_rows.output_ids, output_delta.effective_delta
    )

    documents = subject_stage1.load_frequency_documents(
        args.wikidata_dir, args.frequency_doc_start, args.frequency_docs
    )
    if len(documents) < args.frequency_docs:
        raise RuntimeError(
            f"V2 loaded {len(documents)} Wikipedia documents, expected {args.frequency_docs}"
        )
    corpus = corpus_partitions(
        documents,
        fit=args.corpus_fit_prompts,
        development=args.corpus_development_prompts,
        certification=args.corpus_certification_prompts,
        seed=args.seed,
    )
    frequency_counts = subject_stage1.token_frequency_counts(
        tok, documents, int(input_layer.weight.shape[0])
    )
    selected_counts = frequency_counts.index_select(
        0, torch.tensor(endpoint_rows.input_ids, dtype=torch.long)
    )
    input_caps = geometry.frequency_adjusted_caps(
        base_input_rows.cpu(),
        selected_counts,
        relative_cap=args.input_relative_cap,
        alpha=args.input_frequency_alpha,
    ).to(device)
    output_caps = geometry.relative_caps(
        base_output_rows, relative_cap=args.output_relative_cap
    ).to(device)

    fit_cases = protection_cases(
        protection_fit,
        corpus["fit"],
        forget,
        role="fit",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    development_cases = protection_cases(
        protection_development,
        corpus["development"],
        forget,
        role="development",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    print(
        f"Stage 1: cache {len(fit_cases)} fit and {len(development_cases)} development protection prompts"
    )
    fit_cache = cache_protection(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    development_cache = cache_protection(
        model,
        tok,
        development_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )

    print("Stage 1a: construct protected input-Jacobian and output-state subspaces")
    input_bases, input_basis_report = build_input_bases(
        model,
        tok,
        fit_cache,
        device,
        input_delta=input_delta,
        output_delta=output_delta,
        sketches=args.input_jacobian_sketches,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
        max_rank=args.input_rank_cap,
    )
    hidden_sketch = compress_hidden_states(
        model,
        tok,
        fit_cases,
        device,
        batch_size=args.capture_batch_size,
        rows=args.output_rank_cap,
    )
    output_basis = geometry.common_basis(
        hidden_sketch, max_rank=args.output_rank_cap
    ).to(device)
    basis_report = {
        "input": input_basis_report,
        "output": {
            "protected_prompt_states": len(fit_cases),
            "sketch_rows": int(hidden_sketch.shape[0]),
            "rank": int(output_basis.shape[0]),
            "rank_cap": args.output_rank_cap,
        },
    }
    write_json(output / "method" / "protected_subspace_report.json", basis_report)

    print(
        "Stage 1b: verify one preservation-safe first-order direction per forget fact"
    )
    feasibility = feasible_gradient_report(
        model,
        tok,
        forget,
        device,
        llama_like=llama_like,
        input_delta=input_delta,
        output_delta=output_delta,
        input_bases=input_bases,
        output_basis=output_basis,
        minimum=args.minimum_projected_gradient_norm,
    )
    write_json(output / "method" / "safe_direction_feasibility.json", feasibility)
    if not feasibility["passed"]:
        completion_failure(
            output, phase="safe_direction_feasibility", detail=feasibility
        )
        raise RuntimeError("V2 has a forget fact with no protected safe direction")

    synthetic_prefixes = synthetic.corpus_context_prefixes(
        documents, count=256, seed=args.seed + 71
    )
    synthetic_records = synthetic.build_synthetic_records(
        forget,
        count=args.synthetic_paraphrases,
        context_prefixes=synthetic_prefixes,
    )
    training_records = list(forget) + list(synthetic_records)
    if input_delta.raw_delta is None or output_delta.raw_delta is None:
        raise RuntimeError("V2 endpoint deltas unexpectedly parameterized")
    with torch.no_grad():
        input_delta.raw_delta.zero_()
        output_delta.raw_delta.zero_()
    optimizer = torch.optim.AdamW(
        [
            {"params": [input_delta.raw_delta], "lr": args.embedding_lr},
            {"params": [output_delta.raw_delta], "lr": args.lm_head_lr},
        ],
        weight_decay=0.0,
    )
    rng = random.Random(args.seed + 9901)
    development_reports: List[Dict[str, Any]] = []
    states: Dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    print("Stage 2: jointly train sparse embedding and LM-head rows from exact zero")
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        forget_indices = next_indices(
            rng, len(training_records), args.forget_batch_size
        )
        forget_batch = [training_records[index] for index in forget_indices]
        margins = records_margin_tensor(
            model, tok, forget_batch, device, llama_like=llama_like
        )
        forget_loss = F.relu(args.forget_margin_target - margins).square().mean()
        protect_indices = next_indices(
            rng, len(fit_cache.cases), args.protection_batch_size
        )
        protect_batch = cache_batch(fit_cache, protect_indices, device)
        protect_logits = canonical.forward_last_logits(
            model, tok, protect_batch["cases"], device
        )
        kl, drift = geometry.protection_loss(
            protect_logits,
            **{key: value for key, value in protect_batch.items() if key != "cases"},
        )
        l2 = (
            input_delta.raw_delta.square().mean()
            + output_delta.raw_delta.square().mean()
        )
        loss = (
            args.forget_margin_weight * forget_loss
            + args.protection_kl_weight * kl.mean()
            + args.protection_top1_weight * drift.square().mean()
            + args.delta_l2_weight * l2
        )
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite V2 loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [input_delta.raw_delta, output_delta.raw_delta], args.gradient_clip
        )
        optimizer.step()
        geometry.project_rowwise_(input_delta.raw_delta, input_bases)
        geometry.project_common_(output_delta.raw_delta, output_basis)
        geometry.apply_row_caps_(input_delta.raw_delta, input_caps)
        geometry.apply_row_caps_(output_delta.raw_delta, output_caps)

        if step % args.check_every == 0 or step == 1:
            direct = margin_report(
                model,
                tok,
                forget,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            synthetic_report = margin_report(
                model,
                tok,
                synthetic_records,
                device,
                llama_like=llama_like,
                batch_size=args.capture_batch_size,
                minimum=args.minimum_forget_margin,
            )
            development = evaluate_protection(
                model,
                tok,
                development_cache,
                device,
                batch_size=args.capture_batch_size,
                args=args,
            )
            input_cap_report = geometry.cap_report(input_delta.raw_delta, input_caps)
            output_cap_report = geometry.cap_report(output_delta.raw_delta, output_caps)
            total_norm = float(
                (
                    input_delta.raw_delta.square().sum()
                    + output_delta.raw_delta.square().sum()
                )
                .sqrt()
                .detach()
                .cpu()
            )
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "forget_loss": float(forget_loss.detach().cpu()),
                "direct": direct,
                "synthetic": synthetic_report,
                "protection_development": development,
                "input_caps": input_cap_report,
                "output_caps": output_cap_report,
                "total_delta_norm": total_norm,
            }
            row["passed"] = bool(
                direct["passed"]
                and synthetic_report["passed"]
                and development["passed"]
                and input_cap_report["passed"]
                and output_cap_report["passed"]
            )
            development_reports.append(row)
            states[step] = (
                input_delta.raw_delta.detach().cpu().clone(),
                output_delta.raw_delta.detach().cpu().clone(),
            )
            print(
                f"  step {step:4d}: loss {row['loss']:.5f}, "
                f"direct fail {direct['failures']}, synth fail {synthetic_report['failures']}, "
                f"dev KL max {development['topk_kl_max']:.6f}, "
                f"dev top1 {development['top1_logprob_abs_max']:.6f}, pass={row['passed']}"
            )
    del optimizer
    write_json(
        output / "method" / "development_training.json",
        {"reports": development_reports},
    )
    selected_step = geometry.select_development_candidate(development_reports)
    if selected_step is None:
        completion_failure(
            output,
            phase="development_selection",
            detail={"reports": development_reports},
        )
        raise RuntimeError("V2 produced no development-passing checkpoint")
    selected_input, selected_output = states[selected_step]
    with torch.no_grad():
        input_delta.raw_delta.copy_(selected_input.to(device))
        output_delta.raw_delta.copy_(selected_output.to(device))

    print(
        f"Stage 3: open disjoint certification once for selected step {selected_step}"
    )
    protection_certification = load_partition(
        protocol_dir, manifest, "protection_certification", permitted=True
    )
    certification_cases = protection_cases(
        protection_certification,
        corpus["certification"],
        forget,
        role="certification",
        tok=tok,
        llama_like=llama_like,
        same_subject_count=args.same_subject_prompts,
    )
    with torch.no_grad():
        input_delta.raw_delta.zero_()
        output_delta.raw_delta.zero_()
    certification_cache = cache_protection(
        model,
        tok,
        certification_cases,
        device,
        batch_size=args.capture_batch_size,
        topk=args.protection_topk,
    )
    with torch.no_grad():
        input_delta.raw_delta.copy_(selected_input.to(device))
        output_delta.raw_delta.copy_(selected_output.to(device))

    def complete_certificate() -> Dict[str, Any]:
        direct = margin_report(
            model,
            tok,
            forget,
            device,
            llama_like=llama_like,
            batch_size=args.capture_batch_size,
            minimum=args.minimum_forget_margin,
        )
        synth = margin_report(
            model,
            tok,
            synthetic_records,
            device,
            llama_like=llama_like,
            batch_size=args.capture_batch_size,
            minimum=args.minimum_forget_margin,
        )
        protection = evaluate_protection(
            model,
            tok,
            certification_cache,
            device,
            batch_size=args.capture_batch_size,
            args=args,
        )
        input_cap = geometry.cap_report(input_delta.raw_delta, input_caps)
        output_cap = geometry.cap_report(output_delta.raw_delta, output_caps)
        result = {
            "direct": direct,
            "synthetic": synth,
            "protection": protection,
            "input_caps": input_cap,
            "output_caps": output_cap,
        }
        result["passed"] = bool(
            direct["passed"]
            and synth["passed"]
            and protection["passed"]
            and input_cap["passed"]
            and output_cap["passed"]
        )
        return result

    pre_materialization = complete_certificate()
    write_json(
        output / "method" / "pre_materialization_certificate.json",
        pre_materialization,
    )
    if not pre_materialization["passed"]:
        completion_failure(output, phase="certification", detail=pre_materialization)
        raise RuntimeError("V2 selected candidate failed one-shot certification")

    input_handle.remove()
    output_handle.remove()
    endpoint_hooks.materialize_input_delta(
        input_layer, endpoint_rows.input_ids, selected_input
    )
    canonical.materialize_output_delta(
        output_layer, endpoint_rows.output_ids, selected_output
    )
    post_materialization = complete_certificate()
    transformer_after = transformer_sha256(
        model, excluded_parameters=[input_layer.weight, output_layer.weight]
    )
    acceptance_match = {
        "direct_failures_match": pre_materialization["direct"]["failures"]
        == post_materialization["direct"]["failures"],
        "synthetic_failures_match": pre_materialization["synthetic"]["failures"]
        == post_materialization["synthetic"]["failures"],
        "protection_acceptance_match": pre_materialization["protection"]["passed"]
        == post_materialization["protection"]["passed"],
        "complete_acceptance_match": pre_materialization["passed"]
        == post_materialization["passed"],
    }
    acceptance_match["passed"] = all(acceptance_match.values())
    post_report = {
        **post_materialization,
        "pre_post_acceptance": acceptance_match,
        "transformer_sha256_before": transformer_before,
        "transformer_sha256_after": transformer_after,
        "transformer_bit_identical": transformer_before == transformer_after,
    }
    post_report["passed"] = bool(
        post_materialization["passed"]
        and acceptance_match["passed"]
        and transformer_before == transformer_after
    )
    write_json(output / "method" / "post_materialization_certificate.json", post_report)
    if not post_report["passed"]:
        completion_failure(output, phase="native_materialization", detail=post_report)
        raise RuntimeError("V2 native materialization failed its locked certificate")

    candidate_path = output / "method" / "v2_candidate_sparse_rows.pt"
    torch.save(
        {
            "schema_version": 1,
            "protocol": geometry.PROTOCOL,
            "base_model_path": str(args.model_path),
            "input_row_ids": endpoint_rows.input_ids,
            "input_rows": input_layer.weight.index_select(0, input_ids_tensor)
            .detach()
            .cpu(),
            "output_row_ids": endpoint_rows.output_ids,
            "output_rows": output_layer.weight.index_select(0, output_ids_tensor)
            .detach()
            .cpu(),
            "selected_step": selected_step,
            "split_manifest_sha256": sha256_file(manifest_path),
            "registry_sha256": sha256_file(registry_path),
            "transformer_sha256": transformer_after,
        },
        candidate_path,
    )
    completion = {
        "schema_version": 1,
        "protocol": geometry.PROTOCOL,
        "passed": True,
        "architecture": {
            "selected_input_embedding_rows": len(endpoint_rows.input_ids),
            "selected_lm_head_rows": len(endpoint_rows.output_ids),
            "transformer_frozen": True,
            "contextual_classifier": False,
            "runtime_gate": False,
            "external_sidecar": False,
        },
        "selected_development_step": selected_step,
        "safe_direction_feasibility_passed": True,
        "certification_passed": True,
        "native_materialization_passed": True,
        "transformer_bit_identical": True,
        "candidate_saved": True,
        "candidate_path": str(candidate_path),
        "candidate_sha256": sha256_file(candidate_path),
        "eligible_for_separate_official_evaluation": True,
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "claim_scope": "internal_sparse_counterfactual_rewiring_candidate_not_yet_latent_erasure",
    }
    write_json(output / "method" / "completion.json", completion)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
