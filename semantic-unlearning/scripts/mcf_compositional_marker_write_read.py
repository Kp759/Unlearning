#!/usr/bin/env python3
"""Router-free context-composed sparse embedding writer for MCF.

Architecture (and the only trainable/model-changing objects)::

    sparse subject embedding-row deltas
        -> completely frozen Transformer
        -> context-composed marker v_i
        -> diagnostic record-level marker reader
        -> bounded writer-residual LM-head readers
        -> exact per-row factorization Delta W_y = -beta_y q_y

There is no router, exact-subject runtime, sidecar, logit bias, adapter, LoRA,
or trained Transformer parameter.  Each sparse output row is constrained to a
small basis derived from its paired writer-induced residuals and to a hard
relative norm cap.  The scientific hypothesis is that a frozen
Transformer can decode a globally shared sparse embedding perturbation as a
record-specific marker only when the complete subject/relation composition is
present.

Data firewall
-------------
This program requires the direct-only locked MCF split.  Its positive contexts
are the direct prompt, hand-authored relation templates, corpus-prefix variants,
and (optionally) an independently generated surrogate artifact whose receipt
declares zero official-probe access.  Its negatives use only other locked
forget subjects and corruptions of those subjects.  Official CounterFact
paraphrases, neighborhoods, retain records, and the official PPL prefix are
never loaded.  Final evaluation is a separate process against the original
source after the checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import build_mcf_sure_target_aware_direct_split as locked_split
import mcf_compositional_marker_core as compositional
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_surrogate_answer_guard as answer_guard
import mcf_synthetic_paraphrase_templates as synthetic
import sure_canonical_core as canonical


METHOD = "Context-Composed Sparse Embedding Writing"
PROTOCOL = compositional.PROTOCOL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    # Training-safe positive contexts.
    p.add_argument("--synthetic-paraphrases-per-record", type=int, default=6)
    p.add_argument(
        "--surrogate-prompts-path",
        default="",
        help=(
            "Optional semantically generated surrogate artifact. It must match "
            "the locked records and declare zero access to all official probes."
        ),
    )
    p.add_argument(
        "--require-semantic-surrogates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail unless --surrogate-prompts-path is supplied and semantically validated.",
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--frequency-doc-start", type=int, default=20)
    p.add_argument("--frequency-docs", type=int, default=5000)
    p.add_argument("--corpus-context-prefixes", type=int, default=256)
    p.add_argument("--max-subject-token-frequency", type=int, default=10**9)

    # Whole-subject / collision negatives.
    p.add_argument("--max-shared-subword-negatives", type=int, default=8)
    p.add_argument("--max-leave-one-out-negatives", type=int, default=8)
    p.add_argument("--max-fragment-negatives", type=int, default=4)
    p.add_argument("--max-unrelated-negatives", type=int, default=4)

    # Contrastive multi-context reachability.
    p.add_argument("--reach-probes", type=int, default=12)
    p.add_argument("--reach-sigma", type=float, default=0.02)
    p.add_argument("--reach-positive-contexts", type=int, default=4)
    p.add_argument("--reach-negative-contexts", type=int, default=8)
    p.add_argument("--marker-max-rank", type=int, default=96)
    p.add_argument("--marker-ridge", type=float, default=1e-4)

    # Stage 1: sparse embedding writer only.
    p.add_argument(
        "--resume-stage1-state",
        default="",
        help=(
            "Optional stage1_writer.pt from the same seed/context protocol. "
            "When supplied with --writer-steps 0, reuse its sparse embedding "
            "delta and markers without retraining."
        ),
    )
    p.add_argument("--writer-steps", type=int, default=1200)
    p.add_argument("--writer-lr", type=float, default=2e-4)
    p.add_argument("--writer-record-batch", type=int, default=4)
    p.add_argument("--positive-context-batch", type=int, default=4)
    p.add_argument("--negative-context-batch", type=int, default=5)
    p.add_argument("--write-alpha", type=float, default=6.0)
    p.add_argument("--lambda-consistency", type=float, default=2.0)
    p.add_argument("--lambda-negative", type=float, default=10.0)
    p.add_argument("--lambda-state", type=float, default=1.0)
    p.add_argument("--lambda-cross", type=float, default=2.0)
    p.add_argument("--lambda-off-axis", type=float, default=0.1)
    p.add_argument("--lambda-kl", type=float, default=0.1)
    p.add_argument("--kl-topk", type=int, default=64)
    p.add_argument("--writer-grad-clip", type=float, default=1.0)
    p.add_argument("--writer-eval-every", type=int, default=50)
    p.add_argument("--row-norm-cap", type=float, default=8.0)
    p.add_argument("--row-norm-cap-frequency-alpha", type=float, default=0.15)
    p.add_argument("--writer-amplitude-min-frac", type=float, default=0.75)
    p.add_argument("--writer-marker-kappa-max", type=float, default=0.10)

    # Distributional q reader.
    p.add_argument("--reader-ridge", type=float, default=0.05)
    p.add_argument("--reader-anchor-weight", type=float, default=0.05)
    p.add_argument("--reader-consistency-weight", type=float, default=2.0)
    p.add_argument("--reader-negative-weight", type=float, default=2.0)
    p.add_argument("--reader-refine-steps", type=int, default=600)
    p.add_argument("--reader-refine-lr", type=float, default=0.03)
    p.add_argument("--reader-positive-floor", type=float, default=0.02)
    p.add_argument("--reader-cross-positive-cap", type=int, default=256)

    # Pre-Stage-2 portability gate.
    p.add_argument("--kappa-train-max", type=float, default=0.10)
    p.add_argument("--portability-min", type=float, default=0.50)
    p.add_argument(
        "--cos-marker-reader-min", type=float, default=0.0,
        help=(
            "Diagnostic lower bound after negative-nullspace projection. The first "
            "run showed that forcing cos(v,q) near one preserves base-state leakage; "
            "zero allows the distributional reader to remove that geometry."
        ),
    )
    p.add_argument("--gate-pass-frac", type=float, default=1.0)
    p.add_argument(
        "--gate-policy", choices=("strict", "report"), default="strict",
        help="strict refuses Stage 2 if the predeclared reader gate fails.",
    )

    # Stage 2: joint sparse LM-head rows, each exactly factorized as -beta_y q_y.
    p.add_argument("--forget-margin", type=float, default=1.0)
    p.add_argument("--stage2-steps", type=int, default=500)
    p.add_argument("--stage2-lr", type=float, default=0.05)
    p.add_argument("--stage2-batch-size", type=int, default=8)
    p.add_argument("--stage2-margin-weight", type=float, default=100.0)
    p.add_argument("--stage2-negative-weight", type=float, default=1e-2)
    p.add_argument(
        "--stage2-base-positive-weight",
        type=float,
        default=1.0,
        help=(
            "Penalty on output-row shifts at the same positive teacher-forced "
            "states with the embedding writer removed. This makes the sparse "
            "output reader depend causally on the learned input marker."
        ),
    )
    p.add_argument("--stage2-beta-l2", type=float, default=1e-3)
    p.add_argument(
        "--stage2-reference-nll-weight",
        type=float,
        default=10.0,
        help=(
            "Penalize absolute target_new NLL drift relative to the writer-only "
            "model. This protects the reference answer without incorrectly "
            "declaring its same-prompt predictor state out of scope."
        ),
    )
    p.add_argument("--stage2-reference-nll-tolerance", type=float, default=0.05)
    p.add_argument("--stage2-check-every", type=int, default=25)
    p.add_argument(
        "--stage2-protection-rank",
        type=int,
        default=512,
        help=(
            "Project every learned output row away from the leading span of "
            "writer-off positives, hard negatives, and disjoint corpus states; "
            "0 disables the hard projection."
        ),
    )
    p.add_argument("--stage2-protection-states", type=int, default=1024)
    p.add_argument("--stage2-corpus-protection-prompts", type=int, default=256)
    p.add_argument(
        "--stage2-residual-rank",
        type=int,
        default=4,
        help="Maximum writer-residual basis rank for each sensitive output row.",
    )
    p.add_argument(
        "--stage2-row-negative-rank",
        type=int,
        default=32,
        help="Maximum row-semantic hard-negative basis rank per output token.",
    )
    p.add_argument(
        "--stage2-row-norm-caps",
        default="0.05,0.10,0.20,0.40",
        help=(
            "Ascending hard caps on ||Delta W_y||/||W_y||. Stage 2 selects the "
            "smallest cap satisfying every training-only constraint and never "
            "falls back to an unconstrained row."
        ),
    )

    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--save-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(list(argv) if argv is not None else None)

    positive_ints = (
        a.forget_num,
        a.reach_probes,
        a.reach_positive_contexts,
        a.reach_negative_contexts,
        a.marker_max_rank,
        a.writer_record_batch,
        a.positive_context_batch,
        a.negative_context_batch,
        a.kl_topk,
        a.stage2_steps,
        a.stage2_batch_size,
        a.stage2_protection_states,
        a.cache_batch_size,
    )
    if any(int(x) <= 0 for x in positive_ints):
        p.error("counts, ranks, and optimization steps must be positive")
    if int(a.writer_steps) < 0:
        p.error("--writer-steps must be non-negative")
    if int(a.writer_steps) == 0 and not str(a.resume_stage1_state).strip():
        p.error("--writer-steps 0 requires --resume-stage1-state")
    if int(a.synthetic_paraphrases_per_record) < 0 or int(a.reader_refine_steps) < 0:
        p.error("synthetic paraphrase and reader-refinement counts must be non-negative")
    if int(a.stage2_protection_rank) < 0 or int(a.stage2_corpus_protection_prompts) < 0:
        p.error("Stage-2 protection rank/counts must be non-negative")
    if int(a.stage2_residual_rank) <= 0 or int(a.stage2_row_negative_rank) < 0:
        p.error("Stage-2 residual rank must be positive and negative rank non-negative")
    try:
        caps = sorted(
            {
                float(piece.strip())
                for piece in str(a.stage2_row_norm_caps).split(",")
                if piece.strip()
            }
        )
    except ValueError:
        p.error("--stage2-row-norm-caps must be comma-separated numbers")
    if not caps or any(not math.isfinite(value) or value <= 0.0 for value in caps):
        p.error("Stage-2 row-norm caps must be finite and positive")
    a.stage2_row_norm_cap_values = caps
    if int(a.reach_probes) < 4 or int(a.reach_probes) % 2:
        p.error("--reach-probes must be an even integer >= 4")
    if int(a.frequency_doc_start) < 20:
        p.error("--frequency-doc-start must be >=20 to exclude official PPL documents")
    if a.require_semantic_surrogates and not str(a.surrogate_prompts_path).strip():
        p.error("--require-semantic-surrogates needs --surrogate-prompts-path")
    fractions = (
        a.gate_pass_frac,
        a.portability_min,
        a.cos_marker_reader_min,
        a.writer_amplitude_min_frac,
    )
    if any(not 0 <= float(x) <= 1 for x in fractions):
        p.error("gate fractions, portability, and cosine thresholds must be in [0,1]")
    if float(a.kappa_train_max) < 0:
        p.error("--kappa-train-max must be non-negative")
    if float(a.writer_marker_kappa_max) < 0:
        p.error("--writer-marker-kappa-max must be non-negative")
    if min(
        float(a.stage2_margin_weight),
        float(a.stage2_negative_weight),
        float(a.stage2_base_positive_weight),
        float(a.stage2_beta_l2),
        float(a.stage2_reference_nll_weight),
        float(a.stage2_reference_nll_tolerance),
    ) < 0:
        p.error("Stage-2 loss weights must be non-negative")
    return a


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_views(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    views: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        rewrite = record["requested_rewrite"]
        views.append(
            {
                "position": int(position),
                "case_id": int(record.get("case_id", position)),
                "subject": str(rewrite["subject"]),
                "prompt_template": str(rewrite["prompt"]),
                "direct_prompt": str(rewrite["prompt"]).format(str(rewrite["subject"])),
                "answer": str(rewrite["target_true"]["str"]),
                "reference": str(rewrite["target_new"]["str"]),
                "relation_id": str(rewrite.get("relation_id") or ""),
            }
        )
    return views


def load_surrogate_prompts(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    require_semantic: bool,
) -> Tuple[List[List[str]], Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", -1)) != 1:
        raise RuntimeError("unsupported surrogate artifact schema")
    if int(data.get("seed", -1)) != int(seed):
        raise RuntimeError("surrogate artifact seed mismatch")
    if int(data.get("forget_num", -1)) != len(records):
        raise RuntimeError("surrogate artifact forget count mismatch")
    access = data.get("data_access", {})
    forbidden = {
        "official_paraphrase_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "official_PPL_seen": False,
    }
    for key, expected in forbidden.items():
        if access.get(key) != expected:
            raise RuntimeError(f"surrogate artifact violates data firewall: {key}")
    semantic = data.get("semantic_validation", {})
    robust_adapter = str(data.get("protocol", "")).startswith(
        "mcf_direct_only_robust_prompt_adapter_v"
    )
    semantic_enabled = bool(
        semantic.get("enabled", False) or semantic.get("protocol")
    )
    semantic_required = bool(
        semantic.get("required_for_every_surrogate", False) or robust_adapter
    )
    if require_semantic and not semantic_enabled:
        raise RuntimeError("semantic surrogate validation was required but is absent")
    if require_semantic and not semantic_required:
        raise RuntimeError("semantic validation was not required for every surrogate")

    rows = data.get("records")
    if not isinstance(rows, list) or len(rows) != len(records):
        raise RuntimeError("surrogate records do not match the locked forget set")
    prompts_by_record: List[List[str]] = []
    for position, (record, row) in enumerate(zip(records, rows)):
        rewrite = record["requested_rewrite"]
        case_id = int(record.get("case_id", position))
        subject = str(rewrite["subject"])
        direct = str(rewrite["prompt"]).format(subject)
        if int(row.get("case_id", -1)) != case_id:
            raise RuntimeError(f"surrogate case mismatch at position {position}")
        if int(row.get("sampled_position", -1)) != position:
            raise RuntimeError(f"surrogate position mismatch at {position}")
        if compositional.normalized_key(row.get("subject", "")) != compositional.normalized_key(subject):
            raise RuntimeError(f"surrogate subject mismatch at position {position}")
        if compositional.normalized_key(row.get("direct_prompt", "")) != compositional.normalized_key(direct):
            raise RuntimeError(f"surrogate direct-prompt mismatch at position {position}")
        candidates = row.get("surrogate_prompts")
        if not isinstance(candidates, list):
            raise RuntimeError(f"surrogate prompts missing at position {position}")
        if not candidates and row.get("augmentation_status") != "direct_only":
            raise RuntimeError(f"empty surrogate set is not declared direct_only at {position}")
        answers = [str(rewrite["target_true"]["str"]), str(rewrite["target_new"]["str"])]
        clean = compositional.ordered_unique([str(x) for x in candidates])
        for candidate in clean:
            if compositional.normalized_key(subject) not in compositional.normalized_key(candidate):
                raise RuntimeError(f"surrogate dropped the subject at position {position}")
            if answer_guard.introduced_answer_occurrences(candidate, direct, answers):
                raise RuntimeError(f"surrogate introduced an answer at position {position}")
        prompts_by_record.append(clean)
    return prompts_by_record, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "protocol": data.get("protocol"),
        "builder_protocol": data.get("builder_protocol"),
        "semantic_validation": semantic,
        "generator": data.get("generator"),
    }


def prompt_token_ids(tok: Any, prompt: str) -> List[int]:
    value = tok(str(prompt), add_special_tokens=True)["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one prompt")
        value = value[0]
    return [int(x) for x in value]


def answer_token_ids(tok: Any, answer: str) -> List[int]:
    return gagd.token_ids_for_text(tok, gagd.normalize_answer(str(answer)))


def forward_last_hidden_logits(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(prompts), device=device)
        return output.hidden_states[-1][rows, positions, :], output.logits[rows, positions, :]
    finally:
        tok.padding_side = old_side


def forward_last_hidden_only(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    """Differentiable final states without paying for the 128k-row LM head."""
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
        backbone = getattr(model, "model", None)
        if backbone is not None and backbone is not model:
            output = backbone(**encoded, use_cache=False, return_dict=True)
            hidden = output.last_hidden_state
        else:
            output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states[-1]
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(prompts), device=device)
        return hidden[rows, positions, :]
    finally:
        tok.padding_side = old_side


@torch.no_grad()
def batched_last_hidden_only(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        rows.append(
            forward_last_hidden_only(
                model, tok, prompts[start : start + int(batch_size)], device
            ).detach().float().cpu()
        )
    return torch.cat(rows, dim=0) if rows else torch.empty((0, 0))


@torch.no_grad()
def cache_prompt_baselines(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
    topk: int,
) -> Dict[str, Dict[str, torch.Tensor]]:
    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    unique = compositional.ordered_unique(list(prompts))
    for start in range(0, len(unique), int(batch_size)):
        batch = unique[start : start + int(batch_size)]
        hidden, logits = forward_last_hidden_logits(model, tok, batch, device)
        values, ids = logits.float().topk(min(int(topk), logits.shape[-1]), dim=-1)
        log_probs = torch.log_softmax(values, dim=-1)
        for row, prompt in enumerate(batch):
            cache[prompt] = {
                "hidden": hidden[row].detach().float().cpu(),
                "top_ids": ids[row].detach().cpu(),
                "top_log_probs": log_probs[row].detach().cpu(),
            }
    return cache


@torch.no_grad()
def multi_context_reachability(
    model: torch.nn.Module,
    tok: Any,
    delta_module: canonical.SelectedRowDelta,
    slots: Sequence[int],
    positive_prompts: Sequence[str],
    negative_prompts: Sequence[str],
    base_cache: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    *,
    probes: int,
    sigma: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if delta_module.raw_delta is None:
        raise RuntimeError("reachability requires a full sparse row delta")
    prompts = list(positive_prompts) + list(negative_prompts)
    if not prompts:
        raise ValueError("reachability prompt set is empty")
    base = torch.stack([base_cache[p]["hidden"] for p in prompts])
    positive_rows: List[torch.Tensor] = []
    negative_rows: List[torch.Tensor] = []
    saved = delta_module.raw_delta.detach().clone()
    hidden_size = int(delta_module.raw_delta.shape[1])
    try:
        for _ in range(int(probes) // 2):
            noise = torch.randn(
                (len(slots), hidden_size), generator=generator, dtype=torch.float32
            ).to(delta_module.raw_delta.device) * float(sigma)
            for sign in (1.0, -1.0):
                delta_module.raw_delta.zero_()
                delta_module.raw_delta[list(slots)] = sign * noise
                moved = forward_last_hidden_only(model, tok, prompts, device)
                displacement = moved.detach().float().cpu() - base
                positive_rows.append(displacement[: len(positive_prompts)])
                if negative_prompts:
                    negative_rows.append(displacement[len(positive_prompts) :])
    finally:
        delta_module.raw_delta.copy_(saved)
    positive = torch.cat(positive_rows, dim=0)
    negative = (
        torch.cat(negative_rows, dim=0)
        if negative_rows
        else torch.empty((0, positive.shape[1]), dtype=torch.float32)
    )
    return positive, negative


@torch.no_grad()
def teacher_forced_state_groups(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    answer: str,
    device: torch.device,
    *,
    batch_size: int,
) -> Tuple[List[torch.Tensor], List[int]]:
    target_ids = answer_token_ids(tok, answer)
    if not target_ids:
        raise RuntimeError(f"answer {answer!r} tokenized to an empty sequence")
    pad = int(tok.pad_token_id if tok.pad_token_id is not None else 0)
    groups: List[torch.Tensor] = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = list(prompts[start : start + int(batch_size)])
        prefix_ids = [prompt_token_ids(tok, prompt) for prompt in batch]
        sequences = [prefix + target_ids for prefix in prefix_ids]
        width = max(len(x) for x in sequences)
        ids = torch.tensor(
            [x + [pad] * (width - len(x)) for x in sequences],
            dtype=torch.long,
            device=device,
        )
        attention = torch.tensor(
            [[1] * len(x) + [0] * (width - len(x)) for x in sequences],
            dtype=torch.long,
            device=device,
        )
        backbone = getattr(model, "model", None)
        if backbone is not None and backbone is not model:
            hidden = backbone(
                input_ids=ids,
                attention_mask=attention,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
        else:
            hidden = model(
                input_ids=ids,
                attention_mask=attention,
                output_hidden_states=True,
                use_cache=False,
            ).hidden_states[-1]
        for row, prefix in enumerate(prefix_ids):
            predictors = [len(prefix) + offset - 1 for offset in range(len(target_ids))]
            groups.append(hidden[row, predictors, :].detach().float().cpu())
    return groups, target_ids


def build_prompt_instances(
    records: Sequence[Mapping[str, Any]],
    context_sets: Mapping[int, Mapping[str, Any]],
) -> Tuple[List[mcf_repair.MCFPromptInstance], List[int], List[bool]]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    owners: List[int] = []
    direct_flags: List[bool] = []
    for position, record in enumerate(records):
        case_id = int(record["case_id"])
        prompts = list(context_sets[case_id]["positive_prompts"])
        for prompt_index, prompt in enumerate(prompts):
            instances.append(
                mcf_repair.MCFPromptInstance(
                    record_index=case_id,
                    sampled_position=position,
                    prompt_type="direct" if prompt_index == 0 else "training_safe_positive",
                    prompt_index=prompt_index,
                    prompt=prompt,
                    target_new=str(record["reference"]),
                    target_true=str(record["answer"]),
                )
            )
            owners.append(position)
            direct_flags.append(prompt_index == 0)
    return instances, owners, direct_flags


def differentiable_instance_nlls(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded, target_ids, prefix_lens = mcf_repair.official_batch_components(
        tok, instances, device, llama_like
    )
    logits = model(**encoded, use_cache=False).logits
    if llama_like:
        logits = logits[:, 1:, :]
    losses: List[torch.Tensor] = []
    for row, (tokens, prefix_len) in enumerate(zip(target_ids, prefix_lens)):
        pieces = [
            -F.log_softmax(logits[row, prefix_len + offset - 1, :].float(), dim=-1)[token]
            for offset, token in enumerate(tokens)
        ]
        losses.append(torch.stack(pieces).mean())
    paired = torch.stack(losses).reshape(len(instances), 2)
    return paired[:, 0], paired[:, 1]  # reference new, sensitive true


def differentiable_instance_margins(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
) -> torch.Tensor:
    target_new_nll, target_true_nll = differentiable_instance_nlls(
        model, tok, instances, device, llama_like=llama_like
    )
    return target_true_nll - target_new_nll


@torch.no_grad()
def evaluate_instance_nlls(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target_new: List[torch.Tensor] = []
    target_true: List[torch.Tensor] = []
    for start in range(0, len(instances), int(batch_size)):
        new_batch, true_batch = mcf_repair.official_prompt_instance_nll_tensors(
            model,
            tok,
            instances[start : start + int(batch_size)],
            device,
            llama_like,
        )
        target_new.append(new_batch.detach().cpu())
        target_true.append(true_batch.detach().cpu())
    if not target_new:
        return torch.empty(0), torch.empty(0)
    return torch.cat(target_new), torch.cat(target_true)


@torch.no_grad()
def evaluate_instance_margins(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> torch.Tensor:
    """Evaluate with the same native-dtype arithmetic as the official scorer.

    Stage 2 needs a differentiable approximation while optimizing ``beta``,
    but scale selection and checkpoint acceptance must not use that
    approximation.  The earlier span-writer run lost five direct prompts only
    after official evaluation because its internal float32/BOS path was not the
    official path.  Reuse the repository's exact-compatible NLL helper here so
    every non-gradient decision sees the same token positions, native-dtype
    ``log_softmax``, and float32 sequential accumulation as the evaluator.
    """
    target_new_nll, target_true_nll = evaluate_instance_nlls(
        model,
        tok,
        instances,
        device,
        llama_like=llama_like,
        batch_size=batch_size,
    )
    return target_true_nll - target_new_nll


def absolute_nll_drift_penalty(
    current_nll: torch.Tensor,
    baseline_nll: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    """Squared penalty outside a symmetric reference-answer NLL band."""
    if current_nll.shape != baseline_nll.shape:
        raise ValueError("current and baseline NLL tensors must have equal shape")
    if not math.isfinite(float(tolerance)) or float(tolerance) < 0.0:
        raise ValueError("NLL drift tolerance must be finite and non-negative")
    return F.relu(
        (current_nll - baseline_nll).abs() - float(tolerance)
    ).square().mean()


def distribution(values: Sequence[float]) -> Dict[str, float]:
    finite = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not finite:
        return {"n": 0}
    return {
        "n": len(finite),
        "min": finite[0],
        "p10": finite[max(0, len(finite) // 10)],
        "median": finite[len(finite) // 2],
        "p90": finite[min(len(finite) - 1, 9 * len(finite) // 10)],
        "max": finite[-1],
        "mean": sum(finite) / len(finite),
    }


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)
    generator = torch.Generator().manual_seed(int(a.seed) + 710031)
    py_rng = random.Random(int(a.seed) + 99173)
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    locked_records, split_manifest = directional.validate_locked(
        visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError(
            "compositional writer requires the target-aware direct-only split "
            f"{locked_split.PROTOCOL!r}, got {split_manifest.get('protocol')!r}"
        )
    locked_split.assert_direct_only_training_view(locked_records)
    records = _record_views(locked_records)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embeddings and LM head remain tied")
    device = gagd.first_device(model)
    llama_like = canonical.is_llama_like(model, tok)
    hidden_size = int(input_layer.weight.shape[1])

    transformer_fingerprint = sum(
        float(parameter.detach().float().abs().sum())
        for name, parameter in model.named_parameters()
        if "embed_tokens" not in name and "lm_head" not in name
    )

    print("\nStage 0: training-safe context construction")
    frequency_documents = subject_writer.load_frequency_documents(
        a.wikidata_dir, int(a.frequency_doc_start), int(a.frequency_docs)
    )
    if int(a.frequency_docs) > 0 and not frequency_documents:
        raise RuntimeError(f"no disjoint frequency documents loaded from {a.wikidata_dir!r}")
    token_counts = subject_writer.token_frequency_counts(
        tok, frequency_documents, int(input_layer.weight.shape[0])
    )
    corpus_prefixes = synthetic.corpus_context_prefixes(
        frequency_documents,
        count=int(a.corpus_context_prefixes),
        seed=int(a.seed),
    )

    external_surrogates: List[List[str]] = [[] for _ in records]
    surrogate_receipt: Dict[str, Any] | None = None
    if str(a.surrogate_prompts_path).strip():
        external_surrogates, surrogate_receipt = load_surrogate_prompts(
            Path(a.surrogate_prompts_path).resolve(),
            locked_records,
            seed=int(a.seed),
            require_semantic=bool(a.require_semantic_surrogates),
        )

    positives_by_case: Dict[int, List[str]] = {}
    synthetic_coverage = synthetic.coverage_report(locked_records)
    for position, (locked, record) in enumerate(zip(locked_records, records)):
        rewrite = locked["requested_rewrite"]
        templates = synthetic.synthetic_prompt_templates(
            relation_id=str(rewrite.get("relation_id") or ""),
            canonical_prompt=str(rewrite["prompt"]),
            case_id=int(record["case_id"]),
            count=int(a.synthetic_paraphrases_per_record),
            context_prefixes=corpus_prefixes or None,
        )
        generated = [template.format(str(record["subject"])) for template in templates]
        positives_by_case[int(record["case_id"])] = compositional.ordered_unique(
            [str(record["direct_prompt"]), *generated, *external_surrogates[position]]
        )

    direct_live = subject_writer.live_prompt_token_ids(locked_records, tok)
    positive_live: Dict[int, set[int]] = {}
    for record in records:
        case_id = int(record["case_id"])
        ids: set[int] = set()
        for prompt in positives_by_case[case_id]:
            ids.update(gagd.token_ids_for_text(tok, prompt))
        positive_live[case_id] = ids
    selected_rows, row_reports = subject_writer.select_subject_rows(
        locked_records,
        tok,
        llama_like=llama_like,
        counts=token_counts,
        max_frequency=int(a.max_subject_token_frequency),
        direct_live_ids=direct_live,
        paraphrase_live_ids=positive_live,
    )
    selected_by_case = {
        int(row["case_id"]): [int(x) for x in row["kept_token_ids"]]
        for row in row_reports
    }
    context_sets, context_report = compositional.build_compositional_contexts(
        records,
        positives_by_case,
        selected_by_case,
        tok,
        seed=int(a.seed),
        max_shared_subjects=int(a.max_shared_subword_negatives),
        max_leave_one_out=int(a.max_leave_one_out_negatives),
        max_fragments=int(a.max_fragment_negatives),
        max_unrelated=int(a.max_unrelated_negatives),
    )
    context_manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "source_training_visible_path": str(visible_path),
        "source_training_visible_sha256": sha256_file(visible_path),
        "source_split_manifest": str(manifest_path),
        "source_split_manifest_sha256": sha256_file(manifest_path),
        "surrogate_receipt": surrogate_receipt,
        "synthetic_coverage": synthetic_coverage,
        "corpus_prefixes": len(corpus_prefixes),
        "frequency_document_range": [
            int(a.frequency_doc_start),
            int(a.frequency_doc_start) + len(frequency_documents),
        ],
        "selected_embedding_rows": selected_rows,
        "subject_row_selection": row_reports,
        "summary": context_report,
        "records": [context_sets[int(record["case_id"])] for record in records],
        "data_access": {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
    }
    gagd.write_json(out_dir / "context_manifest.json", context_manifest)
    print(
        f"  {context_report['positive_prompts']} positives, "
        f"{context_report['negative_prompts']} negatives, "
        f"{len(selected_rows)} sparse embedding rows"
    )

    all_prompts: List[str] = []
    for record in records:
        context = context_sets[int(record["case_id"])]
        all_prompts.extend(context["positive_prompts"])
        all_prompts.extend(row["prompt"] for row in context["negative_contexts"])
    base_cache = cache_prompt_baselines(
        model,
        tok,
        all_prompts,
        device,
        batch_size=int(a.cache_batch_size),
        topk=int(a.kl_topk),
    )

    delta_module = canonical.SelectedRowDelta(
        len(selected_rows), hidden_size, direction_basis=None, device=input_layer.weight.device
    )
    if delta_module.raw_delta is None:
        raise AssertionError("embedding writer unexpectedly lacks raw deltas")
    embedding_hook = directional.register_input_embedding_delta_hook(
        input_layer, selected_rows, delta_module.effective_delta
    )
    slot_of = {int(token_id): slot for slot, token_id in enumerate(selected_rows)}

    print("\nStage 0B: contrastive multi-context marker selection")
    marker_start = time.time()
    prior_markers: List[torch.Tensor] = []
    marker_reports: List[Dict[str, Any]] = []
    for record in records:
        case_id = int(record["case_id"])
        context = context_sets[case_id]
        positives = list(context["positive_prompts"][: int(a.reach_positive_contexts)])
        collision_negatives = [
            row["prompt"]
            for row in context["negative_contexts"]
            if row["contains_selected_row"]
        ]
        if not collision_negatives:
            collision_negatives = [row["prompt"] for row in context["negative_contexts"]]
        negatives = collision_negatives[: int(a.reach_negative_contexts)]
        slots = [slot_of[token_id] for token_id in selected_by_case[case_id]]
        positive_reach, negative_reach = multi_context_reachability(
            model,
            tok,
            delta_module,
            slots,
            positives,
            negatives,
            base_cache,
            device,
            probes=int(a.reach_probes),
            sigma=float(a.reach_sigma),
            generator=generator,
        )
        forbidden = (
            compositional.orthonormal_row_basis(torch.stack(prior_markers))
            if prior_markers
            else torch.empty((0, hidden_size), dtype=torch.float32)
        )
        marker, report = compositional.select_contrastive_marker(
            positive_reach,
            negative_reach,
            forbidden_basis=forbidden,
            ridge=float(a.marker_ridge),
            max_rank=int(a.marker_max_rank),
        )
        record["marker"] = marker
        prior_markers.append(marker)
        marker_reports.append({"case_id": case_id, **report})
        print(
            f"  case {case_id}: rank {int(report['candidate_rank'])}, "
            f"reach ratio {report['contrastive_ratio']:.3f}"
        )
    marker_bank = torch.stack([record["marker"] for record in records])
    delta_module.raw_delta.detach().zero_()
    resumed_stage1: Dict[str, Any] | None = None
    resume_path: Path | None = None
    if str(a.resume_stage1_state).strip():
        resume_path = Path(a.resume_stage1_state).resolve()
        resumed_stage1 = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        resume_protocol = str(resumed_stage1.get("protocol") or "")
        compatible_protocols = {
            "mcf_context_composed_sparse_embedding_writer_v2",
            "mcf_context_composed_sparse_embedding_writer_v3",
            "mcf_context_composed_sparse_embedding_writer_v4",
            PROTOCOL,
        }
        if resume_protocol not in compatible_protocols:
            raise RuntimeError(
                f"resumed Stage-1 protocol {resume_protocol!r} is incompatible"
            )
        if [int(x) for x in resumed_stage1.get("selected_embedding_rows", [])] != [
            int(x) for x in selected_rows
        ]:
            raise RuntimeError("resumed Stage-1 selected embedding rows do not match")
        resumed_delta = resumed_stage1.get("embedding_delta")
        if not isinstance(resumed_delta, torch.Tensor) or tuple(resumed_delta.shape) != tuple(
            delta_module.raw_delta.shape
        ):
            raise RuntimeError("resumed Stage-1 embedding delta has incompatible shape")
        resumed_markers = resumed_stage1.get("markers")
        if not isinstance(resumed_markers, Mapping):
            raise RuntimeError("resumed Stage-1 state lacks its marker map")
        loaded_markers: List[torch.Tensor] = []
        for record in records:
            case_id = int(record["case_id"])
            marker = resumed_markers.get(case_id, resumed_markers.get(str(case_id)))
            if not isinstance(marker, torch.Tensor) or tuple(marker.shape) != (hidden_size,):
                raise RuntimeError(f"resumed marker mismatch for case {case_id}")
            record["marker"] = marker.float().cpu()
            loaded_markers.append(record["marker"])
        marker_bank = torch.stack(loaded_markers)
        with torch.no_grad():
            delta_module.raw_delta.copy_(
                resumed_delta.to(
                    device=delta_module.raw_delta.device,
                    dtype=delta_module.raw_delta.dtype,
                )
            )
        print(f"  resumed sparse Stage 1: {resume_path}")
    print(f"  marker selection wall time: {time.time() - marker_start:.1f}s")

    print("\nStage 1: compositional sparse embedding writer")
    optimizer = torch.optim.AdamW(
        delta_module.parameters(), lr=float(a.writer_lr), weight_decay=0.0
    )
    sampler = canonical.IndexSampler(
        len(records), int(a.writer_record_batch), int(a.seed) + 447
    )
    row_frequencies = torch.tensor(
        [float(token_counts[token_id]) for token_id in selected_rows], dtype=torch.float32
    )
    if float(a.row_norm_cap) > 0:
        row_caps = float(a.row_norm_cap) / (
            1.0 + row_frequencies
        ).pow(float(a.row_norm_cap_frequency_alpha))
    else:
        row_caps = torch.zeros_like(row_frequencies)
    log_path = out_dir / "stage1_writer_log.jsonl"
    hard_positive_prompts = {
        index: context_sets[int(record["case_id"])]["positive_prompts"][0]
        for index, record in enumerate(records)
    }
    hard_negative_prompts = {
        index: context_sets[int(record["case_id"])]["negative_contexts"][0]["prompt"]
        for index, record in enumerate(records)
    }
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(1, int(a.writer_steps) + 1):
            record_indices = sampler.next()
            entries: List[Tuple[int, str, str]] = []
            for record_index in record_indices:
                record = records[record_index]
                context = context_sets[int(record["case_id"])]
                positives = list(context["positive_prompts"])
                selected_pos = [positives[0]]
                hard_positive = hard_positive_prompts[record_index]
                if hard_positive not in selected_pos:
                    selected_pos.append(hard_positive)
                remainder = positives[1:]
                py_rng.shuffle(remainder)
                for prompt in remainder:
                    if len(selected_pos) >= int(a.positive_context_batch):
                        break
                    if prompt not in selected_pos:
                        selected_pos.append(prompt)
                negatives = [row["prompt"] for row in context["negative_contexts"]]
                py_rng.shuffle(negatives)
                selected_neg = [hard_negative_prompts[record_index]]
                for prompt in negatives:
                    if len(selected_neg) >= int(a.negative_context_batch):
                        break
                    if prompt not in selected_neg:
                        selected_neg.append(prompt)
                entries.extend((record_index, "positive", prompt) for prompt in selected_pos)
                entries.extend((record_index, "negative", prompt) for prompt in selected_neg)

            optimizer.zero_grad(set_to_none=True)
            prompts = [entry[2] for entry in entries]
            hidden, logits = forward_last_hidden_logits(model, tok, prompts, device)
            base_hidden = torch.stack(
                [base_cache[prompt]["hidden"] for prompt in prompts]
            ).to(device)
            displacement = hidden.float() - base_hidden
            l_write = hidden.sum() * 0.0
            l_consistency = hidden.sum() * 0.0
            l_negative = hidden.sum() * 0.0
            l_state = hidden.sum() * 0.0
            l_cross = hidden.sum() * 0.0
            l_off = hidden.sum() * 0.0
            for record_index in record_indices:
                own = [i for i, entry in enumerate(entries) if entry[0] == record_index]
                pos_indices = [i for i in own if entries[i][1] == "positive"]
                neg_indices = [i for i in own if entries[i][1] == "negative"]
                marker = marker_bank[record_index].to(device)
                pos_delta = displacement[pos_indices]
                amplitudes = pos_delta @ marker
                shortfall = F.relu(float(a.write_alpha) - amplitudes)
                alpha_scale = max(float(a.write_alpha), 1e-6)
                # Squared shortfall makes the currently worst context own the
                # gradient. The first run's mean hinge tolerated A_min=-10
                # while its median climbed, exactly the wrong optimization
                # behavior for zero Gen.
                l_write = l_write + shortfall.square().mean() / alpha_scale
                l_consistency = l_consistency + (
                    amplitudes.var(unbiased=False) / (alpha_scale * alpha_scale)
                )
                parallel = amplitudes.unsqueeze(1) * marker.unsqueeze(0)
                l_off = l_off + (pos_delta - parallel).square().mean()
                if neg_indices:
                    neg_delta = displacement[neg_indices]
                    l_negative = l_negative + (neg_delta @ marker).square().mean()
                    denominator = base_hidden[neg_indices].norm(dim=1).clamp_min(1e-9)
                    l_state = l_state + (
                        neg_delta.norm(dim=1) / denominator
                    ).square().mean()
                peers = torch.cat(
                    [marker_bank[:record_index], marker_bank[record_index + 1 :]], dim=0
                ).to(device)
                if peers.shape[0]:
                    l_cross = l_cross + (pos_delta @ peers.T).square().mean()

            count = max(1, len(record_indices))
            l_write = l_write / count
            l_consistency = l_consistency / count
            l_negative = l_negative / count
            l_state = l_state / count
            l_cross = l_cross / count
            l_off = l_off / count

            l_kl = hidden.sum() * 0.0
            if float(a.lambda_kl) > 0:
                terms: List[torch.Tensor] = []
                for row, prompt in enumerate(prompts):
                    ids = base_cache[prompt]["top_ids"].to(device)
                    target = base_cache[prompt]["top_log_probs"].to(device)
                    observed = torch.log_softmax(logits[row].float()[ids], dim=-1)
                    terms.append(
                        F.kl_div(observed, target, log_target=True, reduction="sum")
                    )
                l_kl = torch.stack(terms).mean()

            loss = (
                l_write
                + float(a.lambda_consistency) * l_consistency
                + float(a.lambda_negative) * l_negative
                + float(a.lambda_state) * l_state
                + float(a.lambda_cross) * l_cross
                + float(a.lambda_off_axis) * l_off
                + float(a.lambda_kl) * l_kl
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite writer loss at step {step}")
            loss.backward()
            if float(a.writer_grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(
                    delta_module.parameters(), float(a.writer_grad_clip)
                )
            optimizer.step()
            if float(a.row_norm_cap) > 0:
                with torch.no_grad():
                    norms = delta_module.raw_delta.norm(dim=1)
                    caps = row_caps.to(delta_module.raw_delta.device)
                    scale = (caps / norms.clamp_min(1e-12)).clamp(max=1.0)
                    delta_module.raw_delta.mul_(scale.unsqueeze(1))

            if step == 1 or step % 25 == 0 or step == int(a.writer_steps):
                row = {
                    "step": step,
                    "loss": float(loss.detach()),
                    "write": float(l_write.detach()),
                    "consistency": float(l_consistency.detach()),
                    "negative": float(l_negative.detach()),
                    "state": float(l_state.detach()),
                    "cross": float(l_cross.detach()),
                    "off_axis": float(l_off.detach()),
                    "topk_kl": float(l_kl.detach()),
                    "writer_delta_norm": float(delta_module.raw_delta.detach().norm()),
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                print(
                    f"  step {step:>4}: loss {row['loss']:.4f}, write {row['write']:.4f}, "
                    f"neg {row['negative']:.4f}, consistency {row['consistency']:.4f}"
                )
            if int(a.writer_eval_every) > 0 and step % int(a.writer_eval_every) == 0:
                per_record_minima: List[float] = []
                per_record_medians: List[float] = []
                for record_index, record in enumerate(records):
                    eval_prompts = list(
                        context_sets[int(record["case_id"])]["positive_prompts"]
                    )
                    eval_hidden = batched_last_hidden_only(
                        model,
                        tok,
                        eval_prompts,
                        device,
                        batch_size=int(a.cache_batch_size),
                    )
                    eval_base = torch.stack(
                        [base_cache[prompt]["hidden"] for prompt in eval_prompts]
                    )
                    amplitudes = (
                        (eval_hidden - eval_base)
                        @ marker_bank[record_index].float()
                    )
                    per_record_minima.append(float(amplitudes.min()))
                    per_record_medians.append(float(amplitudes.median()))
                    hard_positive_prompts[record_index] = eval_prompts[
                        int(amplitudes.argmin())
                    ]
                    negative_prompts = [
                        row["prompt"]
                        for row in context_sets[int(record["case_id"])][
                            "negative_contexts"
                        ]
                    ]
                    negative_hidden = batched_last_hidden_only(
                        model,
                        tok,
                        negative_prompts,
                        device,
                        batch_size=int(a.cache_batch_size),
                    )
                    negative_base = torch.stack(
                        [base_cache[prompt]["hidden"] for prompt in negative_prompts]
                    )
                    negative_amplitudes = (
                        (negative_hidden - negative_base)
                        @ marker_bank[record_index].float()
                    )
                    hard_negative_prompts[record_index] = negative_prompts[
                        int(negative_amplitudes.abs().argmax())
                    ]
                sorted_minima = sorted(per_record_minima)
                sorted_medians = sorted(per_record_medians)
                print(
                    f"    all-context marker A_min: min {sorted_minima[0]:+.3f}, "
                    f"median {sorted_minima[len(sorted_minima)//2]:+.3f}; "
                    f"A_median median {sorted_medians[len(sorted_medians)//2]:+.3f}"
                )
    del optimizer

    trained_embedding_delta = delta_module.effective_delta().detach().cpu()
    torch.save(
        {
            "protocol": PROTOCOL,
            "seed": int(a.seed),
            "case_ids": [int(record["case_id"]) for record in records],
            "selected_embedding_rows": selected_rows,
            "embedding_delta": trained_embedding_delta,
            "markers": {int(r["case_id"]): r["marker"] for r in records},
            "context_manifest_sha256": sha256_file(out_dir / "context_manifest.json"),
            "resumed_from": str(resume_path) if resume_path is not None else None,
        },
        out_dir / "stage1_writer.pt",
    )

    writer_reports: List[Dict[str, Any]] = []
    for record_index, record in enumerate(records):
        context = context_sets[int(record["case_id"])]
        positive_prompts = list(context["positive_prompts"])
        negative_prompts = [row["prompt"] for row in context["negative_contexts"]]
        positive_hidden = batched_last_hidden_only(
            model,
            tok,
            positive_prompts,
            device,
            batch_size=int(a.cache_batch_size),
        )
        negative_hidden = batched_last_hidden_only(
            model,
            tok,
            negative_prompts,
            device,
            batch_size=int(a.cache_batch_size),
        )
        positive_base = torch.stack(
            [base_cache[prompt]["hidden"] for prompt in positive_prompts]
        )
        negative_base = torch.stack(
            [base_cache[prompt]["hidden"] for prompt in negative_prompts]
        )
        positive_delta = positive_hidden - positive_base
        negative_delta = negative_hidden - negative_base
        marker = marker_bank[record_index]
        positive_amplitudes = positive_delta @ marker
        negative_amplitudes = negative_delta @ marker
        negative_drift = negative_delta.norm(dim=1) / negative_base.norm(
            dim=1
        ).clamp_min(1e-9)
        peer_markers = torch.cat(
            [marker_bank[:record_index], marker_bank[record_index + 1 :]], dim=0
        )
        cross_activation = (
            float((positive_delta @ peer_markers.T).abs().max())
            if peer_markers.shape[0]
            else 0.0
        )
        positive_min = float(positive_amplitudes.min())
        marker_kappa = float(negative_amplitudes.abs().max()) / (
            max(abs(positive_min), 1e-9)
        )
        writer_reports.append(
            {
                "case_id": int(record["case_id"]),
                "positive_contexts": len(positive_prompts),
                "negative_contexts": len(negative_prompts),
                "positive_amplitude_min": positive_min,
                "positive_amplitude_median": float(positive_amplitudes.median()),
                "positive_amplitude_max": float(positive_amplitudes.max()),
                "positive_amplitude_variance": float(
                    positive_amplitudes.var(unbiased=False)
                ),
                "negative_marker_abs_max": float(negative_amplitudes.abs().max()),
                "marker_kappa": marker_kappa,
                "negative_state_drift_median": float(negative_drift.median()),
                "negative_state_drift_max": float(negative_drift.max()),
                "cross_record_marker_abs_max": cross_activation,
            }
        )
    writer_report = {
        "write_alpha": float(a.write_alpha),
        "criterion": {
            "positive_amplitude_min": (
                float(a.writer_amplitude_min_frac) * float(a.write_alpha)
            ),
            "marker_kappa_max": float(a.writer_marker_kappa_max),
        },
        "positive_amplitude_min": distribution(
            [row["positive_amplitude_min"] for row in writer_reports]
        ),
        "positive_amplitude_variance": distribution(
            [row["positive_amplitude_variance"] for row in writer_reports]
        ),
        "negative_marker_abs_max": distribution(
            [row["negative_marker_abs_max"] for row in writer_reports]
        ),
        "marker_kappa": distribution(
            [row["marker_kappa"] for row in writer_reports]
        ),
        "negative_state_drift_max": distribution(
            [row["negative_state_drift_max"] for row in writer_reports]
        ),
        "cross_record_marker_abs_max": distribution(
            [row["cross_record_marker_abs_max"] for row in writer_reports]
        ),
        "per_record": writer_reports,
    }
    writer_pass_flags = [
        float(row["positive_amplitude_min"])
        >= float(a.writer_amplitude_min_frac) * float(a.write_alpha)
        and float(row["marker_kappa"]) <= float(a.writer_marker_kappa_max)
        for row in writer_reports
    ]
    writer_report["passed_records"] = int(sum(writer_pass_flags))
    writer_report["total_records"] = len(writer_pass_flags)
    writer_report["pass_fraction"] = sum(writer_pass_flags) / max(
        1, len(writer_pass_flags)
    )
    gagd.write_json(out_dir / "stage1_writer_report.json", writer_report)

    print("\nReader: distributional q over all training-safe contexts")
    positive_state_groups: Dict[int, List[torch.Tensor]] = {}
    negative_state_groups: Dict[int, List[torch.Tensor]] = {}
    answer_rows_by_record: List[List[int]] = []
    for position, record in enumerate(records):
        context = context_sets[int(record["case_id"])]
        positives = list(context["positive_prompts"])
        negatives = [row["prompt"] for row in context["negative_contexts"]]
        positive_groups, answer_rows = teacher_forced_state_groups(
            model,
            tok,
            positives,
            str(record["answer"]),
            device,
            batch_size=int(a.cache_batch_size),
        )
        negative_groups, negative_answer_rows = teacher_forced_state_groups(
            model,
            tok,
            negatives,
            str(record["answer"]),
            device,
            batch_size=int(a.cache_batch_size),
        )
        if answer_rows != negative_answer_rows:
            raise AssertionError("positive/negative answer tokenization diverged")
        positive_state_groups[position] = positive_groups
        negative_state_groups[position] = negative_groups
        answer_rows_by_record.append(answer_rows)

    # Cache the exact same sensitive-answer states with the sparse embedding
    # writer removed. Stage 2 penalizes output-row activation on these states,
    # so success must use the writer-induced contextual displacement instead
    # of learning a stand-alone output-head edit from pre-existing geometry.
    embedding_hook.remove()
    base_positive_state_groups: Dict[int, List[torch.Tensor]] = {}
    corpus_protection_rows = torch.empty((0, hidden_size), dtype=torch.float32)
    try:
        for position, record in enumerate(records):
            context = context_sets[int(record["case_id"])]
            base_groups, base_answer_rows = teacher_forced_state_groups(
                model,
                tok,
                list(context["positive_prompts"]),
                str(record["answer"]),
                device,
                batch_size=int(a.cache_batch_size),
            )
            if base_answer_rows != answer_rows_by_record[position]:
                raise AssertionError("base/edited answer tokenization diverged")
            base_positive_state_groups[position] = base_groups
        corpus_prompts = list(
            corpus_prefixes[: int(a.stage2_corpus_protection_prompts)]
        )
        if corpus_prompts:
            corpus_protection_rows = batched_last_hidden_only(
                model,
                tok,
                corpus_prompts,
                device,
                batch_size=int(a.cache_batch_size),
            )
    finally:
        embedding_hook = directional.register_input_embedding_delta_hook(
            input_layer, selected_rows, delta_module.effective_delta
        )

    readers: List[torch.Tensor] = []
    reader_reports: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        own_positive = torch.cat(positive_state_groups[position], dim=0)
        # The same-prompt reference sequence is deliberately not a hidden-state
        # negative. Its first predictor is exactly the positive predictor, so
        # including it makes selectivity impossible by construction. Reference
        # answer preservation is enforced directly as an NLL constraint in
        # Stage 2 instead.
        own_negative_parts = [
            *negative_state_groups[position],
            *base_positive_state_groups[position],
        ]
        own_answer_rows = set(answer_rows_by_record[position])
        own_subject_rows = set(selected_by_case[int(record["case_id"])])
        cross_candidates: List[torch.Tensor] = []
        for other_position, other in enumerate(records):
            if other_position == position:
                continue
            answer_collision = bool(
                own_answer_rows & set(answer_rows_by_record[other_position])
            )
            subject_collision = bool(
                own_subject_rows & set(selected_by_case[int(other["case_id"])])
            )
            if answer_collision or subject_collision:
                cross_candidates.extend(positive_state_groups[other_position])
        if cross_candidates:
            flat_cross = torch.cat(cross_candidates, dim=0)
            cap = min(int(a.reader_cross_positive_cap), flat_cross.shape[0])
            own_negative_parts.append(flat_cross[:cap])
        own_negative = torch.cat(own_negative_parts, dim=0)
        reader, fit_report = compositional.distributional_reader(
            record["marker"].to(device),
            own_positive.to(device),
            own_negative.to(device),
            ridge=float(a.reader_ridge),
            anchor_weight=float(a.reader_anchor_weight),
            consistency_weight=float(a.reader_consistency_weight),
            negative_weight=float(a.reader_negative_weight),
            refine_steps=int(a.reader_refine_steps),
            refine_lr=float(a.reader_refine_lr),
            positive_floor=float(a.reader_positive_floor),
        )
        metrics = compositional.reader_metrics(reader, own_positive, own_negative)
        readers.append(reader)
        reader_reports.append(
            {
                "case_id": int(record["case_id"]),
                "positive_states": int(own_positive.shape[0]),
                "negative_states": int(own_negative.shape[0]),
                **fit_report,
                **metrics,
                "abs_cos_marker_q": abs(float(fit_report["cos_marker_q"])),
            }
        )
        print(
            f"  case {record['case_id']}: kappa {metrics['kappa_train']:.4f}, "
            f"R {metrics['portability_ratio']:.4f}, cos(v,q) {fit_report['cos_marker_q']:.4f}"
        )

    reader_bank = torch.stack(readers)
    pass_flags = [
        bool(writer_pass_flags[index])
        and bool(row["positive_sign_consistent"])
        and float(row["kappa_train"]) <= float(a.kappa_train_max)
        and float(row["portability_ratio"]) >= float(a.portability_min)
        and float(row["abs_cos_marker_q"]) >= float(a.cos_marker_reader_min)
        for index, row in enumerate(reader_reports)
    ]
    pass_fraction = sum(pass_flags) / max(1, len(pass_flags))
    gate_report = {
        "criterion": {
            "positive_sign_consistent": True,
            "kappa_train_max": float(a.kappa_train_max),
            "portability_min": float(a.portability_min),
            "abs_cos_marker_reader_min": float(a.cos_marker_reader_min),
            "writer_amplitude_min_frac": float(a.writer_amplitude_min_frac),
            "writer_marker_kappa_max": float(a.writer_marker_kappa_max),
            "pass_fraction_required": float(a.gate_pass_frac),
        },
        "pass_fraction": pass_fraction,
        "passed_records": int(sum(pass_flags)),
        "total_records": len(pass_flags),
        "passed": pass_fraction >= float(a.gate_pass_frac),
        "kappa_train": distribution([row["kappa_train"] for row in reader_reports]),
        "portability_ratio": distribution(
            [row["portability_ratio"] for row in reader_reports]
        ),
        "cos_marker_q": distribution([row["cos_marker_q"] for row in reader_reports]),
        "abs_cos_marker_q": distribution(
            [row["abs_cos_marker_q"] for row in reader_reports]
        ),
        "per_record": reader_reports,
    }
    gagd.write_json(out_dir / "reader_gate_report.json", gate_report)
    print(
        f"  portability gate: {sum(pass_flags)}/{len(pass_flags)} records "
        f"({pass_fraction:.1%})"
    )
    if not gate_report["passed"] and a.gate_policy == "strict":
        embedding_hook.remove()
        raise SystemExit(
            f"reader portability gate failed; refusing Stage 2. See "
            f"{out_dir / 'reader_gate_report.json'}"
        )

    print("\nStage 2: bounded writer-residual LM-head reader solve")
    positive_instances, _instance_owners, direct_flags = build_prompt_instances(
        records, context_sets
    )
    pre_target_new_nll, pre_target_true_nll = evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(a.stage2_batch_size),
    )
    pre_margins = pre_target_true_nll - pre_target_new_nll
    pre_margin_report = distribution([float(x) for x in pre_margins])
    selected_output_rows = sorted(
        {int(token_id) for rows in answer_rows_by_record for token_id in rows}
        - set(gagd.special_token_ids(tok))
    )
    output_row_index = torch.tensor(
        selected_output_rows, dtype=torch.long, device=output_layer.weight.device
    )
    base_selected_output_rows = output_layer.weight.index_select(
        0, output_row_index
    ).detach().clone()
    output_slot = {
        token_id: slot for slot, token_id in enumerate(selected_output_rows)
    }
    base_output_row_norms = base_selected_output_rows.float().norm(dim=1)
    if bool((base_output_row_norms <= 1e-12).any()):
        raise RuntimeError("selected LM-head row has zero base norm")

    # V5 never searches the full hidden space.  Common writer-off and corpus
    # geometry is removed first; each output token then gets its own bounded
    # basis made only from paired writer residuals for that token.
    base_positive_rows_cpu = torch.cat(
        [
            torch.cat(base_positive_state_groups[i], dim=0)
            for i in range(len(records))
        ],
        dim=0,
    ).float()
    common_protection_rows_cpu = base_positive_rows_cpu
    if corpus_protection_rows.numel():
        common_protection_rows_cpu = torch.cat(
            [common_protection_rows_cpu, corpus_protection_rows.float()], dim=0
        )
    if common_protection_rows_cpu.shape[0] > int(a.stage2_protection_states):
        indices = torch.linspace(
            0,
            common_protection_rows_cpu.shape[0] - 1,
            steps=int(a.stage2_protection_states),
        ).round().long()
        common_protection_rows_cpu = common_protection_rows_cpu.index_select(
            0, indices
        )
    if int(a.stage2_protection_rank) > 0:
        common_protection_basis = compositional.orthonormal_row_basis(
            common_protection_rows_cpu.to(output_layer.weight.device),
            max_rank=min(
                int(a.stage2_protection_rank),
                hidden_size - 1,
            ),
        )
    else:
        common_protection_basis = torch.empty(
            (0, hidden_size),
            dtype=torch.float32,
            device=output_layer.weight.device,
        )
    common_protection_basis_cpu = common_protection_basis.cpu()
    print(
        f"  common protected subspace: rank {common_protection_basis_cpu.shape[0]} "
        f"from {common_protection_rows_cpu.shape[0]} writer-off/corpus states"
    )

    def row_state_key(
        prompt: str, answer_rows: Sequence[int], offset: int
    ) -> Tuple[str, Tuple[int, ...], int]:
        return (
            compositional.normalized_key(prompt),
            tuple(int(x) for x in answer_rows[:offset]),
            int(answer_rows[offset]),
        )

    positive_keys_by_token: Dict[int, set[Tuple[str, Tuple[int, ...], int]]] = {
        token_id: set() for token_id in selected_output_rows
    }
    for position, record in enumerate(records):
        prompts = list(
            context_sets[int(record["case_id"])]["positive_prompts"]
        )
        answer_rows = answer_rows_by_record[position]
        for prompt in prompts:
            for offset, token_id in enumerate(answer_rows):
                if int(token_id) in positive_keys_by_token:
                    positive_keys_by_token[int(token_id)].add(
                        row_state_key(prompt, answer_rows, offset)
                    )

    residual_basis_bank_cpu = torch.zeros(
        (
            len(selected_output_rows),
            int(a.stage2_residual_rank),
            hidden_size,
        ),
        dtype=torch.float32,
    )
    row_negative_states_cpu: List[torch.Tensor] = []
    residual_basis_reports: List[Dict[str, Any]] = []
    for token_id in selected_output_rows:
        residual_parts: List[torch.Tensor] = []
        row_negative_parts: List[torch.Tensor] = []
        for position, record in enumerate(records):
            answer_rows = answer_rows_by_record[position]
            offsets = [
                offset
                for offset, owned_token in enumerate(answer_rows)
                if int(owned_token) == int(token_id)
            ]
            if not offsets:
                continue
            edited_groups = positive_state_groups[position]
            writer_off_groups = base_positive_state_groups[position]
            if len(edited_groups) != len(writer_off_groups):
                raise AssertionError("edited/writer-off positive groups diverged")
            for edited_group, writer_off_group in zip(
                edited_groups, writer_off_groups
            ):
                residual_parts.extend(
                    (edited_group[offset] - writer_off_group[offset]).unsqueeze(0)
                    for offset in offsets
                )

            negative_prompts = [
                row["prompt"]
                for row in context_sets[int(record["case_id"])][
                    "negative_contexts"
                ]
            ]
            owned_negative_groups = negative_state_groups[position]
            if len(negative_prompts) != len(owned_negative_groups):
                raise AssertionError("negative prompt/state groups diverged")
            for prompt, group in zip(negative_prompts, owned_negative_groups):
                for offset in offsets:
                    # Token-row semantics: a state that is a positive for y in
                    # any record cannot simultaneously be a negative for q_y.
                    if (
                        row_state_key(prompt, answer_rows, offset)
                        in positive_keys_by_token[int(token_id)]
                    ):
                        continue
                    row_negative_parts.append(group[offset].unsqueeze(0))

        residuals = torch.cat(residual_parts, dim=0).float()
        row_negatives = (
            torch.cat(row_negative_parts, dim=0).float()
            if row_negative_parts
            else torch.empty((0, hidden_size), dtype=torch.float32)
        )
        basis_device, basis_report = compositional.residual_reader_basis(
            residuals.to(output_layer.weight.device),
            common_protection_basis,
            row_negatives.to(output_layer.weight.device),
            residual_rank=int(a.stage2_residual_rank),
            row_negative_rank=int(a.stage2_row_negative_rank),
        )
        basis = basis_device.cpu()
        slot = output_slot[int(token_id)]
        residual_basis_bank_cpu[slot, : basis.shape[0]] = basis
        row_negative_states_cpu.append(row_negatives)
        residual_norms = residuals.norm(dim=1)
        residual_basis_reports.append(
            {
                "token_id": int(token_id),
                **basis_report,
                "writer_residual_norm": distribution(
                    [float(x) for x in residual_norms]
                ),
            }
        )
        print(
            f"  token {token_id}: residual rank {basis.shape[0]}, "
            f"row-negative rank {basis_report['row_negative_rank']}, "
            f"safe energy {basis_report['safe_residual_energy_fraction']:.4f}"
        )

    residual_basis_bank = residual_basis_bank_cpu.to(
        output_layer.weight.device
    )
    output_coefficients = torch.nn.Parameter(
        torch.zeros(
            (len(selected_output_rows), int(a.stage2_residual_rank)),
            dtype=torch.float32,
            device=output_layer.weight.device,
        )
    )

    def raw_output_delta() -> torch.Tensor:
        return compositional.row_basis_deltas(
            output_coefficients, residual_basis_bank
        )

    def current_output_delta() -> torch.Tensor:
        return compositional.materialized_row_delta_ste(
            raw_output_delta(), base_selected_output_rows
        )

    output_hook = canonical.register_output_delta_hook(
        output_layer, selected_output_rows, current_output_delta
    )
    stage2_optimizer = torch.optim.AdamW(
        [output_coefficients], lr=float(a.stage2_lr), weight_decay=0.0
    )
    common_protection_rows = common_protection_rows_cpu.to(
        output_layer.weight.device
    )
    base_positive_rows = base_positive_rows_cpu.to(output_layer.weight.device)
    row_negative_states = [
        rows.to(output_layer.weight.device) for rows in row_negative_states_cpu
    ]
    maximum_relative_cap = max(a.stage2_row_norm_cap_values)
    instance_sampler = canonical.IndexSampler(
        len(positive_instances), int(a.stage2_batch_size), int(a.seed) + 811
    )
    stage2_log: List[Dict[str, Any]] = []
    for step in range(1, int(a.stage2_steps) + 1):
        picked = instance_sampler.next()
        batch = [positive_instances[i] for i in picked]
        stage2_optimizer.zero_grad(set_to_none=True)
        target_new_nll, target_true_nll = differentiable_instance_nlls(
            model, tok, batch, device, llama_like=llama_like
        )
        margins = target_true_nll - target_new_nll
        hinge = F.relu(float(a.forget_margin) - margins).mean()
        baseline_target_new = pre_target_new_nll[picked].to(
            device=target_new_nll.device, dtype=target_new_nll.dtype
        )
        reference_nll_drift = absolute_nll_drift_penalty(
            target_new_nll,
            baseline_target_new,
            float(a.stage2_reference_nll_tolerance),
        )
        delta_rows = current_output_delta()
        protected_shift = common_protection_rows.float() @ delta_rows.float().T
        locality = protected_shift.square().sum(dim=1).mean()
        base_positive_shift = base_positive_rows.float() @ delta_rows.float().T
        base_positive_locality = base_positive_shift.square().sum(dim=1).mean()
        per_row_negative_losses = [
            (rows.float() @ delta_rows[slot].float()).square().mean()
            for slot, rows in enumerate(row_negative_states)
            if rows.numel()
        ]
        row_negative_locality = (
            torch.stack(per_row_negative_losses).mean()
            if per_row_negative_losses
            else delta_rows.sum() * 0.0
        )
        relative_row_norms = (
            delta_rows.float().norm(dim=1)
            / base_output_row_norms.to(delta_rows).clamp_min(1e-30)
        )
        output_l2 = relative_row_norms.square().mean()
        loss = (
            float(a.stage2_margin_weight) * hinge
            + float(a.stage2_negative_weight) * locality
            + float(a.stage2_negative_weight) * row_negative_locality
            + float(a.stage2_base_positive_weight) * base_positive_locality
            + float(a.stage2_reference_nll_weight) * reference_nll_drift
            + float(a.stage2_beta_l2) * output_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
        loss.backward()
        stage2_optimizer.step()
        cap_projection = compositional.clamp_materialized_basis_coefficients_(
            output_coefficients,
            residual_basis_bank,
            base_selected_output_rows,
            maximum_relative_cap,
        )

        if step == 1 or step % int(a.stage2_check_every) == 0 or step == int(a.stage2_steps):
            full_target_new_nll, full_target_true_nll = evaluate_instance_nlls(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(a.stage2_batch_size),
            )
            full = full_target_true_nll - full_target_new_nll
            full_reference_nll_abs_drift_max = float(
                (full_target_new_nll - pre_target_new_nll).abs().max()
            )
            failures = int((full < float(a.forget_margin) - 1e-6).sum())
            direct_failures = sum(
                int(direct_flags[i] and float(full[i]) < float(a.forget_margin) - 1e-6)
                for i in range(len(full))
            )
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "batch_hinge": float(hinge.detach()),
                "locality": float(locality.detach()),
                "row_negative_locality": float(row_negative_locality.detach()),
                "base_positive_locality": float(base_positive_locality.detach()),
                "reference_nll_drift": float(
                    reference_nll_drift.detach()
                ),
                "output_delta_l2": float(output_l2.detach()),
                "maximum_relative_row_norm": float(
                    cap_projection["materialized_max_relative_norm"]
                ),
                "positive_failures": failures,
                "direct_failures": direct_failures,
                "minimum_margin": float(full.min()),
                "reference_nll_abs_drift_max": full_reference_nll_abs_drift_max,
            }
            stage2_log.append(row)
            print(
                f"  step {step:>4}: direct fail {direct_failures}, all-positive fail "
                f"{failures}, min margin {row['minimum_margin']:+.4f}, "
                f"max |ref dNLL| {full_reference_nll_abs_drift_max:.4f}"
            )
            if (
                failures == 0
                and full_reference_nll_abs_drift_max
                <= float(a.stage2_reference_nll_tolerance) + 1e-6
            ):
                break
    del stage2_optimizer

    solved_output_coefficients = output_coefficients.detach().clone()
    cap_reports: List[Dict[str, Any]] = []
    cap_candidates: List[torch.Tensor] = []
    for relative_cap in a.stage2_row_norm_cap_values:
        with torch.no_grad():
            output_coefficients.copy_(solved_output_coefficients)
        cap_projection = compositional.clamp_materialized_basis_coefficients_(
            output_coefficients,
            residual_basis_bank,
            base_selected_output_rows,
            float(relative_cap),
        )
        candidate_target_new_nll, candidate_target_true_nll = evaluate_instance_nlls(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(a.stage2_batch_size),
        )
        margins = candidate_target_true_nll - candidate_target_new_nll
        reference_nll_abs_drift_max = float(
            (candidate_target_new_nll - pre_target_new_nll).abs().max()
        )
        direct_failures = sum(
            int(direct_flags[i] and float(margins[i]) < float(a.forget_margin) - 1e-6)
            for i in range(len(margins))
        )
        all_failures = int((margins < float(a.forget_margin) - 1e-6).sum())
        candidate_delta = current_output_delta().detach()
        relative_norms = (
            candidate_delta.float().norm(dim=1)
            / base_output_row_norms.to(candidate_delta).clamp_min(1e-30)
        )
        cap_reports.append(
            {
                "relative_cap": float(relative_cap),
                "direct_failures": direct_failures,
                "positive_failures": all_failures,
                "minimum_margin": float(margins.min()),
                "reference_nll_abs_drift_max": reference_nll_abs_drift_max,
                "maximum_relative_row_norm": float(relative_norms.max()),
                "mean_relative_row_norm": float(relative_norms.mean()),
                "materialized_violating_rows": int(
                    cap_projection["materialized_violating_rows"]
                ),
            }
        )
        cap_candidates.append(output_coefficients.detach().clone())
    feasible = [
        (index, row)
        for index, row in enumerate(cap_reports)
        if row["positive_failures"] == 0
        and float(row["reference_nll_abs_drift_max"])
        <= float(a.stage2_reference_nll_tolerance) + 1e-6
        and int(row["materialized_violating_rows"]) == 0
    ]
    if feasible:
        selected_cap_index, _selected_cap_report = min(
            feasible, key=lambda pair: float(pair[1]["relative_cap"])
        )
    else:
        selected_cap_index, _selected_cap_report = min(
            enumerate(cap_reports),
            key=lambda pair: (
                int(pair[1]["direct_failures"]),
                int(pair[1]["positive_failures"]),
                max(
                    0.0,
                    float(pair[1]["reference_nll_abs_drift_max"])
                    - float(a.stage2_reference_nll_tolerance),
                ),
                -float(pair[1]["minimum_margin"]),
            ),
        )
    selected_relative_cap = float(
        cap_reports[selected_cap_index]["relative_cap"]
    )
    with torch.no_grad():
        output_coefficients.copy_(cap_candidates[selected_cap_index])
        final_output_delta = current_output_delta().detach().cpu()
        final_raw_output_delta = raw_output_delta().detach().cpu()
        final_output_rows = (
            base_selected_output_rows
            + final_raw_output_delta.to(
                device=base_selected_output_rows.device,
                dtype=base_selected_output_rows.dtype,
            )
        )
    hooked_final_margins = evaluate_instance_margins(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(a.stage2_batch_size),
    )

    # Fold both ordinary global parameter changes into a standard HF checkpoint.
    # V5 has no post-hoc scale retry: the exact BF16/FP16 target rows were part
    # of the hard-cap optimization and are serialized once.
    output_hook.remove()
    embedding_hook.remove()
    input_row_index = torch.tensor(
        selected_rows, dtype=torch.long, device=input_layer.weight.device
    )
    base_selected_input_rows = input_layer.weight.index_select(
        0, input_row_index
    ).detach().clone()
    directional.materialize_input_delta(
        input_layer, selected_rows, trained_embedding_delta
    )
    with torch.no_grad():
        output_layer.weight.index_copy_(
            0, output_row_index, final_output_rows
        )
    materialized_target_new_nll, materialized_target_true_nll = evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(a.stage2_batch_size),
    )
    materialized_margins = (
        materialized_target_true_nll - materialized_target_new_nll
    )
    margin_drift = float(
        (materialized_margins - hooked_final_margins).abs().max()
    )
    final_reference_nll_drift = (
        materialized_target_new_nll - pre_target_new_nll
    )
    final_reference_nll_abs_drift_max = float(
        final_reference_nll_drift.abs().max()
    )

    # Causal writer ablation: leave the learned output rows in place but
    # restore the original subject input rows. If this output-only model still
    # satisfies the forget constraints, the flexible LM-head solve bypassed
    # the proposed context-composed embedding code. Always restore the edited
    # input rows before serialization.
    edited_selected_input_rows = input_layer.weight.index_select(
        0, input_row_index
    ).detach().clone()
    with torch.no_grad():
        input_layer.weight.index_copy_(0, input_row_index, base_selected_input_rows)
    try:
        output_only_margins = evaluate_instance_margins(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(a.stage2_batch_size),
        )
    finally:
        with torch.no_grad():
            input_layer.weight.index_copy_(
                0, input_row_index, edited_selected_input_rows
            )
    output_only_direct_failures = sum(
        int(
            direct_flags[i]
            and float(output_only_margins[i]) < float(a.forget_margin) - 1e-6
        )
        for i in range(len(output_only_margins))
    )
    output_only_positive_failures = int(
        (output_only_margins < float(a.forget_margin) - 1e-6).sum()
    )
    causal_writer_gain = materialized_margins - output_only_margins
    causal_writer_ablation = {
        "kind": "restore_original_input_rows_keep_sparse_output_edit",
        "direct_failures_without_writer": output_only_direct_failures,
        "positive_failures_without_writer": output_only_positive_failures,
        "minimum_margin_without_writer": float(output_only_margins.min()),
        "margin_gain_from_writer": distribution(
            [float(x) for x in causal_writer_gain]
        ),
        "writer_is_necessary_for_at_least_one_positive": bool(
            output_only_positive_failures > 0
        ),
        "writer_is_necessary_for_at_least_one_direct": bool(
            output_only_direct_failures > 0
        ),
        "edited_input_rows_restored_before_save": bool(
            torch.equal(
                input_layer.weight.index_select(0, input_row_index),
                edited_selected_input_rows,
            )
        ),
    }
    gagd.write_json(out_dir / "causal_writer_ablation.json", causal_writer_ablation)
    intended_output_delta = final_output_delta
    actual_embedding_delta = (
        edited_selected_input_rows.detach().float().cpu()
        - base_selected_input_rows.detach().float().cpu()
    )
    actual_output_delta = (
        output_layer.weight.index_select(0, output_row_index).detach().float().cpu()
        - base_selected_output_rows.detach().float().cpu()
    )
    output_materialization_delta_error = float(
        (actual_output_delta - intended_output_delta.float()).abs().max()
    )
    output_row_betas, output_row_readers = compositional.factorize_output_rows(
        actual_output_delta
    )
    actual_relative_output_row_norms = (
        actual_output_delta.norm(dim=1)
        / base_selected_output_rows.detach().float().cpu().norm(dim=1).clamp_min(1e-30)
    )
    output_row_cap_passed = bool(
        float(actual_relative_output_row_norms.max())
        <= selected_relative_cap + 1e-6
    )

    # Every jointly learned sparse row has the exact decomposition
    # Delta W_y = -beta_y q_y. Audit the resulting q_y readers on all
    # training-safe states where token y is sensitive, versus the protected
    # compositional, writer-off, and disjoint-corpus pool used by Stage 2.
    output_reader_rows: List[Dict[str, Any]] = []
    common_audit_rows_cpu = base_positive_rows_cpu
    if corpus_protection_rows.numel():
        common_audit_rows_cpu = torch.cat(
            [common_audit_rows_cpu, corpus_protection_rows.float()], dim=0
        )
    basis_report_by_token = {
        int(row["token_id"]): row for row in residual_basis_reports
    }
    for token_id in selected_output_rows:
        slot = output_slot[token_id]
        protected_parts = [common_audit_rows_cpu]
        if row_negative_states_cpu[slot].numel():
            protected_parts.append(row_negative_states_cpu[slot])
        protected_cpu = torch.cat(protected_parts, dim=0)
        beta = float(output_row_betas[slot])
        positive_parts: List[torch.Tensor] = []
        base_positive_parts: List[torch.Tensor] = []
        for position in range(len(records)):
            offsets = [
                offset
                for offset, owned_token in enumerate(answer_rows_by_record[position])
                if int(owned_token) == int(token_id)
            ]
            edited_groups = positive_state_groups[position]
            writer_off_groups = base_positive_state_groups[position]
            if len(edited_groups) != len(writer_off_groups):
                raise AssertionError("edited/writer-off positive groups diverged")
            for group, base_group in zip(edited_groups, writer_off_groups):
                positive_parts.extend(group[offset].unsqueeze(0) for offset in offsets)
                base_positive_parts.extend(
                    base_group[offset].unsqueeze(0) for offset in offsets
                )
        if not positive_parts or beta <= 1e-12:
            output_reader_rows.append(
                {
                    "token_id": int(token_id),
                    "beta": beta,
                    "relative_row_norm": float(
                        actual_relative_output_row_norms[slot]
                    ),
                    "residual_basis": basis_report_by_token[int(token_id)],
                    "active": False,
                    "passed": True,
                }
            )
            continue
        positive_states = torch.cat(positive_parts, dim=0)
        owned_base_states = torch.cat(base_positive_parts, dim=0)
        writer_displacement = positive_states - owned_base_states
        displacement_response = writer_displacement @ output_row_readers[slot]
        positive_response = positive_states @ output_row_readers[slot]
        writer_response_fraction = (
            displacement_response.abs()
            / positive_response.abs().clamp_min(1e-9)
        )
        metrics = compositional.reader_metrics(
            output_row_readers[slot], positive_states, protected_cpu
        )
        passed = bool(
            metrics["positive_sign_consistent"]
            and float(metrics["kappa_train"]) <= float(a.kappa_train_max)
            and float(metrics["portability_ratio"]) >= float(a.portability_min)
        )
        output_reader_rows.append(
            {
                "token_id": int(token_id),
                "beta": beta,
                "relative_row_norm": float(
                    actual_relative_output_row_norms[slot]
                ),
                "residual_basis": basis_report_by_token[int(token_id)],
                "active": True,
                "passed": passed,
                **metrics,
                "writer_displacement_response": distribution(
                    [float(x) for x in displacement_response]
                ),
                "writer_displacement_positive_fraction": float(
                    (displacement_response > 0).float().mean()
                ),
                "writer_response_fraction": distribution(
                    [float(x) for x in writer_response_fraction]
                ),
            }
        )
    active_reader_rows = [row for row in output_reader_rows if row["active"]]
    output_reader_gate = {
        "kind": "post_solve_sparse_lm_head_row_reader_gate",
        "factorization": "Delta W_y = -beta_y q_y",
        "criterion": {
            "positive_sign_consistent": True,
            "kappa_train_max": float(a.kappa_train_max),
            "portability_min": float(a.portability_min),
            "selected_relative_row_norm_cap": selected_relative_cap,
        },
        "hard_row_cap_passed": output_row_cap_passed,
        "active_rows": len(active_reader_rows),
        "total_rows": len(output_reader_rows),
        "passed_rows": sum(int(row["passed"]) for row in active_reader_rows),
        "passed": bool(
            active_reader_rows
            and all(row["passed"] for row in active_reader_rows)
            and output_row_cap_passed
        ),
        "per_row": output_reader_rows,
    }
    gagd.write_json(out_dir / "output_reader_gate_report.json", output_reader_gate)
    direct_failures = sum(
        int(direct_flags[i] and float(materialized_margins[i]) < float(a.forget_margin) - 1e-6)
        for i in range(len(materialized_margins))
    )
    positive_failures = int(
        (materialized_margins < float(a.forget_margin) - 1e-6).sum()
    )
    bounded_training_passed = bool(
        direct_failures == 0
        and positive_failures == 0
        and final_reference_nll_abs_drift_max
        <= float(a.stage2_reference_nll_tolerance) + 1e-6
        and output_row_cap_passed
        and bool(feasible)
    )
    reader_policy_passed = bool(
        output_reader_gate["passed"] or a.gate_policy == "report"
    )

    post_transformer_fingerprint = sum(
        float(parameter.detach().float().abs().sum())
        for name, parameter in model.named_parameters()
        if "embed_tokens" not in name and "lm_head" not in name
    )
    transformer_absdiff = abs(post_transformer_fingerprint - transformer_fingerprint)
    if transformer_absdiff > 1e-3:
        raise RuntimeError("a frozen Transformer parameter changed")
    checkpoint_path = out_dir / "checkpoint"
    checkpoint_saved = bool(
        a.save_checkpoint and bounded_training_passed and reader_policy_passed
    )
    if checkpoint_saved:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tok.save_pretrained(checkpoint_path)

    torch.save(
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "seed": int(a.seed),
            "case_ids": [int(record["case_id"]) for record in records],
            "selected_embedding_rows": selected_rows,
            "embedding_delta": trained_embedding_delta,
            "actual_embedding_delta": actual_embedding_delta,
            "base_selected_embedding_rows": (
                base_selected_input_rows.detach().cpu()
            ),
            "edited_selected_embedding_rows": (
                edited_selected_input_rows.detach().cpu()
            ),
            "markers": marker_bank,
            "readers": output_row_readers,
            "betas": output_row_betas,
            "diagnostic_record_readers": reader_bank,
            "selected_output_rows": selected_output_rows,
            "output_delta": actual_output_delta,
            "residual_basis_bank": residual_basis_bank_cpu,
            "residual_basis_reports": residual_basis_reports,
            "selected_output_coefficients": output_coefficients.detach().cpu(),
            "selected_relative_row_norm_cap": selected_relative_cap,
            "relative_output_row_norms": actual_relative_output_row_norms,
            "cap_reports": cap_reports,
            "unconstrained_fallback_used": False,
            "base_selected_output_rows": base_selected_output_rows.detach().cpu(),
            "edited_selected_output_rows": (
                output_layer.weight.index_select(0, output_row_index).detach().cpu()
            ),
            "context_manifest_sha256": sha256_file(
                out_dir / "context_manifest.json"
            ),
        },
        out_dir / "compositional_marker_state.pt",
    )

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_num": len(records),
        "model_path": str(Path(a.model_path).resolve()),
        "architecture": {
            "router_used": False,
            "sidecar_used": False,
            "logit_bias_used": False,
            "lora_used": False,
            "adapter_used": False,
            "transformer_trainable_parameters": 0,
            "transformer_parameter_absdiff": transformer_absdiff,
            "input_embedding_rows_edited": len(selected_rows),
            "lm_head_rows_edited": len(selected_output_rows),
            "writer": "ordinary global sparse subject embedding-row deltas",
            "reader": (
                "per-sensitive-token q_y constrained to a rank-bounded, "
                "protected writer-residual basis"
            ),
            "decoder": "ordinary sparse LM-head rows, exactly Delta W_y=-beta_y q_y",
        },
        "data_firewall": context_manifest["data_access"],
        "contexts": context_report,
        "stage0_markers": {
            "selection": (
                "generalized eigenvector maximizing multi-context positive "
                "reachability over shared-subword/compositional negative reachability"
            ),
            "per_record": marker_reports,
        },
        "stage1": {
            "steps": int(a.writer_steps),
            "resumed": resume_path is not None,
            "resumed_from": str(resume_path) if resume_path is not None else None,
            "selected_embedding_rows": selected_rows,
            "embedding_delta_norm": float(trained_embedding_delta.norm()),
            "write_alpha": float(a.write_alpha),
            "row_norm_cap": float(a.row_norm_cap),
            "writer_selectivity": writer_report,
        },
        "record_reader_diagnostic": gate_report,
        "reader_gate": output_reader_gate,
        "causal_writer_ablation": causal_writer_ablation,
        "stage2": {
            "parameterization": (
                "row-specific writer-residual bases with hard relative norm "
                "balls; post-solve exact factorization Delta W_y=-beta_y q_y"
            ),
            "initialization": "zero output-row delta",
            "pre_margin": pre_margin_report,
            "exact_margin_optimization": True,
            "forget_margin": float(a.forget_margin),
            "base_positive_writer_dependence_weight": float(
                a.stage2_base_positive_weight
            ),
            "reference_nll_weight": float(a.stage2_reference_nll_weight),
            "reference_nll_tolerance": float(
                a.stage2_reference_nll_tolerance
            ),
            "reference_nll_drift": distribution(
                [float(x) for x in final_reference_nll_drift]
            ),
            "protection_subspace": {
                "common_rank": int(common_protection_basis_cpu.shape[0]),
                "common_source_states": int(common_protection_rows_cpu.shape[0]),
                "writer_off_positive_states": int(base_positive_rows_cpu.shape[0]),
                "corpus_states": int(corpus_protection_rows.shape[0]),
                "row_negative_states": distribution(
                    [int(rows.shape[0]) for rows in row_negative_states_cpu]
                ),
            },
            "residual_basis_rank_limit": int(a.stage2_residual_rank),
            "row_negative_rank_limit": int(a.stage2_row_negative_rank),
            "residual_basis": residual_basis_reports,
            "predeclared_relative_row_norm_caps": [
                float(x) for x in a.stage2_row_norm_cap_values
            ],
            "selected_relative_row_norm_cap": selected_relative_cap,
            "bounded_candidate_feasible": bool(feasible),
            "actual_relative_row_norms": distribution(
                [float(x) for x in actual_relative_output_row_norms]
            ),
            "hard_row_cap_passed": output_row_cap_passed,
            "unconstrained_fallback_used": False,
            "selected_output_rows": selected_output_rows,
            "beta": [float(x) for x in output_row_betas],
            "output_delta_norm": float(actual_output_delta.norm()),
            "output_materialization_delta_abs_error_max": (
                output_materialization_delta_error
            ),
            "optimization_log": stage2_log,
            "cap_reports": cap_reports,
            "direct_failures": direct_failures,
            "training_safe_positive_failures": positive_failures,
            "minimum_margin": float(materialized_margins.min()),
            "hook_to_materialized_margin_drift": margin_drift,
        },
        "acceptance": {
            "direct_eff_training_proxy_zero": direct_failures == 0,
            "all_training_safe_positive_failures_zero": positive_failures == 0,
            "reference_nll_abs_drift_within_tolerance": bool(
                final_reference_nll_abs_drift_max
                <= float(a.stage2_reference_nll_tolerance) + 1e-6
            ),
            "record_reader_diagnostic_passed": bool(gate_report["passed"]),
            "output_reader_gate_passed": bool(output_reader_gate["passed"]),
            "hard_output_row_cap_passed": output_row_cap_passed,
            "bounded_candidate_feasible": bool(feasible),
            "checkpoint_saved": checkpoint_saved,
            "passed": bool(bounded_training_passed and reader_policy_passed),
        },
        "checkpoint": str(checkpoint_path) if checkpoint_saved else None,
        "claim_boundary": (
            "Standard weight-level sparse edit with no runtime scope mechanism. "
            "Official unseen Gen, Spe, retain, and PPL are not known until the "
            "separately frozen checkpoint is evaluated."
        ),
    }
    gagd.write_json(out_dir / "compositional_marker_summary.json", summary)
    print("\n" + "=" * 72)
    print(f"  direct training failures             : {direct_failures}")
    print(f"  all training-safe positive failures  : {positive_failures}")
    print(f"  minimum training margin              : {float(materialized_margins.min()):+.4f}")
    print(
        "  maximum reference NLL regression     : "
        f"{final_reference_nll_abs_drift_max:.4f}"
    )
    print(
        "  output-only direct/positive failures : "
        f"{output_only_direct_failures}/{output_only_positive_failures}"
    )
    print(
        "  selected / actual max row cap        : "
        f"{selected_relative_cap:.3f} / "
        f"{float(actual_relative_output_row_norms.max()):.6f}"
    )
    print(f"  Transformer parameter absdiff        : {transformer_absdiff:.3e}")
    print("  router / bias / sidecar              : False / False / False")
    print("=" * 72)
    if (
        direct_failures
        or positive_failures
        or final_reference_nll_abs_drift_max
        > float(a.stage2_reference_nll_tolerance) + 1e-6
        or not output_row_cap_passed
        or not feasible
        or not reader_policy_passed
    ):
        raise SystemExit(
            "bounded Stage 2 did not satisfy every training-only constraint; "
            "checkpoint is diagnostic only and official evaluation is refused"
        )
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {out_dir / 'compositional_marker_summary.json'}")


if __name__ == "__main__":
    main()
