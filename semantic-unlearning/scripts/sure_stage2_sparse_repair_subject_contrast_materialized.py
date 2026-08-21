#!/usr/bin/env python3
"""Materialization-safe MCF Stage 2 using subject-contrast hidden directions.

This is an isolated SURE-LM ablation for the unswapped MCF contract:

* ``target_true`` is the sensitive / unwanted answer;
* ``target_new`` is the non-sensitive reference answer;
* only direct training-visible forget records are used;
* only sensitive ``target_true`` LM-head rows are editable;
* no paraphrases, neighborhoods, benchmark-retain examples, Wikipedia utility
  examples, or PPL text are used for basis construction or repair selection.

Why subject contrast instead of ``H_target_true - H_target_new``?
For the first answer token of a causal LM, both candidate answers are scored from
exactly the same prompt hidden state, so subtracting candidate-labelled hidden
states is identically zero for single-token MCF answers.  Here we instead
contrast two *different prompt states at the actual answer-prediction point*.

For each active direct record i with relation template q_i and subject s_i, pick
a deterministic donor subject s_j from another direct training-visible forget
record and construct

    d_i = h(q_i, s_i) - h(q_i, s_j).

The leading right-singular directions of the matrix D whose rows are d_i define
a fixed hidden basis.  Sparse output repair is parameterized as

    Delta W_sensitive = C B_subject-contrast.

This isolates subject-conditioned evidence under the same relation wording while
keeping the locked evaluation protocol intact.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
import sure_stage2_sparse_repair as shared
import sure_stage2_sparse_repair_materialized as materialized
from mcf_zero_unlearn_official_eval import is_llama_like


METHOD = "SURE-LM-MCF-target-true-sensitive-subject-contrast-hidden-basis-stage2"
PROTOCOL = "mcf_target_true_sensitive_subject_contrast_hidden_basis_v1"


def _parse_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--subject-control-count", type=int, default=4)
    opts, remaining = p.parse_known_args(list(argv))
    if opts.subject_control_count <= 0:
        p.error("subject-control-count must be positive")
    return opts, remaining


def _parse_shared_stage_args(argv: Sequence[str]) -> argparse.Namespace:
    _wrapper, shared_argv = materialized._split_wrapper_args(argv)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(__file__).resolve()), *shared_argv]
        return shared.parse_args()
    finally:
        sys.argv = old_argv


def _assert_target_contract(stage_args: argparse.Namespace, manifest: Dict[str, Any]) -> None:
    if stage_args.dataset != "mcf":
        raise RuntimeError("subject-contrast hidden-basis wrapper is MCF-only")
    if stage_args.mcf_sensitive_field != "target_true":
        raise RuntimeError("this ablation requires target_true = sensitive/unwanted")
    if stage_args.mcf_reference_field != "target_new":
        raise RuntimeError("this ablation requires target_new = non-sensitive reference")
    ranks = shared.core.parse_rank_list(stage_args.candidate_ranks)
    if 0 in ranks:
        raise RuntimeError(
            "rank 0 is unrestricted and would violate the subject-contrast ablation; "
            "use positive ranks such as 1,2,4,8"
        )

    contract = manifest.get("target_contract", {})
    if contract:
        sensitive = contract.get("sensitive_answer")
        reference = contract.get("non_sensitive_reference")
        swapping = contract.get("field_swapping")
        if sensitive not in (None, "requested_rewrite.target_true"):
            raise RuntimeError(f"split manifest sensitive contract mismatch: {sensitive!r}")
        if reference not in (None, "requested_rewrite.target_new"):
            raise RuntimeError(f"split manifest reference contract mismatch: {reference!r}")
        if swapping not in (None, False):
            raise RuntimeError("split manifest unexpectedly enables field swapping")


@torch.no_grad()
def _last_hidden_for_prompts(
    model: torch.nn.Module,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(prompts), int(batch_size)):
        batch = list(prompts[start : start + int(batch_size)])
        encoded = tok(batch, padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=device)
        chunks.append(
            output.hidden_states[-1][rows, positions, :]
            .detach()
            .float()
            .cpu()
        )
    if not chunks:
        return torch.empty((0, 0), dtype=torch.float32)
    return torch.cat(chunks, dim=0).contiguous()


def _subjects(records: Sequence[Mapping[str, Any]]) -> List[str]:
    values: List[str] = []
    for i, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"record {i} lacks requested_rewrite")
        subject = str(rr.get("subject", "")).strip()
        if not subject:
            raise RuntimeError(f"record {i} has empty subject")
        values.append(subject)
    return values


def _donor_indices(
    position: int,
    subjects: Sequence[str],
    count: int,
) -> List[int]:
    if len(subjects) < 2:
        raise RuntimeError("subject contrast requires at least two direct records")
    donors: List[int] = []
    offset = 1
    while len(donors) < min(int(count), len(subjects) - 1):
        j = (int(position) + offset) % len(subjects)
        offset += 1
        if j == position or subjects[j] == subjects[position] or j in donors:
            if offset > len(subjects) * 3 and not donors:
                raise RuntimeError("could not find a distinct donor subject")
            continue
        donors.append(j)
    return donors


def build_subject_contrast_basis(
    differences: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if differences.ndim != 2:
        raise ValueError("subject-contrast matrix must be rank-2")
    if differences.shape[0] == 0:
        hidden = differences.shape[1] if differences.ndim == 2 else 0
        return differences.new_empty((0, hidden), dtype=torch.float32), {
            "active_fact_count": 0,
            "subject_contrast_rank": 0,
            "singular_values": [],
        }

    d = differences.detach().float().cpu().contiguous()
    _u, singular_values, right = torch.linalg.svd(d, full_matrices=False)
    tolerance = (
        max(d.shape)
        * torch.finfo(d.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    rank = int((singular_values > tolerance).sum().item())
    if rank <= 0:
        raise RuntimeError("subject-contrast hidden differences have zero rank")
    return right[:rank].float().contiguous(), {
        "active_fact_count": int(d.shape[0]),
        "subject_contrast_rank": int(rank),
        "singular_values": [float(v) for v in singular_values[: min(32, rank)].tolist()],
        "direction": "h(original direct prompt) - mean h(subject-swapped direct controls)",
    }


@torch.no_grad()
def _precompute_subject_contrast(
    stage_args: argparse.Namespace,
    *,
    subject_control_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    records, manifest = shared.load_locked(
        "mcf",
        Path(stage_args.training_visible_path).resolve(),
        Path(stage_args.split_manifest).resolve(),
        int(stage_args.seed),
        int(stage_args.forget_num),
    )
    _assert_target_contract(stage_args, manifest)

    ns = argparse.Namespace(
        model_path=stage_args.model_path,
        dtype=stage_args.dtype,
        device_map=stage_args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    instances = shared.mcf_instances(records)

    margins = shared.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        int(stage_args.batch_size),
        "target_true",
        "target_new",
    )
    active_positions = [
        i
        for i, value in enumerate(margins.detach().cpu().tolist())
        if float(value) < float(stage_args.constraint_margin)
    ]

    hidden_size = int(model.get_output_embeddings().weight.shape[1])
    selected_ids = shared.mcf_sensitive_rows(tok, instances, active_positions, "target_true")
    if active_positions and not selected_ids:
        raise RuntimeError("active target_true cases produced no sensitive rows")

    subjects = _subjects(records)
    original_prompts: List[str] = []
    control_groups: List[List[str]] = []
    donor_receipt: List[Dict[str, Any]] = []

    for position in active_positions:
        rr = records[position]["requested_rewrite"]
        template = str(rr["prompt"])
        own_subject = subjects[position]
        original_prompts.append(template.format(own_subject))

        donor_ids = _donor_indices(position, subjects, int(subject_control_count))
        controls = [template.format(subjects[j]) for j in donor_ids]
        control_groups.append(controls)
        donor_receipt.append(
            {
                "record_position": int(position),
                "original_subject": own_subject,
                "donor_positions": [int(j) for j in donor_ids],
                "donor_subjects": [subjects[j] for j in donor_ids],
            }
        )

    if original_prompts:
        original_hidden = _last_hidden_for_prompts(
            model, tok, original_prompts, device, int(stage_args.batch_size)
        )
        flat_controls = [p for group in control_groups for p in group]
        flat_hidden = _last_hidden_for_prompts(
            model, tok, flat_controls, device, int(stage_args.batch_size)
        )
        control_means: List[torch.Tensor] = []
        cursor = 0
        for group in control_groups:
            n = len(group)
            control_means.append(flat_hidden[cursor : cursor + n].mean(dim=0))
            cursor += n
        control_hidden = torch.stack(control_means, dim=0).float().contiguous()
        differences = (original_hidden - control_hidden).float().contiguous()
    else:
        original_hidden = torch.empty((0, hidden_size), dtype=torch.float32)
        control_hidden = torch.empty((0, hidden_size), dtype=torch.float32)
        differences = torch.empty((0, hidden_size), dtype=torch.float32)

    receipt = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "active_constraint_margin": float(stage_args.constraint_margin),
        "active_direct_positions": [int(x) for x in active_positions],
        "active_direct_count": int(len(active_positions)),
        "selected_sensitive_target_true_rows": int(len(selected_ids)),
        "hidden_size": int(hidden_size),
        "subject_control_count_requested": int(subject_control_count),
        "basis_source": (
            "direct prompt final hidden state minus mean of same relation template with deterministic donor subjects"
        ),
        "donors": donor_receipt,
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "external_wikipedia_seen": 0,
        "PPL_seen": False,
    }

    del model
    del tok
    del margins
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return differences, original_hidden, control_hidden, receipt


def _rank_diagnostics(
    basis: torch.Tensor,
    differences: torch.Tensor,
    original_hidden: torch.Tensor,
    control_hidden: torch.Tensor,
) -> Dict[str, float]:
    b = basis.detach().float().cpu()

    def energy(x: torch.Tensor) -> float:
        if b.numel() == 0 or x.numel() == 0:
            return 0.0
        projected = x.detach().float().cpu() @ b.transpose(0, 1)
        return float(projected.square().sum(dim=1).mean())

    return {
        "subject_contrast_projection_energy": energy(differences),
        "original_prompt_projection_energy": energy(original_hidden),
        "control_prompt_projection_energy": energy(control_hidden),
    }


def main(argv: Sequence[str] | None = None) -> None:
    opts, forwarded = _parse_wrapper_args(sys.argv[1:] if argv is None else argv)
    stage_args = _parse_shared_stage_args(forwarded)

    differences, original_hidden, control_hidden, receipt = _precompute_subject_contrast(
        stage_args,
        subject_control_count=int(opts.subject_control_count),
    )
    if differences.shape[0] > 0:
        full_basis, basis_receipt = build_subject_contrast_basis(differences)
    else:
        full_basis = torch.empty((0, int(receipt["hidden_size"])), dtype=torch.float32)
        basis_receipt = {
            "active_fact_count": 0,
            "subject_contrast_rank": 0,
            "singular_values": [],
        }

    requested_rank_reports: Dict[str, Any] = {}
    original_rank_basis = shared._rank_basis

    def subject_rank_basis(
        _shared_active_hidden: torch.Tensor,
        requested_rank: int,
    ) -> Tuple[torch.Tensor | None, int]:
        if requested_rank <= 0:
            raise RuntimeError(
                "subject-contrast repair requires positive rank; unrestricted rank 0 is disabled"
            )
        actual = min(int(requested_rank), int(full_basis.shape[0]))
        if actual <= 0:
            raise RuntimeError("subject-contrast sensitive basis has zero rank")
        basis = full_basis[:actual].to(
            device=_shared_active_hidden.device,
            dtype=torch.float32,
        )
        requested_rank_reports[str(requested_rank)] = {
            "requested_rank": int(requested_rank),
            "actual_rank": int(actual),
            **_rank_diagnostics(
                full_basis[:actual], differences, original_hidden, control_hidden
            ),
        }
        return basis, actual

    try:
        shared._rank_basis = subject_rank_basis
        materialized.main(forwarded)
    finally:
        shared._rank_basis = original_rank_basis

    output_dir = Path(stage_args.output_dir).resolve()
    summary_path = output_dir / "repair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "subject_contrast_basis_method": METHOD,
            "subject_contrast_basis_protocol": PROTOCOL,
            "subject_contrast_basis_mode": "original_subject_vs_same_relation_donor_subjects",
            "subject_contrast_target_contract": receipt["target_contract"],
            "subject_contrast_basis_receipt": basis_receipt,
            "subject_contrast_rank_reports": requested_rank_reports,
            "subject_control_count": int(opts.subject_control_count),
            "subject_contrast_unrestricted_rank0_disabled": True,
            "benchmark_retain_seen_for_basis": 0,
            "heldout_paraphrases_seen_for_basis": 0,
            "locality_or_neighborhood_seen_for_basis": 0,
            "external_wikipedia_seen_for_basis": 0,
            "PPL_seen_for_basis": False,
        }
    )
    shared.core.write_json(summary_path, summary)
    shared.core.write_json(
        output_dir / "subject_contrast_basis_receipt.json",
        {
            **receipt,
            "basis": basis_receipt,
            "rank_reports": requested_rank_reports,
        },
    )
    print(
        "Subject-contrast target_true-sensitive Stage2 complete: "
        f"active facts={basis_receipt.get('active_fact_count')}; "
        f"basis rank={basis_receipt.get('subject_contrast_rank')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
