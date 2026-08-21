#!/usr/bin/env python3
"""Materialization-safe MCF Stage 2 using target_true-minus-target_new hidden directions.

This is an isolated SURE-LM ablation for the unswapped MCF contract:

* ``target_true`` is the sensitive / unwanted answer;
* ``target_new`` is the non-sensitive reference answer;
* only direct training-visible forget records are used;
* only sensitive ``target_true`` LM-head rows are editable;
* no paraphrases, neighborhoods, benchmark-retain examples, or PPL text are
  used for basis construction, optimization, scale selection, or checkpoint
  selection.

For each active direct forget record i, the paired fact direction is

    d_i = mean(H_target_true_i) - mean(H_target_new_i),

where each mean is over that answer's teacher-forced prediction hidden states.
The fixed repair basis is the leading right-singular directions of the matrix
D whose rows are d_i.  Sparse row repair is then parameterized as

    Delta W_sensitive = C B_true-minus-new.

A sequence mean is used because target_true and target_new can tokenize to
different numbers of answer tokens; this yields exactly one paired direction
per active deletion fact without inventing token alignments.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import gagd_compare as gagd
import sure_stage2_sparse_repair as shared
import sure_stage2_sparse_repair_materialized as materialized
from mcf_zero_unlearn_official_eval import is_llama_like


METHOD = "SURE-LM-MCF-target-true-minus-target-new-hidden-basis-stage2"
PROTOCOL = "mcf_target_true_sensitive_true_minus_new_hidden_basis_v1"


def _parse_shared_stage_args(argv: Sequence[str]) -> argparse.Namespace:
    _wrapper, shared_argv = materialized._split_wrapper_args(argv)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(Path(__file__).resolve()), *shared_argv]
        return shared.parse_args()
    finally:
        sys.argv = old_argv


def _assert_target_contract(
    stage_args: argparse.Namespace, manifest: Dict[str, Any]
) -> None:
    if stage_args.dataset != "mcf":
        raise RuntimeError("true-minus-new hidden-basis wrapper is MCF-only")
    if stage_args.mcf_sensitive_field != "target_true":
        raise RuntimeError(
            "this ablation requires target_true = sensitive/unwanted"
        )
    if stage_args.mcf_reference_field != "target_new":
        raise RuntimeError(
            "this ablation requires target_new = non-sensitive reference"
        )
    ranks = shared.core.parse_rank_list(stage_args.candidate_ranks)
    if 0 in ranks:
        raise RuntimeError(
            "rank 0 is unrestricted and would violate the paired-direction "
            "ablation; use positive ranks such as 1,2,4,8"
        )

    contract = manifest.get("target_contract", {})
    if contract:
        sensitive = contract.get("sensitive_answer")
        reference = contract.get("non_sensitive_reference")
        swapping = contract.get("field_swapping")
        if sensitive not in (None, "requested_rewrite.target_true"):
            raise RuntimeError(
                f"split manifest sensitive contract mismatch: {sensitive!r}"
            )
        if reference not in (None, "requested_rewrite.target_new"):
            raise RuntimeError(
                f"split manifest reference contract mismatch: {reference!r}"
            )
        if swapping not in (None, False):
            raise RuntimeError("split manifest unexpectedly enables field swapping")


def build_true_minus_new_basis(
    paired_differences: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return SVD-ordered orthonormal directions spanning paired differences."""
    if paired_differences.ndim != 2:
        raise ValueError("paired difference matrix must be rank-2")
    if paired_differences.shape[0] == 0:
        hidden = paired_differences.shape[1] if paired_differences.ndim == 2 else 0
        return paired_differences.new_empty((0, hidden), dtype=torch.float32), {
            "paired_fact_count": 0,
            "paired_difference_rank": 0,
            "singular_values": [],
        }

    d = paired_differences.detach().float().cpu().contiguous()
    _u, singular_values, right = torch.linalg.svd(d, full_matrices=False)
    tolerance = (
        max(d.shape)
        * torch.finfo(d.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    rank = int((singular_values > tolerance).sum().item())
    if rank <= 0:
        raise RuntimeError("paired target_true-target_new differences have zero rank")
    basis = right[:rank].float().contiguous()
    return basis, {
        "paired_fact_count": int(d.shape[0]),
        "paired_difference_rank": int(rank),
        "singular_values": [float(v) for v in singular_values[: min(32, rank)].tolist()],
        "sequence_reduction": "mean teacher-forced answer hidden states per answer",
        "paired_direction": "mean(H_target_true) - mean(H_target_new)",
    }


@torch.no_grad()
def _precompute_paired_hidden(
    stage_args: argparse.Namespace,
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
    selected_ids = shared.mcf_sensitive_rows(
        tok,
        instances,
        active_positions,
        "target_true",
    )
    if active_positions and not selected_ids:
        raise RuntimeError("active target_true cases produced no sensitive rows")

    true_means: List[torch.Tensor] = []
    new_means: List[torch.Tensor] = []
    differences: List[torch.Tensor] = []
    token_counts: List[Dict[str, int]] = []

    if active_positions:
        caches = shared.mcf_repair.build_prompt_instance_delta_caches(
            model,
            tok,
            instances,
            selected_ids,
            device,
            int(stage_args.batch_size),
            llama_like,
        )
        for position in active_positions:
            true_hidden = caches[position].target_true.hidden.detach().float()
            new_hidden = caches[position].target_new.hidden.detach().float()
            if true_hidden.ndim != 2 or new_hidden.ndim != 2:
                raise RuntimeError("answer hidden cache has unexpected shape")
            h_true = true_hidden.mean(dim=0)
            h_new = new_hidden.mean(dim=0)
            true_means.append(h_true.cpu())
            new_means.append(h_new.cpu())
            differences.append((h_true - h_new).cpu())
            token_counts.append(
                {
                    "record_position": int(position),
                    "target_true_tokens": int(true_hidden.shape[0]),
                    "target_new_tokens": int(new_hidden.shape[0]),
                }
            )
        del caches

    if differences:
        d = torch.stack(differences, dim=0).float().contiguous()
        ht = torch.stack(true_means, dim=0).float().contiguous()
        hn = torch.stack(new_means, dim=0).float().contiguous()
    else:
        d = torch.empty((0, hidden_size), dtype=torch.float32)
        ht = torch.empty((0, hidden_size), dtype=torch.float32)
        hn = torch.empty((0, hidden_size), dtype=torch.float32)

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
        "active_direct_positions": active_positions,
        "active_direct_count": int(len(active_positions)),
        "selected_sensitive_target_true_rows": int(len(selected_ids)),
        "hidden_size": int(hidden_size),
        "paired_hidden_source": (
            "same direct training-visible MCF record; teacher-forced target_true and target_new answer states"
        ),
        "paired_direction": "mean(H_target_true) - mean(H_target_new)",
        "paired_fact_count": int(d.shape[0]),
        "token_counts": token_counts,
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "locality_or_neighborhood_seen": 0,
        "PPL_seen": False,
    }

    del model
    del tok
    del margins
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return d, ht, hn, receipt


def _rank_diagnostics(
    basis: torch.Tensor,
    differences: torch.Tensor,
    true_means: torch.Tensor,
    new_means: torch.Tensor,
) -> Dict[str, float]:
    b = basis.detach().float().cpu()
    if b.numel() == 0:
        return {
            "paired_difference_projection_energy": 0.0,
            "target_true_projection_energy": 0.0,
            "target_new_projection_energy": 0.0,
        }

    def energy(x: torch.Tensor) -> float:
        if x.numel() == 0:
            return 0.0
        projected = x.detach().float().cpu() @ b.transpose(0, 1)
        return float(projected.square().sum(dim=1).mean())

    return {
        "paired_difference_projection_energy": energy(differences),
        "target_true_projection_energy": energy(true_means),
        "target_new_projection_energy": energy(new_means),
    }


def main(argv: Sequence[str] | None = None) -> None:
    forwarded = list(sys.argv[1:] if argv is None else argv)
    stage_args = _parse_shared_stage_args(forwarded)

    differences, true_means, new_means, receipt = _precompute_paired_hidden(stage_args)
    if differences.shape[0] > 0:
        full_basis, basis_receipt = build_true_minus_new_basis(differences)
    else:
        full_basis = torch.empty(
            (0, int(receipt["hidden_size"])), dtype=torch.float32
        )
        basis_receipt = {
            "paired_fact_count": 0,
            "paired_difference_rank": 0,
            "singular_values": [],
        }

    requested_rank_reports: Dict[str, Any] = {}
    original_rank_basis = shared._rank_basis

    def paired_rank_basis(
        _shared_active_hidden: torch.Tensor,
        requested_rank: int,
    ) -> Tuple[torch.Tensor | None, int]:
        if requested_rank <= 0:
            raise RuntimeError(
                "paired-direction repair requires positive rank; unrestricted rank 0 is disabled"
            )
        actual = min(int(requested_rank), int(full_basis.shape[0]))
        if actual <= 0:
            raise RuntimeError("paired true-minus-new basis has zero rank")
        basis = full_basis[:actual].to(
            device=_shared_active_hidden.device,
            dtype=torch.float32,
        )
        requested_rank_reports[str(requested_rank)] = {
            "requested_rank": int(requested_rank),
            "actual_rank": int(actual),
            **_rank_diagnostics(
                full_basis[:actual], differences, true_means, new_means
            ),
        }
        return basis, actual

    try:
        shared._rank_basis = paired_rank_basis
        materialized.main(forwarded)
    finally:
        shared._rank_basis = original_rank_basis

    output_dir = Path(stage_args.output_dir).resolve()
    summary_path = output_dir / "repair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "paired_basis_method": METHOD,
            "paired_basis_protocol": PROTOCOL,
            "paired_basis_mode": "target_true_mean_minus_target_new_mean",
            "paired_target_contract": receipt["target_contract"],
            "paired_hidden_source": receipt["paired_hidden_source"],
            "paired_basis_receipt": basis_receipt,
            "paired_rank_reports": requested_rank_reports,
            "paired_unrestricted_rank0_disabled": True,
            "benchmark_retain_seen_for_basis": 0,
            "heldout_paraphrases_seen_for_basis": 0,
            "locality_or_neighborhood_seen_for_basis": 0,
            "PPL_seen_for_basis": False,
        }
    )
    shared.core.write_json(summary_path, summary)
    shared.core.write_json(
        output_dir / "true_minus_new_basis_receipt.json",
        {
            **receipt,
            "basis": basis_receipt,
            "rank_reports": requested_rank_reports,
        },
    )
    print(
        "Target_true-minus-target_new Stage2 complete: "
        f"paired facts={basis_receipt.get('paired_fact_count')}; "
        f"paired rank={basis_receipt.get('paired_difference_rank')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
