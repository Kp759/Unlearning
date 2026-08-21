#!/usr/bin/env python3
"""Materialization-safe MCF Stage 2 with a target-true-sensitive contrastive basis.

This is an isolated SURE-LM ablation for the unswapped MCF contract:

* ``target_true`` is the sensitive / unwanted answer;
* ``target_new`` is the non-sensitive reference answer;
* residual direct failures are detected exactly as in the canonical target-true
  Stage 2;
* only sensitive ``target_true`` LM-head rows are editable;
* the fixed repair basis is estimated from hidden states that predict
  ``target_true`` on the active direct forget requests;
* 200 external Wikipedia contexts (default) provide a background utility
  covariance, with the first 20 rows excluded so the fixed PPL probe is not
  training-visible;
* the basis maximizes sensitive hidden-state energy relative to utility energy
  inside the span of the sensitive hidden states;
* official paraphrases, neighborhoods, benchmark-retain records, and PPL text
  remain unavailable to repair and basis selection.

For sensitive hidden matrix H_f and utility hidden matrix H_0, first let Q span
row(H_f).  In Q coordinates we solve

    C_f v = lambda (C_0 + eps I) v,

where C_f and C_0 are uncentered second-moment matrices.  The selected top-r
generalized eigenvectors are mapped back through Q and orthonormalized before
parameterizing the sparse row update ``Delta W = C B_sensitive``.

The generalized eigenproblem is solved only inside row(H_f), so its dimension
is bounded by the number of active sensitive prediction states rather than the
full 3072-dimensional model hidden size.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import gagd_compare as gagd
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as shared
import sure_stage2_sparse_repair_materialized as materialized
from mcf_zero_unlearn_official_eval import is_llama_like


METHOD = "SURE-LM-MCF-target-true-sensitive-contrastive-hidden-basis-stage2"
PROTOCOL = "mcf_target_true_sensitive_contrastive_hidden_basis_wikipedia_v1"


def _split_contrastive_args(
    argv: Sequence[str],
) -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--contrastive-utility-wikipedia-dir", required=True)
    p.add_argument("--contrastive-utility-sample-size", type=int, default=200)
    p.add_argument("--contrastive-utility-batch-size", type=int, default=8)
    p.add_argument("--contrastive-utility-max-length", type=int, default=128)
    p.add_argument("--contrastive-utility-seed", type=int, default=1)
    p.add_argument("--contrastive-utility-exclude-first", type=int, default=20)
    p.add_argument("--contrastive-ridge-ratio", type=float, default=1e-3)
    opts, remaining = p.parse_known_args(list(argv))

    if opts.contrastive_utility_sample_size <= 0:
        p.error("contrastive utility sample size must be positive")
    if opts.contrastive_utility_batch_size <= 0:
        p.error("contrastive utility batch size must be positive")
    if opts.contrastive_utility_max_length < 8:
        p.error("contrastive utility max length must be at least 8")
    if opts.contrastive_utility_exclude_first < 20:
        p.error(
            "contrastive utility must exclude at least the first 20 Wikipedia rows "
            "used by the fixed PPL probe"
        )
    if (
        not math.isfinite(opts.contrastive_ridge_ratio)
        or opts.contrastive_ridge_ratio <= 0
    ):
        p.error("contrastive ridge ratio must be finite and positive")
    return opts, remaining


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
        raise RuntimeError("contrastive hidden-basis wrapper is MCF-only")
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
            "rank 0 is unrestricted and would violate the contrastive-basis "
            "ablation; use positive ranks such as 2,8,16"
        )

    contract = manifest.get("target_contract", {})
    if contract:
        sensitive = contract.get("sensitive_answer")
        reference = contract.get("non_sensitive_reference")
        if sensitive not in (None, "requested_rewrite.target_true"):
            raise RuntimeError(
                f"split manifest sensitive contract mismatch: {sensitive!r}"
            )
        if reference not in (None, "requested_rewrite.target_new"):
            raise RuntimeError(
                f"split manifest reference contract mismatch: {reference!r}"
            )


@torch.no_grad()
def _utility_last_hidden(
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
        encoded = tok(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
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


def _precompute_hidden_sets(
    stage_args: argparse.Namespace,
    opts: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
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
        # Critical semantic choice: H_f contains ONLY the hidden states that
        # teacher-force the sensitive target_true answer, never target_new.
        sensitive_hidden = torch.cat(
            [caches[position].target_true.hidden for position in active_positions],
            dim=0,
        ).detach().float().cpu().contiguous()
        del caches
    else:
        sensitive_hidden = torch.empty((0, hidden_size), dtype=torch.float32)

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(opts.contrastive_utility_wikipedia_dir).resolve(),
        sample_size=int(opts.contrastive_utility_sample_size),
        seed=int(opts.contrastive_utility_seed),
        exclude_first=int(opts.contrastive_utility_exclude_first),
        max_length=int(opts.contrastive_utility_max_length),
    )
    print(
        "Contrastive Stage2: collecting "
        f"{len(utility_prompts)} external Wikipedia utility hidden states...",
        flush=True,
    )
    utility_hidden = _utility_last_hidden(
        model,
        tok,
        utility_prompts,
        device,
        int(opts.contrastive_utility_batch_size),
    )

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
        "active_direct_count": len(active_positions),
        "selected_sensitive_target_true_rows": len(selected_ids),
        "sensitive_hidden_source": "active direct target_true teacher-forced answer positions only",
        "sensitive_hidden_count": int(sensitive_hidden.shape[0]),
        "hidden_size": hidden_size,
        "utility_hidden_source": "external Wikipedia final hidden state at last prompt token",
        "utility_hidden_count": int(utility_hidden.shape[0]),
        "utility_receipt": utility_receipt,
        "ridge_ratio": float(opts.contrastive_ridge_ratio),
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
    return sensitive_hidden, utility_hidden, receipt


def build_contrastive_basis(
    sensitive_hidden: torch.Tensor,
    utility_hidden: torch.Tensor,
    *,
    ridge_ratio: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Return ordered sensitive-specific directions in row(H_f)."""
    if sensitive_hidden.ndim != 2 or utility_hidden.ndim != 2:
        raise ValueError("hidden matrices must be rank-2")
    if sensitive_hidden.shape[0] == 0:
        return sensitive_hidden.new_empty((0, sensitive_hidden.shape[1])), {
            "forget_span_rank": 0,
            "generalized_eigenvalues": [],
        }
    if utility_hidden.shape[0] == 0:
        raise ValueError("utility hidden matrix must be non-empty")
    if sensitive_hidden.shape[1] != utility_hidden.shape[1]:
        raise ValueError("sensitive and utility hidden sizes differ")

    hf = sensitive_hidden.detach().float().cpu()
    h0 = utility_hidden.detach().float().cpu()
    # Q has orthonormal rows and spans exactly the sensitive hidden-state row
    # space.  The generalized eigenproblem is therefore small.
    q = shared.core.orthonormal_row_basis(hf, max_rank=None).cpu().double()
    span_rank = int(q.shape[0])
    if span_rank <= 0:
        raise RuntimeError("sensitive hidden states have zero numerical rank")

    f = hf.double() @ q.transpose(0, 1)
    u = h0.double() @ q.transpose(0, 1)
    cf = (f.transpose(0, 1) @ f) / float(max(1, f.shape[0]))
    c0 = (u.transpose(0, 1) @ u) / float(max(1, u.shape[0]))

    utility_scale = float(torch.diagonal(c0).mean().clamp_min(1e-12))
    ridge = float(ridge_ratio) * utility_scale
    eye = torch.eye(span_rank, dtype=torch.float64)
    denom = c0 + ridge * eye
    chol = torch.linalg.cholesky(denom)

    # Whiten denominator: A = L^{-1} C_f L^{-T}.
    left = torch.linalg.solve_triangular(chol, cf, upper=False)
    whitened = torch.linalg.solve_triangular(
        chol,
        left.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    whitened = 0.5 * (whitened + whitened.transpose(0, 1))

    evals, evecs = torch.linalg.eigh(whitened)
    order = torch.argsort(evals, descending=True)
    evals = evals.index_select(0, order)
    y = evecs.index_select(1, order)

    # Generalized vectors x = L^{-T} y, then map from Q coordinates back to
    # model hidden space.  QR preserves the nested top-r spans while making
    # the row basis Euclidean-orthonormal for SelectedRowDelta.
    x = torch.linalg.solve_triangular(
        chol.transpose(0, 1),
        y,
        upper=True,
    )
    raw = x.transpose(0, 1) @ q
    orthogonal_columns, _ = torch.linalg.qr(raw.transpose(0, 1), mode="reduced")
    basis = orthogonal_columns.transpose(0, 1).float().contiguous()

    return basis, {
        "forget_span_rank": span_rank,
        "utility_covariance_scale": utility_scale,
        "ridge_absolute": ridge,
        "generalized_eigenvalues": [
            float(v) for v in evals[: min(32, len(evals))].cpu().tolist()
        ],
    }


def _subspace_diagnostics(
    basis: torch.Tensor,
    sensitive_hidden: torch.Tensor,
    utility_hidden: torch.Tensor,
) -> Dict[str, float]:
    if basis.numel() == 0:
        return {
            "sensitive_projection_energy": 0.0,
            "utility_projection_energy": 0.0,
            "sensitive_to_utility_energy_ratio": 0.0,
        }
    b = basis.detach().float().cpu()
    f = sensitive_hidden.detach().float().cpu() @ b.transpose(0, 1)
    u = utility_hidden.detach().float().cpu() @ b.transpose(0, 1)
    ef = float(f.square().sum(dim=1).mean())
    eu = float(u.square().sum(dim=1).mean())
    return {
        "sensitive_projection_energy": ef,
        "utility_projection_energy": eu,
        "sensitive_to_utility_energy_ratio": ef / max(eu, 1e-12),
    }


def main(argv: Sequence[str] | None = None) -> None:
    opts, forwarded = _split_contrastive_args(
        sys.argv[1:] if argv is None else argv
    )
    stage_args = _parse_shared_stage_args(forwarded)

    sensitive_hidden, utility_hidden, receipt = _precompute_hidden_sets(
        stage_args, opts
    )
    if sensitive_hidden.shape[0] > 0:
        full_basis, eig_receipt = build_contrastive_basis(
            sensitive_hidden,
            utility_hidden,
            ridge_ratio=float(opts.contrastive_ridge_ratio),
        )
    else:
        full_basis = torch.empty(
            (0, int(receipt["hidden_size"])), dtype=torch.float32
        )
        eig_receipt = {
            "forget_span_rank": 0,
            "generalized_eigenvalues": [],
        }

    requested_rank_reports: Dict[str, Any] = {}
    original_rank_basis = shared._rank_basis

    def contrastive_rank_basis(
        _shared_active_hidden: torch.Tensor,
        requested_rank: int,
    ) -> Tuple[torch.Tensor | None, int]:
        if requested_rank <= 0:
            raise RuntimeError(
                "contrastive repair requires positive rank; unrestricted rank 0 is disabled"
            )
        actual = min(int(requested_rank), int(full_basis.shape[0]))
        if actual <= 0:
            raise RuntimeError("contrastive sensitive basis has zero rank")
        basis = full_basis[:actual].to(
            device=_shared_active_hidden.device,
            dtype=torch.float32,
        )
        requested_rank_reports[str(requested_rank)] = {
            "requested_rank": int(requested_rank),
            "actual_rank": int(actual),
            **_subspace_diagnostics(
                full_basis[:actual], sensitive_hidden, utility_hidden
            ),
        }
        return basis, actual

    try:
        shared._rank_basis = contrastive_rank_basis
        materialized.main(forwarded)
    finally:
        shared._rank_basis = original_rank_basis

    output_dir = Path(stage_args.output_dir).resolve()
    summary_path = output_dir / "repair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "contrastive_basis_method": METHOD,
            "contrastive_basis_protocol": PROTOCOL,
            "contrastive_basis_mode": "target_true_sensitive_vs_external_wikipedia",
            "contrastive_target_contract": receipt["target_contract"],
            "contrastive_sensitive_hidden_source": receipt[
                "sensitive_hidden_source"
            ],
            "contrastive_utility_hidden_source": receipt[
                "utility_hidden_source"
            ],
            "contrastive_utility_sample_size": int(
                opts.contrastive_utility_sample_size
            ),
            "contrastive_ridge_ratio": float(opts.contrastive_ridge_ratio),
            "contrastive_eigensystem": eig_receipt,
            "contrastive_rank_reports": requested_rank_reports,
            "contrastive_unrestricted_rank0_disabled": True,
        }
    )
    shared.core.write_json(summary_path, summary)
    shared.core.write_json(
        output_dir / "contrastive_basis_receipt.json",
        {
            **receipt,
            "eigensystem": eig_receipt,
            "rank_reports": requested_rank_reports,
        },
    )
    print(
        "Contrastive target_true-sensitive Stage2 complete: "
        f"forget-span rank={eig_receipt.get('forget_span_rank')}; "
        f"utility contexts={opts.contrastive_utility_sample_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
