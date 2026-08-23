#!/usr/bin/env python3
"""Run covariance SURE-v3 with KL-budgeted external relation locality.

This wrapper keeps the covariance-whitened Stage-2 core unchanged, but replaces
its brittle exact-top1 locality gate with an exact sparse-head KL gate.  The
budget is *not* a new hyperparameter: it is the same --max-protected-kl value
already used for Stage-1-success protection (0.05 by default).

For a locality hidden state h and edited LM-head rows A, only logits in A move.
Therefore the full-vocabulary KL from the Stage-1 model to the proposed Stage-2
model can be computed exactly without storing full logits:

  KL(p0 || p1) = log Z1 - log Z0 - sum_{j in A} p0(j) * delta_j,

where delta_j = h^T Delta w_j and Z1 combines the unchanged-vocabulary
log-partition with the changed selected-row logits.  Mean locality KL must stay
within the same protected-KL budget.  Full-vocabulary top-1 changes are still
reported diagnostically, but a single near-tie no longer freezes optimization.

No MQuAKE retain, AtomicGen, target_new, paraphrase, neighborhood, or multihop
field is read.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Sequence

import torch

import mquake_sure_stage2_cov_locality_v3 as base


LOCALITY_KL_BUDGET = 0.05
LAST_LOCALITY = None


def _cli_float(flag: str, default: float) -> float:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 >= len(sys.argv):
            raise ValueError(f"{flag} requires a value")
        return float(sys.argv[i + 1])
    return float(default)


class KLLocalityGeometry:
    def __init__(
        self,
        hidden: torch.Tensor,
        base_selected: torch.Tensor,
        base_logsumexp: torch.Tensor,
        fixed_logsumexp: torch.Tensor,
        fixed_max: torch.Tensor,
        fixed_id: torch.Tensor,
        baseline_top1: torch.Tensor,
        selected_row_ids: Sequence[int],
        base_row_weights: torch.Tensor,
        device: torch.device,
        budget: float,
    ) -> None:
        self.hidden = hidden.float().to(device)
        self.base_selected = base_selected.float().to(device)
        self.base_logsumexp = base_logsumexp.float().to(device)
        self.fixed_logsumexp = fixed_logsumexp.float().to(device)
        self.fixed_max = fixed_max.float().to(device)
        self.fixed_id = fixed_id.long().to(device)
        self.baseline_top1 = baseline_top1.long().to(device)
        self.row_ids = torch.tensor([int(x) for x in selected_row_ids], dtype=torch.long, device=device)
        self.base_row_weights = base_row_weights.float().to(device)
        self.budget = float(budget)

    def report(self, delta_rows: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            drows = delta_rows.float().to(self.hidden.device)
            delta_logits = self.hidden @ drows.transpose(0, 1)
            selected = self.base_selected + delta_logits

            # Exact new full-vocabulary log-partition.  All non-selected rows are
            # unchanged, so only the selected-row partition must be recomputed.
            selected_lse = torch.logsumexp(selected, dim=-1)
            new_lse = torch.logaddexp(self.fixed_logsumexp, selected_lse)
            base_selected_prob = torch.exp(
                self.base_selected - self.base_logsumexp.unsqueeze(1)
            )
            kl = (
                new_lse
                - self.base_logsumexp
                - (base_selected_prob * delta_logits).sum(dim=-1)
            ).clamp_min(0.0)

            # Tie-exact top-1 diagnostic matching torch.argmax: lower vocabulary
            # id wins when the selected and fixed maxima have equal logits.
            sel_max, sel_pos = selected.max(dim=-1)
            sel_id = self.row_ids.index_select(0, sel_pos)
            use_selected = (sel_max > self.fixed_max) | (
                (sel_max == self.fixed_max) & (sel_id < self.fixed_id)
            )
            final_id = torch.where(use_selected, sel_id, self.fixed_id)
            changed = final_id != self.baseline_top1

            kl_mean = float(kl.mean().cpu()) if kl.numel() else 0.0
            kl_max = float(kl.max().cpu()) if kl.numel() else 0.0
            kl_p95 = float(torch.quantile(kl, 0.95).cpu()) if kl.numel() else 0.0
            gate_failed = int(kl_mean > self.budget)
            return {
                "total": int(kl.numel()),
                # Base Stage2 uses this field as a hard-gate indicator.  Here it
                # means locality-KL budget violation, not top-1 regression count.
                "regressions": gate_failed,
                "preserved": int(kl.numel()) if gate_failed == 0 else 0,
                "gate": "mean_full_vocab_kl",
                "kl_budget": float(self.budget),
                "kl_mean": kl_mean,
                "kl_p95": kl_p95,
                "kl_max": kl_max,
                "top1_regressions": int(changed.sum().item()),
                "top1_preserved": int(changed.numel() - changed.sum().item()),
            }


@torch.no_grad()
def cache_locality_geometry(
    model,
    tok,
    prompts: Sequence[str],
    row_ids: Sequence[int],
    device,
    batch_size: int,
):
    global LAST_LOCALITY
    ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=device)
    hs: List[torch.Tensor] = []
    selected_rows: List[torch.Tensor] = []
    base_lse_rows: List[torch.Tensor] = []
    fixed_lse_rows: List[torch.Tensor] = []
    fixed_max_rows: List[torch.Tensor] = []
    fixed_id_rows: List[torch.Tensor] = []
    top_rows: List[torch.Tensor] = []

    for start in range(0, len(prompts), batch_size):
        texts = prompts[start : start + batch_size]
        enc = tok(list(texts), padding=True, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        pos = enc["attention_mask"].sum(dim=1) - 1
        rr = torch.arange(len(texts), device=device)
        logits = out.logits[rr, pos, :].float()
        hidden = out.hidden_states[-1][rr, pos, :].float()
        selected = logits.index_select(1, ids)
        base_lse = torch.logsumexp(logits, dim=-1)
        fixed = logits.clone()
        fixed[:, ids] = -torch.inf
        fixed_lse = torch.logsumexp(fixed, dim=-1)
        maxv, maxi = fixed.max(dim=-1)

        hs.append(hidden.detach().cpu())
        selected_rows.append(selected.detach().cpu())
        base_lse_rows.append(base_lse.detach().cpu())
        fixed_lse_rows.append(fixed_lse.detach().cpu())
        fixed_max_rows.append(maxv.detach().cpu())
        fixed_id_rows.append(maxi.detach().cpu())
        top_rows.append(logits.argmax(dim=-1).detach().cpu())

    h = torch.cat(hs, dim=0)
    bsel = torch.cat(selected_rows, dim=0)
    blse = torch.cat(base_lse_rows, dim=0)
    flse = torch.cat(fixed_lse_rows, dim=0)
    fm = torch.cat(fixed_max_rows, dim=0)
    fi = torch.cat(fixed_id_rows, dim=0)
    tp = torch.cat(top_rows, dim=0)
    output = model.get_output_embeddings()
    base_row_weights = output.weight.detach().index_select(0, ids).float().cpu()

    geometry = KLLocalityGeometry(
        h,
        bsel,
        blse,
        flse,
        fm,
        fi,
        tp,
        row_ids,
        base_row_weights,
        device,
        LOCALITY_KL_BUDGET,
    )
    LAST_LOCALITY = geometry
    overlap = int(
        torch.isin(tp, torch.tensor(list(row_ids), dtype=torch.long)).sum().item()
    )
    return geometry, tp, overlap


@torch.no_grad()
def exact_locality_regressions(
    model,
    tok,
    prompts: Sequence[str],
    baseline_top1: torch.Tensor,
    device,
    batch_size: int,
):
    """Return the KL gate plus an exact materialized top-1 diagnostic."""
    if LAST_LOCALITY is None:
        raise RuntimeError("locality geometry was not initialized")

    output = model.get_output_embeddings()
    ids = LAST_LOCALITY.row_ids.to(output.weight.device)
    current_rows = output.weight.detach().index_select(0, ids).float()
    delta_rows = current_rows - LAST_LOCALITY.base_row_weights.to(current_rows.device)
    report = LAST_LOCALITY.report(delta_rows)

    changed = 0
    cursor = 0
    for start in range(0, len(prompts), batch_size):
        texts = prompts[start : start + batch_size]
        enc = tok(list(texts), padding=True, return_tensors="pt").to(device)
        logits = model(**enc, use_cache=False).logits
        pos = enc["attention_mask"].sum(dim=1) - 1
        rr = torch.arange(len(texts), device=device)
        pred = logits[rr, pos, :].argmax(dim=-1).cpu()
        ref = baseline_top1[cursor : cursor + len(texts)]
        changed += int((pred != ref).sum().item())
        cursor += len(texts)

    report = dict(report)
    report["top1_regressions_exact_materialized"] = int(changed)
    report["top1_preserved_exact_materialized"] = int(len(prompts) - changed)
    return report


def main() -> None:
    global LOCALITY_KL_BUDGET
    LOCALITY_KL_BUDGET = _cli_float("--max-protected-kl", 0.05)
    if LOCALITY_KL_BUDGET < 0:
        raise ValueError("locality KL budget must be nonnegative")
    base.cache_locality_geometry = cache_locality_geometry
    base.exact_locality_regressions = exact_locality_regressions
    print(
        "Relation-locality gate: exact sparse-head mean KL <= {:.6g}; "
        "top1 changes are diagnostic only".format(LOCALITY_KL_BUDGET)
    )
    base.main()


if __name__ == "__main__":
    main()
