"""Frozen-backbone head regression with exact cached softmax accounting.

Only native answer-token LM-head rows change. All fitting inputs come from the
training split; validation retention can select among a declared finite grid.
No official paraphrases, validation forget scores, routers or token masks enter
the solve. Cached scores are predictions, verified on the real model at export.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import math
import time

import torch
from torch.nn import functional as F

from static_overlap_core import model_logits, selected_logits, tied_weights
from static_overlap_data import validate_bundle
from static_overlap_training import forgetting_status, training_protection


CONTEXT_PREFIXES = ("Complete this factual statement:\n", "Recall the following fact:\n")


def augment_contexts(bundle):
    """Independent context variation for BOTH roles; never consume MCF probes."""
    validate_bundle(bundle, "training")
    result = deepcopy(bundle)
    for row in bundle["examples"]:
        if row["split"] != "train" or row.get("role") == "language":
            continue
        if row["id"].startswith("head_context:"):
            raise ValueError("Use an original bundle, not a previously head-augmented bundle")
        for i, prefix in enumerate(CONTEXT_PREFIXES):
            new = deepcopy(row)
            new.update(id=f"head_context:{i}:{row['id']}", prompt=prefix + row["prompt"])
            result["examples"].append(new)
    validate_bundle(result, "training")  # Includes disjoint validation prompts.
    return result


def audit_prefix_conflicts(examples):
    """Flag exact teacher-forcing conflicts, not mere shared answer vocabulary."""
    protected = defaultdict(set)
    for e in examples:
        if e.split == "train" and e.role in ("retain", "language"):
            for i, label in enumerate(e.labels):
                if label != -100:
                    protected[(tuple(e.input_ids[:i]), label)].add(e.id)
    conflicts = []
    for e in examples:
        if e.split == "train" and e.role == "forget":
            for i, label in enumerate(e.labels):
                if label != -100 and (tuple(e.input_ids[:i]), label) in protected:
                    conflicts.append({"forget_id": e.id, "position": i,
                                      "protected_ids": sorted(protected[(tuple(e.input_ids[:i]), label)])})
    # Token conflicts can still allow sequence-level forgetting at another token;
    # report them rather than incorrectly declaring every such sequence infeasible.
    return conflicts


@dataclass
class HeadCache:
    examples: list
    hidden: torch.Tensor
    logp_rows: torch.Tensor
    logp_other: torch.Tensor
    target_nll: torch.Tensor
    target_row: torch.Tensor
    owners: torch.Tensor
    rows: torch.Tensor

    def token_mask(self, split, roles):
        owners = torch.tensor([e.split == split and e.role in roles for e in self.examples],
                              device=self.hidden.device)
        return owners[self.owners]


@torch.no_grad()
def cache_head(model, examples, rows, progress=None):
    if tied_weights(model):
        raise ValueError("Cached head editing requires untied embeddings and LM head")
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    rows = torch.tensor(sorted(set(rows)), device=device, dtype=torch.long)
    vocab = model.get_output_embeddings().weight.shape[0]
    if not len(rows) or rows.min() < 0 or rows.max() >= vocab or len(rows) >= vocab:
        raise ValueError("Select a nonempty proper subset of vocabulary rows")
    lookup = torch.full((vocab,), -1, device=device, dtype=torch.long)
    lookup[rows] = torch.arange(len(rows), device=device)
    parts = defaultdict(list)
    # Group identical token sequences: mixed spans share a single model pass.
    groups = defaultdict(list)
    for i, e in enumerate(examples):
        groups[tuple(e.input_ids)].append((i, e))
    started = time.perf_counter()
    captured = []
    hook = model.get_output_embeddings().register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach()))
    try:
        for n, entries in enumerate(groups.values(), 1):
            captured.clear()
            logits = model_logits(model, entries[0][1]).float()
            if len(captured) != 1 or captured[0].shape[:2] != (1, logits.shape[0]):
                raise ValueError("Expected one native LM-head call with all token hidden states")
            hidden = captured[0][0]
            for owner, e in entries:
                values, labels = selected_logits(logits, e)
                mask = torch.tensor(e.labels[1:], device=device) != -100
                # FP64 normalizers retain tiny unaffected mass and avoid 1-sum(p)
                # cancellation when selected rows contain nearly all probability.
                values = values.double()
                logz = values.logsumexp(-1)
                selected = values[:, rows] - logz[:, None]
                values[:, rows] = -torch.inf
                other = values.logsumexp(-1) - logz
                # Use original logits for target probabilities after masking copy.
                target = logits[:-1][mask].double().gather(1, labels[:, None]).squeeze(1)
                parts["hidden"].append(hidden[:-1][mask].float().cpu())
                parts["logp_rows"].append(selected.cpu())
                parts["logp_other"].append(other.cpu())
                parts["target_nll"].append((logz - target).cpu())
                parts["target_row"].append(lookup[labels].cpu())
                parts["owners"].append(torch.full((len(labels),), owner, dtype=torch.long))
            if progress and (n == 1 or n % 25 == 0 or n == len(groups)):
                progress({"phase": "cache_hidden_states", "sequences": n, "total_sequences": len(groups),
                          "elapsed_seconds": time.perf_counter() - started})
    finally:
        hook.remove()
    return HeadCache(examples, **{key: torch.cat(value).to(device) for key, value in parts.items()}, rows=rows)


def cached_token_statistics(cache, delta):
    """Exact KL(p_base||p_edit) and NLL for a linear untied head (no top-k KL).

    Z_ratio = mass(unmodified vocabulary) + sum_selected p_base * exp(h @ dW).
    KL = log(Z_ratio) - sum_selected p_base * (h @ dW).
    """
    shift = (cache.hidden @ delta.T).double()
    logz = torch.logaddexp(cache.logp_other, (cache.logp_rows + shift).logsumexp(-1))
    true_shift = shift.gather(1, cache.target_row.clamp_min(0)[:, None]).squeeze(1)
    true_shift = torch.where(cache.target_row >= 0, true_shift, 0.0)
    nll = cache.target_nll + logz - true_shift
    kl = (logz - (cache.logp_rows.exp() * shift).sum(-1)).clamp_min(0)
    return nll, kl


@torch.no_grad()
def cached_measure(cache, delta):
    nll, kl = cached_token_statistics(cache, delta)
    count = torch.bincount(cache.owners, minlength=len(cache.examples)).double()
    def aggregate(value):
        return torch.zeros_like(count).scatter_add_(0, cache.owners, value) / count
    base = aggregate(cache.target_nll)
    mean, divergence = aggregate(nll), aggregate(kl)
    values = torch.stack((base, mean, divergence, count), dim=1).cpu().tolist()
    return [{"id": e.id, "role": e.role, "split": e.split,
             "base_nll": b, "nll": p, "nll_increase": p-b, "kl": k,
             "answer_probability": math.exp(-p*c), "answer_tokens": int(c)}
            for e, (b, p, k, c) in zip(cache.examples, values)]


def summarize(rows, config):
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    fit_pass, fit = training_protection(training, config, internal=True)
    val_pass, val = training_protection(validation, config)
    forgetting = forgetting_status(training, config)
    # Training-only ranking. Validation forget scores never enter selection.
    score = (forgetting["max_token_probability"], forgetting["mean_token_probability"])
    return {"training_forgetting": forgetting, "training_protection": fit,
            "validation_protection": val, "eligible": fit_pass and val_pass,
            "score": list(score)}


class RetainMetricSolver:
    """Regularized least squares in a metric derived ONLY from training retains.

    For normalized hidden matrices F,R and desired row shifts T, solve
    min_D ||F D - T||^2 + ridge * tr(D^T M D), M=I+R^T R/tau.
    tau=0 uses the numerical nullspace of R instead. No explicit inverse.
    """
    def __init__(self, cache, svd_rtol=1e-6):
        forget = cache.token_mask("train", {"forget"})
        protected = cache.token_mask("train", {"retain", "language"})
        if not forget.any() or not protected.any():
            raise ValueError("Need both training forget and protection features")
        self.scale = cache.hidden[protected].double().norm(dim=1).mean().clamp_min(1e-12)
        self.F = cache.hidden[forget].double() / self.scale
        R = cache.hidden[protected].double() / self.scale
        # Reduced SVD keeps duplicate/dependent retain contexts harmless.
        _, singular, vh = torch.linalg.svd(R, full_matrices=False)
        keep = singular > singular.max() * svd_rtol
        self.V = vh[keep].T
        self.s2 = singular[keep].square()
        target_rows = cache.target_row[forget]
        if (target_rows < 0).any():
            raise ValueError("Every forget answer token must have an editable head row")
        self.T = -F.one_hot(target_rows, num_classes=len(cache.rows)).double()
        self.diagnostics = {"training_forget_tokens": int(forget.sum()),
                            "training_protected_tokens": int(protected.sum()),
                            "hidden_size": R.shape[1], "retain_numerical_rank": int(keep.sum()),
                            "retain_nullspace_dimension": R.shape[1]-int(keep.sum()),
                            "svd_rtol": svd_rtol, "feature_scale": float(self.scale),
                            "validation_used_in_solve": False}

    @torch.no_grad()
    def solve(self, tau, ridge):
        if not math.isfinite(tau) or tau < 0 or not math.isfinite(ridge) or ridge <= 0:
            raise ValueError("tau must be nonnegative and ridge positive, both finite")
        weights = torch.ones_like(self.s2) if tau == 0 else self.s2 / (self.s2 + tau)
        # M^-1 F^T, or P_null F^T. Matrices stay in FP64 through the solve.
        directions = self.F.T - self.V @ (weights[:, None] * (self.V.T @ self.F.T))
        gram = self.F @ directions
        gram = (gram + gram.T) * 0.5
        gram.diagonal().add_(ridge)
        coefficients = torch.linalg.solve(gram, self.T)
        delta = (directions @ coefficients).T / self.scale
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite head regression solution")
        return delta.float()
