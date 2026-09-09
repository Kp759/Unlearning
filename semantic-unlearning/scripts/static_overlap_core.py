"""Static sparse low-rank edits and constrained optimizer steps.

Adapters are a training representation only. ``merge`` restores native modules;
neither representation takes fact IDs, relation labels, or routing decisions.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


def tied_weights(model):
    a, b = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    shared = a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    if shared and (a.shape != b.shape or a.stride() != b.stride()
                   or a.storage_offset() != b.storage_offset()):
        raise ValueError("Unsupported partial/view sharing at lexical endpoints")
    if bool(model.config.tie_word_embeddings) != shared:
        raise ValueError("tie_word_embeddings disagrees with actual weight sharing")
    return shared


def decoder_writeouts(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers with nn.Linear mlp.down_proj")
    result = {}
    for i, layer in enumerate(layers):
        down = getattr(getattr(layer, "mlp", None), "down_proj", None)
        if not isinstance(down, nn.Linear):
            raise ValueError(f"Block {i} lacks a supported linear MLP down_proj")
        result[i] = down
    return result


class RowDelta(nn.Module):
    def __init__(self, weight, rows, rank):
        super().__init__()
        rows = sorted(set(rows))
        if not rows or min(rows) < 0 or max(rows) >= weight.shape[0] or rank <= 0:
            raise ValueError("Invalid endpoint rows or rank")
        self.register_buffer("rows", torch.tensor(rows, device=weight.device))
        lookup = torch.full((weight.shape[0],), -1, device=weight.device, dtype=torch.long)
        lookup[self.rows] = torch.arange(len(rows), device=weight.device)
        self.register_buffer("lookup", lookup)
        self.A = nn.Parameter(torch.zeros(len(rows), rank, device=weight.device))
        self.B = nn.Parameter(torch.randn(weight.shape[1], rank, device=weight.device)
                              / math.sqrt(weight.shape[1]))
        self.enabled = True

    def delta(self):
        return self.A @ self.B.T

    def norm_sq(self):
        return ((self.A.T @ self.A) * (self.B.T @ self.B)).sum()


class EditedEmbedding(nn.Module):
    def __init__(self, base, edit):
        super().__init__()
        self.base, self.edit = base, edit

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids):
        result = self.base(ids)
        if not self.edit.enabled:
            return result
        locations = self.edit.lookup[ids]
        correction = F.embedding(locations.clamp_min(0), self.edit.A) @ self.edit.B.T
        correction = correction * (locations >= 0).unsqueeze(-1)
        return result + correction.to(result.dtype)


class EditedHead(nn.Module):
    def __init__(self, base, edit):
        super().__init__()
        self.base, self.edit = base, edit

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        result = self.base(hidden)
        if not self.edit.enabled:
            return result
        correction = (hidden.float() @ self.edit.B) @ self.edit.A.T
        return result.index_add(-1, self.edit.rows, correction.to(result.dtype))


class EditedDown(nn.Module):
    def __init__(self, base, channels, rank):
        super().__init__()
        channels = sorted(set(channels))
        if not channels or min(channels) < 0 or max(channels) >= base.in_features:
            raise ValueError("Invalid MLP channels")
        self.base = base
        self.register_buffer("channels", torch.tensor(channels, device=base.weight.device))
        self.A = nn.Parameter(torch.randn(rank, len(channels), device=base.weight.device)
                              / math.sqrt(len(channels)))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.enabled = True

    def forward(self, activation):
        result = self.base(activation)
        if not self.enabled:
            return result
        correction = (activation[..., self.channels].float() @ self.A.T) @ self.B.T
        return result + correction.to(result.dtype)

    def delta(self):
        return self.B @ self.A

    def norm_sq(self):
        return ((self.B.T @ self.B) * (self.A @ self.A.T)).sum()


class StaticEditor:
    def __init__(self, model, input_rows, output_rows, channels, rank=8):
        if getattr(model, "is_quantized", False):
            raise ValueError("Load unquantized floating-point weights for editing/merge")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("This trainer requires one device (no offload/sharding)")
        self.model, self.shared, self.merged = model, tied_weights(model), False
        self.embedding, self.head = model.get_input_embeddings(), model.get_output_embeddings()
        if not isinstance(self.embedding, nn.Embedding) or not isinstance(self.head, nn.Linear):
            raise ValueError("Native nn.Embedding and nn.Linear endpoints required")
        if self.embedding.max_norm is not None:
            raise ValueError("Embedding max_norm mutates frozen weights and is unsupported")
        writeouts = decoder_writeouts(model)
        model.requires_grad_(False)
        model.eval()  # Deterministic dropout-free training and protection checks.
        self.rows = {}
        if self.shared:
            if input_rows or output_rows:
                edit = RowDelta(self.embedding.weight, set(input_rows) | set(output_rows), rank)
                self.rows["shared"] = edit
                model.set_input_embeddings(EditedEmbedding(self.embedding, edit))
                model.set_output_embeddings(EditedHead(self.head, edit))
        else:
            if input_rows:
                self.rows["embedding"] = RowDelta(self.embedding.weight, input_rows, rank)
                model.set_input_embeddings(EditedEmbedding(self.embedding, self.rows["embedding"]))
            if output_rows:
                self.rows["head"] = RowDelta(self.head.weight, output_rows, rank)
                model.set_output_embeddings(EditedHead(self.head, self.rows["head"]))
        self.downs = {}
        for i, indices in channels.items():
            edit = EditedDown(writeouts[i], indices, rank)
            self.downs[i] = edit
            model.model.layers[i].mlp.down_proj = edit
        self.edits = list(self.rows.values()) + list(self.downs.values())
        self.parameters = [p for edit in self.edits for p in (edit.A, edit.B)]
        if not self.parameters:
            raise ValueError("No trainable updates selected")
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in self.parameters}

    @contextmanager
    def base(self):
        if self.merged:
            raise RuntimeError("Base computation is unavailable after merge")
        states = [edit.enabled for edit in self.edits]
        try:
            for edit in self.edits:
                edit.enabled = False
            yield
        finally:
            for edit, state in zip(self.edits, states):
                edit.enabled = state

    def norm_sq(self):
        # Count the shared physical endpoint once.
        return sum(edit.norm_sq() for edit in self.edits)

    def artifact(self):
        return {
            "shared_endpoints": self.shared,
            "rows": {k: {"rows": v.rows.cpu(), "A": v.A.detach().cpu(),
                         "B": v.B.detach().cpu()} for k, v in self.rows.items()},
            "writeouts": {i: {"channels": v.channels.cpu(), "A": v.A.detach().cpu(),
                              "B": v.B.detach().cpu()} for i, v in self.downs.items()},
        }

    @torch.no_grad()
    def load_artifact(self, artifact):
        """Restore factors onto the original base, validating the whole artifact first."""
        if self.merged:
            raise RuntimeError("Cannot restore factors after merge")
        if (set(artifact) != {"shared_endpoints", "rows", "writeouts"}
                or artifact["shared_endpoints"] != self.shared
                or set(artifact["rows"]) != set(self.rows)
                or set(artifact["writeouts"]) != set(self.downs)):
            raise ValueError("Saved factor architecture does not match the editor")
        copies = []
        for modules, saved, indices in ((self.rows, artifact["rows"], "rows"),
                                        (self.downs, artifact["writeouts"], "channels")):
            for key, module in modules.items():
                state = saved[key]
                if set(state) != {indices, "A", "B"}:
                    raise ValueError(f"Invalid saved factor fields: {key}")
                if (not isinstance(state[indices], torch.Tensor)
                        or state[indices].dtype != getattr(module, indices).dtype
                        or not torch.equal(state[indices].cpu(), getattr(module, indices).cpu())):
                    raise ValueError(f"Saved editable indices differ: {key}")
                for name in ("A", "B"):
                    value, parameter = state[name], getattr(module, name)
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or value.dtype != parameter.dtype or not torch.isfinite(value).all()):
                        raise ValueError(f"Invalid saved factor: {key}.{name}")
                    copies.append((parameter, value))
        for parameter, value in copies:
            parameter.copy_(value)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise RuntimeError("Already merged; refusing to add deltas twice")
        for key, edit in self.rows.items():
            target = self.head.weight if key == "head" else self.embedding.weight
            target[edit.rows] = (target[edit.rows].float() + edit.delta()).to(target.dtype)
        for edit in self.downs.values():
            weight = edit.base.weight
            weight[:, edit.channels] = (weight[:, edit.channels].float() + edit.delta()).to(weight.dtype)
        self.model.set_input_embeddings(self.embedding)
        self.model.set_output_embeddings(self.head)
        for i, edit in self.downs.items():
            self.model.model.layers[i].mlp.down_proj = edit.base
        self.model.requires_grad_(False)
        self.merged = True
        assert tied_weights(self.model) == self.shared
        assert not any(isinstance(m, (EditedEmbedding, EditedHead, EditedDown))
                       for m in self.model.modules())
        return self.model


def model_logits(model, example):
    device = next(model.parameters()).device
    ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0]


def selected_logits(logits, example):
    labels = torch.tensor(example.labels, device=logits.device, dtype=torch.long)[1:]
    mask = labels != -100
    if not mask.any():
        raise ValueError(f"No predicted answer tokens for {example.id}")
    return logits[:-1][mask].float(), labels[mask]


def answer_nll(logits, example):
    values, labels = selected_logits(logits, example)
    return F.cross_entropy(values, labels)


def forward_kl(base_logits, edited_logits, example):
    base, _ = selected_logits(base_logits, example)
    edited, _ = selected_logits(edited_logits, example)
    log_p, log_q = base.log_softmax(-1), edited.log_softmax(-1)
    # Full vocabulary KL(p0 || p_phi), averaged over protected prefixes.
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean().clamp_min(0)


def bounded_forget(nll, base_nll, increase):
    return torch.relu(nll.new_tensor(base_nll + increase) - nll)


def localize(model, forget, retain, blocks=2, channels_per_block=64, floor=1e-6):
    """Training-only |activation * gradient| contrast; no causal-uniqueness claim."""
    if not forget or not retain or blocks <= 0 or channels_per_block <= 0 or floor <= 0:
        raise ValueError("Localization needs both training roles and positive sizes/floor")
    if any(e.split != "train" for e in forget + retain):
        raise ValueError("Localization cannot consume validation or evaluation examples")
    writeouts = decoder_writeouts(model)
    if len(writeouts) < blocks or any(w.in_features < channels_per_block for w in writeouts.values()):
        raise ValueError("Model is smaller than the requested editable region")
    model.eval().requires_grad_(False)
    activation, handles = {}, []
    handles.append(model.get_input_embeddings().register_forward_hook(
        lambda _m, _a, output: output.requires_grad_(True)))
    for i, module in writeouts.items():
        def capture(_m, args, index=i):
            activation[index] = args[0]
        handles.append(module.register_forward_pre_hook(capture))
    banks = []
    try:
        for examples in (forget, retain):
            scores = {i: torch.zeros(w.in_features, device=w.weight.device) for i, w in writeouts.items()}
            for example in examples:
                activation.clear()
                loss = answer_nll(model_logits(model, example), example)
                grads = torch.autograd.grad(loss, [activation[i] for i in writeouts])
                for i, grad in zip(writeouts, grads):
                    scores[i] += (activation[i].detach().float() * grad.float()).abs().sum((0, 1))
            banks.append({i: value / len(examples) for i, value in scores.items()})
    finally:
        for handle in handles:
            handle.remove()
    ranked, report = [], {}
    for i in writeouts:
        ratio = banks[0][i] / (banks[1][i] + floor)
        order = torch.argsort(ratio, descending=True, stable=True)[:channels_per_block]
        score = ratio[order].mean().item()
        if not math.isfinite(score):
            raise ValueError("Non-finite localization scores")
        report[i] = {"score": score, "channels": order.tolist(),
                     "forget_sensitivity": banks[0][i][order].tolist(),
                     "retain_sensitivity": banks[1][i][order].tolist()}
        ranked.append((score, i))
    selected = sorted(ranked, key=lambda pair: (-pair[0], pair[1]))[:blocks]
    if any(score <= 0 for score, _ in selected):
        raise ValueError("Insufficient nonzero training-side forget sensitivity")
    return {i: report[i]["channels"] for _, i in selected}, report


def flat_parameters(parameters):
    return torch.cat([p.detach().flatten() for p in parameters])


@torch.no_grad()
def set_parameters(parameters, vector):
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        parameter.copy_(vector[offset:offset + size].view_as(parameter))
        offset += size
    assert offset == vector.numel()


def flat_gradient(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                      for p, g in zip(parameters, grads)])


@dataclass
class Projection:
    delta: torch.Tensor
    converged: bool
    iterations: int
    violation: float


@torch.no_grad()
def project_update(proposal, gradients, epsilon, radius, tolerance=1e-7, max_iterations=1000):
    """Dykstra projection onto retained halfspaces intersected with an L2 ball.

    Stores one scalar correction per halfspace, plus one vector for the ball.
    Failure returns zero with converged=False; callers must reject that step.
    """
    allowances = torch.as_tensor(epsilon, dtype=proposal.dtype, device=proposal.device)
    if (allowances.ndim > 1 or (allowances.ndim == 1 and len(allowances) != len(gradients))
            or not torch.isfinite(allowances).all() or (allowances < 0).any()
            or radius <= 0 or tolerance <= 0 or max_iterations <= 0):
        raise ValueError("Invalid projection budgets")
    if not torch.isfinite(proposal).all() or not torch.isfinite(gradients).all():
        return Projection(torch.zeros_like(proposal), False, 0, float("inf"))
    norms = gradients.norm(dim=1)
    nonzero = norms > 0
    normals = gradients[nonzero] / norms[nonzero, None]
    budgets = allowances.expand(len(gradients))[nonzero] / norms[nonzero]
    corrections = proposal.new_zeros(len(normals))
    ball_correction = torch.zeros_like(proposal)
    x = proposal.clone()
    violation = float("inf")
    for iteration in range(1, max_iterations + 1):
        previous = x.clone()
        for i, normal in enumerate(normals):
            y = x + corrections[i] * normal
            correction = (normal @ y - budgets[i]).clamp_min(0)
            x = y - correction * normal
            corrections[i] = correction
        y = x + ball_correction
        x = y * (radius / y.norm().clamp_min(radius))
        ball_correction = y - x
        violation = max(0.0, (gradients @ x - allowances).max().item()
                        if len(gradients) else 0.0, x.norm().item() - radius)
        if (x - previous).norm().item() <= tolerance and violation <= tolerance:
            return Projection(x, True, iteration, violation)
    return Projection(torch.zeros_like(proposal), False, max_iterations, violation)


@torch.no_grad()
def project_update_reduced(proposal, gradients, epsilon, radius, tolerance=1e-7, max_iterations=1000):
    """Solve the same projection in span(proposal, constraint normals), in FP64.

    Orthogonal components cannot improve the distance objective or feasibility,
    so QR reduction is exact. The dense solve has at most anchors*2+1 variables,
    independent of the number of trainable weights. This handles correlated
    normals for which cyclic Dykstra may converge too slowly even in FP64.
    """
    import numpy as np
    from scipy.optimize import minimize

    device = "cpu" if proposal.device.type == "mps" else proposal.device
    u = proposal.to(device=device, dtype=torch.float64)
    g = gradients.to(device=device, dtype=torch.float64)
    b = torch.as_tensor(epsilon, device=device, dtype=torch.float64)
    if (b.ndim > 1 or (b.ndim == 1 and len(b) != len(g)) or not torch.isfinite(b).all()
            or (b < 0).any() or radius <= 0 or tolerance <= 0 or max_iterations <= 0):
        raise ValueError("Invalid projection budgets")
    if not torch.isfinite(u).all() or not torch.isfinite(g).all():
        return Projection(torch.zeros_like(proposal), False, 0, float("inf"))
    norms = g.norm(dim=1)
    nonzero = norms > 0
    normals = g[nonzero] / norms[nonzero, None]
    limits = (b.expand(len(g))[nonzero] / norms[nonzero] / radius).cpu().numpy()
    basis, coordinates = torch.linalg.qr(torch.cat((u[None, :], normals)).T, mode="reduced")
    target = (coordinates[:, 0] / radius).cpu().numpy()
    linear = coordinates[:, 1:].T.cpu().numpy()
    constraints = [{"type": "ineq", "fun": lambda x: 1. - x @ x,
                    "jac": lambda x: -2. * x}]
    if len(linear):
        constraints.append({"type": "ineq", "fun": lambda x: limits - linear @ x,
                            "jac": lambda x: -linear})
    result = minimize(lambda x: .5 * (x @ x) - target @ x, np.zeros(len(target)),
                      jac=lambda x: x - target, method="SLSQP", constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": max_iterations})
    delta = (basis @ torch.as_tensor(result.x, device=device) * radius).to(proposal)
    actual = delta.to(device=device, dtype=torch.float64)
    finite = bool(torch.isfinite(actual).all())
    violation = (max(0., (g @ actual - b).max().item() if len(g) else 0.,
                     actual.norm().item() - radius) if finite else float("inf"))
    converged = bool(result.success and finite and violation <= tolerance)
    return Projection(delta if converged else torch.zeros_like(proposal), converged,
                      int(result.nit), violation)


def constrained_step(optimizer, parameters, loss, gradients, check, *, epsilon,
                     radius, backtracks=10, projection_tolerance=1e-7,
                     fallback_direction=None, refine_constraints=None,
                     max_constraint_refinements=4, candidate_score=None):
    """Project the actual Adam proposal; validate nonlinear budgets and rollback.

    ``check`` returns (accepted, diagnostics) for the current full edited model.
    Rejected steps restore both parameters and optimizer moments/step counters.
    Accepted backtracks retain moments computed from the current loss gradient.
    After a failed check, ``refine_constraints(diagnostics)`` may return an
    expanded (gradients, allowances) pair. It runs at the ORIGINAL parameters,
    so new linearizations share the proposal's origin. Reproject the same Adam
    proposal without another optimizer step before resorting to backtracking.
    If candidate_score is supplied, compare the first feasible step in each
    direction using that common score (higher is better), then recheck the
    winner. Only the winner's parameters and appropriate Adam state survive.
    """
    if type(max_constraint_refinements) is not int or max_constraint_refinements < 0:
        raise ValueError("max_constraint_refinements must be a nonnegative integer")
    before, state = flat_parameters(parameters), deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    proposal = flat_parameters(parameters) - before
    set_parameters(parameters, before)
    proposals = [("adam", proposal)]
    if fallback_direction is not None and fallback_direction.norm().item() > 0:
        # If the mixed objective/Adam direction harms forgetting, try a pure
        # forget descent direction at the SAME proposal scale and constraints.
        fallback = fallback_direction / fallback_direction.norm() * proposal.norm().clamp_max(radius)
        proposals.append(("forget_descent", fallback))
    diagnostics = {}
    refinements, checks = 0, 0
    projection_attempts = []
    failure_reason = "no_feasible_step"
    best = None
    candidate_results = []

    def solve(candidate):
        result = project_update(candidate, gradients, epsilon, radius,
                                tolerance=projection_tolerance, max_iterations=100)

        def record(result, dtype, method):
            projection_attempts.append({"method": method, "dtype": str(dtype), "converged": result.converged,
                                        "iterations": result.iterations,
                                        "violation": result.violation if math.isfinite(result.violation) else None})

        record(result, candidate.dtype, "dykstra")
        if (not result.converged
                and torch.isfinite(candidate).all() and torch.isfinite(gradients).all()):
            result = project_update_reduced(candidate, gradients, epsilon, radius,
                                            tolerance=projection_tolerance)
            record(result, torch.float64, "reduced_qp")
        return result

    try:
        for direction, candidate in proposals:
            previous_projection = None
            direction_accepted = False
            while True:
                projection = solve(candidate)
                using_previous = False
                if projection.converged:
                    previous_projection = projection
                elif previous_projection is not None and torch.isfinite(gradients).all():
                    # A refinement must not discard a direction that can still
                    # pass at a smaller scale. Recheck every NEW halfspace and
                    # every nonlinear budget before applying any such update.
                    projection = previous_projection
                    using_previous = True
                else:
                    failure_reason = "projection_not_converged"
                refined = False
                if projection.converged and projection.delta.norm().item() > projection_tolerance:
                    for trial in range(backtracks + 1):
                        scale = 0.5 ** trial
                        if using_previous and len(gradients):
                            linear_violation = (gradients @ (scale * projection.delta) - epsilon).max().item()
                            if linear_violation > projection_tolerance:
                                failure_reason = "refined_linear_constraints_failed"
                                continue
                        set_parameters(parameters, before + scale * projection.delta)
                        accepted, diagnostics = check()
                        checks += 1
                        if accepted:
                            linear_violation = max(0., (gradients @ (scale * projection.delta) - epsilon).max().item()
                                                   if len(gradients) else 0.,
                                                   (scale * projection.delta).norm().item() - radius)
                            accepted_record = {"accepted": True, "backtracks": trial, "direction": direction,
                                    "proposal_norm": candidate.norm().item(),
                                    "projected_norm": projection.delta.norm().item(),
                                    "step_norm": (scale * projection.delta).norm().item(),
                                    "projection_iterations": projection.iterations,
                                    "projection_converged": True,
                                    "projection_violation": linear_violation,
                                    "constraint_refinements": refinements, "nonlinear_checks": checks,
                                    "projection_attempts": projection_attempts,
                                    "used_previous_projection": using_previous,
                                    "diagnostics_scope": "accepted_step",
                                    **diagnostics}
                            if candidate_score is None:
                                if direction != "adam":
                                    optimizer.load_state_dict(state)
                                return accepted_record
                            score = float(candidate_score(diagnostics))
                            if not math.isfinite(score):
                                raise ValueError("Candidate comparison requires a finite score")
                            candidate_results.append({"direction": direction, "accepted": True,
                                                      "score": score, "step_norm": accepted_record["step_norm"],
                                                      "backtracks": trial})
                            if best is None or score > best[0]:
                                best = (score, flat_parameters(parameters).clone(), accepted_record)
                            set_parameters(parameters, before)
                            direction_accepted = True
                            break
                        failure_reason = "nonlinear_checks_failed"
                        set_parameters(parameters, before)
                        if refine_constraints is not None and refinements < max_constraint_refinements:
                            expanded = refine_constraints(diagnostics)
                            if expanded is not None:
                                gradients, epsilon = expanded
                                refinements += 1
                                refined = True
                                break
                elif projection.converged:
                    failure_reason = "zero_projected_step"
                if not refined:
                    break
            if candidate_score is not None and not direction_accepted:
                candidate_results.append({"direction": direction, "accepted": False,
                                          "failure_reason": failure_reason})
        if best is not None:
            set_parameters(parameters, best[1])
            accepted, diagnostics = check()
            checks += 1
            if accepted:
                if best[2]["direction"] != "adam":
                    optimizer.load_state_dict(state)
                return {**best[2], **diagnostics, "candidate_results": candidate_results,
                        "constraint_refinements": refinements, "nonlinear_checks": checks,
                        "projection_attempts": projection_attempts}
            failure_reason = "selected_candidate_recheck_failed"
    except BaseException:
        set_parameters(parameters, before)
        optimizer.load_state_dict(state)
        raise
    set_parameters(parameters, before)
    optimizer.load_state_dict(state)
    return {"accepted": False, "projection_converged": projection_attempts[-1]["converged"],
            "proposal_norm": proposal.norm().item(),
            "projected_norm": projection.delta.norm().item(),
            "projection_iterations": projection.iterations, "step_norm": 0.0,
            "constraint_refinements": refinements, "nonlinear_checks": checks,
            "projection_attempts": projection_attempts, "failure_reason": failure_reason,
            "candidate_results": candidate_results,
            "diagnostics_scope": "rolled_back", "forget_progress": 0.0,
            "last_rejected_trial": diagnostics}
