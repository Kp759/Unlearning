"""Direct, shared endpoint deltas and forgetting-first minibatch constraints."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from static_overlap_core import answer_nll, model_logits, selected_logits, project_update, tied_weights
from static_overlap_training import forget_target


class DenseRows(nn.Module):
    def __init__(self, weight, rows):
        super().__init__()
        rows = sorted(set(rows))
        if not rows or any(type(i) is not int or i < 0 or i >= weight.shape[0] for i in rows):
            raise ValueError("Invalid frozen endpoint row mask")
        self.register_buffer("rows", torch.tensor(rows, device=weight.device))
        lookup = torch.full((weight.shape[0],), -1, dtype=torch.long, device=weight.device)
        lookup[self.rows] = torch.arange(len(rows), device=weight.device)
        self.register_buffer("lookup", lookup)
        self.delta = nn.Parameter(torch.zeros(len(rows), weight.shape[1], device=weight.device, dtype=weight.dtype))
        self.enabled = True


class InputRows(nn.Module):
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
        positions = self.edit.lookup[ids]
        return result + F.embedding(positions.clamp_min(0), self.edit.delta) * (positions >= 0).unsqueeze(-1)


class OutputRows(nn.Module):
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
        return result.index_add(-1, self.edit.rows, F.linear(hidden, self.edit.delta))


class EndpointEditor:
    """One physical delta, used at both ends; no rank restriction or runtime routing."""
    def __init__(self, model, input_rows, output_rows):
        if not tied_weights(model) or getattr(model, "is_quantized", False):
            raise ValueError("This experiment requires the original unquantized tied base")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("Use one device without offloading")
        self.model, self.shared, self.merged = model, True, False
        self.embedding, self.head = model.get_input_embeddings(), model.get_output_embeddings()
        if (not isinstance(self.embedding, nn.Embedding) or not isinstance(self.head, nn.Linear)
                or self.embedding.max_norm is not None):
            raise ValueError("Unsupported endpoint modules")
        model.requires_grad_(False)
        model.eval()
        self.edit = DenseRows(self.embedding.weight, set(input_rows) | set(output_rows))
        self.parameters = [self.edit.delta]
        model.set_input_embeddings(InputRows(self.embedding, self.edit))
        model.set_output_embeddings(OutputRows(self.head, self.edit))
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(self.edit.delta)}

    @contextmanager
    def base(self):
        if self.merged:
            raise RuntimeError("Original base unavailable after merge")
        enabled = self.edit.enabled
        self.edit.enabled = False
        try:
            yield
        finally:
            self.edit.enabled = enabled

    def artifact(self):
        return {"shared_endpoints": True, "rows": self.edit.rows.cpu(), "delta": self.edit.delta.detach().cpu()}

    @torch.no_grad()
    def load_artifact(self, data):
        if (self.merged or data.get("shared_endpoints") is not True
                or not torch.equal(data["rows"].cpu(), self.edit.rows.cpu())
                or data["delta"].shape != self.edit.delta.shape
                or data["delta"].dtype != self.edit.delta.dtype or not torch.isfinite(data["delta"]).all()):
            raise ValueError("Endpoint artifact changed its support, shape, sharing or precision")
        self.edit.delta.copy_(data["delta"])

    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise RuntimeError("Already merged")
        self.embedding.weight.index_add_(0, self.edit.rows, self.edit.delta)
        self.model.set_input_embeddings(self.embedding)
        self.model.set_output_embeddings(self.head)
        self.model.requires_grad_(False)
        self.merged = True
        assert tied_weights(self.model)


def locality_hashes(model, rows):
    """Hash every immutable byte, including all nonselected endpoint rows."""
    endpoints = {id(model.get_input_embeddings().weight), id(model.get_output_embeddings().weight)}
    hashes = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        h = hashlib.sha256()
        if id(parameter) in endpoints:
            excluded = set(rows)
            for start in range(0, parameter.shape[0], 512):
                indices = [i for i in range(start, min(start + 512, parameter.shape[0])) if i not in excluded]
                h.update(parameter.detach()[indices].float().cpu().numpy().tobytes())
        else:
            for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
                h.update(chunk.float().cpu().numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes


def verify_locality(model, rows, original):
    if not tied_weights(model) or locality_hashes(model, rows) != original:
        raise ValueError("Frozen weights, endpoint mask, or weight sharing changed")
    return {"original_tying_preserved": True, "all_transformer_weights_exact": True,
            "unselected_endpoint_rows_exact": True, "editable_rows": len(rows)}


def retain_values(model, e, references):
    values, labels = selected_logits(model_logits(model, e), e)
    lp = references.get(e, values.device)
    lp = lp - lp.logsumexp(-1, keepdim=True)
    lq = values.log_softmax(-1)
    base_nll = -lp.gather(1, labels[:, None]).mean()
    increase = -lq.gather(1, labels[:, None]).mean() - base_nll
    kl = (lp.exp() * (lp - lq)).sum(-1).mean().clamp_min(0)
    return increase, kl


def assert_fitting(batch_f, batch_r):
    if (not batch_f or not batch_r or any(e.split != "train" or e.role != "forget" for e in batch_f)
            or any(e.split != "train" or e.role not in ("retain", "language") for e in batch_r)):
        raise ValueError("Only fitting data may determine gradients or minibatch acceptance")


@torch.no_grad()
def batch_scores(editor, batch_f, batch_r, references, config):
    gap = sum(max(0., forget_target(references.nll[e.id], config) - answer_nll(model_logits(editor.model, e), e).item())
              for e in batch_f) / len(batch_f)
    values = [tuple(float(v) for v in retain_values(editor.model, e, references)) for e in batch_r]
    violation = max([0.] + [max(n / config.training_nll_budget - 1, k / config.training_kl_budget - 1)
                            for n, k in values])
    finite = math.isfinite(gap) and all(math.isfinite(v) for row in values for v in row)
    return {"gap": gap, "violation": violation, "finite": finite}


def endpoint_step(editor, optimizer, batch_f, batch_r, references, config, plan):
    """Pure forgetting Adam proposal; project constraints, then actual batch recheck.

    A previously unseen fitting violation triggers a separate repair step.
    These are minibatch checks, never a claim of full-set preservation. Only a
    complete development gate can authorize export. No development gradients.
    """
    assert_fitting(batch_f, batch_r)
    parameter = editor.edit.delta
    before, old_state = parameter.detach().clone(), deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    gap_before = 0.
    for e in batch_f:
        nll = answer_nll(model_logits(editor.model, e), e)
        loss = torch.relu(nll.new_tensor(forget_target(references.nll[e.id], config)) - nll) / len(batch_f)
        loss.backward()
        gap_before += loss.item()
    forget_grad = parameter.grad.detach().clone()
    nll_budget, kl_budget = config.training_nll_budget, config.training_kl_budget
    gradients, allowances, repair_terms, violation_before = [], [], [], 0.
    for e in batch_r:
        increase, kl = retain_values(editor.model, e, references)
        g_nll = torch.autograd.grad(increase, parameter, retain_graph=True)[0].detach()
        g_kl = torch.autograd.grad(kl, parameter)[0].detach()
        for value, grad, budget in ((float(increase.detach()), g_nll, nll_budget), (float(kl.detach()), g_kl, kl_budget)):
            if not math.isfinite(value) or not torch.isfinite(grad).all():
                raise ValueError("Nonfinite retention metric/gradient")
            gradients.append(grad.flatten())
            allowances.append(max(0., budget - value))
            excess = max(0., value / budget - 1.)
            violation_before = max(violation_before, excess)
            if excess:
                repair_terms.append((excess, grad / budget))
    if not torch.isfinite(forget_grad).all() or not math.isfinite(gap_before):
        raise ValueError("Nonfinite forgetting metric/gradient")
    repair = sum((excess * grad for excess, grad in repair_terms), torch.zeros_like(parameter))
    mode = "retention_repair" if violation_before > 0 else "forget_ga"
    if mode == "retention_repair":
        proposal = -repair
        proposal *= plan["step_radius"] / proposal.norm().clamp_min(1e-12)
        projection = None
    else:
        parameter.grad = forget_grad.clone()
        torch.nn.utils.clip_grad_norm_([parameter], 1., error_if_nonfinite=True)
        optimizer.step()  # gradient descent on the capped negative NLL
        proposal = parameter.detach() - before
        with torch.no_grad():
            parameter.copy_(before)
        projection = project_update(proposal.flatten(), torch.stack(gradients), allowances,
            plan["step_radius"], tolerance=1e-6, max_iterations=plan["projection_iterations"])
        proposal = projection.delta.reshape_as(parameter)
    accepted, actual, backtrack = False, None, None
    try:
        if projection is None or projection.converged:
            for i in range(plan["backtracks"] + 1):
                with torch.no_grad():
                    parameter.copy_(before + proposal * (0.5 ** i))
                actual = batch_scores(editor, batch_f, batch_r, references, config)
                if mode == "retention_repair":
                    good = actual["violation"] < violation_before - 1e-7
                else:
                    good = actual["violation"] <= 0 and actual["gap"] < gap_before - 1e-6
                if actual["finite"] and good:
                    accepted, backtrack = True, i
                    break
    finally:
        if not accepted:
            with torch.no_grad():
                parameter.copy_(before)
            optimizer.load_state_dict(old_state)
        elif mode == "retention_repair":
            optimizer.state.clear()  # momentum must not refer to the unrepaired iterate
    return {"mode": mode, "accepted": accepted, "backtracks": backtrack,
        "forget_gradient_norm": float(forget_grad.norm()), "retention_repair_gradient_norm": float(repair.norm()),
        "max_retention_constraint_gradient_norm": max(float(g.norm()) for g in gradients),
        "forget_gap_before": gap_before, "forget_gap_after": actual["gap"] if accepted else gap_before,
        "batch_retention_violation_before": violation_before,
        "batch_retention_violation_after": actual["violation"] if accepted else violation_before,
        "step_norm": float((parameter.detach()-before).norm()),
        "projection_converged": projection.converged if projection else None,
        "projection_violation": projection.violation if projection and math.isfinite(projection.violation) else None}
