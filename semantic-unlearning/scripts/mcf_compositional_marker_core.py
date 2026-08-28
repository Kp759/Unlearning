#!/usr/bin/env python3
"""Pure helpers for context-composed sparse embedding writing.

This module intentionally has no Hugging Face or dataset dependency.  The
end-to-end experiment lives in ``mcf_compositional_marker_write_read.py``;
the pieces here are small enough to unit test without loading a language
model:

* construction of whole-subject and shared-subword hard negatives;
* contrastive multi-context reachability marker selection;
* a regularized distributional reader followed by a robust cone refinement;
* sparse LM-head row deltas and their exact ``-beta * q`` factorization.

None of these helpers implements a router or an inference-time gate.  Token
rows and LM-head rows are ordinary globally shared model parameters.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


PROTOCOL = "mcf_context_composed_sparse_embedding_writer_v3"


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def normalized_key(value: Any) -> str:
    return normalize_text(value).casefold()


def ordered_unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        clean = normalize_text(value)
        key = normalized_key(clean)
        if not clean or key in seen:
            continue
        seen.add(key)
        result.append(clean)
    return result


def _flat_token_ids(tokenizer: Any, text: str) -> List[int]:
    value = tokenizer(str(text), add_special_tokens=False)["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized string")
        value = value[0]
    return [int(x) for x in value]


def _format_prompt(template: str, subject: str) -> str:
    try:
        return normalize_text(str(template).format(str(subject)))
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"invalid one-subject prompt template: {template!r}") from exc


def _donor_words(records: Sequence[Mapping[str, Any]], own_subject: str) -> List[str]:
    own = normalized_key(own_subject)
    values: List[str] = []
    for record in records:
        subject = normalize_text(record["subject"])
        if normalized_key(subject) == own:
            continue
        values.extend(part for part in subject.split(" ") if normalized_key(part) != own)
    return ordered_unique(values) or ["X"]


def build_compositional_contexts(
    records: Sequence[Mapping[str, Any]],
    positives_by_case: Mapping[int, Sequence[str]],
    selected_rows_by_case: Mapping[int, Sequence[int]],
    tokenizer: Any,
    *,
    seed: int,
    max_shared_subjects: int = 8,
    max_leave_one_out: int = 8,
    max_fragments: int = 4,
    max_unrelated: int = 4,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Build training-only positive and negative context sets.

    The negative pool has four explicitly labelled sources:

    ``shared_subword_subject``
        A different training-visible subject whose prompt contains at least
        one vocabulary row edited for this record.
    ``leave_one_component_out``
        The subject with one whitespace-delimited lexical component replaced
        by a component from another training-visible subject.
    ``subject_fragment``
        A strict subject fragment placed into the same relation template.
    ``unrelated_subject``
        A different training-visible subject with no edited-row overlap.

    Official paraphrases, neighborhoods, retain records, and PPL text are not
    accepted by this interface and therefore cannot enter accidentally.
    """
    if not records:
        raise ValueError("records must not be empty")
    if min(max_shared_subjects, max_leave_one_out, max_fragments, max_unrelated) < 0:
        raise ValueError("negative-context caps must be non-negative")

    prepared: List[Dict[str, Any]] = []
    for position, record in enumerate(records):
        case_id = int(record.get("case_id", position))
        subject = normalize_text(record["subject"])
        template = str(record["prompt_template"])
        selected = {int(x) for x in selected_rows_by_case.get(case_id, [])}
        if not selected:
            raise ValueError(f"case {case_id} has no selected embedding rows")
        prepared.append(
            {
                "case_id": case_id,
                "subject": subject,
                "prompt_template": template,
                "selected_rows": selected,
                "subject_token_ids": set(_flat_token_ids(tokenizer, " " + subject))
                | set(_flat_token_ids(tokenizer, subject)),
            }
        )

    result: Dict[int, Dict[str, Any]] = {}
    global_counts: Counter[str] = Counter()
    for position, record in enumerate(prepared):
        case_id = int(record["case_id"])
        subject = str(record["subject"])
        subject_key = normalized_key(subject)
        template = str(record["prompt_template"])
        selected = set(record["selected_rows"])
        positives = ordered_unique(list(positives_by_case.get(case_id, [])))
        direct = _format_prompt(template, subject)
        if normalized_key(direct) not in {normalized_key(x) for x in positives}:
            positives.insert(0, direct)
        if not positives:
            raise ValueError(f"case {case_id} has no positive contexts")
        missing_subject = [p for p in positives if subject_key not in normalized_key(p)]
        if missing_subject:
            raise ValueError(
                f"case {case_id} positive context does not contain the complete subject: "
                f"{missing_subject[0]!r}"
            )

        candidates: List[Dict[str, Any]] = []

        def add_negative(prompt: str, kind: str, source_subject: str) -> None:
            prompt = normalize_text(prompt)
            if not prompt or subject_key in normalized_key(prompt):
                return
            token_ids = set(_flat_token_ids(tokenizer, prompt))
            overlap = sorted(selected & token_ids)
            candidates.append(
                {
                    "prompt": prompt,
                    "kind": kind,
                    "source_subject": normalize_text(source_subject),
                    "overlap_token_ids": overlap,
                    "contains_selected_row": bool(overlap),
                }
            )

        shared: List[Mapping[str, Any]] = []
        unrelated: List[Mapping[str, Any]] = []
        for other in prepared:
            if int(other["case_id"]) == case_id:
                continue
            prompt = _format_prompt(template, str(other["subject"]))
            overlap = selected & set(_flat_token_ids(tokenizer, prompt))
            (shared if overlap else unrelated).append(other)
        shared.sort(key=lambda x: (-len(selected & set(x["subject_token_ids"])), int(x["case_id"])))
        unrelated.sort(key=lambda x: int(x["case_id"]))
        for other in shared[: int(max_shared_subjects)]:
            add_negative(
                _format_prompt(template, str(other["subject"])),
                "shared_subword_subject",
                str(other["subject"]),
            )

        words = [x for x in subject.split(" ") if x]
        donors = _donor_words(prepared, subject)
        rng = random.Random(int(seed) * 1000003 + case_id * 9176 + position)
        rng.shuffle(donors)
        for component_index in range(min(len(words), int(max_leave_one_out))):
            corrupt = list(words)
            donor = donors[component_index % len(donors)]
            if normalized_key(donor) == normalized_key(words[component_index]):
                donor = "X"
            corrupt[component_index] = donor
            corrupted_subject = " ".join(corrupt)
            add_negative(
                _format_prompt(template, corrupted_subject),
                "leave_one_component_out",
                corrupted_subject,
            )

        fragments: List[str] = []
        if len(words) > 1:
            fragments.extend(words)
            fragments.extend(" ".join(words[start : start + 2]) for start in range(len(words) - 1))
        for fragment in ordered_unique(fragments)[: int(max_fragments)]:
            if normalized_key(fragment) != subject_key:
                add_negative(
                    _format_prompt(template, fragment),
                    "subject_fragment",
                    fragment,
                )

        for other in unrelated[: int(max_unrelated)]:
            add_negative(
                _format_prompt(template, str(other["subject"])),
                "unrelated_subject",
                str(other["subject"]),
            )

        # Preserve the first occurrence and its strongest provenance label.
        dedup: Dict[str, Dict[str, Any]] = {}
        priority = {
            "shared_subword_subject": 0,
            "leave_one_component_out": 1,
            "subject_fragment": 2,
            "unrelated_subject": 3,
        }
        for row in candidates:
            key = normalized_key(row["prompt"])
            current = dedup.get(key)
            if current is None or priority[row["kind"]] < priority[current["kind"]]:
                dedup[key] = row
        negatives = list(dedup.values())
        if not negatives:
            raise ValueError(f"case {case_id} has no compositional negatives")
        counts = Counter(str(row["kind"]) for row in negatives)
        global_counts.update(counts)
        result[case_id] = {
            "case_id": case_id,
            "subject": subject,
            "positive_prompts": positives,
            "negative_contexts": negatives,
            "counts": dict(counts),
            "negative_prompts_with_selected_row": sum(
                int(row["contains_selected_row"]) for row in negatives
            ),
        }

    report = {
        "records": len(result),
        "positive_prompts": sum(len(x["positive_prompts"]) for x in result.values()),
        "negative_prompts": sum(len(x["negative_contexts"]) for x in result.values()),
        "negative_kind_counts": dict(global_counts),
        "official_paraphrases_seen": 0,
        "official_neighborhoods_seen": 0,
        "benchmark_retain_seen": 0,
        "official_ppl_seen": False,
    }
    return result, report


def orthonormal_row_basis(rows: torch.Tensor, max_rank: int | None = None) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError("rows must be a matrix")
    if rows.numel() == 0:
        return rows.new_empty((0, rows.shape[1]), dtype=torch.float32)
    matrix = rows.float()
    _, singular, right = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = (
        max(matrix.shape)
        * torch.finfo(matrix.dtype).eps
        * singular.max().clamp_min(1.0)
    )
    rank = int((singular > tolerance).sum().item())
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return right[:rank].contiguous()


def project_out(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if rows.numel() == 0 or basis.numel() == 0:
        return rows
    return rows - (rows @ basis.T) @ basis


def select_contrastive_marker(
    positive_reach: torch.Tensor,
    negative_reach: torch.Tensor,
    *,
    forbidden_basis: torch.Tensor | None = None,
    ridge: float = 1e-4,
    max_rank: int = 128,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Generalized-eigen marker maximizing positive/negative reachability.

    The generalized problem is solved only in the span reachable across the
    positive contexts, keeping the solve at ``O(rank^3)`` rather than
    ``O(hidden_size^3)``.
    """
    if positive_reach.ndim != 2 or positive_reach.shape[0] == 0:
        raise ValueError("positive_reach must contain at least one row")
    if negative_reach.ndim != 2 or negative_reach.shape[1] != positive_reach.shape[1]:
        raise ValueError("negative_reach has incompatible shape")
    if not math.isfinite(float(ridge)) or float(ridge) <= 0:
        raise ValueError("ridge must be finite and positive")
    forbidden = (
        positive_reach.new_empty((0, positive_reach.shape[1]))
        if forbidden_basis is None
        else orthonormal_row_basis(forbidden_basis.float())
    )
    residual = project_out(positive_reach.float(), forbidden)
    basis = orthonormal_row_basis(residual, max_rank=int(max_rank))
    if basis.shape[0] == 0:
        raise RuntimeError("no positive reachable direction survives the forbidden basis")

    z_pos = positive_reach.float() @ basis.T
    z_neg = negative_reach.float() @ basis.T
    a = (z_pos.T @ z_pos) / max(1, z_pos.shape[0])
    if z_neg.shape[0]:
        b = (z_neg.T @ z_neg) / z_neg.shape[0]
    else:
        b = torch.zeros_like(a)
    scale = float(torch.trace(a).clamp_min(1e-12) / max(1, a.shape[0]))
    effective_ridge = float(ridge) * max(scale, 1e-12)
    b = b + effective_ridge * torch.eye(b.shape[0], dtype=b.dtype, device=b.device)
    chol = torch.linalg.cholesky(b)
    left = torch.linalg.solve_triangular(chol, a, upper=False)
    whitened = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    whitened = 0.5 * (whitened + whitened.T)
    values, vectors = torch.linalg.eigh(whitened)
    y = vectors[:, -1]
    coefficient = torch.linalg.solve_triangular(
        chol.T, y.unsqueeze(1), upper=True
    ).squeeze(1)
    marker = coefficient @ basis
    marker = marker / marker.norm().clamp_min(1e-12)

    pos_energy = float((positive_reach.float() @ marker).square().mean())
    neg_energy = (
        float((negative_reach.float() @ marker).square().mean())
        if negative_reach.shape[0]
        else 0.0
    )
    forbidden_max = (
        float((forbidden @ marker).abs().max()) if forbidden.numel() else 0.0
    )
    report = {
        "positive_rows": int(positive_reach.shape[0]),
        "negative_rows": int(negative_reach.shape[0]),
        "candidate_rank": int(basis.shape[0]),
        "positive_energy": pos_energy,
        "negative_energy": neg_energy,
        "contrastive_ratio": pos_energy / (neg_energy + effective_ridge),
        "effective_ridge": effective_ridge,
        "leading_generalized_eigenvalue": float(values[-1]),
        "forbidden_projection_abs_max": forbidden_max,
    }
    return marker, report


def conjugate_gradient(
    matvec,
    rhs: torch.Tensor,
    *,
    max_steps: int = 256,
    tolerance: float = 1e-7,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if rhs.ndim != 1:
        raise ValueError("conjugate-gradient rhs must be one-dimensional")
    x = torch.zeros_like(rhs)
    residual = rhs - matvec(x)
    direction = residual.clone()
    squared = torch.dot(residual, residual)
    initial = float(torch.sqrt(squared).clamp_min(1e-30))
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        product = matvec(direction)
        denom = torch.dot(direction, product).clamp_min(1e-30)
        alpha = squared / denom
        x = x + alpha * direction
        residual = residual - alpha * product
        next_squared = torch.dot(residual, residual)
        if float(torch.sqrt(next_squared)) <= float(tolerance) * max(initial, 1e-12):
            squared = next_squared
            break
        beta = next_squared / squared.clamp_min(1e-30)
        direction = residual + beta * direction
        squared = next_squared
    return x, {
        "cg_steps": int(steps),
        "cg_initial_residual": initial,
        "cg_final_residual": float(torch.sqrt(squared).clamp_min(0.0)),
    }


def distributional_reader(
    marker: torch.Tensor,
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
    *,
    ridge: float = 0.05,
    anchor_weight: float = 10.0,
    consistency_weight: float = 1.0,
    negative_weight: float = 1.0,
    refine_steps: int = 200,
    refine_lr: float = 0.05,
    positive_floor: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Fit ``q`` from distributions of positive and negative hidden states.

    The initial candidate is a regularized discriminant, projected into the
    exact span-complement of the registered negative states:

    ``(C_neg + lambda*C_pos_centered + ridge*I + gamma*I)^-1
      (mu_pos + gamma*v)``.

    A small fixed-state refinement stays inside that negative nullspace and
    enforces a positive cone over every training-safe positive context. It
    trains no model parameter and consumes no held-out probe.
    """
    if marker.ndim != 1:
        raise ValueError("marker must be one-dimensional")
    hidden = marker.shape[0]
    if positive_states.ndim != 2 or positive_states.shape != (positive_states.shape[0], hidden):
        raise ValueError("positive_states has incompatible shape")
    if positive_states.shape[0] == 0:
        raise ValueError("positive_states must not be empty")
    if negative_states.ndim != 2 or negative_states.shape[1] != hidden:
        raise ValueError("negative_states has incompatible shape")
    for value, name in (
        (ridge, "ridge"),
        (anchor_weight, "anchor_weight"),
        (consistency_weight, "consistency_weight"),
        (negative_weight, "negative_weight"),
    ):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if float(ridge) + float(anchor_weight) <= 0:
        raise ValueError("ridge + anchor_weight must be positive")

    device = positive_states.device
    dtype = torch.float32
    v = F.normalize(marker.to(device=device, dtype=dtype), dim=0, eps=1e-12)
    pos = F.normalize(positive_states.float(), dim=1, eps=1e-12)
    neg = F.normalize(negative_states.float(), dim=1, eps=1e-12)
    mu = pos.mean(dim=0)
    centered = pos - mu

    # The first run showed why a covariance penalty is not enough here:
    # cos(v,q) stayed near one while kappa reached 158 because the large base
    # hidden-state component survived.  In this high-dimensional, low-sample
    # regime we can impose the locality condition directly.  Remove the span
    # of every training-safe negative from both the positive mean and marker,
    # then optimize only inside that negative nullspace.  Thus q.h_negative is
    # zero by construction on the registered controls, while the remaining
    # degrees of freedom maximize the worst positive projection.
    negative_basis = orthonormal_row_basis(
        neg,
        max_rank=min(int(neg.shape[0]), max(0, hidden - 1)),
    )
    residual_pos = project_out(pos, negative_basis)
    residual_marker = project_out(v.unsqueeze(0), negative_basis).squeeze(0)
    residual_mu = residual_pos.mean(dim=0)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        result = (float(ridge) + float(anchor_weight)) * vector
        if neg.shape[0] and float(negative_weight) > 0:
            result = result + float(negative_weight) * (
                neg.T @ (neg @ vector)
            ) / neg.shape[0]
        if centered.shape[0] > 1 and float(consistency_weight) > 0:
            result = result + float(consistency_weight) * (
                centered.T @ (centered @ vector)
            ) / centered.shape[0]
        return result

    rhs = residual_mu + float(anchor_weight) * residual_marker
    initial, cg_report = conjugate_gradient(matvec, rhs)
    initial = project_out(initial.unsqueeze(0), negative_basis).squeeze(0)
    if float(initial.norm()) < 1e-10:
        initial = residual_mu
    if float(initial.norm()) < 1e-10:
        raise RuntimeError("positive states have no direction outside the negative span")
    q = F.normalize(initial, dim=0, eps=1e-12)
    if float((pos @ q).mean()) < 0:
        q = -q

    initial_scores = pos @ q
    initial_min = float(initial_scores.min())
    if int(refine_steps) > 0:
        # Deliberately use an explicit projected-gradient update.  The object
        # being refined is one fixed hidden-space vector, not a model module;
        # pulling in an optimizer (and its torch.compile/SymPy import chain)
        # adds environment sensitivity without buying useful state here.
        parameter = q.detach().clone().requires_grad_(True)
        momentum = torch.zeros_like(parameter)
        for _ in range(int(refine_steps)):
            unit = F.normalize(
                project_out(parameter.unsqueeze(0), negative_basis).squeeze(0),
                dim=0,
                eps=1e-12,
            )
            pos_scores = pos @ unit
            hinge = F.relu(float(positive_floor) - pos_scores).square().mean()
            consistency = pos_scores.var(unbiased=False)
            negative = (neg @ unit).square().mean() if neg.shape[0] else unit.sum() * 0.0
            anchor = 1.0 - torch.dot(unit, v)
            loss = (
                200.0 * hinge
                + 0.1 * float(consistency_weight) * consistency
                + float(negative_weight) * negative
                + float(anchor_weight) * anchor
            )
            gradient = torch.autograd.grad(loss, parameter)[0]
            with torch.no_grad():
                momentum.mul_(0.9).add_(gradient)
                updated = parameter - float(refine_lr) * momentum
                updated = project_out(
                    updated.unsqueeze(0), negative_basis
                ).squeeze(0)
                updated = F.normalize(updated, dim=0, eps=1e-12)
            parameter = updated.detach().requires_grad_(True)
        q = F.normalize(parameter.detach(), dim=0, eps=1e-12)
        if float((pos @ q).mean()) < 0:
            q = -q

    report = {
        **cg_report,
        "initial_normalized_positive_min": initial_min,
        "final_normalized_positive_min": float((pos @ q).min()),
        "final_normalized_positive_max": float((pos @ q).max()),
        "final_normalized_negative_abs_max": (
            float((neg @ q).abs().max()) if neg.shape[0] else 0.0
        ),
        "cos_marker_q": float(torch.dot(v, q)),
        "negative_nullspace_rank": int(negative_basis.shape[0]),
        "refine_steps": int(refine_steps),
    }
    return q.cpu(), report


def reader_metrics(
    reader: torch.Tensor,
    positive_states: torch.Tensor,
    negative_states: torch.Tensor,
) -> Dict[str, float]:
    q = F.normalize(reader.float(), dim=0, eps=1e-12)
    positive = positive_states.float() @ q
    negative = negative_states.float() @ q
    abs_positive = positive.abs()
    s_min = float(abs_positive.min())
    s_max = float(abs_positive.max())
    l_max = float(negative.abs().max()) if negative.numel() else 0.0
    return {
        "positive_signed_min": float(positive.min()),
        "positive_signed_max": float(positive.max()),
        "S_min": s_min,
        "S_max": s_max,
        "L_max": l_max,
        "kappa_train": l_max / (s_min + 1e-9),
        "portability_ratio": s_min / (s_max + 1e-9),
        "positive_sign_consistent": bool(float(positive.min()) > 0.0),
    }


def directional_row_deltas(
    readers: torch.Tensor,
    betas: torch.Tensor,
    answer_rows_by_record: Sequence[Sequence[int]],
    selected_output_rows: Sequence[int],
) -> torch.Tensor:
    """Assemble ``Delta W_y = -sum_i beta_i q_i`` on sparse answer rows."""
    if readers.ndim != 2:
        raise ValueError("readers must be [records, hidden]")
    if betas.ndim != 1 or betas.shape[0] != readers.shape[0]:
        raise ValueError("betas must contain one value per reader")
    if len(answer_rows_by_record) != readers.shape[0]:
        raise ValueError("answer_rows_by_record must contain one row list per reader")
    row_slot = {int(token_id): slot for slot, token_id in enumerate(selected_output_rows)}
    membership = readers.new_zeros((len(selected_output_rows), readers.shape[0]))
    for record_index, token_ids in enumerate(answer_rows_by_record):
        for token_id in set(int(x) for x in token_ids):
            if token_id not in row_slot:
                raise ValueError(f"answer token {token_id} missing from selected output rows")
            membership[row_slot[token_id], record_index] = 1.0
    return -((membership * betas.unsqueeze(0)) @ readers)


def factorize_output_rows(delta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the exact row-wise factorization ``delta_y = -beta_y q_y``.

    ``beta_y`` is the non-negative row norm. Active ``q_y`` rows are unit
    vectors; zero rows receive an all-zero reader. This is an identity, not a
    low-rank approximation, so a jointly optimized sparse LM-head delta keeps
    the linear-reader interpretation without imposing an infeasible shared
    reader across different sensitive output tokens.
    """
    if delta.ndim != 2:
        raise ValueError("delta must be [rows, hidden]")
    betas = delta.norm(dim=1)
    readers = torch.zeros_like(delta)
    active = betas > 1e-12
    readers[active] = -delta[active] / betas[active].unsqueeze(1)
    return betas, readers


def monotone_cover_betas(
    response: torch.Tensor,
    required_margin_gain: torch.Tensor,
    *,
    safety_factor: float = 1.25,
    max_steps: int = 10000,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Solve non-negative linearized margin constraints monotonically.

    ``response[i, j]`` is the predicted increase in instance ``i``'s
    sensitive-vs-reference NLL margin per unit of reader ``j``.  Negative
    cross-reader responses are excluded from the initializer (the exact
    Stage-2 optimizer still sees them).  Repeated projections onto the most
    violated non-negative halfspace cannot undo a previously accumulated
    margin because every retained coefficient is non-negative.
    """
    if response.ndim != 2:
        raise ValueError("response must be [instances, readers]")
    if required_margin_gain.ndim != 1 or required_margin_gain.shape[0] != response.shape[0]:
        raise ValueError("required-margin vector has incompatible shape")
    if not math.isfinite(float(safety_factor)) or float(safety_factor) < 1.0:
        raise ValueError("safety_factor must be finite and >=1")
    matrix = response.float().clamp_min(0.0)
    target = required_margin_gain.float().clamp_min(0.0) * float(safety_factor)
    beta = torch.zeros(matrix.shape[1], dtype=torch.float32, device=matrix.device)
    steps = 0
    for steps in range(1, int(max_steps) + 1):
        deficit = target - matrix @ beta
        worst_value, worst_index = deficit.max(dim=0)
        if float(worst_value) <= 1e-5:
            break
        row = matrix[int(worst_index)]
        norm_sq = torch.dot(row, row)
        if float(norm_sq) <= 1e-12:
            raise RuntimeError(
                f"instance {int(worst_index)} has no positive reader response"
            )
        beta = beta + (worst_value / norm_sq) * row
    residual = target - matrix @ beta
    return beta.cpu(), {
        "steps": int(steps),
        "instances": int(matrix.shape[0]),
        "readers": int(matrix.shape[1]),
        "response_positive_fraction": float((matrix > 0).float().mean()),
        "target_max": float(target.max()) if target.numel() else 0.0,
        "residual_max": float(residual.max()) if residual.numel() else 0.0,
        "beta_min": float(beta.min()) if beta.numel() else 0.0,
        "beta_median": float(beta.median()) if beta.numel() else 0.0,
        "beta_max": float(beta.max()) if beta.numel() else 0.0,
    }
