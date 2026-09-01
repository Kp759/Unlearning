#!/usr/bin/env python3
"""Core math for MCF internal fact-conditional embedding rewiring V1.

This module deliberately contains no dataset or Transformers dependency.  It
implements the overlap-aware subject code, the capacity-limited contextual
classifier, and the frozen-threshold certificate used by the training-only
preflight.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


PROTOCOL = "mcf_internal_fact_conditional_embedding_rewiring_v1"
SCHEMA_VERSION = 1


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    return sha256_bytes(tensor.view(torch.uint8).numpy().tobytes())


def _flat_token_ids(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized sequence")
        value = value[0]
    return [int(item) for item in value]


def subject_token_rows(
    tokenizer: Any,
    subject: str,
    *,
    excluded_token_ids: Iterable[int] = (),
) -> List[int]:
    """Union sentence-initial and whitespace-prefixed subject token rows."""

    text = str(subject).strip()
    if not text:
        raise ValueError("subject cannot be empty")
    excluded = {int(item) for item in excluded_token_ids if item is not None}
    found: set[int] = set()
    for variant in (text, " " + text):
        encoded = tokenizer(variant, add_special_tokens=False)["input_ids"]
        found.update(_flat_token_ids(encoded))
    result = sorted(found - excluded)
    if not result:
        raise ValueError(f"subject {text!r} has no editable token rows")
    return result


def build_subject_incidence(
    token_rows_by_subject: Sequence[Sequence[int]],
) -> Tuple[List[int], torch.Tensor, Dict[int, List[int]]]:
    """Return editable rows and a normalized subject-by-row incidence matrix.

    A token row appears exactly once in the output even when several subjects
    share it.  Each subject row sums to one, so ``A @ C`` is the mean code
    contributed by that complete subject's editable subwords.
    """

    if not token_rows_by_subject:
        raise ValueError("subject token rows cannot be empty")
    normalized: List[List[int]] = []
    ownership: Dict[int, List[int]] = {}
    for subject_index, values in enumerate(token_rows_by_subject):
        rows = sorted({int(item) for item in values})
        if not rows:
            raise ValueError(f"subject {subject_index} has no token rows")
        normalized.append(rows)
        for token_id in rows:
            ownership.setdefault(token_id, []).append(subject_index)
    token_ids = sorted(ownership)
    column = {token_id: index for index, token_id in enumerate(token_ids)}
    incidence = torch.zeros((len(normalized), len(token_ids)), dtype=torch.float64)
    for subject_index, rows in enumerate(normalized):
        weight = 1.0 / len(rows)
        for token_id in rows:
            incidence[subject_index, column[token_id]] = weight
    return token_ids, incidence, ownership


def deterministic_subject_codes(records: int, rank: int, seed: int) -> torch.Tensor:
    """Generate fixed unit-norm subject keys without inspecting model states."""

    if records <= 0 or rank <= 0:
        raise ValueError("subject-code dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729)
    codes = torch.randn((int(records), int(rank)), generator=generator)
    codes = F.normalize(codes, dim=1)
    if int(torch.linalg.matrix_rank(codes).item()) != min(records, rank):
        raise RuntimeError("deterministic subject code unexpectedly lost rank")
    return codes.to(torch.float64)


def deterministic_orthonormal_basis(
    hidden_size: int, rank: int, seed: int
) -> torch.Tensor:
    """Generate a fixed embedding-space basis with orthonormal rows."""

    if not 0 < rank <= hidden_size:
        raise ValueError("embedding basis rank must lie in (0, hidden_size]")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 130363)
    matrix = torch.randn(
        (int(hidden_size), int(rank)),
        generator=generator,
        dtype=torch.float64,
    )
    q, _r = torch.linalg.qr(matrix, mode="reduced")
    basis = q.T.contiguous().to(torch.float64)
    gram = basis @ basis.T
    if not torch.allclose(gram, torch.eye(rank, dtype=gram.dtype), atol=1e-10):
        raise RuntimeError("embedding basis is not orthonormal")
    return basis


def code_reconstruction_certificate(
    achieved_codes: torch.Tensor,
    target_codes: torch.Tensor,
    *,
    nearest_key_margin_floor: float,
) -> Dict[str, Any]:
    """Certify that every achieved complete-subject code decodes to its key."""

    if achieved_codes.shape != target_codes.shape or achieved_codes.ndim != 2:
        raise ValueError("achieved and target subject codes must have equal shape")
    achieved = F.normalize(achieved_codes.float(), dim=1, eps=1e-12)
    target = F.normalize(target_codes.float(), dim=1, eps=1e-12)
    similarities = achieved @ target.T
    own = similarities.diag()
    masked = similarities.clone()
    masked.fill_diagonal_(float("-inf"))
    cross = masked.max(dim=1).values
    margins = own - cross
    nonzero = achieved_codes.float().norm(dim=1).gt(1e-10)
    passed_rows = nonzero & margins.ge(float(nearest_key_margin_floor))
    relative_error = (achieved_codes.float() - target_codes.float()).norm(
        dim=1
    ) / target_codes.float().norm(dim=1).clamp_min(1e-12)
    return {
        "records": int(achieved_codes.shape[0]),
        "rank": int(achieved_codes.shape[1]),
        "nearest_key_margin_floor": float(nearest_key_margin_floor),
        "own_cosine_min": float(own.min()),
        "own_cosine_median": float(own.median()),
        "cross_cosine_max": float(cross.max()),
        "nearest_key_margin_min": float(margins.min()),
        "relative_reconstruction_error_max": float(relative_error.max()),
        "relative_reconstruction_error_median": float(relative_error.median()),
        "zero_code_records": int((~nonzero).sum()),
        "failed_records": [
            int(index)
            for index, value in enumerate(passed_rows.tolist())
            if not bool(value)
        ],
        "passed": bool(passed_rows.all()),
    }


def solve_overlap_aware_embedding_code(
    incidence: torch.Tensor,
    target_codes: torch.Tensor,
    basis: torch.Tensor,
    base_embedding_rows: torch.Tensor,
    token_frequencies: torch.Tensor,
    *,
    ridge_lambda: float,
    relative_row_cap: float,
    frequency_alpha: float,
    nearest_key_margin_floor: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Solve one jointly consistent embedding delta for all shared rows."""

    if incidence.ndim != 2 or target_codes.ndim != 2 or basis.ndim != 2:
        raise ValueError("incidence, target codes and basis must be matrices")
    records, rows = incidence.shape
    if target_codes.shape[0] != records or target_codes.shape[1] != basis.shape[0]:
        raise ValueError("subject-code dimensions are incompatible")
    if base_embedding_rows.shape != (rows, basis.shape[1]):
        raise ValueError("Base embedding rows are incompatible")
    if token_frequencies.shape != (rows,):
        raise ValueError("token frequencies must provide one value per row")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge lambda must be finite and positive")
    if not math.isfinite(relative_row_cap) or relative_row_cap <= 0:
        raise ValueError("relative row cap must be finite and positive")
    if not math.isfinite(frequency_alpha) or frequency_alpha < 0:
        raise ValueError("frequency alpha must be finite and non-negative")

    a = incidence.to(torch.float64)
    k = target_codes.to(torch.float64)
    b = basis.to(torch.float64)
    frequencies = token_frequencies.to(torch.float64).clamp_min(0)
    penalty = (1.0 + frequencies).pow(float(frequency_alpha))
    lhs = a.T @ a + float(ridge_lambda) * torch.diag(penalty)
    rhs = a.T @ k
    coefficients = torch.linalg.solve(lhs, rhs)

    raw_delta = coefficients @ b
    base_norms = base_embedding_rows.detach().to(torch.float64).norm(dim=1)
    caps = (
        float(relative_row_cap)
        * base_norms
        / (1.0 + frequencies).pow(float(frequency_alpha))
    )
    raw_norms = raw_delta.norm(dim=1)
    projection_scale = torch.minimum(
        torch.ones_like(raw_norms), caps / raw_norms.clamp_min(1e-12)
    )
    projected_coefficients = coefficients * projection_scale.unsqueeze(1)
    delta = projected_coefficients @ b
    achieved = a @ projected_coefficients
    certificate = code_reconstruction_certificate(
        achieved,
        k,
        nearest_key_margin_floor=float(nearest_key_margin_floor),
    )
    gram = a @ a.T
    condition = torch.linalg.cond(gram)
    condition_value = float(condition) if bool(torch.isfinite(condition)) else None
    relative_norms = delta.norm(dim=1) / base_norms.clamp_min(1e-12)
    relative_cap_limits = float(relative_row_cap) / (1.0 + frequencies).pow(
        float(frequency_alpha)
    )
    cap_excess = relative_norms - relative_cap_limits
    cap_violations = int(cap_excess.gt(1e-8).sum())
    certificate.update(
        {
            "ridge_lambda": float(ridge_lambda),
            "relative_row_cap": float(relative_row_cap),
            "frequency_alpha": float(frequency_alpha),
            "incidence_rank": int(torch.linalg.matrix_rank(a).item()),
            "incidence_rows": int(records),
            "incidence_columns": int(rows),
            "incidence_gram_condition_number": condition_value,
            "projected_rows": int(projection_scale.lt(1.0 - 1e-12).sum()),
            "delta_relative_norm_max": float(relative_norms.max()),
            "delta_relative_norm_median": float(relative_norms.median()),
            "frequency_adjusted_relative_cap_max": float(relative_cap_limits.max()),
            "frequency_adjusted_relative_cap_min": float(relative_cap_limits.min()),
            "relative_cap_max_excess": float(cap_excess.max()),
            "relative_cap_violations": cap_violations,
            "delta_abs_norm_max": float(delta.norm(dim=1).max()),
            "finite": bool(torch.isfinite(delta).all()),
        }
    )
    certificate["passed"] = bool(
        certificate["passed"] and certificate["finite"] and cap_violations == 0
    )
    return delta.to(torch.float32), certificate


def rms_normalize(hidden: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if hidden.ndim != 2:
        raise ValueError("classifier hidden states must be [rows, hidden]")
    value = hidden.float()
    return value * torch.rsqrt(value.square().mean(dim=1, keepdim=True) + epsilon)


class FactorizedFactClassifier(nn.Module):
    """Shared rank-limited subject/relation towers with a soft AND output.

    The classifier receives contextual hidden states only.  Subject and
    relation metadata are labels used by the loss; they are never inputs to
    ``forward``.
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        fact_relation_index: Sequence[int] | torch.Tensor,
        relation_count: int,
        *,
        softmin_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        relation_index = torch.as_tensor(fact_relation_index, dtype=torch.long)
        if hidden_size <= 0 or rank <= 0 or relation_count <= 0:
            raise ValueError("classifier dimensions must be positive")
        if relation_index.ndim != 1 or relation_index.numel() == 0:
            raise ValueError("fact relation index must cover at least one fact")
        if bool(relation_index.lt(0).any()) or bool(
            relation_index.ge(int(relation_count)).any()
        ):
            raise ValueError("fact relation indices are out of range")
        if not softmin_temperature > 0:
            raise ValueError("softmin temperature must be positive")
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.facts = int(relation_index.numel())
        self.relation_count = int(relation_count)
        self.softmin_temperature = float(softmin_temperature)
        self.subject_projection = nn.Linear(hidden_size, rank, bias=False)
        self.relation_projection = nn.Linear(hidden_size, rank, bias=False)
        self.subject_coefficients = nn.Parameter(torch.empty(self.facts, rank))
        self.subject_bias = nn.Parameter(torch.full((self.facts,), -0.5))
        self.relation_coefficients = nn.Parameter(
            torch.empty(self.relation_count, rank)
        )
        self.relation_bias = nn.Parameter(torch.full((self.relation_count,), -0.5))
        self.register_buffer("fact_relation_index", relation_index)
        nn.init.orthogonal_(self.subject_projection.weight)
        nn.init.orthogonal_(self.relation_projection.weight)
        nn.init.normal_(self.subject_coefficients, mean=0.0, std=0.05)
        nn.init.normal_(self.relation_coefficients, mean=0.0, std=0.05)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        normalized = rms_normalize(hidden)
        subject_latent = torch.tanh(self.subject_projection(normalized))
        relation_latent = torch.tanh(self.relation_projection(normalized))
        subject_scores = F.linear(
            subject_latent, self.subject_coefficients, self.subject_bias
        )
        relation_all = F.linear(
            relation_latent, self.relation_coefficients, self.relation_bias
        )
        relation_scores = relation_all.index_select(1, self.fact_relation_index)
        temperature = self.softmin_temperature
        fact_scores = -temperature * torch.logsumexp(
            torch.stack(
                (-subject_scores / temperature, -relation_scores / temperature),
                dim=0,
            ),
            dim=0,
        )
        return {
            "fact_scores": fact_scores,
            "subject_scores": subject_scores,
            "relation_scores_by_fact": relation_scores,
            "relation_scores": relation_all,
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def balanced_squared_hinge(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    positive_floor: float,
    negative_ceiling: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if scores.shape != labels.shape:
        raise ValueError("classifier scores and labels must have equal shape")
    labels = labels.bool()
    zero = scores.sum() * 0.0
    positive = F.relu(float(positive_floor) - scores[labels]).square()
    negative = F.relu(scores[~labels] - float(negative_ceiling)).square()
    positive_loss = positive.mean() if positive.numel() else zero
    negative_loss = negative.mean() if negative.numel() else zero
    return positive_loss + negative_loss, {
        "positive": positive_loss,
        "negative": negative_loss,
    }


def factorized_classifier_loss(
    output: Mapping[str, torch.Tensor],
    *,
    fact_labels: torch.Tensor,
    subject_labels: torch.Tensor,
    relation_labels: torch.Tensor,
    positive_floor: float,
    negative_ceiling: float,
    auxiliary_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    fact_loss, fact_parts = balanced_squared_hinge(
        output["fact_scores"],
        fact_labels,
        positive_floor=positive_floor,
        negative_ceiling=negative_ceiling,
    )
    subject_loss, subject_parts = balanced_squared_hinge(
        output["subject_scores"],
        subject_labels,
        positive_floor=positive_floor,
        negative_ceiling=negative_ceiling,
    )
    relation_loss, relation_parts = balanced_squared_hinge(
        output["relation_scores"],
        relation_labels,
        positive_floor=positive_floor,
        negative_ceiling=negative_ceiling,
    )
    total = fact_loss + float(auxiliary_weight) * (subject_loss + relation_loss)
    return total, {
        "fact": fact_loss,
        "fact_positive": fact_parts["positive"],
        "fact_negative": fact_parts["negative"],
        "subject": subject_loss,
        "subject_positive": subject_parts["positive"],
        "subject_negative": subject_parts["negative"],
        "relation": relation_loss,
        "relation_positive": relation_parts["positive"],
        "relation_negative": relation_parts["negative"],
    }


def score_separation_report(
    scores: torch.Tensor, labels: torch.Tensor
) -> Dict[str, Any]:
    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("fact scores and labels must be equal matrices")
    labels = labels.bool()
    if not bool(labels.any()) or not bool((~labels).any()):
        raise ValueError("separation report requires positive and negative cells")
    positives = scores[labels].detach().float().cpu()
    negatives = scores[~labels].detach().float().cpu()
    negative_max = float(negatives.max())
    threshold = math.nextafter(negative_max, math.inf)
    positive_failures = int(positives.lt(threshold).sum())
    return {
        "positive_cells": int(positives.numel()),
        "negative_cells": int(negatives.numel()),
        "positive_min": float(positives.min()),
        "positive_median": float(positives.median()),
        "negative_max": negative_max,
        "negative_median": float(negatives.median()),
        "separation_gap": float(positives.min()) - negative_max,
        "provisional_threshold": threshold,
        "positive_failures_at_provisional_threshold": positive_failures,
        "negative_failures_at_provisional_threshold": 0,
        "perfectly_separated": positive_failures == 0,
    }


def fit_negative_standardization(
    scores: torch.Tensor, labels: torch.Tensor, *, minimum_std: float = 1e-4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit per-fact score location/scale from fit negatives only."""

    if scores.shape != labels.shape or scores.ndim != 2:
        raise ValueError("standardization scores and labels must be equal matrices")
    means: List[torch.Tensor] = []
    scales: List[torch.Tensor] = []
    for fact in range(scores.shape[1]):
        values = scores[:, fact][~labels[:, fact].bool()].detach().float()
        if values.numel() < 2:
            raise ValueError(f"fact {fact} lacks two fit-negative scores")
        means.append(values.mean())
        scales.append(values.std(unbiased=False).clamp_min(float(minimum_std)))
    return torch.stack(means), torch.stack(scales)


def standardize_fact_scores(
    scores: torch.Tensor, means: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    if (
        scores.ndim != 2
        or means.shape != (scores.shape[1],)
        or scales.shape != means.shape
    ):
        raise ValueError("score standardization shapes are incompatible")
    return (scores.float() - means.to(scores.device)) / scales.to(scores.device)


def calibrate_global_threshold(
    standardized_scores: torch.Tensor, labels: torch.Tensor
) -> Dict[str, Any]:
    """Set one fail-closed threshold immediately above calibration negatives."""

    report = score_separation_report(standardized_scores, labels)
    threshold = math.nextafter(float(report["negative_max"]), math.inf)
    report.update(
        {
            "threshold": threshold,
            "selection_rule": "nextafter(max_negative_score, +infinity)",
            "threshold_frozen": True,
            "passed": int(report["positive_failures_at_provisional_threshold"]) == 0,
        }
    )
    return report


def frozen_threshold_certificate(
    standardized_scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
    distinct_prompts: int,
    minimum_negative_cells: int,
    minimum_distinct_prompts: int,
) -> Dict[str, Any]:
    """Evaluate a previously frozen threshold without recalibration."""

    if standardized_scores.shape != labels.shape or standardized_scores.ndim != 2:
        raise ValueError("certificate scores and labels must be equal matrices")
    labels = labels.bool()
    positives = standardized_scores[labels].detach().float().cpu()
    negatives = standardized_scores[~labels].detach().float().cpu()
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("certificate requires positive and negative cells")
    positive_failures = int(positives.lt(float(threshold)).sum())
    negative_failures = int(negatives.ge(float(threshold)).sum())
    negative_cells = int(negatives.numel())
    passed = (
        positive_failures == 0
        and negative_failures == 0
        and negative_cells >= int(minimum_negative_cells)
        and int(distinct_prompts) >= int(minimum_distinct_prompts)
    )
    return {
        "threshold": float(threshold),
        "positive_cells": int(positives.numel()),
        "negative_cells": negative_cells,
        "distinct_prompts": int(distinct_prompts),
        "minimum_negative_cells": int(minimum_negative_cells),
        "minimum_distinct_prompts": int(minimum_distinct_prompts),
        "positive_min": float(positives.min()),
        "negative_max": float(negatives.max()),
        "positive_failures": positive_failures,
        "negative_failures": negative_failures,
        "cell_level_rule_of_three_95_percent_upper_bound": 3.0 / negative_cells,
        "prompt_level_rule_of_three_95_percent_upper_bound": 3.0
        / int(distinct_prompts),
        "cell_independence_not_assumed": True,
        "passed": bool(passed),
    }


def per_kind_threshold_audit(
    standardized_scores: torch.Tensor,
    labels: torch.Tensor,
    kinds: Sequence[Sequence[str]],
    *,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    """Report threshold outcomes separately for every registered prompt family.

    A canonical prompt can carry more than one provenance kind.  Such a row is
    intentionally included in each applicable family; this is an audit view,
    not an independent-sample count.
    """

    if standardized_scores.shape != labels.shape or standardized_scores.ndim != 2:
        raise ValueError("kind audit scores and labels must be equal matrices")
    if len(kinds) != standardized_scores.shape[0]:
        raise ValueError("kind audit provenance must cover every prompt row")
    scores = standardized_scores.detach().float().cpu()
    labels = labels.detach().bool().cpu()
    kind_rows: Dict[str, List[int]] = {}
    for row_index, row_kinds in enumerate(kinds):
        for kind in set(str(value) for value in row_kinds):
            kind_rows.setdefault(kind, []).append(row_index)
    report: Dict[str, Dict[str, Any]] = {}
    for kind, row_indices in sorted(kind_rows.items()):
        index = torch.tensor(row_indices, dtype=torch.long)
        selected_scores = scores.index_select(0, index)
        selected_labels = labels.index_select(0, index)
        positive_scores = selected_scores[selected_labels]
        negative_scores = selected_scores[~selected_labels]
        report[kind] = {
            "rows": len(row_indices),
            "positive_cells": int(positive_scores.numel()),
            "negative_cells": int(negative_scores.numel()),
            "positive_failures": int(positive_scores.lt(float(threshold)).sum()),
            "negative_failures": int(negative_scores.ge(float(threshold)).sum()),
            "positive_min": (
                float(positive_scores.min()) if positive_scores.numel() else None
            ),
            "negative_max": (
                float(negative_scores.max()) if negative_scores.numel() else None
            ),
            "passed": bool(
                (
                    positive_scores.numel() == 0
                    or positive_scores.ge(float(threshold)).all()
                )
                and (
                    negative_scores.numel() == 0
                    or negative_scores.lt(float(threshold)).all()
                )
            ),
        }
    return report


@dataclass(frozen=True)
class SemanticPrompt:
    prompt: str
    subject: str | None
    relation_id: str | None
    kind: str
    writer_on: bool = True


@dataclass
class PromptBank:
    prompts: List[str]
    writer_on: torch.Tensor
    fact_labels: torch.Tensor
    subject_labels: torch.Tensor
    relation_labels: torch.Tensor
    kinds: List[List[str]]

    def validate(self) -> None:
        rows = len(self.prompts)
        if self.writer_on.shape != (rows,):
            raise ValueError("prompt writer mask has incompatible shape")
        if self.fact_labels.shape[0] != rows or self.subject_labels.shape[0] != rows:
            raise ValueError("prompt fact/subject labels have incompatible rows")
        if self.relation_labels.shape[0] != rows or len(self.kinds) != rows:
            raise ValueError("prompt relation/kind labels have incompatible rows")
        if not all(self.prompts) or not all(self.kinds):
            raise ValueError("prompt bank contains an empty prompt or kind")


def canonical_multilabel_prompt_bank(
    specs: Sequence[SemanticPrompt],
    *,
    fact_subjects: Sequence[str],
    fact_relation_ids: Sequence[str],
    relation_ids: Sequence[str],
) -> PromptBank:
    """Canonicalize duplicate prompt/state pairs into one multi-label row."""

    if len(fact_subjects) != len(fact_relation_ids) or not fact_subjects:
        raise ValueError("fact subjects and relation ids must cover every fact")
    relation_to_index = {str(value): index for index, value in enumerate(relation_ids)}
    if len(relation_to_index) != len(relation_ids):
        raise ValueError("relation ids must be unique")
    facts = len(fact_subjects)
    by_key: Dict[Tuple[str, bool], Dict[str, Any]] = {}
    for spec in specs:
        prompt = str(spec.prompt).strip()
        if not prompt:
            raise ValueError("semantic prompt cannot be empty")
        key = (prompt, bool(spec.writer_on))
        row = by_key.setdefault(
            key,
            {
                "subject": set(),
                "relation": set(),
                "kinds": set(),
            },
        )
        if spec.subject is not None:
            row["subject"].add(str(spec.subject))
        if spec.relation_id is not None:
            row["relation"].add(str(spec.relation_id))
        row["kinds"].add(str(spec.kind))

    prompts: List[str] = []
    writer_mask: List[bool] = []
    fact_rows: List[torch.Tensor] = []
    subject_rows: List[torch.Tensor] = []
    relation_rows: List[torch.Tensor] = []
    kinds: List[List[str]] = []
    for (prompt, writer_on), metadata in by_key.items():
        fact_label = torch.zeros(facts, dtype=torch.bool)
        subject_label = torch.zeros(facts, dtype=torch.bool)
        relation_label = torch.zeros(len(relation_ids), dtype=torch.bool)
        if writer_on:
            for fact, (subject, relation) in enumerate(
                zip(fact_subjects, fact_relation_ids)
            ):
                subject_match = str(subject) in metadata["subject"]
                relation_match = str(relation) in metadata["relation"]
                subject_label[fact] = subject_match
                fact_label[fact] = subject_match and relation_match
        for relation in metadata["relation"]:
            if relation in relation_to_index:
                relation_label[relation_to_index[relation]] = True
        prompts.append(prompt)
        writer_mask.append(writer_on)
        fact_rows.append(fact_label)
        subject_rows.append(subject_label)
        relation_rows.append(relation_label)
        kinds.append(sorted(metadata["kinds"]))
    bank = PromptBank(
        prompts=prompts,
        writer_on=torch.tensor(writer_mask, dtype=torch.bool),
        fact_labels=torch.stack(fact_rows),
        subject_labels=torch.stack(subject_rows),
        relation_labels=torch.stack(relation_rows),
        kinds=kinds,
    )
    bank.validate()
    return bank
