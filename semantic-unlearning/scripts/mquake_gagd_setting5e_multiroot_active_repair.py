#!/usr/bin/env python3
"""MQuAKE Setting 5e with protected active-pair LM-head repair.

This is a method extension.  The reproducibility baseline in
``mquake_gagd_setting5e_active_repair.py`` is intentionally unchanged.

Only sampled ``requested_rewrite`` cloze facts are visible to Setting 5e and
repair.  Natural-language atomic questions, the three record-level questions,
answers/aliases, and counterfactual ``target_new`` values remain held out until
the checkpoint selection decision has been written.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as baseline
import mquake_multihop_unlearning_eval as multihop
import mquake_zero_unlearn_official_eval as mquake
import zsre_gagd_setting5e_active_repair as repair


METHOD = "mquake_gagd_setting5e_active_pair_repair"
METHOD_LABEL = "Setting 5e + protected active LM-head repair"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE
ACTIVE_SOURCE = "sampled_requested_rewrite_cloze_teacher_forced_prefixes"
REPAIR_TYPE = "active_pair"


@dataclass
class ActivePairCache:
    case: mquake.PredictionCase
    hidden: torch.Tensor
    sensitive_token_id: int
    competitor_token_id: int
    sensitive_base_logit: torch.Tensor
    competitor_base_logit: torch.Tensor

    @property
    def identity(self) -> Tuple[int, str, int, int, int, int]:
        return (*self.case.identity, self.sensitive_token_id, self.competitor_token_id)


@dataclass
class ProtectedPairState:
    case: mquake.PredictionCase
    hidden: torch.Tensor
    correct_token_id: int
    correct_base_logit: torch.Tensor
    modified_row_base_logits: torch.Tensor
    correct_modified_row_index: int


@dataclass(frozen=True)
class ActivePairTensors:
    """Active-pair data packed once on the optimization device."""

    hidden: torch.Tensor
    base_margin: torch.Tensor
    sensitive_row_index: torch.Tensor
    competitor_row_index: torch.Tensor
    hidden_rank: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class ProtectedPairTensors:
    """Protected retain-pair data packed once on the optimization device."""

    hidden: torch.Tensor
    base_modified_logits: torch.Tensor
    correct_base: torch.Tensor
    correct_modified_row_index: torch.Tensor
    competitor_mask: torch.Tensor
    hidden_rank: Optional[torch.Tensor] = None


def build_parser() -> argparse.ArgumentParser:
    parser = baseline.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="outputs/mquake_setting5e_multiroot_active/seed0",
        steps=600,
        batch_size=1,
        retain_batch_size=4,
        emb_lm_lr=1e-4,
        forget_weight=2.0,
        retain_weight=1.0,
        forget_margin=1.0,
        emb_lm_optimizer="adamw",
        sampling_strategy="epoch",
        repair_steps=600,
        repair_lr=2e-3,
        repair_optimizer="adamw",
        active_logit_margin=0.50,
        selection_logit_margin=0.10,
        repair_rank=0,
        repair_l2_lambda=1e-6,
        retain_calibration_num=1000,
        retain_calibration_seed=1729,
        project_away_protected_hidden=False,
        stop_when_all_satisfied=True,
        target_eff_max=0.0,
        utility_drop_tolerance=0.10,
        max_ppl_ratio=1.02,
        strict_utility_gates=True,
    )
    parser.add_argument(
        "--repair-mode",
        choices=("active_pair",),
        default="active_pair",
        help="Joint sensitive/true-runner-up active-pair LM-head repair.",
    )
    parser.add_argument(
        "--protected-logit-margin",
        type=float,
        default=0.0,
        help="Nonnegative target-vs-modified-row margin for protected retain states.",
    )
    parser.add_argument(
        "--forget-sampling",
        choices=("instance_balanced", "atomic_epoch"),
        default="instance_balanced",
        help=(
            "Setting 5e forget sampling. instance_balanced samples an instance "
            "uniformly and then one of its requested_rewrite atoms uniformly."
        ),
    )
    parser.add_argument(
        "--protected-logit-drift-weight",
        type=float,
        default=1.0,
        help="Penalty on modified-row logit drift over protected retain states.",
    )
    parser.add_argument("--multihop-prompt-dir", default="data/mquake_prompts")
    parser.add_argument("--multihop-batch-size", type=int, default=4)
    parser.add_argument("--standard-max-new-tokens", type=int, default=32)
    parser.add_argument("--cot-max-new-tokens", type=int, default=128)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    repair.validate_args(args)
    if args.forget_num != 1000 or args.retain_num != 1000:
        raise ValueError(
            "The publication protocol requires 1000 forget and 1000 retain instances"
        )
    if args.skip_ppl:
        raise ValueError("The fixed Base-relative PPL gate cannot be skipped")
    if args.mquake_url != mquake.MQUAKE_URL:
        raise ValueError("The publication protocol requires the pinned MQuAKE revision")
    pinned = {
        "steps": 600,
        "batch_size": 1,
        "retain_batch_size": 4,
        "emb_lm_lr": 1e-4,
        "forget_weight": 2.0,
        "retain_weight": 1.0,
        "forget_margin": 1.0,
        "emb_lm_optimizer": "adamw",
        "sampling_strategy": "epoch",
    }
    for name, expected in pinned.items():
        if getattr(args, name) != expected:
            raise ValueError(
                f"Pinned aggressive Setting 5e requires --{name.replace('_', '-')} "
                f"{expected}"
            )
    if args.repair_mode != REPAIR_TYPE:
        raise ValueError("The canonical repair mode must be active_pair")
    if args.protected_logit_margin < 0:
        raise ValueError("protected logit margin must be non-negative")
    if args.protected_logit_drift_weight < 0:
        raise ValueError("protected logit-drift weight must be non-negative")
    if args.multihop_batch_size <= 0:
        raise ValueError("multi-hop batch size must be positive")
    if args.target_eff_max != 0.0:
        raise ValueError("The fixed MQuAKE candidate gate requires target Eff 0.0")
    if args.utility_drop_tolerance != 0.10:
        raise ValueError("The fixed MQuAKE retain tolerance is 0.10 percentage points")
    if args.max_ppl_ratio != 1.02:
        raise ValueError("The fixed Base-relative PPL multiplier is 1.02")
    if not args.strict_utility_gates:
        raise ValueError("The publication path requires strict utility gates")


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_repair_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
) -> List[mquake.PredictionCase]:
    """Build only cloze teacher-forced states; never read evaluation prompts."""

    cases: List[mquake.PredictionCase] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        subject = str(rewrite["subject"])
        sensitive = str(rewrite["target_true"]["str"])
        target_ids = mquake.original_answer_token_ids(
            tok, sensitive, llama_like=llama_like
        )
        prompt = str(rewrite["prompt"]).format(subject)
        for token_index, token_id in enumerate(target_ids):
            decoded_prefix = tok.decode(target_ids[:token_index])
            evaluated_prompt = (
                prompt + " " + decoded_prefix
                if llama_like and token_index > 0
                else prompt + decoded_prefix
            )
            cases.append(
                mquake.PredictionCase(
                    case_id=int(record["case_id"]),
                    prompt_type="rewrite",
                    prompt_index=0,
                    token_index=token_index,
                    prompt=evaluated_prompt,
                    target_text=tok.decode([token_id]),
                )
            )
    return cases


def selection_visible_records(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Copy only cloze/source fields needed by pre-selection official Eff.

    The shared evaluator currently constructs its prompt lookup eagerly, even
    when ``include_atomic_gen=False``.  A literal sentinel therefore occupies
    that unused key so the real atomic question, record-level questions, and
    answer aliases never cross the checkpoint-selection boundary.
    """

    visible: List[Dict[str, Any]] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        visible.append(
            {
                "case_id": int(record["case_id"]),
                "mquake_case_id": int(record["mquake_case_id"]),
                "source_index": int(record["source_index"]),
                "rewrite_index": int(record["rewrite_index"]),
                "requested_rewrite": {
                    "prompt": str(rewrite["prompt"]),
                    "subject": str(rewrite["subject"]),
                    "target_true": {"str": str(rewrite["target_true"]["str"])},
                },
                "atomic_gen_prompt": "<withheld-until-post-selection>",
            }
        )
    return visible


def instance_balanced_training_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    steps: int,
    seed: int,
) -> Tuple[List[gagd.Example], Dict[str, Any]]:
    """Pre-sample one instance and then one atom per Setting 5e step."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    by_instance: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[int(record["source_index"])].append(record)
    if not by_instance:
        raise ValueError("No MQuAKE forget instances were supplied")
    instance_ids = sorted(by_instance)
    for values in by_instance.values():
        values.sort(key=lambda row: int(row["rewrite_index"]))

    rng = random.Random(seed)
    sampled_records: List[Mapping[str, Any]] = []
    instance_counts: Counter[int] = Counter()
    atom_counts: Counter[int] = Counter()
    for _ in range(steps):
        instance_id = instance_ids[rng.randrange(len(instance_ids))]
        atoms = by_instance[instance_id]
        record = atoms[rng.randrange(len(atoms))]
        sampled_records.append(record)
        instance_counts[instance_id] += 1
        atom_counts[int(record["case_id"])] += 1
    examples = baseline.canonical_examples(sampled_records, tok)
    return examples, {
        "strategy": "instance_balanced",
        "seed": int(seed),
        "steps": int(steps),
        "instance_draw_counts": {str(k): v for k, v in sorted(instance_counts.items())},
        "atomic_draw_counts": {str(k): v for k, v in sorted(atom_counts.items())},
    }


def setting5_forget_examples(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    strategy: str,
    steps: int,
    seed: int,
) -> Tuple[List[gagd.Example], Dict[str, Any]]:
    if strategy == "instance_balanced":
        return instance_balanced_training_examples(
            records, tok, steps=steps, seed=seed
        )
    if strategy == "atomic_epoch":
        examples = baseline.canonical_examples(records, tok)
        return examples, {
            "strategy": "atomic_epoch",
            "seed": int(seed),
            "steps": int(steps),
            "atomic_fact_count": len(examples),
        }
    raise ValueError(f"Unsupported forget sampling strategy: {strategy}")


def residual_active_caches(
    caches: Sequence[repair.TokenLogitCache],
) -> List[repair.TokenLogitCache]:
    """Exact official failures: sensitive target_true is still top-1."""

    return [cache for cache in caches if cache.predicted_token_id == cache.target_token_id]


def true_runner_up_token_id(logits: torch.Tensor, sensitive_token_id: int) -> int:
    """Return the current top logit after excluding the sensitive row only."""

    if logits.ndim != 1:
        raise ValueError("runner-up logits must be a one-dimensional vocabulary row")
    if not 0 <= int(sensitive_token_id) < logits.shape[0]:
        raise ValueError("sensitive token ID is outside the vocabulary")
    competitor_logits = logits.clone()
    competitor_logits[int(sensitive_token_id)] = -torch.inf
    return int(competitor_logits.argmax().item())


@torch.no_grad()
def cache_active_pairs(
    model: nn.Module,
    tok: Any,
    context_cases: Sequence[mquake.PredictionCase],
    active_caches: Sequence[repair.TokenLogitCache],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> List[ActivePairCache]:
    """Cache sensitive/true-runner-up pairs under official token indexing."""

    active_by_identity = {cache.case.identity: cache for cache in active_caches}
    if len(active_by_identity) != len(active_caches):
        raise ValueError("Residual active cache identities are not unique")
    pairs_by_identity: Dict[Tuple[int, str, int, int], ActivePairCache] = {}
    for batch in _chunks(list(context_cases), batch_size):
        encoded = tok(
            [case.prompt for case in batch], padding=True, return_tensors="pt"
        ).to(device)
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][batch_indices, last_non_masked, :].float()
        logits = output.logits[batch_indices, last_non_masked, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        for index, case in enumerate(batch):
            cached = active_by_identity.get(case.identity)
            if cached is None:
                continue
            sensitive_id = int(target_ids[index].item())
            if sensitive_id != int(cached.target_token_id):
                raise RuntimeError("Active-pair target differs from official cache")
            if int(predicted_ids[index].item()) != sensitive_id:
                raise RuntimeError("Residual active token was not reproducibly top-1")
            competitor_id = true_runner_up_token_id(logits[index], sensitive_id)
            pairs_by_identity[case.identity] = ActivePairCache(
                case=case,
                hidden=hidden[index].detach(),
                sensitive_token_id=sensitive_id,
                competitor_token_id=competitor_id,
                sensitive_base_logit=logits[index, sensitive_id].detach(),
                competitor_base_logit=logits[index, competitor_id].detach(),
            )
    missing = set(active_by_identity) - set(pairs_by_identity)
    if missing:
        raise RuntimeError(f"Failed to cache {len(missing)} active-pair states")
    return [pairs_by_identity[cache.case.identity] for cache in active_caches]


def active_pair_row_ids(
    pairs: Sequence[ActivePairCache],
) -> List[int]:
    """Union of both sides of every pair; no token has a special role."""

    return sorted(
        {
            token_id
            for pair in pairs
            for token_id in (pair.sensitive_token_id, pair.competitor_token_id)
        }
    )


def active_pair_row_counts(
    pairs: Sequence[ActivePairCache],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    sensitive = Counter(int(pair.sensitive_token_id) for pair in pairs)
    competitors = Counter(int(pair.competitor_token_id) for pair in pairs)
    return dict(sorted(sensitive.items())), dict(sorted(competitors.items()))


def active_pair_report(pair: ActivePairCache, tok: Any) -> Dict[str, Any]:
    initial_margin = pair.competitor_base_logit - pair.sensitive_base_logit
    return {
        **asdict(pair.case),
        "active_source": ACTIVE_SOURCE,
        "sensitive_token_id": int(pair.sensitive_token_id),
        "sensitive_token": tok.decode([int(pair.sensitive_token_id)]),
        "competitor_token_id": int(pair.competitor_token_id),
        "competitor_token": tok.decode([int(pair.competitor_token_id)]),
        "sensitive_base_logit": float(pair.sensitive_base_logit.cpu()),
        "competitor_base_logit": float(pair.competitor_base_logit.cpu()),
        "initial_pair_margin": float(initial_margin.cpu()),
        "pair_identity": list(pair.identity),
    }


def active_pair_identity_sha256(pairs: Sequence[ActivePairCache]) -> str:
    identities = [list(pair.identity) for pair in pairs]
    encoded = json.dumps(identities, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_retain_instances(
    records: Sequence[Mapping[str, Any]],
    count: int,
    seed: int,
) -> List[Mapping[str, Any]]:
    """Sample instance identities, then retain every atom in each instance."""

    if count < 0:
        raise ValueError("retain calibration count must be non-negative")
    by_instance: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_instance[int(record["source_index"])].append(record)
    instance_ids = sorted(by_instance)
    if count >= len(instance_ids):
        selected_ids = instance_ids
    else:
        selected_ids = sorted(random.Random(seed).sample(instance_ids, count))
    return [
        record
        for instance_id in selected_ids
        for record in sorted(
            by_instance[instance_id], key=lambda row: int(row["rewrite_index"])
        )
    ]


def freeze_model_for_multirow_repair(model: nn.Module) -> nn.Module:
    """Untie the output head without changing logits, then freeze the model."""

    return active.freeze_model_for_output_repair(model)


def prepare_active_pair_tensors(
    pairs: Sequence[ActivePairCache],
    row_ids: Sequence[int],
    *,
    device: torch.device,
    direction_basis: Optional[torch.Tensor] = None,
) -> ActivePairTensors:
    """Resolve row IDs and move active-pair data to the device exactly once."""

    hidden_size = int(pairs[0].hidden.numel()) if pairs else 0
    if not pairs:
        hidden = torch.empty((0, hidden_size), dtype=torch.float32, device=device)
        empty_float = torch.empty((0,), dtype=torch.float32, device=device)
        empty_long = torch.empty((0,), dtype=torch.long, device=device)
        hidden_rank = (
            hidden @ direction_basis.to(device=device, dtype=torch.float32).T
            if direction_basis is not None
            else None
        )
        return ActivePairTensors(
            hidden=hidden,
            base_margin=empty_float,
            sensitive_row_index=empty_long,
            competitor_row_index=empty_long,
            hidden_rank=hidden_rank,
        )
    row_index = {int(token_id): index for index, token_id in enumerate(row_ids)}
    for pair in pairs:
        if (
            int(pair.sensitive_token_id) not in row_index
            or int(pair.competitor_token_id) not in row_index
        ):
            raise ValueError("Every active-pair row must be jointly trainable")
    hidden = torch.stack([pair.hidden for pair in pairs]).to(
        device=device, dtype=torch.float32
    )
    base_margin = torch.stack(
        [pair.competitor_base_logit - pair.sensitive_base_logit for pair in pairs]
    ).to(device=device, dtype=torch.float32)
    sensitive_row_index = torch.tensor(
        [row_index[int(pair.sensitive_token_id)] for pair in pairs],
        dtype=torch.long,
        device=device,
    )
    competitor_row_index = torch.tensor(
        [row_index[int(pair.competitor_token_id)] for pair in pairs],
        dtype=torch.long,
        device=device,
    )
    hidden_rank = (
        hidden @ direction_basis.to(device=device, dtype=torch.float32).T
        if direction_basis is not None
        else None
    )
    return ActivePairTensors(
        hidden=hidden,
        base_margin=base_margin,
        sensitive_row_index=sensitive_row_index,
        competitor_row_index=competitor_row_index,
        hidden_rank=hidden_rank,
    )


def active_pair_margins_from_delta(
    tensors: ActivePairTensors, delta_rows: torch.Tensor
) -> torch.Tensor:
    """Vectorized full-dimensional active-pair margins."""

    if not tensors.base_margin.numel():
        return delta_rows.new_empty((0,))
    sensitive_delta = delta_rows.index_select(0, tensors.sensitive_row_index)
    competitor_delta = delta_rows.index_select(0, tensors.competitor_row_index)
    return (
        tensors.base_margin
        + (tensors.hidden * competitor_delta).sum(dim=1)
        - (tensors.hidden * sensitive_delta).sum(dim=1)
    )


def active_pair_margins_from_coefficients(
    tensors: ActivePairTensors, coefficients: torch.Tensor
) -> torch.Tensor:
    """Vectorized rank-r margins without materializing full hidden-size rows."""

    if tensors.hidden_rank is None:
        raise ValueError("Rank-space active hidden states were not prepared")
    if not tensors.base_margin.numel():
        return coefficients.new_empty((0,))
    sensitive_coeff = coefficients.index_select(0, tensors.sensitive_row_index)
    competitor_coeff = coefficients.index_select(0, tensors.competitor_row_index)
    return (
        tensors.base_margin
        + (tensors.hidden_rank * competitor_coeff).sum(dim=1)
        - (tensors.hidden_rank * sensitive_coeff).sum(dim=1)
    )


def active_pair_margins(
    pairs: Sequence[ActivePairCache],
    delta_rows: torch.Tensor,
    row_ids: Sequence[int],
) -> torch.Tensor:
    """Compatibility wrapper; the optimizer uses prepacked tensors directly."""

    tensors = prepare_active_pair_tensors(
        pairs, row_ids, device=delta_rows.device
    )
    return active_pair_margins_from_delta(tensors, delta_rows)


def active_pair_squared_hinge_loss(
    margins: torch.Tensor, active_logit_margin: float
) -> torch.Tensor:
    if not margins.numel():
        return margins.new_zeros(())
    return torch.relu(float(active_logit_margin) - margins).square().mean()


@torch.no_grad()
def cache_protected_pair_states(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    row_ids: Sequence[int],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> List[ProtectedPairState]:
    """Cache only Setting5-correct cloze retain states and modified-row logits."""

    row_index = {int(token_id): index for index, token_id in enumerate(row_ids)}
    selected = torch.tensor(row_ids, dtype=torch.long, device=device)
    states: List[ProtectedPairState] = []
    for batch in _chunks(list(cases), batch_size):
        encoded = tok(
            [case.prompt for case in batch], padding=True, return_tensors="pt"
        ).to(device)
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][batch_indices, last_non_masked, :].float()
        logits = output.logits[batch_indices, last_non_masked, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        selected_logits = logits.index_select(1, selected)
        for index, case in enumerate(batch):
            target_id = int(target_ids[index].item())
            if int(predicted_ids[index].item()) != target_id:
                continue
            states.append(
                ProtectedPairState(
                    case=case,
                    hidden=hidden[index].detach(),
                    correct_token_id=target_id,
                    correct_base_logit=logits[index, target_id].detach(),
                    modified_row_base_logits=selected_logits[index].detach(),
                    correct_modified_row_index=row_index.get(target_id, -1),
                )
            )
    return states


def prepare_protected_pair_tensors(
    states: Sequence[ProtectedPairState],
    row_ids: Sequence[int],
    *,
    hidden_size: int,
    device: torch.device,
    direction_basis: Optional[torch.Tensor] = None,
) -> ProtectedPairTensors:
    """Pack every static protected-state tensor once before optimization."""

    if not states:
        hidden = torch.empty((0, hidden_size), dtype=torch.float32, device=device)
        base_modified = torch.empty(
            (0, len(row_ids)), dtype=torch.float32, device=device
        )
        correct_base = torch.empty((0,), dtype=torch.float32, device=device)
        correct_index = torch.empty((0,), dtype=torch.long, device=device)
        competitor_mask = torch.empty(
            (0, len(row_ids)), dtype=torch.bool, device=device
        )
        hidden_rank = (
            hidden @ direction_basis.to(device=device, dtype=torch.float32).T
            if direction_basis is not None
            else None
        )
        return ProtectedPairTensors(
            hidden=hidden,
            base_modified_logits=base_modified,
            correct_base=correct_base,
            correct_modified_row_index=correct_index,
            competitor_mask=competitor_mask,
            hidden_rank=hidden_rank,
        )
    hidden = torch.stack([state.hidden for state in states]).to(
        device=device, dtype=torch.float32
    )
    base_modified = torch.stack(
        [state.modified_row_base_logits for state in states]
    ).to(device=device, dtype=torch.float32)
    correct_base = torch.stack([state.correct_base_logit for state in states]).to(
        device=device, dtype=torch.float32
    )
    correct_modified_row_index = torch.tensor(
        [state.correct_modified_row_index for state in states],
        dtype=torch.long,
        device=device,
    )
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=device)
    correct_ids = torch.tensor(
        [state.correct_token_id for state in states],
        dtype=torch.long,
        device=device,
    )
    competitor_mask = row_tensor[None, :] != correct_ids[:, None]
    hidden_rank = (
        hidden @ direction_basis.to(device=device, dtype=torch.float32).T
        if direction_basis is not None
        else None
    )
    return ProtectedPairTensors(
        hidden=hidden,
        base_modified_logits=base_modified,
        correct_base=correct_base,
        correct_modified_row_index=correct_modified_row_index,
        competitor_mask=competitor_mask,
        hidden_rank=hidden_rank,
    )


def protected_pair_margins_from_delta_logits(
    tensors: ProtectedPairTensors, delta_logits: torch.Tensor
) -> torch.Tensor:
    """Vectorize correct-row adjustments and all protected competitors."""

    if not tensors.correct_base.numel() or not delta_logits.shape[1]:
        return delta_logits.new_empty((0,))
    index = tensors.correct_modified_row_index
    valid = index >= 0
    safe_index = index.clamp_min(0)
    target_delta = delta_logits.gather(1, safe_index[:, None]).squeeze(1)
    target_delta = torch.where(valid, target_delta, torch.zeros_like(target_delta))
    correct_after = tensors.correct_base + target_delta
    margins = correct_after[:, None] - (
        tensors.base_modified_logits + delta_logits
    )
    return margins[tensors.competitor_mask]


def protected_delta_logits_from_delta(
    tensors: ProtectedPairTensors, delta_rows: torch.Tensor
) -> torch.Tensor:
    return tensors.hidden @ delta_rows.transpose(0, 1)


def protected_delta_logits_from_coefficients(
    tensors: ProtectedPairTensors, coefficients: torch.Tensor
) -> torch.Tensor:
    if tensors.hidden_rank is None:
        raise ValueError("Rank-space protected hidden states were not prepared")
    return tensors.hidden_rank @ coefficients.transpose(0, 1)


def protected_pair_margins(
    states: Sequence[ProtectedPairState],
    delta_rows: torch.Tensor,
    row_ids: Sequence[int],
) -> torch.Tensor:
    """Compatibility wrapper; the optimizer reuses prepacked static tensors."""

    tensors = prepare_protected_pair_tensors(
        states,
        row_ids,
        hidden_size=delta_rows.shape[1],
        device=delta_rows.device,
    )
    delta_logits = protected_delta_logits_from_delta(tensors, delta_rows)
    return protected_pair_margins_from_delta_logits(tensors, delta_logits)


def protected_pair_squared_hinge_loss(
    margins: torch.Tensor, protected_logit_margin: float
) -> torch.Tensor:
    if not margins.numel():
        return margins.new_zeros(())
    return torch.relu(float(protected_logit_margin) - margins).square().mean()


def protected_logit_drift_loss(
    delta_rows: torch.Tensor,
    protected_caches: Sequence[ProtectedPairState],
) -> torch.Tensor:
    """Mean squared Setting-5e logit drift for every modified output row."""

    if not protected_caches:
        return delta_rows.new_zeros(())
    hidden = torch.stack([cache.hidden for cache in protected_caches]).to(
        device=delta_rows.device, dtype=delta_rows.dtype
    )
    drift = hidden @ delta_rows.transpose(0, 1)
    return drift.square().mean()


def protected_logit_drift_from_delta_logits(delta_logits: torch.Tensor) -> torch.Tensor:
    if not delta_logits.numel():
        return delta_logits.new_zeros(())
    return delta_logits.square().mean()


def basis_is_orthonormal(
    basis: torch.Tensor, *, atol: float = 1e-5, rtol: float = 1e-5
) -> bool:
    if basis.ndim != 2:
        return False
    gram = basis @ basis.transpose(0, 1)
    identity = torch.eye(
        basis.shape[0], dtype=basis.dtype, device=basis.device
    )
    return bool(torch.allclose(gram, identity, atol=atol, rtol=rtol))


def low_rank_delta_l2(
    coefficients: torch.Tensor,
    effective_direction_basis: torch.Tensor,
    *,
    orthonormal: bool,
) -> torch.Tensor:
    """Use coefficient-space L2 exactly when the effective basis is orthonormal."""

    if orthonormal:
        return coefficients.square().sum()
    return (coefficients @ effective_direction_basis).square().sum()


@torch.no_grad()
def _constrain_parameterized_delta_norm(
    module: active.SelectedRowDelta,
    delta_l2: torch.Tensor,
    max_delta_norm: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the existing global norm constraint without a host sync."""

    before = delta_l2.clamp_min(0.0).sqrt()
    if max_delta_norm is None:
        return before, before, torch.zeros((), dtype=torch.bool, device=before.device)
    maximum = before.new_tensor(float(max_delta_norm))
    scale = torch.where(
        before > maximum,
        maximum / before.clamp_min(torch.finfo(before.dtype).tiny),
        torch.ones_like(before),
    )
    parameter = module.coefficients if module.coefficients is not None else module.raw_delta
    if parameter is None:
        raise RuntimeError("SelectedRowDelta has no trainable parameter")
    parameter.mul_(scale)
    return before, before * scale, scale < 1.0


def _assert_finite_without_cuda_host_sync(value: torch.Tensor, step: int) -> None:
    condition = torch.isfinite(value)
    if value.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, f"Non-finite active-pair loss at step {step}")
        return
    if not bool(condition.item()):
        raise FloatingPointError(f"Non-finite active-pair loss at step {step}")


def _materialize_repair_logs(
    rows: Sequence[torch.Tensor],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    values = torch.stack(list(rows)).detach().cpu().tolist()
    result: List[Dict[str, Any]] = []
    for step, row in enumerate(values, 1):
        result.append(
            {
                "step": step,
                "total_loss": float(row[0]),
                "active_pair_squared_hinge": float(row[1]),
                "protected_pair_squared_hinge": float(row[2]),
                "protected_logit_drift_loss": float(row[3]),
                "delta_l2": float(row[4]),
                "active_pair_violations": int(row[5]),
                "protected_pair_violations": int(row[6]),
                "effective_delta_norm_before_projection": float(row[7]),
                "effective_delta_norm": float(row[8]),
                "delta_norm_projected": bool(row[9]),
            }
        )
    return result


def optimize_active_pair_delta(
    active_pairs: Sequence[ActivePairCache],
    protected_caches: Sequence[ProtectedPairState],
    *,
    row_ids: Sequence[int],
    hidden_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    if not active_pairs:
        zeros = torch.zeros(
            (len(row_ids), hidden_size), dtype=torch.float32, device=device
        )
        return zeros, [], {
            "steps_completed": 0,
            "stopped_early": True,
            "all_satisfied": True,
            "reason": "no_residual_active_pairs",
            "repair_type": REPAIR_TYPE,
        }

    active_tensors = prepare_active_pair_tensors(
        active_pairs, row_ids, device=device
    )
    protected_tensors = prepare_protected_pair_tensors(
        protected_caches,
        row_ids,
        hidden_size=hidden_size,
        device=device,
    )
    protected_hidden = protected_tensors.hidden
    retained_basis = None
    if args.project_away_protected_hidden and protected_hidden.numel():
        retained_basis = active.orthonormal_row_basis(protected_hidden)

    projected_active = active.project_rows_away(
        active_tensors.hidden, retained_basis
    )
    direction_basis = None
    if args.repair_rank > 0:
        direction_basis = active.orthonormal_row_basis(
            projected_active, max_rank=args.repair_rank
        )
        if direction_basis.numel() == 0:
            raise RuntimeError("Protected projection removed every active direction")

    module = active.SelectedRowDelta(
        n_rows=len(row_ids),
        hidden_size=hidden_size,
        direction_basis=direction_basis,
        retained_basis=retained_basis,
        device=device,
    )

    effective_direction_basis = None
    rank_basis_orthonormal = False
    if module.coefficients is not None:
        if module.direction_basis is None:
            raise RuntimeError("Low-rank repair has no shared direction basis")
        effective_direction_basis = active.project_rows_away(
            module.direction_basis, module.retained_basis
        )
        rank_basis_orthonormal = basis_is_orthonormal(effective_direction_basis)
        active_tensors = ActivePairTensors(
            hidden=active_tensors.hidden,
            base_margin=active_tensors.base_margin,
            sensitive_row_index=active_tensors.sensitive_row_index,
            competitor_row_index=active_tensors.competitor_row_index,
            hidden_rank=(
                active_tensors.hidden @ effective_direction_basis.transpose(0, 1)
            ),
        )
        protected_tensors = ProtectedPairTensors(
            hidden=protected_tensors.hidden,
            base_modified_logits=protected_tensors.base_modified_logits,
            correct_base=protected_tensors.correct_base,
            correct_modified_row_index=(
                protected_tensors.correct_modified_row_index
            ),
            competitor_mask=protected_tensors.competitor_mask,
            hidden_rank=(
                protected_tensors.hidden
                @ effective_direction_basis.transpose(0, 1)
            ),
        )

    def objective_tensors() -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        if module.coefficients is not None:
            coefficients = module.coefficients
            active_margins = active_pair_margins_from_coefficients(
                active_tensors, coefficients
            )
            protected_delta_logits = protected_delta_logits_from_coefficients(
                protected_tensors, coefficients
            )
            if effective_direction_basis is None:
                raise RuntimeError("Missing effective rank-space basis")
            delta_l2 = low_rank_delta_l2(
                coefficients,
                effective_direction_basis,
                orthonormal=rank_basis_orthonormal,
            )
        else:
            delta_rows = module.effective_delta()
            active_margins = active_pair_margins_from_delta(
                active_tensors, delta_rows
            )
            protected_delta_logits = protected_delta_logits_from_delta(
                protected_tensors, delta_rows
            )
            delta_l2 = delta_rows.square().sum()
        protected_margins = protected_pair_margins_from_delta_logits(
            protected_tensors, protected_delta_logits
        )
        return (
            active_margins,
            protected_margins,
            protected_delta_logits,
            delta_l2,
        )

    optimizer = active.make_repair_optimizer(
        module, args.repair_optimizer, args.repair_lr
    )
    with torch.no_grad():
        initial_active, initial_protected, _, _ = objective_tensors()
    stopped_early = False
    tensor_logs: List[torch.Tensor] = []
    for step in range(1, args.repair_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        (
            active_margins,
            protected_margins,
            protected_delta_logits,
            delta_l2,
        ) = objective_tensors()
        active_hinge = active_pair_squared_hinge_loss(
            active_margins, args.active_logit_margin
        )
        protected_hinge = protected_pair_squared_hinge_loss(
            protected_margins, args.protected_logit_margin
        )
        drift = protected_logit_drift_from_delta_logits(
            protected_delta_logits
        )
        total = (
            active_hinge
            + protected_hinge
            + args.protected_logit_drift_weight * drift
            + args.repair_l2_lambda * delta_l2
        )
        _assert_finite_without_cuda_host_sync(total, step)
        total.backward()
        optimizer.step()
        with torch.no_grad():
            (
                active_after,
                protected_after,
                _,
                updated_l2_before_projection,
            ) = objective_tensors()
            before_norm, after_norm, projected = _constrain_parameterized_delta_norm(
                module, updated_l2_before_projection, args.max_delta_norm
            )
            if args.max_delta_norm is not None:
                active_after, protected_after, _, _ = objective_tensors()
            active_violation_tensor = (
                active_after < args.active_logit_margin
            ).sum()
            protected_violation_tensor = (
                protected_after < args.protected_logit_margin
            ).sum()
            # The single host synchronization in the hot loop preserves exact
            # early-stopping semantics and supplies the per-step counts.
            violation_values = torch.stack(
                [active_violation_tensor, protected_violation_tensor]
            ).detach().cpu().tolist()
            active_violations = int(violation_values[0])
            protected_violations = int(violation_values[1])
            tensor_logs.append(
                torch.stack(
                    [
                        total.detach(),
                        active_hinge.detach(),
                        protected_hinge.detach(),
                        drift.detach(),
                        delta_l2.detach(),
                        active_violation_tensor.float(),
                        protected_violation_tensor.float(),
                        before_norm,
                        after_norm,
                        projected.float(),
                    ]
                )
            )
        if (
            args.stop_when_all_satisfied
            and active_violations == 0
            and protected_violations == 0
        ):
            stopped_early = True
            break
    logs = _materialize_repair_logs(tensor_logs)
    delta = module.effective_delta().detach()
    with torch.no_grad():
        final_active, final_protected, _, _ = objective_tensors()
    norm_projection_steps = sum(
        int(row["delta_norm_projected"]) for row in logs
    )
    summary = {
        "repair_type": REPAIR_TYPE,
        "steps_completed": len(logs),
        "stopped_early": stopped_early,
        "all_satisfied": bool(
            (final_active >= args.active_logit_margin).all().item()
            and (
                not final_protected.numel()
                or (final_protected >= args.protected_logit_margin).all().item()
            )
        ),
        "delta_norm_projection_steps": norm_projection_steps,
        "active_pair_violations_before": int(
            (initial_active < args.active_logit_margin).sum().item()
        ),
        "active_pair_violations_after_optimization": int(
            (final_active < args.active_logit_margin).sum().item()
        ),
        "protected_pair_violations_before": int(
            (initial_protected < args.protected_logit_margin).sum().item()
        ),
        "protected_pair_violations_after_optimization": int(
            (final_protected < args.protected_logit_margin).sum().item()
        ),
    }
    summary.update(
        {
            "modified_row_count": len(row_ids),
            "row_ids": [int(value) for value in row_ids],
            "active_pair_count": len(active_pairs),
            "protected_state_count": len(protected_caches),
            "protected_pair_count": int(initial_protected.numel()),
            "protected_hidden_rank": (
                0 if retained_basis is None else int(retained_basis.shape[0])
            ),
            "protected_logit_drift_weight": float(
                args.protected_logit_drift_weight
            ),
            "optimization_tensorization": {
                "active_pairs_prepacked_on_device": True,
                "protected_states_prepacked_on_device": True,
                "python_pair_loops_per_optimizer_step": 0,
                "python_protected_state_loops_per_optimizer_step": 0,
                "repair_rank": int(args.repair_rank),
                "rank_space_fast_path": module.coefficients is not None,
                "rank_basis_orthonormal": rank_basis_orthonormal,
                "full_hidden_size_delta_materialized_per_step": bool(
                    module.coefficients is None
                ),
                "cuda_host_synchronizations_per_step": 1,
            },
        }
    )
    return delta, logs, summary


@torch.no_grad()
def materialize_multirow_scale(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
) -> None:
    if len(row_ids) != original_rows.shape[0] or len(row_ids) != delta_rows.shape[0]:
        raise ValueError("row IDs, originals, and deltas must have matching rows")
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    updated = original_rows + float(scale) * delta_rows.to(
        device=original_rows.device, dtype=original_rows.dtype
    )
    output_weight.index_copy_(0, ids, updated)
    if float(scale) == 0.0 and not torch.equal(
        output_weight.index_select(0, ids), original_rows
    ):
        raise RuntimeError("Scale 0 did not exactly restore Setting 5e rows")


def scale_report_is_locally_safe(report: Mapping[str, Any]) -> bool:
    return bool(
        int(report["active_correct_tokens"]) == 0
        and int(report["active_pair_margin_violations"]) == 0
        and int(report["protected_incremental_regressions_vs_zero"]) == 0
        and int(report["protected_pair_margin_violations"]) == 0
    )


@torch.no_grad()
def _evaluate_active_pairs_exact(
    model: nn.Module,
    tok: Any,
    context_cases: Sequence[mquake.PredictionCase],
    pairs: Sequence[ActivePairCache],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """Evaluate fixed active pairs with the official teacher-forced indexing."""

    by_identity = {pair.case.identity: pair for pair in pairs}
    rows: Dict[Tuple[int, str, int, int], Dict[str, Any]] = {}
    for batch in _chunks(list(context_cases), batch_size):
        encoded = tok(
            [case.prompt for case in batch], padding=True, return_tensors="pt"
        ).to(device)
        logits = model(**encoded, use_cache=False).logits
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        logits = logits[batch_indices, last_non_masked, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        for index, case in enumerate(batch):
            pair = by_identity.get(case.identity)
            if pair is None:
                continue
            sensitive_id = int(target_ids[index].item())
            if sensitive_id != int(pair.sensitive_token_id):
                raise RuntimeError("Exact active-pair target identity changed")
            competitor_id = int(pair.competitor_token_id)
            sensitive_logit = float(logits[index, sensitive_id].cpu())
            competitor_logit = float(logits[index, competitor_id].cpu())
            rows[case.identity] = {
                "pair_identity": list(pair.identity),
                "predicted_token_id": int(predicted_ids[index].item()),
                "sensitive_token_id": sensitive_id,
                "competitor_token_id": competitor_id,
                "sensitive_logit": sensitive_logit,
                "competitor_logit": competitor_logit,
                "pair_margin": competitor_logit - sensitive_logit,
                "sensitive_is_top1": int(predicted_ids[index].item())
                == sensitive_id,
            }
    missing = set(by_identity) - set(rows)
    if missing:
        raise RuntimeError(f"Exact materialization omitted {len(missing)} active pairs")
    return [rows[pair.case.identity] for pair in pairs]


@torch.no_grad()
def _evaluate_protected_pairs_exact(
    model: nn.Module,
    tok: Any,
    context_cases: Sequence[mquake.PredictionCase],
    states: Sequence[ProtectedPairState],
    *,
    row_ids: Sequence[int],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    protected_logit_margin: float,
) -> List[Dict[str, Any]]:
    """Audit Setting5-correct retain targets against every modified row."""

    by_identity = {state.case.identity: state for state in states}
    selected = torch.tensor(row_ids, dtype=torch.long, device=device)
    rows: Dict[Tuple[int, str, int, int], Dict[str, Any]] = {}
    for batch in _chunks(list(context_cases), batch_size):
        encoded = tok(
            [case.prompt for case in batch], padding=True, return_tensors="pt"
        ).to(device)
        logits = model(**encoded, use_cache=False).logits
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        logits = logits[batch_indices, last_non_masked, :].float()
        target_ids = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        selected_logits = logits.index_select(1, selected)
        for index, case in enumerate(batch):
            state = by_identity.get(case.identity)
            if state is None:
                continue
            target_id = int(target_ids[index].item())
            if target_id != int(state.correct_token_id):
                raise RuntimeError("Exact protected target identity changed")
            target_logit = float(logits[index, target_id].cpu())
            pair_margins = [
                target_logit - float(selected_logits[index, row_index].cpu())
                for row_index, token_id in enumerate(row_ids)
                if int(token_id) != target_id
            ]
            rows[case.identity] = {
                "case_identity": list(case.identity),
                "correct_token_id": target_id,
                "predicted_token_id": int(predicted_ids[index].item()),
                "correct": int(predicted_ids[index].item()) == target_id,
                "modified_competitor_count": len(pair_margins),
                "pair_margin_min": min(pair_margins) if pair_margins else None,
                "pair_margin_violations": sum(
                    value < protected_logit_margin for value in pair_margins
                ),
            }
    missing = set(by_identity) - set(rows)
    if missing:
        raise RuntimeError(
            f"Exact materialization omitted {len(missing)} protected states"
        )
    return [rows[state.case.identity] for state in states]


@torch.no_grad()
def exact_bf16_active_pair_scale_sweep(
    *,
    model: nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    active_pairs: Sequence[ActivePairCache],
    protected_states: Sequence[ProtectedPairState],
    active_context_cases: Sequence[mquake.PredictionCase],
    protected_context_cases: Sequence[mquake.PredictionCase],
    scales: Sequence[float],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    minimum_active_margin: float,
    protected_logit_margin: float,
) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
    """Apply each scale jointly from immutable Setting-5e rows."""

    normalized = sorted({float(value) for value in scales}, reverse=True)
    if 0.0 not in normalized:
        raise ValueError("The exact scale sweep must include 0.0")

    materialize_multirow_scale(
        output_weight, row_ids, original_rows, delta_rows, 0.0
    )
    zero_active = _evaluate_active_pairs_exact(
        model,
        tok,
        active_context_cases,
        active_pairs,
        device=device,
        llama_like=llama_like,
        batch_size=batch_size,
    )
    zero_protected = _evaluate_protected_pairs_exact(
        model,
        tok,
        protected_context_cases,
        protected_states,
        row_ids=row_ids,
        device=device,
        llama_like=llama_like,
        batch_size=batch_size,
        protected_logit_margin=protected_logit_margin,
    )
    zero_protected_correct = [bool(row["correct"]) for row in zero_protected]
    reports: List[Dict[str, Any]] = []
    for scale in normalized:
        materialize_multirow_scale(
            output_weight, row_ids, original_rows, delta_rows, scale
        )
        if scale == 0.0:
            active_rows, protected_rows = zero_active, zero_protected
        else:
            active_rows = _evaluate_active_pairs_exact(
                model,
                tok,
                active_context_cases,
                active_pairs,
                device=device,
                llama_like=llama_like,
                batch_size=batch_size,
            )
            protected_rows = _evaluate_protected_pairs_exact(
                model,
                tok,
                protected_context_cases,
                protected_states,
                row_ids=row_ids,
                device=device,
                llama_like=llama_like,
                batch_size=batch_size,
                protected_logit_margin=protected_logit_margin,
            )
        margins = [float(row["pair_margin"]) for row in active_rows]
        protected_correct = [bool(row["correct"]) for row in protected_rows]
        materialized = output_weight.index_select(
            0, torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
        ).float() - original_rows.float()
        reports.append(
            {
                "repair_type": REPAIR_TYPE,
                "scale": float(scale),
                "active_pair_count": len(active_rows),
                "active_correct_tokens": int(
                    sum(bool(row["sensitive_is_top1"]) for row in active_rows)
                ),
                "active_pair_margin_violations": int(
                    sum(value < minimum_active_margin for value in margins)
                ),
                "active_pair_margin_min": (
                    min(margins) if margins else None
                ),
                "active_pair_margin_mean": (
                    sum(margins) / len(margins) if margins else None
                ),
                "active_pair_margin_max": (
                    max(margins) if margins else None
                ),
                "protected_state_count": len(protected_rows),
                "protected_pair_count": int(
                    sum(row["modified_competitor_count"] for row in protected_rows)
                ),
                "protected_pair_margin_violations": int(
                    sum(row["pair_margin_violations"] for row in protected_rows)
                ),
                "protected_incremental_regressions_vs_zero": int(
                    sum(
                        before and not after
                        for before, after in zip(
                            zero_protected_correct, protected_correct
                        )
                    )
                ),
                "joint_materialized_delta_norm": float(materialized.norm().cpu()),
                "all_rows_applied_jointly": True,
                "active_pair_audit": active_rows,
                "protected_pair_audit": protected_rows,
            }
        )

    eligible = [
        row for row in reports if scale_report_is_locally_safe(row)
    ]
    selected = min(
        eligible or [row for row in reports if row["scale"] == 0.0],
        key=lambda row: (
            float(row["joint_materialized_delta_norm"]),
            float(row["scale"]),
        ),
    )
    selected_scale = float(selected["scale"])
    materialize_multirow_scale(
        output_weight, row_ids, original_rows, delta_rows, selected_scale
    )
    zero_baseline = {
        "active_correct_tokens_at_zero": int(
            sum(bool(row["sensitive_is_top1"]) for row in zero_active)
        ),
        "active_pair_count": len(zero_active),
        "active_pair_margin_violations_at_zero": int(
            sum(float(row["pair_margin"]) < minimum_active_margin for row in zero_active)
        ),
        "protected_correct_tokens_at_zero": int(
            sum(bool(row["correct"]) for row in zero_protected)
        ),
        "protected_state_count": len(zero_protected),
        "protected_pair_count": int(
            sum(row["modified_competitor_count"] for row in zero_protected)
        ),
        "protected_pair_margin_violations_at_zero": int(
            sum(row["pair_margin_violations"] for row in zero_protected)
        ),
    }
    return selected_scale, reports, zero_baseline


def candidate_acceptance_report(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    scale_report: Mapping[str, Any],
    *,
    retain_tolerance: float = 0.10,
    max_ppl_ratio: float = 1.02,
) -> Dict[str, Any]:
    """Frozen selection rule; held-out AtomicGen/MH metrics are not inputs."""

    base_retain = base_result["retain"].get("Eff")
    candidate_retain = candidate_result["retain"].get("Eff")
    base_ppl = base_result.get("forget_PPL")
    candidate_ppl = candidate_result.get("forget_PPL")
    checks = {
        "forget_Eff_exactly_zero": float(candidate_result["forget"]["Eff"]) == 0.0,
        "retain_Eff_within_base_tolerance": (
            base_retain is not None
            and candidate_retain is not None
            and float(candidate_retain) >= float(base_retain) - retain_tolerance
        ),
        "PPL_within_base_ratio": (
            base_ppl is not None
            and candidate_ppl is not None
            and float(candidate_ppl) <= float(base_ppl) * max_ppl_ratio
        ),
        "no_incremental_protected_regression": int(
            scale_report["protected_incremental_regressions_vs_zero"]
        )
        == 0,
        "active_token_constraints_satisfied": scale_report_is_locally_safe(
            scale_report
        ),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "forget_Eff": 0.0,
            "retain_Eff_minimum": (
                None if base_retain is None else float(base_retain) - retain_tolerance
            ),
            "PPL_maximum": (
                None if base_ppl is None else float(base_ppl) * max_ppl_ratio
            ),
            "protected_incremental_regressions": 0,
        },
    }


def select_candidate(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    scale_report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Selection deliberately has no AtomicGen or multi-hop parameter."""

    return candidate_acceptance_report(base_result, candidate_result, scale_report)


def _row_reports(
    row_ids: Sequence[int],
    delta_rows: torch.Tensor,
    tok: Any,
    sensitive_counts: Mapping[int, int],
    competitor_counts: Mapping[int, int],
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for index, token_id in enumerate(row_ids):
        sensitive_count = int(sensitive_counts.get(int(token_id), 0))
        competitor_count = int(competitor_counts.get(int(token_id), 0))
        if sensitive_count and competitor_count:
            role = "sensitive_and_competitor"
        elif sensitive_count:
            role = "sensitive"
        elif competitor_count:
            role = "competitor"
        else:
            raise RuntimeError("Selected row has no active-pair role")
        reports.append({
            "token_id": int(token_id),
            "decoded_token": tok.decode([int(token_id)]),
            "role": role,
            "sensitive_constraint_count": sensitive_count,
            "competitor_constraint_count": competitor_count,
            "active_constraint_count": sensitive_count + competitor_count,
            "delta_norm": float(delta_rows[index].float().norm().cpu()),
        })
    return reports


def _evaluate_multihop_post_selection(
    *,
    model: nn.Module,
    tok: Any,
    mquake_path: Path,
    split_manifest: Path,
    prompt_dir: Path,
    args: argparse.Namespace,
    out_path: Path,
) -> Dict[str, Any]:
    instances, manifest = multihop.load_forget_instances(
        mquake_path, split_manifest
    )
    standard_path = multihop.download_text(
        prompt_dir / "multihop-prompts.txt", multihop.STANDARD_PROMPT_URL
    )
    cot_path = multihop.download_text(
        prompt_dir / "multihop-cot-prompts.txt", multihop.COT_PROMPT_URL
    )
    results: Dict[str, Any] = {}
    raw: Dict[str, Any] = {}
    original_padding_side = getattr(tok, "padding_side", "right")
    try:
        # Decoder-only batched generation must read continuations after a
        # common, left-padded prompt width.  Teacher-forced evaluation retains
        # its existing indexing and padding convention outside this scope.
        tok.padding_side = "left"
        for mode, path, max_new in (
            ("standard", standard_path, args.standard_max_new_tokens),
            ("cot", cot_path, args.cot_max_new_tokens),
        ):
            summary, rows = multihop.evaluate_mode(
                model=model,
                tok=tok,
                instances=instances,
                task_prompt=path.read_text(encoding="utf-8"),
                mode=mode,
                batch_size=args.multihop_batch_size,
                max_new_tokens=max_new,
            )
            results[mode] = summary
            raw[mode] = rows
    finally:
        tok.padding_side = original_padding_side
    payload = {
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": manifest.get("seed"),
        "checkpoint_selection": "completed before this evaluation",
        "results": results,
        "raw": raw,
    }
    gagd.write_json(out_path, payload)
    return payload


def _evaluate_held_out_after_selection(
    *,
    accepted: bool,
    args: argparse.Namespace,
    model: nn.Module,
    tok: Any,
    records: Any,
    mquake_path: Path,
    wikidata_dir: Path,
    split_manifest: Path,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Open held-out reporting only after the durable selection decision."""

    if args.fail_if_target_missed and not accepted:
        raise RuntimeError("No active-pair candidate passed every fixed gate")
    selected_extension = baseline.evaluate_extension(
        method=METHOD_LABEL + " post-selection AtomicGen",
        model=model,
        tok=tok,
        model_dir="in-memory:selected",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "selected_atomic_gen_eval.json",
        args=args,
        records=records,
    )
    multihop_result = _evaluate_multihop_post_selection(
        model=model,
        tok=tok,
        mquake_path=mquake_path,
        split_manifest=split_manifest,
        prompt_dir=gagd.resolve_output_path(args.multihop_prompt_dir),
        args=args,
        out_path=output_dir / "multihop_unlearning_eval.json",
    )
    return selected_extension, multihop_result


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setting5_dir = output_dir / "setting5e"
    repair_dir = output_dir / "multirow_active_repair"
    setting5_dir.mkdir(parents=True, exist_ok=True)
    repair_dir.mkdir(parents=True, exist_ok=True)
    mquake_path = Path(args.mquake_path)
    if not mquake_path.is_absolute():
        mquake_path = gagd.PROJECT_DIR / mquake_path
    mquake_path = mquake.download_mquake(mquake_path, url=args.mquake_url)
    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)

    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "repair_type": REPAIR_TYPE,
            "dataset_revision": mquake.MQUAKE_REV,
            "training_source": "sampled requested_rewrite cloze facts only",
            "repair_source": ACTIVE_SOURCE,
            "evaluation_only_until_selection": [
                "requested_rewrite.question",
                "record questions[0:3]",
                "answer/new_answer and aliases",
                "AtomicGen",
                "standard and CoT multi-hop leakage",
            ],
            "counterfactual_target_new_is_training_target": False,
            "active_pair_competitor": (
                "highest-logit Setting5e token at the same state, excluding "
                "only the sensitive token"
            ),
            "unknown_has_special_repair_role": False,
            "candidate_gates": {
                "forget_Eff": 0.0,
                "retain_Eff_minimum": "Base RetainEff - 0.10 percentage points",
                "PPL_maximum": "Base PPL * 1.02",
                "protected_incremental_regressions": 0,
            },
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading Base model and pinned MQuAKE split")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    forget_records, retain_records = mquake.load_official_eval_records(
        mquake_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        mquake_url=args.mquake_url,
    )
    records = (forget_records, retain_records)
    preselection_records = (
        selection_visible_records(forget_records),
        selection_visible_records(retain_records),
    )
    split_manifest = output_dir / "split_manifest.json"
    mquake.write_split_manifest(
        split_manifest,
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)
    base_result = baseline.evaluate_eff_only(
        method="Base",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "base_official_eval.json",
        args=args,
        records=preselection_records,
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training unchanged 600-step Setting 5e objective")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    if mquake.resolve_neutral_target_token_id(tok) != neutral_token_id:
        raise RuntimeError("Neutral token ID changed between model loads")
    forget_examples, sampling_report = setting5_forget_examples(
        forget_records,
        tok,
        strategy=args.forget_sampling,
        steps=args.steps,
        seed=args.seed,
    )
    retain_examples = baseline.canonical_examples(retain_records, tok)
    gagd.write_json(setting5_dir / "forget_sampling.json", sampling_report)
    args.post_training_excluded_token_ids = [neutral_token_id]
    requested_save = bool(args.save_model)
    args.save_model = False
    train_summary = gagd.train_mode(
        model,
        tok,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=SETTING5_MODE,
        args=args,
        mode_dir=setting5_dir,
    )
    args.save_model = requested_save
    setting5_result = baseline.evaluate_eff_only(
        method="Setting 5e",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=setting5_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    if args.save_setting5_checkpoint:
        baseline.save_checkpoint(model, tok, setting5_dir / "checkpoint")

    print("Caching cloze-only residual active and protected retain states")
    output_layer = freeze_model_for_multirow_repair(model)
    device = next(model.parameters()).device
    llama_like = mquake.is_llama_like(model, tok)
    forget_cases = build_repair_cases(forget_records, tok, llama_like=llama_like)
    forget_caches = repair.cache_prediction_cases(
        model,
        tok,
        forget_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache MQuAKE residual cloze tokens",
    )
    active_caches = residual_active_caches(forget_caches)
    active_pairs = cache_active_pairs(
        model,
        tok,
        forget_cases,
        active_caches,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    row_ids = active_pair_row_ids(active_pairs)
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=output_layer.weight.device)
    original_rows = output_layer.weight.index_select(0, row_tensor).detach().clone()

    calibration_records = sample_retain_instances(
        retain_records,
        args.retain_calibration_num,
        args.retain_calibration_seed,
    )
    retain_cases = build_repair_cases(
        calibration_records, tok, llama_like=llama_like
    )
    protected_states = cache_protected_pair_states(
        model,
        tok,
        retain_cases,
        row_ids=row_ids,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    repair.write_jsonl(
        repair_dir / "active_tokens_before.jsonl",
        [active_pair_report(pair, tok) for pair in active_pairs],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_before.jsonl",
        [
            {
                **asdict(state.case),
                "active_source": "sampled_requested_rewrite_cloze_retain_states",
                "correct_token_id": int(state.correct_token_id),
                "correct_token": tok.decode([int(state.correct_token_id)]),
                "correct_base_logit": float(state.correct_base_logit.cpu()),
                "modified_row_count": len(row_ids),
                "correct_row_is_modified": state.correct_modified_row_index >= 0,
            }
            for state in protected_states
        ],
    )

    delta_rows, repair_logs, optimization = optimize_active_pair_delta(
        active_pairs,
        protected_states,
        row_ids=row_ids,
        hidden_size=output_layer.weight.shape[1],
        device=device,
        args=args,
    )
    for row in repair_logs:
        row["active_source"] = ACTIVE_SOURCE
        row["repair_type"] = REPAIR_TYPE
    repair.write_jsonl(repair_dir / "active_pair_repair_log.jsonl", repair_logs)
    # Preserve the established artifact name for downstream readers while the
    # metadata explicitly identifies the new active-pair implementation.
    repair.write_jsonl(repair_dir / "multirow_repair_log.jsonl", repair_logs)
    sensitive_counts, competitor_counts = active_pair_row_counts(active_pairs)
    rows_report = _row_reports(
        row_ids, delta_rows, tok, sensitive_counts, competitor_counts
    )
    gagd.write_json(repair_dir / "active_rows.json", rows_report)
    gagd.write_json(repair_dir / "row_delta_norms.json", rows_report)

    selected_scale, scale_reports, zero_baseline = exact_bf16_active_pair_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        row_ids=row_ids,
        original_rows=original_rows,
        delta_rows=delta_rows,
        active_pairs=active_pairs,
        protected_states=protected_states,
        active_context_cases=forget_cases,
        protected_context_cases=retain_cases,
        scales=repair.parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        minimum_active_margin=args.selection_logit_margin,
        protected_logit_margin=args.protected_logit_margin,
    )
    gagd.write_json(
        repair_dir / "bf16_exact_active_pair_scale_sweep.json", scale_reports
    )
    gagd.write_json(
        repair_dir / "bf16_exact_multirow_scale_sweep.json", scale_reports
    )
    selected_scale_report = next(
        row for row in scale_reports if float(row["scale"]) == selected_scale
    )

    candidate_result = baseline.evaluate_eff_only(
        method=METHOD_LABEL + " candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:active-pair-candidate",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=repair_dir / "candidate_official_eval.json",
        args=args,
        records=preselection_records,
    )
    gate = select_candidate(base_result, candidate_result, selected_scale_report)
    accepted = bool(gate["accepted"])
    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selection_reason = "candidate_passed_exact_zero_eff_base_retain_ppl_and_protection_gates"
    else:
        materialize_multirow_scale(
            output_layer.weight, row_ids, original_rows, delta_rows, 0.0
        )
        selected_scale = 0.0
        selected_result = copy.deepcopy(setting5_result)
        selection_reason = "candidate_rejected_and_setting5e_exactly_restored"

    selection_commit = {
        "repair_type": REPAIR_TYPE,
        "selection_irrevocable": True,
        "candidate_accepted": accepted,
        "selected_scale": float(selected_scale),
        "selection_reason": selection_reason,
        "held_out_metrics_observed": False,
        "atomic_gen_used_for_selection": False,
        "multihop_used_for_selection": False,
    }
    gagd.write_json(output_dir / "selection_commit.json", selection_commit)
    if args.save_selected_checkpoint:
        baseline.save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    final_materialized_delta = (
        output_layer.weight.index_select(0, row_tensor).detach().float()
        - original_rows.detach().float()
    )
    actual_modified_rows = int(
        (final_materialized_delta.abs().sum(dim=1) != 0).sum().item()
    )
    sensitive_rows = set(sensitive_counts)
    competitor_rows = set(competitor_counts)
    initial_pair_margins = [
        float((pair.competitor_base_logit - pair.sensitive_base_logit).cpu())
        for pair in active_pairs
    ]
    selected_pair_violations = (
        int(selected_scale_report["active_pair_margin_violations"])
        if accepted
        else int(zero_baseline["active_pair_margin_violations_at_zero"])
    )
    selected_protected_violations = (
        int(selected_scale_report["protected_pair_margin_violations"])
        if accepted
        else int(zero_baseline["protected_pair_margin_violations_at_zero"])
    )

    def stage_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "Eff": result["forget"].get("Eff"),
            "RetainEff": result["retain"].get("Eff"),
            "PPL": result.get("forget_PPL"),
        }

    repair_summary = {
        "method": METHOD_LABEL,
        "repair_type": REPAIR_TYPE,
        "repair_rank": int(args.repair_rank),
        "active_source": ACTIVE_SOURCE,
        "active_pair_count": len(active_pairs),
        "active_sensitive_token_count": len(active_pairs),
        "unique_sensitive_row_count": len(sensitive_rows),
        "unique_competitor_row_count": len(competitor_rows),
        "modified_row_count": actual_modified_rows,
        "sensitive_and_competitor_row_overlap_count": len(
            sensitive_rows & competitor_rows
        ),
        "number_of_modified_rows": actual_modified_rows,
        "candidate_modified_row_count": len(row_ids),
        "sensitive_row_IDs": sorted(sensitive_rows),
        "competitor_row_IDs": sorted(competitor_rows),
        "unknown_token_id_for_setting5e_only": int(neutral_token_id),
        "unknown_has_special_repair_role": False,
        "active_constraints_per_sensitive_row": {
            str(k): v for k, v in sensitive_counts.items()
        },
        "active_constraints_per_competitor_row": {
            str(k): v for k, v in competitor_counts.items()
        },
        "delta_norm_per_row": {
            str(row["token_id"]): row["delta_norm"] for row in rows_report
        },
        "initial_pair_margin_min": (
            min(initial_pair_margins) if initial_pair_margins else None
        ),
        "initial_pair_margin_mean": (
            sum(initial_pair_margins) / len(initial_pair_margins)
            if initial_pair_margins
            else None
        ),
        "initial_pair_margin_max": (
            max(initial_pair_margins) if initial_pair_margins else None
        ),
        "active_pair_violations_before": int(
            zero_baseline["active_pair_margin_violations_at_zero"]
        ),
        "active_pair_violations_after": selected_pair_violations,
        "protected_pair_count": int(zero_baseline["protected_pair_count"]),
        "protected_pair_violations_before": int(
            zero_baseline["protected_pair_margin_violations_at_zero"]
        ),
        "protected_pair_violations_after": selected_protected_violations,
        "active_pair_identities": [list(pair.identity) for pair in active_pairs],
        "active_pair_identities_sha256": active_pair_identity_sha256(active_pairs),
        "candidate_BF16_scale": float(selected_scale_report["scale"]),
        "selected_BF16_scale": float(selected_scale),
        "active_failures_before": len(active_caches),
        "active_failures_after": int(
            selected_scale_report["active_correct_tokens"] if accepted else len(active_caches)
        ),
        "protected_regressions_before": 0,
        "protected_regressions_after": int(
            selected_scale_report["protected_incremental_regressions_vs_zero"]
            if accepted
            else 0
        ),
        "Eff_before": setting5_result["forget"]["Eff"],
        "Eff_after": selected_result["forget"]["Eff"],
        "retain_Eff_before": setting5_result["retain"]["Eff"],
        "retain_Eff_after": selected_result["retain"]["Eff"],
        "PPL_before": setting5_result.get("forget_PPL"),
        "PPL_after": selected_result.get("forget_PPL"),
        "candidate_accepted": accepted,
        "candidate_rejected": not accepted,
        "candidate_reason": selection_reason,
        "Base": stage_metrics(base_result),
        "Setting5e": stage_metrics(setting5_result),
        "Candidate": stage_metrics(candidate_result),
        "Selected": stage_metrics(selected_result),
        "gates": gate,
        "optimization": optimization,
        "zero_scale_baseline": zero_baseline,
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "official_atomic_questions_used_for_repair": False,
        "official_multihop_questions_used_for_repair": False,
    }
    gagd.write_json(repair_dir / "repair_summary.json", repair_summary)

    # Persist the complete checkpoint-selection evidence before any held-out
    # prompt is opened.  A rejected canonical run exits from the guarded helper
    # below, leaving this result as its durable terminal diagnostic.
    held_out_status = (
        "pending_post_selection_evaluation"
        if accepted or not args.fail_if_target_missed
        else "not_evaluated_candidate_rejected"
    )
    preselection_reporting = {
        "Eff": selected_result["forget"].get("Eff"),
        "Eff_micro": selected_result["forget"].get("Eff_micro"),
        "Eff_instance_macro": selected_result["forget"].get(
            "Eff_instance_macro"
        ),
        "AtomicGen": None,
        "AtomicGen_micro": None,
        "AtomicGen_instance_macro": None,
        "RetainEff": selected_result["retain"].get("Eff"),
        "RetainAtomicGen": None,
        "PPL": selected_result.get("forget_PPL"),
        "MHLeak_exact_any": None,
        "MHLeak_contains_any": None,
        "MHLeak_by_hop": None,
        "held_out_status": held_out_status,
    }
    preselection_result = {
        "method": METHOD_LABEL,
        "repair_type": REPAIR_TYPE,
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": int(args.seed),
        "training": {
            **asdict(train_summary),
            "forget_sampling": sampling_report,
            "steps": int(args.steps),
        },
        "repair": repair_summary,
        "base": baseline.compact_metrics(base_result),
        "setting5e": baseline.compact_metrics(setting5_result),
        "candidate": baseline.compact_metrics(candidate_result),
        "selected": baseline.compact_metrics(selected_result),
        "selected_extension": None,
        "multihop": None,
        "reporting": preselection_reporting,
        "selection": selection_commit,
    }
    gagd.write_json(output_dir / "mquake_results.json", preselection_result)

    # The checkpoint decision and all available diagnostics are now durable.
    # The helper fails here for a rejected canonical run, before it can invoke
    # AtomicGen or either multi-hop generation mode.
    selected_extension, multihop_result = _evaluate_held_out_after_selection(
        accepted=accepted,
        args=args,
        model=model,
        tok=tok,
        records=records,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        split_manifest=split_manifest,
        output_dir=output_dir,
    )

    forget_extension = selected_extension["forget"]
    retain_extension = selected_extension["retain"]
    multihop_summaries = multihop_result["results"]
    reporting = {
        "Eff": selected_result["forget"].get("Eff"),
        "Eff_micro": selected_result["forget"].get("Eff_micro"),
        "Eff_instance_macro": selected_result["forget"].get(
            "Eff_instance_macro"
        ),
        "AtomicGen": forget_extension.get("AtomicGen"),
        "AtomicGen_micro": forget_extension.get("AtomicGen_micro"),
        "AtomicGen_instance_macro": forget_extension.get(
            "AtomicGen_instance_macro"
        ),
        "RetainEff": selected_result["retain"].get("Eff"),
        "RetainAtomicGen": retain_extension.get("AtomicGen"),
        "PPL": selected_result.get("forget_PPL"),
        "MHLeak_exact_any": {
            mode: summary.get("MHLeak_exact_any")
            for mode, summary in multihop_summaries.items()
        },
        "MHLeak_contains_any": {
            mode: summary.get("MHLeak_contains_any")
            for mode, summary in multihop_summaries.items()
        },
        "MHLeak_by_hop": {
            str(hop): {
                mode: summary.get("by_hop", {}).get(str(hop))
                for mode, summary in multihop_summaries.items()
            }
            for hop in (2, 3, 4)
        },
    }

    result = {
        "method": METHOD_LABEL,
        "repair_type": REPAIR_TYPE,
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": int(args.seed),
        "training": {
            **asdict(train_summary),
            "forget_sampling": sampling_report,
            "steps": int(args.steps),
        },
        "repair": repair_summary,
        "base": baseline.compact_metrics(base_result),
        "setting5e": baseline.compact_metrics(setting5_result),
        "candidate": baseline.compact_metrics(candidate_result),
        "selected": baseline.compact_metrics(selected_result),
        "selected_extension": baseline.compact_metrics(selected_extension),
        "multihop": multihop_result["results"],
        "reporting": reporting,
        "selection": selection_commit,
    }
    gagd.write_json(output_dir / "mquake_results.json", result)
    print(
        f"Selected Eff={selected_result['forget']['Eff']}; "
        f"AtomicGen={selected_extension['forget'].get('AtomicGen')}; "
        f"RetainEff={selected_result['retain'].get('Eff')}; "
        f"PPL={selected_result.get('forget_PPL')}; accepted={accepted}"
    )
    if args.require_atomic_gen_zero:
        atomic_gen = selected_extension["forget"].get("AtomicGen")
        if atomic_gen is None or float(atomic_gen) > 0.0:
            raise RuntimeError(
                "Post-selection AtomicGen was not zero; the checkpoint remains "
                f"unchanged by this diagnostic (AtomicGen={atomic_gen})"
            )


if __name__ == "__main__":
    main()
