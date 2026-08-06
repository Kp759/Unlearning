#!/usr/bin/env python3
"""Gate-aware sensitive-row LM-head repair for saved ZsRE Setting 5e models.

This experiment deliberately does not rerun Setting 5e.  It loads a verified
600-step checkpoint, freezes the transformer and input embeddings, and learns
FP32 deltas only for sensitive answer-token rows that remain correct on the
official rewrite/paraphrase cases.  Candidate scales are materialized in the
actual BF16 LM head and selected with the native ZsRE evaluator.

The experiment is evaluation-conditioned repair: official correctness selects
active/protected constraints and official metrics select the final scale.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as legacy_repair
import zsre_zero_unlearn_official_eval as zsre


METHOD = "zsre_gate_aware_sensitive_row_repair"
PROTOCOL_STATUS = "native_data_and_metrics_but_evaluation_conditioned_repair"
SETTING5_STEPS = 600
FIXED_UTILITY_DROP_TOLERANCE = 0.10
FIXED_MAX_PPL_RATIO = 1.02
FIXED_TARGET_EFF_MAX = 0.0
FIXED_TARGET_GEN_MAX = 0.0
LIVE_PROGRESS_REQUIRED_FIELDS = frozenset(
    {
        "step",
        "total_steps",
        "total_loss",
        "active_hinge",
        "protected_hinge",
        "retain_kl",
        "delta_l2",
        "active_violation_count_full_set",
        "protected_violation_count_current_batch",
        "effective_delta_norm",
        "cuda_allocated_bytes",
        "cuda_reserved_bytes",
        "full_active_violation_count",
        "full_protected_violation_count",
        "minimum_active_slack",
        "minimum_protected_slack",
    }
)


@dataclass
class CaseBaselineCache:
    """The exact Setting 5e state needed by all repair objectives."""

    case: zsre.PredictionCase
    hidden: torch.Tensor
    target_token_id: int
    predicted_token_id: int
    target_logit: torch.Tensor
    best_other_token_id: int
    best_other_logit: torch.Tensor
    strongest_unchanged_token_id: int
    strongest_unchanged_logit: torch.Tensor
    selected_logits: torch.Tensor
    selected_probs: torch.Tensor
    correct: bool


@dataclass
class ActiveConstraintTensors:
    hidden: torch.Tensor
    sensitive_logits: torch.Tensor
    best_other_logits: torch.Tensor
    selected_row_columns: torch.Tensor


@dataclass
class ProtectedConstraintTensors:
    hidden: torch.Tensor
    target_logits: torch.Tensor
    strongest_unchanged_logits: torch.Tensor
    selected_logits: torch.Tensor
    target_selected_columns: torch.Tensor
    required_margins: torch.Tensor


@dataclass
class RetainKLTensors:
    """Flattened token tensors with contiguous offsets for each retain record."""

    hidden: torch.Tensor
    candidate_selected_probs: torch.Tensor
    reference_selected_probs: torch.Tensor
    baseline_kl: torch.Tensor
    record_ids: Tuple[int, ...]
    record_offsets: Tuple[int, ...]

    @property
    def record_count(self) -> int:
        return len(self.record_ids)

    @property
    def token_count(self) -> int:
        return int(self.hidden.shape[0])


@dataclass(frozen=True)
class CyclicBatch:
    indices: Tuple[int, ...]
    cycle: int
    start: int
    stop: int
    completed_cycle: bool


@dataclass
class InterruptionState:
    """Mutable outer-run state used to write a truthful Ctrl+C receipt."""

    output_dir: Path
    latest_completed_step: int = 0
    phase: str = "initializing"
    run_completed: bool = False


class DeterministicCyclicBatcher:
    """Visit each item exactly once per deterministic cycle."""

    def __init__(self, total_items: int, batch_size: int) -> None:
        if total_items < 0:
            raise ValueError("cyclic batch item count must be non-negative")
        if batch_size <= 0:
            raise ValueError("cyclic batch size must be positive")
        self.total_items = int(total_items)
        self.batch_size = int(batch_size)
        self.cursor = 0
        self.cycle = 0

    def next_batch(self) -> CyclicBatch:
        if self.total_items == 0:
            cycle = self.cycle
            self.cycle += 1
            return CyclicBatch((), cycle, 0, 0, True)
        start = self.cursor
        stop = min(start + self.batch_size, self.total_items)
        cycle = self.cycle
        completed = stop == self.total_items
        indices = tuple(range(start, stop))
        if completed:
            self.cursor = 0
            self.cycle += 1
        else:
            self.cursor = stop
        return CyclicBatch(indices, cycle, start, stop, completed)


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting5-checkpoint", required=True)
    parser.add_argument("--source-results", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)

    parser.add_argument("--repair-steps", type=int, default=3000)
    parser.add_argument("--repair-lr", type=float, default=1e-3)
    parser.add_argument(
        "--repair-optimizer",
        choices=("sgd", "adam", "adamw"),
        default="adamw",
    )
    parser.add_argument("--active-margin", type=float, default=0.02)
    parser.add_argument("--protected-margin-cap", type=float, default=0.05)
    parser.add_argument("--active-hinge-weight", type=float, default=2.0)
    parser.add_argument("--protected-hinge-weight", type=float, default=50.0)
    parser.add_argument("--retain-kl-mu", type=float, default=10.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    parser.add_argument("--retain-calibration-num", type=int, default=1000)
    parser.add_argument("--retain-calibration-seed", type=int, default=1729)
    parser.add_argument("--protected-batch-size", type=int, default=256)
    parser.add_argument("--retain-kl-batch-size", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--full-constraint-check-every", type=int, default=100)
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--repair-rank", type=int, default=0)
    parser.add_argument(
        "--edit-unknown-row",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exploratory override; the primary experiment excludes Unknown.",
    )

    parser.add_argument("--candidate-scale-step", type=float, default=0.025)
    parser.add_argument(
        "--utility-drop-tolerance",
        type=float,
        default=FIXED_UTILITY_DROP_TOLERANCE,
    )
    parser.add_argument("--max-ppl-ratio", type=float, default=FIXED_MAX_PPL_RATIO)
    parser.add_argument("--target-eff-max", type=float, default=FIXED_TARGET_EFF_MAX)
    parser.add_argument("--target-gen-max", type=float, default=FIXED_TARGET_GEN_MAX)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument(
        "--fail-if-target-missed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("forget and retain counts must be positive")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("repair steps and learning rate must be positive")
    if args.active_margin < 0 or args.protected_margin_cap < 0:
        raise ValueError("active/protected margins must be non-negative")
    if args.active_hinge_weight <= 0 or args.protected_hinge_weight <= 0:
        raise ValueError("hinge weights must be positive")
    if args.retain_kl_mu < 0 or args.delta_l2_lambda < 0:
        raise ValueError("KL and L2 weights must be non-negative")
    if args.repair_rank < 0:
        raise ValueError("repair rank must be non-negative")
    if args.eval_batch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("evaluation/cache batch sizes must be positive")
    if (
        args.protected_batch_size <= 0
        or args.retain_kl_batch_size <= 0
        or args.progress_every <= 0
        or args.full_constraint_check_every <= 0
    ):
        raise ValueError(
            "optimization batch sizes and progress/check intervals must be positive"
        )
    if args.cache_batch_size != args.eval_batch_size:
        raise ValueError(
            "Exact BF16 cache alignment requires --cache-batch-size to equal "
            "--eval-batch-size"
        )
    if args.retain_calibration_num != args.retain_num:
        raise ValueError(
            "Gate-aware repair requires KL caches over the full official retain "
            "set: --retain-calibration-num must equal --retain-num"
        )
    if not math.isclose(
        args.utility_drop_tolerance,
        FIXED_UTILITY_DROP_TOLERANCE,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("utility_drop_tolerance is fixed at 0.10 percentage points")
    if not math.isclose(
        args.max_ppl_ratio,
        FIXED_MAX_PPL_RATIO,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("max_ppl_ratio is fixed at 1.02")
    if args.target_eff_max != FIXED_TARGET_EFF_MAX:
        raise ValueError("target_eff_max is fixed at 0.0")
    if args.target_gen_max != FIXED_TARGET_GEN_MAX:
        raise ValueError("target_gen_max is fixed at 0.0")
    candidate_scales(args.candidate_scale_step)


def dtype_for_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def candidate_scales(step: float) -> List[float]:
    """Return an exact descending 1..0 grid without cumulative float drift."""

    if not math.isfinite(step) or step <= 0 or step > 1:
        raise ValueError("candidate scale step must be finite and in (0, 1]")
    count = int(round(1.0 / step))
    if not math.isclose(count * step, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("candidate scale step must divide 1.0 exactly")
    return [round(index * step, 12) for index in range(count, -1, -1)]


def all_special_token_ids(tok: Any) -> set[int]:
    values = set(gagd.special_token_ids(tok))
    values.update(int(value) for value in getattr(tok, "all_special_ids", []) or [])
    return values


def identity_payload(identity: Tuple[int, str, int, int]) -> List[Any]:
    return [identity[0], identity[1], identity[2], identity[3]]


def directory_sha256(path: Path) -> str:
    """Hash relative names, sizes, and contents for a saved checkpoint tree."""

    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Checkpoint directory is empty: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def discover_source_results(
    checkpoint: Path,
    explicit_path: Optional[str],
) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(gagd.resolve_output_path(explicit_path))
    seed_root = checkpoint.parent.parent
    candidates.extend(
        [
            seed_root / "zsre_results.json",
            seed_root / "source_zsre_results.json",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            payload = _load_json(candidate)
            if payload.get("dataset") != "ZsRE":
                raise ValueError(f"Source result is not ZsRE: {candidate}")
            return candidate.resolve(), payload
    raise FileNotFoundError(
        "A source zsre_results.json is required to verify the 600-step Setting "
        "5e checkpoint and recover the matching Base metrics. Pass "
        "--source-results explicitly. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def verify_source_result(
    source: Mapping[str, Any],
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
) -> None:
    if int(source.get("seed", -1)) != seed:
        raise ValueError("Source result seed does not match the requested seed")
    if int(source.get("forget_num", -1)) != forget_num:
        raise ValueError("Source result forget count does not match")
    if int(source.get("retain_num", -1)) != retain_num:
        raise ValueError("Source result retain count does not match")
    training = source.get("training")
    steps = training.get("steps") if isinstance(training, Mapping) else None
    if int(steps or -1) != SETTING5_STEPS:
        raise ValueError(
            f"Source result does not verify a {SETTING5_STEPS}-step Setting 5e "
            f"checkpoint (reported steps={steps!r})"
        )
    if not isinstance(source.get("base"), Mapping):
        raise ValueError("Source result lacks the matching Base metric block")
    if not isinstance(source.get("setting5e"), Mapping):
        raise ValueError("Source result lacks the Setting 5e metric block")


def compare_compact_metrics(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches: List[str] = []
    for split in ("forget", "retain"):
        for metric in ("Eff", "Gen", "Spe"):
            before = float(expected[split][metric])
            after = float(actual[split][metric])
            if not math.isclose(before, after, rel_tol=0.0, abs_tol=1e-6):
                mismatches.append(f"{split}.{metric}: {before} != {after}")
    expected_ppl = expected.get("PPL")
    actual_ppl = actual.get("PPL")
    if expected_ppl is None or actual_ppl is None:
        if expected_ppl != actual_ppl:
            mismatches.append(f"PPL: {expected_ppl!r} != {actual_ppl!r}")
    elif not math.isclose(
        float(expected_ppl), float(actual_ppl), rel_tol=0.0, abs_tol=1e-6
    ):
        mismatches.append(f"PPL: {expected_ppl} != {actual_ppl}")
    if mismatches:
        raise RuntimeError(f"{label} metric identity check failed: " + "; ".join(mismatches))


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> Tuple[nn.Module, Any, Path]:
    checkpoint = gagd.resolve_output_path(args.setting5_checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Setting 5e checkpoint not found: {checkpoint}")
    kwargs: Dict[str, Any] = {"dtype": dtype_for_name(args.dtype)}
    if args.device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    legacy_repair.validate_neutral_target_checkpoint(checkpoint, tok)
    if model.get_output_embeddings().weight.shape[0] < len(tok):
        raise ValueError("Tokenizer vocabulary exceeds the checkpoint LM head")
    model.config.use_cache = False
    model.eval()
    return model, tok, checkpoint


def target_ids_for_cases(
    tok: Any,
    cases: Sequence[zsre.PredictionCase],
    *,
    llama_like: bool,
    batch_size: int,
) -> Dict[Tuple[int, str, int, int], int]:
    output: Dict[Tuple[int, str, int, int], int] = {}
    for batch in _chunks(list(cases), batch_size):
        target_ids = zsre.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=torch.device("cpu"),
        ).tolist()
        for case, target_id in zip(batch, target_ids):
            if case.identity in output:
                raise ValueError(f"Duplicate prediction-case identity: {case.identity}")
            output[case.identity] = int(target_id)
    return output


@torch.no_grad()
def cache_case_baselines(
    model: nn.Module,
    tok: Any,
    cases: Sequence[zsre.PredictionCase],
    *,
    selected_ids: Sequence[int],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    desc: str,
) -> List[CaseBaselineCache]:
    """Cache exact official-order hidden/logit state with no trainable model."""

    if batch_size <= 0:
        raise ValueError("cache batch size must be positive")
    selected = torch.tensor(selected_ids, dtype=torch.long, device=device)
    output: List[CaseBaselineCache] = []
    batches = list(_chunks(list(cases), batch_size))
    for batch in tqdm(batches, desc=desc, leave=False):
        encoded = tok(
            [case.prompt for case in batch],
            padding=True,
            return_tensors="pt",
        ).to(device)
        model_output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
        last_non_masked = encoded["attention_mask"].sum(dim=1) - 1
        batch_indices = torch.arange(len(batch), device=device)
        hidden = model_output.hidden_states[-1][
            batch_indices, last_non_masked, :
        ].float()
        logits = model_output.logits[batch_indices, last_non_masked, :].float()
        target_ids = zsre.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predicted_ids = logits.argmax(dim=-1)
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)

        best_other_values = logits.clone()
        best_other_values.scatter_(1, target_ids[:, None], -torch.inf)
        best_other_logits, best_other_ids = best_other_values.max(dim=-1)

        unchanged_values = best_other_values
        if selected.numel():
            unchanged_values = unchanged_values.clone()
            unchanged_values.index_fill_(1, selected, -torch.inf)
        unchanged_logits, unchanged_ids = unchanged_values.max(dim=-1)
        if not torch.isfinite(unchanged_logits).all():
            raise RuntimeError("No unchanged LM-head competitor remained")

        if selected.numel():
            selected_logits = logits.index_select(1, selected)
            selected_probs = F.softmax(logits, dim=-1).index_select(1, selected)
        else:
            selected_logits = logits.new_empty((len(batch), 0))
            selected_probs = logits.new_empty((len(batch), 0))

        for index, case in enumerate(batch):
            target_id = int(target_ids[index].item())
            predicted_id = int(predicted_ids[index].item())
            output.append(
                CaseBaselineCache(
                    case=case,
                    hidden=hidden[index].detach(),
                    target_token_id=target_id,
                    predicted_token_id=predicted_id,
                    target_logit=target_logits[index].detach(),
                    best_other_token_id=int(best_other_ids[index].item()),
                    best_other_logit=best_other_logits[index].detach(),
                    strongest_unchanged_token_id=int(unchanged_ids[index].item()),
                    strongest_unchanged_logit=unchanged_logits[index].detach(),
                    selected_logits=selected_logits[index].detach(),
                    selected_probs=selected_probs[index].detach(),
                    correct=bool(target_id == predicted_id),
                )
            )
    return output


def active_constraint_tensors(
    caches: Sequence[CaseBaselineCache],
    selected_ids: Sequence[int],
    *,
    device: torch.device,
) -> ActiveConstraintTensors:
    columns = {token_id: index for index, token_id in enumerate(selected_ids)}
    missing = sorted(
        {cache.target_token_id for cache in caches if cache.target_token_id not in columns}
    )
    if missing:
        raise ValueError(f"Active sensitive rows were not selected: {missing}")
    if not caches:
        hidden_size = 0
        return ActiveConstraintTensors(
            hidden=torch.empty(
                (0, hidden_size), dtype=torch.float32, device=device
            ),
            sensitive_logits=torch.empty((0,), dtype=torch.float32, device=device),
            best_other_logits=torch.empty((0,), dtype=torch.float32, device=device),
            selected_row_columns=torch.empty(
                (0,), dtype=torch.long, device=device
            ),
        )
    return ActiveConstraintTensors(
        hidden=torch.stack([cache.hidden for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        sensitive_logits=torch.stack([cache.target_logit for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        best_other_logits=torch.stack([cache.best_other_logit for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        selected_row_columns=torch.tensor(
            [columns[cache.target_token_id] for cache in caches],
            dtype=torch.long,
            device=device,
        ),
    )


def active_constraint_margins(
    hidden: torch.Tensor,
    sensitive_logits: torch.Tensor,
    best_other_logits: torch.Tensor,
    selected_row_columns: torch.Tensor,
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if hidden.shape[0] == 0:
        return delta_rows.new_empty((0,))
    corrections = hidden @ delta_rows.transpose(0, 1)
    sensitive_correction = corrections.gather(
        1, selected_row_columns[:, None]
    ).squeeze(1)
    return best_other_logits - (sensitive_logits + sensitive_correction)


def build_protected_constraint_tensors(
    caches: Sequence[CaseBaselineCache],
    selected_ids: Sequence[int],
    *,
    protected_margin_cap: float,
    device: torch.device,
) -> ProtectedConstraintTensors:
    columns = {token_id: index for index, token_id in enumerate(selected_ids)}
    if not caches:
        return ProtectedConstraintTensors(
            hidden=torch.empty((0, 0), dtype=torch.float32, device=device),
            target_logits=torch.empty((0,), dtype=torch.float32, device=device),
            strongest_unchanged_logits=torch.empty(
                (0,), dtype=torch.float32, device=device
            ),
            selected_logits=torch.empty(
                (0, len(selected_ids)), dtype=torch.float32, device=device
            ),
            target_selected_columns=torch.empty(
                (0,), dtype=torch.long, device=device
            ),
            required_margins=torch.empty(
                (0,), dtype=torch.float32, device=device
            ),
        )
    original_margins = torch.stack(
        [cache.target_logit - cache.best_other_logit for cache in caches]
    ).to(device=device, dtype=torch.float32)
    if (original_margins < -1e-5).any():
        raise RuntimeError("A protected token is not top-1 in the exact baseline cache")
    required = torch.minimum(
        original_margins.clamp_min(0.0),
        torch.full_like(original_margins, float(protected_margin_cap)),
    )
    return ProtectedConstraintTensors(
        hidden=torch.stack([cache.hidden for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        target_logits=torch.stack([cache.target_logit for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        strongest_unchanged_logits=torch.stack(
            [cache.strongest_unchanged_logit for cache in caches]
        ).to(device=device, dtype=torch.float32),
        selected_logits=torch.stack([cache.selected_logits for cache in caches]).to(
            device=device, dtype=torch.float32
        ),
        target_selected_columns=torch.tensor(
            [columns.get(cache.target_token_id, -1) for cache in caches],
            dtype=torch.long,
            device=device,
        ),
        required_margins=required,
    )


def protected_constraint_margins(
    hidden: torch.Tensor,
    target_logits: torch.Tensor,
    strongest_unchanged_logits: torch.Tensor,
    selected_logits: torch.Tensor,
    target_selected_columns: torch.Tensor,
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Exact target-vs-unchanged/edited-row top-1 preservation margins."""

    if hidden.shape[0] == 0:
        return delta_rows.new_empty((0,))
    corrections = hidden @ delta_rows.transpose(0, 1)
    corrected_selected = selected_logits + corrections
    selected_target = target_selected_columns.ge(0)
    safe_columns = target_selected_columns.clamp_min(0)
    target_correction = corrections.gather(1, safe_columns[:, None]).squeeze(1)
    corrected_target = target_logits + torch.where(
        selected_target,
        target_correction,
        torch.zeros_like(target_correction),
    )

    if corrected_selected.shape[1]:
        selected_competitors = corrected_selected.clone()
        if selected_target.any():
            row_indices = torch.arange(hidden.shape[0], device=hidden.device)[
                selected_target
            ]
            selected_competitors[
                row_indices, target_selected_columns[selected_target]
            ] = -torch.inf
        strongest_selected = selected_competitors.max(dim=1).values
    else:
        strongest_selected = torch.full_like(corrected_target, -torch.inf)
    strongest_competitor = torch.maximum(
        strongest_unchanged_logits,
        strongest_selected,
    )
    return corrected_target - strongest_competitor


def protected_constraint_slice(
    tensors: ProtectedConstraintTensors,
    batch: CyclicBatch,
) -> ProtectedConstraintTensors:
    """Return a zero-copy contiguous cyclic optimization slice."""

    selection = slice(batch.start, batch.stop)
    return ProtectedConstraintTensors(
        hidden=tensors.hidden[selection],
        target_logits=tensors.target_logits[selection],
        strongest_unchanged_logits=tensors.strongest_unchanged_logits[selection],
        selected_logits=tensors.selected_logits[selection],
        target_selected_columns=tensors.target_selected_columns[selection],
        required_margins=tensors.required_margins[selection],
    )


def build_retain_kl_caches(
    caches: Sequence[CaseBaselineCache],
) -> RetainKLTensors:
    """Flatten all tokens while retaining contiguous record boundaries."""

    if not caches:
        raise ValueError("Full-retain KL cache cannot be empty")
    record_ids: List[int] = []
    record_offsets: List[int] = [0]
    seen: set[int] = set()
    previous: Optional[int] = None
    for position, cache in enumerate(caches):
        case_id = int(cache.case.case_id)
        if case_id == previous:
            continue
        if case_id in seen:
            raise ValueError(
                "Retain prediction cases for one record must remain contiguous"
            )
        if previous is not None:
            record_offsets.append(position)
        seen.add(case_id)
        record_ids.append(case_id)
        previous = case_id
    record_offsets.append(len(caches))
    hidden = torch.stack([cache.hidden for cache in caches]).float()
    probabilities = torch.stack([cache.selected_probs for cache in caches]).float()
    return RetainKLTensors(
        hidden=hidden,
        candidate_selected_probs=probabilities,
        reference_selected_probs=probabilities.clone(),
        baseline_kl=torch.zeros(
            len(caches), dtype=torch.float32, device=hidden.device
        ),
        record_ids=tuple(record_ids),
        record_offsets=tuple(record_offsets),
    )


def retain_kl_from_tensors(
    tensors: RetainKLTensors,
    delta_rows: torch.Tensor,
    *,
    record_start: int = 0,
    record_stop: Optional[int] = None,
) -> torch.Tensor:
    """Vectorized selected-row KL for a contiguous set of retain records."""

    stop = tensors.record_count if record_stop is None else int(record_stop)
    start = int(record_start)
    if not 0 <= start <= stop <= tensors.record_count:
        raise ValueError("Invalid retain-record KL slice")
    token_start = tensors.record_offsets[start]
    token_stop = tensors.record_offsets[stop]
    if token_start == token_stop:
        return delta_rows.new_zeros(())
    hidden = tensors.hidden[token_start:token_stop]
    candidate_probs = tensors.candidate_selected_probs[token_start:token_stop]
    reference_probs = tensors.reference_selected_probs[token_start:token_stop]
    baseline_kl = tensors.baseline_kl[token_start:token_stop]
    corrections = hidden @ delta_rows.transpose(0, 1)
    log_shift = active._log_partition_shift(candidate_probs, corrections)
    return (
        baseline_kl
        + log_shift
        - (reference_probs * corrections).sum(dim=-1)
    ).mean()


def retain_kl_from_caches(
    caches: Sequence[active.RetainKLCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    return active.retain_kl_from_caches(caches, delta_rows)


def _squared_hinge(margins: torch.Tensor, required: torch.Tensor) -> torch.Tensor:
    if margins.numel() == 0:
        return required.new_zeros(())
    return F.relu(required - margins).square().mean()


def _minimum(values: torch.Tensor) -> Optional[float]:
    return None if values.numel() == 0 else float(values.min().detach().cpu())


def _format_optional_float(value: Optional[float]) -> str:
    return "none" if value is None else f"{value:.6g}"


@torch.no_grad()
def full_constraint_check_chunked(
    active_tensors: ActiveConstraintTensors,
    protected_tensors: ProtectedConstraintTensors,
    delta_rows: torch.Tensor,
    *,
    active_margin: float,
    chunk_size: int,
) -> Dict[str, Any]:
    """Evaluate every constraint with bounded no-gradient matrix products."""

    if chunk_size <= 0:
        raise ValueError("full constraint check chunk size must be positive")

    full_active_violation_count = 0
    full_protected_violation_count = 0
    minimum_active_slack: Optional[float] = None
    minimum_protected_slack: Optional[float] = None
    minimum_active_margin: Optional[float] = None
    minimum_protected_margin: Optional[float] = None
    active_chunks_evaluated = 0
    protected_chunks_evaluated = 0

    active_count = int(active_tensors.hidden.shape[0])
    for start in range(0, active_count, chunk_size):
        stop = min(start + chunk_size, active_count)
        margins = active_constraint_margins(
            active_tensors.hidden[start:stop],
            active_tensors.sensitive_logits[start:stop],
            active_tensors.best_other_logits[start:stop],
            active_tensors.selected_row_columns[start:stop],
            delta_rows,
        )
        slack = margins - float(active_margin)
        full_active_violation_count += int(slack.lt(0).sum().item())
        margin_minimum = _minimum(margins)
        if margin_minimum is not None:
            minimum_active_margin = (
                margin_minimum
                if minimum_active_margin is None
                else min(minimum_active_margin, margin_minimum)
            )
        chunk_minimum = _minimum(slack)
        if chunk_minimum is not None:
            minimum_active_slack = (
                chunk_minimum
                if minimum_active_slack is None
                else min(minimum_active_slack, chunk_minimum)
            )
        active_chunks_evaluated += 1

    protected_count = int(protected_tensors.hidden.shape[0])
    for start in range(0, protected_count, chunk_size):
        stop = min(start + chunk_size, protected_count)
        margins = protected_constraint_margins(
            protected_tensors.hidden[start:stop],
            protected_tensors.target_logits[start:stop],
            protected_tensors.strongest_unchanged_logits[start:stop],
            protected_tensors.selected_logits[start:stop],
            protected_tensors.target_selected_columns[start:stop],
            delta_rows,
        )
        slack = margins - protected_tensors.required_margins[start:stop]
        full_protected_violation_count += int(slack.lt(0).sum().item())
        margin_minimum = _minimum(margins)
        if margin_minimum is not None:
            minimum_protected_margin = (
                margin_minimum
                if minimum_protected_margin is None
                else min(minimum_protected_margin, margin_minimum)
            )
        chunk_minimum = _minimum(slack)
        if chunk_minimum is not None:
            minimum_protected_slack = (
                chunk_minimum
                if minimum_protected_slack is None
                else min(minimum_protected_slack, chunk_minimum)
            )
        protected_chunks_evaluated += 1

    return {
        "full_active_violation_count": full_active_violation_count,
        "full_protected_violation_count": full_protected_violation_count,
        "minimum_active_slack": minimum_active_slack,
        "minimum_protected_slack": minimum_protected_slack,
        "minimum_active_margin": minimum_active_margin,
        "minimum_protected_margin": minimum_protected_margin,
        "active_chunks_evaluated": active_chunks_evaluated,
        "protected_chunks_evaluated": protected_chunks_evaluated,
        "chunk_size": int(chunk_size),
    }


def optimize_gate_aware_delta(
    active_tensors: ActiveConstraintTensors,
    protected_tensors: ProtectedConstraintTensors,
    retain_kl_tensors: RetainKLTensors,
    *,
    selected_row_count: int,
    hidden_size: int,
    args: argparse.Namespace,
    device: torch.device,
    live_progress_path: Path,
    interruption_state: Optional[InterruptionState] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Joint FP32 optimization with deterministic cyclic utility batches."""

    live_progress_path = Path(live_progress_path)
    live_progress_path.parent.mkdir(parents=True, exist_ok=True)
    if interruption_state is not None:
        interruption_state.phase = "optimization"
        interruption_state.latest_completed_step = 0

    if selected_row_count == 0:
        if active_tensors.hidden.shape[0]:
            raise RuntimeError("Active constraints exist but no sensitive row is editable")
        with live_progress_path.open("w", encoding="utf-8"):
            pass
        return (
            torch.empty((0, hidden_size), dtype=torch.float32, device=device),
            [],
            {
                "steps_completed": 0,
                "stopped_early": True,
                "all_active_satisfied": True,
                "all_protected_satisfied": True,
                "full_retain_kl_after": 0.0,
                "reason": "setting5e_has_no_active_sensitive_tokens",
            },
        )

    direction_basis = None
    if args.repair_rank > 0:
        direction_basis = active.orthonormal_row_basis(
            active_tensors.hidden,
            max_rank=args.repair_rank,
        )
        if direction_basis.numel() == 0:
            raise RuntimeError("Active hidden-state basis has zero rank")
    module = active.SelectedRowDelta(
        n_rows=selected_row_count,
        hidden_size=hidden_size,
        direction_basis=direction_basis,
        retained_basis=None,
        device=device,
    )
    optimizer = active.make_repair_optimizer(
        module,
        args.repair_optimizer,
        args.repair_lr,
    )
    active_required = torch.full(
        (active_tensors.hidden.shape[0],),
        float(args.active_margin),
        dtype=torch.float32,
        device=device,
    )
    protected_required = protected_tensors.required_margins
    protected_batcher = DeterministicCyclicBatcher(
        int(protected_required.numel()),
        args.protected_batch_size,
    )
    retain_batcher = DeterministicCyclicBatcher(
        retain_kl_tensors.record_count,
        args.retain_kl_batch_size,
    )
    logs: List[Dict[str, Any]] = []
    stopped_early = False

    with live_progress_path.open("w", encoding="utf-8", buffering=1) as live_handle:
        for step in range(1, args.repair_steps + 1):
            protected_batch = protected_batcher.next_batch()
            retain_batch = retain_batcher.next_batch()
            protected_current = protected_constraint_slice(
                protected_tensors,
                protected_batch,
            )

            optimizer.zero_grad(set_to_none=True)
            delta_rows = module.effective_delta()
            active_margins = active_constraint_margins(
                active_tensors.hidden,
                active_tensors.sensitive_logits,
                active_tensors.best_other_logits,
                active_tensors.selected_row_columns,
                delta_rows,
            )
            protected_margins = protected_constraint_margins(
                protected_current.hidden,
                protected_current.target_logits,
                protected_current.strongest_unchanged_logits,
                protected_current.selected_logits,
                protected_current.target_selected_columns,
                delta_rows,
            )
            active_hinge = _squared_hinge(active_margins, active_required)
            protected_hinge = _squared_hinge(
                protected_margins,
                protected_current.required_margins,
            )
            retain_kl = retain_kl_from_tensors(
                retain_kl_tensors,
                delta_rows,
                record_start=retain_batch.start,
                record_stop=retain_batch.stop,
            )
            delta_l2 = delta_rows.square().sum()
            total = (
                args.active_hinge_weight * active_hinge
                + args.protected_hinge_weight * protected_hinge
                + args.retain_kl_mu * retain_kl
                + args.delta_l2_lambda * delta_l2
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Non-finite repair loss at step {step}")
            total.backward()
            optimizer.step()
            if interruption_state is not None:
                interruption_state.latest_completed_step = step

            progress_due = (
                step == 1
                or step % args.progress_every == 0
                or step == args.repair_steps
            )
            full_check_due = (
                step % args.full_constraint_check_every == 0
                or step == args.repair_steps
            )
            if not (progress_due or full_check_due):
                continue

            with torch.no_grad():
                updated = module.effective_delta()
                active_after = active_constraint_margins(
                    active_tensors.hidden,
                    active_tensors.sensitive_logits,
                    active_tensors.best_other_logits,
                    active_tensors.selected_row_columns,
                    updated,
                )
                protected_batch_after = protected_constraint_margins(
                    protected_current.hidden,
                    protected_current.target_logits,
                    protected_current.strongest_unchanged_logits,
                    protected_current.selected_logits,
                    protected_current.target_selected_columns,
                    updated,
                )
                active_violations = int(
                    (active_after < active_required).sum().item()
                )
                protected_batch_violations = int(
                    (
                        protected_batch_after
                        < protected_current.required_margins
                    )
                    .sum()
                    .item()
                )
                full_check: Optional[Dict[str, Any]] = None
                all_satisfied = False
                if full_check_due:
                    full_check = full_constraint_check_chunked(
                        active_tensors,
                        protected_tensors,
                        updated,
                        active_margin=args.active_margin,
                        chunk_size=args.protected_batch_size,
                    )
                    all_satisfied = (
                        full_check["full_active_violation_count"] == 0
                        and full_check["full_protected_violation_count"] == 0
                    )

                if device.type == "cuda":
                    gpu_allocated_bytes = int(torch.cuda.memory_allocated(device))
                    gpu_reserved_bytes = int(torch.cuda.memory_reserved(device))
                else:
                    gpu_allocated_bytes = 0
                    gpu_reserved_bytes = 0
                gpu_allocated = float(gpu_allocated_bytes / (1024**2))
                gpu_reserved = float(gpu_reserved_bytes / (1024**2))
                full_active_violations = (
                    None
                    if full_check is None
                    else int(full_check["full_active_violation_count"])
                )
                full_protected_violations = (
                    None
                    if full_check is None
                    else int(full_check["full_protected_violation_count"])
                )
                minimum_active_slack = (
                    None if full_check is None else full_check["minimum_active_slack"]
                )
                minimum_protected_slack = (
                    None
                    if full_check is None
                    else full_check["minimum_protected_slack"]
                )
                row = {
                    "step": step,
                    "total_steps": args.repair_steps,
                    "total_loss": float(total.detach().cpu()),
                    "active_hinge": float(active_hinge.detach().cpu()),
                    "protected_hinge": float(protected_hinge.detach().cpu()),
                    "retain_kl": float(retain_kl.detach().cpu()),
                    "active_squared_hinge": float(active_hinge.detach().cpu()),
                    "protected_squared_hinge": float(
                        protected_hinge.detach().cpu()
                    ),
                    "retain_kl_setting5e_to_repaired": float(
                        retain_kl.detach().cpu()
                    ),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "active_violation_count_full_set": active_violations,
                    "protected_violation_count_current_batch": (
                        protected_batch_violations
                    ),
                    "active_violations_full": active_violations,
                    "protected_violations_current_batch": (
                        protected_batch_violations
                    ),
                    "full_active_violation_count": full_active_violations,
                    "full_protected_violation_count": full_protected_violations,
                    "minimum_active_slack": minimum_active_slack,
                    "minimum_protected_slack": minimum_protected_slack,
                    "protected_violations_full": full_protected_violations,
                    "retain_kl_full": None,
                    "effective_delta_norm": float(
                        updated.norm().detach().cpu()
                    ),
                    "cuda_allocated_bytes": gpu_allocated_bytes,
                    "cuda_reserved_bytes": gpu_reserved_bytes,
                    "gpu_allocated_mib": gpu_allocated,
                    "gpu_reserved_mib": gpu_reserved,
                    "protected_batch": {
                        "cycle": protected_batch.cycle,
                        "start": protected_batch.start,
                        "stop": protected_batch.stop,
                        "size": len(protected_batch.indices),
                        "completed_cycle": protected_batch.completed_cycle,
                    },
                    "retain_kl_batch": {
                        "cycle": retain_batch.cycle,
                        "record_start": retain_batch.start,
                        "record_stop": retain_batch.stop,
                        "record_count": len(retain_batch.indices),
                        "completed_cycle": retain_batch.completed_cycle,
                    },
                    "full_constraint_check": full_check_due,
                    "full_constraint_check_chunks": (
                        None
                        if full_check is None
                        else {
                            "active": full_check["active_chunks_evaluated"],
                            "protected": full_check[
                                "protected_chunks_evaluated"
                            ],
                            "chunk_size": full_check["chunk_size"],
                        }
                    ),
                    "all_constraints_satisfied_on_full_check": (
                        all_satisfied if full_check_due else None
                    ),
                }
                missing_progress_fields = LIVE_PROGRESS_REQUIRED_FIELDS - row.keys()
                if missing_progress_fields:
                    raise RuntimeError(
                        "Internal live-progress schema error; missing fields: "
                        f"{sorted(missing_progress_fields)}"
                    )
                logs.append(row)
                live_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                live_handle.flush()
                full_check_text = ""
                if full_check is not None:
                    full_check_text = (
                        " full_active_violations="
                        f"{full_active_violations}"
                        " full_protected_violations="
                        f"{full_protected_violations}"
                        " minimum_active_slack="
                        f"{_format_optional_float(minimum_active_slack)}"
                        " minimum_protected_slack="
                        f"{_format_optional_float(minimum_protected_slack)}"
                    )
                print(
                    "[gate-aware repair] "
                    f"step={step}/{args.repair_steps} "
                    f"total_loss={row['total_loss']:.6g} "
                    f"active_hinge={row['active_hinge']:.6g} "
                    f"protected_hinge={row['protected_hinge']:.6g} "
                    f"retain_kl={row['retain_kl']:.6g} "
                    f"delta_l2={row['delta_l2']:.6g} "
                    f"active_violations_full={active_violations} "
                    "protected_violations_batch="
                    f"{protected_batch_violations} "
                    f"delta_norm={row['effective_delta_norm']:.6g} "
                    f"gpu_allocated_mib={gpu_allocated:.1f} "
                    f"gpu_reserved_mib={gpu_reserved:.1f}"
                    f"{full_check_text}",
                    flush=True,
                )
            if (
                args.stop_when_all_satisfied
                and full_check_due
                and all_satisfied
            ):
                stopped_early = True
                break

    delta = module.effective_delta().detach()
    final_check = full_constraint_check_chunked(
        active_tensors,
        protected_tensors,
        delta,
        active_margin=args.active_margin,
        chunk_size=args.protected_batch_size,
    )
    final_retain_kl = retain_kl_from_tensors(retain_kl_tensors, delta)
    return delta, logs, {
        "steps_completed": step,
        "progress_records": len(logs),
        "stopped_early": stopped_early,
        "all_active_satisfied": final_check["full_active_violation_count"] == 0,
        "all_protected_satisfied": (
            final_check["full_protected_violation_count"] == 0
        ),
        "active_constraint_count": int(active_required.numel()),
        "protected_constraint_count": int(protected_required.numel()),
        "active_violations_after": final_check["full_active_violation_count"],
        "protected_violations_after": final_check[
            "full_protected_violation_count"
        ],
        "minimum_active_slack_after": final_check["minimum_active_slack"],
        "minimum_protected_slack_after": final_check[
            "minimum_protected_slack"
        ],
        "minimum_active_margin_after": final_check["minimum_active_margin"],
        "minimum_protected_margin_after": final_check[
            "minimum_protected_margin"
        ],
        "full_retain_kl_after": float(final_retain_kl.detach().cpu()),
        "protected_batch_size": args.protected_batch_size,
        "retain_kl_batch_size_records": args.retain_kl_batch_size,
        "protected_cycles_completed": protected_batcher.cycle,
        "retain_kl_cycles_completed": retain_batcher.cycle,
        "live_progress_path": str(live_progress_path),
        "effective_delta_norm": float(delta.norm().detach().cpu()),
        "direction_rank": (
            hidden_size if direction_basis is None else int(direction_basis.shape[0])
        ),
    }


@torch.no_grad()
def materialize_sensitive_rows(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
) -> float:
    """Set, never accumulate, a candidate in the actual output-weight dtype."""

    if len(selected_ids) != original_rows.shape[0] or len(selected_ids) != delta_rows.shape[0]:
        raise ValueError("selected row snapshots and deltas do not align")
    if not selected_ids:
        return 0.0
    ids = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    materialized = original_rows + (float(scale) * delta_rows).to(
        device=original_rows.device,
        dtype=original_rows.dtype,
    )
    output_weight.index_copy_(0, ids, materialized)
    effective = output_weight.index_select(0, ids).float() - original_rows.float()
    return float(effective.norm().detach().cpu())


def _metric_check(
    *,
    setting5: Optional[float],
    candidate: Optional[float],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Dict[str, Any]:
    if setting5 is None or candidate is None:
        passed = False
    elif minimum is not None:
        passed = float(candidate) >= float(minimum)
    elif maximum is not None:
        passed = float(candidate) <= float(maximum)
    else:
        raise ValueError("Metric check needs a bound")
    return {
        "setting5": setting5,
        "candidate": candidate,
        "minimum": minimum,
        "maximum": maximum,
        "passed": bool(passed),
    }


def non_ppl_gate_report(
    setting5: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    tolerance = FIXED_UTILITY_DROP_TOLERANCE
    checks = {
        "forget_Eff_target": _metric_check(
            setting5=setting5["forget"]["Eff"],
            candidate=candidate["forget"]["Eff"],
            maximum=FIXED_TARGET_EFF_MAX,
        ),
        "forget_Gen_target": _metric_check(
            setting5=setting5["forget"]["Gen"],
            candidate=candidate["forget"]["Gen"],
            maximum=FIXED_TARGET_GEN_MAX,
        ),
        "forget_Spe": _metric_check(
            setting5=setting5["forget"]["Spe"],
            candidate=candidate["forget"]["Spe"],
            minimum=float(setting5["forget"]["Spe"]) - tolerance,
        ),
    }
    for metric in ("Eff", "Gen", "Spe"):
        checks[f"retain_{metric}"] = _metric_check(
            setting5=setting5["retain"][metric],
            candidate=candidate["retain"][metric],
            minimum=float(setting5["retain"][metric]) - tolerance,
        )
    return {
        "passed": all(check["passed"] for check in checks.values()),
        "utility_drop_tolerance_percentage_points": tolerance,
        "target_eff_max": FIXED_TARGET_EFF_MAX,
        "target_gen_max": FIXED_TARGET_GEN_MAX,
        "checks": checks,
    }


def full_gate_report(
    setting5: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    non_ppl = non_ppl_gate_report(setting5, candidate)
    setting_ppl = setting5.get("forget_PPL")
    candidate_ppl = candidate.get("forget_PPL")
    ppl = _metric_check(
        setting5=setting_ppl,
        candidate=candidate_ppl,
        maximum=(
            None
            if setting_ppl is None
            else float(setting_ppl) * FIXED_MAX_PPL_RATIO
        ),
    )
    return {
        "passed": bool(non_ppl["passed"] and ppl["passed"]),
        "non_ppl": non_ppl,
        "PPL": ppl,
        "max_ppl_ratio": FIXED_MAX_PPL_RATIO,
    }


def setting5_already_meets_target(setting5: Mapping[str, Any]) -> bool:
    return bool(
        float(setting5["forget"]["Eff"]) <= FIXED_TARGET_EFF_MAX
        and float(setting5["forget"]["Gen"]) <= FIXED_TARGET_GEN_MAX
    )


def select_all_gates_candidate(
    candidates: Sequence[Mapping[str, Any]],
    *,
    setting5_target_already_met: bool,
) -> Optional[Mapping[str, Any]]:
    passing = [
        candidate
        for candidate in candidates
        if bool(candidate["full_gate"]["passed"])
        and (
            float(candidate["scale"]) > 0.0
            or setting5_target_already_met
        )
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda candidate: (
            float(candidate["materialized_delta_norm"]),
            float(candidate["scale"]),
        ),
    )


def save_selected_checkpoint_if_accepted(
    *,
    accepted: bool,
    model: Any,
    tok: Any,
    output_dir: Path,
) -> Optional[Path]:
    if not accepted:
        return None
    checkpoint = Path(output_dir) / "selected_checkpoint"
    legacy_repair.save_checkpoint(model, tok, checkpoint)
    return checkpoint


def _discard_selected_checkpoint_after_interruption(output_dir: Path) -> bool:
    """Remove only the publishable checkpoint from an interrupted run."""

    checkpoint = Path(output_dir) / "selected_checkpoint"
    if checkpoint.is_symlink() or checkpoint.is_file():
        checkpoint.unlink()
        return True
    if checkpoint.is_dir():
        shutil.rmtree(checkpoint)
        return True
    return False


def write_interruption_receipt(state: InterruptionState) -> Path:
    """Record an outer-level Ctrl+C and prevent checkpoint publication."""

    interrupted_phase = state.phase
    selected_checkpoint_removed = _discard_selected_checkpoint_after_interruption(
        state.output_dir
    )
    state.phase = "interrupted"
    state.run_completed = False
    path = state.output_dir / "optimization" / "interrupted.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    gagd.write_json(
        path,
        {
            "status": "interrupted",
            "exit_status": 130,
            "latest_completed_step": int(state.latest_completed_step),
            "phase_at_interrupt": interrupted_phase,
            "interrupted_at_utc": datetime.now(timezone.utc).isoformat(),
            "selected_checkpoint_removed": selected_checkpoint_removed,
            "selected_checkpoint_emitted": False,
        },
    )
    return path


def execute_with_interrupt_receipt(
    execute: Callable[[], None],
    state: InterruptionState,
) -> None:
    """Run one seed and handle KeyboardInterrupt only at the outer boundary."""

    try:
        execute()
    except KeyboardInterrupt:
        write_interruption_receipt(state)
        raise SystemExit(130) from None


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return legacy_repair.compact_metrics(result)


def _ensure_fresh_output(output_dir: Path) -> None:
    protected_names = (
        "config_used.json",
        "sampled_case_ids.json",
        "setting5e_official_eval.json",
        "zsre_results.json",
        "active_candidate_checkpoint",
        "selected_checkpoint",
        "optimization",
        "scale_sweep",
    )
    existing = [name for name in protected_names if (output_dir / name).exists()]
    if existing:
        raise FileExistsError(
            "Refusing to mix a new repair with existing run artifacts: "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _case_cache_report(cache: CaseBaselineCache, tok: Any) -> Dict[str, Any]:
    return {
        **asdict(cache.case),
        "target_token_id": cache.target_token_id,
        "target_token": tok.decode([cache.target_token_id]),
        "predicted_token_id": cache.predicted_token_id,
        "target_logit": float(cache.target_logit.detach().cpu()),
        "best_other_token_id": cache.best_other_token_id,
        "best_other_token": tok.decode([cache.best_other_token_id]),
        "best_other_logit": float(cache.best_other_logit.detach().cpu()),
        "correct": cache.correct,
    }


def _validate_official_cache_alignment(
    caches: Sequence[CaseBaselineCache],
    identities: set[Tuple[int, str, int, int]],
    *,
    label: str,
) -> List[CaseBaselineCache]:
    selected = [cache for cache in caches if cache.case.identity in identities]
    observed = {cache.case.identity for cache in selected}
    if observed != identities or len(selected) != len(identities):
        missing = sorted(identities - observed)
        raise RuntimeError(f"Missing exact {label} caches: {missing[:10]}")
    misaligned = [cache.case.identity for cache in selected if not cache.correct]
    if misaligned:
        raise RuntimeError(
            f"Exact BF16 {label} cache disagrees with official correctness: "
            f"{misaligned[:10]}"
        )
    return selected


def _verify_sample_identity(
    source_results_path: Path,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
) -> None:
    sample_path = source_results_path.parent / "sampled_case_ids.json"
    if not sample_path.is_file():
        return
    source = _load_json(sample_path)
    expected_forget = [int(record["case_id"]) for record in forget_records]
    expected_retain = [int(record["case_id"]) for record in retain_records]
    if source.get("forget_case_ids") != expected_forget:
        raise RuntimeError("Source checkpoint forget sample does not match")
    if source.get("retain_case_ids") != expected_retain:
        raise RuntimeError("Source checkpoint retain sample does not match")


def _official_metrics_equal(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    return all(
        float(first[split][metric]) == float(second[split][metric])
        for split in ("forget", "retain")
        for metric in ("Eff", "Gen", "Spe")
    )


def execute_run(
    args: argparse.Namespace,
    *,
    interruption_state: InterruptionState,
) -> None:
    validate_args(args)
    gagd.set_seed(args.seed)
    output_dir = interruption_state.output_dir
    expected_output_dir = gagd.resolve_output_path(args.output_dir)
    if output_dir != expected_output_dir:
        raise ValueError("Interruption state output directory does not match arguments")
    interruption_state.phase = "preparing"
    _ensure_fresh_output(output_dir)
    optimization_dir = output_dir / "optimization"
    scale_sweep_dir = output_dir / "scale_sweep"
    optimization_dir.mkdir(parents=True, exist_ok=True)
    scale_sweep_dir.mkdir(parents=True, exist_ok=True)

    interruption_state.phase = "loading_setting5_checkpoint"
    model, tok, checkpoint = load_model_and_tokenizer(args)
    source_results_path, source_result = discover_source_results(
        checkpoint,
        args.source_results,
    )
    verify_source_result(
        source_result,
        seed=args.seed,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
    )
    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "protocol_status": PROTOCOL_STATUS,
            "source_setting5_checkpoint": str(checkpoint),
            "source_results": str(source_results_path),
            "setting5_steps_verified": SETTING5_STEPS,
            "stage1_rerun": False,
            "editable_parameters": "selected_sensitive_lm_head_rows_only",
            "unknown_row_edited": bool(args.edit_unknown_row),
            "fixed_gates": {
                "utility_drop_tolerance_percentage_points": FIXED_UTILITY_DROP_TOLERANCE,
                "max_ppl_ratio": FIXED_MAX_PPL_RATIO,
                "target_eff_max": FIXED_TARGET_EFF_MAX,
                "target_gen_max": FIXED_TARGET_GEN_MAX,
            },
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    zsre_path = zsre.download_zsre(
        gagd.resolve_output_path(args.zsre_path),
        url=args.zsre_url,
    )
    forget_records, retain_records = zsre.load_official_eval_records(
        zsre_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        zsre_url=args.zsre_url,
    )
    records = (forget_records, retain_records)
    _verify_sample_identity(
        source_results_path,
        forget_records,
        retain_records,
    )
    dataset_hash = zsre.file_sha256(zsre_path)
    source_hash = source_result.get("zsre_sha256")
    if source_hash is not None and str(source_hash) != dataset_hash:
        raise RuntimeError("Source checkpoint ZsRE hash does not match current data")
    gagd.write_json(
        output_dir / "sampled_case_ids.json",
        {
            "seed": args.seed,
            "sampling_protocol": (
                "forget sampled first from second half; retain sampled second "
                "from first half"
            ),
            "forget_case_ids": [record["case_id"] for record in forget_records],
            "retain_case_ids": [record["case_id"] for record in retain_records],
            "zsre_sha256": dataset_hash,
        },
    )

    interruption_state.phase = "evaluating_setting5_checkpoint"
    print("Evaluating the immutable Setting 5e checkpoint")
    setting5_result = zsre.evaluate_loaded_model_official(
        method="Setting 5e (600-step saved checkpoint)",
        model=model,
        tok=tok,
        model_dir=checkpoint,
        zsre_path=zsre_path,
        wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
        out_path=output_dir / "setting5e_official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=False,
        zsre_url=args.zsre_url,
        records=records,
    )
    if setting5_result.get("forget_PPL") is None:
        raise RuntimeError(
            "Setting 5e PPL is unresolved; exact PPL gating requires the official "
            "Wikipedia calibration corpus"
        )
    setting5_compact = compact_metrics(setting5_result)
    compare_compact_metrics(
        source_result["setting5e"],
        setting5_compact,
        label="Saved Setting 5e checkpoint",
    )

    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = zsre.resolve_neutral_target_token_id(tok)
    forget_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    retain_cases = [
        case
        for record in retain_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]
    active_identities = legacy_repair.official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("rewrite", "paraphrase"),
    )
    forget_protected_identities = legacy_repair.official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("neighborhood",),
    )
    retain_protected_identities = legacy_repair.official_correct_case_identities(
        retain_records,
        setting5_result["retain_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("rewrite", "paraphrase", "neighborhood"),
    )
    target_ids = target_ids_for_cases(
        tok,
        forget_cases,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
    )
    special_ids = all_special_token_ids(tok)
    excluded_ids = set(special_ids)
    if not args.edit_unknown_row:
        excluded_ids.add(neutral_token_id)
    selected_ids = sorted(
        {
            target_ids[identity]
            for identity in active_identities
            if target_ids[identity] not in excluded_ids
        }
    )
    excluded_active = [
        {
            "identity": identity_payload(identity),
            "token_id": target_ids[identity],
            "decoded_token": tok.decode([target_ids[identity]]),
            "reason": (
                "unknown_row_excluded"
                if target_ids[identity] == neutral_token_id
                else "tokenizer_special_row_excluded"
            ),
        }
        for identity in sorted(active_identities)
        if target_ids[identity] in excluded_ids
    ]
    selected_rows_payload = {
        "method": METHOD,
        "unknown_row_edited": bool(args.edit_unknown_row),
        "neutral_token_id": neutral_token_id,
        "special_token_ids": sorted(special_ids),
        "selected_row_count": len(selected_ids),
        "selected_rows": [
            {
                "token_id": token_id,
                "decoded_token": tok.decode([token_id]),
                "active_case_count": sum(
                    target_ids[identity] == token_id for identity in active_identities
                ),
            }
            for token_id in selected_ids
        ],
        "excluded_active_cases": excluded_active,
    }
    gagd.write_json(
        output_dir / "selected_sensitive_rows.json",
        selected_rows_payload,
    )

    output_layer = active.freeze_model_for_output_repair(model)
    input_embeddings = model.get_input_embeddings()
    if input_embeddings.weight.requires_grad:
        raise RuntimeError("Input embeddings are trainable during repair")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("The model contains trainable parameters during delta repair")
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=device)
    original_rows = (
        output_layer.weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_layer.weight.new_empty((0, output_layer.weight.shape[1]))
    )

    interruption_state.phase = "caching_constraints"
    print("Caching official-order forget and full-retain constraints")
    forget_caches = cache_case_baselines(
        model,
        tok,
        forget_cases,
        selected_ids=selected_ids,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache exact forget constraints",
    )
    retain_caches = cache_case_baselines(
        model,
        tok,
        retain_cases,
        selected_ids=selected_ids,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache exact full-retain constraints/KL",
    )
    active_caches = _validate_official_cache_alignment(
        forget_caches,
        active_identities,
        label="active rewrite/paraphrase",
    )
    forget_protected = _validate_official_cache_alignment(
        forget_caches,
        forget_protected_identities,
        label="forget-neighborhood protected",
    )
    retain_protected = _validate_official_cache_alignment(
        retain_caches,
        retain_protected_identities,
        label="full-retain protected",
    )
    protected_caches = forget_protected + retain_protected
    active_tensors = active_constraint_tensors(
        active_caches,
        selected_ids,
        device=device,
    )
    protected_tensors = build_protected_constraint_tensors(
        protected_caches,
        selected_ids,
        protected_margin_cap=args.protected_margin_cap,
        device=device,
    )
    retain_kl_tensors = build_retain_kl_caches(retain_caches)
    if retain_kl_tensors.record_count != len(retain_records):
        raise RuntimeError(
            "Full-retain KL cache does not cover every official retain record: "
            f"{retain_kl_tensors.record_count} != {len(retain_records)}"
        )
    zero_delta = torch.zeros(
        (len(selected_ids), output_layer.weight.shape[1]),
        dtype=torch.float32,
        device=device,
    )
    active_before = active_constraint_margins(
        active_tensors.hidden,
        active_tensors.sensitive_logits,
        active_tensors.best_other_logits,
        active_tensors.selected_row_columns,
        zero_delta,
    )
    protected_before = protected_constraint_margins(
        protected_tensors.hidden,
        protected_tensors.target_logits,
        protected_tensors.strongest_unchanged_logits,
        protected_tensors.selected_logits,
        protected_tensors.target_selected_columns,
        zero_delta,
    )

    print(
        "Optimizing sensitive rows jointly: "
        f"rows={len(selected_ids)}, active={len(active_caches)}, "
        f"protected={len(protected_caches)}, retain_KL_records={len(retain_records)}",
        flush=True,
    )
    delta_rows, repair_logs, optimization_summary = optimize_gate_aware_delta(
        active_tensors,
        protected_tensors,
        retain_kl_tensors,
        selected_row_count=len(selected_ids),
        hidden_size=output_layer.weight.shape[1],
        args=args,
        device=device,
        live_progress_path=optimization_dir / "live_progress.jsonl",
        interruption_state=interruption_state,
    )
    interruption_state.phase = "candidate_evaluation"
    write_jsonl(optimization_dir / "repair_log.jsonl", repair_logs)
    active_after = active_constraint_margins(
        active_tensors.hidden,
        active_tensors.sensitive_logits,
        active_tensors.best_other_logits,
        active_tensors.selected_row_columns,
        delta_rows,
    )
    protected_after = protected_constraint_margins(
        protected_tensors.hidden,
        protected_tensors.target_logits,
        protected_tensors.strongest_unchanged_logits,
        protected_tensors.selected_logits,
        protected_tensors.target_selected_columns,
        delta_rows,
    )
    constraint_summary = {
        "active_constraint_count": len(active_caches),
        "protected_constraint_count": len(protected_caches),
        "forget_neighborhood_protected_count": len(forget_protected),
        "full_retain_protected_count": len(retain_protected),
        "retain_kl_record_count": len(retain_records),
        "retain_kl_token_count": retain_kl_tensors.token_count,
        "active_violations_before": int(
            (active_before < args.active_margin).sum().item()
        ),
        "active_violations_after_unscaled": int(
            (active_after < args.active_margin).sum().item()
        ),
        "protected_violations_before": int(
            (protected_before < protected_tensors.required_margins).sum().item()
        ),
        "protected_violations_after_unscaled": int(
            (protected_after < protected_tensors.required_margins).sum().item()
        ),
        "minimum_active_margin_before": _minimum(active_before),
        "minimum_active_margin_after_unscaled": _minimum(active_after),
        "minimum_protected_margin_before": _minimum(protected_before),
        "minimum_protected_margin_after_unscaled": _minimum(protected_after),
        "optimization": optimization_summary,
    }
    gagd.write_json(
        optimization_dir / "constraint_summary.json",
        constraint_summary,
    )
    write_jsonl(
        optimization_dir / "active_cases.jsonl",
        [_case_cache_report(cache, tok) for cache in active_caches],
    )

    # The unscaled FP32 optimizer result is the named active candidate.
    materialize_sensitive_rows(
        output_layer.weight,
        selected_ids,
        original_rows,
        delta_rows,
        1.0,
    )
    active_candidate_checkpoint = output_dir / "active_candidate_checkpoint"
    legacy_repair.save_checkpoint(model, tok, active_candidate_checkpoint)

    scales = candidate_scales(args.candidate_scale_step)
    non_ppl_rows: List[Dict[str, Any]] = []
    non_ppl_results: Dict[float, Dict[str, Any]] = {}
    for scale in scales:
        norm = materialize_sensitive_rows(
            output_layer.weight,
            selected_ids,
            original_rows,
            delta_rows,
            scale,
        )
        result = zsre.evaluate_loaded_model_official(
            method=f"Gate-aware sensitive-row candidate scale={scale:g}",
            model=model,
            tok=tok,
            model_dir=f"in-memory:gate-aware-scale-{scale:g}",
            zsre_path=zsre_path,
            wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
            out_path=None,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            batch_size=args.eval_batch_size,
            skip_ppl=True,
            zsre_url=args.zsre_url,
            records=records,
        )
        if scale == 0.0 and not _official_metrics_equal(setting5_result, result):
            raise RuntimeError(
                "Exact BF16 scale-0 metrics do not reproduce the immutable "
                "Setting 5e evaluation"
            )
        gate = non_ppl_gate_report(setting5_result, result)
        non_ppl_results[scale] = result
        non_ppl_rows.append(
            {
                "scale": scale,
                "materialized_delta_norm": norm,
                "metrics": compact_metrics(result),
                "non_ppl_gate": gate,
                "survived_non_ppl_gates": bool(gate["passed"]),
            }
        )
    gagd.write_json(
        scale_sweep_dir / "non_ppl_gate_sweep.json",
        non_ppl_rows,
    )

    full_rows: List[Dict[str, Any]] = []
    full_results: Dict[float, Dict[str, Any]] = {}
    for row in non_ppl_rows:
        if not row["survived_non_ppl_gates"]:
            continue
        scale = float(row["scale"])
        materialize_sensitive_rows(
            output_layer.weight,
            selected_ids,
            original_rows,
            delta_rows,
            scale,
        )
        full_result = zsre.evaluate_loaded_model_official(
            method=f"Gate-aware sensitive-row full-gate scale={scale:g}",
            model=model,
            tok=tok,
            model_dir=f"in-memory:gate-aware-full-scale-{scale:g}",
            zsre_path=zsre_path,
            wikidata_dir=gagd.resolve_output_path(args.wikidata_dir),
            out_path=None,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            batch_size=args.eval_batch_size,
            skip_ppl=False,
            zsre_url=args.zsre_url,
            records=records,
        )
        if not _official_metrics_equal(non_ppl_results[scale], full_result):
            raise RuntimeError(
                f"Non-PPL and full evaluation metrics disagree at scale {scale:g}"
            )
        gate = full_gate_report(setting5_result, full_result)
        full_results[scale] = full_result
        full_rows.append(
            {
                "scale": scale,
                "materialized_delta_norm": row["materialized_delta_norm"],
                "metrics": compact_metrics(full_result),
                "full_gate": gate,
            }
        )
    gagd.write_json(scale_sweep_dir / "full_gate_sweep.json", full_rows)

    selected_gate_row = select_all_gates_candidate(
        full_rows,
        setting5_target_already_met=setting5_already_meets_target(setting5_result),
    )
    selected_scale: Optional[float] = None
    selected_result: Optional[Dict[str, Any]] = None
    selected_checkpoint: Optional[Path] = None
    selected_checkpoint_hash: Optional[str] = None
    if selected_gate_row is not None:
        selected_scale = float(selected_gate_row["scale"])
        materialize_sensitive_rows(
            output_layer.weight,
            selected_ids,
            original_rows,
            delta_rows,
            selected_scale,
        )
        selected_result = copy.deepcopy(full_results[selected_scale])
        selected_result["method"] = "Gate-aware sensitive-row LM-head repair"
    else:
        materialize_sensitive_rows(
            output_layer.weight,
            selected_ids,
            original_rows,
            delta_rows,
            0.0,
        )

    if selected_ids:
        exact_materialized_delta = (
            output_layer.weight.index_select(0, selected_tensor).detach().float()
            - original_rows.detach().float()
        )
    else:
        exact_materialized_delta = zero_delta
    # Mandatory full-set diagnostics are evaluated after exact BF16
    # materialization and before the gate-passing candidate is accepted/saved.
    interruption_state.phase = "final_acceptance_validation"
    final_constraint_check = full_constraint_check_chunked(
        active_tensors,
        protected_tensors,
        exact_materialized_delta,
        active_margin=args.active_margin,
        chunk_size=args.protected_batch_size,
    )
    final_retain_kl = retain_kl_from_tensors(
        retain_kl_tensors,
        exact_materialized_delta,
    )
    accepted = selected_gate_row is not None
    if accepted:
        if selected_result is None:
            raise RuntimeError("Gate-selected candidate lacks official metrics")
        interruption_state.phase = "saving_selected_checkpoint"
        gagd.write_json(output_dir / "selected_official_eval.json", selected_result)
        selected_checkpoint = save_selected_checkpoint_if_accepted(
            accepted=True,
            model=model,
            tok=tok,
            output_dir=output_dir,
        )
        if selected_checkpoint is None:
            raise RuntimeError("Accepted candidate did not produce a checkpoint")
        selected_checkpoint_hash = directory_sha256(selected_checkpoint)
        selection_reason = "smallest_materialized_delta_norm_passing_every_gate"
    else:
        selection_reason = "no_materialized_candidate_passed_every_fixed_gate"
    interruption_state.phase = "finalizing_results"
    constraint_summary.update(
        {
            "final_materialized_scale": selected_scale if accepted else None,
            "final_materialized_delta_norm": float(
                exact_materialized_delta.norm().detach().cpu()
            ),
            "active_violations_after_final_materialization": (
                final_constraint_check["full_active_violation_count"]
            ),
            "protected_violations_after_final_materialization": (
                final_constraint_check["full_protected_violation_count"]
            ),
            "minimum_active_slack_after_final_materialization": (
                final_constraint_check["minimum_active_slack"]
            ),
            "minimum_protected_slack_after_final_materialization": (
                final_constraint_check["minimum_protected_slack"]
            ),
            "minimum_active_margin_after_final_materialization": (
                final_constraint_check["minimum_active_margin"]
            ),
            "minimum_protected_margin_after_final_materialization": (
                final_constraint_check["minimum_protected_margin"]
            ),
            "final_constraint_check_chunks": {
                "active": final_constraint_check["active_chunks_evaluated"],
                "protected": final_constraint_check[
                    "protected_chunks_evaluated"
                ],
                "chunk_size": final_constraint_check["chunk_size"],
            },
            "full_retain_kl_after_final_materialization": float(
                final_retain_kl.detach().cpu()
            ),
            "full_retain_records_checked_before_acceptance": (
                retain_kl_tensors.record_count
            ),
        }
    )
    gagd.write_json(
        optimization_dir / "constraint_summary.json",
        constraint_summary,
    )

    scale_one_result = non_ppl_results[1.0]
    if 1.0 in full_results:
        scale_one_result = full_results[1.0]
    official_metric_gates: Dict[str, Any]
    if selected_gate_row is None:
        official_metric_gates = {
            "passed": False,
            "reason": selection_reason,
            "full_candidate_count": len(full_rows),
            "fixed_thresholds": config["fixed_gates"],
        }
    else:
        official_metric_gates = copy.deepcopy(selected_gate_row["full_gate"])

    repair_summary = {
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "protocol_status_reason": (
            "Official ZsRE correctness defines active/protected cases and native "
            "metrics select the exact BF16 candidate scale."
        ),
        "candidate_accepted": accepted,
        "selected_scale": selected_scale,
        "selection_reason": selection_reason,
        "official_metric_gates": official_metric_gates,
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "active_constraint_count": len(active_caches),
        "protected_constraint_count": len(protected_caches),
        "retain_kl_record_count": len(retain_records),
        "input_embeddings_frozen_during_repair": True,
        "transformer_frozen_during_repair": True,
        "unknown_row_edited": bool(args.edit_unknown_row),
        "selected_checkpoint_sha256": selected_checkpoint_hash,
        "selected_checkpoint": (
            None if selected_checkpoint is None else str(selected_checkpoint)
        ),
        "active_candidate_checkpoint": str(active_candidate_checkpoint),
        "optimization": optimization_summary,
    }
    base_block = copy.deepcopy(source_result["base"])
    final_result = {
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "dataset": "ZsRE",
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "zsre_sha256": dataset_hash,
        "base": base_block,
        "setting5e": setting5_compact,
        "active_candidate": compact_metrics(scale_one_result),
        "selected": (
            None if selected_result is None else compact_metrics(selected_result)
        ),
        "repair": repair_summary,
    }
    gagd.write_json(output_dir / "zsre_results.json", final_result)
    print(
        f"Gate-aware ZsRE seed {args.seed}: accepted={accepted}; "
        f"selected_scale={selected_scale}; selected_checkpoint_sha256="
        f"{selected_checkpoint_hash}"
    )
    interruption_state.phase = "complete"
    interruption_state.run_completed = True
    if args.fail_if_target_missed and not accepted:
        raise RuntimeError(
            "No gate-aware sensitive-row candidate passed zero Eff/Gen, all "
            "0.10-point utility gates, and the 1.02 PPL gate. Diagnostics were "
            f"written to {output_dir}."
        )


def main() -> None:
    args = build_parser().parse_args()
    state = InterruptionState(
        output_dir=gagd.resolve_output_path(args.output_dir),
    )
    execute_with_interrupt_receipt(
        lambda: execute_run(args, interruption_state=state),
        state,
    )


if __name__ == "__main__":
    main()
