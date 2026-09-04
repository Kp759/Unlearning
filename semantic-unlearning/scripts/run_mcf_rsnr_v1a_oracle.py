#!/usr/bin/env python3
"""RSNR-V1A: oracle (subject, relation) routing with a latent null adapter.

Development-only MCF experiment. The Base model is frozen. For an oracle-known
forget (subject, relation) query, a small residual null adapter is activated at
one decoder layer. For every non-target query the adapter is strictly off, so
the computation follows the Base path exactly.

The sensitive branch optimizes two complementary objectives over the existing
leakage-safe V1.3 five-view corpus:
  * natural abstention: increase P("I don't know.")
  * true-object unlikelihood: directly suppress target_true

No target_new objective is used. No official paraphrase/neighborhood prompt is
accepted by this runner. This is relation-scoped behavioral suppression, not a
claim of latent knowledge deletion; disabling the oracle intervention recovers
the frozen Base model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_mcf_private_vocab_rewiring_v1 as base_runner
import run_mcf_private_vocab_rewiring_v1_3_multiview as multiview

PROTOCOL = "mcf_rsnr_v1a_oracle_null_adapter"
ABSTENTION = "I don't know."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--view-corpus", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--case-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adapter-rank", type=int, default=16)
    p.add_argument("--adapter-alpha", type=float, default=16.0)
    p.add_argument("--layer-index", type=int, default=-4,
                   help="Decoder layer index; negative values count from the end.")
    p.add_argument("--abstain-weight", type=float, default=1.0)
    p.add_argument("--unlikelihood-weight", type=float, default=1.0)
    p.add_argument("--anchor-weight", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--minimum-abstain-vs-true-margin", type=float, default=0.1)
    p.add_argument("--minimum-true-logprob-drop", type=float, default=2.0)
    p.add_argument("--gate-off-logit-drift-max", type=float, default=0.0)
    p.add_argument("--save-base-copy", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("RSNR-V1A is development-only and locked to consumed seed 1 / forget50")
    if args.steps <= 0 or args.case_batch_size <= 0 or args.check_every <= 0:
        p.error("step/batch/check values must be positive")
    if args.adapter_rank <= 0 or args.adapter_alpha <= 0:
        p.error("adapter rank/alpha must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        p.error("invalid optimizer configuration")
    if args.abstain_weight < 0 or args.unlikelihood_weight < 0 or args.anchor_weight < 0:
        p.error("loss weights must be non-negative")
    if args.abstain_weight == 0 and args.unlikelihood_weight == 0:
        p.error("at least one sensitive objective must be active")
    return args


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_decoder_layers(model: Any) -> Sequence[nn.Module]:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is not None and len(layers) > 0:
            return layers
    raise RuntimeError("could not locate decoder layers")


def resolve_layer_index(requested: int, count: int) -> int:
    index = int(requested)
    if index < 0:
        index = count + index
    if index < 0 or index >= count:
        raise ValueError(f"layer index {requested} resolves outside 0..{count - 1}")
    return index


class NullResidualAdapter(nn.Module):
    """Small residual bottleneck initialized as an exact no-op."""

    def __init__(self, hidden_size: int, rank: int, alpha: float, device: torch.device):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.down = nn.Linear(hidden_size, rank, bias=False, device=device, dtype=torch.float32)
        self.up = nn.Linear(rank, hidden_size, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(F.linear(hidden.float(), self.down.weight))
        delta = F.linear(z, self.up.weight).mul(self.scaling)
        return delta.to(dtype=hidden.dtype)


@dataclass
class OracleNullHook:
    adapter: NullResidualAdapter
    handle: Any
    gate_mask: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None

    @classmethod
    def install(cls, layer: nn.Module, adapter: NullResidualAdapter) -> "OracleNullHook":
        state = cls(adapter=adapter, handle=None)

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if state.gate_mask is None:
                return output
            if isinstance(output, tuple):
                hidden = output[0]
                tail = output[1:]
            else:
                hidden = output
                tail = None
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                return output
            gate = state.gate_mask.to(device=hidden.device, dtype=hidden.dtype).view(-1, 1, 1)
            if gate.shape[0] != hidden.shape[0]:
                raise RuntimeError("oracle gate batch size does not match decoder hidden batch")
            if state.position_mask is not None:
                pos = state.position_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
                if tuple(pos.shape[:2]) != tuple(hidden.shape[:2]):
                    raise RuntimeError("oracle position mask does not match hidden-state shape")
                gate = gate * pos
            if not bool(torch.any(gate != 0).item()):
                return output
            edited = hidden + gate * state.adapter(hidden)
            if tail is None:
                return edited
            return (edited, *tail)

        state.handle = layer.register_forward_hook(hook)
        return state

    def set(self, gate_mask: torch.Tensor | None, position_mask: torch.Tensor | None = None) -> None:
        self.gate_mask = gate_mask
        self.position_mask = position_mask

    def clear(self) -> None:
        self.gate_mask = None
        self.position_mask = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def load_protocol(protocol_dir: Path, forget_num: int) -> Dict[str, list[Dict[str, Any]]]:
    return base_runner.load_protocol(protocol_dir, forget_num)


def load_training_views(path: Path) -> tuple[Dict[int, list[str]], Dict[str, Any]]:
    view_map, meta = multiview.load_view_corpus(path)
    if int(meta["views_per_case"]) != 5:
        raise RuntimeError(f"RSNR-V1A requires the locked 5-view corpus, got {meta['views_per_case']}")
    return view_map, meta


def validate_case_alignment(forget: Sequence[Mapping[str, Any]], view_map: Mapping[int, Sequence[str]]) -> None:
    direct_ids = {int(row["case_id"]) for row in forget}
    view_ids = set(int(k) for k in view_map)
    if direct_ids != view_ids:
        raise RuntimeError(
            f"forget/view case mismatch: missing={sorted(direct_ids-view_ids)[:10]}, "
            f"extra={sorted(view_ids-direct_ids)[:10]}"
        )


def fact_key(record: Mapping[str, Any]) -> tuple[str, str]:
    rr = record["requested_rewrite"]
    return str(rr["subject"]), str(rr["relation_id"])


def oracle_membership(record: Mapping[str, Any], forget_pairs: set[tuple[str, str]]) -> bool:
    return fact_key(record) in forget_pairs


def build_oracle_negative_audit(
    forget: Sequence[Mapping[str, Any]], protection_fit: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    forget_pairs = {fact_key(row) for row in forget}
    forget_subjects = {s for s, _ in forget_pairs}
    forget_relations = {r for _, r in forget_pairs}
    same_subject_other_relation = []
    same_relation_other_subject = []
    for row in protection_fit:
        key = fact_key(row)
        if key in forget_pairs:
            continue
        subject, relation = key
        if subject in forget_subjects:
            same_subject_other_relation.append(int(row["case_id"]))
        if relation in forget_relations:
            same_relation_other_subject.append(int(row["case_id"]))
    return {
        "forget_pairs": len(forget_pairs),
        "same_subject_different_relation_negatives": len(same_subject_other_relation),
        "same_relation_different_subject_negatives": len(same_relation_other_subject),
        "generic_subject_mentions_without_relation": 4 * len(forget_subjects),
        "oracle_false_positive_rate_by_construction": 0.0,
        "atomic_query_scope": True,
    }


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    return base_runner.answer_ids(tokenizer, answer)


def encode_prompt_answer_batch(
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[list[int]], torch.Tensor]:
    rows: list[list[int]] = []
    starts: list[int] = []
    targets: list[list[int]] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for prompt, answer in zip(prompts, answers):
        pids = tokenizer(prompt, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        pids = [int(v) for v in pids]
        if bos is not None:
            pids = [int(bos)] + pids
        aids = answer_ids(tokenizer, answer)
        rows.append(pids + aids)
        starts.append(len(pids))
        targets.append(aids)
    max_len = max(len(row) for row in rows)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", 0)
    input_ids = torch.full((len(rows), max_len), int(pad), device=device, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    positions = torch.zeros_like(input_ids, dtype=torch.float32)
    for i, (row, start, aids) in enumerate(zip(rows, starts, targets)):
        input_ids[i, : len(row)] = torch.tensor(row, device=device, dtype=torch.long)
        attention[i, : len(row)] = 1
        for pos in range(start - 1, start - 1 + len(aids)):
            positions[i, pos] = 1.0
    return input_ids, attention, starts, targets, positions


def sequence_logprobs(
    model: Any,
    hook: OracleNullHook,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    device: torch.device,
    gated: bool,
) -> torch.Tensor:
    input_ids, attention, starts, targets, positions = encode_prompt_answer_batch(
        tokenizer, prompts, answers, device=device
    )
    gate = torch.ones(len(prompts), device=device) if gated else torch.zeros(len(prompts), device=device)
    hook.set(gate, positions if gated else None)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    finally:
        hook.clear()
    log_probs = F.log_softmax(logits, dim=-1)
    values = []
    for i, (start, aids) in enumerate(zip(starts, targets)):
        pos = torch.arange(start - 1, start - 1 + len(aids), device=device, dtype=torch.long)
        tok = torch.tensor(aids, device=device, dtype=torch.long)
        values.append(log_probs[i, pos, tok].mean())
    return torch.stack(values)


def sequence_unlikelihood(
    model: Any,
    hook: OracleNullHook,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention, starts, targets, positions = encode_prompt_answer_batch(
        tokenizer, prompts, answers, device=device
    )
    hook.set(torch.ones(len(prompts), device=device), positions)
    try:
        logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    finally:
        hook.clear()
    log_probs = F.log_softmax(logits, dim=-1)
    losses = []
    eps = 1e-6
    for i, (start, aids) in enumerate(zip(starts, targets)):
        pos = torch.arange(start - 1, start - 1 + len(aids), device=device, dtype=torch.long)
        tok = torch.tensor(aids, device=device, dtype=torch.long)
        p = log_probs[i, pos, tok].exp().clamp(max=1.0 - eps)
        losses.append((-torch.log1p(-p)).mean())
    return torch.stack(losses)


def prompts_for_cases(
    cases: Sequence[Mapping[str, Any]], view_map: Mapping[int, Sequence[str]]
) -> tuple[list[str], list[str], list[int]]:
    prompts: list[str] = []
    true_answers: list[str] = []
    owners: list[int] = []
    for local_index, row in enumerate(cases):
        rr = row["requested_rewrite"]
        subject = str(rr["subject"])
        true = str(rr["target_true"]["str"])
        for template in view_map[int(row["case_id"])]:
            prompts.append(str(template).format(subject))
            true_answers.append(true)
            owners.append(local_index)
    return prompts, true_answers, owners


def worst_by_owner(values: torch.Tensor, owners: Sequence[int], count: int, *, maximum: bool) -> torch.Tensor:
    out = []
    for i in range(int(count)):
        idx = torch.tensor([j for j, owner in enumerate(owners) if owner == i], device=values.device)
        selected = values.index_select(0, idx)
        out.append(selected.max() if maximum else selected.min())
    return torch.stack(out)


def base_true_logprobs_for_all_views(
    model: Any,
    hook: OracleNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    *,
    device: torch.device,
    batch_cases: int,
) -> Dict[int, list[float]]:
    out: Dict[int, list[float]] = {}
    with torch.no_grad():
        for start in range(0, len(forget), batch_cases):
            cases = forget[start : start + batch_cases]
            prompts, answers, owners = prompts_for_cases(cases, view_map)
            values = sequence_logprobs(model, hook, tokenizer, prompts, answers,
                                       device=device, gated=False)
            for local, row in enumerate(cases):
                out[int(row["case_id"])] = [
                    float(values[j].item()) for j, owner in enumerate(owners) if owner == local
                ]
    return out


def evaluate_sensitive(
    model: Any,
    hook: OracleNullHook,
    tokenizer: Any,
    forget: Sequence[Mapping[str, Any]],
    view_map: Mapping[int, Sequence[str]],
    base_true: Mapping[int, Sequence[float]],
    *,
    device: torch.device,
    batch_cases: int,
    margin_threshold: float,
    drop_threshold: float,
) -> Dict[str, Any]:
    rows = []
    with torch.no_grad():
        for start in range(0, len(forget), batch_cases):
            cases = forget[start : start + batch_cases]
            prompts, true_answers, owners = prompts_for_cases(cases, view_map)
            abstain_answers = [ABSTENTION] * len(prompts)
            true_lp = sequence_logprobs(model, hook, tokenizer, prompts, true_answers,
                                        device=device, gated=True)
            abstain_lp = sequence_logprobs(model, hook, tokenizer, prompts, abstain_answers,
                                           device=device, gated=True)
            for local, row in enumerate(cases):
                idxs = [j for j, owner in enumerate(owners) if owner == local]
                margins = [float((abstain_lp[j] - true_lp[j]).item()) for j in idxs]
                drops = [
                    float(base_true[int(row["case_id"])][k] - true_lp[j].item())
                    for k, j in enumerate(idxs)
                ]
                worst_margin = min(margins)
                worst_drop = min(drops)
                rows.append({
                    "case_id": int(row["case_id"]),
                    "subject": str(row["requested_rewrite"]["subject"]),
                    "relation_id": str(row["requested_rewrite"]["relation_id"]),
                    "worst_abstain_vs_true_margin": worst_margin,
                    "worst_true_logprob_drop": worst_drop,
                    "margin_pass": worst_margin >= float(margin_threshold),
                    "suppression_pass": worst_drop >= float(drop_threshold),
                    "joint_pass": worst_margin >= float(margin_threshold)
                                  and worst_drop >= float(drop_threshold),
                })
    return {
        "count": len(rows),
        "joint_passed": sum(bool(r["joint_pass"]) for r in rows),
        "joint_failures": sum(not bool(r["joint_pass"]) for r in rows),
        "margin_passed": sum(bool(r["margin_pass"]) for r in rows),
        "suppression_passed": sum(bool(r["suppression_pass"]) for r in rows),
        "minimum_worst_abstain_vs_true_margin": min(r["worst_abstain_vs_true_margin"] for r in rows),
        "minimum_worst_true_logprob_drop": min(r["worst_true_logprob_drop"] for r in rows),
        "margin_threshold": float(margin_threshold),
        "true_drop_threshold": float(drop_threshold),
        "per_case": rows,
    }


def gate_off_equivalence(
    model: Any,
    hook: OracleNullHook,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
) -> Dict[str, float]:
    if not texts:
        return {"contexts": 0, "max_abs_logit_drift": 0.0}
    maximum = 0.0
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hook.clear()
            base_logits = model(**encoded).logits.float()
            hook.set(torch.zeros(base_logits.shape[0], device=device), None)
            try:
                off_logits = model(**encoded).logits.float()
            finally:
                hook.clear()
            maximum = max(maximum, float((base_logits - off_logits).abs().max().item()))
    return {"contexts": len(texts), "max_abs_logit_drift": maximum}


def adapter_norm(adapter: NullResidualAdapter) -> Dict[str, float]:
    with torch.no_grad():
        return {
            "down_fro": float(adapter.down.weight.float().norm().item()),
            "up_fro": float(adapter.up.weight.float().norm().item()),
        }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    method_dir = output / "method"
    method_dir.mkdir()

    protocol = load_protocol(Path(args.protocol_dir), int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]
    view_path = Path(args.view_corpus).resolve()
    view_map, view_meta = load_training_views(view_path)
    validate_case_alignment(forget, view_map)
    oracle_audit = build_oracle_negative_audit(forget, protection_fit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    dtype = base_runner.dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = get_decoder_layers(model)
    layer_index = resolve_layer_index(int(args.layer_index), len(layers))
    hidden_size = int(getattr(model.config, "hidden_size"))
    adapter = NullResidualAdapter(hidden_size, int(args.adapter_rank),
                                  float(args.adapter_alpha), device).to(device)
    hook = OracleNullHook.install(layers[layer_index], adapter)

    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(json.dumps({
        "protocol": PROTOCOL,
        "oracle_gate": "exact (subject, relation) membership supplied by experiment metadata",
        "sensitive_action": "activate latent null residual adapter",
        "non_sensitive_action": "adapter off; exact Base path",
        "layer_index": layer_index,
        "decoder_layers": len(layers),
        "adapter_rank": int(args.adapter_rank),
        "trainable_adapter_parameters": trainable,
        "frozen_base_parameters": total,
        "views_per_case": int(view_meta["views_per_case"]),
        "abstention": ABSTENTION,
        "target_new_used": False,
        "atomic_query_scope": True,
        "heldout_probe_text_used": False,
    }, indent=2), flush=True)

    generic_contexts = []
    for row in forget:
        subject = str(row["requested_rewrite"]["subject"])
        generic_contexts.extend([
            subject,
            f"Tell me about {subject}.",
            f"Information about {subject}:",
            f"The following statement concerns {subject}.",
        ])
    forget_pairs = {fact_key(row) for row in forget}
    forget_subjects = {s for s, _ in forget_pairs}
    forget_relations = {r for _, r in forget_pairs}
    negative_contexts = []
    for row in protection_fit:
        if fact_key(row) in forget_pairs:
            continue
        subject, relation = fact_key(row)
        if subject in forget_subjects or relation in forget_relations:
            negative_contexts.append(base_runner.render_prompt(row))
        if len(negative_contexts) >= 128:
            break
    equivalence_contexts = list(dict.fromkeys(generic_contexts + negative_contexts))[:192]
    equivalence = gate_off_equivalence(model, hook, tokenizer, equivalence_contexts,
                                       device=device)
    if equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError(
            f"gate-off path is not Base-identical: {equivalence['max_abs_logit_drift']} > "
            f"{args.gate_off_logit_drift_max}"
        )

    base_true = base_true_logprobs_for_all_views(
        model, hook, tokenizer, forget, view_map, device=device,
        batch_cases=int(args.case_batch_size)
    )

    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay)
    )
    rng = random.Random(int(args.seed) + 44017)
    best_state = copy.deepcopy(adapter.state_dict())
    best_key = (10**9, float("inf"), float("inf"))
    training_log = []

    for step in range(1, int(args.steps) + 1):
        cases = rng.sample(forget, min(int(args.case_batch_size), len(forget)))
        prompts, true_answers, owners = prompts_for_cases(cases, view_map)
        abstain_answers = [ABSTENTION] * len(prompts)
        optimizer.zero_grad(set_to_none=True)

        abstain_lp = sequence_logprobs(
            model, hook, tokenizer, prompts, abstain_answers,
            device=device, gated=True
        )
        per_case_abstain = worst_by_owner(-abstain_lp, owners, len(cases), maximum=True)
        abstain_loss = per_case_abstain.mean()

        unlikelihood = sequence_unlikelihood(
            model, hook, tokenizer, prompts, true_answers, device=device
        )
        per_case_unlikelihood = worst_by_owner(
            unlikelihood, owners, len(cases), maximum=True
        )
        unlikelihood_loss = per_case_unlikelihood.mean()

        anchor = adapter.up.weight.float().pow(2).mean()
        loss = (
            float(args.abstain_weight) * abstain_loss
            + float(args.unlikelihood_weight) * unlikelihood_loss
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(adapter.parameters()), float(args.grad_clip))
        optimizer.step()

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            metrics = evaluate_sensitive(
                model, hook, tokenizer, forget, view_map, base_true,
                device=device, batch_cases=int(args.case_batch_size),
                margin_threshold=float(args.minimum_abstain_vs_true_margin),
                drop_threshold=float(args.minimum_true_logprob_drop),
            )
            eq = gate_off_equivalence(
                model, hook, tokenizer, equivalence_contexts[:32], device=device
            )
            key = (
                int(metrics["joint_failures"]),
                -float(metrics["minimum_worst_abstain_vs_true_margin"]),
                -float(metrics["minimum_worst_true_logprob_drop"]),
            )
            if key < best_key:
                best_key = key
                best_state = copy.deepcopy(adapter.state_dict())
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "abstain_loss": float(abstain_loss.detach().item()),
                "unlikelihood_loss": float(unlikelihood_loss.detach().item()),
                "anchor": float(anchor.detach().item()),
                "joint_passed": int(metrics["joint_passed"]),
                "joint_failures": int(metrics["joint_failures"]),
                "minimum_worst_abstain_vs_true_margin": float(metrics["minimum_worst_abstain_vs_true_margin"]),
                "minimum_worst_true_logprob_drop": float(metrics["minimum_worst_true_logprob_drop"]),
                "gate_off_max_abs_logit_drift": float(eq["max_abs_logit_drift"]),
                **adapter_norm(adapter),
            }
            training_log.append(row)
            print(
                f"step {step:4d}: joint pass={metrics['joint_passed']}/50, "
                f"worst abstain-true={metrics['minimum_worst_abstain_vs_true_margin']:.4f}, "
                f"worst true-drop={metrics['minimum_worst_true_logprob_drop']:.4f}, "
                f"gate-off drift={eq['max_abs_logit_drift']:.3g}",
                flush=True,
            )
            if metrics["joint_failures"] == 0:
                print("all 50 cases pass all 5 training views; stopping early", flush=True)
                break

    adapter.load_state_dict(best_state)
    final_metrics = evaluate_sensitive(
        model, hook, tokenizer, forget, view_map, base_true,
        device=device, batch_cases=int(args.case_batch_size),
        margin_threshold=float(args.minimum_abstain_vs_true_margin),
        drop_threshold=float(args.minimum_true_logprob_drop),
    )
    final_equivalence = gate_off_equivalence(
        model, hook, tokenizer, equivalence_contexts, device=device
    )
    if final_equivalence["max_abs_logit_drift"] > float(args.gate_off_logit_drift_max):
        raise RuntimeError("final gate-off Base equivalence failed")

    torch.save({
        "protocol": PROTOCOL,
        "base_model": str(args.model_path),
        "layer_index": layer_index,
        "hidden_size": hidden_size,
        "adapter_rank": int(args.adapter_rank),
        "adapter_alpha": float(args.adapter_alpha),
        "adapter_state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "abstention": ABSTENTION,
        "forget_membership": [
            {"case_id": int(row["case_id"]), "subject": fact_key(row)[0],
             "relation_id": fact_key(row)[1]}
            for row in forget
        ],
    }, method_dir / "rsnr_oracle_null_adapter.pt")

    sidecar = {
        "protocol": PROTOCOL,
        "routing": "oracle_exact_subject_relation_membership",
        "atomic_query_scope": True,
        "non_target_behavior": "adapter_off_exact_base_path",
        "sensitive_behavior": "activate_latent_null_adapter",
        "abstention_text": ABSTENTION,
        "target_new_used": False,
        "forget_membership": [
            {"case_id": int(row["case_id"]), "subject": fact_key(row)[0],
             "relation_id": fact_key(row)[1]}
            for row in forget
        ],
    }
    (method_dir / "relation_scoped_null_routing.json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report = {
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "development_only": True,
        "architecture": {
            "name": "Relation-Scoped Null Routing (RSNR)",
            "variant": "V1A oracle gate",
            "base_model_frozen": True,
            "lm_head_frozen": True,
            "input_embeddings_frozen": True,
            "oracle_gate": "exact (subject, relation) forget-membership",
            "intervention": "low-rank latent null residual adapter",
            "intervention_layer": layer_index,
            "adapter_rank": int(args.adapter_rank),
            "adapter_alpha": float(args.adapter_alpha),
            "trainable_parameters": trainable,
            "non_target_path": "exact Base path; adapter disabled",
            "atomic_factual_queries_only": True,
        },
        "objective": {
            "abstention_text": ABSTENTION,
            "abstention_weight": float(args.abstain_weight),
            "true_object_unlikelihood_weight": float(args.unlikelihood_weight),
            "adapter_anchor_weight": float(args.anchor_weight),
            "target_new_used": False,
            "worst_of_5_training_views": True,
            "true_aliases_used_for_training": False,
            "true_aliases_note": "sanitized direct protocol exposes target_true string only; aliases are reserved for evaluation if available",
        },
        "training_view_corpus": {
            **view_meta,
            "path": str(view_path),
            "heldout_probe_text_used": False,
            "official_paraphrase_text_used": False,
            "official_neighborhood_text_used": False,
        },
        "oracle_gate_audit": oracle_audit,
        "gate_off_equivalence": final_equivalence,
        "final_training_view_metrics": final_metrics,
        "adapter_norm": adapter_norm(adapter),
        "training_log": training_log,
        "claim_boundary": {
            "relation_scoped_behavioral_unlearning": True,
            "conditional_knowledge_suppression": True,
            "latent_knowledge_erasure_claimed": False,
            "oracle_gate_is_not_learned": True,
            "disabling_intervention_recovers_base": True,
            "external_refusal_only": False,
        },
    }
    (method_dir / "rsnr_v1a_oracle.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.save_base_copy:
        model_dir = output / "base_model_copy"
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

    completion = {
        "protocol": PROTOCOL,
        "joint_passed": int(final_metrics["joint_passed"]),
        "joint_failures": int(final_metrics["joint_failures"]),
        "margin_passed": int(final_metrics["margin_passed"]),
        "suppression_passed": int(final_metrics["suppression_passed"]),
        "minimum_worst_abstain_vs_true_margin": float(final_metrics["minimum_worst_abstain_vs_true_margin"]),
        "minimum_worst_true_logprob_drop": float(final_metrics["minimum_worst_true_logprob_drop"]),
        "gate_off_max_abs_logit_drift": float(final_equivalence["max_abs_logit_drift"]),
        "adapter_saved": True,
        "base_weights_modified": False,
        "target_new_used": False,
        "heldout_probe_text_used": False,
    }
    (method_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)
    hook.remove()


if __name__ == "__main__":
    main()
