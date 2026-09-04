#!/usr/bin/env python3
"""Shared utilities for the RSNR direct-logit baseline.

The baseline leaves embeddings, every Transformer block, final norm, and LM-head
weights unchanged. For an oracle-routed forgotten (subject, relation) prompt it
operates only on the LM-head logits:

    z'_t = z_t - delta   for token ids in the canonical sensitive answer
    z'_u = z_u + gamma   for token ids in the registered abstention

The primary baseline masks only the canonical target_true surface exposed by the
training-visible forget records. Held-out aliases are never used to construct the
mask; they are evaluation-only leakage probes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mcf_zero_unlearn_official_eval as official
import mcf_rsnr_v1a_official_eval_fresh_retain as ev
import run_mcf_rsnr_v1a_oracle as rsnr


PROTOCOL = "mcf_rsnr_v1a_oracle_direct_logit_mask"
ABSTENTION = rsnr.ABSTENTION
INTERVENTION_SITE = "post_lm_head_logits"


def fact_key(row: Mapping[str, Any]) -> tuple[str, str]:
    rr = row["requested_rewrite"]
    return str(rr["subject"]), str(rr["relation_id"])


def canonical_answer_map(rows: Sequence[Mapping[str, Any]]) -> Dict[tuple[str, str], str]:
    out: Dict[tuple[str, str], str] = {}
    for row in rows:
        key = fact_key(row)
        answer = str(row["requested_rewrite"]["target_true"]["str"])
        if key in out and out[key] != answer:
            raise RuntimeError(f"inconsistent canonical target_true for {key}")
        out[key] = answer
    return out


def canonical_token_sets(tokenizer: Any, answers: Mapping[tuple[str, str], str]) -> Dict[tuple[str, str], tuple[int, ...]]:
    out: Dict[tuple[str, str], tuple[int, ...]] = {}
    for key, answer in answers.items():
        ids = tuple(sorted(set(int(v) for v in rsnr.answer_ids(tokenizer, answer))))
        if not ids:
            raise RuntimeError(f"empty sensitive token set for {key}: {answer!r}")
        out[key] = ids
    return out


def abstention_token_set(tokenizer: Any) -> tuple[int, ...]:
    ids = tuple(sorted(set(int(v) for v in rsnr.answer_ids(tokenizer, ABSTENTION))))
    if not ids:
        raise RuntimeError("registered abstention token set is empty")
    return ids


@dataclass
class DirectLogitMaskHook:
    """Conditional post-LM-head logit intervention.

    `suppress_token_sets` is supplied per batch row. IDK token ids are global.
    Gate OFF returns the original LM-head tensor object without cloning, giving
    exact Base behavior by construction.
    """

    true_penalty: float
    idk_boost: float
    idk_token_ids: tuple[int, ...]
    handle: Any = None
    gate_mask: torch.Tensor | None = None
    position_mask: torch.Tensor | None = None
    suppress_token_sets: Sequence[Sequence[int]] | None = None

    @classmethod
    def install(
        cls,
        lm_head: nn.Module,
        *,
        true_penalty: float,
        idk_boost: float,
        idk_token_ids: Sequence[int],
    ) -> "DirectLogitMaskHook":
        state = cls(
            true_penalty=float(true_penalty),
            idk_boost=float(idk_boost),
            idk_token_ids=tuple(int(v) for v in idk_token_ids),
        )

        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
            if state.gate_mask is None or not torch.is_tensor(output) or output.ndim != 3:
                return output
            gate = state.gate_mask.to(device=output.device).view(-1)
            if gate.numel() != output.shape[0]:
                raise RuntimeError("direct-logit gate batch size mismatch")
            if not bool(torch.any(gate != 0).item()):
                return output

            if state.position_mask is None:
                positions = torch.ones(output.shape[:2], dtype=torch.bool, device=output.device)
            else:
                positions = state.position_mask.to(device=output.device).bool()
                if tuple(positions.shape) != tuple(output.shape[:2]):
                    raise RuntimeError("direct-logit position mask shape mismatch")

            if state.suppress_token_sets is None or len(state.suppress_token_sets) != output.shape[0]:
                raise RuntimeError("direct-logit suppress-token sets missing or batch-mismatched")

            edited = output.clone()
            for i in range(output.shape[0]):
                if float(gate[i].item()) == 0.0:
                    continue
                active_positions = torch.nonzero(positions[i], as_tuple=False).flatten()
                if active_positions.numel() == 0:
                    continue
                suppress_ids = tuple(int(v) for v in state.suppress_token_sets[i])
                for pos in active_positions.tolist():
                    if state.true_penalty != 0.0 and suppress_ids:
                        edited[i, pos, list(suppress_ids)] -= float(state.true_penalty)
                    if state.idk_boost != 0.0 and state.idk_token_ids:
                        edited[i, pos, list(state.idk_token_ids)] += float(state.idk_boost)
            return edited

        state.handle = lm_head.register_forward_hook(hook)
        return state

    def set(
        self,
        gate_mask: torch.Tensor | None,
        position_mask: torch.Tensor | None = None,
        suppress_token_sets: Sequence[Sequence[int]] | None = None,
    ) -> None:
        self.gate_mask = gate_mask
        self.position_mask = position_mask
        self.suppress_token_sets = suppress_token_sets

    def clear(self) -> None:
        self.gate_mask = None
        self.position_mask = None
        self.suppress_token_sets = None

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()


def get_lm_head(model: Any) -> nn.Module:
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is None:
        head = getattr(model, "lm_head", None)
    if not isinstance(head, nn.Module):
        raise RuntimeError("could not locate LM head")
    return head


def _pair_from_match(match: Any) -> tuple[str, str] | None:
    if match is None:
        return None
    _case_id, subject, relation = match
    return str(subject), str(relation)


def suppress_sets_for_pairs(
    pairs: Sequence[tuple[str, str] | None],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
) -> list[tuple[int, ...]]:
    return [tuple(token_sets[pair]) if pair is not None else tuple() for pair in pairs]


@torch.no_grad()
def sequence_logprobs_masked(
    model: Any,
    hook: DirectLogitMaskHook,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    pairs: Sequence[tuple[str, str] | None],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    *,
    device: torch.device,
    gated: bool,
) -> torch.Tensor:
    if len(prompts) != len(answers) or len(prompts) != len(pairs):
        raise ValueError("prompt/answer/pair length mismatch")
    input_ids, attention, starts, targets, positions = rsnr.encode_prompt_answer_batch(
        tokenizer, prompts, answers, device=device
    )
    gate = torch.ones(len(prompts), device=device) if gated else torch.zeros(len(prompts), device=device)
    suppress = suppress_sets_for_pairs(pairs, token_sets)
    hook.set(gate, positions if gated else None, suppress if gated else [tuple()] * len(prompts))
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


@torch.no_grad()
def legacy_test_batch_prediction(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    matches: Sequence[Any],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    llama_like: bool,
):
    if not prefixes:
        return []
    if len(prefixes) != len(matches):
        raise ValueError("prefix/match length mismatch")

    raw_prefix_lens = [len(x) for x in tok(list(prefixes))["input_ids"]]
    texts = [f"{prefix} {suffix}" for prefix in prefixes for suffix in (target_new, target_true)]
    batch = tok(texts, padding=True, return_tensors="pt").to(device)
    a_tok, b_tok = (tok(f" {x}")["input_ids"] for x in (target_new, target_true))
    score_prefix_lens = list(raw_prefix_lens)
    if llama_like:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        score_prefix_lens = [x - 1 for x in score_prefix_lens]

    positions = torch.zeros_like(batch["input_ids"], dtype=torch.float32)
    gate_rows: list[float] = []
    suppress_rows: list[tuple[int, ...]] = []
    for seq_index in range(len(texts)):
        prefix_index = seq_index // 2
        pair = _pair_from_match(matches[prefix_index])
        gate_rows.append(1.0 if pair is not None else 0.0)
        suppress_rows.append(tuple(token_sets[pair]) if pair is not None else tuple())
        cur_len = len(a_tok) if seq_index % 2 == 0 else len(b_tok)
        raw_start = int(raw_prefix_lens[prefix_index]) - 1
        for pos in range(raw_start, raw_start + cur_len):
            if 0 <= pos < positions.shape[1]:
                positions[seq_index, pos] = 1.0

    hook.set(torch.tensor(gate_rows, device=device), positions, suppress_rows)
    try:
        logits = model(**batch).logits
    finally:
        hook.clear()
    if llama_like:
        logits = logits[:, 1:, :]

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    for i in range(logits.size(0)):
        cur_tokens = a_tok if i % 2 == 0 else b_tok
        for j, cur_tok in enumerate(cur_tokens):
            pos = score_prefix_lens[i // 2] + j - 1
            probs[i] += -F.log_softmax(logits[i, pos, :], dim=0)[cur_tok].item()
        probs[i] /= max(1, len(cur_tokens))
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ]


@torch.no_grad()
def compute_record(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    record: Mapping[str, Any],
    router: ev.OraclePromptRouter,
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    llama_like: bool,
):
    rr = record["requested_rewrite"]
    target_new = str(rr["target_new"]["str"])
    target_true = str(rr["target_true"]["str"])
    routed = ev.route_record_prompts(record, router)
    out: Dict[str, Any] = {}
    for group, items in routed.items():
        prompts = [str(item["prompt"]) for item in items]
        matches = [item["match"] for item in items]
        out[f"{group}_prompts_probs"] = legacy_test_batch_prediction(
            model, hook, tok, prompts, target_new, target_true, matches,
            token_sets, device, llama_like=llama_like,
        )
    return out, routed


def evaluate_split(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    router: ev.OraclePromptRouter,
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    llama_like: bool,
    split_name: str,
):
    metrics = []
    counts: Dict[str, Any] = {
        "rewrite_total": 0,
        "rewrite_gated": 0,
        "paraphrase_total": 0,
        "paraphrase_gated": 0,
        "neighborhood_total": 0,
        "neighborhood_gated": 0,
        "matching_fact_records": 0,
        "sensitive_neighborhood_examples": [],
    }
    sensitive_prompts: list[Dict[str, Any]] = []
    for record in records:
        post, routed = compute_record(
            model, hook, tok, record, router, token_sets, device, llama_like=llama_like
        )
        if fact_key(record) in router.forget_pairs:
            counts["matching_fact_records"] += 1
        rr = record["requested_rewrite"]
        for group, items in routed.items():
            counts[f"{group}_total"] += len(items)
            for item in items:
                if not item["gated"]:
                    continue
                counts[f"{group}_gated"] += 1
                matched_case, matched_subject, matched_relation = item["match"]
                sensitive_prompts.append({
                    "split": split_name,
                    "group": group,
                    "parent_case_id": int(record["case_id"]),
                    "parent_subject": str(rr["subject"]),
                    "parent_relation_id": str(rr["relation_id"]),
                    "prompt": str(item["prompt"]),
                    "matched_case_id": int(matched_case),
                    "matched_subject": str(matched_subject),
                    "matched_relation_id": str(matched_relation),
                })
                if group == "neighborhood" and len(counts["sensitive_neighborhood_examples"]) < 10:
                    counts["sensitive_neighborhood_examples"].append({
                        "prompt": str(item["prompt"]),
                        "matched_subject": str(matched_subject),
                        "matched_relation_id": str(matched_relation),
                    })
        metrics.append({"requested_rewrite": record["requested_rewrite"], "post": post})
    return official.official_summarize(split_name, metrics), metrics, counts, sensitive_prompts


@torch.no_grad()
def native_teacher_forced_audit(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    sensitive_prompts: Sequence[Mapping[str, Any]],
    canonical_answers: Mapping[tuple[str, str], str],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for batch_rows in ev._chunks(list(sensitive_prompts), batch_size):
        prompts = [str(row["prompt"]) for row in batch_rows]
        pairs = [(str(row["matched_subject"]), str(row["matched_relation_id"])) for row in batch_rows]
        true_answers = [canonical_answers[pair] for pair in pairs]
        idk_answers = [ABSTENTION] * len(batch_rows)
        masked_true = sequence_logprobs_masked(
            model, hook, tok, prompts, true_answers, pairs, token_sets, device=device, gated=True
        )
        masked_idk = sequence_logprobs_masked(
            model, hook, tok, prompts, idk_answers, pairs, token_sets, device=device, gated=True
        )
        base_true = sequence_logprobs_masked(
            model, hook, tok, prompts, true_answers, pairs, token_sets, device=device, gated=False
        )
        for i, source in enumerate(batch_rows):
            margin = float((masked_idk[i] - masked_true[i]).item())
            drop = float((base_true[i] - masked_true[i]).item())
            rows.append({
                **dict(source),
                "true_answer_scored": true_answers[i],
                "idk_logprob": float(masked_idk[i].item()),
                "masked_true_logprob": float(masked_true[i].item()),
                "base_true_logprob": float(base_true[i].item()),
                "idk_vs_true_margin": margin,
                "true_logprob_drop": drop,
                "idk_margin_pass_0p1": margin >= 0.1,
                "true_drop_pass_2p0": drop >= 2.0,
            })
    return {
        "prompt_count": len(rows),
        "idk_margin_passed_0p1": sum(bool(r["idk_margin_pass_0p1"]) for r in rows),
        "true_drop_passed_2p0": sum(bool(r["true_drop_pass_2p0"]) for r in rows),
        "joint_passed": sum(bool(r["idk_margin_pass_0p1"] and r["true_drop_pass_2p0"]) for r in rows),
        "minimum_idk_vs_true_margin": min((r["idk_vs_true_margin"] for r in rows), default=None),
        "minimum_true_logprob_drop": min((r["true_logprob_drop"] for r in rows), default=None),
        "per_prompt": rows,
    }


@torch.no_grad()
def greedy_generate_logitmask(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    prompts: Sequence[str],
    pairs: Sequence[tuple[str, str] | None],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    device: torch.device,
    *,
    max_new_tokens: int,
) -> list[str]:
    if len(prompts) != len(pairs):
        raise ValueError("prompt/pair length mismatch")
    if not prompts:
        return []
    sequences = [
        [int(v) for v in tok(str(prompt), add_special_tokens=True, return_attention_mask=False)["input_ids"]]
        for prompt in prompts
    ]
    if any(not seq for seq in sequences):
        raise RuntimeError("cannot generate from empty prompt")
    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos if eos is not None else 0
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False] * len(sequences)

    for _ in range(int(max_new_tokens)):
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), int(pad), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        positions = torch.zeros_like(input_ids, dtype=torch.float32)
        gate = torch.zeros(len(sequences), dtype=torch.float32, device=device)
        suppress_rows: list[tuple[int, ...]] = []
        last_positions = []
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention[i, : len(seq)] = 1
            last = len(seq) - 1
            last_positions.append(last)
            pair = pairs[i]
            suppress_rows.append(tuple(token_sets[pair]) if pair is not None else tuple())
            if not finished[i] and pair is not None:
                gate[i] = 1.0
                positions[i, last] = 1.0

        hook.set(gate, positions, suppress_rows)
        try:
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        finally:
            hook.clear()
        active = False
        for i, last in enumerate(last_positions):
            if finished[i]:
                continue
            token = int(torch.argmax(logits[i, last, :]).item())
            generated[i].append(token)
            sequences[i].append(token)
            if eos is not None and token == int(eos):
                finished[i] = True
            else:
                active = True
        if not active:
            break
    return [tok.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]


def generation_audit(
    model: Any,
    hook: DirectLogitMaskHook,
    tok: Any,
    sensitive_prompts: Sequence[Mapping[str, Any]],
    token_sets: Mapping[tuple[str, str], Sequence[int]],
    alias_map: Mapping[tuple[str, str], Sequence[str]],
    device: torch.device,
    *,
    max_new_tokens: int,
    batch_size: int,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for batch_rows in ev._chunks(list(sensitive_prompts), batch_size):
        prompts = [str(row["prompt"]) for row in batch_rows]
        pairs = [(str(row["matched_subject"]), str(row["matched_relation_id"])) for row in batch_rows]
        outputs = greedy_generate_logitmask(
            model, hook, tok, prompts, pairs, token_sets, device, max_new_tokens=max_new_tokens
        )
        for source, pair, output in zip(batch_rows, pairs, outputs):
            aliases = list(alias_map[pair])
            abstain = ev.detect_abstention(output)
            leaks = ev.leaked_aliases(output, aliases)
            rows.append({
                **dict(source),
                "generation": output,
                "exact_abstention": bool(abstain["exact"]),
                "semantic_abstention": bool(abstain["semantic"]),
                "true_answer_aliases_checked": aliases,
                "leaked_aliases": leaks,
                "true_or_alias_leak": bool(leaks),
                "nondisclosure_pass": not bool(leaks),
                "abstaining_nondisclosure_pass": bool(abstain["semantic"]) and not bool(leaks),
            })
    total = len(rows)
    exact = sum(bool(r["exact_abstention"]) for r in rows)
    semantic = sum(bool(r["semantic_abstention"]) for r in rows)
    leaks = sum(bool(r["true_or_alias_leak"]) for r in rows)
    by_group: Dict[str, Dict[str, int]] = {}
    for key in sorted({f"{r['split']}_{r['group']}" for r in rows}):
        subset = [r for r in rows if f"{r['split']}_{r['group']}" == key]
        by_group[key] = {
            "prompts": len(subset),
            "exact_abstentions": sum(bool(r["exact_abstention"]) for r in subset),
            "semantic_abstentions": sum(bool(r["semantic_abstention"]) for r in subset),
            "true_or_alias_leaks": sum(bool(r["true_or_alias_leak"]) for r in subset),
        }
    return {
        "prompt_count": total,
        "greedy_decoding": True,
        "max_new_tokens": int(max_new_tokens),
        "exact_abstention_count": exact,
        "exact_abstention_rate": exact / total if total else None,
        "semantic_abstention_count": semantic,
        "semantic_abstention_rate": semantic / total if total else None,
        "true_or_alias_leak_count": leaks,
        "true_or_alias_leak_rate": leaks / total if total else None,
        "nondisclosure_pass_rate": (total - leaks) / total if total else None,
        "by_split_group": by_group,
        "per_prompt": rows,
    }


def summarize_method_aligned(
    teacher_rows: Sequence[Mapping[str, Any]], generation_rows: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "definition": {
            "Eff_IDK_Gen_IDK": "percent with logP(IDK)-logP(true) <= 0; lower is better",
            "Sensitive_Eff_Sensitive_Gen": "percent greedy generations leaking true answer or explicit alias; lower is better",
        },
        "registered_abstention": ABSTENTION,
    }
    for group, suffix in (("rewrite", "Eff"), ("paraphrase", "Gen")):
        trows = [r for r in teacher_rows if r.get("split") == "forget" and r.get("group") == group]
        grows = [r for r in generation_rows if r.get("split") == "forget" and r.get("group") == group]
        margins = [float(r["idk_vs_true_margin"]) for r in trows]
        drops = [float(r["true_logprob_drop"]) for r in trows]
        fail = sum(m <= 0.0 for m in margins)
        buffered = sum(m < 0.1 for m in margins)
        leaks = sum(bool(r["true_or_alias_leak"]) for r in grows)
        abstain = sum(bool(r["semantic_abstention"]) for r in grows)
        n_t, n_g = len(trows), len(grows)
        out[f"{suffix}_IDK"] = round(100.0 * fail / n_t, 4) if n_t else None
        out[f"{suffix}_IDK_buffered_0p1"] = round(100.0 * buffered / n_t, 4) if n_t else None
        out[f"{suffix}_IDK_prompt_count"] = n_t
        out[f"{suffix}_minimum_idk_vs_true_margin"] = min(margins) if margins else None
        out[f"{suffix}_mean_true_logprob_drop"] = sum(drops) / n_t if n_t else None
        out[f"{suffix}_minimum_true_logprob_drop"] = min(drops) if drops else None
        out[f"Sensitive_{suffix}"] = round(100.0 * leaks / n_g, 4) if n_g else None
        out[f"Sensitive_{suffix}_leak_count"] = leaks
        out[f"Sensitive_{suffix}_prompt_count"] = n_g
        out[f"{suffix}_semantic_abstention_rate"] = abstain / n_g if n_g else None
    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("direct-logit config protocol mismatch")
    if payload.get("calibration_passed") is not True:
        raise RuntimeError("direct-logit config did not pass its registered calibration gate")
    if payload.get("heldout_probe_text_used") is not False:
        raise RuntimeError("direct-logit config does not certify heldout_probe_text_used=false")
    if payload.get("aliases_used_for_mask") is not False:
        raise RuntimeError("primary direct-logit baseline must not use aliases in mask construction")
    return payload
