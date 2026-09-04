#!/usr/bin/env python3
"""Official-compatible + RSNR-native MCF evaluation for RSNR-V1A.

This evaluator reloads the frozen Base model plus the saved oracle-null adapter.
Unlike the first evaluator draft, routing is resolved PER PROMPT, not per
benchmark prompt group.  The parent record supplies the relation id, while the
actual prompt text is checked for any registered forgotten subject on that
relation.  Rewrite/paraphrase prompts of a forgotten parent fact retain an
oracle metadata fallback if the subject is not literally present.  Therefore a
neighborhood prompt such as ``BMW M5 is developed by`` activates the adapter
whenever (BMW M5, P176) is itself in the forget set.

It reports two complementary views:

1. The repository's unchanged ZeroUnlearn-compatible Eff/Gen/Spe/PPL metrics.
   These still compare target_true with CounterFact target_new for historical
   comparability, even though RSNR never trains on target_new.
2. RSNR-native nondisclosure metrics on every official prompt that resolves to
   a forgotten pair: IDK-vs-true teacher-forced margin, Base-to-RSNR true-answer
   suppression, greedy generation, exact/rule-semantic abstention, and true
   answer / alias leakage.

The evaluator also validates artifact correspondence across the adapter,
routing sidecar, completion.json, locked training-visible forget records, and
split manifest before any scoring is accepted.

Seed 1 remains DEVELOPMENT ONLY because its official aggregates are consumed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import run_mcf_rsnr_v1a_oracle as rsnr
from mcf_sampling import sample_official_mcf_records


PROTOCOL = rsnr.PROTOCOL
ABSTENTION = rsnr.ABSTENTION


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--out", required=True)
    p.add_argument("--unlearn-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fresh-retain-seed", type=int, default=700002)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--generation-max-new-tokens", type=int, default=20)
    p.add_argument("--generation-batch-size", type=int, default=8)
    args = p.parse_args()
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        p.error("RSNR-V1A development evaluation is locked to seed=1, forget=50, retain=1000")
    if args.generation_max_new_tokens <= 0 or args.generation_batch_size <= 0:
        p.error("generation limits must be positive")
    return args


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _load_manifest(protocol_dir: Path) -> Dict[str, Any]:
    payload = _load_json(protocol_dir / "split_manifest.json")
    if int(payload.get("seed", -1)) != 1:
        raise RuntimeError("RSNR-V1A eval expects the locked seed-1 protocol")
    return payload


def _load_locked_forget(protocol_dir: Path) -> list[Dict[str, Any]]:
    path = protocol_dir / "training_visible_forget_direct.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 50:
        raise RuntimeError("locked RSNR forget-direct file must contain exactly 50 records")
    return rows


def _excluded_retain_ids(manifest: Mapping[str, Any]) -> set[int]:
    excluded = {int(v) for v in manifest.get("official_retain_case_ids_only", [])}
    case_ids = manifest.get("case_ids", {})
    if not isinstance(case_ids, Mapping):
        raise RuntimeError("split manifest lacks case_ids mapping")
    for values in case_ids.values():
        excluded.update(int(v) for v in values)
    return excluded


def _expected_forget_ids(manifest: Mapping[str, Any]) -> list[int]:
    return [int(v) for v in manifest.get("case_ids", {}).get("forget", [])]


def _load_adapter(run_dir: Path):
    path = run_dir / "method" / "rsnr_oracle_null_adapter.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("RSNR adapter protocol mismatch")
    return payload, path


def _load_sidecar(run_dir: Path) -> Dict[str, Any]:
    payload = _load_json(run_dir / "method" / "relation_scoped_null_routing.json")
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("RSNR routing sidecar protocol mismatch")
    if payload.get("routing") != "oracle_exact_subject_relation_membership":
        raise RuntimeError("RSNR-V1A evaluator requires oracle exact-pair routing")
    return payload


def _load_completion(run_dir: Path) -> Dict[str, Any]:
    payload = _load_json(run_dir / "method" / "completion.json")
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("RSNR completion protocol mismatch")
    return payload


def _fact_key(record: Mapping[str, Any]) -> tuple[str, str]:
    rr = record["requested_rewrite"]
    return str(rr["subject"]), str(rr["relation_id"])


def _membership_rows(rows: Sequence[Mapping[str, Any]], *, source: str) -> set[tuple[int, str, str]]:
    values: list[tuple[int, str, str]] = []
    for row in rows:
        if "requested_rewrite" in row:
            subject, relation = _fact_key(row)
            values.append((int(row["case_id"]), subject, relation))
        else:
            values.append((int(row["case_id"]), str(row["subject"]), str(row["relation_id"])))
    if len(set(values)) != len(values):
        raise RuntimeError(f"duplicate forget membership rows in {source}")
    return set(values)


def validate_artifact_correspondence(
    *,
    adapter_payload: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    completion: Mapping[str, Any],
    locked_forget: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    expected_count: int,
) -> Dict[str, Any]:
    checkpoint = _membership_rows(adapter_payload.get("forget_membership", []), source="adapter")
    routed = _membership_rows(sidecar.get("forget_membership", []), source="sidecar")
    locked = _membership_rows(locked_forget, source="training_visible_forget_direct")
    manifest_ids = set(_expected_forget_ids(manifest))
    locked_ids = {case_id for case_id, _subject, _relation in locked}

    if len(checkpoint) != expected_count or len(routed) != expected_count or len(locked) != expected_count:
        raise RuntimeError("RSNR artifact membership count mismatch")
    if checkpoint != routed:
        raise RuntimeError("adapter and routing sidecar forget membership disagree")
    if checkpoint != locked:
        raise RuntimeError("adapter/sidecar membership does not match locked forget records")
    if locked_ids != manifest_ids:
        raise RuntimeError("locked forget records do not match split manifest forget IDs")

    joint_failures = int(completion.get("joint_failures", -1))
    joint_passed = int(completion.get("joint_passed", -1))
    explicit_gate = completion.get("training_gate_passed")
    if joint_failures != 0 or joint_passed != expected_count:
        raise RuntimeError(
            f"RSNR training gate failed: joint_passed={joint_passed}, joint_failures={joint_failures}"
        )
    if explicit_gate is False:
        raise RuntimeError("RSNR completion explicitly marks training_gate_passed=false")
    if completion.get("adapter_saved") is not True:
        raise RuntimeError("RSNR completion does not certify adapter_saved=true")
    if completion.get("base_weights_modified") is not False:
        raise RuntimeError("RSNR completion does not certify frozen Base weights")
    if completion.get("heldout_probe_text_used") is not False:
        raise RuntimeError("RSNR completion does not certify heldout_probe_text_used=false")

    if str(adapter_payload.get("abstention")) != ABSTENTION:
        raise RuntimeError("adapter abstention text does not match registered RSNR abstention")
    if str(sidecar.get("abstention_text")) != ABSTENTION:
        raise RuntimeError("sidecar abstention text does not match registered RSNR abstention")

    return {
        "passed": True,
        "membership_count": len(locked),
        "adapter_sidecar_locked_forget_exact_match": True,
        "manifest_forget_ids_exact_match": True,
        "completion_training_gate_passed": True,
        "completion_joint_passed": joint_passed,
        "completion_joint_failures": joint_failures,
    }


def _normalize_text(text: str) -> str:
    value = str(text).casefold().replace("’", "'")
    value = re.sub(r"[^\w']+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def _contains_surface(text: str, surface: str) -> bool:
    hay = _normalize_text(text)
    needle = _normalize_text(surface)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", hay) is not None


class OraclePromptRouter:
    """Resolve oracle forget membership from each actual prompt.

    Relation id is supplied by the benchmark record.  Subject identity is
    resolved from the prompt text against forgotten subjects on that relation.
    For rewrite/paraphrase prompts only, an exact forgotten parent pair is an
    oracle fallback when the paraphrase omits the literal subject.
    """

    def __init__(self, membership: Iterable[tuple[int, str, str]]):
        self.by_relation: Dict[str, list[tuple[int, str, str]]] = defaultdict(list)
        self.by_pair: Dict[tuple[str, str], tuple[int, str, str]] = {}
        for case_id, subject, relation in membership:
            item = (int(case_id), str(subject), str(relation))
            self.by_relation[str(relation)].append(item)
            pair = (str(subject), str(relation))
            if pair in self.by_pair:
                raise RuntimeError(f"duplicate oracle pair {pair}")
            self.by_pair[pair] = item
        for relation in self.by_relation:
            self.by_relation[relation].sort(key=lambda x: (-len(x[1]), x[1], x[0]))

    @property
    def forget_pairs(self) -> set[tuple[str, str]]:
        return set(self.by_pair)

    def resolve(
        self,
        prompt: str,
        relation_id: str,
        *,
        parent_subject: str | None = None,
        allow_parent_fallback: bool = False,
    ) -> tuple[int, str, str] | None:
        relation = str(relation_id)
        matches = [item for item in self.by_relation.get(relation, []) if _contains_surface(prompt, item[1])]
        # Remove exact duplicate pair matches defensively.
        unique = {(case_id, subject, rel) for case_id, subject, rel in matches}
        if len(unique) > 1:
            raise RuntimeError(
                "atomic-query RSNR router found multiple forgotten subjects for one relation in prompt: "
                f"{prompt!r} -> {sorted(unique)}"
            )
        if unique:
            return next(iter(unique))
        if allow_parent_fallback and parent_subject is not None:
            return self.by_pair.get((str(parent_subject), relation))
        return None


def route_record_prompts(
    record: Mapping[str, Any], router: OraclePromptRouter
) -> Dict[str, list[Dict[str, Any]]]:
    rr = record["requested_rewrite"]
    subject = str(rr["subject"])
    relation = str(rr["relation_id"])
    groups = {
        "rewrite": [str(rr["prompt"]).format(subject)],
        "paraphrase": list(record.get("paraphrase_prompts", [])),
        "neighborhood": list(record.get("neighborhood_prompts", [])),
    }
    routed: Dict[str, list[Dict[str, Any]]] = {}
    for group, prompts in groups.items():
        values = []
        for prompt in prompts:
            match = router.resolve(
                str(prompt),
                relation,
                parent_subject=subject,
                allow_parent_fallback=group in {"rewrite", "paraphrase"},
            )
            values.append({"prompt": str(prompt), "match": match, "gated": match is not None})
        routed[group] = values
    return routed


@torch.no_grad()
def rsnr_test_batch_prediction(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    prefixes: Sequence[str],
    target_new: str,
    target_true: str,
    gated_flags: Sequence[bool],
    device: torch.device,
    *,
    llama_like: bool,
):
    """ZeroUnlearn-compatible target NLLs with per-prefix RSNR routing."""
    if len(prefixes) == 0:
        return []
    if len(prefixes) != len(gated_flags):
        raise ValueError("prefix/gate length mismatch")

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
    gate_rows = []
    for seq_index in range(len(texts)):
        prefix_index = seq_index // 2
        gate_rows.append(1.0 if gated_flags[prefix_index] else 0.0)
        cur_len = len(a_tok) if seq_index % 2 == 0 else len(b_tok)
        raw_start = int(raw_prefix_lens[prefix_index]) - 1
        for pos in range(raw_start, raw_start + cur_len):
            if 0 <= pos < positions.shape[1]:
                positions[seq_index, pos] = 1.0

    hook.set(torch.tensor(gate_rows, device=device), positions)
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
            probs[i] += -torch.nn.functional.log_softmax(logits[i, pos, :], dim=0)[cur_tok].item()
        probs[i] /= max(1, len(cur_tokens))
    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ]


@torch.no_grad()
def compute_record(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    record: Mapping[str, Any],
    router: OraclePromptRouter,
    device: torch.device,
    *,
    llama_like: bool,
):
    rr = record["requested_rewrite"]
    target_new = str(rr["target_new"]["str"])
    target_true = str(rr["target_true"]["str"])
    routed = route_record_prompts(record, router)
    out: Dict[str, Any] = {}
    for group, items in routed.items():
        prompts = [item["prompt"] for item in items]
        flags = [bool(item["gated"]) for item in items]
        out[f"{group}_prompts_probs"] = rsnr_test_batch_prediction(
            model, hook, tok, prompts, target_new, target_true, flags, device,
            llama_like=llama_like,
        )
    return out, routed


def evaluate_split(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    router: OraclePromptRouter,
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
            model, hook, tok, record, router, device, llama_like=llama_like
        )
        if _fact_key(record) in router.forget_pairs:
            counts["matching_fact_records"] += 1
        rr = record["requested_rewrite"]
        for group, items in routed.items():
            counts[f"{group}_total"] += len(items)
            for item in items:
                if item["gated"]:
                    counts[f"{group}_gated"] += 1
                    matched_case, matched_subject, matched_relation = item["match"]
                    sensitive_prompts.append({
                        "split": split_name,
                        "group": group,
                        "parent_case_id": int(record["case_id"]),
                        "parent_subject": str(rr["subject"]),
                        "parent_relation_id": str(rr["relation_id"]),
                        "prompt": item["prompt"],
                        "matched_case_id": int(matched_case),
                        "matched_subject": str(matched_subject),
                        "matched_relation_id": str(matched_relation),
                    })
                    if group == "neighborhood" and len(counts["sensitive_neighborhood_examples"]) < 10:
                        counts["sensitive_neighborhood_examples"].append({
                            "prompt": item["prompt"],
                            "matched_subject": str(matched_subject),
                            "matched_relation_id": str(matched_relation),
                        })
        metrics.append({"requested_rewrite": record["requested_rewrite"], "post": post})
    return official.official_summarize(split_name, metrics), metrics, counts, sensitive_prompts


def fresh_split(
    data: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    unlearn_num: int,
    retain_num: int,
    seed: int,
    fresh_retain_seed: int,
):
    forget_records, _ = sample_official_mcf_records(data, unlearn_num, 0, seed, strict=True)
    expected = _expected_forget_ids(manifest)
    forget_ids = [int(row["case_id"]) for row in forget_records]
    if forget_ids != expected:
        raise RuntimeError("official forget sample does not match locked RSNR/V1.3 split")

    excluded = _excluded_retain_ids(manifest)
    half = len(data) // 2
    candidates = [row for row in data[:half] if int(row["case_id"]) not in excluded]
    if len(candidates) < retain_num:
        raise RuntimeError(f"fresh retain pool has {len(candidates)} records; need {retain_num}")
    retain_records = random.Random(int(fresh_retain_seed)).sample(candidates, k=retain_num)
    retain_ids = [int(row["case_id"]) for row in retain_records]
    if excluded.intersection(retain_ids):
        raise AssertionError("fresh retain sample overlaps excluded protocol records")
    return (
        [official.normalize_record(row) for row in forget_records],
        [official.normalize_record(row) for row in retain_records],
        {
            "forget_case_ids": forget_ids,
            "fresh_retain_case_ids": retain_ids,
            "fresh_retain_seed": int(fresh_retain_seed),
            "excluded_case_id_count": len(excluded),
            "fresh_retain_disjoint_from_all_excluded": True,
        },
    )


def _explicit_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for key in ("str", "name", "label"):
            if isinstance(value.get(key), str):
                out.append(str(value[key]))
        for key in ("aliases", "alias", "names", "labels", "str_aliases"):
            out.extend(_explicit_aliases(value.get(key)))
        return out
    if isinstance(value, Sequence):
        out: list[str] = []
        for item in value:
            out.extend(_explicit_aliases(item))
        return out
    return []


def build_true_alias_map(
    data: Sequence[Mapping[str, Any]], forget_records: Sequence[Mapping[str, Any]]
) -> Dict[tuple[str, str], list[str]]:
    by_target_id: Dict[str, set[str]] = defaultdict(set)
    for row in data:
        rr = row.get("requested_rewrite", {})
        target = rr.get("target_true", {}) if isinstance(rr, Mapping) else {}
        if isinstance(target, Mapping) and target.get("id") is not None:
            target_id = str(target["id"])
            by_target_id[target_id].update(_explicit_aliases(target))

    result: Dict[tuple[str, str], list[str]] = {}
    for row in forget_records:
        rr = row["requested_rewrite"]
        key = _fact_key(row)
        target = rr["target_true"]
        aliases = set(_explicit_aliases(target))
        if isinstance(target, Mapping) and target.get("id") is not None:
            aliases.update(by_target_id.get(str(target["id"]), set()))
        # Only use explicit true-answer fields; never infer aliases from target_new.
        aliases = {a.strip() for a in aliases if isinstance(a, str) and a.strip()}
        if not aliases:
            aliases = {str(target["str"]).strip()}
        result[key] = sorted(aliases, key=lambda x: (-len(x), x.casefold()))
    return result


def detect_abstention(text: str) -> Dict[str, bool]:
    normalized = _normalize_text(text)
    canonical = _normalize_text(ABSTENTION)
    exact = normalized == canonical or normalized.startswith(canonical + " ")
    semantic_patterns = (
        r"\bi do not know\b",
        r"\bi don't know\b",
        r"\bi am not sure\b",
        r"\bi'm not sure\b",
        r"\bunknown\b",
        r"\bcannot determine\b",
        r"\bcan't determine\b",
        r"\bcannot answer\b",
        r"\bcan't answer\b",
        r"\bunable to answer\b",
        r"\binsufficient information\b",
    )
    semantic = exact or any(re.search(pattern, normalized) for pattern in semantic_patterns)
    return {"exact": bool(exact), "semantic": bool(semantic)}


def leaked_aliases(text: str, aliases: Sequence[str]) -> list[str]:
    return [alias for alias in aliases if _contains_surface(text, alias)]


@torch.no_grad()
def greedy_generate_rsnr(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    prompts: Sequence[str],
    gated_flags: Sequence[bool],
    device: torch.device,
    *,
    max_new_tokens: int,
) -> list[str]:
    """Greedy decode while applying RSNR only at each next-token position.

    This deliberately runs without KV cache so the hook can mark the current
    final attended position exactly, matching the answer-position intervention
    geometry used in training.
    """
    if len(prompts) != len(gated_flags):
        raise ValueError("prompt/gate length mismatch")
    if not prompts:
        return []

    sequences: list[list[int]] = []
    for prompt in prompts:
        ids = tok(str(prompt), add_special_tokens=True, return_attention_mask=False)["input_ids"]
        if not ids:
            raise RuntimeError("cannot generate from empty prompt")
        sequences.append([int(v) for v in ids])

    eos = getattr(tok, "eos_token_id", None)
    pad = getattr(tok, "pad_token_id", None)
    if pad is None:
        pad = eos if eos is not None else 0
    generated: list[list[int]] = [[] for _ in sequences]
    finished = [False] * len(sequences)

    for _step in range(int(max_new_tokens)):
        max_len = max(len(seq) for seq in sequences)
        input_ids = torch.full((len(sequences), max_len), int(pad), dtype=torch.long, device=device)
        attention = torch.zeros_like(input_ids)
        positions = torch.zeros_like(input_ids, dtype=torch.float32)
        gate = torch.zeros(len(sequences), dtype=torch.float32, device=device)
        last_positions = []
        for i, seq in enumerate(sequences):
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            attention[i, : len(seq)] = 1
            last = len(seq) - 1
            last_positions.append(last)
            if not finished[i] and gated_flags[i]:
                gate[i] = 1.0
                positions[i, last] = 1.0

        hook.set(gate, positions)
        try:
            logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        finally:
            hook.clear()

        any_active = False
        for i, last in enumerate(last_positions):
            if finished[i]:
                continue
            token = int(torch.argmax(logits[i, last, :]).item())
            generated[i].append(token)
            sequences[i].append(token)
            if eos is not None and token == int(eos):
                finished[i] = True
            else:
                any_active = True
        if not any_active:
            break

    return [tok.decode(tokens, skip_special_tokens=True).strip() for tokens in generated]


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def rsnr_native_teacher_forced_audit(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    sensitive_prompts: Sequence[Mapping[str, Any]],
    true_alias_map: Mapping[tuple[str, str], Sequence[str]],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for batch_rows in _chunks(list(sensitive_prompts), batch_size):
        prompts = [str(row["prompt"]) for row in batch_rows]
        true_answers = [
            str(true_alias_map[(str(row["matched_subject"]), str(row["matched_relation_id"]))][0])
            for row in batch_rows
        ]
        idk_answers = [ABSTENTION] * len(batch_rows)
        gated_true = rsnr.sequence_logprobs(
            model, hook, tok, prompts, true_answers, device=device, gated=True
        )
        gated_idk = rsnr.sequence_logprobs(
            model, hook, tok, prompts, idk_answers, device=device, gated=True
        )
        base_true = rsnr.sequence_logprobs(
            model, hook, tok, prompts, true_answers, device=device, gated=False
        )
        for i, source in enumerate(batch_rows):
            margin = float((gated_idk[i] - gated_true[i]).item())
            drop = float((base_true[i] - gated_true[i]).item())
            rows.append({
                **dict(source),
                "true_answer_scored": true_answers[i],
                "idk_logprob": float(gated_idk[i].item()),
                "rsnr_true_logprob": float(gated_true[i].item()),
                "base_true_logprob": float(base_true[i].item()),
                "idk_vs_true_margin": margin,
                "true_logprob_drop": drop,
                "idk_margin_pass_0p1": margin >= 0.1,
                "true_drop_pass_2p0": drop >= 2.0,
            })
    return {
        "prompt_count": len(rows),
        "idk_margin_passed_0p1": sum(bool(row["idk_margin_pass_0p1"]) for row in rows),
        "true_drop_passed_2p0": sum(bool(row["true_drop_pass_2p0"]) for row in rows),
        "joint_passed": sum(
            bool(row["idk_margin_pass_0p1"] and row["true_drop_pass_2p0"]) for row in rows
        ),
        "minimum_idk_vs_true_margin": min((row["idk_vs_true_margin"] for row in rows), default=None),
        "minimum_true_logprob_drop": min((row["true_logprob_drop"] for row in rows), default=None),
        "per_prompt": rows,
    }


def generation_audit(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    sensitive_prompts: Sequence[Mapping[str, Any]],
    true_alias_map: Mapping[tuple[str, str], Sequence[str]],
    device: torch.device,
    *,
    max_new_tokens: int,
    batch_size: int,
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for batch_rows in _chunks(list(sensitive_prompts), batch_size):
        prompts = [str(row["prompt"]) for row in batch_rows]
        outputs = greedy_generate_rsnr(
            model, hook, tok, prompts, [True] * len(prompts), device,
            max_new_tokens=max_new_tokens,
        )
        for source, output in zip(batch_rows, outputs):
            pair = (str(source["matched_subject"]), str(source["matched_relation_id"]))
            aliases = list(true_alias_map[pair])
            abstain = detect_abstention(output)
            leaks = leaked_aliases(output, aliases)
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

    by_group: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = f"{row['split']}_{row['group']}"
        by_group[key]["prompts"] += 1
        by_group[key]["exact_abstentions"] += int(row["exact_abstention"])
        by_group[key]["semantic_abstentions"] += int(row["semantic_abstention"])
        by_group[key]["true_or_alias_leaks"] += int(row["true_or_alias_leak"])
        by_group[key]["abstaining_nondisclosure_passes"] += int(row["abstaining_nondisclosure_pass"])

    total = len(rows)
    exact = sum(bool(row["exact_abstention"]) for row in rows)
    semantic = sum(bool(row["semantic_abstention"]) for row in rows)
    leaks = sum(bool(row["true_or_alias_leak"]) for row in rows)
    abstain_clean = sum(bool(row["abstaining_nondisclosure_pass"]) for row in rows)
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
        "abstaining_nondisclosure_pass_count": abstain_clean,
        "abstaining_nondisclosure_pass_rate": abstain_clean / total if total else None,
        "by_split_group": {key: dict(value) for key, value in sorted(by_group.items())},
        "per_prompt": rows,
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = _load_manifest(protocol_dir)
    locked_forget = _load_locked_forget(protocol_dir)
    sidecar = _load_sidecar(run_dir)
    completion = _load_completion(run_dir)
    adapter_payload, adapter_path = _load_adapter(run_dir)

    artifact_validation = validate_artifact_correspondence(
        adapter_payload=adapter_payload,
        sidecar=sidecar,
        completion=completion,
        locked_forget=locked_forget,
        manifest=manifest,
        expected_count=args.unlearn_num,
    )
    membership = _membership_rows(locked_forget, source="training_visible_forget_direct")
    router = OraclePromptRouter(membership)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RSNR official MCF evaluation")
    base_model = str(adapter_payload["base_model"])
    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = official.dtype_from_str(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    layers = rsnr.get_decoder_layers(model)
    layer_index = int(adapter_payload["layer_index"])
    if layer_index < 0 or layer_index >= len(layers):
        raise RuntimeError("saved RSNR intervention layer is invalid for Base model")
    adapter = rsnr.NullResidualAdapter(
        int(adapter_payload["hidden_size"]),
        int(adapter_payload["adapter_rank"]),
        float(adapter_payload["adapter_alpha"]),
        device,
    ).to(device)
    adapter.load_state_dict(adapter_payload["adapter_state_dict"])
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    hook = rsnr.OracleNullHook.install(layers[layer_index], adapter)

    data = official.load_mcf(official.download_mcf(args.mcf_path))
    forget_records, retain_records, selection = fresh_split(
        data,
        manifest,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        fresh_retain_seed=args.fresh_retain_seed,
    )
    # Full official records must also correspond to the locked membership.
    official_membership = _membership_rows(forget_records, source="official forget sample")
    if official_membership != membership:
        raise RuntimeError("official forget sample pair metadata does not match locked RSNR artifacts")

    true_alias_map = build_true_alias_map(data, forget_records)
    llama_like = official.is_llama_like(model, tok)
    forget_summary, forget_raw, forget_routing, forget_sensitive = evaluate_split(
        model, hook, tok, forget_records, router, device,
        llama_like=llama_like, split_name="forget"
    )
    retain_summary, retain_raw, retain_routing, retain_sensitive = evaluate_split(
        model, hook, tok, retain_records, router, device,
        llama_like=llama_like, split_name="retain"
    )
    sensitive_prompts = [*forget_sensitive, *retain_sensitive]

    native_teacher = rsnr_native_teacher_forced_audit(
        model, hook, tok, sensitive_prompts, true_alias_map, device,
        batch_size=args.generation_batch_size,
    )
    generated = generation_audit(
        model, hook, tok, sensitive_prompts, true_alias_map, device,
        max_new_tokens=args.generation_max_new_tokens,
        batch_size=args.generation_batch_size,
    )

    ppl = None
    if not args.skip_ppl:
        ppl_text = official.load_official_ppl_text(args.wikidata_dir)
        if ppl_text is None:
            print(f"[warning] wikidata dir {args.wikidata_dir} not found. PPL set to null.")
        else:
            hook.clear()
            ppl = official.official_perplexity(model, tok, ppl_text, device, max_input_length=100)

    retain_pair_overlap = sum(_fact_key(row) in router.forget_pairs for row in retain_records)
    result = {
        "method": "rsnr_v1a_oracle_fresh_retain_idk_aware",
        "model_dir": base_model,
        "adapter_path": str(adapter_path),
        "dataset": "MCF",
        "sample_mode": "official_compatible_fresh_disjoint_retain",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "retain_num": int(args.retain_num),
        "development_only": True,
        "oracle_gate": True,
        "atomic_query_scope": True,
        "artifact_validation": artifact_validation,
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "rsnr_native_teacher_forced": native_teacher,
        "rsnr_generation_audit": generated,
        "routing_audit": {
            "forget": forget_routing,
            "retain": retain_routing,
            "sensitive_prompt_count_total": len(sensitive_prompts),
            "fresh_retain_records_matching_a_forget_pair": int(retain_pair_overlap),
            "routing_policy": "per_prompt_subject_resolution_plus_relation_metadata",
            "neighborhood_gate_policy": "per_prompt; ON when prompt resolves to any forgotten (subject, relation) pair",
            "ppl_gate_policy": "off_atomic_query_scope",
        },
        "fresh_retain_selection": selection,
        "claim_boundary": {
            "relation_scoped_behavioral_suppression": True,
            "latent_knowledge_erasure_claimed": False,
            "oracle_gate_not_learned": True,
            "target_new_used_for_training": False,
            "official_eff_gen_still_compare_target_true_vs_target_new": True,
            "idk_generation_directly_audited": True,
            "true_answer_alias_generation_leakage_audited": True,
        },
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selection_path = out.with_name(out.stem + "_fresh_retain_manifest.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    summary = official.result_to_comparison_row(result)
    print(json.dumps(summary, indent=2))
    print(json.dumps({
        "artifact_validation": artifact_validation,
        "routing_audit": result["routing_audit"],
        "rsnr_native_teacher_forced": {
            key: value for key, value in native_teacher.items() if key != "per_prompt"
        },
        "rsnr_generation_audit": {
            key: value for key, value in generated.items() if key != "per_prompt"
        },
    }, indent=2))
    print(f"Official-compatible + IDK-aware result: {out}")
    print(f"Fresh retain manifest: {selection_path}")
    hook.remove()


if __name__ == "__main__":
    main()
