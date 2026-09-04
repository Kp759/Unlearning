#!/usr/bin/env python3
"""Official-compatible MCF Eff/Gen/Spe/PPL evaluation for RSNR-V1A.

This evaluator reloads the frozen Base model plus the saved RSNR oracle-null
adapter.  Routing is defined by exact (subject, relation) membership from the
saved sidecar/checkpoint:

* matching rewrite/canonical and paraphrase queries: adapter ON
* neighborhood queries: adapter OFF (other subjects; specificity probe)
* non-matching retain records: adapter OFF
* Wikidata PPL: adapter OFF (RSNR-V1A is scoped to atomic factual queries)

It preserves the repository's ZeroUnlearn-compatible Eff/Gen/Spe formulas and
uses the same fresh, disjoint 1000-retain sampling protocol as the V1.1/V1.3
development evaluator.  Seed 1 is DEVELOPMENT ONLY because official aggregate
metrics for this seed have already been consumed in earlier experiments.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official
import run_mcf_rsnr_v1a_oracle as rsnr
from mcf_sampling import sample_official_mcf_records


PROTOCOL = rsnr.PROTOCOL


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
    args = p.parse_args()
    if args.seed != 1 or args.unlearn_num != 50 or args.retain_num != 1000:
        p.error("RSNR-V1A development evaluation is locked to seed=1, forget=50, retain=1000")
    return args


def _load_manifest(protocol_dir: Path) -> Dict[str, Any]:
    path = protocol_dir / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("seed", -1)) != 1:
        raise RuntimeError("RSNR-V1A eval expects the locked seed-1 protocol")
    return payload


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


def _load_adapter(run_dir: Path, device: torch.device):
    path = run_dir / "method" / "rsnr_oracle_null_adapter.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("RSNR adapter protocol mismatch")
    return payload, path


def _load_sidecar(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "method" / "relation_scoped_null_routing.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL:
        raise RuntimeError("RSNR routing sidecar protocol mismatch")
    if payload.get("routing") != "oracle_exact_subject_relation_membership":
        raise RuntimeError("RSNR-V1A evaluator requires oracle exact-pair routing")
    return payload


def _fact_key(record: Mapping[str, Any]) -> tuple[str, str]:
    rr = record["requested_rewrite"]
    return str(rr["subject"]), str(rr["relation_id"])


def routing_flags_for_record(
    record: Mapping[str, Any], forget_pairs: set[tuple[str, str]]
) -> Dict[str, Any]:
    """Return oracle routing for official prompt groups.

    Rewrite/paraphrase prompts inherit the record's (subject, relation) pair.
    Neighborhood prompts are specificity probes for other subjects and must stay
    on the exact Base path even when the parent record is a forget fact.
    """
    sensitive = _fact_key(record) in forget_pairs
    return {
        "rewrite": bool(sensitive),
        "paraphrase": bool(sensitive),
        "neighborhood": False,
    }


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
    """ZeroUnlearn-compatible target NLLs with per-prefix RSNR oracle routing."""
    if len(prefixes) == 0:
        return []
    if len(prefixes) != len(gated_flags):
        raise ValueError("prefix/gate length mismatch")

    raw_prefix_lens = [len(x) for x in tok(list(prefixes))["input_ids"]]
    texts = [
        f"{prefix} {suffix}"
        for prefix in prefixes
        for suffix in (target_new, target_true)
    ]
    batch = tok(texts, padding=True, return_tensors="pt").to(device)

    a_tok, b_tok = (tok(f" {x}")["input_ids"] for x in (target_new, target_true))
    score_prefix_lens = list(raw_prefix_lens)
    if llama_like:
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        score_prefix_lens = [x - 1 for x in score_prefix_lens]

    # The null intervention is applied only at hidden positions whose logits
    # score the target completion, matching RSNR-V1A training geometry.
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
        cur_len = len(cur_tokens)
        for j in range(cur_len):
            cur_tok = cur_tokens[j]
            pos = score_prefix_lens[i // 2] + j - 1
            probs[i] += -torch.nn.functional.log_softmax(logits[i, pos, :], dim=0)[cur_tok].item()
        probs[i] /= max(1, cur_len)

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
    forget_pairs: set[tuple[str, str]],
    device: torch.device,
    *,
    llama_like: bool,
):
    rr = record["requested_rewrite"]
    subject = str(rr["subject"])
    target_new = str(rr["target_new"]["str"])
    target_true = str(rr["target_true"]["str"])
    groups = {
        "rewrite": [str(rr["prompt"]).format(subject)],
        "paraphrase": list(record.get("paraphrase_prompts", [])),
        "neighborhood": list(record.get("neighborhood_prompts", [])),
    }
    flags = routing_flags_for_record(record, forget_pairs)
    out = {}
    for name, prompts in groups.items():
        out[f"{name}_prompts_probs"] = rsnr_test_batch_prediction(
            model,
            hook,
            tok,
            prompts,
            target_new,
            target_true,
            [bool(flags[name])] * len(prompts),
            device,
            llama_like=llama_like,
        )
    return out, flags


def evaluate_split(
    model: Any,
    hook: rsnr.OracleNullHook,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    forget_pairs: set[tuple[str, str]],
    device: torch.device,
    *,
    llama_like: bool,
    split_name: str,
):
    metrics = []
    counts = {
        "rewrite_total": 0,
        "rewrite_gated": 0,
        "paraphrase_total": 0,
        "paraphrase_gated": 0,
        "neighborhood_total": 0,
        "neighborhood_gated": 0,
        "matching_fact_records": 0,
    }
    for record in records:
        post, flags = compute_record(
            model, hook, tok, record, forget_pairs, device, llama_like=llama_like
        )
        if _fact_key(record) in forget_pairs:
            counts["matching_fact_records"] += 1
        for group in ("rewrite", "paraphrase", "neighborhood"):
            n = len(post[f"{group}_prompts_probs"])
            counts[f"{group}_total"] += n
            if flags[group]:
                counts[f"{group}_gated"] += n
        metrics.append({"requested_rewrite": record["requested_rewrite"], "post": post})
    return official.official_summarize(split_name, metrics), metrics, counts


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


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    protocol_dir = Path(args.protocol_dir).resolve()
    manifest = _load_manifest(protocol_dir)
    sidecar = _load_sidecar(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for RSNR official MCF evaluation")
    adapter_payload, adapter_path = _load_adapter(run_dir, device)
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

    checkpoint_membership = {
        (str(row["subject"]), str(row["relation_id"]))
        for row in adapter_payload["forget_membership"]
    }
    sidecar_membership = {
        (str(row["subject"]), str(row["relation_id"]))
        for row in sidecar["forget_membership"]
    }
    if checkpoint_membership != sidecar_membership:
        raise RuntimeError("adapter and routing sidecar forget membership disagree")
    forget_pairs = checkpoint_membership

    data = official.load_mcf(official.download_mcf(args.mcf_path))
    forget_records, retain_records, selection = fresh_split(
        data,
        manifest,
        unlearn_num=args.unlearn_num,
        retain_num=args.retain_num,
        seed=args.seed,
        fresh_retain_seed=args.fresh_retain_seed,
    )

    llama_like = official.is_llama_like(model, tok)
    forget_summary, forget_raw, forget_routing = evaluate_split(
        model, hook, tok, forget_records, forget_pairs, device,
        llama_like=llama_like, split_name="forget"
    )
    retain_summary, retain_raw, retain_routing = evaluate_split(
        model, hook, tok, retain_records, forget_pairs, device,
        llama_like=llama_like, split_name="retain"
    )

    ppl = None
    if not args.skip_ppl:
        ppl_text = official.load_official_ppl_text(args.wikidata_dir)
        if ppl_text is None:
            print(f"[warning] wikidata dir {args.wikidata_dir} not found. PPL set to null.")
        else:
            # RSNR-V1A is defined for atomic factual queries; generic PPL text is
            # explicitly gate OFF and therefore follows the exact frozen Base path.
            hook.clear()
            ppl = official.official_perplexity(model, tok, ppl_text, device, max_input_length=100)

    retain_pair_overlap = sum(_fact_key(row) in forget_pairs for row in retain_records)
    result = {
        "method": "rsnr_v1a_oracle_fresh_retain",
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
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "routing_audit": {
            "forget": forget_routing,
            "retain": retain_routing,
            "fresh_retain_records_matching_a_forget_pair": int(retain_pair_overlap),
            "neighborhood_gate_policy": "always_off_other_subject_specificity_probe",
            "ppl_gate_policy": "off_atomic_query_scope",
        },
        "fresh_retain_selection": selection,
        "claim_boundary": {
            "relation_scoped_behavioral_suppression": True,
            "latent_knowledge_erasure_claimed": False,
            "oracle_gate_not_learned": True,
            "target_new_used_for_training": False,
            "official_eff_gen_still_compare_target_true_vs_target_new": True,
        },
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selection_path = out.with_name(out.stem + "_fresh_retain_manifest.json")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")

    summary = official.result_to_comparison_row(result)
    print(json.dumps(summary, indent=2))
    print(json.dumps({"routing_audit": result["routing_audit"]}, indent=2))
    print(f"Official-compatible result: {out}")
    print(f"Fresh retain manifest: {selection_path}")
    hook.remove()


if __name__ == "__main__":
    main()
