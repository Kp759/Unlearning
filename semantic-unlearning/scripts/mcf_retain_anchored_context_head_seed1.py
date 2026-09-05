#!/usr/bin/env python3
"""Locked Seed-1 MCF pilot for the retain-anchored contextual output head.

Training-visible information:
  * official sampled rewrite prompt + subject
  * target_true answer, used only to identify the sensitive continuation tokens
  * sampled retain rewrite prompt + target_true for protection anchors

The fit path intentionally does NOT read official paraphrase prompts,
neighborhood prompts, or target_new.  Those fields are opened only by the
unchanged official evaluator after the contextual head has been fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import mcf_zero_unlearn_official_eval as official_eval  # noqa: E402
from mcf_sampling import sample_official_mcf_records  # noqa: E402
from retain_anchored_context_head import (  # noqa: E402
    AnchoredFeatureMap,
    ContextualCorrectionModel,
    FactIndexedLogitCorrection,
    FrozenRandomProjector,
)


@dataclass
class SequenceSpec:
    text: str
    event_positions: List[int]
    event_token_ids: List[int]
    record_index: int


def dtype_from_str(value: str) -> torch.dtype:
    value = str(value).lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def _normalize_record(record):
    return official_eval.normalize_record(record)


def _answer_token_ids(tokenizer, answer: str) -> List[int]:
    # add_special_tokens=False is equivalent to the evaluator's Llama BOS strip
    # for the continuation itself and avoids architecture-specific BOS logic.
    return list(tokenizer(f" {answer}", add_special_tokens=False)["input_ids"])


def _sequence_spec(record, tokenizer, record_index: int, max_events: int | None) -> SequenceSpec:
    rr = record["requested_rewrite"]
    prefix = rr["prompt"].format(rr["subject"])
    target_true = rr["target_true"]["str"]
    target_ids = _answer_token_ids(tokenizer, target_true)
    if not target_ids:
        raise ValueError(f"record {record_index} has empty target_true tokenization")

    text = f"{prefix} {target_true}"
    full_ids = list(tokenizer(text)["input_ids"])
    if len(full_ids) < len(target_ids) or full_ids[-len(target_ids) :] != target_ids:
        raise ValueError(
            "target_true tokenization is not an exact suffix of the formatted rewrite; "
            f"record_index={record_index}"
        )
    answer_start = len(full_ids) - len(target_ids)
    n_events = len(target_ids) if max_events is None else min(len(target_ids), int(max_events))
    positions = [answer_start + j - 1 for j in range(n_events)]
    if min(positions) < 0:
        raise ValueError(f"record {record_index} has invalid prediction position")
    return SequenceSpec(
        text=text,
        event_positions=positions,
        event_token_ids=target_ids[:n_events],
        record_index=int(record_index),
    )


def build_specs(records, tokenizer, *, max_events_per_record: int | None) -> List[SequenceSpec]:
    return [
        _sequence_spec(record, tokenizer, i, max_events=max_events_per_record)
        for i, record in enumerate(records)
    ]


@torch.no_grad()
def extract_event_descriptors(
    model,
    tokenizer,
    projector: FrozenRandomProjector,
    specs: Sequence[SequenceSpec],
    *,
    batch_size: int,
    device: torch.device,
):
    descriptors = []
    token_ids: List[int] = []
    event_records: List[int] = []
    model.eval()
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(specs), int(batch_size)):
            batch_specs = list(specs[start : start + int(batch_size)])
            encoded = tokenizer(
                [spec.text for spec in batch_specs],
                padding=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            selected_hidden = []
            for row, spec in enumerate(batch_specs):
                seq_len = int(encoded["attention_mask"][row].sum().item())
                for position, token_id in zip(spec.event_positions, spec.event_token_ids):
                    if position >= seq_len:
                        raise RuntimeError(
                            f"event position {position} >= sequence length {seq_len}"
                        )
                    selected_hidden.append(hidden[row, position, :])
                    token_ids.append(int(token_id))
                    event_records.append(int(spec.record_index))
            if selected_hidden:
                selected_hidden = torch.stack(selected_hidden, dim=0)
                descriptors.append(projector(selected_hidden).float())
    finally:
        tokenizer.padding_side = old_padding_side

    if not descriptors:
        raise RuntimeError("no contextual events were extracted")
    return torch.cat(descriptors, dim=0), token_ids, event_records


def _hard_overlap_records(retain_records, tokenizer, sensitive_token_ids: set[int]):
    hard = []
    for record in retain_records:
        target_true = record["requested_rewrite"]["target_true"]["str"]
        ids = set(_answer_token_ids(tokenizer, target_true))
        if ids & sensitive_token_ids:
            hard.append(record)
    return hard


def _evaluate_subset(model, tokenizer, records, *, split_name: str):
    if not records:
        return None, []
    device = next(model.parameters()).device
    llama_like = official_eval.is_llama_like(model, tokenizer)
    return official_eval.evaluate_record_split(
        model, tokenizer, records, device, llama_like, split_name
    )


def save_sidecar(
    path: Path,
    *,
    projector,
    feature_map,
    correction,
    forget_event_token_ids,
    forget_event_records,
    retain_event_records,
    args,
):
    payload = {
        "schema_version": 1,
        "kind": "retain_anchored_context_head_seed1",
        "seed": int(args.seed),
        "forget_num": int(args.forget_num),
        "retain_num": int(args.retain_num),
        "descriptor_dim": int(args.descriptor_dim),
        "radius": float(args.radius),
        "logit_penalty": float(args.logit_penalty),
        "retain_jitter": float(args.retain_jitter),
        "cardinal_jitter": float(args.cardinal_jitter),
        "projection": projector.matrix.detach().cpu(),
        "retain_descriptors": feature_map.retain.detach().cpu(),
        "forget_descriptors": feature_map.forget.detach().cpu(),
        "selected_token_ids": correction.selected_token_ids.detach().cpu(),
        "coefficients": correction.coefficients.detach().cpu(),
        "fact_enabled": correction.fact_enabled.detach().cpu(),
        "forget_event_token_ids": list(map(int, forget_event_token_ids)),
        "forget_event_records": list(map(int, forget_event_records)),
        "retain_event_records": list(map(int, retain_event_records)),
        "training_contract": {
            "used_rewrite_prompt": True,
            "used_target_true": True,
            "used_target_new": False,
            "used_official_paraphrases": False,
            "used_official_neighborhoods": False,
        },
    }
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mcf-path", required=True)
    ap.add_argument("--wikidata-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--forget-num", type=int, default=50)
    ap.add_argument("--retain-num", type=int, default=1000)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--descriptor-dim", type=int, default=32)
    ap.add_argument("--projection-seed", type=int, default=1729)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--logit-penalty", type=float, default=12.0)
    ap.add_argument("--retain-jitter", type=float, default=1e-4)
    ap.add_argument("--cardinal-jitter", type=float, default=1e-4)
    ap.add_argument("--retain-events-per-record", type=int, default=1)
    ap.add_argument("--extract-batch-size", type=int, default=16)
    ap.add_argument("--alpha-chunk-size", type=int, default=256)
    ap.add_argument("--eval-base", action="store_true")
    ap.add_argument("--skip-ppl", action="store_true")
    args = ap.parse_args()

    if args.seed != 1:
        raise ValueError("This pilot is intentionally locked to development seed 1")
    if args.forget_num != 50 or args.retain_num != 1000:
        raise ValueError("Seed-1 pilot is locked to forget=50 and retain=1000")
    if args.descriptor_dim <= 0 or args.radius <= 0 or args.logit_penalty <= 0:
        raise ValueError("descriptor_dim, radius, and logit_penalty must be positive")

    model_path = Path(args.model_path)
    mcf_path = Path(args.mcf_path)
    wikidata_dir = Path(args.wikidata_dir)
    for label, path in (("model", model_path), ("MCF", mcf_path), ("Wikidata", wikidata_dir)):
        if not path.exists():
            raise FileNotFoundError(f"local {label} path does not exist: {path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype_from_str(args.dtype),
        local_files_only=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    with mcf_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    forget_records, retain_records = sample_official_mcf_records(
        data,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        strict=True,
    )
    forget_records = [_normalize_record(r) for r in forget_records]
    retain_records = [_normalize_record(r) for r in retain_records]

    # Build fit artifacts from rewrite+target_true only.  No official paraphrase,
    # neighborhood, or target_new field is inspected before the head is fixed.
    forget_specs = build_specs(forget_records, tokenizer, max_events_per_record=None)
    retain_specs = build_specs(
        retain_records,
        tokenizer,
        max_events_per_record=args.retain_events_per_record,
    )

    hidden_size = int(model.config.hidden_size)
    projector = FrozenRandomProjector(
        input_dim=hidden_size,
        output_dim=args.descriptor_dim,
        seed=args.projection_seed,
        device=device,
        dtype=torch.float32,
    )

    forget_desc, forget_event_token_ids, forget_event_records = extract_event_descriptors(
        model,
        tokenizer,
        projector,
        forget_specs,
        batch_size=args.extract_batch_size,
        device=device,
    )
    retain_desc, _, retain_event_records = extract_event_descriptors(
        model,
        tokenizer,
        projector,
        retain_specs,
        batch_size=args.extract_batch_size,
        device=device,
    )

    feature_map = AnchoredFeatureMap.fit(
        retain=retain_desc.float(),
        forget=forget_desc.float(),
        radius=args.radius,
        retain_jitter=args.retain_jitter,
        cardinal_jitter=args.cardinal_jitter,
    )
    selected_token_ids = sorted(set(map(int, forget_event_token_ids)))
    correction = FactIndexedLogitCorrection(
        feature_map=feature_map,
        selected_token_ids=selected_token_ids,
        vocab_size=int(model.config.vocab_size),
    )
    token_to_row = {
        int(token_id): row
        for row, token_id in enumerate(correction.selected_token_ids.tolist())
    }
    with torch.no_grad():
        correction.coefficients.zero_()
        for event_index, token_id in enumerate(forget_event_token_ids):
            correction.coefficients[token_to_row[int(token_id)], event_index] = -float(
                args.logit_penalty
            )

    diagnostics = correction.diagnostics()
    diagnostics_json = {
        "max_abs_retain_alpha": diagnostics.max_abs_retain_alpha,
        "max_abs_cardinal_error": diagnostics.max_abs_cardinal_error,
        "num_forget_events": int(forget_desc.shape[0]),
        "num_retain_anchors": int(retain_desc.shape[0]),
        "num_selected_tokens": diagnostics.num_selected_tokens,
        "descriptor_dim": int(args.descriptor_dim),
        "radius": float(args.radius),
        "logit_penalty": float(args.logit_penalty),
    }
    (output_dir / "fit_diagnostics.json").write_text(
        json.dumps(diagnostics_json, indent=2) + "\n", encoding="utf-8"
    )

    sensitive_token_ids = set(map(int, forget_event_token_ids))
    hard_overlap_records = _hard_overlap_records(
        retain_records, tokenizer, sensitive_token_ids
    )

    base_result = None
    base_hard_summary = None
    if args.eval_base:
        base_result = official_eval.evaluate_loaded_model_official(
            method="base",
            model=model,
            tok=tokenizer,
            model_dir=model_path,
            mcf_path=mcf_path,
            wikidata_dir=wikidata_dir,
            out_path=output_dir / "base_official_eval.json",
            unlearn_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            sample_mode="official",
            skip_ppl=args.skip_ppl,
        )
        base_hard_summary, _ = _evaluate_subset(
            model, tokenizer, hard_overlap_records, split_name="hard_overlap_retain_base"
        )

    wrapped = ContextualCorrectionModel(
        base_model=model,
        projector=projector,
        correction=correction,
        alpha_chunk_size=args.alpha_chunk_size,
    ).to(device)
    wrapped.eval()

    corrected_result = official_eval.evaluate_loaded_model_official(
        method="retain_anchored_context_head",
        model=wrapped,
        tok=tokenizer,
        model_dir=model_path,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "context_head_official_eval.json",
        unlearn_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode="official",
        skip_ppl=args.skip_ppl,
    )
    hard_summary, hard_raw = _evaluate_subset(
        wrapped, tokenizer, hard_overlap_records, split_name="hard_overlap_retain"
    )

    save_sidecar(
        output_dir / "context_head_sidecar.pt",
        projector=projector,
        feature_map=feature_map,
        correction=correction,
        forget_event_token_ids=forget_event_token_ids,
        forget_event_records=forget_event_records,
        retain_event_records=retain_event_records,
        args=args,
    )

    summary = {
        "schema_version": 1,
        "kind": "mcf_seed1_retain_anchored_context_head_pilot",
        "training_contract": {
            "seed": 1,
            "forget_num": 50,
            "retain_num": 1000,
            "sample_mode": "official",
            "target_new_used_for_fit": False,
            "official_paraphrases_used_for_fit": False,
            "official_neighborhoods_used_for_fit": False,
            "transformer_frozen": True,
            "base_embeddings_modified": False,
            "base_lm_head_modified": False,
        },
        "fit_diagnostics": diagnostics_json,
        "hard_overlap_retain_records": len(hard_overlap_records),
        "base": base_result,
        "base_hard_overlap": base_hard_summary,
        "context_head": corrected_result,
        "context_head_hard_overlap": hard_summary,
        "context_head_hard_overlap_raw": hard_raw,
    }
    (output_dir / "seed1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    def compact(result):
        if result is None:
            return None
        return {
            "Eff": result["forget"]["Eff"],
            "Gen": result["forget"]["Gen"],
            "Spe": result["forget"]["Spe"],
            "Spe_success": result["forget"].get("Spe_success"),
            "PPL": result.get("retain_PPL"),
            "retain_Eff": result["retain"]["Eff"],
            "retain_Gen": result["retain"]["Gen"],
            "retain_Spe": result["retain"]["Spe"],
            "retain_Spe_success": result["retain"].get("Spe_success"),
        }

    print(json.dumps({
        "base": compact(base_result),
        "context_head": compact(corrected_result),
        "hard_overlap_retain_records": len(hard_overlap_records),
        "hard_overlap_base": base_hard_summary,
        "hard_overlap_context_head": hard_summary,
        "fit_diagnostics": diagnostics_json,
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
