#!/usr/bin/env python3
"""Trace the post-hoc uniform-beta frontier of a frozen compositional edit.

This diagnostic keeps the learned sparse embedding writer fixed and replaces
the selected LM-head rows with

    W_y(scale) = W_y(base) + scale * (W_y(edited) - W_y(base)).

For each predeclared scale it measures output-only PPL, combined PPL, and the
official forget Eff/Gen/Spe split. It does not train or save a checkpoint.
Because it reads official MCF/PPL probes, its selected scale is exploratory and
must not be reported as a confirmatory held-out result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_zero_unlearn_official_eval as official


DEFAULT_SCALES = "0,0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1"


def parse_scales(value: str) -> list[float]:
    values: list[float] = []
    for piece in str(value).split(","):
        piece = piece.strip()
        if not piece:
            continue
        scale = float(piece)
        if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("beta scales must be finite and in [0, 1]")
        if scale not in values:
            values.append(scale)
    if not values:
        raise ValueError("at least one beta scale is required")
    return sorted(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--scales", default=DEFAULT_SCALES)
    parser.add_argument("--ppl-limit-percent", type=float, default=5.0)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--device-map", choices=("single", "auto"), default="single"
    )
    value = parser.parse_args(list(argv) if argv is not None else None)
    try:
        value.parsed_scales = parse_scales(value.scales)
    except ValueError as exc:
        parser.error(str(exc))
    if int(value.forget_num) <= 0:
        parser.error("--forget-num must be positive")
    if not math.isfinite(float(value.ppl_limit_percent)) or float(
        value.ppl_limit_percent
    ) < 0.0:
        parser.error("--ppl-limit-percent must be finite and non-negative")
    return value


def _required_rows(state: Mapping[str, Any], key: str) -> list[int]:
    values = state.get(key)
    if not isinstance(values, (list, tuple)) or not values:
        raise RuntimeError(f"method state lacks non-empty {key!r}")
    return [int(value) for value in values]


@torch.no_grad()
def selected_rows(layer: torch.nn.Module, token_ids: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(token_ids, dtype=torch.long, device=layer.weight.device)
    return layer.weight.index_select(0, index).detach().cpu()


@torch.no_grad()
def replace_rows(
    layer: torch.nn.Module,
    token_ids: Sequence[int],
    values: torch.Tensor,
) -> None:
    if values.ndim != 2 or values.shape[0] != len(token_ids):
        raise ValueError("row values do not match token ids")
    index = torch.tensor(token_ids, dtype=torch.long, device=layer.weight.device)
    layer.weight.index_copy_(
        0,
        index,
        values.to(device=layer.weight.device, dtype=layer.weight.dtype),
    )


def relative_row_norms(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if delta.shape != base.shape or delta.ndim != 2:
        raise ValueError("delta/base row matrices must have equal shape")
    return delta.float().norm(dim=1) / base.float().norm(dim=1).clamp_min(1e-12)


def distribution(values: torch.Tensor) -> Dict[str, Any]:
    flat = values.detach().float().flatten().cpu()
    if not flat.numel():
        return {"n": 0}
    ordered = flat.sort().values

    def quantile(fraction: float) -> float:
        position = round(fraction * (len(ordered) - 1))
        return float(ordered[position])

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p10": quantile(0.10),
        "median": quantile(0.50),
        "p90": quantile(0.90),
        "max": float(ordered[-1]),
        "mean": float(ordered.mean()),
        "values": [float(value) for value in ordered],
    }


def percent_delta(value: float, base: float) -> float:
    return 100.0 * (float(value) - float(base)) / float(base)


def choose_largest_ppl_safe_scale(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit_percent: float,
) -> float | None:
    safe = [
        float(row["scale"])
        for row in rows
        if abs(float(row["output_only_ppl_percent_delta"]))
        <= float(limit_percent) + 1e-12
    ]
    return max(safe) if safe else None


def _load_model(path: Path, dtype: torch.dtype, device_map: str):
    kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    if device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    model.eval()
    model.config.use_cache = False
    return model


def _failed_paraphrase_subjects(raw_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    failed: list[str] = []
    for row in raw_rows:
        values = row.get("post", {}).get("paraphrase_prompts_probs", [])
        if any(
            float(item["target_true"]) < float(item["target_new"])
            for item in values
        ):
            failed.append(str(row.get("requested_rewrite", {}).get("subject", "")))
    return failed


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    base_model_path = Path(args.base_model_path).resolve()
    state_path = Path(args.state).resolve()
    for path in (model_dir, base_model_path, state_path, Path(args.wikidata_dir)):
        if not path.exists():
            raise FileNotFoundError(path)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("method state must be a mapping")
    input_ids = _required_rows(state, "selected_embedding_rows")
    output_ids = _required_rows(state, "selected_output_rows")
    dtype = official.dtype_from_str(args.dtype)

    print("Loading exact base selected rows on CPU")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path), torch_dtype=dtype
    )
    base_input = selected_rows(base_model.get_input_embeddings(), input_ids)
    base_output = selected_rows(base_model.get_output_embeddings(), output_ids)
    del base_model
    gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = _load_model(model_dir, dtype, args.device_map)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("frontier requires the checkpoint's untied LM head")
    edited_input = selected_rows(input_layer, input_ids)
    edited_output = selected_rows(output_layer, output_ids)
    input_delta = edited_input.float() - base_input.float()
    output_delta = edited_output.float() - base_output.float()

    saved_output_delta = state.get("output_delta")
    state_delta_abs_error = None
    if isinstance(saved_output_delta, torch.Tensor) and saved_output_delta.shape == output_delta.shape:
        state_delta_abs_error = float(
            (saved_output_delta.float() - output_delta).abs().max()
        )

    forget_records, _ = official.load_official_eval_records(
        args.mcf_path,
        int(args.forget_num),
        0,
        int(args.seed),
        "official",
    )
    ppl_text = official.load_official_ppl_text(args.wikidata_dir)
    if ppl_text is None:
        raise RuntimeError(f"could not load PPL text from {args.wikidata_dir}")
    device = next(model.parameters()).device
    llama_like = official.is_llama_like(model, tokenizer)

    rows: list[Dict[str, Any]] = []
    try:
        # Reconstruct the exact base inside the edited model to establish the
        # denominator independently of any previously serialized JSON.
        replace_rows(input_layer, input_ids, base_input)
        replace_rows(output_layer, output_ids, base_output)
        reconstructed_base_ppl = float(
            official.official_perplexity(model, tokenizer, ppl_text, device)
        )

        for scale in args.parsed_scales:
            scaled_output = base_output.float() + float(scale) * output_delta

            replace_rows(input_layer, input_ids, base_input)
            replace_rows(output_layer, output_ids, scaled_output)
            output_only_ppl = float(
                official.official_perplexity(model, tokenizer, ppl_text, device)
            )

            replace_rows(input_layer, input_ids, edited_input)
            replace_rows(output_layer, output_ids, scaled_output)
            combined_ppl = float(
                official.official_perplexity(model, tokenizer, ppl_text, device)
            )
            forget_summary, forget_raw = official.evaluate_record_split(
                model,
                tokenizer,
                forget_records,
                device,
                llama_like,
                "forget",
            )
            row = {
                "scale": float(scale),
                "output_only_ppl": output_only_ppl,
                "output_only_ppl_percent_delta": percent_delta(
                    output_only_ppl, reconstructed_base_ppl
                ),
                "combined_ppl": combined_ppl,
                "combined_ppl_percent_delta": percent_delta(
                    combined_ppl, reconstructed_base_ppl
                ),
                "Eff": float(forget_summary["Eff"]),
                "Gen": float(forget_summary["Gen"]),
                "Spe": float(forget_summary["Spe"]),
                "Spe_success": float(forget_summary["Spe_success"]),
                "rewrite_failures": int(
                    forget_summary["post_rewrite_failure_prompt_instances"]
                ),
                "paraphrase_failures": int(
                    forget_summary["post_paraphrase_failure_prompt_instances"]
                ),
                "failed_paraphrase_subjects": _failed_paraphrase_subjects(
                    forget_raw
                ),
            }
            rows.append(row)
            print(
                f"scale={scale:>7g} output-PPL={output_only_ppl:.6g} "
                f"combined-PPL={combined_ppl:.6g} Eff={row['Eff']:.2f} "
                f"Gen={row['Gen']:.2f} Spe={row['Spe']:.2f}"
            )
    finally:
        replace_rows(input_layer, input_ids, edited_input)
        replace_rows(output_layer, output_ids, edited_output)

    safe_scale = choose_largest_ppl_safe_scale(
        rows, limit_percent=float(args.ppl_limit_percent)
    )
    safe_row = next(
        (row for row in rows if float(row["scale"]) == safe_scale), None
    )
    payload = {
        "schema_version": 1,
        "kind": "mcf_compositional_uniform_beta_frontier",
        "source": {
            "model_dir": str(model_dir),
            "base_model_path": str(base_model_path),
            "state": str(state_path),
            "protocol": state.get("protocol"),
        },
        "sparse_rows": {
            "input_embedding": len(input_ids),
            "lm_head": len(output_ids),
        },
        "norms": {
            "input_delta_frobenius": float(input_delta.norm()),
            "output_delta_frobenius": float(output_delta.norm()),
            "input_relative_row_norm": distribution(
                relative_row_norms(input_delta, base_input)
            ),
            "output_relative_row_norm": distribution(
                relative_row_norms(output_delta, base_output)
            ),
            "saved_output_delta_abs_error_max": state_delta_abs_error,
        },
        "reconstructed_base_ppl": reconstructed_base_ppl,
        "ppl_limit_percent": float(args.ppl_limit_percent),
        "largest_ppl_safe_scale": safe_scale,
        "largest_ppl_safe_result": safe_row,
        "frontier": rows,
        "diagnostic_only": True,
        "used_for_training_or_checkpoint_selection": False,
        "confirmatory_claim_allowed": False,
        "reason": (
            "Official MCF and PPL probes were used to characterize and select "
            "the displayed scale. A future cap must be frozen on disjoint data."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"largest |PPL delta| <= {args.ppl_limit_percent:g}% scale: {safe_scale}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
