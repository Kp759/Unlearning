#!/usr/bin/env python3
"""V1.3 multi-view relation-robust private-vocabulary rewiring.

This entrypoint keeps V1.1's architecture and objective, but replaces each
canonical forget case's training margin with the WORST margin across a
training-only multi-view corpus.  The corpus must have been generated solely
from the sanitized direct forget split by
``build_mcf_private_vocab_rewiring_v1_3_training_views.py``.

No official paraphrase/neighborhood/evaluation prompt is accepted by this
process.  Relation-aware inference routing is still forbidden: the same private
subject sequence is used in every context.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import run_mcf_private_vocab_rewiring_v1_1 as runner
import run_mcf_private_vocab_rewiring_v1_1_relational as relational


PROTOCOL = "mcf_private_vocab_rewiring_v1_3_multiview_relation_robust"
VIEW_CORPUS_PROTOCOL = "mcf_private_vocab_rewiring_v1_3_training_multiview_corpus"
_ORIGINAL_MARGIN_BATCH = runner.margin_batch
_VIEW_MAP: Dict[int, list[str]] = {}
_VIEW_CORPUS_META: Dict[str, Any] = {}
_VIEW_CHUNK = 16


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_registry(registry: Mapping[str, Any]) -> None:
    arch = registry.get("architecture", {})
    mv = registry.get("training_multiview", {})
    if (
        registry.get("protocol") != PROTOCOL
        or arch.get("trainable_parameter_families") != ["private_subject_embedding_rows"]
        or arch.get("private_tokens_per_subject_token") != "one_to_one"
        or arch.get("subject_token_count_preserved") is not True
        or arch.get("transformer_frozen") is not True
        or arch.get("lm_head_frozen_bit_identical") is not True
        or arch.get("original_input_embedding_rows_frozen") is not True
        or arch.get("reserved_vocab_slots_repurposed") is not True
        or arch.get("relation_aware_router") is not False
        or arch.get("subject_sequence_rewrite") is not True
        or mv.get("heldout_probe_text_used") is not False
        or mv.get("worst_view_objective") is not True
        or int(mv.get("views_per_case", 0)) < 2
    ):
        raise RuntimeError("V1.3 registry contract mismatch")


def load_view_corpus(path: Path) -> tuple[Dict[int, list[str]], Dict[str, Any]]:
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if payload.get("protocol") != VIEW_CORPUS_PROTOCOL:
        raise RuntimeError("V1.3 training-view corpus protocol mismatch")
    leakage = payload.get("leakage_contract", {})
    required_false = (
        "full_mcf_path_accepted",
        "official_paraphrase_prompts_read",
        "official_neighborhood_prompts_read",
        "official_generation_prompts_read",
        "official_retain_records_read",
        "generator_received_target_true",
        "generator_received_target_new",
    )
    if any(leakage.get(key) is not False for key in required_false):
        raise RuntimeError("V1.3 training-view corpus fails leakage contract")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("V1.3 training-view corpus has no cases")
    expected_views = int(payload.get("views_per_case", 0))
    if expected_views < 2:
        raise RuntimeError("V1.3 requires canonical + synthetic training views")

    view_map: Dict[int, list[str]] = {}
    for item in cases:
        case_id = int(item["case_id"])
        if case_id in view_map:
            raise RuntimeError(f"duplicate view-corpus case_id {case_id}")
        views = item.get("views")
        if not isinstance(views, list) or len(views) != expected_views:
            raise RuntimeError(
                f"case {case_id} has {len(views) if isinstance(views, list) else 'invalid'} "
                f"views, expected {expected_views}"
            )
        templates: list[str] = []
        for view in views:
            template = str(view.get("template", ""))
            if template.count("{}") != 1:
                raise RuntimeError(f"case {case_id} has invalid training-view template")
            templates.append(template)
        if len(set(templates)) != len(templates):
            raise RuntimeError(f"case {case_id} has duplicate training views")
        view_map[case_id] = templates

    meta = {
        "corpus_sha256": sha256_bytes(raw_bytes),
        "cases": len(view_map),
        "views_per_case": expected_views,
        "synthetic_views_per_case": int(payload.get("synthetic_views_per_case", expected_views - 1)),
        "generator_seed": int(payload.get("seed", -1)),
        "source_sha256": str(payload.get("source_sha256", "")),
        "leakage_contract": dict(leakage),
        "semantic_filter": dict(payload.get("semantic_filter", {})),
    }
    return view_map, meta


def view_records_for_case(record: Mapping[str, Any]) -> list[Dict[str, Any]]:
    case_id = int(record["case_id"])
    templates = _VIEW_MAP.get(case_id)
    if templates is None:
        raise RuntimeError(f"V1.3 view corpus missing forget case {case_id}")
    out: list[Dict[str, Any]] = []
    for template in templates:
        clone = copy.deepcopy(dict(record))
        clone["requested_rewrite"]["prompt"] = str(template)
        out.append(clone)
    return out


def multiview_worst_margin_batch(
    model: Any,
    prompt_tokenizer: Any,
    base_tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one differentiable worst-view new-vs-true margin per case."""
    flat: list[Dict[str, Any]] = []
    spans: list[tuple[int, int]] = []
    for record in records:
        start = len(flat)
        local = view_records_for_case(record)
        flat.extend(local)
        spans.append((start, len(flat)))

    values: list[torch.Tensor] = []
    for start in range(0, len(flat), int(_VIEW_CHUNK)):
        chunk = flat[start : start + int(_VIEW_CHUNK)]
        values.append(
            _ORIGINAL_MARGIN_BATCH(
                model,
                prompt_tokenizer,
                base_tokenizer,
                chunk,
                device=device,
            )
        )
    all_values = torch.cat(values, dim=0)
    worst = [all_values[start:stop].min() for start, stop in spans]
    return torch.stack(worst)


def verify_protocol_cases(protocol_dir: Path) -> Dict[str, Any]:
    direct = json.loads(
        (protocol_dir / "training_visible_forget_direct.json").read_text(encoding="utf-8")
    )
    direct_ids = [int(row["case_id"]) for row in direct]
    corpus_ids = list(_VIEW_MAP)
    if set(direct_ids) != set(corpus_ids):
        missing = sorted(set(direct_ids) - set(corpus_ids))
        extra = sorted(set(corpus_ids) - set(direct_ids))
        raise RuntimeError(
            f"V1.3 training-view case mismatch: missing={missing[:10]}, extra={extra[:10]}"
        )
    return {
        "direct_cases": len(direct_ids),
        "view_cases": len(corpus_ids),
        "case_ids_exact_set_match": True,
    }


def postprocess_output(output_dir: Path, protocol_check: Mapping[str, Any]) -> None:
    method_dir = output_dir / "method"
    old_report = method_dir / "private_vocab_rewiring_v1_1.json"
    report = json.loads(old_report.read_text(encoding="utf-8"))
    report["protocol"] = PROTOCOL
    report["training_multiview"] = {
        **_VIEW_CORPUS_META,
        **dict(protocol_check),
        "objective": "minimum_new_minus_true_margin_across_training_views",
        "worst_view_objective": True,
        "target_new_gradient": True,
        "target_true_gradient": True,
        "heldout_probe_text_used": False,
        "official_paraphrase_text_used": False,
        "official_neighborhood_text_used": False,
        "relation_aware_inference_router": False,
    }
    margins = report.get("margins", {})
    report["worst_view_margins"] = margins
    report["margins"] = {
        "note": "V1.3 training/checkpoint summaries are worst-case across the registered training-only views. Official canonical/paraphrase behavior must be evaluated separately.",
    }
    report["claim_boundary"]["training_only_synthetic_paraphrases"] = True
    report["claim_boundary"]["official_paraphrases_used_for_training"] = False
    report["claim_boundary"]["official_paraphrases_used_for_model_selection"] = False
    report["claim_boundary"]["seed1_is_development_only_after_prior_eval"] = True

    new_report = method_dir / "private_vocab_rewiring_v1_3_multiview.json"
    new_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    old_report.unlink()

    completion_path = method_dir / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    worst_view = completion.pop("direct")
    completion["protocol"] = PROTOCOL
    completion["worst_training_view_margin"] = worst_view
    completion["views_per_case"] = int(_VIEW_CORPUS_META["views_per_case"])
    completion["heldout_probe_text_used"] = False
    completion_path.write_text(json.dumps(completion, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "worst_training_view_margin": worst_view,
                "retain_mean_kl": completion.get("retain_mean_kl"),
                "views_per_case": completion["views_per_case"],
                "heldout_probe_text_used": False,
                "integrity_passed": completion.get("integrity_passed"),
                "checkpoint_saved": completion.get("checkpoint_saved"),
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    global _VIEW_MAP, _VIEW_CORPUS_META, _VIEW_CHUNK
    corpus_env = os.environ.get("MCF_V13_VIEW_CORPUS")
    if not corpus_env:
        raise RuntimeError("MCF_V13_VIEW_CORPUS is required")
    corpus_path = Path(corpus_env).resolve()
    _VIEW_MAP, _VIEW_CORPUS_META = load_view_corpus(corpus_path)
    _VIEW_CHUNK = int(os.environ.get("MCF_V13_VIEW_CHUNK", "16"))
    if _VIEW_CHUNK <= 0:
        raise RuntimeError("MCF_V13_VIEW_CHUNK must be positive")

    # The Base V1.1 runner continues to own all architecture, integrity, saving,
    # and optimizer logic.  We swap only the forget margin geometry and retain
    # context builder.
    runner.margin_batch = multiview_worst_margin_batch
    runner.validate_registry = validate_registry
    runner.v1.make_retain_contexts = relational.make_relation_preserving_retain_contexts

    # Locate protocol/output args without adding any MCF/evaluation-data input.
    import sys

    argv = sys.argv[1:]
    try:
        protocol_dir = Path(argv[argv.index("--protocol-dir") + 1]).resolve()
        output_dir = Path(argv[argv.index("--output-dir") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("V1.3 requires --protocol-dir and --output-dir") from exc
    protocol_check = verify_protocol_cases(protocol_dir)

    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "training_view_cases": _VIEW_CORPUS_META["cases"],
                "views_per_case": _VIEW_CORPUS_META["views_per_case"],
                "objective": "worst_view_new_minus_true_margin",
                "heldout_probe_text_used": False,
                "official_paraphrase_text_used": False,
                "official_neighborhood_text_used": False,
            },
            indent=2,
        ),
        flush=True,
    )

    runner.main()
    postprocess_output(output_dir, protocol_check)


if __name__ == "__main__":
    main()
