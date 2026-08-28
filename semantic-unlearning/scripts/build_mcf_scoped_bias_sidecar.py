#!/usr/bin/env python3
"""Build an exact-subject-scoped constant-logit suppression sidecar.

This is the geometry-independent reader ablation for ``span_gated``.  The
complete subject sequence still decides scope, but once the gate is open the
sensitive answer's discriminative token rows receive a fixed negative logit
bias.  Consequently paraphrase portability does not depend on ``q . h`` at an
unseen answer position.  Closed scope remains exactly the frozen base model.

The artifact is scoped suppression/model editing, not weight-level unlearning:
the base checkpoint is untouched and omitting the sidecar recovers it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from mcf_sampling import sample_first_mcf_records, sample_official_mcf_records
import scoped_span_edit as scoped


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--mcf-path", default="data/multi_counterfact.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--writer-layer", type=int, default=8)
    parser.add_argument(
        "--penalty",
        type=float,
        default=512.0,
        help=(
            "Fixed negative logit bias for sensitive-only token rows while "
            "the subject gate is open. The default is deliberately "
            "geometry-independent and bf16-safe; it is not calibrated on "
            "official paraphrases."
        ),
    )
    return parser.parse_args(argv)


def _rewrite(record):
    value = record["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def _answer_token_ids(tokenizer, text):
    value = str(text)
    if value and not value.startswith(" "):
        value = " " + value
    return [
        int(token_id)
        for token_id in tokenizer(value, add_special_tokens=False)["input_ids"]
    ]


def build_state(tokenizer, hidden_size, records, writer_layer, penalty, seed):
    subjects = [str(_rewrite(record)["subject"]) for record in records]
    patterns = scoped.build_subject_patterns(tokenizer, subjects)

    selected_rows = []
    audit_rows = []
    for record in records:
        rr = _rewrite(record)
        sensitive = str(rr["target_true"]["str"])
        reference = str(rr["target_new"]["str"])
        sensitive_ids = _answer_token_ids(tokenizer, sensitive)
        reference_ids = _answer_token_ids(tokenizer, reference)
        # Penalize only rows absent from the reference answer.  Penalizing all
        # sensitive rows can move both alternatives equally when their
        # subword tokenizations overlap, silently eliminating the margin gain.
        discriminative = sorted(set(sensitive_ids) - set(reference_ids))
        if not discriminative:
            raise RuntimeError(
                "No sensitive-only token row for case "
                f"{record.get('case_id')}: target_true={sensitive!r}, "
                f"target_new={reference!r}. A row-bias reader cannot "
                "distinguish these answer tokenizations without sequence "
                "conditioning."
            )
        selected_rows.append(discriminative)
        audit_rows.append(
            {
                "case_id": int(record["case_id"]),
                "subject": str(rr["subject"]),
                "target_true": sensitive,
                "target_new": reference,
                "target_true_token_ids": sensitive_ids,
                "target_new_token_ids": reference_ids,
                "penalized_token_ids": discriminative,
                "penalized_tokens": [
                    tokenizer.decode([token_id]) for token_id in discriminative
                ],
            }
        )

    max_rows = max(len(rows) for rows in selected_rows)
    row_ids = torch.full((len(records), max_rows), -1, dtype=torch.long)
    deltas = torch.zeros(
        (len(records), max_rows, int(hidden_size)), dtype=torch.float32
    )
    biases = torch.zeros((len(records), max_rows), dtype=torch.float32)
    for record_index, rows in enumerate(selected_rows):
        for row_index, token_id in enumerate(rows):
            row_ids[record_index, row_index] = int(token_id)
            biases[record_index, row_index] = -float(penalty)

    state = scoped.build_sidecar_state(
        subjects=subjects,
        subject_patterns=patterns,
        writer_layer=int(writer_layer),
        writer_delta=torch.zeros(
            (len(records), int(hidden_size)), dtype=torch.float32
        ),
        reader_row_ids=row_ids,
        reader_deltas=deltas,
        reader_biases=biases,
        reader_scale=1.0,
        metadata={
            "method": "MCF-exact-subject-scoped-constant-logit-suppression",
            "protocol": "mcf_scoped_bias_reader_v1",
            "scope": "exact complete subject token sequence",
            "reader_mode": "constant sensitive-only token-row bias",
            "penalty": float(penalty),
            "seed": int(seed),
            "case_ids": [int(record["case_id"]) for record in records],
            "base_weights_modified": False,
            "official_paraphrases_or_neighborhoods_used": False,
            "developed_after_observing_seed1_official_gen_failure": True,
            "claim": "scoped suppression/model editing, not weight-level unlearning",
            "per_record": audit_rows,
        },
    )
    return state, audit_rows


def main(argv=None):
    args = parse_args(argv)
    if not math.isfinite(args.penalty) or args.penalty <= 0.0:
        raise ValueError("--penalty must be finite and positive")

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    config = AutoConfig.from_pretrained(str(model_dir))
    hidden_size = int(getattr(config, "hidden_size"))

    with Path(args.mcf_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if args.sample_mode == "official":
        records, _ = sample_official_mcf_records(
            data, args.unlearn_num, 0, args.seed, strict=True
        )
    else:
        records, _ = sample_first_mcf_records(
            data, args.unlearn_num, 0, strict=True
        )

    state, audit_rows = build_state(
        tokenizer,
        hidden_size,
        records,
        args.writer_layer,
        args.penalty,
        args.seed,
    )
    output = scoped.save_sidecar(args.out, state)
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit = {
        "sidecar": str(output),
        "model_dir": str(model_dir),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "records": len(records),
        "penalty": float(args.penalty),
        "base_weights_modified": False,
        "official_paraphrases_or_neighborhoods_used": False,
        "developed_after_observing_seed1_official_gen_failure": True,
        "per_record": audit_rows,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"sidecar: {output}")
    print(f"audit:   {audit_path}")
    print(f"scopes:  {len(records)}")
    print("base weights modified: False")


if __name__ == "__main__":
    main()
