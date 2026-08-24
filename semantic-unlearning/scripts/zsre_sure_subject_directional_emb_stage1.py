#!/usr/bin/env python3
"""SURE-ZsRE subject-keyed directional GA on input embeddings only.

Third dataset for the same architecture: train only the subject's
input-embedding rows, keep every transformer block and the LM head frozen and
untied.

ZsRE groups with MQuAKE rather than MCF. MCF supplies ``target_new`` and scores
a pairwise preference, but the canonical ZsRE protocol supplies the sensitive
answer only -- ``build_zsre_zerounlearn_locked_no_neutral_split.py`` asserts
``target_new leaked into no-neutral ZsRE split`` if one appears, and
``sure_stage2_sparse_repair.load_locked`` raises ``Canonical ZsRE Stage 2
forbids target_new/neutral targets``. Its official Eff is
``post_rewrite_acc[0]``, an accuracy on the sensitive answer, matching MQuAKE's
argmax-accuracy criterion rather than MCF's margin.

So the objective is the competitor form, shared with MQuAKE and imported from
it rather than reimplemented:

    L = margin_weight * relu( train_margin
                              - [ logp(competitor) - logp(target_true) ] )
      + delta_l2 * ||Delta||^2

where the competitor is the highest-logit non-sensitive token cached once on
the base model. Because only input-embedding rows are ever edited, no output
row changes and that competitor stays a stable reference by construction.

Note the older ``build_zsre_zerounlearn_locked_split.py`` writes
``target_new = "Unknown"``; that variant is *not* the canonical path here,
since the loader rejects it.

Data firewall: only ``requested_rewrite`` prompt/subject/target_true of the
locked forget split are read. ZsRE's ``rephrase`` probe (which the official
evaluator scores as Gen) and the ``loc``/``loc_ans`` locality probes are never
loaded at training time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple

import gagd_compare as gagd
import mquake_sure_subject_directional_emb_stage1 as competitor
import sure_stage2_sparse_repair as stage2

METHOD = "SURE-ZsRE-subject-keyed-directional-embedding-stage1"
PROTOCOL = "zsre_target_true_sensitive_subject_keyed_embedding_ga_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Identical knobs to the MQuAKE port, since the objective is the same."""
    return competitor.parse_args(argv)


def validate_locked_zsre(
    visible_path: Path, manifest_path: Path, seed: int, forget_num: int
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    """Locked ZsRE forget view, with the no-neutral contract enforced here too.

    ``load_locked`` already raises on ``target_new`` for non-MCF datasets; this
    repeats the check with a message naming this script, and additionally
    guards the ZsRE-specific probes so a mis-built split fails loudly at
    training time rather than quietly training on evaluation data.
    """
    records, manifest = stage2.load_locked(
        "zsre", visible_path, manifest_path, seed, forget_num
    )
    for index, record in enumerate(records):
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise RuntimeError(f"record {index} lacks requested_rewrite")
        if not str(rewrite.get("subject", "")).strip():
            raise RuntimeError(f"record {index} lacks a subject")
        target_true = str((rewrite.get("target_true") or {}).get("str", "")).strip()
        if not target_true:
            raise RuntimeError(f"record {index} lacks target_true")
        if target_true.lower() == "unknown":
            raise RuntimeError(
                f"record {index} has target_true 'Unknown' -- that is the neutral "
                "replacement target, not a sensitive answer. Build the split with "
                "build_zsre_zerounlearn_locked_no_neutral_split.py."
            )
        if "target_new" in rewrite:
            raise RuntimeError(
                f"record {index} carries target_new; the canonical ZsRE protocol "
                "is no-neutral"
            )
        for probe in ("rephrase", "loc", "loc_ans"):
            if record.get(probe):
                raise RuntimeError(
                    f"record {index} exposes the held-out ZsRE probe {probe!r}"
                )
    return records, manifest


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = validate_locked_zsre(
        Path(a.training_visible_path).resolve(),
        Path(a.split_manifest).resolve(),
        int(a.seed),
        int(a.forget_num),
    )

    competitor.run_competitor_stage1(
        a,
        records,
        manifest,
        dataset="zsre",
        method=METHOD,
        protocol=PROTOCOL,
        extra_config={
            "neutral_target_used": False,
            "zsre_protocol_note": (
                "canonical no-neutral ZsRE: the locked view carries target_true "
                "only, so forgetting is scored against the model's own top "
                "non-sensitive token rather than a supplied replacement"
            ),
            "rephrase_probe_seen": 0,
            "locality_probe_seen": 0,
        },
    )


if __name__ == "__main__":
    main()
