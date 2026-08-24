#!/usr/bin/env python3
"""Check that selected subject embedding rows actually occur in the prompts.

``mcf_sure_subject_directional_emb_stage1.py`` selects subject rows from the
union of two tokenizations (" Danielle Darrieux" and "Danielle Darrieux"),
then drops rows above ``--max-subject-token-frequency``.  If that filter
keeps only rows belonging to the standalone variant, those rows never appear
in the real prompt, so ``register_input_embedding_delta_hook`` never fires
for them: the record receives literally zero gradient and can never be
fixed, no matter how long training runs.

Two real runs (98e34f4, 017174c) reported a bit-identical
``stage1_minimum_margin`` of -10.671875 across different synthetic training
data, which is what a permanently-unedited record looks like.

Tokenizer only -- no model weights, no GPU, runs in seconds.

    python scripts/diagnose_mcf_subject_row_alignment.py \
      --model-path /path/to/model \
      --stage1-config outputs/.../stage1_subject_emb/stage1_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="tokenizer source")
    p.add_argument(
        "--stage1-config",
        required=True,
        help="stage1_config.json written by the subject-keyed Stage 1",
    )
    p.add_argument("--multi-counterfact", default="data/multi_counterfact.json")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    cfg = json.loads(Path(a.stage1_config).read_text(encoding="utf-8"))
    selection: List[Dict[str, Any]] = cfg.get("subject_row_selection") or []
    if not selection:
        raise SystemExit("config has no subject_row_selection block")
    failing = set(int(x) for x in cfg.get("stage1_failing_positions") or [])

    tok = AutoTokenizer.from_pretrained(a.model_path)
    records = json.loads(Path(a.multi_counterfact).read_text(encoding="utf-8"))
    by_case = {int(r["case_id"]): r for r in records}

    dead: List[Dict[str, Any]] = []
    partial: List[Dict[str, Any]] = []
    healthy = 0

    for entry in selection:
        case_id = int(entry["case_id"])
        # Prefer the direct-prompt subset when present (newer configs); older
        # configs only recorded the pooled kept set.
        kept = set(
            int(x)
            for x in (entry.get("kept_direct_token_ids") or entry["kept_token_ids"])
        )
        record = by_case.get(case_id)
        if record is None:
            continue
        rewrite = record["requested_rewrite"]
        prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        prompt_ids = set(int(x) for x in tok(prompt, add_special_tokens=False)["input_ids"])
        live = kept & prompt_ids
        row = {
            "record_position": entry["record_position"],
            "case_id": case_id,
            "subject": entry["subject"],
            "kept": sorted(kept),
            "live_in_prompt": sorted(live),
            "fallback": bool(entry.get("rarest_token_fallback")),
            "is_eff_failure": int(entry["record_position"]) in failing,
        }
        if not live:
            dead.append(row)
        elif len(live) < len(kept):
            partial.append(row)
        else:
            healthy += 1

    total = len(selection)
    print(f"\nrecords analysed                : {total}")
    print(f"  all kept rows live in prompt  : {healthy}")
    print(f"  SOME kept rows dead in prompt : {len(partial)}")
    print(f"  ALL kept rows dead in prompt  : {len(dead)}   <-- unfixable by training")

    if dead:
        print("\nDEAD records (zero gradient reaches these subjects):")
        for r in dead:
            flag = " [Eff FAIL]" if r["is_eff_failure"] else ""
            fb = " [fallback]" if r["fallback"] else ""
            print(f"  pos {r['record_position']:>3} {r['subject'][:40]:42s}{fb}{flag}")
            print(f"      kept={r['kept']}  live_in_prompt=[]")

    if partial:
        print("\nPARTIAL records (some rows never fire):")
        for r in partial[:15]:
            flag = " [Eff FAIL]" if r["is_eff_failure"] else ""
            print(
                f"  pos {r['record_position']:>3} {r['subject'][:34]:36s}"
                f" live {len(r['live_in_prompt'])}/{len(r['kept'])}{flag}"
            )

    if failing:
        dead_fail = sum(1 for r in dead if r["is_eff_failure"])
        part_fail = sum(1 for r in partial if r["is_eff_failure"])
        print(
            f"\nof {len(failing)} Eff failures: {dead_fail} fully dead, "
            f"{part_fail} partially dead, "
            f"{len(failing) - dead_fail - part_fail} have all rows live "
            "(those are an optimization-strength problem, not an alignment bug)"
        )

    print(
        "\nInterpretation: any 'dead' record proves the frequency filter kept "
        "only rows from the standalone subject tokenization, which never occurs "
        "in the real prompt. Fix = intersect the selection with the prompt's own "
        "tokenization. If instead every row is live, alignment is fine and the "
        "weak edit is about optimization strength (--surgical-weight, "
        "--train-margin, --lr) rather than a bug."
    )


if __name__ == "__main__":
    main()
