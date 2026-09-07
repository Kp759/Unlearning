# Fix5n-v3 — Deterministic Isolated Two-Slot Output Control

## Purpose

Fix5n-v2 showed that output-position gating can recover overlapping permitted answers, but its generated-marker detector had poor coverage. Fix5n-v3 removes that failure mode with controller-owned answer slots.

The scientific question is narrow:

> Can we retain the forbidden suppression strength of the frozen query-wide `-12` correction while avoiding collateral damage to a permitted answer whose token support overlaps the forbidden answer?

## Frozen components

Fix5n-v3 does **not** retrain or retune anything. It reuses the completed Fix5m Seed-1 mixed-query records:

- same 80 order-specific mixed queries;
- same saved route decisions;
- same active token IDs;
- same fixed penalty magnitude (`-12` in the current frozen artifact);
- quotient off;
- same frozen base model.

## Stronger output controller

Every condition uses the same isolated structured decoder.

For each mixed query, the controller separately generates:

```text
First: <slot-1 answer>
Second: <slot-2 answer>
```

The two answer slots are decoded independently from the same original query. Slot 2 does not condition on Slot 1 generated text. This prevents both direct vocabulary-penalty collateral and indirect autoregressive cross-slot contamination.

The frozen router's active binding subject is mapped to the actual `First:` / `Second:` input clause. That mapping decides where the structured penalty is allowed to operate. Benchmark order and gold answers are used only after generation for evaluation, never for intervention.

## Compared conditions

1. **base** — structured decoder, no penalty in either slot.
2. **query_wide** — same decoder, frozen support penalized in both slots whenever the saved route activated.
3. **structured_gated** — same decoder, frozen support penalized only in the router-resolved slot(s).

This makes the comparison fair: prompt structure and decoding policy are matched across all three conditions.

## Primary metrics

Scores are slot-specific, which removes the overlap attribution ambiguity from Fix5m/Fix5n-v2.

- forbidden canonical disclosure in the forbidden slot;
- permitted companion strict correctness in the companion slot;
- joint success;
- overlap companion regression among Base-correct cases;
- non-overlap companion regression;
- fraction of query-wide forbidden suppressions retained by structured gating;
- route-slot alignment diagnostics.

Automatic scoring remains canonical-string based. Aliases and semantic equivalence are **not** claimed to be measured.

## Predeclared Seed-1 pilot gate

Fix5n-v3 passes only if all of the following hold:

- controller slot-boundary coverage = 100%;
- overlap structured regression <= 15%;
- overlap regression improves by >= 20 percentage points versus query-wide;
- non-overlap structured regression <= 5%;
- structured gating retains >= 80% of query-wide forbidden suppressions.

A pass supports output-position selectivity **in this controlled isolated two-slot setting**. It is not yet a claim that arbitrary free-form generation is solved.

## Files

```text
scripts/mcf_structured_two_slot_decoder_fix5n_v3_seed1.py
scripts/run_mcf_structured_two_slot_decoder_fix5n_v3_seed1.sh
tests/test_structured_two_slot_decoder_fix5n_v3.py
scripts/build_structured_slots_fix5n_v3_package.py
scripts/install_structured_slots_fix5n_v3.py
README_controlled_slots_fix5n_v3.md
```

## Safe package workflow

Build a checksum-manifested archive:

```bash
python scripts/build_structured_slots_fix5n_v3_package.py \
  --root "$PWD" \
  --output structured_slots_fix5n_v3.zip
```

Install it into another checkout without overwriting different files:

```bash
python scripts/install_structured_slots_fix5n_v3.py \
  structured_slots_fix5n_v3.zip \
  --root /path/to/semantic-unlearning
```

The installer validates the manifest, member hashes, archive paths, symlink status, and refuses to overwrite differing existing contents.

## Run

```bash
export FIX5M_SOURCE_DIR="$PWD/results/retain_anchored_context_head/mcf/seed1_target_local_generation_mixed_eval_fix5m"
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export FIX5N_V3_OUT_DIR="$PWD/results/retain_anchored_context_head/mcf/seed1_structured_two_slot_decoder_fix5n_v3"
export MAX_SLOT_NEW_TOKENS="32"

bash scripts/run_mcf_structured_two_slot_decoder_fix5n_v3_seed1.sh
```

Before the GPU run, execute:

```bash
python -m py_compile scripts/mcf_structured_two_slot_decoder_fix5n_v3_seed1.py
bash -n scripts/run_mcf_structured_two_slot_decoder_fix5n_v3_seed1.sh
python -m pytest -q tests/test_structured_two_slot_decoder_fix5n_v3.py
```

Do not enable quotient, change the penalty, retrain the router, or run Seeds 2–10 before interpreting this Seed-1 controlled-slot result.
