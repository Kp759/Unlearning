#!/bin/bash
# Side-by-side: exact-match L19 router + question-side rows (E_L16_query_logit) vs the
# v2.1 factorized router + retrained rows (E_L16_query_logit_v21S<L>).
ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
L="${SUBJECT_LAYER:-8}"
A=outputs/mcf_early_write_seed1/E_L16_query_logit
B=outputs/mcf_early_write_seed1/E_L16_query_logit_v21S$L
echo "== official MCF (answer probability)"
python scripts/compare_official_mcf.py "exact-match router=$A/linear_global" "v2.1 router S$L=$B/linear_global"
echo; echo "== held-out generation (leak / refusal / routed; neighbourhood kept)"
grep -E "^\| model|^\|---|L16_query_logit, |L16_query_logit_v21S$L" outputs/mcf_heldout_generation_seed1/summary.md | head -12
echo; echo "== same-subject collateral"
grep -E "^\| run|^\|---|^\| E_L16_query_logit \||E_L16_query_logit_v21S$L" outputs/mcf_subject_collateral_seed1/summary.md | head -6
echo; echo "== lens eval (answer readable inside the model)"
grep -E "^\| run|^\|---|^\| base|^\| E_L16_query_logit \||E_L16_query_logit_v21S$L" outputs/mcf_lens_eval_seed1/summary.md | head -12
