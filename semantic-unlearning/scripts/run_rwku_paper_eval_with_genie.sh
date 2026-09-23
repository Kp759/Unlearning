#!/usr/bin/env bash
# RWKU evaluation only: Base, actual V2, trained-row genie, and held-out best-of-K genie.
# Usage (from any directory):
#   bash semantic-unlearning/scripts/run_rwku_paper_eval_with_genie.sh [RUN_DIR]
#   or from semantic-unlearning/: bash scripts/run_rwku_paper_eval_with_genie.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="${1:-outputs/rwku_fact_assoc_router_v2_seed1_direct}"
PARA="$RUN/same50_paraphrases.json"
[[ -s "$PARA" ]] || {
    echo "Missing $PARA. Build and manually audit the frozen same-50 rewording set first." >&2
    exit 1
}
[[ -s "$RUN/fact_association_embeddings.pt" ]] || {
    echo "Missing trained artifact in $RUN" >&2
    exit 1
}

python - "$PARA" <<'PY'
import json, sys
from collections import Counter
p=json.load(open(sys.argv[1]))
rows=p["rows"]
n=Counter(str(x["paraphrase_of_source_record_sha256"]) for x in rows)
print("Reworded probes:",len(rows),"covered trained associations:",len(n),"/50")
print("Paraphrases per association:",dict(Counter(n.values())))
print("Generator:",(p.get("manifest") or {}).get("generator"))
if len(n)!=50:
    raise SystemExit("Refusing final Gen evaluation: one or more of the 50 trained facts have no accepted rewording")
PY

mkdir -p logs
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
sha256sum "$RUN/fact_association_embeddings.pt" "$PARA"

for N in 1 2; do
    echo "====== RWKU FINAL REPLAY $N / 2 ======"
    python -u scripts/evaluate_rwku_router_decomposition.py \
        --run-dir "$RUN" \
        --data-root data/rwku \
        --same50-paraphrases "$PARA" \
        --arms base,v2,genie_exact,genie_subject \
        --genie-select generation \
        --groups same50,heldout_level1,heldout_level2,heldout_paraphrase,neighbors \
        --dtype bfloat16 \
        --max-new-tokens 30 \
        --output-dir "$RUN/paper_final_$N" \
        --local-files-only --no-download \
        2>&1 | tee "logs/rwku_paper_final_$N.log"
done

# PPL is measured by the registered V2 evaluator. It does not independently
# evaluate base PPL: use inactive-route counters before inferring equality.
python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
    --run-dir "$RUN" \
    --data-root data/rwku \
    --wikidata-dir data/wikidata \
    --dtype bfloat16 \
    --out "$RUN/paper_final_v2_ppl.json" \
    --local-files-only --no-download \
    2>&1 | tee logs/rwku_paper_final_ppl.log

python - "$RUN" <<'PY'
import json,sys
from pathlib import Path
from collections import defaultdict
run=Path(sys.argv[1])
groups=[
 ("Eff: trained same-50","same50"),
 ("Gen: reworded same-50","same50_paraphrase"),
 ("Unseen Level-1","heldout_level1"),
 ("Unseen Level-2","heldout_level2"),
 ("Unseen paraphrase","heldout_paraphrase"),
 ("Spe: neighbor recovery","neighbors"),
]
arms=("base","v2","genie_exact","genie_subject")
for n in (1,2):
    report=json.loads((run/f"paper_final_{n}"/"rwku_router_decomposition.json").read_text())
    rows=json.loads((run/f"paper_final_{n}"/"rwku_router_decomposition_rows.json").read_text())
    print(f"\n====== REPLAY {n} GENERATED-ANSWER RECOVERY (%) ======")
    print(f"{'Group':28s}"+"".join(f"{a:>17s}" for a in arms))
    for label,group in groups:
        vals=[]
        for arm in arms:
            if group=="same50_paraphrase":
                per_fact=defaultdict(list)
                for item in rows.get(arm,{}).get(group,[]):
                    per_fact[str(item["paraphrase_of_source_record_sha256"])].append(float(item["recovered"]))
                value=(100*sum(sum(v)/len(v) for v in per_fact.values())/len(per_fact)) if per_fact else None
            else:
                block=report["summaries"].get(arm,{}).get(group)
                value=block.get("recovery_percent") if block else None
            vals.append(f"{value:>16.2f}%" if value is not None else f"{'—':>17s}")
        print(f"{label:28s}"+''.join(vals))
    print("Gen note: association-macro average when two accepted paraphrases exist.")
p=run/"paper_final_v2_ppl.json"
if p.exists():
    x=json.loads(p.read_text())
    print("\nV2 runtime-aligned PPL:",x.get("runtime_aligned_PPL"))
    print("PPL route activity:",x.get("runtime_aligned_PPL_route_activity"))
    print("Base PPL is not independently computed by this evaluator.")
PY
