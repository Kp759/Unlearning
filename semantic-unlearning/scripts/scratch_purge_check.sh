#!/usr/bin/env bash
# Wulver /scratch deletes files ~30 days old (sooner when it is >80% full). This reports what
# the paper runs depend on that is already gone. Login node, no conda env needed (system python):
#
#   bash scripts/scratch_purge_check.sh
#
# Also saves the broken env's pip package versions (from its *.dist-info dir names) to
# ~/semantic_unlearning_pip_versions.txt (home is not purged) so the env can be rebuilt as it was.
set -uo pipefail
S=/scratch/yl258/kp759
REPO=$S/Unlearning
R=$REPO/semantic-unlearning
ENV="${ENV_DIR:-$S/conda_envs/semantic_unlearning}"
PY=/usr/bin/python3; [[ -x "$PY" ]] || PY="$(command -v python3)"

echo "== git: tracked files deleted from the working tree"
git -C "$REPO" ls-files --deleted | sed 's/^/  /' | head -30
echo "  total: $(git -C "$REPO" ls-files --deleted | wc -l)"
echo "== git: objects missing from .git (repo integrity)"
echo "  $(git -C "$REPO" fsck --connectivity-only 2>&1 | grep -ciE 'missing|invalid|broken') problem lines"

"$PY" - "$S" "$R" "$ENV" "$HOME/semantic_unlearning_pip_versions.txt" <<'EOF'
import glob, json, os, re, sys
S, R, ENV, LOCK = sys.argv[1:5]

def ex(p):
    return os.path.exists(p)          # follows symlinks: a dangling HF snapshot link = gone

print(f"== conda env {ENV}")
miss = tot = 0
bad = {}
for m in glob.glob(f"{ENV}/conda-meta/*.json"):
    try:
        files = json.load(open(m)).get("files", [])
    except Exception:
        continue
    g = [f for f in files if not ex(os.path.join(ENV, f))]
    tot += len(files); miss += len(g)
    if g:
        bad[os.path.basename(m)[:-5]] = len(g)
print(f"  conda packages: {miss}/{tot} files missing; worst: {sorted(bad.items(), key=lambda x: -x[1])[:6]}")
sp = glob.glob(f"{ENV}/lib/python3*/site-packages")
pins, pbad = [], {}
for d in sorted(glob.glob(f"{sp[0]}/*.dist-info")) if sp else []:
    m = re.match(r"(.+?)-(\d[^-]*)\.dist-info$", os.path.basename(d))
    if m:
        pins.append(f"{m.group(1)}=={m.group(2)}")
    rec = os.path.join(d, "RECORD")
    if ex(rec):
        files = [l.split(",")[0] for l in open(rec, errors="ignore") if l.strip()]
        g = [f for f in files if not ex(os.path.normpath(os.path.join(sp[0], f)))]
        if g:
            pbad[os.path.basename(d)] = f"{len(g)}/{len(files)}"
    else:
        pbad[os.path.basename(d)] = "RECORD gone"
print(f"  pip packages: {len(pins)} found, {len(pbad)} with missing files; e.g. {list(pbad.items())[:6]}")
if pins:
    open(LOCK, "w").write("\n".join(pins) + "\n")
    print(f"  versions saved -> {LOCK}  (torch: {[p for p in pins if p.lower().startswith('torch==')]}, "
          f"transformers: {[p for p in pins if p.lower().startswith('transformers==')]})")

print("== model weights (HF snapshots: every file must resolve)")
for snap in sorted(glob.glob(f"{S}/hf/models--*/snapshots/*")):
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(snap) for f in fs]
    g = [os.path.relpath(f, snap) for f in files if not ex(f)]
    print(f"  {'OK  ' if not g else 'GONE'} {snap.split('/hf/')[1][:70]}  missing {len(g)}/{len(files)} {g[:3]}")
for d in (f"{S}/hf-materialized/Llama-3.2-3B-Instruct-clean",
          f"{S}/models/Llama-3.2-3B-Instruct-0cb88a4f764b7a12671c53f0838cd831a0843b95-runtime"):
    if os.path.isdir(d):
        print(f"  {'OK  ' if ex(d + '/config.json') else 'GONE'} {d}")

print("== data the paper evaluators read")
for f in ("data/multi_counterfact.json", "data/zsre_mend_eval.json", "data/MQuAKE-CF-3k-v2.json",
          "data/wikidata/train/data-00000-of-00001.arrow"):
    print(f"  {'OK  ' if ex(os.path.join(R, f)) else 'GONE'} {f}")
print(f"  data/rwku: {sum(len(fs) for _, _, fs in os.walk(os.path.join(R, 'data/rwku')))} files; "
      f"datasets/wikipedia_sure_50020: {sum(len(fs) for _, _, fs in os.walk(S + '/datasets/wikipedia_sure_50020'))} files")

print("== routers and results")
def count(pattern):
    return len(glob.glob(os.path.join(R, pattern)))
for label, pat, want in (
        ("L19 routers (4 ds x 5 seeds)", "outputs/*_multiseed_regular_v1/seed*/L19/router/fact_association_embeddings.pt", 20),
        ("ZsRE reworded_v2 routers", "outputs/zsre_multiseed_reworded_v2/seed*/L19/router/fact_association_embeddings.pt", 5),
        ("MCF v2.2 routers", "outputs/l16_paper_v1/mcf/seed*/router_v22/S*/fact_association_embeddings.pt", 5),
        ("MCF eval reference manifest", "outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json", 1),
        ("paper results (official evals)", "outputs/l16_paper_v1/*/seed*/*/official_*_eval.json", 160),
        ("ours rows", "outputs/l16_paper_v1/*/seed*/ours/fact_association_embeddings.pt", 20)):
    n = count(pat)
    print(f"  {'OK  ' if n >= want else 'LOW '} {label}: {n}/{want}")
EOF
echo "== where to move things: /project (2 TB per PI group, backed up, not purged)"
ls -ld /project/yl258 /project/yl258/kp759 2>&1 | sed 's/^/  /'
