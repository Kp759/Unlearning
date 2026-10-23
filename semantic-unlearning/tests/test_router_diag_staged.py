"""router_diag_staged.slurm with stubbed python / sbatch / squeue (the gates run for real):
step order, stops at the hook-parity and L19-agreement checks, the conditional downstream
step (written, submitted once, resumed), downstream mode, and the refusals."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

PY_STUB = r"""#!/bin/bash
echo "python $*" >> "$LOGF"
case "$1" in
  -) exec "$REAL_PY" "$@" ;;
  -c) exit 0 ;;
  -u) shift ;;
esac
script="$1"; shift
args=("$@"); get() { for i in "${!args[@]}"; do [[ "${args[$i]}" == "$1" ]] && { echo "${args[$((i+1))]}"; return; }; done; }
has() { for a in "${args[@]}"; do [[ "$a" == "$1" ]] && return 0; done; return 1; }
case "$script" in
  scripts/cache_router_states.py)
    out="$(get --out-dir)"; mkdir -p "$out"
    [[ -f "$out/manifest.json" ]] && { echo "$out/manifest.json exists"; exit 0; }
    printf '{"dataset": "%s", "seed": %s, "router_layer": 19, "n_layers": 28, "hook_parity": {"batched": {"route_mismatches": %s, "examples": []}, "single": {"route_mismatches": 0, "examples": []}}}' \
      "$(get --dataset)" "$(get --seed)" "${PARITY:-0}" > "$out/manifest.json"
    exit "${CACHE_EXIT:-0}" ;;
  scripts/router_diag_report.py)
    out="$(get --cache-dir)"; mkdir -p "$out/report"
    printf '# report\n## 1. Scores against the decision line\n## 6. Positives vs negatives at L16 and L19\n' > "$out/report/summary.md"
    exit "${REPORT_EXIT:-0}" ;;
  scripts/router_layer_sweep_cached.py)
    if has --combine-only; then mkdir -p "$(get --out-root)"; echo combined > "$(get --out-root)/ROUTER_LAYER_CHOICE.md"; exit 0; fi
    out="$(get --cache-dirs)"; d="$out/layer_sweep"; mkdir -p "$d"
    if [[ -z "$(get --variants)" ]]; then
      printf '{"sweep": [{"variant": "baseline", "layers": [19], "rule": "deployed", "agreement_with_shipped_router": %s}], "robust": []}' "${AGREE:-1.0}" > "$d/sweep.json"
      printf '{"deployed": {"routing_improves": %s}}' "${IMPROVES:-false}" > "$d/penultimate_vs_L19.json"
      printf 'Best F1: L19 (0.9).\nL19 vs penultimate L26: F1 0.9 vs 0.8\n' > "$d/layer_choice.md"
    else
      echo robust > "$d/robustness.md"
    fi
    exit 0 ;;
esac
echo "unexpected python call: $script" >&2; exit 9
"""


def _layout(tmp, fixed=True):
    root = tmp / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "router_diag_staged.slurm", root)
    src = (ROOT / "scripts" / "router_layer_sweep_cached.py").read_text()
    (root / "scripts" / "router_layer_sweep_cached.py").write_text(src if fixed else src.replace("have_sweep", "have"))
    for r in ("mcf_multiseed_regular_v1", "rwku_multiseed_subject_v1"):
        d = root / "outputs" / r / "seed1" / "L19" / "router"
        d.mkdir(parents=True)
        (d / "fact_association_embeddings.pt").touch()
    env = tmp / "env"
    (env / "bin").mkdir(parents=True)
    stubs = {"python": PY_STUB,
             "sbatch": '#!/bin/bash\necho "sbatch $*" >> "$LOGF"\necho 4242\n',
             "squeue": '#!/bin/bash\nprintf "%s" "${SQUEUE_OUT:-}"\n',
             "jq": "#!/bin/bash\nexit 0\n"}
    for name, body in stubs.items():
        (env / "bin" / name).write_text(body)
        (env / "bin" / name).chmod(0o755)
    return root, env


def _run(tmp, root, env, task=0, **extra):
    log = tmp / "calls.log"
    log.write_text("")
    e = {"PATH": f"{env / 'bin'}:/usr/bin:/bin", "HOME": str(tmp), "USER": "kp", "LOGF": str(log),
         "REAL_PY": sys.executable, "REPO_DIR": str(root), "ENV_DIR": str(env),
         "SLURM_ARRAY_TASK_ID": str(task), **{k: str(v) for k, v in extra.items()}}
    r = subprocess.run(["bash", str(root / "router_diag_staged.slurm")], cwd=root, env=e,
                       capture_output=True, text=True)
    calls = [line.split()[1] if line.startswith("python ") and len(line.split()) > 1 else line
             for line in log.read_text().splitlines()]
    return r, log.read_text(), calls


def _steps(stdout):
    return [int(line.split("STEP ")[1].split("/")[0]) for line in stdout.splitlines() if "== STEP " in line]


def test_all_steps_in_order_and_l19_kept(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _steps(r.stdout) == [1, 2, 3, 4, 5, 6]
    scripts = [l.split()[2] for l in log.splitlines() if l.startswith("python -u ")]
    assert scripts == ["scripts/cache_router_states.py", "scripts/router_diag_report.py",
                       "scripts/router_layer_sweep_cached.py", "scripts/router_layer_sweep_cached.py",
                       "scripts/router_layer_sweep_cached.py"]
    sweeps = [l for l in log.splitlines() if "router_layer_sweep_cached.py" in l]
    assert sweeps[0].endswith("--device cuda --variants ")          # step 3: layer sweep only
    assert "--variants baseline,hardneg,aug,hardneg_aug,concat,unmasked" in sweeps[1]
    assert "--combine-only" in sweeps[2]
    assert "PASS hook parity" in r.stdout and "PASS L19 refit" in r.stdout and "KEEP_L19" in r.stdout
    out = root / "outputs" / "router_diag_v1"
    assert (out / "ROUTER_LAYER_CHOICE.md").is_file()
    assert not (out / "mcf" / "seed1" / "layer_sweep" / "downstream_penultimate.sh").exists()
    assert "sbatch" not in log


def test_hook_parity_failure_stops_before_the_report(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, PARITY=3)
    assert r.returncode == 1
    assert "FAIL hook parity" in r.stdout and "STOPPED at step 1" in r.stdout
    assert "router_diag_report.py" not in log and _steps(r.stdout) == [1]


def test_agreement_gate(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, AGREE=0.95)
    assert r.returncode == 1 and "STOPPED at step 3" in r.stdout
    assert "--variants baseline" not in log and _steps(r.stdout) == [1, 2, 3]
    shutil.rmtree(root / "outputs" / "router_diag_v1")
    r, log, _ = _run(tmp_path, root, env, AGREE=0.995)
    assert r.returncode == 0 and "WARN L19 refit" in r.stdout and _steps(r.stdout)[-1] == 6


def test_report_failure_does_not_block_the_sweep(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, REPORT_EXIT=1)
    assert r.returncode == 1 and _steps(r.stdout) == [1, 2, 3, 4, 5, 6]


def test_downstream_written_not_submitted_by_default(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, IMPROVES="true")
    assert r.returncode == 0, r.stdout + r.stderr
    script = root / "outputs" / "router_diag_v1" / "mcf" / "seed1" / "layer_sweep" / "downstream_penultimate.sh"
    text = script.read_text()
    assert "TAG=l16_paper_router26_v1 MCF_ROUTER=linear ROUTER_LAYER=26 bash scripts/run_l16_paper_one.sh mcf 1" in text
    assert f"cd {root}" in text
    assert "sbatch" not in log and "not submitted (RUN_DOWNSTREAM=0)" in r.stdout


def test_downstream_submitted_once_and_resume_skips_cache(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, IMPROVES="true", RUN_DOWNSTREAM=1)
    assert r.returncode == 0, r.stdout + r.stderr
    sb = [l for l in log.splitlines() if l.startswith("sbatch")]
    assert len(sb) == 1 and "DOWNSTREAM_SCRIPT=" in sb[0] and "--time=24:00:00" in sb[0]
    assert "submitted downstream job 4242" in r.stdout
    r, log, _ = _run(tmp_path, root, env, IMPROVES="true", RUN_DOWNSTREAM=1)
    assert r.returncode == 0 and "already submitted" in r.stdout and "manifest.json exists" in r.stdout
    assert not [l for l in log.splitlines() if l.startswith("sbatch")]


def test_rwku_downstream_command(tmp_path):
    root, env = _layout(tmp_path)
    r, _, _ = _run(tmp_path, root, env, task=3, IMPROVES="true")
    assert r.returncode == 0, r.stdout + r.stderr
    text = (root / "outputs" / "router_diag_v1" / "rwku" / "seed1" / "layer_sweep" / "downstream_penultimate.sh").read_text()
    assert "RWKU_GATE=subject ROUTER_LAYER=26 bash scripts/run_l16_paper_one.sh rwku 1" in text


def test_downstream_mode_runs_the_script_only(tmp_path):
    root, env = _layout(tmp_path)
    script = tmp_path / "down.sh"
    script.write_text(f'#!/bin/bash\necho ran >> "{tmp_path}/ran"\n')
    r, log, _ = _run(tmp_path, root, env, DOWNSTREAM_SCRIPT=script)
    assert r.returncode == 0 and (tmp_path / "ran").read_text() == "ran\n"
    assert "cache_router_states.py" not in log and "STEP" not in r.stdout


def test_refuses_next_to_router_diag_and_without_the_fix(tmp_path):
    root, env = _layout(tmp_path)
    r, log, _ = _run(tmp_path, root, env, SQUEUE_OUT="1437819_0 PENDING")
    assert r.returncode == 2 and "scancel" in r.stderr and "cache_router_states.py" not in log
    r, _, _ = _run(tmp_path, root, env, SQUEUE_OUT="1437819_0 PENDING", FORCE=1)
    assert r.returncode == 0
    old, env2 = _layout(tmp_path / "old", fixed=False)
    r, log, _ = _run(tmp_path, old, env2)
    assert r.returncode == 2 and "resume-key fix" in r.stderr and "cache_router_states.py" not in log
