from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MCF_SCRIPTS = SCRIPTS / "MCF_Scripts"
for path in (SCRIPTS, MCF_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module(
    "mcf_zsre_arch_w1k_runner",
    MCF_SCRIPTS / "run_mcf_zsre_arch_target_true_w1k.py",
)
stage1 = load_module(
    "sure_stage1_gagd_w1k_test",
    SCRIPTS / "sure_stage1_gagd_w1k.py",
)
base = sys.modules["run_mcf_zsre_arch_target_true"]


def test_w1k_plan_locks_medium_stage1_lr_and_external_utility(tmp_path):
    args = base.parse_args(
        [
            "--model-path",
            "/model",
            "--mcf-path",
            "/mcf.json",
            "--wikidata-dir",
            "/wikidata",
            "--output-root",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    args.stage1_lr = runner.LOCKED_STAGE1_LR
    w1k, remainder = runner._split_w1k_args(
        ["--utility-wikipedia-dir", "/external-wikipedia"]
    )
    assert remainder == []
    paths = base.seed_paths(tmp_path / "out", 1)
    plan = runner._patched_plan(base.seed_command_plan, args, paths, 1, w1k)
    command = plan[0].command
    joined = " ".join(command)
    assert "sure_stage1_gagd_w1k.py" in joined
    assert "--emb-lm-lr 4e-05" in joined
    assert "--utility-sample-size 1000" in joined
    assert "--utility-wikipedia-dir" in command
    assert "--utility-kl-weight 1.0" in joined
    # Stage 2 remains the baseline repair optimizer at this layer; the
    # materialization wrapper is applied one level above this plan.
    assert "--repair-lr 0.005" in " ".join(plan[1].command)


def test_architecture_records_w1k_without_changing_stage2_lr(tmp_path):
    args = base.parse_args(["--model-path", "/model", "--dry-run"])
    args.stage1_lr = runner.LOCKED_STAGE1_LR
    w1k, _ = runner._split_w1k_args(
        ["--utility-wikipedia-dir", "/external-wikipedia"]
    )
    architecture = runner._patched_architecture(base.effective_architecture, args, w1k)
    assert architecture["stage1"]["learning_rate"] == 4e-5
    assert architecture["stage1"]["external_utility"]["sample_size"] == 1000
    assert architecture["stage1"]["external_utility"]["kl_weight"] == 1.0
    assert architecture["stage2"]["learning_rate"] == 0.005


def test_materialization_options_are_removed_before_base_parse():
    _, forwarded = runner._split_w1k_args(
        [
            "--utility-wikipedia-dir",
            "/external-wikipedia",
            "--model-path",
            "/model",
            "--solver-margin",
            "0.25",
            "--solver-retry-margin",
            "0.5",
            "--materialization-guard-margin",
            "0.1",
            "--dry-run",
        ]
    )
    options, base_forwarded = runner.materialized._split_wrapper_args(forwarded)
    parsed = base.parse_args(base_forwarded)
    assert options.solver_margin == 0.25
    assert options.solver_retry_margin == 0.5
    assert options.materialization_guard_margin == 0.1
    assert parsed.model_path == "/model"
    assert parsed.dry_run is True


def test_utility_kl_is_zero_for_identical_logits():
    logits = torch.tensor([[0.1, 0.2, -0.3], [0.0, 1.0, 2.0]], dtype=torch.float32)
    value = stage1.utility_kl(logits, logits.to(torch.float16))
    assert abs(float(value)) < 1e-5


def test_first_sentence_is_deterministic_and_nontrivial():
    text = "This is a sufficiently long first sentence for a utility example. Second sentence here."
    assert stage1._first_sentence(text) == (
        "This is a sufficiently long first sentence for a utility example."
    )
