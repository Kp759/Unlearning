from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repair = load_module(
    "mcf_unknown_fullblock_locality_repair_test",
    SCRIPTS / "mcf_unknown_fullblock_locality_repair.py",
)


def test_locality_hidden_loss_zero_at_base():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ref = x.detach().clone()
    loss = repair.locality_hidden_loss(x, ref)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_locality_hidden_loss_positive_after_drift():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    ref = x.detach().clone()
    loss = repair.locality_hidden_loss(x + 1.0, ref)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_locality_kl_zero_when_logits_match():
    z = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    ref = z.detach().clone()
    loss = repair.locality_kl(z, ref)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_build_locality_prompts_uses_only_other_training_visible_subjects():
    records = []
    for i, subject in enumerate(["Alice", "Bob", "Carol"]):
        records.append(
            {
                "case_id": i,
                "requested_rewrite": {
                    "subject": subject,
                    "prompt": "{} was born in",
                    "target_true": {"str": "Sensitive"},
                    "target_new": {"str": "Reference"},
                },
            }
        )
    prompts, receipt = repair.build_locality_prompts(records, control_count=2)
    assert len(prompts) == 6
    assert len(receipt) == 6
    for row in receipt:
        assert row["donor_subject"] != row["original_subject"]
        assert row["prompt"] == f"{row['donor_subject']} was born in"


def test_parser_has_no_rank_or_adapter_argument():
    args = repair.parse_args(
        [
            "--model-path", "/base",
            "--training-visible-path", "/visible.json",
            "--split-manifest", "/manifest.json",
            "--output-dir", "/out",
            "--utility-wikipedia-dir", "/wiki",
        ]
    )
    assert not hasattr(args, "adapter_rank")
    assert args.lr == 5e-6
    assert args.locality_hidden_weight == 10.0
    assert args.locality_kl_weight == 2.0
    assert args.utility_kl_weight == 2.0
    assert args.neutral_answer == "Unknown"
