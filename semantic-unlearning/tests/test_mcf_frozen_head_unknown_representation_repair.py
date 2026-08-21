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
    "mcf_frozen_head_unknown_representation_repair_test",
    SCRIPTS / "mcf_frozen_head_unknown_representation_repair.py",
)


def test_unknown_margin_is_sensitive_nll_minus_unknown_nll():
    sensitive = torch.tensor([3.0, 1.0], requires_grad=True)
    unknown = torch.tensor([1.0, 1.2], requires_grad=True)
    loss, margins = repair.unknown_margin_loss(sensitive, unknown, 0.05)
    assert torch.allclose(margins, torch.tensor([2.0, -0.2]))
    expected = (0.0**2 + 0.25**2) / 2.0
    assert torch.allclose(loss, torch.tensor(expected))
    loss.backward()
    # Failed example should push sensitive NLL up and Unknown NLL down.
    assert sensitive.grad is not None
    assert unknown.grad is not None
    assert float(sensitive.grad[1]) < 0.0
    assert float(unknown.grad[1]) > 0.0


def test_unknown_instances_do_not_copy_counterfact_target_new():
    records = [
        {
            "case_id": 7,
            "requested_rewrite": {
                "prompt": "{} was born in",
                "subject": "Ada",
                "target_true": {"str": "London"},
                "target_new": {"str": "Paris"},
            },
        }
    ]
    instances = repair.build_unknown_instances(records, "Unknown")
    assert len(instances) == 1
    item = instances[0]
    assert item.prompt == "Ada was born in"
    assert item.target_true == "London"
    assert item.target_new == "Unknown"
    assert item.target_new != "Paris"


def test_parser_defaults_encode_exact_unknown_objective():
    args = repair.parse_args(
        [
            "--model-path",
            "/model",
            "--training-visible-path",
            "/visible.json",
            "--split-manifest",
            "/manifest.json",
            "--output-dir",
            "/out",
            "--utility-wikipedia-dir",
            "/wiki",
        ]
    )
    assert args.neutral_answer == "Unknown"
    assert args.repair_scope == "active"
    assert args.forget_margin == 0.05
    assert args.forget_weight == 1.0
    assert args.utility_kl_weight == 2.0
    assert args.delta_weight == 1e-8
    assert args.benchmark_pair_margin == 0.05
