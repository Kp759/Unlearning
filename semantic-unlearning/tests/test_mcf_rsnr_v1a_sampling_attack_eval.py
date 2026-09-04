from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_rsnr_v1a_sampling_attack_eval as sample_ev  # noqa: E402


def test_temperature_parser():
    assert sample_ev.parse_temperatures("0.7,1.0") == [0.7, 1.0]
    with pytest.raises(ValueError):
        sample_ev.parse_temperatures("")
    with pytest.raises(ValueError):
        sample_ev.parse_temperatures("0")


def test_top_p_sampler_returns_valid_token():
    torch.manual_seed(3)
    logits = torch.tensor([0.0, 1.0, 5.0])
    token = sample_ev.sample_from_logits(logits, temperature=0.7, top_p=0.95)
    assert token in {0, 1, 2}


def test_sampling_script_is_retrieval_only():
    source = Path(sample_ev.__file__).read_text(encoding="utf-8")
    assert '"retrieval_attacks_only": True' in source
    assert '"true_answer_present_in_attack_prompt": False' in source
    assert '"checkpoint_not_retrained_or_selected": True' in source
