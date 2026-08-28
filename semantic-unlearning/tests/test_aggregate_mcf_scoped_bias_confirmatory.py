from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import aggregate_mcf_scoped_bias_confirmatory as aggregate  # noqa: E402


def _metrics(eff, gen, spe, spe_success):
    return {
        "Eff": eff,
        "Gen": gen,
        "Spe": spe,
        "Spe_success": spe_success,
    }


def _payload(seed, *, scoped):
    payload = {
        "seed": seed,
        "forget": _metrics(80.0 if not scoped else 0.0, 85.0 if not scoped else 0.0, 12.0, 88.0),
        "retain": _metrics(87.0, 84.0, 11.5 if not scoped else 11.48, 83.0),
        "PPL": 16.625,
    }
    if scoped:
        payload["post_reload_acceptance"] = {"passed": True}
        payload["scoped_span_edit"] = {
            "loaded": True,
            "record_scopes": 50,
            "metadata": {
                "protocol": aggregate.EXPECTED_PROTOCOL,
                "penalty": aggregate.EXPECTED_PENALTY,
                "base_weights_modified": False,
            },
            "per_split_prompt_fire_audit": {
                "used_for_training_or_checkpoint_selection": False,
                "groups": {
                    "forget_rewrite": {
                        "matched_prompts": 50,
                        "prompt_count": 50,
                    },
                    "retain_rewrite": {
                        "matched_prompts": 0,
                        "prompt_count": 1000,
                    },
                },
            },
        }
    return payload


def _write_seed(root, seed):
    seed_dir = root / f"seed{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "base_official_eval.json").write_text(
        json.dumps(_payload(seed, scoped=False)), encoding="utf-8"
    )
    (seed_dir / "official_eval.json").write_text(
        json.dumps(_payload(seed, scoped=True)), encoding="utf-8"
    )


def test_collect_and_aggregate_require_matched_frozen_results(tmp_path):
    _write_seed(tmp_path, 2)
    _write_seed(tmp_path, 3)
    rows = aggregate.collect(tmp_path, [2, 3])
    assert len(rows) == 2
    assert rows[0]["delta_forget_Eff"] == -80.0
    assert rows[0]["delta_forget_Gen"] == -85.0
    assert rows[0]["delta_forget_Spe"] == 0.0
    assert rows[0]["delta_retain_Spe"] == pytest.approx(-0.02)
    assert rows[0]["delta_PPL"] == 0.0
    assert rows[0]["route_forget_rewrite_matched"] == 50
    summary = aggregate.aggregate(rows)
    assert summary["n_seeds"] == 2
    assert summary["scoped_forget_Eff_mean"] == 0.0
    assert summary["delta_retain_Spe_mean"] == pytest.approx(-0.02)


def test_seed1_is_forbidden_from_confirmatory_aggregate(tmp_path):
    _write_seed(tmp_path, 1)
    with pytest.raises(ValueError, match="seed 1 is exploratory"):
        aggregate.collect(tmp_path, [1])


def test_changed_penalty_is_rejected(tmp_path):
    _write_seed(tmp_path, 2)
    path = tmp_path / "seed2" / "official_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["scoped_span_edit"]["metadata"]["penalty"] = 64.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="penalty is not frozen"):
        aggregate.collect(tmp_path, [2])
