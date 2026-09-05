import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch

from rsnr_v1a_frozen_spec import validate_adapter_checkpoint
from run_mcf_rsnr_v1a_learned_router import (
    binary_metrics,
    choose_threshold,
    split_training_safe_positive_views,
)


def _checkpoint():
    return {
        "variant": "RSNR-V1A-PreHead",
        "protocol": "mcf_rsnr_v1a_prehead_oracle_null_adapter",
        "intervention_site": "pre_lm_head_final_hidden_state",
        "adapter_rank": 16,
        "adapter_alpha": 16.0,
        "abstention": "I don't know.",
        "transformer_weights_modified": False,
        "lm_head_weights_modified": False,
        "adapter_state_dict": {
            "down.weight": torch.zeros(16, 3072),
            "up.weight": torch.zeros(3072, 16),
        },
    }


def test_frozen_checkpoint_accepts_v1a_prehead_and_rejects_rank_change():
    good = _checkpoint()
    validate_adapter_checkpoint(good)
    bad = dict(good)
    bad["adapter_rank"] = 32
    try:
        validate_adapter_checkpoint(bad)
    except RuntimeError as exc:
        assert "adapter_rank" in str(exc)
    else:
        raise AssertionError("rank change should violate frozen architecture")


def test_positive_view_split_holds_out_one_training_safe_view_per_case():
    forget = [
        {
            "case_id": 7,
            "requested_rewrite": {
                "subject": "Ada Lovelace",
                "relation_id": "P19",
            },
        }
    ]
    views = {7: ["v0 {}", "v1 {}", "v2 {}", "v3 {}", "v4 {}"]}
    train, calib = split_training_safe_positive_views(forget, views)
    assert len(train) == 4
    assert len(calib) == 1
    assert set(train).isdisjoint(calib)
    assert all("Ada Lovelace" in x for x in train + calib)


def test_threshold_selection_preserves_required_recall_then_reduces_fpr():
    # Two positives at 0.95/0.90 and negatives at 0.80/0.20.  With recall=1,
    # threshold 0.90 gives zero FPR and is the desired calibrated threshold.
    probs = [0.95, 0.90, 0.80, 0.20]
    labels = [1, 1, 0, 0]
    threshold, metrics = choose_threshold(probs, labels, minimum_recall=1.0)
    assert threshold == 0.90
    assert metrics["recall"] == 1.0
    assert metrics["false_positive_rate"] == 0.0


def test_binary_metrics_reports_precision_recall_and_false_positive_rate():
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
