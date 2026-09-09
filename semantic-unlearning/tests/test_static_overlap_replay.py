"""Regression tests for hardest-target regression and retention context coverage."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from augment_static_overlap_retention import augment_retention, main as augment_main, PREFIXES
from static_overlap_training import (TrainConfig, hard_replay_examples, worst_forget_guard,
                                     worst_forget_status, weighted_forget_loss)
from static_overlap_data import validate_bundle


def test_average_improvement_cannot_mask_worse_hardest_target():
    before = {"hard": .25, "easy": 3.}
    after = {"hard": .20, "easy": 5.}
    targets, weights = {"hard": 14., "easy": 14.}, {"hard": 4., "easy": 1.}
    assert weighted_forget_loss(after, targets, weights) < weighted_forget_loss(before, targets, weights)
    passed, report = worst_forget_guard(after, targets, worst_forget_status(before, targets))
    assert not passed and report["worst_forget_violating_ids"] == ["hard"]
    assert report["worst_forget_after"]["max_token_probability"] > report["worst_forget_before"]["max_token_probability"]


def test_guard_checks_probability_and_relative_target_separately_without_slack_ratcheting():
    targets = {"relative": 30., "probability": 14.}
    before = {"relative": 17., "probability": 3.}
    status = worst_forget_status(before, targets)
    assert not worst_forget_guard({"relative": 18., "probability": 2.}, targets, status)[0]
    assert not worst_forget_guard({"relative": 16., "probability": 4.}, targets, status)[0]
    assert not worst_forget_guard({"relative": 17. - 1e-10, "probability": 4.}, targets, status)[0]
    assert worst_forget_guard({"relative": 18., "probability": 4.}, targets, status)[0]
    with pytest.raises(ValueError, match="finite"):
        worst_forget_guard({"relative": math.nan, "probability": 4.}, targets, status)


def test_replay_refreshes_hardness_without_dropping_coverage_or_duplicating_ids():
    examples = [SimpleNamespace(id=key, fact_id=fact) for key, fact in
                [("a", "a"), ("a_view", "a"), ("b", "b"), ("c", "c"), ("d", "d")]]
    coverage = [examples[2], examples[3]]
    targets = {e.id: 14. for e in examples}
    nlls = dict(a=.2, a_view=.3, b=2., c=3., d=4.)
    replay = hard_replay_examples(examples, coverage, nlls, targets, 2)
    assert [e.id for e in replay] == ["a", "d"]
    assert len({e.id for e in coverage + replay}) == len(coverage + replay)
    nlls.update(a=5., a_view=5., d=.1)
    assert hard_replay_examples(examples, coverage, nlls, targets, 2)[0].id == "d"
    assert hard_replay_examples(examples, coverage, nlls, targets, 0) == []


@pytest.mark.parametrize("kwargs", [{"hard_replay_size": -1}, {"hard_replay_size": 1.5},
                                    {"guard_worst_forget": 1}, {"lambda_worst_forget": -1.}])
def test_invalid_replay_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate()


def test_retention_augmentation_preserves_validation_and_original_forget_supervision(tmp_path):
    original = json.loads((ROOT / "config/static_overlap_training.example.json").read_text())
    unchanged = deepcopy(original)
    augmented, report = augment_retention(original)
    facts = validate_bundle(augmented)
    assert original == unchanged
    assert augmented["facts"] == original["facts"]
    assert [r for r in augmented["examples"] if r["split"] == "validation"] == [
        r for r in original["examples"] if r["split"] == "validation"]
    added = augmented["examples"][len(original["examples"]):]
    assert len(added) == report["added_rows"] > 0
    for row, source in zip(added, report["sources"]):
        assert row["split"] == "train"
        assert all(facts[s["fact_id"]]["role"] == "retain" for s in row["spans"])
        for span in row["spans"]:
            assert row["completion"][span["start"]:span["end"]] == facts[span["fact_id"]]["object"]
        source_row = next(r for r in original["examples"] if r["id"] == source["source_training_id"])
        assert source_row["split"] == "train"
        assert row["prompt"] == PREFIXES[source["prefix_index"]] + source_row["prompt"]
    assert any("I don't know." in r["completion"] for r in added)
    assert not report["official_evaluation_read"] and not report["added_forget_supervision"]
    with pytest.raises(ValueError, match="already contains"):
        augment_retention(augmented)
    source, out = tmp_path / "input.json", tmp_path / "augmented.json"
    source.write_text(json.dumps(original))
    augment_main(["--training-bundle", str(source), "--out", str(out)])
    assert json.loads(out.read_text()) == augmented
    assert out.with_suffix(".augmentation.json").is_file()
    with pytest.raises(SystemExit):
        augment_main(["--training-bundle", str(source), "--out", str(out)])


def test_augmented_prompt_collision_with_validation_is_rejected():
    bundle = json.loads((ROOT / "config/static_overlap_training.example.json").read_text())
    train_row = next(r for r in bundle["examples"] if r["split"] == "train" and r["id"].startswith("train_retain"))
    validation_row = next(r for r in bundle["examples"] if r["split"] == "validation" and "prompt" in r)
    validation_row["prompt"] = PREFIXES[0] + train_row["prompt"]
    with pytest.raises(ValueError, match="prompt cannot cross|Duplicate text"):
        augment_retention(bundle)
