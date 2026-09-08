from pathlib import Path
import sys

import pytest
from datasets import Dataset, DatasetDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_static_overlap_mcf_bundles as builder


def corpus(monkeypatch, texts):
    dataset = DatasetDict({"train": Dataset.from_dict({"text": texts})})
    monkeypatch.setattr(builder, "load_from_disk", lambda _path: dataset)


def test_repeated_ten_sentence_fixture_splits_without_leakage(monkeypatch):
    original = [f"Distinct source sentence {i}." for i in range(10)]
    corpus(monkeypatch, original * 20)
    train, validation, test, report = builder.language_rows("fixture", return_summary=True)
    assert (len(train), len(validation), len(test)) == (4, 2, 4)
    assert train + validation + test == original
    assert not set(train) & set(validation)
    assert not set(train + validation) & set(test)
    assert report["raw_rows"] == 200
    assert report["unique_rows"] == 10
    assert report["duplicate_rows_removed"] == 190
    assert report["requested_rows"] == {"train": 12, "validation": 6, "test": 12}
    assert report["actual_rows"] == {"train": 4, "validation": 2, "test": 4}
    assert report["reduced_to_available_rows"]
    assert not report["official_ppl_held_out_from_fitting_and_validation"]
    assert report["official_ppl_first_20_overlap_rows"] == {"train": 4, "validation": 2, "test": 4}
    assert builder.language_rows("fixture") == (train, validation, test)


def test_full_corpus_keeps_requested_counts(monkeypatch):
    original = [f"Original document {i}." for i in range(40)]
    corpus(monkeypatch, original)
    train, validation, test, report = builder.language_rows("full", strict=True, return_summary=True)
    assert train == original[:12]
    assert validation == original[12:18]
    assert test == original[18:30]
    assert not report["reduced_to_available_rows"]


def test_strict_mode_preserves_insufficient_count_error(monkeypatch):
    corpus(monkeypatch, [f"Sentence {i}." for i in range(10)] * 20)
    with pytest.raises(ValueError, match="10 unique rows, need 30"):
        builder.language_rows("fixture", strict=True)


@pytest.mark.parametrize("size", [0, 1, 2])
def test_cannot_duplicate_a_tiny_corpus_into_three_splits(monkeypatch, size):
    corpus(monkeypatch, [f"Sentence {i}." for i in range(size)])
    with pytest.raises(ValueError, match="at least 3"):
        builder.language_rows("too_small")


def test_whitespace_case_duplicates_and_empty_rows_are_not_extra_evidence(monkeypatch):
    corpus(monkeypatch, ["First sentence.", " FIRST   sentence. ", "Second sentence.",
                         "Third sentence.", "", "  ", None])
    train, validation, test, report = builder.language_rows("normalized", return_summary=True)
    assert (train, validation, test) == (["First sentence."], ["Second sentence."], ["Third sentence."])
    assert report["nonempty_text_rows"] == 4
    assert report["unique_rows"] == 3


@pytest.mark.parametrize("requested", [(12, 6, 12), (1, 2, 30), (30, 1, 1), (1, 1, 1)])
def test_allocations_are_nonempty_bounded_and_deterministic(requested):
    for available in range(3, sum(requested) + 4):
        counts = builder.language_split_counts(available, requested)
        assert sum(counts) == min(available, sum(requested))
        assert all(1 <= actual <= wanted for actual, wanted in zip(counts, requested))
        assert counts == builder.language_split_counts(available, requested)


@pytest.mark.parametrize("requested", [(0, 6, 12), (12, -1, 12), (12, 6, 1.5)])
def test_requested_counts_must_be_positive_integers(requested):
    with pytest.raises(ValueError, match="positive integers"):
        builder.language_split_counts(10, requested)


def test_loads_saved_dataset_dict(tmp_path):
    path = tmp_path / "corpus"
    DatasetDict({"train": Dataset.from_dict({"text": [f"Document {i}." for i in range(10)] * 20})}).save_to_disk(path)
    train, validation, test = builder.language_rows(path)
    assert (len(train), len(validation), len(test)) == (4, 2, 4)
