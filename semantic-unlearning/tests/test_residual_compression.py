"""Invariants of residual-bank compression.

The claims these protect:
  * a compressed variant changes only the residual rows: router, threshold,
    subject patterns and facts are identical, so routing decisions are too;
  * the rows written to the artifact are exactly reconstruct(compact);
  * rank-K at full rank is the original bank, and norm rescaling restores
    every row's norm;
  * int8 error is bounded by half a quantization step;
  * tied variants share one direction per group and keep each fact's norm;
  * the shuffled control never gives a fact its own row;
  * storage accounting matches the formulas.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compress_residual_bank import (  # noqa: E402
    ARTIFACT_NAME,
    MANIFEST_NAME,
    REPORT_NAME,
    break_even_facts,
    build_variants,
    check_frozen_config,
    derangement,
    int8_compact,
    low_rank_compact,
    low_rank_int8_compact,
    main,
    per_n_bytes,
    quantize_int8,
    dequantize_int8,
    reconstruct,
    shuffled_compact,
    spectrum,
    storage,
    tied_compact,
)
from linear_router import (  # noqa: E402
    LinearClassifierAssociationBank,
    load_linear_classifier_artifact,
)
from summarize_residual_compression import summarize  # noqa: E402

HIDDEN = 16
FACTS = 6
WIDTH = 6


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


def _facts():
    objects = ["French", "French", "English", "Paris", "Paris", "Berlin"]
    relations = ["P103", "P103", "P103", "P36", "P36", "P36"]
    return [
        {"id": f"mcf_{i}", "subject": f"S{i}", "relation": relations[i], "object": objects[i]}
        for i in range(FACTS)
    ]


def _bank(per_head=None, gate_mode="threshold", seed=0):
    torch.manual_seed(seed)
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, torch.randn(FACTS, HIDDEN), torch.zeros(FACTS),
        feature_mean=torch.zeros(HIDDEN), feature_components=None,
        threshold=0.0, subject_patterns=[[(10 + i,)] for i in range(FACTS)],
        facts=_facts(), rows=torch.randn(FACTS, HIDDEN) * 3.0,
        ambiguity_margin=0.5, gate_mode=gate_mode, per_head_thresholds=per_head,
    )


def _ids():
    return torch.tensor([
        [10, 1, 2, 3, 4, 5],
        [11, 1, 2, 3, 4, 5],
        [99, 1, 2, 3, 4, 5],
        [13, 14, 2, 3, 4, 5],
    ])


def _run(bank, hidden):
    ids = _ids()
    bank.bind(ids, attention_mask=torch.ones_like(ids))
    edited = bank._hook(None, None, (hidden.clone(),))[0]
    active = [list(x) for x in bank.last_active_fact_indices]
    bank.unbind()
    return edited, active


def _rows(seed=0, n=FACTS, d=HIDDEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g, dtype=torch.float64) * 2.0


def _write_run(tmp_path, bank):
    run = tmp_path / "run"
    run.mkdir()
    torch.save(bank.artifact(), run / ARTIFACT_NAME)
    (run / MANIFEST_NAME).write_text(json.dumps({"protocol_id": "test"}))
    return run


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def test_full_rank_is_exact_and_truncation_keeps_norms():
    rows = _rows()
    full = low_rank_compact(rows, FACTS, rescale=False)
    assert torch.allclose(reconstruct(full).double(), rows, atol=1e-5)
    truncated = low_rank_compact(rows, 2, rescale=True)
    rebuilt = reconstruct(truncated).double()
    assert truncated["codes"].shape == (FACTS, 2)
    assert truncated["basis"].shape == (2, HIDDEN)
    assert torch.allclose(rebuilt.norm(dim=1), rows.norm(dim=1), rtol=1e-5)
    shrunk = reconstruct(low_rank_compact(rows, 2, rescale=False)).double()
    assert bool((shrunk.norm(dim=1) <= rows.norm(dim=1) + 1e-6).all())


def test_spectrum_is_monotone_and_complete():
    info = spectrum(_rows())
    cumulative = info["cumulative_energy"]
    assert all(b >= a - 1e-12 for a, b in zip(cumulative, cumulative[1:]))
    assert cumulative[-1] == pytest.approx(1.0)
    assert 1 <= info["rank_for_energy"]["0.9"] <= FACTS


def test_int8_error_is_at_most_half_a_step():
    rows = _rows()
    q, scale = quantize_int8(rows)
    assert q.dtype == torch.int8
    error = (dequantize_int8(q, scale).double() - rows).abs()
    assert bool((error <= scale.double()[:, None] / 2 + 1e-6).all())
    lr8 = low_rank_int8_compact(rows, 3)
    assert lr8["codes_q"].dtype == torch.int8 and lr8["basis_q"].dtype == torch.int8
    close = reconstruct(low_rank_compact(rows, 3)).double()
    assert torch.allclose(reconstruct(lr8).double(), close, atol=0.1 * float(close.abs().max()))


def test_tied_rows_share_direction_and_keep_norms():
    rows = _rows()
    keys = [f["object"].casefold() for f in _facts()]
    compact = tied_compact(rows, keys)
    rebuilt = reconstruct(compact).double()
    assert compact["groups"] == 4
    assert torch.allclose(rebuilt.norm(dim=1), rows.norm(dim=1), rtol=1e-5)
    units = torch.nn.functional.normalize(rebuilt, dim=1)
    assert float(units[0] @ units[1]) == pytest.approx(1.0, abs=1e-5)   # both "french"
    assert float(units[3] @ units[4]) == pytest.approx(1.0, abs=1e-5)   # both "paris"
    assert float(units[0] @ units[2]) < 0.999                            # french vs english
    single = tied_compact(rows, ["all"] * FACTS)
    assert single["directions"].shape == (1, HIDDEN)


def test_shuffled_control_is_a_derangement_at_own_norm():
    for n in (2, 3, 7, 50):
        perm = derangement(n, seed=3)
        assert sorted(perm.tolist()) == list(range(n))
        assert all(int(perm[i]) != i for i in range(n))
    rows = _rows()
    rebuilt = reconstruct(shuffled_compact(rows, 0)).double()
    assert torch.allclose(rebuilt.norm(dim=1), rows.norm(dim=1), rtol=1e-5)
    cos = torch.nn.functional.cosine_similarity(rebuilt, rows, dim=1)
    assert bool((cos < 0.999).all())


def test_storage_accounting():
    rows = _rows(n=50, d=3072)
    lr = low_rank_compact(rows, 32)
    s = storage(lr, 50, 3072)
    assert s["bytes_as_saved"] == 4 * (50 * 32 + 32 * 3072)
    assert s["full_rows_fp32_bytes"] == 4 * 50 * 3072
    assert per_n_bytes(lr, 100_000, 3072) == 4 * (100_000 * 32 + 32 * 3072)
    n_star = break_even_facts(lr, 3072)
    assert per_n_bytes(lr, n_star, 3072) < 4 * n_star * 3072
    assert per_n_bytes(lr, n_star - 1, 3072) >= 4 * (n_star - 1) * 3072
    i8 = int8_compact(rows)
    assert storage(i8, 50, 3072)["bytes_as_saved"] == 50 * 3072 + 4 * 50
    lr8 = low_rank_int8_compact(rows, 32)
    m = break_even_facts(lr8, 3072)
    assert per_n_bytes(lr8, m, 3072) < 4 * m * 3072 <= per_n_bytes(lr8, m - 1, 3072) + 4 * 3072


def test_variant_set_skips_exact_ranks_and_single_relation_duplicates():
    rows = _rows()
    names = [name for name, _ in build_variants(rows, _facts(), ranks=(1, 2, 6, 99))]
    assert "rank1" in names and "rank2" in names
    assert "rank6" not in names and "rank99" not in names
    assert {"int8", "tied_answer", "tied_relation", "tied_single",
            "control_shuffled", "control_random"} <= set(names)
    one_relation = [dict(f, relation="zsre_direct_request_context") for f in _facts()]
    names = [name for name, _ in build_variants(rows, one_relation, ranks=(1,))]
    assert "tied_relation" not in names and "tied_single" in names


# ---------------------------------------------------------------------------
# Frozen configuration and artifacts
# ---------------------------------------------------------------------------

def test_refuses_per_head_and_non_linear_artifacts():
    per_head = _bank(per_head=torch.zeros(FACTS)).artifact()
    with pytest.raises(ValueError, match="per-head"):
        check_frozen_config(per_head)
    assert check_frozen_config(per_head, allow_per_head=True) == ("threshold", "per_head")
    assert check_frozen_config(_bank().artifact()) == ("threshold", "global")
    assert check_frozen_config(_bank(gate_mode="subject").artifact()) == ("subject", "subject_gate")
    with pytest.raises(ValueError, match="linear-router"):
        check_frozen_config({"architecture": "relation_prototype_fact_association_bank_v2"})


def test_end_to_end_variants_route_identically_and_store_reconstructed_rows(tmp_path):
    source_bank = _bank()
    run = _write_run(tmp_path, source_bank)
    out = tmp_path / "comp"
    assert main(["--run-dir", str(run), "--output-dir", str(out), "--ranks", "1,2,4"]) == 0

    report = json.loads((out / REPORT_NAME).read_text())
    assert report["benchmark"] == "mcf" and report["n_facts"] == FACTS
    assert (out / "run_evals.sh").is_file()
    script = (out / "run_evals.sh").read_text()
    assert "evaluate_static_overlap_fact_association_embeddings_official.py" in script

    torch.manual_seed(5)
    hidden = torch.randn(_ids().shape[0], WIDTH, HIDDEN)
    source_edit, source_routes = _run(source_bank, hidden)
    assert any(source_routes)                     # something routes in this batch
    source_artifact = torch.load(run / ARTIFACT_NAME, weights_only=False)

    for name, meta in report["variants"].items():
        artifact = torch.load(Path(meta["run_dir"]) / ARTIFACT_NAME, weights_only=False)
        for key in ("router_weight", "router_bias", "feature_mean"):
            assert torch.equal(artifact[key], source_artifact[key]), (name, key)
        assert artifact["threshold"] == source_artifact["threshold"]
        assert artifact["subject_patterns"] == source_artifact["subject_patterns"]
        assert torch.equal(artifact["rows"], reconstruct(artifact["residual_compact"]))
        _, bank = load_linear_classifier_artifact(_FakeModel(), artifact)
        edited, routes = _run(bank, hidden)
        assert routes == source_routes, name
        unrouted = [i for i, r in enumerate(routes) if not r]
        for i in unrouted:
            assert torch.equal(edited[i], hidden[i])
        manifest = json.loads((Path(meta["run_dir"]) / MANIFEST_NAME).read_text())
        assert manifest["router_unchanged_from_source"] is True
        assert manifest["protocol_id"] == "test"

    with pytest.raises(FileExistsError):
        main(["--run-dir", str(run), "--output-dir", str(out)])


def test_compressed_variant_cannot_be_compressed_again(tmp_path):
    run = _write_run(tmp_path, _bank())
    out = tmp_path / "comp"
    main(["--run-dir", str(run), "--output-dir", str(out), "--ranks", "2", "--no-int8",
          "--no-tied", "--no-controls"])
    with pytest.raises(ValueError, match="already a compressed"):
        main(["--run-dir", str(out / "variants" / "rank2"), "--output-dir", str(tmp_path / "x")])


def test_summary_table_reads_source_and_variant_evals(tmp_path):
    run = _write_run(tmp_path, _bank())
    out = tmp_path / "comp"
    main(["--run-dir", str(run), "--output-dir", str(out), "--ranks", "2", "--no-int8",
          "--no-tied", "--no-controls"])
    fake = {"forget": {"Eff": 0.0002, "Gen": 0.003, "Spe": 20.4},
            "retain": {"Eff": 12.0, "Gen": 12.0, "Spe": 19.6},
            "static_branch_display_zero_check": {"passed": True}, "forget_PPL": 11.31}
    (run / "official_mcf_eval.json").write_text(json.dumps(fake))
    (out / "variants" / "rank2" / "official_mcf_eval.json").write_text(json.dumps(
        dict(fake, forget={"Eff": 0.5, "Gen": 0.6, "Spe": 20.4})))
    summary = summarize(out)
    assert [r["variant"] for r in summary["rows"]] == ["source (uncompressed)", "rank2"]
    assert summary["rows"][1]["metrics"]["forget_Gen"] == 0.6
    assert summary["missing_evals"] == []
    assert "rank2" in summary["table_markdown"]
