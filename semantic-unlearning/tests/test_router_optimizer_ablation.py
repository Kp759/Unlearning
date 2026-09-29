"""Router optimizer ablation: SGD fitter, CLI plumbing and the comparison report."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_router_optimizers as cmp  # noqa: E402
import fit_linear_router_sgd as sgd  # noqa: E402
import linear_router  # noqa: E402
from test_linear_router import FACTS, HIDDEN, _synthetic  # noqa: E402

CONFIG = dict(lr="auto", lr_scale=1.0, momentum=0.9, nesterov=False, batch_size=16,
              epochs=300, schedule="cosine", seed=0, dtype="float64", log_every=50, twin=True)


def _fit(l2, config=CONFIG):
    queries, labels, eligible, _, _ = _synthetic()
    original = linear_router._fit_heads
    linear_router._fit_heads = lambda *a: sgd.sgd_fit_heads(*a, config=config)
    try:
        return linear_router.fit_linear_router(queries, labels, eligible, l2=l2)
    finally:
        linear_router._fit_heads = original


def test_sgd_reaches_the_lbfgs_optimum_when_well_conditioned():
    info = _fit(1e-2)["info"]
    twin = info["lbfgs_twin_same_features"]
    assert info["optimizer"] == "sgd"
    assert abs(twin["relative_objective_gap"]) < 1e-4
    assert twin["weight_cosine"]["min"] > 0.999
    assert twin["fit_pairs_sign_disagreement_at_zero"] == 0


def test_sgd_objective_is_never_below_the_lbfgs_optimum():
    for l2 in (1e-1, 1e-4):
        twin = _fit(l2)["info"]["lbfgs_twin_same_features"]
        assert twin["objective_gap_sgd_minus_lbfgs"] >= -1e-9


def test_sgd_is_deterministic_for_a_seed():
    a, b = _fit(1e-3), _fit(1e-3)
    assert torch.equal(a["weight"], b["weight"]) and torch.equal(a["bias"], b["bias"])


def test_minibatch_objective_with_all_rows_equals_full_objective():
    queries, labels, eligible, _, _ = _synthetic()
    phi = linear_router.apply_feature_map(queries.double(), *linear_router.fit_feature_map(queries))
    pw = linear_router._pair_weights(labels.bool() & eligible, eligible, True)
    w = torch.randn(FACTS, phi.shape[1], dtype=torch.float64)
    b = torch.randn(FACTS, dtype=torch.float64)
    full = sgd._objective(phi, labels.double(), pw, w, b, 1e-3)
    every = sgd._objective(phi, labels.double(), pw, w, b, 1e-3, torch.arange(phi.shape[0]))
    assert torch.allclose(full, every)


def test_smoothness_bound_orders():
    queries, labels, eligible, _, _ = _synthetic()
    phi = linear_router.apply_feature_map(queries.double(), *linear_router.fit_feature_map(queries))
    pw = linear_router._pair_weights(labels.bool() & eligible, eligible, True)
    bounds = sgd.smoothness_bounds(phi, pw, 1e-3, 8)
    assert bounds["minibatch_worst_case"] >= bounds["full_batch"] > 0


def test_patch_reaches_grouped_cv_through_the_module():
    queries, labels, eligible, _, splits = _synthetic()
    groups = [f"g{i % 3}" for i in range(queries.shape[0])]
    seen = []
    original = linear_router._fit_heads

    def spy(*a):
        seen.append(1)
        return sgd.sgd_fit_heads(*a, config={**CONFIG, "epochs": 5, "twin": False})

    linear_router._fit_heads = spy
    try:
        linear_router.select_hyperparameters(queries, labels, eligible, groups,
                                             lambdas=(1e-2,), pca_dims=(0,), folds=3)
    finally:
        linear_router._fit_heads = original
    assert len(seen) == 3


def _reference_router_dir(tmp_path, command):
    ref = tmp_path / "ref_router"
    ref.mkdir()
    (ref / "linear_router_report.json").write_text(json.dumps({
        "command": command,
        "router_fit": {"selected_l2": 1e-05, "selected_pca_dim": 64},
    }))
    return ref


def test_fit_argv_copies_reference_flags_and_pins_hyperparameters(tmp_path):
    ref = _reference_router_dir(tmp_path, [
        "scripts/fit_linear_router.py", "--run-dir", "/x/prep", "--output-dir", "/x/router",
        "--device", "cuda", "--min-recall", "0.98", "--lambdas", "1,2",
    ])
    argv, pinned = sgd.build_fit_argv(ref, "pinned", "/y/out", ["--batch-size", "8"])
    assert pinned == {"l2": 1e-05, "pca_dim": 64}
    assert argv.count("--output-dir") == 1 and argv[argv.index("--output-dir") + 1] == "/y/out"
    assert argv[argv.index("--lambdas") + 1] == "1e-05" and argv.count("--lambdas") == 1
    assert argv[argv.index("--pca-dims") + 1] == "64"
    assert argv[argv.index("--min-recall") + 1] == "0.98"
    assert argv[-2:] == ["--batch-size", "8"]
    full, _ = sgd.build_fit_argv(ref, "full", "/y/out", [])
    assert full[full.index("--lambdas") + 1] == "1,2"


def test_cli_patches_fit_and_restores_it(tmp_path, monkeypatch):
    ref = _reference_router_dir(tmp_path, ["--run-dir", "/x/prep", "--output-dir", "/x/router"])
    queries, labels, eligible, _, _ = _synthetic()

    def fake_main(argv):
        out = Path(argv[argv.index("--output-dir") + 1])
        out.mkdir(parents=True)
        model = linear_router.fit_linear_router(queries, labels, eligible, l2=1e-2)
        (out / "linear_router_report.json").write_text(json.dumps({"router_fit": {
            "selected_l2": 1e-2, "selected_pca_dim": 0, "fit_info": model["info"]}}))
        return 0

    monkeypatch.setitem(sys.modules, "fit_linear_router", types.SimpleNamespace(main=fake_main))
    original = linear_router._fit_heads
    out = tmp_path / "router_sgd"
    assert sgd.main(["--like", str(ref), "--output-dir", str(out), "--sgd-epochs", "50"]) == 0
    assert linear_router._fit_heads is original
    record = json.loads((out / "optimizer_ablation.json").read_text())
    assert record["optimizer"] == "sgd" and record["sgd"]["epochs"] == 50
    assert record["lbfgs_twin_same_features"] is not None
    control = tmp_path / "router_lbfgs"
    assert sgd.main(["--optimizer", "lbfgs", "--like", str(ref), "--output-dir", str(control)]) == 0
    assert json.loads((control / "optimizer_ablation.json").read_text())["final_fit"]["optimizer"] == "lbfgs"


# ---------------------------------------------------------------------------
# Comparison report on fake run directories
# ---------------------------------------------------------------------------

def _router(path, weight, bias, routes, components=None):
    path.mkdir(parents=True)
    facts = [{"id": f"mcf_{i}"} for i in range(FACTS)]
    torch.save({
        "facts": facts, "router_weight": weight, "router_bias": bias,
        "feature_mean": torch.zeros(HIDDEN), "feature_components": components,
        "bias_calibration": {"stage1_bias": bias + 1.0, "global_shift": 1.0},
    }, path / "fact_association_embeddings.pt")
    rows = [{"prompt": f"p{k}", "split": split, "owner_fact_id": owner, "kind": "x",
             "negative_for": [], "group": "g", "linear_routes_to": to,
             "linear_best_eligible_logit": 1.0, "linear_best_eligible_stage1_logit": 2.0,
             "linear_best_eligible_fact_id": owner or "mcf_0"}
            for k, (split, owner, to) in enumerate(routes)]
    (path / "linear_router_dataset.json").write_text(json.dumps(rows))
    (path / "linear_router_report.json").write_text(json.dumps({"router_fit": {
        "selected_l2": 1e-3, "selected_pca_dim": 0, "fit_info": {"objective": 0.1}},
        "route_outcomes_by_split": {"audit": {"correct_route": {"rate": 1.0}}}}))


def _run(path, gen, excluded=()):
    path.mkdir(parents=True)
    (path / "association_manifest.json").write_text(json.dumps({
        "views_excluded_unrouted": list(excluded), "untrainable_fact_ids": []}))
    (path / "official_mcf_eval.json").write_text(json.dumps({
        "forget": {"Eff": 0.0, "Gen": gen, "Spe": 50.0}, "retain": {"Eff": 90.0, "Gen": 80.0},
        "forget_PPL": 10.0}))


def test_comparison_reports_route_flips_and_noise_floor(tmp_path):
    w = torch.randn(FACTS, HIDDEN)
    b = torch.zeros(FACTS)
    base = [("audit", "mcf_0", "mcf_0"), ("audit", None, None), ("fit", "mcf_1", "mcf_1")]
    flipped = [("audit", "mcf_0", None), ("audit", None, "mcf_2"), ("fit", "mcf_1", "mcf_1")]
    _router(tmp_path / "ref", w, b, base)
    _router(tmp_path / "ctl", w.clone(), b.clone(), base)
    _router(tmp_path / "sgd", 0.5 * w, b + 0.2, flipped)
    _run(tmp_path / "ref_run", gen=1.0)
    _run(tmp_path / "ctl_run", gen=1.5)
    _run(tmp_path / "swap", gen=1.2)
    _run(tmp_path / "sgd_run", gen=3.0, excluded=["v1"])
    prefix = tmp_path / "out" / "comparison"
    cmp.main(["--dataset", "mcf",
              "--reference-router", str(tmp_path / "ref"), "--reference-run", str(tmp_path / "ref_run"),
              "--control-router", str(tmp_path / "ctl"), "--control-run", str(tmp_path / "ctl_run"),
              "--candidate-router", str(tmp_path / "sgd"),
              "--candidate-swap-run", str(tmp_path / "swap"),
              "--candidate-run", str(tmp_path / "sgd_run"), "--out-prefix", str(prefix)])
    result = json.loads(prefix.with_suffix(".json").read_text())
    assert result["routes"]["control"]["route_changes_total"] == 0
    audit = result["routes"]["candidate"]["by_split"]["audit"]
    assert audit["route_changes"] == 2 and audit["positives_newly_missed"] == 1
    assert audit["negatives_firing"] == {"reference": 0, "other": 1}
    assert result["parameters"]["control"]["hidden_space_weight_cosine"]["min"] == pytest.approx(1.0)
    assert result["parameters"]["candidate"]["weight_norm_ratio"] == pytest.approx(0.5)
    gen = next(r for r in result["metrics"] if r["metric"] == "forget_Gen")
    assert gen["noise_floor"] == pytest.approx(0.5)
    assert gen["verdict[candidate_swap]"] == "within noise"
    assert gen["verdict[candidate_full]"] == "differs"
    assert "DIFFERENT" in result["overall"]["pipeline"] and "forget_Gen" in result["overall"]["pipeline"]
    assert result["training_views"]["candidate_full"]["excluded_views"]["only_other"] == ["v1"]
    assert prefix.with_suffix(".md").read_text().startswith("# Router optimizer ablation")


def test_pca_sign_flip_is_not_a_difference(tmp_path):
    w = torch.randn(FACTS, 4)
    c = torch.linalg.qr(torch.randn(HIDDEN, 4))[0].T.contiguous()
    flip = torch.tensor([1.0, -1.0, 1.0, -1.0])
    routes = [("audit", "mcf_0", "mcf_0")]
    _router(tmp_path / "a", w, torch.zeros(FACTS), routes, components=c)
    _router(tmp_path / "b", w * flip, torch.zeros(FACTS), routes, components=c * flip[:, None])
    diff = cmp.parameter_diff(tmp_path / "a", tmp_path / "b")
    assert diff["hidden_space_weight_cosine"]["min"] == pytest.approx(1.0)


def test_identical_runs_read_as_same(tmp_path):
    w, b = torch.randn(FACTS, HIDDEN), torch.zeros(FACTS)
    routes = [("audit", "mcf_0", "mcf_0")]
    for name in ("ref", "sgd"):
        _router(tmp_path / name, w, b, routes)
    _run(tmp_path / "ref_run", gen=1.0)
    _run(tmp_path / "sgd_run", gen=1.0)
    prefix = tmp_path / "abl" / "mcf" / "seed1" / "L19" / "comparison"
    cmp.main(["--dataset", "mcf", "--reference-router", str(tmp_path / "ref"),
              "--reference-run", str(tmp_path / "ref_run"),
              "--candidate-router", str(tmp_path / "sgd"),
              "--candidate-run", str(tmp_path / "sgd_run"), "--out-prefix", str(prefix)])
    result = json.loads(prefix.with_suffix(".json").read_text())
    assert result["overall"]["pipeline"].startswith("pipeline: SAME")
    assert "identically" in result["overall"]["router"]
    assert cmp.main(["--collect", str(tmp_path / "abl")]) == 0
    assert (tmp_path / "abl" / "optimizer_ablation_summary.md").is_file()
