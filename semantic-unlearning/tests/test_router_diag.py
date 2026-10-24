"""Router diagnostics end to end on a tiny Llama with a byte-level tokenizer.

The claims these protect:
  * the eval probes are exactly the official evaluators' groups and sizes, with genie
    ground truth (MCF / ZsRE / MQuAKE / RWKU on the real local data, MQuAKE synthetic);
  * cached last-token states equal extract_prompt_queries (the router's own features)
    and the real bank hook routes exactly as the cached L19 decision;
  * the cached refit at the router layer reproduces the shipped fit_linear_router.py
    router on the official prompts;
  * an ours-recipe router (fuzzy subject matching, fixed positive / negative sets) is
    cached under its own rule: hook parity holds on typo variants too, and the refit
    under that rule reproduces it;
  * metrics, confusion matrices, calibration and separability numbers are what they say.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("RWKU_CHAT_DATE_STRING", "26 Jul 2024")

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

import cache_router_states as cache_mod  # noqa: E402
import router_diag_probes as probes_lib  # noqa: E402
import router_diag_report as report  # noqa: E402
import router_layer_sweep_cached as sweep  # noqa: E402
from static_overlap_fact_association_embeddings import (  # noqa: E402
    extract_prompt_queries,
    subject_token_patterns,
)

MCF = ROOT / "data" / "multi_counterfact.json"
ZSRE = ROOT / "data" / "zsre_mend_eval.json"
RWKU = ROOT / "data" / "rwku"
WIKI = ROOT / "data" / "wikidata"
CHAT = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "{{ messages[0]['content'] }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
LAYERS, HIDDEN, ROUTER_LAYER = 4, 32, 2


# ---------------------------------------------------------------------------
# Tiny model + tokenizer
# ---------------------------------------------------------------------------

def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"]
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {tok: i for i, tok in enumerate(specials + alphabet)}
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    core.decoder = decoders.ByteLevel()
    core.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A", special_tokens=[("<|begin_of_text|>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=core, bos_token="<|begin_of_text|>",
                                  eos_token="<|eot_id|>", pad_token="<pad>",
                                  additional_special_tokens=specials[3:])
    tok.add_tokens(["Unknown"])            # MQuAKE's neutral target must be one token
    tok.chat_template = CHAT
    return tok


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_llama")
    tok = _tokenizer()
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    transformers.LlamaForCausalLM(config).eval().save_pretrained(path)
    tok.save_pretrained(path)
    return path


def _load(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32,
                                                 attn_implementation="eager").eval()
    return model, tok


def _fit_router(tmp, model_dir, facts, tok, extra=None, manifest_extra=None, fit_args=()):
    """A real router: fit_linear_router.py on a prep dir, as run_<ds>_layer_sweep_one.sh."""
    import fit_linear_router

    prep, router = tmp / "prep", tmp / "router"
    prep.mkdir()
    artifact = {"architecture": "untrained_association_rows_v1", "layer": ROUTER_LAYER,
                "facts": facts, "rows": torch.zeros(len(facts), HIDDEN),
                "subject_patterns": [subject_token_patterns(tok, f["subject"]) for f in facts],
                **(extra or {})}
    torch.save(artifact, prep / "fact_association_embeddings.pt")
    (prep / "association_manifest.json").write_text(json.dumps(
        {"model_path": str(model_dir), **(manifest_extra or {})}))
    fit_linear_router.main(["--run-dir", str(prep), "--output-dir", str(router), "--device", "cpu",
                            "--threshold-policy", "global", "--min-recall", "0.98",
                            "--threshold-placement-fraction", "0.1", "--lambdas", "1e-3,1e-1",
                            "--pca-dims", "0", "--cv-folds", "2", "--skip-runtime-parity", *fit_args])
    return router


def _cache(tmp, router, model_dir, dataset, extra_args=()):
    out = tmp / "cache"
    cache_mod.main(["--dataset", dataset, "--seed", "1", "--router-dir", str(router),
                    "--out-dir", str(out), "--model-path", str(model_dir), "--device", "cpu",
                    "--eval-dtype", "float32", "--batch-size", "8", "--retain-num", "3",
                    "--null-per-fact", "1", "--perturb-per-probe", "1", "--aug-per-probe", "1",
                    "--hardneg-per-fact", "1", "--parity-sample", "48", "--time-layers", "1,2",
                    *extra_args])
    return out


def _sweep(cache_dir, tmp, variants="baseline", hparams=("--hparams", "fixed")):
    sweep.main(["--cache-dirs", str(cache_dir), "--out-root", str(tmp / "root"), *hparams,
                "--variants", variants, "--robust-layers", f"{ROUTER_LAYER}", "--concat-layers", "1,2"])
    return json.loads((cache_dir / "layer_sweep" / "sweep.json").read_text())


def _check_cache(out, model_dir):
    manifest = json.loads((out / "manifest.json").read_text())
    for mode, rep in manifest["hook_parity"].items():
        assert rep["route_mismatches"] == 0, (mode, rep["examples"])
    model, tok = _load(model_dir)
    probes = json.loads((out / "probes_eval.json").read_text())
    states = torch.load(out / "states_eval.pt", weights_only=False)
    assert states["states"].shape[:2] == (LAYERS, len(probes))
    picked = [p["prompt"] for p in probes[:12] if p["prompt"] is not None]
    queries = extract_prompt_queries(model, tok, picked, ROUTER_LAYER, batch_size=4)
    cached = torch.nn.functional.normalize(states["states"][ROUTER_LAYER][:len(picked)].float(), dim=-1)
    assert torch.allclose(cached, queries, atol=1e-5)
    return manifest, probes


def _counts(probes):
    out = {}
    for p in probes:
        out[(p["split"], p["group"])] = out.get((p["split"], p["group"]), 0) + 1
    return out


# ---------------------------------------------------------------------------
# End to end, per benchmark
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not MCF.exists(), reason="data/multi_counterfact.json not present")
def test_mcf_end_to_end(tmp_path, model_dir):
    import mcf_zero_unlearn_official_eval as mcf

    _, tok = _load(model_dir)
    forget, _ = mcf.load_official_eval_records(str(MCF), 50, 1000, 1, "official")
    facts = [{"id": f"mcf_forget_{r['case_id']}", "subject": r["requested_rewrite"]["subject"],
              "relation": r["requested_rewrite"]["relation_id"], "case_id": int(r["case_id"]),
              "canonical_prompt": r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"])}
             for r in forget]
    router = _fit_router(tmp_path, model_dir, facts, tok, manifest_extra={"mcf_path": str(MCF)})
    out = _cache(tmp_path, router, model_dir, "mcf", ["--wikidata-dir", str(WIKI)])
    manifest, probes = _check_cache(out, model_dir)
    counts = _counts(probes)
    assert counts[("forget", "forget_rewrite")] == 50
    assert counts[("paraphrase", "forget_paraphrase")] == 100
    assert counts[("neighborhood", "forget_neighborhood")] == 500
    assert counts[("retain", "retain_rewrite")] == 3
    assert all(p["true_facts"] for p in probes if p["split"] in ("forget", "paraphrase"))
    nulls = json.loads((out / "probes_null.json").read_text())
    groups = {p["group"] for p in nulls}
    assert {"null_generic", "null_unseen_subject", "null_protected_other_relation"} <= groups
    assert not any(p["should_fire"] for p in nulls)
    protected = {f["subject"].casefold() for f in facts}
    assert not any(p["subject"].casefold() in protected for p in nulls if p["group"] == "null_unseen_subject")
    aug = json.loads((out / "probes_aug.json").read_text())
    assert {"hardneg_same_subject", "hardneg_same_relation"} <= {p["group"] for p in aug}
    assert all(p["split"] == "fit" for p in aug)
    eval_prompts = {p["prompt"] for p in probes}
    assert not any(p["prompt"] in eval_prompts for p in aug)
    assert manifest["router_pass_timing_ms_per_prompt"]["fraction_of_full"]["stop_after_L1"] > 0

    metrics = report.build_report(out)
    for name in ("summary.md", "metrics.json", "scores_eval.png", "confusion_binary.png",
                 "confusion_nway.png", "umap_L16_L19.png", "reliability.png", "scores_null.png"):
        assert (out / "report" / name).is_file(), name
    total = metrics["eval"]["by_split"]["all"]
    assert total["tp"] + total["fn"] + total["fp"] + total["tn"] == len(probes)

    result = _sweep(out, tmp_path, "baseline,hardneg,aug,concat,unmasked")
    at = next(e for e in result["sweep"] if e["layers"] == [ROUTER_LAYER] and e["rule"] == "deployed")
    assert at["agreement_with_shipped_router"] >= 0.99
    assert {e["variant"] for e in result["robust"]} == {"baseline", "hardneg", "aug", "concat", "unmasked"}
    assert (out / "layer_sweep" / "layer_choice.md").is_file()
    assert (tmp_path / "root" / "ROUTER_LAYER_CHOICE.md").is_file()
    before = (out / "layer_sweep" / "sweep.json").read_text()
    sweep.main(["--cache-dirs", str(out), "--out-root", str(tmp_path / "combined"), "--combine-only"])
    assert (out / "layer_sweep" / "sweep.json").read_text() == before      # nothing refit
    assert (tmp_path / "combined" / "ROUTER_LAYER_CHOICE.md").is_file()


@pytest.mark.skipif(not ZSRE.exists(), reason="data/zsre_mend_eval.json not present")
def test_zsre_end_to_end(tmp_path, model_dir):
    import zsre_zero_unlearn_official_eval as zsre

    _, tok = _load(model_dir)
    forget, _ = zsre.load_official_eval_records(ZSRE, tok, forget_num=50, retain_num=1000, seed=1)
    facts = [{"id": f"zsre_forget_{r['case_id']}", "subject": r["requested_rewrite"]["subject"],
              "relation": "zsre_direct_request_context", "case_id": int(r["case_id"]),
              "canonical_prompt": r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"])}
             for r in forget]
    router = _fit_router(tmp_path, model_dir, facts, tok)
    out = _cache(tmp_path, router, model_dir, "zsre", ["--zsre-path", str(ZSRE), "--no-ppl"])
    _, probes = _check_cache(out, model_dir)
    counts = _counts(probes)
    assert counts[("forget", "forget_rewrite")] == 50
    assert counts[("paraphrase", "forget_paraphrase")] == 50
    assert counts[("neighborhood", "forget_neighborhood")] >= 50
    assert counts[("retain", "retain_neighborhood")] >= 3
    report.build_report(out)
    # grouped CV with the shipped folds over a sub-grid: the shipped L2 is re-selected
    result = _sweep(out, tmp_path, hparams=("--hparams", "cv", "--lambdas", "1e-1", "--cv-folds", "2"))
    at = next(e for e in result["sweep"] if e["layers"] == [ROUTER_LAYER])
    shipped = json.loads((router / "association_manifest.json").read_text())["router"]["selected_l2"]
    assert at["l2"] == shipped
    assert at["agreement_with_shipped_router"] >= 0.99


@pytest.mark.skipif(not ZSRE.exists(), reason="data/zsre_mend_eval.json not present")
def test_ours_fuzzy_router_end_to_end(tmp_path, model_dir):
    import zsre_zero_unlearn_official_eval as zsre

    _, tok = _load(model_dir)
    forget, _ = zsre.load_official_eval_records(ZSRE, tok, forget_num=50, retain_num=1000, seed=1)
    facts = [{"id": f"zsre_forget_{r['case_id']}", "subject": r["requested_rewrite"]["subject"],
              "relation": "zsre_direct_request_context", "case_id": int(r["case_id"]),
              "canonical_prompt": r["requested_rewrite"]["prompt"].format(r["requested_rewrite"]["subject"])}
             for r in forget]
    router = _fit_router(tmp_path, model_dir, facts, tok, fit_args=(
        "--subject-match", "fuzzy", "--positives-per-fact", "6", "--negatives-per-fact", "4",
        "--max-prefix-positives", "4"))
    out = _cache(tmp_path, router, model_dir, "zsre",
                 ["--zsre-path", str(ZSRE), "--no-ppl", "--router-recipe", "ours"])
    manifest, _ = _check_cache(out, model_dir)          # hook parity 0, eval and perturbed
    assert manifest["subject_match"] == "fuzzy" and manifest["router_recipe"] == "ours"
    assert {"batched", "single", "perturbed_batched", "perturbed_single"} <= set(manifest["hook_parity"])
    pert = torch.load(out / "router_perturbed.pt", weights_only=False)
    assert bool((pert["eligible"] & ~pert["eligible_exact"]).any())     # typos only fuzzy catches
    assert not bool((pert["eligible_exact"] & ~pert["eligible"]).any())  # fuzzy contains exact
    trained = json.loads((router / "linear_router_dataset.json").read_text())
    assert len(json.loads((out / "probes_train.json").read_text())) == len(trained)
    report.build_report(out)
    assert "subject match `fuzzy`" in (out / "report" / "summary.md").read_text()
    result = _sweep(out, tmp_path)
    at = next(e for e in result["sweep"] if e["layers"] == [ROUTER_LAYER] and e["rule"] == "deployed")
    assert at["deployed_eligibility"] == "fuzzy" and set(at["by_eligibility"]) == {"fuzzy"}
    assert at["agreement_with_shipped_router"] >= 0.99
    modes = result["robust"][0]["by_eligibility"]
    assert set(modes) == {"exact", "fuzzy", "none"}
    assert modes["fuzzy"]["perturbed_recall"] >= modes["exact"]["perturbed_recall"]
    doc = (out / "layer_sweep" / "layer_choice.md").read_text()
    assert "Fuzzy subject eligibility" in doc and "ONE pass" in doc
    import router_layer_sweep_cached as rls
    assert "ONEPASS=1 WRITE_LAYER=19 ROUTER_LAYER=26" in rls.downstream_command(manifest, 26)


@pytest.mark.skipif(not RWKU.exists(), reason="data/rwku not present")
def test_rwku_end_to_end(tmp_path, model_dir):
    from rwku_batch50 import build_batch_split
    from rwku_fact_association_embeddings import build_association_facts, make_rwku_subject_patterns

    _, tok = _load(model_dir)
    split = build_batch_split(data_root=RWKU.resolve(), batch_seed=1, allow_download=False)
    facts, _, _ = build_association_facts(list(split["efficacy_forget"]), tok)
    router = _fit_router(tmp_path, model_dir, facts, tok,
                         extra={"subject_patterns": make_rwku_subject_patterns(tok, facts)})
    out = _cache(tmp_path, router, model_dir, "rwku", ["--rwku-root", str(RWKU), "--no-ppl"])
    manifest, probes = _check_cache(out, model_dir)
    assert manifest["gate_mode"] == "subject"
    counts = _counts(probes)
    assert counts[("forget", "same50")] == 50
    assert counts[("paraphrase", "heldout_level1")] == 43
    assert counts[("paraphrase", "heldout_level2")] == 50
    assert counts[("paraphrase", "adversarial_level3")] == 173
    assert counts[("neighborhood", "neighbors")] == 279
    types = {p["probe_type"] for p in probes}
    assert {"fill_in_blank", "qa", "qa_paraphrase", "neighbor_fill_in_blank", "neighbor_qa"} <= types
    assert any(t.startswith("adversarial:") for t in types)
    heldout = [p for p in probes if p["group"] == "heldout_level2"]
    assert all(len(p["true_facts"]) > 1 for p in heldout)        # entity level: any of the person's rows
    metrics = report.build_report(out)
    assert "rwku" in metrics and (out / "report" / "confusion_person.png").is_file()
    result = _sweep(out, tmp_path)
    assert {e["rule"] for e in result["sweep"]} == {"deployed", "threshold"}


def _synthetic_mquake(path):
    relations = [("P17", "{} is located in the country of", "Which country is {} located in?"),
                 ("P36", "The capital of {} is", "What is the capital of {}?"),
                 ("P19", "{} was born in", "Where was {} born?"),
                 ("P27", "{} is a citizen of", "Which country is {} a citizen of?")]
    rng = random.Random(0)
    raw = []
    for i in range(3000):
        rewrites = []
        for k in range(1 + i % 2):
            rel, prompt, question = relations[(i + k) % len(relations)]
            subject = f"Entity {i % 1500} {k}"            # i and i + 1500 share rewrites
            rewrites.append({"prompt": prompt, "subject": subject, "relation_id": rel,
                             "target_true": {"str": f"Place {(i % 1500) * 7 % 97}"},
                             "target_new": {"str": f"Place {rng.randrange(97)}"},
                             "question": question.format(subject)})
        raw.append({"case_id": i, "requested_rewrite": rewrites, "questions": [], "answer": ""})
    path.write_text(json.dumps(raw))


def test_mquake_end_to_end(tmp_path, model_dir):
    import mquake_zero_unlearn_official_eval as mquake
    from mquake_fact_association_embeddings import build_association_facts

    data = tmp_path / "mquake.json"
    _synthetic_mquake(data)
    _, tok = _load(model_dir)
    forget, _ = mquake.load_official_eval_records(data, tok, forget_num=50, retain_num=1000, seed=1)
    facts, case_to_fact, _ = build_association_facts(forget)
    for f in facts:
        f.setdefault("canonical_prompt", f["canonical_prompts"][0] if f.get("canonical_prompts") else "")
    router = _fit_router(tmp_path, model_dir, facts, tok, extra={
        "atomic_case_to_association_id": {str(k): v for k, v in case_to_fact.items()}})
    out = _cache(tmp_path, router, model_dir, "mquake", ["--mquake-path", str(data), "--no-ppl"])
    _, probes = _check_cache(out, model_dir)
    counts = _counts(probes)
    assert counts[("forget", "forget_rewrite")] == len(forget)
    assert counts[("paraphrase", "forget_atomic_gen")] == len(forget)
    overlap = [p for p in probes if p["split"] == "retain" and p["meta"].get("identical_to_forget_prompt")]
    assert all(p["should_fire"] and p["true_facts"] for p in overlap)
    report.build_report(out)
    _sweep(out, tmp_path)


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------

def _routes(logits, eligible, threshold=0.0, margin=0.5):
    from linear_router import decide_routes

    logits = torch.tensor(logits)
    eligible = torch.tensor(eligible)
    return {"logits": logits, "eligible": eligible,
            **decide_routes(logits, eligible, threshold, margin)}


def _p(split, should, true=()):
    return probes_lib.probe("x", "eval", split, split, should, true)


def test_outcomes_and_metrics():
    probes = [_p("forget", True, [0]), _p("forget", True, [1]), _p("forget", True, [0]),
              _p("neighborhood", False), _p("neighborhood", False)]
    routes = _routes([[2.0, -1.0], [3.0, -2.0], [-1.0, -1.0], [1.0, 0.0], [5.0, 5.0]],
                     [[True, False], [True, True], [True, False], [False, False], [True, False]])
    rows = report.probe_outcomes(probes, routes)
    assert [r["category"] for r in rows] == ["correct", "misroute", "false_negative", "correct",
                                             "false_positive"]
    assert rows[2]["reason"] == "below_threshold" and not rows[3]["relevant_eligible"]
    m = report.binary_metrics(rows)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["fpr"] == 0.5 and abs(m["fnr"] - 1 / 3) < 1e-12 and m["correct_row_rate"] == 1 / 3
    matrix = report.nway_confusion(rows, 2)
    assert matrix.tolist() == [[1, 0, 1], [1, 0, 0], [1, 0, 1]]


def test_ece_and_separability():
    ece, _ = report.expected_calibration_error([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0], bins=10)
    assert abs(ece - 0.1) < 1e-12
    x = torch.cat([torch.randn(10, 8) + 4, torch.randn(10, 8) - 4])
    labels = [1] * 10 + [0] * 10
    assert report.silhouette(x, labels, "euclidean") > 0.5
    loo = report.loo_probe_accuracy(x, labels)
    assert loo["logistic"] == 1.0 and loo["nearest_neighbour"] == 1.0


def test_fuzzy_eligibility_catches_typos_not_other_names():
    surfaces = [[sweep._plain_words("Danielle Darrieux")]]
    probes = [probes_lib.probe(t, "eval", "forget", "g", True, [0]) for t in (
        "The mother tongue of Danielle Darieux is",      # deletion
        "The mother tongue of danielle darrieux is",     # case
        "The mother tongue of Daniel Craig is")]
    out = sweep.fuzzy_eligibility(probes, surfaces, torch.zeros((3, 1), dtype=torch.bool))
    assert out[:, 0].tolist() == [True, True, False]


def test_swap_person_replaces_every_surface():
    text = probes_lib.swap_person("Warren Buffett said Buffett would", "Warren Buffett", "Stephen King")
    assert text == "Stephen King said King would"
    assert probes_lib.swap_person("no name here", "Warren Buffett", "Stephen King") is None
