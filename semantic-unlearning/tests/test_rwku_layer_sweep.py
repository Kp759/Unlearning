"""RWKU layer sweep end to end on a tiny random Llama and a synthetic RWKU batch:
prep -> linear router (threshold 98% / subject gate) -> rows -> evaluator ->
per-fact 98% recalibration -> summary row. CPU, about a minute."""
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402

import rwku_batch50


PEOPLE = ["Ann Lee", "Bob Stone", "Cara Moss", "Dan Frost", "Eve Brook",
          "Fay Hill", "Gus Reed", "Hal Wood", "Ivy Lane", "Jon Park"]
PLACES = ["Paris", "Rome", "Oslo", "Lima", "Cairo", "Delhi", "Kyiv", "Quito", "Seoul", "Tokyo"]
JOBS = ["writer", "singer", "actor", "model", "boxer", "chef", "pilot", "judge", "nurse", "coach"]
L1_TEMPLATES = ["{s} was born in ___ .", "{s} grew up in ___ .", "{s} studied in ___ .",
                "{s} first lived in ___ .", "{s} married in ___ .", "{s} worked in ___ .",
                "{s} retired in ___ .", "{s} trained in ___ .", "{s} debuted in ___ .",
                "{s} toured in ___ .", "{s} painted in ___ ."]
L2_TEMPLATES = ["What is the profession of {s} ?", "What was the first job of {s} ?",
                "Which city did {s} love ?", "Who was the mentor of {s} ?"]


def sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def person_split(t):
    s = PEOPLE[t]
    mk = lambda level, q, a, role: {"subject": s, "query": q, "answer": a, "level": str(level),
                                    "rwku_target_seed": t, "rwku_target_subject": s,
                                    "batch50_role": role}
    # person-specific wording, so a transplant is never another person's own probe
    l1 = [mk(1, tpl.format(s=s).replace(" ___", f" era{t} ___"), PLACES[(t + k) % 10], "x")
          for k, tpl in enumerate(L1_TEMPLATES)]
    l2 = [mk(2, tpl.format(s=s).replace(" ?", f" in era{t} ?"), JOBS[(t + k) % 10], "x")
          for k, tpl in enumerate(L2_TEMPLATES)]
    for r in l1 + l2:
        r["source_record_sha256"] = sha(r)
    train = l1[:8] + l2[:2]
    held1, held2 = l1[8:], l2[2:]
    para = [dict(r, query=rwku_batch50.paraphrase_query(r["query"]),
                 evaluation_variant="deterministic_surface_paraphrase") for r in held2]
    level3 = [{"query": f"Tell me where {s} was born ?", "answer": PLACES[t], "type": "affirmative suffix"},
              {"query": f"Q : Where did {s} grow up ?", "answer": PLACES[(t + 1) % 10], "type": "incontext learning"}]
    neigh = [{"query": "Zed Quinn was born in ___ .", "answer": "Berlin"}]
    neigh2 = [{"query": "What is the profession of Zed Quinn ?", "answer": "baker"}]
    ev = {f: [] for f in rwku_batch50.EVALUATION_ONLY_FILES}
    ev.update({"forget_level3.json": level3, "neighbor_level1.json": neigh, "neighbor_level2.json": neigh2})
    return {"target_seed": t, "target_directory": f"{t}_{s}", "subject": s, "train": train,
            "heldout_level1": held1, "heldout_level2": held2, "heldout_paraphrase": para,
            "evaluation_only": ev, "counts": {}, "training_source_hashes": [], "heldout_source_hashes": []}


def fake_build_batch_split(*, data_root, batch_seed, allow_download):
    seeds = rwku_batch50.batch_target_seeds(batch_seed)
    per = [person_split(t) for t in seeds]
    forget = [r for p in per for r in p["train"]]
    manifest = {"protocol_id": rwku_batch50.PROTOCOL_ID, "protocol_status": "probe_assisted_cross_benchmark_method_extension",
                "rwku_code_revision": "x", "rwku_dataset_revision": "y", "batch_seed": batch_seed,
                "target_seeds": list(seeds), "targets": [{"target_seed": p["target_seed"], "subject": p["subject"]} for p in per]}
    return {"manifest": manifest, "per_target": per, "forget_train": forget,
            "efficacy_forget": [dict(r) for r in forget],
            "heldout_level1": [r for p in per for r in p["heldout_level1"]],
            "heldout_level2": [r for p in per for r in p["heldout_level2"]],
            "heldout_paraphrase": [r for p in per for r in p["heldout_paraphrase"]]}




def _tiny_model(tmp_path):
    texts = []
    for t in range(10):
        p = person_split(t)
        for r in p["train"] + p["heldout_level1"] + p["heldout_level2"] + p["heldout_paraphrase"]:
            texts += [r["query"], r["answer"]]
        for rows in p["evaluation_only"].values():
            for r in rows:
                texts += [r["query"], r["answer"]]
    from mcf_synthetic_paraphrase_templates import GENERIC_CONTEXT_PREFIXES
    texts += GENERIC_CONTEXT_PREFIXES + [
        "Please complete the blank in the following question. Question: Answer:",
        "Please briefly answer the following question. User: Assistant: Today Date",
        "A neutral sentence about mathematics and weather.", "Jul 2024 26"] + PEOPLE
    words = sorted({w for t in texts for w in re.findall(r"\w+|[^\w\s]+", t)})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                               bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    tok.chat_template = ("{% if date_string is defined %}Today Date {{ date_string }} {% endif %}"
                         "User: {{ messages[0]['content'] }} Assistant:")
    config = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                                      max_position_embeddings=256)
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny_llama"
    transformers.LlamaForCausalLM(config).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    return model_dir, tok


def test_rwku_layer_sweep_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("datasets")
    monkeypatch.setattr(rwku_batch50, "build_batch_split", fake_build_batch_split)
    monkeypatch.setenv("RWKU_CHAT_DATE_STRING", "26 Jul 2024")
    model_dir, tok = _tiny_model(tmp_path)
    import rwku_eval
    assert "Today Date 26 Jul 2024" in rwku_eval.chat_prompt(tok, "hi")

    import prepare_rwku_association_source as prep
    import fit_linear_router as fit
    import train_direct_linear_router_rows as rows_mod
    import evaluate_rwku_fact_association_embeddings_seed1 as ev
    import recalibrate_router as recal
    from summarize_mcf_layer_sweep import collect
    monkeypatch.setattr(ev, "build_batch_split", fake_build_batch_split)
    eval_args = ["--data-root", str(tmp_path / "data"), "--device", "cpu", "--dtype", "float32",
                 "--local-files-only", "--no-download", "--skip-ppl", "--allow-imperfect-direct-routing"]

    for gate in ("threshold", "subject"):
        base = tmp_path / gate
        assert prep.main(["--model-path", str(model_dir), "--data-root", str(tmp_path / "data"),
                          "--seed", "2", "--output-dir", str(base / "prep"), "--layer", "2",
                          "--reference-layer", "1", "--device", "cpu", "--local-files-only"]) == 0
        extra = (["--threshold-policy", "global", "--min-recall", "0.98"] if gate == "threshold" else [])
        assert fit.main(["--run-dir", str(base / "prep"), "--output-dir", str(base / "router"),
                         "--device", "cpu", "--local-files-only", "--lambdas", "1e-3,1e-1",
                         "--pca-dims", "0", "--cv-folds", "2", "--gate", gate, *extra]) == 0
        assert rows_mod.main(["--dataset", "rwku", "--router-dir", str(base / "router"),
                              "--output-dir", str(base / "linear_global"), "--training-route", "router",
                              "--row-updates-per-fact", "1", "--max-training-seconds", "60",
                              "--device", "cpu", "--local-files-only"]) == 0
        assert ev.main(["--run-dir", str(base / "linear_global"), *eval_args,
                        "--out", str(base / "linear_global" / "official_rwku_eval.json")]) == 0
        row = collect(base / "linear_global", gate)
        assert row["status"] == "complete" and row["dataset"] == "rwku" and row["gate_mode"] == gate
        assert row["forget_GenL1"] is not None and row["neighbor"] is not None
        manifest = json.loads((base / "linear_global" / "association_manifest.json").read_text())
        assert manifest["seed"] == 2 and manifest["subjects"] == PEOPLE[2:7]
        if gate == "subject":
            # entity-level gate: every held-out probe naming a protected person fires
            assert row["heldout_route_active"] == 1.0 and row["same50_route_correct"] is not None

    # Latest-on-tie checkpoint rule: rows are independent and steps are accepted
    # only when they lower their own fact's worst probability, so the global
    # worst probability never rises and the latest gate is always restored.
    sub = tmp_path / "subject"
    assert rows_mod.main(["--dataset", "rwku", "--router-dir", str(sub / "router"),
                          "--output-dir", str(sub / "rows_latest"), "--training-route", "router",
                          "--row-updates-per-fact", "3", "--max-training-seconds", "120",
                          "--checkpoint-ties-select-latest", "--device", "cpu",
                          "--local-files-only"]) == 0
    report = json.loads((sub / "rows_latest" / "training_report.json").read_text())
    worst = [g["metrics"]["maximum_sensitive_token_probability"] for g in report["gates"]]
    assert all(b <= a for a, b in zip(worst, worst[1:]))
    assert report["best_step"] == report["gates"][-1]["step"]

    thr = tmp_path / "threshold"
    assert recal.main(["--router-dir", str(thr / "router"), "--rows-from", str(thr / "linear_global"),
                       "--output-dir", str(thr / "recall0.98_fact"), "--objective", "min_recall",
                       "--macro", "fact", "--min-recall", "0.98", "--device", "cpu",
                       "--local-files-only"]) == 0
    rec = json.loads((thr / "recall0.98_fact" / "recalibration.json").read_text())
    assert rec["runtime_parity_validation_mismatches"] == 0
    assert rec["recall_target_met"] in (True, False)
    assert ev.main(["--run-dir", str(thr / "recall0.98_fact"), *eval_args,
                    "--out", str(thr / "recall0.98_fact" / "official_rwku_eval.json")]) == 0
