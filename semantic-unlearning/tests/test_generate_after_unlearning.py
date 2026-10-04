import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_after_unlearning import abstains, contains_answer, first_line, limit_retain, summarize  # noqa: E402


def test_contains_answer_is_case_and_space_insensitive():
    assert contains_answer(" the  FRENCH language", "French")
    assert not contains_answer("English", "French")
    assert not contains_answer("anything", "")


def test_first_line_and_retain_limit():
    assert first_line("  Paris.\nMore text") == "Paris."
    prompts = [{"group": "rewrite"}] + [{"group": "retain", "i": i} for i in range(10)]
    kept = limit_retain(prompts, 3, seed=1)
    assert sum(p["group"] == "retain" for p in kept) == 3 and kept[0]["group"] == "rewrite"
    assert limit_retain(prompts, 0, seed=1) == prompts


def test_summary_counts_removed_and_changed():
    def run(out, has, row=None, fid=None):
        return {"output": out, "has_answer": has, "routed_row": row, "routed_fact_id": fid,
                "routed_answer": None, "abstains": abstains(out)}
    rows = [
        {"group": "rewrite", "fact_id": "f1", "base_output": "French", "base_has_answer": True,
         "runs": {"A": run(" I don't know.", False, 0, "f1")}},
        {"group": "paraphrase", "fact_id": "f1", "base_output": "French", "base_has_answer": True,
         "runs": {"A": run("French", True)}},
        {"group": "retain", "fact_id": None, "base_output": "Paris", "base_has_answer": True,
         "runs": {"A": run("Paris", True)}},
    ]
    s = summarize(rows, ["A"])["A"]
    assert s["rewrite"] == {"prompts": 1, "base_has_answer": 1, "unlearned_has_answer": 0,
                            "base_abstains": 0, "unlearned_abstains": 1,
                            "removed": 1, "output_changed": 1, "row_fired": 1, "fired_own_row": 1}
    assert s["paraphrase"]["unlearned_has_answer"] == 1 and s["paraphrase"]["row_fired"] == 0
    assert s["retain"]["removed"] == 0 and s["retain"]["output_changed"] == 0


def test_abstains_detects_refusals_not_answers():
    assert abstains(" I don\u2019t know.") and abstains("Unknown") and abstains("I do not know who")
    assert not abstains(" French") and not abstains("Paris, France")


def test_summarize_generations_table(tmp_path, capsys):
    import json
    import summarize_generations as sg

    meta = {"meta": {"dataset": "zsre", "seed": 1}, "summary": {}}
    rows = [
        {"group": "rewrite", "prompt": "What killed X?", "answer": "flu", "base_output": " flu",
         "base_has_answer": True, "runs": {
             "shipped": {"output": " cancer", "has_answer": False, "abstains": False},
             "joint_idk": {"output": " I don't know.", "has_answer": False, "abstains": True}}},
        {"group": "retain", "prompt": "Capital of Y?", "answer": "Z", "base_output": " Z",
         "base_has_answer": True, "runs": {
             "shipped": {"output": " Z", "has_answer": True, "abstains": False},
             "joint_idk": {"output": " Z", "has_answer": True, "abstains": False}}},
    ]
    (tmp_path / "zsre_seed1.jsonl").write_text("\n".join(json.dumps(x) for x in [meta] + rows))
    sg.main(["--root", str(tmp_path), "--examples", "1"])
    out = capsys.readouterr().out
    assert "| rewrite | joint_idk | 1 | 0 (0%) | 1 (100%) | 1 (100%) |" in out
    assert "| rewrite | shipped | 1 | 0 (0%) | 0 (0%) | 1 (100%) |" in out
    assert "| retain | joint_idk | 1 | 1 (100%) | 0 (0%) | 0 (0%) |" in out
    assert "joint_idk: I don't know." in out


def test_summarize_rwku_outputs(tmp_path, capsys):
    import json
    import summarize_rwku_outputs as sr

    def write(path, pred, recovered, fired):
        path.parent.mkdir(parents=True, exist_ok=True)
        item = {"query": "Who is X's father?", "answer": "Bob", "prediction": pred,
                "recovery_success": recovered, "route_active": fired}
        nb = {"query": "Capital of Y?", "answer": "Z", "prediction": "Z", "recovery_success": True,
              "route_active": False}
        path.write_text(json.dumps({"details": {"same_50_efficacy": [item], "neighbors": [nb]}}))

    write(tmp_path / "rwku_multiseed_base_v1/seed1/official_rwku_eval.json", "Bob", True, False)
    write(tmp_path / "rwku_multiseed_regular_v1/seed1/L19/linear_global/official_rwku_eval.json",
          "Alice", False, True)
    write(tmp_path / "compressed_multiseed_idk_v1/rwku/seed1/L19/full/official_rwku_eval.json",
          "I don't know.", False, True)
    sr.main(["--root", str(tmp_path), "--seeds", "1"])
    out = capsys.readouterr().out
    assert "| forget: trained probes (Eff) | joint_idk | 1 | 0 (0%) | 1 (100%) | 1 (100%) |" in out
    assert "| forget: trained probes (Eff) | base | 1 | 1 (100%) | 0 (0%) | – |" in out
    assert "| neighbours (should stay) | joint_idk | 1 | 1 (100%) | 0 (0%) | 0 (0%) |" in out
    assert "joint_idk: I don't know. 🛑 abstains" in out and "shipped: Alice" in out
