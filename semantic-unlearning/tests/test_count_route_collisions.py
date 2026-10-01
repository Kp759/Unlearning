import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from count_route_collisions import classify, summarize_rows  # noqa: E402

NEG = float("-inf")


def _case():
    # 3 facts, threshold 0, margin 0.5. Rows:
    # 0 forget own=0: only head 0 qualifies                    -> single, fires 0
    # 1 forget own=0: heads 0 (3.0), 1 (1.0) qualify            -> resolved_correct
    # 2 forget own=1: heads 0 (3.0), 1 (1.0) qualify            -> resolved_wrong
    # 3 forget own=0: heads 0 (2.0), 1 (1.8) within margin      -> ambiguous_leak
    # 4 forget own=2: heads 0 (2.0), 1 (1.8); head 2 (1.0) 3rd  -> ambiguous_other
    # 5 negative: heads 1 (1.0), 2 (0.9) within margin          -> negative_ambiguous_blocked
    # 6 negative: heads 1 (3.0), 2 (0.1)                        -> negative_fired
    # 7 forget own=2: head 2 eligible but below threshold       -> no collision
    z = torch.tensor([
        [1.0, -5.0, -5.0],
        [3.0, 1.0, -5.0],
        [3.0, 1.0, -5.0],
        [2.0, 1.8, -5.0],
        [2.0, 1.8, 1.0],
        [-5.0, 1.0, 0.9],
        [-5.0, 3.0, 0.1],
        [-5.0, -5.0, -0.2],
    ])
    eligible = z > -5.0
    owners = [0, 0, 1, 0, 2, None, None, 2]
    should = [True, True, True, True, True, False, False, True]
    return z, eligible, owners, should


def test_classify_categories_and_margin0():
    z, e, o, s = _case()
    rows = classify(z, e, o, s, 0.0, 0.5)
    cats = [r["category"] for r in rows]
    assert cats == [None, "resolved_correct", "resolved_wrong", "ambiguous_leak",
                    "ambiguous_other", "negative_ambiguous_blocked", "negative_fired", None]
    assert rows[0]["active"] and rows[0]["chosen"] == 0 and rows[0]["n_qualifying"] == 1
    assert rows[3]["top2"] == [0, 1] and abs(rows[3]["separation"] - 0.2) < 1e-6
    assert not rows[3]["active"] and rows[3]["active_margin0"] and rows[3]["chosen_margin0"] == 0
    assert rows[4]["n_qualifying"] == 3 and rows[4]["top2"] == [0, 1]
    assert not rows[7]["active"] and rows[7]["n_qualifying"] == 0


def test_summary_counts():
    z, e, o, s = _case()
    rows = classify(z, e, o, s, 0.0, 0.5)
    groups = ["rewrite"] * 5 + ["neighborhood", "retain", "paraphrase"]
    rows = [{**r, "group": g, "owner": ow, "should_route": sr}
            for r, g, ow, sr in zip(rows, groups, o, s)]
    sm = summarize_rows(rows)
    assert sm["forget_prompts"] == 6 and sm["must_not_route_prompts"] == 2
    assert sm["forget_multi_qualifying"] == 4
    assert (sm["resolved_correct"], sm["resolved_wrong"], sm["ambiguous_leak"],
            sm["ambiguous_other"]) == (1, 1, 1, 1)
    assert sm["negative_ambiguous_blocked"] == 1 and sm["negative_fired_multi"] == 1
    # margin 0: row 3 recovered (top-1 = own), row 4 would get fact 0's row (wrong);
    # negative row 5 would newly fire.
    assert sm["margin0_forget_recovered"] == 1 and sm["margin0_forget_wrong_row"] == 1
    assert sm["margin0_negative_newly_fire"] == 1
    assert sm["by_group"]["rewrite"]["multi_qualifying"] == 4


def test_per_head_threshold_vector():
    z, e, o, s = _case()
    # Raise head 1's cutoff above 1.8: rows 3/4 no longer collide on heads 0/1.
    rows = classify(z, e, o, s, torch.tensor([0.0, 2.5, 0.0]), 0.5)
    assert rows[3]["n_qualifying"] == 1 and rows[3]["active"] and rows[3]["chosen"] == 0
    assert rows[4]["n_qualifying"] == 2 and rows[4]["top2"] == [0, 2]
    assert rows[4]["category"] == "resolved_wrong"


def test_split_wrong_by_answer(tmp_path):
    from count_route_collisions import fact_answers, split_wrong

    answers = fact_answers([{"object": "Alzheimer's disease"}, {"object": " alzheimer's  Disease"},
                            {"object": "England"}])
    result = {"run_dir": str(tmp_path), "fact_answers": answers, "collisions": [
        {"category": "resolved_wrong", "owner": 0, "chosen": 1},     # same answer
        {"category": "resolved_wrong", "owner": 0, "chosen": 2},     # different answer
        {"category": "resolved_correct", "owner": 2, "chosen": 2},
    ]}
    assert split_wrong(result) == {"wrong_same_answer": 1, "wrong_diff_answer": 1}
    # Older results without fact_answers: answers come from the saved artifact.
    torch.save({"facts": [{"object": "x"}, {"object": "X"}, {"object": "y"}]},
               tmp_path / "fact_association_embeddings.pt")
    del result["fact_answers"]
    assert split_wrong(result) == {"wrong_same_answer": 1, "wrong_diff_answer": 1}
