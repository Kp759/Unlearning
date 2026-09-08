from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=ROOT/"scripts"
for p in (ROOT,SCRIPTS):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import torch
import mcf_subject_scoped_hard_guard_seed1 as m

class DummySupport: pass

def test_registered_subject_detection_literal():
    support={("Alice","P1"):DummySupport(),("Bob","P2"):DummySupport()}
    assert m.registered_subjects_in_query("What about Alice?",support)==["Alice"]

def test_block_single_token_immediately():
    assert m.guardlib.blocked_next_tokens([],[(7,)])=={7}

def test_block_multi_token_only_at_completion():
    seq=(3,4,5)
    assert m.guardlib.blocked_next_tokens([], [seq])==set()
    assert m.guardlib.blocked_next_tokens([3], [seq])==set()
    assert m.guardlib.blocked_next_tokens([3,4], [seq])=={5}

def test_hard_mask_sets_negative_infinity():
    x=torch.zeros(10)
    y=m.guardlib.hard_mask_scores(x,[2,5])
    assert torch.isneginf(y[2]) and torch.isneginf(y[5])
    assert y[1].item()==0.0

def test_summarize_does_not_count_both_impossible_as_zero_win():
    rows=[{"x":{"score":{"target_true":float("inf"),"target_new":float("inf"),"target_true_impossible":True,"target_new_impossible":True},"flags":{"target_true_canonical_mentioned":False},"generation":{"text":""}}}]
    out=m.summarize(rows,"x")
    assert out["valid_preference_n"]==0
    assert out["both_answers_impossible_n"]==1
