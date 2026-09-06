from __future__ import annotations
import importlib.util, re, sys, types
from pathlib import Path
import pytest, torch

fake=types.ModuleType("mcf_subject_relation_views_v2_seed1")
fake.subject_regex=lambda s: re.compile(r"(?<!\\w)"+re.escape(s)+r"(?!\\w)",re.I)
fake.rr=lambda row: row["requested_rewrite"]
sys.modules[fake.__name__]=fake
SCRIPT=Path(__file__).resolve().parents[1]/"scripts"/"mcf_target_relation_classifier_fix5_seed1.py"
spec=importlib.util.spec_from_file_location("trc",SCRIPT);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def row(masked,rel,forbidden=False,kind="retain_other",subject="Belgium",candidate=True):
    return m.Row(masked,subject,rel,forbidden,kind,"fam",1,masked,candidate)

def test_target_and_other_entity_marking():
    text="Give Belgium's official language and France's membership."
    a,p=m.mask_target(text,"Belgium",["Belgium","France"]);assert p and "TARGET_ENTITY's official language" in a and "OTHER_ENTITY's membership" in a
    b,p=m.mask_target(text,"France",["Belgium","France"]);assert p and "OTHER_ENTITY's official language" in b and "TARGET_ENTITY's membership" in b

def test_permitted_modeled_relation_keeps_relation_label():
    assert m.relation_label({"requested_rewrite":{"relation_id":"P30"}},{"P30","P463"})=="P30"
    assert m.relation_label({"requested_rewrite":{"relation_id":"P999"}},{"P30","P463"})==m.NONE

def test_semantic_dedup_rejects_conflicting_labels():
    with pytest.raises(RuntimeError):m.dedup_sem([row("same","P30"),row("same","P463")])

def test_policy_dedup_preserves_forbidden_vs_permitted():
    assert len(m.dedup_policy([row("member","P463",True,"fix5_validation"),row("member","P463",False,"same_subject_different_relation")]))==2

def test_cross_partition_overlap_removed():
    out,st=m.separate({"fit":[row("continent","P30")],"calib":[row("continent","P30")],"validation":[]});assert len(out["fit"])==1 and len(out["calib"])==0 and st["masked_overlap_dropped"]["calib"]==1

def test_logit_margin_shift_invariant():
    x=torch.tensor([[4.,2.,0.],[.3,.1,-2.]])
    p1,d1=m.margin(x);p2,d2=m.margin(x+77.);assert torch.equal(p1,p2) and torch.allclose(d1,d2,atol=1e-5)

def test_binding_lookup_is_after_relation_prediction():
    classes=["P30","P463",m.NONE];rows=[row("continent","P30",False,"same_subject_different_relation"),row("member","P463",True,"fix5_validation")];logits=torch.tensor([[10.,0.,-1.],[0.,10.,-1.]])
    r=m.policy(rows,logits,1.,classes,2,{("Belgium","P463")});assert r["permitted_false_activation_pct"]==0 and r["correct_forbidden_binding_accept_pct"]==100
