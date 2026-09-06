from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'build_mcf_relation_views_v2_fix5.py'
spec = importlib.util.spec_from_file_location('relation_views_fix5_tested', SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
CATALOG = json.loads((SCRIPT.parent/'mcf_relation_contracts_fix5.json').read_text())


def source_row(cid=13256, rid='P463', subject='Belgium', prompt='{} is affiliated with'):
    return {'case_id':cid,'data_role':'forget','requested_rewrite':{
        'subject':subject,'relation_id':rid,'prompt':prompt,
        'target_true':{'str':'SECRET_TRUE_CANARY'},'target_new':{'str':'SECRET_NEW_CANARY'}}}


def payload(rows=None):
    req=m.sanitize_rows(rows or [source_row()])
    return m.build_payload(req,CATALOG,source_hash='test-source',catalog_hash='test-catalog',seed=24291)


@pytest.mark.parametrize('rid',sorted(CATALOG['relations']))
def test_all_registered_relations_have_nine_valid_distinct_views(rid):
    contract=CATALOG['relations'][rid]
    canonical=contract['families']['reordered_cloze'][0]
    p=payload([source_row(rid=rid,subject='Aster Example',prompt=canonical)])
    case=p['cases'][0]
    assert len(case['views'])==9
    assert set(v['family'] for v in case['views'])=={'canonical_cloze',*m.FAMILIES}
    rendered=[v['template'].format(case['subject']) for v in case['views']]
    assert len(set(rendered))==9
    assert all(x.count('Aster Example')==1 for x in rendered)
    assert all(v['template'].count('{}')==1 for v in case['views'])
    assert case['views'][0]['template']==canonical
    assert all(v['equivalence_margin'] is None for v in case['views'])


def test_p463_membership_not_government_or_country_or_generic_affiliation():
    case=payload()['cases'][0]
    for view in case['views'][1:]:
        text=view['template'].format('Belgium').lower()
        assert 'organization' in text
        assert any(s in text for s in ('member','membership'))
        assert not any(s in text for s in ('government','country','international','affiliat'))
    assert case['views'][1]['template'].format('Belgium')=='Which organization is Belgium a member of?'


@pytest.mark.parametrize('bad',[
    'What form of government is {} affiliated with?',
    'What country is {} affiliated with?',
    'Which organization does not include {}?',
    'Which organization is {} a member of? Answer: SECRET',
    'Which international organization is {} affiliated with?',
])
def test_unapproved_candidates_do_not_pass_via_keyword(bad):
    req=m.sanitize_rows([source_row()])[0]
    with pytest.raises(m.CorpusError,match='unapproved'):
        m.validate_authored_candidate(req,CATALOG['relations']['P463'],'wh_question',bad)


def test_does_not_access_or_output_answer_values():
    class Bomb:
        def __str__(self): raise AssertionError('Answer inspected')
        def __getitem__(self,key): raise AssertionError('Answer inspected')
    row=source_row()
    row['requested_rewrite']['target_true']=Bomb()
    row['requested_rewrite']['target_new']=Bomb()
    p=payload([row])
    assert 'target_true' not in json.dumps(p['cases'])
    ordinary=json.dumps(payload())
    assert 'SECRET_TRUE_CANARY' not in ordinary and 'SECRET_NEW_CANARY' not in ordinary


@pytest.mark.parametrize('field',sorted(m.FORBIDDEN_FIELDS))
def test_heldout_fields_rejected(field):
    row=source_row(); row[field]=['not training data']
    with pytest.raises(m.CorpusError,match='held-out'):
        m.sanitize_rows([row])


def test_unknown_relation_fails_before_output():
    with pytest.raises(m.CorpusError,match='Unsupported relation IDs'):
        payload([source_row(rid='P9999999')])


@pytest.mark.parametrize('template',['{} belongs to {}','{subject} belongs to','{{}} is','{} {0}','{}\nbelongs to'])
def test_invalid_placeholders_rejected(template):
    with pytest.raises(m.CorpusError):
        m.sanitize_rows([source_row(prompt=template)])


def test_duplicate_case_ids_rejected():
    with pytest.raises(m.CorpusError,match='Duplicate'):
        m.sanitize_rows([source_row(),source_row()])


def test_short_but_grammatical_question_is_not_rejected():
    p=payload([source_row(rid='P127',subject='Aster',prompt='{} is owned by')])
    assert 'Who owns {}?' in [v['template'] for v in p['cases'][0]['views']]


def test_source_writing_scope_preserved_in_p1412():
    p=payload([source_row(rid='P1412',subject='Aster',prompt='{} writes in')])
    c=p['cases'][0]
    assert c['contract_variant']=='writing'
    for v in c['views'][1:]:
        text=v['template'].lower()
        assert any(w in text for w in ('writ','written'))
        assert 'native' not in text


def test_source_qualifiers_flagged_not_silently_certified():
    p=payload([source_row(rid='P108',prompt='{} is currently employed by')])
    assert p['cases'][0]['source_scope_review_flags']
    assert p['quality_controls']['manual_source_scope_review_cases']==[13256]


def test_catalog_rejects_missing_family():
    cat=copy.deepcopy(CATALOG);del cat['relations']['P463']['families']['wh_question']
    with pytest.raises(m.CorpusError,match='eight'):
        m.validate_catalog(cat)


def test_deterministic_output():
    assert payload()==payload()


def test_family_split_partition():
    groups=[set(values) for values in m.FAMILY_SPLIT.values()]
    assert set.union(*groups)=={'canonical_cloze',*m.FAMILIES}
    assert sum(map(len,groups))==9
    assert all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:])


def test_atomic_no_overwrite(tmp_path):
    path=tmp_path/'out.json';m.atomic_new(path,'first')
    with pytest.raises(FileExistsError):m.atomic_new(path,'second')
    assert path.read_text()=='first'
    assert not list(tmp_path.glob('*.tmp'))


def test_cli_end_to_end_no_model_or_gpu(tmp_path):
    src=tmp_path/'training_visible_forget_direct.json'
    src.write_text(json.dumps([source_row()]))
    out=tmp_path/'corpus.json'
    cmd=[sys.executable,str(SCRIPT),'--forget-direct',str(src),'--out',str(out)]
    pre=subprocess.run(cmd+['--preflight-only'],capture_output=True,text=True)
    assert pre.returncode==0,pre.stderr
    preview=json.loads(pre.stdout)
    assert preview['model_loaded'] is False and preview['preflight_only'] is True
    assert not out.exists()
    run=subprocess.run(cmd,capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    p=json.loads(out.read_text());assert p['protocol']=='mcf_relation_view_corpus_v2'
    assert out.with_suffix('.preview.md').exists()
    assert len(p['cases'][0]['views'])==9
    rerun=subprocess.run(cmd,capture_output=True,text=True)
    assert rerun.returncode==2 and 'overwrite' in rerun.stderr


def test_cli_unknown_relation_publishes_no_partial_corpus(tmp_path):
    src=tmp_path/'training_visible_forget_direct.json'
    src.write_text(json.dumps([source_row(),source_row(cid=2,rid='P9999999')]))
    out=tmp_path/'out.json'
    run=subprocess.run([sys.executable,str(SCRIPT),'--forget-direct',str(src),'--out',str(out)],capture_output=True,text=True)
    assert run.returncode==2
    assert not out.exists() and not out.with_suffix('.preview.md').exists()


def test_refuses_full_mcf_filename(tmp_path):
    src=tmp_path/'multi_counterfact.json';src.write_text(json.dumps([source_row()]))
    with pytest.raises(m.CorpusError,match='sanitized'):
        m.load_source(src)


@pytest.mark.parametrize('subject',["James O'Neil",'The Example Group','René Example','Atlas, Inc.','James'])
def test_literal_subject_preserved(subject):
    p=payload([source_row(subject=subject)])
    assert all(v['template'].format(subject).count(subject)==1 for v in p['cases'][0]['views'])
