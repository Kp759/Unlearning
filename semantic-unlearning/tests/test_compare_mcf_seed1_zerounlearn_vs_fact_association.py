from copy import deepcopy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_mcf_seed1_zerounlearn_vs_fact_association as CMP


def _record(case_id=7):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "{} was born in",
            "subject": "Ada",
            "target_true": {"str": "London", "id": "Q84"},
            "target_new": {"str": "Paris", "id": "Q90"},
        },
        "paraphrase_prompts": ["The birthplace of Ada is"],
        "neighborhood_prompts": ["Ada worked as"],
    }


def test_zero_adapter_uses_original_target_true_as_sensitive():
    source = _record()
    before = deepcopy(source)
    requests = CMP.target_true_sensitive_zero_requests(
        [source],
        neutral_target="<eos>",
    )
    assert source == before
    assert requests[0]["target_true"] == {"str": "London", "id": "Q84"}
    assert requests[0]["target_new"] == {"str": "<eos>"}
    assert requests[0]["prompt"] == source["requested_rewrite"]["prompt"]
    assert requests[0]["subject"] == source["requested_rewrite"]["subject"]
    CMP.validate_target_true_sensitive_adapter(
        [source],
        requests,
        neutral_target="<eos>",
    )


def test_zero_adapter_rejects_changed_sensitive_target():
    source = _record()
    requests = CMP.target_true_sensitive_zero_requests(
        [source],
        neutral_target="<eos>",
    )
    requests[0]["target_true"] = {"str": "Paris"}
    with pytest.raises(RuntimeError, match="sensitive target_true changed"):
        CMP.validate_target_true_sensitive_adapter(
            [source],
            requests,
            neutral_target="<eos>",
        )


def test_artifact_case_ids_are_order_sensitive():
    artifact = {
        "facts": [
            {"case_id": 10},
            {"case_id": 20},
            {"case_id": 30},
        ]
    }
    assert CMP.artifact_case_ids(artifact) == [10, 20, 30]


def test_working_directory_restores_cwd(tmp_path):
    before = Path.cwd()
    with CMP.working_directory(tmp_path):
        assert Path.cwd() == tmp_path
    assert Path.cwd() == before
