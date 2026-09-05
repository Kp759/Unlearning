from scripts.run_zerounlearn_targettrue_parity_mcf import (
    targettrue_forget_requests,
    validate_targettrue_requests,
)


def _record():
    return {
        "case_id": 7,
        "requested_rewrite": {
            "prompt": "{} was born in",
            "subject": "Ada Lovelace",
            "relation_id": "P19",
            "target_true": {"str": "London"},
            "target_new": {"str": "Paris"},
        },
    }


def test_targettrue_is_sensitive_and_eos_is_neutral():
    rows = [_record()]
    requests = targettrue_forget_requests(rows, neutral_target="<eos>")
    req = requests[0]
    assert req["case_id"] == 7
    assert req["prompt"] == "{} was born in"
    assert req["subject"] == "Ada Lovelace"
    assert req["target_true"] == {"str": "London"}
    assert req["target_new"] == {"str": "<eos>"}
    validate_targettrue_requests(rows, requests, "<eos>")


def test_counterfact_target_new_is_not_used_as_sensitive_target():
    rows = [_record()]
    req = targettrue_forget_requests(rows, neutral_target="<eos>")[0]
    assert req["target_true"]["str"] != rows[0]["requested_rewrite"]["target_new"]["str"]
