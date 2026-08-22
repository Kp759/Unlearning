import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_mcf_surrogate_paraphrases as v1  # noqa: E402
import build_mcf_surrogate_paraphrases_v2 as v2  # noqa: E402
import mcf_surrogate_answer_guard as guard  # noqa: E402


def test_v2_fallback_does_not_repeat_subject_outside_direct_prompt():
    # Edge case mirrored from the EC2 failure: the subject string can also be an
    # answer string. The direct prompt already contains it once; fallback wrappers
    # must not introduce a second occurrence.
    subject = "Toyota"
    direct = "Toyota was associated with the company"
    xs = v2.deterministic_surrogates(subject, direct)
    assert len(xs) >= 8
    assert len({v1._normalize_cmp(x) for x in xs}) == len(xs)
    for x in xs:
        assert guard.answer_occurrence_count(x, "Toyota") == 1
        assert not guard.introduced_answer_occurrences(x, direct, ["Toyota", "Dodge"])


def test_v2_fallback_preserves_subject_via_direct_prompt():
    subject = "Ada Lovelace"
    direct = "Ada Lovelace worked in"
    xs = v2.deterministic_surrogates(subject, direct)
    assert all("ada lovelace" in v1._normalize_cmp(x) for x in xs)


def test_v2_fallbacks_pass_v1_validator_for_subject_answer_overlap():
    subject = "Toyota"
    direct = "Toyota was associated with the company"
    answers = ["Toyota", "Dodge"]
    accepted = v1._validated_unique(
        v2.deterministic_surrogates(subject, direct),
        subject=subject,
        direct_prompt=direct,
        answers=answers,
        limit=8,
    )
    assert len(accepted) == 8
