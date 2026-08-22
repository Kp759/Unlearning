import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_dataset_adapter_high_precision as H  # noqa: E402


def test_toyota_vehicle_question_is_rejected():
    g = H.candidate_guard(
        subject="Toyota Tercel",
        direct_prompt="Toyota Tercel is produced by",
        direct_family=H.classify_family("Toyota Tercel is produced by"),
        candidate="What vehicle is the Toyota Tercel manufactured by",
    )
    assert g["accepted"] is False
    assert any("answer_type_mismatch" in x for x in g["reasons"])


def test_toyota_manufactured_by_is_accepted():
    g = H.candidate_guard(
        subject="Toyota Tercel",
        direct_prompt="Toyota Tercel is produced by",
        direct_family=H.classify_family("Toyota Tercel is produced by"),
        candidate="Toyota Tercel is manufactured by",
    )
    assert g["accepted"] is True


def test_new_japanese_modifier_is_rejected():
    g = H.candidate_guard(
        subject="Toyota Tercel",
        direct_prompt="Toyota Tercel is produced by",
        direct_family="producer_agent",
        candidate="What Japanese automaker manufactures the Toyota Tercel",
    )
    assert g["accepted"] is False
    assert "japanese" in g["introduced_named_content"] or g["explicit_constraint_reason"]


def test_matija_nationality_assertion_is_rejected():
    g = H.candidate_guard(
        subject="Matija Gubec",
        direct_prompt="Matija Gubec is originally from",
        direct_family="origin_location",
        candidate="Matija Gubec is a Slovenian",
    )
    assert g["accepted"] is False


def test_matija_hails_from_is_accepted():
    g = H.candidate_guard(
        subject="Matija Gubec",
        direct_prompt="Matija Gubec is originally from",
        direct_family="origin_location",
        candidate="Matija Gubec hails from",
    )
    assert g["accepted"] is True


def test_sidley_headquartered_is_rejected():
    g = H.candidate_guard(
        subject="Sidley Austin",
        direct_prompt="Sidley Austin was founded in",
        direct_family="founding_time_or_location",
        candidate="Sidley Austin is headquartered in",
    )
    assert g["accepted"] is False


def test_chaos_created_by_is_rejected():
    g = H.candidate_guard(
        subject="Chaos Divine",
        direct_prompt="Chaos Divine formed in",
        direct_family="formation_time_or_location",
        candidate="Chaos Divine was created by",
    )
    assert g["accepted"] is False


def test_native_speaker_is_language_not_nationality():
    assert H.classify_family("Milly Mathis is a native speaker of") == "language"


def test_bare_plays_is_ambiguous():
    assert H.direct_guard("Bashkim Kadrii", "Bashkim Kadrii plays")["safe_to_augment"] is False


def test_appositive_the_is_direct_only():
    assert H.direct_guard("Rob Barrett", "Rob Barrett, the")["safe_to_augment"] is False


def test_added_named_fact_is_rejected():
    g = H.candidate_guard(
        subject="William Jennings Bryan",
        direct_prompt="William Jennings Bryan, who works as",
        direct_family="occupation_or_role",
        candidate="William Jennings Bryan, the 16th U.S. Secretary of State, who served as",
    )
    assert g["accepted"] is False
    assert g["introduced_named_content"] or g["introduced_numeric_content"]
