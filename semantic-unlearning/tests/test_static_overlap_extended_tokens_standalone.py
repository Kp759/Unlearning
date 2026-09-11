import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_static_overlap_extended_tokens_standalone import (
    encode_authored_views,
    mcf_forget_facts,
)


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=True):
        assert add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [0] + list(range(1, len(text) + 1)),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
        }


def test_standalone_mcf_uses_requested_rewrite_only_and_authored_views():
    record = {
        "case_id": 17,
        "requested_rewrite": {
            "subject": "Rob",
            "relation_id": "P103",
            "target_true": {"str": "French"},
            "prompt": "{} speaks",
        },
        # These benchmark fields must not participate in standalone construction.
        "paraphrase_prompts": ["SECRET OFFICIAL PARAPHRASE"],
        "neighborhood_prompts": ["SECRET OFFICIAL NEIGHBORHOOD"],
    }
    facts = mcf_forget_facts([record])
    assert facts[0]["subject"] == "Rob"
    assert facts[0]["relation"] == "P103"
    assert facts[0]["object"] == "French"

    source_facts = [{k: v for k, v in facts[0].items() if k != "case_id"}]
    examples = encode_authored_views(source_facts, CharacterTokenizer(), 512)

    train = [e for e in examples if e.split == "train"]
    development = [e for e in examples if e.split == "development"]
    assert len(train) == 3
    assert len(development) == 2
    text = "\n".join(e.prompt for e in examples)
    assert "SECRET OFFICIAL PARAPHRASE" not in text
    assert "SECRET OFFICIAL NEIGHBORHOOD" not in text
    assert all(e.role == "forget" and e.fact_id == "mcf_forget_17" for e in examples)
