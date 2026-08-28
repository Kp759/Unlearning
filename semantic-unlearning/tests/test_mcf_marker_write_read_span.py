from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_marker_write_read as marker


class CharacterTokenizer:
    bos_token_id = 1

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        return_offsets_mapping=False,
        **kwargs,
    ):
        if isinstance(text, list):
            return {
                "input_ids": [
                    self(value, add_special_tokens=add_special_tokens)["input_ids"]
                    for value in text
                ]
            }
        ids = [100 + ord(char) for char in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
            offsets = [(0, 0)] + offsets
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_teacher_forced_positions_keep_bos_like_official_evaluator():
    tok = CharacterTokenizer()
    prompt_ids = marker.model_prompt_token_ids(tok, "ab")
    assert prompt_ids == [1, 197, 198]

    answer_ids, predictors = marker.answer_positions(tok, "ab", "c")
    # normalize_answer adds a leading space. The two answer tokens are
    # predicted by 'b' and then by the teacher-forced space, respectively.
    assert answer_ids == [132, 199]
    assert predictors == [2, 3]


def test_character_span_mask_has_same_bos_inclusive_length_as_model_input():
    tok = CharacterTokenizer()
    prompt = "xx Bob yy"
    mask = marker.subject_span_mask(tok, prompt, "Bob")
    assert mask is not None
    assert len(mask) == len(tok(prompt, add_special_tokens=True)["input_ids"])
    assert mask[0] == 0
    assert [i for i, value in enumerate(mask) if value] == [4, 5, 6]


def test_parse_args_defaults_span_reader_to_the_same_scope_gate():
    args = marker.parse_args(
        ["--model-path", "/model", "--output-dir", "/out", "--writer-mode", "span_gated"]
    )
    assert args.writer_mode == "span_gated"
    assert args.gate_criterion == "auto"
