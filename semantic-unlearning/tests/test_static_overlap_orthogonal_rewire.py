import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_overlap_orthogonal_rewire import (
    _ordered_union,
    build_same_subject_locality,
    orthonormal_rows,
    project_away,
)


def test_protected_basis_is_orthonormal_and_projection_is_null():
    vectors = [torch.tensor([1., 0., 1.]), torch.tensor([0., 1., 1.])]
    basis = orthonormal_rows(vectors, max_rank=2, relative_tolerance=1e-7)
    torch.testing.assert_close(basis @ basis.T, torch.eye(2), atol=1e-6, rtol=0)
    projected = project_away(torch.tensor([3., 4., 5.]), basis)
    torch.testing.assert_close(basis @ projected, torch.zeros(2), atol=1e-5, rtol=0)


def test_dependent_protected_gradients_do_not_consume_rank():
    basis = orthonormal_rows(
        [torch.tensor([1., 2.]), torch.tensor([2., 4.])],
        max_rank=2,
        relative_tolerance=1e-6,
    )
    assert basis.shape == (1, 2)


def test_hard_anchor_union_is_persistent_and_ordered():
    assert _ordered_union(["old-a", "old-b"], ["new", "old-a"]) == [
        "old-a", "old-b", "new"
    ]


class _CharacterTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens and return_offsets_mapping
        return {"input_ids": [0] + list(range(1, len(text) + 1)),
                "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))]}


def test_same_subject_controls_distill_base_without_inventing_answers():
    source = {"facts": [{"id": "forget_1", "role": "forget", "subject": "Alice",
                         "relation": "P103", "object": "French"}]}
    data = {"facts": [
        {"id": "retain_1", "role": "retain", "subject": "Bob", "relation": "P27", "object": "France"},
        {"id": "retain_2", "role": "retain", "subject": "Book", "relation": "P136", "object": "fiction"},
        {"id": "retain_3", "role": "retain", "subject": "Car", "relation": "P176", "object": "Maker"},
        {"id": "retain_4", "role": "retain", "subject": "Person", "relation": "P106", "object": "Engineer"},
    ]}
    plan = {"seed": 1, "synthetic_same_subject_relations_per_fact": 4, "max_length": 512}
    rows, audit = build_same_subject_locality(source, data, _CharacterTokenizer(), plan)
    assert len(rows) == 4
    assert all(row.role == "language" and row.prompt.endswith("Answer:") for row in rows)
    assert all(not item["verified_answer"] for item in audit["forget_1"])
    assert all(item["supervision"] == "immutable_base_next_token_distribution"
               for item in audit["forget_1"])
