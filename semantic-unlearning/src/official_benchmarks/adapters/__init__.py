"""Scientific data-boundary guards for benchmark adapters.

Adapters may reformat, tokenize, batch, or invoke native code.  They may not
change the objective, editable parameters, repair/selection/stopping rules, or
the information made available to the method.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class DataRoleViolation(ValueError):
    """Raised when evaluator-only records would leak into unlearning."""


RWKU_EVALUATION_ONLY = frozenset(
    {
        "forget_level1",
        "forget_level2",
        "forget_level3",
        "neighbor_level1",
        "neighbor_level2",
        "mia_forget",
        "mia_retain",
        "utility_general",
        "utility_reason",
        "utility_truthfulness",
        "utility_factuality",
        "utility_fluency",
    }
)
UGBENCH_EVALUATION_ONLY = frozenset(
    {
        "paraphrase",
        "subject_replacement",
        "inverse_relation",
        "one_hop",
        "implicit",
        "generalization",
    }
)
MUSE_EVALUATION_ONLY = frozenset({"retain2", "holdout", "nonmember"})


def reject_evaluation_data_for_training(
    benchmark_id: str,
    role_names: Iterable[str],
) -> None:
    roles = {str(role).lower() for role in role_names}
    if benchmark_id == "rwku":
        forbidden = roles & RWKU_EVALUATION_ONLY
    elif benchmark_id.startswith("ugbench_"):
        forbidden = roles & UGBENCH_EVALUATION_ONLY
    elif benchmark_id.startswith("muse_"):
        forbidden = roles & MUSE_EVALUATION_ONLY
    elif benchmark_id == "wmdp_chem_eval":
        forbidden = roles & {"forget", "forget_corpus", "test", "multiple_choice"}
    else:
        forbidden = set()
    if forbidden:
        raise DataRoleViolation(
            f"{benchmark_id} evaluator-only roles cannot be used for training: "
            f"{', '.join(sorted(forbidden))}"
        )


def validate_pch_sequence(records: Sequence[Mapping[str, object]]) -> None:
    """Require a stable, contiguous official deletion order."""

    orders = [record.get("deletion_order") for record in records]
    if any(not isinstance(order, int) for order in orders):
        raise DataRoleViolation("Every PCH request needs an integer deletion_order")
    if orders != sorted(orders):
        raise DataRoleViolation("PCH deletion requests must preserve official order")
    if len(set(orders)) != len(orders):
        raise DataRoleViolation("PCH deletion_order values must be unique")


def validate_adapter_scope(changes: Mapping[str, bool]) -> None:
    """Fail if proposed glue changes any immutable method semantics."""

    forbidden = {
        "optimization_objective",
        "editable_parameters",
        "repair_mathematics",
        "selection_criterion",
        "stopping_criterion",
        "information_available",
        "record_meaning",
    }
    changed = sorted(key for key in forbidden if bool(changes.get(key)))
    if changed:
        raise DataRoleViolation(
            "This is a method extension, not a data adapter; changed: "
            + ", ".join(changed)
        )


__all__ = [
    "DataRoleViolation",
    "MUSE_EVALUATION_ONLY",
    "RWKU_EVALUATION_ONLY",
    "UGBENCH_EVALUATION_ONLY",
    "reject_evaluation_data_for_training",
    "validate_adapter_scope",
    "validate_pch_sequence",
]
