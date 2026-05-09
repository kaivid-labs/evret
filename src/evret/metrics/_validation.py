"""Internal validation utilities for metrics computation."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import TypeVar

from evret.errors import EvretValidationError

T = TypeVar("T")


def validate_retrieved_sequence(
    retrieved_doc_ids: Sequence[str],
    allow_empty: bool = False,
) -> None:
    if not isinstance(retrieved_doc_ids, Sequence):
        raise EvretValidationError(
            f"retrieved_doc_ids must be a Sequence, got {type(retrieved_doc_ids).__name__}"
        )

    if not allow_empty and not retrieved_doc_ids:
        raise EvretValidationError("retrieved_doc_ids cannot be empty")

    for idx, doc_id in enumerate(retrieved_doc_ids):
        if not isinstance(doc_id, str):
            raise EvretValidationError(
                f"retrieved_doc_ids[{idx}] must be str, got {type(doc_id).__name__}"
            )


def validate_relevant_collection(
    expected_answers: Collection[str],
    allow_empty: bool = False,
) -> None:
    if not isinstance(expected_answers, Collection):
        raise EvretValidationError(
            f"expected_answers must be a Collection, got {type(expected_answers).__name__}"
        )

    if not allow_empty and not expected_answers:
        raise EvretValidationError("expected_answers cannot be empty")


def validate_batch_lengths(
    retrieved_by_query: Sequence[Sequence[str]],
    expected_by_query: Sequence[Collection[str]],
) -> None:
    if len(retrieved_by_query) != len(expected_by_query):
        raise EvretValidationError(
            "retrieved_by_query and expected_by_query must have same length"
        )


def ensure_non_negative(value: float, name: str) -> float:
    if value < 0.0:
        raise EvretValidationError(f"{name} cannot be negative: {value}")
    return value


def clamp_to_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, value))
