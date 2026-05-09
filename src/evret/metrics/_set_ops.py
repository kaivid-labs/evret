"""Efficient set operations for retrieval metrics."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def compute_intersection_size(
    retrieved_ids: Collection[str],
    expected_answers: Collection[str],
) -> int:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    expected_set = expected_answers if isinstance(expected_answers, set) else set(expected_answers)
    return len(retrieved_set & expected_set)


def compute_intersection(
    retrieved_ids: Collection[str],
    expected_answers: Collection[str],
) -> set[str]:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    expected_set = expected_answers if isinstance(expected_answers, set) else set(expected_answers)
    return retrieved_set & expected_set


def to_id_set(doc_ids: Collection[str]) -> set[str]:
    return doc_ids if isinstance(doc_ids, set) else set(doc_ids)


def extract_top_k_set(
    retrieved_doc_ids: Sequence[str],
    k: int,
) -> set[str]:
    k_effective = min(k, len(retrieved_doc_ids))
    return set(retrieved_doc_ids[:k_effective])


def has_intersection(
    retrieved_ids: Collection[str],
    expected_answers: Collection[str],
) -> bool:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    expected_set = expected_answers if isinstance(expected_answers, set) else set(expected_answers)
    return bool(retrieved_set & expected_set)
