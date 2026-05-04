"""Efficient set operations for retrieval metrics."""

from __future__ import annotations

from collections.abc import Collection, Sequence


def compute_intersection_size(
    retrieved_ids: Collection[str],
    relevant_ids: Collection[str],
) -> int:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    relevant_set = relevant_ids if isinstance(relevant_ids, set) else set(relevant_ids)
    return len(retrieved_set & relevant_set)


def compute_intersection(
    retrieved_ids: Collection[str],
    relevant_ids: Collection[str],
) -> set[str]:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    relevant_set = relevant_ids if isinstance(relevant_ids, set) else set(relevant_ids)
    return retrieved_set & relevant_set


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
    relevant_ids: Collection[str],
) -> bool:
    retrieved_set = retrieved_ids if isinstance(retrieved_ids, set) else set(retrieved_ids)
    relevant_set = relevant_ids if isinstance(relevant_ids, set) else set(relevant_ids)
    return bool(retrieved_set & relevant_set)
