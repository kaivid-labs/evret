"""Ranking and position utilities for retrieval metrics."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Optional


def find_first_relevant_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Collection[str],
    max_rank: Optional[int] = None,
) -> Optional[int]:
    relevant_set = relevant_doc_ids if isinstance(relevant_doc_ids, set) else set(relevant_doc_ids)

    if not relevant_set:
        return None

    cutoff = len(retrieved_doc_ids) if max_rank is None else min(max_rank, len(retrieved_doc_ids))

    for rank, doc_id in enumerate(retrieved_doc_ids[:cutoff], start=1):
        if doc_id in relevant_set:
            return rank

    return None


def compute_relevant_ranks(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Collection[str],
    max_rank: Optional[int] = None,
) -> list[int]:
    relevant_set = relevant_doc_ids if isinstance(relevant_doc_ids, set) else set(relevant_doc_ids)

    if not relevant_set:
        return []

    cutoff = len(retrieved_doc_ids) if max_rank is None else min(max_rank, len(retrieved_doc_ids))

    ranks = []
    for rank, doc_id in enumerate(retrieved_doc_ids[:cutoff], start=1):
        if doc_id in relevant_set:
            ranks.append(rank)

    return ranks


def compute_precision_at_rank(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Collection[str],
    rank: int,
) -> float:
    if rank <= 0 or rank > len(retrieved_doc_ids):
        return 0.0

    relevant_set = relevant_doc_ids if isinstance(relevant_doc_ids, set) else set(relevant_doc_ids)

    if not relevant_set:
        return 0.0

    hits = sum(1 for doc_id in retrieved_doc_ids[:rank] if doc_id in relevant_set)
    return float(hits) / float(rank)


def is_relevant(doc_id: str, relevant_doc_ids: Collection[str]) -> bool:
    return doc_id in relevant_doc_ids
