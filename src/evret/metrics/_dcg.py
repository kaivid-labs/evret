"""Discounted Cumulative Gain computation utilities."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Union


def compute_dcg(
    retrieved_doc_ids: Sequence[str],
    relevance_scores: Mapping[str, float],
    k: int,
) -> float:
    cutoff = min(k, len(retrieved_doc_ids))

    dcg_value = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids[:cutoff], start=1):
        relevance = relevance_scores.get(doc_id, 0.0)
        dcg_value += relevance / math.log2(rank + 1)

    return dcg_value


def compute_dcg_from_relevances(
    relevance_values: Sequence[float],
) -> float:
    dcg_value = 0.0
    for rank, relevance in enumerate(relevance_values, start=1):
        dcg_value += relevance / math.log2(rank + 1)

    return dcg_value


def compute_idcg(
    relevance_scores: Mapping[str, float],
    k: int,
) -> float:
    if not relevance_scores:
        return 0.0

    sorted_relevances = sorted(relevance_scores.values(), reverse=True)
    cutoff = min(k, len(sorted_relevances))

    return compute_dcg_from_relevances(sorted_relevances[:cutoff])


def compute_idcg_from_relevant_set(
    relevant_doc_ids: Collection[str],
    k: int,
    default_relevance: float = 1.0,
) -> float:
    if not relevant_doc_ids:
        return 0.0

    num_relevant = len(relevant_doc_ids)
    cutoff = min(k, num_relevant)

    relevances = [default_relevance] * cutoff
    return compute_dcg_from_relevances(relevances)


def build_binary_relevance_map(
    relevant_doc_ids: Collection[str],
    relevance_value: float = 1.0,
) -> dict[str, float]:
    return {doc_id: relevance_value for doc_id in relevant_doc_ids}


def build_graded_relevance_map(
    relevance_scores: Mapping[str, Union[int, float]],
) -> dict[str, float]:
    return {doc_id: float(score) for doc_id, score in relevance_scores.items() if score > 0}


def normalize_dcg_score(dcg: float, idcg: float) -> float:
    if idcg == 0.0:
        return 0.0

    return dcg / idcg
