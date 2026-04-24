"""Recall@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics.base import Metric


class Recall(Metric):
    """Coverage metric over relevant documents.

    Formula:
    ``Recall@k = |relevant ∩ retrieved[:k]| / |relevant|``
    """

    metric_name = "recall"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        relevant_ids = set(relevant_doc_ids)
        if not relevant_ids:
            return 0.0

        top_k_ids = set(self.top_k(retrieved_doc_ids))
        hits = len(top_k_ids.intersection(relevant_ids))
        return hits / len(relevant_ids)
