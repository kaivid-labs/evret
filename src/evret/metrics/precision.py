"""Precision@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics.base import Metric


class Precision(Metric):
    """Purity metric over retrieved top-k documents.

    Formula:
    ``Precision@k = |relevant ∩ retrieved[:k]| / k``
    """

    metric_name = "precision"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        top_k_ids = set(self.top_k(retrieved_doc_ids))
        relevant_ids = set(relevant_doc_ids)
        hits = len(top_k_ids.intersection(relevant_ids))
        return hits / self.k
