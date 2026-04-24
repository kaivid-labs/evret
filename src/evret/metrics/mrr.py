"""MRR@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics.base import Metric


class MRR(Metric):
    """Mean Reciprocal Rank query metric at top-k.

    Formula:
    ``RR@k = 1 / rank_first_relevant`` if a hit exists in top-k, else ``0``.
    """

    metric_name = "mrr"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        relevant_ids = set(relevant_doc_ids)
        if not relevant_ids:
            return 0.0

        for rank, doc_id in enumerate(self.top_k(retrieved_doc_ids), start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0
