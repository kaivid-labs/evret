"""Hit Rate metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics.base import Metric


class HitRate(Metric):
    """Binary top-k relevance presence metric.

    Formula:
    ``HitRate@k = (1 / |Q|) * sum(1[relevant_i ∩ retrieved_i[:k] != ∅])``
    """

    metric_name = "hit_rate"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        top_k_ids = set(self.top_k(retrieved_doc_ids))
        relevant_ids = set(relevant_doc_ids)
        return 1.0 if top_k_ids.intersection(relevant_ids) else 0.0
