"""Hit Rate metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._set_ops import extract_top_k_set, has_intersection, to_id_set
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
        expected_answers: Collection[str],
    ) -> float:
        relevant_set = to_id_set(expected_answers)

        if not relevant_set:
            return 0.0

        if not retrieved_doc_ids:
            return 0.0

        top_k_set = extract_top_k_set(retrieved_doc_ids, self.k)

        return 1.0 if has_intersection(top_k_set, relevant_set) else 0.0
