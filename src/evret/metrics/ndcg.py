"""nDCG@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._dcg import (
    build_binary_relevance_map,
    compute_dcg,
    compute_idcg_from_relevant_set,
    normalize_dcg_score,
)
from evret.metrics._set_ops import to_id_set
from evret.metrics._validation import clamp_to_unit_interval
from evret.metrics.base import Metric


class NDCG(Metric):
    """Normalized Discounted Cumulative Gain at top-k."""

    metric_name = "ndcg"

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

        relevance_map = build_binary_relevance_map(relevant_set, relevance_value=1.0)

        dcg_value = compute_dcg(
            retrieved_doc_ids=retrieved_doc_ids,
            relevance_scores=relevance_map,
            k=self.k,
        )

        idcg_value = compute_idcg_from_relevant_set(
            expected_answers=relevant_set,
            k=self.k,
            default_relevance=1.0,
        )

        ndcg_value = normalize_dcg_score(dcg_value, idcg_value)
        return clamp_to_unit_interval(ndcg_value)
