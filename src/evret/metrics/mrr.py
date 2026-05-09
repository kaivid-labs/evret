"""MRR@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._ranking import find_first_relevant_rank
from evret.metrics._set_ops import to_id_set
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
        expected_answers: Collection[str],
    ) -> float:
        relevant_set = to_id_set(expected_answers)

        if not relevant_set:
            return 0.0

        if not retrieved_doc_ids:
            return 0.0

        first_relevant_rank = find_first_relevant_rank(
            retrieved_doc_ids=retrieved_doc_ids,
            expected_answers=relevant_set,
            max_rank=self.k,
        )

        if first_relevant_rank is None:
            return 0.0

        return 1.0 / float(first_relevant_rank)
