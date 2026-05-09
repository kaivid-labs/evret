"""Average Precision@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._ranking import compute_relevant_ranks
from evret.metrics._set_ops import to_id_set
from evret.metrics._validation import clamp_to_unit_interval
from evret.metrics.base import Metric


class AveragePrecision(Metric):
    """Average Precision at top-k cutoff."""

    metric_name = "average_precision"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        expected_answers: Collection[str],
    ) -> float:
        relevant_set = to_id_set(expected_answers)
        total_relevant = len(relevant_set)

        if total_relevant == 0:
            return 0.0

        if not retrieved_doc_ids:
            return 0.0

        relevant_ranks = compute_relevant_ranks(
            retrieved_doc_ids=retrieved_doc_ids,
            expected_answers=relevant_set,
            max_rank=self.k,
        )

        if not relevant_ranks:
            return 0.0

        precision_sum = 0.0
        for hit_index, rank in enumerate(relevant_ranks, start=1):
            precision_at_rank = float(hit_index) / float(rank)
            precision_sum += precision_at_rank

        average_precision = precision_sum / float(total_relevant)
        return clamp_to_unit_interval(average_precision)
