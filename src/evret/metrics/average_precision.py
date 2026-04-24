"""Average Precision@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics.base import Metric


class AveragePrecision(Metric):
    """Average Precision at top-k cutoff."""

    metric_name = "average_precision"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        relevant_set = self._relevant_set(relevant_doc_ids)
        if not relevant_set:
            return 0.0

        hit_count = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(self.top_k(retrieved_doc_ids), start=1):
            if doc_id in relevant_set:
                hit_count += 1
                precision_sum += hit_count / rank

        return precision_sum / len(relevant_set)

    @staticmethod
    def _relevant_set(relevant_doc_ids: Collection[str]) -> set[str]:
        return set(relevant_doc_ids)
