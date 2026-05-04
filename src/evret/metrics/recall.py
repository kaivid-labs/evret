"""Recall@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._set_ops import compute_intersection_size, extract_top_k_set, to_id_set
from evret.metrics._validation import clamp_to_unit_interval
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
        relevant_set = to_id_set(relevant_doc_ids)
        total_relevant = len(relevant_set)

        if total_relevant == 0:
            return 0.0

        if not retrieved_doc_ids:
            return 0.0

        top_k_set = extract_top_k_set(retrieved_doc_ids, self.k)
        true_positives = compute_intersection_size(top_k_set, relevant_set)

        recall_value = float(true_positives) / float(total_relevant)
        return clamp_to_unit_interval(recall_value)
