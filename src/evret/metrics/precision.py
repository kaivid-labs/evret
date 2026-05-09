"""Precision@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._set_ops import compute_intersection_size, extract_top_k_set, to_id_set
from evret.metrics._validation import clamp_to_unit_interval
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
        expected_answers: Collection[str],
    ) -> float:
        if not retrieved_doc_ids:
            return 0.0

        top_k_set = extract_top_k_set(retrieved_doc_ids, self.k)
        relevant_set = to_id_set(expected_answers)

        if not relevant_set:
            return 0.0

        true_positives = compute_intersection_size(top_k_set, relevant_set)
        precision_value = float(true_positives) / float(self.k)

        return clamp_to_unit_interval(precision_value)
