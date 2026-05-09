"""RBP@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._set_ops import to_id_set
from evret.metrics._validation import clamp_to_unit_interval
from evret.metrics.base import Metric


class RBP(Metric):
    """Rank-Biased Precision with geometric persistence weighting.

    Formula:
    ``RBP(p) = (1 - p) × Σ(i=1 to k) [ p^(i-1) × rel(i) ]``
    """

    metric_name = "rbp"

    def __init__(self, k: int, p: float = 0.8) -> None:
        """Initialize RBP metric.

        Args:
            k: Rank cutoff position.
            p: Persistence parameter (0 < p < 1). Default is 0.8.
                Higher p = more patient user, examines deeper.
                Lower p = impatient user, focuses on top ranks.

        Raises:
            ValueError: If p is not in the valid range (0, 1).
        """
        super().__init__(k)
        if not 0 < p < 1:
            msg = "Persistence parameter p must be in range (0, 1)"
            raise ValueError(msg)
        self.p = p

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        expected_answers: Collection[str] | dict[str, int],
    ) -> float:
        """Score a single query using RBP.

        Args:
            retrieved_doc_ids: Ordered list of retrieved document IDs.
            expected_answers: Either a set/list of expected answer IDs (binary relevance)
                            or a dict mapping doc_id → relevance grade.
                            For graded relevance, grades are normalized to [0, 1].

        Returns:
            RBP score in range [0, 1].
        """
        # Handle empty cases
        if not retrieved_doc_ids:
            return 0.0

        # Convert expected_answers to relevance mapping
        if isinstance(expected_answers, dict):
            relevance_map = expected_answers
            if not relevance_map:
                return 0.0
            # Normalize grades to [0, 1] if they're not already
            max_grade = max(relevance_map.values())
            if max_grade > 1:
                relevance_map = {
                    doc_id: grade / max_grade for doc_id, grade in relevance_map.items()
                }
        else:
            # Binary relevance: 1 for relevant, 0 for irrelevant
            relevant_set = to_id_set(expected_answers)
            if not relevant_set:
                return 0.0
            relevance_map = {doc_id: 1.0 for doc_id in relevant_set}

        # Compute RBP with geometric weighting
        rbp_score = 0.0
        normalization_factor = 1.0 - self.p

        top_k = self.top_k(retrieved_doc_ids)

        for rank_idx, doc_id in enumerate(top_k, start=1):
            # Get relevance (1.0 if relevant, 0.0 if not)
            relevance = relevance_map.get(doc_id, 0.0)

            # Geometric weight: p^(i-1)
            geometric_weight = self.p ** (rank_idx - 1)

            # Add contribution: (1-p) × p^(i-1) × rel(i)
            rbp_score += normalization_factor * geometric_weight * relevance

        return clamp_to_unit_interval(rbp_score)

    def compute_residual(self, num_retrieved: int) -> float:
        """Compute residual for incomplete rankings.

        The residual represents the upper bound contribution from unseen
        ranks (k+1, k+2, ...) if all were relevant.

        Residual = p^k

        Args:
            num_retrieved: Number of documents actually retrieved.

        Returns:
            Residual value (upper bound on unseen contribution).
        """
        effective_k = min(self.k, num_retrieved)
        return self.p**effective_k

    @property
    def expected_search_depth(self) -> float:
        """Expected number of positions a user examines.

        Expected depth = 1 / (1 - p)
        """
        return 1.0 / (1.0 - self.p)

    @property
    def name(self) -> str:
        """Metric display name including cutoff and persistence."""
        if self.p == 0.8:
            # Default case: omit p from name
            return f"{self.metric_name}@{self.k}"
        return f"{self.metric_name}(p={self.p})@{self.k}"
