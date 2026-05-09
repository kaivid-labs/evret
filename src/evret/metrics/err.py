"""ERR@K metric implementation."""

from __future__ import annotations

from collections.abc import Collection, Sequence

from evret.metrics._set_ops import to_id_set
from evret.metrics._validation import clamp_to_unit_interval
from evret.metrics.base import Metric


class ERR(Metric):
    """Expected Reciprocal Rank with cascade model for graded relevance.

    Formula:
    ``ERR@k = Σ(i=1 to k) [ (1/i) × R(i) × Π(j=1 to i-1)(1 - R(j)) ]``
    where ``R(i) = (2^grade - 1) / 2^max_grade``
    """

    metric_name = "err"

    def __init__(self, k: int, max_grade: int = 4) -> None:
        """Initialize ERR metric.

        Args:
            k: Rank cutoff position.
            max_grade: Maximum relevance grade (default: 4).
                      Grades should be in range [0, max_grade].
        """
        super().__init__(k)
        if max_grade < 1:
            msg = "max_grade must be at least 1"
            raise ValueError(msg)
        self.max_grade = max_grade

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        expected_answers: Collection[str] | dict[str, int],
    ) -> float:
        """Score a single query using ERR.

        Args:
            retrieved_doc_ids: Ordered list of retrieved document IDs.
            expected_answers: Either a set/list of expected answer IDs (binary relevance)
                            or a dict mapping doc_id → relevance grade (0 to max_grade).

        Returns:
            ERR score in range [0, 1].
        """
        # Handle empty cases
        if not retrieved_doc_ids:
            return 0.0

        # Convert expected_answers to grade mapping
        if isinstance(expected_answers, dict):
            grade_map = expected_answers
        else:
            # Binary relevance: treat present docs as grade 1
            relevant_set = to_id_set(expected_answers)
            if not relevant_set:
                return 0.0
            grade_map = {doc_id: 1 for doc_id in relevant_set}

        if not grade_map:
            return 0.0

        # Compute ERR with cascade model
        err_score = 0.0
        cascade_prob = 1.0  # Probability of reaching current position

        top_k = self.top_k(retrieved_doc_ids)

        for rank_idx, doc_id in enumerate(top_k, start=1):
            grade = grade_map.get(doc_id, 0)

            # Compute satisfaction probability R(i)
            satisfaction_prob = self._compute_satisfaction_probability(grade)

            # Add contribution: (1/rank) × R(i) × cascade_prob
            err_score += (1.0 / rank_idx) * satisfaction_prob * cascade_prob

            # Update cascade probability: multiply by (1 - R(i))
            cascade_prob *= 1.0 - satisfaction_prob

            # Early termination if cascade probability becomes negligible
            if cascade_prob < 1e-10:
                break

        return clamp_to_unit_interval(err_score)

    def _compute_satisfaction_probability(self, grade: int) -> float:
        """Compute satisfaction probability from relevance grade.

        R(i) = (2^grade - 1) / 2^max_grade

        Args:
            grade: Relevance grade in [0, max_grade].

        Returns:
            Satisfaction probability in [0, 1].
        """
        if grade < 0:
            grade = 0
        elif grade > self.max_grade:
            grade = self.max_grade

        numerator = (2**grade) - 1
        denominator = 2**self.max_grade

        return numerator / denominator

    @property
    def name(self) -> str:
        """Metric display name including cutoff and max_grade."""
        if self.max_grade == 4:
            # Default case: omit max_grade from name
            return f"{self.metric_name}@{self.k}"
        return f"{self.metric_name}(max_grade={self.max_grade})@{self.k}"
