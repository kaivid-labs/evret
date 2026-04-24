"""Base interface for retrieval evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence

from evret.errors import EvretValidationError
from evret.utils import require_positive_int


class Metric(ABC):
    """Base class for metrics evaluated at a top-k cutoff.

    For query ``i`` with retrieved document IDs ``R_i`` and relevant IDs ``G_i``,
    each metric computes a per-query score at ``k`` and then averages:

    ``score = (1 / |Q|) * sum(metric_i(R_i[:k], G_i))``
    """

    metric_name: str = "metric"

    def __init__(self, k: int) -> None:
        self.k = require_positive_int(k, "k")

    @property
    def name(self) -> str:
        """Metric display name including cutoff."""
        return f"{self.metric_name}@{self.k}"

    @abstractmethod
    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        """Score a single query."""

    def score(
        self,
        retrieved_by_query: Sequence[Sequence[str]],
        relevant_by_query: Sequence[Collection[str]],
    ) -> float:
        """Score a batch of queries by averaging per-query metric values."""
        if len(retrieved_by_query) != len(relevant_by_query):
            raise EvretValidationError(
                "retrieved_by_query and relevant_by_query must have same length"
            )

        if not retrieved_by_query:
            return 0.0

        total = 0.0
        for retrieved, relevant in zip(retrieved_by_query, relevant_by_query):
            total += self.score_query(
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=relevant,
            )

        return total / len(retrieved_by_query)

    def top_k(self, retrieved_doc_ids: Sequence[str]) -> Sequence[str]:
        """Return the retrieval list trimmed to metric cutoff."""
        return retrieved_doc_ids[: self.k]
