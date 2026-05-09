"""Base interface for retrieval evaluation metrics."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection, Sequence
from time import perf_counter

from evret.logging import get_logger
from evret.metrics._validation import validate_batch_lengths
from evret.utils import require_positive_int

logger = get_logger(__name__)


class Metric(ABC):
    """Base class for metrics evaluated at a top-k cutoff.

    For query ``i`` with retrieved labels ``R_i`` and expected labels ``G_i``,
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
        expected_answers: Collection[str],
    ) -> float:
        """Score a single query."""

    def score(
        self,
        retrieved_by_query: Sequence[Sequence[str]],
        expected_by_query: Sequence[Collection[str]],
    ) -> float:
        """Score a batch of queries by averaging per-query metric values."""
        started_at = perf_counter()
        validate_batch_lengths(retrieved_by_query, expected_by_query)

        if not retrieved_by_query:
            logger.debug("Metric score called with empty batch", extra={"metric": self.name})
            return 0.0

        total_score = 0.0
        num_queries = len(retrieved_by_query)
        is_debug = logger.isEnabledFor(logging.DEBUG)

        logger.info(
            "Computing metric over query batch",
            extra={"metric": self.name, "queries": num_queries, "k": self.k},
        )
        for retrieved, relevant in zip(retrieved_by_query, expected_by_query, strict=True):
            query_score = self.score_query(
                retrieved_doc_ids=retrieved,
                expected_answers=relevant,
            )
            total_score += query_score
            if is_debug:
                logger.debug(
                    "Computed per-query metric score",
                    extra={
                        "metric": self.name,
                        "retrieved_count": len(retrieved),
                        "expected_count": len(relevant),
                        "query_score": query_score,
                    },
                )

        final_score = total_score / num_queries
        logger.info(
            "Finished metric computation",
            extra={
                "metric": self.name,
                "queries": num_queries,
                "score": final_score,
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return final_score

    def top_k(self, retrieved_doc_ids: Sequence[str]) -> Sequence[str]:
        """Return the retrieval list trimmed to metric cutoff."""
        k_effective = min(self.k, len(retrieved_doc_ids))
        return retrieved_doc_ids[:k_effective]

    def _extract_top_k_size(self, retrieved_doc_ids: Sequence[str]) -> int:
        """Return the effective top-k size after considering retrieval list length."""
        return min(self.k, len(retrieved_doc_ids))
