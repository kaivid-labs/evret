"""Base interface and data models for retriever integrations."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evret.logging import get_logger
from evret.utils import require_non_empty_str, require_positive_int

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Standard retrieval output for all retriever backends."""

    doc_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract retriever interface used by evaluation pipelines."""

    @abstractmethod
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        """Return top-``k`` results for a single query."""

    def batch_retrieve(self, queries: Sequence[str], k: int) -> list[list[RetrievalResult]]:
        """Retrieve for each query using the same cutoff ``k``."""
        self._validate_k(k)

        logger.debug(f"Batch retrieve: {len(queries)} queries, k={k}")
        start_time = time.perf_counter()

        results = [self.retrieve(query=self._validate_query(query), k=k) for query in queries]

        elapsed = time.perf_counter() - start_time
        total_docs = sum(len(r) for r in results)
        logger.debug(
            f"Batch retrieve complete: {len(queries)} queries, {total_docs} docs "
            f"in {elapsed:.2f}s ({elapsed/len(queries):.3f}s per query)"
        )

        return results

    def _validate_k(self, k: int) -> None:
        require_positive_int(k, "k")

    def _validate_query(self, query: str) -> str:
        return require_non_empty_str(query, "query")
