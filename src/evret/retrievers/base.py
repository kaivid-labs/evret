"""Base interface and data models for retriever integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import perf_counter
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
        started_at = perf_counter()
        logger.info(
            "Starting retriever batch",
            extra={
                "retriever": type(self).__name__,
                "queries": len(queries),
                "k": k,
            },
        )

        results: list[list[RetrievalResult]] = []
        try:
            for query_index, query in enumerate(queries):
                normalized_query = self._validate_query(query)
                query_results = self.retrieve(query=normalized_query, k=k)
                results.append(query_results)
                logger.debug(
                    "Retrieved query results",
                    extra={
                        "retriever": type(self).__name__,
                        "query_index": query_index,
                        "k": k,
                        "results": len(query_results),
                    },
                )
        except Exception:
            logger.exception(
                "Retriever batch failed",
                extra={
                    "retriever": type(self).__name__,
                    "queries": len(queries),
                    "completed_queries": len(results),
                    "k": k,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
            raise

        total_results = sum(len(query_results) for query_results in results)
        logger.info(
            "Finished retriever batch",
            extra={
                "retriever": type(self).__name__,
                "queries": len(queries),
                "k": k,
                "total_results": total_results,
                "empty_result_queries": sum(1 for query_results in results if not query_results),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return results

    def _validate_k(self, k: int) -> None:
        require_positive_int(k, "k")

    def _validate_query(self, query: str) -> str:
        return require_non_empty_str(query, "query")
