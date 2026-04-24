"""Base interface and data models for retriever integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evret.utils import require_non_empty_str, require_positive_int


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
        return [self.retrieve(query=self._validate_query(query), k=k) for query in queries]

    def _validate_k(self, k: int) -> None:
        require_positive_int(k, "k")

    def _validate_query(self, query: str) -> str:
        return require_non_empty_str(query, "query")
