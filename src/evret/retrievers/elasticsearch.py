"""Elasticsearch retriever adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Callable

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.logging import get_logger
from evret.retrievers.base import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str

logger = get_logger(__name__)


class ElasticsearchRetriever(BaseRetriever):
    """Retrieve documents from Elasticsearch kNN search with a unified Evret interface."""

    def __init__(
        self,
        index_name: str,
        vector_field: str,
        query_encoder: Callable[[str], Sequence[float]],
        client: Any | None = None,
        *,
        id_field: str = "doc_id",
        filter: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        num_candidates: int | None = None,
        source: bool | Sequence[str] | Mapping[str, Any] = True,
        fields: Sequence[str] | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.index_name = require_non_empty_str(index_name, "index_name")
        self.vector_field = require_non_empty_str(vector_field, "vector_field")
        self.id_field = require_non_empty_str(id_field, "id_field")
        self.query_encoder = query_encoder
        self.filter = filter
        self.num_candidates = num_candidates
        self.source = source
        self.fields = list(fields) if fields else None
        self.client = client or self._create_client(client_kwargs=client_kwargs)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        started_at = perf_counter()
        self._validate_k(k)
        normalized_query = self._validate_query(query)
        query_vector = list(self.query_encoder(normalized_query))
        if not query_vector:
            raise EvretValidationError("query_encoder must return a non-empty vector")

        knn: dict[str, Any] = {
            "field": self.vector_field,
            "query_vector": query_vector,
            "k": k,
            "num_candidates": self.num_candidates or max(k * 10, k),
        }
        if self.filter is not None:
            knn["filter"] = self.filter

        search_kwargs: dict[str, Any] = {
            "index": self.index_name,
            "knn": knn,
            "size": k,
            "_source": self.source,
        }
        if self.fields is not None:
            search_kwargs["fields"] = self.fields

        response = self.client.search(**search_kwargs)
        hits = self._hits(response)
        results = [self._to_result(hit) for hit in hits]
        logger.debug(
            "Elasticsearch retrieval completed",
            extra={
                "retriever": type(self).__name__,
                "index": self.index_name,
                "k": k,
                "num_candidates": knn["num_candidates"],
                "vector_dimensions": len(query_vector),
                "has_filter": self.filter is not None,
                "has_fields": self.fields is not None,
                "results": len(results),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return results

    def _create_client(self, client_kwargs: dict[str, Any] | None) -> Any:
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise OptionalDependencyError(
                "elasticsearch is required for ElasticsearchRetriever. "
                "Install with: pip install elasticsearch"
            ) from exc

        return Elasticsearch(**dict(client_kwargs or {}))

    def _to_result(self, hit: Any) -> RetrievalResult:
        source_raw = self._value(hit, "_source", {})
        metadata = dict(source_raw) if isinstance(source_raw, Mapping) else {}

        fields_raw = self._value(hit, "fields", {})
        if isinstance(fields_raw, Mapping) and fields_raw:
            metadata.setdefault("fields", dict(fields_raw))

        raw_id = metadata.get(self.id_field, self._value(hit, "_id", None))
        if raw_id is None:
            raise EvretValidationError(
                f"Elasticsearch hit is missing both `_id` and source field `{self.id_field}`"
            )

        score = float(self._value(hit, "_score", 0.0))
        return RetrievalResult(doc_id=str(raw_id), score=score, metadata=metadata)

    @classmethod
    def _hits(cls, response: Any) -> list[Any]:
        hits_obj = cls._value(response, "hits", {})
        hits = cls._value(hits_obj, "hits", [])
        return list(hits or [])

    @staticmethod
    def _value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)
