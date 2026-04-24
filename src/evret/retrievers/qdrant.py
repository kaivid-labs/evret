"""Qdrant retriever adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.retrievers.base import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str


class QdrantRetriever(BaseRetriever):
    """Retrieve documents from Qdrant with a unified Evret interface."""

    def __init__(
        self,
        collection_name: str,
        query_encoder: Callable[[str], Sequence[float]],
        client: Any | None = None,
        *,
        url: str = "http://localhost:6333",
        id_field: str = "doc_id",
        query_filter: Any | None = None,
        search_params: Any | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        normalized_collection = require_non_empty_str(collection_name, "collection_name")
        normalized_id_field = require_non_empty_str(id_field, "id_field")

        self.collection_name = normalized_collection
        self.query_encoder = query_encoder
        self.id_field = normalized_id_field
        self.query_filter = query_filter
        self.search_params = search_params
        self.client = client or self._create_client(url=url, client_kwargs=client_kwargs)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        normalized_query = self._validate_query(query)
        query_vector = list(self.query_encoder(normalized_query))
        if not query_vector:
            raise EvretValidationError("query_encoder must return a non-empty vector")

        raw_points = self._query_points(query_vector=query_vector, k=k)
        return [self._to_result(point) for point in raw_points]

    def _create_client(self, url: str, client_kwargs: dict[str, Any] | None) -> Any:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise OptionalDependencyError(
                "qdrant-client is required for QdrantRetriever. Install with: pip install qdrant-client"
            ) from exc

        kwargs = dict(client_kwargs or {})
        kwargs.setdefault("url", url)
        return QdrantClient(**kwargs)

    def _query_points(self, query_vector: list[float], k: int) -> list[Any]:
        query_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "limit": k,
        }
        if self.query_filter is not None:
            query_kwargs["query_filter"] = self.query_filter
        if self.search_params is not None:
            query_kwargs["search_params"] = self.search_params

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(query=query_vector, **query_kwargs)
            points = getattr(response, "points", response)
            return list(points)

        if hasattr(self.client, "search"):
            return list(self.client.search(query_vector=query_vector, **query_kwargs))

        raise AttributeError("Qdrant client must provide query_points or search")

    def _to_result(self, point: Any) -> RetrievalResult:
        point_id = self._point_value(point, "id")
        score = float(self._point_value(point, "score", 0.0))
        payload_raw = self._point_value(point, "payload", {})
        metadata = dict(payload_raw) if isinstance(payload_raw, dict) else {}
        if point_id is None and self.id_field not in metadata:
            raise EvretValidationError(
                f"Qdrant point is missing both `id` and payload field `{self.id_field}`"
            )
        doc_id = metadata.get(self.id_field, point_id)
        return RetrievalResult(doc_id=str(doc_id), score=score, metadata=metadata)

    @staticmethod
    def _point_value(point: Any, key: str, default: Any = None) -> Any:
        if isinstance(point, dict):
            return point.get(key, default)
        return getattr(point, key, default)
