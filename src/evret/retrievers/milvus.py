"""Milvus retriever adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.retrievers.base import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str


class MilvusRetriever(BaseRetriever):
    """Retrieve documents from Milvus with a unified Evret interface."""

    def __init__(
        self,
        collection_name: str,
        query_encoder: Callable[[str], Sequence[float]],
        client: Any | None = None,
        *,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        id_field: str = "doc_id",
        search_filter: str | None = None,
        output_fields: Sequence[str] | None = None,
        search_params: dict[str, Any] | None = None,
        anns_field: str | None = None,
        partition_names: Sequence[str] | None = None,
        distance_to_score: Callable[[float], float] | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        normalized_collection = require_non_empty_str(collection_name, "collection_name")
        normalized_id_field = require_non_empty_str(id_field, "id_field")

        self.collection_name = normalized_collection
        self.query_encoder = query_encoder
        self.id_field = normalized_id_field
        self.search_filter = search_filter
        self.output_fields = list(output_fields) if output_fields else None
        self.search_params = dict(search_params) if search_params is not None else None
        self.anns_field = anns_field
        self.partition_names = list(partition_names) if partition_names else None
        self.distance_to_score = distance_to_score or self._default_distance_to_score
        self.client = client or self._create_client(uri=uri, token=token, client_kwargs=client_kwargs)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        normalized_query = self._validate_query(query)
        query_vector = list(self.query_encoder(normalized_query))
        if not query_vector:
            raise EvretValidationError("query_encoder must return a non-empty vector")

        search_kwargs: dict[str, Any] = {
            "collection_name": self.collection_name,
            "data": [query_vector],
            "limit": k,
        }
        if self.search_filter:
            search_kwargs["filter"] = self.search_filter
        if self.output_fields is not None:
            search_kwargs["output_fields"] = self.output_fields
        if self.search_params is not None:
            search_kwargs["search_params"] = self.search_params
        if self.anns_field is not None:
            search_kwargs["anns_field"] = self.anns_field
        if self.partition_names is not None:
            search_kwargs["partition_names"] = self.partition_names

        raw_response = self.client.search(**search_kwargs)
        hits = self._extract_hits(raw_response)
        return [self._to_result(hit) for hit in hits]

    def _create_client(
        self, uri: str, token: str | None, client_kwargs: dict[str, Any] | None
    ) -> Any:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:
            raise OptionalDependencyError(
                "pymilvus is required for MilvusRetriever. Install with: pip install pymilvus"
            ) from exc

        kwargs = dict(client_kwargs or {})
        kwargs.setdefault("uri", require_non_empty_str(uri, "uri"))
        if token is not None:
            kwargs.setdefault("token", token)
        return MilvusClient(**kwargs)

    def _to_result(self, hit: Any) -> RetrievalResult:
        metadata_raw = self._value(hit, "entity", {})
        metadata = dict(metadata_raw) if isinstance(metadata_raw, Mapping) else {}

        raw_id = self._value(hit, "id", None)
        if raw_id is None:
            raw_id = metadata.get(self.id_field)
        if raw_id is None:
            raise EvretValidationError(
                f"Milvus hit is missing both `id` and entity field `{self.id_field}`"
            )

        distance = self._value(hit, "distance", self._value(hit, "score", 0.0))
        score = self.distance_to_score(float(distance))
        return RetrievalResult(doc_id=str(raw_id), score=score, metadata=metadata)

    @staticmethod
    def _extract_hits(response: Any) -> list[Any]:
        if not isinstance(response, list):
            return []
        if not response:
            return []
        first = response[0]
        if isinstance(first, list):
            return first
        return response

    @staticmethod
    def _default_distance_to_score(distance: float) -> float:
        return float(distance)

    @staticmethod
    def _value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)
