"""Weaviate retriever adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable
from urllib.parse import urlparse

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.retrievers.base import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str


class WeaviateRetriever(BaseRetriever):
    """Retrieve documents from Weaviate with a unified Evret interface."""

    def __init__(
        self,
        collection_name: str,
        query_encoder: Callable[[str], Sequence[float]],
        client: Any | None = None,
        *,
        url: str = "http://localhost:8080",
        grpc_port: int = 50051,
        id_field: str = "doc_id",
        query_filter: Any | None = None,
        return_properties: Sequence[str] | None = None,
        return_metadata: Any | None = None,
        distance_to_score: Callable[[float], float] | None = None,
        near_vector_kwargs: dict[str, Any] | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        normalized_collection = require_non_empty_str(collection_name, "collection_name")
        normalized_id_field = require_non_empty_str(id_field, "id_field")

        self.collection_name = normalized_collection
        self.query_encoder = query_encoder
        self.id_field = normalized_id_field
        self.query_filter = query_filter
        self.return_properties = list(return_properties) if return_properties else None
        self.return_metadata = return_metadata
        self.distance_to_score = distance_to_score or self._default_distance_to_score
        self.near_vector_kwargs = dict(near_vector_kwargs or {})
        self.collection = self._resolve_collection(
            collection_name=normalized_collection,
            client=client,
            url=url,
            grpc_port=grpc_port,
            client_kwargs=client_kwargs,
        )

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        normalized_query = self._validate_query(query)
        query_vector = list(self.query_encoder(normalized_query))
        if not query_vector:
            raise EvretValidationError("query_encoder must return a non-empty vector")

        if not hasattr(self.collection, "query") or not hasattr(self.collection.query, "near_vector"):
            raise AttributeError("Weaviate collection must provide query.near_vector")

        query_kwargs: dict[str, Any] = {"near_vector": query_vector, "limit": k}
        if self.query_filter is not None:
            query_kwargs["filters"] = self.query_filter
        if self.return_properties is not None:
            query_kwargs["return_properties"] = self.return_properties
        if self.return_metadata is not None:
            query_kwargs["return_metadata"] = self.return_metadata
        query_kwargs.update(self.near_vector_kwargs)

        response = self.collection.query.near_vector(**query_kwargs)
        objects = getattr(response, "objects", response)
        return [self._to_result(obj) for obj in list(objects)]

    def _resolve_collection(
        self,
        collection_name: str,
        client: Any | None,
        url: str,
        grpc_port: int,
        client_kwargs: dict[str, Any] | None,
    ) -> Any:
        if client is None:
            client = self._create_client(url=url, grpc_port=grpc_port, client_kwargs=client_kwargs)

        if hasattr(client, "query"):
            return client

        collections = getattr(client, "collections", None)
        if collections is None:
            raise TypeError("client must be a Weaviate collection or provide collections.use/get")

        if hasattr(collections, "use"):
            return collections.use(collection_name)
        if hasattr(collections, "get"):
            return collections.get(collection_name)

        raise TypeError("client must be a Weaviate collection or provide collections.use/get")

    def _create_client(self, url: str, grpc_port: int, client_kwargs: dict[str, Any] | None) -> Any:
        try:
            import weaviate
        except ImportError as exc:
            raise OptionalDependencyError(
                "weaviate-client is required for WeaviateRetriever. Install with: pip install weaviate-client"
            ) from exc

        parsed = self._parse_url(url)
        secure = parsed.scheme == "https"
        default_http_port = 443 if secure else 8080

        kwargs = dict(client_kwargs or {})
        if hasattr(weaviate, "connect_to_custom"):
            kwargs.setdefault("http_host", parsed.hostname)
            kwargs.setdefault("http_port", parsed.port or default_http_port)
            kwargs.setdefault("http_secure", secure)
            kwargs.setdefault("grpc_host", parsed.hostname)
            kwargs.setdefault("grpc_port", grpc_port)
            kwargs.setdefault("grpc_secure", secure)
            return weaviate.connect_to_custom(**kwargs)

        if hasattr(weaviate, "connect_to_local"):
            kwargs.setdefault("host", parsed.hostname)
            kwargs.setdefault("port", parsed.port or default_http_port)
            kwargs.setdefault("grpc_port", grpc_port)
            return weaviate.connect_to_local(**kwargs)

        raise OptionalDependencyError(
            "Unsupported weaviate-client version: expected connect_to_custom or connect_to_local"
        )

    def _to_result(self, item: Any) -> RetrievalResult:
        properties_raw = self._value(item, "properties", {})
        properties = dict(properties_raw) if isinstance(properties_raw, Mapping) else {}

        metadata_raw = self._value(item, "metadata", {})
        metadata = self._metadata_to_dict(metadata_raw)

        raw_id = self._value(item, "uuid", self._value(item, "id", None))
        doc_id = properties.get(self.id_field, raw_id)
        if doc_id is None:
            raise EvretValidationError(
                f"Weaviate result is missing both object id and property field `{self.id_field}`"
            )

        score = self._score_from_metadata(metadata)
        result_metadata = dict(properties)
        if metadata:
            result_metadata["weaviate_metadata"] = metadata

        return RetrievalResult(doc_id=str(doc_id), score=score, metadata=result_metadata)

    def _score_from_metadata(self, metadata: dict[str, Any]) -> float:
        certainty = metadata.get("certainty")
        if certainty is not None:
            return float(certainty)

        distance = metadata.get("distance")
        if distance is not None:
            return self.distance_to_score(float(distance))

        score = metadata.get("score")
        if score is not None:
            return float(score)

        return 0.0

    @staticmethod
    def _metadata_to_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)

        metadata: dict[str, Any] = {}
        for key in ("distance", "certainty", "score", "explain_score"):
            attr = getattr(value, key, None)
            if attr is not None:
                metadata[key] = attr
        return metadata

    @staticmethod
    def _parse_url(url: str) -> Any:
        normalized = require_non_empty_str(url, "url")
        if "://" not in normalized:
            normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
        if parsed.hostname is None:
            raise EvretValidationError(f"url must include a valid host: {url}")
        return parsed

    @staticmethod
    def _default_distance_to_score(distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _value(item: Any, key: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(key, default)
        return getattr(item, key, default)
