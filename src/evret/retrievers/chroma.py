"""ChromaDB retriever adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.retrievers.base import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str


class ChromaRetriever(BaseRetriever):
    """Retrieve documents from ChromaDB with a unified Evret interface."""

    def __init__(
        self,
        collection_name: str,
        client: Any | None = None,
        *,
        query_encoder: Callable[[str], Sequence[float]] | None = None,
        distance_to_score: Callable[[float], float] | None = None,
        client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        normalized_collection = require_non_empty_str(collection_name, "collection_name")

        self.collection_name = normalized_collection
        self.query_encoder = query_encoder
        self.distance_to_score = distance_to_score or self._default_distance_to_score
        self.collection = self._resolve_collection(
            collection_name=normalized_collection,
            client=client,
            client_kwargs=client_kwargs,
        )

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        normalized_query = self._validate_query(query)
        query_kwargs: dict[str, Any] = {
            "n_results": k,
            "include": ["metadatas", "distances", "documents"],
        }
        if self.query_encoder is None:
            query_kwargs["query_texts"] = [normalized_query]
        else:
            query_vector = list(self.query_encoder(normalized_query))
            if not query_vector:
                raise EvretValidationError("query_encoder must return a non-empty vector")
            query_kwargs["query_embeddings"] = [query_vector]

        response = self.collection.query(**query_kwargs)
        ids = self._first(response.get("ids"))
        metadatas = self._first(response.get("metadatas"))
        distances = self._first(response.get("distances"))
        documents = self._first(response.get("documents"))

        results: list[RetrievalResult] = []
        for idx, doc_id in enumerate(ids):
            metadata_raw = metadatas[idx] if idx < len(metadatas) else {}
            metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

            if idx < len(documents) and documents[idx] is not None:
                metadata.setdefault("document", documents[idx])

            distance_raw = distances[idx] if idx < len(distances) else None
            score = self.distance_to_score(float(distance_raw)) if distance_raw is not None else 0.0
            results.append(RetrievalResult(doc_id=str(doc_id), score=score, metadata=metadata))

        return results

    def _resolve_collection(
        self,
        collection_name: str,
        client: Any | None,
        client_kwargs: dict[str, Any] | None,
    ) -> Any:
        if client is None:
            client = self._create_client(client_kwargs=client_kwargs)

        if hasattr(client, "query"):
            return client
        if hasattr(client, "get_collection"):
            return client.get_collection(name=collection_name)
        if hasattr(client, "get_or_create_collection"):
            return client.get_or_create_collection(name=collection_name)
        raise TypeError(
            "client must be a Chroma collection or provide get_collection/get_or_create_collection"
        )

    def _create_client(self, client_kwargs: dict[str, Any] | None) -> Any:
        try:
            import chromadb
        except ImportError as exc:
            raise OptionalDependencyError(
                "chromadb is required for ChromaRetriever. Install with: pip install chromadb"
            ) from exc

        return chromadb.Client(**dict(client_kwargs or {}))

    @staticmethod
    def _default_distance_to_score(distance: float) -> float:
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _first(value: Any) -> list[Any]:
        if not value:
            return []
        return value[0]
