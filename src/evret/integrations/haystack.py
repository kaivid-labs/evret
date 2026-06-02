"""Haystack adapter for Evret retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evret.errors import OptionalDependencyError
from evret.retrievers import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str, require_positive_int

try:
    from haystack import Document, component

    _HAYSTACK_AVAILABLE = True
except ImportError:
    _HAYSTACK_AVAILABLE = False

    class Document:
        pass

    class _Component:
        def __call__(self, cls):
            return cls

        @staticmethod
        def output_types(**_types):
            def decorator(func):
                return func

            return decorator

    component = _Component()


@component
class HaystackRetrieverAdapter:
    """Bridge Evret retrievers and Haystack 2.x retriever components."""

    def __init__(
        self,
        evret_retriever: BaseRetriever | None = None,
        haystack_retriever: Any = None,
        k: int = 4,
        *,
        text_field: str = "document",
    ) -> None:
        if not _HAYSTACK_AVAILABLE:
            raise OptionalDependencyError(
                "haystack-ai is required for HaystackRetrieverAdapter. "
                "Install with: pip install haystack-ai"
            )
        if (evret_retriever is None) == (haystack_retriever is None):
            raise ValueError("provide exactly one of evret_retriever or haystack_retriever")

        self.evret_retriever = evret_retriever
        self.haystack_retriever = haystack_retriever
        self.k = require_positive_int(k, "k")
        self.text_field = require_non_empty_str(text_field, "text_field")

    @component.output_types(documents=list[Document])
    def run(self, query: str, top_k: int | None = None) -> dict[str, list[Document]]:
        if self.evret_retriever is None:
            raise ValueError("evret_retriever is required for Haystack retrieval")
        normalized_query = require_non_empty_str(query, "query")
        k = self._coerce_k(top_k if top_k is not None else self.k)

        results = self.evret_retriever.retrieve(query=normalized_query, k=k)
        documents: list[Document] = []
        for result in results:
            metadata = dict(result.metadata)
            content = str(metadata.pop(self.text_field, ""))
            metadata.setdefault("doc_id", result.doc_id)
            metadata.setdefault("score", result.score)
            documents.append(
                Document(
                    id=result.doc_id,
                    content=content,
                    meta=metadata,
                    score=result.score,
                )
            )
        return {"documents": documents}

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if self.haystack_retriever is None:
            raise ValueError("haystack_retriever is required for Evret retrieval")
        normalized_query = require_non_empty_str(query, "query")
        k = self._coerce_k(k)

        output = self.haystack_retriever.run(query=normalized_query, top_k=k)
        documents = output.get("documents", [])[:k]
        results: list[RetrievalResult] = []
        for index, document in enumerate(documents):
            metadata = dict(getattr(document, "meta", {}) or {})
            doc_id = str(
                getattr(document, "id", None)
                or metadata.get("doc_id")
                or metadata.get("id")
                or f"doc_{index}"
            )
            score = float(getattr(document, "score", None) or metadata.get("score", 0.0))
            content = getattr(document, "content", None)
            if content is not None:
                metadata[self.text_field] = content
            results.append(RetrievalResult(doc_id=doc_id, score=score, metadata=metadata))
        return results

    def batch_retrieve(self, queries: Sequence[str], k: int) -> list[list[RetrievalResult]]:
        self._coerce_k(k)
        return [self.retrieve(query=query, k=k) for query in queries]

    @staticmethod
    def _coerce_k(value: Any) -> int:
        return require_positive_int(value, "k")
