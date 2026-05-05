"""LangChain adapter for Evret retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evret.errors import OptionalDependencyError
from evret.retrievers import BaseRetriever, RetrievalResult
from evret.utils import require_non_empty_str, require_positive_int

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
    from pydantic import ConfigDict

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

    class LangChainBaseRetriever:
        pass

    class Document:
        pass

    class ConfigDict:
        pass

class LangChainRetrieverAdapter(LangChainBaseRetriever):
    """Bridge Evret and LangChain retrievers."""

    if _LANGCHAIN_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    evret_retriever: BaseRetriever | None = None
    langchain_retriever: Any = None
    k: int = 4

    def __init__(
        self,
        evret_retriever: BaseRetriever | None = None,
        langchain_retriever: Any = None,
        k: int = 4,
        **kwargs: Any,
    ) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise OptionalDependencyError(
                "langchain is required for LangChainRetrieverAdapter. "
                "Install with: pip install langchain"
            )
        if (evret_retriever is None) == (langchain_retriever is None):
            raise ValueError("provide exactly one of evret_retriever or langchain_retriever")
        super().__init__(
            evret_retriever=evret_retriever,
            langchain_retriever=langchain_retriever,
            k=self._coerce_k(k),
            **kwargs,
        )

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[LangChainDocument]:
        if self.evret_retriever is None:
            raise ValueError("evret_retriever is required for LangChain retrieval")
        normalized_query = require_non_empty_str(query, "query")
        k = self._coerce_k(kwargs.get("k", self.k))

        results = self.evret_retriever.retrieve(query=normalized_query, k=k)
        documents: list[Document] = []
        for result in results:
            metadata = dict(result.metadata)
            page_content = str(metadata.pop("document", ""))
            metadata.setdefault("doc_id", result.doc_id)
            metadata.setdefault("score", result.score)
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents

    async def _aget_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> list[LangChainDocument]:
        return self._get_relevant_documents(query=query, **kwargs)

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        if self.langchain_retriever is None:
            raise ValueError("langchain_retriever is required for Evret retrieval")
        normalized_query = require_non_empty_str(query, "query")
        k = self._coerce_k(k)

        documents = self.langchain_retriever.invoke(normalized_query)[:k]
        results: list[RetrievalResult] = []
        for index, document in enumerate(documents):
            metadata = dict(document.metadata)
            doc_id = str(metadata.get("doc_id") or metadata.get("id") or f"doc_{index}")
            score = float(metadata.get("score", metadata.get("_score", 0.0)))
            results.append(
                RetrievalResult(
                    doc_id=doc_id,
                    score=score,
                    metadata={**metadata, "document": document.page_content},
                )
            )
        return results

    def batch_retrieve(self, queries, k: int) -> list[list[RetrievalResult]]:
        self._coerce_k(k)
        return [self.retrieve(query=query, k=k) for query in queries]

    @staticmethod
    def _coerce_k(value: Any) -> int:
        return require_positive_int(value, "k")
