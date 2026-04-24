"""LangChain adapter for Evret retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evret.errors import OptionalDependencyError
from evret.retrievers import BaseRetriever
from evret.utils import require_non_empty_str, require_positive_int

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
    from pydantic import ConfigDict

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    Document = None
    LangChainBaseRetriever = object  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]
    _LANGCHAIN_AVAILABLE = False

if TYPE_CHECKING:
    from langchain_core.documents import Document as LangChainDocument


class LangChainRetrieverAdapter(LangChainBaseRetriever):
    """Wrap an Evret retriever as a LangChain-compatible retriever."""

    if _LANGCHAIN_AVAILABLE:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    evret_retriever: BaseRetriever
    k: int = 4

    def __init__(self, evret_retriever: BaseRetriever, k: int = 4, **kwargs: Any) -> None:
        if not _LANGCHAIN_AVAILABLE:
            raise OptionalDependencyError(
                "langchain-core is required for LangChainRetrieverAdapter. "
                "Install with: pip install langchain-core"
            )
        super().__init__(evret_retriever=evret_retriever, k=self._coerce_k(k), **kwargs)

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[LangChainDocument]:
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

    @staticmethod
    def _coerce_k(value: Any) -> int:
        return require_positive_int(value, "k")
