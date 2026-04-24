"""LlamaIndex adapter for Evret retrievers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evret.errors import OptionalDependencyError
from evret.retrievers import BaseRetriever
from evret.utils import require_non_empty_str, require_positive_int

try:
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever as LlamaIndexBaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    _LLAMAINDEX_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when optional dependency is absent
    QueryBundle = None  # type: ignore[assignment]
    LlamaIndexBaseRetriever = object  # type: ignore[assignment]
    NodeWithScore = None  # type: ignore[assignment]
    TextNode = None  # type: ignore[assignment]
    _LLAMAINDEX_AVAILABLE = False

if TYPE_CHECKING:
    from llama_index.core import QueryBundle as LlamaIndexQueryBundle
    from llama_index.core.schema import NodeWithScore as LlamaIndexNodeWithScore


class LlamaIndexRetrieverAdapter(LlamaIndexBaseRetriever):
    """Wrap an Evret retriever as a LlamaIndex-compatible retriever."""

    def __init__(
        self,
        evret_retriever: BaseRetriever,
        k: int = 4,
        *,
        text_field: str = "document",
        **kwargs: Any,
    ) -> None:
        if not _LLAMAINDEX_AVAILABLE:
            raise OptionalDependencyError(
                "llama-index-core is required for LlamaIndexRetrieverAdapter. "
                "Install with: pip install llama-index-core"
            )

        super().__init__(**kwargs)
        self.evret_retriever = evret_retriever
        self.k = require_positive_int(k, "k")
        self.text_field = require_non_empty_str(text_field, "text_field")

    def _retrieve(self, query_bundle: LlamaIndexQueryBundle) -> list[LlamaIndexNodeWithScore]:
        query = require_non_empty_str(query_bundle.query_str, "query")
        results = self.evret_retriever.retrieve(query=query, k=self.k)

        nodes: list[NodeWithScore] = []
        for result in results:
            metadata = dict(result.metadata)
            node_text = str(metadata.pop(self.text_field, ""))
            metadata.setdefault("doc_id", result.doc_id)
            metadata.setdefault("score", result.score)
            node = TextNode(text=node_text, id_=result.doc_id, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=result.score))
        return nodes

    async def _aretrieve(self, query_bundle: LlamaIndexQueryBundle) -> list[LlamaIndexNodeWithScore]:
        return self._retrieve(query_bundle)
