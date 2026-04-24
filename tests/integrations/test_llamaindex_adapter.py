import pytest

pytest.importorskip("llama_index.core")

from evret.integrations import LlamaIndexRetrieverAdapter
from evret.retrievers import BaseRetriever, RetrievalResult


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self.calls.append((query, k))
        return [
            RetrievalResult(
                doc_id="doc_1",
                score=0.87,
                metadata={"document": "alpha content", "source": "unit-test"},
            )
        ]


def test_llamaindex_adapter_converts_evret_results_to_nodes() -> None:
    retriever = DummyRetriever()
    adapter = LlamaIndexRetrieverAdapter(evret_retriever=retriever, k=2)

    nodes = adapter.retrieve("alpha query")

    assert retriever.calls == [("alpha query", 2)]
    assert len(nodes) == 1
    node = nodes[0].node
    text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
    assert text == "alpha content"
    assert node.metadata["doc_id"] == "doc_1"
    assert node.metadata["score"] == 0.87
    assert node.metadata["source"] == "unit-test"


def test_llamaindex_adapter_raises_for_non_positive_k() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="k must be a positive integer"):
        LlamaIndexRetrieverAdapter(evret_retriever=retriever, k=0)
