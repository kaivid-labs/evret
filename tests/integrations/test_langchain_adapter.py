import pytest

pytest.importorskip("langchain")

from evret.integrations import LangChainRetrieverAdapter
from evret.retrievers import BaseRetriever, RetrievalResult
from langchain_core.documents import Document


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


class DummyLangChainRetriever:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, query: str):
        self.calls.append(query)
        return [
            Document(
                page_content="alpha content",
                metadata={"doc_id": "doc_1", "score": 0.91, "source": "unit-test"},
            ),
            Document(page_content="beta content", metadata={"doc_id": "doc_2"}),
        ]


def test_langchain_adapter_converts_evret_results_to_documents() -> None:
    retriever = DummyRetriever()
    adapter = LangChainRetrieverAdapter(evret_retriever=retriever, k=2)

    docs = adapter.invoke("alpha query")

    assert retriever.calls == [("alpha query", 2)]
    assert len(docs) == 1
    assert docs[0].page_content == "alpha content"
    assert docs[0].metadata["doc_id"] == "doc_1"
    assert docs[0].metadata["score"] == 0.87
    assert docs[0].metadata["source"] == "unit-test"


def test_langchain_adapter_allows_request_level_k_override() -> None:
    retriever = DummyRetriever()
    adapter = LangChainRetrieverAdapter(evret_retriever=retriever, k=4)

    adapter._get_relevant_documents("beta query", k=1)

    assert retriever.calls == [("beta query", 1)]


def test_langchain_adapter_raises_for_non_positive_k() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="k must be a positive integer"):
        LangChainRetrieverAdapter(evret_retriever=retriever, k=0)


def test_langchain_adapter_supports_lcel_chains() -> None:
    runnables = pytest.importorskip("langchain_core.runnables")
    runnable_lambda = runnables.RunnableLambda

    retriever = DummyRetriever()
    adapter = LangChainRetrieverAdapter(evret_retriever=retriever, k=2)
    chain = adapter | runnable_lambda(lambda docs: docs[0].metadata["doc_id"])

    result = chain.invoke("alpha query")

    assert result == "doc_1"


def test_langchain_adapter_converts_langchain_documents_to_evret_results() -> None:
    retriever = DummyLangChainRetriever()
    adapter = LangChainRetrieverAdapter(langchain_retriever=retriever)

    results = adapter.retrieve("alpha query", k=1)

    assert retriever.calls == ["alpha query"]
    assert results == [
        RetrievalResult(
            doc_id="doc_1",
            score=0.91,
            metadata={
                "doc_id": "doc_1",
                "score": 0.91,
                "source": "unit-test",
                "document": "alpha content",
            },
        )
    ]
