import pytest

from evret.retrievers import BaseRetriever, RetrievalResult


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        self.calls.append((query, k))
        return [RetrievalResult(doc_id=f"{query}-{k}", score=1.0)]


def test_batch_retrieve_calls_single_query_retrieve_for_each_query() -> None:
    retriever = DummyRetriever()

    results = retriever.batch_retrieve(queries=["q1", "q2"], k=3)

    assert retriever.calls == [("q1", 3), ("q2", 3)]
    assert results == [
        [RetrievalResult(doc_id="q1-3", score=1.0)],
        [RetrievalResult(doc_id="q2-3", score=1.0)],
    ]


def test_batch_retrieve_raises_for_non_positive_k() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="k must be a positive integer"):
        retriever.batch_retrieve(queries=["q1"], k=0)


def test_batch_retrieve_raises_for_empty_query() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="query must be a non-empty string"):
        retriever.batch_retrieve(queries=["   "], k=1)
