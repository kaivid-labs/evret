from types import SimpleNamespace

import pytest

from evret.retrievers import QdrantRetriever


class FakeQueryPointsClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def query_points(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            points=[
                {"id": "p1", "score": 0.91, "payload": {"doc_id": "doc_1", "source": "a"}},
                {"id": "p2", "score": 0.52, "payload": {"source": "b"}},
            ]
        )


class FakeSearchClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def search(self, **kwargs: object) -> list[object]:
        self.kwargs = kwargs
        return [
            SimpleNamespace(id="x1", score=0.7, payload={"source": "legacy"}),
        ]


def test_qdrant_retrieve_uses_query_points_when_available() -> None:
    client = FakeQueryPointsClient()
    retriever = QdrantRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.1, 0.2],
        client=client,
    )

    results = retriever.retrieve("hello", k=2)

    assert client.kwargs["collection_name"] == "docs"
    assert client.kwargs["query"] == [0.1, 0.2]
    assert client.kwargs["limit"] == 2
    assert [result.doc_id for result in results] == ["doc_1", "p2"]
    assert [result.score for result in results] == [0.91, 0.52]
    assert results[0].metadata == {"doc_id": "doc_1", "source": "a"}


def test_qdrant_retrieve_falls_back_to_search() -> None:
    client = FakeSearchClient()
    retriever = QdrantRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.5, 0.4],
        client=client,
    )

    results = retriever.retrieve("hello", k=1)

    assert client.kwargs["collection_name"] == "docs"
    assert client.kwargs["query_vector"] == [0.5, 0.4]
    assert [result.doc_id for result in results] == ["x1"]
    assert results[0].metadata == {"source": "legacy"}


def test_qdrant_retrieve_raises_for_empty_embedding() -> None:
    client = FakeQueryPointsClient()
    retriever = QdrantRetriever(
        collection_name="docs",
        query_encoder=lambda _: [],
        client=client,
    )

    with pytest.raises(ValueError, match="query_encoder must return a non-empty vector"):
        retriever.retrieve("hello", k=1)


def test_qdrant_retrieve_raises_for_missing_query_methods() -> None:
    retriever = QdrantRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.1],
        client=object(),
    )

    with pytest.raises(AttributeError, match="query_points or search"):
        retriever.retrieve("hello", k=1)
