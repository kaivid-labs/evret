from types import SimpleNamespace

import pytest

from evret.retrievers import WeaviateRetriever


class FakeCollection:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}
        self.query = SimpleNamespace(near_vector=self.near_vector)

    def near_vector(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return SimpleNamespace(
            objects=[
                SimpleNamespace(
                    uuid="uuid-1",
                    properties={"doc_id": "doc_1", "source": "a"},
                    metadata=SimpleNamespace(distance=0.25),
                ),
                {
                    "uuid": "uuid-2",
                    "properties": {"source": "b"},
                    "metadata": {"certainty": 0.91},
                },
            ]
        )


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.collections = SimpleNamespace(use=self.use)
        self.requested_name: str | None = None

    def use(self, name: str) -> FakeCollection:
        self.requested_name = name
        return self.collection


def test_weaviate_retrieve_uses_near_vector_and_normalizes_results() -> None:
    collection = FakeCollection()
    client = FakeClient(collection=collection)
    retriever = WeaviateRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.1, 0.2],
        client=client,
    )

    results = retriever.retrieve("hello", k=2)

    assert client.requested_name == "docs"
    assert collection.kwargs["near_vector"] == [0.1, 0.2]
    assert collection.kwargs["limit"] == 2
    assert [result.doc_id for result in results] == ["doc_1", "uuid-2"]
    assert [round(result.score, 2) for result in results] == [0.8, 0.91]
    assert results[0].metadata["source"] == "a"


def test_weaviate_retrieve_raises_for_empty_embedding() -> None:
    collection = FakeCollection()
    retriever = WeaviateRetriever(
        collection_name="docs",
        query_encoder=lambda _: [],
        client=collection,
    )

    with pytest.raises(ValueError, match="query_encoder must return a non-empty vector"):
        retriever.retrieve("hello", k=1)


def test_weaviate_retrieve_raises_when_collection_has_no_near_vector() -> None:
    retriever = WeaviateRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.1],
        client=SimpleNamespace(query=SimpleNamespace()),
    )

    with pytest.raises(AttributeError, match="query.near_vector"):
        retriever.retrieve("hello", k=1)
