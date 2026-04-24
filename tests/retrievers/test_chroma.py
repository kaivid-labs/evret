import pytest

from evret.retrievers import ChromaRetriever


class FakeCollection:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
        self.kwargs = kwargs
        return {
            "ids": [["doc_1", "doc_2"]],
            "metadatas": [[{"source": "a"}, {"source": "b"}]],
            "distances": [[0.0, 1.0]],
            "documents": [["text one", "text two"]],
        }


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.requested_name: str | None = None

    def get_or_create_collection(self, name: str) -> FakeCollection:
        self.requested_name = name
        return self.collection


def test_chroma_retrieve_uses_query_texts_without_encoder() -> None:
    collection = FakeCollection()
    client = FakeClient(collection=collection)
    retriever = ChromaRetriever(collection_name="docs", client=client)

    results = retriever.retrieve("hello world", k=2)

    assert client.requested_name == "docs"
    assert collection.kwargs["query_texts"] == ["hello world"]
    assert collection.kwargs["n_results"] == 2
    assert [result.doc_id for result in results] == ["doc_1", "doc_2"]
    assert [result.score for result in results] == [1.0, 0.5]
    assert results[0].metadata == {"source": "a", "document": "text one"}


def test_chroma_retrieve_uses_query_embeddings_with_encoder() -> None:
    collection = FakeCollection()
    retriever = ChromaRetriever(
        collection_name="docs",
        client=collection,
        query_encoder=lambda _: [0.1, 0.2],
    )

    retriever.retrieve("hello world", k=1)

    assert collection.kwargs["query_embeddings"] == [[0.1, 0.2]]
    assert "query_texts" not in collection.kwargs


def test_chroma_retrieve_raises_for_empty_embedding() -> None:
    collection = FakeCollection()
    retriever = ChromaRetriever(
        collection_name="docs",
        client=collection,
        query_encoder=lambda _: [],
    )

    with pytest.raises(ValueError, match="query_encoder must return a non-empty vector"):
        retriever.retrieve("hello", k=1)


def test_chroma_retrieve_uses_custom_distance_to_score() -> None:
    collection = FakeCollection()
    retriever = ChromaRetriever(
        collection_name="docs",
        client=collection,
        distance_to_score=lambda distance: round(10 - distance, 2),
    )

    results = retriever.retrieve("hello", k=2)

    assert [result.score for result in results] == [10.0, 9.0]
