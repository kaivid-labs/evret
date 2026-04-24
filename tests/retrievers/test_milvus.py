import pytest

from evret.retrievers import MilvusRetriever


class FakeMilvusClient:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] = {}

    def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
        self.kwargs = kwargs
        return [
            [
                {
                    "id": "doc_1",
                    "distance": 0.87,
                    "entity": {"source": "a"},
                },
                {
                    "distance": 0.42,
                    "entity": {"doc_id": "doc_2", "source": "b"},
                },
            ]
        ]


def test_milvus_retrieve_uses_search_and_normalizes_results() -> None:
    client = FakeMilvusClient()
    retriever = MilvusRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.1, 0.2],
        client=client,
        search_filter='source == "a"',
        output_fields=["source"],
        search_params={"metric_type": "IP"},
        anns_field="vector",
    )

    results = retriever.retrieve("hello", k=2)

    assert client.kwargs["collection_name"] == "docs"
    assert client.kwargs["data"] == [[0.1, 0.2]]
    assert client.kwargs["limit"] == 2
    assert client.kwargs["filter"] == 'source == "a"'
    assert client.kwargs["output_fields"] == ["source"]
    assert client.kwargs["search_params"] == {"metric_type": "IP"}
    assert client.kwargs["anns_field"] == "vector"
    assert [result.doc_id for result in results] == ["doc_1", "doc_2"]
    assert [result.score for result in results] == [0.87, 0.42]
    assert results[0].metadata == {"source": "a"}


def test_milvus_retrieve_raises_for_empty_embedding() -> None:
    client = FakeMilvusClient()
    retriever = MilvusRetriever(
        collection_name="docs",
        query_encoder=lambda _: [],
        client=client,
    )

    with pytest.raises(ValueError, match="query_encoder must return a non-empty vector"):
        retriever.retrieve("hello", k=1)


def test_milvus_retrieve_raises_for_missing_doc_id() -> None:
    class MissingIdClient(FakeMilvusClient):
        def search(self, **kwargs: object) -> list[list[dict[str, object]]]:
            return [[{"distance": 0.1, "entity": {"source": "x"}}]]

    retriever = MilvusRetriever(
        collection_name="docs",
        query_encoder=lambda _: [0.3],
        client=MissingIdClient(),
    )

    with pytest.raises(ValueError, match="missing both `id` and entity field `doc_id`"):
        retriever.retrieve("hello", k=1)
