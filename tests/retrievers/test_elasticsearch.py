from types import SimpleNamespace

import pytest

from evret.retrievers import ElasticsearchRetriever


class FakeElasticsearchClient:
    def __init__(self, response: object | None = None) -> None:
        self.kwargs: dict[str, object] = {}
        self.response = response or {
            "hits": {
                "hits": [
                    {
                        "_id": "es_1",
                        "_score": 0.91,
                        "_source": {"doc_id": "doc_1", "source": "a"},
                        "fields": {"title": ["A"]},
                    },
                    {
                        "_id": "es_2",
                        "_score": 0.52,
                        "_source": {"source": "b"},
                    },
                ]
            }
        }

    def search(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.response


def test_elasticsearch_retrieve_uses_knn_search_and_normalizes_results() -> None:
    client = FakeElasticsearchClient()
    retriever = ElasticsearchRetriever(
        index_name="docs",
        vector_field="embedding",
        query_encoder=lambda _: [0.1, 0.2],
        client=client,
        filter={"term": {"file_type": "pdf"}},
        num_candidates=20,
        fields=["title"],
    )

    results = retriever.retrieve("hello", k=2)

    assert client.kwargs["index"] == "docs"
    assert client.kwargs["size"] == 2
    assert client.kwargs["_source"] is True
    assert client.kwargs["fields"] == ["title"]
    assert client.kwargs["knn"] == {
        "field": "embedding",
        "query_vector": [0.1, 0.2],
        "k": 2,
        "num_candidates": 20,
        "filter": {"term": {"file_type": "pdf"}},
    }
    assert [result.doc_id for result in results] == ["doc_1", "es_2"]
    assert [result.score for result in results] == [0.91, 0.52]
    assert results[0].metadata == {
        "doc_id": "doc_1",
        "source": "a",
        "fields": {"title": ["A"]},
    }


def test_elasticsearch_retrieve_supports_object_style_response() -> None:
    response = SimpleNamespace(
        hits=SimpleNamespace(
            hits=[
                SimpleNamespace(
                    _id="es_1",
                    _score=0.75,
                    _source={"source": "object"},
                )
            ]
        )
    )
    client = FakeElasticsearchClient(response=response)
    retriever = ElasticsearchRetriever(
        index_name="docs",
        vector_field="embedding",
        query_encoder=lambda _: [0.3],
        client=client,
    )

    results = retriever.retrieve("hello", k=1)

    assert client.kwargs["knn"]["num_candidates"] == 10
    assert results[0].doc_id == "es_1"
    assert results[0].score == 0.75
    assert results[0].metadata == {"source": "object"}


def test_elasticsearch_retrieve_raises_for_empty_embedding() -> None:
    retriever = ElasticsearchRetriever(
        index_name="docs",
        vector_field="embedding",
        query_encoder=lambda _: [],
        client=FakeElasticsearchClient(),
    )

    with pytest.raises(ValueError, match="query_encoder must return a non-empty vector"):
        retriever.retrieve("hello", k=1)


def test_elasticsearch_retrieve_raises_for_missing_doc_id() -> None:
    client = FakeElasticsearchClient(response={"hits": {"hits": [{"_score": 0.1, "_source": {}}]}})
    retriever = ElasticsearchRetriever(
        index_name="docs",
        vector_field="embedding",
        query_encoder=lambda _: [0.3],
        client=client,
    )

    with pytest.raises(ValueError, match="missing both `_id` and source field `doc_id`"):
        retriever.retrieve("hello", k=1)
