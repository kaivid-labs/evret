# Elasticsearch Retriever

Use `ElasticsearchRetriever` to evaluate dense-vector kNN search results from Elasticsearch.

## Install

```bash
pip install "evret[elasticsearch]"
```

## Basic Usage

```python
from evret.retrievers import ElasticsearchRetriever

def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = ElasticsearchRetriever(
    index_name="docs",
    vector_field="embedding",
    query_encoder=encode_query,
    client_kwargs={"hosts": ["http://localhost:9200"]},
    id_field="doc_id",
)

results = retriever.retrieve("what is retriever evaluation?", k=5)
for item in results:
    print(item.doc_id, item.score, item.metadata)
```

## With an Existing Elasticsearch Client

```python
from elasticsearch import Elasticsearch
from evret.retrievers import ElasticsearchRetriever

client = Elasticsearch("http://localhost:9200")

retriever = ElasticsearchRetriever(
    index_name="docs",
    vector_field="embedding",
    query_encoder=encode_query,
    client=client,
    filter={"term": {"file_type": "pdf"}},
    num_candidates=50,
)
```

## Notes

- `query_encoder` must return a non-empty vector
- Search uses Elasticsearch's `knn` option with `field`, `query_vector`, `k`, and `num_candidates`
- `doc_id` is resolved from source `id_field` first, then Elasticsearch `_id`
- `filter` is passed inside the `knn` object so Elasticsearch applies it during vector search
