# Qdrant Retriever

Use `QdrantRetriever` to evaluate collections stored in Qdrant.

## Install

```bash
pip install "evret[qdrant]"
```

## Basic Usage

```python
from evret.retrievers import QdrantRetriever

def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = QdrantRetriever(
    collection_name="docs",
    query_encoder=encode_query,
    url="http://localhost:6333",
    id_field="doc_id",
)

results = retriever.retrieve("what is retriever evaluation?", k=5)
for item in results:
    print(item.doc_id, item.score, item.metadata)
```

## With an Existing Qdrant Client

```python
from qdrant_client import QdrantClient
from evret.retrievers import QdrantRetriever

client = QdrantClient(url="http://localhost:6333")

retriever = QdrantRetriever(
    collection_name="docs",
    query_encoder=encode_query,
    client=client,
    query_filter=my_filter,
)
```

## Notes

- `query_encoder` must return a non-empty vector
- `doc_id` is resolved from payload `id_field` first, then point id
- Both `query_points` and `search` style client APIs are supported
