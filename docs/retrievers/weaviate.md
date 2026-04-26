# Weaviate Retriever

Use `WeaviateRetriever` to evaluate objects in Weaviate collections.

## Install

```bash
pip install "evret[weaviate]"
```

## Basic Usage

```python
from evret.retrievers import WeaviateRetriever

def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = WeaviateRetriever(
    collection_name="Docs",
    query_encoder=encode_query,
    url="http://localhost:8080",
    id_field="doc_id",
    return_properties=["doc_id", "text", "source"],
)

results = retriever.retrieve("how to tune ndcg", k=5)
for item in results:
    print(item.doc_id, item.score, item.metadata)
```

## With an Existing Weaviate Client

```python
import weaviate

client = weaviate.connect_to_local()

retriever = WeaviateRetriever(
    collection_name="Docs",
    query_encoder=encode_query,
    client=client,
    query_filter=my_filter,
)
```

## Notes

- `query_encoder` must return a non-empty vector
- Scores are derived from metadata (`certainty`, `distance`, or `score`)
- Works with clients that expose `collections.use/get`
