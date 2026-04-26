# Milvus Retriever

Use `MilvusRetriever` to evaluate Milvus search results.

## Install

```bash
pip install "evret[milvus]"
```

## Basic Usage

```python
from evret.retrievers import MilvusRetriever

def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = MilvusRetriever(
    collection_name="docs",
    query_encoder=encode_query,
    uri="http://localhost:19530",
    id_field="doc_id",
    anns_field="embedding",
    output_fields=["doc_id", "text", "source"],
    search_params={"metric_type": "COSINE", "params": {"nprobe": 10}},
)

results = retriever.retrieve("difference between mrr and ndcg", k=5)
for item in results:
    print(item.doc_id, item.score, item.metadata)
```

## With Auth

```python
retriever = MilvusRetriever(
    collection_name="docs",
    query_encoder=encode_query,
    uri="http://localhost:19530",
    token="username:password",
)
```

## Notes

- `query_encoder` must return a non-empty vector
- You can pass `search_filter` for server-side filtering
- Milvus `distance` is mapped to `score` using `distance_to_score` (identity by default)
