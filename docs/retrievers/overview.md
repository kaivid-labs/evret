# Retriever Overview

Evret uses one retriever contract across all vector backends.

## Base Contract

Every retriever follows `BaseRetriever`:

- `retrieve(query: str, k: int) -> list[RetrievalResult]`
- `batch_retrieve(queries: Sequence[str], k: int) -> list[list[RetrievalResult]]`

`RetrievalResult` has:

- `doc_id: str`
- `score: float`
- `metadata: dict`

## Supported Retrievers

| Retriever | Backend | Optional Dependency | Usage Guide |
| --- | --- | --- | --- |
| `QdrantRetriever` | Qdrant | `qdrant-client` | [Qdrant Usage](qdrant.md) |
| `ChromaRetriever` | ChromaDB | `chromadb` | [Chroma Usage](chroma.md) |
| `WeaviateRetriever` | Weaviate | `weaviate-client` | [Weaviate Usage](weaviate.md) |
| `MilvusRetriever` | Milvus | `pymilvus` | [Milvus Usage](milvus.md) |

## Common Evaluation Pattern

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR
from evret.retrievers import QdrantRetriever

def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = QdrantRetriever(
    collection_name="docs",
    query_encoder=encode_query,
    url="http://localhost:6333",
)

dataset = EvaluationDataset.from_json("eval_data.json")
evaluator = Evaluator(
    retriever=retriever,
    metrics=[HitRate(k=4), MRR(k=4)],
)
results = evaluator.evaluate(dataset)
print(results.summary())
```

## Why This Helps

- Same metric code works for every backend
- You can switch vector database with less rewrite
- Evaluation code does not depend on backend-specific response shapes

## Minimal Custom Retriever

```python
from evret.retrievers import BaseRetriever, RetrievalResult

class MyRetriever(BaseRetriever):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        q = self._validate_query(query)
        return [RetrievalResult(doc_id="doc_1", score=0.9, metadata={"query": q})]
```
