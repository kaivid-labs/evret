# Chroma Retriever

Use `ChromaRetriever` to evaluate ChromaDB collections.

## Install

```bash
pip install "evret[chroma]"
```

## Basic Usage

```python
import chromadb
from evret.retrievers import ChromaRetriever

client = chromadb.PersistentClient(path="./chroma-data")

retriever = ChromaRetriever(
    collection_name="docs",
    client=client,
)

results = retriever.retrieve("ranking metrics for rag", k=5)
for item in results:
    print(item.doc_id, item.score, item.metadata.get("document"))
```

By default, Chroma text query mode is used (`query_texts`).

## Using a Query Encoder

```python
def encode_query(query: str) -> list[float]:
    return embedding_model.embed_query(query)

retriever = ChromaRetriever(
    collection_name="docs",
    client=client,
    query_encoder=encode_query,
)
```

## Notes

- You can pass a Chroma collection directly as `client`
- If `query_encoder` is set, Evret uses `query_embeddings`
- Distance values are converted to scores with `1 / (1 + distance)` by default
