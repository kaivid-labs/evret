from __future__ import annotations

from qdrant_client import models

from common import (
    BM25,
    COLBERT,
    COLLECTION_NAME,
    DENSE,
    SPLADE,
    embed_bm25_docs,
    embed_colbert_docs,
    embed_dense_docs,
    embed_splade_docs,
    get_client,
    load_raw_dataset,
)

BATCH = 4


def main() -> None:
    documents = load_raw_dataset()["documents"]
    texts = [doc["text"] for doc in documents]
    print(f"Embedding {len(texts)} chunks (dense + bm25 + splade + colbert)...")

    dense = embed_dense_docs(texts)
    bm25 = embed_bm25_docs(texts)
    splade = embed_splade_docs(texts)
    colbert = embed_colbert_docs(texts)

    client = get_client()
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE: models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE),
            COLBERT: models.VectorParams(
                size=len(colbert[0][0]),
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={
            BM25: models.SparseVectorParams(modifier=models.Modifier.IDF),
            SPLADE: models.SparseVectorParams(),
        },
    )

    points = [
        models.PointStruct(
            id=index,
            vector={
                DENSE: dense[index],
                COLBERT: colbert[index],
                BM25: bm25[index],
                SPLADE: splade[index],
            },
            payload={
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "source": doc.get("metadata", {}).get("source", ""),
            },
        )
        for index, doc in enumerate(documents)
    ]

    for start in range(0, len(points), BATCH):
        client.upsert(collection_name=COLLECTION_NAME, points=points[start : start + BATCH])

    print(f"Indexed {len(points)} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
