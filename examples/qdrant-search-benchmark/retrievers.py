from __future__ import annotations

from typing import Any

from qdrant_client import models

from common import (
    BM25,
    COLBERT,
    COLLECTION_NAME,
    DENSE,
    PREFETCH_LIMIT,
    SPLADE,
    embed_bm25_query,
    embed_colbert_query,
    embed_dense_query,
    embed_splade_query,
)
from evret.retrievers import BaseRetriever, RetrievalResult


def _to_results(points: list[Any]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for point in points:
        payload = point.payload or {}
        results.append(
            RetrievalResult(
                doc_id=str(payload.get("doc_id", point.id)),
                score=float(point.score),
                metadata=dict(payload),
            )
        )
    return results


def _dense_prefetch(query: str) -> models.Prefetch:
    return models.Prefetch(query=embed_dense_query(query), using=DENSE, limit=PREFETCH_LIMIT)


def _bm25_prefetch(query: str) -> models.Prefetch:
    return models.Prefetch(query=embed_bm25_query(query), using=BM25, limit=PREFETCH_LIMIT)


class DenseRetriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_dense_query(query),
            using=DENSE,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class Bm25Retriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_bm25_query(query),
            using=BM25,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class SpladeRetriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_splade_query(query),
            using=SPLADE,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class ColbertRetriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_colbert_query(query),
            using=COLBERT,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class HybridNoFusionRetriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[_bm25_prefetch(query)],
            query=embed_dense_query(query),
            using=DENSE,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class HybridFusionRetriever(BaseRetriever):
    def __init__(self, client: Any, fusion: models.Fusion) -> None:
        self.client = client
        self.fusion = fusion

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[_dense_prefetch(query), _bm25_prefetch(query)],
            query=models.FusionQuery(fusion=self.fusion),
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)


class HybridColbertRerankRetriever(BaseRetriever):
    def __init__(self, client: Any) -> None:
        self.client = client

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[_dense_prefetch(query), _bm25_prefetch(query)],
            query=embed_colbert_query(query),
            using=COLBERT,
            with_payload=True,
            limit=k,
        )
        return _to_results(response.points)
