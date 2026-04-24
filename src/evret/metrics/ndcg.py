"""nDCG@K metric implementation."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

from evret.metrics.base import Metric


class NDCG(Metric):
    """Normalized Discounted Cumulative Gain at top-k."""

    metric_name = "ndcg"

    def score_query(
        self,
        retrieved_doc_ids: Sequence[str],
        relevant_doc_ids: Collection[str],
    ) -> float:
        relevance_lookup = self._relevance_lookup(relevant_doc_ids)
        top_k_ids = self.top_k(retrieved_doc_ids)
        dcg = self._dcg(top_k_ids, relevance_lookup)

        ideal_relevances = sorted(relevance_lookup.values(), reverse=True)[: self.k]
        idcg = self._dcg_from_relevances(ideal_relevances)

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _relevance_lookup(relevant_doc_ids: Collection[str]) -> dict[str, float]:
        lookup: dict[str, float] = {}
        for doc_id in relevant_doc_ids:
            lookup[doc_id] = 1.0
        return lookup

    def _dcg(self, ranked_doc_ids: Sequence[str], relevance_lookup: Mapping[str, float]) -> float:
        relevances = [relevance_lookup.get(doc_id, 0.0) for doc_id in ranked_doc_ids]
        return self._dcg_from_relevances(relevances)

    @staticmethod
    def _dcg_from_relevances(relevances: Sequence[float]) -> float:
        total = 0.0
        for rank, relevance in enumerate(relevances, start=1):
            total += relevance / math.log2(rank + 1)
        return total
