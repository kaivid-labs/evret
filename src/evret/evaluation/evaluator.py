"""Evaluation orchestrator for retriever metrics."""

from __future__ import annotations

from collections.abc import Sequence

from evret.errors import EvretValidationError
from evret.evaluation.dataset import EvaluationDataset
from evret.evaluation.results import EvaluationResults
from evret.metrics import Metric
from evret.retrievers import BaseRetriever
from evret.utils import find_duplicates


class Evaluator:
    """Run a list of metrics over a retriever and dataset."""

    def __init__(self, retriever: BaseRetriever, metrics: Sequence[Metric]) -> None:
        if not metrics:
            raise EvretValidationError("metrics must contain at least one metric")

        metric_names = [metric.name for metric in metrics]
        duplicate_metric_names = sorted(find_duplicates(metric_names))
        if duplicate_metric_names:
            duplicates_text = ", ".join(duplicate_metric_names)
            raise EvretValidationError(
                f"duplicate metric names are not allowed: {duplicates_text}"
            )

        self.retriever = retriever
        self.metrics = list(metrics)

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationResults:
        if not dataset.queries:
            raise EvretValidationError("dataset must contain at least one query")

        max_k = max(metric.k for metric in self.metrics)
        retrieved_results = self.retriever.batch_retrieve(
            queries=[query.query_text for query in dataset.queries],
            k=max_k,
        )
        if len(retrieved_results) != len(dataset.queries):
            raise EvretValidationError(
                "retriever returned a different number of query results than input queries"
            )
        retrieved_ids = [
            [result.doc_id for result in query_results]
            for query_results in retrieved_results
        ]
        relevant_sets = [set(query.relevant_doc_ids) for query in dataset.queries]

        metric_scores: dict[str, float] = {}
        for metric in self.metrics:
            metric_scores[metric.name] = metric.score(
                retrieved_by_query=retrieved_ids,
                relevant_by_query=relevant_sets,
            )

        return EvaluationResults(
            metric_scores=metric_scores,
            query_count=len(dataset.queries),
        )
