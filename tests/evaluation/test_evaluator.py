import pytest

from evret.evaluation import EvaluationDataset, Evaluator, QueryExample
from evret.metrics import HitRate, Recall
from evret.retrievers import BaseRetriever, RetrievalResult


class DummyRetriever(BaseRetriever):
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        mapping = {
            "alpha query": [
                RetrievalResult(doc_id="doc_1", score=0.9),
                RetrievalResult(doc_id="doc_3", score=0.2),
            ],
            "beta query": [
                RetrievalResult(doc_id="doc_9", score=0.8),
                RetrievalResult(doc_id="doc_2", score=0.5),
            ],
        }
        return mapping[query][:k]

    def batch_retrieve(self, queries: list[str], k: int) -> list[list[RetrievalResult]]:
        self.calls.append((queries, k))
        return [self.retrieve(query, k) for query in queries]


def test_evaluator_runs_metrics_and_returns_summary() -> None:
    dataset = EvaluationDataset(
        queries=[
            QueryExample(
                query_id="q1",
                query_text="alpha query",
                relevant_doc_ids=["doc_1", "doc_2"],
            ),
            QueryExample(
                query_id="q2",
                query_text="beta query",
                relevant_doc_ids=["doc_2"],
            ),
        ]
    )
    retriever = DummyRetriever()
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=1), Recall(k=2)])

    results = evaluator.evaluate(dataset)

    assert retriever.calls == [(["alpha query", "beta query"], 2)]
    assert results.summary() == {"hit_rate@1": 0.5, "recall@2": 0.75}
    assert results.query_count == 2


def test_evaluator_requires_non_empty_metric_list() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="at least one metric"):
        Evaluator(retriever=retriever, metrics=[])


def test_evaluator_requires_non_empty_dataset() -> None:
    retriever = DummyRetriever()
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=1)])

    with pytest.raises(ValueError, match="at least one query"):
        evaluator.evaluate(EvaluationDataset(queries=[]))


def test_evaluator_rejects_duplicate_metric_names() -> None:
    retriever = DummyRetriever()

    with pytest.raises(ValueError, match="duplicate metric names"):
        Evaluator(retriever=retriever, metrics=[HitRate(k=1), HitRate(k=1)])
