import re

import pytest

from evret.evaluation import DocumentExample, EvaluationDataset, Evaluator, QueryExample
from evret.metrics import HitRate, Recall
from evret.retrievers import BaseRetriever, RetrievalResult

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class CorpusRetriever(BaseRetriever):
    def __init__(self, documents: list[DocumentExample]) -> None:
        self.documents = documents
        self.calls: list[tuple[list[str], int]] = []

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        query_tokens = set(self._tokenize(query))
        results: list[RetrievalResult] = []

        for document in self.documents:
            document_tokens = set(self._tokenize(document.text))
            overlap = len(query_tokens.intersection(document_tokens))
            if overlap == 0:
                continue
            results.append(
                RetrievalResult(
                    doc_id=document.doc_id,
                    score=overlap / max(len(query_tokens), 1),
                    metadata={"text": document.text, **document.metadata},
                )
            )

        results.sort(key=lambda row: (-row.score, row.doc_id))
        return results[:k]

    def batch_retrieve(self, queries: list[str], k: int) -> list[list[RetrievalResult]]:
        self.calls.append((queries, k))
        return [self.retrieve(query, k) for query in queries]

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return TOKEN_PATTERN.findall(value.lower())


def build_dataset() -> EvaluationDataset:
    documents = [
        DocumentExample(
            doc_id="travel_policy_1",
            text="Employees must submit travel expenses within 30 days of trip completion.",
            metadata={"source": "travel_policy.md", "section": "expense_deadline"},
        ),
        DocumentExample(
            doc_id="travel_policy_2",
            text="Flights above 500 dollars require manager approval before booking business travel.",
            metadata={"source": "travel_policy.md", "section": "flight_approval"},
        ),
        DocumentExample(
            doc_id="travel_policy_3",
            text="Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.",
            metadata={"source": "travel_policy.md", "section": "hotel_cap"},
        ),
        DocumentExample(
            doc_id="support_runbook_1",
            text="Urgent support tickets must receive a first response within 1 hour.",
            metadata={"source": "support_runbook.md", "section": "urgent_sla"},
        ),
        DocumentExample(
            doc_id="security_handbook_1",
            text="Production API keys must be rotated every 90 days.",
            metadata={"source": "security_handbook.md", "section": "api_key_rotation"},
        ),
    ]
    queries = [
        QueryExample(
            query_id="q1",
            query_text="When do employees need to submit travel expenses?",
            relevant_docs=[documents[0].text],
        ),
        QueryExample(
            query_id="q2",
            query_text="Does a flight above 500 dollars need manager approval?",
            relevant_docs=[documents[1].text],
        ),
        QueryExample(
            query_id="q3",
            query_text="What approval and hotel reimbursement limits apply to business travel?",
            relevant_docs=[documents[1].text, documents[2].text],
        ),
    ]
    return EvaluationDataset(queries=queries, documents=documents)


def test_evaluator_runs_metrics_and_returns_summary() -> None:
    dataset = build_dataset()
    retriever = CorpusRetriever(dataset.documents)
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=1), Recall(k=1)])

    results = evaluator.evaluate(dataset)

    assert retriever.calls == [
        (
            [
                "When do employees need to submit travel expenses?",
                "Does a flight above 500 dollars need manager approval?",
                "What approval and hotel reimbursement limits apply to business travel?",
            ],
            1,
        )
    ]
    assert results.summary() == {"hit_rate@1": 1.0, "recall@1": pytest.approx(5 / 6)}
    assert results.query_count == 3


def test_evaluator_requires_non_empty_metric_list() -> None:
    dataset = build_dataset()
    retriever = CorpusRetriever(dataset.documents)

    with pytest.raises(ValueError, match="at least one metric"):
        Evaluator(retriever=retriever, metrics=[])


def test_evaluator_requires_non_empty_dataset() -> None:
    dataset = build_dataset()
    retriever = CorpusRetriever(dataset.documents)
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=1)])

    with pytest.raises(ValueError, match="at least one query"):
        evaluator.evaluate(EvaluationDataset(queries=[]))


def test_evaluator_rejects_duplicate_metric_names() -> None:
    dataset = build_dataset()
    retriever = CorpusRetriever(dataset.documents)

    with pytest.raises(ValueError, match="duplicate metric names"):
        Evaluator(retriever=retriever, metrics=[HitRate(k=1), HitRate(k=1)])


def test_evaluator_supports_id_based_relevance_labels() -> None:
    dataset = build_dataset()
    dataset.queries = [
        QueryExample(
            query_id="q1",
            query_text="When do employees need to submit travel expenses?",
            relevant_docs=["travel_policy_1"],
        )
    ]
    retriever = CorpusRetriever(dataset.documents)
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=1)])

    results = evaluator.evaluate(dataset)

    assert results.summary() == {"hit_rate@1": 1.0}


def test_evaluator_scores_top_4_contexts_against_gold_chunks() -> None:
    dataset = build_dataset()
    dataset.queries = [
        QueryExample(
            query_id="q4",
            query_text="What approval and hotel reimbursement limits apply to business travel?",
            relevant_docs=[dataset.documents[1].text, dataset.documents[2].text],
        )
    ]
    retriever = CorpusRetriever(dataset.documents)
    evaluator = Evaluator(retriever=retriever, metrics=[HitRate(k=4), Recall(k=4)])

    results = evaluator.evaluate(dataset)

    assert results.summary() == {"hit_rate@4": 1.0, "recall@4": 1.0}


def test_evaluator_uses_custom_relevance_judge() -> None:
    from evret.judges.base import Judge, JudgmentContext

    class ParaphraseRetriever(BaseRetriever):
        def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
            return [
                RetrievalResult(
                    doc_id="x",
                    score=1.0,
                    metadata={"text": "Travel expenses have to be filed no later than 30 days after the trip."},
                )
            ][:k]

    class ParaphraseJudge(Judge):
        """Custom judge that matches on '30 days' phrase."""

        @property
        def name(self) -> str:
            return "paraphrase_judge"

        def judge(self, context: JudgmentContext) -> bool:
            return "30 days" in context.expected_text and "30 days" in context.retrieved_text

    dataset = EvaluationDataset(
        queries=[
            QueryExample(
                query_id="q1",
                query_text="When do I file travel expenses?",
                relevant_docs=["Employees must submit travel expenses within 30 days of trip completion."],
            )
        ]
    )
    evaluator = Evaluator(
        retriever=ParaphraseRetriever(),
        metrics=[HitRate(k=1)],
        judge=ParaphraseJudge(),
    )

    results = evaluator.evaluate(dataset)

    assert results.summary() == {"hit_rate@1": 1.0}
