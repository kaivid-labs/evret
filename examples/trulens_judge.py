"""Evaluate a retriever with evret using TruLens feedback functions as the judge.
pip install "evret[trulens-openai]"  # or evret[trulens-litellm]
export OPENAI_API_KEY=sk-...
"""
from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import DocumentExample, QueryExample
from evret.judges import TruLensJudge
from evret.judges.base import JudgmentContext
from evret.metrics import NDCG, HitRate, MRR, Recall
from evret.retrievers import BaseRetriever, RetrievalResult

from dotenv import load_dotenv
load_dotenv()

class InMemoryRetriever(BaseRetriever):
    """Tiny retriever over a fixed document set for a self-contained example."""

    def __init__(self, documents: list[DocumentExample]) -> None:
        self._documents = documents

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        # Naive lexical overlap ranking, just enough to produce a ranked list.
        # The evaluator resolves each result's text from the dataset by doc_id.
        query_tokens = set(query.lower().split())
        scored = [
            (
                len(query_tokens & set(doc.text.lower().split())),
                RetrievalResult(doc_id=doc.doc_id, score=0.0),
            )
            for doc in self._documents
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [result for _, result in scored[:k]]


def main() -> None:
    documents = [
        DocumentExample(
            "doc_0",
            "The Python GIL is a mutex that allows only one thread to execute "
            "Python bytecode at a time, which simplifies memory management.",
        ),
        DocumentExample(
            "doc_1",
            "In HTTP, status code 404 indicates that the requested resource "
            "could not be found on the server.",
        ),
        DocumentExample(
            "doc_2",
            "Photosynthesis converts carbon dioxide and water into glucose and "
            "oxygen using energy captured from sunlight.",
        ),
        DocumentExample(
            "doc_3",
            "Mount Everest, on the border of Nepal and Tibet, rises about 8,849 "
            "meters above sea level, making it the tallest mountain on Earth.",
        ),
        DocumentExample(
            "doc_4",
            "TCP is a connection-oriented protocol that guarantees ordered, "
            "reliable delivery, whereas UDP is connectionless and best-effort.",
        ),
    ]

    dataset = EvaluationDataset(
        documents=documents,
        queries=[
            QueryExample(
                query_id="q1",
                query_text="What is the purpose of the Python GIL?",
                expected_answers=[
                    "The GIL lets only one thread run Python bytecode at a time."
                ],
            ),
            QueryExample(
                query_id="q2",
                query_text="Which HTTP status code means the resource was not found?",
                expected_answers=[
                    "HTTP 404 means the requested resource was not found."
                ],
            ),
            QueryExample(
                query_id="q3",
                query_text="How tall is Mount Everest?",
                expected_answers=[
                    "Mount Everest is roughly 8,849 meters above sea level."
                ],
            ),
            QueryExample(
                query_id="q4",
                query_text="How does TCP differ from UDP?",
                expected_answers=[
                    "TCP is reliable and connection-oriented; UDP is connectionless and best-effort."
                ],
            ),
        ],
    )

    judge = TruLensJudge(
        provider="openai",
        model="gpt-6-luna",
        metric="context_relevance",
        threshold=0.5,
    )

    score = judge.score(
        JudgmentContext(
            query="What captures sunlight to make glucose in plants?",
            expected_text="",  # ignored by TruLens feedback functions
            retrieved_text=documents[2].text,
        )
    )
    print(f"context_relevance score for the photosynthesis passage: {score:.3f}")

    results = Evaluator(
        retriever=InMemoryRetriever(documents),
        metrics=[HitRate(k=3), MRR(k=3), Recall(k=3), NDCG(k=3)],
        judge=judge,
    ).evaluate(dataset)
    print(results.summary())


if __name__ == "__main__":
    main()
