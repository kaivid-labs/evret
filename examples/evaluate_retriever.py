"""Minimal evaluation example using Evret."""

from __future__ import annotations

from pathlib import Path

from evret import (
    AveragePrecision,
    EvaluationDataset,
    Evaluator,
    HitRate,
    MRR,
    NDCG,
    Precision,
    Recall,
    RetrievalResult,
)
from evret.retrievers import BaseRetriever


class DemoRetriever(BaseRetriever):
    """Simple deterministic retriever for example usage."""

    def __init__(self) -> None:
        self.lookup: dict[str, list[RetrievalResult]] = {
            "what is retrieval augmented generation": [
                RetrievalResult(
                    doc_id="doc_1",
                    score=0.95,
                    metadata={"document": "RAG combines retrieval with generation."},
                ),
                RetrievalResult(doc_id="doc_3", score=0.42),
                RetrievalResult(doc_id="doc_2", score=0.33),
            ],
            "how does vector search work": [
                RetrievalResult(doc_id="doc_3", score=0.91),
                RetrievalResult(doc_id="doc_2", score=0.35),
                RetrievalResult(doc_id="doc_1", score=0.2),
            ],
        }

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        return self.lookup.get(query, [])[:k]


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset = EvaluationDataset.from_json(base_dir / "eval_data.json")
    retriever = DemoRetriever()
    metrics = [
        HitRate(k=2),
        Recall(k=3),
        Precision(k=2),
        MRR(k=3),
        NDCG(k=3),
        AveragePrecision(k=3),
    ]

    results = Evaluator(retriever=retriever, metrics=metrics).evaluate(dataset)
    results.to_json(base_dir / "results.json")
    results.to_csv(base_dir / "results.csv")

    print("Summary:", results.summary())
    print("JSON output:", base_dir / "results.json")
    print("CSV output:", base_dir / "results.csv")


if __name__ == "__main__":
    main()
