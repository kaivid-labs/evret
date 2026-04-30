"""Corpus-backed retrieval evaluation example with Judge system."""

from __future__ import annotations

import re
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
from evret.judges import TokenOverlapJudge
from evret.retrievers import BaseRetriever

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class KeywordCorpusRetriever(BaseRetriever):
    """Retrieve top chunks from a document corpus using lexical overlap."""

    def __init__(self, dataset: EvaluationDataset) -> None:
        self.documents = dataset.documents

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        query_tokens = set(self._tokenize(query))
        scored_results: list[RetrievalResult] = []

        for document in self.documents:
            document_tokens = set(self._tokenize(document.text))
            overlap = len(query_tokens.intersection(document_tokens))
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            metadata = {"text": document.text, **document.metadata}
            scored_results.append(
                RetrievalResult(doc_id=document.doc_id, score=score, metadata=metadata)
            )

        scored_results.sort(key=lambda row: (-row.score, row.doc_id))
        return scored_results[:k]

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        return TOKEN_PATTERN.findall(value.lower())


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    dataset = EvaluationDataset.from_json(base_dir / "eval_data.json")
    retriever = KeywordCorpusRetriever(dataset)
    metrics = [
        HitRate(k=4),
        Recall(k=4),
        Precision(k=4),
        MRR(k=4),
        NDCG(k=4),
        AveragePrecision(k=4),
    ]

    # Use TokenOverlapJudge for text-based relevance matching
    judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)

    print(f"Using judge: {judge.name}")
    print("-" * 50)

    results = Evaluator(
        retriever=retriever,
        metrics=metrics,
        judge=judge,  # Text-based matching instead of ID matching
    ).evaluate(dataset)

    results.to_json(base_dir / "results.json")
    results.to_csv(base_dir / "results.csv")

    print("Summary:", results.summary())
    print("JSON output:", base_dir / "results.json")
    print("CSV output:", base_dir / "results.csv")
    print("-" * 50)
    print("✓ Evaluation complete with text-based judge!")


if __name__ == "__main__":
    main()
