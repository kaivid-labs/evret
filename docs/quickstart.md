# Quickstart

This guide shows a small end-to-end evaluation with Evret.

## Install

```bash
pip install evret
```

Optional integrations:

```bash
pip install "evret[qdrant]"
pip install "evret[langchain]"
pip install "evret[semantic]"
pip install "evret[all]"
```

## 1. Create An Evaluation Dataset

Create `eval_data.json`:

```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "does a flight above 500 dollars need manager approval",
      "expected_answers": [
        "Flights above 500 dollars require manager approval before booking business travel."
      ]
    },
    {
      "query_id": "q2",
      "query_text": "what hotel reimbursement limit applies to business travel",
      "expected_answers": [
        "Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception."
      ]
    }
  ],
  "documents": [
    {
      "doc_id": "travel_policy_1",
      "text": "Employees must submit travel expenses within 30 days of trip completion."
    },
    {
      "doc_id": "travel_policy_2",
      "text": "Flights above 500 dollars require manager approval before booking business travel."
    },
    {
      "doc_id": "travel_policy_3",
      "text": "Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception."
    }
  ]
}
```

Use `expected_answers` when you want a judge to match retrieved text against gold text. Use `relevant_doc_ids` when your dataset already has exact ground truth document ids.

## 2. Define A Retriever

Evret evaluates any retriever that implements `BaseRetriever`.

```python
import re

from evret import RetrievalResult
from evret.retrievers import BaseRetriever


class KeywordRetriever(BaseRetriever):
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        query_tokens = set(re.findall(r"[a-z0-9]+", self._validate_query(query).lower()))
        results = []

        for document in self.documents:
            doc_tokens = set(re.findall(r"[a-z0-9]+", document.text.lower()))
            overlap = len(query_tokens & doc_tokens)
            if overlap == 0:
                continue
            results.append(
                RetrievalResult(
                    doc_id=document.doc_id,
                    score=overlap / max(len(query_tokens), 1),
                    metadata={"text": document.text},
                )
            )

        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:k]
```

In a real app, this can be a `QdrantRetriever`, `ChromaRetriever`, `WeaviateRetriever`, `MilvusRetriever`, or your own retriever.

## 3. Run Evaluation

```python
from evret import (
    AveragePrecision,
    EvaluationDataset,
    Evaluator,
    HitRate,
    MRR,
    NDCG,
    Precision,
    Recall,
    TokenOverlapJudge,
)

dataset = EvaluationDataset.from_json("eval_data.json")
retriever = KeywordRetriever(dataset.documents)

evaluator = Evaluator(
    retriever=retriever,
    metrics=[
        HitRate(k=2),
        Recall(k=2),
        Precision(k=2),
        MRR(k=2),
        NDCG(k=2),
        AveragePrecision(k=2),
    ],
    judge=TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6),
)

results = evaluator.evaluate(dataset)
print(results.summary())

results.to_json("results.json")
results.to_csv("results.csv")
```

`Evaluator` calls the retriever once per query using the largest `k` from your metric list. Then it computes each metric over the same retrieved results.

## 4. Read The Scores

Metric names include their cutoff:

```python
{
    "hit_rate@2": 1.0,
    "recall@2": 1.0,
    "precision@2": 0.5,
    "mrr@2": 1.0,
    "ndcg@2": 1.0,
    "average_precision@2": 1.0,
}
```

The exact numbers depend on your retriever and judge. Every built-in metric returns a score between `0.0` and `1.0`.

## Choose Metrics

| Goal | Metric |
| --- | --- |
| Check if any relevant result appears | `HitRate` |
| Check how much relevant content is recovered | `Recall` |
| Check how clean the returned context is | `Precision` |
| Check how early the first relevant hit appears | `MRR` |
| Check ranking quality across positions | `NDCG` |
| Check rank-aware precision over relevant hits | `AveragePrecision` |

## Choose A Judge

`Evaluator` defaults to `TokenOverlapJudge()`. Pass `judge=` when you want explicit matching behavior.

```python
from evret.judges import LLMJudge, SemanticJudge, TokenOverlapJudge

token_judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)
semantic_judge = SemanticJudge(threshold=0.75)
llm_judge = LLMJudge(provider="openai", model="gpt-4o-mini")
```

| Judge | Best For |
| --- | --- |
| `TokenOverlapJudge` | Fast local checks with no external dependency |
| `SemanticJudge` | Embedding similarity and paraphrase tolerance |
| `LLMJudge` | Complex semantic judgment with an LLM provider |

## Next Steps

- Read [Metrics Overview](metrics/index.md) for formulas.
- Read [Dataset Format](evaluation/dataset-format.md) for JSON and CSV fields.
- Read [Retriever Overview](retrievers/overview.md) to connect a vector database.
- Read [Judges](judges.md) for judge parameters.
