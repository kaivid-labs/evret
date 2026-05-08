# Evret Documentation

Evret is a lightweight Python framework for evaluating retrievers in information retrieval and search systems.

It provides standard Information Retrieval metrics, a judge system for text-based relevance matching, and retriever adapters for common vector databases.

## What Evret Does

Evret turns retrieved results into metric scores:

1. Load an evaluation dataset.
2. Ask your retriever for top-`k` results.
3. Match retrieved content against gold labels with a judge.
4. Compute IR metrics over the matched relevant ids.
5. Export a summary to JSON or CSV.

The evaluator supports two relevance styles:

- `relevant_doc_ids` for classic IR evaluation with known document ids
- `expected_answers` for judge-based information retrieval evaluation with gold text snippets

## Key Features

- **Standard IR metrics**: Hit Rate, Recall, Precision, MRR, nDCG, ERR, RBP, and Average Precision
- **Judge-based matching**: token overlap by default, with optional semantic and LLM judges
- **Vector database support**: Qdrant, Chroma, Weaviate, and Milvus retrievers
- **Framework integrations**: LangChain and LlamaIndex adapters
- **Simple result exports**: `summary()`, `to_json()`, and `to_csv()`

## Quick Example

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR, NDCG, TokenOverlapJudge

dataset = EvaluationDataset.from_json("eval_data.json")

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), MRR(k=4), NDCG(k=4)],
    judge=TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6),
)

results = evaluator.evaluate(dataset)
print(results.summary())
```

## Metrics

| Metric | What It Answers | Best Use |
| --- | --- | --- |
| Hit Rate@k | Did at least one relevant result appear? | Basic retrieval sanity checks |
| Recall@k | How much relevant content was found? | Coverage and completeness |
| Precision@k | How clean are the top-k results? | LLM context quality |
| MRR@k | How early is the first relevant result? | Single-answer QA |
| nDCG@k | Are relevant results ranked high? | Ranking and reranker quality |
| ERR@k | How satisfied is a user likely to be? | Graded relevance and cascade browsing |
| RBP@k | How good are results for a patience level? | Tunable user persistence |
| Average Precision@k | How good is precision across relevant ranks? | Benchmark-style ranking comparison |

See [Metrics Overview](metrics/index.md) for formulas and examples.

## Judges

Judges decide whether retrieved text matches the expected text in your dataset. `Evaluator` uses `TokenOverlapJudge()` by default.

| Judge | Best For | Dependency |
| --- | --- | --- |
| `TokenOverlapJudge` | Fast local checks | none |
| `SemanticJudge` | Embedding similarity | `evret[semantic]` |
| `LLMJudge` | Complex paraphrase matching | LLM provider extra |

See [Judges](judges.md) for parameters and provider defaults.

## Installation

```bash
pip install evret
```

Optional extras:

```bash
pip install "evret[qdrant]"
pip install "evret[langchain]"
pip install "evret[semantic]"
pip install "evret[all]"
```

## Docs Map

- [Quickstart](quickstart.md): end-to-end first evaluation
- [Metrics](metrics/index.md): metric formulas and examples
- [Evaluation](evaluation/overview.md): evaluator flow and result object
- [Retrievers](retrievers/overview.md): retriever contract and backend guides
- [Integrations](integrations/overview.md): framework adapters
- [API Reference](api/package.md): generated API docs
