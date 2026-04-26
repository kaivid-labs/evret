# Evaluation Overview

Evret evaluates a retriever against a dataset of queries and gold supporting chunks.

## Evaluation Flow

1. Load dataset from JSON or CSV
2. Choose metric list
3. Evaluator asks retriever for top results
4. Metrics are computed per query
5. Final score is mean across queries
6. Export results to JSON or CSV

## Minimal Example

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR

retriever = my_retriever

dataset = EvaluationDataset.from_json("examples/eval_data.json")
evaluator = Evaluator(
    retriever=retriever,
    metrics=[HitRate(k=4), MRR(k=4)],
)

results = evaluator.evaluate(dataset)
print(results.summary())
```

## Output Object

`EvaluationResults` contains:

- `metric_scores`: dictionary of metric names and scores
- `query_count`: total evaluated queries
- `generated_at`: UTC timestamp

Export methods:

- `results.to_json("results.json")`
- `results.to_csv("results.csv")`

## Important Notes

- Evaluator validates that metric names are unique
- Evaluator validates dataset has at least one query
- Evaluator asks retriever using `max(k)` across selected metrics
- Mainline RAG evaluation stores gold supporting chunks in `relevant_docs`
- For top-4 context evaluation, set metrics with `k=4`
- Use `relevance_judge` only when your gold labels are paraphrases or metadata references instead of exact chunk text
