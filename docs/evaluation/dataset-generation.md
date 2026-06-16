# Dataset Generation

Evret can generate evaluation datasets from source documents with a simple LLM-assisted flow:

1. Split source documents into retrieval-sized chunks.
2. Ask a user-selected LLM to generate query text and expected answers from each chunk.
3. Attach chunk-derived `expected_context` and `expected_doc_ids` in Evret code.
4. Convert the generated rows into `EvaluationDataset`.

## Basic Usage

```python
from evret import DatasetGenerator, SourceDocument

generator = DatasetGenerator.from_provider(
    provider="openai",
    model="gpt-5.4-nano",
    examples_per_chunk=5,
)

generated = generator.generate(
    [
        SourceDocument(
            source="travel_policy.md",
            text="Flights above 500 dollars require manager approval before booking.",
        )
    ]
)

dataset = generated.to_evaluation_dataset()
```

## Evaluating Generated Datasets

Generated datasets include `expected_doc_ids`, so `Evaluator` can score retrieved document IDs directly without calling a judge.

If your evaluation workflow uses the generated `expected_answers` for text-based matching, use `LLMJudge` instead of `TokenOverlapJudge`. Generated answers can be paraphrased or compressed from the source chunk, and token overlap is too brittle for that judgment.

```python
from evret import Evaluator, HitRate, Recall
from evret.judges import LLMJudge

judge = LLMJudge(provider="openai", model="gpt-5.4-nano")

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=5), Recall(k=5)],
    judge=judge,
)

results = evaluator.evaluate(dataset)
```

## Generated Categories

The generator uses one prompt per chunk and asks for diverse categories:

| Category | Description |
| --- | --- |
| `direct_fact` | Direct factual question answerable from the chunk. |
| `paraphrase` | Same fact asked with different wording. |
| `keyword_search` | Short search-style query. |
| `specific_detail` | Query about a condition, value, exception, date, field, or parameter. |
| `broad_summary` | Broader question answerable from the chunk as a whole. |
| `out_of_context` | Plausible domain question not answered by the chunk. |

The LLM is not asked to generate `expected_context` or document IDs. For answerable rows, Evret stores the source chunk text as `expected_context` and the chunk UUID as `expected_doc_ids`. For `out_of_context`, the generator requires an empty expected answer; those rows have `expected_answers=[]`, `expected_doc_ids=[]`, and `expected_context=""`.

## Rich Output

`GeneratedDataset.to_dict()` preserves generation metadata:

```python
{
    "query_id": "q1",
    "query_text": "When does a flight require manager approval?",
    "expected_answers": ["Flights above 500 dollars require manager approval."],
    "category": "specific_detail",
    "expected_context": "Flights above 500 dollars require manager approval before booking.",
    "expected_doc_ids": ["0182f1e8-2f9a-5f7b-a23d-65ad3f7c7f7b"],
}
```

Use `to_evaluation_dataset()` when you want the standard Evret runtime format.
