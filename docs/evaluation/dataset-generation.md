# Dataset Generation

Evret can generate evaluation datasets from source documents with a simple LLM-assisted flow:

1. Split source documents into retrieval-sized chunks.
2. Ask a user-selected LLM to generate diverse examples from each chunk.
3. Convert the generated rows into `EvaluationDataset`.

## Basic Usage

```python
from evret import DatasetGenerator, SourceDocument

generator = DatasetGenerator.from_provider(
    provider="openai",
    model="gpt-4o-mini",
    examples_per_chunk=6,
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

For `out_of_context`, the generator requires an empty expected answer and empty expected context. When converted to `EvaluationDataset`, those rows have `expected_answers=[]`.

## Chunking Defaults

`chunk_documents()` uses structure-aware chunking:

- normal prose target: 250-450 tokens
- maximum chunk size: 700 tokens
- overlap: 40-80 tokens
- minimum useful chunk size: 80 tokens

Markdown headings are preserved as `heading_path` metadata, and every chunk receives a stable `doc_id`.

## Rich Output

`GeneratedDataset.to_dict()` preserves generation metadata:

```python
{
    "query_id": "q1",
    "query_text": "When does a flight require manager approval?",
    "expected_answers": ["Flights above 500 dollars require manager approval."],
    "category": "specific_detail",
    "expected_context": "Flights above 500 dollars require manager approval.",
    "source_chunk_id": "travel_policy_md_1",
}
```

Use `to_evaluation_dataset()` when you want the standard Evret runtime format.
