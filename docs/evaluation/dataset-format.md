# Dataset Format

Evret supports JSON and CSV for evaluation datasets.

## JSON Format

Top level object has:

- `queries`: required list
- `documents`: optional list

### Query Item Fields

- `query_id` or `id`: string (required)
- `query_text` or `query`: string (required)
- `relevant_doc_ids`: list of strings (optional)
  - Use when you have pre-labeled document IDs as ground truth
  - For classic IR evaluation with exact doc ID matching
- `expected_answers`: list of strings (optional)
  - Use when you want a judge to determine relevance
  - Store gold supporting text snippets that judges match against retrieved docs
- `relevant_docs`: list of strings (deprecated, backward compatible)
  - Legacy field name, automatically mapped to `relevant_doc_ids`

**Note:** Provide either `relevant_doc_ids` OR `expected_answers`, not both

### Document Item Fields

- `doc_id`: string
- `text`: string
- `metadata`: object, optional

### JSON Example with Expected Answers (Judge-Based Evaluation)

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
      "doc_id": "travel_policy_2",
      "text": "Flights above 500 dollars require manager approval before booking business travel.",
      "metadata": {
        "source": "travel_policy.md",
        "section": "flight_approval"
      }
    },
    {
      "doc_id": "travel_policy_3",
      "text": "Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.",
      "metadata": {
        "source": "travel_policy.md",
        "section": "hotel_cap"
      }
    }
  ]
}
```

### JSON Example with Document IDs (Classic IR Evaluation)

```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "does a flight above 500 dollars need manager approval",
      "relevant_doc_ids": ["travel_policy_2"]
    },
    {
      "query_id": "q2",
      "query_text": "what hotel reimbursement limit applies to business travel",
      "relevant_doc_ids": ["travel_policy_3"]
    }
  ],
  "documents": [
    {
      "doc_id": "travel_policy_2",
      "text": "Flights above 500 dollars require manager approval before booking business travel.",
      "metadata": {
        "source": "travel_policy.md",
        "section": "flight_approval"
      }
    },
    {
      "doc_id": "travel_policy_3",
      "text": "Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.",
      "metadata": {
        "source": "travel_policy.md",
        "section": "hotel_cap"
      }
    }
  ]
}
```

## CSV Format

Required columns:

- `query_text` or `query`
- `relevant_docs`

Optional columns:

- `query_id` or `id`
- `relevant_doc_ids` (for classic IR evaluation)
- `expected_answers` (for judge-based evaluation)

**Note:** The old `relevant_docs` column is still supported for backward compatibility.

The relevance field (`relevant_doc_ids`, `expected_answers`, or legacy `relevant_docs`) can be:

- JSON list string like `"[\"Flights above 500 dollars require manager approval before booking business travel.\"]"`
- Comma separated values when the labels are short and unambiguous

### CSV Example with Expected Answers

```csv
query_id,query_text,expected_answers
q1,does a flight above 500 dollars need manager approval,"[""Flights above 500 dollars require manager approval before booking business travel.""]"
q2,what hotel reimbursement limit applies to business travel,"[""Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.""]"
```

### CSV Example with Document IDs

```csv
query_id,query_text,relevant_doc_ids
q1,does a flight above 500 dollars need manager approval,"[""travel_policy_2""]"
q2,what hotel reimbursement limit applies to business travel,"[""travel_policy_3""]"
```

## Loader Methods

```python
from evret import EvaluationDataset

dataset_json = EvaluationDataset.from_json("eval_data.json")
dataset_csv = EvaluationDataset.from_csv("eval_data.csv")
```
