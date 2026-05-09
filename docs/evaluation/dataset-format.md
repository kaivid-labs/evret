# Dataset Format

Evret supports JSON and CSV for evaluation datasets.

## JSON Format

Top level object has:

- `queries`: required list
- `documents`: optional list

### Query Item Fields

- `query_id` or `id`: string (required)
- `query_text` or `query`: string (required)
- `expected_answers`: list of strings (optional)
  - Use gold supporting text snippets or expected facts
  - Judges match these strings against retrieved content

### Document Item Fields

- `doc_id`: string
- `text`: string
- `metadata`: object, optional

### JSON Example

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

## CSV Format

Required columns:

- `query_text` or `query`

Optional columns:

- `query_id` or `id`
- `expected_answers`

The `expected_answers` field can be:

- JSON list string like `"[\"Flights above 500 dollars require manager approval before booking business travel.\"]"`
- Comma separated values when the answers are short and unambiguous

### CSV Example

```csv
query_id,query_text,expected_answers
q1,does a flight above 500 dollars need manager approval,"[""Flights above 500 dollars require manager approval before booking business travel.""]"
q2,what hotel reimbursement limit applies to business travel,"[""Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.""]"
```

## Loader Methods

```python
from evret import EvaluationDataset

dataset_json = EvaluationDataset.from_json("eval_data.json")
dataset_csv = EvaluationDataset.from_csv("eval_data.csv")
```
