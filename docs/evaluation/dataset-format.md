# Dataset Format

Evret supports JSON and CSV for evaluation datasets.

## JSON Format

Top level object has:

- `queries`: required list
- `documents`: optional list

### Query Item Fields

- `query_id` or `id`: string
- `query_text` or `query`: string
- `relevant_docs` or `relevant_doc_ids`: list of strings
  - In real RAG evaluation, store the gold supporting chunk text here
  - Use ids or fuzzy labels only if you also control how matching is judged

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
      "relevant_docs": [
        "Flights above 500 dollars require manager approval before booking business travel."
      ]
    },
    {
      "query_id": "q2",
      "query_text": "what hotel reimbursement limit applies to business travel",
      "relevant_docs": [
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
- `relevant_docs`

Optional columns:

- `query_id` or `id`

`relevant_docs` can be:

- JSON list string like `"[\"Flights above 500 dollars require manager approval before booking business travel.\"]"`
- Comma separated values when the labels are short and unambiguous

### CSV Example

```csv
query_id,query_text,relevant_docs
q1,does a flight above 500 dollars need manager approval,"[""Flights above 500 dollars require manager approval before booking business travel.""]"
q2,what hotel reimbursement limit applies to business travel,"[""Hotel reimbursement is capped at 180 dollars per night unless finance approves an exception.""]"
```

## Loader Methods

```python
from evret import EvaluationDataset

dataset_json = EvaluationDataset.from_json("eval_data.json")
dataset_csv = EvaluationDataset.from_csv("eval_data.csv")
```
