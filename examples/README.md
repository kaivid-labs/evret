# Evret Examples

Practical examples for evaluating RAG systems.

## Files

### `create_eval_dataset.py` - Create Your Own Dataset

**Use when:** You don't have an eval dataset

Two functions:
1. `create_dataset_manually()` - Write queries and expected answers
2. `save_dataset()` - Export to JSON

```bash
python create_eval_dataset.py
```

Creates `my_eval_dataset.json` with queries, expected answers, and documents.

### `demo_qdrant.py` - Full Evaluation Pipeline

**Use when:** You have a dataset and retriever

Shows:
- Index documents in Qdrant
- Run evaluation with metrics
- Export results

```bash
python demo_qdrant.py
```

## Quick Start

```bash
# 1. Create dataset
python create_eval_dataset.py

# 2. Edit my_eval_dataset.json - add your queries and expected answers

# 3. Run evaluation
python demo_qdrant.py
```

## Dataset Format

```json
{
  "queries": [{
    "query_id": "q1",
    "query_text": "Your question",
    "relevant_docs": ["expected answer text"]
  }],
  "documents": [{
    "doc_id": "doc1",
    "text": "Document content",
    "metadata": {}
  }]
}
```

## Understanding relevant_docs (Expected Answers)

**DON'T** manually label which document IDs are relevant.

**DO** provide the expected answer text you want to find in retrieved documents.

### How It Works

1. **You provide expected answer text** (not document IDs)
   ```python
   QueryExample(
       query_id="q1",
       query_text="How to install Python packages?",
       relevant_docs=["pip install package-name"]  # Expected answer text
   )
   ```

2. **The judge automatically compares** retrieved docs against your expected answers
   - If retrieved doc contains "pip install", judge marks it as relevant
   - If retrieved doc doesn't match, judge marks it as irrelevant

3. **Metrics are calculated** based on judge's relevance decisions
   - Precision: How many retrieved docs matched your expected answers?
   - Recall: Did the retriever find all expected answers?
   - MRR: How quickly did it find the first match?

### Example

```python
# Your documents
documents = [
    DocumentExample(doc_id="doc1", text="Use pip install to add packages"),
    DocumentExample(doc_id="doc2", text="Use venv for virtual environments"),
]

# Your query with expected answer
queries = [
    QueryExample(
        query_id="q1",
        query_text="How do I install packages?",
        relevant_docs=["pip install"]  # Just the answer text!
    )
]

# During evaluation:
# - Retriever returns doc1
# - Judge compares doc1 text ("Use pip install...") with expected ("pip install")
# - Judge says: MATCH! ✓
# - Metrics calculated: Precision@1 = 1.0
```

The framework handles the matching automatically - you just provide what answers you expect!
