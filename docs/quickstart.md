# Evret Quickstart Guide

Get started with Evret in 5 minutes! This guide covers basic usage with the Judge system.

---

## Installation

```bash
# Core package (includes TokenOverlapJudge)
pip install evret

# With semantic judge
pip install evret[semantic]

# With LLM judges
pip install evret[llm-openai]      # OpenAI
pip install evret[llm-anthropic]   # Anthropic
pip install evret[llm-google]      # Google Gen AI

# Everything
pip install evret[all]
```

---

## Basic Evaluation

### 1. Prepare Your Dataset

Create `eval_data.json` with text-based relevance labels:

```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "What is RAG?",
      "relevant_docs": [
        "RAG combines retrieval with generation for better accuracy",
        "Retrieval-augmented generation improves LLM responses"
      ]
    },
    {
      "query_id": "q2",
      "query_text": "How do vector databases work?",
      "relevant_docs": [
        "Vector databases store embeddings as high-dimensional vectors",
        "They use similarity search to find nearest neighbors"
      ]
    }
  ]
}
```

### 2. Run Evaluation

```python
from evret import EvaluationDataset, Evaluator, HitRate, Recall, Precision, MRR
from evret.judges import TokenOverlapJudge

# Load dataset
dataset = EvaluationDataset.from_json("eval_data.json")

# Create evaluator with TokenOverlapJudge (default)
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), Recall(k=4), Precision(k=4), MRR(k=4)]
)

# Evaluate
results = evaluator.evaluate(dataset)

# View results
print(results.summary())
# {'hit_rate@4': 0.85, 'recall@4': 0.75, 'precision@4': 0.5, 'mrr@4': 0.92}

# Export
results.to_json("results.json")
results.to_csv("results.csv")
```

---

## Using Different Judges

### TokenOverlapJudge (Default)

Fast keyword-based matching, no external dependencies:

```python
from evret.judges import TokenOverlapJudge

judge = TokenOverlapJudge(
    min_tokens=2,        # Minimum shared tokens
    overlap_ratio=0.6,   # Minimum 60% overlap
    query_boost=True     # Allow query tokens to relax threshold
)

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[Recall(k=4)],
    judge=judge
)
```

**Best for:**
- Quick evaluation
- Keyword-based relevance
- No API costs

### SemanticJudge

Embedding-based semantic similarity:

```python
from evret.judges import SemanticJudge

judge = SemanticJudge(
    model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.75,  # Cosine similarity threshold
    device="cpu"     # or "cuda"
)

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[Recall(k=4)],
    judge=judge
)
```

**Best for:**
- Semantic similarity matching
- No API costs
- Better accuracy than token overlap

### LLMJudge

LLM-powered semantic judgment (highest accuracy):

```python
from evret.judges import LLMJudge

# OpenAI
judge = LLMJudge(
    provider="openai",
    model="gpt-4o-mini",  # or gpt-4o
    api_key="sk-..."      # or set OPENAI_API_KEY env var
)

# Anthropic
judge = LLMJudge(
    provider="anthropic",
    model="claude-3-5-haiku-20241022",
    api_key="..."  # or set ANTHROPIC_API_KEY env var
)

# Google Gen AI
judge = LLMJudge(
    provider="google",
    model="gemini-2.5-flash",
    api_key="..."  # or set GEMINI_API_KEY env var
)

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[Recall(k=4)],
    judge=judge
)
```

**Best for:**
- Maximum accuracy
- Complex reasoning
- Paraphrase detection

---

## Comparison Table

| Judge | Speed | Accuracy | Cost | Dependencies |
|-------|-------|----------|------|--------------|
| **TokenOverlapJudge** | ⚡️ Fastest | Good | Free | None |
| **SemanticJudge** | 🔄 Medium | Better | Free | sentence-transformers |
| **LLMJudge** | 🐢 Slowest | Best | $$$ | openai/anthropic/google-genai |

---

## Custom Judge

Implement your own matching logic:

```python
from evret.judges.base import Judge, JudgmentContext

class ExactMatchJudge(Judge):
    """Match only if texts are exactly the same (case-insensitive)."""

    @property
    def name(self) -> str:
        return "exact_match"

    def judge(self, context: JudgmentContext) -> bool:
        return context.expected_text.lower() == context.retrieved_text.lower()

# Use it
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[Recall(k=4)],
    judge=ExactMatchJudge()
)
```

---

## Complete Example

```python
from evret import (
    EvaluationDataset,
    Evaluator,
    HitRate,
    Recall,
    Precision,
    MRR,
    NDCG,
)
from evret.judges import TokenOverlapJudge

# 1. Load dataset
dataset = EvaluationDataset.from_json("eval_data.json")

# 2. Create judge
judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)

# 3. Define metrics
metrics = [
    HitRate(k=4),
    Recall(k=4),
    Precision(k=4),
    MRR(k=4),
    NDCG(k=4),
]

# 4. Create evaluator
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=metrics,
    judge=judge
)

# 5. Evaluate
results = evaluator.evaluate(dataset)

# 6. Results
print(f"Evaluated {results.query_count} queries")
print(f"Results: {results.summary()}")

results.to_json("results.json")
results.to_csv("results.csv")
```

---

## Next Steps

- Read [Architecture Guide](architecture.md) for design details
- See [examples/](../examples/) for complete working code
- Check [API Reference](api_reference.md) for detailed documentation

---

## Tips

### Choosing the Right Judge

1. **Start with TokenOverlapJudge** - Fast and good enough for most cases
2. **Upgrade to SemanticJudge** - If you need better semantic matching
3. **Use LLMJudge** - For final production evaluation or when accuracy is critical

### Performance Optimization

- Use `batch_judge()` for bulk evaluation (done automatically)
- Cache embeddings for SemanticJudge if evaluating multiple times
- Use `gpt-4o-mini` for LLMJudge to reduce costs

### Debugging

```python
# Print judge decisions
contexts = [
    JudgmentContext(
        query="test query",
        expected_text="expected",
        retrieved_text="retrieved"
    )
]
decisions = judge.batch_judge(contexts)
print(f"Judge: {judge.name}")
print(f"Decisions: {decisions}")
```

---

Happy evaluating! 🎯
