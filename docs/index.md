# Evret Documentation

**Evret** - An information retrieval evaluation framework with text-based semantic matching.

---

## What is Evret?

Evret is a modern Python framework for evaluating retrieval systems in RAG (Retrieval-Augmented Generation) pipelines. Unlike traditional evaluation frameworks that rely on document IDs, Evret supports **text-based relevance matching** for real-world scenarios.

### Key Features

✅ **Standard IR Metrics** - Hit Rate, Recall, Precision, MRR, NDCG, Average Precision
✅ **Judge System** - TokenOverlap, Semantic, and LLM-based matching
✅ **Production-Ready** - Text-based evaluation for real RAG systems
✅ **Vector DB Support** - Native adapters for Qdrant, Chroma, Weaviate, Milvus
✅ **Framework Integration** - Seamless adapters for LangChain and LlamaIndex
✅ **Type-Safe** - Fully typed with comprehensive test coverage

---

## Quick Example

```python
from evret import EvaluationDataset, Evaluator, HitRate, Recall, MRR
from evret.judges import TokenOverlapJudge

# Load dataset with text-based labels
dataset = EvaluationDataset.from_json("eval_data.json")

# Create evaluator with judge
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), Recall(k=4), MRR(k=4)],
    judge=TokenOverlapJudge()  # Text-based matching
)

# Evaluate
results = evaluator.evaluate(dataset)
print(results.summary())
# {'hit_rate@4': 0.85, 'recall@4': 0.75, 'mrr@4': 0.92}
```

---

## Three Judge Types

Evret supports three types of judges for text-based relevance matching:

### 1. TokenOverlapJudge (Default)
Fast keyword/token-based matching with configurable thresholds.

```python
from evret.judges import TokenOverlapJudge

judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)
```

**Best for**: Quick evaluation, keyword-based relevance, no external dependencies

### 2. SemanticJudge
Embedding-based semantic similarity using sentence-transformers.

```python
from evret.judges import SemanticJudge

judge = SemanticJudge(threshold=0.75)
```

**Best for**: Semantic similarity, better accuracy, no API costs

### 3. LLMJudge
LLM-powered semantic judgment (OpenAI, Anthropic, Google Gen AI).

```python
from evret.judges import LLMJudge

judge = LLMJudge(provider="openai", api_key="sk-...")
judge = LLMJudge(provider="google", model="gemini-2.5-flash")
```

**Best for**: Maximum accuracy, complex reasoning, paraphrase detection

---

## Judge Comparison

| Judge | Speed | Accuracy | Cost | Dependencies |
|-------|-------|----------|------|--------------|
| **TokenOverlapJudge** | ⚡️ Fastest | Good | Free | None |
| **SemanticJudge** | 🔄 Medium | Better | Free | sentence-transformers |
| **LLMJudge** | 🐢 Slowest | Best | $$$ | openai/anthropic/google-genai |

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

## Documentation Structure

### Getting Started
- **[Quickstart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Architecture Overview](architecture.md)** - Understand the judge system design

### Core Guides
- [Metrics](metrics/overview.md) - IR metrics reference
- [Evaluation](evaluation/dataset.md) - Dataset and evaluation
- [Retrievers](retrievers/base.md) - Retriever implementations
- [Integrations](integrations/langchain.md) - Framework integrations

### Reference
- [API Reference](api/modules.md) - Detailed API documentation

---

## Why Evret?

### Real-World RAG Evaluation

Traditional evaluation frameworks require pre-labeled document IDs:

```python
# ❌ Toy approach: Requires IDs
relevant_docs = ["doc_123", "doc_456"]
```

Evret works with **actual text content**:

```python
# ✅ Real-world approach: Uses text
relevant_docs = [
    "RAG combines retrieval with generation for better accuracy",
    "Retrieval-augmented generation improves LLM responses"
]
```

### Clean Architecture

Evret separates concerns:

```
User Data (Text) → Judge (Matching) → Evaluator (Mapping) → Metrics (IR Math)
```

- **Judges** handle: "Does this text match that text?"
- **Metrics** handle: "How good is the ranking?"
- **Clean separation** = Extensible, testable, maintainable
