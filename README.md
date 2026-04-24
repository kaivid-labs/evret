<div align="center">

# 🎯 Evret

**A focused, lightweight retriever evaluation framework with standard Information Retrieval metrics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/evret.svg)](https://badge.fury.io/py/evret)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/example/evret/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Evret brings standard Information Retrieval metrics to your RAG and search systems. Evaluate retrievers with Hit Rate, Recall, Precision, MRR, NDCG, and Average Precision in just a few lines of code. Built for simplicity, extensibility, and seamless integration with vector databases and AI frameworks.

</div>

---

## 🌟 Overview

**Evret** is a modern Python framework designed for evaluating retrieval systems in RAG (Retrieval-Augmented Generation) pipelines and search applications. It provides:

- **Standard IR Metrics**: Hit Rate, Recall, Precision, MRR, NDCG, and Average Precision
- **Vector Database Support**: Native adapters for Qdrant and other vector databases
- **Framework Integration**: Seamless adapters for LangChain
- **Production-Ready**: Type-safe, well-tested, and optimized for real-world use cases

Whether you're building a semantic search engine, evaluating RAG systems, or benchmarking retrieval models, Evret gives you the tools to measure what matters.

---

## 🚀 Quick Start

### Installation

```bash
pip install evret
```

For optional integrations:

```bash
# Install all optional integrations
pip install evret[all]
```

### 5-Minute Evaluation

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR, NDCG
# Load your evaluation dataset
dataset = EvaluationDataset.from_json("eval_data.json")

# Evaluate your retriever
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=5), MRR(k=10), NDCG(k=10)]
)

results = evaluator.evaluate(dataset)
print(results.summary())

# Export results
results.to_json("results.json")
results.to_csv("results.csv")
```

### Minimal Metrics Example

```python
from evret import HitRate, Recall, Precision, MRR, NDCG, AveragePrecision

retrieved = [
    ["doc_1", "doc_5", "doc_2", "doc_9"],
    ["doc_8", "doc_7", "doc_6", "doc_3"],
]
relevant = [
    {"doc_1", "doc_2"},
    {"doc_3"},
]

metrics = [HitRate(k=3), Recall(k=3), Precision(k=3), MRR(k=3), NDCG(k=3)]

for metric in metrics:
    score = metric.score(
        retrieved_by_query=retrieved,
        relevant_by_query=relevant,
    )
    print(f"{metric.name}: {score:.4f}")
```

---

## 🛠 Local Development Setup

Use `uv` for local development:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install all optional integrations:

```bash
uv pip install -e ".[all]"
```

Run tests:

```bash
pytest
```

---

## 📊 Metrics

Evret supports all standard Information Retrieval metrics:

| Metric | Description | Use Case |
|--------|-------------|----------|
| **Hit Rate@k** | % of queries with at least one relevant doc in top-k | Binary relevance, recall-focused |
| **Recall@k** | % of relevant docs found in top-k | Comprehensive retrieval |
| **Precision@k** | % of top-k results that are relevant | Precision-focused systems |
| **MRR@k** | Mean Reciprocal Rank of first relevant doc | Single-answer retrieval |
| **NDCG@k** | Normalized Discounted Cumulative Gain | Rank-aware binary relevance quality |
| **Average Precision@k** | Area under precision-recall curve | Overall ranking quality |

### Usage Example

```python
from evret import HitRate, Recall, Precision, MRR, NDCG, AveragePrecision

metrics = [
    HitRate(k=5),
    Recall(k=10),
    Precision(k=5),
    MRR(k=10),
    NDCG(k=10),
    AveragePrecision(k=10),
]

# Score a single query
retrieved = ["doc_1", "doc_5", "doc_2"]
relevant = {"doc_1", "doc_2"}

for metric in metrics:
    score = metric.score(
        retrieved_by_query=[retrieved],
        relevant_by_query=[relevant],
    )
    print(f"{metric.name}: {score:.4f}")
```

---

## 🔌 Integrations

### Qdrant Vector Database

```python
from evret.retrievers import QdrantRetriever

retriever = QdrantRetriever(
    collection_name="docs",
    query_encoder=embed_query,
    url="http://localhost:6333",
    id_field="doc_id",
)
```

### LangChain Integration

```python
from evret.integrations import LangChainRetrieverAdapter

# Wrap any Evret retriever for use in LangChain
lc_retriever = LangChainRetrieverAdapter(evret_retriever=retriever, k=5)
docs = lc_retriever.invoke("what is RAG?")
```

---

## 📁 Examples

### Basic Evaluation Pipeline

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR

dataset = EvaluationDataset.from_json("eval_data.json")
evaluator = Evaluator(retriever=my_retriever, metrics=[HitRate(k=5), MRR(k=10)])
results = evaluator.evaluate(dataset)

results.to_json("results.json")
results.to_csv("results.csv")
print(results.summary())
```

### Custom Retriever

```python
from evret.retrievers import BaseRetriever
from evret import RetrievalResult

class MyCustomRetriever(BaseRetriever):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        self._validate_k(k)
        # Your retrieval logic here
        return [
            RetrievalResult(doc_id="doc_1", score=0.95, metadata={"text": "..."}),
            RetrievalResult(doc_id="doc_2", score=0.87),
        ]
```

### Run Examples Locally

```bash
# Basic evaluation example
python examples/evaluate_retriever.py

# Jupyter notebook quickstart
jupyter notebook examples/langchain_rag_evaluation.ipynb
```

---

## 🧪 Testing

### Run Unit Tests

```bash
pytest
```

### Run Integration Tests

Integration tests require Docker to be running:

```bash
# Start Docker Desktop/daemon first, then run:
EVRET_RUN_INTEGRATION=1 pytest -m integration
```

This will spin up Docker containers for Qdrant and Chroma to run end-to-end tests.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

---

## 📚 Citation

If you use Evret in your research, please cite:

```bibtex
@software{evret2026,
  title={Evret: A Focused Retriever Evaluation Framework},
  author={lucifertrj},
  year={2026},
}
```

---

<div align="center">

**Built with ❤️ for the RAG and IR community**

[GitHub](https://github.com/example/evret) • [Issues](https://github.com/example/evret/issues) • [Discussions](https://github.com/example/evret/discussions)

</div>
