<div align="center">

# 🎯 Evret

**A focused, lightweight retriever evaluation framework with standard Information Retrieval metrics**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/evret)](https://pypi.org/project/evret/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/kaivid-labs/evret/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Evret brings standard Information Retrieval metrics to your recommendation, RAG and search systems. Evaluate retrievers with Hit Rate, Recall, Precision, MRR, NDCG, and Average Precision in just a few lines of code. Built for simplicity, extensibility, and seamless integration with vector databases and AI frameworks.

</div>

---

## 🌟 Overview

**Evret** is a modern Python framework designed for evaluating retrieval systems in Information Retrieval pipelines and search applications. It provides:

- **Standard IR Metrics**: Hit Rate, Recall, Precision, MRR, NDCG, and Average Precision
- **Judge-Based Matching**: Token overlap, semantic, and LLM judges for text relevance
- **Vector Database Support**: Native adapters for Qdrant and other vector databases
- **Framework Integration**:  Adapters for LangChain and LlamaIndex

---

## 🚀 Quick Start

### Installation

```bash
pip install evret
```

For optional integrations:

```bash
pip install evret[all]

# Install specific integrations
pip install "evret[qdrant]"
pip install "evret[langchain]"
pip install "evret[semantic]"
```

### 5-Minute Evaluation

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR, NDCG, TokenOverlapJudge

dataset = EvaluationDataset.from_json("eval_data.json")

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), MRR(k=4), NDCG(k=4)],
    judge=TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6),
)

results = evaluator.evaluate(dataset)
print(results.summary())

# Optional: Export results
results.to_json("results.json")
results.to_csv("results.csv")
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

Run the default test suite:

```bash
pytest
```

See [tests/README.md](tests/README.md) for coverage areas, optional integration tests, and test setup notes.

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

---

## 📋 Evaluation Datasets

Create evaluation datasets with queries and expected answers for judge-based evaluation:

```python
from evret import EvaluationDataset, QueryExample, DocumentExample

dataset = EvaluationDataset(
    documents=[
        DocumentExample(doc_id="doc_1", text="Python uses pip for packages."),
        DocumentExample(doc_id="doc_2", text="Virtual environments isolate dependencies."),
    ],
    queries=[
        QueryExample(
            query_id="q1",
            query_text="How to install Python packages?",
            expected_answers=["pip install"]  # Judge matches this against retrieved text
        )
    ]
)
```

Load datasets from JSON or CSV files:

```python
dataset = EvaluationDataset.from_json("eval_data.json")
dataset = EvaluationDataset.from_csv("eval_data.csv")
```

For detailed dataset format documentation, classic IR evaluation with document IDs, and more examples, see the [Dataset Format Guide](https://github.com/kaivid-labs/evret/blob/main/docs/evaluation/dataset-format.md)

---

## ⚖️ Judges

Judges decide whether retrieved text matches the expected text in your evaluation dataset. `Evaluator` uses `TokenOverlapJudge()` by default, and you can pass `judge=` when you want explicit matching behavior.

```python
from evret import Evaluator, HitRate, Recall
from evret.judges import TokenOverlapJudge

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), Recall(k=4)],
    judge=TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6),
)
```

Use `SemanticJudge` for embedding similarity and `LLMJudge` for LLM-provider based judgment.

```python
from evret.judges import LLMJudge, SemanticJudge

semantic_judge = SemanticJudge(threshold=0.75)
llm_judge = LLMJudge(provider="openai", model="gpt-4o-mini")
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
docs = lc_retriever.invoke("what is information retrieval?")
```

---

## 🧪 Testing

Run the default suite:

```bash
pytest
```

Run Docker-backed integration tests:

```bash
EVRET_RUN_INTEGRATION=1 pytest -m integration
```

More details are in [tests/README.md](tests/README.md).

---

## 📚 Documentation

Evret docs use MkDocs with Material theme.

Install docs dependencies:

```bash
uv pip install -e ".[docs]"
```

Run docs locally:

```bash
mkdocs serve
```

Build static docs:

```bash
mkdocs build
```

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
**Built with ❤️ for the Information Retrieval community**
</div>
