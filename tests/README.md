# Evret Test Suite

This directory covers the core behavior of Evret across metrics, evaluation, retrievers, framework adapters, judges, and Docker-backed integrations.

## What Is Covered

| Area | Path | Coverage |
|------|------|----------|
| Metrics | `tests/metrics/` | Hit Rate, Recall, Precision, MRR, NDCG, and Average Precision scoring behavior. |
| Evaluation | `tests/evaluation/` | Dataset loading, evaluator orchestration, result summaries, exports, and default relevance matching. |
| Judges | `tests/judges/` | Token overlap judging and LLM provider wiring with mocked clients. |
| Retrievers | `tests/retrievers/` | Base retriever validation plus Qdrant, Chroma, Milvus, and Weaviate adapter behavior with mocked clients. |
| Framework integrations | `tests/integrations/` | LangChain and LlamaIndex adapter behavior when optional packages are installed. |
| Docker integrations | `tests/integration/` | End-to-end Qdrant and Chroma retrieval against real containers. |
| Shared utilities | `tests/test_utils.py` | Name, k-value, and dataset path validation helpers. |

## Install Test Dependencies

For the default test suite:

```bash
uv pip install -e ".[dev]"
```

For all optional integration coverage:

```bash
uv pip install -e ".[integration]"
```

## Run Tests

Run the default suite:

```bash
pytest
```

Run a focused area:

```bash
pytest tests/metrics
pytest tests/evaluation
pytest tests/retrievers/test_qdrant.py
```

Run Docker-backed integration tests:

```bash
EVRET_RUN_INTEGRATION=1 pytest -m integration
```

The integration tests require Docker to be running. Without `EVRET_RUN_INTEGRATION=1`, they are skipped.

Run everything except Docker-backed tests:

```bash
pytest -m "not integration"
```
