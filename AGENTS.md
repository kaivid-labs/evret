# AGENTS.md

Guidance for AI coding agents working on **Evret** — a focused, lightweight retriever
evaluation framework providing standard Information Retrieval metrics for RAG, search,
and recommendation systems. This file complements `README.md` (for humans) with
agent-focused context. See [agents.md](https://agents.md) for the format.

## Project overview

Evret evaluates retrievers with standard IR metrics (Hit Rate, Recall, Precision, MRR,
NDCG, ERR, RBP, Average Precision). It is a pure-Python library published to PyPI as
`evret`, built around three pluggable concepts:

- **Retrievers** — adapters over vector DBs (Qdrant, Chroma, Elasticsearch, Weaviate,
  Milvus) plus framework adapters (LangChain, LlamaIndex, Haystack).
- **Judges** — decide whether a retrieved chunk matches a relevant answer: token
  overlap, semantic (embeddings), and LLM-based (OpenAI / Anthropic / Google / TruLens).
- **Metrics + Evaluator** — the orchestrator runs a dataset through a retriever, applies
  a judge, and computes the configured metrics.

## Setup commands

Use `uv` for local development (Python >= 3.10):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[all,dev]"   # editable install with every optional integration
```

Plain pip works too: `pip install -e ".[all,dev]"`.

Optional dependency groups are defined in `pyproject.toml` — install only what you need
(e.g. `.[qdrant]`, `.[langchain]`, `.[semantic]`, `.[llm-openai]`, `.[trulens-openai]`).

## Testing

```bash
pytest                                  # all unit tests (config in pyproject.toml)
pytest tests/metrics/test_hit_rate.py   # a single file
pytest --cov=evret --cov-report=html    # with coverage
EVRET_RUN_INTEGRATION=1 pytest -m integration   # Docker-backed integration tests
```

- Tests live under `tests/`, mirroring the `src/evret/` layout.
- `integration` is the only custom marker; those tests need Docker and are skipped
  unless `EVRET_RUN_INTEGRATION=1` is set.
- Run `pytest` before committing and fix all failures. Add tests for any new metric,
  judge, or retriever.
- Name tests descriptively: `test_<function>_<scenario>_<expected_result>`.

## Code style

- Follow PEP 8; max line length **100**.
- Type-hint all function signatures. `from __future__ import annotations` at the top.
- Format with `black src/ tests/`; type-check with `mypy src/` (both recommended).
- Imports: absolute only, grouped stdlib → third-party → local, alphabetized per group.
- Guard optional third-party imports and raise `OptionalDependencyError` (see
  `src/evret/errors.py`) when a needed extra isn't installed — never let the base
  package hard-import an optional dependency.

## Project structure

```
src/evret/
├── __init__.py          # public API surface — keep __all__ in sync with new exports
├── errors.py            # EvretError, EvretValidationError, OptionalDependencyError
├── logging.py           # configure_logging, get_logger
├── metrics/             # hit_rate, recall, precision, mrr, ndcg, err, rbp,
│                        #   average_precision (+ base.py; _-prefixed helpers are private)
├── retrievers/          # base + qdrant/chroma/elasticsearch/weaviate/milvus adapters
├── judges/              # base, token_overlap, semantic, trulens
│   └── llm/             # LLM judge: factory, prompts, providers/{openai,anthropic,google}
├── integrations/        # langchain, llamaindex, haystack retriever adapters
├── evaluation/          # Evaluator, EvaluationDataset, EvaluationResults, examples
└── generation/          # synthetic dataset generation (chunking + LLM prompts)

docs/        # MkDocs (Material) site — mirror public API changes here
examples/    # runnable examples (qdrant demo, trulens-judge, streamlit dashboard, …)
tests/       # pytest suite mirroring src/
```

## Public API conventions

- The supported entry point is `from evret import ...`. Everything users should reach is
  re-exported in `src/evret/__init__.py`. When you add or rename a public class/function,
  update both the relevant subpackage `__init__.py` **and** the top-level `__all__`.
- Current exports include: metrics (`HitRate`, `Recall`, `Precision`, `MRR`, `NDCG`,
  `ERR`, `RBP`, `AveragePrecision`), retrievers (`QdrantRetriever`, `ChromaRetriever`,
  `ElasticsearchRetriever`, `WeaviateRetriever`, `MilvusRetriever`, `BaseRetriever`,
  `RetrievalResult`), judges (`Judge`, `JudgmentContext`, `TokenOverlapJudge`),
  evaluation (`Evaluator`, `EvaluationDataset`, `EvaluationResults`, `QueryExample`,
  `DocumentExample`), generation, and integration adapters.

### Minimal usage shape

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR, NDCG, TokenOverlapJudge

dataset = EvaluationDataset.from_json("eval_data.json")
evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), MRR(k=4), NDCG(k=4)],
    judge=TokenOverlapJudge(min_tokens=15, overlap_ratio=0.6),
)
results = evaluator.evaluate(dataset)
print(results.summary())
```

## LLM / provider notes

- Provider SDKs (OpenAI, Anthropic, Google GenAI, TruLens) are optional extras. When
  touching LLM judges, prefer the latest Claude models for Anthropic defaults and keep
  model identifiers current — do not reintroduce stale model names.
- Never hard-code API keys; read them from the environment.

## Documentation

- Docs are built with MkDocs Material (`mkdocs.yml`, sources under `docs/`). Keep
  `docs/` in sync when you change the public API, add a metric/judge/retriever, or change
  behavior. `README.md` holds the human quick start.

## Commit & PR guidelines

- Branch from `main`; never commit directly to it.
- Write clear, imperative commit messages (e.g. `Add ERR metric`, `Fix NDCG tie handling`).
- Before opening a PR: `pytest` passes, new code is tested, docs and `__all__` updated.
- PRs should state what changed, why, and reference any related issue.
- See `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md` for the full process.

## References

- Human quick start: [README.md](README.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Architecture notes: [docs/architecture.md](docs/architecture.md)
- Test setup & coverage areas: [tests/README.md](tests/README.md)
