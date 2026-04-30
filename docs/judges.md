# Judges

Judges decide whether retrieved text matches the expected text for a query. The evaluator maps those judge decisions into the ID sets used by the standard IR metrics.

## TokenOverlapJudge

Use `TokenOverlapJudge` for fast local checks with no external dependency.

```python
from evret.judges import TokenOverlapJudge

judge = TokenOverlapJudge(
    min_tokens=2,
    overlap_ratio=0.6,
    query_boost=True,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `min_tokens` | `int` | `2` | Minimum number of shared tokens required before a match can pass. |
| `overlap_ratio` | `float` | `0.6` | Minimum fraction of expected tokens that must appear in retrieved text. |
| `query_boost` | `bool` | `True` | Allows query-token overlap to relax the overlap threshold. |

## SemanticJudge

Use `SemanticJudge` when exact words differ but the meaning should match.

```python
from evret.judges import SemanticJudge

judge = SemanticJudge(
    model="sentence-transformers/all-MiniLM-L6-v2",
    threshold=0.75,
    device="cpu",
)
```

Install the optional dependency first:

```bash
pip install evret[semantic]
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `model` | `str` | `sentence-transformers/all-MiniLM-L6-v2` | SentenceTransformers model used for embeddings. |
| `threshold` | `float` | `0.75` | Cosine similarity cutoff. Values must be in `[0, 1]`. |
| `device` | `str` | `cpu` | Device passed to SentenceTransformers, such as `cpu` or `cuda`. |

## LLMJudge

Use `LLMJudge` when the expected and retrieved text need semantic judgment from an LLM.

```python
from evret.judges import LLMJudge

judge = LLMJudge(
    provider="openai",
    model="gpt-4o-mini",
    api_key=None,
    temperature=0.0,
    max_retries=3,
)
```

Install the provider dependency first:

```bash
pip install evret[llm-openai]
pip install evret[llm-anthropic]
pip install evret[llm-google]
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `str` | `openai` | LLM provider. Supported values are `openai`, `anthropic`, and `google`. |
| `model` | `str | None` | `None` | Model name. When omitted, Evret uses the provider default. |
| `api_key` | `str | None` | `None` | API key. When omitted, Evret reads the provider environment variable. |
| `temperature` | `float` | `0.0` | Sampling temperature for deterministic relevance decisions. |
| `max_retries` | `int` | `3` | Retry attempts for failed API calls. |

Provider defaults:

| Provider | Default Model | Environment Variable |
| --- | --- | --- |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |
| `google` | `gemini-2.5-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

## Use A Judge With Evaluator

```python
from evret import EvaluationDataset, Evaluator, HitRate, Recall
from evret.judges import TokenOverlapJudge

dataset = EvaluationDataset.from_json("examples/eval_data.json")
judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), Recall(k=4)],
    judge=judge,
)

results = evaluator.evaluate(dataset)
print(results.summary())
```
