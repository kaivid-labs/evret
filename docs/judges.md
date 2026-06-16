# Judges

Judges decide whether retrieved text matches the expected text for a query. The evaluator maps those judge decisions into the expected labels used by the standard IR metrics.

## TokenOverlapJudge

Use `TokenOverlapJudge` for fast local checks with no external dependency. It removes common stopwords, scores weighted token overlap, adds small phrase and query-match bonuses, and rejects negation mismatches.

Do not use `TokenOverlapJudge` as the text judge for LLM-generated evaluation datasets. Generated expected answers may be paraphrased or summarized from the source chunk, so use `LLMJudge` for text-based judgment. If the generated dataset keeps `expected_doc_ids`, the evaluator scores those document IDs directly and does not need a judge for metric matching.

```python
from evret.judges import TokenOverlapJudge

judge = TokenOverlapJudge(
    min_tokens=15,
    overlap_ratio=0.6,
    query_boost=True,
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `min_tokens` | `int` | `15` | Minimum number of shared tokens required before a match can pass. |
| `overlap_ratio` | `float` | `0.6` | Minimum score required for a match. |
| `query_boost` | `bool` | `True` | Adds a small score boost when matched tokens also appear in the query. |
| `stopwords` | `Iterable[str] \| None` | `None` | Optional custom stopword list. Defaults to Evret's built-in English stopwords. |

### RAG Use Case: Configuring for Different Chunk Sizes

The default `min_tokens=15` is suitable for medium-sized chunks. Adjust based on your chunk size:

```python
# For ~1000 token chunks
judge = TokenOverlapJudge(
    min_tokens=50,      # 5% of chunk size
    overlap_ratio=0.6,
)

# For ~500 token chunks (default is good)
judge = TokenOverlapJudge(
    min_tokens=15,      # default
    overlap_ratio=0.6,
)

# For smaller chunks (~200 tokens)
judge = TokenOverlapJudge(
    min_tokens=10,      # 5% of chunk size
    overlap_ratio=0.6,
)
```

**Rule of thumb:** Set `min_tokens` to 5-10% of your average chunk size to ensure meaningful overlap beyond common stopwords.

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

Use `LLMJudge` for text-based evaluation of LLM-generated datasets. It is a better fit than token overlap when expected answers were generated from chunks and may not share exact wording with retrieved content.

```python
from evret.judges import LLMJudge

judge = LLMJudge(
    provider="openai",
    model="gpt-5.4-nano",
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
| `openai` | `gpt-5.4-nano` | `OPENAI_API_KEY` |
| `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| `google` | `gemini-3-flash-preview` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

## Use A Judge With Evaluator

```python
from evret import EvaluationDataset, Evaluator, HitRate, Recall
from evret.judges import TokenOverlapJudge

dataset = EvaluationDataset.from_json("examples/eval_data.json")
judge = TokenOverlapJudge(min_tokens=15, overlap_ratio=0.6)

evaluator = Evaluator(
    retriever=my_retriever,
    metrics=[HitRate(k=4), Recall(k=4)],
    judge=judge,
)

results = evaluator.evaluate(dataset)
print(results.summary())
```
