# Evret Architecture: Judge System Design

## Overview

Evret implements a **judge-based architecture** for text-based relevance matching in RAG evaluation. This document explains the design decisions, data flow, and why we chose boolean judgments over continuous scores.

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│  Dataset with text-based relevance labels                        │
│  {                                                                │
│    "query": "What is RAG?",                                      │
│    "relevant_docs": [                                            │
│      "RAG combines retrieval with generation...",               │
│      "Retrieval-augmented generation improves accuracy..."      │
│    ]                                                             │
│  }                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RETRIEVER                                   │
│  Returns: Retrieved documents with text/metadata                 │
│  [                                                               │
│    RetrievalResult(                                             │
│      doc_id="doc_123",                                          │
│      score=0.95,                                                │
│      metadata={"text": "RAG is retrieval-augmented..."}        │
│    )                                                            │
│  ]                                                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    JUDGE SYSTEM                                  │
│                                                                  │
│  Input:  JudgmentContext(                                       │
│            query="What is RAG?",                                │
│            expected_text="RAG combines retrieval...",           │
│            retrieved_text="RAG is retrieval-augmented..."       │
│          )                                                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐      │
│  │  TokenOverlapJudge / SemanticJudge / LLMJudge        │      │
│  │  Determines: Is retrieved text relevant?             │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                  │
│  Output: Boolean (True/False)                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EVALUATOR                                   │
│  Maps boolean judgments → ID sets                                │
│                                                                  │
│  retrieved_ids = ["relevant_doc_1", "relevant_doc_2", ...]      │
│  relevant_ids = {"relevant_doc_1", "relevant_doc_2"}            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                       METRICS                                    │
│  Compute standard IR metrics on ID sets                          │
│                                                                  │
│  - Precision@k = |relevant ∩ retrieved[:k]| / k                 │
│  - Recall@k = |relevant ∩ retrieved[:k]| / |relevant|           │
│  - MRR@k = 1 / rank_of_first_relevant                           │
│  - NDCG@k, HitRate@k, Average Precision@k                       │
│                                                                  │
│  Output: {"recall@4": 0.75, "precision@4": 0.5, ...}           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Flow

### 1. Judge System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                       Judge Interface                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  abstract class Judge:                                    │  │
│  │    def judge(context: JudgmentContext) -> bool           │  │
│  │    def batch_judge(contexts: List[Context]) -> List[bool]│  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ TokenOverlap    │ │ SemanticJudge   │ │   LLMJudge      │
│    Judge        │ │                 │ │                 │
│                 │ │                 │ │                 │
│ • Fast          │ │ • Embeddings    │ │ • LLM API       │
│ • Token-based   │ │ • Cosine sim    │ │ • Highest acc   │
│ • No deps       │ │ • Batched       │ │ • Async batch   │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2. Evaluator Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Evaluator.evaluate()                          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Retrieve Documents                                      │
│  ────────────────────────────                                    │
│  retrieved_results = retriever.batch_retrieve(queries, k=max_k) │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Build Judgment Contexts                                 │
│  ────────────────────────────────                                │
│  For each query:                                                 │
│    For each retrieved result:                                    │
│      For each expected relevant text:                            │
│        Create JudgmentContext(                                   │
│          query=query_text,                                       │
│          expected_text=relevant_label,                           │
│          retrieved_text=result.metadata["text"]                  │
│        )                                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Batch Judge All Contexts                                │
│  ──────────────────────────────────                              │
│  all_judgments = judge.batch_judge(all_contexts)                │
│                                                                  │
│  Returns: [True, False, True, True, False, ...]                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Map Judgments to IDs                                    │
│  ─────────────────────────────                                   │
│  For each query:                                                 │
│    For each retrieved result:                                    │
│      Find first matching expected text (judgment=True)           │
│      Assign matched ID to retrieved result                       │
│                                                                  │
│  retrieved_ids = ["relevant_1", "relevant_2", "irrelevant_0"]   │
│  relevant_ids = {"relevant_1", "relevant_2"}                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Compute Metrics                                         │
│  ────────────────────────                                        │
│  For each metric:                                                │
│    score = metric.score(retrieved_ids, relevant_ids)            │
│                                                                  │
│  Results: {"recall@4": 0.75, "precision@4": 0.5, ...}          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Why Boolean Judgments?

### Design Decision: Boolean vs. Continuous Scores

We chose **boolean judgments** (`True`/`False`) over continuous relevance scores (e.g., `0.0` to `1.0`) for several critical reasons:

### 1. **Alignment with IR Metrics Semantics**

Traditional Information Retrieval metrics are **set-based**, not score-based:

```python
# Recall: What fraction of relevant documents were retrieved?
Recall@k = |relevant ∩ retrieved[:k]| / |relevant|

# Precision: What fraction of retrieved documents are relevant?
Precision@k = |relevant ∩ retrieved[:k]| / k
```

These metrics operate on **binary relevance**: a document is either relevant or not. There's no "50% relevant" in the classic IR formulation.

**Example:**
```python
# Boolean approach (what we use)
relevant = {"doc_1", "doc_2", "doc_3"}
retrieved = ["doc_1", "doc_5", "doc_2", "doc_9"]
recall = len({"doc_1", "doc_2"}) / 3 = 0.667

# Continuous approach (problematic)
# How do you compute recall with scores?
# Do you sum? Average? Threshold?
```

### 2. **Clean Separation of Concerns**

```
┌─────────────────────────────────────────────────────────────┐
│  Judge: Answers "Is this relevant?"                          │
│  • Input: (query, expected_text, retrieved_text)            │
│  • Output: Boolean decision                                 │
│  • Responsibility: Relevance matching logic                 │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  Metrics: Answers "How good is the retrieval?"              │
│  • Input: (retrieved_ids, relevant_ids)                     │
│  • Output: Metric score (0.0 to 1.0)                       │
│  • Responsibility: Ranking quality measurement             │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Judges focus on: **"Does this match?"** (domain-specific)
- Metrics focus on: **"How good is the ranking?"** (domain-agnostic)
- No confusion between "relevance score" and "metric score"

### 3. **Ambiguity of Continuous Scores**

Continuous scores introduce ambiguity:

```python
# What does a score of 0.7 mean?
judge.judge_with_score(context) → 0.7

# Is this:
# - 70% confident it's relevant? → Use threshold (boolean anyway)
# - 70% semantically similar? → Metric already measures this
# - 70% of tokens overlap? → Implementation detail, not user concern
```

**With boolean:**
```python
# Clear decision boundary
judge.judge(context) → True  # It's relevant
judge.judge(context) → False # It's not relevant
```

Users can configure the threshold internally:
```python
SemanticJudge(threshold=0.7)  # Internally uses 0.7, returns boolean
```

### 4. **Simplicity for Custom Judges**

**Boolean approach** (what we use):
```python
class MyCustomJudge(Judge):
    def judge(self, context: JudgmentContext) -> bool:
        # Simple: return True or False
        return my_matching_logic(context.expected_text, context.retrieved_text)
```

**Continuous approach** (problematic):
```python
class MyCustomJudge(Judge):
    def judge(self, context: JudgmentContext) -> float:
        # Complex: What scale? How to interpret?
        # How does this interact with metrics?
        return ???  # 0.0 to 1.0? Unbounded? Normalized?
```

### 5. **Consistency with NDCG Binary Relevance**

NDCG (Normalized Discounted Cumulative Gain) in Evret uses **binary relevance**:

```python
# Document is either relevant (1) or not (0)
relevance_scores = [1, 0, 1, 1, 0]  # Binary

# Not continuous:
relevance_scores = [0.8, 0.3, 0.9, 0.7, 0.1]  # Would require graded relevance
```

Our boolean approach naturally maps:
- `judge() → True` → `relevance = 1`
- `judge() → False` → `relevance = 0`

### 6. **Extensibility Without Breaking Changes**

Boolean is a **strict interface**. If we need graded relevance in the future:

```python
# Future: Graded relevance (backward compatible)
class GradedJudge(Judge):
    def judge(self, context: JudgmentContext) -> bool:
        # Still implements boolean interface
        return self.grade(context) >= self.threshold

    def grade(self, context: JudgmentContext) -> float:
        # Optional: Additional method for continuous scores
        return self._compute_relevance_score(context)
```

Boolean is the **minimal contract**. We can always add more, but can't remove.

---

## Comparison: Boolean vs. Continuous

| Aspect | Boolean (Our Choice) | Continuous |
|--------|---------------------|------------|
| **Simplicity** | ✅ Clear True/False | ❌ Ambiguous scale |
| **IR Metrics Alignment** | ✅ Natural set operations | ❌ Requires thresholding |
| **User Experience** | ✅ Easy to understand | ❌ "What does 0.7 mean?" |
| **Custom Judges** | ✅ Simple to implement | ❌ Complex interface |
| **Performance** | ✅ Fast comparisons | ⚠️ Needs normalization |
| **Extensibility** | ✅ Can add grading later | ❌ Breaking change to simplify |

---

## Judge Implementation Details

### TokenOverlapJudge Algorithm

```
Input: JudgmentContext(query, expected_text, retrieved_text)

Step 1: Exact Match Check
├─ IF normalized(expected_text) == normalized(retrieved_text)
│  └─ RETURN True
│
Step 2: Substring Match
├─ IF expected_text in retrieved_text OR retrieved_text in expected_text
│  └─ RETURN True
│
Step 3: Token Overlap Computation
├─ expected_tokens = tokenize(expected_text)
├─ retrieved_tokens = tokenize(retrieved_text)
├─ shared_tokens = expected_tokens ∩ retrieved_tokens
│
├─ IF len(shared_tokens) < min_tokens
│  └─ RETURN False
│
├─ overlap_ratio = len(shared_tokens) / len(expected_tokens)
│
├─ IF overlap_ratio >= threshold
│  └─ RETURN True
│
└─ IF query_boost AND query shares tokens
   ├─ relaxed_threshold = threshold × 0.75
   └─ RETURN overlap_ratio >= relaxed_threshold

Otherwise: RETURN False
```

### SemanticJudge Algorithm

```
Input: JudgmentContext(query, expected_text, retrieved_text)

Step 1: Encode Texts
├─ emb_expected = model.encode(expected_text)
├─ emb_retrieved = model.encode(retrieved_text)

Step 2: Compute Cosine Similarity
├─ similarity = dot(emb_expected, emb_retrieved) /
│               (norm(emb_expected) × norm(emb_retrieved))

Step 3: Apply Threshold
└─ RETURN similarity >= threshold

Batch Optimization:
├─ Encode all texts at once (vectorized)
├─ Compute all similarities (matrix operation)
└─ Apply threshold to all results
```

### LLMJudge Algorithm

```
Input: JudgmentContext(query, expected_text, retrieved_text)

Step 1: Build Prompt
├─ prompt = TEMPLATE.format(
│     query=query,
│     expected_text=expected_text,
│     retrieved_text=retrieved_text
│   )

Step 2: Call LLM API
├─ response = provider.complete(prompt)

Step 3: Parse Response
├─ IF response starts with "YES" → RETURN True
├─ IF response starts with "NO" → RETURN False
├─ IF response contains positive keywords → RETURN True
├─ IF response contains negative keywords → RETURN False
└─ OTHERWISE → RETURN False (conservative)

Batch Optimization:
├─ Build prompts for all contexts
├─ Call API concurrently (asyncio.gather)
└─ Parse all responses
```

---

## Performance Characteristics

### Time Complexity

| Judge | Single | Batch (n contexts) |
|-------|--------|-------------------|
| **TokenOverlap** | O(t) | O(n·t) |
| **Semantic** | O(m) | O(n·m) |
| **LLM** | O(a) | O(a) with async |

Where:
- `t` = average text length (token operations)
- `m` = model inference time
- `a` = API latency
- `n` = number of contexts

### Space Complexity

| Judge | Memory Usage |
|-------|--------------|
| **TokenOverlap** | O(t) - temporary token sets |
| **Semantic** | O(d·n) - embeddings (d=dimensions) |
| **LLM** | O(1) - no local storage |

---

## Example: Complete Evaluation Flow

### Input Data
```python
dataset = EvaluationDataset(
    queries=[
        QueryExample(
            query_id="q1",
            query_text="What is RAG?",
            relevant_docs=[
                "RAG combines retrieval with generation for better accuracy",
                "Retrieval-augmented generation improves LLM responses"
            ]
        )
    ]
)
```

### Retrieval Results
```python
retrieved = [
    RetrievalResult(
        doc_id="doc_123",
        score=0.95,
        metadata={"text": "RAG is a technique that combines retrieval with generation"}
    ),
    RetrievalResult(
        doc_id="doc_456",
        score=0.87,
        metadata={"text": "Vector databases store embeddings"}
    )
]
```

### Judgment Phase
```python
# Context 1: Expected text 1 vs. Retrieved doc 1
context_1_1 = JudgmentContext(
    query="What is RAG?",
    expected_text="rag combines retrieval with generation for better accuracy",
    retrieved_text="rag is a technique that combines retrieval with generation"
)
judge.judge(context_1_1) → True  # Match!

# Context 1: Expected text 2 vs. Retrieved doc 1
context_1_2 = JudgmentContext(
    query="What is RAG?",
    expected_text="retrieval augmented generation improves llm responses",
    retrieved_text="rag is a technique that combines retrieval with generation"
)
judge.judge(context_1_2) → True  # Match!

# Context 2: Expected text 1 vs. Retrieved doc 2
context_2_1 = JudgmentContext(
    query="What is RAG?",
    expected_text="rag combines retrieval with generation for better accuracy",
    retrieved_text="vector databases store embeddings"
)
judge.judge(context_2_1) → False  # No match

# Context 2: Expected text 2 vs. Retrieved doc 2
context_2_2 = JudgmentContext(
    query="What is RAG?",
    expected_text="retrieval augmented generation improves llm responses",
    retrieved_text="vector databases store embeddings"
)
judge.judge(context_2_2) → False  # No match
```

### ID Mapping
```python
# Retrieved doc 1 matches expected text 1
retrieved_ids = ["relevant_doc_1", "retrieved_1:vector databases store embeddings"]
relevant_ids = {"relevant_doc_1", "relevant_doc_2"}
```

### Metric Computation
```python
# Recall@2
recall = len({"relevant_doc_1"} ∩ retrieved_ids[:2]) / 2
       = 1 / 2
       = 0.5

# Precision@2
precision = len({"relevant_doc_1"} ∩ retrieved_ids[:2]) / 2
          = 1 / 2
          = 0.5

# HitRate@2
hit_rate = 1.0  # At least one relevant doc found
```

---

## Summary

The boolean judge design provides:

✅ **Clarity** - Binary decisions are unambiguous
✅ **Correctness** - Aligns with IR metric semantics
✅ **Simplicity** - Easy to implement and understand
✅ **Performance** - Fast comparisons, efficient batching
✅ **Extensibility** - Can add graded relevance later
✅ **Separation** - Clean boundary between matching and measurement

This architecture enables **production-ready RAG evaluation** with text-based matching while maintaining mathematical rigor and user-friendly interfaces.
