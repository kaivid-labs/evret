# Evret Agent & Skills Documentation

## Overview

This document describes the agent skills available in the Evret project. These skills are used by Claude Code to provide specialized guidance on retrieval evaluation metrics and RAG pipeline development.

---

## Agent Skills Directory Structure

```
.agents/
└── skills/
    └── retriever-evals/
        ├── SKILL.md                          # Main RAG metrics skill
        └── metrics/
            ├── HITRATE_SKILL.md              # Hit Rate metric guidance
            ├── RECALL_SKILL.md               # Recall@K metric guidance
            ├── PRECISION_SKILL.md            # Precision@K metric guidance
            ├── MRR_SKILL.md                  # MRR metric guidance
            ├── NDCG_SKILL.md                 # nDCG metric guidance
            └── AVERAGE_PRECISION_SKILL.md    # Average Precision/MAP guidance
```

---

## Available Skills

### 1. RAG Metrics Skill (`rag-metrics`)

**Location**: [.agents/skills/retriever-evals/SKILL.md](.agents/skills/retriever-evals/SKILL.md)

**Trigger Keywords**:
- RAG pipeline, retrieval system, vector search
- MRR, nDCG, Hit Rate, Recall, retrieval metrics, ranking metrics
- Debugging hallucinating LLM
- Tuning retriever or reranker
- Choosing between retrieval evaluation metrics

**Purpose**: Provides high-level guidance on choosing the right retrieval evaluation metric for different use cases.

**Key Topics**:
- Metric selection guidance based on use case
- Stage-based recommendations (debugging, tuning, eval)
- Comparison table of all metrics
- Best practices for multi-objective evaluation

**Quick Reference**:
| Use Case | Recommended Metric | Why |
|----------|-------------------|-----|
| Check if any correct result retrieved | Hit Rate | Binary success check |
| Measure completeness of results | Recall | Coverage metric |
| Know how early first correct result appears | MRR | Single-answer QA optimization |
| Care about ranking quality | nDCG | Multi-relevance ranking quality |

---

### 2. Hit Rate Metric Skill (`hitrate-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/HITRATE_SKILL.md](.agents/skills/retriever-evals/metrics/HITRATE_SKILL.md)

**Trigger Keywords**:
- Hit Rate
- Binary check on retrieval
- Debugging hallucinating LLM
- Simplest retrieval metric
- Validating chunks in top-k results

**Purpose**: Deep dive into Hit Rate metric - the first-line diagnostic for RAG systems.

**Formula**:
```
Hit Rate = (1/|Q|) × Σ 𝟙[relevant ∩ retrieved ≠ ∅]
```

**When to Use**:
- ✅ Initial RAG debugging / baseline check
- ✅ Embedding model or chunking strategy swap
- ✅ Reranker evaluation (alongside nDCG)
- ❌ Evaluating ranking quality (use MRR or nDCG)
- ❌ Completeness-critical retrieval (use Recall)

**Best Practices**:
1. Use as entry-point metric, not final one
2. Choose k that mirrors production context window
3. Segment by query difficulty and domain
4. Pair with MRR to separate presence from position
5. Use in CI/CD to catch regressions

---

### 3. Recall@K Metric Skill (`recall-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/RECALL_SKILL.md](.agents/skills/retriever-evals/metrics/RECALL_SKILL.md)

**Trigger Keywords**:
- Recall, Recall@K
- Coverage metric
- Completeness of retrieval
- Multi-document QA

**Purpose**: Measures what fraction of all relevant documents were retrieved.

**Formula**:
```
Recall@k = |relevant ∩ retrieved_k| / |relevant|
```

**When to Use**:
- ✅ Multi-document QA or synthesis tasks
- ✅ Completeness-critical domains (medical, financial, safety)
- ✅ Measuring coverage improvements
- ❌ When you only care about first hit (use MRR)
- ❌ When ranking quality matters more (use nDCG)

---

### 4. Precision@K Metric Skill (`precision-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/PRECISION_SKILL.md](.agents/skills/retriever-evals/metrics/PRECISION_SKILL.md)

**Trigger Keywords**:
- Precision, Precision@K
- Purity metric
- LLM context quality
- Noise reduction

**Purpose**: Measures what fraction of retrieved documents are relevant.

**Formula**:
```
Precision@k = |relevant ∩ retrieved_k| / k
```

**When to Use**:
- ✅ LLM context quality optimization
- ✅ Noise reduction in retrieved results
- ✅ Limited context window scenarios
- ❌ When you need to measure coverage (use Recall)
- ❌ When ranking order matters (use MRR or nDCG)

---

### 5. MRR (Mean Reciprocal Rank) Skill (`mrr-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/MRR_SKILL.md](.agents/skills/retriever-evals/metrics/MRR_SKILL.md)

**Trigger Keywords**:
- MRR, Mean Reciprocal Rank
- First-hit metric
- Single-answer QA
- Fast-hit evaluation

**Purpose**: Measures how early the first relevant result appears in rankings.

**Formula**:
```
MRR = (1/|Q|) × Σ (1/rank_first_relevant)
```

**Scoring**:
- Rank 1 → score = 1.0
- Rank 2 → score = 0.5
- Rank 5 → score = 0.2

**When to Use**:
- ✅ Single-answer QA systems
- ✅ Intent classification, slot filling
- ✅ Fast-hit optimization
- ❌ Multi-document tasks (use nDCG)
- ❌ When you need all relevant results (use Recall)

---

### 6. nDCG (Normalized Discounted Cumulative Gain) Skill (`ndcg-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/NDCG_SKILL.md](.agents/skills/retriever-evals/metrics/NDCG_SKILL.md)

**Trigger Keywords**:
- nDCG, DCG, Normalized Discounted Cumulative Gain
- Ranking quality
- Graded relevance
- Reranker tuning

**Purpose**: Measures ranking quality across all relevant results with support for graded relevance.

**Formula**:
```
DCG@k = Σ (rel_i / log₂(i+1))
nDCG@k = DCG@k / IDCG@k
```

**Positional Discount**:
- Rank 1: discount = 1.0
- Rank 2: discount ≈ 0.63
- Rank 5: discount ≈ 0.39

**When to Use**:
- ✅ Reranker evaluation
- ✅ Multi-document QA or synthesis
- ✅ Search systems with graded relevance
- ✅ System-level RAG evaluation
- ❌ Simple RAG debugging (use Hit Rate)
- ❌ Single-answer QA (use MRR)

**Best Practices**:
1. Invest in graded relevance labels (exact/partial/irrelevant)
2. Match k to reranker's output size
3. Pair with Hit Rate to prevent coverage regression
4. Compare against BM25 baseline
5. Normalize relevance scale across annotators

---

### 7. Average Precision (MAP) Skill (`average-precision-metric`)

**Location**: [.agents/skills/retriever-evals/metrics/AVERAGE_PRECISION_SKILL.md](.agents/skills/retriever-evals/metrics/AVERAGE_PRECISION_SKILL.md)

**Trigger Keywords**:
- Average Precision, AP, MAP
- Mean Average Precision
- Rank-aware precision
- Benchmark comparison

**Purpose**: Rank-aware precision metric that considers all relevant results and their positions.

**Formula**:
```
AP = (1/R) × Σ P(k) × 𝟙[doc_k is relevant]
```

Where R is the total number of relevant documents.

**When to Use**:
- ✅ Benchmark comparison across systems
- ✅ Multi-relevant query evaluation
- ✅ Academic/research contexts
- ❌ When only first hit matters (use MRR)
- ❌ When graded relevance is needed (use nDCG)

---

## Metric Selection Decision Tree

```
Start: What are you trying to optimize?

├─ Basic functionality check?
│  └─ Use: Hit Rate
│
├─ How early is the first relevant result?
│  └─ Use: MRR
│
├─ How many relevant results did we find?
│  └─ Use: Recall
│
├─ How pure are the results?
│  └─ Use: Precision
│
├─ How well are results ranked? (binary relevance)
│  └─ Use: Average Precision (MAP)
│
└─ How well are results ranked? (graded relevance)
   └─ Use: nDCG
```

---

## Integration with Evret Framework

These skills guide the implementation and usage of the metrics in the Evret framework:

**Implementation Reference**:
- Metric base class: [src/evret/metrics/base.py](src/evret/metrics/base.py)
- Individual metrics: [src/evret/metrics/](src/evret/metrics/)
- Evaluation orchestrator: [src/evret/evaluation/evaluator.py](src/evret/evaluation/evaluator.py)

**Usage Example**:
```python
from evret.retrievers import QdrantRetriever
from evret.evaluation import Evaluator, Dataset
from evret.metrics import HitRate, MRR, NDCG

# Setup retriever
retriever = QdrantRetriever(
    url="http://localhost:6333",
    collection_name="my_docs"
)

# Load evaluation dataset
dataset = Dataset.from_json("eval_data.json")

# Run evaluation with multiple metrics
evaluator = Evaluator(
    retriever=retriever,
    metrics=[
        HitRate(k=5),    # Binary presence check
        MRR(k=10),       # First-hit optimization
        NDCG(k=5)        # Ranking quality
    ]
)

results = evaluator.evaluate(dataset)
print(results.summary())
```

---

## Metric Comparison Summary

| Metric | Type | Rank-Sensitive | First Hit Only | Best Use Case |
|--------|------|----------------|----------------|---------------|
| Hit Rate | Binary | No (threshold) | No | RAG debugging, presence check |
| Recall | Binary/Graded | No | No | Completeness, coverage |
| Precision | Binary/Graded | No | No | Context quality, noise reduction |
| MRR | Binary | Yes | Yes | Fast hits, single-answer QA |
| nDCG | Graded | Yes | No | Ranking quality, search systems |
| Average Precision | Binary | Yes | No | Benchmark comparison, multi-relevant |

---

## Contributing New Skills

To add a new skill to the Evret agent system:

1. Create a new `.md` file in [.agents/skills/retriever-evals/metrics/](.agents/skills/retriever-evals/metrics/) (for metric skills) or [.agents/skills/retriever-evals/](.agents/skills/retriever-evals/) (for general skills)

2. Follow the frontmatter format:
   ```markdown
   ---
   name: skill-name
   description: Trigger conditions and use cases
   ---
   ```

3. Include these sections:
   - **What is the use of this metric/skill**
   - **Mathematical Formula** (for metrics)
   - **When to use this metric/skill**
   - **Best practices**

4. Update this `Agents.md` file with the new skill documentation

---

## References

- Project prompt: [PROMPT.md](PROMPT.md)
- Project progress: [progress.md](progress.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Main README: [README.md](README.md)
