# Getting Started

## Install

```bash
pip install evret
```

For local development:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

For docs development:

```bash
uv pip install -e ".[docs]"
```

## First Metric Run

```python
from evret.metrics import HitRate, Recall, Precision, MRR, NDCG, AveragePrecision

retrieved = [
    ["d1", "d3", "d2", "d5"],
    ["d10", "d7", "d8", "d9"],
]
relevant = [
    {"d1", "d2"},
    {"d9"},
]

metrics = [
    HitRate(k=3),
    Recall(k=3),
    Precision(k=3),
    MRR(k=3),
    NDCG(k=3),
    AveragePrecision(k=3),
]

for metric in metrics:
    print(metric.name, metric.score(retrieved, relevant))
```

## First Dataset Evaluation

```python
from evret import EvaluationDataset, Evaluator, HitRate, MRR

# Provide your own retriever implementation or adapter
retriever = my_retriever

dataset = EvaluationDataset.from_json("examples/eval_data.json")
evaluator = Evaluator(
    retriever=retriever,
    metrics=[HitRate(k=4), MRR(k=4)],
)

results = evaluator.evaluate(dataset)
print(results.summary())
results.to_json("results.json")
results.to_csv("results.csv")
```

Each query in `examples/eval_data.json` stores the gold supporting chunks that should appear in the retrieved top-4 contexts.

## Build Docs Locally

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000`.
