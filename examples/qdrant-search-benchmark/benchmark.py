from __future__ import annotations

import json

from qdrant_client import models

from common import (
    RESULTS_PATH,
    category_by_query_id,
    get_client,
    load_raw_dataset,
)
from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import QueryExample
from evret.metrics import MRR, NDCG, HitRate, Precision, Recall
from retrievers import (
    Bm25Retriever,
    ColbertRetriever,
    DenseRetriever,
    HybridColbertRerankRetriever,
    HybridFusionRetriever,
    HybridNoFusionRetriever,
    SpladeRetriever,
)

K = 5
PRIMARY_METRIC = f"ndcg@{K}"


def build_techniques(client):
    return {
        "dense": DenseRetriever(client),
        "bm25 (lexical)": Bm25Retriever(client),
        "splade (learned sparse)": SpladeRetriever(client),
        "hybrid (no fusion)": HybridNoFusionRetriever(client),
        "hybrid (RRF)": HybridFusionRetriever(client, models.Fusion.RRF),
        "hybrid (DBSF)": HybridFusionRetriever(client, models.Fusion.DBSF),
        "colbert": ColbertRetriever(client),
        "hybrid + colbert rerank": HybridColbertRerankRetriever(client),
    }


def answerable_queries(raw) -> list[QueryExample]:
    return [
        QueryExample(
            query_id=q["query_id"],
            query_text=q["query_text"],
            expected_doc_ids=q["expected_doc_ids"],
        )
        for q in raw["queries"]
        if q.get("expected_doc_ids")
    ]


def metrics():
    return [HitRate(k=K), Precision(k=K), Recall(k=K), MRR(k=K), NDCG(k=K)]


def evaluate(retriever, queries) -> dict[str, float]:
    dataset = EvaluationDataset(queries=queries)
    return Evaluator(retriever=retriever, metrics=metrics()).evaluate(dataset).summary()


def print_table(title, columns, rows) -> None:
    widths = [len(col) for col in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print(f"\n{title}")
    print("  ".join(col.ljust(widths[i]) for i, col in enumerate(columns)))
    print("  ".join("-" * widths[i] for i in range(len(columns))))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def main() -> None:
    raw = load_raw_dataset()
    queries = answerable_queries(raw)
    if not queries:
        raise SystemExit("No answerable queries in the dataset. Re-run generate_dataset.py.")

    categories = category_by_query_id(raw)
    queries_by_category: dict[str, list[QueryExample]] = {}
    for query in queries:
        queries_by_category.setdefault(categories.get(query.query_id, "unknown"), []).append(query)

    client = get_client()
    techniques = build_techniques(client)

    print(f"Evaluating {len(techniques)} techniques on {len(queries)} answerable queries (k={K})\n")

    overall: dict[str, dict[str, float]] = {}
    per_category: dict[str, dict[str, float]] = {}
    for name, retriever in techniques.items():
        overall[name] = evaluate(retriever, queries)
        per_category[name] = {
            category: evaluate(retriever, subset)[f"hit_rate@{K}"]
            for category, subset in sorted(queries_by_category.items())
        }
        print(f"  done: {name}")

    metric_names = [f"hit_rate@{K}", f"precision@{K}", f"recall@{K}", f"mrr@{K}", f"ndcg@{K}"]
    ranking = sorted(techniques, key=lambda n: overall[n][PRIMARY_METRIC], reverse=True)

    print_table(
        f"LEADERBOARD (ranked by {PRIMARY_METRIC})",
        ["#", "technique", *metric_names],
        [
            [str(rank), name, *[f"{overall[name][m]:.3f}" for m in metric_names]]
            for rank, name in enumerate(ranking, start=1)
        ],
    )

    category_names = sorted(queries_by_category)
    print_table(
        f"HIT RATE@{K} BY QUERY CATEGORY",
        ["technique", *category_names],
        [[name, *[f"{per_category[name][c]:.3f}" for c in category_names]] for name in ranking],
    )

    winner = ranking[0]
    print(f"\nBest technique by {PRIMARY_METRIC}: {winner} ({overall[winner][PRIMARY_METRIC]:.3f})")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "k": K,
                "primary_metric": PRIMARY_METRIC,
                "query_count": len(queries),
                "category_counts": {c: len(q) for c, q in sorted(queries_by_category.items())},
                "overall": overall,
                "hit_rate_by_category": per_category,
                "ranking": ranking,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
