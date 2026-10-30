import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import QueryExample
from evret.judges import TruLensJudge
from evret.judges.base import JudgmentContext
from evret.metrics import NDCG, HitRate, MRR, Recall
from evret.retrievers import QdrantRetriever

load_dotenv()

HERE = Path(__file__).parent
DATASET_PATH = HERE / "generated_eval_dataset.json"
COLLECTION_NAME = "western_digital_2026"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
EMBEDDER = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")


def encode(query: str) -> list[float]:
    return list(next(EMBEDDER.embed([query])))


def load_answerable_dataset(path: Path) -> EvaluationDataset:
    raw = EvaluationDataset.from_json(path)
    queries = [
        QueryExample(
            query_id=query.query_id,
            query_text=query.query_text,
            expected_answers=query.expected_answers,
        )
        for query in raw.queries
        if query.expected_answers
    ]
    return EvaluationDataset(queries=queries, documents=raw.documents)


def main() -> None:
    dataset = load_answerable_dataset(DATASET_PATH)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    retriever = QdrantRetriever(
        collection_name=COLLECTION_NAME,
        client=client,
        query_encoder=encode,
    )

    judge = TruLensJudge(
        provider="openai",
        model="gpt-6-luna",
        metric="context_relevance",
        threshold=0.5,
    )

    sample = dataset.queries[0]
    hits = retriever.retrieve(sample.query_text, k=1)
    if hits:
        score = judge.score(
            JudgmentContext(
                query=sample.query_text,
                expected_text="",
                retrieved_text=hits[0].metadata.get("text", ""),
            )
        )
        print(f"context_relevance for top hit of q='{sample.query_text}': {score:.3f}")

    results = Evaluator(
        retriever=retriever,
        metrics=[HitRate(k=3), MRR(k=3), Recall(k=3), NDCG(k=3)],
        judge=judge,
    ).evaluate(dataset)
    print(results.summary())


if __name__ == "__main__":
    main()
