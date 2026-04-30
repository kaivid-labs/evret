from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from evret import EvaluationDataset, Evaluator
from evret.judges import TokenOverlapJudge
from evret.metrics import AveragePrecision, HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever


def index() -> tuple[EvaluationDataset, QdrantRetriever]:
    base_dir = Path(__file__).resolve().parent
    dataset = EvaluationDataset.from_json(base_dir / "eval_data.json")
    collection_name = "evret_eval"
    texts = [document.text for document in dataset.documents]
    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=str(base_dir / ".fastembed_cache"),
    )
    vectors = [list(vector) for vector in model.embed(texts)]

    client = QdrantClient(path="db")
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=vector,
            payload={"doc_id": document.doc_id, "text": document.text, **document.metadata},
        )
        for index, (vector, document) in enumerate(zip(vectors, dataset.documents), start=1)
    ]
    client.upsert(collection_name=collection_name, points=points)

    retriever = QdrantRetriever(
        collection_name=collection_name,
        client=client,
        query_encoder=lambda query: list(next(model.embed([query]))),
    )
    return dataset, retriever


def run_eval() -> None:
    dataset, retriever = index()
    metrics = [HitRate(k=4), Recall(k=4), Precision(k=4), MRR(k=4), NDCG(k=4), AveragePrecision(k=4)]
    judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)
    results = Evaluator(retriever=retriever, metrics=metrics, judge=judge).evaluate(dataset)
    base_dir = Path(__file__).resolve().parent
    results.to_json(base_dir / "results.json")
    results.to_csv(base_dir / "results.csv")
    print(results.summary())


if __name__ == "__main__":
    run_eval()
