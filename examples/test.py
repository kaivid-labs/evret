from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from evret import EvaluationDataset, Evaluator, QueryExample
from evret.metrics import AveragePrecision, HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever

def index() -> QdrantRetriever:
    collection_name = "evret_eval"
    docs = [
        {"doc_id": "doc_python", "document": "Python is a programming language used for AI and data."},
        {"doc_id": "doc_qdrant", "document": "Qdrant is a vector database for semantic retrieval."},
        {"doc_id": "doc_sql", "document": "SQL databases store structured records in relational tables."},
        {"doc_id": "doc_rag", "document": "RAG combines retrieval with generation for grounded answers."},
    ]
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = [list(v) for v in model.embed([d["document"] for d in docs])]
    
    client = QdrantClient(path="db")
    client.create_collection(collection_name=collection_name, 
                             vectors_config=models.VectorParams(size=len(vectors[0]), 
                                                                distance=models.Distance.COSINE))
    points = [
        models.PointStruct(id=i, vector=vector, payload=doc)
        for i, (vector, doc) in enumerate(zip(vectors, docs), start=1)
    ]
    client.upsert(collection_name=collection_name, points=points)

    retriever = QdrantRetriever(collection_name=collection_name, client=client, query_encoder=lambda q: list(next(model.embed([q]))))
    return retriever

def run_eval() -> None:
    dataset = EvaluationDataset(
        queries=[
            QueryExample(query_id="q1", query_text="vector retrieval with qdrant", relevant_doc_ids=["doc_qdrant", "doc_rag"]),
            QueryExample(query_id="q2", query_text="what is python used for", relevant_doc_ids=["doc_python"]),
            QueryExample(query_id="q3", query_text="relational table records", relevant_doc_ids=["doc_sql"]),
        ]
    )
    metrics = [HitRate(k=3), Recall(k=3), Precision(k=3), MRR(k=3), NDCG(k=3), AveragePrecision(k=3)]
    print(Evaluator(retriever=index(), metrics=metrics).evaluate(dataset).summary())

if __name__ == "__main__":
    run_eval()