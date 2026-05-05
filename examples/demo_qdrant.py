"""Minimal example of evaluating a Qdrant retriever with evret.
pip install fastembed pypdfium2 evret[qdrant]
"""
from fastembed import TextEmbedding
import pypdfium2 as pdfium
from qdrant_client import QdrantClient, models

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import DocumentExample, QueryExample
from evret.judges import TokenOverlapJudge
from evret.metrics import HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever

def load_pdf_chunks(pdf_path, chunk_size: int = 500) -> list[str]:
    """Load PDF and split into chunks."""
    pdf = pdfium.PdfDocument(pdf_path)
    text = " ".join(page.get_textpage().get_text_range() for page in pdf)
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_dataset(chunks: list[str]) -> EvaluationDataset:
    """Create a simple evaluation dataset with document IDs."""
    documents = [DocumentExample(doc_id=f"doc_{i}", text=chunk) for i, chunk in enumerate(chunks)]
    queries = [
        QueryExample(
            query_id="q1",
            query_text="What is the ReAct framework?",
            relevant_doc_ids=["doc_0", "doc_1"],
        ),
        QueryExample(
            query_id="q2",
            query_text="How does reasoning help in decision making?",
            relevant_doc_ids=["doc_2", "doc_3"],
        ),
    ]
    return EvaluationDataset(documents=documents, queries=queries)


def index_documents(chunks: list[str], collection_name: str) -> QdrantRetriever:
    """Index documents in Qdrant and return retriever."""
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = [list(vector) for vector in model.embed(chunks)]

    client = QdrantClient(path="/tmp/app") # or use https://cloud.qdrant.io/. Create free cluster and get endpoint and API key
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(id=i, vector=vec, payload={"doc_id": f"doc_{i}", "text": chunk})
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)

    return QdrantRetriever(
        collection_name=collection_name,
        client=client,
        query_encoder=lambda query: list(next(model.embed([query]))),
    )

def main():
    """Run the evaluation."""
    # Load PDF and create chunks
    chunks = load_pdf_chunks("react_agent_paper.pdf")
    # Create dataset and index documents
    dataset = create_dataset(chunks)
    retriever = index_documents(chunks, collection_name="react_paper")
    print("Indexing done")
    # Run evaluation
    metrics = [HitRate(k=5), Precision(k=5), Recall(k=5), MRR(k=5), NDCG(k=5)]
    judge = TokenOverlapJudge(min_tokens=10, overlap_ratio=0.5)
    results = Evaluator(retriever=retriever, metrics=metrics, judge=judge).evaluate(dataset)
    # Print results
    print(results.summary())

if __name__ == "__main__":
    main()