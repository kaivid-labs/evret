"""Evaluate retrieval using the indexed Qdrant collection.
Prerequisites:
1. Run 'index.py' first to create the index
"""
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import QueryExample
from evret.judges import TokenOverlapJudge
from evret.metrics import HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever

QDRANT_API_KEY = "ey******" # same as used in index.py
QDRANT_URL = "https://***cloud.qdrant.io:6333" # same as used in index.py
QUERY_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def encode_query(query: str) -> list[float]:
    """Encode a single query into a vector."""
    return list(next(QUERY_MODEL.embed([query])))

def create_retriever(collection_name: str) -> QdrantRetriever:
    """Connect to existing Qdrant collection and create retriever"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    return QdrantRetriever(
        collection_name=collection_name,
        client=client,
        query_encoder=encode_query,
    )

def create_dataset() -> EvaluationDataset:
    """Create a minimal demo dataset for illustration."""
    dataset = EvaluationDataset(
        queries=[
            QueryExample(
                query_id="q1",
                query_text="What is the ReAct framework?",
                expected_answers=[
                    "ReAct combines reasoning traces with task-specific actions."
                ],
            ),
            QueryExample(
                query_id="q2",
                query_text="How does ReAct improve performance over reasoning-only or acting-only baselines?",
                expected_answers=[
                    "Interleaving reasoning and action helps the model gather external information and reduces hallucinations."
                ],
            ),
            QueryExample(
                query_id="q3",
                query_text="Why are reasoning traces useful in ReAct agents?",
                expected_answers=[
                    "Reasoning traces help the model induce, track, and update action plans while improving interpretability."
                ],
            ),
        ],
    )
    return dataset

def inference():
    """Run evaluation on the indexed collection."""
    COLLECTION_NAME = "react_paper"
    retriever = create_retriever(COLLECTION_NAME)
    dataset = create_dataset()

    # Debug: retriever response
    print("\n=== DEBUGGING RETRIEVAL ===\n")
    for query_example in dataset.queries:
        print(f"Query: {query_example.query_text}")
        print(f"Expected Answer: {query_example.expected_answers[0][:100]}...")
        # Retrieve documents
        retrieved_docs = retriever.retrieve(query_example.query_text, k=4)
        print(f"\nRetrieved {len(retrieved_docs)} documents:")
        for doc in retrieved_docs:
            print(doc)
        print("\n" + "="*50 + "\n")
        
    metrics = [
        HitRate(k=4),Precision(k=4),
        Recall(k=4),MRR(k=4), NDCG(k=4),
    ]
    judge = TokenOverlapJudge(min_tokens=5, overlap_ratio=0.5)
    evaluator = Evaluator(retriever=retriever, metrics=metrics, judge=judge)

    results = evaluator.evaluate(dataset)
    print(results.summary())

if __name__ == "__main__":
    inference()