"""Step 2: Evaluate retrieval using the indexed Qdrant collection.

This script shows how to use a PREVIOUSLY INDEXED Qdrant collection for evaluation.
The query_encoder is initialized here at inference time - separate from indexing.

Prerequisites:
1. Run 'qdrant_1_index.py' first to create the index
2. Create expected answer labels (see 'qdrant_3_create_ground_truth.py')
"""
import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import DocumentExample, QueryExample
from evret.judges import TokenOverlapJudge
from evret.metrics import HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever


def create_query_encoder():
    """Create the query encoder for inference.

    This initializes the embedding model that will be used to encode queries
    at retrieval time. This is SEPARATE from the indexing process.
    """
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def encode(query: str) -> list[float]:
        """Encode a single query into a vector."""
        return list(next(model.embed([query])))

    return encode


def create_retriever(collection_name: str, persist_path: str) -> QdrantRetriever:
    """Connect to existing Qdrant collection and create retriever.

    This connects to an ALREADY INDEXED collection. The query_encoder
    is used at inference time to encode incoming queries.
    """
    # Connect to the existing Qdrant database
    client = QdrantClient(path=persist_path)

    # Create query encoder for inference
    query_encoder = create_query_encoder()

    # Create retriever with the query encoder
    return QdrantRetriever(
        collection_name=collection_name,
        client=client,
        query_encoder=query_encoder,  # This will be called during retrieve()
    )


def load_ground_truth(ground_truth_path: str) -> EvaluationDataset:
    """Load expected answer labels from JSON file.

    The file should contain queries and expected answer text snippets.
    See 'qdrant_3_create_ground_truth.py' for how to create this file.
    """
    with open(ground_truth_path) as f:
        data = json.load(f)

    queries = [
        QueryExample(
            query_id=q["query_id"],
            query_text=q["query_text"],
            expected_answers=q.get("expected_answers", []),
        )
        for q in data["queries"]
    ]

    documents = [
        DocumentExample(doc_id=d["doc_id"], text=d.get("text", ""))
        for d in data["documents"]
    ]

    return EvaluationDataset(documents=documents, queries=queries)


def create_demo_dataset() -> EvaluationDataset:
    """Create a minimal demo dataset for illustration.

    WARNING: These expected answers are generic examples.
    This is just for demonstration. For real evaluation, use load_ground_truth().
    """
    print("WARNING: Using demo dataset with generic expected answers!")
    print("   For real evaluation, create expected answers using:")
    print("   python examples/qdrant_3_create_ground_truth.py\n")

    # Documents (just references, actual content is in Qdrant)
    documents = [
        DocumentExample(doc_id=f"doc_{i}", text="")
        for i in range(100)
    ]

    # Demo queries for illustration only
    queries = [
        QueryExample(
            query_id="q1",
            query_text="What is the ReAct framework?",
            expected_answers=["ReAct combines reasoning traces and actions for language agents."],
        ),
    ]

    return EvaluationDataset(documents=documents, queries=queries)


def main():
    """Run evaluation on the indexed collection."""
    # Configuration (must match indexing script)
    COLLECTION_NAME = "react_paper"
    PERSIST_PATH = "/tmp/qdrant_demo"
    GROUND_TRUTH_PATH = "ground_truth.json"

    print("📊 Starting evaluation...\n")

    # Connect to indexed collection
    retriever = create_retriever(COLLECTION_NAME, PERSIST_PATH)

    # Load evaluation dataset
    if Path(GROUND_TRUTH_PATH).exists():
        print(f"✓ Loading ground truth from {GROUND_TRUTH_PATH}")
        dataset = load_ground_truth(GROUND_TRUTH_PATH)
    else:
        print(f"✗ Ground truth file not found: {GROUND_TRUTH_PATH}")
        print(f"  Using demo dataset instead (NOT verified!)\n")
        dataset = create_demo_dataset()

    # Configure metrics and judge
    metrics = [
        HitRate(k=5),
        Precision(k=5),
        Recall(k=5),
        MRR(k=5),
        NDCG(k=5),
    ]
    judge = TokenOverlapJudge(min_tokens=10, overlap_ratio=0.5)

    # Run evaluation
    evaluator = Evaluator(retriever=retriever, metrics=metrics, judge=judge)
    results = evaluator.evaluate(dataset)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(results.summary())


if __name__ == "__main__":
    main()
