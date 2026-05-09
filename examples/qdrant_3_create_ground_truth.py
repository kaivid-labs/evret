"""Step 3 (Optional): Create expected answer labels.

This script shows how to create practical evaluation labels.
You need this before running evaluation!

Three approaches:
1. Manual annotation (write the expected answer text for each query)
2. Load from benchmark or QA dataset
3. Use LLM-assisted labeling (but validate!)
"""
import json
from qdrant_client import QdrantClient

from evret import EvaluationDataset
from evret.evaluation.dataset import DocumentExample, QueryExample


def inspect_documents(collection_name: str, persist_path: str, num_docs: int = 10):
    """Inspect indexed documents to understand what's in them.

    Run this first to see what documents exist before creating ground truth.
    """
    client = QdrantClient(path=persist_path)

    # Retrieve some sample documents
    points = client.scroll(
        collection_name=collection_name,
        limit=num_docs,
        with_payload=True,
        with_vectors=False,
    )[0]

    print("=" * 80)
    print(f"SAMPLE DOCUMENTS FROM '{collection_name}'")
    print("=" * 80)

    for point in points:
        doc_id = point.payload.get("doc_id", point.id)
        text = point.payload.get("text", "")
        preview = text[:200] + "..." if len(text) > 200 else text

        print(f"\n📄 {doc_id}")
        print(f"   {preview}")
        print("-" * 80)


def create_ground_truth_manually() -> EvaluationDataset:
    """Approach 1: Manual annotation.

    After inspecting documents, manually write the answer text that a good
    retrieval result should contain.
    """
    # Step 1: Define your test queries
    test_queries = [
        "What is the ReAct framework?",
        "How does reasoning help in decision making?",
        "What are the benefits of combining reasoning and acting?",
    ]

    # Step 2: For EACH query, write the expected fact or supporting answer text

    print("\n" + "=" * 80)
    print("MANUAL ANNOTATION WORKFLOW")
    print("=" * 80)
    print("\n1. For each query, search your Qdrant collection")
    print("2. Manually read the top-k results")
    print("3. Write the expected answer text")
    print("4. Create QueryExample with expected_answers\n")

    queries = [
        QueryExample(
            query_id="q1",
            query_text="What is the ReAct framework?",
            expected_answers=["ReAct combines reasoning traces and task-specific actions."],
        ),
        QueryExample(
            query_id="q2",
            query_text="How does reasoning help in decision making?",
            expected_answers=["Reasoning helps the model plan, track progress, and decide the next action."],
        ),
    ]

    # Documents list (just references, actual content is in Qdrant)
    documents = [
        DocumentExample(doc_id=f"doc_{i}", text="")
        for i in range(100)
    ]

    return EvaluationDataset(documents=documents, queries=queries)


def load_from_benchmark() -> EvaluationDataset:
    """Approach 2: Load from existing benchmark or QA datasets.

    If your benchmark has gold answers, load those answers into expected_answers.
    """
    # Example: Loading from a benchmark dataset (pseudo-code)
    # from datasets import load_dataset
    # dataset = load_dataset("natural_questions")

    queries = [
        QueryExample(
            query_id="benchmark_q1",
            query_text="what is a corporation",
            expected_answers=["A corporation is a legal entity that is separate from its owners."],
        ),
    ]

    documents = [
        DocumentExample(doc_id="doc_1234", text="A corporation is..."),
        DocumentExample(doc_id="doc_5678", text="Corporate structure..."),
    ]

    return EvaluationDataset(documents=documents, queries=queries)


def save_ground_truth(dataset: EvaluationDataset, output_path: str):
    """Save expected answer labels to JSON for reuse."""
    data = {
        "queries": [
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "expected_answers": q.expected_answers,
            }
            for q in dataset.queries
        ],
        "documents": [
            {
                "doc_id": d.doc_id,
                "text": d.text,
            }
            for d in dataset.documents
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Ground truth saved to: {output_path}")


def load_ground_truth(input_path: str) -> EvaluationDataset:
    """Load expected answer labels from JSON."""
    with open(input_path) as f:
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
        DocumentExample(doc_id=d["doc_id"], text=d["text"])
        for d in data["documents"]
    ]

    return EvaluationDataset(documents=documents, queries=queries)


def main():
    """Demonstrate expected answer creation workflow."""
    COLLECTION_NAME = "react_paper"
    PERSIST_PATH = "/tmp/qdrant_demo"

    print("EXPECTED ANSWER CREATION WORKFLOW\n")

    # Step 1: Inspect documents to understand what's indexed
    print("Step 1: Inspecting indexed documents...\n")
    inspect_documents(COLLECTION_NAME, PERSIST_PATH, num_docs=5)

    # Step 2: Create expected answers
    print("\n\nStep 2: Creating expected answer annotations...\n")
    dataset = create_ground_truth_manually()

    # Step 3: Save for reuse
    print("\nStep 3: Saving expected answers...\n")
    save_ground_truth(dataset, "ground_truth.json")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review the expected answers in 'ground_truth.json'")
    print("2. Validate that expected answers are correct and specific")
    print("3. Use this ground truth in 'qdrant_2_evaluate.py'")
    print("\n⚠️  RAG Evaluation Note:")
    print("   Adjust TokenOverlapJudge min_tokens based on chunk size:")
    print("   - Default min_tokens=30 (good for ~500 token chunks)")
    print("   - For ~1000 token chunks: min_tokens=50-100")
    print("   - For ~200 token chunks: min_tokens=10")
    print("\nYour evaluation is only as good as your expected answers.")


if __name__ == "__main__":
    main()
