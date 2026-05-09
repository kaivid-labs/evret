"""
Create evaluation dataset when you don't have one.
The judge automatically determines if retrieved docs match your expected answers.
You only provide the answer text you expect.
"""
from pathlib import Path
import json
from evret import EvaluationDataset
from evret.evaluation.dataset import DocumentExample, QueryExample

def create_dataset_manually():
    """Create evaluation dataset by providing expected answer text."""
    documents = [
        DocumentExample(
            doc_id="doc1",
            text="Python packages are installed using pip install package-name.",
            metadata={"source": "guide.md"}
        ),
        DocumentExample(
            doc_id="doc2",
            text="Virtual environments are created with python -m venv env_name.",
            metadata={"source": "guide.md"}
        ),
        DocumentExample(
            doc_id="doc3",
            text="Run tests using pytest in the terminal.",
            metadata={"source": "guide.md"}
        ),
    ]

    queries = [
        QueryExample(
            query_id="q1",
            query_text="How do I install packages in Python?",
            expected_answers=["pip install package-name"]
        ),
        QueryExample(
            query_id="q2",
            query_text="How to create a virtual environment?",
            expected_answers=["python -m venv"]
        ),
        QueryExample(
            query_id="q3",
            query_text="What command runs tests?",
            expected_answers=["pytest"]
        ),
    ]
    return EvaluationDataset(queries=queries, documents=documents)


def save_dataset(dataset: EvaluationDataset, filename: str):
    """Save dataset to JSON file."""
    output = {
        "queries": [
            {
                "query_id": q.query_id,
                "query_text": q.query_text,
                "expected_answers": q.expected_answers
            }
            for q in dataset.queries
        ],
        "documents": [
            {
                "doc_id": d.doc_id,
                "text": d.text,
                "metadata": d.metadata
            }
            for d in dataset.documents
        ]
    }
    Path(filename).write_text(json.dumps(output, indent=2))

if __name__ == "__main__":
    dataset = create_dataset_manually()
    save_dataset(dataset, "my_eval_dataset.json")
    print("Saved my_eval_dataset.json")