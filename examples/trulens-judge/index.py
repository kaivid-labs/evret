import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

load_dotenv()

HERE = Path(__file__).parent
DATASET_PATH = HERE / "generated_eval_dataset.json"
COLLECTION_NAME = "western_digital_2026"
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
EMBEDDER = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")


def main() -> None:
    documents = json.loads(DATASET_PATH.read_text(encoding="utf-8"))["documents"]
    texts = [doc["text"] for doc in documents]
    vectors = [list(vector) for vector in EMBEDDER.embed(texts)]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=index,
                vector=vector,
                payload={"doc_id": doc["doc_id"], "text": doc["text"]},
            )
            for index, (doc, vector) in enumerate(zip(documents, vectors))
        ],
    )
    print(f"Indexed {len(documents)} chunks into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
