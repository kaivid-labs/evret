"""Step 1: Index documents into Qdrant.

This script shows how to index documents once and save them to Qdrant.
Run this first to create your vector database.

pip install fastembed pypdfium2 evret[qdrant]
"""
from fastembed import TextEmbedding
import pypdfium2 as pdfium
from qdrant_client import QdrantClient, models


def load_pdf_chunks(pdf_path: str, chunk_size: int = 500) -> list[str]:
    """Load PDF and split into chunks."""
    pdf = pdfium.PdfDocument(pdf_path)
    text = " ".join(page.get_textpage().get_text_range() for page in pdf)
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def index_documents(chunks: list[str], collection_name: str, persist_path: str) -> None:
    """Index documents in Qdrant - this only needs to be done once."""
    print(f"Indexing {len(chunks)} chunks...")

    # Initialize embedding model for indexing
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = [list(vector) for vector in model.embed(chunks)]

    # Create Qdrant client and collection
    client = QdrantClient(path=persist_path)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE
        ),
    )

    # Index all chunks with metadata
    points = [
        models.PointStruct(
            id=i,
            vector=vec,
            payload={"doc_id": f"doc_{i}", "text": chunk}
        )
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)

    print(f"✓ Indexed {len(points)} documents to collection '{collection_name}'")
    print(f"✓ Persisted to: {persist_path}")


def main():
    """Index the documents."""
    # Configuration
    COLLECTION_NAME = "react_paper"
    PERSIST_PATH = "/tmp/qdrant_demo"  # or use Qdrant Cloud URL
    PDF_PATH = "react_agent_paper.pdf"

    # Load and index
    chunks = load_pdf_chunks(PDF_PATH)
    index_documents(chunks, COLLECTION_NAME, PERSIST_PATH)

    print("\n📦 Indexing complete! Now run 'qdrant_2_evaluate.py' to evaluate.")


if __name__ == "__main__":
    main()
