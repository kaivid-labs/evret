"""Index documents into Qdrant.
pip install fastembed pypdfium2 evret[qdrant]
"""
from fastembed import TextEmbedding
import pypdfium2 as pdfium
from qdrant_client import QdrantClient, models

QDRANT_API_KEY = "ey******" # get the API key from cloud.qdrant.io
QDRANT_URL = "https://***cloud.qdrant.io:6333" # once you create a free cluster, you will be find the Endpoint URL and API key

def load_pdf_chunks(pdf_path: str, chunk_size: int = 500) -> list[str]:
    """Load PDF and split into chunks."""
    pdf = pdfium.PdfDocument(pdf_path)
    text = " ".join(page.get_textpage().get_text_range() for page in pdf)
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def index_documents(chunks: list[str], collection_name: str) -> None:
    """Index documents in Qdrant - this only needs to be done once."""
    print(f"Indexing {len(chunks)} chunks...")

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = [list(vector) for vector in model.embed(chunks)]

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=len(vectors[0]),
            distance=models.Distance.COSINE
        )
    )
    points = [
        models.PointStruct(
            id=i,
            vector=vec,
            payload={"doc_id": f"doc_{i}", "text": chunk}
        )
        for i, (vec, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)
    print(f"indexing done")

def main():
    """Index the documents."""
    COLLECTION_NAME = "react_paper"
    PDF_PATH = "../react_agent_paper.pdf" # replace based on your file path
    chunks = load_pdf_chunks(PDF_PATH)
    index_documents(chunks, COLLECTION_NAME)

if __name__ == "__main__":
    main()