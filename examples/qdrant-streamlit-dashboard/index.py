"""Index documents into Qdrant.
pip install fastembed pypdfium2 "evret[qdrant]" python-dotenv 
"""
import os
from pathlib import Path
import pypdfium2 as pdfium
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "react_paper"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
PDF_PATH = Path(__file__).resolve().parent.parent / "react_agent_paper.pdf"

def load_pdf_chunks(pdf_path: Path, chunk_size: int = 500) -> list[str]:
    pdf = pdfium.PdfDocument(pdf_path)
    text = " ".join(page.get_textpage().get_text_range() for page in pdf)
    words = text.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_client() -> QdrantClient:
    kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return QdrantClient(**kwargs)

def index_documents(chunks: list[str], collection_name: str) -> None:
    if not chunks:
        raise ValueError("no chunks were loaded from the PDF")

    print(f"Indexing {len(chunks)} chunks into {collection_name!r}")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    vectors = [list(vector) for vector in model.embed(chunks)]

    client = create_client()
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384, distance=models.Distance.COSINE,
        ),
    )
    points = [
        models.PointStruct(
            id=index,
            vector=vector,
            payload={"doc_id": f"doc_{index}", "text": chunk},
        )
        for index, (vector, chunk) in enumerate(zip(vectors, chunks))
    ]
    client.upsert(collection_name=collection_name, points=points)
    print("Indexing complete")

def main() -> None:
    chunks = load_pdf_chunks(PDF_PATH)
    index_documents(chunks, COLLECTION_NAME)

if __name__ == "__main__":
    main()