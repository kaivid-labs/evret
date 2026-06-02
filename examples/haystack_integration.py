"""Evaluate a Haystack Qdrant retriever with evret.

Install the example dependencies:
pip install "evret[haystack]" qdrant-haystack fastembed-haystack pypdfium2
"""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium
from haystack import Document
from haystack_integrations.components.embedders.fastembed import (
    FastembedDocumentEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import DocumentExample, QueryExample
from evret.integrations import HaystackRetrieverAdapter
from evret.metrics import HitRate, MRR


PDF_PATH = Path(__file__).with_name("react_agent_paper.pdf")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class HaystackQdrantTextRetriever:
    """Expose Haystack's embedding retriever through a text-query run method."""

    def __init__(self, document_store: QdrantDocumentStore, model: str = MODEL_NAME) -> None:
        self.text_embedder = FastembedTextEmbedder(
            model=model,
            prefix="Represent this sentence for searching relevant passages: ",
        )
        self.retriever = QdrantEmbeddingRetriever(document_store=document_store)
        self.text_embedder.warm_up()

    def run(self, query: str, top_k: int = 5) -> dict[str, list[Document]]:
        embedding = self.text_embedder.run(text=query)["embedding"]
        return self.retriever.run(query_embedding=embedding, top_k=top_k)


def load_pdf_chunks(path: Path, chunk_size: int = 2000, chunk_overlap: int = 200) -> list[Document]:
    text = "\n".join(page.get_textpage().get_text_range() for page in pdfium.PdfDocument(path))
    chunks: list[Document] = []
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        content = text[start:end].strip()
        if content:
            chunks.append(
                Document(
                    id=f"doc_{index}",
                    content=content,
                    meta={"doc_id": f"doc_{index}", "source": path.name},
                )
            )
            index += 1
        if end == len(text):
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def index_documents(documents: list[Document]) -> QdrantDocumentStore:
    document_store = QdrantDocumentStore(
        ":memory:",
        recreate_index=True,
        return_embedding=True,
        wait_result_from_api=True,
        embedding_dim=EMBEDDING_DIM,
    )
    document_embedder = FastembedDocumentEmbedder(model=MODEL_NAME)
    document_embedder.warm_up()
    embedded_documents = document_embedder.run(documents=documents)["documents"]
    document_store.write_documents(embedded_documents)
    return document_store


def build_dataset(documents: list[Document]) -> EvaluationDataset:
    return EvaluationDataset(
        documents=[
            DocumentExample(str(document.id), document.content or "")
            for document in documents
        ],
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


def main() -> None:
    documents = load_pdf_chunks(PDF_PATH)
    document_store = index_documents(documents)
    haystack_retriever = HaystackQdrantTextRetriever(document_store=document_store)

    results = Evaluator(
        retriever=HaystackRetrieverAdapter(haystack_retriever=haystack_retriever),
        metrics=[HitRate(k=5), MRR(k=5)],
    ).evaluate(build_dataset(documents))
    print(results.summary())


if __name__ == "__main__":
    main()
