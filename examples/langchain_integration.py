"""Evaluate a LangChain Qdrant retriever with evret.
pip install "evret[langchain]" fastembed pypdfium langchain-qdrant
"""
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import DocumentExample, QueryExample
from evret.integrations import LangChainRetrieverAdapter
from evret.metrics import HitRate, MRR

def main() -> None:
    docs = PyPDFium2Loader("react_agent_paper.pdf").load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = f"doc_{i}"

    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        path="/tmp/db",
        collection_name="react_paper")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    dataset = EvaluationDataset(
        documents=[
            DocumentExample(chunk.metadata["doc_id"], chunk.page_content)
            for chunk in chunks
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

    results = Evaluator(
        retriever=LangChainRetrieverAdapter(langchain_retriever=retriever),
        metrics=[HitRate(k=5), MRR(k=5)],
    ).evaluate(dataset)
    print(results.summary())

if __name__ == "__main__":
    main()