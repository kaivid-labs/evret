"""
pip install streamlit pandas openai
streamlit run run_evals_ui.py
"""
import os
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

from evret import EvaluationDataset, Evaluator
from evret.evaluation.dataset import QueryExample
from evret.judges import LLMJudge
from evret.metrics import HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import QdrantRetriever

from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "react_paper"
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY","")
MODEL_NAME = "gpt-5.4-mini"

@st.cache_resource(show_spinner=False)
def query_model() -> TextEmbedding:
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def encode_query(query: str) -> list[float]:
    return list(next(query_model().embed([query])))

def create_dataset() -> EvaluationDataset:
    return EvaluationDataset(
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
                query_text=(
                    "How does ReAct improve performance over reasoning-only or "
                    "acting-only baselines?"
                ),
                expected_answers=[
                    "Interleaving reasoning and action helps the model gather external "
                    "information and reduces hallucinations."
                ],
            ),
            QueryExample(
                query_id="q3",
                query_text="Why are reasoning traces useful in ReAct agents?",
                expected_answers=[
                    "Reasoning traces help the model induce, track, and update action "
                    "plans while improving interpretability."
                ],
            ),
        ],
    )

def dataset_rows(dataset: EvaluationDataset) -> list[dict[str, Any]]:
    return [
        {
            "query_id": query.query_id,
            "query_text": query.query_text,
            "expected_answers": "\n".join(query.expected_answers),
        }
        for query in dataset.queries
    ]

def metric_rows(metric_scores: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {"metric": metric_name, "score": round(score, 4)}
        for metric_name, score in sorted(metric_scores.items())
    ]

def result_rows(results) -> list[dict[str, Any]]:
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "doc_id": result.doc_id,
                "score": round(result.score, 4),
                "text": result.metadata.get("text", ""),
            })
    return rows

def create_retriever(collection_name: str, qdrant_url: str, qdrant_api_key: str):
    client_kwargs: dict[str, Any] = {"url": qdrant_url}
    if qdrant_api_key:
        client_kwargs["api_key"] = qdrant_api_key

    client = QdrantClient(**client_kwargs)
    return QdrantRetriever(
        collection_name=collection_name,
        client=client,
        query_encoder=encode_query,
    )

def run_evaluation(dataset,retriever,k=2):
    metrics = [
        HitRate(k=k),
        Precision(k=k),
        Recall(k=k),
        MRR(k=k),
        NDCG(k=k),
    ]
    judge = LLMJudge(
        provider="openai",
        model=MODEL_NAME,
    )
    evaluator = Evaluator(retriever=retriever, metrics=metrics, judge=judge)
    return evaluator.evaluate(dataset), judge

def render_metric_cards(metric_scores: dict[str, float]) -> None:
    columns = st.columns(len(metric_scores))
    for column, (metric_name, score) in zip(columns, sorted(metric_scores.items())):
        column.metric(metric_name, f"{score:.3f}")

def main() -> None:
    st.set_page_config(page_title="Evret Evaluation",layout="wide")
    st.title("Evret Evaluations Dashboard")
    dataset = create_dataset()

    st.subheader("Evaluation Dataset")
    with st.expander("Show dataset", expanded=False):
        st.dataframe(dataset_rows(dataset), hide_index=True, width="stretch")
        st.json({"queries": [asdict(query) for query in dataset.queries]})

    try:
        retriever = create_retriever(
            collection_name=COLLECTION_NAME,
            qdrant_url=QDRANT_URL,
            qdrant_api_key=QDRANT_API_KEY,
        )
        with st.spinner("Running retrieval and LLM judgments..."):
            results, judge = run_evaluation(
                dataset=dataset,
                retriever=retriever
            )
            retrieved_by_query = retriever.batch_retrieve(
                [query.query_text for query in dataset.queries],
                k=2,
            )
    except Exception as exc:
        st.error(f"Evaluation failed: {exc}")
        st.stop()

    st.subheader("Metrics")
    render_metric_cards(results.metric_scores)

    metrics_df = pd.DataFrame(metric_rows(results.metric_scores))
    st.bar_chart(metrics_df, x="metric", y="score", width="stretch")

    left, right = st.columns(2)
    left.metric("Queries evaluated", results.query_count)
    right.metric("Judge", judge.name)

    st.subheader("Retrieved Evidence")
    for query, query_results in zip(dataset.queries, retrieved_by_query):
        with st.expander(f"{query.query_id}: {query.query_text}", expanded=False):
            st.markdown("**Expected answer**")
            st.write("\n\n".join(query.expected_answers))
            st.dataframe(
                result_rows(query_results),
                hide_index=True,
                width="stretch",
            )

if __name__ == "__main__":
    main()