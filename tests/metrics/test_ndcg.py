import pytest

from evret.metrics import NDCG


def test_ndcg_binary_relevance() -> None:
    metric = NDCG(k=3)
    score = metric.score_query(
        retrieved_doc_ids=["d1", "d2", "d3"],
        relevant_doc_ids={"d1", "d3"},
    )
    assert score == pytest.approx(0.9197207891, rel=1e-9)


def test_ndcg_all_relevant_is_perfect_score() -> None:
    metric = NDCG(k=3)
    score = metric.score_query(
        retrieved_doc_ids=["d2", "d1", "d3"],
        relevant_doc_ids={"d1", "d2", "d3"},
    )
    assert score == 1.0


def test_ndcg_returns_zero_when_no_relevance_signal() -> None:
    metric = NDCG(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["d1", "d2", "d3"],
        relevant_doc_ids=set(),
    )
    assert score == 0.0


def test_ndcg_batch_score_mean() -> None:
    metric = NDCG(k=3)
    score = metric.score(
        retrieved_by_query=[
            ["d1", "d2", "d3"],
            ["d9", "d8", "d7"],
        ],
        relevant_by_query=[
            {"d1", "d3"},
            {"d4"},
        ],
    )
    assert score == pytest.approx(0.45986039455, rel=1e-9)
