import pytest

from evret.metrics import AveragePrecision


def test_average_precision_query_value() -> None:
    metric = AveragePrecision(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["a", "b", "c", "d", "e"],
        relevant_doc_ids={"a", "c", "f"},
    )
    assert score == pytest.approx(0.5555555556, rel=1e-9)


def test_average_precision_respects_k_cutoff() -> None:
    metric = AveragePrecision(k=3)
    score = metric.score_query(
        retrieved_doc_ids=["a", "x", "y", "b"],
        relevant_doc_ids={"a", "b"},
    )
    assert score == 0.5


def test_average_precision_binary_relevance() -> None:
    metric = AveragePrecision(k=4)
    score = metric.score_query(
        retrieved_doc_ids=["d4", "d1", "d2", "d3"],
        relevant_doc_ids={"d1", "d2"},
    )
    assert score == pytest.approx(0.5833333333, rel=1e-9)


def test_average_precision_batch_score_mean() -> None:
    metric = AveragePrecision(k=4)
    score = metric.score(
        retrieved_by_query=[
            ["a", "b", "c", "d"],
            ["x", "y", "z", "w"],
        ],
        relevant_by_query=[
            {"a", "c"},
            {"q"},
        ],
    )
    assert score == pytest.approx(0.4166666667, rel=1e-9)
