import pytest

from evret.metrics import Recall


def test_recall_query_value() -> None:
    metric = Recall(k=3)
    score = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_9", "doc_2", "doc_3"],
        relevant_doc_ids={"doc_1", "doc_2", "doc_5", "doc_8"},
    )
    assert score == 0.5


def test_recall_empty_relevant_is_zero() -> None:
    metric = Recall(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_2", "doc_3"],
        relevant_doc_ids=set(),
    )
    assert score == 0.0


def test_recall_batch_score_mean() -> None:
    metric = Recall(k=2)
    score = metric.score(
        retrieved_by_query=[
            ["doc_1", "doc_2", "doc_3"],
            ["doc_3", "doc_4", "doc_5"],
        ],
        relevant_by_query=[{"doc_1", "doc_8"}, {"doc_3", "doc_4"}],
    )
    assert score == 0.75


def test_metric_score_length_mismatch_raises() -> None:
    metric = Recall(k=2)
    with pytest.raises(ValueError, match="same length"):
        metric.score(
            retrieved_by_query=[["doc_1"], ["doc_2"]],
            relevant_by_query=[{"doc_1"}],
        )
