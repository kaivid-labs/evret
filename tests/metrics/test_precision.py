from evret.metrics import Precision


def test_precision_query_value() -> None:
    metric = Precision(k=4)
    score = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_3", "doc_5", "doc_9", "doc_10"],
        relevant_doc_ids={"doc_1", "doc_9", "doc_88"},
    )
    assert score == 0.5


def test_precision_uses_configured_k_denominator() -> None:
    metric = Precision(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_2"],
        relevant_doc_ids={"doc_1", "doc_2"},
    )
    assert score == 0.4


def test_precision_batch_score_mean() -> None:
    metric = Precision(k=3)
    score = metric.score(
        retrieved_by_query=[
            ["doc_a", "doc_b", "doc_c"],
            ["doc_1", "doc_2", "doc_3"],
        ],
        relevant_by_query=[{"doc_a", "doc_d"}, {"doc_9"}],
    )
    assert score == (1 / 3) / 2
