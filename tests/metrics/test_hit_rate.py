from evret.metrics import HitRate

def test_hit_rate_query_hit_inside_k() -> None:
    metric = HitRate(k=3)
    score = metric.score_query(
        retrieved_doc_ids=["doc_9", "doc_2", "doc_5", "doc_7"],
        relevant_doc_ids={"doc_5", "doc_88"},
    )
    assert score == 1.0


def test_hit_rate_query_miss_inside_k() -> None:
    metric = HitRate(k=2)
    score = metric.score_query(
        retrieved_doc_ids=["doc_9", "doc_2", "doc_5"],
        relevant_doc_ids={"doc_5"},
    )
    assert score == 0.0


def test_hit_rate_batch_score_mean() -> None:
    metric = HitRate(k=3)
    score = metric.score(
        retrieved_by_query=[
            ["doc_1", "doc_2", "doc_3"],
            ["doc_7", "doc_8", "doc_9"],
            ["doc_4", "doc_5", "doc_6"],
        ],
        relevant_by_query=[{"doc_2"}, {"doc_11"}, {"doc_6"}],
    )
    assert score == 2 / 3
