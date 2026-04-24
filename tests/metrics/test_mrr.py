from evret.metrics import MRR


def test_mrr_query_first_hit_rank_one() -> None:
    metric = MRR(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["doc_2", "doc_1", "doc_3"],
        relevant_doc_ids={"doc_2", "doc_7"},
    )
    assert score == 1.0


def test_mrr_query_first_hit_rank_three() -> None:
    metric = MRR(k=5)
    score = metric.score_query(
        retrieved_doc_ids=["doc_x", "doc_y", "doc_z", "doc_1"],
        relevant_doc_ids={"doc_z", "doc_2"},
    )
    assert score == 1 / 3


def test_mrr_query_miss_or_empty_relevance() -> None:
    metric = MRR(k=2)
    score_miss = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_2", "doc_3"],
        relevant_doc_ids={"doc_3"},
    )
    score_empty = metric.score_query(
        retrieved_doc_ids=["doc_1", "doc_2"],
        relevant_doc_ids=set(),
    )
    assert score_miss == 0.0
    assert score_empty == 0.0


def test_mrr_batch_score_mean() -> None:
    metric = MRR(k=4)
    score = metric.score(
        retrieved_by_query=[
            ["doc_2", "doc_5", "doc_8"],
            ["doc_9", "doc_3", "doc_4"],
        ],
        relevant_by_query=[{"doc_5"}, {"doc_x"}],
    )
    assert score == 0.25
