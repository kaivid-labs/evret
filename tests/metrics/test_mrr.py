import pytest

from evret.metrics import MRR


class TestMRRBasicScoring:
    def test_first_rank_perfect_score(self) -> None:
        metric = MRR(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc_2", "doc_1", "doc_3"],
            relevant_doc_ids={"doc_2", "doc_7"},
        )
        assert score == 1.0

    def test_second_rank(self) -> None:
        metric = MRR(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.5

    def test_third_rank(self) -> None:
        metric = MRR(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc_x", "doc_y", "doc_z", "doc_1"],
            relevant_doc_ids={"doc_z", "doc_2"},
        )
        assert score == 1 / 3


class TestMRRRankPosition:
    @pytest.mark.parametrize(
        "rank,expected_score",
        [
            (1, 1.0),
            (2, 0.5),
            (3, 1 / 3),
            (4, 0.25),
            (5, 0.2),
            (10, 0.1),
        ],
    )
    def test_reciprocal_rank_calculation(self, rank: int, expected_score: float) -> None:
        metric = MRR(k=20)
        retrieved = [f"irr{i}" for i in range(rank - 1)] + ["rel"]
        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids={"rel"},
        )
        assert score == pytest.approx(expected_score)


class TestMRRFirstRelevantOnly:
    def test_only_first_relevant_counts(self) -> None:
        metric = MRR(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["irr", "rel1", "rel2", "rel3"],
            relevant_doc_ids={"rel1", "rel2", "rel3"},
        )
        assert score == 0.5

    def test_multiple_relevant_same_as_single(self) -> None:
        metric = MRR(k=5)

        score_multiple = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc2", "doc3"},
        )
        score_single = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc2"},
        )

        assert score_multiple == score_single == 0.5


class TestMRRWithKCutoff:
    def test_relevant_at_k_cutoff(self) -> None:
        metric = MRR(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc3"},
        )
        assert score == pytest.approx(1 / 3)

    def test_relevant_beyond_k_cutoff_returns_zero(self) -> None:
        metric = MRR(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc4"},
        )
        assert score == 0.0

    def test_k_larger_than_list(self) -> None:
        metric = MRR(k=10)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.5


class TestMRREdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = MRR(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = MRR(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc_1", "doc_2"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = MRR(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_no_relevant_in_retrieved(self) -> None:
        metric = MRR(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc_1", "doc_2", "doc_3"],
            relevant_doc_ids={"doc_3"},
        )
        assert score == 0.0

    def test_single_document_hit(self) -> None:
        metric = MRR(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_single_document_miss(self) -> None:
        metric = MRR(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.0


class TestMRRBatchScoring:
    def test_batch_score_averages_correctly(self) -> None:
        metric = MRR(k=4)
        score = metric.score(
            retrieved_by_query=[
                ["doc_2", "doc_5", "doc_8"],
                ["doc_9", "doc_3", "doc_4"],
            ],
            relevant_by_query=[{"doc_5"}, {"doc_x"}],
        )
        assert score == 0.25

    def test_batch_with_varied_ranks(self) -> None:
        metric = MRR(k=5)
        score = metric.score(
            retrieved_by_query=[
                ["rel", "irr1", "irr2"],
                ["irr1", "rel", "irr2"],
                ["irr1", "irr2", "rel"],
            ],
            relevant_by_query=[{"rel"}, {"rel"}, {"rel"}],
        )
        expected = (1.0 + 0.5 + (1 / 3)) / 3
        assert score == pytest.approx(expected)

    def test_empty_batch_returns_zero(self) -> None:
        metric = MRR(k=5)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0


class TestMRRKParameter:
    @pytest.mark.parametrize("k", [1, 5, 10, 50, 100])
    def test_different_k_values(self, k: int) -> None:
        metric = MRR(k=k)
        retrieved = [f"doc{i}" for i in range(k + 5)]
        relevant = {f"doc{min(k - 1, len(retrieved) - 1)}"}

        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )
        assert 0.0 <= score <= 1.0

    def test_k_equals_one(self) -> None:
        metric = MRR(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0


class TestMRRRankingSensitivity:
    def test_earlier_rank_better_score(self) -> None:
        metric = MRR(k=5)

        score_rank1 = metric.score_query(
            retrieved_doc_ids=["rel", "irr1", "irr2"],
            relevant_doc_ids={"rel"},
        )
        score_rank2 = metric.score_query(
            retrieved_doc_ids=["irr1", "rel", "irr2"],
            relevant_doc_ids={"rel"},
        )
        score_rank3 = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel"],
            relevant_doc_ids={"rel"},
        )

        assert score_rank1 > score_rank2 > score_rank3
        assert score_rank1 == 1.0
        assert score_rank2 == 0.5
        assert score_rank3 == pytest.approx(1 / 3)


class TestMRRSemantics:
    def test_mrr_rewards_early_hits(self) -> None:
        metric = MRR(k=10)

        score_early = metric.score_query(
            retrieved_doc_ids=["rel"] + [f"irr{i}" for i in range(9)],
            relevant_doc_ids={"rel"},
        )
        score_late = metric.score_query(
            retrieved_doc_ids=[f"irr{i}" for i in range(9)] + ["rel"],
            relevant_doc_ids={"rel"},
        )

        assert score_early == 1.0
        assert score_late == 0.1
        assert score_early > score_late
