import pytest

from evret.metrics import NDCG


class TestNDCGBasicScoring:
    def test_perfect_ndcg_all_relevant_at_top(self) -> None:
        metric = NDCG(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["d2", "d1", "d3"],
            relevant_doc_ids={"d1", "d2", "d3"},
        )
        assert score == 1.0

    def test_zero_ndcg_no_relevant(self) -> None:
        metric = NDCG(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_partial_ndcg_binary_relevance(self) -> None:
        metric = NDCG(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1", "d3"},
        )
        assert score == pytest.approx(0.9197207891, rel=1e-9)


class TestNDCGRankingSensitivity:
    def test_better_ranking_higher_score(self) -> None:
        metric = NDCG(k=3)

        score_optimal = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1", "d2"},
        )
        score_suboptimal = metric.score_query(
            retrieved_doc_ids=["d3", "d1", "d2"],
            relevant_doc_ids={"d1", "d2"},
        )

        assert score_optimal > score_suboptimal
        assert score_optimal == 1.0

    def test_ordering_affects_score(self) -> None:
        metric = NDCG(k=5)
        relevant = {"rel1", "rel2", "rel3"}

        score_best = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "rel3", "irr1", "irr2"],
            relevant_doc_ids=relevant,
        )
        score_worst = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel1", "rel2", "rel3"],
            relevant_doc_ids=relevant,
        )

        assert score_best == 1.0
        assert score_worst < score_best


class TestNDCGDiscounting:
    def test_later_positions_discounted(self) -> None:
        metric = NDCG(k=5)

        score_early = metric.score_query(
            retrieved_doc_ids=["rel", "irr1", "irr2", "irr3", "irr4"],
            relevant_doc_ids={"rel"},
        )
        score_late = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "irr3", "irr4", "rel"],
            relevant_doc_ids={"rel"},
        )

        assert score_early > score_late
        assert score_early == 1.0


class TestNDCGWithKCutoff:
    def test_respects_k_cutoff(self) -> None:
        metric = NDCG(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4"],
            relevant_doc_ids={"d1", "d2"},
        )
        assert score == 1.0

    def test_relevant_beyond_k_ignored(self) -> None:
        metric = NDCG(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel1", "rel2"],
            relevant_doc_ids={"rel1", "rel2"},
        )
        assert score == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        metric = NDCG(k=10)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            relevant_doc_ids={"d1", "d2"},
        )
        assert score == 1.0


class TestNDCGEdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = NDCG(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = NDCG(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = NDCG(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_single_relevant_hit(self) -> None:
        metric = NDCG(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_single_relevant_miss(self) -> None:
        metric = NDCG(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.0


class TestNDCGBatchScoring:
    def test_batch_score_averages_correctly(self) -> None:
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

    def test_batch_with_perfect_and_zero_scores(self) -> None:
        metric = NDCG(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["d1", "d2", "d3"],
                ["d4", "d5", "d6"],
            ],
            relevant_by_query=[
                {"d1", "d2", "d3"},
                {"d7", "d8"},
            ],
        )
        expected = (1.0 + 0.0) / 2
        assert score == 0.5

    def test_empty_batch_returns_zero(self) -> None:
        metric = NDCG(k=5)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0


class TestNDCGKParameter:
    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_different_k_values(self, k: int) -> None:
        metric = NDCG(k=k)
        retrieved = [f"doc{i}" for i in range(k)]
        relevant = {f"doc{i}" for i in range(k)}

        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )
        assert score == 1.0

    def test_k_equals_one(self) -> None:
        metric = NDCG(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0


class TestNDCGNormalization:
    def test_score_bounded_between_zero_and_one(self) -> None:
        metric = NDCG(k=5)

        for i in range(10):
            retrieved = [f"doc{j}" for j in range(5)]
            relevant = {f"doc{j % 7}" for j in range(3)}

            score = metric.score_query(
                retrieved_doc_ids=retrieved,
                relevant_doc_ids=relevant,
            )
            assert 0.0 <= score <= 1.0

    def test_idcg_normalization(self) -> None:
        metric = NDCG(k=5)

        score_perfect = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_doc_ids={"d1", "d2", "d3", "d4", "d5"},
        )
        score_partial = metric.score_query(
            retrieved_doc_ids=["d1", "irr1", "d2", "irr2", "d3"],
            relevant_doc_ids={"d1", "d2", "d3"},
        )

        assert score_perfect == 1.0
        assert score_partial < 1.0
        assert score_partial > 0.0


class TestNDCGVsOtherMetrics:
    def test_ndcg_position_sensitive_unlike_precision(self) -> None:
        metric = NDCG(k=3)

        score_good_order = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr"],
            relevant_doc_ids={"rel1", "rel2"},
        )
        score_bad_order = metric.score_query(
            retrieved_doc_ids=["irr", "rel1", "rel2"],
            relevant_doc_ids={"rel1", "rel2"},
        )

        assert score_good_order > score_bad_order
