import pytest

from evret.metrics import AveragePrecision


class TestAveragePrecisionBasicScoring:
    def test_perfect_average_precision(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            expected_answers={"doc1", "doc2", "doc3"},
        )
        assert score == 1.0

    def test_zero_average_precision(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            expected_answers={"doc4", "doc5"},
        )
        assert score == 0.0

    def test_partial_average_precision(self) -> None:
        metric = AveragePrecision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["a", "b", "c", "d", "e"],
            expected_answers={"a", "c", "f"},
        )
        assert score == pytest.approx(0.5555555556, rel=1e-9)


class TestAveragePrecisionInterpolation:
    def test_precision_at_each_relevant_position(self) -> None:
        metric = AveragePrecision(k=4)
        score = metric.score_query(
            retrieved_doc_ids=["d4", "d1", "d2", "d3"],
            expected_answers={"d1", "d2"},
        )
        assert score == pytest.approx(0.5833333333, rel=1e-9)

    def test_early_expected_answers_weighted_more(self) -> None:
        metric = AveragePrecision(k=5)

        score_early = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr1", "irr2", "irr3"],
            expected_answers={"rel1", "rel2"},
        )
        score_late = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "irr3", "rel1", "rel2"],
            expected_answers={"rel1", "rel2"},
        )

        assert score_early > score_late


class TestAveragePrecisionWithKCutoff:
    def test_respects_k_cutoff(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["a", "x", "y", "b"],
            expected_answers={"a", "b"},
        )
        assert score == 0.5

    def test_relevant_beyond_k_ignored(self) -> None:
        metric = AveragePrecision(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel1", "rel2"],
            expected_answers={"rel1", "rel2"},
        )
        assert score == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        metric = AveragePrecision(k=10)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            expected_answers={"doc1", "doc2"},
        )
        assert score == 1.0


class TestAveragePrecisionEdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = AveragePrecision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            expected_answers={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            expected_answers=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            expected_answers=set(),
        )
        assert score == 0.0

    def test_single_relevant_hit(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            expected_answers={"doc1"},
        )
        assert score == 1.0

    def test_single_relevant_miss(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            expected_answers={"doc4"},
        )
        assert score == 0.0


class TestAveragePrecisionBatchScoring:
    def test_batch_score_averages_correctly(self) -> None:
        metric = AveragePrecision(k=4)
        score = metric.score(
            retrieved_by_query=[
                ["a", "b", "c", "d"],
                ["x", "y", "z", "w"],
            ],
            expected_by_query=[
                {"a", "c"},
                {"q"},
            ],
        )
        assert score == pytest.approx(0.4166666667, rel=1e-9)

    def test_batch_with_varied_scores(self) -> None:
        metric = AveragePrecision(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
                ["doc7", "doc8", "doc9"],
            ],
            expected_by_query=[
                {"doc1", "doc2", "doc3"},
                {"doc4"},
                set(),
            ],
        )
        expected = (1.0 + 1.0 + 0.0) / 3
        assert score == pytest.approx(expected)

    def test_empty_batch_returns_zero(self) -> None:
        metric = AveragePrecision(k=5)
        score = metric.score(retrieved_by_query=[], expected_by_query=[])
        assert score == 0.0


class TestAveragePrecisionKParameter:
    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_different_k_values(self, k: int) -> None:
        metric = AveragePrecision(k=k)
        retrieved = [f"doc{i}" for i in range(k)]
        relevant = {f"doc{i}" for i in range(k)}

        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            expected_answers=relevant,
        )
        assert score == 1.0

    def test_k_equals_one(self) -> None:
        metric = AveragePrecision(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            expected_answers={"doc1"},
        )
        assert score == 1.0


class TestAveragePrecisionRankingSensitivity:
    def test_better_ranking_higher_score(self) -> None:
        metric = AveragePrecision(k=5)

        score_good = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr1", "irr2", "irr3"],
            expected_answers={"rel1", "rel2"},
        )
        score_bad = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "irr3", "rel1", "rel2"],
            expected_answers={"rel1", "rel2"},
        )

        assert score_good > score_bad

    def test_position_affects_interpolation(self) -> None:
        metric = AveragePrecision(k=4)

        score1 = metric.score_query(
            retrieved_doc_ids=["rel", "irr1", "irr2", "irr3"],
            expected_answers={"rel"},
        )
        score2 = metric.score_query(
            retrieved_doc_ids=["irr1", "rel", "irr2", "irr3"],
            expected_answers={"rel"},
        )
        score3 = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel", "irr3"],
            expected_answers={"rel"},
        )

        assert score1 > score2 > score3


class TestAveragePrecisionSemantics:
    def test_ap_averages_precision_at_relevant_positions(self) -> None:
        metric = AveragePrecision(k=5)

        score = metric.score_query(
            retrieved_doc_ids=["rel1", "irr1", "rel2", "irr2", "irr3"],
            expected_answers={"rel1", "rel2"},
        )

        precision_at_1 = 1 / 1
        precision_at_3 = 2 / 3
        expected = (precision_at_1 + precision_at_3) / 2

        assert score == pytest.approx(expected)

    def test_ap_normalized_by_total_relevant(self) -> None:
        metric = AveragePrecision(k=5)

        score_two_relevant = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr1", "irr2", "irr3"],
            expected_answers={"rel1", "rel2"},
        )
        score_four_relevant = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr1", "irr2", "irr3"],
            expected_answers={"rel1", "rel2", "rel3", "rel4"},
        )

        assert score_two_relevant == 1.0
        assert score_four_relevant < 1.0


class TestAveragePrecisionBoundary:
    def test_score_bounded_between_zero_and_one(self) -> None:
        metric = AveragePrecision(k=5)

        for _ in range(10):
            retrieved = [f"doc{j}" for j in range(5)]
            relevant = {f"doc{j % 7}" for j in range(3)}

            score = metric.score_query(
                retrieved_doc_ids=retrieved,
                expected_answers=relevant,
            )
            assert 0.0 <= score <= 1.0
