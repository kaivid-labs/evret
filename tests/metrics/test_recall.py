import pytest

from evret.metrics import Recall


class TestRecallBasicScoring:
    def test_perfect_recall_all_retrieved(self) -> None:
        metric = Recall(k=4)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc1", "doc2", "doc3", "doc4"},
        )
        assert score == 1.0

    def test_zero_recall_no_hits(self) -> None:
        metric = Recall(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc4", "doc5", "doc6"},
        )
        assert score == 0.0

    def test_partial_recall(self) -> None:
        metric = Recall(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc_1", "doc_9", "doc_2", "doc_3"],
            relevant_doc_ids={"doc_1", "doc_2", "doc_5", "doc_8"},
        )
        assert score == 0.5


class TestRecallWithKCutoff:
    def test_recall_respects_k_cutoff(self) -> None:
        metric = Recall(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc1", "doc3", "doc4"},
        )
        assert score == pytest.approx(1 / 3)

    def test_recall_with_relevant_beyond_k(self) -> None:
        metric = Recall(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "doc1", "doc2"],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_recall_k_larger_than_retrieved(self) -> None:
        metric = Recall(k=10)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1", "doc2", "doc3", "doc4"},
        )
        assert score == 0.5


class TestRecallEdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = Recall(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = Recall(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = Recall(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_single_relevant_found(self) -> None:
        metric = Recall(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_single_relevant_not_found(self) -> None:
        metric = Recall(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc3"},
        )
        assert score == 0.0


class TestRecallBatchScoring:
    def test_batch_score_averages_correctly(self) -> None:
        metric = Recall(k=2)
        score = metric.score(
            retrieved_by_query=[
                ["doc_1", "doc_2", "doc_3"],
                ["doc_3", "doc_4", "doc_5"],
            ],
            relevant_by_query=[{"doc_1", "doc_8"}, {"doc_3", "doc_4"}],
        )
        assert score == 0.75

    def test_batch_with_mixed_results(self) -> None:
        metric = Recall(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
                ["doc7", "doc8", "doc9"],
            ],
            relevant_by_query=[
                {"doc1", "doc2", "doc3"},
                {"doc4", "doc10"},
                {"doc10", "doc11"},
            ],
        )
        assert score == 0.5

    def test_empty_batch_returns_zero(self) -> None:
        metric = Recall(k=5)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0

    def test_batch_length_mismatch_raises(self) -> None:
        metric = Recall(k=2)
        with pytest.raises(ValueError, match="same length"):
            metric.score(
                retrieved_by_query=[["doc_1"], ["doc_2"]],
                relevant_by_query=[{"doc_1"}],
            )


class TestRecallKParameter:
    @pytest.mark.parametrize("k", [1, 5, 10, 50, 100])
    def test_different_k_values(self, k: int) -> None:
        metric = Recall(k=k)
        retrieved = [f"doc{i}" for i in range(k)]
        relevant = {f"doc{i}" for i in range(k * 2)}
        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )
        assert 0.0 <= score <= 1.0
        assert score == 0.5

    def test_k_equals_one(self) -> None:
        metric = Recall(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.5


class TestRecallVsPrecisionDifference:
    def test_high_recall_low_precision_scenario(self) -> None:
        metric = Recall(k=10)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"] + [f"irr{i}" for i in range(8)],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 1.0

    def test_many_relevant_few_retrieved(self) -> None:
        metric = Recall(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={f"doc{i}" for i in range(1, 11)},
        )
        assert score == 0.2


class TestRecallRankingSensitivity:
    def test_recall_insensitive_to_ranking_order(self) -> None:
        metric = Recall(k=5)
        relevant = {"doc1", "doc3", "doc5"}

        score1 = metric.score_query(
            retrieved_doc_ids=["doc1", "doc3", "doc5", "doc2", "doc4"],
            relevant_doc_ids=relevant,
        )
        score2 = metric.score_query(
            retrieved_doc_ids=["doc5", "doc3", "doc1", "doc2", "doc4"],
            relevant_doc_ids=relevant,
        )
        score3 = metric.score_query(
            retrieved_doc_ids=["doc2", "doc4", "doc1", "doc3", "doc5"],
            relevant_doc_ids=relevant,
        )

        assert score1 == score2 == score3 == 1.0
