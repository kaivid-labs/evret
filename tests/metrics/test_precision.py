import pytest

from evret.metrics import Precision


class TestPrecisionBasicScoring:
    def test_perfect_precision_all_relevant(self) -> None:
        metric = Precision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc1", "doc2", "doc3"},
        )
        assert score == 1.0

    def test_zero_precision_no_relevant(self) -> None:
        metric = Precision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc4", "doc5"},
        )
        assert score == 0.0

    def test_partial_precision(self) -> None:
        metric = Precision(k=4)
        score = metric.score_query(
            retrieved_doc_ids=["doc_1", "doc_3", "doc_5", "doc_9", "doc_10"],
            relevant_doc_ids={"doc_1", "doc_9", "doc_88"},
        )
        assert score == 0.5


class TestPrecisionDenominatorBehavior:
    def test_uses_k_as_denominator_even_when_fewer_retrieved(self) -> None:
        metric = Precision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc_1", "doc_2"],
            relevant_doc_ids={"doc_1", "doc_2"},
        )
        assert score == 0.4

    def test_uses_k_as_denominator_when_more_retrieved(self) -> None:
        metric = Precision(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 1.0


class TestPrecisionEdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = Precision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = Precision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = Precision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_single_document_hit(self) -> None:
        metric = Precision(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_single_document_miss(self) -> None:
        metric = Precision(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.0


class TestPrecisionBatchScoring:
    def test_batch_score_averages_correctly(self) -> None:
        metric = Precision(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc_a", "doc_b", "doc_c"],
                ["doc_1", "doc_2", "doc_3"],
            ],
            relevant_by_query=[{"doc_a", "doc_d"}, {"doc_9"}],
        )
        assert score == (1 / 3) / 2

    def test_batch_with_mixed_results(self) -> None:
        metric = Precision(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
                ["doc7", "doc8", "doc9"],
            ],
            relevant_by_query=[
                {"doc1", "doc2", "doc3"},
                {"doc4"},
                set(),
            ],
        )
        expected = (1.0 + (1 / 3) + 0.0) / 3
        assert score == pytest.approx(expected)

    def test_empty_batch_returns_zero(self) -> None:
        metric = Precision(k=5)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0


class TestPrecisionKParameter:
    @pytest.mark.parametrize("k", [1, 5, 10, 50, 100])
    def test_different_k_values(self, k: int) -> None:
        metric = Precision(k=k)
        retrieved = [f"doc{i}" for i in range(k * 2)]
        relevant = {f"doc{i}" for i in range(0, k * 2, 2)}
        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )
        assert 0.0 <= score <= 1.0

    def test_k_equals_one(self) -> None:
        metric = Precision(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_large_k_value(self) -> None:
        metric = Precision(k=1000)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.002


class TestPrecisionRankingPosition:
    def test_relevant_at_beginning(self) -> None:
        metric = Precision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr1", "irr2", "irr3"],
            relevant_doc_ids={"rel1", "rel2"},
        )
        assert score == 0.4

    def test_relevant_at_end(self) -> None:
        metric = Precision(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "irr3", "rel1", "rel2"],
            relevant_doc_ids={"rel1", "rel2"},
        )
        assert score == 0.4

    def test_relevant_beyond_k_cutoff_ignored(self) -> None:
        metric = Precision(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            relevant_doc_ids={"doc4", "doc5"},
        )
        assert score == 0.0
