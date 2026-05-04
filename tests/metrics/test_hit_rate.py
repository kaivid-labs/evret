import pytest

from evret.metrics import HitRate


class TestHitRateBasicScoring:
    def test_hit_when_relevant_found(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 1.0

    def test_miss_when_no_relevant_found(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc4", "doc5"},
        )
        assert score == 0.0

    def test_hit_with_multiple_relevant(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc1", "doc2", "doc3"},
        )
        assert score == 1.0


class TestHitRateBinaryBehavior:
    def test_returns_only_zero_or_one(self) -> None:
        metric = HitRate(k=5)

        score_hit = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc3"},
        )
        score_miss = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc4"},
        )

        assert score_hit in {0.0, 1.0}
        assert score_miss in {0.0, 1.0}
        assert score_hit == 1.0
        assert score_miss == 0.0

    def test_insensitive_to_number_of_hits(self) -> None:
        metric = HitRate(k=5)

        score_one_hit = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc1"},
        )
        score_three_hits = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc1", "doc2", "doc3"},
        )

        assert score_one_hit == score_three_hits == 1.0


class TestHitRateWithKCutoff:
    def test_hit_within_k_cutoff(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc_9", "doc_2", "doc_5", "doc_7"],
            relevant_doc_ids={"doc_5", "doc_88"},
        )
        assert score == 1.0

    def test_miss_beyond_k_cutoff(self) -> None:
        metric = HitRate(k=2)
        score = metric.score_query(
            retrieved_doc_ids=["doc_9", "doc_2", "doc_5"],
            relevant_doc_ids={"doc_5"},
        )
        assert score == 0.0

    def test_hit_at_position_k(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids={"doc3"},
        )
        assert score == 1.0

    def test_miss_at_position_k_plus_one(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4"],
            relevant_doc_ids={"doc4"},
        )
        assert score == 0.0


class TestHitRateEdgeCases:
    def test_empty_retrieved_returns_zero(self) -> None:
        metric = HitRate(k=5)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"doc1", "doc2"},
        )
        assert score == 0.0

    def test_empty_relevant_returns_zero(self) -> None:
        metric = HitRate(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3"],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_both_empty_returns_zero(self) -> None:
        metric = HitRate(k=3)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids=set(),
        )
        assert score == 0.0

    def test_single_document_hit(self) -> None:
        metric = HitRate(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0

    def test_single_document_miss(self) -> None:
        metric = HitRate(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1"],
            relevant_doc_ids={"doc2"},
        )
        assert score == 0.0


class TestHitRateBatchScoring:
    def test_batch_score_averages_binary_results(self) -> None:
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

    def test_batch_all_hits(self) -> None:
        metric = HitRate(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
            ],
            relevant_by_query=[{"doc1"}, {"doc4"}],
        )
        assert score == 1.0

    def test_batch_all_misses(self) -> None:
        metric = HitRate(k=3)
        score = metric.score(
            retrieved_by_query=[
                ["doc1", "doc2", "doc3"],
                ["doc4", "doc5", "doc6"],
            ],
            relevant_by_query=[{"doc7"}, {"doc8"}],
        )
        assert score == 0.0

    def test_empty_batch_returns_zero(self) -> None:
        metric = HitRate(k=5)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0


class TestHitRateKParameter:
    @pytest.mark.parametrize("k", [1, 5, 10, 50, 100])
    def test_different_k_values(self, k: int) -> None:
        metric = HitRate(k=k)
        retrieved = [f"doc{i}" for i in range(k)]
        relevant = {f"doc{k - 1}"}

        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            relevant_doc_ids=relevant,
        )
        assert score == 1.0

    def test_k_equals_one(self) -> None:
        metric = HitRate(k=1)
        score = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2"],
            relevant_doc_ids={"doc1"},
        )
        assert score == 1.0


class TestHitRatePositionInsensitivity:
    def test_first_position_hit(self) -> None:
        metric = HitRate(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["rel", "irr1", "irr2", "irr3", "irr4"],
            relevant_doc_ids={"rel"},
        )
        assert score == 1.0

    def test_last_position_hit(self) -> None:
        metric = HitRate(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "irr3", "irr4", "rel"],
            relevant_doc_ids={"rel"},
        )
        assert score == 1.0

    def test_middle_position_hit(self) -> None:
        metric = HitRate(k=5)
        score = metric.score_query(
            retrieved_doc_ids=["irr1", "irr2", "rel", "irr3", "irr4"],
            relevant_doc_ids={"rel"},
        )
        assert score == 1.0


class TestHitRateVsOtherMetrics:
    def test_hit_rate_simpler_than_precision(self) -> None:
        metric = HitRate(k=5)

        score_one_relevant = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            relevant_doc_ids={"doc1"},
        )
        score_five_relevant = metric.score_query(
            retrieved_doc_ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
            relevant_doc_ids={"doc1", "doc2", "doc3", "doc4", "doc5"},
        )

        assert score_one_relevant == score_five_relevant == 1.0
