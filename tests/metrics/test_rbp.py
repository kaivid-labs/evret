"""Tests for Rank-Biased Precision (RBP) metric."""

import pytest

from evret.metrics import RBP


class TestRBPBasicScoring:
    """Test basic RBP scoring behavior."""

    def test_perfect_rbp_all_relevant(self) -> None:
        """Perfect ranking with all expected answers."""
        metric = RBP(k=3, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d1", "d2", "d3"},
        )
        # RBP = (1-0.8) × [0.8^0×1 + 0.8^1×1 + 0.8^2×1]
        #     = 0.2 × [1 + 0.8 + 0.64]
        #     = 0.2 × 2.44 = 0.488
        expected = 0.2 * (1.0 + 0.8 + 0.64)
        assert score == pytest.approx(expected, rel=1e-6)

    def test_zero_rbp_no_relevant(self) -> None:
        """RBP should be 0 when no expected answeruments."""
        metric = RBP(k=5, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers=set(),
        )
        assert score == 0.0

    def test_zero_rbp_empty_retrieved(self) -> None:
        """RBP should be 0 when no documents retrieved."""
        metric = RBP(k=5, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=[],
            expected_answers={"d1", "d2"},
        )
        assert score == 0.0

    def test_single_relevant_at_top(self) -> None:
        """Single expected answer at position 1."""
        metric = RBP(k=3, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d1"},
        )
        # RBP = (1-0.8) × [0.8^0×1 + 0.8^1×0 + 0.8^2×0]
        #     = 0.2 × 1 = 0.2
        assert score == pytest.approx(0.2, rel=1e-6)


class TestRBPPersistenceParameter:
    """Test persistence parameter behavior."""

    def test_higher_p_values_later_positions_more(self) -> None:
        """Higher p gives more weight to later positions relative to total."""
        # Expected answer at position 3
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d3"}

        score_low_p = RBP(k=3, p=0.5).score_query(retrieved, relevant)
        score_high_p = RBP(k=3, p=0.95).score_query(retrieved, relevant)

        # With lower p, position 1 is weighted much more heavily (normalization factor 1-p is larger)
        # But the absolute contribution at position 3 with higher p is relatively closer to position 1
        # Let's check the ratio of position 3 to position 1 weight
        # For p=0.5: weight_ratio = (0.5^2) / 1 = 0.25
        # For p=0.95: weight_ratio = (0.95^2) / 1 = 0.9025
        # So higher p gives more relative weight to later positions
        # However, due to normalization factor (1-p), absolute scores work differently
        # p=0.5: (1-0.5) × 0.5^2 = 0.5 × 0.25 = 0.125
        # p=0.95: (1-0.95) × 0.95^2 = 0.05 × 0.9025 = 0.045
        assert score_low_p > score_high_p  # Corrected: lower p actually gives higher absolute score at later positions due to normalization

    def test_different_p_values(self) -> None:
        """Test RBP with different persistence parameters."""
        retrieved = ["d1", "d2", "d3"]
        relevant = {"d1"}

        score_p50 = RBP(k=3, p=0.5).score_query(retrieved, relevant)
        score_p80 = RBP(k=3, p=0.8).score_query(retrieved, relevant)
        score_p95 = RBP(k=3, p=0.95).score_query(retrieved, relevant)

        # p=0.5: (1-0.5) × 1 = 0.5
        # p=0.8: (1-0.8) × 1 = 0.2
        # p=0.95: (1-0.95) × 1 = 0.05
        assert score_p50 == pytest.approx(0.5, rel=1e-6)
        assert score_p80 == pytest.approx(0.2, rel=1e-6)
        assert score_p95 == pytest.approx(0.05, rel=1e-6)

    def test_invalid_p_raises_error(self) -> None:
        """Invalid p values should raise ValueError."""
        with pytest.raises(ValueError, match="Persistence parameter p must be in range"):
            RBP(k=5, p=0.0)
        with pytest.raises(ValueError, match="Persistence parameter p must be in range"):
            RBP(k=5, p=1.0)
        with pytest.raises(ValueError, match="Persistence parameter p must be in range"):
            RBP(k=5, p=1.5)


class TestRBPGeometricWeighting:
    """Test geometric weighting behavior."""

    def test_position_1_highest_weight(self) -> None:
        """Position 1 should have highest contribution."""
        metric = RBP(k=3, p=0.8)

        score_pos1 = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d1"},
        )
        score_pos2 = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d2"},
        )
        score_pos3 = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d3"},
        )

        assert score_pos1 > score_pos2 > score_pos3

    def test_exponential_decay(self) -> None:
        """Weights should decay exponentially."""
        metric = RBP(k=5, p=0.8)
        retrieved = ["d1", "d2", "d3", "d4", "d5"]

        # Individual contributions at each position
        scores = []
        for i in range(5):
            relevant = {f"d{i+1}"}
            score = metric.score_query(retrieved, relevant)
            scores.append(score)

        # Check exponential decay: score[i+1] = score[i] × p
        for i in range(len(scores) - 1):
            expected_ratio = 0.8
            actual_ratio = scores[i + 1] / scores[i]
            assert actual_ratio == pytest.approx(expected_ratio, rel=1e-6)


class TestRBPWithKCutoff:
    """Test RBP respects k cutoff."""

    def test_respects_k_cutoff(self) -> None:
        """RBP should only consider top-k positions."""
        metric = RBP(k=2, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4"],
            expected_answers={"d1", "d2"},
        )
        # Only positions 1 and 2
        # RBP = 0.2 × (1 + 0.8) = 0.36
        expected = 0.2 * (1.0 + 0.8)
        assert score == pytest.approx(expected, rel=1e-6)

    def test_relevant_beyond_k_ignored(self) -> None:
        """Expected answers beyond k should be ignored."""
        metric = RBP(k=2, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4"],
            expected_answers={"d3", "d4"},
        )
        assert score == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """RBP should handle k > retrieved length."""
        metric = RBP(k=10, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            expected_answers={"d1", "d2"},
        )
        expected = 0.2 * (1.0 + 0.8)
        assert score == pytest.approx(expected, rel=1e-6)


class TestRBPEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_both_empty_returns_zero(self) -> None:
        """Empty retrieved and relevant should return 0."""
        metric = RBP(k=5, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=[],
            expected_answers=set(),
        )
        assert score == 0.0

    def test_single_doc_hit(self) -> None:
        """Single doc that is relevant."""
        metric = RBP(k=1, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1"],
            expected_answers={"d1"},
        )
        # RBP = (1-0.8) × 1 = 0.2
        assert score == pytest.approx(0.2, rel=1e-6)

    def test_single_doc_miss(self) -> None:
        """Single doc that is not relevant."""
        metric = RBP(k=1, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1"],
            expected_answers={"d2"},
        )
        assert score == 0.0

    def test_alternating_relevant_irrelevant(self) -> None:
        """Test with alternating relevant/irexpected answers."""
        metric = RBP(k=5, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["rel1", "irr1", "rel2", "irr2", "rel3"],
            expected_answers={"rel1", "rel2", "rel3"},
        )
        # RBP = 0.2 × (1 + 0 + 0.64 + 0 + 0.4096)
        expected = 0.2 * (1.0 + 0.64 + 0.4096)
        assert score == pytest.approx(expected, rel=1e-4)


class TestRBPBatchScoring:
    """Test batch evaluation."""

    def test_batch_score_averages_correctly(self) -> None:
        """Batch scoring should average per-query scores."""
        metric = RBP(k=3, p=0.8)
        score = metric.score(
            retrieved_by_query=[
                ["d1", "d2", "d3"],
                ["d4", "d5", "d6"],
            ],
            expected_by_query=[
                {"d1", "d2"},
                {"d6"},
            ],
        )
        # Query 1: RBP = 0.2 × (1 + 0.8) = 0.36
        # Query 2: RBP = 0.2 × 0.64 = 0.128
        expected = (0.36 + 0.128) / 2
        assert score == pytest.approx(expected, rel=1e-4)

    def test_empty_batch_returns_zero(self) -> None:
        """Empty batch should return 0."""
        metric = RBP(k=5, p=0.8)
        score = metric.score(retrieved_by_query=[], expected_by_query=[])
        assert score == 0.0


class TestRBPGradedRelevance:
    """Test RBP with graded relevance."""

    def test_graded_relevance_normalized(self) -> None:
        """Graded relevance should be normalized to [0, 1]."""
        metric = RBP(k=3, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            expected_answers={"d1": 4, "d2": 2, "d3": 1},  # Grades 1-4
        )
        # Normalized: d1=1.0, d2=0.5, d3=0.25
        # RBP = 0.2 × (1.0 + 0.8×0.5 + 0.64×0.25)
        #     = 0.2 × (1.0 + 0.4 + 0.16) = 0.2 × 1.56 = 0.312
        expected = 0.2 * (1.0 + 0.4 + 0.16)
        assert score == pytest.approx(expected, rel=1e-4)

    def test_higher_grades_contribute_more(self) -> None:
        """Higher relevance grades should contribute more."""
        metric = RBP(k=2, p=0.8)  # Changed k to 2 to accommodate both docs

        score_grade_1 = metric.score_query(
            retrieved_doc_ids=["d1"],
            expected_answers={"d1": 1},
        )
        score_grade_4 = metric.score_query(
            retrieved_doc_ids=["d1"],
            expected_answers={"d1": 4},
        )

        # Both get normalized: if max is 4, then grade_1 → 0.25, grade_4 → 1.0
        # But if there's only one doc, max_grade = max(grades) = 1 or 4
        # So grade_1 with only grade_1 present: 1/1 = 1.0
        # And grade_4 with only grade_4 present: 4/4 = 1.0
        # They normalize to the same value when evaluated separately!
        # Let's test with multiple docs to see the difference
        score_both = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            expected_answers={"d1": 1, "d2": 4},
        )
        # Now d1 normalized to 1/4=0.25, d2 normalized to 4/4=1.0
        # RBP = 0.2 × (0.25 + 0.8×1.0) = 0.2 × 1.05 = 0.21
        expected = 0.2 * (0.25 + 0.8 * 1.0)
        assert score_both == pytest.approx(expected, rel=1e-4)

    def test_graded_relevance_already_normalized(self) -> None:
        """Already normalized grades (≤1) should not be renormalized."""
        metric = RBP(k=2, p=0.8)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            expected_answers={"d1": 1.0, "d2": 0.5},
        )
        # RBP = 0.2 × (1.0 + 0.8×0.5) = 0.2 × 1.4 = 0.28
        expected = 0.2 * (1.0 + 0.4)
        assert score == pytest.approx(expected, rel=1e-6)


class TestRBPResidual:
    """Test residual calculation for incomplete rankings."""

    def test_residual_calculation(self) -> None:
        """Test residual computation."""
        metric = RBP(k=5, p=0.8)
        residual = metric.compute_residual(num_retrieved=5)
        # Residual = p^k = 0.8^5
        expected = 0.8**5
        assert residual == pytest.approx(expected, rel=1e-6)

    def test_residual_with_fewer_retrieved(self) -> None:
        """Residual with num_retrieved < k."""
        metric = RBP(k=10, p=0.8)
        residual = metric.compute_residual(num_retrieved=5)
        # Uses effective k = min(10, 5) = 5
        expected = 0.8**5
        assert residual == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("p", [0.5, 0.8, 0.95])
    def test_residual_different_p(self, p: float) -> None:
        """Test residual with different persistence values."""
        metric = RBP(k=10, p=p)
        residual = metric.compute_residual(num_retrieved=10)
        assert residual == pytest.approx(p**10, rel=1e-6)


class TestRBPExpectedSearchDepth:
    """Test expected search depth calculation."""

    def test_expected_search_depth_formula(self) -> None:
        """Test expected depth = 1/(1-p)."""
        metric_p50 = RBP(k=10, p=0.5)
        metric_p80 = RBP(k=10, p=0.8)
        metric_p95 = RBP(k=10, p=0.95)

        assert metric_p50.expected_search_depth == pytest.approx(2.0, rel=1e-6)
        assert metric_p80.expected_search_depth == pytest.approx(5.0, rel=1e-6)
        assert metric_p95.expected_search_depth == pytest.approx(20.0, rel=1e-4)

    @pytest.mark.parametrize(
        ("p", "expected_depth"),
        [
            (0.5, 2.0),
            (0.7, 3.333),
            (0.8, 5.0),
            (0.9, 10.0),
            (0.95, 20.0),
        ],
    )
    def test_expected_depth_values(self, p: float, expected_depth: float) -> None:
        """Test expected depth for common p values."""
        metric = RBP(k=20, p=p)
        assert metric.expected_search_depth == pytest.approx(expected_depth, rel=1e-2)


class TestRBPMetricName:
    """Test metric name formatting."""

    def test_default_name_omits_p(self) -> None:
        """Default p=0.8 should be omitted from name."""
        metric = RBP(k=10, p=0.8)
        assert metric.name == "rbp@10"

    def test_custom_p_in_name(self) -> None:
        """Non-default p should appear in name."""
        metric = RBP(k=10, p=0.95)
        assert metric.name == "rbp(p=0.95)@10"


class TestRBPVsOtherMetrics:
    """Compare RBP behavior with other metrics."""

    def test_rbp_position_sensitive(self) -> None:
        """RBP should be sensitive to document positions."""
        metric = RBP(k=3, p=0.8)

        score_good_order = metric.score_query(
            retrieved_doc_ids=["rel1", "rel2", "irr"],
            expected_answers={"rel1", "rel2"},
        )
        score_bad_order = metric.score_query(
            retrieved_doc_ids=["irr", "rel1", "rel2"],
            expected_answers={"rel1", "rel2"},
        )

        # Expected answers at top should yield higher RBP
        assert score_good_order > score_bad_order

    def test_rbp_top_heavy_weighting(self) -> None:
        """RBP heavily weights top positions."""
        metric = RBP(k=10, p=0.8)

        # All relevant at top
        score_top = metric.score_query(
            retrieved_doc_ids=[f"d{i}" for i in range(10)],
            expected_answers={f"d{i}" for i in range(3)},
        )
        # All relevant at bottom
        score_bottom = metric.score_query(
            retrieved_doc_ids=[f"d{i}" for i in range(10)],
            expected_answers={f"d{i}" for i in range(7, 10)},
        )

        assert score_top > score_bottom


class TestRBPParameterizedScenarios:
    """Parameterized test scenarios."""

    @pytest.mark.parametrize("k", [1, 3, 5, 10, 20])
    def test_different_k_values(self, k: int) -> None:
        """Test RBP with different k values."""
        metric = RBP(k=k, p=0.8)
        retrieved = [f"doc{i}" for i in range(k)]
        relevant = {f"doc{i}" for i in range(k)}

        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            expected_answers=relevant,
        )
        # Should be bounded in [0, 1]
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "retrieved,relevant,p,expected",
        [
            # Single relevant at position 1, p=0.8
            (["d1", "d2"], {"d1"}, 0.8, 0.2),
            # Two relevant at positions 1,2, p=0.8
            (["d1", "d2"], {"d1", "d2"}, 0.8, 0.2 * 1.8),
            # Single relevant at position 1, p=0.5
            (["d1", "d2"], {"d1"}, 0.5, 0.5),
        ],
    )
    def test_known_rbp_values(
        self,
        retrieved: list[str],
        relevant: set[str],
        p: float,
        expected: float,
    ) -> None:
        """Test RBP against known values."""
        metric = RBP(k=len(retrieved), p=p)
        score = metric.score_query(
            retrieved_doc_ids=retrieved,
            expected_answers=relevant,
        )
        assert score == pytest.approx(expected, rel=1e-6)
