"""Tests for Expected Reciprocal Rank (ERR) metric."""

import pytest

from evret.metrics import ERR


class TestERRBasicScoring:
    """Test basic ERR scoring behavior."""

    def test_perfect_err_most_relevant_at_top(self) -> None:
        """Perfect ranking with max-grade doc at rank 1."""
        metric = ERR(k=5, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_doc_ids={"d1": 4},  # Max grade at position 1
        )
        # With grade 4 at position 1: R(1) = (2^4 - 1) / 2^4 = 15/16 = 0.9375
        # ERR = (1/1) × 0.9375 × 1 = 0.9375
        assert score == pytest.approx(0.9375, rel=1e-6)

    def test_zero_err_no_relevant(self) -> None:
        """ERR should be 0 when no relevant documents."""
        metric = ERR(k=5, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={},
        )
        assert score == 0.0

    def test_zero_err_empty_retrieved(self) -> None:
        """ERR should be 0 when no documents retrieved."""
        metric = ERR(k=5, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={"d1": 4},
        )
        assert score == 0.0

    def test_binary_relevance_as_grade_1(self) -> None:
        """Binary relevance (set) should be treated as grade 1."""
        metric = ERR(k=3, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1"},  # Binary: grade 1
        )
        # R(1) = (2^1 - 1) / 2^4 = 1/16 = 0.0625
        # ERR = (1/1) × 0.0625 × 1 = 0.0625
        assert score == pytest.approx(0.0625, rel=1e-6)


class TestERRGradedRelevance:
    """Test ERR with graded relevance."""

    def test_higher_grade_higher_contribution(self) -> None:
        """Higher relevance grade should contribute more."""
        metric = ERR(k=1, max_grade=4)

        score_grade_1 = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": 1},
        )
        score_grade_2 = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": 2},
        )
        score_grade_4 = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": 4},
        )

        assert score_grade_1 < score_grade_2 < score_grade_4

    def test_graded_relevance_cascade_model(self) -> None:
        """Test cascade model with multiple grades."""
        metric = ERR(k=3, max_grade=4)
        # Rank 1: grade=3, Rank 2: grade=4, Rank 3: grade=1
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1": 3, "d2": 4, "d3": 1},
        )
        # R(1) = (2^3 - 1) / 2^4 = 7/16 = 0.4375
        # R(2) = (2^4 - 1) / 2^4 = 15/16 = 0.9375
        # R(3) = (2^1 - 1) / 2^4 = 1/16 = 0.0625
        #
        # Position 1: (1/1) × 0.4375 × 1 = 0.4375
        # Position 2: (1/2) × 0.9375 × (1-0.4375) = 0.5 × 0.9375 × 0.5625 = 0.2637
        # Position 3: (1/3) × 0.0625 × (1-0.4375) × (1-0.9375)
        #           = 0.3333 × 0.0625 × 0.5625 × 0.0625 = 0.0007
        expected = 0.4375 + 0.2637 + 0.0007
        assert score == pytest.approx(expected, abs=0.001)

    def test_grade_zero_irrelevant(self) -> None:
        """Grade 0 should be treated as irrelevant."""
        metric = ERR(k=3, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1": 0, "d2": 4},
        )
        # Position 1: grade 0, R(1) = 0
        # Position 2: grade 4, R(2) = 0.9375
        # ERR = 0 + (1/2) × 0.9375 × 1 = 0.46875
        assert score == pytest.approx(0.46875, rel=1e-6)


class TestERRCascadeModel:
    """Test cascade probability behavior."""

    def test_cascade_probability_decreases(self) -> None:
        """Later positions have lower contribution due to cascade."""
        metric = ERR(k=3, max_grade=4)

        # Same grade at different positions
        score_position_1 = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1": 2},
        )
        score_position_3 = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d3": 2},
        )

        assert score_position_1 > score_position_3

    def test_high_grade_blocks_later_positions(self) -> None:
        """High-grade doc at top reduces contribution from later positions."""
        metric = ERR(k=5, max_grade=4)

        # Max grade at position 1 (blocks later)
        score_top_heavy = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_doc_ids={"d1": 4, "d5": 4},
        )
        # Max grade at position 5 (no blocking)
        score_bottom_heavy = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4", "d5"],
            relevant_doc_ids={"d5": 4},
        )

        # Top-heavy should have higher score due to position weighting
        assert score_top_heavy > score_bottom_heavy


class TestERRWithKCutoff:
    """Test ERR respects k cutoff."""

    def test_respects_k_cutoff(self) -> None:
        """ERR should only consider top-k positions."""
        metric = ERR(k=2, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4"],
            relevant_doc_ids={"d1": 4, "d2": 3},
        )
        # Only positions 1 and 2 are considered
        # R(1) = 0.9375, R(2) = 0.4375
        # ERR = 1.0 × 0.9375 + 0.5 × 0.4375 × (1-0.9375)
        expected = 0.9375 + 0.5 * 0.4375 * 0.0625
        assert score == pytest.approx(expected, rel=1e-6)

    def test_relevant_beyond_k_ignored(self) -> None:
        """Relevant docs beyond k should be ignored."""
        metric = ERR(k=2, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3", "d4"],
            relevant_doc_ids={"d3": 4, "d4": 4},  # Beyond k=2
        )
        assert score == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """ERR should handle k > retrieved length."""
        metric = ERR(k=10, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            relevant_doc_ids={"d1": 4},
        )
        assert score == pytest.approx(0.9375, rel=1e-6)


class TestERREdgeCases:
    """Test edge cases and boundary conditions."""

    def test_both_empty_returns_zero(self) -> None:
        """Empty retrieved and relevant should return 0."""
        metric = ERR(k=5, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=[],
            relevant_doc_ids={},
        )
        assert score == 0.0

    def test_single_doc_perfect_grade(self) -> None:
        """Single doc with max grade."""
        metric = ERR(k=1, max_grade=4)
        score = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": 4},
        )
        assert score == pytest.approx(0.9375, rel=1e-6)

    def test_negative_grade_clamped_to_zero(self) -> None:
        """Negative grades should be treated as 0."""
        metric = ERR(k=3, max_grade=4)
        # Implementation should handle this gracefully
        score = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": -1},
        )
        assert score == 0.0

    def test_grade_exceeds_max_clamped(self) -> None:
        """Grades above max_grade should be clamped."""
        metric = ERR(k=1, max_grade=4)
        # Grade 5 should be clamped to 4
        score = metric.score_query(
            retrieved_doc_ids=["d1"],
            relevant_doc_ids={"d1": 5},
        )
        expected_max_grade = (2**4 - 1) / 2**4
        assert score == pytest.approx(expected_max_grade, rel=1e-6)


class TestERRBatchScoring:
    """Test batch evaluation."""

    def test_batch_score_averages_correctly(self) -> None:
        """Batch scoring should average per-query scores."""
        metric = ERR(k=3, max_grade=4)
        score = metric.score(
            retrieved_by_query=[
                ["d1", "d2", "d3"],
                ["d4", "d5", "d6"],
            ],
            relevant_by_query=[
                {"d1": 4},
                {"d6": 2},
            ],
        )
        # Query 1: ERR = 0.9375 (grade 4 at pos 1)
        # Query 2: ERR = (1/3) × (3/16) × 1 = 0.0625
        expected = (0.9375 + 0.0625) / 2
        assert score == pytest.approx(expected, rel=1e-4)

    def test_empty_batch_returns_zero(self) -> None:
        """Empty batch should return 0."""
        metric = ERR(k=5, max_grade=4)
        score = metric.score(retrieved_by_query=[], relevant_by_query=[])
        assert score == 0.0


class TestERRMaxGradeParameter:
    """Test different max_grade values."""

    @pytest.mark.parametrize("max_grade", [1, 2, 3, 4, 5])
    def test_different_max_grades(self, max_grade: int) -> None:
        """Test ERR with different max_grade values."""
        metric = ERR(k=3, max_grade=max_grade)
        score = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            relevant_doc_ids={"d1": max_grade},
        )
        # R(1) = (2^max_grade - 1) / 2^max_grade
        expected = (2**max_grade - 1) / 2**max_grade
        assert score == pytest.approx(expected, rel=1e-6)

    def test_invalid_max_grade_raises_error(self) -> None:
        """max_grade < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_grade must be at least 1"):
            ERR(k=5, max_grade=0)


class TestERRSatisfactionProbability:
    """Test satisfaction probability calculation."""

    def test_satisfaction_probability_grade_0(self) -> None:
        """Grade 0 should have 0 satisfaction probability."""
        metric = ERR(k=5, max_grade=4)
        prob = metric._compute_satisfaction_probability(0)
        assert prob == 0.0

    def test_satisfaction_probability_max_grade(self) -> None:
        """Max grade should have high satisfaction probability."""
        metric = ERR(k=5, max_grade=4)
        prob = metric._compute_satisfaction_probability(4)
        # R(4) = (2^4 - 1) / 2^4 = 15/16
        assert prob == pytest.approx(15 / 16, rel=1e-6)

    def test_satisfaction_probability_increases_with_grade(self) -> None:
        """Satisfaction probability should increase with grade."""
        metric = ERR(k=5, max_grade=4)
        probs = [metric._compute_satisfaction_probability(g) for g in range(5)]
        # Ensure monotonic increase
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]


class TestERRMetricName:
    """Test metric name formatting."""

    def test_default_name_omits_max_grade(self) -> None:
        """Default max_grade=4 should be omitted from name."""
        metric = ERR(k=10, max_grade=4)
        assert metric.name == "err@10"

    def test_custom_max_grade_in_name(self) -> None:
        """Non-default max_grade should appear in name."""
        metric = ERR(k=10, max_grade=5)
        assert metric.name == "err(max_grade=5)@10"


class TestERRVsOtherMetrics:
    """Compare ERR behavior with other metrics."""

    def test_err_position_sensitive(self) -> None:
        """ERR should be sensitive to document positions."""
        metric = ERR(k=3, max_grade=4)

        score_good_order = metric.score_query(
            retrieved_doc_ids=["d1", "d2", "d3"],
            relevant_doc_ids={"d1": 4, "d3": 2},
        )
        score_bad_order = metric.score_query(
            retrieved_doc_ids=["d2", "d3", "d1"],
            relevant_doc_ids={"d1": 4, "d3": 2},
        )

        # Higher grade at top should yield higher ERR
        assert score_good_order > score_bad_order

    def test_err_handles_graded_unlike_binary_metrics(self) -> None:
        """ERR should differentiate between relevance grades."""
        metric = ERR(k=2, max_grade=4)

        score_high_grades = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            relevant_doc_ids={"d1": 4, "d2": 4},
        )
        score_low_grades = metric.score_query(
            retrieved_doc_ids=["d1", "d2"],
            relevant_doc_ids={"d1": 1, "d2": 1},
        )

        assert score_high_grades > score_low_grades
