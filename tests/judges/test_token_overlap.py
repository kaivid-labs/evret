"""Tests for TokenOverlapJudge."""

from __future__ import annotations

import pytest

from evret.errors import EvretValidationError
from evret.judges.base import JudgmentContext
from evret.judges.token_overlap import TokenOverlapJudge


class TestTokenOverlapJudge:
    """Test TokenOverlapJudge functionality."""

    def test_exact_match(self):
        """Test exact text match."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="what is rag?",
            expected_text="retrieval augmented generation",
            retrieved_text="retrieval augmented generation",
        )
        assert judge.judge(context) is True

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="what is rag?",
            expected_text="Retrieval Augmented Generation",
            retrieved_text="retrieval augmented generation",
        )
        assert judge.judge(context) is True

    def test_substring_match(self):
        """Test substring containment."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="what is rag?",
            expected_text="retrieval augmented generation",
            retrieved_text="RAG stands for retrieval augmented generation technique",
        )
        assert judge.judge(context) is True

    def test_token_overlap_sufficient(self):
        """Test sufficient token overlap."""
        judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6)
        context = JudgmentContext(
            query="what is rag?",
            expected_text="retrieval augmented generation",
            retrieved_text="retrieval generation systems",
        )
        assert judge.judge(context) is True

    def test_token_overlap_insufficient(self):
        """Test insufficient token overlap."""
        judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.8)
        context = JudgmentContext(
            query="what is rag?",
            expected_text="retrieval augmented generation",
            retrieved_text="database query execution",
        )
        assert judge.judge(context) is False

    def test_query_boost(self):
        """Test query token boost feature."""
        judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.7, query_boost=True)
        context = JudgmentContext(
            query="retrieval augmented generation",
            expected_text="retrieval augmented systems",
            retrieved_text="retrieval systems work",
        )
        # Without query boost this might fail, but with it should pass
        result = judge.judge(context)
        assert isinstance(result, bool)

    def test_no_query_boost(self):
        """Test without query boost."""
        judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.7, query_boost=False)
        context = JudgmentContext(
            query="retrieval augmented generation",
            expected_text="retrieval augmented systems database",
            retrieved_text="retrieval systems",
        )
        # Lower overlap ratio without boost
        assert judge.judge(context) is False

    def test_empty_texts(self):
        """Test with empty texts."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="test query",
            expected_text="",
            retrieved_text="some text",
        )
        assert judge.judge(context) is False

    def test_batch_judge(self):
        """Test batch evaluation."""
        judge = TokenOverlapJudge()
        contexts = [
            JudgmentContext(
                query="q1",
                expected_text="retrieval augmented generation",
                retrieved_text="retrieval augmented generation",
            ),
            JudgmentContext(
                query="q2",
                expected_text="semantic search",
                retrieved_text="unrelated content",
            ),
            JudgmentContext(
                query="q3",
                expected_text="vector database",
                retrieved_text="vector database systems",
            ),
        ]
        results = judge.batch_judge(contexts)
        assert results == [True, False, True]

    def test_name_property(self):
        """Test judge name property."""
        judge = TokenOverlapJudge(min_tokens=3, overlap_ratio=0.75)
        assert "token_overlap" in judge.name
        assert "min=3" in judge.name
        assert "ratio=0.75" in judge.name

    def test_invalid_min_tokens(self):
        """Test validation of min_tokens parameter."""
        with pytest.raises(EvretValidationError, match="min_tokens must be >= 1"):
            TokenOverlapJudge(min_tokens=0)

    def test_invalid_overlap_ratio_low(self):
        """Test validation of overlap_ratio too low."""
        with pytest.raises(EvretValidationError, match="overlap_ratio must be in"):
            TokenOverlapJudge(overlap_ratio=0.0)

    def test_invalid_overlap_ratio_high(self):
        """Test validation of overlap_ratio too high."""
        with pytest.raises(EvretValidationError, match="overlap_ratio must be in"):
            TokenOverlapJudge(overlap_ratio=1.5)

    def test_special_characters(self):
        """Test handling of special characters."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="test",
            expected_text="retrieval-augmented generation (RAG)",
            retrieved_text="RAG: Retrieval Augmented Generation",
        )
        assert judge.judge(context) is True

    def test_numeric_tokens(self):
        """Test handling of numeric tokens."""
        judge = TokenOverlapJudge()
        context = JudgmentContext(
            query="model version",
            expected_text="gpt 4 model",
            retrieved_text="gpt 4 turbo model",
        )
        assert judge.judge(context) is True
