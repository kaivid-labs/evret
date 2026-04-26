"""Token overlap judge for keyword-based relevance matching."""

from __future__ import annotations

from evret.errors import EvretValidationError
from evret.judges.base import Judge, JudgmentContext
from evret.judges.utils import tokenize, normalize_text


class TokenOverlapJudge(Judge):
    """Fast keyword/token-based relevance matching.

    Suitable for exact/fuzzy text matching without semantic understanding.
    Uses token overlap with configurable thresholds to determine relevance.

    Algorithm:
        1. Try exact match
        2. Try substring containment
        3. Check token overlap with minimum token and ratio thresholds
        4. Optionally boost with query token overlap

    Examples:
        >>> judge = TokenOverlapJudge()  # Default settings
        >>> judge = TokenOverlapJudge(min_tokens=3, overlap_ratio=0.7)
        >>> judge = TokenOverlapJudge(min_tokens=2, overlap_ratio=0.6, query_boost=False)

    Args:
        min_tokens: Minimum shared tokens required (default: 2)
        overlap_ratio: Minimum overlap ratio 0-1 (default: 0.6)
        query_boost: Allow query tokens to relax threshold (default: True)
    """

    def __init__(
        self,
        min_tokens: int = 2,
        overlap_ratio: float = 0.6,
        query_boost: bool = True,
    ):
        """Initialize token overlap judge with configurable thresholds."""
        self._validate_params(min_tokens, overlap_ratio)
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
        self.query_boost = query_boost

    @property
    def name(self) -> str:
        """Judge display name."""
        return f"token_overlap(min={self.min_tokens},ratio={self.overlap_ratio})"

    def judge(self, context: JudgmentContext) -> bool:
        """Judge using token overlap algorithm.

        Args:
            context: Judgment context with query and texts

        Returns:
            True if retrieved text matches expected text
        """
        expected_norm = normalize_text(context.expected_text)
        retrieved_norm = normalize_text(context.retrieved_text)

        # 1. Exact match
        if expected_norm == retrieved_norm:
            return True

        # 2. Substring match
        if self._substring_match(expected_norm, retrieved_norm):
            return True

        # 3. Token overlap
        return self._token_overlap_match(context)

    def _token_overlap_match(self, context: JudgmentContext) -> bool:
        """Core token overlap algorithm."""
        expected_tokens = set(tokenize(context.expected_text))
        retrieved_tokens = set(tokenize(context.retrieved_text))

        if not expected_tokens or not retrieved_tokens:
            return False

        shared = expected_tokens & retrieved_tokens

        # Check minimum tokens
        if len(shared) < self.min_tokens:
            return False

        # Check overlap ratio
        ratio = len(shared) / len(expected_tokens)
        if ratio >= self.overlap_ratio:
            return True

        # Query boost fallback
        if self.query_boost:
            query_tokens = set(tokenize(context.query))
            if query_tokens and len(shared & query_tokens) >= 1:
                relaxed_ratio = self.overlap_ratio * 0.75
                return ratio >= relaxed_ratio

        return False

    @staticmethod
    def _substring_match(expected: str, retrieved: str) -> bool:
        """Check if either text contains the other."""
        if len(expected) > 0 and len(retrieved) > 0:
            return expected in retrieved or retrieved in expected
        return False

    @staticmethod
    def _validate_params(min_tokens: int, overlap_ratio: float) -> None:
        """Validate initialization parameters."""
        if min_tokens < 1:
            raise EvretValidationError("min_tokens must be >= 1")
        if not 0 < overlap_ratio <= 1:
            raise EvretValidationError("overlap_ratio must be in (0, 1]")
