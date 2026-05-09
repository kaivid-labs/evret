"""Token overlap judge for keyword-based relevance matching."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from evret.errors import EvretValidationError
from evret.judges.base import Judge, JudgmentContext
from evret.judges.constants import DEFAULT_STOPWORDS, NEGATION_TOKENS
from evret.judges.utils import tokenize, normalize_text
from evret.logging import get_logger

logger = get_logger(__name__)


class TokenOverlapJudge(Judge):
    """Fast keyword/token-based relevance matching.

    Suitable for exact/fuzzy text matching without semantic understanding.
    Uses weighted token overlap with configurable thresholds to determine relevance.

    Algorithm:
        1. Try exact match
        2. Try substring containment
        3. Remove common stopwords
        4. Score weighted token overlap
        5. Add small phrase and query overlap bonuses
        6. Penalize negation mismatches

    Examples:
        >>> judge = TokenOverlapJudge()  # Default settings
        >>> judge = TokenOverlapJudge(min_tokens=50, overlap_ratio=0.7)
        >>> judge = TokenOverlapJudge(min_tokens=30, overlap_ratio=0.6, query_boost=False)

    Args:
        min_tokens: Minimum shared tokens required (default: 30, suitable for RAG chunks)
        overlap_ratio: Minimum overlap ratio 0-1 (default: 0.6)
        query_boost: Allow query tokens to relax threshold (default: True)
        stopwords: Tokens ignored during overlap scoring
    """

    def __init__(
        self,
        min_tokens: int = 30,
        overlap_ratio: float = 0.6,
        query_boost: bool = True,
        stopwords: Iterable[str] | None = None,
    ):
        """Initialize token overlap judge with configurable thresholds."""
        self._validate_params(min_tokens, overlap_ratio)
        self.min_tokens = min_tokens
        self.overlap_ratio = overlap_ratio
        self.query_boost = query_boost
        self.stopwords = DEFAULT_STOPWORDS if stopwords is None else frozenset(stopwords)
        logger.debug(
            "Initialized TokenOverlapJudge",
            extra={
                "judge": self.name,
                "min_tokens": self.min_tokens,
                "overlap_ratio": self.overlap_ratio,
                "query_boost": self.query_boost,
            },
        )

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
        relevance_score = self.score(context)
        decision = relevance_score >= self.overlap_ratio
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Token overlap judgment computed",
                extra={
                    "judge": self.name,
                    "score": relevance_score,
                    "threshold": self.overlap_ratio,
                    "decision": decision,
                },
            )
        return decision

    def score(self, context: JudgmentContext) -> float:
        """Return a relevance score in the range [0, 1]."""
        expected_norm = normalize_text(context.expected_text)
        retrieved_norm = normalize_text(context.retrieved_text)

        if expected_norm == retrieved_norm:
            return 1.0

        if self._has_negation_mismatch(context.expected_text, context.retrieved_text):
            return 0.0

        if self._substring_match(expected_norm, retrieved_norm):
            return 1.0

        return self._token_overlap_score(context)

    def _token_overlap_score(self, context: JudgmentContext) -> float:
        """Core token overlap scoring algorithm."""
        expected_token_list = self._content_tokens(context.expected_text)
        retrieved_token_list = self._content_tokens(context.retrieved_text)
        expected_tokens = set(expected_token_list)
        retrieved_tokens = set(retrieved_token_list)

        if not expected_tokens or not retrieved_tokens:
            return 0.0

        shared = expected_tokens & retrieved_tokens

        if len(shared) < self.min_tokens:
            return 0.0

        score = self._weighted_overlap(shared, expected_tokens)
        score += self._phrase_bonus(expected_token_list, context.retrieved_text)

        if self.query_boost:
            query_tokens = set(self._content_tokens(context.query))
            if query_tokens and len(shared & query_tokens) >= 1:
                score += 0.1

        return self._clamp(score)

    def _content_tokens(self, text: str) -> list[str]:
        tokens = tokenize(text)
        filtered = [token for token in tokens if token not in self.stopwords]
        return filtered if filtered else []

    @staticmethod
    def _weighted_overlap(shared: set[str], expected_tokens: set[str]) -> float:
        shared_weight = sum(TokenOverlapJudge._token_weight(token) for token in shared)
        expected_weight = sum(TokenOverlapJudge._token_weight(token) for token in expected_tokens)
        if expected_weight == 0:
            return 0.0
        return shared_weight / expected_weight

    @staticmethod
    def _token_weight(token: str) -> float:
        if token.isdigit():
            return 1.5
        if len(token) <= 2:
            return 0.75
        if len(token) >= 8:
            return 1.35
        return 1.0

    def _phrase_bonus(self, expected_tokens: list[str], retrieved_text: str) -> float:
        if len(expected_tokens) < 2:
            return 0.0

        retrieved_norm = normalize_text(retrieved_text)
        max_bonus = 0.0
        for size in range(min(4, len(expected_tokens)), 1, -1):
            for start in range(0, len(expected_tokens) - size + 1):
                phrase = " ".join(expected_tokens[start : start + size])
                if phrase in retrieved_norm:
                    max_bonus = max(max_bonus, 0.05 * (size - 1))
            if max_bonus:
                break
        return min(max_bonus, 0.15)

    @staticmethod
    def _has_negation_mismatch(expected_text: str, retrieved_text: str) -> bool:
        expected_has_negation = bool(set(tokenize(expected_text)) & NEGATION_TOKENS)
        retrieved_has_negation = bool(set(tokenize(retrieved_text)) & NEGATION_TOKENS)
        return expected_has_negation != retrieved_has_negation

    @staticmethod
    def _clamp(score: float) -> float:
        return max(0.0, min(1.0, score))

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
