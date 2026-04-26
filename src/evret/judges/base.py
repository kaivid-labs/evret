"""Base interface for relevance judges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class JudgmentContext:
    """Context passed to judge for relevance decision.

    Attributes:
        query: User query text
        expected_text: Expected/ground-truth relevant text
        retrieved_text: Retrieved candidate text to judge
    """
    query: str
    expected_text: str
    retrieved_text: str


class Judge(ABC):
    """Base interface for relevance judges.

    All judges implement a simple contract:
    - judge(context) → bool (is relevant?)
    - batch_judge(contexts) → list[bool] (batch evaluation)

    Subclasses should override judge() and optionally batch_judge()
    for optimized batch processing.
    """

    @abstractmethod
    def judge(self, context: JudgmentContext) -> bool:
        """Return True if retrieved_text is relevant to expected_text given query.

        Args:
            context: Judgment context with query and texts

        Returns:
            True if retrieved text is relevant, False otherwise
        """

    def batch_judge(self, contexts: list[JudgmentContext]) -> list[bool]:
        """Batch evaluation of multiple contexts.

        Default implementation calls judge() for each context sequentially.
        Override this method for optimized batch processing (e.g., vectorized
        operations, async API calls, etc.).

        Args:
            contexts: List of judgment contexts

        Returns:
            List of boolean judgments (same order as input)
        """
        return [self.judge(ctx) for ctx in contexts]

    @property
    @abstractmethod
    def name(self) -> str:
        """Judge display name for logging/debugging."""
