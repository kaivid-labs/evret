"""Utility functions for judge implementations."""

from __future__ import annotations

import re

# Token pattern: alphanumeric sequences only
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase alphanumeric tokens

    Examples:
        >>> tokenize("Hello World!")
        ['hello', 'world']
        >>> tokenize("RAG-based search: 123")
        ['rag', 'based', 'search', '123']
    """
    return _TOKEN_PATTERN.findall(text.lower())


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace).

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return " ".join(text.strip().lower().split())
