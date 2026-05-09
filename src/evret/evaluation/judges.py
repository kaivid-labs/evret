"""Relevance judge helpers for text-based evaluation matching."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from evret.errors import EvretValidationError
from evret.logging import get_logger

RelevanceJudge = Callable[[str, str, str], bool]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
logger = get_logger(__name__)


def default_relevance_judge(query_text: str, relevant_label: str, candidate_text: str) -> bool:
    """Return whether one candidate matches a relevance label."""
    if relevant_label == candidate_text:
        logger.debug("Default relevance judge exact match", extra={"strategy": "exact"})
        return True
    if " " in relevant_label or " " in candidate_text:
        if relevant_label in candidate_text or candidate_text in relevant_label:
            logger.debug(
                "Default relevance judge substring match",
                extra={"strategy": "substring"},
            )
            return True
    result = token_overlap_relevance_judge(
        query_text=query_text,
        relevant_label=relevant_label,
        candidate_text=candidate_text,
    )
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Default relevance judge token overlap fallback",
            extra={"strategy": "token_overlap", "decision": result},
        )
    return result


def token_overlap_relevance_judge(
    query_text: str,
    relevant_label: str,
    candidate_text: str,
    *,
    min_shared_tokens: int = 2,
    min_overlap_ratio: float = 0.6,
) -> bool:
    """Return True when token overlap passes configured thresholds."""
    if min_shared_tokens < 1:
        raise EvretValidationError("min_shared_tokens must be >= 1")
    if min_overlap_ratio <= 0 or min_overlap_ratio > 1:
        raise EvretValidationError("min_overlap_ratio must be in (0, 1]")

    label_tokens = set(_tokenize(relevant_label))
    candidate_tokens = set(_tokenize(candidate_text))
    if not label_tokens or not candidate_tokens:
        logger.debug(
            "Token overlap relevance judge has empty token set",
            extra={"decision": False},
        )
        return False

    shared_tokens = label_tokens.intersection(candidate_tokens)
    if len(shared_tokens) < min_shared_tokens:
        logger.debug(
            "Token overlap relevance judge below shared token threshold",
            extra={
                "shared_tokens": len(shared_tokens),
                "min_shared_tokens": min_shared_tokens,
                "decision": False,
            },
        )
        return False

    overlap_ratio = len(shared_tokens) / len(label_tokens)
    if overlap_ratio >= min_overlap_ratio:
        logger.debug(
            "Token overlap relevance judge passed overlap ratio",
            extra={"overlap_ratio": overlap_ratio, "decision": True},
        )
        return True

    query_tokens = set(_tokenize(query_text))
    if not query_tokens:
        logger.debug("Token overlap relevance judge missing query tokens", extra={"decision": False})
        return False

    query_condition = len(shared_tokens.intersection(query_tokens)) >= 1
    decision = query_condition and overlap_ratio >= (min_overlap_ratio * 0.75)
    logger.debug(
        "Token overlap relevance judge relaxed threshold decision",
        extra={
            "query_condition": query_condition,
            "overlap_ratio": overlap_ratio,
            "decision": decision,
        },
    )
    return decision


def make_token_overlap_judge(
    *,
    min_shared_tokens: int = 2,
    min_overlap_ratio: float = 0.6,
) -> RelevanceJudge:
    """Build a token-overlap relevance judge with custom thresholds."""
    if min_shared_tokens < 1:
        raise EvretValidationError("min_shared_tokens must be >= 1")
    if min_overlap_ratio <= 0 or min_overlap_ratio > 1:
        raise EvretValidationError("min_overlap_ratio must be in (0, 1]")

    def judge(query_text: str, relevant_label: str, candidate_text: str) -> bool:
        return token_overlap_relevance_judge(
            query_text=query_text,
            relevant_label=relevant_label,
            candidate_text=candidate_text,
            min_shared_tokens=min_shared_tokens,
            min_overlap_ratio=min_overlap_ratio,
        )

    return judge


def _tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(value.lower())
