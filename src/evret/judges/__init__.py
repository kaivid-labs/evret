"""Relevance judges for text-based evaluation matching."""

from __future__ import annotations

from evret.judges.base import Judge, JudgmentContext
from evret.judges.token_overlap import TokenOverlapJudge

__all__ = [
    "Judge",
    "JudgmentContext",
    "TokenOverlapJudge",
]

def __getattr__(name: str):
    if name == "SemanticJudge":
        from evret.judges.semantic import SemanticJudge
        return SemanticJudge
    elif name == "LLMJudge":
        from evret.judges.llm import LLMJudge
        return LLMJudge
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
