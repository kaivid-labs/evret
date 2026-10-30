"""Tests for the TruLens-backed judge."""

from __future__ import annotations

import pytest

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.judges import trulens
from evret.judges.base import JudgmentContext


class FakeTruLensProvider:
    """Stand-in for a TruLens provider exposing feedback functions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def context_relevance(self, prompt, context):
        self.calls.append(("context_relevance", prompt, context))
        return 0.8

    def relevance(self, prompt, context):
        self.calls.append(("relevance", prompt, context))
        return 0.3


@pytest.fixture
def patched_openai(monkeypatch):
    monkeypatch.setattr(trulens, "HAS_TRULENS_OPENAI", True)
    monkeypatch.setattr(trulens, "_TruLensOpenAI", FakeTruLensProvider)
    return FakeTruLensProvider


def _context(query="q", retrieved="r", expected="e"):
    return JudgmentContext(query=query, expected_text=expected, retrieved_text=retrieved)


def test_defaults_use_openai_and_context_relevance(patched_openai):
    judge = trulens.TruLensJudge()

    assert judge.provider_name == "openai"
    assert judge.metric == "context_relevance"
    assert judge.threshold == 0.5
    # model is None -> model_engine not forwarded to provider
    assert "model_engine" not in judge._provider.kwargs


def test_model_is_forwarded_as_model_engine(patched_openai):
    judge = trulens.TruLensJudge(model="gpt-6-luna")

    assert judge._provider.kwargs["model_engine"] == "gpt-6-luna"


def test_score_returns_raw_float_and_uses_query_and_retrieved(patched_openai):
    judge = trulens.TruLensJudge()

    value = judge.score(_context(query="what is react", retrieved="react is..."))

    assert value == 0.8
    assert judge._provider.calls == [("context_relevance", "what is react", "react is...")]


def test_judge_thresholds_score_to_bool(patched_openai):
    assert trulens.TruLensJudge(threshold=0.5).judge(_context()) is True
    assert trulens.TruLensJudge(threshold=0.9).judge(_context()) is False


def test_relevance_metric_selected(patched_openai):
    judge = trulens.TruLensJudge(metric="relevance")

    assert judge.score(_context()) == 0.3
    assert judge._provider.calls[0][0] == "relevance"


def test_invalid_metric_raises():
    with pytest.raises(EvretValidationError, match="Unsupported TruLens metric"):
        trulens.TruLensJudge(metric="groundedness")


def test_invalid_threshold_raises(patched_openai):
    with pytest.raises(EvretValidationError, match="threshold must be in"):
        trulens.TruLensJudge(threshold=1.5)


def test_unsupported_provider_raises(patched_openai):
    with pytest.raises(ValueError, match="Unsupported TruLens provider"):
        trulens.TruLensJudge(provider="bedrock")


def test_missing_dependency_raises(monkeypatch):
    monkeypatch.setattr(trulens, "HAS_TRULENS_OPENAI", False)
    with pytest.raises(OptionalDependencyError, match="trulens-providers-openai"):
        trulens.TruLensJudge(provider="openai")


def test_litellm_provider_requires_package(monkeypatch):
    monkeypatch.setattr(trulens, "HAS_TRULENS_LITELLM", False)
    with pytest.raises(OptionalDependencyError, match="trulens-providers-litellm"):
        trulens.TruLensJudge(provider="litellm", model="claude-3-5-sonnet-latest")


def test_litellm_provider_builds(monkeypatch):
    monkeypatch.setattr(trulens, "HAS_TRULENS_LITELLM", True)
    monkeypatch.setattr(trulens, "_TruLensLiteLLM", FakeTruLensProvider)

    judge = trulens.TruLensJudge(provider="litellm", model="gemini/gemini-3.5-flash-lite")

    assert judge._provider.kwargs["model_engine"] == "gemini/gemini-3.5-flash-lite"


def test_exported_via_judges_namespace():
    from evret.judges import TruLensJudge

    assert TruLensJudge is trulens.TruLensJudge
