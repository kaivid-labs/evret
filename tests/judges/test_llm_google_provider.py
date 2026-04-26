"""Tests for Google Gen AI LLM provider."""

from __future__ import annotations

import pytest

from evret.judges.llm.factory import llm_provider_factory
from evret.judges.llm.providers import google


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponse:
    text = "YES"


class FakeModels:
    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeAsyncModels:
    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeAsyncClient:
    def __init__(self):
        self.models = FakeAsyncModels()


class FakeClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.models = FakeModels()
        self.aio = FakeAsyncClient()


class FakeGenAI:
    Client = FakeClient


class FakeTypes:
    GenerateContentConfig = FakeGenerateContentConfig


def test_google_provider_uses_default_model_and_api_key(monkeypatch) -> None:
    monkeypatch.setattr(google, "HAS_GOOGLE_GENAI", True)
    monkeypatch.setattr(google, "genai", FakeGenAI)
    monkeypatch.setattr(google, "types", FakeTypes)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    provider = llm_provider_factory("google")

    assert provider.default_model == "gemini-2.5-flash"
    assert provider.model == "gemini-2.5-flash"
    assert provider._client.api_key == "test-key"


def test_google_provider_supports_aliases(monkeypatch) -> None:
    monkeypatch.setattr(google, "HAS_GOOGLE_GENAI", True)
    monkeypatch.setattr(google, "genai", FakeGenAI)
    monkeypatch.setattr(google, "types", FakeTypes)

    assert llm_provider_factory("google-genai", api_key="key").model == "gemini-2.5-flash"
    assert llm_provider_factory("gemini", api_key="key").model == "gemini-2.5-flash"


def test_google_provider_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr(google, "HAS_GOOGLE_GENAI", True)
    monkeypatch.setattr(google, "genai", FakeGenAI)
    monkeypatch.setattr(google, "types", FakeTypes)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Google Gen AI API key not provided"):
        llm_provider_factory("google")


def test_google_provider_complete(monkeypatch) -> None:
    monkeypatch.setattr(google, "HAS_GOOGLE_GENAI", True)
    monkeypatch.setattr(google, "genai", FakeGenAI)
    monkeypatch.setattr(google, "types", FakeTypes)

    provider = llm_provider_factory("google", api_key="key", temperature=0.2)

    assert provider.complete("Judge this") == "YES"
    assert provider._client.models.kwargs["model"] == "gemini-2.5-flash"
    assert provider._client.models.kwargs["contents"] == "Judge this"
    assert provider._client.models.kwargs["config"].kwargs == {
        "temperature": 0.2,
        "max_output_tokens": 10,
    }
