"""Tests for OpenAI LLM provider."""

from __future__ import annotations

import asyncio

import pytest

from evret.judges.llm.factory import llm_provider_factory
from evret.judges.llm.providers import openai


class FakeMessage:
    content = "  YES  "


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeAsyncCompletions:
    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeAsyncChat:
    def __init__(self):
        self.completions = FakeAsyncCompletions()


class FakeOpenAI:
    def __init__(self, api_key, max_retries):
        self.api_key = api_key
        self.max_retries = max_retries
        self.chat = FakeChat()


class FakeAsyncOpenAI:
    def __init__(self, api_key, max_retries):
        self.api_key = api_key
        self.max_retries = max_retries
        self.chat = FakeAsyncChat()


def _patch_openai(monkeypatch) -> None:
    monkeypatch.setattr(openai, "HAS_OPENAI", True)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeAsyncOpenAI)


def test_openai_provider_uses_default_model_and_api_key(monkeypatch) -> None:
    _patch_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    provider = llm_provider_factory("openai")

    assert provider.default_model == "gpt-5.4-nano"
    assert provider.model == "gpt-5.4-nano"
    assert provider._client.api_key == "test-key"
    assert provider._client.max_retries == 3


def test_openai_provider_requires_api_key(monkeypatch) -> None:
    _patch_openai(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OpenAI API key not provided"):
        llm_provider_factory("openai")


def test_openai_provider_complete(monkeypatch) -> None:
    _patch_openai(monkeypatch)

    provider = llm_provider_factory("openai", api_key="key", temperature=0.2)

    assert provider.complete("Judge this") == "YES"
    kwargs = provider._client.chat.completions.kwargs
    assert kwargs["model"] == "gpt-5.4-nano"
    assert kwargs["messages"] == [{"role": "user", "content": "Judge this"}]
    # temperature is no longer forwarded (gpt-5.4-nano rejects a custom value)
    assert "temperature" not in kwargs


def test_openai_provider_acomplete(monkeypatch) -> None:
    _patch_openai(monkeypatch)

    provider = llm_provider_factory("openai", api_key="key", temperature=0.2)

    assert asyncio.run(provider.acomplete("Judge this")) == "YES"
    kwargs = provider._async_client.chat.completions.kwargs
    assert kwargs["model"] == "gpt-5.4-nano"
    assert kwargs["messages"] == [{"role": "user", "content": "Judge this"}]
    assert "temperature" not in kwargs
