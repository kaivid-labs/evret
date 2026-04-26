"""Anthropic LLM provider implementation."""

from __future__ import annotations

import os

from evret.errors import OptionalDependencyError
from evret.judges.llm.providers.base import LLMProvider

try:
    from anthropic import Anthropic, AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    Anthropic = None
    AsyncAnthropic = None
    HAS_ANTHROPIC = False


class AnthropicProvider(LLMProvider):
    """Anthropic API provider for LLM judge."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """Initialize Anthropic provider.

        Args:
            model: Model name (default: claude-3-5-haiku-20241022)
            api_key: Anthropic API key (reads from ANTHROPIC_API_KEY env if None)
            temperature: Sampling temperature
            max_retries: Max retry attempts for failed requests
        """
        if not HAS_ANTHROPIC:
            raise OptionalDependencyError(
                "Anthropic provider requires anthropic package. "
                "Install with: pip install anthropic"
            )

        self.model = model or self.default_model
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not provided. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )

        self._client = Anthropic(api_key=api_key, max_retries=max_retries)
        self._async_client = AsyncAnthropic(api_key=api_key, max_retries=max_retries)

    @property
    def default_model(self) -> str:
        """Default Anthropic model."""
        return "claude-3-5-haiku-20241022"

    def complete(self, prompt: str) -> str:
        """Synchronous completion via Anthropic API."""
        response = self._client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=10,  # "YES" or "NO" response
        )
        return response.content[0].text.strip()

    async def acomplete(self, prompt: str) -> str:
        """Asynchronous completion via Anthropic API."""
        response = await self._async_client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=10,
        )
        return response.content[0].text.strip()
