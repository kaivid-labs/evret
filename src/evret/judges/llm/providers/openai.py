"""OpenAI LLM provider implementation."""

from __future__ import annotations

import os

from evret.errors import OptionalDependencyError
from evret.judges.llm.providers.base import LLMProvider

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    AsyncOpenAI = None
    HAS_OPENAI = False


class OpenAIProvider(LLMProvider):
    """OpenAI API provider for LLM judge."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (default: gpt-4o-mini)
            api_key: OpenAI API key (reads from OPENAI_API_KEY env if None)
            temperature: Sampling temperature
            max_retries: Max retry attempts for failed requests
        """
        if not HAS_OPENAI:
            raise OptionalDependencyError(
                "OpenAI provider requires openai package. "
                "Install with: pip install openai"
            )

        self.model = model or self.default_model
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        self._client = OpenAI(api_key=api_key, max_retries=max_retries)
        self._async_client = AsyncOpenAI(api_key=api_key, max_retries=max_retries)

    @property
    def default_model(self) -> str:
        """Default OpenAI model."""
        return "gpt-4o-mini"

    def complete(self, prompt: str) -> str:
        """Synchronous completion via OpenAI API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=10,  # "YES" or "NO" response
        )
        return response.choices[0].message.content.strip()

    async def acomplete(self, prompt: str) -> str:
        """Asynchronous completion via OpenAI API."""
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=10,
        )
        return response.choices[0].message.content.strip()
