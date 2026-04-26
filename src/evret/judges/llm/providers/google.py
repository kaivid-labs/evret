"""Google Gen AI LLM provider implementation."""

from __future__ import annotations

import os

from evret.errors import OptionalDependencyError
from evret.judges.llm.providers.base import LLMProvider

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    genai = None
    types = None
    HAS_GOOGLE_GENAI = False


class GoogleGenAIProvider(LLMProvider):
    """Google Gen AI API provider for LLM judge."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """Initialize Google Gen AI provider.

        Args:
            model: Model name (default: gemini-2.5-flash)
            api_key: Google Gen AI API key (reads from GEMINI_API_KEY or GOOGLE_API_KEY)
            temperature: Sampling temperature
            max_retries: Max retry attempts for failed requests
        """
        if not HAS_GOOGLE_GENAI:
            raise OptionalDependencyError(
                "Google Gen AI provider requires google-genai package. "
                "Install with: pip install google-genai"
            )

        self.model = model or self.default_model
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google Gen AI API key not provided. "
                "Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        self._client = genai.Client(api_key=api_key)
        self._async_client = self._client.aio

    @property
    def default_model(self) -> str:
        """Default Google Gen AI model."""
        return "gemini-2.5-flash"

    def complete(self, prompt: str) -> str:
        """Synchronous completion via Google Gen AI API."""
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=10,
            ),
        )
        return (response.text or "").strip()

    async def acomplete(self, prompt: str) -> str:
        """Asynchronous completion via Google Gen AI API."""
        response = await self._async_client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=10,
            ),
        )
        return (response.text or "").strip()
