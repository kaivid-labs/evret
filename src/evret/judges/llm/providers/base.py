"""Base LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base interface for LLM providers."""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model name for this provider."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Synchronous completion.

        Args:
            prompt: Input prompt text

        Returns:
            Model response text
        """

    @abstractmethod
    async def acomplete(self, prompt: str) -> str:
        """Asynchronous completion.

        Args:
            prompt: Input prompt text

        Returns:
            Model response text
        """
