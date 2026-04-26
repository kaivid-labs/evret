"""LLM provider factory."""

from __future__ import annotations

from evret.judges.llm.providers.base import LLMProvider


def llm_provider_factory(
    provider: str,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> LLMProvider:
    """Create LLM provider instance.

    Args:
        provider: Provider name ("openai", "anthropic", or "google")
        model: Model name (uses provider default if None)
        api_key: API key (reads from env if None)
        temperature: Sampling temperature
        max_retries: Max retry attempts

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()

    if provider_lower == "openai":
        from evret.judges.llm.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
        )
    elif provider_lower == "anthropic":
        from evret.judges.llm.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
        )
    elif provider_lower in {"google", "google-genai", "gemini"}:
        from evret.judges.llm.providers.google import GoogleGenAIProvider
        return GoogleGenAIProvider(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: openai, anthropic, google"
        )
