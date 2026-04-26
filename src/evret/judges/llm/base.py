"""LLM-powered semantic relevance judge."""

from __future__ import annotations

import asyncio

from evret.judges.base import Judge, JudgmentContext
from evret.judges.llm.factory import llm_provider_factory
from evret.judges.llm.prompts import RELEVANCE_PROMPT_TEMPLATE


class LLMJudge(Judge):
    """LLM-powered semantic relevance judgment.

    Uses GPT/Claude/other LLMs to determine if retrieved text matches
    expected content semantically. Most accurate but slowest judge option.

    Examples:
        >>> judge = LLMJudge(provider="openai")  # Uses OPENAI_API_KEY env
        >>> judge = LLMJudge(provider="openai", api_key="sk-...")
        >>> judge = LLMJudge(provider="anthropic", model="claude-3-5-sonnet-20241022")
        >>> judge = LLMJudge(provider="google", model="gemini-2.5-flash")

    Args:
        provider: LLM provider ("openai", "anthropic", or "google")
        model: Model name (uses provider default if None)
        api_key: API key (reads from env if None)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_retries: Max retry attempts for failed API calls

    Requires:
        pip install openai  # for OpenAI
        pip install anthropic  # for Anthropic
        pip install google-genai  # for Google Gen AI
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """Initialize LLM judge with specified provider."""
        self.provider_name = provider
        self.model_name = model
        self._provider = llm_provider_factory(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
        )

    @property
    def name(self) -> str:
        """Judge display name."""
        model_display = self.model_name or self._provider.default_model
        return f"llm({self.provider_name}:{model_display})"

    def judge(self, context: JudgmentContext) -> bool:
        """Judge using LLM prompt.

        Args:
            context: Judgment context with query and texts

        Returns:
            True if LLM determines texts are semantically relevant
        """
        prompt = self._build_prompt(context)
        response = self._provider.complete(prompt)
        return self._parse_response(response)

    async def ajudge(self, context: JudgmentContext) -> bool:
        """Async judge using LLM.

        Args:
            context: Judgment context

        Returns:
            Boolean relevance judgment
        """
        prompt = self._build_prompt(context)
        response = await self._provider.acomplete(prompt)
        return self._parse_response(response)

    def batch_judge(self, contexts: list[JudgmentContext]) -> list[bool]:
        """Batch evaluation with concurrent async API calls.

        Args:
            contexts: List of judgment contexts

        Returns:
            List of boolean judgments
        """
        if not contexts:
            return []
        return asyncio.run(self._abatch_judge(contexts))

    async def _abatch_judge(self, contexts: list[JudgmentContext]) -> list[bool]:
        """Internal async batch implementation."""
        tasks = [self.ajudge(ctx) for ctx in contexts]
        return await asyncio.gather(*tasks)

    def _build_prompt(self, context: JudgmentContext) -> str:
        """Build relevance judgment prompt from template."""
        return RELEVANCE_PROMPT_TEMPLATE.format(
            query=context.query,
            expected_text=context.expected_text,
            retrieved_text=context.retrieved_text,
        )

    @staticmethod
    def _parse_response(response: str) -> bool:
        """Parse LLM response into boolean relevance judgment.

        Args:
            response: LLM response text

        Returns:
            True if response indicates relevance, False otherwise
        """
        response_lower = response.strip().lower()

        # Check for explicit yes/no at start
        if response_lower.startswith("yes"):
            return True
        if response_lower.startswith("no"):
            return False

        # Check for boolean keywords
        positive_keywords = ["relevant", "match", "correct", "accurate", "true"]
        negative_keywords = ["irrelevant", "mismatch", "incorrect", "inaccurate", "false"]

        has_positive = any(kw in response_lower for kw in positive_keywords)
        has_negative = any(kw in response_lower for kw in negative_keywords)

        if has_positive and not has_negative:
            return True
        if has_negative and not has_positive:
            return False

        # Default to False if ambiguous
        return False
