"""LLM-as-a-judge relevance judge backed by TruLens feedback functions."""

from __future__ import annotations

import logging
from time import perf_counter

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.judges.base import Judge, JudgmentContext
from evret.logging import get_logger

try:
    from trulens.providers.openai import OpenAI as _TruLensOpenAI
    HAS_TRULENS_OPENAI = True
except ImportError:
    _TruLensOpenAI = None
    HAS_TRULENS_OPENAI = False

try:
    from trulens.providers.litellm import LiteLLM as _TruLensLiteLLM
    HAS_TRULENS_LITELLM = True
except ImportError:
    _TruLensLiteLLM = None
    HAS_TRULENS_LITELLM = False

logger = get_logger(__name__)

# TruLens feedback functions that take (query, retrieved_text) -> float in [0, 1].
SUPPORTED_METRICS = frozenset({"context_relevance", "relevance"})


class TruLensJudge(Judge):
    """Relevance judgment using a TruLens feedback function.

    Scores the query against the retrieved text with a TruLens provider's
    feedback function (default: ``context_relevance``) and thresholds the
    resulting 0.0-1.0 score to a boolean so it plugs into the evaluator.
    Use :meth:`score` directly when you want the raw graded score.

    Only feedback functions that take two strings and return a 0.0-1.0 score
    map onto :class:`JudgmentContext`: ``context_relevance`` and ``relevance``.
    Groundedness and answer-relevance against a generated response are not
    exposed because :class:`JudgmentContext` has no generated-answer field.

    Note: unlike the other judges, TruLens feedback functions score ``query``
    against ``retrieved_text`` directly and ignore ``expected_text``.

    Examples:
        >>> judge = TruLensJudge()  # OpenAI provider, context_relevance
        >>> judge = TruLensJudge(provider="openai", model="gpt-6-lina")
        >>> judge = TruLensJudge(provider="litellm", model="claude-3-5-sonnet-latest")
        >>> judge = TruLensJudge(metric="relevance", threshold=0.6)

    Args:
        provider: TruLens provider ("openai" or "litellm"). LiteLLM routes to
            Anthropic, Gemini, Cohere, Mistral, Ollama, and more via its model
            string (e.g. "claude-3-5-sonnet-latest", "gemini/gemini-3.5-flash-lite").
        model: Model engine name (uses the provider default if None). Required
            in practice for the litellm provider.
        metric: Feedback function to use (one of ``SUPPORTED_METRICS``,
            default "context_relevance").
        threshold: Score cutoff in [0, 1]; scores >= threshold judge as
            relevant (default: 0.5).
        provider_kwargs: Extra keyword arguments forwarded to the TruLens
            provider constructor (e.g. ``api_base`` for a local LiteLLM model).

    Requires:
        pip install trulens-providers-openai  # for provider="openai"
        pip install trulens-providers-litellm  # for provider="litellm"
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        metric: str = "context_relevance",
        threshold: float = 0.5,
        provider_kwargs: dict | None = None,
    ):
        """Initialize the TruLens-backed judge."""
        self._validate_threshold(threshold)
        self._validate_metric(metric)

        self.provider_name = provider
        self.model_name = model
        self.metric = metric
        self.threshold = threshold

        self._provider = _trulens_provider_factory(
            provider=provider,
            model=model,
            provider_kwargs=provider_kwargs or {},
        )
        self._feedback_fn = self._resolve_feedback_fn(self._provider, metric)
        logger.info(
            "Initialized TruLensJudge",
            extra={
                "judge": self.name,
                "provider": self.provider_name,
                "model": self.model_name,
                "metric": self.metric,
                "threshold": self.threshold,
            },
        )

    @property
    def name(self) -> str:
        """Judge display name."""
        model_display = self.model_name or "default"
        return f"trulens({self.provider_name}:{model_display},{self.metric},threshold={self.threshold})"

    def score(self, context: JudgmentContext) -> float:
        """Return the raw TruLens feedback score in [0, 1].

        The feedback function scores ``context.query`` against
        ``context.retrieved_text``; ``context.expected_text`` is not used.

        Args:
            context: Judgment context with query and texts

        Returns:
            Graded relevance score in [0.0, 1.0]
        """
        return float(self._feedback_fn(context.query, context.retrieved_text))

    def judge(self, context: JudgmentContext) -> bool:
        """Judge relevance by thresholding the TruLens feedback score.

        Args:
            context: Judgment context with query and texts

        Returns:
            True if the feedback score >= threshold
        """
        started_at = perf_counter()
        value = self.score(context)
        decision = value >= self.threshold
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "TruLens judgment computed",
                extra={
                    "judge": self.name,
                    "metric": self.metric,
                    "score": value,
                    "threshold": self.threshold,
                    "decision": decision,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                },
            )
        return decision

    @staticmethod
    def _resolve_feedback_fn(provider, metric: str):
        """Resolve the bound feedback-function method from the provider."""
        feedback_fn = getattr(provider, metric, None)
        if feedback_fn is None or not callable(feedback_fn):
            raise EvretValidationError(
                f"TruLens provider {type(provider).__name__} does not expose a "
                f"callable feedback function named {metric!r}"
            )
        return feedback_fn

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        """Validate threshold parameter."""
        if not 0 <= threshold <= 1:
            raise EvretValidationError("threshold must be in [0, 1]")

    @staticmethod
    def _validate_metric(metric: str) -> None:
        """Validate the requested feedback function is supported."""
        if metric not in SUPPORTED_METRICS:
            supported = ", ".join(sorted(SUPPORTED_METRICS))
            raise EvretValidationError(
                f"Unsupported TruLens metric: {metric!r}. "
                f"Supported metrics: {supported}"
            )


def _trulens_provider_factory(
    provider: str,
    model: str | None,
    provider_kwargs: dict,
):
    """Create a TruLens provider instance.

    Args:
        provider: Provider name ("openai" or "litellm")
        model: Model engine name (uses provider default if None)
        provider_kwargs: Extra keyword arguments for the provider constructor

    Returns:
        A TruLens provider exposing feedback-function methods

    Raises:
        OptionalDependencyError: If the provider package is not installed
        ValueError: If the provider name is not supported
    """
    provider_lower = provider.lower()
    kwargs = dict(provider_kwargs)
    if model is not None:
        kwargs["model_engine"] = model

    if provider_lower == "openai":
        if not HAS_TRULENS_OPENAI:
            raise OptionalDependencyError(
                "TruLensJudge with provider='openai' requires the "
                "trulens-providers-openai package. "
                "Install with: pip install trulens-providers-openai"
            )
        return _TruLensOpenAI(**kwargs)

    if provider_lower == "litellm":
        if not HAS_TRULENS_LITELLM:
            raise OptionalDependencyError(
                "TruLensJudge with provider='litellm' requires the "
                "trulens-providers-litellm package. "
                "Install with: pip install trulens-providers-litellm"
            )
        return _TruLensLiteLLM(**kwargs)

    raise ValueError(
        f"Unsupported TruLens provider: {provider}. "
        f"Supported providers: openai, litellm"
    )
