"""Semantic similarity judge using sentence embeddings."""

from __future__ import annotations

import logging
from time import perf_counter

from evret.errors import EvretValidationError, OptionalDependencyError
from evret.judges.base import Judge, JudgmentContext
from evret.logging import get_logger

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    np = None
    HAS_SENTENCE_TRANSFORMERS = False

logger = get_logger(__name__)


class SemanticJudge(Judge):
    """Embedding-based semantic similarity matching.

    Uses sentence-transformers to compute dense embeddings and cosine
    similarity for relevance judgment. More accurate than token overlap
    but requires additional dependencies and computation.

    Examples:
        >>> judge = SemanticJudge()  # Default model
        >>> judge = SemanticJudge(model="all-MiniLM-L6-v2", threshold=0.8)
        >>> judge = SemanticJudge(threshold=0.7, device="cuda")

    Args:
        model: HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)
        threshold: Cosine similarity threshold 0-1 (default: 0.75)
        device: Device for computation: "cpu" or "cuda" (default: "cpu")

    Requires:
        pip install sentence-transformers
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        device: str = "cpu",
    ):
        """Initialize semantic judge with embedding model."""
        self._check_dependencies()
        self._validate_threshold(threshold)

        self.model_name = model
        self.threshold = threshold
        self.device = device
        model_load_start = perf_counter()
        self._model = SentenceTransformer(model, device=device)
        logger.info(
            "Initialized SemanticJudge model",
            extra={
                "judge": self.name,
                "model": self.model_name,
                "device": self.device,
                "elapsed_ms": round((perf_counter() - model_load_start) * 1000, 2),
            },
        )

    @property
    def name(self) -> str:
        """Judge display name."""
        model_short = self.model_name.split("/")[-1]
        return f"semantic({model_short},threshold={self.threshold})"

    def judge(self, context: JudgmentContext) -> bool:
        """Judge using embedding cosine similarity.

        Args:
            context: Judgment context with query and texts

        Returns:
            True if cosine similarity >= threshold
        """
        similarity = self._compute_similarity(
            context.expected_text,
            context.retrieved_text
        )
        decision = similarity >= self.threshold
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Semantic judgment computed",
                extra={
                    "judge": self.name,
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "decision": decision,
                },
            )
        return decision

    def batch_judge(self, contexts: list[JudgmentContext]) -> list[bool]:
        """Optimized batch evaluation using vectorized embeddings.

        Args:
            contexts: List of judgment contexts

        Returns:
            List of boolean judgments
        """
        if not contexts:
            return []

        started_at = perf_counter()
        expected_texts = [ctx.expected_text for ctx in contexts]
        retrieved_texts = [ctx.retrieved_text for ctx in contexts]

        expected_embs = self._model.encode(expected_texts, convert_to_numpy=True)
        retrieved_embs = self._model.encode(retrieved_texts, convert_to_numpy=True)

        similarities = self._batch_cosine_similarity(expected_embs, retrieved_embs)
        results = [float(sim) >= self.threshold for sim in similarities]
        logger.debug(
            "Semantic batch judgment computed",
            extra={
                "judge": self.name,
                "batch_size": len(contexts),
                "positives": sum(1 for value in results if value),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return results

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self._model.encode([text1], convert_to_numpy=True)[0]
        emb2 = self._model.encode([text2], convert_to_numpy=True)[0]
        return float(
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )

    @staticmethod
    def _batch_cosine_similarity(embs1, embs2):
        """Vectorized cosine similarity computation."""
        dot_products = np.sum(embs1 * embs2, axis=1)
        norms1 = np.linalg.norm(embs1, axis=1)
        norms2 = np.linalg.norm(embs2, axis=1)
        return dot_products / (norms1 * norms2)

    @staticmethod
    def _check_dependencies() -> None:
        """Ensure sentence-transformers is installed."""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise OptionalDependencyError(
                "SemanticJudge requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

    @staticmethod
    def _validate_threshold(threshold: float) -> None:
        """Validate threshold parameter."""
        if not 0 <= threshold <= 1:
            raise EvretValidationError("threshold must be in [0, 1]")
