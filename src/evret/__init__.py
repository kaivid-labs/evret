"""Evret package."""

from importlib.metadata import PackageNotFoundError, version

from evret.evaluation import (
    DocumentExample,
    EvaluationDataset,
    EvaluationResults,
    Evaluator,
    QueryExample,
)
from evret.errors import EvretError, EvretValidationError, OptionalDependencyError
from evret.integrations import LangChainRetrieverAdapter, LlamaIndexRetrieverAdapter
from evret.judges import Judge, JudgmentContext, TokenOverlapJudge
from evret.metrics import AveragePrecision, HitRate, MRR, NDCG, Precision, Recall
from evret.retrievers import (
    BaseRetriever,
    ChromaRetriever,
    MilvusRetriever,
    QdrantRetriever,
    RetrievalResult,
    WeaviateRetriever,
)

try:
    __version__ = version("evret")
except PackageNotFoundError:  # pragma: no cover - only when package metadata is unavailable
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "AveragePrecision",
    "BaseRetriever",
    "ChromaRetriever",
    "DocumentExample",
    "EvaluationDataset",
    "EvaluationResults",
    "Evaluator",
    "EvretError",
    "EvretValidationError",
    "HitRate",
    "Judge",
    "JudgmentContext",
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "MilvusRetriever",
    "MRR",
    "NDCG",
    "OptionalDependencyError",
    "Precision",
    "QueryExample",
    "QdrantRetriever",
    "Recall",
    "RetrievalResult",
    "TokenOverlapJudge",
    "WeaviateRetriever",
]
