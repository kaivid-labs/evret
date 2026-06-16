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
from evret.generation import (
    ChunkingConfig,
    DatasetGenerator,
    GeneratedChunk,
    GeneratedDataset,
    GeneratedExample,
    SourceDocument,
    build_generation_prompt,
    chunk_documents,
)
from evret.integrations import (
    HaystackRetrieverAdapter,
    LangChainRetrieverAdapter,
    LlamaIndexRetrieverAdapter,
)
from evret.judges import Judge, JudgmentContext, TokenOverlapJudge
from evret.logging import configure_logging, get_logger
from evret.metrics import AveragePrecision, ERR, HitRate, MRR, NDCG, Precision, RBP, Recall
from evret.retrievers import (
    BaseRetriever,
    ChromaRetriever,
    ElasticsearchRetriever,
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
    "ChunkingConfig",
    "DatasetGenerator",
    "DocumentExample",
    "ERR",
    "EvaluationDataset",
    "EvaluationResults",
    "Evaluator",
    "ElasticsearchRetriever",
    "EvretError",
    "EvretValidationError",
    "GeneratedChunk",
    "GeneratedDataset",
    "GeneratedExample",
    "HaystackRetrieverAdapter",
    "HitRate",
    "Judge",
    "JudgmentContext",
    "LangChainRetrieverAdapter",
    "LlamaIndexRetrieverAdapter",
    "MilvusRetriever",
    "MRR",
    "NDCG",
    "OptionalDependencyError",
    "configure_logging",
    "get_logger",
    "Precision",
    "QueryExample",
    "QdrantRetriever",
    "RBP",
    "Recall",
    "RetrievalResult",
    "SourceDocument",
    "TokenOverlapJudge",
    "WeaviateRetriever",
    "build_generation_prompt",
    "chunk_documents",
]
