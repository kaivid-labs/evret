from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DATASET_PATH = HERE / "generated_eval_dataset.json"
RESULTS_PATH = HERE / "benchmark_results.json"

COLLECTION_NAME = "evret_search_benchmark"

DENSE = "dense"
BM25 = "bm25"
SPLADE = "splade"
COLBERT = "colbert"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
BM25_MODEL = "Qdrant/bm25"
SPLADE_MODEL = "prithivida/Splade_PP_en_v1"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

MAX_WORDS_PER_DOC = 4000
PREFETCH_LIMIT = 50

SUPPORTED_SUFFIXES = {".htm", ".html", ".pdf", ".txt"}


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise SystemExit("Set QDRANT_URL and QDRANT_API_KEY (env or .env) for Qdrant Cloud.")
    return QdrantClient(url=url, api_key=api_key, timeout=120)


class _HTMLTextExtractor(HTMLParser):
    _BLOCK = {"p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "h4", "table"}
    _SKIP = {"script", "style", "head", "title", "ix:header"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.parts.append(text + " ")

    def text(self) -> str:
        joined = re.sub(r"[ \t]+", " ", "".join(self.parts))
        return re.sub(r"\n\s*\n+", "\n\n", joined).strip()


def _html_to_text(raw: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    return parser.text()


def _pdf_to_text(path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    return "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))


def _read_document(path: Path) -> str:
    if path.suffix.lower() in {".htm", ".html"}:
        return _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    if path.suffix.lower() == ".pdf":
        return _pdf_to_text(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def load_documents(max_words_per_doc: int = MAX_WORDS_PER_DOC) -> list[tuple[str, str]]:
    paths = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not paths:
        raise SystemExit(f"No documents found in {DATA_DIR}. Add .htm/.pdf/.txt files.")
    documents = []
    for path in paths:
        text = _cap_words(_read_document(path), max_words_per_doc)
        if text.strip():
            documents.append((path.name, text))
    return documents


@lru_cache(maxsize=1)
def _dense():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=DENSE_MODEL)


@lru_cache(maxsize=1)
def _bm25():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=BM25_MODEL)


@lru_cache(maxsize=1)
def _splade():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=SPLADE_MODEL)


@lru_cache(maxsize=1)
def _colbert():
    from fastembed import LateInteractionTextEmbedding

    return LateInteractionTextEmbedding(model_name=COLBERT_MODEL)


def embed_dense_docs(texts: list[str]) -> list[list[float]]:
    return [vector.tolist() for vector in _dense().embed(texts)]


def embed_dense_query(text: str) -> list[float]:
    return next(iter(_dense().query_embed(text))).tolist()


def _to_sparse(sparse: Any) -> models.SparseVector:
    return models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist())


def embed_bm25_docs(texts: list[str]) -> list[models.SparseVector]:
    return [_to_sparse(sparse) for sparse in _bm25().embed(texts)]


def embed_bm25_query(text: str) -> models.SparseVector:
    return _to_sparse(next(iter(_bm25().query_embed(text))))


def embed_splade_docs(texts: list[str]) -> list[models.SparseVector]:
    return [_to_sparse(sparse) for sparse in _splade().embed(texts)]


def embed_splade_query(text: str) -> models.SparseVector:
    return _to_sparse(next(iter(_splade().query_embed(text))))


def embed_colbert_docs(texts: list[str]) -> list[list[list[float]]]:
    return [vectors.tolist() for vectors in _colbert().embed(texts)]


def embed_colbert_query(text: str) -> list[list[float]]:
    return next(iter(_colbert().query_embed(text))).tolist()


def load_raw_dataset() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"{DATASET_PATH.name} not found. Run `python generate_dataset.py` first.")
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def category_by_query_id(raw: dict[str, Any]) -> dict[str, str]:
    return {q["query_id"]: q.get("category", "unknown") for q in raw["queries"]}


def answerable_category_counts(raw: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for query in raw["queries"]:
        if query.get("expected_doc_ids"):
            counts[query.get("category", "unknown")] += 1
    return dict(counts)
