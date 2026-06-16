"""Evaluation dataset models and loaders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from evret.errors import EvretValidationError
from evret.logging import get_logger
from evret.utils import (
    normalize_unique_non_empty_strings,
    require_file_exists,
    require_non_empty_str,
)

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QueryExample:
    """One query item in an evaluation dataset.

    Provide expected_doc_ids when gold relevant document IDs are known.
    Provide expected_answers as gold answer text snippets for the judge to match
    against retrieved content when document IDs are not available.
    """

    query_id: str
    query_text: str
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_answers: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class DocumentExample:
    """One document entry in an evaluation dataset."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationDataset:
    """Evaluation dataset containing query examples and optional documents."""

    queries: list[QueryExample]
    documents: list[DocumentExample] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> EvaluationDataset:
        started_at = perf_counter()
        dataset_path = require_file_exists(path, "dataset")
        raw_data = json.loads(dataset_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise EvretValidationError("root JSON value must be an object")

        raw_queries = raw_data.get("queries", [])
        raw_documents = raw_data.get("documents", [])
        if not isinstance(raw_queries, list):
            raise EvretValidationError("queries must be a list")
        if not isinstance(raw_documents, list):
            raise EvretValidationError("documents must be a list")

        queries = [cls._parse_query_item(item) for item in raw_queries]
        documents = [cls._parse_document_item(item) for item in raw_documents]
        logger.info(
            "Loaded evaluation dataset",
            extra={
                "path": str(dataset_path),
                "format": "json",
                "queries": len(queries),
                "documents": len(documents),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return cls(queries=queries, documents=documents)

    @classmethod
    def from_csv(cls, path: str | Path) -> EvaluationDataset:
        started_at = perf_counter()
        dataset_path = require_file_exists(path, "dataset")
        query_items: list[QueryExample] = []
        with dataset_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise EvretValidationError("CSV must include a header row")
            for row_index, row in enumerate(reader, start=1):
                query_text = (row.get("query_text") or row.get("query") or "").strip()
                if not query_text:
                    raise EvretValidationError(f"query_text is required in row {row_index}")

                query_id = (
                    row.get("query_id")
                    or row.get("id")
                    or f"q{row_index}"
                )
                query_id = require_non_empty_str(query_id, "query_id")

                expected_answers = cls._parse_answer_list(row.get("expected_answers", ""))
                expected_doc_ids = cls._parse_answer_list(row.get("expected_doc_ids", ""))

                query_items.append(
                    QueryExample(
                        query_id=query_id,
                        query_text=query_text,
                        expected_doc_ids=expected_doc_ids,
                        expected_answers=expected_answers,
                    )
                )

        logger.info(
            "Loaded evaluation dataset",
            extra={
                "path": str(dataset_path),
                "format": "csv",
                "queries": len(query_items),
                "documents": 0,
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            },
        )
        return cls(queries=query_items, documents=[])

    @staticmethod
    def _parse_query_item(item: Any) -> QueryExample:
        if not isinstance(item, dict):
            raise EvretValidationError("each query item must be an object")

        query_id_raw = item.get("query_id") or item.get("id")
        query_text_raw = item.get("query_text") or item.get("query")

        query_id = require_non_empty_str(query_id_raw or "", "query_id")
        query_text = require_non_empty_str(query_text_raw or "", "query_text")

        expected_answers_raw = item.get("expected_answers") or []
        if not isinstance(expected_answers_raw, list):
            raise EvretValidationError("expected_answers must be a list")
        expected_answers = normalize_unique_non_empty_strings(expected_answers_raw)

        expected_doc_ids_raw = item.get("expected_doc_ids") or item.get("relevant_doc_ids") or []
        if not isinstance(expected_doc_ids_raw, list):
            raise EvretValidationError("expected_doc_ids must be a list")
        expected_doc_ids = normalize_unique_non_empty_strings(expected_doc_ids_raw)

        return QueryExample(
            query_id=query_id,
            query_text=query_text,
            expected_doc_ids=expected_doc_ids,
            expected_answers=expected_answers,
        )

    @staticmethod
    def _parse_document_item(item: Any) -> DocumentExample:
        if not isinstance(item, dict):
            raise EvretValidationError("each document item must be an object")

        doc_id = require_non_empty_str(item.get("doc_id") or "", "doc_id")
        text = str(item.get("text") or "")
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise EvretValidationError("document metadata must be an object")

        return DocumentExample(doc_id=doc_id, text=text, metadata=dict(metadata))

    @staticmethod
    def _parse_answer_list(raw_value: str) -> list[str]:
        value = raw_value.strip()
        if not value:
            return []

        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            loaded = [part.strip() for part in value.split(",") if part.strip()]
            logger.debug(
                "Parsed expected answers from comma-separated CSV field",
                extra={"answers": len(loaded)},
            )

        if isinstance(loaded, list):
            return normalize_unique_non_empty_strings(loaded)
        raise EvretValidationError("expected_answers must be a JSON list or comma-separated string")
