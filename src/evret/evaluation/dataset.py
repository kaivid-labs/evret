"""Evaluation dataset models and loaders."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evret.errors import EvretValidationError
from evret.utils import (
    normalize_unique_non_empty_strings,
    require_file_exists,
    require_non_empty_str,
)


@dataclass(frozen=True, slots=True)
class QueryExample:
    """One query item in an evaluation dataset."""

    query_id: str
    query_text: str
    relevant_doc_ids: list[str]


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
        return cls(queries=queries, documents=documents)

    @classmethod
    def from_csv(cls, path: str | Path) -> EvaluationDataset:
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
                relevant_doc_ids = cls._parse_relevant_docs(row.get("relevant_docs", ""))
                query_items.append(
                    QueryExample(
                        query_id=query_id,
                        query_text=query_text,
                        relevant_doc_ids=relevant_doc_ids,
                    )
                )

        return cls(queries=query_items, documents=[])

    @staticmethod
    def _parse_query_item(item: Any) -> QueryExample:
        if not isinstance(item, dict):
            raise EvretValidationError("each query item must be an object")

        query_id_raw = item.get("query_id") or item.get("id")
        query_text_raw = item.get("query_text") or item.get("query")
        relevant_docs_raw = item.get("relevant_docs") or item.get("relevant_doc_ids") or []

        query_id = require_non_empty_str(query_id_raw or "", "query_id")
        query_text = require_non_empty_str(query_text_raw or "", "query_text")
        if not isinstance(relevant_docs_raw, list):
            raise EvretValidationError("relevant_docs must be a list")

        relevant_doc_ids = normalize_unique_non_empty_strings(relevant_docs_raw)
        return QueryExample(
            query_id=query_id,
            query_text=query_text,
            relevant_doc_ids=relevant_doc_ids,
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
    def _parse_relevant_docs(raw_value: str) -> list[str]:
        value = raw_value.strip()
        if not value:
            return []

        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            loaded = [part.strip() for part in value.split(",") if part.strip()]

        if isinstance(loaded, list):
            return normalize_unique_non_empty_strings(loaded)
        raise EvretValidationError("relevant_docs must be a JSON list or comma-separated string")
