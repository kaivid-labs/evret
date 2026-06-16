"""LLM-assisted evaluation dataset generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from evret.errors import EvretValidationError
from evret.evaluation import DocumentExample, EvaluationDataset, QueryExample
from evret.utils import require_non_empty_str

ANSWERABLE_CATEGORIES = frozenset(
    {
        "direct_fact",
        "paraphrase",
        "keyword_search",
        "specific_detail",
        "broad_summary",
    }
)
OUT_OF_CONTEXT_CATEGORY = "out_of_context"
GENERATION_CATEGORIES = ANSWERABLE_CATEGORIES | {OUT_OF_CONTEXT_CATEGORY}


class CompletionProvider(Protocol):
    """Minimal LLM interface used by the dataset generator."""

    def complete(self, prompt: str) -> str:
        """Return a completion for the prompt."""


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Input document for dataset generation."""

    text: str
    source: str = "document"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Settings for structure-aware text chunking."""

    target_min_tokens: int = 250
    target_max_tokens: int = 450
    max_tokens: int = 700
    overlap_tokens: int = 60
    min_tokens: int = 80

    def __post_init__(self) -> None:
        if self.target_min_tokens <= 0:
            raise EvretValidationError("target_min_tokens must be positive")
        if self.target_max_tokens < self.target_min_tokens:
            raise EvretValidationError("target_max_tokens must be >= target_min_tokens")
        if self.max_tokens < self.target_max_tokens:
            raise EvretValidationError("max_tokens must be >= target_max_tokens")
        if self.overlap_tokens < 0:
            raise EvretValidationError("overlap_tokens must be non-negative")
        if self.min_tokens <= 0:
            raise EvretValidationError("min_tokens must be positive")


@dataclass(frozen=True, slots=True)
class GeneratedChunk:
    """Chunk emitted by the dataset-generation chunker."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_document_example(self) -> DocumentExample:
        """Convert to Evret's evaluation document shape."""
        return DocumentExample(doc_id=self.doc_id, text=self.text, metadata=dict(self.metadata))


@dataclass(frozen=True, slots=True)
class GeneratedExample:
    """Rich generated query example before conversion to `EvaluationDataset`."""

    query_id: str
    query_text: str
    category: str
    expected_answer: str
    expected_context: str
    source_chunk_id: str

    def to_query_example(self) -> QueryExample:
        """Convert to Evret's evaluation query shape."""
        expected_answers = [self.expected_answer] if self.expected_answer else []
        return QueryExample(
            query_id=self.query_id,
            query_text=self.query_text,
            expected_answers=expected_answers,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the rich JSON-serializable example."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "category": self.category,
            "expected_answer": self.expected_answer,
            "expected_context": self.expected_context,
            "source_chunk_id": self.source_chunk_id,
        }


@dataclass(frozen=True, slots=True)
class GeneratedDataset:
    """Generated dataset with rich examples and Evret-compatible export."""

    examples: list[GeneratedExample]
    chunks: list[GeneratedChunk]

    def to_evaluation_dataset(self) -> EvaluationDataset:
        """Convert to Evret's existing evaluation dataset model."""
        return EvaluationDataset(
            queries=[example.to_query_example() for example in self.examples],
            documents=[chunk.to_document_example() for chunk in self.chunks],
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a rich JSON-serializable dataset."""
        return {
            "queries": [
                {
                    "query_id": example.query_id,
                    "query_text": example.query_text,
                    "expected_answers": (
                        [example.expected_answer] if example.expected_answer else []
                    ),
                    "category": example.category,
                    "expected_context": example.expected_context,
                    "source_chunk_id": example.source_chunk_id,
                }
                for example in self.examples
            ],
            "documents": [
                {
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "metadata": dict(chunk.metadata),
                }
                for chunk in self.chunks
            ],
        }


class DatasetGenerator:
    """Generate diverse retrieval-evaluation examples from documents."""

    def __init__(
        self,
        llm: CompletionProvider,
        *,
        chunking_config: ChunkingConfig | None = None,
        examples_per_chunk: int = 6,
    ) -> None:
        self.llm = llm
        self.chunking_config = chunking_config or ChunkingConfig()
        if examples_per_chunk <= 0:
            raise EvretValidationError("examples_per_chunk must be positive")
        self.examples_per_chunk = examples_per_chunk

    @classmethod
    def from_provider(
        cls,
        provider: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 3,
        chunking_config: ChunkingConfig | None = None,
        examples_per_chunk: int = 6,
    ) -> DatasetGenerator:
        """Create a generator using Evret's configured LLM providers."""
        from evret.judges.llm.factory import llm_provider_factory

        llm = llm_provider_factory(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_retries=max_retries,
        )
        return cls(
            llm,
            chunking_config=chunking_config,
            examples_per_chunk=examples_per_chunk,
        )

    def generate(self, documents: Sequence[SourceDocument | str]) -> GeneratedDataset:
        """Chunk documents and generate a rich evaluation dataset."""
        chunks = chunk_documents(documents, config=self.chunking_config)
        examples: list[GeneratedExample] = []
        seen_queries: set[str] = set()

        for chunk in chunks:
            prompt = build_generation_prompt(chunk, num_examples=self.examples_per_chunk)
            raw_examples = _parse_llm_json_array(self.llm.complete(prompt))
            for raw_example in raw_examples:
                example = _normalize_generated_example(
                    raw_example,
                    chunk=chunk,
                    query_index=len(examples) + 1,
                )
                if example is None:
                    continue
                query_key = _normalize_query_key(example.query_text)
                if query_key in seen_queries:
                    continue
                seen_queries.add(query_key)
                examples.append(example)

        return GeneratedDataset(examples=examples, chunks=chunks)


def chunk_documents(
    documents: Sequence[SourceDocument | str],
    *,
    config: ChunkingConfig | None = None,
) -> list[GeneratedChunk]:
    """Split documents into structure-aware chunks."""
    chunking_config = config or ChunkingConfig()
    chunks: list[GeneratedChunk] = []

    for document_index, document in enumerate(documents, start=1):
        source_document = _coerce_source_document(document, document_index=document_index)
        sections = _split_markdown_sections(source_document.text)
        chunk_index = 1
        for section in sections:
            section_chunks = _chunk_section(section["text"], chunking_config)
            for section_chunk in section_chunks:
                text = section_chunk.strip()
                if not text:
                    continue
                doc_id = _make_chunk_id(source_document.source, document_index, chunk_index)
                metadata = dict(source_document.metadata)
                metadata.update(
                    {
                        "doc_id": doc_id,
                        "source": source_document.source,
                        "heading_path": section["heading_path"],
                        "chunk_index": chunk_index,
                    }
                )
                chunks.append(
                    GeneratedChunk(
                        doc_id=doc_id,
                        text=text,
                        metadata=metadata,
                    )
                )
                chunk_index += 1

    return _merge_short_chunks(chunks, chunking_config)


def build_generation_prompt(chunk: GeneratedChunk, *, num_examples: int = 6) -> str:
    """Build the single diverse generation prompt for one chunk."""
    metadata = json.dumps(chunk.metadata, sort_keys=True)
    return f"""You are creating a retriever evaluation dataset.

Use the provided context to create a diverse set of retrieval evaluation examples.

Source chunk id:
{chunk.doc_id}

Source metadata:
{metadata}

Context:
{chunk.text}

Generate {num_examples} evaluation examples.

Requirements:
- Each example must include category, query_text, expected_answer, expected_context, and source_chunk_id.
- Use these categories when possible:
  - direct_fact
  - paraphrase
  - keyword_search
  - specific_detail
  - broad_summary
  - out_of_context
- For every category except out_of_context:
  - The query must be answerable from the context.
  - The expected_answer must be directly supported by the context.
  - The expected_context must be the exact supporting text or the smallest supporting snippet from the context.
- For out_of_context:
  - The query must be plausible for the same domain, but not answered by the context.
  - The expected_answer must be an empty string.
  - The expected_context must be an empty string.
- Do not use outside knowledge to answer any query.
- Avoid duplicate questions.
- Avoid yes/no questions unless the context contains a clear condition or rule.
- If the context does not contain enough information for answerable examples, only return out_of_context examples.

Return JSON only:
[
  {{
    "category": "direct_fact",
    "query_text": "...",
    "expected_answer": "...",
    "expected_context": "...",
    "source_chunk_id": "{chunk.doc_id}"
  }},
  {{
    "category": "out_of_context",
    "query_text": "...",
    "expected_answer": "",
    "expected_context": "",
    "source_chunk_id": "{chunk.doc_id}"
  }}
]"""


def _coerce_source_document(document: SourceDocument | str, *, document_index: int) -> SourceDocument:
    if isinstance(document, SourceDocument):
        text = require_non_empty_str(document.text, "document text")
        source = require_non_empty_str(document.source, "document source")
        return SourceDocument(text=text, source=source, metadata=dict(document.metadata))
    return SourceDocument(
        text=require_non_empty_str(document, "document text"),
        source=f"document_{document_index}",
    )


def _split_markdown_sections(text: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_heading_path: list[str] = []
    current_lines: list[str] = []

    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading_match:
            if current_lines:
                sections.append(
                    {
                        "heading_path": list(current_heading_path),
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            current_heading_path = current_heading_path[: level - 1] + [heading]
            continue
        current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "heading_path": list(current_heading_path),
                "text": "\n".join(current_lines).strip(),
            }
        )

    if not sections and text.strip():
        sections.append({"heading_path": [], "text": text.strip()})
    return sections


def _chunk_section(text: str, config: ChunkingConfig) -> list[str]:
    if _token_count(text) <= config.max_tokens:
        return [text]

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    chunks: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join([*current, paragraph]).strip()
        if not current or _token_count(candidate) <= config.target_max_tokens:
            current.append(paragraph)
            continue
        chunks.extend(_flush_chunk_group(current, config))
        current = [paragraph]

    if current:
        chunks.extend(_flush_chunk_group(current, config))

    return _add_overlap(chunks, config)


def _flush_chunk_group(paragraphs: list[str], config: ChunkingConfig) -> list[str]:
    text = "\n\n".join(paragraphs).strip()
    if _token_count(text) <= config.max_tokens:
        return [text]
    return _split_by_sentences(text, config)


def _split_by_sentences(text: str, config: ChunkingConfig) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if not sentences:
        return _split_by_tokens(text, config)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*current, sentence]).strip()
        if not current or _token_count(candidate) <= config.target_max_tokens:
            current.append(sentence)
            continue
        chunks.append(" ".join(current).strip())
        current = [sentence]
    if current:
        chunks.append(" ".join(current).strip())

    final_chunks: list[str] = []
    for chunk in chunks:
        if _token_count(chunk) > config.max_tokens:
            final_chunks.extend(_split_by_tokens(chunk, config))
        else:
            final_chunks.append(chunk)
    return final_chunks


def _split_by_tokens(text: str, config: ChunkingConfig) -> list[str]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(config.target_max_tokens - config.overlap_tokens, 1)
    return [
        " ".join(tokens[start : start + config.target_max_tokens])
        for start in range(0, len(tokens), step)
    ]


def _add_overlap(chunks: list[str], config: ChunkingConfig) -> list[str]:
    if config.overlap_tokens == 0 or len(chunks) <= 1:
        return chunks
    overlapped = [chunks[0]]
    for previous, current in zip(chunks, chunks[1:]):
        previous_tail = " ".join(previous.split()[-config.overlap_tokens :]).strip()
        if previous_tail and not current.startswith(previous_tail):
            overlapped.append(f"{previous_tail}\n\n{current}")
        else:
            overlapped.append(current)
    return overlapped


def _merge_short_chunks(chunks: list[GeneratedChunk], config: ChunkingConfig) -> list[GeneratedChunk]:
    merged: list[GeneratedChunk] = []
    for chunk in chunks:
        if (
            merged
            and _token_count(chunk.text) < config.min_tokens
            and merged[-1].metadata.get("source") == chunk.metadata.get("source")
            and merged[-1].metadata.get("heading_path") == chunk.metadata.get("heading_path")
            and _token_count(merged[-1].text) + _token_count(chunk.text) <= config.max_tokens
        ):
            previous = merged[-1]
            merged[-1] = GeneratedChunk(
                doc_id=previous.doc_id,
                text=f"{previous.text}\n\n{chunk.text}".strip(),
                metadata=dict(previous.metadata),
            )
            continue
        merged.append(chunk)
    return merged


def _parse_llm_json_array(response: str) -> list[Any]:
    value = response.strip()
    if not value:
        return []
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvretValidationError("LLM response must be a JSON array") from exc
    if not isinstance(parsed, list):
        raise EvretValidationError("LLM response must be a JSON array")
    return parsed


def _normalize_generated_example(
    raw_example: Any,
    *,
    chunk: GeneratedChunk,
    query_index: int,
) -> GeneratedExample | None:
    if not isinstance(raw_example, dict):
        return None

    category = str(raw_example.get("category") or "").strip()
    if category not in GENERATION_CATEGORIES:
        return None

    query_text = str(raw_example.get("query_text") or "").strip()
    if not query_text:
        return None

    expected_answer = str(raw_example.get("expected_answer") or "").strip()
    expected_context = str(raw_example.get("expected_context") or "").strip()

    if category == OUT_OF_CONTEXT_CATEGORY:
        if expected_answer or expected_context:
            return None
    else:
        if not expected_answer or not expected_context:
            return None
        if not _contains_normalized(chunk.text, expected_context):
            return None

    return GeneratedExample(
        query_id=f"q{query_index}",
        query_text=query_text,
        category=category,
        expected_answer=expected_answer,
        expected_context=expected_context,
        source_chunk_id=chunk.doc_id,
    )


def _contains_normalized(haystack: str, needle: str) -> bool:
    return _normalize_space(needle).lower() in _normalize_space(haystack).lower()


def _normalize_query_key(query_text: str) -> str:
    return _normalize_space(query_text).lower()


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def _token_count(text: str) -> int:
    return len(text.split())


def _make_chunk_id(source: str, document_index: int, chunk_index: int) -> str:
    source_slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    if not source_slug:
        source_slug = f"document_{document_index}"
    return f"{source_slug}_{chunk_index}"
