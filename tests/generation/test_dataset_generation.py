import json
from uuid import UUID

import pytest

import evret.generation.dataset as dataset_module
from evret import (
    ChunkingConfig,
    DatasetGenerator,
    SourceDocument,
    build_generation_prompt,
    chunk_documents,
)
from evret.errors import EvretValidationError


class FakeLLM:
    def __init__(self, response: object) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(self.response)


def test_chunk_documents_preserves_heading_metadata_and_source() -> None:
    document = SourceDocument(
        source="policy.md",
        text="# Travel\n\n" + "Flights above 500 dollars require manager approval. " * 20,
        metadata={"team": "finance"},
    )

    chunks = chunk_documents(
        [document],
        config=ChunkingConfig(
            target_min_tokens=20,
            target_max_tokens=40,
            max_tokens=60,
            overlap_tokens=5,
            min_tokens=5,
        ),
    )

    assert chunks
    assert UUID(chunks[0].doc_id)
    assert chunks[0].metadata["source"] == "policy.md"
    assert chunks[0].metadata["heading_path"] == ["Travel"]
    assert chunks[0].metadata["team"] == "finance"


def test_chunk_documents_uses_unique_ids_for_repeated_sources() -> None:
    documents = [
        SourceDocument(source="policy.md", text="Alpha policy text."),
        SourceDocument(source="policy.md", text="Beta policy text."),
    ]

    chunks = chunk_documents(
        documents,
        config=ChunkingConfig(
            target_min_tokens=5,
            target_max_tokens=20,
            max_tokens=40,
            overlap_tokens=0,
            min_tokens=1,
        ),
    )

    assert len({chunk.doc_id for chunk in chunks}) == 2
    assert all(UUID(chunk.doc_id) for chunk in chunks)


def test_chunk_documents_uses_stable_ids_without_leaking_path_sources() -> None:
    config = ChunkingConfig(
        target_min_tokens=5,
        target_max_tokens=20,
        max_tokens=40,
        overlap_tokens=0,
        min_tokens=1,
    )
    path_source = "/Users/tarunjain/Documents/Work/kaivid/evret/examples/react_agent_paper.pdf"
    chunks = chunk_documents(
        [SourceDocument(source=path_source, text="ReAct combines reasoning and acting.")],
        config=config,
    )
    repeated_chunks = chunk_documents(
        [SourceDocument(source=path_source, text="ReAct combines reasoning and acting.")],
        config=config,
    )

    assert UUID(chunks[0].doc_id)
    assert chunks[0].doc_id == repeated_chunks[0].doc_id
    assert "users_tarunjain" not in chunks[0].doc_id
    assert "react_agent_paper" not in chunks[0].doc_id


def test_build_generation_prompt_includes_categories_and_negative_rules() -> None:
    chunk = chunk_documents(["Flights above 500 dollars require manager approval."])[0]

    prompt = build_generation_prompt(chunk, num_examples=6)

    assert "direct_fact" in prompt
    assert "out_of_context" in prompt
    assert "The expected_answer must be an empty string" in prompt
    assert chunk.doc_id in prompt
    assert "expected_context" not in prompt
    assert "source_chunk_id" not in prompt


def test_dataset_generator_derives_expected_context_from_chunk() -> None:
    chunk_text = "Flights above 500 dollars require manager approval."
    llm = FakeLLM(
        [
            {
                "category": "specific_detail",
                "query_text": "When does a flight need manager approval?",
                "expected_answer": "Flights above 500 dollars require manager approval.",
                "expected_context": "LLM supplied context must be ignored.",
            },
            {
                "category": "out_of_context",
                "query_text": "What is the office pet policy?",
                "expected_answer": "",
                "expected_context": "LLM supplied context must be ignored.",
            },
        ]
    )
    generator = DatasetGenerator(llm, examples_per_chunk=2)

    generated = generator.generate(
        [SourceDocument(text=chunk_text, source="travel")]
    )
    dataset = generated.to_evaluation_dataset()

    assert [example.category for example in generated.examples] == [
        "specific_detail",
        "out_of_context",
    ]
    assert dataset.queries[0].expected_answers == [
        "Flights above 500 dollars require manager approval."
    ]
    assert dataset.queries[1].expected_answers == []
    assert generated.examples[0].expected_doc_ids == [generated.chunks[0].doc_id]
    assert generated.examples[1].expected_doc_ids == []
    assert generated.examples[0].expected_context == chunk_text
    assert generated.to_dict()["queries"][1]["expected_context"] == ""
    assert generated.to_dict()["queries"][1]["expected_doc_ids"] == []


def test_dataset_generator_filters_invalid_answers_and_invalid_negative_examples() -> None:
    llm = FakeLLM(
        [
            {
                "category": "direct_fact",
                "query_text": "What approval is required?",
                "expected_answer": "Manager approval is required.",
            },
            {
                "category": "out_of_context",
                "query_text": "What is the office pet policy?",
                "expected_answer": "Pets are allowed.",
            },
            {
                "category": "keyword_search",
                "query_text": "flight approval threshold",
                "expected_answer": "Flights above 500 dollars require manager approval.",
            },
        ]
    )

    generated = DatasetGenerator(llm).generate(
        [SourceDocument(text="Flights above 500 dollars require manager approval.", source="travel")]
    )

    assert len(generated.examples) == 1
    assert generated.examples[0].category == "keyword_search"


def test_dataset_generator_raises_for_non_json_array_response() -> None:
    llm = FakeLLM({"query_text": "not an array"})

    with pytest.raises(EvretValidationError, match="JSON array"):
        DatasetGenerator(llm).generate(["Flights above 500 dollars require manager approval."])


def test_dataset_generator_wraps_chunks_with_progress(monkeypatch) -> None:
    chunks = [
        dataset_module.GeneratedChunk(doc_id="doc_1", text="alpha"),
        dataset_module.GeneratedChunk(doc_id="doc_2", text="beta"),
    ]
    tqdm_calls = []

    def fake_chunk_documents(documents, *, config=None):
        return chunks

    def fake_tqdm(iterable, **kwargs):
        tqdm_calls.append(kwargs)
        return iterable

    monkeypatch.setattr(dataset_module, "chunk_documents", fake_chunk_documents)
    monkeypatch.setattr(dataset_module, "tqdm", fake_tqdm)

    DatasetGenerator(FakeLLM([]), show_progress=False).generate(["ignored"])

    assert tqdm_calls == [
        {
            "desc": "Generating dataset",
            "unit": "chunk",
            "disable": True,
        }
    ]
