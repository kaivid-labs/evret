import json

import pytest

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
    assert chunks[0].doc_id == "policy_md_1"
    assert chunks[0].metadata["source"] == "policy.md"
    assert chunks[0].metadata["heading_path"] == ["Travel"]
    assert chunks[0].metadata["team"] == "finance"


def test_build_generation_prompt_includes_categories_and_negative_rules() -> None:
    chunk = chunk_documents(["Flights above 500 dollars require manager approval."])[0]

    prompt = build_generation_prompt(chunk, num_examples=6)

    assert "direct_fact" in prompt
    assert "out_of_context" in prompt
    assert "expected_context" in prompt
    assert "The expected_answer must be an empty string" in prompt
    assert chunk.doc_id in prompt


def test_dataset_generator_keeps_answerable_and_out_of_context_examples() -> None:
    llm = FakeLLM(
        [
            {
                "category": "specific_detail",
                "query_text": "When does a flight need manager approval?",
                "expected_answer": "Flights above 500 dollars require manager approval.",
                "expected_context": "Flights above 500 dollars require manager approval.",
                "source_chunk_id": "ignored",
            },
            {
                "category": "out_of_context",
                "query_text": "What is the office pet policy?",
                "expected_answer": "",
                "expected_context": "",
                "source_chunk_id": "ignored",
            },
        ]
    )
    generator = DatasetGenerator(llm, examples_per_chunk=2)

    generated = generator.generate(
        [SourceDocument(text="Flights above 500 dollars require manager approval.", source="travel")]
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
    assert generated.to_dict()["queries"][1]["expected_context"] == ""


def test_dataset_generator_filters_invalid_context_and_invalid_negative_examples() -> None:
    llm = FakeLLM(
        [
            {
                "category": "direct_fact",
                "query_text": "What approval is required?",
                "expected_answer": "Manager approval is required.",
                "expected_context": "This text is not in the source chunk.",
                "source_chunk_id": "travel_1",
            },
            {
                "category": "out_of_context",
                "query_text": "What is the office pet policy?",
                "expected_answer": "Pets are allowed.",
                "expected_context": "",
                "source_chunk_id": "travel_1",
            },
            {
                "category": "keyword_search",
                "query_text": "flight approval threshold",
                "expected_answer": "Flights above 500 dollars require manager approval.",
                "expected_context": "Flights above 500 dollars require manager approval.",
                "source_chunk_id": "travel_1",
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
