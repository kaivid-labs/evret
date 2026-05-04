import json

import pytest

from evret.evaluation import EvaluationDataset


def test_dataset_from_json_loads_queries_and_documents(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "expected_answers": ["alpha text", "beta snippet"],
                    }
                ],
                "documents": [
                    {
                        "doc_id": "doc_1",
                        "text": "alpha text",
                        "metadata": {"source": "unit"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert len(dataset.queries) == 1
    assert dataset.queries[0].query_id == "q1"
    assert dataset.queries[0].expected_answers == ["alpha text", "beta snippet"]
    assert len(dataset.documents) == 1
    assert dataset.documents[0].doc_id == "doc_1"


def test_dataset_from_csv_supports_json_and_csv_columns(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,expected_answers",
                'q1,alpha,"[""alpha text"",""beta snippet""]"',
                "q2,beta,\"gamma context,delta context\"",
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert [query.query_id for query in dataset.queries] == ["q1", "q2"]
    assert dataset.queries[0].expected_answers == ["alpha text", "beta snippet"]
    assert dataset.queries[1].expected_answers == ["gamma context", "delta context"]


def test_dataset_from_json_raises_for_invalid_query_type(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"queries": [123]}), encoding="utf-8")

    with pytest.raises(ValueError, match="query item must be an object"):
        EvaluationDataset.from_json(dataset_path)


def test_dataset_from_json_accepts_relevant_doc_ids_field(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "relevant_doc_ids": ["doc_1"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == ["doc_1"]


def test_dataset_from_json_raises_when_file_missing(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="dataset file not found"):
        EvaluationDataset.from_json(missing_path)


def test_dataset_from_json_backward_compatible_with_relevant_docs(tmp_path) -> None:
    """Test backward compatibility with old relevant_docs field name."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "relevant_docs": ["doc_1", "doc_2"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == ["doc_1", "doc_2"]
    assert dataset.queries[0].expected_answers == []


def test_dataset_from_csv_with_expected_answers_column(tmp_path) -> None:
    """Test CSV loading with expected_answers column."""
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,expected_answers",
                'q1,alpha,"[""answer one"",""answer two""]"',
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert dataset.queries[0].query_id == "q1"
    assert dataset.queries[0].expected_answers == ["answer one", "answer two"]
    assert dataset.queries[0].relevant_doc_ids == []


def test_dataset_from_csv_with_relevant_doc_ids_column(tmp_path) -> None:
    """Test CSV loading with relevant_doc_ids column."""
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,relevant_doc_ids",
                'q1,alpha,"[""doc_1"",""doc_2""]"',
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert dataset.queries[0].query_id == "q1"
    assert dataset.queries[0].relevant_doc_ids == ["doc_1", "doc_2"]
    assert dataset.queries[0].expected_answers == []


def test_dataset_from_csv_prefers_relevant_doc_ids_over_relevant_docs(tmp_path) -> None:
    """Test that relevant_doc_ids takes priority over legacy relevant_docs."""
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,relevant_doc_ids,relevant_docs",
                'q1,alpha,"[""doc_new""]","[""doc_old""]"',
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == ["doc_new"]


def test_dataset_from_csv_prefers_expected_answers_over_relevant_docs(tmp_path) -> None:
    """Test that expected_answers takes priority over legacy relevant_docs."""
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,expected_answers,relevant_docs",
                'q1,alpha,"[""answer text""]","[""doc_old""]"',
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert dataset.queries[0].expected_answers == ["answer text"]
    assert dataset.queries[0].relevant_doc_ids == []


def test_dataset_from_json_prefers_relevant_doc_ids_over_relevant_docs(tmp_path) -> None:
    """Test that relevant_doc_ids takes priority over legacy relevant_docs in JSON."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "relevant_doc_ids": ["doc_new"],
                        "relevant_docs": ["doc_old"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == ["doc_new"]


def test_dataset_from_json_prefers_expected_answers_over_relevant_docs(tmp_path) -> None:
    """Test that expected_answers takes priority over legacy relevant_docs in JSON."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "expected_answers": ["answer text"],
                        "relevant_docs": ["doc_old"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].expected_answers == ["answer text"]
    assert dataset.queries[0].relevant_doc_ids == []


def test_dataset_from_json_allows_both_fields_empty(tmp_path) -> None:
    """Test that both relevant_doc_ids and expected_answers can be empty lists."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "relevant_doc_ids": [],
                        "expected_answers": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == []
    assert dataset.queries[0].expected_answers == []


def test_dataset_from_json_accepts_both_relevant_doc_ids_and_expected_answers(tmp_path) -> None:
    """Test that both relevant_doc_ids and expected_answers can be provided together."""
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "relevant_doc_ids": ["doc_1"],
                        "expected_answers": ["answer text"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].relevant_doc_ids == ["doc_1"]
    assert dataset.queries[0].expected_answers == ["answer text"]
