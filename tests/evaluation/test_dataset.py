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


def test_dataset_from_csv_supports_json_and_csv_answer_columns(tmp_path) -> None:
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


def test_dataset_from_json_raises_when_file_missing(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="dataset file not found"):
        EvaluationDataset.from_json(missing_path)


def test_dataset_from_csv_with_expected_answers_column(tmp_path) -> None:
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


def test_dataset_from_json_allows_empty_expected_answers(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "expected_answers": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].expected_answers == []


def test_dataset_from_json_ignores_unknown_legacy_answer_fields(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query_id": "q1",
                        "query_text": "alpha",
                        "expected_answers": ["answer text"],
                        "legacy_answers": ["old answer"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_json(dataset_path)

    assert dataset.queries[0].expected_answers == ["answer text"]


def test_dataset_from_csv_defaults_expected_answers_to_empty_list(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text",
                "q1,alpha",
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert dataset.queries[0].expected_answers == []
