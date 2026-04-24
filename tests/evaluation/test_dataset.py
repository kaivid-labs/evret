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
                        "relevant_docs": ["doc_1", "doc_2"],
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
    assert dataset.queries[0].relevant_doc_ids == ["doc_1", "doc_2"]
    assert len(dataset.documents) == 1
    assert dataset.documents[0].doc_id == "doc_1"


def test_dataset_from_csv_supports_json_and_csv_columns(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "query_id,query_text,relevant_docs",
                'q1,alpha,"[""doc_1"",""doc_2""]"',
                "q2,beta,\"doc_3,doc_4\"",
            ]
        ),
        encoding="utf-8",
    )

    dataset = EvaluationDataset.from_csv(dataset_path)

    assert [query.query_id for query in dataset.queries] == ["q1", "q2"]
    assert dataset.queries[0].relevant_doc_ids == ["doc_1", "doc_2"]
    assert dataset.queries[1].relevant_doc_ids == ["doc_3", "doc_4"]


def test_dataset_from_json_raises_for_invalid_query_type(tmp_path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"queries": [123]}), encoding="utf-8")

    with pytest.raises(ValueError, match="query item must be an object"):
        EvaluationDataset.from_json(dataset_path)


def test_dataset_from_json_raises_when_file_missing(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="dataset file not found"):
        EvaluationDataset.from_json(missing_path)
