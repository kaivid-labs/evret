import csv
import json

from evret.evaluation import EvaluationResults


def test_results_to_json_and_to_dict(tmp_path) -> None:
    results = EvaluationResults(metric_scores={"hit_rate@5": 0.8}, query_count=10)
    output_path = tmp_path / "results.json"

    results.to_json(output_path)
    parsed = json.loads(output_path.read_text(encoding="utf-8"))

    assert parsed["query_count"] == 10
    assert parsed["metrics"] == {"hit_rate@5": 0.8}
    assert "generated_at" in parsed
    assert results.summary() == {"hit_rate@5": 0.8}


def test_results_to_csv_writes_metric_rows(tmp_path) -> None:
    results = EvaluationResults(
        metric_scores={"hit_rate@5": 0.8, "mrr@10": 0.65},
        query_count=5,
    )
    output_path = tmp_path / "results.csv"

    results.to_csv(output_path)

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["metric", "score"]
    assert rows[1] == ["hit_rate@5", "0.8"]
    assert rows[2] == ["mrr@10", "0.65"]


def test_results_to_json_creates_parent_directories(tmp_path) -> None:
    results = EvaluationResults(metric_scores={"hit_rate@5": 0.8}, query_count=1)
    output_path = tmp_path / "nested" / "results.json"

    results.to_json(output_path)

    assert output_path.exists()
