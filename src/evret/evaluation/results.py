"""Evaluation result container and exporters."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from evret.errors import EvretValidationError
from evret.utils import ensure_parent_dir, require_non_empty_str


@dataclass(slots=True)
class EvaluationResults:
    """Aggregated metric results for an evaluation run."""

    metric_scores: dict[str, float]
    query_count: int
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.query_count < 0:
            raise EvretValidationError("query_count must be non-negative")

        normalized_scores: dict[str, float] = {}
        for metric_name, score in self.metric_scores.items():
            normalized_metric_name = require_non_empty_str(metric_name, "metric name")
            normalized_scores[normalized_metric_name] = float(score)
        self.metric_scores = normalized_scores

    def summary(self) -> dict[str, float]:
        """Return metric summary map."""
        return dict(self.metric_scores)

    def to_dict(self) -> dict[str, object]:
        """Return serializable representation of this result object."""
        return {
            "query_count": self.query_count,
            "generated_at": self.generated_at,
            "metrics": self.summary(),
        }

    def to_json(self, path: str | Path) -> None:
        """Write results as JSON."""
        output_path = ensure_parent_dir(path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def to_csv(self, path: str | Path) -> None:
        """Write metric rows as CSV (`metric`, `score`)."""
        output_path = ensure_parent_dir(path)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "score"])
            for metric_name in sorted(self.metric_scores):
                score = self.metric_scores[metric_name]
                writer.writerow([metric_name, f"{score:.12g}"])
