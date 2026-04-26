"""Evaluation orchestrator for retriever metrics."""

from __future__ import annotations

from collections.abc import Sequence

from evret.errors import EvretValidationError
from evret.evaluation.dataset import EvaluationDataset
from evret.evaluation.results import EvaluationResults
from evret.judges.base import Judge, JudgmentContext
from evret.judges.token_overlap import TokenOverlapJudge
from evret.metrics import Metric
from evret.retrievers import BaseRetriever, RetrievalResult
from evret.utils import find_duplicates


class Evaluator:
    """Run a list of metrics over a retriever and dataset.

    Uses pluggable Judge system for text-based relevance matching.

    Args:
        retriever: Retriever to evaluate
        metrics: List of metrics to compute
        judge: Relevance judge (defaults to TokenOverlapJudge if None)

    Examples:
        >>> from evret import Evaluator, HitRate, Recall
        >>> from evret.judges import TokenOverlapJudge, SemanticJudge, LLMJudge
        >>>
        >>> # Default: TokenOverlapJudge
        >>> evaluator = Evaluator(retriever, [HitRate(k=4), Recall(k=4)])
        >>>
        >>> # Custom judge
        >>> evaluator = Evaluator(
        ...     retriever,
        ...     [Recall(k=4)],
        ...     judge=SemanticJudge(threshold=0.8)
        ... )
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        metrics: Sequence[Metric],
        judge: Judge | None = None,
    ) -> None:
        if not metrics:
            raise EvretValidationError("metrics must contain at least one metric")

        metric_names = [metric.name for metric in metrics]
        duplicate_metric_names = sorted(find_duplicates(metric_names))
        if duplicate_metric_names:
            duplicates_text = ", ".join(duplicate_metric_names)
            raise EvretValidationError(
                f"duplicate metric names are not allowed: {duplicates_text}"
            )

        self.retriever = retriever
        self.metrics = list(metrics)
        self.judge = judge or TokenOverlapJudge()

    def evaluate(self, dataset: EvaluationDataset) -> EvaluationResults:
        if not dataset.queries:
            raise EvretValidationError("dataset must contain at least one query")

        max_k = max(metric.k for metric in self.metrics)
        retrieved_results = self.retriever.batch_retrieve(
            queries=[query.query_text for query in dataset.queries],
            k=max_k,
        )
        if len(retrieved_results) != len(dataset.queries):
            raise EvretValidationError(
                "retriever returned a different number of query results than input queries"
            )
        retrieved_ids, relevant_sets = self._build_metric_inputs(
            dataset=dataset,
            retrieved_results=retrieved_results,
        )

        metric_scores: dict[str, float] = {}
        for metric in self.metrics:
            metric_scores[metric.name] = metric.score(
                retrieved_by_query=retrieved_ids,
                relevant_by_query=relevant_sets,
            )

        return EvaluationResults(
            metric_scores=metric_scores,
            query_count=len(dataset.queries),
        )

    def _build_metric_inputs(
        self,
        dataset: EvaluationDataset,
        retrieved_results: Sequence[Sequence[RetrievalResult]],
    ) -> tuple[list[list[str]], list[set[str]]]:
        """Build metric inputs using Judge system for text-based matching."""
        document_text_by_id = {
            self._normalize_label(document.doc_id): self._normalize_label(document.text)
            for document in dataset.documents
            if document.text.strip()
        }

        retrieved_by_query: list[list[str]] = []
        relevant_by_query: list[set[str]] = []

        for query, query_results in zip(dataset.queries, retrieved_results):
            normalized_query_text = self._normalize_label(query.query_text)
            relevant_labels = [
                self._normalize_label(label)
                for label in query.relevant_docs
                if str(label).strip()
            ]
            unique_relevant_labels = list(dict.fromkeys(relevant_labels))
            relevant_set = set(unique_relevant_labels)

            # Build judgment contexts for batch evaluation
            contexts_per_result = []
            candidates_per_result = []

            for rank, result in enumerate(query_results):
                candidates = self._candidate_labels(result, document_text_by_id)
                candidates_per_result.append(candidates)

                # Create judgment contexts for all relevant_label x candidate pairs
                result_contexts = []
                for label in unique_relevant_labels:
                    for candidate in candidates:
                        result_contexts.append(
                            JudgmentContext(
                                query=normalized_query_text,
                                expected_text=label,
                                retrieved_text=candidate,
                            )
                        )
                contexts_per_result.append(result_contexts)

            # Batch judge all contexts at once for efficiency
            all_contexts = [ctx for contexts in contexts_per_result for ctx in contexts]
            if all_contexts:
                all_judgments = self.judge.batch_judge(all_contexts)
            else:
                all_judgments = []

            # Parse judgments back into structure
            judgment_idx = 0
            available_labels = set(unique_relevant_labels)
            normalized_retrieved: list[str] = []

            for rank, candidates in enumerate(candidates_per_result):
                num_contexts = len(unique_relevant_labels) * len(candidates)
                result_judgments = all_judgments[judgment_idx:judgment_idx + num_contexts]
                judgment_idx += num_contexts

                # Find first matching label
                matched_label = self._find_match(
                    judgments=result_judgments,
                    ordered_relevant_labels=unique_relevant_labels,
                    candidates=candidates,
                    remaining_relevant_labels=available_labels,
                )

                if matched_label is not None:
                    normalized_retrieved.append(matched_label)
                    available_labels.discard(matched_label)
                else:
                    fallback_label = candidates[0] if candidates else f"retrieved_{rank}"
                    normalized_retrieved.append(f"retrieved_{rank}:{fallback_label}")

            retrieved_by_query.append(normalized_retrieved)
            relevant_by_query.append(relevant_set)

        return retrieved_by_query, relevant_by_query

    def _candidate_labels(
        self,
        result: RetrievalResult,
        document_text_by_id: dict[str, str],
    ) -> list[str]:
        """Extract candidate text labels from retrieval result."""
        candidate_labels: list[str] = [self._normalize_label(result.doc_id)]
        doc_text = document_text_by_id.get(candidate_labels[0], "")
        if doc_text:
            candidate_labels.append(doc_text)

        text_keys = (
            "text",
            "document",
            "content",
            "page_content",
            "chunk",
            "passage",
        )
        for key in text_keys:
            value = result.metadata.get(key)
            if value is None:
                continue
            normalized_value = self._normalize_label(value)
            if normalized_value:
                candidate_labels.append(normalized_value)

        return list(dict.fromkeys(candidate_labels))

    def _find_match(
        self,
        judgments: list[bool],
        ordered_relevant_labels: Sequence[str],
        candidates: list[str],
        remaining_relevant_labels: set[str],
    ) -> str | None:
        """Find first matching label from batch judgments."""
        idx = 0
        for label in ordered_relevant_labels:
            if label not in remaining_relevant_labels:
                idx += len(candidates)
                continue

            for _ in candidates:
                if judgments[idx]:
                    return label
                idx += 1

        return None

    @staticmethod
    def _normalize_label(value: object) -> str:
        return " ".join(str(value).strip().lower().split())
