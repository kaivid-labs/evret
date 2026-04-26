import pytest

from evret.evaluation.judges import default_relevance_judge, make_token_overlap_judge


def test_default_relevance_judge_matches_by_substring() -> None:
    assert default_relevance_judge(
        query_text="what is rag",
        relevant_label="retrieval augmented generation",
        candidate_text="retrieval augmented generation improves grounded outputs",
    )


def test_default_relevance_judge_matches_by_token_overlap() -> None:
    assert default_relevance_judge(
        query_text="how do vector dbs work",
        relevant_label="embeddings semantic search ranking",
        candidate_text="semantic ranking with embeddings in vector search pipelines",
    )


def test_make_token_overlap_judge_honors_thresholds() -> None:
    judge = make_token_overlap_judge(min_shared_tokens=3, min_overlap_ratio=0.75)

    assert judge(
        query_text="retriever relevance",
        relevant_label="retriever relevance scoring",
        candidate_text="retriever relevance scoring with overlap",
    )
    assert not judge(
        query_text="retriever relevance",
        relevant_label="retriever relevance scoring",
        candidate_text="retriever ranking only",
    )


def test_make_token_overlap_judge_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_shared_tokens"):
        make_token_overlap_judge(min_shared_tokens=0)
    with pytest.raises(ValueError, match="min_overlap_ratio"):
        make_token_overlap_judge(min_overlap_ratio=0)
