"""Internal LLM prompt templates (not exposed in public API)."""

from __future__ import annotations

RELEVANCE_PROMPT_TEMPLATE = """You are an expert evaluator for retrieval systems.

**Task**: Determine if the retrieved text is relevant to the expected content for the given query.

**Query**: {query}

**Expected Content**: {expected_text}

**Retrieved Text**: {retrieved_text}

**Instructions**:
- Respond with "YES" if the retrieved text semantically matches or covers the expected content
- Respond with "NO" if the retrieved text is irrelevant or off-topic
- Focus on semantic meaning, not exact wording
- Consider the query context when making your judgment

**Response** (YES or NO):"""
