# Qdrant Streamlit Evaluation Dashboard

This is a standalone example for indexing a Qdrant collection and then
evaluating it from a Streamlit UI with Evret metrics and `LLMJudge`.

It is separate from `examples/demo-qdrant`. The dashboard only follows that
example's Qdrant retriever syntax.

## Files

- `index.py`: indexes the ReAct paper into Qdrant.
- `run_evals_ui.py`: runs the Streamlit evaluation dashboard.

The UI does not index documents. Run `index.py` separately when you need to
create or refresh the collection.

## Install

For OpenAI as the LLM judge provider:

```bash
pip install streamlit pandas fastembed pypdfium2 "evret[qdrant,llm-openai]"
export OPENAI_API_KEY="..."
```

For other providers:

```bash
pip install streamlit pandas fastembed pypdfium2 "evret[qdrant,llm-anthropic]"
export ANTHROPIC_API_KEY="..."

pip install streamlit pandas fastembed pypdfium2 "evret[qdrant,llm-google]"
export GOOGLE_API_KEY="..."
```

## 1. Index Documents

Configure Qdrant with environment variables:

```bash
export QDRANT_URL="http://localhost:6333"
export QDRANT_API_KEY=""  # optional for local Qdrant
export QDRANT_COLLECTION="react_paper"
```

Then index the PDF:

```bash
python examples/qdrant-streamlit-dashboard/index.py
```

`index.py` uses `examples/react_agent_paper.pdf`, embeds chunks with
`BAAI/bge-small-en-v1.5`, and stores each chunk in the Qdrant payload as `text`.

## 2. Run Evals UI

```bash
streamlit run examples/qdrant-streamlit-dashboard/run_evals_ui.py
```

The dashboard displays aggregate Hit Rate, Precision, Recall, MRR, and nDCG
scores, keeps the evaluation dataset inside a collapsed expander, and shows
retrieved Qdrant evidence per query.
