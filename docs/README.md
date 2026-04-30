# Docs Development

This folder contains the MkDocs source for the Evret documentation site.

## Install Docs Dependencies

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[docs]"
```

If your environment is already created, only run:

```bash
uv pip install -e ".[docs]"
```

## Run Locally

From the repository root:

```bash
uv run mkdocs serve
```

Open:

```text
http://127.0.0.1:8000
```

## Build And Check

Run the strict build before publishing docs changes:

```bash
uv run mkdocs build --strict
```

This checks navigation, internal links, API reference pages, and Markdown syntax that MkDocs can validate.

## Docs Layout

- `index.md`: documentation home page
- `getting-started.md`: install and first run
- `quickstart.md`: end-to-end evaluation guide
- `judges.md`: judge configuration and parameter reference
- `metrics/`: metric explanations and formulas
- `evaluation/`: datasets and evaluator flow
- `retrievers/`: vector database retriever examples
- `integrations/`: framework adapters
- `api/`: mkdocstrings API reference pages
