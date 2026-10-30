# TruLens judge + Qdrant example

End-to-end retrieval evaluation with [evret](https://github.com/kaivid-labs/evret):
generate an eval dataset from a real PDF, index the chunks into Qdrant Cloud, and
score retrieval using [TruLens](https://github.com/truera/trulens)
`context_relevance` as the relevance judge (`TruLensJudge`).

The sample document is Western Digital's 2026 Annual Report (Form ARS), filed
with the SEC.

```
data/western_digital_2026_annual_report.pdf   source document (downloaded below)
generate_dataset.py                           PDF -> chunks -> LLM-generated queries/answers (JSON)
index.py                                       chunks -> embeddings -> Qdrant Cloud collection
app.py                                         retrieve from Qdrant + evaluate with TruLensJudge
```

## Install

From this folder, with uv (uses the local evret source):

```bash
uv sync
```

Or with pip:

```bash
pip install "evret[trulens-openai]" qdrant-client fastembed pypdfium2 python-dotenv
```

## Get the data

SEC EDGAR rejects requests (HTTP 403) that don't identify the caller with a
contact email in the `User-Agent` — so put *your own* email below (this is an
SEC requirement, not a normal curl flag):

```bash
curl -L -A "your-name your-email@domain.com" \
  -o data/western_digital_2026_annual_report.pdf \
  https://www.sec.gov/Archives/edgar/data/106040/000162828026064921/wdc015705-ars.pdf
```

The filing is 100+ pages, so `generate_dataset.py` caps generation to the first
`MAX_PAGES` (default 12). Any text PDF works — drop it in `data/` and update
`PDF_PATH` in `generate_dataset.py`.

## Configure

Set these as environment variables or in a `.env` file in this folder:

```bash
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

## Run

```bash
uv run generate_dataset.py
uv run index.py
uv run app.py
```

## Results

```
context_relevance for top hit of q='What was the end date of Western Digital's fiscal year covered by this annual report?': 0.333
{'hit_rate@3': 1.0, 'mrr@3': 0.6666666666666666, 'recall@3': 1.0, 'ndcg@3': 0.75}
```

## Notes

- `app.py` drops `expected_doc_ids` and keeps only answerable queries so the
  judge is actually used — otherwise the evaluator matches on document IDs and
  never calls TruLens.
- `TruLensJudge` runs one LLM call per judged pair sequentially, so keep `k` and
  `examples_per_chunk` small to control latency and cost.
