# PyConFR Sprint: Benchmarking Qdrant search techniques with evret

Every search method hands back a different set of chunks, so which one is right
for your data? This cookbook runs eight Qdrant search methods over the same
documents and scores each one with [evret](https://github.com/kaivid-labs/evret).

The corpus is three real Netflix 2025 SEC filings (10-K, 10-Q, 8-K) under
`data/`. Storage runs on Qdrant Cloud.

## The techniques

Each chunk is stored four times in one collection (dense, BM25 sparse, SPLADE
learned sparse, ColBERT multivector), so only the retrieval method changes.

| # | Technique | How it works |
|---|-----------|--------------|
| 1 | `dense` | One dense embedding, cosine similarity |
| 2 | `bm25` | Exact-term BM25, IDF-weighted |
| 3 | `splade` | Neural term expansion, no IDF modifier |
| 4 | `hybrid (no fusion)` | BM25 finds candidates, dense re-scores them |
| 5 | `hybrid (RRF)` | Dense + BM25 merged with Reciprocal Rank Fusion |
| 6 | `hybrid (DBSF)` | Dense + BM25 merged with Distribution-Based Score Fusion |
| 7 | `colbert` | Late-interaction retrieval with ColBERT MaxSim |
| 8 | `hybrid + colbert rerank` | Dense + BM25 candidates reranked by ColBERT MaxSim |

## How scoring works

An LLM writes one question per chunk, and that source chunk is the correct answer.
The benchmark then scores retrieval by matching document IDs (Hit Rate@k,
Precision@k, Recall@k, MRR, NDCG), so no LLM judge runs while scoring. evret's
`DatasetGenerator` produces the chunks and questions, and tags each question with
a category so the leaderboard can show a per-category Hit Rate.

## Setup

```bash
uv sync   # or: pip install "evret[qdrant,llm-openai]" qdrant-client fastembed pypdfium2 python-dotenv
```

Put an LLM key and a [Qdrant Cloud](https://cloud.qdrant.io) URL + key in a `.env`
file here:

```bash
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxxx.cloud.qdrant.io:6333
QDRANT_API_KEY=...
```

## Get the data

The filings are not in the repo, so download them first. EDGAR needs a contact
email in the `User-Agent`, so put your own name and email below.

```bash
mkdir -p data
UA="your-name your-email@example.com"
curl -sL -A "$UA" -o data/netflix_2025_10k.htm \
  https://www.sec.gov/Archives/edgar/data/1065280/000106528026000034/nflx-20251231.htm
curl -sL -A "$UA" -o data/netflix_2025_10q_q2.htm \
  https://www.sec.gov/Archives/edgar/data/1065280/000106528025000323/nflx-20250630.htm
curl -sL -A "$UA" -o data/netflix_2025_8k_oct.htm \
  https://www.sec.gov/Archives/edgar/data/1065280/000106528025000404/nflx-20251021.htm
```

Any `.htm`, `.html`, `.pdf`, or `.txt` file in `data/` works, so you can drop in
your own documents instead.

## Run

```bash
uv run generate_dataset.py   # documents -> generated_eval_dataset.json
uv run index.py              # embed + upsert dense/bm25/splade/colbert
uv run benchmark.py          # evaluate every technique, print leaderboard
```

`benchmark.py` prints a leaderboard (ranked by `ndcg@5`) plus a per-category Hit
Rate table, and writes `benchmark_results.json`.

## Results

| # | Technique | hit_rate@5 | precision@5 | recall@5 | mrr@5 | ndcg@5 |
|---|-----------|------------|-------------|----------|-------|--------|
| 1 | hybrid (DBSF) | 0.958 | 0.192 | 0.958 | 0.917 | 0.928 |
| 2 | hybrid (RRF) | 0.958 | 0.192 | 0.958 | 0.828 | 0.860 |
| 3 | bm25 (lexical) | 1.000 | 0.200 | 1.000 | 0.802 | 0.852 |
| 4 | colbert | 0.875 | 0.175 | 0.875 | 0.802 | 0.821 |
| 5 | hybrid + colbert rerank | 0.875 | 0.175 | 0.875 | 0.802 | 0.821 |
| 6 | splade (learned sparse) | 0.875 | 0.175 | 0.875 | 0.725 | 0.761 |
| 7 | dense | 0.833 | 0.167 | 0.833 | 0.574 | 0.638 |
| 8 | hybrid (no fusion) | 0.833 | 0.167 | 0.833 | 0.574 | 0.638 |

Hit Rate@5 by query category:

| Technique | broad_summary | direct_fact | keyword_search | paraphrase | specific_detail |
|-----------|---------------|-------------|----------------|------------|-----------------|
| hybrid (DBSF) | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| hybrid (RRF) | 1.000 | 0.875 | 1.000 | 1.000 | 1.000 |
| bm25 (lexical) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| colbert | 0.667 | 0.875 | 1.000 | 0.800 | 1.000 |
| hybrid + colbert rerank | 0.667 | 0.875 | 1.000 | 0.800 | 1.000 |
| splade (learned sparse) | 0.667 | 0.875 | 1.000 | 0.800 | 1.000 |
| dense | 1.000 | 0.875 | 1.000 | 0.800 | 0.600 |
| hybrid (no fusion) | 1.000 | 0.875 | 1.000 | 0.800 | 0.600 |

A few things to read from this. DBSF fusion ranked first because it kept the high
recall of the hybrid candidates and ordered the right chunk higher, which lifts
MRR and NDCG. BM25 found every gold chunk (recall 1.0), since many of the answers
are distinctive tokens like dollar figures and share counts that reward exact
matching. Dense alone was weakest on `specific_detail`, where those numeric facts
are easy to retrieve by keyword but hard to separate in embedding space.

Two rows are identical on purpose, and it is a property of this small corpus, not
a bug. `hybrid (no fusion)` matches `dense` because the BM25 prefetch limit (50)
is larger than the whole 23-chunk corpus, so the dense re-score sees every chunk
and behaves like plain dense. For the same reason `hybrid + colbert rerank`
matches `colbert`: the prefetch covers the full corpus, so ColBERT reranks the
same set it would have retrieved on its own. On a larger corpus these paths
separate.
