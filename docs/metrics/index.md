# Metrics Overview

This section explains each retrieval metric in plain words.

Each metric answers a different question. Use the metric that matches your product goal.

## Quick Pick

| Goal | Metric |
| --- | --- |
| Check if at least one relevant result appears | Hit Rate |
| Check how much relevant content is recovered | Recall |
| Check how clean the top results are | Precision |
| Check how early the first relevant hit appears | MRR |
| Check ranking quality across positions | nDCG |
| Check rank aware precision over all relevant hits | Average Precision |

## Shared Setup

For one query, define:

- `retrieved@k`: first `k` retrieved document ids
- `relevant`: set of ground truth relevant ids

Most metric pages in this section use this small example:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

In that case, relevant hits in top 5 are `d2` and `d8`.

## Batch Score

Evret computes each metric for every query, then returns the average.

For `N` queries:

`final_score = (score_query_1 + score_query_2 + ... + score_query_N) / N`

This is why each metric class has:

- `score_query(...)` for one query
- `score(...)` for a list of queries
