# Metrics Overview

Evret metrics score ranked retrieval results at a top-`k` cutoff.

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

- \(R_i^{(k)}\): first `k` retrieved document ids for query \(i\)
- \(G_i\): ground truth relevant ids for query \(i\)
- \(Q\): set of evaluation queries

Most metric pages in this section use this small example:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

In that case, relevant hits in top 5 are `d2` and `d8`.

## Batch Score

Evret computes each metric for every query, then returns the average.

For \(|Q|\) queries:

\[
\operatorname{score} =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{score}_i
\]

This is why each metric class has:

- `score_query(...)` for one query
- `score(...)` for a list of queries

## Implementation Notes

- Hit Rate, Recall, Precision, MRR, nDCG, and Average Precision all return scores from `0.0` to `1.0`.
- nDCG currently uses binary relevance, where relevant ids have relevance `1.0`.
- Precision divides by the configured `k`, not by the number of returned results.
- Average Precision divides by the total number of relevant ids for the query.
- Empty relevant sets score `0.0`.
