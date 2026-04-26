# MRR

## What It Measures

MRR means Mean Reciprocal Rank.

It answers:

"How early does the first relevant result appear?"

It only cares about the first relevant hit.

## Mathematical Formula

$$
MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

For a single query:

$$
RR = \frac{1}{\text{rank of first relevant doc}}
\quad \Rightarrow \quad
RR_{\text{rank=1}} = 1.0, \quad RR_{\text{rank=2}} = 0.5, \quad RR_{\text{rank=5}} = 0.2
$$

If no relevant document appears in top-`k`, reciprocal rank is `0`.

## Formula Breakdown

- `|Q|`: total number of queries
- `rank_i`: 1-based rank of first relevant doc for query `i`
- Rank 1 gives score `1.0`
- Rank 2 gives score `0.5`
- Rank 5 gives score `0.2`
- No hit gives `0`

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: first relevant hit is `d2` at rank `3`

Step 2: `RR@5(query) = 1 / 3 = 0.3333`

If query scores over 3 queries are `[1.0, 0.5, 0.3333]`, then:

`MRR@k = (1.0 + 0.5 + 0.3333) / 3 = 0.6111`

## When To Use

- Single answer QA
- Search experience where users click early results
- Compare retrievers on first hit speed
