# MRR

## What It Measures

MRR means Mean Reciprocal Rank.

It answers:

"How early does the first relevant result appear?"

It only cares about the first relevant hit.

## Mathematical Formula

For query \(i\):

\[
\operatorname{RR@}k_i =
\begin{cases}
\frac{1}{\operatorname{rank}_i}, & \text{if the first relevant result appears in top-}k \\
0, & \text{otherwise}
\end{cases}
\]

Across all queries:

\[
\operatorname{MRR@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{RR@}k_i
\]

## Formula Breakdown

- \(Q\): set of evaluation queries
- \(\operatorname{rank}_i\): 1-based rank of the first relevant result for query \(i\)
- Rank 1 gives score `1.0`
- Rank 2 gives score `0.5`
- Rank 5 gives score `0.2`
- No relevant result in top-`k` gives `0`

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: first relevant hit is `d2` at rank `3`.

\[
\operatorname{RR@}5 = \frac{1}{3} = 0.3333
\]

If query scores over 3 queries are `[1.0, 0.5, 0.3333]`, then:

\[
\operatorname{MRR@}k =
\frac{1.0 + 0.5 + 0.3333}{3}
= 0.6111
\]

## When To Use

- Single-answer QA
- Search experience where users click early results
- Compare retrievers on first hit speed
