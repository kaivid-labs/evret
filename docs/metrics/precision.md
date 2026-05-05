# Precision

## What It Measures

Precision answers:

"How many of the top-k results are relevant?"

It focuses on result purity. High precision means the LLM context receives less irrelevant content.

## Mathematical Formula

For query \(i\):

\[
\operatorname{Precision@}k_i =
\frac{|G_i \cap R_i^{(k)}|}{k}
\]

Across all queries:

\[
\operatorname{MeanPrecision@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{Precision@}k_i
\]

## Formula Breakdown

- \(Q\): set of evaluation queries
- \(G_i\): ground truth relevant ids for query \(i\)
- \(R_i^{(k)}\): first \(k\) retrieved ids for query \(i\)
- \(|G_i \cap R_i^{(k)}|\): number of relevant hits in top-`k`
- \(k\): configured cutoff

Evret divides by the configured `k`, not by the number of results returned. If a retriever returns fewer than `k` results, missing slots still count against Precision@k.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: relevant hits are `{d2, d8}`, so hits = `2`.

Step 2: divide by `k = 5`.

\[
\operatorname{Precision@}5 = \frac{2}{5} = 0.4
\]

## When To Use

- Keep LLM context clean
- Reduce noisy retrieval outputs
- Evaluate top slot quality when context is limited
