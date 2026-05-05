# Recall

## What It Measures

Recall answers:

"Out of all relevant documents, how many did we retrieve in top-k?"

It focuses on coverage. High recall means the retriever found most of the relevant evidence.

## Mathematical Formula

For query \(i\):

\[
\operatorname{Recall@}k_i =
\frac{|G_i \cap R_i^{(k)}|}{|G_i|}
\]

Across all queries:

\[
\operatorname{MeanRecall@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{Recall@}k_i
\]

When \(k\) is smaller than the number of relevant documents, recall has a ceiling:

\[
\operatorname{Recall@}k_i \leq \frac{k}{|G_i|}
\quad \text{when } k < |G_i|
\]

## Formula Breakdown

- \(Q\): set of evaluation queries
- \(G_i\): ground truth relevant ids for query \(i\)
- \(R_i^{(k)}\): first \(k\) retrieved ids for query \(i\)
- \(|G_i \cap R_i^{(k)}|\): number of relevant hits in top-`k`
- \(|G_i|\): total relevant ids for that query

Evret returns `0.0` for a query when its relevant set is empty.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: relevant hits in top 5 are `{d2, d8}`, so hits = `2`.

Step 2: total relevant ids = `3`.

\[
\operatorname{Recall@}5 = \frac{2}{3} = 0.6667
\]

## When To Use

- Multi-document QA
- Domains where missing information is risky
- Comparing retrievers for completeness
