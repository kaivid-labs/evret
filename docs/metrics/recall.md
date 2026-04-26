# Recall

## What It Measures

Recall answers:

"Out of all relevant documents, how many did we retrieve in top-k?"

It focuses on coverage.

## Mathematical Formula

Recall@k for a single query:

$$
\text{Recall@}k = \frac{|\text{relevant docs} \cap \text{retrieved top-}k|}{|\text{relevant docs}|}
$$

Mean Recall@k across all queries:

$$
\text{Mean Recall@}k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{Recall@}k_i
$$

Recall ceiling when `k` is smaller than the number of relevant docs:

$$
\text{Recall@}k \leq \frac{k}{|\text{relevant docs}|} \quad \text{(when } k < |\text{relevant docs}| \text{)}
$$

## Formula Breakdown

- `|Q|`: total number of queries
- `|relevant ∩ retrieved@k|`: number of relevant hits in top-`k`
- `|relevant|`: total number of relevant docs for that query
- Query score range is `0` to `1`
- If `relevant` is empty, Evret returns `0.0` for that query

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: relevant hits in top 5 are `{d2, d8}` so hits = `2`

Step 2: total relevant ids = `3`

Step 3: `Recall@5(query) = 2 / 3 = 0.6667`

## When To Use

- Multi document QA
- Domains where missing information is risky
- Comparing retrievers for completeness
