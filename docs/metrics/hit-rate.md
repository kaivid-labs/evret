# Hit Rate

## What It Measures

Hit Rate answers one question:

"Did we get at least one relevant document in top-k?"

It does not care whether the first hit is at rank 1 or rank 5.

## Mathematical Formula

$$
\text{Hit Rate} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{1}\left[\text{rel}_i \cap \text{retrieved}_i^{(k)} \neq \emptyset\right]
$$

For a single query:

$$
\text{Hit}_i =
\begin{cases}
1 & \text{if any relevant doc appears in top-}k \\
0 & \text{otherwise}
\end{cases}
$$

## Formula Breakdown

- `|Q|`: total number of queries
- `rel_i`: ground truth relevant set for query `i`
- `retrieved_i^(k)`: top-`k` retrieved set for query `i`
- `1[ ... ]`: indicator function (`1` if true, else `0`)

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: intersection is `{d2, d8}`

Step 2: intersection is not empty, so query score is `1`

If another query has no overlap, that query score is `0`.

If scores over 4 queries are `[1, 0, 1, 1]`, final Hit Rate is:

`(1 + 0 + 1 + 1) / 4 = 0.75`

## When To Use

- First debugging step in RAG retrieval
- Fast check after changing chunking or embeddings
- CI sanity check to catch obvious retrieval regressions
