# Hit Rate

## What It Measures

Hit Rate answers one question:

"Did we retrieve at least one relevant document in the top-k results?"

It does not care whether the first hit is at rank 1 or rank 5. A query gets `1` when any relevant item appears in top-`k`, otherwise it gets `0`.

## Mathematical Formula

For query \(i\):

\[
\operatorname{Hit@}k_i =
\begin{cases}
1, & \text{if } G_i \cap R_i^{(k)} \neq \varnothing \\
0, & \text{otherwise}
\end{cases}
\]

Across all queries:

\[
\operatorname{HitRate@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{Hit@}k_i
\]

## Formula Breakdown

- \(Q\): set of evaluation queries
- \(G_i\): ground truth relevant ids for query \(i\)
- \(R_i^{(k)}\): first \(k\) retrieved ids for query \(i\)
- \(G_i \cap R_i^{(k)}\): relevant ids found in the top-`k`
- \(\varnothing\): empty set

Evret returns `0.0` for a query when its relevant set is empty or no retrieved result matches.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: intersection is `{d2, d8}`.

Step 2: intersection is not empty, so query score is `1`.

If scores over 4 queries are `[1, 0, 1, 1]`, final Hit Rate is:

\[
\frac{1 + 0 + 1 + 1}{4} = 0.75
\]

## When To Use

- First debugging step in information retrieval
- Fast check after changing chunking or embeddings
- CI sanity check to catch obvious retrieval regressions
