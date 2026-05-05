# Average Precision

## What It Measures

Average Precision answers:

"Across the ranking, how good is precision each time we hit a relevant document?"

It is rank aware and rewards placing relevant documents earlier.

## Mathematical Formula

Precision at rank \(j\):

\[
P(j) =
\frac{|G_i \cap R_i^{(j)}|}{j}
\]

Average Precision for query \(i\):

\[
\operatorname{AP@}k_i =
\frac{1}{|G_i|}
\sum_{j=1}^{k}
P(j)
\cdot
\mathbf{1}[r_{ij} \in G_i]
\]

Mean Average Precision across queries:

\[
\operatorname{MAP@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{AP@}k_i
\]

## Formula Breakdown

- \(Q\): set of evaluation queries
- \(G_i\): ground truth relevant ids for query \(i\)
- \(R_i^{(j)}\): first \(j\) retrieved ids for query \(i\)
- \(r_{ij}\): retrieved id at rank \(j\) for query \(i\)
- \(P(j)\): precision at rank \(j\)
- \(\mathbf{1}[\cdot]\): indicator function

Evret divides AP by the total number of relevant ids, not only the number found in top-`k`. If no relevant ids exist, Evret returns `0.0`.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Relevant hits in top 5 are at ranks `3` and `5`.

### Step 1: Precision At Relevant Ranks

- At rank 3, hits so far = 1, so `P@3 = 1 / 3 = 0.3333`
- At rank 5, hits so far = 2, so `P@5 = 2 / 5 = 0.4`

### Step 2: Sum Precision Values

\[
0.3333 + 0.4 = 0.7333
\]

### Step 3: Divide By Total Relevant Count

Total relevant ids = `3`.

\[
\operatorname{AP@}5 =
\frac{0.7333}{3}
= 0.2444
\]

## When To Use

- Compare ranking models on full ranking behavior
- Evaluate tradeoff between early hits and overall relevance
- Benchmark retrieval quality with one rank-aware score
