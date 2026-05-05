# nDCG

## What It Measures

nDCG means Normalized Discounted Cumulative Gain.

It answers:

"How good is the ranking, with more credit for relevant results at higher positions?"

Evret's current `NDCG` metric uses binary relevance:

- relevant document: `1`
- non-relevant document: `0`

## Mathematical Formula

Discounted Cumulative Gain:

\[
\operatorname{DCG@}k =
\sum_{j=1}^{k}
\frac{\operatorname{rel}_j}{\log_2(j + 1)}
\]

Ideal DCG:

\[
\operatorname{IDCG@}k =
\sum_{j=1}^{\min(k, |G_i|)}
\frac{1}{\log_2(j + 1)}
\]

Normalized DCG:

\[
\operatorname{nDCG@}k_i =
\frac{\operatorname{DCG@}k_i}{\operatorname{IDCG@}k_i}
\]

Across all queries:

\[
\operatorname{Mean nDCG@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{nDCG@}k_i
\]

## Formula Breakdown

- \(\operatorname{rel}_j\): relevance value at rank \(j\)
- \(\log_2(j + 1)\): rank discount term
- \(G_i\): ground truth relevant ids for query \(i\)
- `DCG`: score for the actual ranking
- `IDCG`: best possible DCG for that query

If `IDCG` is `0`, Evret returns `0.0` for that query.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

### Step 1: Build Relevance By Rank

- Rank 1 `d1`: `0`
- Rank 2 `d4`: `0`
- Rank 3 `d2`: `1`
- Rank 4 `d9`: `0`
- Rank 5 `d8`: `1`

### Step 2: Compute DCG

\[
\operatorname{DCG@}5 =
\frac{0}{\log_2(2)}
+ \frac{0}{\log_2(3)}
+ \frac{1}{\log_2(4)}
+ \frac{0}{\log_2(5)}
+ \frac{1}{\log_2(6)}
= 0.8869
\]

### Step 3: Compute IDCG

Best ranking puts relevant ids first:

\[
\operatorname{IDCG@}5 =
\frac{1}{\log_2(2)}
+ \frac{1}{\log_2(3)}
+ \frac{1}{\log_2(4)}
= 2.1309
\]

### Step 4: Normalize

\[
\operatorname{nDCG@}5 =
\frac{0.8869}{2.1309}
= 0.4162
\]

## When To Use

- Rank quality comparison
- Reranker tuning
- Search and recommendation style ranking tasks
