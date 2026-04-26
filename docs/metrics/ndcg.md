# nDCG

## What It Measures

nDCG means Normalized Discounted Cumulative Gain.

It answers:

"How good is the ranking, with more credit for higher positions?"

In Evret today, relevance is binary.

- Relevant document has relevance `1`
- Non relevant document has relevance `0`

## Mathematical Formula

DCG at rank `k`:

$$
DCG@k = \sum_{i=1}^{k} \frac{\text{rel}_i}{\log_2(i + 1)}
$$

Ideal DCG (`IDCG`) is DCG computed on the perfectly sorted ranking.

Normalized DCG:

$$
nDCG@k = \frac{DCG@k}{IDCG@k}
$$

Mean nDCG across queries:

$$
\text{Mean nDCG@}k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{nDCG@}k_i
$$

## Formula Breakdown

- `rel_i`: relevance at rank `i` in retrieved list
- `log_2(i + 1)`: rank discount term
- Higher rank gets lower denominator, so it gets more weight
- `IDCG`: best possible DCG for the query
- If `IDCG` is `0`, Evret returns `0.0` for that query

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

### Step 1: Build relevance by rank

- Rank 1 `d1`: `rel_1 = 0`
- Rank 2 `d4`: `rel_2 = 0`
- Rank 3 `d2`: `rel_3 = 1`
- Rank 4 `d9`: `rel_4 = 0`
- Rank 5 `d8`: `rel_5 = 1`

### Step 2: Compute DCG

`DCG@5 = 0/log2(2) + 0/log2(3) + 1/log2(4) + 0/log2(5) + 1/log2(6)`

`DCG@5 = 0 + 0 + 0.5 + 0 + 0.3869 = 0.8869`

### Step 3: Compute IDCG

Best ranking puts relevant ids first.

`IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4)`

`IDCG@5 = 1 + 0.6309 + 0.5 = 2.1309`

### Step 4: Normalize

`nDCG@5 = 0.8869 / 2.1309 = 0.4162`

## When To Use

- Rank quality comparison
- Reranker tuning
- Search and recommendation style ranking tasks
