# Average Precision

## What It Measures

Average Precision answers:

"Across the ranking, how good is precision each time we hit a relevant document?"

It is rank aware and rewards placing relevant documents earlier.

## Mathematical Formula

Precision at rank `k` (used in AP):

$$
P(k) = \frac{\text{number of relevant docs in top-}k}{k}
$$

Average Precision for a single query:

$$
AP = \frac{1}{R} \sum_{k=1}^{n} P(k) \cdot \mathbb{1}[\text{doc at rank } k \text{ is relevant}]
$$

Mean Average Precision across queries:

$$
MAP = \frac{1}{|Q|} \sum_{i=1}^{|Q|} AP_i
$$

## Formula Breakdown

- `R`: total number of relevant docs for the query
- `n`: total number of retrieved docs
- `P(k)`: precision at rank `k`
- `1[ ... ]`: indicator function (`1` when rank `k` doc is relevant)
- AP averages precision values only at relevant ranks
- If no relevant ids exist, Evret returns `0.0`

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Relevant hits in top 5 are at ranks `3` and `5`.

### Step 1: Precision at relevant ranks

- At rank 3, hits so far = 1 -> `P@3 = 1/3 = 0.3333`
- At rank 5, hits so far = 2 -> `P@5 = 2/5 = 0.4`

### Step 2: Sum precision values

`0.3333 + 0.4 = 0.7333`

### Step 3: Divide by total relevant count

Total relevant ids = `3`

`AP@5(query) = 0.7333 / 3 = 0.2444`

## When To Use

- Compare ranking models on full ranking behavior
- Evaluate tradeoff between early hits and overall relevance
- Benchmark retrieval quality with one score that is rank aware
