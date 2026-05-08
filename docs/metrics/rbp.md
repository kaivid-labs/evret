# RBP

## What It Measures

RBP means Rank-Biased Precision.

It answers:

"How good are the results, assuming users have a certain patience level?"

RBP uses a persistence parameter `p` (0 < p < 1):

- `p = 0.5`: impatient users, ~2 positions examined
- `p = 0.8`: typical web search, ~5 positions examined
- `p = 0.95`: patient users, ~20 positions examined

Expected search depth: \(\frac{1}{1-p}\)

## Mathematical Formula

Rank-Biased Precision:

\[
\operatorname{RBP}(p)@k =
(1 - p) \times \sum_{i=1}^{k} p^{i-1} \times \operatorname{rel}(i)
\]

Across all queries:

\[
\operatorname{Mean RBP}(p)@k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{RBP}(p)@k_i
\]

Residual (for incomplete rankings):

\[
\operatorname{Residual} = p^k
\]

## Formula Breakdown

- \(p\): persistence parameter (probability of continuing to next rank)
- \(\operatorname{rel}(i)\): relevance at rank \(i\) (1 if relevant, 0 otherwise)
- \(p^{i-1}\): geometric weight at rank \(i\)
- \((1 - p)\): normalization factor
- Residual: upper bound contribution from unseen ranks

If there are no relevant documents, Evret returns `0.0` for that query.

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d2, d3, d4, d5]`
- `relevant = {d1, d3, d4}` (binary relevance)
- `p = 0.8`

### Step 1: Identify Relevant Ranks

- Rank 1 `d1`: relevant (1)
- Rank 2 `d2`: not relevant (0)
- Rank 3 `d3`: relevant (1)
- Rank 4 `d4`: relevant (1)
- Rank 5 `d5`: not relevant (0)

### Step 2: Compute Geometric Weights

- Rank 1: \(0.8^0 = 1.0\)
- Rank 2: \(0.8^1 = 0.8\)
- Rank 3: \(0.8^2 = 0.64\)
- Rank 4: \(0.8^3 = 0.512\)
- Rank 5: \(0.8^4 = 0.4096\)

### Step 3: Compute Contributions

- Rank 1: \((1 - 0.8) \times 1.0 \times 1 = 0.2\)
- Rank 2: \((1 - 0.8) \times 0.8 \times 0 = 0.0\)
- Rank 3: \((1 - 0.8) \times 0.64 \times 1 = 0.128\)
- Rank 4: \((1 - 0.8) \times 0.512 \times 1 = 0.1024\)
- Rank 5: \((1 - 0.8) \times 0.4096 \times 0 = 0.0\)

### Step 4: Sum Contributions

\[
\operatorname{RBP}(0.8)@5 = 0.2 + 0.0 + 0.128 + 0.1024 + 0.0 = 0.4304
\]

### Step 5: Compute Residual

\[
\operatorname{Residual} = 0.8^5 = 0.3277
\]

Total RBP with residual bound: `[0.4304, 0.7581]`

## When To Use

- User persistence/patience varies
- Need tunable position weighting
- Comparing systems with different result lengths
- Incomplete rankings need evaluation
