# Precision

## What It Measures

Precision answers:

"How many of the top-k results are relevant?"

It focuses on result purity.

## Mathematical Formula

Precision@k for a single query:

$$
\text{Precision@}k = \frac{|\text{relevant docs} \cap \text{retrieved top-}k|}{k}
$$

Mean Precision@k across all queries:

$$
\text{Mean Precision@}k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \text{Precision@}k_i
$$

Precision-Recall tradeoff intuition:

$$
\uparrow k \Rightarrow \uparrow \text{Recall}, \downarrow \text{Precision}
\qquad
\downarrow k \Rightarrow \uparrow \text{Precision}, \downarrow \text{Recall}
$$

## Formula Breakdown

- `|Q|`: total number of queries
- `|relevant ∩ retrieved@k|`: number of relevant hits in top-`k`
- Denominator is fixed at `k`
- Query score range is `0` to `1`
- In Evret, denominator remains `k` by design

## Worked Example

Given:

- `k = 5`
- `retrieved@5 = [d1, d4, d2, d9, d8]`
- `relevant = {d2, d8, d10}`

Step 1: relevant hits are `{d2, d8}` so hits = `2`

Step 2: divide by `k = 5`

Step 3: `Precision@5(query) = 2 / 5 = 0.4`

## When To Use

- Keep LLM context clean
- Reduce noisy retrieval outputs
- Evaluate top slot quality when context is limited
