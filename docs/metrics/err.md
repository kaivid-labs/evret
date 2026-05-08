# ERR

## What It Measures

ERR means Expected Reciprocal Rank.

It answers:

"How satisfied will users be, assuming they examine results sequentially and stop when satisfied?"

ERR uses a cascade model with graded relevance:

- grade `0`: not relevant
- grade `1-4`: increasing relevance levels
- higher grades mean higher satisfaction probability

## Mathematical Formula

Satisfaction probability:

\[
R(i) = \frac{2^{\operatorname{grade}(i)} - 1}{2^{\max\_grade}}
\]

Expected Reciprocal Rank:

\[
\operatorname{ERR@}k =
\sum_{i=1}^{k}
\frac{1}{i} \times R(i) \times \prod_{j=1}^{i-1} (1 - R(j))
\]

Across all queries:

\[
\operatorname{Mean ERR@}k =
\frac{1}{|Q|}
\sum_{i=1}^{|Q|}
\operatorname{ERR@}k_i
\]

## Formula Breakdown

- \(R(i)\): probability user is satisfied at rank \(i\)
- \(\operatorname{grade}(i)\): relevance grade at rank \(i\) (0 to max_grade)
- \(\max\_grade\): maximum relevance grade (default: 4)
- \(\frac{1}{i}\): reciprocal rank weight
- \(\prod_{j=1}^{i-1} (1 - R(j))\): cascade probability (user wasn't satisfied before rank \(i\))

If there are no relevant documents, Evret returns `0.0` for that query.

## Worked Example

Given:

- `k = 3`
- `retrieved@3 = [d1, d2, d3]`
- `relevant = {d1: 3, d2: 4, d3: 1}` (graded relevance)
- `max_grade = 4`

### Step 1: Compute Satisfaction Probabilities

- Rank 1 `d1` (grade 3): \(R(1) = \frac{2^3 - 1}{2^4} = \frac{7}{16} = 0.4375\)
- Rank 2 `d2` (grade 4): \(R(2) = \frac{2^4 - 1}{2^4} = \frac{15}{16} = 0.9375\)
- Rank 3 `d3` (grade 1): \(R(3) = \frac{2^1 - 1}{2^4} = \frac{1}{16} = 0.0625\)

### Step 2: Compute ERR Contributions

Position 1:

\[
\frac{1}{1} \times 0.4375 \times 1 = 0.4375
\]

Position 2:

\[
\frac{1}{2} \times 0.9375 \times (1 - 0.4375) = 0.5 \times 0.9375 \times 0.5625 = 0.2637
\]

Position 3:

\[
\frac{1}{3} \times 0.0625 \times (1 - 0.4375) \times (1 - 0.9375) = 0.3333 \times 0.0625 \times 0.5625 \times 0.0625 = 0.0007
\]

### Step 3: Sum Contributions

\[
\operatorname{ERR@}3 = 0.4375 + 0.2637 + 0.0007 = 0.7019
\]

## When To Use

- Graded relevance judgments available
- Modeling user satisfaction behavior
- Cascade browsing patterns matter
- Earlier positions should be weighted heavily
