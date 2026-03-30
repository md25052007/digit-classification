# Step 4 — SVD Basis Classifier

## What this step does

Builds a **low-dimensional subspace** for each digit class using Singular Value Decomposition (SVD), then classifies test images by finding which subspace they fit into best — measured by how much of the image's energy is *not* explained (the residual).

This is the most mathematically elegant of the three methods — it's entirely grounded in linear algebra.

---

## The Core Idea

Imagine all handwritten `3`s live roughly in some low-dimensional "space of 3-ness" inside the 256-dimensional pixel space. SVD finds the best axes (directions) that span that space.

If a test image truly is a `3`, projecting it onto the `3` subspace should explain most of it — leaving a **small residual**. Projecting it onto the `7` subspace would leave a **large residual** because the image doesn't look like a 7.

**→ Classify by: which class subspace gives the smallest residual?**

---

## Training Phase — Building Per-Class SVD Bases

For each digit class `j` (0 through 9):

1. Collect all training images of digit `j` → matrix `X_j` of shape `256 × n_j`
2. Compute the SVD:
   ```
   X_j = U · Σ · Vᵀ
   ```
   where `U` has shape `256 × 256` — its columns are the **left singular vectors**
3. Keep only the first `k` columns: `U_k` (shape `256 × k`)

`U_k` is an **orthonormal basis** for the most important k-dimensional subspace of digit class `j`'s training data. The first column captures the most variation, the second captures the next most, and so on — like "eigen-digits."

This gives 10 basis sets: `U_0, U_1, ..., U_9`, one per digit.

**SVD computation time: 0.69 seconds** for all 10 classes.

---

## Classification Phase — Residual Error

For each test image `y`, and each digit class `j`:

1. Project `y` onto class `j`'s subspace:
   ```
   projection = U_k · (U_kᵀ · y)
   ```
2. The **residual** is the part of `y` not explained by that subspace:
   ```
   residual_j(y) = || y - U_k · U_kᵀ · y ||₂
   ```
3. Pick the class with **minimum residual**:
   ```
   predicted class = argmin_j  residual_j(y)
   ```

### Efficient computation

Since `U_k` has orthonormal columns, we use the identity:
```
||y - U U.T y||² = ||y||² - ||U.T y||²
```

This means instead of computing a full projection, we just compute:
- `||y||²` once (same for all classes)
- `||U_kᵀ y||²` per class (a cheap dot product)

No 256×256 matrix is ever formed. Clean and fast.

---

## Choosing k — The Sweep

We swept `k` from 1 to 20 and measured accuracy at each value:

```
k= 1: 84.66%           k=11: 96.47%
k= 2: 88.06%           k=12: 96.54%
k= 3: 92.06%           k=13: 96.36%
k= 4: 93.35%           k=14: 96.47%
k= 5: 94.24%           k=15: 96.58%
k= 6: 95.18%           k=16: 96.43%
k= 7: 95.61%           k=17: 96.62%  <-- best
k= 8: 95.74%           k=18: 96.43%
k= 9: 95.96%           k=19: 96.24%
k=10: 96.45%           k=20: 95.98%
```

### Interesting pattern:
- **k=1** gives 84.66% — exactly the same as the mean classifier! This is not a coincidence: with k=1, the only "basis vector" is the first left singular vector, which points in the direction of the mean.
- Accuracy **rises sharply** from k=1 to k=10 as more variation is captured
- Accuracy **peaks at k=17** then starts dropping — too many basis vectors start fitting noise and class-overlap regions
- The sweet spot is k=17: enough to capture genuine digit variation, not so many that bases start overlapping across classes

---

## Results

```
Best k        : 17
Best accuracy : 96.62%
Expected (paper): 96.62% at k=17   <-- exact match
```

### Confusion Matrix (k=17)

```
        0     1     2     3     4     5     6     7     8     9
  0 |  772     2     1     3     1     1     2     1     3     0
  1 |    0   646     0     0     0     0     0     0     0     1
  2 |    3     6   431     6     0     3     1     2     2     0
  3 |    1     1     4   401     0     7     0     0     4     0
  4 |    2     8     1     0   424     1     1     5     0     1
  5 |    2     0     0     5     2   335     7     1     1     2
  6 |    6     4     0     0     2     3   399     0     0     0
  7 |    0     2     0     0     2     0     0   387     0    11
  8 |    2     9     1     5     1     1     0     0   309     3
  9 |    0     5     0     1     0     0     0     4     1   388
```

### Per-Class Accuracy (k=17)

| Digit | Correct | Total | Accuracy |
|---|---|---|---|
| 0 | 772 | 786 | 98.2% |
| 1 | 646 | 647 | **99.8%** ← best |
| 2 | 431 | 454 | 94.9% |
| 3 | 401 | 418 | 95.9% |
| 4 | 424 | 443 | 95.7% |
| 5 | 335 | 355 | 94.4% |
| 6 | 399 | 414 | 96.4% |
| 7 | 387 | 402 | 96.3% |
| 8 | 309 | 331 | **93.4%** ← hardest |
| 9 | 388 | 399 | 97.2% |

---

## Key Observations

- **Digit `1` hits 99.8%** — `1`s are so simple that even a 17-dimensional subspace perfectly captures all their variation
- **Digit `7` confused with `9` (11 times)** — both have diagonal strokes at the top; their subspaces partially overlap
- **Digit `4` confused with `1` (8 times)** — some people write `4`s with a single vertical stroke resembling a `1`
- **SVD is far more compact than k-NN**: to classify, you only need 10 matrices of size `256 × 17` = 43,520 numbers. k-NN stores 4,649 × 256 = 1,190,144 numbers — **27× more memory**

---

## Why the SVD Approach Is Powerful

| Property | Value |
|---|---|
| Memory needed | 10 × (256 × 17) = ~43K floats |
| Test time | One matrix multiply per class per image |
| Accuracy | 96.62% |
| Mathematical basis | Subspace projection / least squares |

The SVD classifier generalizes better as image dimensionality grows. k-NN's distance computations become exponentially harder in high dimensions ("curse of dimensionality"), while SVD only cares about the structure within each class.

---

*Previous: [Step 3 — k-NN Classifier](readme_step3.md)*  
*Next: [Final Summary](readme_summary.md)*
