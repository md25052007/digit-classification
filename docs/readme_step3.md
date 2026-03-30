# Step 3 — k-Nearest Neighbor (k-NN) Classifier

## What this step does

Instead of comparing a test image to just 10 class means, k-NN compares it to **every single training image** and lets the `k` closest ones vote on the label. More data → better decisions.

---

## The Idea

Given an unknown test digit `y`:
1. Measure its Euclidean distance to all 4,649 training images
2. Find the `k` nearest training images (smallest distances)
3. Look at their true labels
4. **Majority vote** → the most common label among the k neighbors is the prediction

---

## No Explicit Training Phase

Unlike the mean classifier, k-NN has **no training step**. You just store the entire training set and do all the work at test time. This is called a *lazy learner*.

---

## The Distance Matrix Trick

Naively, you'd loop over all 4,649 test images and for each one compute 4,649 distances — slow. Instead, we compute the **full 4649 × 4649 distance matrix in one shot** using the identity:

```
||a - b||² = ||a||² + ||b||² - 2·(aᵀb)
```

In code:
```python
train_sq = sum of squared pixel values per training image   # (4649,)
test_sq  = sum of squared pixel values per test image       # (4649,)
cross    = test_patterns.T @ train_patterns                 # (4649 x 4649) dot products

dist_sq  = test_sq[:, newaxis] + train_sq[newaxis, :] - 2 * cross  # (4649 x 4649)
```

This computes all **21.6 million distances** in **0.34 seconds** — pure NumPy, no loops.

---

## Results

```
k=1:  96.99%   (paper: 96.99%)
k=3:  97.07%   (paper: 97.07%)  <-- best
k=5:  96.62%   (paper: 96.62%)
```

All three match the paper exactly ✅

### Why k=3 is best?
- **k=1** is sensitive to noise — one mislabeled or weird training sample can throw off the prediction
- **k=3** smooths out that noise with a vote, without averaging over too many potentially irrelevant neighbors
- **k=5** starts pulling in neighbors that are farther away = less similar, introducing errors

---

## Confusion Matrix (k=3)

```
        0     1     2     3     4     5     6     7     8     9
  0 |  778     0     4     2     0     1     0     0     0     1
  1 |    0   643     0     0     1     0     1     0     1     1
  2 |    3     1   435     5     2     1     0     6     1     0
  3 |    1     0     1   402     0     5     0     1     5     3
  4 |    0     2     1     0   420     0     3     1     0    16
  5 |    4     0     1    10     0   332     3     1     2     2
  6 |    3     1     1     0     2     2   405     0     0     0
  7 |    0     0     0     0     1     0     0   394     1     6
  8 |    1     1     1     3     2     3     1     2   314     3
  9 |    0     0     0     1     2     0     0     5     1   390
```

### Per-Class Accuracy (k=3)

| Digit | Correct | Total | Accuracy |
|---|---|---|---|
| 0 | 778 | 786 | **99.0%** |
| 1 | 643 | 647 | 99.4% |
| 2 | 435 | 454 | 95.8% |
| 3 | 402 | 418 | 96.2% |
| 4 | 420 | 443 | 94.8% |
| 5 | 332 | 355 | **93.5%** ← still hardest |
| 6 | 405 | 414 | 97.8% |
| 7 | 394 | 402 | 98.0% |
| 8 | 314 | 331 | 94.9% |
| 9 | 390 | 399 | 97.7% |

Compare digit `5` here (93.5%) vs the mean classifier (76.3%) — k-NN is dramatically better because it doesn't rely on one blurry average; actual training examples guide the decision.

---

## Key Observations

- **Digit `4` often gets confused with `9` (16 times)** — both have a loop at the top and a stem; easy to mix up in handwriting
- **Digit `5` still causes the most errors** — confusion with `3` (10 cases) and `6` (3 cases)
- **k-NN essentially eliminates the `0→6` confusion** (73 errors in mean classifier → 0 errors here)

---

## The Tradeoff

| Property | Mean Classifier | k-NN (k=3) |
|---|---|---|
| Training time | < 1s | None |
| Test time | < 1s | ~0.5s total (vectorized) |
| Memory | 256 × 10 floats | All 4649 training images |
| Accuracy | 84.66% | **97.07%** |

k-NN wins on accuracy but must store **and search** the full training set. For 4,649 images it's fine — for millions of images it becomes impractical.

---

*Previous: [Step 2 — Mean Classifier](readme_step2.md)*  
*Next: [Step 4 — SVD Basis Classifier](readme_step4.md)*
