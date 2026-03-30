# Step 2 — Mean (Centroid) Classifier

## What this step does

Computes the **average image** for each digit class from the training set, then classifies every test image by finding which mean it is closest to (Euclidean distance). This is the simplest possible baseline classifier.

---

## The Idea

Think of it this way: if you've seen hundreds of handwritten `3`s, what does the "average 3" look like? You compute that average image and use it as the representative for the class. At test time, you compare an unknown digit to all 10 averages and pick the closest one.

---

## Training Phase

For each digit class `i` (0 through 9):

1. Collect all training images belonging to class `i` → a submatrix of shape `256 × nᵢ`
2. Average across all `nᵢ` columns → one 256-dimensional **mean image vector** `mᵢ`

This gives a `means` matrix of shape **256 × 10** (one mean per digit).

```
Number of training samples used per class:
  Digit 0: 767    Digit 5: 361
  Digit 1: 622    Digit 6: 420
  Digit 2: 475    Digit 7: 390
  Digit 3: 406    Digit 8: 377
  Digit 4: 409    Digit 9: 422
```

---

## Classification Phase

For each test image `y`:

1. Compute Euclidean distance to all 10 mean images:
   ```
   distance_i = || y - mᵢ ||₂    for i = 0, ..., 9
   ```
2. Assign the class with the **smallest distance**:
   ```
   predicted_class = argmin_i  distance_i
   ```

No training loop here — just 10 distance calculations per test image.

---

## Implementation Note

Instead of looping over each of the 4649 test images, we vectorized it using broadcasting:

```python
diff      = test_patterns[:, :, np.newaxis] - means[:, np.newaxis, :]  # 256 x 4649 x 10
distances = np.linalg.norm(diff, axis=0)                                # 4649 x 10
predicted = np.argmin(distances, axis=1)                                # 4649 predictions
```

This computes all 4649 × 10 = 46,490 distances in one shot — fast and clean.

---

## Results

```
Correct predictions : 3936 / 4649
Classification rate : 84.66%
Expected (from paper): 84.66%   <-- exact match
```

### Per-Class Accuracy

| Digit | Correct | Total | Accuracy |
|---|---|---|---|
| 0 | 656 | 786 | 83.5% |
| 1 | 644 | 647 | **99.5%** ← easiest |
| 2 | 362 | 454 | 79.7% |
| 3 | 368 | 418 | 88.0% |
| 4 | 363 | 443 | 81.9% |
| 5 | 271 | 355 | **76.3%** ← hardest |
| 6 | 354 | 414 | 85.5% |
| 7 | 351 | 402 | 87.3% |
| 8 | 253 | 331 | 76.4% |
| 9 | 314 | 399 | 78.7% |

### Confusion Matrix

```
        0     1     2     3     4     5     6     7     8     9
  0 |  656     1     3     4    10    19    73     2    17     1
  1 |    0   644     0     1     0     0     1     0     1     0
  2 |   14     4   362    13    25     5     4     9    18     0
  3 |    1     3     4   368     1    17     0     3    14     7
  4 |    3    16     6     0   363     1     8     1     5    40
  5 |   13     3     3    20    14   271     9     0    16     6
  6 |   23    11    13     0     9     3   354     0     1     0
  7 |    0     5     1     0     7     1     0   351     3    34
  8 |    9    19     5    12     6     6     0     1   253    20
  9 |    1    15     0     1    39     2     0    24     3   314
```

---

## Key Observations

- **Digit `1` is easiest (99.5%)** — handwritten `1`s are nearly always a simple vertical stroke, very consistent. The mean of all `1`s is a clean representative.
- **Digits `5` and `8` are hardest (~76%)** — people write these very differently. The mean image is a blurry blur that doesn't represent any particular style well.
- **Digit `0` often gets misclassified as `6` (73 times!)** — visually, a poorly written zero can look like a six. The mean classifier can't distinguish fine shape details.
- **The single mean is a lossy summary.** If a digit class has high intra-class variation (many writing styles), one average image fails to capture all of them.

---

## Why This Is a Baseline

- ✅ Extremely fast — training is just 10 averages, inference is 10 distance calculations
- ✅ Needs almost no memory — stores only 10 vectors of length 256
- ❌ 84.66% accuracy — misclassifies ~1 in 6 digits
- ❌ Cannot handle multiple "modes" within a class (e.g., people write `4` in two very different ways)

This sets the floor. Steps 3 and 4 will significantly improve on this.

---

*Previous: [Step 1 — Load Data](readme_step1.md)*  
*Next: [Step 3 — k-Nearest Neighbor Classifier](readme_step3.md)*
