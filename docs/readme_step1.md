# Step 1 — Load & Inspect the USPS Dataset

## What this step does

Loads the raw USPS Handwritten Digit dataset from the downloaded `.mat` file and verifies it matches the expected structure before any classification is done.

---

## The Dataset

The **USPS Handwritten Digit Dataset** was collected at CEDAR (Center of Excellence in Document Analysis and Recognition) at SUNY Buffalo, sponsored by the US Postal Service.

- **Source:** http://www.gaussianprocess.org/gpml/data/
- **Format:** MATLAB `.mat` file (`usps_resampled.mat`)
- **Contents:** 9,298 grayscale images of handwritten digits (0–9)

---

## What's Inside the `.mat` File

When loaded, it contains 4 matrices:

| Variable | Shape | What it is |
|---|---|---|
| `train_patterns` | 256 × 4649 | Training images. Each **column** is one flattened 16×16 image |
| `test_patterns` | 256 × 4649 | Test images. Same format |
| `train_labels` | 10 × 4649 | Labels for training images |
| `test_labels` | 10 × 4649 | Labels for test images |

### Why 256 rows?
Each image is **16 × 16 pixels = 256 pixels**, flattened into a single column vector. To visualize a digit, you'd reshape that vector back to 16×16.

### Why 4649 columns?
The dataset is split 50/50 — 4649 training images and 4649 test images.

### Pixel values
Pixels are **normalized to the range [−1, 1]** (not 0–255). Confirmed from output: `[-1.00, 1.00]`.

---

## Label Encoding

Labels use a **±1 one-hot** scheme — NOT the usual 0/1 one-hot:

- Each label column has **10 rows** (one per digit class 0–9)
- The row corresponding to the correct digit = **+1**
- All other rows = **−1**

Example: if an image is the digit `3`, then row index 3 = +1, rows 0,1,2,4,...,9 = −1.

To recover the actual digit class: `argmax(label_column)` → gives the index of the +1.

---

## Output We Got

```
train_patterns shape : (256, 4649)
test_patterns  shape : (256, 4649)
train_labels   shape : (10, 4649)
test_labels    shape : (10, 4649)

Pixel value range    : [-1.00, 1.00]

Training samples per digit class:
  Digit 0: 767    Digit 5: 361
  Digit 1: 622    Digit 6: 420
  Digit 2: 475    Digit 7: 390
  Digit 3: 406    Digit 8: 377
  Digit 4: 409    Digit 9: 422

Test samples per digit class:
  Digit 0: 786    Digit 5: 355
  Digit 1: 647    Digit 6: 414
  Digit 2: 454    Digit 7: 402
  Digit 3: 418    Digit 8: 331
  Digit 4: 443    Digit 9: 399
```

### Observations
- Classes are **imbalanced** — digit `0` has the most samples (767 train / 786 test), digit `5` the fewest (361 / 355). This matters when evaluating per-class accuracy.
- Everything matches the expected structure from the methodology doc exactly.

---

## Why This Step Matters

Before running any classifier, you need to confirm:
1. The file loaded without corruption
2. Shapes are exactly what the math expects (256-dimensional vectors, 10-class labels)
3. Pixel normalization is correct (algorithms assume [−1, 1])
4. You know how many samples exist per class

All 4 checks passed ✅

---

*Next: [Step 2 — Mean (Centroid) Classifier](readme_step2.md)*
