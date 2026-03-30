# Classification of Handwritten Digits — Methodology & Workflow

**Course:** MAT 167: Applied Linear Algebra — Lecture 21  
**Instructor:** Naoki Saito, Department of Mathematics, University of California, Davis  
**Date:** May 17, 2017

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Data Representation & Notation](#2-data-representation--notation)
3. [Algorithm 1 — Mean (Centroid) Classifier](#3-algorithm-1--mean-centroid-classifier)
4. [Algorithm 2 — k-Nearest Neighbor (k-NN) Classifier](#4-algorithm-2--k-nearest-neighbor-k-nn-classifier)
5. [Algorithm 3 — SVD Basis Classifier](#5-algorithm-3--svd-basis-classifier)
6. [Performance Summary](#6-performance-summary)

---

## 1. Dataset Overview

### Source

The **USPS Handwritten Digit Dataset** was collected at the Center of Excellence in Document Analysis and Recognition (CEDAR) at SUNY Buffalo, under a project sponsored by the US Postal Service.

> Reference: J. J. Hull, *"A database for handwritten text recognition research,"* IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 16, no. 5, pp. 550–554, 1994.

Dataset available at: http://www.gaussianprocess.org/gpml/data/

### Structure

| Property | Value |
|---|---|
| Total digits | 9,298 handwritten single digits (0–9) |
| Image size | 16 × 16 pixels |
| Pixel value range | Normalized to [−1, 1] |
| Training set size | 4,649 digits |
| Test set size | 4,649 digits |

### Storage Format (MATLAB)

Each 16×16 image is **flattened to a 256-dimensional vector** and stored as a column in one of two matrices:

- `train_patterns` — size **256 × 4649** (training images)
- `test_patterns` — size **256 × 4649** (test images)

> **Visualization note:** To display a digit, reshape the 256-length column vector to 16×16, **transpose** it, then render with `imagesc`.

### Label Encoding

Labels use a **±1 one-hot** scheme stored in `train_labels` and `test_labels`, each of size **10 × 4649**.

For the j-th digit:
- If the digit represents class *i*, then row *(i+1)* of column *j* = **+1**
- All other rows of column *j* = **−1**

---

## 2. Data Representation & Notation

| Symbol | Meaning |
|---|---|
| `d = 256` | Dimensionality of each image (16×16 pixels flattened) |
| `n = 4649` | Number of training samples |
| `m = 4649` | Number of test samples |
| `X = [x₁ ··· xₙ] ∈ ℝ^(d×n)` | Training data matrix (columns = training images) |
| `Y = [y₁ ··· yₘ] ∈ ℝ^(d×m)` | Test data matrix (columns = test images) |
| `X^(j)` | Sub-matrix of X containing only training images of digit class j |
| `nⱼ` | Number of training images for digit j |

---

## 3. Algorithm 1 — Mean (Centroid) Classifier

### Concept

The simplest possible approach: represent each digit class by its average (mean) image, then classify by nearest mean.

### Workflow

```
TRAINING
─────────────────────────────────────────────────────────────
For each digit class i = 0, 1, ..., 9:
  1. Collect all training images belonging to class i
  2. Compute the mean image:
       mᵢ = (1/nᵢ) · Σ xⱼ    (sum over all j where label = i)

CLASSIFICATION
─────────────────────────────────────────────────────────────
For each test digit yⱼ:
  1. Compute Euclidean distance to each mean:
       d(mᵢ, yⱼ) = ‖mᵢ − yⱼ‖₂    for i = 0, ..., 9
  2. Assign class label k where:
       k = argmin_i ‖mᵢ − yⱼ‖₂
```

### Distance Metric

The **Euclidean (ℓ₂) distance** was chosen as the simplest option, though other choices are valid: ℓ₁-norm, ℓ∞-norm, cosine similarity, etc.

### Result

**Overall classification rate: 84.66%**

Confusion Matrix (rows = true class, columns = predicted class):

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 656 | 1 | 3 | 4 | 10 | 19 | 73 | 2 | 17 | 1 |
| **1** | 0 | 644 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 |
| **2** | 14 | 4 | 362 | 13 | 25 | 5 | 4 | 9 | 18 | 0 |
| **3** | 1 | 3 | 4 | 368 | 1 | 17 | 0 | 3 | 14 | 7 |
| **4** | 3 | 16 | 6 | 0 | 363 | 1 | 8 | 1 | 5 | 40 |
| **5** | 13 | 3 | 3 | 20 | 14 | 271 | 9 | 0 | 16 | 6 |
| **6** | 23 | 11 | 13 | 0 | 9 | 3 | 354 | 0 | 1 | 0 |
| **7** | 0 | 5 | 1 | 0 | 7 | 1 | 0 | 351 | 3 | 34 |
| **8** | 9 | 19 | 5 | 12 | 6 | 6 | 0 | 1 | 253 | 20 |
| **9** | 1 | 15 | 0 | 1 | 39 | 2 | 0 | 24 | 3 | 314 |

**Limitation:** A single mean image is a very lossy summary of an entire digit class. It struggles when digit shapes have high intra-class variation (e.g., digit '5' varies a lot in writing style).

---

## 4. Algorithm 2 — k-Nearest Neighbor (k-NN) Classifier

### Concept

Instead of comparing to a single class prototype, compare to all training examples and let the *k* closest ones vote.

### Workflow

```
SETUP
─────────────────────────────────────────────────────────────
  Select k from small odd integers: {1, 3, 5, ...}

CLASSIFICATION (no explicit training phase)
─────────────────────────────────────────────────────────────
For each test digit yⱼ:
  1. Compute Euclidean distance from yⱼ to every training digit xᵢ:
       d(xᵢ, yⱼ) = ‖xᵢ − yⱼ‖₂    for i = 1, ..., n (n = 4649)
  2. Find the k nearest training digits (k smallest distances)
  3. Look up the labels of these k neighbors
  4. Take a majority vote → assign as class label of yⱼ
```

> **Implementation note:** Tested using MATLAB's `knnclassify` function from the Bioinformatics Toolbox.

### Results

| k value | Classification Rate |
|---|---|
| k = 1 | 96.99% |
| **k = 3** | **97.07%** ← best |
| k = 5 | 96.62% |

Confusion Matrix for k = 3:

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 778 | 0 | 4 | 2 | 0 | 1 | 0 | 0 | 0 | 1 |
| **1** | 0 | 643 | 0 | 0 | 1 | 0 | 1 | 0 | 1 | 1 |
| **2** | 3 | 1 | 435 | 5 | 2 | 1 | 0 | 6 | 1 | 0 |
| **3** | 1 | 0 | 1 | 402 | 0 | 5 | 0 | 1 | 5 | 3 |
| **4** | 0 | 2 | 1 | 0 | 420 | 0 | 3 | 1 | 0 | 16 |
| **5** | 4 | 0 | 1 | 10 | 0 | 332 | 3 | 1 | 2 | 2 |
| **6** | 3 | 1 | 1 | 0 | 2 | 2 | 405 | 0 | 0 | 0 |
| **7** | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 394 | 1 | 6 |
| **8** | 1 | 1 | 1 | 3 | 2 | 3 | 1 | 2 | 314 | 3 |
| **9** | 0 | 0 | 0 | 1 | 2 | 0 | 0 | 5 | 1 | 390 |

**Limitation:** k-NN is computationally expensive at test time — every prediction requires computing distances to all 4,649 training examples. It also stores the entire training set.

---

## 5. Algorithm 3 — SVD Basis Classifier

### Concept

Build a **low-dimensional orthonormal subspace** for each digit class using Singular Value Decomposition. Classify a test image by finding which class subspace it projects onto with the **smallest residual error**.

This approach is grounded in linear algebra: if a test image truly belongs to digit class j, it should be well-approximated by the basis vectors that span the space of all training images of class j.

---

### Phase 1 — Training: Build Per-Class SVD Bases

```
For each digit class j = 0, 1, ..., 9:

  1. Collect all training images of digit j into a matrix:
       X^(j) ∈ ℝ^(d × nⱼ)
       where d = 256 and nⱼ = number of training images for digit j

  2. Choose rank parameter k (same k for all 10 classes)

  3. Compute the rank-k SVD of X^(j):
       X^(j) ≈ U^(j)_k  Σ^(j)_k  V^(j)ᵀ_k

       where:
         U^(j)_k  ∈ ℝ^(d×k)    ← first k left singular vectors  [KEEP THIS]
         Σ^(j)_k  ∈ ℝ^(k×k)    ← top k singular values (diagonal)
         V^(j)_k  ∈ ℝ^(nⱼ×k)  ← first k right singular vectors

  4. Store U^(j)_k for classification use
     (Use MATLAB's svds(X, k) for efficiency — computes only top k terms)
```

This gives 10 basis matrices: `U^(0)_k, U^(1)_k, ..., U^(9)_k`

Each `U^(j)_k` spans the **most important k-dimensional subspace** of digit class j's training data.

---

### Why Left Singular Vectors?

Three mathematical justifications:

1. **Subspace coverage:** If k is chosen appropriately, `range(X^(j)) ≈ range(U^(j)_k)`. At k = min(d, nⱼ), equality holds exactly.

2. **Efficient projection:** The columns of `U^(j)_k` are orthonormal, so the projection of any image `xᵢ` onto this subspace is simply:
   - Expansion coefficients: `U^(j)ᵀ_k xᵢ` (k-vector)
   - Projected image: `U^(j)_k (U^(j)ᵀ_k xᵢ)` (back in ℝ^d)

3. **Best approximation:** `U^(j)_k (U^(j)ᵀ_k xᵢ)` is the **best rank-k least-squares approximation** of `xᵢ` within digit j's subspace.

---

### Phase 2 — Classification: Residual Error Minimization

```
For each test digit yₗ:

  1. For each digit class j = 0, 1, ..., 9:
       a. Project yₗ onto class j's subspace:
            projection = U^(j)_k · (U^(j)ᵀ_k · yₗ)

            ← IMPORTANT: compute U^(j)ᵀ_k · yₗ first (cheap k-vector),
              then multiply by U^(j)_k. Never form d×d matrix U^(j)_k U^(j)ᵀ_k.

       b. Compute the residual (orthogonal complement of projection):
            Eⱼ(yₗ) = ‖yₗ − U^(j)_k (U^(j)ᵀ_k yₗ)‖₂

            Equivalently: Eⱼ(yₗ) = ‖(I − U^(j)_k U^(j)ᵀ_k) yₗ‖₂

  2. Find the class with minimum residual:
            j* = argmin_j  Eⱼ(yₗ)

  3. Decision:
       - If E_{j*} is significantly smaller than all others → classify yₗ as digit j*
       - Otherwise → abstain (give up / no prediction)
```

---

### Key Assumptions

For this algorithm to work well, two conditions must hold:

- **Intra-class compactness:** Each digit class X^(j) is well-captured by a rank-k approximation (small reconstruction error for same-class images)
- **Inter-class separability:** Approximating X^(m) using U^(j)_k with m ≠ j produces **large** residual errors (the bases are discriminative across classes)

---

### Choosing k

- k is swept over a range (tested k = 1 to 20 in this study)
- **k = 17 gave the best result: 96.62%**
- The singular vectors `U^(j)_k` (visualized as "eigen-digit" images) capture progressively finer modes of variation within each digit class

---

### Result

**Overall classification rate at k = 17: 96.62%**

Confusion Matrix for k = 17:

|  | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 772 | 2 | 1 | 3 | 1 | 1 | 2 | 1 | 3 | 0 |
| **1** | 0 | 646 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| **2** | 3 | 6 | 431 | 6 | 0 | 3 | 1 | 2 | 2 | 0 |
| **3** | 1 | 1 | 4 | 401 | 0 | 7 | 0 | 0 | 4 | 0 |
| **4** | 2 | 8 | 1 | 0 | 424 | 1 | 1 | 5 | 0 | 1 |
| **5** | 2 | 0 | 0 | 5 | 2 | 335 | 7 | 1 | 1 | 2 |
| **6** | 6 | 4 | 0 | 0 | 2 | 3 | 399 | 0 | 0 | 0 |
| **7** | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 387 | 0 | 11 |
| **8** | 2 | 9 | 1 | 5 | 1 | 1 | 0 | 0 | 309 | 3 |
| **9** | 0 | 5 | 0 | 1 | 0 | 0 | 0 | 4 | 1 | 388 |

---

## 6. Performance Summary

| Algorithm | Key Parameter | Classification Rate |
|---|---|---|
| Mean (Centroid) Classifier | — | 84.66% |
| k-Nearest Neighbor | k = 1 | 96.99% |
| k-Nearest Neighbor | **k = 3** | **97.07%** ← overall best |
| k-Nearest Neighbor | k = 5 | 96.62% |
| SVD Basis Classifier | k = 17 | 96.62% |

### Key Takeaways

- The **mean classifier** is fast and simple but significantly underperforms (~85%) because a single average image cannot capture the variability within each digit class.
- **k-NN** achieves the best accuracy (~97%) and is conceptually simple, but is computationally expensive: test time scales with the full training set size.
- The **SVD classifier** matches k-NN accuracy (~97% at its best k) and is more compact: only 10 small matrices `U^(j)_k` of size 256×k need to be stored and used at test time, making it more scalable to large datasets and high-dimensional images.
- The SVD method also has a clean mathematical interpretation — it classifies based on **subspace geometry** rather than point distances, which generalizes better as dimensionality increases.

---

*Source: MAT 167 Lecture 21 — Naoki Saito, UC Davis, May 17, 2017*
