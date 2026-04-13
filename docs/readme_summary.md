# Final Summary — Handwritten Digit Classification

**Course:** IT256-Applied Linear Algebra — Swaroop Georgy Zachariah
**Dataset:** USPS Handwritten Digits (9,298 images, 16×16 px)

---

## The Problem

Given a 16×16 grayscale image of a handwritten digit (0–9), predict which digit it is.  
Each image is flattened to a 256-dimensional vector. We have 4,649 training images and 4,649 test images.

---

## Three Algorithms, One Dataset

### Algorithm 1 — Mean (Centroid) Classifier
> *"What does the average digit look like?"*

- **Training:** Compute the mean image for each of the 10 digit classes → 10 vectors
- **Classification:** Assign the class whose mean is closest (Euclidean distance)
- **Accuracy: 84.66%**

### Algorithm 2 — k-Nearest Neighbor (k-NN)
> *"What do the most similar training digits look like?"*

- **Training:** None — store all 4,649 training images
- **Classification:** Find the k closest training images, take majority vote
- **Best at k=3 → Accuracy: 97.07%**

### Algorithm 3 — SVD Basis Classifier
> *"Does this digit fit into the geometric subspace of this class?"*

- **Training:** Compute a rank-k SVD for each digit class → 10 small basis matrices
- **Classification:** Project the test image onto each subspace, pick smallest residual
- **Best at k=17 → Accuracy: 96.62%**

---

## Results at a Glance

| Algorithm | Key Parameter | Accuracy | Match Paper? |
|---|---|---|---|
| Mean Classifier | — | 84.66% | ✅ |
| k-NN | k=1 | 96.99% | ✅ |
| k-NN | **k=3** | **97.07%** ← overall best | ✅ |
| k-NN | k=5 | 96.62% | ✅ |
| SVD Basis | k=17 | 96.62% | ✅ |

All results reproduced exactly from the original 2017 UC Davis lecture.

---

## Accuracy vs k — SVD Classifier

The SVD accuracy rises steadily as more basis vectors capture digit variation, peaks at k=17, then drops as the bases become noisy:

```
k=1:  84.66%  |████████████████░░░░|
k=5:  94.24%  |███████████████████░|
k=10: 96.45%  |████████████████████|
k=17: 96.62%  |████████████████████| <-- best
k=20: 95.98%  |███████████████████░|
```

---

## Per-Class Accuracy Comparison

| Digit | Mean | k-NN (k=3) | SVD (k=17) |
|---|---|---|---|
| 0 | 83.5% | 99.0% | 98.2% |
| 1 | 99.5% | 99.4% | **99.8%** |
| 2 | 79.7% | 95.8% | 94.9% |
| 3 | 88.0% | 96.2% | 95.9% |
| 4 | 81.9% | 94.8% | 95.7% |
| 5 | 76.3% | 93.5% | 94.4% |
| 6 | 85.5% | 97.8% | 96.4% |
| 7 | 87.3% | 98.0% | 96.3% |
| 8 | 76.4% | 94.9% | 93.4% |
| 9 | 78.7% | 97.7% | 97.2% |

---

## What Each Method Needs (Memory & Speed)

| Property | Mean | k-NN (k=3) | SVD (k=17) |
|---|---|---|---|
| Training time | < 1s | None | 0.69s |
| Test time (all 4649) | < 1s | ~0.5s | ~2s |
| Storage | 256 × 10 floats | 256 × 4649 floats | 256 × 17 × 10 floats |
| Storage (numbers) | **2,560** | 1,190,144 | **43,520** |

- k-NN stores **464× more** than the mean classifier
- SVD stores only **17× more** than the mean classifier but achieves k-NN-level accuracy

---

## Key Takeaways

1. **The mean classifier is fast but weak** — a single average image can't represent the diversity of handwriting styles within a class. Digit `5` at only 76.3% is the clearest example.

2. **k-NN is the most accurate** — by using actual training examples as references it makes fine-grained comparisons. But it scales badly: for 1 million training images, you'd compute 1M distances per test image.

3. **SVD matches k-NN accuracy while being far more scalable** — it compresses each 400–700 image class into just 17 basis vectors. Projection is a simple matrix multiply. This approach generalizes to very high-dimensional data far better than k-NN.

4. **The SVD k=1 case == mean classifier (84.66%)** — not a coincidence. The first left singular vector points in the direction of the data mean. Adding more singular vectors captures more shape variation.

5. **Linear algebra is enough** — no neural networks, no GPUs, no deep learning. Pure SVD, distance metrics, and matrix operations give ~97% accuracy on handwritten digit classification.

---

## File Index

| File | Description |
|---|---|
| `step1_load_data.py` | Load and inspect the dataset |
| `step2_mean_classifier.py` | Mean classifier |
| `step3_knn_classifier.py` | k-NN classifier (k=1,3,5) |
| `step4_svd_classifier.py` | SVD basis classifier (k=1..20) |
| `readme_step1.md` | Explanation of Step 1 |
| `readme_step2.md` | Explanation of Step 2 |
| `readme_step3.md` | Explanation of Step 3 |
| `readme_step4.md` | Explanation of Step 4 |
| `readme_summary.md` | This file |
| `usps_resampled/usps_resampled.mat` | Dataset (MATLAB format) |

---

## How to Run (in order)

```bash
python step1_load_data.py
python step2_mean_classifier.py
python step3_knn_classifier.py
python step4_svd_classifier.py
```

Each script is self-contained and runs independently.

---

*Source: MAT 167 Lecture 21 — Naoki Saito, UC Davis, May 17, 2017*
