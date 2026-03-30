# MAT 167: Applied Linear Algebra - Digit Classification

This repository contains Python implementations of three mathematical algorithms for handwritten digit classification, relying entirely on **Linear Algebra** rather than deep learning or modern neural networks.

This is based on the MAT 167 lecture from UC Davis (Naoki Saito), reproducing the exact classification accuracies from the original 2017 study.

---

## 📊 The Algorithms
We classify 16x16 grayscale handwritten digits from the USPS dataset using three geometric models:

1. **Mean (Centroid) Classifier (84.66%)**: Computes the average "mean vector" for each digit class and classifies tests based on minimum Euclidean distance.
2. **k-Nearest Neighbor (97.07%)**: Computes the $L_2$ Euclidean distance to all 4,649 training images and takes a majority vote among the top $k=3$ neighbors.
3. **SVD Basis Classifier (96.62%)**: Computes a low-dimensional orthonormal subspace for each digit class using Singular Value Decomposition (SVD), then classifies tests by finding which class subspace yields the lowest projection residual error (at $k=17$ rank).

---

## 📁 Repository Structure

```text
├── scripts/
│   ├── step1_load_data.py          # Script to load and verify matrix shapes
│   ├── step2_mean_classifier.py    # Implements Centroid distance classifier
│   ├── step3_knn_classifier.py     # Implements k-NN classifier
│   ├── step4_svd_classifier.py     # Implements SVD subspace classifier
│   ├── predict.py                  # CLI tool to test YOUR OWN drawn digit images
│   ├── visualize_dataset.py        # Generates a PNG preview of the dataset
│   └── generate_test_images.py     # Extracts 10 authentic test images
├── docs/                           # Detailed markdown explanations for each algorithm
├── test_images/                    # Authentic 16x16 reference USPS test digits
├── usps_resampled/                 # Raw USPS datasets (.mat format)
└── results/                        # Cached .npy matrices of classification results
```

---

## ⚙️ Requirements & Installation

You need Python 3 and a few standard numerical libraries installed.

```bash
# Clone the repository
git clone https://github.com/md25052007/digit-classification.git
cd digit-classification

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

Navigate to the `scripts` folder before running anything:
```bash
cd scripts
```

**Evaluate the Models on the Test Set:**
Run any of the step scripts to automatically train on the training set and classify the 4,649 item testing set.
```bash
python step2_mean_classifier.py
python step3_knn_classifier.py
python step4_svd_classifier.py
```

**Generate Visualizations:**
```bash
python visualize_dataset.py       # Creates dataset_samples.png
python generate_test_images.py    # Extracts 10 images into /test_images
```

**Classify an Arbitrary Image:**
You can pass your own image to test against all three models simultaneously.
```bash
python predict.py ../test_images/test_digit_5.png
```
*(Note: Because the algorithms rely on precise Euclidean distances rather than translation-invariant CNN convolutions, hand-drawn predictions will only succeed if the digit is drawn thick and squarely centered, identically resembling the USPS scanning geometry).*
