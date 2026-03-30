"""
Step 3 - k-Nearest Neighbor (k-NN) Classifier
===============================================
For each test image, find the k closest training images by Euclidean distance.
The majority class among those k neighbors becomes the prediction.
Tested for k = 1, 3, 5.
"""

import scipy.io as sio
import numpy as np
from collections import Counter
import time

# -- Load data -----------------------------------------------------------------
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
train_patterns = mat["train_patterns"]   # 256 x 4649
test_patterns  = mat["test_patterns"]    # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649
test_labels    = mat["test_labels"]      # 10  x 4649

train_class = np.argmax(train_labels, axis=0)  # (4649,)
test_class  = np.argmax(test_labels,  axis=0)  # (4649,)

n_train = train_patterns.shape[1]
n_test  = test_patterns.shape[1]

# -- Precompute full distance matrix (4649 x 4649) ----------------------------
# ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b   (fast, vectorized)
print("Computing full distance matrix (4649 x 4649)...")
t0 = time.time()

train_sq = np.sum(train_patterns ** 2, axis=0)   # (4649,)
test_sq  = np.sum(test_patterns  ** 2, axis=0)   # (4649,)
cross    = test_patterns.T @ train_patterns       # (4649 x 4649)

dist_sq = test_sq[:, np.newaxis] + train_sq[np.newaxis, :] - 2 * cross
dist_sq = np.maximum(dist_sq, 0)                 # clip tiny negatives from float errors
dist    = np.sqrt(dist_sq)                        # (4649 x 4649)

print(f"Done in {time.time()-t0:.2f}s\n")

# -- Run k-NN for k = 1, 3, 5 -------------------------------------------------
results = {}

for k in [1, 3, 5]:
    print(f"Running k-NN with k={k}...")
    t1 = time.time()

    predicted = np.zeros(n_test, dtype=int)
    for i in range(n_test):
        neighbor_idx   = np.argpartition(dist[i], k)[:k]  # k nearest indices
        neighbor_labels = train_class[neighbor_idx]
        predicted[i]   = Counter(neighbor_labels).most_common(1)[0][0]

    elapsed  = time.time() - t1
    correct  = np.sum(predicted == test_class)
    accuracy = correct / n_test * 100
    results[k] = {"acc": accuracy, "pred": predicted.copy()}

    print(f"  Accuracy: {accuracy:.2f}%  |  Time: {elapsed:.2f}s")

# -- Summary -------------------------------------------------------------------
print()
print("=" * 50)
print("k-NN RESULTS SUMMARY")
print("=" * 50)
expected = {1: 96.99, 3: 97.07, 5: 96.62}
for k in [1, 3, 5]:
    flag = "<-- best" if k == 3 else ""
    print(f"  k={k}: {results[k]['acc']:.2f}%   (paper: {expected[k]}%)  {flag}")

# -- Confusion matrix for best k=3 --------------------------------------------
best_pred = results[3]["pred"]
conf = np.zeros((10, 10), dtype=int)
for true, pred in zip(test_class, best_pred):
    conf[true, pred] += 1

print()
print("Confusion Matrix for k=3 (rows = true, cols = predicted):")
header = "     " + "  ".join(f"{i:4d}" for i in range(10))
print(header)
print("     " + "-" * 46)
for i in range(10):
    row = "  ".join(f"{conf[i,j]:4d}" for j in range(10))
    print(f"  {i} | {row}")

print()
print("Per-class accuracy (k=3):")
for d in range(10):
    total_d   = conf[d].sum()
    correct_d = conf[d, d]
    print(f"  Digit {d}: {correct_d}/{total_d}  ({correct_d/total_d*100:.1f}%)")

# -- Save results --------------------------------------------------------------
np.save("../results/results_step3_conf.npy", conf)
np.save("../results/results_step3_accs.npy", np.array([results[k]["acc"] for k in [1,3,5]]))
print("\nResults saved.")
