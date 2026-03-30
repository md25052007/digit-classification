"""
Step 4 - SVD Basis Classifier
===============================
For each digit class, compute a rank-k SVD of the training images.
The left singular vectors U_k form an orthonormal basis for that class.
Classify a test image by finding which class subspace it projects onto
with the SMALLEST residual error.
Sweep k = 1 to 20, report best result.
"""

import scipy.io as sio
import numpy as np
import time

# -- Load data -----------------------------------------------------------------
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
train_patterns = mat["train_patterns"]   # 256 x 4649
test_patterns  = mat["test_patterns"]    # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649
test_labels    = mat["test_labels"]      # 10  x 4649

train_class = np.argmax(train_labels, axis=0)   # (4649,)
test_class  = np.argmax(test_labels,  axis=0)   # (4649,)

n_test = test_patterns.shape[1]

# -- TRAINING: compute rank-k SVD basis for each digit class ------------------
# U_bases[j] will hold the full left singular matrix for class j
# We compute full SVD once and slice at classification time (faster than re-running svd per k)

print("Computing SVD for each digit class...")
t0 = time.time()

U_full = []   # list of 10 matrices, each shape (256 x n_j)

for digit in range(10):
    idx        = np.where(train_class == digit)[0]
    X_j        = train_patterns[:, idx]          # 256 x n_j
    U, S, Vt   = np.linalg.svd(X_j, full_matrices=False)  # U: 256 x min(256, n_j)
    U_full.append(U)
    print(f"  Digit {digit}: {X_j.shape[1]} training images, SVD basis shape {U.shape}")

print(f"SVD done in {time.time()-t0:.2f}s\n")

# -- CLASSIFICATION: sweep k = 1 to 20 ----------------------------------------
print("Sweeping k from 1 to 20...")
print("-" * 50)

best_k   = -1
best_acc = -1
accs     = {}

for k in range(1, 21):
    # For each test image, compute residual error against each class subspace
    # residual_j(y) = || y - U_j_k @ U_j_k.T @ y ||_2
    #               = || (I - U_j_k @ U_j_k.T) y ||_2
    # But: || y - U U.T y ||^2 = ||y||^2 - ||U.T y||^2   (since U is orthonormal)
    # So we minimize ||U_j_k.T y||^2 to find MAX projection → MIN residual

    # Precompute ||y||^2 once (same for all classes)
    y_sq = np.sum(test_patterns ** 2, axis=0)   # (4649,)

    residuals = np.zeros((n_test, 10))

    for digit in range(10):
        U_k      = U_full[digit][:, :k]              # 256 x k
        proj_sq  = np.sum((U_k.T @ test_patterns)**2, axis=0)   # (4649,)  = ||U.T y||^2
        residuals[:, digit] = y_sq - proj_sq         # ||y||^2 - ||U.T y||^2

    # Clip tiny negatives from float errors
    residuals = np.maximum(residuals, 0)

    predicted = np.argmin(residuals, axis=1)   # class with smallest residual
    correct   = np.sum(predicted == test_class)
    accuracy  = correct / n_test * 100
    accs[k]   = accuracy

    marker = " <-- best so far" if accuracy > best_acc else ""
    print(f"  k={k:2d}: {accuracy:.2f}%{marker}")

    if accuracy > best_acc:
        best_acc  = accuracy
        best_k    = k
        best_pred = predicted.copy()

# -- Summary -------------------------------------------------------------------
print()
print("=" * 50)
print("SVD CLASSIFIER RESULTS")
print("=" * 50)
print(f"Best k        : {best_k}")
print(f"Best accuracy : {best_acc:.2f}%")
print(f"Expected (paper): 96.62% at k=17")
print()

# -- Confusion matrix for best k -----------------------------------------------
conf = np.zeros((10, 10), dtype=int)
for true, pred in zip(test_class, best_pred):
    conf[true, pred] += 1

print(f"Confusion Matrix for k={best_k} (rows = true, cols = predicted):")
header = "     " + "  ".join(f"{i:4d}" for i in range(10))
print(header)
print("     " + "-" * 46)
for i in range(10):
    row = "  ".join(f"{conf[i,j]:4d}" for j in range(10))
    print(f"  {i} | {row}")

print()
print(f"Per-class accuracy (k={best_k}):")
for d in range(10):
    total_d   = conf[d].sum()
    correct_d = conf[d, d]
    print(f"  Digit {d}: {correct_d}/{total_d}  ({correct_d/total_d*100:.1f}%)")

# -- Save results --------------------------------------------------------------
np.save("../results/results_step4_conf.npy", conf)
np.save("../results/results_step4_accs.npy", np.array(list(accs.values())))
np.save("../results/results_step4_best_k.npy", np.array([best_k]))
print("\nResults saved.")
