"""
Step 2 - Mean (Centroid) Classifier
=====================================
For each digit class, compute the mean image from training data.
Classify each test image by finding the nearest mean (Euclidean distance).
"""

import scipy.io as sio
import numpy as np

# -- Load data -----------------------------------------------------------------
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
train_patterns = mat["train_patterns"]   # 256 x 4649
test_patterns  = mat["test_patterns"]    # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649
test_labels    = mat["test_labels"]      # 10  x 4649

train_class = np.argmax(train_labels, axis=0)  # (4649,) true digit per column
test_class  = np.argmax(test_labels,  axis=0)

# -- TRAINING: compute one mean image per digit class -------------------------
means = np.zeros((256, 10))   # each column = mean image for that digit

for digit in range(10):
    idx = np.where(train_class == digit)[0]
    means[:, digit] = train_patterns[:, idx].mean(axis=1)

print("Mean images computed for digits 0-9.")
print(f"  means matrix shape: {means.shape}  (256 x 10)\n")

# -- CLASSIFICATION: nearest mean by Euclidean distance -----------------------
diff      = test_patterns[:, :, np.newaxis] - means[:, np.newaxis, :]  # 256 x 4649 x 10
distances = np.linalg.norm(diff, axis=0)                                # 4649 x 10
predicted = np.argmin(distances, axis=1)                                # (4649,)

# -- Results -------------------------------------------------------------------
correct  = np.sum(predicted == test_class)
total    = len(test_class)
accuracy = correct / total * 100

print("=" * 50)
print("MEAN CLASSIFIER RESULTS")
print("=" * 50)
print(f"Correct predictions : {correct} / {total}")
print(f"Classification rate : {accuracy:.2f}%")
print(f"Expected (from paper): 84.66%")
print()

# -- Confusion matrix ----------------------------------------------------------
conf = np.zeros((10, 10), dtype=int)
for true, pred in zip(test_class, predicted):
    conf[true, pred] += 1

print("Confusion Matrix (rows = true digit, cols = predicted digit):")
header = "     " + "  ".join(f"{i:4d}" for i in range(10))
print(header)
print("     " + "-" * 46)
for i in range(10):
    row = "  ".join(f"{conf[i,j]:4d}" for j in range(10))
    print(f"  {i} | {row}")

print()
print("Per-class accuracy:")
for d in range(10):
    total_d   = conf[d].sum()
    correct_d = conf[d, d]
    print(f"  Digit {d}: {correct_d}/{total_d}  ({correct_d/total_d*100:.1f}%)")

# -- Save confusion matrix for README ------------------------------------------
np.save("../results/results_step2_conf.npy", conf)
np.save("../results/results_step2_acc.npy", np.array([accuracy]))
print("\nResults saved.")
