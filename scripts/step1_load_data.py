"""
Step 1 — Load & Inspect the USPS Handwritten Digit Dataset
===========================================================
Loads usps_resampled.mat and prints key stats about the data.
"""

import scipy.io as sio
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")

train_patterns = mat["train_patterns"]   # 256 x 4649
test_patterns  = mat["test_patterns"]    # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649
test_labels    = mat["test_labels"]      # 10  x 4649

# ── Basic stats ───────────────────────────────────────────────────────────────
print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)
print(f"train_patterns shape : {train_patterns.shape}   (256 pixel dims x 4649 images)")
print(f"test_patterns  shape : {test_patterns.shape}")
print(f"train_labels   shape : {train_labels.shape}    (10 classes x 4649 images)")
print(f"test_labels    shape : {test_labels.shape}")
print()
print(f"Pixel value range    : [{train_patterns.min():.2f}, {train_patterns.max():.2f}]  (should be ~[-1, 1])")
print()

# ── Class distribution ────────────────────────────────────────────────────────
# label encoding: row i+1 == +1 means digit i
train_class = np.argmax(train_labels, axis=0)   # shape (4649,)
test_class  = np.argmax(test_labels,  axis=0)

print("Training samples per digit class:")
for d in range(10):
    count = np.sum(train_class == d)
    print(f"  Digit {d}: {count} samples")

print()
print("Test samples per digit class:")
for d in range(10):
    count = np.sum(test_class == d)
    print(f"  Digit {d}: {count} samples")

print()
print("Step 1 complete. Data is ready for classification.")
