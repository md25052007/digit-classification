"""
predict.py - Classify your own handwritten digit image
=======================================================
Usage:
    python predict.py <image_path>

Examples:
    python predict.py my_digit.png
    python predict.py my_digit.jpg
    python predict.py scan.bmp
    python predict.py photo.tiff

The script will:
  1. Load your image (any common format) and convert it to 16x16 grayscale
  2. Normalize pixel values to [-1, 1]
  3. Run all 3 classifiers (Mean, k-NN k=3, SVD k=17) trained on USPS data
  4. Print what each classifier thinks the digit is
  5. Show a visual of how your image looks after preprocessing

Supported formats: PNG, JPG/JPEG, BMP, TIFF, GIF, WEBP, ICO, and more.
RGBA (transparent PNG) and CMYK are handled automatically.
"""

import sys
import os
import scipy.io as sio
import numpy as np
from PIL import Image

# ── Argument check ─────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    print("Example: python predict.py my_digit.png")
    print("Supported: PNG, JPG, BMP, TIFF, GIF, WEBP, ICO, ...")
    sys.exit(1)

image_path = sys.argv[1]

if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    sys.exit(1)

# ── Load & preprocess image ────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif",
                         ".gif", ".webp", ".ico", ".ppm", ".pgm", ".pbm"}
ext = os.path.splitext(image_path)[1].lower()
if ext not in SUPPORTED_EXTENSIONS:
    print(f"Warning: '{ext}' may not be supported. Trying anyway...")
    print(f"Known supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

print(f"\nLoading image: {image_path}")
try:
    raw = Image.open(image_path)
except Exception as e:
    print(f"Error: Could not open image — {e}")
    sys.exit(1)

print(f"  Original mode : {raw.mode}  |  Original size: {raw.size[0]}x{raw.size[1]}px")

# Handle all colour modes robustly before converting to grayscale
if raw.mode == "RGBA":
    # Transparent PNG: composite onto white background so digit stays visible
    background = Image.new("RGB", raw.size, (255, 255, 255))
    background.paste(raw, mask=raw.split()[3])   # use alpha channel as mask
    raw = background
elif raw.mode == "CMYK":
    raw = raw.convert("RGB")    # PIL can convert CMYK->RGB cleanly
elif raw.mode == "P":          # palette/indexed-colour GIF etc.
    raw = raw.convert("RGB")
elif raw.mode == "1":          # 1-bit black & white
    raw = raw.convert("RGB")

img = raw.convert("L")                             # now safely to grayscale

# -- Auto-crop to bounding box and scale to USPS style --
# Skip auto-crop if the image is already 16x16 (meaning it comes from the dataset)
if img.size != (16, 16):
    # Invert temporarily to find the bounding box of the dark ink
    inverted_for_bbox = Image.eval(img, lambda p: 255 - p) if np.array(img).mean() > 127 else img
    bbox = inverted_for_bbox.getbbox()

    if bbox:
        # Crop to the tightly drawn digit ink
        img = img.crop(bbox)
        
        # Pad slightly to make it roughly square and centered
        w, h = img.size
        side = int(max(w, h) * 1.2)  # add ~10% padding on each side
        new_img = Image.new("L", (side, side), color=255 if np.array(img).mean() > 127 else 0)
        new_img.paste(img, ((side - w) // 2, (side - h) // 2))
        img = new_img

# Resize the tightly cropped and padded digit to 16x16
img = img.resize((16, 16), Image.LANCZOS)
img_array = np.array(img, dtype=float)              # shape (16, 16)

# Normalize to [-1, 1]  (USPS dataset range)
# Input pixels are 0-255, we map: 255 -> -1, 0 -> +1  (white=background=-1, dark digit=+1)
# OR if image has dark background: 0 -> -1, 255 -> +1
# We auto-detect: if background is light (mean > 127), invert
if img_array.mean() > 127:
    # Light background (white paper), dark digit — invert so digit = high value
    img_array = 255 - img_array

# Scale to [-1, 1]
img_normalized = (img_array / 127.5) - 1.0         # [0,255] -> [-1, 1]
y = img_normalized.flatten()                         # shape (256,) — our input vector

# ── Show ASCII preview of the processed image ──────────────────────────────────
print("\nPreprocessed image (16x16, bright = digit stroke):")
print("  +" + "-" * 32 + "+")
for row in img_normalized:
    line = ""
    for px in row:
        if px > 0.5:
            line += "##"
        elif px > 0.0:
            line += "++"
        elif px > -0.5:
            line += ".."
        else:
            line += "  "
    print(f"  |{line}|")
print("  +" + "-" * 32 + "+")

# ── Load training data ─────────────────────────────────────────────────────────
print("\nLoading USPS training data...")
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
train_patterns = mat["train_patterns"]   # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649
train_class    = np.argmax(train_labels, axis=0)   # (4649,)

print("Training data loaded.\n")
print("=" * 50)
print("CLASSIFICATION RESULTS")
print("=" * 50)

# ── Classifier 1: Mean Classifier ─────────────────────────────────────────────
means = np.zeros((256, 10))
for digit in range(10):
    idx = np.where(train_class == digit)[0]
    means[:, digit] = train_patterns[:, idx].mean(axis=1)

distances_mean = np.linalg.norm(means - y[:, np.newaxis], axis=0)   # (10,)
pred_mean      = np.argmin(distances_mean)
conf_mean      = 1 - (distances_mean[pred_mean] / distances_mean.sum())

print(f"\n[1] Mean Classifier        --> Digit: {pred_mean}")
print(f"    Distances to each class:")
for d in range(10):
    bar = "#" * int(20 * (1 - distances_mean[d]/distances_mean.max()))
    marker = " <-- predicted" if d == pred_mean else ""
    print(f"      {d}: {distances_mean[d]:6.2f}  {bar}{marker}")

# ── Classifier 2: k-NN (k=3) ──────────────────────────────────────────────────
from collections import Counter

diffs    = train_patterns - y[:, np.newaxis]          # 256 x 4649
dists_knn = np.linalg.norm(diffs, axis=0)             # (4649,)
k        = 3
nn_idx   = np.argpartition(dists_knn, k)[:k]         # indices of 3 nearest
nn_labels = train_class[nn_idx]
vote      = Counter(nn_labels)
pred_knn  = vote.most_common(1)[0][0]

print(f"\n[2] k-NN (k=3)             --> Digit: {pred_knn}")
print(f"    3 nearest neighbors: digits {nn_labels.tolist()} at distances {dists_knn[nn_idx].round(2).tolist()}")
print(f"    Votes: {dict(vote)}")

# ── Classifier 3: SVD Basis (k=17) ────────────────────────────────────────────
k_svd   = 17
y_sq    = np.dot(y, y)
residuals_svd = np.zeros(10)

for digit in range(10):
    idx    = np.where(train_class == digit)[0]
    X_j    = train_patterns[:, idx]
    U, _, _ = np.linalg.svd(X_j, full_matrices=False)
    U_k    = U[:, :k_svd]
    proj_sq = np.dot(U_k.T @ y, U_k.T @ y)
    residuals_svd[digit] = y_sq - proj_sq

residuals_svd = np.maximum(residuals_svd, 0)
pred_svd      = np.argmin(residuals_svd)

print(f"\n[3] SVD Classifier (k=17)  --> Digit: {pred_svd}")
print(f"    Residuals per class (lower = better fit):")
for d in range(10):
    bar    = "#" * int(20 * (1 - residuals_svd[d]/residuals_svd.max()))
    marker = " <-- predicted" if d == pred_svd else ""
    print(f"      {d}: {residuals_svd[d]:8.2f}  {bar}{marker}")

# ── Final verdict ──────────────────────────────────────────────────────────────
print()
print("=" * 50)
print("FINAL VERDICT")
print("=" * 50)
votes_final = Counter([pred_mean, pred_knn, pred_svd])
final       = votes_final.most_common(1)[0][0]

print(f"  Mean Classifier : {pred_mean}")
print(f"  k-NN (k=3)      : {pred_knn}")
print(f"  SVD (k=17)      : {pred_svd}")
print(f"\n  --> Majority vote: DIGIT  {final}  <--")

if pred_mean == pred_knn == pred_svd:
    print("  (All 3 classifiers agree!)")
elif votes_final[final] == 2:
    disagree = [p for p in [pred_mean, pred_knn, pred_svd] if p != final]
    print(f"  (2 out of 3 agree. One classifier said {disagree[0]})")
else:
    print("  (All 3 classifiers disagree — the image may be ambiguous or unclear)")
