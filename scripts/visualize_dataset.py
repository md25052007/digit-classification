"""
visualize_dataset.py - Generate an image showing sample digits from the dataset
================================================================================
This script extracts random examples of each digit (0-9) from the USPS dataset 
and saves them into a single image file (dataset_samples.png) so you can see 
what the actual USPS handwritten digits look like.
"""

import scipy.io as sio
import numpy as np
from PIL import Image

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading USPS dataset...")
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
train_patterns = mat["train_patterns"]   # 256 x 4649
train_labels   = mat["train_labels"]     # 10  x 4649

train_class = np.argmax(train_labels, axis=0)  # (4649,)

# ── Image Configuration ───────────────────────────────────────────────────────
rows = 10        # One row per digit (0-9)
cols = 20        # 20 samples per digit
cell_size = 16   # Each digit is 16x16
scale = 3        # Scale up the image by 3x so it's easier to look at

grid_w = cols * cell_size
grid_h = rows * cell_size

# Create a blank white image for the grid
grid_img = Image.new('L', (grid_w, grid_h), color=255)

print(f"Extracting {cols} samples for each digit...")

for digit in range(10):
    # Find all training images for this digit
    idx = np.where(train_class == digit)[0]
    
    # Pick random samples (or just the first few)
    np.random.seed(42 + digit) # Keep it reproducible
    chosen_idx = np.random.choice(idx, cols, replace=False)
    
    for c, i in enumerate(chosen_idx):
        # Extract the 256-dimensional vector
        vec = train_patterns[:, i]
        
        # The dataset vectors were flattened row-by-row (C-style). 
        # Python's reshape is already C-style, so we DO NOT need to transpose it.
        # (The assignment methodology says to transpose it because MATLAB's reshape 
        # is column-major, which would sideways the image. In Python, it's correct natively.)
        img_array = vec.reshape((16, 16))
        
        # Convert [-1, 1] range to [0, 255]
        # -1 represents background/white, +1 represents digit/black
        # So we map +1 -> 0 (black ink) and -1 -> 255 (white paper)
        img_array = ((1 - img_array) / 2.0) * 255.0
        
        # Create a PIL image for this single digit
        digit_img = Image.fromarray(img_array.astype(np.uint8), mode='L')
        
        # Paste it into the grid
        x = c * cell_size
        y = digit * cell_size
        grid_img.paste(digit_img, (x, y))

# Scale up the image so it's not tiny
final_size = (grid_w * scale, grid_h * scale)
grid_img = grid_img.resize(final_size, Image.NEAREST)

output_file = "../dataset_samples.png"
grid_img.save(output_file)

print(f"\nDone! Open '{output_file}' on your Desktop to see the images.")
print("Each row is a different digit (0-9).")
print(f"Showing {cols} examples per digit.")
