"""
generate_test_images.py - Generate 10 test images (0-9) from the testing dataset
================================================================================
This script extracts one random example of each digit (0-9) from the 
USPS test dataset and saves them as 0.png, 1.png, ..., 9.png.
Since they are taken directly from the test set, predict.py should 
have no problem identifying them!
"""

import os
import scipy.io as sio
import numpy as np
from PIL import Image

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading USPS test dataset...")
mat = sio.loadmat("../usps_resampled/usps_resampled.mat")
test_patterns = mat["test_patterns"]   # 256 x 4649
test_labels   = mat["test_labels"]     # 10  x 4649

test_class = np.argmax(test_labels, axis=0)  # (4649,)

# Ensure output directory exists (we'll save them to a new folder)
output_dir = "../test_images"
os.makedirs(output_dir, exist_ok=True)

for digit in range(10):
    # Find all test images for this digit
    idx = np.where(test_class == digit)[0]
    
    # Pick one random sample
    np.random.seed(123 + digit) # Keep it reproducible
    chosen_idx = np.random.choice(idx)
    
    # Extract the 256-dimensional vector
    vec = test_patterns[:, chosen_idx]
    
    # Reshape (already C-style compatible)
    img_array = vec.reshape((16, 16))
    
    # Convert [-1, 1] range to [0, 255]
    img_array = ((1 - img_array) / 2.0) * 255.0
    
    # Create and save PIL image
    digit_img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    
    # Save the authentic 16x16 image size
    output_file = os.path.join(output_dir, f"test_digit_{digit}.png")
    digit_img.save(output_file)
    print(f"Saved: {output_file}")

print("\nDone! Generated 10 images in the 'test_images' folder.")
