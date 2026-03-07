import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# -----------------------------
# MODIFY THESE PATHS
# -----------------------------
DATASET1_PATH = "D:\\Coding\\SEM-8-NEW\\OIL-SPILL\\dataset_1"
DATASET2_PATH = "D:\\Coding\\SEM-8-NEW\\OIL-SPILL\\dataset_2"

# -----------------------------
# FUNCTION TO ANALYZE DATASET
# -----------------------------
def analyze_dataset(dataset_path, dataset_name):
    # Search recursively for image files
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
        image_paths.extend(glob(os.path.join(dataset_path, "**", ext), recursive=True))
    
    # Limit to first 100 images to avoid memory issues
    image_paths = image_paths[:100]
    
    heights = []
    widths = []
    rgb_means = []
    brightness_vals = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w, c = img.shape
        heights.append(h)
        widths.append(w)
        
        rgb_means.append(np.mean(img, axis=(0,1)))
        brightness_vals.append(np.mean(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)))

    heights = np.array(heights)
    widths = np.array(widths)
    rgb_means = np.array(rgb_means)
    brightness_vals = np.array(brightness_vals)

    print(f"\n===== {dataset_name} =====")
    print("Total Images:", len(image_paths))
    
    if len(heights) > 0:
        print("Resolution Stats:")
        print("  Mean Height:", np.mean(heights))
        print("  Mean Width:", np.mean(widths))
        print("  Min Height:", np.min(heights))
        print("  Max Height:", np.max(heights))
        
        print("\nRGB Mean (Average across dataset):", np.mean(rgb_means, axis=0))
        print("Average Brightness:", np.mean(brightness_vals))
        
        return heights, widths, rgb_means, brightness_vals
    else:
        print("No valid images found!")
        return np.array([]), np.array([]), np.array([]), np.array([])


# -----------------------------
# RUN ANALYSIS
# -----------------------------
h1, w1, rgb1, bright1 = analyze_dataset(DATASET1_PATH, "Zenodo Dataset")
h2, w2, rgb2, bright2 = analyze_dataset(DATASET2_PATH, "Kaggle Dataset")


# -----------------------------
# PLOT COMPARISON
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(bright1, bins=30, alpha=0.6, label="Zenodo")
plt.hist(bright2, bins=30, alpha=0.6, label="Kaggle")
plt.title("Brightness Distribution")
plt.legend()

plt.subplot(1,2,2)
plt.hist([h1, h2], bins=30, label=["Zenodo", "Kaggle"])
plt.title("Height Distribution")
plt.legend()

plt.tight_layout()
plt.show()