import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# DATASET PATHS
# -----------------------------
DATASET_1 = r"D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_3"
DATASET_2 = r"D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_4"


# -----------------------------
# IMAGE EXTENSIONS
# -----------------------------
VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -----------------------------
# RECURSIVE IMAGE COLLECTOR
# -----------------------------
def collect_images(dataset_path):

    image_paths = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(VALID_EXT):
                image_paths.append(os.path.join(root, file))

    return image_paths


# -----------------------------
# DATASET ANALYSIS
# -----------------------------
def analyze_dataset(dataset_path, name):

    image_paths = collect_images(dataset_path)

    heights = []
    widths = []
    channels = []

    rgb_means = []
    rgb_stds = []
    brightness_vals = []

    for path in image_paths:

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            continue

        if len(img.shape) == 2:
            c = 1
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            c = img.shape[2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]

        heights.append(h)
        widths.append(w)
        channels.append(c)

        rgb_means.append(np.mean(img_rgb, axis=(0,1)))
        rgb_stds.append(np.std(img_rgb, axis=(0,1)))

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        brightness_vals.append(np.mean(gray))

    heights = np.array(heights)
    widths = np.array(widths)
    channels = np.array(channels)
    rgb_means = np.array(rgb_means)
    rgb_stds = np.array(rgb_stds)
    brightness_vals = np.array(brightness_vals)

    print("\n==============================")
    print("DATASET:", name)
    print("==============================")

    print("Total images found:", len(image_paths))
    print("Unique channel types:", np.unique(channels))

    print("\nResolution Stats")
    print("Mean height:", heights.mean())
    print("Mean width:", widths.mean())
    print("Min height:", heights.min())
    print("Max height:", heights.max())

    print("\nRGB Mean (dataset avg):", rgb_means.mean(axis=0))
    print("RGB Std  (dataset avg):", rgb_stds.mean(axis=0))

    print("\nAverage Brightness:", brightness_vals.mean())

    return brightness_vals, heights


# -----------------------------
# RUN ANALYSIS
# -----------------------------
b1, h1 = analyze_dataset(DATASET_1, "Dataset 1 (Zenodo)")
b2, h2 = analyze_dataset(DATASET_2, "Dataset 2 (Kaggle)")


# -----------------------------
# VISUAL DOMAIN COMPARISON
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(b1, bins=40, alpha=0.6, label="Dataset 1")
plt.hist(b2, bins=40, alpha=0.6, label="Dataset 2")
plt.title("Brightness Distribution")
plt.legend()

plt.subplot(1,2,2)
plt.hist(h1, bins=40, alpha=0.6, label="Dataset 1")
plt.hist(h2, bins=40, alpha=0.6, label="Dataset 2")
plt.title("Resolution Height Distribution")
plt.legend()

plt.tight_layout()
plt.show()