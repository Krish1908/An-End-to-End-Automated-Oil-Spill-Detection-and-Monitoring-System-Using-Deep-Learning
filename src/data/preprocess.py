# D:\Coding\SEM-8-NEW\OIL-SPILL\src\data\preprocess.py

import os
import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------
# ABSOLUTE PATHS FOR GOOGLE DRIVE
# ---------------------------------------------------

INPUT_DIR = r"D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_1"
OUTPUT_DIR = r"D:\Coding\SEM-8-NEW\OIL-SPILL\src\data\processed"

IMG_SIZE = (256, 256)


# ---------------------------------------------------
# Create processed directories
# ---------------------------------------------------
def create_dirs():
    subsets = ["train", "val", "test"]
    for subset in subsets:
        os.makedirs(os.path.join(OUTPUT_DIR, subset), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, subset, "masks"), exist_ok=True)


# ---------------------------------------------------
# Image preprocessing (UNCHANGED)
# ---------------------------------------------------
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img


# ---------------------------------------------------
# Mask preprocessing (FIXED)
# ---------------------------------------------------
def preprocess_mask(mask):
    # Resize using NEAREST to preserve boundaries
    mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

    # Normalize to [0,1] — DO NOT binarize
    mask = mask.astype(np.uint8)   # preserve class labels
    return mask


# ---------------------------------------------------
# Process each split (train / val / test)
# ---------------------------------------------------
def process_split(split):
    print(f"\nProcessing {split}...")

    img_input_dir = os.path.join(INPUT_DIR, split, "images")
    mask_input_dir = os.path.join(INPUT_DIR, split, "masks")

    img_output_dir = os.path.join(OUTPUT_DIR, split, "images")
    mask_output_dir = os.path.join(OUTPUT_DIR, split, "masks")

    img_files = [f for f in os.listdir(img_input_dir) if f.lower().endswith(".jpg")]

    images_list = []
    masks_list = []
    labels_list = []

    for file in tqdm(img_files):
        base_name = os.path.splitext(file)[0]

        img_path = os.path.join(img_input_dir, file)
        mask_path = os.path.join(mask_input_dir, base_name + ".png")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[SKIP] Could not read image: {file}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[SKIP] Missing mask for: {base_name}.png")
            continue

        # Preprocess
        img = preprocess_image(img)
        mask = preprocess_mask(mask)

        # CNN label based on OIL AREA RATIO (FIXED)
#        oil_ratio = np.sum(mask > 0.1) / mask.size
#        label = 1 if oil_ratio > 0.02 else 0   # ≥2% oil → oil image
        label = 1 if np.any(mask > 0) else 0

        # Save individual files
        np.save(os.path.join(img_output_dir, base_name + ".npy"), img)
        np.save(os.path.join(mask_output_dir, base_name + ".npy"), mask)

        images_list.append(img)
        masks_list.append(mask)
        labels_list.append(label)

    images_np = np.array(images_list, dtype=np.float32)
    masks_np = np.array(masks_list, dtype=np.float32)
    labels_np = np.array(labels_list, dtype=np.uint8)

    np.save(os.path.join(OUTPUT_DIR, split, "images.npy"), images_np)
    np.save(os.path.join(OUTPUT_DIR, split, "masks.npy"), masks_np)
    np.save(os.path.join(OUTPUT_DIR, split, "labels.npy"), labels_np)

    print(f"✔ Saved {split}: images={images_np.shape}, masks={masks_np.shape}, labels={labels_np.shape}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    print("Creating directories...")
    create_dirs()

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n✅ DONE!")
    print("Processed dataset saved at:")
    print(OUTPUT_DIR)
