# /content/drive/MyDrive/OIL-SPILL-8/src/data/preprocess.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

BASE_DIR = "/content/drive/MyDrive/OIL-SPILL-8"
RAW_DATASET_DIR = os.path.join(BASE_DIR, "dataset_1")
OUTPUT_DIR = os.path.join(BASE_DIR, "src/data/processed")
COLOR_TXT_PATH = os.path.join(RAW_DATASET_DIR, "label_colors.txt")

IMG_SIZE = None  # KEEP ORIGINAL SIZE (important for YOLO)

# ---------------------------------------------------
# LOAD COLOR → CLASS MAPPING
# ---------------------------------------------------

def load_color_mapping(txt_path):
    color_map = {}
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            r, g, b, label = parts
            color_map[label.lower()] = (int(r), int(g), int(b))
    return color_map

# ---------------------------------------------------
# CREATE OUTPUT STRUCTURE
# ---------------------------------------------------

def create_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)

# ---------------------------------------------------
# EXTRACT BINARY OIL MASK
# ---------------------------------------------------

def extract_oil_mask(mask_bgr, oil_color_rgb):
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    oil_mask = np.all(mask_rgb == oil_color_rgb, axis=-1).astype(np.uint8) * 255
    return oil_mask

# ---------------------------------------------------
# PROCESS SPLIT
# ---------------------------------------------------

def process_split(split, oil_color):
    print(f"\nProcessing {split} set...")

    raw_img_dir = os.path.join(RAW_DATASET_DIR, split, "images")
    raw_mask_dir = os.path.join(RAW_DATASET_DIR, split, "masks")

    out_img_dir = os.path.join(OUTPUT_DIR, split, "images")
    out_mask_dir = os.path.join(OUTPUT_DIR, split, "masks")
    label_file = os.path.join(OUTPUT_DIR, split, "labels.txt")

    img_files = sorted([
        f for f in os.listdir(raw_img_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    with open(label_file, "w") as lf:
        for fname in tqdm(img_files):
            img_path = os.path.join(raw_img_dir, fname)
            mask_path = os.path.join(raw_mask_dir, os.path.splitext(fname)[0] + ".png")

            if not os.path.exists(mask_path):
                print(f"[SKIP] Missing mask for {fname}")
                continue

            # Copy image as-is (YOLO & CNN friendly)
            shutil.copy(img_path, os.path.join(out_img_dir, fname))

            # Load and convert mask
            mask = cv2.imread(mask_path)
            if mask is None:
                print(f"[SKIP] Invalid mask for {fname}")
                continue

            oil_mask = extract_oil_mask(mask, oil_color)

            cv2.imwrite(
                os.path.join(out_mask_dir, os.path.splitext(fname)[0] + ".png"),
                oil_mask
            )

            # CNN label: oil if any oil pixel exists
            label = 1 if np.any(oil_mask > 0) else 0
            lf.write(f"{fname} {label}\n")

    print(f"✔ {split} completed")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":
    print("🔹 Loading label color mapping...")
    color_map = load_color_mapping(COLOR_TXT_PATH)

    if "oil" not in color_map:
        raise RuntimeError("Oil class not found in label_colors.txt")

    oil_color = color_map["oil"]

    print("🔹 Creating unified processed dataset...")
    create_dirs()

    for split in ["train", "val", "test"]:
        process_split(split, oil_color)

    print("\n✅ Unified preprocessing completed successfully.")
    print("Dataset ready for YOLO, CNN, and U-Net.")
