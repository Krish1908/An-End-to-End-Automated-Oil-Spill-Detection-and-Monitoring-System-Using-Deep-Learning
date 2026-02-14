#/content/drive/MyDrive/OIL-SPILL-8/src/data/mask_to_yolo.py

import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/OIL-SPILL-8"

DATASET_DIR = f"{BASE_DIR}/dataset_1"
YOLO_DIR = f"{BASE_DIR}/yolo_dataset"

IMG_EXT = ".jpg"

# --------------------------------------------------
# CREATE YOLO FOLDERS
# --------------------------------------------------
def create_yolo_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{YOLO_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{YOLO_DIR}/labels/{split}", exist_ok=True)

# --------------------------------------------------
# MASK → YOLO LABEL
# --------------------------------------------------
def mask_to_yolo(mask, img_w, img_h, min_area=30):
    """
    Converts segmentation mask to YOLO bounding boxes.
    Returns list of YOLO label strings.
    """
    labels = []

    # Binary mask
    bin_mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Skip very tiny noise
        if area < min_area:
            continue

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Optional safety: skip absurd full-image boxes
        if w_norm > 0.95 or h_norm > 0.95:
            continue

        labels.append(
            f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )

    return labels

# --------------------------------------------------
# PROCESS SPLIT
# --------------------------------------------------
def process_split(split):
    print(f"\nProcessing {split} set...")

    img_dir = f"{DATASET_DIR}/{split}/images"
    mask_dir = f"{DATASET_DIR}/{split}/masks"

    out_img_dir = f"{YOLO_DIR}/images/{split}"
    out_lbl_dir = f"{YOLO_DIR}/labels/{split}"

    for fname in tqdm(os.listdir(img_dir)):
        if not fname.lower().endswith(IMG_EXT):
            continue

        base = os.path.splitext(fname)[0]

        img_path = f"{img_dir}/{fname}"
        mask_path = f"{mask_dir}/{base}.png"

        # Always copy image (oil OR no-oil)
        shutil.copy(img_path, f"{out_img_dir}/{fname}")

        label_path = f"{out_lbl_dir}/{base}.txt"

        # If mask does not exist → NO OIL
        if not os.path.exists(mask_path):
            open(label_path, "w").close()  # empty label
            continue

        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        h, w = img.shape[:2]

        yolo_labels = mask_to_yolo(mask, w, h)

        # Write labels (empty file = no oil)
        with open(label_path, "w") as f:
            if len(yolo_labels) > 0:
                f.write("\n".join(yolo_labels))

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("🔹 Creating YOLO dataset structure...")
    create_yolo_dirs()

    for split in ["train", "val", "test"]:
        process_split(split)

    print("\n✅ YOLO dataset prepared with background (no-oil) learning enabled.")
