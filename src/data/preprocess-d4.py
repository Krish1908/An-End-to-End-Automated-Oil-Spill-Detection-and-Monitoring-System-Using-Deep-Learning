# /content/drive/MyDrive/OIL-SPILL-8/src/data/preprocess_d4.py
#
# D4 SPECIFIC PREPROCESSING — CSIRO Sentinel-1 SAR Oil Spill Dataset

import os
import cv2
import numpy as np
import json
import random
from tqdm import tqdm
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR    = "."
DATASET_DIR = "dataset_4"
D3_DIR      = "dataset_3"           # for oil crop extraction
OUTPUT_DIR  = "src/data/processed-d4"

CNN_SIZE    = (256, 256)
LEE_KERNEL  = 7

# Train/val/test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Minimum oil region area to extract as crop (pixels, at original 256x256 scale)
MIN_CROP_AREA = 32 * 32

# How many D3 oil crops to add to CNN training data
# Set to 0 to disable D3 crop augmentation
MAX_D3_OIL_CROPS = 500
MAX_D3_BG_CROPS  = 500

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
    print("✔ Output directories created")

# ─────────────────────────────────────────────────────────────────────────────
# SAR PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(img):
    """Convert 3-channel grayscale-stored SAR image to true single channel."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def lee_filter(img, kernel_size=LEE_KERNEL):
    """
    Lee speckle filter for SAR images.
    Reduces high-frequency speckle noise while preserving edges.
    """
    img      = img.astype(np.float32)
    k        = kernel_size
    mean     = cv2.blur(img, (k, k))
    mean_sq  = cv2.blur(img ** 2, (k, k))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    noise_var = np.mean(variance)
    weight   = variance / (variance + noise_var + 1e-8)
    filtered = mean + weight * (img - mean)
    return filtered.astype(np.float32)


def normalize(img):
    """Normalize to 0-255 uint8. Handles 8-bit and 16-bit input."""
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)


def process_sar_image(img_path):
    """
    Full SAR preprocessing for a single D4 image:
    1. Load JPEG (IMREAD_UNCHANGED)
    2. Convert 3ch → 1ch grayscale
    3. Normalize 0-255
    4. Lee speckle filter
    5. Re-normalize
    6. Resize to 256x256
    Returns uint8 grayscale 256x256 image or None on failure.
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    gray     = to_grayscale(img)
    normed   = normalize(gray.astype(np.float32))
    filtered = lee_filter(normed.astype(np.float32), LEE_KERNEL)
    result   = normalize(filtered)
    resized  = cv2.resize(result, CNN_SIZE, interpolation=cv2.INTER_LINEAR)
    return resized

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT D4 FILES WITH LABELS
# ─────────────────────────────────────────────────────────────────────────────

def collect_d4_files():
    """
    Walk D4 folder structure and collect (path, label) pairs.
    Class_0 → label 0 (no oil)
    Class_1 → label 1 (oil)
    Skips Samples folder and any non-image files.
    """
    class_map = {
        "S1SAR_UnBalanced_400by400_Class_0": 0,
        "S1SAR_UnBalanced_400by400_Class_1": 1,
    }
    IMG_EXT = {".jpg", ".jpeg", ".png"}
    all_files = []

    data_dir = f"{DATASET_DIR}/data"
    for class_folder, label in class_map.items():
        class_path = Path(data_dir) / class_folder
        if not class_path.exists():
            print(f"  [WARN] Folder not found: {class_path}")
            continue
        # Images are inside a subfolder named "0" or "1"
        for sub in class_path.iterdir():
            if not sub.is_dir():
                continue
            for f in sorted(sub.iterdir()):
                if f.suffix.lower() in IMG_EXT and f.is_file():
                    all_files.append((str(f), label))

    print(f"  D4 total files found: {len(all_files)}")
    oil    = sum(1 for _, l in all_files if l == 1)
    no_oil = sum(1 for _, l in all_files if l == 0)
    print(f"  Oil: {oil}   No-oil: {no_oil}")
    return all_files


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(all_files):
    """
    Split into train/val/test maintaining class ratio in each split.
    Returns dict: {"train": [...], "val": [...], "test": [...]}
    Each item is (path, label).
    """
    oil_files    = [(p, l) for p, l in all_files if l == 1]
    no_oil_files = [(p, l) for p, l in all_files if l == 0]

    random.shuffle(oil_files)
    random.shuffle(no_oil_files)

    def split_list(lst):
        n      = len(lst)
        n_tr   = int(n * TRAIN_RATIO)
        n_val  = int(n * VAL_RATIO)
        return lst[:n_tr], lst[n_tr:n_tr+n_val], lst[n_tr+n_val:]

    oil_tr,    oil_val,    oil_te    = split_list(oil_files)
    noil_tr,   noil_val,   noil_te   = split_list(no_oil_files)

    splits = {
        "train": oil_tr + noil_tr,
        "val":   oil_val + noil_val,
        "test":  oil_te + noil_te,
    }
    for k in splits:
        random.shuffle(splits[k])

    for split, items in splits.items():
        oil    = sum(1 for _, l in items if l == 1)
        no_oil = sum(1 for _, l in items if l == 0)
        print(f"  {split:6s} → total={len(items)}  oil={oil}  no_oil={no_oil}")

    return splits

# ─────────────────────────────────────────────────────────────────────────────
# D3 OIL CROP EXTRACTION (fixes CNN scale mismatch)
# ─────────────────────────────────────────────────────────────────────────────

def extract_d3_crops(max_oil=MAX_D3_OIL_CROPS, max_bg=MAX_D3_BG_CROPS):
    """
    Extract oil-region crops from D3 training images using masks.
    Also extracts background (no-oil) crops from D3 images.
    These are added to CNN training data to fix the scale mismatch
    between full D4 scenes and YOLO crop inputs at inference.

    Returns list of (crop_array, label) — NOT file paths, actual arrays.
    """
    if max_oil == 0 and max_bg == 0:
        return []

    print("\n  Extracting D3 crops for CNN scale mismatch fix...")

    d3_img_dir  = f"{D3_DIR}/images/images/train"
    d3_mask_dir = f"{D3_DIR}/masks/masks/train"

    img_files = sorted([
        f for f in os.listdir(d3_img_dir)
        if f.lower().endswith(".png")
    ])
    random.shuffle(img_files)

    oil_crops = []
    bg_crops  = []

    for fname in tqdm(img_files, desc="  D3 crops"):
        if len(oil_crops) >= max_oil and len(bg_crops) >= max_bg:
            break

        base      = os.path.splitext(fname)[0]
        img_path  = f"{d3_img_dir}/{fname}"
        mask_path = f"{d3_mask_dir}/{base}.png"

        if not os.path.exists(mask_path):
            continue

        # Load and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        gray     = to_grayscale(img)
        normed   = normalize(gray.astype(np.float32))
        filtered = lee_filter(normed.astype(np.float32), LEE_KERNEL)
        processed = normalize(filtered)   # uint8 256x256

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        has_oil = np.any(binary > 0)

        # ── Oil crops ────────────────────────────────────────
        if has_oil and len(oil_crops) < max_oil:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                if len(oil_crops) >= max_oil:
                    break
                x, y, w, h = cv2.boundingRect(cnt)
                if w * h < MIN_CROP_AREA:
                    continue
                # Add padding around crop
                pad = 10
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(processed.shape[1], x + w + pad)
                y2 = min(processed.shape[0], y + h + pad)
                crop = processed[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, CNN_SIZE,
                                          interpolation=cv2.INTER_LINEAR)
                oil_crops.append((crop_resized, 1))

        # ── Background crops ──────────────────────────────────
        if not has_oil and len(bg_crops) < max_bg:
            # Full image is background — resize and use
            bg = cv2.resize(processed, CNN_SIZE, interpolation=cv2.INTER_LINEAR)
            bg_crops.append((bg, 0))

    print(f"  D3 oil crops extracted  : {len(oil_crops)}")
    print(f"  D3 background crops     : {len(bg_crops)}")
    return oil_crops + bg_crops

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split, file_list, d3_crops=None):
    """
    Process all D4 files for one split.
    For train split, also saves D3 crops.
    """
    print(f"\n── Processing {split} ──────────────────────────────")

    out_img_dir = f"{OUTPUT_DIR}/{split}/images"
    label_file  = f"{OUTPUT_DIR}/{split}/labels.txt"

    skipped = 0

    with open(label_file, "w") as lf:

        # ── D4 images ─────────────────────────────────────────
        for img_path, label in tqdm(file_list, desc=f"  D4 {split}"):
            fname     = os.path.basename(img_path)
            out_fname = fname

            processed = process_sar_image(img_path)
            if processed is None:
                print(f"  [SKIP] Cannot load: {fname}")
                skipped += 1
                continue

            cv2.imwrite(f"{out_img_dir}/{out_fname}", processed)
            lf.write(f"{out_fname} {label}\n")

        # ── D3 crops (train only) ─────────────────────────────
        if split == "train" and d3_crops:
            print(f"  Adding {len(d3_crops)} D3 crops to train set...")
            for idx, (crop_array, label) in enumerate(
                tqdm(d3_crops, desc="  D3 crops → train")
            ):
                crop_fname = f"d3_crop_{idx:05d}.png"
                cv2.imwrite(f"{out_img_dir}/{crop_fname}", crop_array)
                lf.write(f"{crop_fname} {label}\n")

    total   = len(file_list) + (len(d3_crops) if split == "train" and d3_crops else 0)
    print(f"  ✔ {split} done — {total} images saved  skipped={skipped}")

# ─────────────────────────────────────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train_label_file):
    """
    Compute class weights for weighted loss in CNN training.
    Inverse frequency weighting: weight_i = total / (n_classes * count_i)
    Saves to class_weights.json.
    """
    counts = {0: 0, 1: 0}
    with open(train_label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                counts[int(parts[1])] += 1

    total    = counts[0] + counts[1]
    weight_0 = total / (2 * counts[0]) if counts[0] > 0 else 1.0
    weight_1 = total / (2 * counts[1]) if counts[1] > 0 else 1.0

    weights = {
        "class_0_no_oil": round(weight_0, 4),
        "class_1_oil":    round(weight_1, 4),
        "counts": counts,
        "total":  total,
        "note": "Use as pos_weight in BCEWithLogitsLoss or class_weight in CrossEntropyLoss"
    }

    out_path = f"{OUTPUT_DIR}/class_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\n  Class weights saved → {out_path}")
    print(f"  no_oil weight : {weight_0:.4f}")
    print(f"  oil weight    : {weight_1:.4f}")
    print(f"  counts        : {counts}")
    return weights

# ─────────────────────────────────────────────────────────────────────────────
# SPLIT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def write_split_summary(splits):
    lines = ["D4 Split Summary", "=" * 40]
    for split, items in splits.items():
        oil    = sum(1 for _, l in items if l == 1)
        no_oil = sum(1 for _, l in items if l == 0)
        total  = len(items)
        lines.append(
            f"{split:6s}  total={total:5d}  oil={oil:4d}  "
            f"no_oil={no_oil:4d}  oil_ratio={oil/total:.2%}"
        )
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(f"{OUTPUT_DIR}/split_summary.txt", "w") as f:
        f.write(summary)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  D4 SAR PREPROCESSING")
    print("  CSIRO Sentinel-1 SAR Oil Spill Dataset")
    print("=" * 60)

    print("\n[1/5] Creating output directories...")
    create_dirs()

    print("\n[2/5] Collecting D4 files...")
    all_files = collect_d4_files()
    if not all_files:
        raise RuntimeError("No D4 files found. Check DATASET_DIR path.")

    print("\n[3/5] Stratified train/val/test split...")
    splits = stratified_split(all_files)
    write_split_summary(splits)

    print("\n[4/5] Extracting D3 oil crops for CNN scale mismatch fix...")
    d3_crops = extract_d3_crops(
        max_oil=MAX_D3_OIL_CROPS,
        max_bg=MAX_D3_BG_CROPS
    )

    print("\n[5/5] Processing splits...")
    for split, file_list in splits.items():
        process_split(
            split,
            file_list,
            d3_crops=d3_crops if split == "train" else None
        )

    print("\nComputing class weights for CNN training...")
    compute_class_weights(f"{OUTPUT_DIR}/train/labels.txt")

    print("\n" + "=" * 60)
    print("  OUTPUT STRUCTURE")
    print("=" * 60)
    print(f"  {OUTPUT_DIR}/")
    print(f"    train/images/     ← D4 images + D3 crops")
    print(f"    train/labels.txt  ← filename label (0/1)")
    print(f"    val/images/")
    print(f"    val/labels.txt")
    print(f"    test/images/")
    print(f"    test/labels.txt")
    print(f"    class_weights.json")
    print(f"    split_summary.txt")
    print("\n✅ D4 preprocessing complete.")


if __name__ == "__main__":
    main()