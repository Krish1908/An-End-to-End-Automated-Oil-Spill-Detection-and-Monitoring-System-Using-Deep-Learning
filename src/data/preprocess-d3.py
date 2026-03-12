# /content/drive/MyDrive/OIL-SPILL-8/src/data/preprocess_d3.py

# D3 SPECIFIC PREPROCESSING — Zenodo 15298010 (Refined Deep-SAR SOS)

import os
import cv2
import numpy as np
from tqdm import tqdm
import yaml

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = "D:\Coding\SEM-8-NEW\OIL-SPILL"
DATASET_DIR = "D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_3"
OUTPUT_DIR = "D:\Coding\SEM-8-NEW\OIL-SPILL\dataset_3\processed-d3"

CNN_UNET_SIZE = (256, 256)
YOLO_SIZE     = (640, 640)

# Lee filter kernel size — 7x7 is standard for Sentinel-1
LEE_KERNEL    = 7

# Minimum bbox area in pixels to include in YOLO labels (filters noise)
MIN_BBOX_AREA = 30

SPLITS = ["train", "val"]

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY CREATION
# ─────────────────────────────────────────────────────────────────────────────

def create_dirs():
    for split in SPLITS:
        # CNN / UNet outputs
        os.makedirs(f"{OUTPUT_DIR}/cnn_unet/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/cnn_unet/{split}/masks",  exist_ok=True)
        # YOLO outputs
        os.makedirs(f"{OUTPUT_DIR}/yolo/images/{split}",     exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/yolo/labels/{split}",     exist_ok=True)
    print("✔ Output directories created")

# ─────────────────────────────────────────────────────────────────────────────
# SAR PREPROCESSING FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(img):
    """
    Convert 3-channel grayscale-stored SAR image to true single channel.
    D3 images are grayscale data stored as 3ch PNG (R=G=B).
    """
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def lee_filter(img, kernel_size=LEE_KERNEL):
    """
    Lee speckle filter for SAR images.
    Reduces high-frequency speckle noise while preserving edges.
    
    Formula:
        output = mean + k * (pixel - mean)
        where k = variance_local / (variance_local + variance_noise)
    
    img must be float32.
    """
    img = img.astype(np.float32)
    k        = kernel_size
    mean     = cv2.blur(img, (k, k))
    mean_sq  = cv2.blur(img ** 2, (k, k))
    variance = mean_sq - mean ** 2
    variance = np.maximum(variance, 0)

    # Estimate noise variance as median of local variances
    noise_var = np.mean(variance)

    # Lee weight
    weight = variance / (variance + noise_var + 1e-8)

    filtered = mean + weight * (img - mean)
    return filtered.astype(np.float32)


def normalize(img):
    """
    Normalize pixel values to 0-255 uint8.
    Input can be any range (handles 8-bit and 16-bit SAR).
    """
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    normalized = (img - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)


def process_sar_image(img_path):
    """
    Full SAR preprocessing pipeline for a single image:
    1. Load with IMREAD_UNCHANGED (handles 8-bit and 16-bit)
    2. Convert to grayscale
    3. Normalize to 0-255
    4. Apply Lee speckle filter
    Returns: uint8 grayscale image
    """
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    gray     = to_grayscale(img)
    normed   = normalize(gray.astype(np.float32))
    filtered = lee_filter(normed.astype(np.float32), LEE_KERNEL)
    result   = normalize(filtered)   # re-normalize after filter
    return result


def process_mask(mask_path):
    """
    Load and clean binary mask.
    D3 masks are already binary (0=ocean, 255=oil).
    Ensures clean binary output regardless of stored values.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    # Threshold to strict binary — handles any near-zero noise
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary

# ─────────────────────────────────────────────────────────────────────────────
# YOLO BBOX GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def mask_to_yolo_labels(mask, img_w, img_h, min_area=MIN_BBOX_AREA):
    """
    Convert binary mask to YOLO format bounding boxes.
    Finds all oil contours and generates one bbox per contour.
    Returns list of YOLO label strings.
    """
    labels = []
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm   = w / img_w
        h_norm   = h / img_h

        # Skip near-full-image boxes (mask artifacts)
        if w_norm > 0.95 or h_norm > 0.95:
            continue

        labels.append(
            f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
        )
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# PROCESS ONE SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def process_split(split):
    print(f"\n── Processing {split} ──────────────────────────────")

    raw_img_dir  = f"{DATASET_DIR}/images/images/{split}"
    raw_mask_dir = f"{DATASET_DIR}/masks/masks/{split}"

    out_cnn_img  = f"{OUTPUT_DIR}/cnn_unet/{split}/images"
    out_cnn_mask = f"{OUTPUT_DIR}/cnn_unet/{split}/masks"
    out_yolo_img = f"{OUTPUT_DIR}/yolo/images/{split}"
    out_yolo_lbl = f"{OUTPUT_DIR}/yolo/labels/{split}"
    label_file   = f"{OUTPUT_DIR}/cnn_unet/{split}/labels.txt"

    img_files = sorted([
        f for f in os.listdir(raw_img_dir)
        if f.lower().endswith(".png")
    ])

    skipped  = 0
    oil_count    = 0
    no_oil_count = 0

    with open(label_file, "w") as lf:
        for fname in tqdm(img_files, desc=f"  {split}"):
            base      = os.path.splitext(fname)[0]
            img_path  = f"{raw_img_dir}/{fname}"
            mask_path = f"{raw_mask_dir}/{base}.png"

            # ── Process SAR image ────────────────────────────
            processed = process_sar_image(img_path)
            if processed is None:
                print(f"  [SKIP] Cannot load image: {fname}")
                skipped += 1
                continue

            # ── Process mask ─────────────────────────────────
            if not os.path.exists(mask_path):
                print(f"  [SKIP] Missing mask: {base}.png")
                skipped += 1
                continue

            mask = process_mask(mask_path)
            if mask is None:
                print(f"  [SKIP] Cannot load mask: {base}.png")
                skipped += 1
                continue

            # ── CNN / UNet output (256x256) ───────────────────
            img_256  = cv2.resize(processed, CNN_UNET_SIZE,
                                  interpolation=cv2.INTER_LINEAR)
            mask_256 = cv2.resize(mask, CNN_UNET_SIZE,
                                  interpolation=cv2.INTER_NEAREST)  # nearest for masks

            cv2.imwrite(f"{out_cnn_img}/{fname}",        img_256)
            cv2.imwrite(f"{out_cnn_mask}/{base}.png",    mask_256)

            # CNN label
            has_oil = int(np.any(mask_256 > 0))
            lf.write(f"{fname} {has_oil}\n")
            if has_oil:
                oil_count += 1
            else:
                no_oil_count += 1

            # ── YOLO output (640x640) ─────────────────────────
            img_640  = cv2.resize(processed, YOLO_SIZE,
                                  interpolation=cv2.INTER_LINEAR)
            mask_640 = cv2.resize(mask, YOLO_SIZE,
                                  interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(f"{out_yolo_img}/{fname}", img_640)

            # Generate YOLO labels from 640x640 mask
            yolo_labels = mask_to_yolo_labels(mask_640, YOLO_SIZE[0], YOLO_SIZE[1])
            lbl_path = f"{out_yolo_lbl}/{base}.txt"
            with open(lbl_path, "w") as yf:
                if yolo_labels:
                    yf.write("\n".join(yolo_labels))
                # empty file = no oil (valid YOLO convention)

    print(f"  ✔ Done — oil={oil_count}  no_oil={no_oil_count}  skipped={skipped}")
    return oil_count, no_oil_count

# ─────────────────────────────────────────────────────────────────────────────
# YOLO dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────

def write_yolo_yaml():
    yaml_data = {
        "path":  f"{OUTPUT_DIR}/yolo",
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": ["oil_spill"]
    }
    yaml_path = f"{OUTPUT_DIR}/yolo/dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    print(f"  ✔ YOLO dataset.yaml saved → {yaml_path}")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  D3 SAR PREPROCESSING")
    print("  Zenodo 15298010 — Refined Deep-SAR SOS Dataset")
    print("=" * 60)

    print("\n[1/3] Creating output directories...")
    create_dirs()

    print("\n[2/3] Processing splits...")
    stats = {}
    for split in SPLITS:
        oil, no_oil = process_split(split)
        stats[split] = {"oil": oil, "no_oil": no_oil}

    print("\n[3/3] Writing YOLO dataset.yaml...")
    write_yolo_yaml()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for split, s in stats.items():
        total = s["oil"] + s["no_oil"]
        print(f"  {split:6s} → total={total}  oil={s['oil']}  "
              f"no_oil={s['no_oil']}  "
              f"oil_ratio={s['oil']/total:.2%}")

    print(f"\n  CNN/UNet data → {OUTPUT_DIR}/cnn_unet/")
    print(f"  YOLO data     → {OUTPUT_DIR}/yolo/")
    print("\n✅ D3 preprocessing complete.")


if __name__ == "__main__":
    main()