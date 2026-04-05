# evaluate_unet_sar_kaggle_gpu.py

import os
import sys
import numpy as np
import tensorflow as tf
import cv2

# ---------------------------------------------------
# GPU CONFIG (IMPORTANT)
# ---------------------------------------------------

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU not detected — running on CPU")

# ---------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------

PROJECT_ROOT = "."

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------

MODEL_PATH = "models-sar/unet/unet_sar.keras"

IMG_DIR = "src/data/processed-d3/cnn_unet/val/images"

MASK_DIR = "src/data/processed-d3/cnn_unet/val/masks"

SAVE_DIR = "results-d3/unet_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------
# SETTINGS
# ---------------------------------------------------

IMG_SIZE = (256, 256)
BATCH_SIZE = 32
THRESHOLD = 0.35

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print("📦 Loading U-Net model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("✅ Model loaded")

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

img_files = sorted(os.listdir(IMG_DIR))

print("📊 Total images:", len(img_files))

def load_batch(files):

    images = []
    masks = []

    for fname in files:

        img = cv2.imread(os.path.join(IMG_DIR, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype("float32") / 255.0

        mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMG_SIZE)
        mask = (mask > 127).astype(np.uint8)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------

def compute_metrics(y_true, y_pred):

    intersection = np.sum(y_true * y_pred)

    dice = (2 * intersection + 1e-7) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

    iou = (intersection + 1e-7) / (np.sum(y_true) + np.sum(y_pred) - intersection + 1e-7)

    tp = intersection
    fp = np.sum(y_pred) - tp
    fn = np.sum(y_true) - tp

    precision = (tp + 1e-7) / (tp + fp + 1e-7)
    recall = (tp + 1e-7) / (tp + fn + 1e-7)

    pixel_acc = np.mean(y_true == y_pred)

    return dice, iou, precision, recall, pixel_acc

# ---------------------------------------------------
# EVALUATION LOOP
# ---------------------------------------------------

dice_scores = []
iou_scores = []
precision_scores = []
recall_scores = []
pixel_acc_scores = []

print("🚀 Starting evaluation...")

for i in range(0, len(img_files), BATCH_SIZE):

    batch_files = img_files[i:i+BATCH_SIZE]

    images, masks = load_batch(batch_files)

    preds = model.predict(images, verbose=0)

    preds = preds[...,0]

    preds_bin = (preds > THRESHOLD).astype(np.uint8)

    for j in range(len(batch_files)):

        dice, iou, precision, recall, pixel_acc = compute_metrics(
            masks[j],
            preds_bin[j]
        )

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        pixel_acc_scores.append(pixel_acc)

    print(f"Processed {min(i+BATCH_SIZE,len(img_files))}/{len(img_files)}")

# ---------------------------------------------------
# FINAL METRICS
# ---------------------------------------------------

dice = np.mean(dice_scores)
iou = np.mean(iou_scores)
precision = np.mean(precision_scores)
recall = np.mean(recall_scores)
pixel_acc = np.mean(pixel_acc_scores)

print("\n📊 FINAL RESULTS")

print("Dice:", dice)
print("IoU:", iou)
print("Precision:", precision)
print("Recall:", recall)
print("Pixel Accuracy:", pixel_acc)

with open(os.path.join(SAVE_DIR, "metrics.txt"), "w") as f:

    f.write(f"Dice: {dice:.4f}\n")
    f.write(f"IoU: {iou:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")

print("\n✅ Evaluation finished")
print("Results saved to:", SAVE_DIR)