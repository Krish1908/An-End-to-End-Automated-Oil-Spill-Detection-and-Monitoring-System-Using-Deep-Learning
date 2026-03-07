# /content/drive/MyDrive/OIL-SPILL-8/src/testing/test_unet.py

# D1 SPECIFIC U-NET TESTING SCRIPT

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# ---------------------------------------------------
# ADD PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from models.unet import dice_coef

# ---------------------------------------------------
# PATH CONFIG
# ---------------------------------------------------
MODEL_PATH = "/content/drive/MyDrive/OIL-SPILL-8/models/unet/unet_segmentation.keras"

TEST_IMG_DIR = "/content/drive/MyDrive/OIL-SPILL-8/src/data/processed/test/images"
SAVE_DIR = "/content/drive/MyDrive/OIL-SPILL-8/results/unet_predictions"

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

# 🔑 IMPORTANT: lower threshold helps thin oil detection
THRESHOLD = 0.35

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
print("📌 Loading trained U-Net model...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"dice_coef": dice_coef},
    compile=False
)
print("✅ Model loaded successfully!")

# ---------------------------------------------------
# IMAGE LOADER
# ---------------------------------------------------
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img

# ---------------------------------------------------
# OVERLAY FUNCTION
# ---------------------------------------------------
def create_overlay(image, mask):
    overlay = image.copy()
    red = np.zeros_like(image)
    red[..., 0] = 1.0

    alpha = 0.4
    overlay[mask == 1] = (
        (1 - alpha) * overlay[mask == 1] +
        alpha * red[mask == 1]
    )
    return overlay

# ---------------------------------------------------
# MAIN TEST FUNCTION
# ---------------------------------------------------
def test_unet():

    img_files = sorted(os.listdir(TEST_IMG_DIR))
    print(f"📌 Found {len(img_files)} test images")

    for fname in img_files:
        img_path = os.path.join(TEST_IMG_DIR, fname)
        img = load_image(img_path)

        # -------- Inference --------
        pred = model.predict(img[None, ...], verbose=0)[0, ..., 0]

        pred_bin = (pred > THRESHOLD).astype(np.uint8)

        # -------- Overlay --------
        overlay = create_overlay(img, pred_bin)

        # -------- Visualization --------
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(pred, cmap="jet")
        plt.title("Oil Probability Map")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Oil Spill Overlay")
        plt.axis("off")

        save_path = os.path.join(SAVE_DIR, fname)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print(f"✔ Saved: {save_path}")

    print("\n🎉 U-Net Inference Completed Successfully!")

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    test_unet()
