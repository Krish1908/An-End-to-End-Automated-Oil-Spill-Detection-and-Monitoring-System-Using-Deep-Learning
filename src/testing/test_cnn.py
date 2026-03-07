# /content/drive/MyDrive/OIL-SPILL-8/src/testing/test_cnn.py

# D1 SPECIFIC CNN TESTING SCRIPT

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

TEST_IMG_DIR = "/content/drive/MyDrive/OIL-SPILL-8/src/data/processed/test/images"
MODEL_PATH   = "/content/drive/MyDrive/OIL-SPILL-8/models/cnn/cnn_classifier.keras"
SAVE_DIR     = "/content/drive/MyDrive/OIL-SPILL-8/models/cnn/test_results"

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

# ---------------------------------------------------
# LOAD TEST IMAGES ONLY
# ---------------------------------------------------
def load_test_images():
    img_files = sorted(os.listdir(TEST_IMG_DIR))
    X_test = []

    for fname in img_files:
        img = tf.keras.utils.load_img(
            os.path.join(TEST_IMG_DIR, fname),
            target_size=IMG_SIZE
        )
        img = tf.keras.utils.img_to_array(img) / 255.0
        X_test.append(img)

    return img_files, np.array(X_test, dtype=np.float32)


# ---------------------------------------------------
# VISUALIZATION
# ---------------------------------------------------
def visualize_prediction(img, prob, save_path):
    plt.figure(figsize=(5, 5))
    plt.imshow((img * 255).astype(np.uint8))
    plt.axis("off")
    plt.title(f"Oil Probability: {prob:.2f}")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":

    print("📌 Loading test images...")
    img_files, X_test = load_test_images()

    if len(X_test) == 0:
        raise RuntimeError("❌ No test images found")

    print(f"✔ Loaded {len(X_test)} test images")

    print("📌 Loading CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📌 Running CNN inference...")
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob > 0.5).astype(int)

    # ---------------------------------------------------
    # SAVE VISUAL RESULTS
    # ---------------------------------------------------
    print("📌 Saving prediction visualizations...")
    for i, fname in enumerate(img_files):
        save_path = os.path.join(SAVE_DIR, f"pred_{fname}")
        visualize_prediction(X_test[i], y_prob[i], save_path)

    print("✅ CNN inference completed successfully")
    print(f"Positive predictions (oil): {np.sum(y_pred)} / {len(y_pred)}")
