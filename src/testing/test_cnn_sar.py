# /content/drive/MyDrive/OIL-SPILL-8/src/testing/test_cnn.py

# CNN TESTING SCRIPT FOR DATASET-D4

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ---------------------------------------------------
# PATH SETUP
# ---------------------------------------------------

PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.dataloader_cnn import create_dataset

MODEL_PATH = "/content/drive/MyDrive/OIL-SPILL-8/models-sar/cnn/cnn_classifier.keras"

SAVE_DIR = "/content/drive/MyDrive/OIL-SPILL-8/models-sar/cnn/test_results"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------

print("📂 Loading test dataset...")

test_ds = create_dataset(
    split="test",
    batch_size=16,
    augment_data=False
)

# collect full dataset

X_test = []
y_true = []

for imgs, labels in test_ds:
    X_test.append(imgs.numpy())
    y_true.append(labels.numpy())

X_test = np.concatenate(X_test)
y_true = np.concatenate(y_true)

print("✔ Test samples:", len(X_test))


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

print("📦 Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

# ---------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------

print("🚀 Running inference...")

y_prob = model.predict(X_test).flatten()
y_pred = (y_prob > 0.5).astype(int)


# ---------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------

print("\n📊 Classification Report\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=["No Oil", "Oil"]
)

print(report)

with open(os.path.join(SAVE_DIR, "classification_report.txt"), "w") as f:
    f.write(report)


# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5,5))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0,1], ["No Oil", "Oil"])
plt.yticks([0,1], ["No Oil", "Oil"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("True")

plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()


# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(os.path.join(SAVE_DIR, "roc_curve.png"))
plt.close()


# ---------------------------------------------------
# SAMPLE PREDICTIONS
# ---------------------------------------------------

print("🖼 Saving sample predictions...")

for i in range(min(20, len(X_test))):

    img = X_test[i]
    prob = y_prob[i]

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.axis("off")

    label = "Oil" if prob > 0.5 else "No Oil"

    plt.title(f"{label} ({prob:.2f})")

    plt.savefig(os.path.join(SAVE_DIR, f"sample_{i}.png"))
    plt.close()


print("\n✅ CNN testing completed")
print("Results saved in:", SAVE_DIR)