# /content/drive/MyDrive/OIL-SPILL-8/src/training/train_cnn.py

# D1 SPECIFIC CNN TRAINING SCRIPT

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# GPU CONFIGURATION (GOOGLE COLAB)
# ---------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU enabled: {gpus}")
    except RuntimeError as e:
        print("⚠️ GPU configuration error:", e)
else:
    print("⚠️ No GPU detected, running on CPU")

# ---------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# ---------------------------------------------------
PROJECT_ROOT = "src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.dataloader_rgb import create_cnn_dataset
from models.cnn import build_cnn_model

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4

MODEL_DIR = "models-d1/cnn"
MODEL_SAVE_PATH = f"{MODEL_DIR}/cnn_classifier.keras"
PLOT_SAVE_PATH = f"{MODEL_DIR}/cnn_training_plot.png"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_cnn():

    print("📌 Loading CNN datasets...")
    train_ds = create_cnn_dataset(
        split="train",
        batch_size=BATCH_SIZE,
        augment=True
    )

    val_ds = create_cnn_dataset(
        split="val",
        batch_size=BATCH_SIZE,
        augment=False
    )

    print("📌 Building CNN model...")
    model = build_cnn_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    model.summary()

    # ---------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print("🚀 Starting CNN training on GPU (Colab)...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # TRAINING SUMMARY
    # ---------------------------------------------------
    best_val_acc = max(history.history["val_accuracy"])
    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🏆 CNN TRAINING SUMMARY")
    print(f"Best Epoch            : {best_epoch}")
    print(f"Best Validation Acc   : {best_val_acc:.5f}")
    print(f"Best Validation AUC   : {best_val_auc:.5f}")
    print(f"Model Saved At        : {MODEL_SAVE_PATH}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # ---------------------------------------------------
    # SAVE MODEL + PLOT
    # ---------------------------------------------------
    print("💾 Saving CNN model...")
    model.save(MODEL_SAVE_PATH)

    print("📊 Saving training plot...")
    plot_history(history)

    print("🎉 CNN training completed successfully!")

# ---------------------------------------------------
# TRAINING PLOT
# ---------------------------------------------------
def plot_history(history):
    # -------- Accuracy --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "cnn_accuracy.png"))
    plt.close()

    # -------- AUC --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.title("CNN AUC")
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "cnn_auc.png"))
    plt.close()

    # -------- Loss --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("CNN Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "cnn_loss.png"))
    plt.close()


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    train_cnn()
