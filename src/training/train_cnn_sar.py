# /content/drive/MyDrive/OIL-SPILL-8/src/training/train_cnn_sar.py

# CNN TRAINING SCRIPT FOR DATASET-D4

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# GPU CONFIG
# ---------------------------------------------------

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU enabled:", gpus)
    except RuntimeError as e:
        print("GPU setup error:", e)
else:
    print("⚠️ No GPU detected — running on CPU")


# ---------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------

PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


# ---------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------

from data.dataloader_cnn_sar import create_dataset, load_class_weights
from models.cnn import build_cnn_model


# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 1e-4

MODEL_DIR = "/content/drive/MyDrive/OIL-SPILL-8/models-sar/cnn"
MODEL_SAVE_PATH = f"{MODEL_DIR}/cnn_classifier.keras"

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------
# TRAIN FUNCTION
# ---------------------------------------------------

def train_cnn():

    print("📂 Loading CNN datasets...")

    train_ds = create_dataset(
        split="train",
        batch_size=BATCH_SIZE,
        augment_data=True
    )

    val_ds = create_dataset(
        split="val",
        batch_size=BATCH_SIZE,
        augment_data=False
    )


    # ---------------------------------------------------
    # CLASS WEIGHTS (IMPORTANT FOR IMBALANCED DATASET)
    # ---------------------------------------------------

    class_weights = load_class_weights()

    if class_weights:
        print("⚖️ Using class weights:", class_weights)
    else:
        print("⚠️ No class weights file found")


    # ---------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------

    print("🧠 Building CNN model...")

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
            monitor="val_accuracy",
            save_best_only=True,
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


    print("\n🚀 Starting CNN training...\n")


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights
    )


    # ---------------------------------------------------
    # TRAINING SUMMARY
    # ---------------------------------------------------

    best_val_acc = max(history.history["val_accuracy"])
    best_val_auc = max(history.history["val_auc"])
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1


    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🏆 CNN TRAINING SUMMARY")
    print("Best Epoch:", best_epoch)
    print("Best Val Accuracy:", round(best_val_acc,5))
    print("Best Val AUC:", round(best_val_auc,5))
    print("Model saved at:", MODEL_SAVE_PATH)
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


    print("💾 Saving model...")
    model.save(MODEL_SAVE_PATH)

    print("📊 Saving training plots...")
    plot_history(history)

    print("🎉 CNN training completed!")


# ---------------------------------------------------
# TRAINING PLOTS
# ---------------------------------------------------

def plot_history(history):

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "cnn_accuracy.png"))
    plt.close()


    # AUC
    plt.figure(figsize=(8,5))
    plt.plot(history.history["auc"], label="Train AUC")
    plt.plot(history.history["val_auc"], label="Val AUC")
    plt.title("CNN AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "cnn_auc.png"))
    plt.close()


    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("CNN Loss")
    plt.xlabel("Epoch")
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