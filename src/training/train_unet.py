# /content/drive/MyDrive/OIL-SPILL-8/src/training/train_unet.py

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# ---------------------------------------------------
# GPU CONFIGURATION (COLAB SAFE)
# ---------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU enabled: {gpus}")
    except RuntimeError as e:
        print("⚠️ GPU config error:", e)
else:
    print("⚠️ No GPU detected, running on CPU")

# ---------------------------------------------------
# ADD PROJECT ROOT
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8/src"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.dataloader import create_unet_dataset
from models.unet import build_unet, dice_coef, bce_dice_loss

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4

MODEL_DIR = "/content/drive/MyDrive/OIL-SPILL-8/models/unet"
MODEL_SAVE_PATH = f"{MODEL_DIR}/unet_segmentation.keras"

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------
def train_unet():

    print("📌 Loading U-Net datasets...")

    train_ds = create_unet_dataset(
        split="train",
        batch_size=BATCH_SIZE,
        augment=True
    )

    val_ds = create_unet_dataset(
        split="val",
        batch_size=BATCH_SIZE,
        augment=False
    )

    print("📌 Building U-Net model...")
    model = build_unet()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[
            "accuracy",
            dice_coef,
            tf.keras.metrics.MeanIoU(num_classes=2)
        ]
    )

    # ---------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor="val_dice_coef",
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

    print("🚀 Starting U-Net training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ---------------------------------------------------
    # TRAINING SUMMARY
    # ---------------------------------------------------
    best_dice = max(history.history["val_dice_coef"])
    best_epoch = history.history["val_dice_coef"].index(best_dice) + 1

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🏆 U-NET TRAINING SUMMARY")
    print(f"Best Epoch           : {best_epoch}")
    print(f"Best Val Dice        : {best_dice:.5f}")
    print(f"Model Saved At       : {MODEL_SAVE_PATH}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    print("💾 Saving final U-Net model...")
    model.save(MODEL_SAVE_PATH)

    print("📊 Saving training plots...")
    plot_history(history)

    print("🎉 U-Net training completed successfully!")

# ---------------------------------------------------
# TRAINING PLOTS
# ---------------------------------------------------
def plot_history(history):

    # -------- Loss --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("U-Net Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "unet_loss.png"))
    plt.close()

    # -------- Dice --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["dice_coef"], label="Train Dice")
    plt.plot(history.history["val_dice_coef"], label="Val Dice")
    plt.title("U-Net Dice Coefficient")
    plt.xlabel("Epochs")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "unet_dice.png"))
    plt.close()

    # -------- Accuracy --------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("U-Net Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_DIR, "unet_accuracy.png"))
    plt.close()

# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
if __name__ == "__main__":
    train_unet()
