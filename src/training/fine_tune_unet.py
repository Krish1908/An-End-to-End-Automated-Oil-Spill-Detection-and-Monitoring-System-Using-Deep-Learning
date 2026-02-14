# /content/drive/MyDrive/OIL-SPILL-8/src/training/fine_tune_unet.py

import os
import sys
import tensorflow as tf

# ---------------------------------------------------
# ADD PROJECT ROOT TO PATH
# ---------------------------------------------------
PROJECT_ROOT = "/content/drive/MyDrive/OIL-SPILL-8"
sys.path.append(PROJECT_ROOT)

from src.models.unet import build_unet, bce_dice_loss, dice_coef
from src.data.dataloader import create_unet_dataset

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-5  # low LR for fine-tuning
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "models/unet/unet_finetuned.keras")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
print("🔹 Loading datasets...")
train_ds = create_unet_dataset("train", batch_size=BATCH_SIZE, augment=True)
val_ds   = create_unet_dataset("val", batch_size=BATCH_SIZE, augment=False)

print("Train batches:", len(train_ds))
print("Val batches:", len(val_ds))

# ---------------------------------------------------
# BUILD MODEL
# ---------------------------------------------------
print("🔹 Building U-Net...")
model = build_unet()

# OPTIONAL: Load existing weights if present
PRETRAINED_PATH = os.path.join(PROJECT_ROOT, "models/unet/unet_segmentation.keras")

if os.path.exists(PRETRAINED_PATH):
    print("🔹 Loading pretrained weights...")
    model.load_weights(PRETRAINED_PATH)
else:
    print("⚠ No pretrained model found. Training from scratch.")

# ---------------------------------------------------
# COMPILE
# ---------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)

# ---------------------------------------------------
# CALLBACKS
# ---------------------------------------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_dice_coef",
        save_best_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_dice_coef",
        patience=5,
        mode="max",
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef",
        factor=0.5,
        patience=3,
        mode="max",
        verbose=1
    )
]

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
print("🚀 Starting Fine-Tuning...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Fine-tuning completed.")
print("Best model saved at:", MODEL_SAVE_PATH)
