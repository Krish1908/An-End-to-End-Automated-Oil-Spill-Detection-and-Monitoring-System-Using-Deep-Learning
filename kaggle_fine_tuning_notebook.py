import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

# ---------------------------------------------------
# CONFIGURATION - Edit these paths if needed
# ---------------------------------------------------

PROCESSED_DIR       = "/kaggle/input/datasets/sanjaykrishnatn/oil-for-kaggle/OIL-SPILL/src/data/processed"
EXISTING_MODEL_PATH = "/kaggle/input/datasets/sanjaykrishnatn/oil-for-kaggle/OIL-SPILL/models/unet/unet_segmentation.keras"
MODEL_DIR           = "/kaggle/working/models/unet"

# test-new: images inside main dataset, masks in a SEPARATE dataset
TEST_NEW_IMG_DIR    = "/kaggle/input/datasets/sanjaykrishnatn/oil-for-kaggle/OIL-SPILL/src/data/processed/test-new/images"
TEST_NEW_MASK_DIR   = "/kaggle/input/datasets/sanjaykrishnatn/test-newmasks/masks"

IMG_SIZE   = (256, 256)
BATCH_SIZE = 8
EPOCHS     = 15
LR         = 5e-5
AUTOTUNE   = tf.data.AUTOTUNE

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------
# LOSSES & METRICS
# ---------------------------------------------------

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# ---------------------------------------------------
# DATA LOADING
# ---------------------------------------------------

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, IMG_SIZE) / 255.0
    return img

def load_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, IMG_SIZE) / 255.0
    return tf.cast(mask > 0.5, tf.float32)

def augment_fn(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)
    return image, mask

def make_dataset(img_dir, mask_dir, use_augmentation=False, min_oil_ratio=0.01):
    """
    Build a tf.data pipeline from explicit img_dir and mask_dir paths.
    This avoids any path-guessing and works for test-new whose masks
    live in a different Kaggle dataset than the images.
    """
    print(f"  img_dir : {img_dir}")
    print(f"  mask_dir: {mask_dir}")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image dir not found: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Mask dir not found:  {mask_dir}")

    img_files  = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    image_paths = [os.path.join(img_dir,  f) for f in img_files]
    mask_paths  = [os.path.join(mask_dir, f) for f in mask_files]

    print(f"  Found {len(image_paths)} images, {len(mask_paths)} masks")

    ds = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(image_paths),
        tf.data.Dataset.from_tensor_slices(mask_paths)
    ))
    ds = ds.map(lambda x, y: (load_image(x), load_mask(y)), num_parallel_calls=AUTOTUNE)

    if min_oil_ratio > 0:
        ds = ds.filter(lambda img, mask:
            tf.reduce_sum(mask) / tf.cast(tf.size(mask), tf.float32) >= min_oil_ratio)
        print(f"  Filtered: oil ratio >= {min_oil_ratio}")

    if use_augmentation:
        ds = ds.map(augment_fn, num_parallel_calls=AUTOTUNE)
        print("  Augmentation: ON")

    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# ---------------------------------------------------
# BUILD DATASETS
# ---------------------------------------------------

print("=" * 55)
print("📂 Building datasets...")
print("=" * 55)

print("\n[TRAIN]")
train_ds = make_dataset(
    img_dir          = os.path.join(PROCESSED_DIR, "train", "images"),
    mask_dir         = os.path.join(PROCESSED_DIR, "train", "masks"),
    use_augmentation = True
)

print("\n[VAL]")
val_ds = make_dataset(
    img_dir  = os.path.join(PROCESSED_DIR, "val", "images"),
    mask_dir = os.path.join(PROCESSED_DIR, "val", "masks")
)

print("\n[TEST]")
test_ds = make_dataset(
    img_dir  = os.path.join(PROCESSED_DIR, "test", "images"),
    mask_dir = os.path.join(PROCESSED_DIR, "test", "masks")
)

print("\n[TEST-NEW]  ← images and masks from different Kaggle datasets")
test_new_ds = make_dataset(
    img_dir  = TEST_NEW_IMG_DIR,
    mask_dir = TEST_NEW_MASK_DIR
)

# ---------------------------------------------------
# LOAD / BUILD MODEL
# ---------------------------------------------------

print("\n" + "=" * 55)
print("🏗️  Loading model...")
print("=" * 55)

if os.path.exists(EXISTING_MODEL_PATH):
    print(f"Loading existing model from:\n  {EXISTING_MODEL_PATH}")
    model = tf.keras.models.load_model(
        EXISTING_MODEL_PATH,
        custom_objects={"dice_coef": dice_coef, "bce_dice_loss": bce_dice_loss},
        compile=False
    )
    print("✅ Model loaded.")
else:
    print("No existing model found — building from scratch.")
    def build_unet():
        inputs = layers.Input(shape=(*IMG_SIZE, 3))
        x = layers.Conv2D(64,  3, activation='relu', padding='same')(inputs); x = layers.Conv2D(64,  3, activation='relu', padding='same')(x); s1 = x; x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x);      x = layers.Conv2D(128, 3, activation='relu', padding='same')(x); s2 = x; x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(256, 3, activation='relu', padding='same')(x);      x = layers.Conv2D(256, 3, activation='relu', padding='same')(x); s3 = x; x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(512, 3, activation='relu', padding='same')(x);      x = layers.Conv2D(512, 3, activation='relu', padding='same')(x); s4 = x; x = layers.MaxPooling2D(2)(x)
        x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x);     x = layers.Conv2D(1024, 3, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(x);  x = layers.Concatenate()([x, s4]); x = layers.Conv2D(512, 3, activation='relu', padding='same')(x); x = layers.Conv2D(512, 3, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(x);  x = layers.Concatenate()([x, s3]); x = layers.Conv2D(256, 3, activation='relu', padding='same')(x); x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(x);  x = layers.Concatenate()([x, s2]); x = layers.Conv2D(128, 3, activation='relu', padding='same')(x); x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(64,  2, strides=2, padding='same')(x);  x = layers.Concatenate()([x, s1]); x = layers.Conv2D(64,  3, activation='relu', padding='same')(x); x = layers.Conv2D(64,  3, activation='relu', padding='same')(x)
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)
        return Model(inputs, outputs)
    model = build_unet()
    print("✅ Model built from scratch.")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss=bce_dice_loss,
    metrics=[dice_coef]
)
print(f"✅ Compiled with lr={LR}")

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------

print("\n" + "=" * 55)
print("🚀 Starting fine-tuning...")
print("=" * 55)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath    = f"{MODEL_DIR}/unet_fine_tuned.keras",
        save_best_only = True,
        monitor     = "val_dice_coef",
        mode        = "max",
        verbose     = 1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_dice_coef",
        factor   = 0.5,
        patience = 3,
        mode     = "max",
        verbose  = 1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor             = "val_dice_coef",
        patience            = 5,
        mode                = "max",
        restore_best_weights = True,
        verbose             = 1
    )
]

history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS,
    callbacks       = callbacks
)

# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------

print("\n" + "=" * 55)
print("📊 Evaluation")
print("=" * 55)

orig_loss, orig_dice = model.evaluate(test_ds,     verbose=0)
new_loss,  new_dice  = model.evaluate(test_new_ds, verbose=0)

print(f"\n  Original test  → Loss: {orig_loss:.4f}  Dice: {orig_dice:.4f}")
print(f"  Test-new       → Loss: {new_loss:.4f}   Dice: {new_dice:.4f}")
print(f"  Improvement    → {new_dice - orig_dice:+.4f}")

# ---------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------

report = f"""# Fine-Tuning Results

| Dataset       | Loss   | Dice   |
|---------------|--------|--------|
| Original Test | {orig_loss:.4f} | {orig_dice:.4f} |
| Test-New      | {new_loss:.4f}  | {new_dice:.4f}  |

Improvement: {new_dice - orig_dice:+.4f}

Model saved to: {MODEL_DIR}/unet_fine_tuned.keras
Epochs run: {len(history.history['loss'])} / {EPOCHS}
"""
with open(f"{MODEL_DIR}/results.md", "w") as f:
    f.write(report)
print(f"\n📄 Report saved to {MODEL_DIR}/results.md")

# ---------------------------------------------------
# VISUALIZE
# ---------------------------------------------------

print("\n📈 Generating visualizations...")
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i, (imgs, masks) in enumerate(test_new_ds.take(3)):
    preds = model.predict(imgs, verbose=0)
    axes[i, 0].imshow(imgs[0].numpy());          axes[i, 0].set_title("Image");      axes[i, 0].axis('off')
    axes[i, 1].imshow(masks[0].numpy(), cmap='gray'); axes[i, 1].set_title("Ground Truth"); axes[i, 1].axis('off')
    axes[i, 2].imshow(preds[0].squeeze(), cmap='viridis'); axes[i, 2].set_title(f"Pred (Dice={new_dice:.3f})"); axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/visualization.png", dpi=150)
plt.show()
print(f"✅ Saved to {MODEL_DIR}/visualization.png")

print("\n🎉 All done!")