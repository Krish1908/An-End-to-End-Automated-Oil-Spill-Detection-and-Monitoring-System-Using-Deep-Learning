# D:\Coding\SEM-8-NEW\OIL-SPILL\src\data\dataloader.py

import os
import numpy as np
import tensorflow as tf

# ---------------------------------------------------
# ABSOLUTE PATH FOR PROCESSED DATASET (Google Drive)
# ---------------------------------------------------
PROCESSED_DIR = r"D:\Coding\SEM-8-NEW\OIL-SPILL\src\data\processed"
IMG_SIZE = (256, 256)

# ---------------------------------------------------
# Helper: load npy file
# ---------------------------------------------------
def load_npy(path):
    return np.load(path)

# ---------------------------------------------------
# CNN Dataset (Classification)
# ---------------------------------------------------
def create_cnn_dataset(split, batch_size=16, augment=False):
    """
    Returns tf.data.Dataset of (image, label)

    Label definition:
    0 → Mostly water (normal marine scene)
    1 → Significant non-water presence (ships / land / structures)
    """

    img_dir = os.path.join(PROCESSED_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DIR, split, "masks")

    img_files = sorted(os.listdir(img_dir))

    image_paths = [os.path.join(img_dir, f) for f in img_files]
    mask_paths = [os.path.join(mask_dir, f) for f in img_files]

    def load_item(img_path, mask_path):
        img = load_npy(img_path.numpy().decode())
        mask = load_npy(mask_path.numpy().decode())

        # 🔑 IMPORTANT: WATER CLASS ID (verify once visually)
        WATER_CLASS = 174   # change ONLY if your mask visualization proves otherwise

        non_water_ratio = np.sum(mask != WATER_CLASS) / mask.size
        label = 1 if non_water_ratio > 0.05 else 0

        return img.astype("float32"), np.array(label, dtype="float32")

    def tf_wrapper(img_path, mask_path):
        img, label = tf.py_function(
            load_item,
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape((*IMG_SIZE, 3))
        label.set_shape(())
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_cnn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------------------------------------
# CNN Augmentation
# ---------------------------------------------------
def augment_cnn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

# ---------------------------------------------------
# U-Net Dataset (UNCHANGED — handled later)
# ---------------------------------------------------
def create_unet_dataset(split, batch_size=8, augment=False):
    """
    Returns tf.data.Dataset of (image, mask)
    """
    img_dir = os.path.join(PROCESSED_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DIR, split, "masks")

    img_files = sorted(os.listdir(img_dir))

    image_paths = [os.path.join(img_dir, f) for f in img_files]
    mask_paths = [os.path.join(mask_dir, f) for f in img_files]

    def load_item(img_path, mask_path):
        img = load_npy(img_path.numpy().decode())
        mask = load_npy(mask_path.numpy().decode())

        mask = np.expand_dims(mask, axis=-1)  # (256,256,1)
        return img.astype("float32"), mask.astype("float32")

    def tf_wrapper(img_path, mask_path):
        img, mask = tf.py_function(
            load_item,
            inp=[img_path, mask_path],
            Tout=[tf.float32, tf.float32]
        )
        img.set_shape((*IMG_SIZE, 3))
        mask.set_shape((*IMG_SIZE, 1))
        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(tf_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_unet, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(500).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------------------------------------
# U-Net Augmentation
# ---------------------------------------------------
def augment_unet(img, mask):
    img = tf.image.random_flip_left_right(img)
    mask = tf.image.random_flip_left_right(mask)

    img = tf.image.random_flip_up_down(img)
    mask = tf.image.random_flip_up_down(mask)

    return img, mask

# ---------------------------------------------------
# Debug Test
# ---------------------------------------------------
if __name__ == "__main__":
    print("Testing CNN dataloader...")
    train_cnn = create_cnn_dataset("train", augment=True)
    for img, label in train_cnn.take(1):
        print("CNN batch:", img.shape, label.numpy())

    print("\nTesting UNET dataloader...")
    train_unet = create_unet_dataset("train", augment=True)
    for img, mask in train_unet.take(1):
        print("UNET batch:", img.shape, mask.shape)
