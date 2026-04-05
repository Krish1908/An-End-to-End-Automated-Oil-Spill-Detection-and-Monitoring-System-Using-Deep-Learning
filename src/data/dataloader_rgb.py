# /content/drive/MyDrive/OIL-SPILL-8/src/data/dataloader.py

# D1 SPECIFIC DATALOADER

import os
import tensorflow as tf
import numpy as np

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

PROCESSED_DIR = "src/data/processed-d1"
IMG_SIZE = (256, 256)

AUTOTUNE = tf.data.AUTOTUNE

# ---------------------------------------------------
# COMMON IMAGE LOADER
# ---------------------------------------------------

def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img

# ---------------------------------------------------
# LOAD CNN LABELS (labels.txt)
# ---------------------------------------------------

def load_cnn_labels(split):
    label_file = os.path.join(PROCESSED_DIR, split, "labels.txt")
    label_map = {}

    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            label = int(parts[-1])
            fname = " ".join(parts[:-1])

            label_map[fname] = label

    return label_map


# ---------------------------------------------------
# CNN DATASET (IMAGE + LABEL ONLY)
# ---------------------------------------------------

def create_cnn_dataset(split, batch_size=16, augment=False):
    """
    Returns: tf.data.Dataset (image, label)

    CNN sees:
    - Image
    - Image-level oil label

    CNN ignores:
    - Masks
    """

    img_dir = os.path.join(PROCESSED_DIR, split, "images")
    label_map = load_cnn_labels(split)

    image_paths = []
    labels = []

    for fname, label in label_map.items():
        img_path = os.path.join(img_dir, fname)
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(label)

    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels, dtype=tf.int32)

    def load_item(img_path, label):
        img = load_image(img_path)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_cnn, num_parallel_calls=AUTOTUNE)

    dataset = dataset.shuffle(500).batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# ---------------------------------------------------
# CNN AUGMENTATION
# ---------------------------------------------------

def augment_cnn(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, label

# ---------------------------------------------------
# U-NET DATASET (IMAGE + MASK)
# ---------------------------------------------------

def create_unet_dataset(split, batch_size=8, augment=False, min_oil_ratio=0.01):
    """
    Returns: tf.data.Dataset (image, mask)

    U-Net sees:
    - Image
    - Binary oil mask

    Filters:
    - Removes pure black / pure white masks
    """

    img_dir = os.path.join(PROCESSED_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DIR, split, "masks")

    img_files = sorted(os.listdir(img_dir))

    image_paths = []
    mask_paths = []

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        mask_path = os.path.join(mask_dir, os.path.splitext(fname)[0] + ".png")

        if not os.path.exists(mask_path):
            continue

        # quick oil-pixel filter (numpy, once)
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.cast(mask, tf.float32) / 255.0
        oil_ratio = tf.reduce_mean(mask)

        if oil_ratio > min_oil_ratio and oil_ratio < 0.95:
            image_paths.append(img_path)
            mask_paths.append(mask_path)

    image_paths = tf.constant(image_paths)
    mask_paths = tf.constant(mask_paths)

    def load_item(img_path, mask_path):
        img = load_image(img_path)

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
        mask = tf.cast(mask, tf.float32) / 255.0

        return img, mask

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_unet, num_parallel_calls=AUTOTUNE)

    dataset = dataset.shuffle(200).batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# ---------------------------------------------------
# U-NET AUGMENTATION
# ---------------------------------------------------

def augment_unet(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    return img, mask

# ---------------------------------------------------
# DEBUG TEST
# ---------------------------------------------------

if __name__ == "__main__":
    print("🔹 Testing CNN dataloader...")
    cnn_ds = create_cnn_dataset("train", augment=True)
    for img, label in cnn_ds.take(1):
        print("CNN batch:", img.shape, label.numpy())

    print("\n🔹 Testing U-Net dataloader...")
    unet_ds = create_unet_dataset("train", augment=True)
    for img, mask in unet_ds.take(1):
        print("U-Net batch:", img.shape, mask.shape)
