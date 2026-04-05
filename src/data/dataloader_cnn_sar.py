# /content/drive/MyDrive/OIL-SPILL-8/src/data/dataloader_cnn.py
# CNN DATALOADER FOR DATASET-D4

import os
import tensorflow as tf
import numpy as np
import json

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

PROCESSED_DIR = "src/data/processed-d4"
IMG_SIZE = (256, 256)

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------

def load_image(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=1, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)

    img = tf.cast(img, tf.float32) / 255.0

    # convert grayscale → 3 channels (better for CNNs)
    img = tf.repeat(img, 3, axis=-1)

    return img


# ---------------------------------------------------
# READ LABEL FILE
# ---------------------------------------------------

def load_labels(split):

    label_file = os.path.join(PROCESSED_DIR, split, "labels.txt")

    image_paths = []
    labels = []

    with open(label_file, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            fname, label = line.rsplit(" ", 1)

            img_path = os.path.join(PROCESSED_DIR, split, "images", fname)

            if os.path.exists(img_path):

                image_paths.append(img_path)
                labels.append(int(label))

    return image_paths, labels


# ---------------------------------------------------
# CNN AUGMENTATION
# ---------------------------------------------------

def augment(img, label):

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)

    return img, label


# ---------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------

def create_dataset(split, batch_size=16, augment_data=False):

    image_paths, labels = load_labels(split)

    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels, dtype=tf.int32)

    def load_item(img_path, label):

        img = load_image(img_path)

        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)

    if augment_data and split == "train":
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    if split == "train":
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset


# ---------------------------------------------------
# CLASS WEIGHTS
# ---------------------------------------------------

def load_class_weights():

    weight_file = os.path.join(PROCESSED_DIR, "class_weights.json")

    if not os.path.exists(weight_file):
        return None

    with open(weight_file) as f:
        weights = json.load(f)

    return {
        0: weights["class_0_no_oil"],
        1: weights["class_1_oil"]
    }


# ---------------------------------------------------
# DEBUG TEST
# ---------------------------------------------------

if __name__ == "__main__":

    print("Testing CNN dataloader...")

    train_ds = create_dataset("train", batch_size=8, augment_data=True)

    for imgs, labels in train_ds.take(1):

        print("Images shape:", imgs.shape)
        print("Labels:", labels.numpy())

    weights = load_class_weights()
    print("Class weights:", weights)