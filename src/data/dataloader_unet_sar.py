# /content/drive/MyDrive/OIL-SPILL-8/src/data/dataloader_unet.py
# U-NET DATALOADER FOR DATASET-D3

import os
import tensorflow as tf

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

PROCESSED_DIR = "/content/drive/MyDrive/OIL-SPILL-8/src/data/processed-d3/cnn_unet"

IMG_SIZE = (256, 256)

AUTOTUNE = tf.data.AUTOTUNE


# ---------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------

def load_image(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)

    img = tf.cast(img, tf.float32) / 255.0

    return img


# ---------------------------------------------------
# LOAD MASK
# ---------------------------------------------------

def load_mask(mask_path):

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)

    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")

    mask = tf.cast(mask, tf.float32) / 255.0

    # convert to binary mask
    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return mask


# ---------------------------------------------------
# PAIR AUGMENTATION (image + mask)
# ---------------------------------------------------

def augment(img, mask):

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    return img, mask


# ---------------------------------------------------
# CREATE DATASET
# ---------------------------------------------------

def create_dataset(split, batch_size=8, augment_data=False):

    img_dir = os.path.join(PROCESSED_DIR, split, "images")
    mask_dir = os.path.join(PROCESSED_DIR, split, "masks")

    img_files = sorted(os.listdir(img_dir))

    image_paths = []
    mask_paths = []

    for fname in img_files:

        img_path = os.path.join(img_dir, fname)

        mask_name = os.path.splitext(fname)[0] + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):

            image_paths.append(img_path)
            mask_paths.append(mask_path)

    image_paths = tf.constant(image_paths)
    mask_paths = tf.constant(mask_paths)


    def load_item(img_path, mask_path):

        img = load_image(img_path)
        mask = load_mask(mask_path)

        return img, mask


    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    dataset = dataset.map(load_item, num_parallel_calls=AUTOTUNE)

    if augment_data and split == "train":
        dataset = dataset.map(augment, num_parallel_calls=AUTOTUNE)

    if split == "train":
        dataset = dataset.shuffle(500)

    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset


# ---------------------------------------------------
# DEBUG TEST
# ---------------------------------------------------

if __name__ == "__main__":

    print("Testing UNet dataloader...")

    train_ds = create_dataset("train", batch_size=8, augment_data=True)

    for imgs, masks in train_ds.take(1):

        print("Images shape:", imgs.shape)
        print("Masks shape:", masks.shape)