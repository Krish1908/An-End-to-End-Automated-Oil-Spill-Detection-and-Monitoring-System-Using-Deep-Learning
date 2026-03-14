# ============================================================
# TRAIN UNET SAR SEGMENTATION (SINGLE FILE FOR KAGGLE)
# ============================================================

import os
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# GPU CONFIG
# ============================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs:", gpus)

# ============================================================
# DATASET PATH (KAGGLE INPUT)
# ============================================================

PROCESSED_DIR = "/kaggle/input/datasets/sanjaykrishnatn/processed-oil-spill/processed-d3/processed-d3/cnn_unet"
IMG_SIZE = (256,256)
AUTOTUNE = tf.data.AUTOTUNE

# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(img_path):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)

    img = tf.image.resize(img, IMG_SIZE)

    img = tf.cast(img, tf.float32) / 255.0

    return img


def load_mask(mask_path):

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)

    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")

    mask = tf.cast(mask, tf.float32) / 255.0

    mask = tf.where(mask > 0.5, 1.0, 0.0)

    return mask


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment(img, mask):

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    return img, mask


# ============================================================
# CREATE DATASET
# ============================================================

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


# ============================================================
# UNET MODEL
# ============================================================

from tensorflow.keras import layers, Model


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2,2))(f)
    return f, p


def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def dice_coef(y_true, y_pred, smooth=1):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])

    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):

    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)

    return bce + dice_loss(y_true, y_pred)


def build_unet(input_shape=(256,256,3)):

    inputs = layers.Input(shape=input_shape)

    s1, p1 = encoder_block(inputs,64)
    s2, p2 = encoder_block(p1,128)
    s3, p3 = encoder_block(p2,256)
    s4, p4 = encoder_block(p3,512)

    b = conv_block(p4,1024)

    d1 = decoder_block(b,s4,512)
    d2 = decoder_block(d1,s3,256)
    d3 = decoder_block(d2,s2,128)
    d4 = decoder_block(d3,s1,64)

    outputs = layers.Conv2D(1,1,activation="sigmoid")(d4)

    model = Model(inputs,outputs,name="OilSpill_UNet")

    return model


# ============================================================
# TRAIN CONFIG
# ============================================================

BATCH_SIZE = 8
EPOCHS = 40
LR = 1e-4

OUTPUT_DIR = "/kaggle/working/unet_output"
os.makedirs(OUTPUT_DIR,exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR,"unet_sar.keras")


# ============================================================
# LOAD DATA
# ============================================================

print("Loading datasets...")

train_ds = create_dataset("train",BATCH_SIZE,augment_data=True)
val_ds = create_dataset("val",BATCH_SIZE,augment_data=False)


# ============================================================
# BUILD MODEL
# ============================================================

print("Building UNet...")

model = build_unet()

model.compile(

    optimizer=tf.keras.optimizers.Adam(LR),

    loss=bce_dice_loss,

    metrics=[
        "accuracy",
        dice_coef
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

callbacks=[

    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_dice_coef",
        mode="max",
        save_best_only=True,
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


# ============================================================
# TRAIN
# ============================================================

print("Training UNet...")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=callbacks
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)


# ============================================================
# PLOTS
# ============================================================

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("UNet Loss")
plt.savefig(os.path.join(OUTPUT_DIR,"unet_loss.png"))
plt.close()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("UNet Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR,"unet_accuracy.png"))
plt.close()

plt.figure()
plt.plot(history.history["dice_coef"])
plt.plot(history.history["val_dice_coef"])
plt.title("UNet Dice")
plt.savefig(os.path.join(OUTPUT_DIR,"unet_dice.png"))
plt.close()


print("Training complete")
print("Saved to:", OUTPUT_DIR)