# src/models/unet.py

# D1 SPECIFIC UNET ARCHITECTURE

import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------------------------------------------------
# CONVOLUTION BLOCK
# ---------------------------------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

# ---------------------------------------------------
# ENCODER BLOCK
# ---------------------------------------------------
def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(f)
    return f, p

# ---------------------------------------------------
# DECODER BLOCK
# ---------------------------------------------------
def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

# ---------------------------------------------------
# DICE METRIC
# ---------------------------------------------------
def dice_coef(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])

    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))

# ---------------------------------------------------
# DICE LOSS
# ---------------------------------------------------
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# ---------------------------------------------------
# COMBINED LOSS (BEST PRACTICE)
# ---------------------------------------------------
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ---------------------------------------------------
# U-NET MODEL
# ---------------------------------------------------
def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="OilSpill_UNet")
    return model

# ---------------------------------------------------
# DEBUG TEST
# ---------------------------------------------------
if __name__ == "__main__":
    model = build_unet()
    model.summary()
