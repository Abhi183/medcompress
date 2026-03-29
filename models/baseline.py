"""
models/baseline.py
-------------------
Baseline model architectures for ISIC (classification) and BraTS (segmentation).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =========================================================================== #
#  ISIC: EfficientNetB0 classifier                                            #
# =========================================================================== #

def build_efficientnetb0(num_classes: int = 1, dropout: float = 0.3,
                          input_shape: tuple = (224, 224, 3)) -> keras.Model:
    """
    EfficientNetB0 fine-tuned for binary melanoma classification.
    Unfreeze the top 20 layers of the backbone for domain adaptation.
    """
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling=None,
    )
    # Freeze all but the last 20 backbone layers
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    inputs = keras.Input(shape=input_shape, name="image_input")
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout / 2)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
        loss = "binary_crossentropy"
        metrics = [keras.metrics.AUC(name="auc"), "accuracy"]
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    model = keras.Model(inputs, outputs, name="efficientnetb0_isic")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=loss,
        metrics=metrics,
    )
    return model


# =========================================================================== #
#  BraTS: 2.5D U-Net segmentation                                             #
# =========================================================================== #

def conv_block(x, filters: int, name: str):
    """Two consecutive Conv2D → BN → ReLU."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                       name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def encoder_block(x, filters: int, name: str):
    """Conv block + max pooling → returns (skip, pooled)."""
    skip = conv_block(x, filters, name=f"{name}_block")
    pooled = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters: int, name: str):
    """Upsample + concatenate skip + conv block."""
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_block")
    return x


def build_unet_full(num_classes: int = 4, n_channels: int = 12,
                     input_size: int = 128) -> keras.Model:
    """
    Full-resolution 2.5D U-Net: 4 encoder stages, filters 64→128→256→512.
    Input: (batch, H, W, n_slices * n_modalities)
    Output: (batch, H, W, num_classes) softmax segmentation map.
    """
    inputs = keras.Input(shape=(input_size, input_size, n_channels), name="slice_input")

    s1, p1 = encoder_block(inputs, 64,  name="enc1")
    s2, p2 = encoder_block(p1,     128, name="enc2")
    s3, p3 = encoder_block(p2,     256, name="enc3")
    s4, p4 = encoder_block(p3,     512, name="enc4")

    bottleneck = conv_block(p4, 1024, name="bottleneck")

    x = decoder_block(bottleneck, s4, 512, name="dec4")
    x = decoder_block(x,          s3, 256, name="dec3")
    x = decoder_block(x,          s2, 128, name="dec2")
    x = decoder_block(x,          s1, 64,  name="dec1")

    outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                             name="seg_output")(x)

    model = keras.Model(inputs, outputs, name="unet_full")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=dice_ce_loss(num_classes),
        metrics=[DiceCoefficient(num_classes, name="dice")],
    )
    return model


def build_unet_lite(num_classes: int = 4, n_channels: int = 12,
                     input_size: int = 128) -> keras.Model:
    """
    Lightweight 2.5D U-Net: 3 encoder stages, filters 32→64→128.
    ~8× fewer parameters than unet_full. Used as KD student.
    """
    inputs = keras.Input(shape=(input_size, input_size, n_channels), name="slice_input")

    s1, p1 = encoder_block(inputs, 32,  name="enc1")
    s2, p2 = encoder_block(p1,     64,  name="enc2")
    s3, p3 = encoder_block(p2,     128, name="enc3")

    bottleneck = conv_block(p3, 256, name="bottleneck")

    x = decoder_block(bottleneck, s3, 128, name="dec3")
    x = decoder_block(x,          s2, 64,  name="dec2")
    x = decoder_block(x,          s1, 32,  name="dec1")

    outputs = layers.Conv2D(num_classes, 1, activation="softmax",
                             name="seg_output")(x)

    model = keras.Model(inputs, outputs, name="unet_lite")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=dice_ce_loss(num_classes),
        metrics=[DiceCoefficient(num_classes, name="dice")],
    )
    return model


# =========================================================================== #
#  Loss & metric utilities                                                     #
# =========================================================================== #

def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice loss averaged over classes (expects one-hot y_true)."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1, y_true.shape[-1]]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1, y_pred.shape[-1]])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    dice = (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + smooth
    )
    return 1.0 - tf.reduce_mean(dice)


def dice_ce_loss(num_classes: int):
    """Combined Dice + CrossEntropy loss (standard for medical segmentation)."""
    cce = keras.losses.CategoricalCrossentropy()

    def loss_fn(y_true, y_pred):
        return dice_loss(y_true, y_pred) + cce(y_true, y_pred)

    loss_fn.__name__ = "dice_ce_loss"
    return loss_fn


class DiceCoefficient(keras.metrics.Metric):
    """Mean Dice coefficient across classes (excludes background class 0)."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dice_sum = self.add_weight("dice_sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_oh = tf.one_hot(tf.argmax(y_pred, axis=-1), self.num_classes)
        # Exclude background (class 0)
        y_true_fg = y_true[..., 1:]
        y_pred_fg = y_pred_oh[..., 1:]
        smooth = 1e-6
        intersection = tf.reduce_sum(y_true_fg * y_pred_fg)
        union = tf.reduce_sum(y_true_fg) + tf.reduce_sum(y_pred_fg)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / self.count

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)
