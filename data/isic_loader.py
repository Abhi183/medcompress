"""
data/isic_loader.py
--------------------
ISIC 2020 Melanoma Classification dataset loader.
Dataset: https://www.kaggle.com/c/siim-isic-melanoma-classification

Expected directory structure:
    data/isic/
        train/          <- JPEG images
        test/           <- JPEG images (no labels)
        train.csv       <- columns: image_name, target (0=benign, 1=melanoma)
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf


class ISICDataset:
    """Loads and preprocesses ISIC 2020 images for TF training."""

    def __init__(self, config: dict):
        self.root = config["data"]["root"]
        self.image_size = config["data"]["image_size"]
        self.batch_size = config["data"]["batch_size"]
        self.val_split = config["data"]["val_split"]
        self.test_split = config["data"]["test_split"]
        self.augmentation = config["data"].get("augmentation", False)

        self.df = pd.read_csv(os.path.join(self.root, "train.csv"))
        self._compute_class_weights()
        self._build_splits()

    # ------------------------------------------------------------------ #
    #  Class imbalance                                                     #
    # ------------------------------------------------------------------ #

    def _compute_class_weights(self):
        """Compute inverse-frequency class weights for imbalanced melanoma data."""
        counts = self.df["target"].value_counts()
        total = len(self.df)
        self.class_weights = {
            int(cls): total / (len(counts) * cnt)
            for cls, cnt in counts.items()
        }
        print(f"[ISICDataset] Class weights: {self.class_weights}")

    # ------------------------------------------------------------------ #
    #  Train / val / test splits                                           #
    # ------------------------------------------------------------------ #

    def _build_splits(self):
        train_df, test_df = train_test_split(
            self.df, test_size=self.test_split,
            stratify=self.df["target"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_df, test_size=self.val_split / (1 - self.test_split),
            stratify=train_df["target"], random_state=42
        )
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        print(
            f"[ISICDataset] Splits — "
            f"train: {len(self.train_df)}, "
            f"val: {len(self.val_df)}, "
            f"test: {len(self.test_df)}"
        )

    # ------------------------------------------------------------------ #
    #  tf.data pipeline                                                    #
    # ------------------------------------------------------------------ #

    def _load_image(self, path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.cast(img, tf.float32) / 255.0
        # EfficientNet preprocessing (zero-center to [-1, 1])
        img = (img - 0.5) / 0.5
        return img, label

    def _augment(self, img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    def _make_dataset(self, df: pd.DataFrame, augment: bool) -> tf.data.Dataset:
        paths = [
            os.path.join(self.root, "train", f"{name}.jpg")
            for name in df["image_name"]
        ]
        labels = df["target"].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.train_df, augment=self.augmentation)

    def get_val_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.val_df, augment=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.test_df, augment=False)

    # ------------------------------------------------------------------ #
    #  Calibration dataset for PTQ / QAT                                  #
    # ------------------------------------------------------------------ #

    def get_calibration_generator(self, n_samples: int = 200):
        """Yields batches of representative images for TFLite INT8 calibration."""
        subset = self.val_df.sample(n=min(n_samples, len(self.val_df)), random_state=0)
        ds = self._make_dataset(subset, augment=False)
        for images, _ in ds:
            yield [images]
