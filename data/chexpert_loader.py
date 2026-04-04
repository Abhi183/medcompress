"""
data/chexpert_loader.py
-----------------------
CheXpert chest X-ray multi-label classification dataset loader.
Dataset: Stanford CheXpert (Kaggle: stanfordmlgroup/chexpert)

Expected directory structure:
    data/chexpert/
        train/          <- images referenced by train.csv Path column
        valid/          <- images referenced by valid.csv Path column
        train.csv       <- columns: Path, Sex, Age, Frontal/Lateral, AP/PA, + 14 pathology labels
        valid.csv       <- same columns as train.csv

Label policy:
    - Uncertain (-1) -> 1  (U-Ones)
    - Blank / NaN    -> 0  (negative)
"""

import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


COMPETITION_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class CheXpertDataset:
    """Loads and preprocesses CheXpert chest X-rays for TF training."""

    def __init__(self, config: dict):
        self.root = config["data"]["root"]
        self.image_size = config["data"]["image_size"]
        self.batch_size = config["data"]["batch_size"]
        self.val_split = config["data"]["val_split"]
        self.augmentation = config["data"].get("augmentation", False)

        self.train_csv = os.path.join(self.root, "train.csv")
        self.valid_csv = os.path.join(self.root, "valid.csv")

        self._load_and_clean()
        self._build_splits()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def competition_labels(self) -> List[str]:
        """Return the 5 CheXpert competition label names."""
        return list(COMPETITION_LABELS)

    # ------------------------------------------------------------------ #
    #  Data loading & label policy                                         #
    # ------------------------------------------------------------------ #

    def _load_and_clean(self):
        """Read CSVs, apply U-Ones policy, and resolve image paths."""
        self.train_df_full = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.valid_csv)

        for df in (self.train_df_full, self.test_df):
            # U-Ones: treat uncertain (-1) as positive (1)
            for label in COMPETITION_LABELS:
                df[label] = df[label].fillna(0.0)
                df[label] = df[label].replace(-1.0, 1.0).astype(np.float32)

            # Resolve absolute image paths
            df["abs_path"] = df["Path"].apply(
                lambda p: os.path.join(self.root, p)
            )

        print(
            f"[CheXpertDataset] Loaded — "
            f"train_full: {len(self.train_df_full)}, "
            f"test (valid.csv): {len(self.test_df)}"
        )

    # ------------------------------------------------------------------ #
    #  Train / val splits (from train.csv)                                 #
    # ------------------------------------------------------------------ #

    def _build_splits(self):
        """Split train.csv into train and val (stratified on Atelectasis)."""
        # Use the first competition label for stratification since
        # multi-label stratification is non-trivial; Atelectasis has
        # reasonable class balance for this purpose.
        stratify_col = self.train_df_full["Atelectasis"].astype(int)

        train_df, val_df = train_test_split(
            self.train_df_full,
            test_size=self.val_split,
            stratify=stratify_col,
            random_state=42,
        )
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        print(
            f"[CheXpertDataset] Splits — "
            f"train: {len(self.train_df)}, "
            f"val: {len(self.val_df)}, "
            f"test: {len(self.test_df)}"
        )

    # ------------------------------------------------------------------ #
    #  tf.data pipeline                                                    #
    # ------------------------------------------------------------------ #

    def _load_image(self, path, labels):
        """Read, decode, resize, and normalize a single chest X-ray."""
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=1)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        # Convert grayscale to 3-channel by repeating
        img = tf.repeat(img, repeats=3, axis=-1)
        img = tf.cast(img, tf.float32) / 255.0
        # Normalize to [-1, 1]
        img = (img - 0.5) / 0.5
        return img, labels

    def _augment(self, img, labels):
        """Apply training-time augmentations."""
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, labels

    def _make_dataset(
        self, df: pd.DataFrame, augment: bool
    ) -> tf.data.Dataset:
        paths = df["abs_path"].tolist()
        labels = df[COMPETITION_LABELS].values.astype(np.float32)
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
        subset = self.val_df.sample(
            n=min(n_samples, len(self.val_df)), random_state=0
        )
        ds = self._make_dataset(subset, augment=False)
        for images, _ in ds:
            yield [images]
