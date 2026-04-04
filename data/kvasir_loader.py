"""
data/kvasir_loader.py
----------------------
Kvasir-SEG polyp segmentation dataset loader.
Dataset: https://datasets.simula.no/kvasir-seg/

Expected directory structure:
    data/kvasir-seg/
        images/          <- RGB JPEG/PNG polyp endoscopy images
        masks/           <- Binary segmentation masks (white=polyp, black=background)

Notes:
    - 1000 image/mask pairs, varying sizes (332x487 to 1920x1072)
    - Masks share the same filename as corresponding images
    - Task: binary segmentation (polyp vs background)
    - Output: single-channel sigmoid (not 2-channel softmax)
"""

import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


NUM_CLASSES = 2  # polyp vs background (single-channel sigmoid output)
IMAGE_SIZE_DEFAULT = 256


class KvasirSEGDataset:
    """Loads and preprocesses Kvasir-SEG images + masks for TF training."""

    def __init__(self, config: dict):
        self.root = config["data"]["root"]
        self.image_size = config["data"].get("image_size", IMAGE_SIZE_DEFAULT)
        self.batch_size = config["data"]["batch_size"]
        self.val_split = config["data"]["val_split"]
        self.test_split = config["data"]["test_split"]
        self.augmentation = config["data"].get("augmentation", True)
        self.seed = config.get("seed", 42)

        self._discover_samples()
        self._build_splits()

    # ------------------------------------------------------------------ #
    #  Sample discovery                                                    #
    # ------------------------------------------------------------------ #

    def _discover_samples(self):
        """Find matched image/mask pairs by filename."""
        image_dir = os.path.join(self.root, "images")
        mask_dir = os.path.join(self.root, "masks")

        image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg"))
            + glob.glob(os.path.join(image_dir, "*.png"))
            + glob.glob(os.path.join(image_dir, "*.jpeg"))
        )

        self.image_paths = []
        self.mask_paths = []

        for img_path in image_paths:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            # Try common mask extensions
            mask_path = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = os.path.join(mask_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break

            if mask_path is not None:
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)

        print(f"[KvasirSEGDataset] Found {len(self.image_paths)} image/mask pairs")

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No image/mask pairs found. Check that {image_dir} and "
                f"{mask_dir} exist and contain matching filenames."
            )

    # ------------------------------------------------------------------ #
    #  Train / val / test splits                                           #
    # ------------------------------------------------------------------ #

    def _build_splits(self):
        indices = np.arange(len(self.image_paths))

        train_idx, test_idx = train_test_split(
            indices, test_size=self.test_split, random_state=self.seed
        )
        val_ratio = self.val_split / (1 - self.test_split)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=val_ratio, random_state=self.seed
        )

        self.train_pairs = [
            (self.image_paths[i], self.mask_paths[i]) for i in train_idx
        ]
        self.val_pairs = [
            (self.image_paths[i], self.mask_paths[i]) for i in val_idx
        ]
        self.test_pairs = [
            (self.image_paths[i], self.mask_paths[i]) for i in test_idx
        ]

        print(
            f"[KvasirSEGDataset] Splits — "
            f"train: {len(self.train_pairs)}, "
            f"val: {len(self.val_pairs)}, "
            f"test: {len(self.test_pairs)}"
        )

    # ------------------------------------------------------------------ #
    #  Single-sample loading (tf.data map function)                        #
    # ------------------------------------------------------------------ #

    def _load_sample(self, image_path: tf.Tensor, mask_path: tf.Tensor):
        """Load and preprocess one image/mask pair inside tf.data pipeline."""
        # --- Image ---
        raw_img = tf.io.read_file(image_path)
        img = tf.image.decode_image(raw_img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.cast(img, tf.float32) / 255.0  # normalize to [0, 1]

        # --- Mask ---
        raw_mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(raw_mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [self.image_size, self.image_size], method="nearest")
        # Binarize: threshold at 128 (mask pixel > 0.5 after /255)
        mask = tf.cast(mask, tf.float32) / 255.0
        mask = tf.cast(mask > 0.5, tf.float32)  # binary {0.0, 1.0}, shape (H, W, 1)

        return img, mask

    # ------------------------------------------------------------------ #
    #  Augmentation                                                        #
    # ------------------------------------------------------------------ #

    def _augment(self, image: tf.Tensor, mask: tf.Tensor):
        """Apply identical spatial augmentations to image and mask."""
        # Concatenate along channels so spatial transforms apply identically
        combined = tf.concat([image, mask], axis=-1)  # (H, W, 4)

        combined = tf.image.random_flip_left_right(combined)
        combined = tf.image.random_flip_up_down(combined)

        image = combined[:, :, :3]
        mask = combined[:, :, 3:]

        # Brightness (image only, not mask)
        image = tf.image.random_brightness(image, max_delta=0.15)
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, mask

    # ------------------------------------------------------------------ #
    #  tf.data pipeline                                                    #
    # ------------------------------------------------------------------ #

    def _make_dataset(
        self, pairs: list[tuple[str, str]], shuffle: bool = False, augment: bool = False
    ) -> tf.data.Dataset:
        image_paths = [p[0] for p in pairs]
        mask_paths = [p[1] for p in pairs]

        ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

        if shuffle:
            ds = ds.shuffle(buffer_size=len(pairs), seed=self.seed)

        ds = ds.map(self._load_sample, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.train_pairs, shuffle=True, augment=self.augmentation)

    def get_val_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.val_pairs, shuffle=False, augment=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.test_pairs, shuffle=False, augment=False)

    # ------------------------------------------------------------------ #
    #  Calibration dataset for PTQ / QAT                                   #
    # ------------------------------------------------------------------ #

    def get_calibration_generator(self, n_samples: int = 100):
        """Yield batches of representative images for TFLite INT8 calibration."""
        rng = np.random.RandomState(0)
        n = min(n_samples, len(self.val_pairs))
        selected = rng.choice(len(self.val_pairs), size=n, replace=False)
        subset_pairs = [self.val_pairs[i] for i in selected]

        ds = self._make_dataset(subset_pairs, shuffle=False, augment=False)
        for images, _ in ds:
            yield [images]
