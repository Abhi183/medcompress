"""
data/brats_loader.py
---------------------
BraTS 2021 Brain Tumor Segmentation dataset loader (2.5D slice approach).
Dataset: https://www.synapse.org/#!Synapse:syn27046444/wiki/616992

Expected directory structure:
    data/brats/BraTS2021_Training_Data/
        BraTS2021_00000/
            BraTS2021_00000_t1.nii.gz
            BraTS2021_00000_t1ce.nii.gz
            BraTS2021_00000_t2.nii.gz
            BraTS2021_00000_flair.nii.gz
            BraTS2021_00000_seg.nii.gz
        ...

Label map (BraTS 2021):
    0: Background
    1: Necrotic Core (NCR)
    2: Edema (ED)
    4: Enhancing Tumor (ET)  ← note: label 3 is not used
"""

import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Remap BraTS labels {0,1,2,4} → {0,1,2,3} for one-hot encoding
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}
NUM_CLASSES = 4


def _normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Z-score normalize a single MRI volume (non-zero voxels only)."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol.astype(np.float32)
    mu, sigma = vol[mask].mean(), vol[mask].std()
    sigma = sigma if sigma > 0 else 1.0
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - mu) / sigma
    return out


def _remap_labels(seg: np.ndarray) -> np.ndarray:
    """Remap BraTS integer labels to contiguous 0–3 range."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in LABEL_REMAP.items():
        out[seg == src] = dst
    return out


class BraTSDataset:
    """
    Loads BraTS 2021 volumes and produces 2.5D axial slices for TF training.
    2.5D strategy: stack N adjacent slices across 4 modalities → input shape
    (H, W, N*4). This avoids full 3D convolutions while preserving some
    volumetric context, making models deployable on mobile/WASM.
    """

    def __init__(self, config: dict):
        self.root = config["data"]["root"]
        self.image_size = config["data"]["image_size"]
        self.patch_size = config["data"]["patch_size"]
        self.batch_size = config["data"]["batch_size"]
        self.n_slices = config["data"].get("n_slices", 3)   # must be odd
        self.val_split = config["data"]["val_split"]
        self.test_split = config["data"]["test_split"]
        self.modalities = config["data"].get("modalities", ["t1", "t1ce", "t2", "flair"])

        assert self.n_slices % 2 == 1, "n_slices must be odd (symmetric context)"
        self.half = self.n_slices // 2

        self._discover_cases()
        self._build_splits()

    # ------------------------------------------------------------------ #
    #  Case discovery                                                      #
    # ------------------------------------------------------------------ #

    def _discover_cases(self):
        pattern = os.path.join(self.root, "BraTS2021_*")
        self.case_dirs = sorted(glob.glob(pattern))
        print(f"[BraTSDataset] Found {len(self.case_dirs)} cases")

    def _build_splits(self):
        train_cases, test_cases = train_test_split(
            self.case_dirs, test_size=self.test_split, random_state=42
        )
        train_cases, val_cases = train_test_split(
            train_cases, test_size=self.val_split / (1 - self.test_split), random_state=42
        )
        self.train_cases = train_cases
        self.val_cases = val_cases
        self.test_cases = test_cases
        print(
            f"[BraTSDataset] Splits — "
            f"train: {len(train_cases)}, "
            f"val: {len(val_cases)}, "
            f"test: {len(test_cases)}"
        )

    # ------------------------------------------------------------------ #
    #  Volume loading & slice extraction                                   #
    # ------------------------------------------------------------------ #

    def _load_case(self, case_dir: str):
        """Load all modalities + segmentation for one case."""
        case_id = os.path.basename(case_dir)
        vols = []
        for mod in self.modalities:
            path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
            vol = nib.load(path).get_fdata().astype(np.float32)
            vols.append(_normalize_volume(vol))
        # Stack → (H, W, D, num_modalities)
        volume = np.stack(vols, axis=-1)

        seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.int32)
        seg = _remap_labels(seg)
        return volume, seg   # shapes: (240, 240, 155, 4), (240, 240, 155)

    def _extract_slices(self, volume: np.ndarray, seg: np.ndarray):
        """
        Extract 2.5D axial slices. For each axial index z (ignoring borders),
        stack [z-half .. z+half] slices across all modalities.
        Returns list of (image, mask) pairs; skips empty (all-background) slices.
        """
        D = volume.shape[2]
        slices = []
        for z in range(self.half, D - self.half):
            # Multi-slice input: (H, W, n_slices * num_modalities)
            img = volume[:, :, z - self.half: z + self.half + 1, :]  # (H, W, n, M)
            img = img.reshape(img.shape[0], img.shape[1], -1)         # (H, W, n*M)
            mask = seg[:, :, z]   # (H, W,)

            # Skip slices with no foreground (reduces class imbalance in batches)
            if mask.sum() == 0:
                continue

            # Resize to patch_size
            img_t = tf.image.resize(img, [self.patch_size, self.patch_size]).numpy()
            mask_t = tf.image.resize(
                mask[..., None].astype(np.float32),
                [self.patch_size, self.patch_size],
                method="nearest"
            ).numpy()[..., 0].astype(np.uint8)

            slices.append((img_t, mask_t))
        return slices

    # ------------------------------------------------------------------ #
    #  tf.data pipeline                                                    #
    # ------------------------------------------------------------------ #

    def _case_generator(self, case_dirs):
        """Python generator: yields (image_slice, one_hot_mask) pairs."""
        for case_dir in case_dirs:
            try:
                volume, seg = self._load_case(case_dir)
                for img, mask in self._extract_slices(volume, seg):
                    # One-hot encode mask → (H, W, NUM_CLASSES)
                    mask_oh = tf.one_hot(mask, NUM_CLASSES).numpy()
                    yield img.astype(np.float32), mask_oh.astype(np.float32)
            except Exception as e:
                print(f"[BraTSDataset] Skipping {case_dir}: {e}")

    def _make_dataset(self, case_dirs, shuffle: bool = False) -> tf.data.Dataset:
        n_channels = self.n_slices * len(self.modalities)
        output_signature = (
            tf.TensorSpec(shape=(self.patch_size, self.patch_size, n_channels), dtype=tf.float32),
            tf.TensorSpec(shape=(self.patch_size, self.patch_size, NUM_CLASSES), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: self._case_generator(case_dirs),
            output_signature=output_signature
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=500)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def get_train_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.train_cases, shuffle=True)

    def get_val_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.val_cases, shuffle=False)

    def get_test_dataset(self) -> tf.data.Dataset:
        return self._make_dataset(self.test_cases, shuffle=False)

    def get_calibration_generator(self, n_samples: int = 100):
        """Yield batches for TFLite INT8 calibration."""
        count = 0
        for img, _ in self._case_generator(self.val_cases):
            if count >= n_samples:
                break
            yield [img[np.newaxis]]   # add batch dim
            count += 1
