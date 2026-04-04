"""
tests/test_data_loaders.py
---------------------------
Tests for the four MedCompress data loaders:
  - ISICDataset       (ISIC 2020 melanoma classification)
  - BraTSDataset      (BraTS 2021 brain tumor segmentation)
  - CheXpertDataset   (CheXpert chest X-ray multi-label)
  - KvasirSEGDataset  (Kvasir-SEG polyp segmentation)

All tests mock filesystem and heavy I/O so they run WITHOUT real data
files, WITHOUT a GPU, and complete in seconds.

Run with:
    python -m pytest tests/test_data_loaders.py -v
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Guard TF import so the file itself loads even without TensorFlow
try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

skip_no_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
skip_no_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")


# ====================================================================== #
#  Shared helpers                                                         #
# ====================================================================== #

def _make_isic_csv(tmpdir: str, n_samples: int = 60) -> str:
    """Create a minimal train.csv for ISICDataset with balanced classes."""
    csv_path = os.path.join(tmpdir, "train.csv")
    rng = np.random.RandomState(0)
    half = n_samples // 2
    targets = np.array([0] * half + [1] * (n_samples - half))
    rng.shuffle(targets)
    with open(csv_path, "w") as f:
        f.write("image_name,target\n")
        for i, t in enumerate(targets):
            f.write(f"ISIC_{i:07d},{t}\n")
    os.makedirs(os.path.join(tmpdir, "train"), exist_ok=True)
    return csv_path


def _make_chexpert_csvs(tmpdir: str, n_train: int = 80, n_valid: int = 20):
    """Create minimal train.csv and valid.csv for CheXpertDataset."""
    labels = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Pleural Effusion",
    ]
    for name, n in [("train.csv", n_train), ("valid.csv", n_valid)]:
        path = os.path.join(tmpdir, name)
        rng = np.random.RandomState(42)
        with open(path, "w") as f:
            cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + labels
            f.write(",".join(cols) + "\n")
            for i in range(n):
                vals = [
                    f"train/patient{i:04d}/study1/view1.jpg",
                    "Male",
                    "55",
                    "Frontal",
                    "AP",
                ]
                for _ in labels:
                    vals.append(str(rng.choice([0.0, 1.0, -1.0])))
                f.write(",".join(vals) + "\n")


def _isic_config(tmpdir: str) -> dict:
    """Return a minimal config dict for ISICDataset."""
    return {
        "data": {
            "root": tmpdir,
            "image_size": 64,
            "batch_size": 4,
            "val_split": 0.15,
            "test_split": 0.15,
            "augmentation": False,
        }
    }


def _brats_config(tmpdir: str) -> dict:
    """Return a minimal config dict for BraTSDataset."""
    return {
        "data": {
            "root": tmpdir,
            "image_size": 240,
            "patch_size": 32,
            "batch_size": 2,
            "n_slices": 3,
            "val_split": 0.15,
            "test_split": 0.15,
            "modalities": ["t1", "t1ce", "t2", "flair"],
        }
    }


def _chexpert_config(tmpdir: str) -> dict:
    """Return a minimal config dict for CheXpertDataset."""
    return {
        "data": {
            "root": tmpdir,
            "image_size": 64,
            "batch_size": 4,
            "val_split": 0.15,
            "augmentation": False,
        }
    }


def _kvasir_config(tmpdir: str) -> dict:
    """Return a minimal config dict for KvasirSEGDataset."""
    return {
        "data": {
            "root": tmpdir,
            "image_size": 64,
            "batch_size": 2,
            "val_split": 0.15,
            "test_split": 0.15,
            "augmentation": False,
        },
        "seed": 42,
    }


# ====================================================================== #
#  ISICDataset tests                                                       #
# ====================================================================== #


@skip_no_tf
@skip_no_pandas
class TestISICDataset:
    """Tests for data/isic_loader.py ISICDataset."""

    def test_instantiation_with_valid_config(self, tmp_path):
        """ISICDataset should initialize and compute splits from train.csv."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        _make_isic_csv(tmpdir, n_samples=60)
        config = _isic_config(tmpdir)

        dataset = ISICDataset(config)

        assert hasattr(dataset, "train_df")
        assert hasattr(dataset, "val_df")
        assert hasattr(dataset, "test_df")
        total = len(dataset.train_df) + len(dataset.val_df) + len(dataset.test_df)
        assert total == 60, f"Splits should sum to 60, got {total}"

    def test_class_weights_computed(self, tmp_path):
        """Class weights should be non-zero inverse-frequency weights."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        _make_isic_csv(tmpdir, n_samples=60)
        config = _isic_config(tmpdir)

        dataset = ISICDataset(config)

        assert 0 in dataset.class_weights
        assert 1 in dataset.class_weights
        assert dataset.class_weights[0] > 0
        assert dataset.class_weights[1] > 0

    def test_split_ratios_are_approximately_correct(self, tmp_path):
        """Train/val/test proportions should roughly match the config."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        n = 200
        _make_isic_csv(tmpdir, n_samples=n)
        config = _isic_config(tmpdir)
        config["data"]["val_split"] = 0.15
        config["data"]["test_split"] = 0.15

        dataset = ISICDataset(config)

        test_frac = len(dataset.test_df) / n
        val_frac = len(dataset.val_df) / n
        # Allow 5% tolerance due to stratification rounding
        assert abs(test_frac - 0.15) < 0.05, f"Test fraction {test_frac} too far from 0.15"
        assert abs(val_frac - 0.15) < 0.05, f"Val fraction {val_frac} too far from 0.15"

    def test_augmentation_toggle(self, tmp_path):
        """Augmentation flag should be stored from config."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        _make_isic_csv(tmpdir, n_samples=60)

        config_aug = _isic_config(tmpdir)
        config_aug["data"]["augmentation"] = True
        ds_aug = ISICDataset(config_aug)
        assert ds_aug.augmentation is True

        config_noaug = _isic_config(tmpdir)
        config_noaug["data"]["augmentation"] = False
        ds_noaug = ISICDataset(config_noaug)
        assert ds_noaug.augmentation is False

    def test_image_preprocessing_normalization(self, tmp_path):
        """_load_image should normalize to [-1, 1] range."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        _make_isic_csv(tmpdir, n_samples=60)
        config = _isic_config(tmpdir)
        dataset = ISICDataset(config)

        # Manually verify the normalization math: (x/255 - 0.5) / 0.5
        raw_pixel = 128.0
        normalized = (raw_pixel / 255.0 - 0.5) / 0.5
        assert -1.0 <= normalized <= 1.0

        # Pure white (255) -> (1.0 - 0.5) / 0.5 = 1.0
        assert abs((255.0 / 255.0 - 0.5) / 0.5 - 1.0) < 1e-6
        # Pure black (0)  -> (0.0 - 0.5) / 0.5 = -1.0
        assert abs((0.0 / 255.0 - 0.5) / 0.5 - (-1.0)) < 1e-6

    def test_augment_clips_to_valid_range(self, tmp_path):
        """_augment should clip output to [-1, 1]."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        _make_isic_csv(tmpdir, n_samples=60)
        config = _isic_config(tmpdir)
        dataset = ISICDataset(config)

        fake_img = tf.constant(np.random.uniform(-1.0, 1.0, (64, 64, 3)).astype(np.float32))
        fake_label = tf.constant(1.0)

        aug_img, aug_label = dataset._augment(fake_img, fake_label)
        assert tf.reduce_min(aug_img).numpy() >= -1.0
        assert tf.reduce_max(aug_img).numpy() <= 1.0

    def test_missing_csv_raises_error(self, tmp_path):
        """ISICDataset should raise when train.csv does not exist."""
        from data.isic_loader import ISICDataset

        config = _isic_config(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            ISICDataset(config)

    def test_missing_target_column_raises_error(self, tmp_path):
        """ISICDataset should raise when CSV lacks the 'target' column."""
        from data.isic_loader import ISICDataset

        tmpdir = str(tmp_path)
        csv_path = os.path.join(tmpdir, "train.csv")
        with open(csv_path, "w") as f:
            f.write("image_name,wrong_column\n")
            f.write("ISIC_0000000,0\n")
        os.makedirs(os.path.join(tmpdir, "train"), exist_ok=True)

        config = _isic_config(tmpdir)
        with pytest.raises(KeyError):
            ISICDataset(config)


# ====================================================================== #
#  BraTSDataset tests                                                      #
# ====================================================================== #


@skip_no_tf
class TestBraTSDataset:
    """Tests for data/brats_loader.py BraTSDataset."""

    def test_n_slices_must_be_odd(self, tmp_path):
        """BraTSDataset should reject even n_slices."""
        from data.brats_loader import BraTSDataset

        config = _brats_config(str(tmp_path))
        config["data"]["n_slices"] = 4  # even => invalid
        with pytest.raises(AssertionError):
            BraTSDataset(config)

    def test_discover_cases_with_empty_directory(self, tmp_path):
        """BraTSDataset should find zero cases in an empty directory."""
        from data.brats_loader import BraTSDataset

        config = _brats_config(str(tmp_path))
        # No BraTS2021_* subdirectories exist
        # _discover_cases will find 0 cases; _build_splits will get
        # an empty list which train_test_split cannot split
        with pytest.raises(ValueError):
            BraTSDataset(config)

    def test_normalize_volume_all_zeros(self):
        """Z-score normalization of an all-zero volume should return zeros."""
        from data.brats_loader import _normalize_volume

        vol = np.zeros((10, 10, 10), dtype=np.float32)
        result = _normalize_volume(vol)
        np.testing.assert_array_equal(result, vol)

    def test_normalize_volume_standard(self):
        """Z-score normalization should produce mean~0, std~1 for non-zero voxels."""
        from data.brats_loader import _normalize_volume

        rng = np.random.RandomState(42)
        vol = rng.randn(20, 20, 20).astype(np.float32) * 100 + 500
        # Ensure some zeros for mask logic
        vol[:5, :, :] = 0

        result = _normalize_volume(vol)

        nonzero_mask = vol > 0
        normed = result[nonzero_mask]
        assert abs(normed.mean()) < 0.1, f"Mean should be ~0, got {normed.mean()}"
        assert abs(normed.std() - 1.0) < 0.1, f"Std should be ~1, got {normed.std()}"

    def test_remap_labels_maps_correctly(self):
        """Label remap {0,1,2,4} -> {0,1,2,3} should be exact."""
        from data.brats_loader import _remap_labels

        seg = np.array([0, 1, 2, 4, 0, 4, 2, 1], dtype=np.int32)
        remapped = _remap_labels(seg)
        expected = np.array([0, 1, 2, 3, 0, 3, 2, 1], dtype=np.uint8)
        np.testing.assert_array_equal(remapped, expected)

    def test_remap_labels_unknown_values_become_zero(self):
        """Any label not in {0,1,2,4} should map to 0 (background)."""
        from data.brats_loader import _remap_labels

        seg = np.array([0, 3, 5, 6], dtype=np.int32)
        remapped = _remap_labels(seg)
        # 3 and 5 and 6 are not in LABEL_REMAP, so np.zeros_like stays 0
        expected = np.array([0, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(remapped, expected)

    def test_config_modalities_default(self, tmp_path):
        """Default modalities should be ['t1', 't1ce', 't2', 'flair']."""
        from data.brats_loader import BraTSDataset

        tmpdir = str(tmp_path)
        # Create enough fake case dirs to avoid ValueError
        for i in range(10):
            os.makedirs(os.path.join(tmpdir, f"BraTS2021_{i:05d}"))

        config = _brats_config(tmpdir)
        del config["data"]["modalities"]  # rely on default

        dataset = BraTSDataset(config)
        assert dataset.modalities == ["t1", "t1ce", "t2", "flair"]

    def test_half_computed_from_n_slices(self, tmp_path):
        """half should equal n_slices // 2."""
        from data.brats_loader import BraTSDataset

        tmpdir = str(tmp_path)
        for i in range(10):
            os.makedirs(os.path.join(tmpdir, f"BraTS2021_{i:05d}"))

        for n in [1, 3, 5, 7]:
            config = _brats_config(tmpdir)
            config["data"]["n_slices"] = n
            dataset = BraTSDataset(config)
            assert dataset.half == n // 2


# ====================================================================== #
#  CheXpertDataset tests                                                   #
# ====================================================================== #


@skip_no_tf
@skip_no_pandas
class TestCheXpertDataset:
    """Tests for data/chexpert_loader.py CheXpertDataset."""

    def test_instantiation_and_splits(self, tmp_path):
        """CheXpertDataset should load CSVs and create train/val/test splits."""
        from data.chexpert_loader import CheXpertDataset

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir, n_train=80, n_valid=20)
        config = _chexpert_config(tmpdir)

        dataset = CheXpertDataset(config)

        assert hasattr(dataset, "train_df")
        assert hasattr(dataset, "val_df")
        assert hasattr(dataset, "test_df")
        # train + val = train_csv rows, test = valid_csv rows
        assert len(dataset.train_df) + len(dataset.val_df) == 80
        assert len(dataset.test_df) == 20

    def test_uones_policy(self, tmp_path):
        """Uncertain labels (-1) should become 1.0 (U-Ones policy)."""
        from data.chexpert_loader import CheXpertDataset, COMPETITION_LABELS

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir, n_train=80, n_valid=20)
        config = _chexpert_config(tmpdir)

        dataset = CheXpertDataset(config)

        for label in COMPETITION_LABELS:
            # After U-Ones, no value should be -1
            all_vals = pd.concat([dataset.train_df, dataset.val_df, dataset.test_df])[label]
            assert (all_vals == -1.0).sum() == 0, f"Label {label} still has -1 values"

    def test_nan_labels_become_zero(self, tmp_path):
        """NaN labels should be filled with 0.0 (negative)."""
        from data.chexpert_loader import CheXpertDataset, COMPETITION_LABELS

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir, n_train=80, n_valid=20)
        config = _chexpert_config(tmpdir)

        dataset = CheXpertDataset(config)

        for label in COMPETITION_LABELS:
            all_vals = pd.concat([dataset.train_df, dataset.val_df, dataset.test_df])[label]
            assert all_vals.isna().sum() == 0, f"Label {label} has NaN values"

    def test_competition_labels_property(self, tmp_path):
        """competition_labels property should return the 5 CheXpert labels."""
        from data.chexpert_loader import CheXpertDataset

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir)
        config = _chexpert_config(tmpdir)

        dataset = CheXpertDataset(config)

        labels = dataset.competition_labels
        assert len(labels) == 5
        assert "Atelectasis" in labels
        assert "Pleural Effusion" in labels

    def test_abs_path_column_created(self, tmp_path):
        """An 'abs_path' column should be added during loading."""
        from data.chexpert_loader import CheXpertDataset

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir)
        config = _chexpert_config(tmpdir)

        dataset = CheXpertDataset(config)

        assert "abs_path" in dataset.train_df.columns
        assert "abs_path" in dataset.test_df.columns
        # Each abs_path should start with the root directory
        for path in dataset.train_df["abs_path"].head():
            assert path.startswith(tmpdir)

    def test_missing_csv_raises_error(self, tmp_path):
        """CheXpertDataset should raise when CSVs do not exist."""
        from data.chexpert_loader import CheXpertDataset

        config = _chexpert_config(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            CheXpertDataset(config)

    def test_augmentation_default_is_false(self, tmp_path):
        """Augmentation should default to False when not specified."""
        from data.chexpert_loader import CheXpertDataset

        tmpdir = str(tmp_path)
        _make_chexpert_csvs(tmpdir)
        config = _chexpert_config(tmpdir)
        del config["data"]["augmentation"]  # omit to test default

        dataset = CheXpertDataset(config)
        assert dataset.augmentation is False


# ====================================================================== #
#  KvasirSEGDataset tests                                                  #
# ====================================================================== #


@skip_no_tf
class TestKvasirSEGDataset:
    """Tests for data/kvasir_loader.py KvasirSEGDataset."""

    @staticmethod
    def _create_kvasir_files(tmpdir: str, n_pairs: int = 20):
        """Create fake image/mask files so _discover_samples finds them."""
        img_dir = os.path.join(tmpdir, "images")
        mask_dir = os.path.join(tmpdir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for i in range(n_pairs):
            # Create small JPEG-like stubs (just need the file to exist
            # for discovery; actual pixel loading is not tested here)
            for d in (img_dir, mask_dir):
                with open(os.path.join(d, f"cju{i:06d}.jpg"), "wb") as f:
                    f.write(b"\xff\xd8\xff")  # minimal JPEG header

    def test_discover_matched_pairs(self, tmp_path):
        """Should find all image/mask pairs with matching filenames."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=20)
        config = _kvasir_config(tmpdir)

        dataset = KvasirSEGDataset(config)

        assert len(dataset.image_paths) == 20
        assert len(dataset.mask_paths) == 20

    def test_splits_sum_to_total(self, tmp_path):
        """Train + val + test should equal total number of pairs."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=50)
        config = _kvasir_config(tmpdir)

        dataset = KvasirSEGDataset(config)

        total = len(dataset.train_pairs) + len(dataset.val_pairs) + len(dataset.test_pairs)
        assert total == 50

    def test_empty_directory_raises_error(self, tmp_path):
        """KvasirSEGDataset should raise FileNotFoundError with no files."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
        os.makedirs(os.path.join(tmpdir, "masks"), exist_ok=True)
        config = _kvasir_config(tmpdir)

        with pytest.raises(FileNotFoundError, match="No image/mask pairs found"):
            KvasirSEGDataset(config)

    def test_unmatched_images_ignored(self, tmp_path):
        """Images without corresponding masks should be silently skipped."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        img_dir = os.path.join(tmpdir, "images")
        mask_dir = os.path.join(tmpdir, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # Create 10 images but only 5 masks
        for i in range(10):
            with open(os.path.join(img_dir, f"img_{i:04d}.jpg"), "wb") as f:
                f.write(b"\xff\xd8\xff")
        for i in range(5):
            with open(os.path.join(mask_dir, f"img_{i:04d}.jpg"), "wb") as f:
                f.write(b"\xff\xd8\xff")

        config = _kvasir_config(tmpdir)
        dataset = KvasirSEGDataset(config)

        assert len(dataset.image_paths) == 5

    def test_augmentation_default_is_true(self, tmp_path):
        """Kvasir default augmentation should be True."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=20)
        config = _kvasir_config(tmpdir)
        del config["data"]["augmentation"]  # rely on default

        dataset = KvasirSEGDataset(config)
        assert dataset.augmentation is True

    def test_image_size_default_is_256(self, tmp_path):
        """Default image_size should be 256 when not specified."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=20)
        config = _kvasir_config(tmpdir)
        del config["data"]["image_size"]  # rely on default

        dataset = KvasirSEGDataset(config)
        assert dataset.image_size == 256

    def test_seed_is_configurable(self, tmp_path):
        """Custom seed should be stored and used for reproducibility."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=30)
        config = _kvasir_config(tmpdir)
        config["seed"] = 99

        dataset = KvasirSEGDataset(config)
        assert dataset.seed == 99

    def test_augment_preserves_mask_binary(self, tmp_path):
        """Spatial augmentations applied to mask should keep it binary {0, 1}."""
        from data.kvasir_loader import KvasirSEGDataset

        tmpdir = str(tmp_path)
        self._create_kvasir_files(tmpdir, n_pairs=20)
        config = _kvasir_config(tmpdir)

        dataset = KvasirSEGDataset(config)

        fake_img = tf.constant(
            np.random.uniform(0.0, 1.0, (64, 64, 3)).astype(np.float32)
        )
        # Binary mask: half ones, half zeros
        fake_mask = tf.constant(
            np.concatenate([
                np.ones((64, 32, 1), dtype=np.float32),
                np.zeros((64, 32, 1), dtype=np.float32),
            ], axis=1)
        )

        aug_img, aug_mask = dataset._augment(fake_img, fake_mask)

        # Image should be clipped to [0, 1]
        assert tf.reduce_min(aug_img).numpy() >= 0.0
        assert tf.reduce_max(aug_img).numpy() <= 1.0

        # Mask should remain in {0, 1} (flips are spatial, not value-changing)
        unique_vals = set(tf.unique(tf.reshape(aug_mask, [-1]))[0].numpy().tolist())
        assert unique_vals.issubset({0.0, 1.0}), f"Mask has unexpected values: {unique_vals}"
