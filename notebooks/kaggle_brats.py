"""
MedCompress BraTS 2021 Brain Tumor Segmentation — Kaggle Notebook
=================================================================
GPU: T4 x2 (use 1)
Dataset: BraTS 2021 (add on Kaggle, e.g. "dschettler8845/brats-2021-task1"
         or "awsaf49/brats2020-training-data" — auto-discovers path)

2.5D U-Net segmentation with 4 MRI modalities (T1, T1ce, T2, FLAIR),
stacking 3 adjacent axial slices -> 12 input channels.

Experiments:
  1. Baseline U-Net Full (teacher)
  2. QAT INT8 on teacher
  3. Student U-Net Lite from scratch (no KD)
  4. KD student (T=3, alpha=0.6)
  5. KD + QAT INT8 combined

Metrics: Dice (per-class + mean), IoU, Hausdorff distance 95th,
         Sensitivity, Specificity

SETUP ON KAGGLE:
1. Create new notebook
2. Add dataset: search "brats 2021" or "brats2021"
3. Accelerator: GPU T4 x2
4. Paste this entire file, or split at "%%" markers into cells

OUTPUT: trained models, TFLite exports, CSV results, latency profiles
"""

# %% [markdown]
# # MedCompress: BraTS 2021 Brain Tumor Segmentation
# **Author: Abhishek Shekhar**
#
# 2.5D U-Net compression benchmark — Baseline, QAT, KD, KD+QAT.

# %% Cell 1: Setup and Imports
!pip install -q tensorflow-model-optimization SimpleITK nibabel

import os
import time
import json
import random
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import nibabel as nib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
from scipy.ndimage import distance_transform_edt

print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

# Use single GPU
if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices(
        tf.config.list_physical_devices('GPU')[0], 'GPU')

# %% Cell 2: Configuration
OUT = "/kaggle/working/brats_results"
os.makedirs(f"{OUT}/models", exist_ok=True)
os.makedirs(f"{OUT}/tflite", exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)

PATCH_SIZE = 128           # spatial resolution of each 2D slice
N_SLICES = 3               # adjacent axial slices for 2.5D
MODALITIES = ["t1", "t1ce", "t2", "flair"]
N_CHANNELS = N_SLICES * len(MODALITIES)  # 12
NUM_CLASSES = 4            # background, NCR, ED, ET
BATCH = 8
SEEDS = [42, 123, 456, 789, 1024]

# BraTS label remap: {0,1,2,4} -> {0,1,2,3} for contiguous one-hot
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 4: 3}
CLASS_NAMES = ["Background", "NCR", "ED", "ET"]

# Class weights to address heavy class imbalance (background >> foreground)
# Approximate ratios from BraTS: bg~98%, NCR~0.5%, ED~1.0%, ET~0.5%
CLASS_WEIGHTS = np.array([0.1, 3.0, 1.5, 3.0], dtype=np.float32)

config = {
    "baseline": {"epochs": 50, "lr": 1e-4, "patience": 10},
    "qat":      {"epochs": 10, "lr": 1e-5},
    "student":  {"epochs": 50, "lr": 1e-4, "patience": 10},
    "kd":       {"epochs": 50, "lr": 1e-4, "temperature": 3.0, "alpha": 0.6},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# %% Cell 3: Discover BraTS Dataset
print("=" * 60)
print("DISCOVERING BraTS 2021 DATASET")
print("=" * 60)

DATA_ROOT = None
KAGGLE_INPUT = "/kaggle/input"

# Walk Kaggle input directories to find BraTS case folders
for top_dir in os.listdir(KAGGLE_INPUT):
    top_path = os.path.join(KAGGLE_INPUT, top_dir)
    if not os.path.isdir(top_path):
        continue
    # Look for BraTS2021_XXXXX directories anywhere under this dataset
    candidates = glob.glob(os.path.join(top_path, "**", "BraTS2021_*"), recursive=True)
    # Filter to actual case directories (contain .nii.gz files)
    case_dirs = []
    for c in candidates:
        if os.path.isdir(c):
            nii_files = glob.glob(os.path.join(c, "*.nii.gz"))
            if len(nii_files) >= 4:
                case_dirs.append(c)
    if case_dirs:
        DATA_ROOT = os.path.dirname(case_dirs[0])
        break

# Fallback: try BraTS2020 naming (BraTS20_Training_*)
if DATA_ROOT is None:
    for top_dir in os.listdir(KAGGLE_INPUT):
        top_path = os.path.join(KAGGLE_INPUT, top_dir)
        if not os.path.isdir(top_path):
            continue
        candidates = glob.glob(os.path.join(top_path, "**", "BraTS20*"), recursive=True)
        case_dirs = []
        for c in candidates:
            if os.path.isdir(c):
                nii_files = glob.glob(os.path.join(c, "*.nii.gz"))
                if len(nii_files) >= 4:
                    case_dirs.append(c)
        if case_dirs:
            DATA_ROOT = os.path.dirname(case_dirs[0])
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find BraTS case directories under /kaggle/input/. "
        "Please add a BraTS 2021 dataset to this notebook."
    )

# Discover all case directories
all_case_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*")))
all_case_dirs = [d for d in all_case_dirs if os.path.isdir(d) and
                 len(glob.glob(os.path.join(d, "*.nii.gz"))) >= 4]

print(f"Data root: {DATA_ROOT}")
print(f"Total cases: {len(all_case_dirs)}")
if all_case_dirs:
    sample = all_case_dirs[0]
    print(f"Sample case: {os.path.basename(sample)}")
    print(f"  Files: {os.listdir(sample)}")


# %% Cell 4: Volume Loading and Slice Extraction
def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Z-score normalization on non-zero voxels only."""
    mask = vol > 0
    if mask.sum() == 0:
        return vol.astype(np.float32)
    mu = vol[mask].mean()
    sigma = max(vol[mask].std(), 1e-8)
    out = np.zeros_like(vol, dtype=np.float32)
    out[mask] = (vol[mask] - mu) / sigma
    return out


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Remap BraTS labels {0,1,2,4} -> {0,1,2,3}."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in LABEL_REMAP.items():
        out[seg == src] = dst
    return out


def load_case(case_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load all modalities + segmentation for one BraTS case.

    Returns:
        volume: (H, W, D, 4) float32
        seg: (H, W, D) uint8, remapped labels
    """
    case_id = os.path.basename(case_dir)
    vols = []
    for mod in MODALITIES:
        # Try both naming conventions
        path = os.path.join(case_dir, f"{case_id}_{mod}.nii.gz")
        if not os.path.exists(path):
            # Some datasets use lowercase/different naming
            candidates = glob.glob(os.path.join(case_dir, f"*_{mod}.nii.gz"))
            if not candidates:
                candidates = glob.glob(os.path.join(case_dir, f"*{mod}*.nii.gz"))
            if candidates:
                path = candidates[0]
        vol = nib.load(path).get_fdata().astype(np.float32)
        vols.append(normalize_volume(vol))

    volume = np.stack(vols, axis=-1)  # (H, W, D, 4)

    # Load segmentation
    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
    if not os.path.exists(seg_path):
        candidates = glob.glob(os.path.join(case_dir, "*seg*.nii.gz"))
        if candidates:
            seg_path = candidates[0]
    seg = nib.load(seg_path).get_fdata().astype(np.int32)
    seg = remap_labels(seg)

    return volume, seg


def extract_slices(
    volume: np.ndarray, seg: np.ndarray, skip_empty: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Extract 2.5D axial slices from a volume.

    For each axial index z, stack [z-1, z, z+1] across 4 modalities
    -> (H, W, 12) input. Skip all-background slices by default.

    Returns list of (image, mask_onehot) pairs.
    """
    half = N_SLICES // 2
    depth = volume.shape[2]
    slices = []

    for z in range(half, depth - half):
        # Multi-slice input: (H, W, N_SLICES, 4) -> (H, W, 12)
        img = volume[:, :, z - half: z + half + 1, :]
        img = img.reshape(img.shape[0], img.shape[1], -1)
        mask = seg[:, :, z]  # (H, W)

        if skip_empty and mask.sum() == 0:
            continue

        # Resize to PATCH_SIZE
        img_resized = tf.image.resize(
            img, [PATCH_SIZE, PATCH_SIZE]).numpy()
        mask_resized = tf.image.resize(
            mask[..., None].astype(np.float32),
            [PATCH_SIZE, PATCH_SIZE],
            method="nearest"
        ).numpy()[..., 0].astype(np.uint8)

        # One-hot encode: (PATCH_SIZE, PATCH_SIZE, NUM_CLASSES)
        mask_oh = np.eye(NUM_CLASSES, dtype=np.float32)[mask_resized]

        slices.append((img_resized, mask_oh))

    return slices


# Quick validation: load one case
if all_case_dirs:
    print("\nValidating data loading...")
    vol_test, seg_test = load_case(all_case_dirs[0])
    print(f"  Volume shape: {vol_test.shape}")  # (240, 240, 155, 4) typical
    print(f"  Seg shape: {seg_test.shape}")
    print(f"  Unique labels (remapped): {np.unique(seg_test)}")
    slices_test = extract_slices(vol_test, seg_test)
    print(f"  Foreground slices: {len(slices_test)}")
    if slices_test:
        print(f"  Slice image shape: {slices_test[0][0].shape}")
        print(f"  Slice mask shape: {slices_test[0][1].shape}")
    del vol_test, seg_test, slices_test


# %% Cell 5: Data Pipeline
print("\n" + "=" * 60)
print("BUILDING DATA PIPELINE")
print("=" * 60)

# Split cases (not slices) into train/val/test
train_cases, test_cases = train_test_split(
    all_case_dirs, test_size=0.15, random_state=42)
train_cases, val_cases = train_test_split(
    train_cases, test_size=0.15 / 0.85, random_state=42)

print(f"Case splits: train={len(train_cases)}, "
      f"val={len(val_cases)}, test={len(test_cases)}")


def case_generator(case_dirs: List[str], skip_empty: bool = True):
    """Python generator yielding (image, mask_onehot) from cases."""
    for case_dir in case_dirs:
        try:
            volume, seg = load_case(case_dir)
            for img, mask_oh in extract_slices(volume, seg, skip_empty):
                yield img.astype(np.float32), mask_oh.astype(np.float32)
        except Exception as e:
            print(f"  [WARN] Skipping {os.path.basename(case_dir)}: {e}")


def make_dataset(
    case_dirs: List[str],
    shuffle: bool = False,
    augment: bool = False,
    skip_empty: bool = True,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset from BraTS case directories."""
    output_signature = (
        tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, NUM_CLASSES), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(
        lambda: case_generator(case_dirs, skip_empty),
        output_signature=output_signature,
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=500)

    if augment:
        def aug_fn(img, mask):
            # Random horizontal flip
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_left_right(img)
                mask = tf.image.flip_left_right(mask)
            # Random vertical flip
            if tf.random.uniform([]) > 0.5:
                img = tf.image.flip_up_down(img)
                mask = tf.image.flip_up_down(mask)
            # Random brightness on image only
            img = img + tf.random.uniform([], -0.1, 0.1)
            return img, mask
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_cases, shuffle=True, augment=True)
val_ds = make_dataset(val_cases)
test_ds = make_dataset(test_cases)

# Count slices (approximate from first batch)
n_train_batches = 0
for _ in train_ds:
    n_train_batches += 1
print(f"Training batches: ~{n_train_batches} (batch_size={BATCH})")
print(f"Approximate training slices: ~{n_train_batches * BATCH}")


# %% Cell 6: U-Net Architectures (Self-Contained)
def conv_block(x, filters: int, name: str):
    """Two consecutive Conv2D -> BN -> ReLU."""
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def encoder_block(x, filters: int, name: str):
    """Conv block + max pooling. Returns (skip, pooled)."""
    skip = conv_block(x, filters, name=f"{name}_block")
    pooled = layers.MaxPooling2D(2, name=f"{name}_pool")(skip)
    return skip, pooled


def decoder_block(x, skip, filters: int, name: str):
    """Upsample + concatenate skip + conv block."""
    x = layers.UpSampling2D(2, interpolation="bilinear", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_block")
    return x


def build_unet_full(
    num_classes: int = NUM_CLASSES,
    n_channels: int = N_CHANNELS,
    input_size: int = PATCH_SIZE,
) -> keras.Model:
    """U-Net Full (teacher): 4 encoder stages, filters [64,128,256,512], bottleneck 1024."""
    inputs = keras.Input(
        shape=(input_size, input_size, n_channels), name="slice_input")

    s1, p1 = encoder_block(inputs, 64,  name="enc1")
    s2, p2 = encoder_block(p1,     128, name="enc2")
    s3, p3 = encoder_block(p2,     256, name="enc3")
    s4, p4 = encoder_block(p3,     512, name="enc4")

    bottleneck = conv_block(p4, 1024, name="bottleneck")

    x = decoder_block(bottleneck, s4, 512, name="dec4")
    x = decoder_block(x,          s3, 256, name="dec3")
    x = decoder_block(x,          s2, 128, name="dec2")
    x = decoder_block(x,          s1, 64,  name="dec1")

    outputs = layers.Conv2D(
        num_classes, 1, activation="softmax", name="seg_output")(x)

    return keras.Model(inputs, outputs, name="unet_full")


def build_unet_lite(
    num_classes: int = NUM_CLASSES,
    n_channels: int = N_CHANNELS,
    input_size: int = PATCH_SIZE,
) -> keras.Model:
    """U-Net Lite (student): 3 encoder stages, filters [32,64,128], bottleneck 256."""
    inputs = keras.Input(
        shape=(input_size, input_size, n_channels), name="slice_input")

    s1, p1 = encoder_block(inputs, 32,  name="enc1")
    s2, p2 = encoder_block(p1,     64,  name="enc2")
    s3, p3 = encoder_block(p2,     128, name="enc3")

    bottleneck = conv_block(p3, 256, name="bottleneck")

    x = decoder_block(bottleneck, s3, 128, name="dec3")
    x = decoder_block(x,          s2, 64,  name="dec2")
    x = decoder_block(x,          s1, 32,  name="dec1")

    outputs = layers.Conv2D(
        num_classes, 1, activation="softmax", name="seg_output")(x)

    return keras.Model(inputs, outputs, name="unet_lite")


# Print model summaries
teacher_test = build_unet_full()
student_test = build_unet_lite()
print(f"U-Net Full (teacher): {teacher_test.count_params():,} params")
print(f"U-Net Lite (student): {student_test.count_params():,} params")
print(f"Compression ratio: {teacher_test.count_params() / student_test.count_params():.1f}x")
del teacher_test, student_test


# %% Cell 7: Loss Functions
def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice loss averaged over classes (expects one-hot y_true)."""
    y_true_f = tf.cast(tf.reshape(y_true, [-1, NUM_CLASSES]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    denom = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice_per_class = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - tf.reduce_mean(dice_per_class)


def weighted_ce_loss(y_true, y_pred):
    """Categorical cross-entropy weighted by class frequencies."""
    weights = tf.constant(CLASS_WEIGHTS, dtype=tf.float32)
    # y_true: (B, H, W, C), y_pred: (B, H, W, C)
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce = -tf.reduce_sum(y_true * tf.math.log(y_pred_clipped) * weights, axis=-1)
    return tf.reduce_mean(ce)


def dice_ce_loss(y_true, y_pred):
    """Combined Dice + weighted cross-entropy loss."""
    return dice_loss(y_true, y_pred) + weighted_ce_loss(y_true, y_pred)


# Name attribute needed for model serialization
dice_ce_loss.__name__ = "dice_ce_loss"


class MeanDiceMetric(keras.metrics.Metric):
    """Mean Dice coefficient excluding background (class 0)."""

    def __init__(self, **kwargs):
        super().__init__(name="mean_dice", **kwargs)
        self.dice_sum = self.add_weight("dice_sum", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        pred_oh = tf.one_hot(tf.argmax(y_pred, axis=-1), NUM_CLASSES)
        # Foreground classes only (1, 2, 3)
        y_true_fg = y_true[..., 1:]
        pred_fg = pred_oh[..., 1:]
        smooth = 1e-6
        intersection = tf.reduce_sum(y_true_fg * pred_fg)
        union = tf.reduce_sum(y_true_fg) + tf.reduce_sum(pred_fg)
        dice = (2.0 * intersection + smooth) / (union + smooth)
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice_sum / (self.count + 1e-10)

    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)


# %% Cell 8: Comprehensive Evaluation
def hausdorff_distance_95(
    pred_mask: np.ndarray, true_mask: np.ndarray
) -> float:
    """Compute 95th percentile Hausdorff distance between two binary masks."""
    if pred_mask.sum() == 0 and true_mask.sum() == 0:
        return 0.0
    if pred_mask.sum() == 0 or true_mask.sum() == 0:
        return np.inf

    # Distance from pred boundary to nearest true voxel
    pred_border = pred_mask.astype(bool) & ~(
        distance_transform_edt(pred_mask) > 1)
    true_border = true_mask.astype(bool) & ~(
        distance_transform_edt(true_mask) > 1)

    # Use distance transform approach
    dt_true = distance_transform_edt(~true_mask.astype(bool))
    dt_pred = distance_transform_edt(~pred_mask.astype(bool))

    d_pred_to_true = dt_true[pred_mask.astype(bool)]
    d_true_to_pred = dt_pred[true_mask.astype(bool)]

    if len(d_pred_to_true) == 0 or len(d_true_to_pred) == 0:
        return np.inf

    hd95 = max(
        np.percentile(d_pred_to_true, 95),
        np.percentile(d_true_to_pred, 95),
    )
    return float(hd95)


def evaluate_segmentation(
    model, dataset, name: str = ""
) -> Dict[str, float]:
    """Full segmentation evaluation: per-class Dice, IoU, HD95, Sens, Spec."""
    # Accumulate per-class metrics
    class_tp = np.zeros(NUM_CLASSES)
    class_fp = np.zeros(NUM_CLASSES)
    class_fn = np.zeros(NUM_CLASSES)
    class_tn = np.zeros(NUM_CLASSES)
    hd95_values = {c: [] for c in range(1, NUM_CLASSES)}  # skip bg
    n_samples = 0

    for imgs, masks in dataset:
        preds = model(imgs, training=False).numpy()
        pred_labels = np.argmax(preds, axis=-1)       # (B, H, W)
        true_labels = np.argmax(masks.numpy(), axis=-1)  # (B, H, W)

        for i in range(len(pred_labels)):
            for c in range(NUM_CLASSES):
                p_c = (pred_labels[i] == c).astype(np.float32)
                t_c = (true_labels[i] == c).astype(np.float32)

                tp = (p_c * t_c).sum()
                fp = (p_c * (1 - t_c)).sum()
                fn = ((1 - p_c) * t_c).sum()
                tn = ((1 - p_c) * (1 - t_c)).sum()

                class_tp[c] += tp
                class_fp[c] += fp
                class_fn[c] += fn
                class_tn[c] += tn

            # HD95 for foreground classes (sample-level, skip bg)
            for c in range(1, NUM_CLASSES):
                p_c = (pred_labels[i] == c).astype(np.uint8)
                t_c = (true_labels[i] == c).astype(np.uint8)
                if t_c.sum() > 0:  # only compute if ground truth has this class
                    hd = hausdorff_distance_95(p_c, t_c)
                    if hd != np.inf:
                        hd95_values[c].append(hd)

            n_samples += 1

    # Compute per-class metrics (skip background class 0)
    smooth = 1e-7
    results = {}
    dice_scores = []
    iou_scores = []
    sens_scores = []
    spec_scores = []
    hd95_scores = []

    for c in range(1, NUM_CLASSES):
        tp, fp, fn, tn = class_tp[c], class_fp[c], class_fn[c], class_tn[c]

        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        iou = (tp + smooth) / (tp + fp + fn + smooth)
        sens = (tp + smooth) / (tp + fn + smooth)
        spec = (tn + smooth) / (tn + fp + smooth)

        results[f"dice_{CLASS_NAMES[c]}"] = float(dice)
        results[f"iou_{CLASS_NAMES[c]}"] = float(iou)
        results[f"sens_{CLASS_NAMES[c]}"] = float(sens)
        results[f"spec_{CLASS_NAMES[c]}"] = float(spec)

        dice_scores.append(dice)
        iou_scores.append(iou)
        sens_scores.append(sens)
        spec_scores.append(spec)

        if hd95_values[c]:
            hd95_mean = float(np.mean(hd95_values[c]))
            results[f"hd95_{CLASS_NAMES[c]}"] = hd95_mean
            hd95_scores.append(hd95_mean)

    # Mean across foreground classes
    results["dice_mean"] = float(np.mean(dice_scores))
    results["iou_mean"] = float(np.mean(iou_scores))
    results["sensitivity_mean"] = float(np.mean(sens_scores))
    results["specificity_mean"] = float(np.mean(spec_scores))
    if hd95_scores:
        results["hd95_mean"] = float(np.mean(hd95_scores))

    results["n_samples"] = n_samples

    print(f"[{name}] Dice={results['dice_mean']:.4f}  "
          f"IoU={results['iou_mean']:.4f}  "
          f"Sens={results['sensitivity_mean']:.4f}  "
          f"Spec={results['specificity_mean']:.4f}"
          + (f"  HD95={results.get('hd95_mean', 0):.2f}" if hd95_scores else ""))
    print(f"  Per-class Dice — NCR={results['dice_NCR']:.4f}  "
          f"ED={results['dice_ED']:.4f}  ET={results['dice_ET']:.4f}")

    return results


# %% Cell 9: TFLite Export Utilities
def export_tflite(
    model, output_path: str, quantize: str = "fp32", calib_ds=None
) -> float:
    """Export Keras model to TFLite. Returns size in MB."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calib_ds is not None:
            def representative_dataset():
                count = 0
                for images, _ in calib_ds:
                    for i in range(len(images)):
                        if count >= 200:
                            return
                        yield [np.expand_dims(images[i].numpy(), axis=0)]
                        count += 1
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
    elif quantize == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Exported: {output_path} ({size_mb:.1f} MB)")
    return size_mb


def profile_tflite_latency(
    tflite_path: str, n_warmup: int = 10, n_runs: int = 100
) -> Dict[str, float]:
    """Profile TFLite model CPU latency."""
    interpreter = tf.lite.Interpreter(
        model_path=tflite_path, num_threads=4)
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]

    # Generate dummy input matching expected dtype
    if inp_detail["dtype"] == np.uint8:
        dummy = np.random.randint(
            0, 255, size=inp_detail["shape"]).astype(np.uint8)
    else:
        dummy = np.random.randn(*inp_detail["shape"]).astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        interpreter.set_tensor(inp_detail["index"], dummy)
        interpreter.invoke()

    # Measure
    times = []
    for _ in range(n_runs):
        interpreter.set_tensor(inp_detail["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times_arr = np.array(times)
    return {
        "latency_median_ms": float(np.median(times_arr)),
        "latency_p95_ms": float(np.percentile(times_arr, 95)),
        "latency_min_ms": float(np.min(times_arr)),
        "latency_max_ms": float(np.max(times_arr)),
        "size_mb": os.path.getsize(tflite_path) / 1e6,
    }


# %% Cell 10: EXPERIMENT 1 — Baseline U-Net Full (Teacher)
print("\n" + "=" * 60)
print("EXPERIMENT 1: BASELINE U-Net Full (Teacher)")
print("=" * 60)

baseline_results = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    teacher = build_unet_full()
    teacher.compile(
        optimizer=keras.optimizers.Adam(config["baseline"]["lr"]),
        loss=dice_ce_loss,
        metrics=[MeanDiceMetric()],
    )
    teacher.fit(
        train_ds, validation_data=val_ds,
        epochs=config["baseline"]["epochs"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_mean_dice",
                patience=config["baseline"]["patience"],
                mode="max",
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_mean_dice", factor=0.5,
                patience=5, mode="max", min_lr=1e-6,
            ),
        ],
        verbose=1,
    )
    metrics = evaluate_segmentation(teacher, test_ds, f"Baseline s{seed}")
    baseline_results.append(metrics)

# Save last teacher (used for KD and QAT)
teacher.save(f"{OUT}/models/brats_unet_full_teacher.keras")

# Export teacher FP32 TFLite
teacher_fp32_path = f"{OUT}/tflite/brats_unet_full_fp32.tflite"
export_tflite(teacher, teacher_fp32_path, quantize="fp32")

print("\n--- Baseline Summary (5 seeds) ---")
for key in ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean", "hd95_mean"]:
    vals = [r.get(key, 0) for r in baseline_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
print(f"  Params: {teacher.count_params():,}")


# %% Cell 11: EXPERIMENT 2 — QAT INT8 on Teacher
print("\n" + "=" * 60)
print("EXPERIMENT 2: QAT INT8 on U-Net Full")
print("=" * 60)

qat_results = []
for seed in SEEDS:
    print(f"\n--- QAT Seed {seed} ---")
    set_seed(seed)

    base = keras.models.load_model(
        f"{OUT}/models/brats_unet_full_teacher.keras",
        custom_objects={
            "dice_ce_loss": dice_ce_loss,
            "MeanDiceMetric": MeanDiceMetric,
        },
    )

    qat_model = tfmot.quantization.keras.quantize_model(base)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(config["qat"]["lr"]),
        loss=dice_ce_loss,
        metrics=[MeanDiceMetric()],
    )
    qat_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config["qat"]["epochs"],
        verbose=1,
    )

    stripped = tfmot.quantization.keras.strip_pruning(qat_model)

    # Evaluate stripped Keras model
    metrics = evaluate_segmentation(stripped, test_ds, f"QAT s{seed}")

    # Export INT8 TFLite
    tflite_path = f"{OUT}/tflite/brats_unet_full_qat_int8_s{seed}.tflite"
    sz = export_tflite(stripped, tflite_path, quantize="int8", calib_ds=val_ds)
    metrics["size_mb"] = sz

    # Latency profile
    lat = profile_tflite_latency(tflite_path)
    metrics.update(lat)
    qat_results.append(metrics)

print("\n--- QAT INT8 Summary ---")
for key in ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean",
            "hd95_mean", "latency_median_ms", "size_mb"]:
    vals = [r.get(key, 0) for r in qat_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


# %% Cell 12: EXPERIMENT 3 — Student U-Net Lite from Scratch (No KD)
print("\n" + "=" * 60)
print("EXPERIMENT 3: Student U-Net Lite from Scratch")
print("=" * 60)

scratch_results = []
for seed in SEEDS:
    print(f"\n--- Scratch Seed {seed} ---")
    set_seed(seed)
    student = build_unet_lite()
    student.compile(
        optimizer=keras.optimizers.Adam(config["student"]["lr"]),
        loss=dice_ce_loss,
        metrics=[MeanDiceMetric()],
    )
    student.fit(
        train_ds, validation_data=val_ds,
        epochs=config["student"]["epochs"],
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_mean_dice",
                patience=config["student"]["patience"],
                mode="max",
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_mean_dice", factor=0.5,
                patience=5, mode="max", min_lr=1e-6,
            ),
        ],
        verbose=1,
    )
    metrics = evaluate_segmentation(student, test_ds, f"Scratch s{seed}")
    scratch_results.append(metrics)

student.save(f"{OUT}/models/brats_unet_lite_scratch.keras")

# Export student FP32 TFLite
student_fp32_path = f"{OUT}/tflite/brats_unet_lite_fp32.tflite"
export_tflite(student, student_fp32_path, quantize="fp32")

print("\n--- Student Scratch Summary ---")
for key in ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean", "hd95_mean"]:
    vals = [r.get(key, 0) for r in scratch_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
print(f"  Params: {student.count_params():,}")


# %% Cell 13: EXPERIMENT 4 — Knowledge Distillation Student
print("\n" + "=" * 60)
print("EXPERIMENT 4: Knowledge Distillation (T=3, alpha=0.6)")
print("=" * 60)

T = config["kd"]["temperature"]
ALPHA = config["kd"]["alpha"]

# Load trained teacher
teacher_kd = keras.models.load_model(
    f"{OUT}/models/brats_unet_full_teacher.keras",
    custom_objects={
        "dice_ce_loss": dice_ce_loss,
        "MeanDiceMetric": MeanDiceMetric,
    },
)

kd_results = []
for seed in SEEDS:
    print(f"\n--- KD Seed {seed} ---")
    set_seed(seed)

    student_kd = build_unet_lite()
    optimizer = keras.optimizers.Adam(config["kd"]["lr"])

    best_val_dice = 0.0
    patience_counter = 0
    best_weights = None

    for epoch in range(config["kd"]["epochs"]):
        # --- Training ---
        epoch_loss = 0.0
        n_batches = 0
        for images, masks in train_ds:
            with tf.GradientTape() as tape:
                # Teacher predictions (frozen)
                teacher_logits = teacher_kd(images, training=False)
                # Student predictions
                student_logits = student_kd(images, training=True)

                # Temperature-scaled softmax for distillation
                # Work in logit space: convert softmax back to logits
                teacher_log = tf.math.log(teacher_logits + 1e-7)
                student_log = tf.math.log(student_logits + 1e-7)

                teacher_soft = tf.nn.softmax(teacher_log / T, axis=-1)
                student_soft = tf.nn.softmax(student_log / T, axis=-1)

                # KL divergence loss (soft targets)
                kl_loss = tf.reduce_sum(
                    teacher_soft * tf.math.log(
                        (teacher_soft + 1e-7) / (student_soft + 1e-7)),
                    axis=-1,
                )
                distill_loss = tf.reduce_mean(kl_loss) * (T ** 2)

                # Hard label loss (Dice + weighted CE)
                hard_loss = dice_ce_loss(masks, student_logits)

                # Combined loss
                loss = ALPHA * distill_loss + (1 - ALPHA) * hard_loss

            grads = tape.gradient(loss, student_kd.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, student_kd.trainable_variables))
            epoch_loss += loss.numpy()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # --- Validation (quick Dice check) ---
        val_dice_sum = 0.0
        val_count = 0
        for v_imgs, v_masks in val_ds:
            v_preds = student_kd(v_imgs, training=False)
            pred_oh = tf.one_hot(tf.argmax(v_preds, axis=-1), NUM_CLASSES)
            # Foreground Dice
            inter = tf.reduce_sum(pred_oh[..., 1:] * v_masks[..., 1:])
            union = tf.reduce_sum(pred_oh[..., 1:]) + tf.reduce_sum(v_masks[..., 1:])
            val_dice_sum += (2 * inter + 1e-6) / (union + 1e-6)
            val_count += 1

        val_dice = float(val_dice_sum / max(val_count, 1))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_weights = [w.numpy() for w in student_kd.trainable_weights]
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  "
                  f"val_dice={val_dice:.4f}  best={best_val_dice:.4f}")

        # Early stopping
        if patience_counter >= config["baseline"]["patience"]:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_weights is not None:
        for w, bw in zip(student_kd.trainable_weights, best_weights):
            w.assign(bw)

    metrics = evaluate_segmentation(student_kd, test_ds, f"KD s{seed}")
    kd_results.append(metrics)

student_kd.save(f"{OUT}/models/brats_unet_lite_kd.keras")

print("\n--- KD Summary ---")
for key in ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean", "hd95_mean"]:
    vals = [r.get(key, 0) for r in kd_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# Distillation gain
scratch_dice = np.mean([r["dice_mean"] for r in scratch_results])
kd_dice = np.mean([r["dice_mean"] for r in kd_results])
print(f"\n*** DISTILLATION GAIN: {(kd_dice - scratch_dice)*100:+.2f}% mean Dice ***")
print(f"    Scratch: {scratch_dice:.4f}  vs  KD: {kd_dice:.4f}")


# %% Cell 14: EXPERIMENT 5 — KD + QAT INT8 Combined
print("\n" + "=" * 60)
print("EXPERIMENT 5: KD + QAT INT8 Combined")
print("=" * 60)

kd_qat_results = []
for seed in SEEDS:
    print(f"\n--- KD+QAT Seed {seed} ---")
    set_seed(seed)

    base_kd = keras.models.load_model(
        f"{OUT}/models/brats_unet_lite_kd.keras",
        custom_objects={
            "dice_ce_loss": dice_ce_loss,
            "MeanDiceMetric": MeanDiceMetric,
        },
    )

    qat_kd = tfmot.quantization.keras.quantize_model(base_kd)
    qat_kd.compile(
        optimizer=keras.optimizers.Adam(config["qat"]["lr"]),
        loss=dice_ce_loss,
        metrics=[MeanDiceMetric()],
    )
    qat_kd.fit(
        train_ds, validation_data=val_ds,
        epochs=config["qat"]["epochs"],
        verbose=1,
    )

    stripped = tfmot.quantization.keras.strip_pruning(qat_kd)

    # Evaluate
    metrics = evaluate_segmentation(stripped, test_ds, f"KD+QAT s{seed}")

    # Export INT8 TFLite
    tflite_path = f"{OUT}/tflite/brats_unet_lite_kd_qat_int8_s{seed}.tflite"
    sz = export_tflite(stripped, tflite_path, quantize="int8", calib_ds=val_ds)
    metrics["size_mb"] = sz

    # Latency profile
    lat = profile_tflite_latency(tflite_path)
    metrics.update(lat)
    kd_qat_results.append(metrics)

print("\n--- KD+QAT INT8 Summary ---")
for key in ["dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean",
            "hd95_mean", "latency_median_ms", "size_mb"]:
    vals = [r.get(key, 0) for r in kd_qat_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


# %% Cell 15: CPU Latency Profiling (All Models)
print("\n" + "=" * 60)
print("CPU LATENCY PROFILING (all exported TFLite models)")
print("=" * 60)

tflite_dir = Path(f"{OUT}/tflite")
latency_rows = []

for tflite_file in sorted(tflite_dir.glob("*.tflite")):
    # For multi-seed models, profile only seed 42
    if "_s" in tflite_file.stem and "_s42" not in tflite_file.stem:
        continue
    print(f"\nProfiling: {tflite_file.name}")
    lat = profile_tflite_latency(str(tflite_file))
    lat["model"] = tflite_file.name
    latency_rows.append(lat)
    print(f"  Median: {lat['latency_median_ms']:.1f} ms  "
          f"P95: {lat['latency_p95_ms']:.1f} ms  "
          f"Size: {lat['size_mb']:.1f} MB")

latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(f"{OUT}/results/brats_cpu_latency.csv", index=False)
print(f"\nLatency results saved to {OUT}/results/brats_cpu_latency.csv")


# %% Cell 16: Compile All Results
print("\n" + "=" * 60)
print("COMPILING FINAL RESULTS")
print("=" * 60)

METRIC_KEYS = [
    "dice_mean", "iou_mean", "sensitivity_mean", "specificity_mean", "hd95_mean",
    "dice_NCR", "dice_ED", "dice_ET",
    "hd95_NCR", "hd95_ED", "hd95_ET",
]


def summarize(results_list: list, method_name: str) -> dict:
    row = {"method": method_name}
    for key in METRIC_KEYS:
        vals = [r.get(key, 0) for r in results_list]
        row[f"{key}_mean"] = round(float(np.mean(vals)), 4)
        row[f"{key}_std"] = round(float(np.std(vals)), 4)

    # Latency and size (if available)
    if "latency_median_ms" in results_list[0]:
        lat_vals = [r["latency_median_ms"] for r in results_list]
        row["latency_median_ms"] = round(float(np.mean(lat_vals)), 1)
    if "size_mb" in results_list[0]:
        sz_vals = [r["size_mb"] for r in results_list]
        row["size_mb"] = round(float(np.mean(sz_vals)), 1)

    return row


all_results = [
    summarize(baseline_results,  "Baseline U-Net Full (FP32)"),
    summarize(qat_results,       "QAT INT8 (U-Net Full)"),
    summarize(scratch_results,   "Student Scratch (U-Net Lite)"),
    summarize(kd_results,        "KD Student (T=3, a=0.6)"),
    summarize(kd_qat_results,    "KD + QAT INT8 (U-Net Lite)"),
]

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{OUT}/results/brats_all_results.csv", index=False)

# Pretty print
print("\n--- Main Results (mean +/- std across 5 seeds) ---")
print(f"{'Method':<32} {'Dice':>10} {'IoU':>10} {'Sens':>10} {'Spec':>10} {'HD95':>10}")
print("-" * 82)
for row in all_results:
    m = row["method"]
    dice = f"{row['dice_mean_mean']:.4f}"
    iou = f"{row['iou_mean_mean']:.4f}"
    sens = f"{row['sensitivity_mean_mean']:.4f}"
    spec = f"{row['specificity_mean_mean']:.4f}"
    hd95 = f"{row['hd95_mean_mean']:.2f}" if row.get("hd95_mean_mean", 0) > 0 else "N/A"
    print(f"{m:<32} {dice:>10} {iou:>10} {sens:>10} {spec:>10} {hd95:>10}")

print("\n--- Per-Class Dice ---")
print(f"{'Method':<32} {'NCR':>10} {'ED':>10} {'ET':>10}")
print("-" * 62)
for row in all_results:
    m = row["method"]
    ncr = f"{row['dice_NCR_mean']:.4f}"
    ed = f"{row['dice_ED_mean']:.4f}"
    et = f"{row['dice_ET_mean']:.4f}"
    print(f"{m:<32} {ncr:>10} {ed:>10} {et:>10}")

# Distillation gain table
gain_data = {
    "student_scratch_dice": scratch_dice,
    "kd_student_dice": kd_dice,
    "gain_dice": kd_dice - scratch_dice,
    "gain_pct": (kd_dice - scratch_dice) * 100,
}
with open(f"{OUT}/results/brats_distillation_gain.json", "w") as f:
    json.dump(gain_data, f, indent=2)

# Full per-seed results for reproducibility
per_seed_rows = []
for seed_idx, seed in enumerate(SEEDS):
    for exp_name, exp_results in [
        ("baseline", baseline_results),
        ("qat_int8", qat_results),
        ("scratch", scratch_results),
        ("kd", kd_results),
        ("kd_qat_int8", kd_qat_results),
    ]:
        row = {"experiment": exp_name, "seed": seed}
        row.update(exp_results[seed_idx])
        per_seed_rows.append(row)

per_seed_df = pd.DataFrame(per_seed_rows)
per_seed_df.to_csv(f"{OUT}/results/brats_per_seed_results.csv", index=False)

# Save summary JSON
summary_json = {}
for row in all_results:
    name = row["method"]
    summary_json[name] = {
        k: v for k, v in row.items() if k != "method"
    }
with open(f"{OUT}/results/brats_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)


# %% Cell 17: Final Report
print("\n" + "=" * 60)
print("ALL BraTS 2021 EXPERIMENTS COMPLETE")
print("=" * 60)
print(f"\nResults directory: {OUT}/results/")
print(f"  brats_all_results.csv       — summary table")
print(f"  brats_per_seed_results.csv  — per-seed detailed results")
print(f"  brats_cpu_latency.csv       — TFLite latency profiling")
print(f"  brats_summary.json          — machine-readable summary")
print(f"  brats_distillation_gain.json — KD vs scratch comparison")
print(f"\nModels:  {OUT}/models/")
print(f"TFLite:  {OUT}/tflite/")
print(f"\nTo run on your Mac:")
print(f"  python deploy/cli.py --model brats_unet_lite_kd_qat_int8_s42.tflite --image <nifti>")
print("\n" + "=" * 60)
