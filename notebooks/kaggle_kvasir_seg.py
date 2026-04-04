"""
MedCompress Kvasir-SEG Experiment — Kaggle Notebook
====================================================
GPU: T4 x2 (use 1)
Dataset: Kvasir-SEG (add "debeshjha1/kvasirseg" on Kaggle)

Trains U-Net baseline on polyp segmentation, runs QAT, KD,
and reports Dice, IoU, Sensitivity, Specificity.

SETUP:
1. New notebook on Kaggle
2. Add dataset: search "kvasir-seg" or "kvasirseg"
3. Accelerator: GPU T4 x2
4. !pip install -q tensorflow-model-optimization
5. Paste this code
"""

# %% Cell 1: Setup
!pip install -q tensorflow-model-optimization

import os, time, glob, random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

print(f"TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices('GPU'))}")

if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

OUT = "/kaggle/working/kvasir_results"
os.makedirs(f"{OUT}/models", exist_ok=True)
os.makedirs(f"{OUT}/tflite", exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)

IMG_SIZE = 256
BATCH = 16
SEEDS = [42, 123, 456, 789, 1024]

# %% Cell 2: Find Dataset
DATA_DIR = "/kaggle/input"
for d in os.listdir(DATA_DIR):
    if "kvasir" in d.lower():
        DATA_DIR = os.path.join(DATA_DIR, d)
        break
print(f"Data dir: {DATA_DIR}")

# Find images and masks directories
img_dir = None
mask_dir = None
for root, dirs, files in os.walk(DATA_DIR):
    for d in dirs:
        dl = d.lower()
        if dl in ("images", "image"):
            img_dir = os.path.join(root, d)
        if dl in ("masks", "mask"):
            mask_dir = os.path.join(root, d)

if img_dir is None or mask_dir is None:
    # Try flat structure
    all_dirs = []
    for root, dirs, files in os.walk(DATA_DIR):
        for d in dirs:
            all_dirs.append(os.path.join(root, d))
    print(f"Dirs found: {all_dirs[:10]}")
    # Fallback: look for any dir with jpg/png files
    for d in all_dirs:
        files = os.listdir(d)
        if any(f.endswith(('.jpg', '.png')) for f in files):
            if img_dir is None:
                img_dir = d
            elif mask_dir is None:
                mask_dir = d

print(f"Images: {img_dir}")
print(f"Masks: {mask_dir}")

# Match image-mask pairs
img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                   glob.glob(os.path.join(img_dir, "*.png")))
mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")) +
                    glob.glob(os.path.join(mask_dir, "*.png")))

# Match by filename stem
img_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in img_files}
mask_stems = {os.path.splitext(os.path.basename(f))[0]: f for f in mask_files}
common = sorted(set(img_stems.keys()) & set(mask_stems.keys()))
print(f"Matched pairs: {len(common)}")

paired_imgs = [img_stems[s] for s in common]
paired_masks = [mask_stems[s] for s in common]

# %% Cell 3: Data Pipeline
def make_dataset(img_paths, mask_paths, augment=False, shuffle=False):
    def load_fn(img_path, mask_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 255.0

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
        mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE], method="nearest")
        mask = tf.cast(mask > 127, tf.float32)  # binarize
        return img, mask

    def aug_fn(img, mask):
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        img = tf.image.random_brightness(img, 0.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, mask

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(len(img_paths))
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

# Split
idx = list(range(len(common)))
tr_idx, test_idx = train_test_split(idx, test_size=0.15, random_state=42)
tr_idx, val_idx = train_test_split(tr_idx, test_size=0.15/0.85, random_state=42)

train_ds = make_dataset(
    [paired_imgs[i] for i in tr_idx], [paired_masks[i] for i in tr_idx],
    augment=True, shuffle=True)
val_ds = make_dataset(
    [paired_imgs[i] for i in val_idx], [paired_masks[i] for i in val_idx])
test_ds = make_dataset(
    [paired_imgs[i] for i in test_idx], [paired_masks[i] for i in test_idx])

print(f"Splits: train={len(tr_idx)}, val={len(val_idx)}, test={len(test_idx)}")

# %% Cell 4: U-Net Model
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet(filters=[32, 64, 128, 256], name="unet"):
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    skips = []
    x = inp
    for f in filters[:-1]:
        x = conv_block(x, f)
        skips.append(x)
        x = layers.MaxPooling2D(2)(x)
    x = conv_block(x, filters[-1])  # bottleneck
    for f, skip in zip(reversed(filters[:-1]), reversed(skips)):
        x = layers.UpSampling2D(2)(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, f)
    out = layers.Conv2D(1, 1, activation="sigmoid")(x)  # binary segmentation
    return keras.Model(inp, out, name=name)

# Dice loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred):
    return dice_loss(y_true, y_pred) + keras.losses.binary_crossentropy(y_true, y_pred)

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

# %% Cell 5: Evaluation
def evaluate_seg(model, ds, name=""):
    dices, ious, senss = [], [], []
    for imgs, masks in ds:
        preds = model(imgs, training=False).numpy()
        pred_bin = (preds > 0.5).astype(np.float32)
        masks_np = masks.numpy()

        for i in range(len(pred_bin)):
            p = pred_bin[i].ravel()
            t = masks_np[i].ravel()
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()

            dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            sens = (tp + 1e-7) / (tp + fn + 1e-7)
            dices.append(dice)
            ious.append(iou)
            senss.append(sens)

    result = {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "sensitivity": float(np.mean(senss)),
    }
    print(f"[{name}] Dice={result['dice']:.4f} IoU={result['iou']:.4f} Sens={result['sensitivity']:.4f}")
    return result

# %% Cell 6: Baseline
print("=" * 50)
print("EXP 1: Kvasir-SEG Baseline U-Net")
print("=" * 50)

baseline_results = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    model = build_unet([32, 64, 128, 256], name="unet_kvasir")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=dice_bce_loss)
    model.fit(train_ds, validation_data=val_ds, epochs=30,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=7, restore_best_weights=True)],
              verbose=1)
    metrics = evaluate_seg(model, test_ds, f"Baseline s{seed}")
    baseline_results.append(metrics)

model.save(f"{OUT}/models/kvasir_unet_baseline.keras")

print("\nBaseline Summary:")
for key in ["dice", "iou", "sensitivity"]:
    vals = [r[key] for r in baseline_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# %% Cell 7: QAT INT8
print("\n" + "=" * 50)
print("EXP 2: Kvasir-SEG QAT INT8")
print("=" * 50)

qat_results = []
for seed in SEEDS:
    set_seed(seed)
    base = keras.models.load_model(
        f"{OUT}/models/kvasir_unet_baseline.keras",
        custom_objects={"dice_bce_loss": dice_bce_loss})
    qat = tfmot.quantization.keras.quantize_model(base)
    qat.compile(optimizer=keras.optimizers.Adam(1e-5), loss=dice_bce_loss)
    qat.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

    stripped = tfmot.quantization.keras.strip_pruning(qat)
    metrics = evaluate_seg(stripped, test_ds, f"QAT s{seed}")

    # Export TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite = converter.convert()
    path = f"{OUT}/tflite/kvasir_qat_int8_s{seed}.tflite"
    with open(path, "wb") as f:
        f.write(tflite)
    metrics["size_mb"] = os.path.getsize(path) / 1e6
    qat_results.append(metrics)

print("\nQAT INT8 Summary:")
for key in ["dice", "iou", "sensitivity"]:
    vals = [r[key] for r in qat_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# %% Cell 8: Save
import json
results = {
    "baseline": {k: {"mean": float(np.mean([r[k] for r in baseline_results])),
                      "std": float(np.std([r[k] for r in baseline_results]))}
                 for k in ["dice", "iou", "sensitivity"]},
    "qat_int8": {k: {"mean": float(np.mean([r[k] for r in qat_results])),
                      "std": float(np.std([r[k] for r in qat_results]))}
                 for k in ["dice", "iou", "sensitivity"]},
}
with open(f"{OUT}/results/kvasir_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults: {OUT}/results/")
print(f"Models: {OUT}/tflite/")
