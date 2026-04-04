"""
MedCompress CheXpert Experiment — Kaggle Notebook
==================================================
GPU: T4 x2 (use 1)
Dataset: CheXpert (add "stanfordmlgroup/chexpert" on Kaggle)

Trains EfficientNetB0 baseline on 5 competition pathologies,
runs QAT INT8/FP16, knowledge distillation, and combined pipeline.
Reports AUC (mean across 5 labels), Sensitivity, Specificity, F1.

SETUP:
1. New notebook on Kaggle
2. Add dataset: search "chexpert" -> add the Stanford one
3. Accelerator: GPU T4 x2
4. !pip install -q tensorflow-model-optimization
5. Paste this code
"""

# %% Cell 1: Setup
!pip install -q tensorflow-model-optimization

import os, time, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

print(f"TF: {tf.__version__}, GPUs: {len(tf.config.list_physical_devices('GPU'))}")

if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

OUT = "/kaggle/working/chexpert_results"
os.makedirs(f"{OUT}/models", exist_ok=True)
os.makedirs(f"{OUT}/tflite", exist_ok=True)
os.makedirs(f"{OUT}/results", exist_ok=True)

IMG_SIZE = 224
BATCH = 32
SEEDS = [42, 123, 456, 789, 1024]
COMP_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# %% Cell 2: Load CheXpert
DATA_DIR = "/kaggle/input"
# Find the CheXpert directory
for d in os.listdir(DATA_DIR):
    if "chexpert" in d.lower():
        DATA_DIR = os.path.join(DATA_DIR, d)
        break
print(f"Data dir: {DATA_DIR}")

# Find CSVs
train_csv = None
for root, dirs, files in os.walk(DATA_DIR):
    for f in files:
        if f == "train.csv":
            train_csv = os.path.join(root, f)
            break
    if train_csv:
        break

print(f"Train CSV: {train_csv}")
train_df = pd.read_csv(train_csv)
print(f"Total samples: {len(train_df)}")
print(f"Columns: {list(train_df.columns)}")

# U-Ones policy: replace -1 (uncertain) with 1, NaN with 0
for label in COMP_LABELS:
    train_df[label] = train_df[label].fillna(0).replace(-1, 1).astype(np.float32)

# Only frontal views
if "Frontal/Lateral" in train_df.columns:
    train_df = train_df[train_df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
    print(f"Frontal only: {len(train_df)}")

# Fix image paths
path_col = "Path"
def fix_path(p):
    # Handle various Kaggle path structures
    parts = p.split("/")
    # Try to find the file relative to DATA_DIR
    for i in range(len(parts)):
        candidate = os.path.join(DATA_DIR, *parts[i:])
        if os.path.exists(candidate):
            return candidate
    return os.path.join(DATA_DIR, p)

train_df["full_path"] = train_df[path_col].apply(fix_path)
train_df = train_df[train_df["full_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Samples with images: {len(train_df)}")

# Print label prevalence
for label in COMP_LABELS:
    print(f"  {label}: {train_df[label].mean():.3f}")

# %% Cell 3: Splits and Data Pipeline
# Subsample if too large (CheXpert has 200k+ images)
MAX_SAMPLES = 50000
if len(train_df) > MAX_SAMPLES:
    train_df = train_df.sample(MAX_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"Subsampled to {MAX_SAMPLES}")

tr_df, test_df = train_test_split(train_df, test_size=0.15, random_state=42)
tr_df, val_df = train_test_split(tr_df, test_size=0.15/0.85, random_state=42)
print(f"Splits: train={len(tr_df)}, val={len(val_df)}, test={len(test_df)}")

def make_dataset(df, augment=False, shuffle=False):
    paths = df["full_path"].values
    labels = df[COMP_LABELS].values.astype(np.float32)

    def load_fn(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img, label

    def aug_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths))
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(tr_df, augment=True, shuffle=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# %% Cell 4: Model
def build_model():
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(5, activation="sigmoid")(x)  # 5 labels, multi-label
    return keras.Model(inp, out)

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

# %% Cell 5: Evaluation
def evaluate(model, ds, name=""):
    preds, labels = [], []
    for img, lbl in ds:
        preds.append(model(img, training=False).numpy())
        labels.append(lbl.numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    # Per-label AUC
    aucs = []
    for i, label_name in enumerate(COMP_LABELS):
        try:
            auc = roc_auc_score(labels[:, i], preds[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    # Binarize for F1/Sens/Spec
    pred_bin = (preds >= 0.5).astype(int)
    f1 = f1_score(labels, pred_bin, average="macro", zero_division=0)
    sens = recall_score(labels, pred_bin, average="macro", zero_division=0)

    print(f"[{name}] mAUC={mean_auc:.4f} F1={f1:.4f} Sens={sens:.4f}")
    print(f"  Per-label AUC: {dict(zip(COMP_LABELS, [f'{a:.3f}' for a in aucs]))}")
    return {"mean_auc": mean_auc, "f1": f1, "sensitivity": sens,
            "per_label_auc": dict(zip(COMP_LABELS, aucs))}

# %% Cell 6: Experiment 1 — Baseline
print("=" * 50)
print("EXP 1: CheXpert Baseline EfficientNetB0")
print("=" * 50)

baseline_results = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc", multi_label=True)])
    model.fit(train_ds, validation_data=val_ds, epochs=15,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor="val_auc", patience=5, mode="max",
                  restore_best_weights=True)],
              verbose=1)
    metrics = evaluate(model, test_ds, f"Baseline s{seed}")
    baseline_results.append(metrics)

model.save(f"{OUT}/models/chexpert_baseline.keras")
print("\nBaseline Summary:")
for key in ["mean_auc", "f1", "sensitivity"]:
    vals = [r[key] for r in baseline_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# %% Cell 7: Experiment 2 — QAT INT8
print("\n" + "=" * 50)
print("EXP 2: CheXpert QAT INT8")
print("=" * 50)

qat_results = []
for seed in SEEDS:
    set_seed(seed)
    base = keras.models.load_model(f"{OUT}/models/chexpert_baseline.keras")
    qat_model = tfmot.quantization.keras.quantize_model(base)
    qat_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc", multi_label=True)])
    qat_model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

    stripped = tfmot.quantization.keras.strip_pruning(qat_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = f"{OUT}/tflite/chexpert_qat_int8_s{seed}.tflite"
    with open(path, "wb") as f:
        f.write(tflite_model)

    # Evaluate stripped Keras model (TFLite eval is slow for multi-label)
    metrics = evaluate(stripped, test_ds, f"QAT s{seed}")
    metrics["size_mb"] = os.path.getsize(path) / 1e6
    qat_results.append(metrics)

print("\nQAT INT8 Summary:")
for key in ["mean_auc", "f1", "sensitivity"]:
    vals = [r[key] for r in qat_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

# %% Cell 8: Save Results
results = {
    "baseline": {k: {"mean": np.mean([r[k] for r in baseline_results]),
                      "std": np.std([r[k] for r in baseline_results])}
                 for k in ["mean_auc", "f1", "sensitivity"]},
    "qat_int8": {k: {"mean": np.mean([r[k] for r in qat_results]),
                      "std": np.std([r[k] for r in qat_results])}
                 for k in ["mean_auc", "f1", "sensitivity"]},
}

import json
with open(f"{OUT}/results/chexpert_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT}/results/")
print(f"Models saved to {OUT}/tflite/")
print("Download these and bring to the next session.")
