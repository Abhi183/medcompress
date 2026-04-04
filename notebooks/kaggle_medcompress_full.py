"""
MedCompress Full Experiment Pipeline — Kaggle Notebook
======================================================
GPU: T4 x2 (use 1)
Dataset: ISIC 2020 (Kaggle native), BraTS 2021 (optional)

This notebook trains baselines, runs compression (QAT + KD),
evaluates with extended metrics (AUC, Sensitivity, Specificity, F1),
multi-seed evaluation, exports to TFLite, and profiles CPU latency.

SETUP ON KAGGLE:
1. Create new notebook
2. Add dataset: "siim-isic-melanoma-classification"
3. Set accelerator: GPU T4 x2
4. Paste this entire file into a single code cell, or split at the
   "# %%" markers into separate cells

OUTPUT: trained models, TFLite exports, results CSVs, latency profiles
"""

# %% [markdown]
# # MedCompress: Full Experimental Pipeline
# **Author: Abhishek Shekhar**
#
# Training, compression, and evaluation on ISIC 2020 melanoma classification.

# %% Cell 1: Setup and Imports
import os
import time
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

print(f"TensorFlow: {tf.__version__}")
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Use single GPU
if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices(
        tf.config.list_physical_devices('GPU')[0], 'GPU')

OUTPUT_DIR = "/kaggle/working/medcompress_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/tflite", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

# %% Cell 2: Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
SEEDS = [42, 123, 456, 789, 1024]
ISIC_DATA_DIR = "/kaggle/input/siim-isic-melanoma-classification"

config = {
    "baseline": {
        "epochs": 20,
        "lr": 1e-4,
        "patience": 7,
    },
    "qat": {
        "epochs": 10,
        "lr": 1e-5,
        "calib_samples": 200,
    },
    "kd": {
        "epochs": 25,
        "lr": 1e-4,
        "temperature": 4.0,
        "alpha": 0.7,
    },
    "student_scratch": {
        "epochs": 25,
        "lr": 1e-4,
        "patience": 10,
    },
}


# %% Cell 3: Utility Functions
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """Full classification metrics."""
    y_pred = (y_prob >= threshold).astype(int).ravel()
    y_true_bin = y_true.astype(int).ravel()

    auc = roc_auc_score(y_true_bin, y_prob.ravel())
    f1 = f1_score(y_true_bin, y_pred, zero_division=0)
    precision = precision_score(y_true_bin, y_pred, zero_division=0)
    sensitivity = recall_score(y_true_bin, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def estimate_flops(model, input_shape):
    """Estimate FLOPs for a Keras model."""
    try:
        concrete = tf.function(lambda x: model(x, training=False))
        concrete = concrete.get_concrete_function(
            tf.TensorSpec([1] + list(input_shape), tf.float32))
        from tensorflow.python.framework.convert_to_constants import (
            convert_variables_to_constants_v2)
        frozen = convert_variables_to_constants_v2(concrete)
        flops = tf.compat.v1.profiler.profile(
            frozen.graph,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        )
        return flops.total_float_ops
    except Exception as e:
        print(f"FLOPs estimation failed: {e}, using param-based estimate")
        return model.count_params() * 2


# %% Cell 4: Load ISIC 2020 Dataset
print("=" * 60)
print("LOADING ISIC 2020 DATASET")
print("=" * 60)

csv_path = os.path.join(ISIC_DATA_DIR, "train.csv")
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
print(f"Class distribution:\n{df['target'].value_counts()}")
print(f"Melanoma prevalence: {df['target'].mean():.4f}")

# Paths
jpeg_dir = os.path.join(ISIC_DATA_DIR, "jpeg", "train")
if not os.path.exists(jpeg_dir):
    # Try alternative Kaggle paths
    jpeg_dir = os.path.join(ISIC_DATA_DIR, "train")
    if not os.path.exists(jpeg_dir):
        # List what we have
        for root, dirs, files in os.walk(ISIC_DATA_DIR):
            print(f"  {root}: {len(files)} files, dirs={dirs[:5]}")
            break

df["image_path"] = df["image_name"].apply(
    lambda x: os.path.join(jpeg_dir, f"{x}.jpg"))
# Filter to existing files
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
print(f"Samples with images found: {len(df)}")

# Class weights
counts = df["target"].value_counts()
total = len(df)
class_weights = {
    0: total / (2 * counts[0]),
    1: total / (2 * counts[1]),
}
print(f"Class weights: {class_weights}")


# %% Cell 5: Data Pipeline
def create_dataset(dataframe, augment=False, shuffle=False):
    """Build tf.data pipeline from dataframe."""
    paths = dataframe["image_path"].values
    labels = dataframe["target"].values.astype(np.float32)

    def load_and_preprocess(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0  # [-1, 1]
        return img, label

    def augment_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# Stratified split
train_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["target"], random_state=42)
train_df, val_df = train_test_split(
    train_df, test_size=0.15 / 0.85, stratify=train_df["target"],
    random_state=42)

print(f"\nSplits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
print(f"Train melanoma: {train_df['target'].mean():.4f}")
print(f"Val melanoma:   {val_df['target'].mean():.4f}")
print(f"Test melanoma:  {test_df['target'].mean():.4f}")

train_ds = create_dataset(train_df, augment=True, shuffle=True)
val_ds = create_dataset(val_df)
test_ds = create_dataset(test_df)


# %% Cell 6: Model Builders
def build_efficientnetb0():
    """EfficientNetB0 classifier for ISIC."""
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in backbone.layers[:-20]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="efficientnetb0_isic")


def build_mobilenetv3_small():
    """MobileNetV3-Small student for ISIC."""
    backbone = keras.applications.MobileNetV3Small(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in backbone.layers[:-10]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="mobilenetv3s_isic")


def build_efficientnetb3():
    """EfficientNetB3 teacher for KD."""
    backbone = keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="efficientnetb3_teacher")


# %% Cell 7: Evaluate Model (Extended Metrics)
def evaluate_model(model, dataset, name="model"):
    """Run evaluation with extended metrics."""
    all_preds, all_labels = [], []
    for images, labels in dataset:
        preds = model(images, training=False).numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_classification_metrics(labels, preds)
    print(f"[{name}] AUC={metrics['auc']:.4f}  Sens={metrics['sensitivity']:.4f}  "
          f"Spec={metrics['specificity']:.4f}  F1={metrics['f1']:.4f}")
    return metrics, preds, labels


# %% Cell 8: TFLite Export and Evaluation
def export_tflite(model, output_path, quantize="fp32", calib_ds=None):
    """Export Keras model to TFLite."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calib_ds is not None:
            def representative_dataset():
                for images, _ in calib_ds.take(config["qat"]["calib_samples"] // BATCH_SIZE + 1):
                    for i in range(len(images)):
                        yield [np.expand_dims(images[i].numpy(), axis=0)]
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


def evaluate_tflite(tflite_path, dataset, n_warmup=5):
    """Evaluate TFLite model with latency measurement."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    input_dtype = inp["dtype"]

    all_preds, all_labels, latencies = [], [], []
    for batch_images, batch_labels in dataset:
        for i in range(len(batch_images)):
            img = np.expand_dims(batch_images[i].numpy(), axis=0)

            if input_dtype == np.uint8:
                scale, zp = inp["quantization"]
                img = (img / scale + zp).astype(np.uint8)

            interpreter.set_tensor(inp["index"], img)
            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            result = interpreter.get_tensor(out["index"])
            if out["dtype"] == np.uint8:
                scale, zp = out["quantization"]
                result = (result.astype(np.float32) - zp) * scale

            all_preds.append(result.squeeze())
            all_labels.append(batch_labels[i].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    lat = np.array(latencies[n_warmup:])  # skip warmup

    metrics = compute_classification_metrics(labels, preds)
    metrics["latency_median_ms"] = float(np.median(lat))
    metrics["latency_p95_ms"] = float(np.percentile(lat, 95))
    metrics["model_size_mb"] = os.path.getsize(tflite_path) / 1e6

    print(f"  TFLite AUC={metrics['auc']:.4f}  "
          f"Latency={metrics['latency_median_ms']:.1f}ms (median)  "
          f"Size={metrics['model_size_mb']:.1f}MB")
    return metrics


# %% Cell 9: EXPERIMENT 1 — Baseline EfficientNetB0
print("\n" + "=" * 60)
print("EXPERIMENT 1: BASELINE EfficientNetB0")
print("=" * 60)

baseline_results = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    model = build_efficientnetb0()
    model.compile(
        optimizer=keras.optimizers.Adam(config["baseline"]["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")])

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=config["baseline"]["epochs"],
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=config["baseline"]["patience"],
                mode="max", restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_auc", factor=0.5, patience=3, mode="max"),
        ],
        verbose=1)

    metrics, _, _ = evaluate_model(model, test_ds, f"Baseline seed={seed}")
    baseline_results.append(metrics)

# Save best model (last seed for simplicity)
model.save(f"{OUTPUT_DIR}/models/efficientnetb0_baseline.keras")

# Report mean +/- std
print("\n--- Baseline Summary (5 seeds) ---")
for key in ["auc", "sensitivity", "specificity", "f1"]:
    values = [r[key] for r in baseline_results]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

# FLOPs
flops = estimate_flops(model, (IMG_SIZE, IMG_SIZE, 3))
print(f"  FLOPs: {flops:,} ({flops/1e9:.2f} GFLOPs)")
print(f"  Params: {model.count_params():,}")


# %% Cell 10: EXPERIMENT 2 — QAT INT8 + FP16
print("\n" + "=" * 60)
print("EXPERIMENT 2: QUANTIZATION-AWARE TRAINING")
print("=" * 60)

qat_results_int8 = []
qat_results_fp16 = []

for seed in SEEDS:
    print(f"\n--- QAT Seed {seed} ---")
    set_seed(seed)

    # Load baseline
    base = keras.models.load_model(
        f"{OUTPUT_DIR}/models/efficientnetb0_baseline.keras")

    # Apply QAT
    qat_model = tfmot.quantization.keras.quantize_model(base)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(config["qat"]["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")])

    qat_model.fit(
        train_ds, validation_data=val_ds,
        epochs=config["qat"]["epochs"],
        class_weight=class_weights,
        verbose=1)

    # Strip and export INT8
    stripped = tfmot.quantization.keras.strip_pruning(qat_model)
    int8_path = f"{OUTPUT_DIR}/tflite/efficientnetb0_qat_int8_s{seed}.tflite"
    export_tflite(stripped, int8_path, quantize="int8", calib_ds=train_ds)
    metrics_int8 = evaluate_tflite(int8_path, test_ds)
    qat_results_int8.append(metrics_int8)

    # Export FP16
    fp16_path = f"{OUTPUT_DIR}/tflite/efficientnetb0_qat_fp16_s{seed}.tflite"
    export_tflite(stripped, fp16_path, quantize="fp16")
    metrics_fp16 = evaluate_tflite(fp16_path, test_ds)
    qat_results_fp16.append(metrics_fp16)

print("\n--- QAT INT8 Summary ---")
for key in ["auc", "sensitivity", "specificity", "f1", "latency_median_ms", "model_size_mb"]:
    values = [r[key] for r in qat_results_int8]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

print("\n--- QAT FP16 Summary ---")
for key in ["auc", "sensitivity", "specificity", "f1", "latency_median_ms", "model_size_mb"]:
    values = [r[key] for r in qat_results_fp16]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# %% Cell 11: EXPERIMENT 3a — Student from Scratch (MobileNetV3-Small)
print("\n" + "=" * 60)
print("EXPERIMENT 3a: STUDENT FROM SCRATCH (MobileNetV3-Small)")
print("=" * 60)

scratch_results = []
for seed in SEEDS:
    print(f"\n--- Scratch Seed {seed} ---")
    set_seed(seed)
    student = build_mobilenetv3_small()
    student.compile(
        optimizer=keras.optimizers.Adam(config["student_scratch"]["lr"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")])

    student.fit(
        train_ds, validation_data=val_ds,
        epochs=config["student_scratch"]["epochs"],
        class_weight=class_weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=config["student_scratch"]["patience"],
                mode="max", restore_best_weights=True),
        ],
        verbose=1)

    metrics, _, _ = evaluate_model(student, test_ds, f"Scratch seed={seed}")
    scratch_results.append(metrics)

student.save(f"{OUTPUT_DIR}/models/mobilenetv3s_scratch.keras")

print("\n--- Student Scratch Summary ---")
for key in ["auc", "sensitivity", "specificity", "f1"]:
    values = [r[key] for r in scratch_results]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# %% Cell 12: EXPERIMENT 3b — Teacher Training (EfficientNetB3)
print("\n" + "=" * 60)
print("EXPERIMENT 3b: TEACHER (EfficientNetB3)")
print("=" * 60)

set_seed(42)
teacher = build_efficientnetb3()
teacher.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")])

teacher.fit(
    train_ds, validation_data=val_ds,
    epochs=20, class_weight=class_weights,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=7, mode="max",
            restore_best_weights=True),
    ],
    verbose=1)

teacher_metrics, _, _ = evaluate_model(teacher, test_ds, "Teacher B3")
teacher.save(f"{OUTPUT_DIR}/models/efficientnetb3_teacher.keras")


# %% Cell 13: EXPERIMENT 3c — Knowledge Distillation
print("\n" + "=" * 60)
print("EXPERIMENT 3c: KNOWLEDGE DISTILLATION")
print("=" * 60)

T = config["kd"]["temperature"]
ALPHA = config["kd"]["alpha"]

kd_results = []
for seed in SEEDS:
    print(f"\n--- KD Seed {seed} ---")
    set_seed(seed)

    student_kd = build_mobilenetv3_small()
    optimizer = keras.optimizers.Adam(config["kd"]["lr"])

    # Custom KD training loop
    for epoch in range(config["kd"]["epochs"]):
        epoch_loss = 0
        n_batches = 0
        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                teacher_logits = teacher(images, training=False)
                student_logits = student_kd(images, training=True)

                # Soft targets (temperature-scaled)
                teacher_soft = tf.nn.sigmoid(
                    tf.math.log(teacher_logits / (1 - teacher_logits + 1e-7) + 1e-7) / T)
                student_soft = tf.nn.sigmoid(
                    tf.math.log(student_logits / (1 - student_logits + 1e-7) + 1e-7) / T)

                # KL divergence (binary)
                eps = 1e-7
                kl = teacher_soft * tf.math.log(
                    (teacher_soft + eps) / (student_soft + eps))
                kl += (1 - teacher_soft) * tf.math.log(
                    (1 - teacher_soft + eps) / (1 - student_soft + eps))
                distill_loss = tf.reduce_mean(kl) * (T ** 2)

                # Hard label loss
                hard_loss = tf.reduce_mean(
                    keras.losses.binary_crossentropy(
                        tf.expand_dims(labels, -1), student_logits))

                loss = ALPHA * distill_loss + (1 - ALPHA) * hard_loss

            grads = tape.gradient(loss, student_kd.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, student_kd.trainable_variables))
            epoch_loss += loss.numpy()
            n_batches += 1

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

    metrics, _, _ = evaluate_model(student_kd, test_ds, f"KD seed={seed}")
    kd_results.append(metrics)

student_kd.save(f"{OUTPUT_DIR}/models/mobilenetv3s_kd.keras")

print("\n--- KD Summary ---")
for key in ["auc", "sensitivity", "specificity", "f1"]:
    values = [r[key] for r in kd_results]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

# Distillation gain
scratch_auc = np.mean([r["auc"] for r in scratch_results])
kd_auc = np.mean([r["auc"] for r in kd_results])
print(f"\n*** DISTILLATION GAIN: {(kd_auc - scratch_auc)*100:+.2f}% AUC ***")
print(f"    Scratch: {scratch_auc:.4f}  vs  KD: {kd_auc:.4f}")


# %% Cell 14: EXPERIMENT 4 — KD + QAT Combined
print("\n" + "=" * 60)
print("EXPERIMENT 4: KD + QAT INT8")
print("=" * 60)

kd_qat_results = []
for seed in SEEDS:
    print(f"\n--- KD+QAT Seed {seed} ---")
    set_seed(seed)

    base_kd = keras.models.load_model(
        f"{OUTPUT_DIR}/models/mobilenetv3s_kd.keras")
    qat_kd = tfmot.quantization.keras.quantize_model(base_kd)
    qat_kd.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")])

    qat_kd.fit(train_ds, validation_data=val_ds,
               epochs=config["qat"]["epochs"],
               class_weight=class_weights, verbose=0)

    stripped = tfmot.quantization.keras.strip_pruning(qat_kd)
    path = f"{OUTPUT_DIR}/tflite/mobilenetv3s_kd_qat_int8_s{seed}.tflite"
    export_tflite(stripped, path, quantize="int8", calib_ds=train_ds)
    metrics = evaluate_tflite(path, test_ds)
    kd_qat_results.append(metrics)

print("\n--- KD+QAT INT8 Summary ---")
for key in ["auc", "sensitivity", "specificity", "f1", "latency_median_ms", "model_size_mb"]:
    values = [r[key] for r in kd_qat_results]
    print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# %% Cell 15: Compile All Results
print("\n" + "=" * 60)
print("COMPILING FINAL RESULTS")
print("=" * 60)

def summarize(results_list, name):
    row = {"method": name}
    for key in ["auc", "sensitivity", "specificity", "f1"]:
        values = [r[key] for r in results_list]
        row[f"{key}_mean"] = round(np.mean(values), 4)
        row[f"{key}_std"] = round(np.std(values), 4)
    if "latency_median_ms" in results_list[0]:
        lat = [r["latency_median_ms"] for r in results_list]
        row["latency_mean"] = round(np.mean(lat), 1)
        sz = [r["model_size_mb"] for r in results_list]
        row["size_mb"] = round(np.mean(sz), 1)
    return row

all_results = [
    summarize(baseline_results, "Baseline FP32"),
    summarize(qat_results_int8, "QAT INT8"),
    summarize(qat_results_fp16, "QAT FP16"),
    summarize(scratch_results, "Student Scratch"),
    summarize(kd_results, "KD (T=4, a=0.7)"),
    summarize(kd_qat_results, "KD + QAT INT8"),
]

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{OUTPUT_DIR}/results/isic_full_results.csv", index=False)
print(results_df.to_string(index=False))

# Distillation gain table
gain_df = pd.DataFrame([
    {"method": "Scratch", "auc": scratch_auc,
     "auc_std": np.std([r["auc"] for r in scratch_results])},
    {"method": "KD", "auc": kd_auc,
     "auc_std": np.std([r["auc"] for r in kd_results])},
    {"method": "Gain", "auc": kd_auc - scratch_auc, "auc_std": 0},
])
gain_df.to_csv(f"{OUTPUT_DIR}/results/distillation_gain.csv", index=False)
print("\nDistillation Gain:")
print(gain_df.to_string(index=False))


# %% Cell 16: Export Final TFLite Models for Endpoint Testing
print("\n" + "=" * 60)
print("EXPORTING FINAL MODELS FOR ENDPOINT DEPLOYMENT")
print("=" * 60)

# Best baseline -> FP32 TFLite
base_fp32_path = f"{OUTPUT_DIR}/tflite/efficientnetb0_baseline_fp32.tflite"
base_model = keras.models.load_model(
    f"{OUTPUT_DIR}/models/efficientnetb0_baseline.keras")
export_tflite(base_model, base_fp32_path, quantize="fp32")

# Best KD student -> FP32 TFLite
kd_fp32_path = f"{OUTPUT_DIR}/tflite/mobilenetv3s_kd_fp32.tflite"
kd_model = keras.models.load_model(
    f"{OUTPUT_DIR}/models/mobilenetv3s_kd.keras")
export_tflite(kd_model, kd_fp32_path, quantize="fp32")

print("\n--- Models ready for endpoint testing ---")
for f in sorted(Path(f"{OUTPUT_DIR}/tflite").glob("*.tflite")):
    print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")

print(f"\nDownload from: {OUTPUT_DIR}/tflite/")
print("Then on your Mac: python deploy/cli.py --model <file>.tflite --image tests/sample_lesion.jpg")


# %% Cell 17: CPU Latency Profiling (on Kaggle CPU)
print("\n" + "=" * 60)
print("CPU LATENCY PROFILING (Kaggle CPU, simulates endpoint)")
print("=" * 60)

# Profile each exported model on CPU
tflite_dir = Path(f"{OUTPUT_DIR}/tflite")
latency_results = []

for tflite_file in sorted(tflite_dir.glob("*_s42.tflite")):
    print(f"\nProfiling: {tflite_file.name}")
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_file), num_threads=4)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]

    # Generate dummy input
    if inp["dtype"] == np.uint8:
        dummy = np.random.randint(0, 255, size=inp["shape"]).astype(np.uint8)
    else:
        dummy = np.random.randn(*inp["shape"]).astype(np.float32)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(inp["index"], dummy)
        interpreter.invoke()

    # Measure
    times = []
    for _ in range(100):
        interpreter.set_tensor(inp["index"], dummy)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times = np.array(times)
    result = {
        "model": tflite_file.name,
        "size_mb": tflite_file.stat().st_size / 1e6,
        "latency_median_ms": float(np.median(times)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_min_ms": float(np.min(times)),
        "latency_max_ms": float(np.max(times)),
    }
    latency_results.append(result)
    print(f"  Median: {result['latency_median_ms']:.1f} ms  "
          f"P95: {result['latency_p95_ms']:.1f} ms  "
          f"Size: {result['size_mb']:.1f} MB")

latency_df = pd.DataFrame(latency_results)
latency_df.to_csv(f"{OUTPUT_DIR}/results/cpu_latency_profile.csv", index=False)


# %% Cell 18: Summary
print("\n" + "=" * 60)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {OUTPUT_DIR}/results/")
print(f"Models saved to: {OUTPUT_DIR}/models/")
print(f"TFLite exports: {OUTPUT_DIR}/tflite/")
print("\nDownload the tflite/ folder and run on your Mac:")
print("  python deploy/cli.py --model mobilenetv3s_kd_qat_int8_s42.tflite --image tests/sample_lesion.jpg")
print("  python demo.py  # then load the .tflite model")
print("\n" + "=" * 60)
