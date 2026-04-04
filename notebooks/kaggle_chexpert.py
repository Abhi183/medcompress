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

# %% Cell 8: Experiment 3 — Student MobileNetV3Small from Scratch
print("\n" + "=" * 50)
print("EXP 3: CheXpert MobileNetV3Small from Scratch (no KD)")
print("=" * 50)

def build_student():
    backbone = keras.applications.MobileNetV3Small(
        include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(5, activation="sigmoid")(x)
    return keras.Model(inp, out)

scratch_results = []
for seed in SEEDS:
    print(f"\n--- Seed {seed} ---")
    set_seed(seed)
    student = build_student()
    student.compile(optimizer=keras.optimizers.Adam(1e-4),
                    loss="binary_crossentropy",
                    metrics=[keras.metrics.AUC(name="auc", multi_label=True)])
    student.fit(train_ds, validation_data=val_ds, epochs=25,
                callbacks=[keras.callbacks.EarlyStopping(
                    monitor="val_auc", patience=7, mode="max",
                    restore_best_weights=True)],
                verbose=1)
    metrics = evaluate(student, test_ds, f"Scratch s{seed}")
    scratch_results.append(metrics)

student.save(f"{OUT}/models/chexpert_mobilenetv3s_scratch.keras")
print("\nScratch Student Summary:")
for key in ["mean_auc", "f1", "sensitivity"]:
    vals = [r[key] for r in scratch_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


# %% Cell 9: Experiment 4 — Knowledge Distillation
print("\n" + "=" * 50)
print("EXP 4: CheXpert Knowledge Distillation (T=4, alpha=0.7)")
print("=" * 50)

TEMPERATURE = 4.0
ALPHA = 0.7

class DistillationModel(keras.Model):
    def __init__(self, student, teacher, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        self.teacher.trainable = False

    def compile(self, optimizer, student_loss, temperature=4.0, alpha=0.7, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)
        self.student_loss_fn = student_loss
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, data):
        x, y = data
        teacher_pred = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            student_pred = self.student(x, training=True)
            # Soft targets (KL divergence on tempered logits)
            t_soft = tf.sigmoid(tf.math.log(teacher_pred / (1 - teacher_pred + 1e-7)) / self.temperature)
            s_soft = tf.sigmoid(tf.math.log(student_pred / (1 - student_pred + 1e-7)) / self.temperature)
            distill_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(t_soft, s_soft)) * (self.temperature ** 2)
            # Hard targets
            hard_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y, student_pred))
            loss = self.alpha * distill_loss + (1 - self.alpha) * hard_loss

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return {"loss": loss, "hard_loss": hard_loss, "distill_loss": distill_loss}

    def call(self, x, training=False):
        return self.student(x, training=training)

kd_results = []
teacher = keras.models.load_model(f"{OUT}/models/chexpert_baseline.keras")

for seed in SEEDS:
    print(f"\n--- KD Seed {seed} ---")
    set_seed(seed)
    student = build_student()
    distiller = DistillationModel(student, teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        student_loss="binary_crossentropy",
        temperature=TEMPERATURE, alpha=ALPHA)
    distiller.fit(train_ds, epochs=25, verbose=1)
    metrics = evaluate(student, test_ds, f"KD s{seed}")
    kd_results.append(metrics)

student.save(f"{OUT}/models/chexpert_mobilenetv3s_kd.keras")

scratch_auc = np.mean([r["mean_auc"] for r in scratch_results])
kd_auc = np.mean([r["mean_auc"] for r in kd_results])
print(f"\n*** DISTILLATION GAIN: {(kd_auc - scratch_auc)*100:+.2f}% mAUC ***")
print(f"    Scratch: {scratch_auc:.4f}  vs  KD: {kd_auc:.4f}")


# %% Cell 10: Experiment 5 — KD + QAT INT8 Combined
print("\n" + "=" * 50)
print("EXP 5: CheXpert KD + QAT INT8")
print("=" * 50)

kd_qat_results = []
for seed in SEEDS:
    print(f"\n--- KD+QAT Seed {seed} ---")
    set_seed(seed)
    kd_model = keras.models.load_model(f"{OUT}/models/chexpert_mobilenetv3s_kd.keras")
    qat_model = tfmot.quantization.keras.quantize_model(kd_model)
    qat_model.compile(optimizer=keras.optimizers.Adam(1e-5),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc", multi_label=True)])
    qat_model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)

    stripped = tfmot.quantization.keras.strip_pruning(qat_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    path = f"{OUT}/tflite/chexpert_kd_qat_int8_s{seed}.tflite"
    with open(path, "wb") as f:
        f.write(tflite_model)

    metrics = evaluate(stripped, test_ds, f"KD+QAT s{seed}")
    metrics["size_mb"] = os.path.getsize(path) / 1e6
    kd_qat_results.append(metrics)

print("\nKD+QAT INT8 Summary:")
for key in ["mean_auc", "f1", "sensitivity"]:
    vals = [r[key] for r in kd_qat_results]
    print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


# %% Cell 11: TFLite CPU Latency Profiling
print("\n" + "=" * 50)
print("CPU LATENCY PROFILING")
print("=" * 50)

def profile_tflite(path, n_runs=50):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    shape = inp_detail["shape"]
    dtype = inp_detail["dtype"]
    dummy = np.random.randn(*shape).astype(dtype)
    # Warmup
    for _ in range(5):
        interpreter.set_tensor(inp_detail["index"], dummy)
        interpreter.invoke()
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.time()
        interpreter.set_tensor(inp_detail["index"], dummy)
        interpreter.invoke()
        times.append((time.time() - start) * 1000)
    return {"median_ms": np.median(times), "p95_ms": np.percentile(times, 95),
            "size_mb": os.path.getsize(path) / 1e6}

# Profile best models
tflite_dir = f"{OUT}/tflite"
for fname in sorted(os.listdir(tflite_dir)):
    if fname.endswith(".tflite") and "s42" in fname:
        path = os.path.join(tflite_dir, fname)
        stats = profile_tflite(path)
        print(f"  {fname}: {stats['median_ms']:.1f} ms (P95: {stats['p95_ms']:.1f} ms), "
              f"{stats['size_mb']:.1f} MB")


# %% Cell 12: Compile and Save All Results
print("\n" + "=" * 50)
print("COMPILING FINAL RESULTS")
print("=" * 50)

def summarize(results_list, name, metric_key="mean_auc"):
    row = {"method": name}
    for key in [metric_key, "f1", "sensitivity"]:
        values = [r[key] for r in results_list]
        row[f"{key}_mean"] = round(np.mean(values), 4)
        row[f"{key}_std"] = round(np.std(values), 4)
    if "size_mb" in results_list[0]:
        row["size_mb"] = round(np.mean([r["size_mb"] for r in results_list]), 1)
    return row

all_results = [
    summarize(baseline_results, "Baseline FP32"),
    summarize(qat_results, "QAT INT8"),
    summarize(scratch_results, "Student Scratch"),
    summarize(kd_results, "KD (T=4, a=0.7)"),
    summarize(kd_qat_results, "KD + QAT INT8"),
]

results_df = pd.DataFrame(all_results)
results_df.to_csv(f"{OUT}/results/chexpert_full_results.csv", index=False)
print(results_df.to_string(index=False))

# Distillation gain table
gain_df = pd.DataFrame([
    {"method": "Scratch", "mean_auc": scratch_auc,
     "std": np.std([r["mean_auc"] for r in scratch_results])},
    {"method": "KD", "mean_auc": kd_auc,
     "std": np.std([r["mean_auc"] for r in kd_results])},
    {"method": "Gain", "mean_auc": kd_auc - scratch_auc, "std": 0},
])
gain_df.to_csv(f"{OUT}/results/chexpert_distillation_gain.csv", index=False)
print("\nDistillation Gain:")
print(gain_df.to_string(index=False))

import json
with open(f"{OUT}/results/chexpert_results.json", "w") as f:
    json.dump({
        "baseline": {k: {"mean": float(np.mean([r[k] for r in baseline_results])),
                          "std": float(np.std([r[k] for r in baseline_results]))}
                     for k in ["mean_auc", "f1", "sensitivity"]},
        "qat_int8": {k: {"mean": float(np.mean([r[k] for r in qat_results])),
                          "std": float(np.std([r[k] for r in qat_results]))}
                     for k in ["mean_auc", "f1", "sensitivity"]},
        "scratch": {k: {"mean": float(np.mean([r[k] for r in scratch_results])),
                         "std": float(np.std([r[k] for r in scratch_results]))}
                    for k in ["mean_auc", "f1", "sensitivity"]},
        "kd": {k: {"mean": float(np.mean([r[k] for r in kd_results])),
                    "std": float(np.std([r[k] for r in kd_results]))}
               for k in ["mean_auc", "f1", "sensitivity"]},
        "kd_qat": {k: {"mean": float(np.mean([r[k] for r in kd_qat_results])),
                        "std": float(np.std([r[k] for r in kd_qat_results]))}
                   for k in ["mean_auc", "f1", "sensitivity"]},
    }, f, indent=2)

print(f"\nAll results saved to {OUT}/results/")
print(f"All models saved to {OUT}/tflite/")
print("Download the results/ folder for paper data.")
