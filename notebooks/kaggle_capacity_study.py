"""
MedCompress Capacity Study — Kaggle Notebook
=============================================
GPU: T4 x2
Dataset: ISIC 2020 (same as main experiment)

Tests 3 student architectures at different capacities,
each trained from scratch AND with KD, to disentangle
capacity-ceiling effects from genuine knowledge transfer.

SETUP:
1. Same as ISIC notebook: add "siim-isic-melanoma-classification"
2. GPU T4 x2
3. !pip install -q tensorflow-model-optimization
"""

# %% Cell 1: Setup
!pip install -q tensorflow-model-optimization

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

print(f"TF: {tf.__version__}")
if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')

OUT = "/kaggle/working/capacity_study"
os.makedirs(OUT, exist_ok=True)

IMG_SIZE = 224
BATCH = 32
SEEDS = [42, 123, 456]  # 3 seeds for speed (capacity study, not primary result)

# %% Cell 2: Load ISIC (same as main notebook)
ISIC_DIR = "/kaggle/input/siim-isic-melanoma-classification"
csv_path = os.path.join(ISIC_DIR, "train.csv")
df = pd.read_csv(csv_path)

jpeg_dir = os.path.join(ISIC_DIR, "jpeg", "train")
if not os.path.exists(jpeg_dir):
    jpeg_dir = os.path.join(ISIC_DIR, "train")

df["image_path"] = df["image_name"].apply(lambda x: os.path.join(jpeg_dir, f"{x}.jpg"))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

# Subsample for speed
if len(df) > 20000:
    df = df.sample(20000, random_state=42).reset_index(drop=True)

tr_df, test_df = train_test_split(df, test_size=0.15, stratify=df["target"], random_state=42)
tr_df, val_df = train_test_split(tr_df, test_size=0.15/0.85, stratify=tr_df["target"], random_state=42)

counts = df["target"].value_counts()
class_weights = {0: len(df)/(2*counts[0]), 1: len(df)/(2*counts[1])}

def make_ds(dataframe, aug=False, shuf=False):
    paths = dataframe["image_path"].values
    labels = dataframe["target"].values.astype(np.float32)
    def load(p, l):
        img = tf.image.decode_jpeg(tf.io.read_file(p), channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img, l
    def augment(img, l):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        return img, l
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuf: ds = ds.shuffle(len(paths))
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if aug: ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(tr_df, aug=True, shuf=True)
val_ds = make_ds(val_df)
test_ds = make_ds(test_df)

# %% Cell 3: Student Architectures
def build_student(arch_name):
    """Build student architecture by name."""
    if arch_name == "mobilenetv3_small":
        backbone = keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        for layer in backbone.layers[:-10]: layer.trainable = False
        hidden = 128
    elif arch_name == "mobilenetv3_large":
        backbone = keras.applications.MobileNetV3Large(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        for layer in backbone.layers[:-15]: layer.trainable = False
        hidden = 256
    elif arch_name == "efficientnetb1":
        backbone = keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
        for layer in backbone.layers[:-25]: layer.trainable = False
        hidden = 256
    else:
        raise ValueError(f"Unknown arch: {arch_name}")

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = backbone(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hidden, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name=arch_name)

def set_seed(s):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)

def eval_auc(model, ds):
    preds, labels = [], []
    for img, lbl in ds:
        preds.append(model(img, training=False).numpy())
        labels.append(lbl.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(preds))

# %% Cell 4: Train Teacher
print("=" * 50)
print("Training Teacher (EfficientNetB3)")
print("=" * 50)
set_seed(42)
teacher = keras.applications.EfficientNetB3(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in teacher.layers[:-30]: layer.trainable = False
inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = teacher(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.15)(x)
out = layers.Dense(1, activation="sigmoid")(x)
teacher_model = keras.Model(inp, out)
teacher_model.compile(optimizer=keras.optimizers.Adam(1e-4),
                      loss="binary_crossentropy",
                      metrics=[keras.metrics.AUC(name="auc")])
teacher_model.fit(train_ds, validation_data=val_ds, epochs=15,
                  class_weight=class_weights,
                  callbacks=[keras.callbacks.EarlyStopping(
                      monitor="val_auc", patience=5, mode="max",
                      restore_best_weights=True)],
                  verbose=1)
teacher_auc = eval_auc(teacher_model, test_ds)
print(f"Teacher AUC: {teacher_auc:.4f}")

# %% Cell 5: Capacity Study
STUDENTS = ["mobilenetv3_small", "mobilenetv3_large", "efficientnetb1"]
T, ALPHA = 4.0, 0.7

results = []

for arch in STUDENTS:
    print(f"\n{'='*50}")
    print(f"Student: {arch}")
    print(f"{'='*50}")

    scratch_aucs = []
    kd_aucs = []

    for seed in SEEDS:
        set_seed(seed)

        # --- From scratch ---
        student_s = build_student(arch)
        student_s.compile(optimizer=keras.optimizers.Adam(1e-4),
                          loss="binary_crossentropy",
                          metrics=[keras.metrics.AUC(name="auc")])
        student_s.fit(train_ds, validation_data=val_ds, epochs=20,
                      class_weight=class_weights,
                      callbacks=[keras.callbacks.EarlyStopping(
                          monitor="val_auc", patience=7, mode="max",
                          restore_best_weights=True)],
                      verbose=0)
        s_auc = eval_auc(student_s, test_ds)
        scratch_aucs.append(s_auc)
        print(f"  Scratch s{seed}: {s_auc:.4f}")

        # --- With KD ---
        set_seed(seed)
        student_kd = build_student(arch)
        opt = keras.optimizers.Adam(1e-4)

        for epoch in range(20):
            for imgs, labels in train_ds:
                with tf.GradientTape() as tape:
                    t_logits = teacher_model(imgs, training=False)
                    s_logits = student_kd(imgs, training=True)
                    eps = 1e-7
                    t_soft = tf.nn.sigmoid(tf.math.log(t_logits/(1-t_logits+eps)+eps)/T)
                    s_soft = tf.nn.sigmoid(tf.math.log(s_logits/(1-s_logits+eps)+eps)/T)
                    kl = t_soft*tf.math.log((t_soft+eps)/(s_soft+eps))
                    kl += (1-t_soft)*tf.math.log((1-t_soft+eps)/(1-s_soft+eps))
                    d_loss = tf.reduce_mean(kl) * T**2
                    h_loss = tf.reduce_mean(keras.losses.binary_crossentropy(
                        tf.expand_dims(labels,-1), s_logits))
                    loss = ALPHA*d_loss + (1-ALPHA)*h_loss
                grads = tape.gradient(loss, student_kd.trainable_variables)
                opt.apply_gradients(zip(grads, student_kd.trainable_variables))

        kd_auc = eval_auc(student_kd, test_ds)
        kd_aucs.append(kd_auc)
        print(f"  KD     s{seed}: {kd_auc:.4f}")

    scratch_mean = np.mean(scratch_aucs)
    kd_mean = np.mean(kd_aucs)
    gain = kd_mean - scratch_mean

    results.append({
        "architecture": arch,
        "params_m": student_s.count_params() / 1e6,
        "scratch_auc_mean": round(scratch_mean, 4),
        "scratch_auc_std": round(np.std(scratch_aucs), 4),
        "kd_auc_mean": round(kd_mean, 4),
        "kd_auc_std": round(np.std(kd_aucs), 4),
        "distillation_gain": round(gain * 100, 2),
    })
    print(f"\n  Scratch: {scratch_mean:.4f}, KD: {kd_mean:.4f}, Gain: {gain*100:+.2f}%")

# %% Cell 6: Summary
print("\n" + "=" * 50)
print("CAPACITY STUDY RESULTS")
print("=" * 50)
res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(f"{OUT}/capacity_study_results.csv", index=False)

print(f"\nTeacher (B3): {teacher_auc:.4f}")
print("\nInterpretation:")
gains = [r["distillation_gain"] for r in results]
if gains[0] > gains[-1] * 1.5:
    print("  Gain DECREASES with capacity -> partially capacity compensation")
else:
    print("  Gain STABLE across sizes -> genuine knowledge transfer")

print(f"\nResults saved to: {OUT}/capacity_study_results.csv")
