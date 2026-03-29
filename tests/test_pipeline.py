"""
tests/test_pipeline.py
-----------------------
Smoke tests — verify the full pipeline runs end-to-end with tiny synthetic data.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.baseline import build_efficientnetb0, build_unet_lite, DiceCoefficient, dice_loss


IMG_SIZE = 64   # tiny for speed
BATCH = 4
N = 16


def make_isic_ds():
    x = np.random.randn(N, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y = np.random.randint(0, 2, N).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)


def make_brats_ds(n_classes=4, n_ch=12, patch=32):
    x = np.random.randn(N, patch, patch, n_ch).astype(np.float32)
    y = tf.one_hot(np.random.randint(0, n_classes, (N, patch, patch)), n_classes).numpy()
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)


# ── Model construction ────────────────────────────────────────────────────── #

def test_efficientnetb0_builds():
    model = build_efficientnetb0(num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    assert model is not None
    out = model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), training=False)
    assert out.shape == (1, 1), f"Expected (1,1), got {out.shape}"


def test_unet_lite_builds():
    model = build_unet_lite(num_classes=4, n_channels=12, input_size=32)
    assert model is not None
    out = model(np.zeros((1, 32, 32, 12), dtype=np.float32), training=False)
    assert out.shape == (1, 32, 32, 4), f"Expected (1,32,32,4), got {out.shape}"


# ── Loss functions ────────────────────────────────────────────────────────── #

def test_dice_loss():
    y_true = tf.one_hot([[0, 1], [2, 3]], 4)
    y_pred = tf.nn.softmax(tf.random.normal([2, 2, 4]))
    loss = dice_loss(y_true, y_pred)
    assert 0.0 <= float(loss) <= 2.0, f"Dice loss out of range: {loss}"


def test_dice_metric():
    metric = DiceCoefficient(4, name="dice")
    y_true = tf.one_hot(np.zeros((2, 32, 32), dtype=np.int32), 4)
    y_pred = tf.nn.softmax(tf.random.normal([2, 32, 32, 4]))
    metric.update_state(y_true, y_pred)
    result = metric.result().numpy()
    assert 0.0 <= result <= 1.0


# ── Training step ──────────────────────────────────────────────────────────── #

def test_baseline_training_step():
    model = build_efficientnetb0(num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    ds = make_isic_ds()
    imgs, lbls = next(iter(ds))
    with tf.GradientTape() as tape:
        preds = model(imgs, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.expand_dims(lbls, -1), preds
        ))
    grads = tape.gradient(loss, model.trainable_variables)
    assert any(g is not None for g in grads), "No gradients computed"


# ── QAT ──────────────────────────────────────────────────────────────────── #

def test_qat_wrapping():
    model = build_efficientnetb0(num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    qat_model = tfmot.quantization.keras.quantize_model(model)
    assert qat_model is not None
    # QAT model should still produce valid output
    out = qat_model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), training=False)
    assert out.shape[0] == 1


# ── TFLite export ─────────────────────────────────────────────────────────── #

def test_tflite_export_fp32(tmp_path):
    model = build_efficientnetb0(num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = str(tmp_path / "model_fp32.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)
    assert os.path.getsize(path) > 0


def test_tflite_inference(tmp_path):
    model = build_efficientnetb0(num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    path = str(tmp_path / "model.tflite")
    with open(path, "wb") as f:
        f.write(tflite_model)

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    img = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    interp.set_tensor(inp["index"], img)
    interp.invoke()
    result = interp.get_tensor(out_det["index"])
    assert result.shape[0] == 1


# ── KD loss ───────────────────────────────────────────────────────────────── #

def test_kd_loss_computation():
    from compression.distillation import kd_loss
    labels = tf.constant([0.0, 1.0, 0.0, 1.0])
    student_logits = tf.random.normal([4, 1])
    teacher_logits = tf.random.normal([4, 1])
    loss = kd_loss(labels, student_logits, teacher_logits,
                   temperature=4.0, alpha=0.7, task="classification")
    assert loss.numpy() >= 0.0, "KD loss should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
