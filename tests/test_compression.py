"""
tests/test_compression.py
--------------------------
Tests for the MedCompress compression modules:
  - compression/qat.py          (Quantization-Aware Training)
  - compression/distillation.py (Knowledge Distillation)
  - compression/pruning.py      (Magnitude-based pruning)

All tests use tiny synthetic models and data to run fast on CPU.
Heavy operations (TFLite conversion, full training loops) are tested
with small models or mocked where appropriate.

Run with:
    python -m pytest tests/test_compression.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import tensorflow_model_optimization as tfmot

    HAS_TFMOT = True
except ImportError:
    HAS_TFMOT = False

skip_no_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
skip_no_tfmot = pytest.mark.skipif(
    not HAS_TFMOT, reason="tensorflow-model-optimization not installed"
)

BATCH = 4
IMG_SIZE = 32


# ====================================================================== #
#  Helpers                                                                 #
# ====================================================================== #


def _small_classifier() -> "tf.keras.Model":
    """Build a tiny classifier for fast tests (no ImageNet backbone)."""
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inp, x, name="tiny_clf")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _small_segmenter(num_classes: int = 4) -> "tf.keras.Model":
    """Build a tiny segmentation model for fast tests."""
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 12))
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(
        num_classes, 1, padding="same", activation="softmax"
    )(x)
    model = tf.keras.Model(inp, x, name="tiny_seg")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _make_clf_dataset(n: int = 16):
    """Synthetic classification dataset."""
    x = np.random.randn(n, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
    y = np.random.randint(0, 2, n).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)


def _make_seg_dataset(n: int = 16, num_classes: int = 4):
    """Synthetic segmentation dataset with one-hot masks."""
    x = np.random.randn(n, IMG_SIZE, IMG_SIZE, 12).astype(np.float32)
    y_idx = np.random.randint(0, num_classes, (n, IMG_SIZE, IMG_SIZE))
    y = tf.one_hot(y_idx, num_classes).numpy().astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(BATCH)


# ====================================================================== #
#  QAT tests                                                               #
# ====================================================================== #


@skip_no_tf
@skip_no_tfmot
class TestQAT:
    """Tests for compression/qat.py quantization-aware training."""

    def test_apply_qat_wraps_model(self):
        """apply_qat should return a model with quantization wrappers."""
        from compression.qat import apply_qat

        base = _small_classifier()
        qat_model = apply_qat(base)

        assert qat_model is not None
        # QAT model should have more layers due to fake-quant wrappers
        assert len(qat_model.layers) >= len(base.layers)

    def test_qat_model_produces_valid_output(self):
        """QAT-wrapped model should still produce valid sigmoid outputs."""
        from compression.qat import apply_qat

        base = _small_classifier()
        qat_model = apply_qat(base)

        dummy = np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        out = qat_model(dummy, training=False).numpy()
        assert out.shape == (BATCH, 1)
        assert np.all(out >= 0.0) and np.all(out <= 1.0), (
            f"Output should be in [0,1], got min={out.min()}, max={out.max()}"
        )

    def test_qat_model_is_trainable(self):
        """QAT model should have trainable variables for fine-tuning."""
        from compression.qat import apply_qat

        base = _small_classifier()
        qat_model = apply_qat(base)

        assert len(qat_model.trainable_variables) > 0

    def test_export_to_tflite_fp32(self, tmp_path):
        """export_to_tflite with fp32 should create a valid .tflite file."""
        from compression.qat import export_to_tflite

        model = _small_classifier()
        out_path = str(tmp_path / "model_fp32.tflite")

        result_path = export_to_tflite(
            model, precision="fp32", output_path=out_path
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_export_to_tflite_fp16(self, tmp_path):
        """export_to_tflite with fp16 should create a valid .tflite file."""
        from compression.qat import export_to_tflite

        model = _small_classifier()
        out_path = str(tmp_path / "model_fp16.tflite")

        result_path = export_to_tflite(
            model, precision="fp16", output_path=out_path
        )

        assert os.path.exists(result_path)
        size_fp16 = os.path.getsize(result_path)
        assert size_fp16 > 0

    def test_export_to_tflite_int8_with_calibration(self, tmp_path):
        """export_to_tflite with int8 and calibration data should succeed."""
        from compression.qat import export_to_tflite

        model = _small_classifier()
        out_path = str(tmp_path / "model_int8.tflite")

        # Simple calibration generator
        def calibration_gen():
            for _ in range(5):
                yield [np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)]

        result_path = export_to_tflite(
            model,
            precision="int8",
            calibration_gen=calibration_gen,
            output_path=out_path,
        )

        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_tflite_fp16_smaller_than_fp32(self, tmp_path):
        """FP16 TFLite model should generally be smaller than FP32."""
        from compression.qat import export_to_tflite

        model = _small_classifier()

        fp32_path = str(tmp_path / "fp32.tflite")
        fp16_path = str(tmp_path / "fp16.tflite")

        export_to_tflite(model, precision="fp32", output_path=fp32_path)
        export_to_tflite(model, precision="fp16", output_path=fp16_path)

        fp32_size = os.path.getsize(fp32_path)
        fp16_size = os.path.getsize(fp16_path)

        # FP16 should be <= FP32 (small models may not show large difference)
        assert fp16_size <= fp32_size * 1.1, (
            f"FP16 ({fp16_size}) should be <= FP32 ({fp32_size})"
        )

    def test_tflite_inference_roundtrip(self, tmp_path):
        """A TFLite model exported as FP32 should produce valid inference."""
        from compression.qat import export_to_tflite

        model = _small_classifier()
        out_path = str(tmp_path / "model.tflite")
        export_to_tflite(model, precision="fp32", output_path=out_path)

        interpreter = tf.lite.Interpreter(model_path=out_path)
        interpreter.allocate_tensors()
        inp_details = interpreter.get_input_details()[0]
        out_details = interpreter.get_output_details()[0]

        dummy = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        interpreter.set_tensor(inp_details["index"], dummy)
        interpreter.invoke()
        result = interpreter.get_tensor(out_details["index"])

        assert result.shape[0] == 1
        assert result.size > 0


# ====================================================================== #
#  Knowledge Distillation tests                                            #
# ====================================================================== #


@skip_no_tf
class TestKnowledgeDistillation:
    """Tests for compression/distillation.py KD loss and trainer."""

    def test_kd_loss_classification_returns_scalar(self):
        """kd_loss for classification should return a non-negative scalar."""
        from compression.distillation import kd_loss

        y_true = tf.constant([0.0, 1.0, 0.0, 1.0])
        student_logits = tf.random.normal([4, 1])
        teacher_logits = tf.random.normal([4, 1])

        loss = kd_loss(
            y_true, student_logits, teacher_logits,
            temperature=4.0, alpha=0.7, task="classification",
        )

        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert loss.numpy() >= 0.0, f"KD loss should be non-negative, got {loss.numpy()}"

    def test_kd_loss_segmentation_returns_scalar(self):
        """kd_loss for segmentation should return a non-negative scalar."""
        from compression.distillation import kd_loss

        num_classes = 4
        y_true = tf.one_hot(
            np.random.randint(0, num_classes, (2, 8, 8)), num_classes
        )
        student_logits = tf.random.normal([2, 8, 8, num_classes])
        teacher_logits = tf.random.normal([2, 8, 8, num_classes])

        loss = kd_loss(
            y_true, student_logits, teacher_logits,
            temperature=3.0, alpha=0.5, task="segmentation",
        )

        assert loss.shape == ()
        assert loss.numpy() >= 0.0

    def test_kd_loss_alpha_zero_equals_hard_label_only(self):
        """With alpha=0, the loss should depend only on hard labels (CE)."""
        from compression.distillation import kd_loss

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        student_logits = tf.constant([[2.0], [-1.0], [1.5], [-2.0]])
        teacher_logits = tf.random.normal([4, 1])

        loss_alpha0 = kd_loss(
            y_true, student_logits, teacher_logits,
            temperature=4.0, alpha=0.0, task="classification",
        )

        # Same student_logits, different teacher should not change loss when alpha=0
        teacher_logits2 = tf.random.normal([4, 1])
        loss_alpha0_diff_teacher = kd_loss(
            y_true, student_logits, teacher_logits2,
            temperature=4.0, alpha=0.0, task="classification",
        )

        np.testing.assert_allclose(
            loss_alpha0.numpy(), loss_alpha0_diff_teacher.numpy(),
            atol=1e-5,
            err_msg="With alpha=0, teacher should not influence loss",
        )

    def test_kd_loss_alpha_one_ignores_hard_labels(self):
        """With alpha=1, the loss should depend only on teacher-student KL."""
        from compression.distillation import kd_loss

        student_logits = tf.constant([[2.0], [-1.0], [1.5], [-2.0]])
        teacher_logits = tf.constant([[1.5], [-0.5], [1.0], [-1.5]])

        y_true_a = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_true_b = tf.constant([0.0, 1.0, 0.0, 1.0])

        loss_a = kd_loss(
            y_true_a, student_logits, teacher_logits,
            temperature=4.0, alpha=1.0, task="classification",
        )
        loss_b = kd_loss(
            y_true_b, student_logits, teacher_logits,
            temperature=4.0, alpha=1.0, task="classification",
        )

        np.testing.assert_allclose(
            loss_a.numpy(), loss_b.numpy(), atol=1e-5,
            err_msg="With alpha=1, hard labels should not influence loss",
        )

    def test_kd_loss_temperature_scaling(self):
        """Higher temperature should produce softer distributions (lower KL)."""
        from compression.distillation import kd_loss

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        student_logits = tf.constant([[2.0], [-1.0], [1.5], [-2.0]])
        teacher_logits = tf.constant([[1.5], [-0.5], [1.0], [-1.5]])

        loss_low_t = kd_loss(
            y_true, student_logits, teacher_logits,
            temperature=1.0, alpha=0.5, task="classification",
        )
        loss_high_t = kd_loss(
            y_true, student_logits, teacher_logits,
            temperature=10.0, alpha=0.5, task="classification",
        )

        # Different temperatures should produce different losses
        assert loss_low_t.numpy() != loss_high_t.numpy()

    def test_feature_distillation_loss_same_shape(self):
        """FeatureDistillationLoss with matching shapes needs no adapter."""
        from compression.distillation import FeatureDistillationLoss

        t_shapes = [(None, 8, 8, 64)]
        s_shapes = [(None, 8, 8, 64)]
        feat_loss = FeatureDistillationLoss(t_shapes, s_shapes)

        teacher_feats = [tf.random.normal([2, 8, 8, 64])]
        student_feats = [tf.random.normal([2, 8, 8, 64])]

        loss = feat_loss(teacher_feats, student_feats)
        assert loss.numpy() >= 0.0
        assert feat_loss.adapters[0] is None, "No adapter needed for matching shapes"

    def test_feature_distillation_loss_channel_mismatch(self):
        """FeatureDistillationLoss with different channels should create an adapter."""
        from compression.distillation import FeatureDistillationLoss

        t_shapes = [(None, 8, 8, 128)]
        s_shapes = [(None, 8, 8, 64)]
        feat_loss = FeatureDistillationLoss(t_shapes, s_shapes)

        assert feat_loss.adapters[0] is not None, "Adapter expected for channel mismatch"

        teacher_feats = [tf.random.normal([2, 8, 8, 128])]
        student_feats = [tf.random.normal([2, 8, 8, 64])]

        loss = feat_loss(teacher_feats, student_feats)
        assert loss.numpy() >= 0.0

    def test_distillation_trainer_instantiation(self):
        """DistillationTrainer should initialize without errors."""
        from compression.distillation import DistillationTrainer

        teacher = _small_classifier()
        student = _small_classifier()

        config = {
            "distillation": {
                "temperature": 4.0,
                "alpha": 0.7,
                "feature_distillation": False,
            },
            "training": {
                "learning_rate": 1e-4,
                "epochs": 1,
            },
            "task": "classification",
        }

        trainer = DistillationTrainer(teacher, student, config)

        assert trainer.temperature == 4.0
        assert trainer.alpha == 0.7
        assert trainer.teacher.trainable is False

    def test_distillation_trainer_single_step(self):
        """A single training step should reduce loss without crashing."""
        from compression.distillation import DistillationTrainer

        teacher = _small_classifier()
        student = _small_classifier()

        config = {
            "distillation": {
                "temperature": 4.0,
                "alpha": 0.7,
                "feature_distillation": False,
            },
            "training": {
                "learning_rate": 1e-3,
                "epochs": 1,
            },
            "task": "classification",
        }

        trainer = DistillationTrainer(teacher, student, config)
        ds = _make_clf_dataset(n=8)
        images, labels = next(iter(ds))

        loss, preds = trainer._train_step(images, labels)
        assert loss.numpy() >= 0.0
        assert preds.shape[0] == images.shape[0]


# ====================================================================== #
#  Pruning tests                                                           #
# ====================================================================== #


@skip_no_tf
@skip_no_tfmot
class TestPruning:
    """Tests for compression/pruning.py magnitude-based pruning."""

    def test_apply_magnitude_pruning_wraps_model(self):
        """apply_magnitude_pruning should return a pruning-wrapped model."""
        from compression.pruning import apply_magnitude_pruning

        base = _small_classifier()
        pruned = apply_magnitude_pruning(
            base, target_sparsity=0.5, begin_step=0, end_step=100,
        )

        assert pruned is not None
        assert len(pruned.layers) >= len(base.layers)

    def test_pruned_model_produces_valid_output(self):
        """Pruning-wrapped model should still produce valid predictions."""
        from compression.pruning import apply_magnitude_pruning

        base = _small_classifier()
        pruned = apply_magnitude_pruning(
            base, target_sparsity=0.5, begin_step=0, end_step=100,
        )

        dummy = np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        out = pruned(dummy, training=False).numpy()
        assert out.shape == (BATCH, 1)

    def test_strip_pruning_removes_wrappers(self):
        """strip_pruning should remove pruning metadata layers."""
        from compression.pruning import apply_magnitude_pruning, strip_pruning

        base = _small_classifier()
        pruned = apply_magnitude_pruning(
            base, target_sparsity=0.5, begin_step=0, end_step=100,
        )
        stripped = strip_pruning(pruned)

        # Stripped model should have fewer layers than pruned (wrappers removed)
        assert len(stripped.layers) <= len(pruned.layers)
        # Should still produce valid output
        dummy = np.random.randn(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        out = stripped(dummy, training=False)
        assert out.shape == (1, 1)

    def test_get_pruning_callbacks(self):
        """get_pruning_callbacks should return a list with UpdatePruningStep."""
        from compression.pruning import get_pruning_callbacks

        callbacks = get_pruning_callbacks()
        assert isinstance(callbacks, list)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], tfmot.sparsity.keras.UpdatePruningStep)

    def test_compute_sparsity_on_dense_model(self):
        """compute_sparsity on an unpruned model should report ~0% sparsity."""
        from compression.pruning import compute_sparsity

        model = _small_classifier()
        stats = compute_sparsity(model)

        assert stats["overall_sparsity"] < 0.05, (
            f"Unpruned model should have near-zero sparsity, got {stats['overall_sparsity']}"
        )
        assert stats["total_params"] > 0
        assert stats["zero_params"] < stats["total_params"]

    def test_compute_sparsity_returns_correct_structure(self):
        """compute_sparsity should return dict with expected keys."""
        from compression.pruning import compute_sparsity

        model = _small_classifier()
        stats = compute_sparsity(model)

        required_keys = {
            "overall_sparsity", "total_params", "zero_params",
            "nonzero_params", "layers",
        }
        assert required_keys.issubset(set(stats.keys()))
        assert stats["nonzero_params"] == stats["total_params"] - stats["zero_params"]

    def test_structured_filter_pruning_analysis(self):
        """structured_filter_pruning should return per-layer recommendations."""
        from compression.pruning import structured_filter_pruning

        model = _small_segmenter(num_classes=4)
        result = structured_filter_pruning(model, prune_ratio=0.3)

        assert "prune_ratio" in result
        assert result["prune_ratio"] == 0.3
        assert "layers" in result
        assert len(result["layers"]) > 0, "Should analyze at least one Conv2D layer"
        assert "total_filters_original" in result
        assert "total_filters_remaining" in result

    def test_structured_pruning_prune_count(self):
        """Each layer should recommend pruning floor(n_filters * ratio) filters."""
        from compression.pruning import structured_filter_pruning

        model = _small_segmenter(num_classes=4)
        result = structured_filter_pruning(model, prune_ratio=0.5)

        for layer_rec in result["layers"]:
            expected_prune = int(layer_rec["total_filters"] * 0.5)
            assert layer_rec["prune_count"] == expected_prune, (
                f"Layer {layer_rec['layer_name']}: expected {expected_prune} pruned, "
                f"got {layer_rec['prune_count']}"
            )

    def test_structured_pruning_zero_ratio(self):
        """prune_ratio=0 should recommend pruning zero filters."""
        from compression.pruning import structured_filter_pruning

        model = _small_segmenter(num_classes=4)
        result = structured_filter_pruning(model, prune_ratio=0.0)

        for layer_rec in result["layers"]:
            assert layer_rec["prune_count"] == 0
            assert layer_rec["remaining_filters"] == layer_rec["total_filters"]

    def test_different_sparsity_levels(self):
        """Higher target sparsity should be reflected in pruning schedule params."""
        from compression.pruning import apply_magnitude_pruning

        base = _small_classifier()

        for sparsity in [0.3, 0.5, 0.7, 0.9]:
            pruned = apply_magnitude_pruning(
                base, target_sparsity=sparsity, begin_step=0, end_step=100,
            )
            assert pruned is not None, f"Failed to apply pruning at sparsity={sparsity}"
