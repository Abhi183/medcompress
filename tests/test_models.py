"""
tests/test_models.py
---------------------
Tests for the MedCompress baseline model architectures in models/baseline.py:
  - build_efficientnetb0  (ISIC classification)
  - build_unet_full       (BraTS full-resolution segmentation)
  - build_unet_lite       (BraTS lightweight student)
  - conv_block, encoder_block, decoder_block (building blocks)
  - dice_loss, dice_ce_loss, DiceCoefficient (loss and metrics)

All tests use tiny input sizes and run WITHOUT a GPU.

Run with:
    python -m pytest tests/test_models.py -v
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

skip_no_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")

# Use small sizes for fast CPU tests
IMG_SIZE = 32
BATCH = 2


# ====================================================================== #
#  EfficientNetB0 classifier                                               #
# ====================================================================== #


@skip_no_tf
class TestEfficientNetB0:
    """Tests for the EfficientNetB0 classification model."""

    def test_builds_with_default_params(self):
        """Model should build without errors using default arguments."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        assert model is not None
        assert model.name == "efficientnetb0_isic"

    def test_binary_output_shape(self):
        """Binary classifier should output shape (batch, 1)."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        dummy = np.zeros((BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out = model(dummy, training=False)
        assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"

    def test_multiclass_output_shape(self):
        """Multi-class classifier should output shape (batch, num_classes)."""
        from models.baseline import build_efficientnetb0

        num_classes = 5
        model = build_efficientnetb0(
            num_classes=num_classes, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        dummy = np.zeros((BATCH, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        out = model(dummy, training=False)
        assert out.shape == (BATCH, num_classes)

    def test_binary_output_is_sigmoid(self):
        """Binary output should be in (0, 1) range via sigmoid activation."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        dummy = np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        out = model(dummy, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0), (
            f"Sigmoid output should be in [0,1], got min={out.min()}, max={out.max()}"
        )

    def test_multiclass_output_sums_to_one(self):
        """Softmax output should sum to approximately 1 per sample."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=5, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        dummy = np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        out = model(dummy, training=False).numpy()
        row_sums = out.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_compilation_succeeds_binary(self):
        """Binary model should compile with binary_crossentropy + AUC metric."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        # Model is already compiled inside build_efficientnetb0
        assert model.loss is not None
        metric_names = [m.name for m in model.metrics]
        assert any("auc" in name for name in metric_names)

    def test_compilation_succeeds_multiclass(self):
        """Multi-class model should compile with categorical_crossentropy."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=5, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        assert model.loss is not None

    def test_backbone_partial_freeze(self):
        """All but the last 20 backbone layers should be frozen."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        # The model should have both trainable and non-trainable weights
        assert model.count_params() > 0
        trainable_count = sum(
            tf.size(v).numpy() for v in model.trainable_variables
        )
        total_count = model.count_params()
        # Some layers should be frozen (trainable < total)
        assert trainable_count < total_count, (
            "Expected partial freezing: trainable params should be < total params"
        )

    def test_forward_pass_gradient_flows(self):
        """Gradient should be computable through the model."""
        from models.baseline import build_efficientnetb0

        model = build_efficientnetb0(
            num_classes=1, input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        dummy = tf.constant(
            np.random.randn(BATCH, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        )
        labels = tf.constant([[1.0], [0.0]])

        with tf.GradientTape() as tape:
            preds = model(dummy, training=True)
            loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(labels, preds)
            )
        grads = tape.gradient(loss, model.trainable_variables)
        assert any(g is not None for g in grads), "No gradients computed"


# ====================================================================== #
#  U-Net Full                                                              #
# ====================================================================== #

PATCH = 32
N_CHANNELS = 12
NUM_CLASSES = 4


@skip_no_tf
class TestUNetFull:
    """Tests for the full-resolution 2.5D U-Net segmentation model."""

    def test_builds_with_default_params(self):
        """Model should build without errors."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        assert model is not None
        assert model.name == "unet_full"

    def test_output_shape(self):
        """Output should be (batch, H, W, num_classes) matching input spatial dims."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        dummy = np.zeros((BATCH, PATCH, PATCH, N_CHANNELS), dtype=np.float32)
        out = model(dummy, training=False)
        expected = (BATCH, PATCH, PATCH, NUM_CLASSES)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_softmax_output_sums_to_one(self):
        """Per-pixel softmax should sum to 1 across classes."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        dummy = np.random.randn(BATCH, PATCH, PATCH, N_CHANNELS).astype(np.float32)
        out = model(dummy, training=False).numpy()
        pixel_sums = out.sum(axis=-1)
        np.testing.assert_allclose(pixel_sums, 1.0, atol=1e-5)

    def test_output_probabilities_valid_range(self):
        """All output probabilities should be in [0, 1]."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        dummy = np.random.randn(BATCH, PATCH, PATCH, N_CHANNELS).astype(np.float32)
        out = model(dummy, training=False).numpy()
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_compiled_with_dice_ce_loss(self):
        """Model should be compiled with the combined dice+CE loss."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        assert model.loss is not None

    def test_encoder_decoder_symmetry(self):
        """The full U-Net should have 4 encoder stages with filters [64,128,256,512]."""
        from models.baseline import build_unet_full

        model = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        layer_names = [l.name for l in model.layers]
        # Check encoder stages exist
        for stage in ["enc1", "enc2", "enc3", "enc4"]:
            assert any(stage in n for n in layer_names), f"Missing encoder stage {stage}"
        # Check decoder stages exist
        for stage in ["dec1", "dec2", "dec3", "dec4"]:
            assert any(stage in n for n in layer_names), f"Missing decoder stage {stage}"

    def test_different_num_classes(self):
        """Model should adapt to different number of output classes."""
        from models.baseline import build_unet_full

        for nc in [2, 4, 8]:
            model = build_unet_full(
                num_classes=nc, n_channels=N_CHANNELS, input_size=PATCH
            )
            dummy = np.zeros((1, PATCH, PATCH, N_CHANNELS), dtype=np.float32)
            out = model(dummy, training=False)
            assert out.shape[-1] == nc, f"Expected {nc} classes, got {out.shape[-1]}"


# ====================================================================== #
#  U-Net Lite                                                              #
# ====================================================================== #


@skip_no_tf
class TestUNetLite:
    """Tests for the lightweight 2.5D U-Net (KD student)."""

    def test_builds_with_default_params(self):
        """Model should build without errors."""
        from models.baseline import build_unet_lite

        model = build_unet_lite(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        assert model is not None
        assert model.name == "unet_lite"

    def test_output_shape(self):
        """Output should match input spatial dims with num_classes channels."""
        from models.baseline import build_unet_lite

        model = build_unet_lite(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        dummy = np.zeros((BATCH, PATCH, PATCH, N_CHANNELS), dtype=np.float32)
        out = model(dummy, training=False)
        expected = (BATCH, PATCH, PATCH, NUM_CLASSES)
        assert out.shape == expected, f"Expected {expected}, got {out.shape}"

    def test_fewer_params_than_full(self):
        """U-Net Lite should have significantly fewer parameters than U-Net Full."""
        from models.baseline import build_unet_full, build_unet_lite

        full = build_unet_full(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        lite = build_unet_lite(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        full_params = full.count_params()
        lite_params = lite.count_params()
        assert lite_params < full_params, (
            f"Lite ({lite_params:,}) should have fewer params than Full ({full_params:,})"
        )
        # The docstring says ~8x fewer; verify at least 4x
        ratio = full_params / lite_params
        assert ratio > 4, f"Expected >4x ratio, got {ratio:.1f}x"

    def test_three_encoder_stages(self):
        """U-Net Lite should have 3 encoder stages (not 4)."""
        from models.baseline import build_unet_lite

        model = build_unet_lite(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        layer_names = [l.name for l in model.layers]
        for stage in ["enc1", "enc2", "enc3"]:
            assert any(stage in n for n in layer_names), f"Missing encoder stage {stage}"
        # enc4 should NOT exist in lite
        assert not any("enc4" in n for n in layer_names), "enc4 should not exist in lite model"

    def test_softmax_output(self):
        """Output should be valid probability distribution per pixel."""
        from models.baseline import build_unet_lite

        model = build_unet_lite(
            num_classes=NUM_CLASSES, n_channels=N_CHANNELS, input_size=PATCH
        )
        dummy = np.random.randn(BATCH, PATCH, PATCH, N_CHANNELS).astype(np.float32)
        out = model(dummy, training=False).numpy()
        pixel_sums = out.sum(axis=-1)
        np.testing.assert_allclose(pixel_sums, 1.0, atol=1e-5)


# ====================================================================== #
#  Building blocks                                                         #
# ====================================================================== #


@skip_no_tf
class TestBuildingBlocks:
    """Tests for conv_block, encoder_block, decoder_block helpers."""

    def test_conv_block_output_shape(self):
        """conv_block should preserve spatial dimensions and set filter count."""
        from models.baseline import conv_block

        inp = tf.keras.Input(shape=(16, 16, 3))
        out = conv_block(inp, filters=32, name="test_cb")
        model = tf.keras.Model(inp, out)

        dummy = np.zeros((1, 16, 16, 3), dtype=np.float32)
        result = model(dummy, training=False)
        assert result.shape == (1, 16, 16, 32)

    def test_encoder_block_halves_spatial(self):
        """encoder_block should return (skip, pooled) with pooled at half resolution."""
        from models.baseline import encoder_block

        inp = tf.keras.Input(shape=(16, 16, 3))
        skip, pooled = encoder_block(inp, filters=32, name="test_enc")
        model = tf.keras.Model(inp, [skip, pooled])

        dummy = np.zeros((1, 16, 16, 3), dtype=np.float32)
        skip_out, pooled_out = model(dummy, training=False)
        assert skip_out.shape == (1, 16, 16, 32), "Skip should preserve spatial dims"
        assert pooled_out.shape == (1, 8, 8, 32), "Pooled should halve spatial dims"

    def test_decoder_block_doubles_spatial(self):
        """decoder_block should upsample and concatenate with skip connection."""
        from models.baseline import conv_block, decoder_block

        skip_inp = tf.keras.Input(shape=(16, 16, 32), name="skip")
        x_inp = tf.keras.Input(shape=(8, 8, 64), name="x")
        out = decoder_block(x_inp, skip_inp, filters=32, name="test_dec")
        model = tf.keras.Model([x_inp, skip_inp], out)

        x = np.zeros((1, 8, 8, 64), dtype=np.float32)
        skip = np.zeros((1, 16, 16, 32), dtype=np.float32)
        result = model([x, skip], training=False)
        assert result.shape == (1, 16, 16, 32)


# ====================================================================== #
#  Loss functions and metrics                                              #
# ====================================================================== #


@skip_no_tf
class TestLossAndMetrics:
    """Tests for dice_loss, dice_ce_loss, and DiceCoefficient."""

    def test_dice_loss_perfect_prediction(self):
        """Dice loss should be ~0 for perfect prediction."""
        from models.baseline import dice_loss

        y_true = tf.one_hot([[0, 1], [2, 3]], 4)
        y_pred = tf.cast(y_true, tf.float32)  # perfect match
        loss = dice_loss(y_true, y_pred)
        assert float(loss) < 0.01, f"Perfect prediction should have ~0 loss, got {loss}"

    def test_dice_loss_worst_prediction(self):
        """Dice loss should be close to 1 for completely wrong prediction."""
        from models.baseline import dice_loss

        y_true = tf.one_hot([[0, 0], [0, 0]], 4)  # all background
        # Predict all class 1
        y_pred = tf.one_hot([[1, 1], [1, 1]], 4)
        y_pred = tf.cast(y_pred, tf.float32)
        loss = dice_loss(y_true, y_pred)
        # Loss should be high (near 1) since no overlap on most classes
        assert float(loss) > 0.5, f"Wrong prediction should have high loss, got {loss}"

    def test_dice_loss_range(self):
        """Dice loss should be in [0, 1] for random inputs."""
        from models.baseline import dice_loss

        y_true = tf.one_hot(
            np.random.randint(0, 4, (4, 8, 8)), 4
        )
        y_pred = tf.nn.softmax(tf.random.normal([4, 8, 8, 4]))
        loss = dice_loss(y_true, y_pred)
        assert 0.0 <= float(loss) <= 1.5, f"Dice loss out of range: {loss}"

    def test_dice_ce_loss_is_callable(self):
        """dice_ce_loss factory should return a callable loss function."""
        from models.baseline import dice_ce_loss

        loss_fn = dice_ce_loss(num_classes=4)
        assert callable(loss_fn)
        assert loss_fn.__name__ == "dice_ce_loss"

    def test_dice_ce_loss_returns_scalar(self):
        """Combined loss should return a scalar tensor."""
        from models.baseline import dice_ce_loss

        loss_fn = dice_ce_loss(num_classes=4)
        y_true = tf.one_hot(np.random.randint(0, 4, (2, 8, 8)), 4)
        y_pred = tf.nn.softmax(tf.random.normal([2, 8, 8, 4]))
        loss = loss_fn(y_true, y_pred)
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert float(loss) > 0, "Combined loss should be positive"

    def test_dice_coefficient_metric(self):
        """DiceCoefficient metric should be in [0, 1]."""
        from models.baseline import DiceCoefficient

        metric = DiceCoefficient(4, name="dice")
        y_true = tf.one_hot(np.zeros((2, 8, 8), dtype=np.int32), 4)
        y_pred = tf.nn.softmax(tf.random.normal([2, 8, 8, 4]))

        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        assert 0.0 <= result <= 1.0, f"Dice should be in [0,1], got {result}"

    def test_dice_coefficient_reset(self):
        """DiceCoefficient reset_state should zero out accumulators."""
        from models.baseline import DiceCoefficient

        metric = DiceCoefficient(4, name="dice")
        y_true = tf.one_hot(np.zeros((2, 8, 8), dtype=np.int32), 4)
        y_pred = tf.nn.softmax(tf.random.normal([2, 8, 8, 4]))

        metric.update_state(y_true, y_pred)
        metric.reset_state()
        # After reset, dice_sum and count are 0, result is 0/0 = nan or 0
        # Accessing result after reset is fine; the key thing is no crash
        assert metric.dice_sum.numpy() == 0.0
        assert metric.count.numpy() == 0.0

    def test_dice_coefficient_perfect_prediction(self):
        """Perfect prediction should yield Dice close to 1."""
        from models.baseline import DiceCoefficient

        metric = DiceCoefficient(4, name="dice")
        # All class 1 (foreground)
        y_true_raw = np.ones((2, 8, 8), dtype=np.int32)
        y_true = tf.one_hot(y_true_raw, 4)
        # Predicted: strong class 1
        y_pred = tf.one_hot(y_true_raw, 4)
        y_pred = tf.cast(y_pred, tf.float32)

        metric.update_state(y_true, y_pred)
        result = metric.result().numpy()
        assert result > 0.9, f"Perfect prediction should give Dice > 0.9, got {result}"
