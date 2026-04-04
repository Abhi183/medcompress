"""
tests/test_evaluation.py
-------------------------
Tests for the MedCompress evaluation metrics:
  - scripts/evaluate.py               (AUC, Dice)
  - scripts/evaluate_extended.py       (classification/segmentation metrics, FLOPs)
  - scripts/evaluate_boundary.py       (Hausdorff Distance 95, Average Surface Distance)
  - scripts/evaluate_calibration.py    (ECE, MCE, Brier, CalibrationShift)

All tests use numpy arrays with known expected values. No TensorFlow
or real data files required for the pure-metric tests; TF-dependent
tests are guarded with skipif.

Run with:
    python -m pytest tests/test_evaluation.py -v
"""

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    from sklearn.metrics import roc_auc_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.ndimage import distance_transform_edt

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

skip_no_tf = pytest.mark.skipif(not HAS_TF, reason="TensorFlow not installed")
skip_no_sklearn = pytest.mark.skipif(not HAS_SKLEARN, reason="scikit-learn not installed")
skip_no_scipy = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


# ====================================================================== #
#  AUC computation (from scripts/evaluate.py)                              #
# ====================================================================== #


@skip_no_sklearn
class TestAUCComputation:
    """Tests for AUC via sklearn.metrics.roc_auc_score."""

    def test_perfect_auc(self):
        """Perfect predictions should yield AUC = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.95])
        auc = roc_auc_score(y_true, y_prob)
        assert auc == 1.0

    def test_random_auc_near_half(self):
        """Random predictions on balanced data should yield AUC near 0.5."""
        rng = np.random.RandomState(42)
        n = 10000
        y_true = rng.randint(0, 2, n)
        y_prob = rng.uniform(0, 1, n)
        auc = roc_auc_score(y_true, y_prob)
        assert abs(auc - 0.5) < 0.05, f"Random AUC should be ~0.5, got {auc}"

    def test_inverted_predictions_auc_zero(self):
        """Inverted predictions should yield AUC = 0.0."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.95, 0.1, 0.05])
        auc = roc_auc_score(y_true, y_prob)
        assert auc == 0.0

    def test_auc_range(self):
        """AUC should always be in [0, 1]."""
        rng = np.random.RandomState(7)
        for _ in range(10):
            y_true = rng.randint(0, 2, 100)
            y_prob = rng.uniform(0, 1, 100)
            auc = roc_auc_score(y_true, y_prob)
            assert 0.0 <= auc <= 1.0


# ====================================================================== #
#  Dice coefficient (inline from evaluate.py)                              #
# ====================================================================== #


class TestDiceComputation:
    """Tests for the Dice coefficient used in segmentation evaluation."""

    @staticmethod
    def _dice_single_class(pred: np.ndarray, true: np.ndarray) -> float:
        """Compute Dice for a single binary class pair (matching evaluate.py logic)."""
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum()
        if union == 0:
            return 0.0
        return float((2.0 * intersection + 1e-6) / (union + 1e-6))

    def test_perfect_overlap_dice_is_one(self):
        """Perfectly matching masks should yield Dice ~= 1.0."""
        mask = np.ones((32, 32), dtype=np.float32)
        dice = self._dice_single_class(mask, mask)
        assert abs(dice - 1.0) < 1e-4

    def test_no_overlap_dice_is_zero(self):
        """Non-overlapping masks should yield Dice ~= 0."""
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[:16, :] = 1.0
        true = np.zeros((32, 32), dtype=np.float32)
        true[16:, :] = 1.0
        dice = self._dice_single_class(pred, true)
        assert dice < 0.01

    def test_partial_overlap(self):
        """50% overlap should yield Dice between 0 and 1."""
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[:, :16] = 1.0  # left half
        true = np.zeros((32, 32), dtype=np.float32)
        true[:, 8:24] = 1.0  # middle
        dice = self._dice_single_class(pred, true)
        assert 0.0 < dice < 1.0

    def test_empty_masks_return_zero(self):
        """Both empty masks should return 0 (union = 0)."""
        pred = np.zeros((32, 32), dtype=np.float32)
        true = np.zeros((32, 32), dtype=np.float32)
        dice = self._dice_single_class(pred, true)
        assert dice == 0.0

    def test_dice_is_symmetric(self):
        """Dice(A, B) should equal Dice(B, A)."""
        rng = np.random.RandomState(42)
        a = (rng.rand(32, 32) > 0.5).astype(np.float32)
        b = (rng.rand(32, 32) > 0.5).astype(np.float32)
        dice_ab = self._dice_single_class(a, b)
        dice_ba = self._dice_single_class(b, a)
        assert abs(dice_ab - dice_ba) < 1e-6


# ====================================================================== #
#  IoU (Intersection over Union) computation                               #
# ====================================================================== #


class TestIoUComputation:
    """Tests for IoU metric (complement to Dice)."""

    @staticmethod
    def _iou(pred: np.ndarray, true: np.ndarray) -> float:
        """Compute IoU for binary masks."""
        intersection = (pred * true).sum()
        union = pred.sum() + true.sum() - intersection
        if union == 0:
            return 0.0
        return float(intersection / union)

    def test_perfect_overlap_iou_is_one(self):
        """Perfectly matching masks should yield IoU = 1.0."""
        mask = np.ones((32, 32), dtype=np.float32)
        iou = self._iou(mask, mask)
        assert abs(iou - 1.0) < 1e-6

    def test_no_overlap_iou_is_zero(self):
        """Non-overlapping masks should yield IoU = 0."""
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[:16, :] = 1.0
        true = np.zeros((32, 32), dtype=np.float32)
        true[16:, :] = 1.0
        iou = self._iou(pred, true)
        assert iou == 0.0

    def test_iou_less_than_or_equal_to_dice(self):
        """IoU <= Dice for the same mask pair (mathematical property)."""
        rng = np.random.RandomState(42)
        pred = (rng.rand(32, 32) > 0.5).astype(np.float32)
        true = (rng.rand(32, 32) > 0.5).astype(np.float32)

        iou = self._iou(pred, true)
        intersection = (pred * true).sum()
        union_dice = pred.sum() + true.sum()
        dice = (2 * intersection + 1e-6) / (union_dice + 1e-6) if union_dice > 0 else 0
        assert iou <= dice + 1e-6

    def test_iou_known_value(self):
        """Test IoU with a known geometry (quarter overlap)."""
        # pred: top-left 16x16, true: all 32x32
        pred = np.zeros((32, 32), dtype=np.float32)
        pred[:16, :16] = 1.0  # 256 pixels
        true = np.ones((32, 32), dtype=np.float32)  # 1024 pixels

        # intersection = 256, union = 1024
        iou = self._iou(pred, true)
        expected = 256.0 / 1024.0
        assert abs(iou - expected) < 1e-6


# ====================================================================== #
#  Hausdorff Distance (scripts/evaluate_boundary.py)                       #
# ====================================================================== #


@skip_no_scipy
class TestHausdorffDistance:
    """Tests for Hausdorff Distance 95 and Average Surface Distance."""

    def test_identical_masks_hd95_is_zero(self):
        """Identical masks should have HD95 = 0."""
        from scripts.evaluate_boundary import hausdorff_distance_95

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1  # centered square

        hd95 = hausdorff_distance_95(mask, mask)
        assert hd95 == 0.0

    def test_empty_mask_returns_inf(self):
        """Empty prediction or ground truth should return inf."""
        from scripts.evaluate_boundary import hausdorff_distance_95

        empty = np.zeros((32, 32), dtype=np.uint8)
        filled = np.ones((32, 32), dtype=np.uint8)

        assert hausdorff_distance_95(empty, filled) == float("inf")
        assert hausdorff_distance_95(filled, empty) == float("inf")

    def test_hd95_increases_with_distance(self):
        """Masks farther apart should have higher HD95."""
        from scripts.evaluate_boundary import hausdorff_distance_95

        # Mask A: left side
        mask_a = np.zeros((32, 64), dtype=np.uint8)
        mask_a[8:24, 4:12] = 1

        # Mask B: close to A
        mask_b_close = np.zeros((32, 64), dtype=np.uint8)
        mask_b_close[8:24, 12:20] = 1

        # Mask C: far from A
        mask_b_far = np.zeros((32, 64), dtype=np.uint8)
        mask_b_far[8:24, 48:56] = 1

        hd_close = hausdorff_distance_95(mask_a, mask_b_close)
        hd_far = hausdorff_distance_95(mask_a, mask_b_far)

        assert hd_far > hd_close, (
            f"Farther mask should have larger HD95: close={hd_close}, far={hd_far}"
        )

    def test_hd95_is_symmetric(self):
        """HD95(A, B) should equal HD95(B, A)."""
        from scripts.evaluate_boundary import hausdorff_distance_95

        rng = np.random.RandomState(42)
        a = (rng.rand(32, 32) > 0.7).astype(np.uint8)
        b = (rng.rand(32, 32) > 0.7).astype(np.uint8)

        hd_ab = hausdorff_distance_95(a, b)
        hd_ba = hausdorff_distance_95(b, a)
        np.testing.assert_allclose(hd_ab, hd_ba, atol=1e-5)

    def test_voxel_spacing_affects_distance(self):
        """Non-unit voxel spacing should scale the HD95 accordingly."""
        from scripts.evaluate_boundary import hausdorff_distance_95

        mask_a = np.zeros((32, 32), dtype=np.uint8)
        mask_a[8:12, 8:12] = 1
        mask_b = np.zeros((32, 32), dtype=np.uint8)
        mask_b[20:24, 20:24] = 1

        hd_unit = hausdorff_distance_95(mask_a, mask_b, voxel_spacing=(1.0, 1.0))
        hd_2mm = hausdorff_distance_95(mask_a, mask_b, voxel_spacing=(2.0, 2.0))

        # With 2mm spacing, distance should be roughly 2x
        assert hd_2mm > hd_unit * 1.5, (
            f"2mm spacing should roughly double HD95: unit={hd_unit}, 2mm={hd_2mm}"
        )

    def test_average_surface_distance_identical(self):
        """Identical masks should have ASD = 0."""
        from scripts.evaluate_boundary import average_surface_distance

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 1

        asd = average_surface_distance(mask, mask)
        assert asd == 0.0

    def test_average_surface_distance_empty_returns_inf(self):
        """Empty mask should give ASD = inf."""
        from scripts.evaluate_boundary import average_surface_distance

        empty = np.zeros((32, 32), dtype=np.uint8)
        filled = np.ones((32, 32), dtype=np.uint8)

        assert average_surface_distance(empty, filled) == float("inf")

    def test_boundary_metrics_dict_structure(self):
        """compute_boundary_metrics should return dict with expected keys."""
        from scripts.evaluate_boundary import compute_boundary_metrics

        num_classes = 4
        pred = np.random.randint(0, num_classes, (5, 32, 32))
        true = np.random.randint(0, num_classes, (5, 32, 32))

        result = compute_boundary_metrics(pred, true, num_classes=num_classes)

        expected_keys = {
            "dice_NCR", "dice_ED", "dice_ET",
            "hd95_NCR", "hd95_ED", "hd95_ET",
            "asd_NCR", "asd_ED", "asd_ET",
            "dice_mean", "hd95_mean", "asd_mean",
        }
        assert expected_keys.issubset(set(result.keys())), (
            f"Missing keys: {expected_keys - set(result.keys())}"
        )


# ====================================================================== #
#  Calibration (scripts/evaluate_calibration.py)                           #
# ====================================================================== #


class TestExpectedCalibrationError:
    """Tests for ECE, MCE, Brier Score computation."""

    def test_perfectly_calibrated_model(self):
        """A model where predicted prob == actual freq should have low ECE."""
        from scripts.evaluate_calibration import expected_calibration_error

        # 1000 samples: predicted probability matches actual label frequency
        rng = np.random.RandomState(42)
        y_prob = rng.uniform(0, 1, 10000)
        y_true = (rng.uniform(0, 1, 10000) < y_prob).astype(float)

        result = expected_calibration_error(y_true, y_prob, n_bins=15)

        assert result.ece < 0.05, f"Calibrated model should have low ECE, got {result.ece}"
        assert result.n_samples == 10000

    def test_overconfident_model_has_high_ece(self):
        """A model that always predicts 0.99 for 50/50 data should have high ECE."""
        from scripts.evaluate_calibration import expected_calibration_error

        n = 1000
        y_true = np.array([0] * (n // 2) + [1] * (n // 2), dtype=float)
        y_prob = np.full(n, 0.99)  # always 99% confident

        result = expected_calibration_error(y_true, y_prob, n_bins=15)

        assert result.ece > 0.3, f"Overconfident model should have high ECE, got {result.ece}"

    def test_brier_score_perfect(self):
        """Perfect predictions should have Brier score = 0."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])

        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert result.brier < 1e-10

    def test_brier_score_worst(self):
        """Inverted predictions should have Brier score = 1.0."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 1.0, 0.0, 1.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])

        result = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert abs(result.brier - 1.0) < 1e-10

    def test_ece_input_validation_shape_mismatch(self):
        """Mismatched array shapes should raise ValueError."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="Shape mismatch"):
            expected_calibration_error(y_true, y_prob)

    def test_ece_input_validation_empty(self):
        """Empty arrays should raise ValueError."""
        from scripts.evaluate_calibration import expected_calibration_error

        with pytest.raises(ValueError, match="must not be empty"):
            expected_calibration_error(np.array([]), np.array([]))

    def test_ece_input_validation_bad_labels(self):
        """Labels not in {0, 1} should raise ValueError."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 2.0, 1.0])
        y_prob = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            expected_calibration_error(y_true, y_prob)

    def test_ece_input_validation_probs_out_of_range(self):
        """Probabilities outside [0, 1] should raise ValueError."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 1.0])
        y_prob = np.array([0.5, 1.5])

        with pytest.raises(ValueError, match="must be in"):
            expected_calibration_error(y_true, y_prob)

    def test_ece_bin_data_structure(self):
        """CalibrationResult should contain correct number of bins."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.1, 0.3, 0.7, 0.9])

        result = expected_calibration_error(y_true, y_prob, n_bins=10)

        assert len(result.bin_data) == 10
        total_count = sum(b.count for b in result.bin_data)
        assert total_count == 4

    def test_mce_is_max_bin_error(self):
        """MCE should equal the maximum bin error across all bins."""
        from scripts.evaluate_calibration import expected_calibration_error

        y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

        result = expected_calibration_error(y_true, y_prob, n_bins=10)

        max_error = max(b.bin_error for b in result.bin_data)
        assert abs(result.mce - max_error) < 1e-10


# ====================================================================== #
#  Calibration Shift                                                       #
# ====================================================================== #


class TestCalibrationShift:
    """Tests for compute_calibration_shift."""

    def test_no_shift_when_models_identical(self):
        """Identical predictions should produce zero shift deltas."""
        from scripts.evaluate_calibration import compute_calibration_shift

        y_true = np.array([0, 0, 1, 1, 0, 1], dtype=float)
        probs = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])

        shift = compute_calibration_shift(probs, probs, y_true)

        assert abs(shift.ece_delta) < 1e-10
        assert abs(shift.mce_delta) < 1e-10
        assert abs(shift.brier_delta) < 1e-10
        assert abs(shift.mean_confidence_shift) < 1e-10

    def test_overconfident_compressed_model_positive_shift(self):
        """A more overconfident compressed model should show positive ECE delta."""
        from scripts.evaluate_calibration import compute_calibration_shift

        rng = np.random.RandomState(42)
        n = 2000
        y_true = rng.randint(0, 2, n).astype(float)
        baseline_probs = rng.uniform(0.3, 0.7, n)  # moderate confidence
        # Compressed: push all predictions toward 0.95
        compressed_probs = np.clip(baseline_probs + 0.3, 0, 1)

        shift = compute_calibration_shift(baseline_probs, compressed_probs, y_true)

        assert shift.mean_confidence_shift > 0, (
            "Compressed model should be more confident on average"
        )

    def test_shift_shape_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        from scripts.evaluate_calibration import compute_calibration_shift

        y_true = np.array([0, 1], dtype=float)
        base = np.array([0.5, 0.5])
        comp = np.array([0.5, 0.5, 0.5])

        with pytest.raises(ValueError, match="same length"):
            compute_calibration_shift(base, comp, y_true)


# ====================================================================== #
#  CSV I/O for calibration                                                 #
# ====================================================================== #


class TestCalibrationCSV:
    """Tests for load_predictions_csv."""

    def test_load_valid_csv(self, tmp_path):
        """load_predictions_csv should read a well-formed CSV."""
        from scripts.evaluate_calibration import load_predictions_csv

        csv_path = str(tmp_path / "preds.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y_true", "y_prob"])
            writer.writerow([0, 0.1])
            writer.writerow([1, 0.9])
            writer.writerow([0, 0.3])
            writer.writerow([1, 0.7])

        y_true, y_prob = load_predictions_csv(csv_path)

        np.testing.assert_array_equal(y_true, [0, 1, 0, 1])
        np.testing.assert_allclose(y_prob, [0.1, 0.9, 0.3, 0.7])

    def test_missing_columns_raises(self, tmp_path):
        """CSV missing required columns should raise ValueError."""
        from scripts.evaluate_calibration import load_predictions_csv

        csv_path = str(tmp_path / "bad.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "score"])
            writer.writerow([0, 0.5])

        with pytest.raises(ValueError, match="missing columns"):
            load_predictions_csv(csv_path)

    def test_empty_csv_raises(self, tmp_path):
        """CSV with header but no data rows should raise ValueError."""
        from scripts.evaluate_calibration import load_predictions_csv

        csv_path = str(tmp_path / "empty.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["y_true", "y_prob"])

        with pytest.raises(ValueError, match="no data rows"):
            load_predictions_csv(csv_path)

    def test_missing_file_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        from scripts.evaluate_calibration import load_predictions_csv

        with pytest.raises(FileNotFoundError):
            load_predictions_csv("/nonexistent/path/to/preds.csv")


# ====================================================================== #
#  Extended classification metrics (scripts/evaluate_extended.py)           #
# ====================================================================== #


@skip_no_sklearn
class TestExtendedClassificationMetrics:
    """Tests for compute_classification_metrics from evaluate_extended.py."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield all metrics at 1.0."""
        from scripts.evaluate_extended import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])

        result = compute_classification_metrics(y_true, y_prob, threshold=0.5)

        assert result["auc"] == 1.0
        assert result["f1"] == 1.0
        assert result["precision"] == 1.0
        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 1.0

    def test_all_positive_predictions(self):
        """Predicting all positive should have sensitivity=1, low specificity."""
        from scripts.evaluate_extended import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9])

        result = compute_classification_metrics(y_true, y_prob, threshold=0.5)

        assert result["sensitivity"] == 1.0
        assert result["specificity"] == 0.0

    def test_all_negative_predictions(self):
        """Predicting all negative should have specificity=1, zero sensitivity."""
        from scripts.evaluate_extended import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.1])

        result = compute_classification_metrics(y_true, y_prob, threshold=0.5)

        assert result["sensitivity"] == 0.0
        assert result["specificity"] == 1.0

    def test_threshold_affects_predictions(self):
        """Different thresholds should change binary predictions and metrics."""
        from scripts.evaluate_extended import compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])

        result_low = compute_classification_metrics(y_true, y_prob, threshold=0.35)
        result_high = compute_classification_metrics(y_true, y_prob, threshold=0.65)

        # Low threshold: more positives
        assert result_low["sensitivity"] >= result_high["sensitivity"]
        # High threshold: more negatives
        assert result_high["specificity"] >= result_low["specificity"]

    def test_result_keys_present(self):
        """Result dict should contain all expected metric keys."""
        from scripts.evaluate_extended import compute_classification_metrics

        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.3, 0.8, 0.2, 0.9])

        result = compute_classification_metrics(y_true, y_prob)

        expected_keys = {"auc", "f1", "precision", "sensitivity", "specificity"}
        assert expected_keys == set(result.keys())


# ====================================================================== #
#  Extended segmentation metrics                                           #
# ====================================================================== #


@skip_no_sklearn
class TestExtendedSegmentationMetrics:
    """Tests for compute_segmentation_metrics from evaluate_extended.py."""

    def test_perfect_segmentation(self):
        """Perfect segmentation should yield Dice = 1.0 per foreground class."""
        from scripts.evaluate_extended import compute_segmentation_metrics

        mask = np.array([
            [[1, 2, 3], [1, 2, 3]],
            [[1, 2, 3], [1, 2, 3]],
        ])

        result = compute_segmentation_metrics(mask, mask, num_classes=4)

        assert result["dice"] > 0.99
        for d in result["per_class_dice"]:
            assert d > 0.99

    def test_completely_wrong_segmentation(self):
        """Completely wrong predictions should yield low Dice."""
        from scripts.evaluate_extended import compute_segmentation_metrics

        true = np.ones((2, 8, 8), dtype=int)   # all class 1
        pred = np.full((2, 8, 8), 2, dtype=int)  # all class 2

        result = compute_segmentation_metrics(true, pred, num_classes=4)

        assert result["dice"] < 0.5

    def test_result_structure(self):
        """Result should contain dice, dice_std, sensitivity, specificity, per_class_dice."""
        from scripts.evaluate_extended import compute_segmentation_metrics

        pred = np.random.randint(0, 4, (3, 16, 16))
        true = np.random.randint(0, 4, (3, 16, 16))

        result = compute_segmentation_metrics(true, pred, num_classes=4)

        assert "dice" in result
        assert "dice_std" in result
        assert "sensitivity" in result
        assert "specificity" in result
        assert "per_class_dice" in result
        assert len(result["per_class_dice"]) == 3  # 3 foreground classes


# ====================================================================== #
#  Multi-seed evaluation                                                   #
# ====================================================================== #


class TestMultiSeedEvaluation:
    """Tests for evaluate_multi_seed and format_mean_std."""

    def test_multi_seed_mean_and_std(self):
        """Multi-seed evaluation should compute correct mean and std."""
        from scripts.evaluate_extended import evaluate_multi_seed

        def eval_fn(seed):
            return {"metric_a": float(seed), "metric_b": float(seed * 2)}

        seeds = [1, 2, 3, 4, 5]
        result = evaluate_multi_seed(eval_fn, seeds)

        assert abs(result["metric_a"]["mean"] - 3.0) < 1e-6
        assert abs(result["metric_b"]["mean"] - 6.0) < 1e-6
        assert len(result["metric_a"]["values"]) == 5

    def test_format_mean_std(self):
        """format_mean_std should produce LaTeX-friendly string."""
        from scripts.evaluate_extended import format_mean_std

        result = format_mean_std(0.852, 0.031, decimals=3)
        assert "0.852" in result
        assert "0.031" in result
        assert "$\\pm$" in result
