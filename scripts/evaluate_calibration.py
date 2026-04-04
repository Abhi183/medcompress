"""
scripts/evaluate_calibration.py
-------------------------------
Evaluate model calibration before and after compression using
Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and Brier Score.

Calibration matters for medical AI because clinicians rely on
confidence scores for triage decisions. A compressed model that
preserves AUC but becomes overconfident (or underconfident) can
lead to dangerous misallocation of clinical attention.

This script is pure numpy/sklearn -- no TensorFlow dependency --
so it runs on any machine including Mac without GPU.

Usage:
    # Compare baseline vs compressed model predictions
    python scripts/evaluate_calibration.py \
        --baseline-preds baseline_preds.csv \
        --compressed-preds compressed_preds.csv

    # Evaluate a single model
    python scripts/evaluate_calibration.py \
        --baseline-preds baseline_preds.csv

CSV format:
    y_true,y_prob
    0,0.12
    1,0.87
    ...

For segmentation, flatten pixel-level predictions into the same
two-column CSV format (one row per pixel).
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =========================================================================== #
#  Data structures                                                             #
# =========================================================================== #

@dataclass(frozen=True)
class BinData:
    """Single bin in a reliability diagram."""
    bin_lower: float
    bin_upper: float
    mean_predicted: float
    mean_actual: float
    count: int
    bin_error: float


@dataclass(frozen=True)
class CalibrationResult:
    """Complete calibration metrics for a single model."""
    ece: float
    mce: float
    brier: float
    bin_data: tuple[BinData, ...]
    n_samples: int


@dataclass(frozen=True)
class CalibrationShift:
    """Change in calibration from baseline to compressed model."""
    baseline: CalibrationResult
    compressed: CalibrationResult
    ece_delta: float
    mce_delta: float
    brier_delta: float
    max_bin_shift: float
    mean_confidence_shift: float


# =========================================================================== #
#  Core calibration computation                                                #
# =========================================================================== #

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> CalibrationResult:
    """Compute Expected Calibration Error using the standard binning approach.

    ECE divides predictions into M equal-width bins by predicted
    probability, then computes the weighted average of
    |average_confidence - accuracy| per bin.

    Also computes:
      - MCE: maximum bin calibration error (worst-case bin)
      - Brier Score: mean squared error between probability and label

    Args:
        y_true: Binary ground truth labels, shape (N,).
                Values must be 0 or 1.
        y_prob: Predicted probabilities for the positive class,
                shape (N,). Values must be in [0, 1].
        n_bins: Number of equal-width bins. Default 15 per
                Guo et al. (2017) "On Calibration of Modern
                Neural Networks".

    Returns:
        CalibrationResult with ece, mce, brier, bin_data, n_samples.

    Raises:
        ValueError: If inputs have mismatched shapes, are empty,
                    or contain values outside expected ranges.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()

    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_prob has {y_prob.shape[0]}"
        )
    if y_true.shape[0] == 0:
        raise ValueError("Inputs must not be empty")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    unique_labels = np.unique(y_true)
    if not np.all(np.isin(unique_labels, [0.0, 1.0])):
        raise ValueError(
            f"y_true must contain only 0 and 1, found {unique_labels}"
        )
    if np.any(y_prob < 0.0) or np.any(y_prob > 1.0):
        raise ValueError("y_prob must be in [0, 1]")

    n_samples = y_true.shape[0]

    # Brier score: mean squared difference
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Build equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data_list: list[BinData] = []
    bin_errors: list[float] = []
    weighted_error_sum = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Include right edge in the last bin to capture prob == 1.0
        if i < n_bins - 1:
            mask = (y_prob >= lower) & (y_prob < upper)
        else:
            mask = (y_prob >= lower) & (y_prob <= upper)

        count = int(mask.sum())

        if count == 0:
            bin_data_list.append(BinData(
                bin_lower=float(lower),
                bin_upper=float(upper),
                mean_predicted=float((lower + upper) / 2.0),
                mean_actual=0.0,
                count=0,
                bin_error=0.0,
            ))
            bin_errors.append(0.0)
            continue

        mean_predicted = float(np.mean(y_prob[mask]))
        mean_actual = float(np.mean(y_true[mask]))
        bin_error = abs(mean_predicted - mean_actual)

        weighted_error_sum += bin_error * count

        bin_data_list.append(BinData(
            bin_lower=float(lower),
            bin_upper=float(upper),
            mean_predicted=mean_predicted,
            mean_actual=mean_actual,
            count=count,
            bin_error=bin_error,
        ))
        bin_errors.append(bin_error)

    ece = weighted_error_sum / n_samples
    mce = max(bin_errors) if bin_errors else 0.0

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        brier=brier,
        bin_data=tuple(bin_data_list),
        n_samples=n_samples,
    )


# =========================================================================== #
#  Calibration shift under compression                                         #
# =========================================================================== #

def compute_calibration_shift(
    baseline_probs: np.ndarray,
    compressed_probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> CalibrationShift:
    """Quantify how calibration changes when a model is compressed.

    Computes calibration metrics for both models and reports the
    deltas. A positive ECE delta means the compressed model is
    worse-calibrated; negative means it improved (rare but possible
    with quantization-aware training).

    Also reports:
      - max_bin_shift: largest change in bin-level error across
        all bins, highlighting where compression hurt most.
      - mean_confidence_shift: average change in predicted
        probability (positive = compressed model is more confident).

    Args:
        baseline_probs: Predicted probabilities from the original model.
        compressed_probs: Predicted probabilities from the compressed model.
        y_true: Binary ground truth labels.
        n_bins: Number of calibration bins.

    Returns:
        CalibrationShift with per-model results and deltas.

    Raises:
        ValueError: If array shapes do not match.
    """
    baseline_probs = np.asarray(baseline_probs, dtype=np.float64).ravel()
    compressed_probs = np.asarray(compressed_probs, dtype=np.float64).ravel()
    y_true = np.asarray(y_true, dtype=np.float64).ravel()

    if not (baseline_probs.shape[0] == compressed_probs.shape[0] == y_true.shape[0]):
        raise ValueError(
            f"All arrays must have the same length. Got baseline={baseline_probs.shape[0]}, "
            f"compressed={compressed_probs.shape[0]}, y_true={y_true.shape[0]}"
        )

    baseline_result = expected_calibration_error(y_true, baseline_probs, n_bins)
    compressed_result = expected_calibration_error(y_true, compressed_probs, n_bins)

    # Per-bin shift: largest absolute change in bin error
    max_bin_shift = 0.0
    for b_bin, c_bin in zip(baseline_result.bin_data, compressed_result.bin_data):
        if b_bin.count > 0 or c_bin.count > 0:
            shift = abs(c_bin.bin_error - b_bin.bin_error)
            max_bin_shift = max(max_bin_shift, shift)

    # Mean confidence shift (positive = compressed is more confident)
    mean_confidence_shift = float(np.mean(compressed_probs - baseline_probs))

    return CalibrationShift(
        baseline=baseline_result,
        compressed=compressed_result,
        ece_delta=compressed_result.ece - baseline_result.ece,
        mce_delta=compressed_result.mce - baseline_result.mce,
        brier_delta=compressed_result.brier - baseline_result.brier,
        max_bin_shift=max_bin_shift,
        mean_confidence_shift=mean_confidence_shift,
    )


# =========================================================================== #
#  CSV I/O                                                                     #
# =========================================================================== #

def load_predictions_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load y_true and y_prob from a two-column CSV file.

    Expected columns: y_true, y_prob (header required).
    For segmentation, flatten pixel predictions into this format
    before saving.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (y_true, y_prob) as numpy arrays.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If columns are missing or data is malformed.
    """
    y_true_list: list[float] = []
    y_prob_list: list[float] = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} appears to be empty")

        required_cols = {"y_true", "y_prob"}
        actual_cols = set(reader.fieldnames)
        missing = required_cols - actual_cols
        if missing:
            raise ValueError(
                f"CSV file {path} is missing columns: {missing}. "
                f"Found: {actual_cols}"
            )

        for row_num, row in enumerate(reader, start=2):
            try:
                y_true_list.append(float(row["y_true"]))
                y_prob_list.append(float(row["y_prob"]))
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Malformed row {row_num} in {path}: {row}"
                ) from exc

    if not y_true_list:
        raise ValueError(f"CSV file {path} has no data rows")

    return np.array(y_true_list), np.array(y_prob_list)


# =========================================================================== #
#  Display utilities                                                           #
# =========================================================================== #

def print_calibration_result(result: CalibrationResult, label: str) -> None:
    """Print calibration metrics in a readable format."""
    print(f"\n[{label}] Calibration Metrics ({result.n_samples:,} samples)")
    print(f"  ECE:   {result.ece:.4f}")
    print(f"  MCE:   {result.mce:.4f}")
    print(f"  Brier: {result.brier:.4f}")
    print()
    print("  Reliability Diagram Data:")
    print(f"  {'Bin':>10}  {'Predicted':>10}  {'Actual':>10}  {'Count':>8}  {'Error':>8}")
    print(f"  {'---':>10}  {'---':>10}  {'---':>10}  {'---':>8}  {'---':>8}")
    for b in result.bin_data:
        if b.count > 0:
            print(
                f"  [{b.bin_lower:.2f},{b.bin_upper:.2f})"
                f"  {b.mean_predicted:10.4f}"
                f"  {b.mean_actual:10.4f}"
                f"  {b.count:8d}"
                f"  {b.bin_error:8.4f}"
            )


def print_calibration_shift(shift: CalibrationShift) -> None:
    """Print calibration shift analysis."""
    print("\n" + "=" * 60)
    print("  CALIBRATION SHIFT ANALYSIS (Baseline -> Compressed)")
    print("=" * 60)
    print(f"  ECE:   {shift.baseline.ece:.4f} -> {shift.compressed.ece:.4f}  "
          f"(delta: {shift.ece_delta:+.4f})")
    print(f"  MCE:   {shift.baseline.mce:.4f} -> {shift.compressed.mce:.4f}  "
          f"(delta: {shift.mce_delta:+.4f})")
    print(f"  Brier: {shift.baseline.brier:.4f} -> {shift.compressed.brier:.4f}  "
          f"(delta: {shift.brier_delta:+.4f})")
    print(f"  Max bin-level error shift: {shift.max_bin_shift:.4f}")
    print(f"  Mean confidence shift:     {shift.mean_confidence_shift:+.4f}")

    if shift.mean_confidence_shift > 0.01:
        print("\n  WARNING: Compressed model is more confident on average.")
        print("  Check high-confidence bins for overconfidence.")
    elif shift.mean_confidence_shift < -0.01:
        print("\n  NOTE: Compressed model is less confident on average.")

    if shift.ece_delta > 0.02:
        print(f"\n  WARNING: ECE increased by {shift.ece_delta:.4f} after compression.")
        print("  Consider temperature scaling or calibration-aware fine-tuning.")

    print("=" * 60)


# =========================================================================== #
#  Main CLI                                                                    #
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate model calibration (ECE, MCE, Brier) before "
            "and after compression. Reads predictions from CSV files "
            "with columns: y_true, y_prob."
        ),
    )
    parser.add_argument(
        "--baseline-preds",
        required=True,
        help="CSV file with baseline model predictions (y_true, y_prob)",
    )
    parser.add_argument(
        "--compressed-preds",
        default=None,
        help="CSV file with compressed model predictions (y_true, y_prob). "
             "If omitted, only baseline calibration is reported.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=15,
        help="Number of calibration bins (default: 15)",
    )
    args = parser.parse_args()

    # Load baseline predictions
    print(f"Loading baseline predictions from {args.baseline_preds}")
    y_true_baseline, y_prob_baseline = load_predictions_csv(args.baseline_preds)
    baseline_result = expected_calibration_error(
        y_true_baseline, y_prob_baseline, n_bins=args.n_bins,
    )
    print_calibration_result(baseline_result, "Baseline")

    # If compressed predictions are provided, compute shift
    if args.compressed_preds is not None:
        print(f"\nLoading compressed predictions from {args.compressed_preds}")
        y_true_compressed, y_prob_compressed = load_predictions_csv(
            args.compressed_preds,
        )

        # Verify labels match between the two files
        if not np.array_equal(y_true_baseline, y_true_compressed):
            print(
                "WARNING: y_true columns differ between baseline and "
                "compressed CSVs. Using baseline y_true for both.",
                file=sys.stderr,
            )

        compressed_result = expected_calibration_error(
            y_true_baseline, y_prob_compressed, n_bins=args.n_bins,
        )
        print_calibration_result(compressed_result, "Compressed")

        shift = compute_calibration_shift(
            y_prob_baseline, y_prob_compressed, y_true_baseline,
            n_bins=args.n_bins,
        )
        print_calibration_shift(shift)
    else:
        print("\nNo compressed predictions provided. "
              "Pass --compressed-preds to compare models.")


if __name__ == "__main__":
    main()
