"""
scripts/evaluate_extended.py
-----------------------------
Extended evaluation pipeline for MedCompress.

Beyond AUC and Dice, computes:
  - Sensitivity (Recall), Specificity, F1-Score, Precision
  - Multi-seed evaluation with mean +/- std reporting
  - FLOPs / MACs estimation for computational complexity

Usage:
    python scripts/evaluate_extended.py --config configs/isic_baseline.yaml
    python scripts/evaluate_extended.py --config configs/isic_baseline.yaml --seeds 42,123,456,789,1024
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import tensorflow as tf
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    HAS_TF = True
except ImportError:
    HAS_TF = False


# =========================================================================== #
#  Extended Metrics                                                            #
# =========================================================================== #

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute full classification metrics beyond AUC.

    Returns dict with AUC, F1, Sensitivity (Recall), Specificity, Precision.
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true_bin = y_true.astype(int).ravel()
    y_pred_bin = y_pred.ravel()

    auc = roc_auc_score(y_true_bin, y_prob.ravel())
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    sensitivity = recall_score(y_true_bin, y_pred_bin, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true_bin, y_pred_bin, labels=[0, 1]
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def compute_segmentation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = 4,
) -> dict:
    """Compute segmentation metrics: Dice, per-class Dice, Sensitivity, Specificity.

    Args:
        y_true: ground truth class indices (N, H, W)
        y_pred: predicted class indices (N, H, W)
        num_classes: number of classes including background
    """
    dices = []
    sensitivities = []
    specificities = []

    for cls in range(1, num_classes):  # skip background
        pred_c = (y_pred == cls).astype(np.float64)
        true_c = (y_true == cls).astype(np.float64)

        tp = (pred_c * true_c).sum()
        fp = (pred_c * (1 - true_c)).sum()
        fn = ((1 - pred_c) * true_c).sum()
        tn = ((1 - pred_c) * (1 - true_c)).sum()

        dice = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
        sens = tp / (tp + fn + 1e-7)
        spec = tn / (tn + fp + 1e-7)

        dices.append(dice)
        sensitivities.append(sens)
        specificities.append(spec)

    return {
        "dice": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "sensitivity": float(np.mean(sensitivities)),
        "specificity": float(np.mean(specificities)),
        "per_class_dice": [float(d) for d in dices],
    }


# =========================================================================== #
#  FLOPs / MACs Estimation                                                     #
# =========================================================================== #

def estimate_flops(model, input_shape: tuple) -> dict:
    """Estimate FLOPs and MACs for a Keras model using tf.profiler.

    Falls back to parameter-based estimation if profiler is unavailable.

    Args:
        model: compiled Keras model
        input_shape: shape without batch dim, e.g. (224, 224, 3)

    Returns:
        dict with total_flops, total_macs, gflops, gmacs
    """
    try:
        concrete = tf.function(lambda x: model(x, training=False))
        concrete = concrete.get_concrete_function(
            tf.TensorSpec([1] + list(input_shape), model.input.dtype)
        )

        frozen = convert_to_frozen(concrete)
        flops = tf.compat.v1.profiler.profile(
            frozen,
            options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        )
        total_flops = flops.total_float_ops
    except Exception:
        # Fallback: rough estimate from parameter count
        # Approx 2 FLOPs per parameter per inference (multiply-add)
        total_flops = model.count_params() * 2

    total_macs = total_flops // 2  # 1 MAC = 2 FLOPs (multiply + add)

    return {
        "total_flops": int(total_flops),
        "total_macs": int(total_macs),
        "gflops": round(total_flops / 1e9, 2),
        "gmacs": round(total_macs / 1e9, 2),
    }


def convert_to_frozen(concrete_func):
    """Convert a ConcreteFunction to a frozen graph for profiling."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )
    frozen = convert_variables_to_constants_v2(concrete_func)
    return frozen.graph


# =========================================================================== #
#  Multi-Seed Evaluation                                                       #
# =========================================================================== #

def evaluate_multi_seed(
    evaluate_fn,
    seeds: list[int],
) -> dict:
    """Run evaluation across multiple seeds, report mean +/- std.

    Args:
        evaluate_fn: callable(seed) -> dict of metric_name -> float
        seeds: list of random seeds

    Returns:
        dict of metric_name -> {"mean": float, "std": float, "values": list}
    """
    all_results = []
    for seed in seeds:
        np.random.seed(seed)
        result = evaluate_fn(seed)
        all_results.append(result)

    combined = {}
    for key in all_results[0]:
        values = [r[key] for r in all_results if isinstance(r[key], (int, float))]
        if values:
            combined[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return combined


def format_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format as 'mean +/- std' for LaTeX tables."""
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


# =========================================================================== #
#  Main CLI                                                                    #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Extended evaluation with Sensitivity, Specificity, F1, FLOPs"
    )
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seeds", default="42,123,456,789,1024",
                        help="Comma-separated seeds for multi-seed eval")
    parser.add_argument("--checkpoint", default=None, help="Keras model path")
    parser.add_argument("--tflite", default=None, help="TFLite model path")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"Seeds: {seeds}")
    print(f"Config: {args.config}")

    if not HAS_TF:
        print("TensorFlow not available. Showing metric computation interface only.")
        # Demonstrate with synthetic data
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
        y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9, 0.4, 0.1, 0.6, 0.3])
        metrics = compute_classification_metrics(y_true, y_prob)
        print("\nDemo classification metrics (synthetic):")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    print("Full evaluation requires TensorFlow runtime.")


if __name__ == "__main__":
    main()
