"""
scripts/evaluate_boundary.py
-----------------------------
Boundary-aware evaluation metrics for medical image segmentation.

Adds Hausdorff Distance 95 (HD95) and Average Surface Distance (ASD)
to quantify the clinical impact of quantization on tumor boundaries.
A model that loses 2.2% Dice may lose 0.5mm or 3mm at boundaries,
and those numbers matter for surgical planning.

Usage:
    python scripts/evaluate_boundary.py --config configs/brats_baseline.yaml
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def hausdorff_distance_95(pred: np.ndarray, true: np.ndarray,
                          voxel_spacing: tuple = (1.0, 1.0)) -> float:
    """Compute 95th percentile Hausdorff Distance between two binary masks.

    HD95 measures the 95th percentile of the surface-to-surface distance
    between predicted and ground truth boundaries. It is less sensitive
    to outliers than max Hausdorff Distance and directly measures boundary
    accuracy in millimeters (given voxel spacing).

    Args:
        pred: Binary prediction mask (H, W) or (H, W, D)
        true: Binary ground truth mask, same shape
        voxel_spacing: Physical size per voxel in mm

    Returns:
        HD95 in mm. Returns inf if either mask is empty.
    """
    if pred.sum() == 0 or true.sum() == 0:
        return float("inf")

    # Surface voxels = boundary of the binary mask
    pred_border = _get_surface(pred)
    true_border = _get_surface(true)

    # Distance transform from each surface
    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_true = distance_transform_edt(~true_border, sampling=voxel_spacing)

    # Surface distances: distance from each surface point to nearest opposite surface
    dist_pred_to_true = dt_true[pred_border]
    dist_true_to_pred = dt_pred[true_border]

    all_distances = np.concatenate([dist_pred_to_true, dist_true_to_pred])

    return float(np.percentile(all_distances, 95))


def average_surface_distance(pred: np.ndarray, true: np.ndarray,
                             voxel_spacing: tuple = (1.0, 1.0)) -> float:
    """Compute Average Surface Distance between two binary masks.

    ASD is the mean distance from all surface points on the prediction
    to the nearest surface point on the ground truth, averaged
    bidirectionally.

    Args:
        pred: Binary prediction mask
        true: Binary ground truth mask
        voxel_spacing: Physical size per voxel in mm

    Returns:
        ASD in mm. Returns inf if either mask is empty.
    """
    if pred.sum() == 0 or true.sum() == 0:
        return float("inf")

    pred_border = _get_surface(pred)
    true_border = _get_surface(true)

    dt_pred = distance_transform_edt(~pred_border, sampling=voxel_spacing)
    dt_true = distance_transform_edt(~true_border, sampling=voxel_spacing)

    dist_pred_to_true = dt_true[pred_border]
    dist_true_to_pred = dt_pred[true_border]

    return float(
        (dist_pred_to_true.mean() + dist_true_to_pred.mean()) / 2.0
    )


def _get_surface(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels from a binary mask using erosion."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=1)
    return mask & ~eroded


def compute_boundary_metrics(
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    num_classes: int = 4,
    voxel_spacing: tuple = (1.0, 1.0),
) -> dict:
    """Compute per-class Dice, HD95, and ASD for segmentation evaluation.

    This is the metric set that clinical reviewers expect for
    segmentation papers. Dice alone does not capture boundary error.

    Args:
        pred_classes: Predicted class indices (N, H, W)
        true_classes: Ground truth class indices (N, H, W)
        num_classes: Number of classes including background
        voxel_spacing: Physical voxel size in mm

    Returns:
        Dict with per-class and mean metrics.
    """
    class_names = {1: "NCR", 2: "ED", 3: "ET"}
    results = {}

    for cls in range(1, num_classes):
        pred_c = (pred_classes == cls).astype(np.uint8)
        true_c = (true_classes == cls).astype(np.uint8)

        # Dice
        intersection = (pred_c & true_c).sum()
        union = pred_c.sum() + true_c.sum()
        dice = (2 * intersection + 1e-7) / (union + 1e-7)

        # Boundary metrics (per sample, then average)
        hd95_values = []
        asd_values = []
        for i in range(len(pred_c)):
            if true_c[i].sum() > 0:  # skip samples without this class
                hd95_values.append(
                    hausdorff_distance_95(pred_c[i], true_c[i], voxel_spacing))
                asd_values.append(
                    average_surface_distance(pred_c[i], true_c[i], voxel_spacing))

        name = class_names.get(cls, f"class_{cls}")
        results[f"dice_{name}"] = float(dice)
        results[f"hd95_{name}"] = (
            float(np.mean(hd95_values)) if hd95_values else float("inf"))
        results[f"asd_{name}"] = (
            float(np.mean(asd_values)) if asd_values else float("inf"))

    # Mean across foreground classes
    fg_dices = [results[f"dice_{class_names[c]}"] for c in range(1, num_classes)]
    fg_hd95 = [results[f"hd95_{class_names[c]}"] for c in range(1, num_classes)
               if results[f"hd95_{class_names[c]}"] < float("inf")]
    fg_asd = [results[f"asd_{class_names[c]}"] for c in range(1, num_classes)
              if results[f"asd_{class_names[c]}"] < float("inf")]

    results["dice_mean"] = float(np.mean(fg_dices))
    results["hd95_mean"] = float(np.mean(fg_hd95)) if fg_hd95 else float("inf")
    results["asd_mean"] = float(np.mean(fg_asd)) if fg_asd else float("inf")

    return results
