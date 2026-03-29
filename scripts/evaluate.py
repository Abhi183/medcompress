"""
scripts/evaluate.py
--------------------
Evaluate a trained or compressed model on the test set.
Reports: AUC / Dice, model size, and TFLite inference latency.

Usage:
    python scripts/evaluate.py --config configs/isic_qat.yaml
    python scripts/evaluate.py --config configs/brats_kd.yaml --tflite path/to/model.tflite
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.isic_loader import ISICDataset
from data.brats_loader import BraTSDataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# =========================================================================== #
#  Keras model evaluation                                                      #
# =========================================================================== #

def evaluate_keras(model: tf.keras.Model, test_ds: tf.data.Dataset,
                   task: str) -> dict:
    print("[Eval] Running Keras model evaluation...")
    all_preds, all_labels = [], []

    for images, labels in test_ds:
        preds = model(images, training=False).numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    if task == "classification":
        auc = roc_auc_score(labels, preds)
        print(f"  AUC: {auc:.4f}")
        return {"auc": auc}
    else:
        # Dice coefficient (mean across foreground classes)
        pred_classes = np.argmax(preds, axis=-1)
        true_classes = np.argmax(labels, axis=-1)
        dices = []
        for cls in range(1, preds.shape[-1]):   # exclude background
            pred_c = (pred_classes == cls).astype(np.float32)
            true_c = (true_classes == cls).astype(np.float32)
            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()
            if union > 0:
                dices.append((2.0 * intersection + 1e-6) / (union + 1e-6))
        mean_dice = float(np.mean(dices)) if dices else 0.0
        print(f"  Mean Dice (excl. BG): {mean_dice:.4f}")
        return {"dice": mean_dice}


# =========================================================================== #
#  TFLite model evaluation                                                     #
# =========================================================================== #

def evaluate_tflite(tflite_path: str, test_ds: tf.data.Dataset,
                    task: str, n_warmup: int = 5) -> dict:
    print(f"[Eval] Loading TFLite model from {tflite_path}")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]

    all_preds, all_labels, latencies = [], [], []

    for batch_images, batch_labels in test_ds:
        for i in range(len(batch_images)):
            img = batch_images[i].numpy()
            img = np.expand_dims(img, axis=0)

            # Handle INT8 quantized input
            if input_dtype == np.uint8:
                scale, zero_point = input_details[0]["quantization"]
                img = (img / scale + zero_point).astype(np.uint8)

            interpreter.set_tensor(input_details[0]["index"], img)

            t0 = time.perf_counter()
            interpreter.invoke()
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms

            out = interpreter.get_tensor(output_details[0]["index"])

            # De-quantize INT8 output
            if output_details[0]["dtype"] == np.uint8:
                scale, zero_point = output_details[0]["quantization"]
                out = (out.astype(np.float32) - zero_point) * scale

            all_preds.append(out.squeeze())
            all_labels.append(batch_labels[i].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    lat = np.array(latencies)

    print(f"  Latency — median: {np.median(lat):.1f} ms  |  p95: {np.percentile(lat, 95):.1f} ms")

    results = {
        "latency_median_ms": float(np.median(lat)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "model_size_mb": os.path.getsize(tflite_path) / 1e6,
    }

    if task == "classification":
        auc = roc_auc_score(labels.ravel(), preds.ravel())
        print(f"  AUC: {auc:.4f}")
        results["auc"] = auc
    else:
        pred_classes = np.argmax(preds, axis=-1)
        true_classes = np.argmax(labels, axis=-1)
        dices = []
        for cls in range(1, preds.shape[-1]):
            pred_c = (pred_classes == cls).astype(np.float32)
            true_c = (true_classes == cls).astype(np.float32)
            intersection = (pred_c * true_c).sum()
            union = pred_c.sum() + true_c.sum()
            if union > 0:
                dices.append((2 * intersection + 1e-6) / (union + 1e-6))
        mean_dice = float(np.mean(dices)) if dices else 0.0
        print(f"  Mean Dice: {mean_dice:.4f}")
        results["dice"] = mean_dice

    return results


# =========================================================================== #
#  Model size utility                                                          #
# =========================================================================== #

def get_keras_model_size(model: tf.keras.Model, tmp_path: str = "/tmp/_tmp_model.keras") -> float:
    """Save model and return file size in MB."""
    model.save(tmp_path)
    size = os.path.getsize(tmp_path) / 1e6
    os.remove(tmp_path)
    return size


# =========================================================================== #
#  Parameter count                                                             #
# =========================================================================== #

def count_params(model: tf.keras.Model) -> dict:
    total = model.count_params()
    trainable = sum(tf.size(v).numpy() for v in model.trainable_variables)
    return {"total_params": total, "trainable_params": int(trainable)}


# =========================================================================== #
#  Main                                                                        #
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None, help="Path to Keras checkpoint")
    parser.add_argument("--tflite", default=None, help="Path to TFLite model")
    args = parser.parse_args()

    config = load_config(args.config)
    task = config.get("task", "classification")
    dataset_name = config["dataset"]

    # Load dataset
    if dataset_name == "isic":
        dataset = ISICDataset(config)
    else:
        dataset = BraTSDataset(config)
    test_ds = dataset.get_test_dataset()

    results = {}

    # Evaluate Keras model if provided
    if args.checkpoint:
        model = tf.keras.models.load_model(args.checkpoint, compile=False)
        params = count_params(model)
        size_mb = get_keras_model_size(model)
        print(f"\n[Eval] Model params: {params['total_params']:,}  |  Size: {size_mb:.1f} MB")
        keras_results = evaluate_keras(model, test_ds, task)
        results["keras"] = {**keras_results, **params, "size_mb": size_mb}

    # Evaluate TFLite model if provided
    tflite_path = args.tflite or config.get("output", {}).get("tflite_path")
    if tflite_path and os.path.exists(tflite_path):
        print(f"\n[Eval] TFLite model size: {os.path.getsize(tflite_path)/1e6:.2f} MB")
        tflite_results = evaluate_tflite(tflite_path, test_ds, task)
        results["tflite"] = tflite_results

    print("\n── Summary ──────────────────────────────────")
    for backend, metrics in results.items():
        print(f"[{backend.upper()}]")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print("─────────────────────────────────────────────\n")

    return results


if __name__ == "__main__":
    main()
