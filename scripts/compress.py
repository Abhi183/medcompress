"""
scripts/compress.py
--------------------
Run a compression pipeline (QAT or KD) as specified by the config.

Usage:
    python scripts/compress.py --config configs/isic_qat.yaml
    python scripts/compress.py --config configs/isic_kd.yaml
    python scripts/compress.py --config configs/brats_kd.yaml
"""

import os
import sys
import argparse
import yaml
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.isic_loader import ISICDataset
from data.brats_loader import BraTSDataset
from models.baseline import (
    build_efficientnetb0, build_unet_full, build_unet_lite
)
from compression.qat import run_qat_pipeline
from compression.distillation import DistillationTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(config: dict):
    if config["dataset"] == "isic":
        return ISICDataset(config)
    return BraTSDataset(config)


def load_model(spec: dict, data_cfg: dict, task: str) -> tf.keras.Model:
    """Load a model from a checkpoint or build a fresh one."""
    name = spec["name"]
    num_classes = spec.get("num_classes", 1)
    checkpoint = spec.get("checkpoint")

    if checkpoint and os.path.exists(checkpoint):
        print(f"  Loading checkpoint: {checkpoint}")
        return tf.keras.models.load_model(checkpoint, compile=False)

    # Build fresh
    if name == "efficientnetb0":
        return build_efficientnetb0(
            num_classes=num_classes,
            dropout=spec.get("dropout", 0.3),
            input_shape=(data_cfg["image_size"], data_cfg["image_size"], 3),
        )
    elif name == "mobilenetv3small":
        backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False, weights="imagenet" if spec.get("pretrained") else None,
            input_shape=(data_cfg["image_size"], data_cfg["image_size"], 3),
            pooling="avg",
        )
        x = backbone.output
        x = tf.keras.layers.Dropout(spec.get("dropout", 0.2))(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(backbone.input, out, name="mobilenetv3small")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")]
        )
        return model
    elif name == "unet_full":
        n_ch = data_cfg["n_slices"] * len(data_cfg["modalities"])
        return build_unet_full(num_classes, n_ch, data_cfg["patch_size"])
    elif name == "unet_lite":
        n_ch = data_cfg["n_slices"] * len(data_cfg["modalities"])
        return build_unet_lite(num_classes, n_ch, data_cfg["patch_size"])
    else:
        raise ValueError(f"Unknown model name: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--export", default=None, choices=["tflite", "onnx"],
                        help="Force export format (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    task = config.get("task", "classification")
    method = config.get("compression", {}).get("method", "kd")

    # Infer method from config structure
    if "distillation" in config:
        method = "kd"
    elif "compression" in config and config["compression"].get("method") == "qat":
        method = "qat"

    print(f"\n{'='*60}")
    print(f"  MedCompress: {method.upper()} Compression")
    print(f"  Task: {task}  |  Dataset: {config['dataset']}")
    print(f"{'='*60}\n")

    dataset = load_dataset(config)
    train_ds = dataset.get_train_dataset()
    val_ds = dataset.get_val_dataset()

    # ── QAT pipeline ─────────────────────────────────────────────────── #
    if method == "qat":
        model_cfg = config.get("model", {})
        base_model = load_model(model_cfg, config["data"], task)

        calib_gen = None
        if config.get("export", {}).get("tflite", {}).get("precision") == "int8":
            n = config["export"]["tflite"].get("calibration_samples", 200)
            calib_gen = dataset.get_calibration_generator(n)

        results = run_qat_pipeline(base_model, train_ds, val_ds, config, calib_gen)
        print(f"\n[compress] QAT complete. TFLite model: {results['tflite_path']}")

    # ── KD pipeline ──────────────────────────────────────────────────── #
    elif method == "kd":
        teacher = load_model(config["teacher"], config["data"], task)
        teacher.trainable = False

        student = load_model(config["student"], config["data"], task)

        trainer = DistillationTrainer(teacher, student, config)
        out_dir = config["output"]["dir"]
        history = trainer.train(train_ds, val_ds, output_dir=out_dir)

        # Export student to TFLite if configured
        export_cfg = config.get("export", {}).get("tflite", {})
        if export_cfg.get("enabled", False):
            from compression.qat import export_to_tflite
            calib_gen = None
            if export_cfg.get("precision") == "int8":
                n = export_cfg.get("calibration_samples", 200)
                calib_gen = dataset.get_calibration_generator(n)

            tflite_out = os.path.join(out_dir, "student.tflite")
            export_to_tflite(
                student,
                precision=export_cfg.get("precision", "fp16"),
                calibration_gen=calib_gen,
                output_path=tflite_out,
            )

        # Export to ONNX if configured
        onnx_cfg = config.get("export", {}).get("onnx", {})
        if onnx_cfg.get("enabled", False):
            try:
                import tf2onnx
                onnx_out = os.path.join(out_dir, "student.onnx")
                tf2onnx.convert.from_keras(student, output_path=onnx_out)
                print(f"[compress] ONNX model saved to {onnx_out}")
            except ImportError:
                print("[compress] tf2onnx not installed. Skipping ONNX export.")

        print(f"\n[compress] KD complete. Best student: {out_dir}/student_best.keras")

    else:
        raise ValueError(f"Unknown compression method: {method}")


if __name__ == "__main__":
    main()
