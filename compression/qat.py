"""
compression/qat.py
-------------------
Quantization-Aware Training (QAT) pipeline using TensorFlow Model Optimization.
Wraps a trained Keras model with fake-quantization nodes and fine-tunes.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot


def apply_qat(model: tf.keras.Model,
              annotate_fn=None) -> tf.keras.Model:
    """
    Wrap a Keras model with QAT fake-quantization nodes.

    Args:
        model: Pretrained Keras model (FP32).
        annotate_fn: Optional custom annotation function for mixed-precision QAT.
                     If None, quantizes all supported layers.

    Returns:
        QAT-wrapped model ready for fine-tuning. After fine-tuning, call
        `strip_qat()` to get a quantized model that can be exported.
    """
    if annotate_fn is not None:
        # Mixed-precision: selectively annotate layers
        annotated = tfmot.quantization.keras.quantize_annotate_model(
            model, annotate_fn
        )
        qat_model = tfmot.quantization.keras.quantize_apply(annotated)
    else:
        # Uniform INT8 QAT for all supported layers
        qat_model = tfmot.quantization.keras.quantize_model(model)

    return qat_model


def strip_qat(qat_model: tf.keras.Model) -> tf.keras.Model:
    """
    Remove fake-quantization wrappers and return a stripped model
    suitable for TFLite conversion.
    """
    return tfmot.quantization.keras.strip_pruning(qat_model)


def run_qat_pipeline(base_model: tf.keras.Model,
                     train_ds: tf.data.Dataset,
                     val_ds: tf.data.Dataset,
                     config: dict,
                     calibration_gen=None) -> dict:
    """
    Full QAT pipeline:
      1. Wrap model with fake-quantization nodes.
      2. Fine-tune for a small number of epochs.
      3. Strip QAT wrappers.
      4. Export to TFLite INT8.

    Args:
        base_model: Pretrained FP32 Keras model.
        train_ds: Training tf.data.Dataset.
        val_ds: Validation tf.data.Dataset.
        config: Experiment config dict (from YAML).
        calibration_gen: Generator yielding representative data for INT8
                         calibration (used if QAT export is static INT8).

    Returns:
        dict with keys: qat_model, stripped_model, tflite_path, history.
    """
    comp_cfg = config.get("compression", {})
    train_cfg = config["training"]
    out_cfg = config["output"]

    os.makedirs(out_cfg["dir"], exist_ok=True)

    # ── Step 1: Apply QAT ──────────────────────────────────────────────── #
    print("[QAT] Wrapping model with fake-quantization nodes...")
    qat_model = apply_qat(base_model)

    # Recompile with lower learning rate for fine-tuning
    lr = train_cfg.get("learning_rate", 1e-5)
    task = config.get("task", "classification")

    if task == "classification":
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.AUC(name="auc")]
        )
    else:
        from models.baseline import dice_ce_loss, DiceCoefficient
        num_classes = base_model.output_shape[-1]
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss=dice_ce_loss(num_classes),
            metrics=[DiceCoefficient(num_classes, name="dice")]
        )

    # ── Step 2: Fine-tune ──────────────────────────────────────────────── #
    print(f"[QAT] Fine-tuning for {train_cfg['epochs']} epochs...")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(out_cfg["dir"], "qat_best.keras"),
            monitor="val_auc" if task == "classification" else "val_dice",
            mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    history = qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
    )

    # ── Step 3: Strip QAT wrappers ────────────────────────────────────── #
    print("[QAT] Stripping fake-quantization wrappers...")
    stripped = tfmot.quantization.keras.strip_pruning(qat_model)

    # ── Step 4: Export to TFLite ──────────────────────────────────────── #
    tflite_path = None
    export_cfg = config.get("export", {}).get("tflite", {})
    if export_cfg.get("enabled", True):
        tflite_path = export_to_tflite(
            stripped,
            precision=export_cfg.get("precision", "int8"),
            calibration_gen=calibration_gen,
            output_path=out_cfg.get("tflite_path",
                                    os.path.join(out_cfg["dir"], "model_int8.tflite"))
        )

    return {
        "qat_model": qat_model,
        "stripped_model": stripped,
        "tflite_path": tflite_path,
        "history": history,
    }


def export_to_tflite(model: tf.keras.Model,
                     precision: str = "int8",
                     calibration_gen=None,
                     output_path: str = "model.tflite") -> str:
    """
    Convert a Keras model to TFLite with specified precision.

    Args:
        model: Keras model (stripped of QAT nodes if applicable).
        precision: One of 'fp32', 'fp16', 'int8'.
        calibration_gen: Generator for INT8 static calibration (required for int8).
        output_path: Path to write the .tflite file.

    Returns:
        Path to the exported TFLite file.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if precision == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif precision == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if calibration_gen is not None:
            converter.representative_dataset = calibration_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    # Default: fp32 (no optimization flags)

    print(f"[TFLite] Converting to {precision.upper()}...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / 1e6
    print(f"[TFLite] Saved to {output_path} ({size_mb:.2f} MB)")
    return output_path
